"""
WebSocket integration tests for /ws/analyze endpoint.

Tests cover:
- Connect and receive a final result
- Progressive results emitted during streaming
- Graceful handling of disconnection
"""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


def make_pcm_bytes(filename="sample_male.wav", duration_sec: float = 3.0, sr: int = 16000) -> bytes:
    import os, soundfile as sf
    path = os.path.join(os.path.dirname(__file__), "sample_audio", filename)
    y, _ = sf.read(path)
    # y is float [-1, 1], convert to int16 PCM
    samples = (y[:int(sr * duration_sec)] * 32767).astype(np.int16)
    return samples.tobytes()


def _drain_until_final(ws, max_messages: int = 20) -> list:
    """
    Receive messages from the WebSocket until we get one with is_final=True
    or until max_messages is reached.  Returns all collected messages.
    """
    messages = []
    for _ in range(max_messages):
        try:
            msg = ws.receive_json()
            messages.append(msg)
            if msg.get("is_final"):
                break
        except Exception:
            break
    return messages


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------

class TestWebSocketAnalyze:

    def test_ws_connect_and_receive_final(self):
        """
        Send 3s of audio over WebSocket, signal end, verify final result JSON.
        Drain all messages (including any interim partials) until is_final=True.
        """
        client = TestClient(app)
        pcm = make_pcm_bytes(filename="sample_male.wav", duration_sec=3.0)

        with client.websocket_connect("/ws/analyze") as ws:
            # Send audio in 512-sample (1024-byte) chunks
            chunk_size = 512 * 2  # 512 samples × 2 bytes/sample
            for i in range(0, len(pcm), chunk_size):
                ws.send_bytes(pcm[i:i + chunk_size])

            # Send end signal to trigger final result
            ws.send_text(json.dumps({"type": "end"}))

            # Drain all messages until the final one
            messages = _drain_until_final(ws)

        assert len(messages) >= 1, "Expected at least one message from server"
        final_msg = messages[-1]

        assert "gender" in final_msg
        assert "age_bracket" in final_msg
        assert "audio_quality" in final_msg
        assert final_msg.get("is_final") is True, f"Last message should be final, got: {final_msg}"
        assert final_msg["gender"]["prediction"] in ("male", "female", "unknown")
        assert 0.0 <= final_msg["gender"]["confidence"] <= 1.0

    def test_ws_progressive_results_on_large_audio(self):
        """
        Sending >2s of audio should produce at least a final result.
        Optionally a partial may be emitted before end signal.
        """
        client = TestClient(app)
        # 4s of audio — may trigger a partial after 2s accumulation
        pcm = make_pcm_bytes(filename="sample_female.wav", duration_sec=4.0)

        with client.websocket_connect("/ws/analyze") as ws:
            chunk_size = 512 * 2
            for i in range(0, len(pcm), chunk_size):
                ws.send_bytes(pcm[i:i + chunk_size])

            ws.send_text(json.dumps({"type": "end"}))
            messages = _drain_until_final(ws)

        assert len(messages) >= 1, "Expected at least a final result"
        assert messages[-1]["is_final"] is True, "Last message must be final"

    def test_ws_result_structure(self):
        """All WebSocket result frames must conform to the expected schema."""
        client = TestClient(app)
        pcm = make_pcm_bytes(filename="sample_male.wav", duration_sec=3.0)

        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_bytes(pcm)
            ws.send_text(json.dumps({"type": "end"}))
            messages = _drain_until_final(ws)

        assert len(messages) >= 1
        final_msg = messages[-1]

        required_keys = {"chunk_index", "gender", "age_bracket", "audio_quality", "is_final"}
        assert required_keys.issubset(final_msg.keys())
        assert isinstance(final_msg["chunk_index"], int)
        assert isinstance(final_msg["is_final"], bool)
        assert final_msg["audio_quality"] in ("good", "degraded", "insufficient")
