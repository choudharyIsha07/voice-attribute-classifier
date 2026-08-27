"""
WebSocket streaming buffer for progressive voice attribute inference.

Design
------
Clients send raw PCM audio (16-bit little-endian, mono, 16kHz) in arbitrary
chunks.  The buffer accumulates samples and runs inference on every 2-second
window, emitting a partial result each time.  A final result is emitted when
the client closes the connection.

Audio format over WebSocket
---------------------------
- Binary frames: raw 16-bit LE PCM at 16 kHz mono
- Text frames:   JSON control messages  {"type": "end"}

Progressive result format (JSON text frame to client)
------------------------------------------------------
See WebSocketChunkResult in schemas.py
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np

from .inference import InferenceProvider
from .quality import assess_audio_quality

logger = logging.getLogger(__name__)

CHUNK_WINDOW_SEC = 2.0   # run inference every 2 seconds of accumulated audio
TARGET_SR = 16000         # expected PCM sample rate from client


class StreamingBuffer:
    """
    Accumulates raw PCM int16 chunks and produces inference results on
    rolling 2-second windows.

    Usage:
        buf = StreamingBuffer(provider)
        for raw_bytes in audio_stream:
            result = buf.push(raw_bytes)
            if result:
                await ws.send_json(result)
        final = buf.finalize()
        await ws.send_json(final)
    """

    def __init__(self, provider: InferenceProvider, sample_rate: int = TARGET_SR):
        self._provider = provider
        self._sr = sample_rate
        self._samples: Deque[np.ndarray] = deque()
        self._total_samples = 0
        self._last_inference_samples = 0
        self._chunk_index = 0
        self._window_samples = int(CHUNK_WINDOW_SEC * sample_rate)

    def _get_current_array(self) -> np.ndarray:
        """Concatenate deque into a single float32 array."""
        if not self._samples:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._samples)).astype(np.float32)

    def push(self, raw_bytes: bytes) -> Optional[dict]:
        """
        Ingest a chunk of raw 16-bit LE PCM bytes.

        Returns a dict result when enough audio has accumulated for inference,
        or None if more data is needed.
        """
        if len(raw_bytes) == 0:
            return None

        # Decode int16 → float32
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        float_arr = int16_arr.astype(np.float32) / 32768.0
        self._samples.append(float_arr)
        self._total_samples += len(float_arr)

        if self._total_samples - self._last_inference_samples >= self._window_samples:
            self._last_inference_samples = self._total_samples
            return self._run_inference(is_final=False)
        return None

    def finalize(self) -> dict:
        """Run a final inference pass over all accumulated audio."""
        return self._run_inference(is_final=True)

    def _run_inference(self, is_final: bool) -> dict:
        samples = self._get_current_array()
        quality = assess_audio_quality(samples, self._sr)

        gender_pred, gender_conf, age_pred, age_conf, language = \
            self._provider.infer_attributes(samples, self._sr)

        self._chunk_index += 1

        # No longer trimming the buffer! We keep all samples so progressive 
        # and final results are evaluated over the entire call duration.

        return {
            "chunk_index": self._chunk_index,
            "gender": {"prediction": gender_pred, "confidence": gender_conf},
            "age_bracket": {"prediction": age_pred, "confidence": age_conf},
            "audio_quality": quality,
            "language": language,
            "is_final": is_final,
        }
