import io
import numpy as np
# pyrefly: ignore [missing-import]
from pydub import AudioSegment
from fastapi.testclient import TestClient
from app.main import app
from app.services.quality import assess_audio_quality

def create_wav_bytes(duration_sec=3.0, freq=440.0, vol=0.5, sample_rate=16000):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    # Generate a sine wave
    audio_data = np.sin(freq * t * 2 * np.pi) * vol
    
    # Convert to 16-bit integer
    audio_data_int16 = np.int16(audio_data * 32767)
    
    # Create an AudioSegment
    audio_segment = AudioSegment(
        audio_data_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    
    buf = io.BytesIO()
    audio_segment.export(buf, format="wav")
    return buf.getvalue()

def test_assess_audio_quality_insufficient_duration():
    # 0.5s audio
    samples = np.random.uniform(-0.1, 0.1, 8000).astype(np.float32)
    assert assess_audio_quality(samples, 16000) == "insufficient"

def test_assess_audio_quality_insufficient_silence():
    # 3s silence
    samples = np.zeros(48000, dtype=np.float32)
    assert assess_audio_quality(samples, 16000) == "insufficient"

def test_assess_audio_quality_degraded_short():
    # 1.5s audio
    samples = np.random.uniform(-0.2, 0.2, 24000).astype(np.float32)
    assert assess_audio_quality(samples, 16000) == "degraded"

def test_assess_audio_quality_degraded_clipping():
    # 3s audio, highly clipped
    samples = np.ones(48000, dtype=np.float32)
    assert assess_audio_quality(samples, 16000) == "degraded"
    
def test_assess_audio_quality_degraded_quiet():
    # 3s audio, quiet but above silence threshold (RMS ~0.008, which is < 0.01 degraded threshold but > 0.005 silence)
    samples = np.random.uniform(-0.015, 0.015, 48000).astype(np.float32)
    assert assess_audio_quality(samples, 16000) == "degraded"

def test_assess_audio_quality_good():
    # 3s audio, good volume
    samples = np.random.uniform(-0.4, 0.4, 48000).astype(np.float32)
    assert assess_audio_quality(samples, 16000) == "good"
