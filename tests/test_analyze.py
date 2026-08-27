import io
import os
import glob
from fastapi.testclient import TestClient
from app.main import app
from tests.test_audio_quality import create_wav_bytes

client = TestClient(app)

def test_analyze_audio_success():
    initial_wavs = set(glob.glob("**/*.wav", recursive=True))
    
    # Create a 3 second valid wav file in memory
    file_content = create_wav_bytes(duration_sec=3.0, vol=0.5)
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}
    
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    
    assert "contact_id" in data
    assert "gender" in data
    assert "age_bracket" in data
    assert "processing_ms" in data
    assert "audio_quality" in data
    
    assert data["gender"]["prediction"] == "unknown"
    assert data["gender"]["confidence"] == 0.0
    assert data["age_bracket"]["prediction"] == "unknown"
    assert data["age_bracket"]["confidence"] == 0.0
    
    assert isinstance(data["processing_ms"], int)
    assert data["processing_ms"] >= 0
    
    # Ensure audio was not persisted
    final_wavs = set(glob.glob("**/*.wav", recursive=True))
    assert initial_wavs == final_wavs

def test_analyze_unsupported_type():
    file_content = b"just some text"
    files = {"audio": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"

def test_analyze_empty_file():
    files = {"audio": ("test.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_analyze_insufficient_quality():
    # Test short audio
    short_content = create_wav_bytes(duration_sec=0.5)
    files = {"audio": ("short.wav", io.BytesIO(short_content), "audio/wav")}
    resp_short = client.post("/analyze", files=files)
    assert resp_short.status_code == 200
    assert resp_short.json()["audio_quality"] == "insufficient"
    
    # Test silent audio
    silent_content = create_wav_bytes(duration_sec=3.0, vol=0.0)
    files2 = {"audio": ("silent.wav", io.BytesIO(silent_content), "audio/wav")}
    resp_silent = client.post("/analyze", files=files2)
    assert resp_silent.status_code == 200
    assert resp_silent.json()["audio_quality"] == "insufficient"
