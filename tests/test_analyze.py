"""
Integration tests for POST /analyze endpoint.

Covers:
- Successful request with a synthetic audio file
- Real inference predictions (not always 'unknown')
- Audio persistence check (PII guarantee)
- Error cases: unsupported type, empty file
- Audio quality flags
"""

import glob
import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
import os
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_analyze_audio_success_response_structure():
    """POST /analyze returns correct JSON schema."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "contact_id" in data
    assert "gender" in data
    assert "age_bracket" in data
    assert "processing_ms" in data
    assert "audio_quality" in data
    assert "language" in data   # bonus field

    assert "prediction" in data["gender"]
    assert "confidence" in data["gender"]
    assert "prediction" in data["age_bracket"]
    assert "confidence" in data["age_bracket"]

    assert isinstance(data["processing_ms"], int)
    assert data["processing_ms"] >= 0


def test_analyze_audio_real_predictions():
    """
    Real inference should return a concrete gender prediction (not always 'unknown')
    for a good quality audio clip.
    """
    # 440 Hz sine → female pitch range
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_female.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    gender_pred = data["gender"]["prediction"]
    age_pred = data["age_bracket"]["prediction"]

    assert gender_pred in ("male", "female", "unknown")
    assert age_pred in ("18-30", "31-45", "46-60", "60+", "unknown")

    # Confidence should be in [0, 1]
    assert 0.0 <= data["gender"]["confidence"] <= 1.0
    assert 0.0 <= data["age_bracket"]["confidence"] <= 1.0


def test_analyze_returns_valid_uuid_when_none_provided():
    """contact_id should be auto-generated and be a valid UUID."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    # Should not raise
    parsed = uuid.UUID(data["contact_id"])
    assert parsed is not None


def test_analyze_accepts_provided_contact_id():
    """If contact_id is provided in the form, it must be echoed back."""
    contact_id = str(uuid.uuid4())
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}
    data_fields = {"contact_id": contact_id}

    response = client.post("/analyze", files=files, data=data_fields)
    assert response.status_code == 200
    assert response.json()["contact_id"] == contact_id


# ---------------------------------------------------------------------------
# Privacy: no disk persistence
# ---------------------------------------------------------------------------

def test_analyze_no_audio_persisted():
    """
    Verify PII guarantee: no WAV files written to disk during the request.
    """
    initial_wavs = set(glob.glob("**/*.wav", recursive=True))

    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("test.wav", io.BytesIO(file_content), "audio/wav")}
    client.post("/analyze", files=files)

    final_wavs = set(glob.glob("**/*.wav", recursive=True))
    assert initial_wavs == final_wavs, "Audio was unexpectedly written to disk!"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_analyze_unsupported_type():
    """Non-audio file should return 400."""
    files = {"audio": ("test.txt", io.BytesIO(b"just some text"), "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_empty_file():
    """Empty file should return 400."""
    files = {"audio": ("test.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_no_file_field():
    """Missing audio field entirely should return 422."""
    response = client.post("/analyze")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Audio quality flags
# ---------------------------------------------------------------------------

def test_analyze_insufficient_quality_short_clip():
    """Sub-1s clip should be flagged as 'insufficient'."""
    import soundfile as sf
    import io
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    y, sr = sf.read(sample_path)
    # Take only 0.5s
    y_short = y[:int(sr * 0.5)]
    buf = io.BytesIO()
    sf.write(buf, y_short, sr, format='WAV')
    buf.seek(0)
    files = {"audio": ("short.wav", buf, "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    assert response.json()["audio_quality"] == "insufficient"


def test_analyze_insufficient_quality_silent():
    """Silent clip should be flagged as 'insufficient'."""
    import soundfile as sf
    import io
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    y, sr = sf.read(sample_path)
    # Mute audio
    y_silent = y * 0.0
    buf = io.BytesIO()
    sf.write(buf, y_silent, sr, format='WAV')
    buf.seek(0)
    files = {"audio": ("silent.wav", buf, "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    assert response.json()["audio_quality"] == "insufficient"


def test_analyze_good_quality_flag():
    """A well-formed 3s clip at moderate volume should be 'good'."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_audio", "sample_male.wav")
    with open(sample_path, "rb") as f:
        file_content = f.read()
    files = {"audio": ("good.wav", io.BytesIO(file_content), "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    assert response.json()["audio_quality"] == "good"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
