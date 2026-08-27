import io
import numpy as np
from fastapi import UploadFile, HTTPException
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

ALLOWED_MIME_TYPES = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4", "video/mp4", "audio/x-m4a"]

async def validate_audio_file(file: UploadFile, max_size: int) -> tuple[np.ndarray, int]:
    if file.content_type not in ALLOWED_MIME_TYPES and not file.filename.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
        
    try:
        # Determine format from filename to help pydub bypass ffprobe when possible (e.g. for wav)
        fmt = file.filename.split('.')[-1].lower() if file.filename else None
        if fmt == "m4a": fmt = "mp4"
        elif fmt == "mpeg": fmt = "mp3"
        
        try:
            audio = AudioSegment.from_file(io.BytesIO(content), format=fmt)
        except:
            # Fallback to auto-detect if explicitly passing format fails
            audio = AudioSegment.from_file(io.BytesIO(content))
    except CouldntDecodeError:
        raise HTTPException(status_code=400, detail="Invalid audio format or unable to decode")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing audio")
        
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
        
    # Extract raw samples as numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize to float32 between -1.0 and 1.0
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        # 8-bit or other format - approximate scaling
        max_val = float(np.max(np.abs(samples))) if np.max(np.abs(samples)) > 0 else 1.0
        samples = samples.astype(np.float32) / max_val
        
    return samples, audio.frame_rate
