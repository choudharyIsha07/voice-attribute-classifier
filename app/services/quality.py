import numpy as np

def assess_audio_quality(samples: np.ndarray, sample_rate: int) -> str:
    """
    Assess the quality of the uploaded audio based on duration, RMS, clipping, and silence.
    Returns: 'good', 'degraded', or 'insufficient'.
    """
    if len(samples) == 0:
        return "insufficient"
        
    duration = len(samples) / sample_rate
    if duration < 1.0:
        return "insufficient"
        
    # Calculate RMS (Root Mean Square) for loudness
    rms = np.sqrt(np.mean(samples**2))
    
    # Calculate clipping ratio (percentage of samples near max value)
    clipping_ratio = np.mean(np.abs(samples) >= 0.99)
    
    # Calculate silence ratio (percentage of 20ms frames below threshold)
    frame_length = int(sample_rate * 0.02) # 20ms
    if frame_length > 0 and len(samples) >= frame_length:
        # Reshape or iterate in chunks
        num_frames = len(samples) // frame_length
        frames = samples[:num_frames * frame_length].reshape(num_frames, frame_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        silence_ratio = np.mean(frame_rms < 0.005)
    else:
        silence_ratio = 0.0
        
    # Evaluate thresholds
    if silence_ratio > 0.90:
        return "insufficient"
        
    if duration < 2.0:
        return "degraded"
        
    if clipping_ratio > 0.05: # >5% of audio is clipped
        return "degraded"
        
    if rms < 0.01: # Too quiet
        return "degraded"
        
    if rms > 0.5: # Extremely loud / likely distorted overall
        return "degraded"
        
    return "good"
