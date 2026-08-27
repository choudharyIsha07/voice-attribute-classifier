from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Voice Attribute Classifier"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_size: int = 10485760  # 10 MB in bytes

    # Inference tuning
    gender_pitch_threshold_hz: float = 165.0   # F0 below → male, above → female
    min_voiced_frames_ratio: float = 0.10       # minimum voiced frames to trust pitch
    inference_target_sr: int = 16000            # resample to 16kHz before inference

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
