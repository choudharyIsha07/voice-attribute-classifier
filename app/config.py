from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Voice Attribute Classifier"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_size: int = 10485760  # 10 MB in bytes

    # Inference tuning
    inference_target_sr: int = 16000            # resample to 16kHz before inference
    model_id_age_gender: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    model_id_language: str = "speechbrain/lang-id-voxlingua107-ecapa"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
