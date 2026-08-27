from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Voice Attribute Classifier"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_size: int = 10485760  # 10 MB in bytes
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
