from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./ai_counter.db"
    UPLOAD_DIR: str = str(Path(__file__).parent.parent.parent.parent / "uploads")
    PROCESSED_DIR: str = str(Path(__file__).parent.parent.parent.parent / "processed")
    MAX_IMAGE_SIZE_MB: int = 15
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MODEL_NAME: str = "yolov8s-world.pt"
    CONFIDENCE_THRESHOLD: float = 0.25

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
