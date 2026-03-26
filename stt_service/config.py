from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    wake_word_model: str = "hey_jarvis"
    wake_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    whisper_model_size: str = "base"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    vad_silence_duration: float = 1.5
    partial_transcript_interval: float = 1.0

    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    audio_device_index: int = -1


settings = Settings()
