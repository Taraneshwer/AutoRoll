"""
Worker Environment Configuration.
"""

import uuid

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    WORKER_ID: str = f"worker_{uuid.uuid4().hex[:8]}"
    SERVER_URL: str = "http://localhost:8000"
    HEARTBEAT_INTERVAL_SEC: float = 5.0
    DEVICE: str = "auto"
    RTSP_RECONNECT_INTERVAL_SEC: float = 3.0
    MAX_CAMERAS_PER_WORKER: int = 4
    RECOGNITION_INTERVAL_FRAMES: int = 10


def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()
