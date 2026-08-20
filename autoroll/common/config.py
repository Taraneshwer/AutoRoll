"""
Centralized Configuration Loader using Pydantic Settings.
Reads strictly from environment variables or .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Environment
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Server Configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    SECRET_KEY: str = "change-this-super-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./autoroll.db"

    # Vector & ML Model Thresholds
    AUTOROLL_ML_MODE: str = "production"  # production or test
    VECTOR_DIMENSION: int = 512
    SIMILARITY_THRESHOLD: float = 0.65
    LIVENESS_THRESHOLD: float = 0.90
    AUTOROLL_RECOGNITION_MODEL: str = "glint360k"  # glint360k or ms1mv2
    MODEL_VERSION: str = "arcface_iresnet50_v1"
    PAD_MODEL_VERSION: str = "minifasnet_v1"
    PAD_MODEL_PATH: str = "./models/minifasnet_v1.onnx"
    TEMPORAL_WINDOW_SIZE: int = 10
    SCRFD_MODEL_PATH: str = "./models/scrfd_10g_bnkps.onnx"
    ARCFACE_MODEL_PATH: str = "./models/arcface_iresnet50.onnx"
    ARCFACE_MS1MV2_PATH: str = "./models/pretrained/arcface_r50_ms1mv2/model.onnx"
    ARCFACE_GLINT_PATH: str = "./models/pretrained/arcface_r50_webface_or_glint/model.onnx"

    # Worker Node Configuration
    WORKER_ID: str = "worker-node-01"
    WORKER_HOST: str = "127.0.0.1"
    WORKER_PORT: int = 8001
    SERVER_WS_URL: str = "ws://127.0.0.1:8000/ws/workers"
    # Hardware Execution Configuration
    AUTOROLL_DEVICE: str = "auto"  # auto, cuda, or cpu
    DEVICE_TYPE: str = "cpu"

    def resolve_device(self) -> str:
        """
        Resolves active hardware execution device according to AUTOROLL_DEVICE.
        Primary choice is GPU ('cuda'); falls back to CPU if GPU execution is non-functional.
        - 'auto': returns 'cuda' if GPU execution is functional, otherwise 'cpu'
        - 'cuda': attempts 'cuda' execution, falls back to 'cpu' if non-functional
        - 'cpu': forces 'cpu' execution
        """
        from autoroll.ml.utils import is_cuda_functional
        dev = self.AUTOROLL_DEVICE.lower().strip()
        cuda_ok = is_cuda_functional()

        if dev == "cuda":
            if not cuda_ok:
                logger.warning(
                    "AUTOROLL_DEVICE='cuda' requested, but GPU execution is non-functional on this system. Falling back to CPU."
                )
                return "cpu"
            return "cuda"
        elif dev == "cpu":
            return "cpu"
        elif dev == "auto":
            return "cuda" if cuda_ok else "cpu"
        else:
            raise ValueError(f"Invalid AUTOROLL_DEVICE='{self.AUTOROLL_DEVICE}'. Expected 'auto', 'cuda', or 'cpu'.")

    # Storage & Privacy
    ENROLLMENT_STORAGE_DIR: str = "./data/enrollments"
    TEMP_STORAGE_DIR: str = "./data/tmp"
    ALLOW_RAW_IMAGE_STORAGE: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
