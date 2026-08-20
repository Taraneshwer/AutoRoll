"""
Environment check and setup utility.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import sys

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("setup_env")


def check_environment():
    logger.info(f"Python Version: {sys.version}")
    settings = get_settings()
    logger.info(f"Loaded Settings - App Env: {settings.APP_ENV}, DB: {settings.DATABASE_URL}")
    logger.info(f"Worker ID: {settings.WORKER_ID}, Device: {settings.DEVICE_TYPE}")
    return True


if __name__ == "__main__":
    check_environment()
