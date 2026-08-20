"""
Environment check and setup utility.
"""

import sys

from autoroll.common.config import get_settings
from autoroll.common.logger import get_logger

logger = get_logger("setup_env")


def check_environment():
    logger.info(f"Python Version: {sys.version}")
    settings = get_settings()
    logger.info(f"Loaded Settings - App Env: {settings.APP_ENV}, DB: {settings.DATABASE_URL}")
    logger.info(f"Worker ID: {settings.WORKER_ID}, Device: {settings.DEVICE_TYPE}")
    return True


if __name__ == "__main__":
    check_environment()
