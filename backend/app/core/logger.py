"""
Structured Logger for AutoRoll components.
"""

import logging
import re
import sys

from app.core.config import get_settings

RTSP_CRED_REGEX = re.compile(r"(rtsp[s]?://)([^:@\s]+):([^:@\s]+)@")


def sanitize_log_message(msg: str) -> str:
    """
    Sanitizes log messages by redacting RTSP stream credentials, passwords, and tokens.
    """
    if not isinstance(msg, str):
        return str(msg)
    # Redact RTSP credentials
    msg = RTSP_CRED_REGEX.sub(r"\1***:***@", msg)
    return msg


class SanitizingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, str):
            record.msg = sanitize_log_message(record.msg)
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with sanitized output formatting.
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = SanitizingFormatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

    return logger
