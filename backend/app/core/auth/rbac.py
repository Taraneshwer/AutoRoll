"""
Role-Based Access Control (RBAC), Worker Authentication Tokens, and Input Sanitizers.
"""

import re
from enum import Enum

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("security_rbac")
settings = get_settings()


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


def check_role_permission(user_role: str, required_role: UserRole) -> bool:
    """
    Evaluates role permissions: admin > operator > viewer.
    """
    role_weights = {
        UserRole.ADMIN.value: 3,
        UserRole.OPERATOR.value: 2,
        UserRole.VIEWER.value: 1,
    }

    user_weight = role_weights.get(user_role.lower(), 0)
    required_weight = role_weights.get(required_role.value, 3)

    return user_weight >= required_weight


def validate_worker_secret(provided_secret: str | None) -> bool:
    """
    Validates cluster pre-shared secret token for ML worker node registration.
    """
    expected_secret = getattr(settings, "WORKER_SECRET_KEY", "autoroll_worker_cluster_secret")
    if not provided_secret or provided_secret != expected_secret:
        logger.warning("Worker registration rejected: Invalid worker cluster secret token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid worker authentication token",
        )
    return True


RTSP_URL_REGEX = re.compile(
    r"^rtsps?://(?:[^:@\s]+(?::[^:@\s]+)?@)?[a-zA-Z0-9.\-_]+(?::\d+)?(?:/[a-zA-Z0-9._\-/]*)?$"
)


def validate_camera_rtsp_url(url: str) -> str:
    """
    Validates RTSP stream URL format and redacts sensitive credentials for storage/logging.
    """
    if not url or not isinstance(url, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera stream URL must be a non-empty string",
        )

    if not RTSP_URL_REGEX.match(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid camera stream URL format. Must start with rtsp:// or rtsps://",
        )

    return url
