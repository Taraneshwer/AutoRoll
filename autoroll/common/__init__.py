"""
Shared utilities, settings, logger, metrics, and data transfer objects.
"""

from .config import Settings, get_settings
from .logger import get_logger
from .schemas import (
    BoundingBox,
    DetectionResult,
    DeviceType,
    FaceLandmarks,
    FrameProcessingResult,
    LivenessResult,
    RecognitionResult,
    TrackedFace,
)

__all__ = [
    "get_settings",
    "Settings",
    "get_logger",
    "BoundingBox",
    "FaceLandmarks",
    "DetectionResult",
    "LivenessResult",
    "RecognitionResult",
    "TrackedFace",
    "FrameProcessingResult",
    "DeviceType",
]
