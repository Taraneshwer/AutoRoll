"""
Camera Ingestion & Streaming Package.
"""

from app.camera.base import CameraSource
from app.camera.local_camera import LocalCameraSource
from app.camera.manager import camera_manager
from app.camera.rtsp_camera import RTSPCameraSource

__all__ = ["CameraSource", "LocalCameraSource", "RTSPCameraSource", "camera_manager"]
