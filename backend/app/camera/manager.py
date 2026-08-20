"""
Camera Manager Singleton for instantiating and controlling active CameraSources.
"""

from typing import Any

from app.core.config import get_settings
from app.core.logger import get_logger
from app.camera.base import CameraSource
from app.camera.local_camera import LocalCameraSource
from app.camera.rtsp_camera import RTSPCameraSource

logger = get_logger("camera_manager")


class CameraManager:
    """
    Manages global camera instance lifecycle.
    """

    def __init__(self):
        self.active_source: CameraSource | None = None

    def initialize_source(
        self, source_type: str = "local", camera_index: int = 0, rtsp_url: str | None = None
    ) -> CameraSource:
        """
        Instantiates and starts a CameraSource.
        """
        if self.active_source and self.active_source.is_opened():
            self.active_source.stop()

        settings = get_settings()
        target_fps = getattr(settings, "AUTOROLL_CAMERA_FPS", 30)

        if source_type.lower() in ("local", "webcam", "usb"):
            self.active_source = LocalCameraSource(camera_index=camera_index, target_fps=target_fps)
        elif source_type.lower() == "rtsp" and rtsp_url:
            self.active_source = RTSPCameraSource(rtsp_url=rtsp_url, target_fps=target_fps)
        else:
            logger.warning(f"Unknown camera source_type='{source_type}'. Defaulting to local webcam (index 0).")
            self.active_source = LocalCameraSource(camera_index=0, target_fps=target_fps)

        try:
            self.active_source.start()
        except Exception as e:
            logger.error(f"Failed to start camera source: {e}")
            self.active_source = None
            raise

        return self.active_source

    def get_source(self) -> CameraSource | None:
        return self.active_source


    def stop_source(self) -> None:
        if self.active_source:
            self.active_source.stop()
            self.active_source = None


camera_manager = CameraManager()
