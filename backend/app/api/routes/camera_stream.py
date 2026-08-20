"""
Camera Video Streaming & Management API Routes.
Provides MJPEG video stream, status, start, and stop endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.camera.manager import camera_manager
from app.services.camera_pipeline_service import camera_pipeline_service

router = APIRouter(prefix="/camera", tags=["Camera Stream"])


class CameraStartRequest(BaseModel):
    source_type: str = "local"  # local or rtsp
    camera_index: int = 0
    rtsp_url: str | None = None


@router.get("/status")
def get_camera_status():
    """
    Returns camera operational status and metrics.
    """
    source = camera_manager.get_source()
    if source is None:
        return {
            "status": "STOPPED",
            "is_opened": False,
            "metrics": {},
        }
    return {
        "status": "RUNNING" if source.is_opened() else "ERROR",
        "is_opened": source.is_opened(),
        "metrics": source.get_metrics(),
    }


@router.post("/start", status_code=status.HTTP_200_OK)
def start_camera(payload: CameraStartRequest):
    """
    Starts the video camera capture source and decoupled ML inference pipeline.
    """
    try:
        source = camera_manager.initialize_source(
            source_type=payload.source_type,
            camera_index=payload.camera_index,
            rtsp_url=payload.rtsp_url,
        )
        camera_pipeline_service.start_pipeline()
        return {
            "status": "STARTED",
            "source_type": payload.source_type,
            "camera_index": payload.camera_index,
            "metrics": source.get_metrics(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {e}")


@router.post("/stop", status_code=status.HTTP_200_OK)
def stop_camera():
    """
    Stops the video camera capture source and inference pipeline.
    """
    camera_pipeline_service.stop_pipeline()
    camera_manager.stop_source()
    return {"status": "STOPPED"}


@router.get("/mjpeg")
def mjpeg_video_feed():
    """
    MJPEG real-time video stream endpoint for HTML <img> rendering.
    """
    source = camera_manager.get_source()
    if source is None or not source.is_opened():
        # Auto-start default camera source if not already running
        try:
            camera_manager.initialize_source(source_type="local", camera_index=0)
            camera_pipeline_service.start_pipeline()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Camera unavailable: {e}")

    return StreamingResponse(
        camera_pipeline_service.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
