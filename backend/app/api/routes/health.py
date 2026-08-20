"""
System Health, ML Status & Observability API Endpoints.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.ml.recognition.factory import get_recognizer
from app.ml.utils import get_execution_device
from app.services.worker_service import worker_control_plane

router = APIRouter(tags=["System Health & ML Status"])
settings = get_settings()


@router.get("/health")
@router.get("/ready")
@router.get("/api/v1/system/health")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    # Check DB Connection
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    recognizer = get_recognizer()
    device_name, _ = get_execution_device(settings.AUTOROLL_DEVICE)

    workers = worker_control_plane.list_workers()
    workers_online = sum(1 for w in workers if w["status"] == "ONLINE")

    cameras = worker_control_plane.list_cameras()
    cameras_online = sum(1 for c in cameras if c.get("status") == "ONLINE")

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "gpu": f"NVIDIA RTX 5060 Laptop GPU ({device_name.upper()})",
        "recognition_model": recognizer.get_model_id(),
        "model_version": recognizer.get_model_version(),
        "recognition_threshold": recognizer.get_recognition_threshold(),
        "workers_online": workers_online,
        "cameras_online": cameras_online,
    }


@router.get("/metrics")
@router.get("/api/v1/system/metrics")
def system_metrics() -> Dict[str, Any]:
    recognizer = get_recognizer()
    return {
        "active_model_id": recognizer.get_model_id(),
        "model_version": recognizer.get_model_version(),
        "embedding_dimension": 512,
        "liveness_threshold": settings.LIVENESS_THRESHOLD,
        "confirmation_frames": settings.AUTOROLL_CONFIRMATION_FRAMES,
        "attendance_cooldown_seconds": settings.AUTOROLL_ATTENDANCE_COOLDOWN_SECONDS,
    }


@router.get("/api/v1/ml/status")
def ml_status_check():
    recognizer = get_recognizer()
    device_name, providers = get_execution_device(settings.AUTOROLL_DEVICE)

    return {
        "status": "ready",
        "active_model_id": recognizer.get_model_id(),
        "model_version": recognizer.get_model_version(),
        "recognition_threshold": recognizer.get_recognition_threshold(),
        "embedding_dimension": 512,
        "device": device_name,
        "providers": providers,
        "liveness_model": settings.PAD_MODEL_VERSION,
        "liveness_threshold": settings.LIVENESS_THRESHOLD,
        "detector_model": "scrfd_10g_bnkps",
        "temporal_required_observations": settings.TEMPORAL_REQUIRED_OBSERVATIONS,
        "temporal_confirmation_window_ms": settings.TEMPORAL_CONFIRMATION_WINDOW_MS,
    }
