"""
Main FastAPI Router aggregation for AutoRoll Central Server.
Registers all API v1 endpoints under /api/v1 namespace.
"""

from fastapi import APIRouter

from app.api.routes.attendance import router as attendance_router
from app.api.routes.auth import router as auth_router
from app.api.routes.cameras import router as cameras_router
from app.api.routes.camera_stream import router as camera_stream_router
from app.api.routes.enrollment import router as enrollment_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.recognition import router as recognition_router
from app.api.routes.scheduler import router as scheduler_router
from app.api.routes.students import router as students_router
from app.api.routes.workers import router as worker_router


api_router = APIRouter()

# Health and System Status
api_router.include_router(health_router)

# Register sub-routers under /api/v1
api_router.include_router(auth_router, prefix="/api/v1")
api_router.include_router(students_router, prefix="/api/v1")
api_router.include_router(enrollment_router, prefix="/api/v1")
api_router.include_router(recognition_router, prefix="/api/v1")
api_router.include_router(attendance_router, prefix="/api/v1")
api_router.include_router(camera_stream_router, prefix="/api/v1")
api_router.include_router(worker_router, prefix="/api/v1")
api_router.include_router(cameras_router, prefix="/api/v1")
api_router.include_router(scheduler_router, prefix="/api/v1")
api_router.include_router(metrics_router, prefix="/api/v1")
