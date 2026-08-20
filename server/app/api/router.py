"""
Main FastAPI Router aggregation for AutoRoll Central Server.
"""

from fastapi import APIRouter

from server.app.api.endpoints.attendance import router as attendance_router
from server.app.api.endpoints.auth import router as auth_router
from server.app.api.endpoints.cameras import router as cameras_router
from server.app.api.endpoints.metrics import router as metrics_router
from server.app.api.endpoints.scheduler import router as scheduler_router
from server.app.api.endpoints.students import router as students_router
from server.app.api.endpoints.workers import router as workers_router

api_router = APIRouter()

# Register sub-routers under /api/v1
api_router.include_router(auth_router, prefix="/api/v1")
api_router.include_router(students_router, prefix="/api/v1")
api_router.include_router(attendance_router)
api_router.include_router(workers_router, prefix="/api/v1")
api_router.include_router(cameras_router, prefix="/api/v1")
api_router.include_router(scheduler_router, prefix="/api/v1")
api_router.include_router(metrics_router, prefix="/api/v1")
