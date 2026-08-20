"""
AutoRoll FastAPI Central Server Main Entry Point.
Control Plane and Application Backend for Privacy-Preserving Face Recognition Attendance.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logger import get_logger
from app.api.router import api_router
from app.database.session import Base, engine
from app.api.websocket.manager import ws_manager

logger = get_logger("server_main")
settings = get_settings()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware injecting unique X-Request-ID trace header for request tracking.
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AutoRoll Central Server Database Schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
    yield
    logger.info("Shutting down AutoRoll Central Server...")


app = FastAPI(
    title="AutoRoll Central Server API",
    description="Privacy-Preserving AI-Powered Distributed Face Recognition Attendance System",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware Pipeline
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

from app.api.websocket.monitoring import router as ws_monitoring_router
app.include_router(ws_monitoring_router)



@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "AutoRoll Central Server Running",
        "docs_url": "/docs",
        "environment": settings.APP_ENV,
    }


@app.get("/health", tags=["System Health"])
@app.get("/api/v1/health", tags=["System Health"])
async def health_check():
    """
    Liveness probe endpoint.
    """
    return {
        "status": "healthy",
        "service": "AutoRoll Central Server",
        "version": "0.1.0",
    }


@app.get("/ready", tags=["System Health"])
async def readiness_check():
    """
    Readiness probe endpoint testing DB connection.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return Response(
            content='{"status": "not_ready", "reason": "database_error"}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )


@app.websocket("/ws/clients")
async def websocket_client_endpoint(websocket: WebSocket):
    await ws_manager.connect_client(websocket)
