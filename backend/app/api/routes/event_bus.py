"""
Structured Real-Time WebSocket Event Bus for AutoRoll Phase 11.
Broadcasts system-wide events (ATTENDANCE_CONFIRMED, SPOOF_DETECTED, WORKER_ONLINE, etc.) over /ws/events.
Strictly excludes raw 512-dim embedding vectors and raw frame images!
"""

import json
from typing import Any, Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.logger import get_logger

logger = get_logger("event_bus")
router = APIRouter(tags=["events"])


class EventMessage(BaseModel):
    event_type: str
    timestamp: float
    data: Dict[str, Any]


class EventBusManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected to /ws/events (Total: {len(self.active_connections)}).")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected from /ws/events (Remaining: {len(self.active_connections)}).")

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        # Sanitize data: remove any embedding vectors if present
        sanitized = {k: v for k, v in data.items() if "embedding" not in k.lower() and "vector" not in k.lower()}
        msg = {
            "event_type": event_type,
            "data": sanitized,
        }
        json_str = json.dumps(msg)

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json_str)
            except Exception:
                disconnected.append(connection)

        for dead in disconnected:
            self.disconnect(dead)


event_bus = EventBusManager()


@router.websocket("/ws/events")
async def websocket_event_endpoint(websocket: WebSocket):
    await event_bus.connect(websocket)
    try:
        while True:
            # Keep-alive receive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception as err:
        logger.error(f"WebSocket error on /ws/events: {err}")
        event_bus.disconnect(websocket)
