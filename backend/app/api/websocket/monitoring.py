"""
WebSocket Endpoint for Real-time Attendance & Camera Monitoring.
Streams live face count, bounding boxes, recognized identity, similarity score,
liveness score, FPS, total latency, and attendance decisions to frontend dashboards.
Strips out raw face embeddings to enforce privacy boundaries.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import get_logger
from app.api.websocket.manager import ws_manager

logger = get_logger("websocket_monitoring")

router = APIRouter(tags=["WebSocket Monitoring"])


@router.websocket("/ws/monitoring")
@router.websocket("/ws/clients")
async def websocket_monitoring_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live stream dashboard monitoring.
    Receives detection/recognition payloads, sanitizes raw embeddings,
    and broadcasts telemetry updates.
    """
    await ws_manager.connect_client(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            # Sanitize telemetry payload: remove raw face embeddings if present
            if "faces" in data and isinstance(data["faces"], list):
                for face in data["faces"]:
                    if isinstance(face, dict):
                        face.pop("embedding", None)
                        if "recognition" in face and isinstance(face["recognition"], dict):
                            face["recognition"].pop("embedding", None)

            # Broadcast sanitized event to connected dashboard clients
            await ws_manager.broadcast_to_clients("telemetry_update", data)
    except WebSocketDisconnect:
        ws_manager.disconnect_client(websocket)
    except Exception as e:
        logger.error(f"WebSocket monitoring error: {e}")
        ws_manager.disconnect_client(websocket)
