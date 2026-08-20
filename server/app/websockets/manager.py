"""
Enhanced WebSocket Connection Manager for ML Workers and Client Dashboards.
Supports Real-Time Event Sequence Tracking, Event Envelopes, and Reconnection Resiliency.
"""

import asyncio
from typing import Any

from fastapi import WebSocket

from autoroll.common.logger import get_logger
from server.app.websockets.protocol import WebSocketEventEnvelope, WebSocketEventType

logger = get_logger("websocket_manager")


class ConnectionManager:
    """
    Manages active WebSocket connections for ML Workers and Frontend Dashboards.
    Wraps telemetry events in standardized sequence-numbered Event Envelopes.
    """

    def __init__(self):
        self.active_workers: dict[str, WebSocket] = {}
        self.active_clients: list[WebSocket] = []
        self._sequence_counter: int = 0

    def _next_sequence(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    async def connect_worker(self, worker_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_workers[worker_id] = websocket
        logger.info(f"ML Worker connected via WebSocket: '{worker_id}'")

    def disconnect_worker(self, worker_id: str) -> None:
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
            logger.info(f"ML Worker disconnected via WebSocket: '{worker_id}'")

    async def connect_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_clients.append(websocket)
        logger.info("Frontend dashboard client connected via WebSocket.")

    def disconnect_client(self, websocket: WebSocket) -> None:
        if websocket in self.active_clients:
            self.active_clients.remove(websocket)
            logger.info("Frontend dashboard client disconnected via WebSocket.")

    def wrap_event(
        self, event_type: WebSocketEventType | str, data: dict[str, Any]
    ) -> WebSocketEventEnvelope:
        if isinstance(event_type, str):
            event_type = WebSocketEventType(event_type)

        return WebSocketEventEnvelope(
            event_type=event_type,
            sequence_number=self._next_sequence(),
            data=data,
        )

    async def broadcast_to_clients(
        self, event_type: WebSocketEventType | str, data: dict[str, Any]
    ) -> None:
        envelope = self.wrap_event(event_type, data)
        payload = envelope.model_dump()

        for connection in list(self.active_clients):
            try:
                await connection.send_json(payload)
            except Exception as e:
                logger.error(f"Error broadcasting WebSocket message to client: {e}")
                self.disconnect_client(connection)

    def broadcast_sync(
        self, event_type: WebSocketEventType | str, data: dict[str, Any]
    ) -> None:
        """
        Synchronous wrapper for broadcasting sequence-numbered websocket messages.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.broadcast_to_clients(event_type, data))
            except RuntimeError:
                pass
        except Exception as e:
            logger.warning(f"WebSocket broadcast_sync error: {e}")


ws_manager = ConnectionManager()
