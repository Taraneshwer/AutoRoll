"""
Worker Network Client Stub for Server Communication.
"""

from typing import Any

from autoroll.common.logger import get_logger
from worker.config import worker_settings

logger = get_logger("worker_client")


class WorkerClient:
    """
    Handles WebSocket connection and registration with Central Server.
    """

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or worker_settings.SERVER_WS_URL
        self.worker_id = worker_settings.WORKER_ID
        self.is_connected = False

    async def connect_and_listen(self):
        logger.info(
            f"Connecting worker '{self.worker_id}' to Central Server at {self.server_url}..."
        )
        # Connection logic stub
        self.is_connected = True
        logger.info("Worker registration handshake successful (Stub).")

    async def send_heartbeat(self, metrics: dict[str, Any]):
        if not self.is_connected:
            return
        logger.debug(f"Heartbeat metrics: {metrics}")
