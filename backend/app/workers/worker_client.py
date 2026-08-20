"""
Worker Client — AutoRoll Phase 14
HTTP and WebSocket client used by remote GPU worker nodes to communicate with the Central Control Server.
"""

import asyncio
import time
import requests
from typing import Any, Dict, Optional
from app.core.logger import get_logger
from app.workers.models import WorkerRegistrationRequest, WorkerHeartbeatRequest, WorkerStatus

logger = get_logger("worker_client")


class WorkerControlPlaneClient:
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        secret_token: str = "autoroll_secret_2026",
    ):
        self.server_url = server_url.rstrip("/")
        self.secret_token = secret_token
        self.is_registered = False

    def register(self, req: WorkerRegistrationRequest) -> bool:
        """Send worker registration payload to central server REST API."""
        try:
            url = f"{self.server_url}/api/v1/workers/register"
            payload = req.model_dump()
            res = requests.post(url, json=payload, timeout=5.0)

            if res.status_code == 200:
                self.is_registered = True
                logger.info(f"Worker '{req.worker_id}' successfully registered with Central Control Server.")
                return True
            else:
                logger.error(f"Registration failed: HTTP {res.status_code} - {res.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Central Control Server at {self.server_url}: {e}")
            return False

    def send_heartbeat(self, req: WorkerHeartbeatRequest) -> Optional[Dict[str, Any]]:
        """Send periodic heartbeat to central server."""
        try:
            url = f"{self.server_url}/api/v1/workers/{req.worker_id}/heartbeat"
            payload = req.model_dump()
            res = requests.post(url, json=payload, timeout=3.0)

            if res.status_code == 200:
                return res.json()
            else:
                logger.warning(f"Heartbeat rejected: HTTP {res.status_code} - {res.text}")
                return None
        except Exception as e:
            logger.error(f"Heartbeat transmission error to {self.server_url}: {e}")
            return None
