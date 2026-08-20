"""
Worker HTTP API Client for Server Communication.
Handles Registration, Heartbeat, and Event Publishing.
"""

from typing import Any

import httpx

from autoroll.common.logger import get_logger
from worker.state import WorkerHealthMetrics

logger = get_logger("worker_api_client")


class WorkerAPIClient:
    """
    HTTP client for communication between Worker and AutoRoll Server.
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.Client(timeout=5.0)

    def register_worker(self, metrics: WorkerHealthMetrics) -> dict[str, Any] | None:
        url = f"{self.server_url}/api/v1/workers/register"
        try:
            resp = self.client.post(url, json=metrics.model_dump())
            if resp.status_code in (200, 201):
                logger.info(f"Worker '{metrics.worker_id}' successfully registered with server.")
                return resp.json()
            else:
                logger.warning(
                    f"Worker registration failed with status {resp.status_code}: {resp.text}"
                )
                return None
        except Exception as e:
            logger.warning(f"Server unreachable during registration ({url}): {e}")
            return None

    def send_heartbeat(self, metrics: WorkerHealthMetrics) -> dict[str, Any] | None:
        url = f"{self.server_url}/api/v1/workers/heartbeat"
        try:
            resp = self.client.post(url, json=metrics.model_dump())
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"Heartbeat rejected by server ({resp.status_code}): {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"Failed to send heartbeat to server: {e}")
            return None

    def publish_recognition_event(
        self, event_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        url = f"{self.server_url}/api/v1/events/recognition"
        try:
            resp = self.client.post(url, json=event_payload)
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.warning(
                    f"Failed to publish recognition event ({resp.status_code}): {resp.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Network error publishing recognition event: {e}")
            return None

    def close(self) -> None:
        self.client.close()
