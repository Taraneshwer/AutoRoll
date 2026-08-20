"""
Worker Health Monitor — AutoRoll Phase 14
Monitors worker heartbeats and transitions unhealthy workers to DEGRADED and then OFFLINE.
- Heartbeat interval: 5 seconds
- Degraded threshold: 10 seconds
- Offline threshold: 15 seconds
"""

import time
from typing import List, Tuple
from app.core.logger import get_logger
from app.workers.models import WorkerStatus, WorkerNodeInfo
from app.workers.worker_registry import WorkerRegistry

logger = get_logger("worker_health")


class WorkerHealthMonitor:
    def __init__(
        self,
        registry: WorkerRegistry,
        degraded_timeout_seconds: float = 10.0,
        offline_timeout_seconds: float = 15.0,
    ):
        self.registry = registry
        self.degraded_timeout = degraded_timeout_seconds
        self.offline_timeout = offline_timeout_seconds

    def check_health(self) -> List[Tuple[WorkerNodeInfo, WorkerStatus, WorkerStatus]]:
        """
        Check heartbeats for all registered workers.
        Returns list of tuples: (worker, old_status, new_status) for workers whose status changed.
        """
        now = time.time()
        status_changes = []

        for worker in self.registry.list_all():
            if worker.status == WorkerStatus.DRAINING:
                continue

            elapsed = now - worker.last_heartbeat
            old_status = worker.status
            new_status = old_status

            if elapsed >= self.offline_timeout:
                if old_status != WorkerStatus.OFFLINE:
                    new_status = WorkerStatus.OFFLINE
                    logger.error(
                        f"Worker '{worker.worker_id}' heartbeat timeout ({elapsed:.1f}s > {self.offline_timeout}s). Transitioning to OFFLINE."
                    )
            elif elapsed >= self.degraded_timeout:
                if old_status == WorkerStatus.ONLINE:
                    new_status = WorkerStatus.DEGRADED
                    logger.warning(
                        f"Worker '{worker.worker_id}' heartbeat delayed ({elapsed:.1f}s > {self.degraded_timeout}s). Transitioning to DEGRADED."
                    )

            if new_status != old_status:
                worker.status = new_status
                status_changes.append((worker, old_status, new_status))

        return status_changes
