"""
Worker Service managing ML Worker node registration and state.
"""

from server.app.db.models import WorkerNode
from server.app.repositories.worker_repository import WorkerRepository


class WorkerService:
    def __init__(self, worker_repo: WorkerRepository):
        self.worker_repo = worker_repo

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        cpu_percent: float = 0.0,
        ram_used_mb: float = 0.0,
        gpu_available: bool = False,
    ) -> WorkerNode:
        return self.worker_repo.create_or_update(
            worker_id=worker_id,
            hostname=hostname,
            state="READY",
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            gpu_available=gpu_available,
        )

    def list_active_workers(self) -> list[WorkerNode]:
        return self.worker_repo.list_active()
