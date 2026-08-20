"""
Worker Repository for database operations.
"""


from sqlalchemy.orm import Session

from server.app.db.models import WorkerNode


class WorkerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, worker_id: str) -> WorkerNode | None:
        return self.db.query(WorkerNode).filter(WorkerNode.id == worker_id).first()

    def list_active(self) -> list[WorkerNode]:
        return self.db.query(WorkerNode).filter(WorkerNode.state != "OFFLINE").all()

    def create_or_update(
        self,
        worker_id: str,
        hostname: str,
        state: str = "READY",
        cpu_percent: float = 0.0,
        ram_used_mb: float = 0.0,
        gpu_available: bool = False,
    ) -> WorkerNode:
        worker = self.get_by_id(worker_id)
        if not worker:
            worker = WorkerNode(
                id=worker_id,
                state=state,
                cpu_percent=cpu_percent,
                ram_used_mb=ram_used_mb,
                gpu_available=gpu_available,
            )
            self.db.add(worker)
        else:
            worker.hostname = hostname
            worker.state = state
            worker.cpu_percent = cpu_percent
            worker.ram_used_mb = ram_used_mb
            worker.gpu_available = gpu_available

        self.db.commit()
        self.db.refresh(worker)
        return worker
