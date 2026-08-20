"""
Worker Registry — AutoRoll Phase 14
Maintains in-memory registry of worker nodes connected to the central control plane.
"""

import time
from typing import Dict, List, Optional
from app.core.logger import get_logger
from app.workers.models import WorkerNodeInfo, WorkerRegistrationRequest, WorkerStatus

logger = get_logger("worker_registry")


class WorkerRegistry:
    def __init__(self):
        self._workers: Dict[str, WorkerNodeInfo] = {}

    def register(self, req: WorkerRegistrationRequest) -> WorkerNodeInfo:
        """Register or update a worker node."""
        node = WorkerNodeInfo(
            worker_id=req.worker_id,
            hostname=req.hostname,
            ip_address=req.ip_address,
            port=req.port,
            status=WorkerStatus.ONLINE,
            gpu_name=req.gpu_name,
            gpu_memory_total=req.gpu_memory_total,
            model_id=req.model_id,
            model_version=req.model_version,
            embedding_dimension=req.embedding_dimension,
            max_camera_capacity=req.max_camera_capacity,
            last_heartbeat=time.time(),
        )

        # Preserve existing camera assignments if re-registering
        if req.worker_id in self._workers:
            node.assigned_cameras = self._workers[req.worker_id].assigned_cameras

        self._workers[req.worker_id] = node
        logger.info(f"Worker '{req.worker_id}' registered successfully on host '{req.hostname}'.")
        return node

    def get(self, worker_id: str) -> Optional[WorkerNodeInfo]:
        return self._workers.get(worker_id)

    def list_all(self) -> List[WorkerNodeInfo]:
        return list(self._workers.values())

    def update_heartbeat(
        self,
        worker_id: str,
        status: WorkerStatus = WorkerStatus.ONLINE,
        active_cameras: Optional[List[str]] = None,
        queue_depth: int = 0,
        inference_fps: float = 0.0,
        average_latency_ms: float = 0.0,
        p95_latency_ms: float = 0.0,
        gpu_utilization: float = 0.0,
        gpu_memory_used: float = 0.0,
    ) -> bool:
        if worker_id not in self._workers:
            return False

        worker = self._workers[worker_id]
        worker.last_heartbeat = time.time()
        worker.status = status
        if active_cameras is not None:
            worker.assigned_cameras = active_cameras
        worker.queue_depth = queue_depth
        worker.inference_fps = inference_fps
        worker.average_latency_ms = average_latency_ms
        worker.p95_latency_ms = p95_latency_ms
        worker.gpu_utilization = gpu_utilization
        worker.gpu_memory_used = gpu_memory_used
        return True

    def remove(self, worker_id: str) -> Optional[WorkerNodeInfo]:
        return self._workers.pop(worker_id, None)
