"""
Worker Load Balancer — AutoRoll Phase 14
Calculates worker load scores based on configurable weights:
load_score = 0.35 * gpu_utilization + 0.25 * queue_depth + 0.20 * camera_count + 0.20 * p95_latency
"""

from typing import Dict, List, Optional
from app.workers.models import WorkerNodeInfo, WorkerStatus


class WorkerLoadBalancer:
    def __init__(
        self,
        gpu_weight: float = 0.35,
        queue_weight: float = 0.25,
        camera_weight: float = 0.20,
        latency_weight: float = 0.20,
    ):
        self.gpu_weight = gpu_weight
        self.queue_weight = queue_weight
        self.camera_weight = camera_weight
        self.latency_weight = latency_weight

    def calculate_score(self, worker: WorkerNodeInfo) -> float:
        """
        Calculate worker load score (lower score = healthier, less loaded worker).
        """
        # Normalize metrics to [0, 1] range for score stability
        norm_gpu = min(1.0, max(0.0, worker.gpu_utilization / 100.0))
        norm_queue = min(1.0, max(0.0, worker.queue_depth / 5.0))
        norm_cameras = min(1.0, max(0.0, len(worker.assigned_cameras) / max(1, worker.max_camera_capacity)))
        norm_latency = min(1.0, max(0.0, worker.p95_latency_ms / 100.0))

        score = (
            self.gpu_weight * norm_gpu
            + self.queue_weight * norm_queue
            + self.camera_weight * norm_cameras
            + self.latency_weight * norm_latency
        )
        return round(score, 4)

    def select_best_worker(
        self,
        workers: Dict[str, WorkerNodeInfo],
        excluded_worker_ids: Optional[List[str]] = None,
    ) -> Optional[WorkerNodeInfo]:
        """
        Select healthy worker with lowest load score.
        Excludes workers that are OFFLINE, DRAINING, ERROR, or at maximum capacity.
        """
        excluded = set(excluded_worker_ids or [])
        eligible = [
            w for w in workers.values()
            if w.worker_id not in excluded
            and w.status in (WorkerStatus.ONLINE, WorkerStatus.STARTING)
            and len(w.assigned_cameras) < w.max_camera_capacity
        ]

        if not eligible:
            return None

        # Sort by calculated load score ascending
        eligible.sort(key=lambda w: self.calculate_score(w))
        return eligible[0]
