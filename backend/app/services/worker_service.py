"""
Worker Service managing ML Worker node registration, heartbeats, load-aware camera assignment,
failover camera reassignment, model compatibility enforcement, and attendance event deduplication.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("worker_service")
settings = get_settings()


class WorkerRegistrationRequest(BaseModel):
    worker_id: str
    hostname: str
    secret: str
    gpu_name: str = "CPU"
    vram_used_mb: float = 0.0
    model_id: str
    model_version: str
    embedding_dimension: int
    threshold: float


class CameraRegistrationRequest(BaseModel):
    camera_id: str
    camera_name: str
    stream_url: str


class RecognitionEvent(BaseModel):
    worker_id: str
    camera_id: str
    timestamp: float
    student_id: str
    similarity: float
    liveness_score: float
    decision: str
    processing_latency_ms: float


class WorkerNodeState:
    def __init__(
        self,
        worker_id: str,
        hostname: str,
        gpu_name: str,
        model_id: str,
        model_version: str,
        embedding_dimension: int,
        threshold: float,
    ):
        self.worker_id = worker_id
        self.hostname = hostname
        self.gpu_name = gpu_name
        self.model_id = model_id
        self.model_version = model_version
        self.embedding_dimension = embedding_dimension
        self.threshold = threshold
        self.status = "ONLINE"
        self.assigned_cameras: List[str] = []
        self.queue_depth = 0
        self.recent_latency_ms = 0.0
        self.gpu_utilization = 0.0
        self.last_heartbeat = time.time()

    def get_load_score(self) -> float:
        """Calculate load score for deterministic scheduling."""
        return (len(self.assigned_cameras) * 2.0) + self.queue_depth + (self.recent_latency_ms / 10.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "status": self.status,
            "gpu_name": self.gpu_name,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "embedding_dimension": self.embedding_dimension,
            "threshold": self.threshold,
            "assigned_cameras": self.assigned_cameras,
            "load_score": round(self.get_load_score(), 2),
            "queue_depth": self.queue_depth,
            "recent_latency_ms": round(self.recent_latency_ms, 2),
            "last_heartbeat": round(self.last_heartbeat, 2),
        }


class CentralWorkerControlPlane:
    def __init__(self, worker_repo=None):
        self.worker_repo = worker_repo
        self.workers: Dict[str, WorkerNodeState] = {}
        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.dedup_events: Dict[str, float] = {}  # student_id -> last_timestamp

    def register_worker(
        self,
        worker_id: str | WorkerRegistrationRequest = None,
        hostname: str = "localhost",
        secret: str = "autoroll_secret_2026",
        cpu_percent: float = 0.0,
        ram_used_mb: float = 0.0,
        gpu_available: bool = False,
        gpu_name: str = "CPU",
        model_id: str = "autoroll_v1",
        model_version: str = "autoroll_arcface_r50_epoch1",
        embedding_dimension: int = 512,
        threshold: float = 0.0540,
    ) -> Any:
        if isinstance(worker_id, WorkerRegistrationRequest):
            req = worker_id
        else:
            req = WorkerRegistrationRequest(
                worker_id=worker_id or "worker-01",
                hostname=hostname,
                secret=secret,
                gpu_name=gpu_name if gpu_available else "CPU",
                vram_used_mb=ram_used_mb,
                model_id=model_id,
                model_version=model_version,
                embedding_dimension=embedding_dimension,
                threshold=threshold,
            )

        expected_secret = getattr(settings, "AUTOROLL_WORKER_SECRET", "autoroll_secret_2026")
        if req.secret != expected_secret:
            logger.error(f"Worker registration rejected: Invalid secret for worker '{req.worker_id}'.")
            raise ValueError("Unauthorized: Invalid worker secret token.")

        # Model consistency guard
        if req.model_id != "autoroll_v1" or req.embedding_dimension != 512 or req.threshold != 0.0540:
            logger.error(f"Worker registration rejected: Incompatible model configuration for '{req.worker_id}'.")
            raise ValueError("Incompatible worker model configuration. Require model_id='autoroll_v1', dim=512, threshold=0.0540.")

        node = WorkerNodeState(
            worker_id=req.worker_id,
            hostname=req.hostname,
            gpu_name=req.gpu_name,
            model_id=req.model_id,
            model_version=req.model_version,
            embedding_dimension=req.embedding_dimension,
            threshold=req.threshold,
        )
        node.id = req.worker_id
        node.state = "READY"
        self.workers[req.worker_id] = node

        if self.worker_repo is not None:
            db_worker = self.worker_repo.create_or_update(
                worker_id=req.worker_id,
                hostname=req.hostname,
                state="READY",
                cpu_percent=0.0,
                ram_used_mb=req.vram_used_mb,
                gpu_available=True if req.gpu_name != "CPU" else False,
            )
            logger.info(f"Worker '{req.worker_id}' persisted to DB repository.")
            return db_worker


        logger.info(f"Worker '{req.worker_id}' registered successfully on host '{req.hostname}'.")
        return node



    def record_heartbeat(
        self,
        worker_id: str,
        queue_depth: int = 0,
        recent_latency_ms: float = 0.0,
        gpu_utilization: float = 0.0,
    ) -> bool:
        if worker_id not in self.workers:
            return False
        node = self.workers[worker_id]
        node.status = "ONLINE"
        node.queue_depth = queue_depth
        node.recent_latency_ms = recent_latency_ms
        node.gpu_utilization = gpu_utilization
        node.last_heartbeat = time.time()
        return True

    def register_camera(self, req: CameraRegistrationRequest) -> Dict[str, Any]:
        self.cameras[req.camera_id] = {
            "camera_id": req.camera_id,
            "camera_name": req.camera_name,
            "stream_url": req.stream_url,
            "worker_id": None,
            "status": "UNASSIGNED",
        }
        self.assign_camera_to_optimal_worker(req.camera_id)
        return self.cameras[req.camera_id]

    def assign_camera_to_optimal_worker(self, camera_id: str) -> Optional[str]:
        if camera_id not in self.cameras:
            return None

        online_workers = [w for w in self.workers.values() if w.status == "ONLINE"]
        if not online_workers:
            logger.warning(f"No online workers available to assign camera '{camera_id}'.")
            self.cameras[camera_id]["status"] = "UNASSIGNED"
            self.cameras[camera_id]["worker_id"] = None
            return None

        # Deterministic Load-Aware Scheduler
        optimal = min(online_workers, key=lambda w: w.get_load_score())
        
        # Remove from previous worker if assigned
        prev_w = self.cameras[camera_id].get("worker_id")
        if prev_w and prev_w in self.workers and camera_id in self.workers[prev_w].assigned_cameras:
            self.workers[prev_w].assigned_cameras.remove(camera_id)

        optimal.assigned_cameras.append(camera_id)
        self.cameras[camera_id]["worker_id"] = optimal.worker_id
        self.cameras[camera_id]["status"] = "ONLINE"
        logger.info(f"Assigned Camera '{camera_id}' to Worker '{optimal.worker_id}' (Load Score: {optimal.get_load_score():.2f}).")
        return optimal.worker_id

    def check_health_and_failover(self) -> List[str]:
        """Detect worker disconnects (> 15s) and reassign cameras automatically."""
        now = time.time()
        reassigned_cameras = []

        for w_id, w_node in list(self.workers.items()):
            if now - w_node.last_heartbeat > 15.0 and w_node.status != "OFFLINE":
                logger.error(f"Worker '{w_id}' heartbeat timeout (>15s). Marking OFFLINE.")
                w_node.status = "OFFLINE"
                affected_cams = list(w_node.assigned_cameras)
                w_node.assigned_cameras.clear()

                for cam_id in affected_cams:
                    new_w = self.assign_camera_to_optimal_worker(cam_id)
                    if new_w:
                        reassigned_cameras.append(cam_id)

        return reassigned_cameras

    def ingest_recognition_event(self, event: RecognitionEvent) -> Dict[str, Any]:
        """Ingest event with temporal deduplication (1500 ms window across cameras)."""
        now = time.time()
        last_seen = self.dedup_events.get(event.student_id, 0.0)

        if now - last_seen < 1.5:
            return {"status": "DUPLICATE_SUPPRESSED", "student_id": event.student_id}

        self.dedup_events[event.student_id] = now
        return {"status": "RECORDED", "student_id": event.student_id, "camera_id": event.camera_id}

    def list_workers(self) -> List[Dict[str, Any]]:
        self.check_health_and_failover()
        return [w.to_dict() for w in self.workers.values()]

    def list_cameras(self) -> List[Dict[str, Any]]:
        return list(self.cameras.values())


# Singleton instance for Central Control Plane
worker_control_plane = CentralWorkerControlPlane()
WorkerService = CentralWorkerControlPlane

