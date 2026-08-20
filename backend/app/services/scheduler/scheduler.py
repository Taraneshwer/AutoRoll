"""
AutoRoll Distributed Camera Scheduler Engine.
Orchestrates worker state tracking, camera assignment, load rebalancing, and failover.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.models import Camera, WorkerNode
from app.services.scheduler.capacity import WorkerCapacityCalculator

logger = get_logger("distributed_camera_scheduler")


class DistributedCameraScheduler:
    """
    Distributed Camera Scheduler for scaling camera processing across ML worker nodes.
    """

    def __init__(self, max_cameras_per_worker: int = 4, heartbeat_timeout_sec: float = 15.0):
        self.capacity_calculator = WorkerCapacityCalculator(
            max_cameras_per_worker=max_cameras_per_worker
        )
        self.heartbeat_timeout_sec = heartbeat_timeout_sec

    def register_worker(self, worker_data: dict[str, Any], db: Session) -> WorkerNode:
        worker_id = worker_data["worker_id"]
        worker = db.query(WorkerNode).filter(WorkerNode.id == worker_id).first()

        if not worker:
            worker = WorkerNode(id=worker_id)
            db.add(worker)

        worker.state = worker_data.get("state", "READY")
        worker.cpu_percent = worker_data.get("cpu_percent", 0.0)
        worker.ram_used_mb = worker_data.get("ram_used_mb", 0.0)
        worker.ram_percent = worker_data.get("ram_percent", 0.0)
        worker.gpu_available = worker_data.get("gpu_available", False)
        worker.gpu_name = worker_data.get("gpu_name")
        worker.gpu_utilization_percent = worker_data.get("gpu_utilization_percent")
        worker.gpu_memory_used_mb = worker_data.get("gpu_memory_used_mb")
        worker.fps = worker_data.get("fps", 0.0)
        worker.avg_inference_latency_ms = worker_data.get("avg_inference_latency_ms", 0.0)
        worker.last_heartbeat_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(worker)
        logger.info(f"Worker '{worker_id}' state updated in scheduler (State: {worker.state}).")
        return worker

    def update_heartbeat(self, worker_data: dict[str, Any], db: Session) -> dict[str, Any]:
        worker = self.register_worker(worker_data, db)

        # Check worker timeouts to trigger failovers if necessary
        self.check_worker_timeouts(db)

        # Fetch cameras currently assigned to this worker
        assigned_cams = (
            db.query(Camera)
            .filter(Camera.assigned_worker_id == worker.id, Camera.is_active.is_(True))
            .all()
        )

        return {
            "worker_id": worker.id,
            "status": "acknowledged",
            "assigned_cameras": [
                {"camera_id": c.id, "rtsp_url": c.rtsp_url, "name": c.name} for c in assigned_cams
            ],
        }

    def assign_camera(
        self, camera_id: str, worker_id: str | None = None, db: Session | None = None
    ) -> str | None:
        if db is None:
            raise ValueError("Database session required for camera assignment.")

        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            logger.error(f"Camera '{camera_id}' not found.")
            return None

        # Prevent duplicate assignment if camera is already assigned to the same worker
        if worker_id and camera.assigned_worker_id == worker_id:
            logger.info(f"Camera '{camera_id}' is already assigned to worker '{worker_id}'.")
            return worker_id

        # Target worker selection
        target_worker_id = worker_id
        if not target_worker_id:
            active_workers = db.query(WorkerNode).filter(WorkerNode.state != "OFFLINE").all()
            workers_info = []
            for w in active_workers:
                cams_count = (
                    db.query(Camera)
                    .filter(Camera.assigned_worker_id == w.id, Camera.is_active.is_(True))
                    .count()
                )
                workers_info.append(
                    {
                        "worker_id": w.id,
                        "state": w.state,
                        "cpu_percent": w.cpu_percent,
                        "active_cameras_count": cams_count,
                        "avg_inference_latency_ms": w.avg_inference_latency_ms,
                        "gpu_utilization_percent": w.gpu_utilization_percent,
                    }
                )

            best_w = self.capacity_calculator.select_best_worker(workers_info)
            if not best_w:
                logger.warning(f"No available capacity to assign camera '{camera_id}'.")
                return None
            target_worker_id = best_w["worker_id"]

        # Enforce rule: Each camera has at most ONE active processing worker
        camera.assigned_worker_id = target_worker_id
        db.commit()
        logger.info(f"Camera '{camera_id}' assigned to worker '{target_worker_id}'.")
        return target_worker_id

    def unassign_camera(self, camera_id: str, db: Session) -> bool:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera or not camera.assigned_worker_id:
            return False

        old_worker = camera.assigned_worker_id
        camera.assigned_worker_id = None
        db.commit()
        logger.info(f"Camera '{camera_id}' unassigned from app.workers '{old_worker}'.")
        return True

    def rebalance_workload(self, db: Session) -> dict[str, Any]:
        """
        Automatically distributes unassigned cameras across active healthy workers.
        """
        self.check_worker_timeouts(db)

        unassigned_cams = (
            db.query(Camera)
            .filter(Camera.assigned_worker_id.is_(None), Camera.is_active.is_(True))
            .all()
        )

        reassigned_count = 0
        for cam in unassigned_cams:
            assigned_id = self.assign_camera(cam.id, db=db)
            if assigned_id:
                reassigned_count += 1

        logger.info(
            f"Workload rebalance complete: Reassigned {reassigned_count}/"
            f"{len(unassigned_cams)} cameras."
        )
        return {
            "unassigned_cameras_found": len(unassigned_cams),
            "reassigned_count": reassigned_count,
        }

    def check_worker_timeouts(self, db: Session) -> list[str]:
        """
        Detects worker heartbeat timeouts, marks workers OFFLINE, and reassigns their cameras.
        """
        now = datetime.now(timezone.utc)
        active_workers = db.query(WorkerNode).filter(WorkerNode.state != "OFFLINE").all()

        dead_workers: list[str] = []
        for w in active_workers:
            if w.last_heartbeat_at:
                age_sec = (now - w.last_heartbeat_at.replace(tzinfo=timezone.utc)).total_seconds()
                if age_sec > self.heartbeat_timeout_sec:
                    dead_workers.append(w.id)
                    w.state = "OFFLINE"
                    logger.warning(
                        f"Worker '{w.id}' missed heartbeat ({age_sec:.1f}s ago). Marked OFFLINE."
                    )

        if dead_workers:
            db.commit()
            # Reassign cameras owned by dead workers
            orphaned_cams = (
                db.query(Camera)
                .filter(Camera.assigned_worker_id.in_(dead_workers), Camera.is_active.is_(True))
                .all()
            )
            for cam in orphaned_cams:
                cam.assigned_worker_id = None
            db.commit()

            # Trigger automatic rebalance to place orphaned cameras on healthy workers
            self.rebalance_workload(db)

        return dead_workers

    def get_scheduler_status(self, db: Session) -> dict[str, Any]:
        """
        Exposes full camera distribution matrix and worker load status.
        """
        self.check_worker_timeouts(db)

        workers = db.query(WorkerNode).all()
        cameras = db.query(Camera).all()

        worker_matrix = []
        for w in workers:
            cams = [c.id for c in cameras if c.assigned_worker_id == w.id]
            worker_matrix.append(
                {
                    "worker_id": w.id,
                    "state": w.state,
                    "cpu_percent": w.cpu_percent,
                    "ram_used_mb": w.ram_used_mb,
                    "gpu_available": w.gpu_available,
                    "gpu_name": w.gpu_name,
                    "fps": w.fps,
                    "avg_inference_latency_ms": w.avg_inference_latency_ms,
                    "assigned_cameras_count": len(cams),
                    "assigned_camera_ids": cams,
                    "last_heartbeat_at": (
                        w.last_heartbeat_at.isoformat() if w.last_heartbeat_at else None
                    ),
                }
            )

        unassigned = [c.id for c in cameras if not c.assigned_worker_id and c.is_active]

        return {
            "total_workers": len(workers),
            "online_workers": sum(1 for w in workers if w.state != "OFFLINE"),
            "total_cameras": len(cameras),
            "unassigned_cameras": len(unassigned),
            "unassigned_camera_ids": unassigned,
            "workers": worker_matrix,
        }
