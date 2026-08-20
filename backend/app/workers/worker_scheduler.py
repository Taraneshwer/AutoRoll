"""
Worker Scheduler & Failover Manager — AutoRoll Phase 14
Handles camera assignment, affinity, graceful draining, and automatic failovers.
"""

import time
from typing import Dict, List, Optional
from app.core.logger import get_logger
from app.workers.models import WorkerNodeInfo, WorkerStatus
from app.workers.load_balancer import WorkerLoadBalancer
from app.workers.worker_registry import WorkerRegistry
from app.workers.worker_protocol import WorkerFailoverEvent, WorkerEventType

logger = get_logger("worker_scheduler")


class WorkerScheduler:
    def __init__(
        self,
        registry: WorkerRegistry,
        load_balancer: Optional[WorkerLoadBalancer] = None,
    ):
        self.registry = registry
        self.load_balancer = load_balancer or WorkerLoadBalancer()
        self.camera_assignments: Dict[str, str] = {}  # camera_id -> worker_id
        self.failover_events: List[WorkerFailoverEvent] = []

    def assign_camera(self, camera_id: str, worker_id: Optional[str] = None) -> Optional[str]:
        """
        Assign camera to specific worker or calculate optimal worker via load balancer.
        Maintains camera affinity unless worker is offline or draining.
        """
        # Existent assignment check
        current_w_id = self.camera_assignments.get(camera_id)
        if current_w_id and not worker_id:
            current_worker = self.registry.get(current_w_id)
            if current_worker and current_worker.status == WorkerStatus.ONLINE:
                logger.info(f"Camera '{camera_id}' retained on current worker '{current_w_id}' (Affinity).")
                return current_w_id

        workers_map = {w.worker_id: w for w in self.registry.list_all()}

        if worker_id:
            target_worker = workers_map.get(worker_id)
            if not target_worker or target_worker.status not in (WorkerStatus.ONLINE, WorkerStatus.STARTING):
                logger.error(f"Cannot assign camera '{camera_id}': Specified worker '{worker_id}' is not online.")
                return None
            chosen_worker = target_worker
        else:
            chosen_worker = self.load_balancer.select_best_worker(workers_map)

        if not chosen_worker:
            logger.warning(f"No available online workers to assign camera '{camera_id}'.")
            return None

        # Unassign from old worker if migrating
        if current_w_id and current_w_id != chosen_worker.worker_id:
            old_worker = self.registry.get(current_w_id)
            if old_worker and camera_id in old_worker.assigned_cameras:
                old_worker.assigned_cameras.remove(camera_id)

        if camera_id not in chosen_worker.assigned_cameras:
            chosen_worker.assigned_cameras.append(camera_id)

        self.camera_assignments[camera_id] = chosen_worker.worker_id
        logger.info(f"Assigned Camera '{camera_id}' to Worker '{chosen_worker.worker_id}'.")
        return chosen_worker.worker_id

    def unassign_camera(self, camera_id: str) -> bool:
        w_id = self.camera_assignments.pop(camera_id, None)
        if w_id:
            worker = self.registry.get(w_id)
            if worker and camera_id in worker.assigned_cameras:
                worker.assigned_cameras.remove(camera_id)
            logger.info(f"Unassigned Camera '{camera_id}' from Worker '{w_id}'.")
            return True
        return False

    def handle_failover(self, offline_worker_id: str) -> List[WorkerFailoverEvent]:
        """
        Reassign all cameras from an offline worker to healthy online workers.
        Emits WORKER_FAILOVER events for audit logs.
        """
        worker = self.registry.get(offline_worker_id)
        if not worker:
            return []

        worker.status = WorkerStatus.OFFLINE
        affected_cameras = list(worker.assigned_cameras)
        worker.assigned_cameras.clear()
        failovers = []


        start_time = time.time()
        for camera_id in affected_cameras:
            self.camera_assignments.pop(camera_id, None)
            new_w_id = self.assign_camera(camera_id)

            if new_w_id:
                latency_ms = (time.time() - start_time) * 1000.0
                event = WorkerFailoverEvent(
                    camera_id=camera_id,
                    old_worker_id=offline_worker_id,
                    new_worker_id=new_w_id,
                    reason=f"Worker '{offline_worker_id}' heartbeat timeout (OFFLINE)",
                    failover_latency_ms=round(latency_ms, 2),
                )
                failovers.append(event)
                self.failover_events.append(event)
                logger.error(
                    f"FAILOVER: Camera '{camera_id}' migrated from '{offline_worker_id}' -> '{new_w_id}' in {latency_ms:.1f}ms."
                )

        return failovers

    def drain_worker(self, worker_id: str) -> bool:
        """
        Gracefully drain a worker node:
        1. Set status to DRAINING.
        2. Reassign cameras gradually to other workers.
        3. Clear assigned camera list when drained.
        """
        worker = self.registry.get(worker_id)
        if not worker:
            return False

        worker.status = WorkerStatus.DRAINING
        logger.info(f"Worker '{worker_id}' entering DRAINING mode.")

        cams = list(worker.assigned_cameras)
        for camera_id in cams:
            self.camera_assignments.pop(camera_id, None)
            new_w_id = self.assign_camera(camera_id)
            if new_w_id:
                worker.assigned_cameras.remove(camera_id)

        if len(worker.assigned_cameras) == 0:
            logger.info(f"Worker '{worker_id}' fully drained of all camera workloads.")

        return True
