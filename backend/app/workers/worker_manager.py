"""
Worker Manager — AutoRoll Phase 14
Manages worker lifecycles, single-machine local mode auto-launch, and cluster process controls.
"""

from typing import Dict, Optional
from app.core.config import get_settings
from app.core.logger import get_logger
from app.workers.worker_registry import WorkerRegistry
from app.workers.worker_health import WorkerHealthMonitor
from app.workers.worker_scheduler import WorkerScheduler
from app.workers.models import WorkerRegistrationRequest, WorkerStatus

logger = get_logger("worker_manager")
settings = get_settings()


class WorkerManager:
    def __init__(self):
        self.registry = WorkerRegistry()
        self.health_monitor = WorkerHealthMonitor(self.registry)
        self.scheduler = WorkerScheduler(self.registry)
        self.mode = getattr(settings, "AUTOROLL_WORKER_MODE", "local")

        # Single-machine mode compatibility: auto-register local worker if in local mode
        if self.mode == "local":
            self.ensure_local_worker()

    def ensure_local_worker(self):
        """Auto-register default local worker node for single-machine deployment."""
        if not self.registry.get("local-worker-01"):
            req = WorkerRegistrationRequest(
                worker_id="local-worker-01",
                hostname="localhost",
                ip_address="127.0.0.1",
                secret=getattr(settings, "AUTOROLL_WORKER_SECRET", "autoroll_secret_2026"),
                gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
                gpu_memory_total=8151.0,
                model_id="autoroll_v1",
                model_version="autoroll_arcface_r50_epoch1",
                embedding_dimension=512,
                max_camera_capacity=4,
            )
            self.registry.register(req)
            logger.info("Single-machine mode: Default local worker 'local-worker-01' registered automatically.")

    def run_health_checks(self):
        """Run health check sweep and trigger camera failovers if needed."""
        changes = self.health_monitor.check_health()
        for worker, old_s, new_s in changes:
            if new_s == WorkerStatus.OFFLINE:
                self.scheduler.handle_failover(worker.worker_id)
        return len(changes)


# Singleton Instance
worker_manager = WorkerManager()
