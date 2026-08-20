"""
Worker Process Execution Engine — AutoRoll Phase 14
Orchestrates local ML inference pipeline tasks (SCRFD, Alignment, MiniFASNet, ArcFace, Vector Matcher, Decision Engine)
with bounded frame queues (max_queue_size = 2).
"""

import asyncio
import queue
import time
from typing import Any, Dict, List, Optional
from app.core.logger import get_logger
from app.workers.models import WorkerStatus, WorkerNodeInfo
from app.workers.worker_metrics import WorkerMetricsCollector

logger = get_logger("worker_engine")


class FrameQueueItem:
    def __init__(self, camera_id: str, frame: Any, timestamp: float):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp


class BoundedFrameQueue:
    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._queue = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0

    def put(self, item: FrameQueueItem):
        """Put frame into bounded queue. Drops oldest frame if queue full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        self._queue.put(item)

    def get(self, timeout: float = 0.1) -> Optional[FrameQueueItem]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        return self._queue.qsize()


class GPUInferenceWorker:
    def __init__(
        self,
        worker_id: str = "worker-001",
        device: str = "cuda",
        max_queue_size: int = 2,
    ):
        self.worker_id = worker_id
        self.device = device
        self.max_queue_size = max_queue_size
        self.status = WorkerStatus.STARTING

        # Shared ML pipeline components (Lazy loaded or passed in)
        self.frame_queue = BoundedFrameQueue(max_size=max_queue_size)
        self.active_cameras: List[str] = []
        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.fps = 0.0

        self._running = False
        self._verify_gpu_device()

    def _verify_gpu_device(self):
        """Explicit GPU failure check if CUDA configured."""
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.error(f"CUDA device requested for worker '{self.worker_id}', but CUDA is unavailable!")
                    self.status = WorkerStatus.ERROR
                    raise RuntimeError("CUDA device requested but unavailable on worker node.")
                logger.info(f"Worker '{self.worker_id}' GPU verified: {torch.cuda.get_device_name(0)}")
            except ImportError:
                logger.warning(f"PyTorch not found for CUDA check. Running in system mode.")

    def add_camera(self, camera_id: str):
        if camera_id not in self.active_cameras:
            self.active_cameras.append(camera_id)
            logger.info(f"Worker '{self.worker_id}' attached to camera stream '{camera_id}'.")

    def remove_camera(self, camera_id: str):
        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
            logger.info(f"Worker '{self.worker_id}' detached from camera stream '{camera_id}'.")

    def process_frame_pipeline(self, item: FrameQueueItem) -> Dict[str, Any]:
        """
        Simulate/Execute bounded frame pipeline:
        Frame -> SCRFD -> Alignment -> MiniFASNet -> ArcFace -> L2 Norm -> Vector Match -> Decision Engine.
        """
        start = time.time()
        # Pipeline execution time
        time.sleep(0.015)
        latency_ms = (time.time() - start) * 1000.0

        self.inference_count += 1
        self.total_latency_ms += latency_ms
        self.p95_latency_ms = max(self.p95_latency_ms, latency_ms)

        return {
            "worker_id": self.worker_id,
            "camera_id": item.camera_id,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        metrics = WorkerMetricsCollector.get_system_metrics()
        avg_lat = (self.total_latency_ms / max(1, self.inference_count))
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "active_cameras": self.active_cameras,
            "queue_depth": self.frame_queue.size(),
            "dropped_frames": self.frame_queue.dropped_frames,
            "inference_fps": self.fps,
            "average_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "gpu_name": metrics["gpu_name"],
            "gpu_utilization": metrics["gpu_utilization"],
            "gpu_memory_used": metrics["gpu_memory_used"],
            "gpu_memory_total": metrics["gpu_memory_total"],
        }
