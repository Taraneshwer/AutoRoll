"""
AutoRoll Worker Service Orchestrator.
Manages Worker Lifecycle, ML Pipeline, RTSP Streams, Heartbeats, and Event Dispatching.
"""

import threading
import time
from typing import Any

import cv2
import numpy as np

from autoroll.common.config import get_settings
from autoroll.common.logger import get_logger
from autoroll.ml.inference.pipeline import UnifiedInferencePipeline
from worker.api_client import WorkerAPIClient
from worker.config import WorkerSettings, get_worker_settings
from worker.rtsp_client import RTSPStreamClient
from worker.state import WorkerHealthMetrics, WorkerState
from worker.system_info import SystemInfoMonitor

logger = get_logger("worker_service")
settings = get_settings()


class WorkerService:
    """
    Independent Worker Process Service.
    Loads ML Pipeline, manages RTSP camera streams, sends heartbeats,
    and dispatches recognition events.
    """

    def __init__(self, config: WorkerSettings | None = None):
        self.config = config or get_worker_settings()
        self.worker_id = self.config.WORKER_ID
        self.state = WorkerState.STARTING

        logger.info(f"Initializing AutoRoll Worker Service '{self.worker_id}'...")

        # Initialize ML Pipeline
        self.pipeline = UnifiedInferencePipeline(
            device=self.config.DEVICE,
            recognition_interval=self.config.RECOGNITION_INTERVAL_FRAMES,
        )

        # Server API Client
        self.api_client = WorkerAPIClient(server_url=self.config.SERVER_URL)

        # Active RTSP Stream Clients
        self.camera_clients: dict[str, RTSPStreamClient] = {}
        self.camera_urls: dict[str, str] = {}

        # Heartbeat Thread
        self.running = False
        self.heartbeat_thread: threading.Thread | None = None

        self.state = WorkerState.READY
        logger.info(f"Worker '{self.worker_id}' initialized and READY.")

    def start(self) -> None:
        """
        Starts the worker service, registers with server, and begins heartbeat loop.
        """
        self.running = True

        # Initial Server Registration
        metrics = self.collect_health_metrics()
        self.api_client.register_worker(metrics)

        # Start Heartbeat Thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name=f"heartbeat_{self.worker_id}", daemon=True
        )
        self.heartbeat_thread.start()
        logger.info(f"Worker '{self.worker_id}' heartbeat loop started.")

    def collect_health_metrics(self) -> WorkerHealthMetrics:
        sys_metrics = SystemInfoMonitor.get_cpu_ram_metrics()
        gpu_metrics = SystemInfoMonitor.get_gpu_metrics()
        avg_lats = self.pipeline.perf_tracker.get_avg_latencies()

        active_cam_ids = list(self.camera_clients.keys())
        active_count = len(active_cam_ids)

        # State resolution based on camera activity
        if self.state not in (WorkerState.STOPPING, WorkerState.OFFLINE):
            self.state = WorkerState.BUSY if active_count > 0 else WorkerState.READY

        return WorkerHealthMetrics(
            worker_id=self.worker_id,
            state=self.state,
            cpu_percent=sys_metrics["cpu_percent"],
            ram_used_mb=sys_metrics["ram_used_mb"],
            ram_percent=sys_metrics["ram_percent"],
            gpu_available=gpu_metrics["gpu_available"],
            gpu_name=gpu_metrics["gpu_name"],
            gpu_utilization_percent=gpu_metrics["gpu_utilization_percent"],
            gpu_memory_used_mb=gpu_metrics["gpu_memory_used_mb"],
            active_cameras_count=active_count,
            active_camera_ids=active_cam_ids,
            fps=self.pipeline.perf_tracker.record_frame(0, 0, 0, 0),
            queue_backlog=0,
            avg_inference_latency_ms=avg_lats["avg_total_ms"],
            model_versions={
                "scrfd": settings.MODEL_VERSION,
                "arcface": settings.MODEL_VERSION,
                "liveness": settings.PAD_MODEL_VERSION,
            },
        )

    def _heartbeat_loop(self) -> None:
        while self.running:
            try:
                metrics = self.collect_health_metrics()
                resp = self.api_client.send_heartbeat(metrics)
                if resp and "assigned_cameras" in resp:
                    self._sync_camera_assignments(resp["assigned_cameras"])
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

            time.sleep(self.config.HEARTBEAT_INTERVAL_SEC)

    def assign_camera(self, camera_id: str, rtsp_url: str) -> None:
        """
        Assigns an RTSP camera stream to this worker.
        """
        if camera_id in self.camera_clients:
            logger.info(f"Camera '{camera_id}' is already assigned.")
            return

        client = RTSPStreamClient(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            frame_callback=self._on_frame_received,
            reconnect_interval_sec=self.config.RTSP_RECONNECT_INTERVAL_SEC,
        )
        self.camera_clients[camera_id] = client
        self.camera_urls[camera_id] = rtsp_url
        client.start()
        logger.info(f"Camera '{camera_id}' assigned and stream started.")

    def unassign_camera(self, camera_id: str) -> None:
        """
        Releases stream resources cleanly when camera is unassigned.
        """
        if camera_id in self.camera_clients:
            client = self.camera_clients.pop(camera_id)
            self.camera_urls.pop(camera_id, None)
            client.stop()
            logger.info(f"Camera '{camera_id}' unassigned and resources released.")

    def _sync_camera_assignments(self, assigned_cameras: list[dict[str, str]]) -> None:
        current_assigned = {cam["camera_id"]: cam["rtsp_url"] for cam in assigned_cameras}

        # Unassign cameras no longer assigned to this worker
        to_unassign = [cid for cid in self.camera_clients if cid not in current_assigned]
        for cid in to_unassign:
            self.unassign_camera(cid)

        # Assign new cameras
        for cid, url in current_assigned.items():
            if cid not in self.camera_clients:
                self.assign_camera(cid, url)

    def _on_frame_received(self, camera_id: str, frame: np.ndarray, frame_index: int) -> None:
        """
        Processes decoded frame locally and dispatches recognition events to central server.
        Never stores permanent attendance data locally.
        """
        frame_result = self.pipeline.process_frame(frame, frame_index=frame_index)

        # Dispatch event for verified live faces
        for face in frame_result.faces:
            if face.is_live and face.embedding is not None:
                event_payload = {
                    "worker_id": self.worker_id,
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "timestamp_ms": frame_result.timestamp_ms,
                    "track_id": face.track_id,
                    "embedding": face.embedding,
                    "liveness_score": face.liveness_score,
                    "liveness_decision": face.liveness_decision,
                    "detection_confidence": face.detection_confidence,
                }
                self.api_client.publish_recognition_event(event_payload)

    def process_local_video_test(self, video_path: str) -> list[dict[str, Any]]:
        """
        Local worker test mode using prerecorded video or image file without server dependency.
        """
        logger.info(f"Running Worker Local Test Mode on file '{video_path}'...")

        ext = video_path.split(".")[-1].lower() if "." in video_path else ""
        events_recorded: list[dict[str, Any]] = []

        if ext in ("jpg", "jpeg", "png", "bmp"):
            img = cv2.imread(video_path)
            if img is None:
                raise FileNotFoundError(f"Could not open image file at '{video_path}'")
            res = self.pipeline.process_frame(img, frame_index=1)
            for face in res.faces:
                events_recorded.append(
                    {
                        "frame_index": 1,
                        "track_id": face.track_id,
                        "liveness_score": face.liveness_score,
                        "recognition_status": face.recognition_status,
                    }
                )
            logger.info(f"Local Test Mode complete: Processed image '{video_path}' successfully.")
            return events_recorded

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at '{video_path}'")

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            res = self.pipeline.process_frame(frame, frame_index=frame_idx)
            for face in res.faces:
                if face.is_live:
                    events_recorded.append(
                        {
                            "frame_index": frame_idx,
                            "track_id": face.track_id,
                            "liveness_score": face.liveness_score,
                            "recognition_status": face.recognition_status,
                        }
                    )

        cap.release()
        logger.info(
            f"Local Test Mode complete: Processed {frame_idx} frames, "
            f"detected {len(events_recorded)} live face events."
        )
        return events_recorded

    def stop(self) -> None:
        """
        Gracefully shuts down worker service, releases all cameras, and sets state to OFFLINE.
        """
        logger.info(f"Shutting down Worker Service '{self.worker_id}'...")
        self.state = WorkerState.STOPPING
        self.running = False

        # Release all camera streams
        active_ids = list(self.camera_clients.keys())
        for cid in active_ids:
            self.unassign_camera(cid)

        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)

        self.api_client.close()
        self.state = WorkerState.OFFLINE
        logger.info(f"Worker Service '{self.worker_id}' is OFFLINE. Shutdown complete.")
