"""
Worker Capacity Calculator for Distributed Camera Workload Placement.
Evaluates CPU, GPU, Active Cameras, and Inference Latency.
"""

from typing import Any

from app.core.logger import get_logger

logger = get_logger("worker_capacity_calculator")


class WorkerCapacityCalculator:
    """
    Evaluates worker health metrics to compute capacity scores for optimal camera placement.
    Lower load score = higher priority for receiving new camera streams.
    """

    def __init__(
        self,
        max_cameras_per_worker: int = 4,
        max_cpu_percent: float = 90.0,
    ):
        self.max_cameras_per_worker = max_cameras_per_worker
        self.max_cpu_percent = max_cpu_percent

    def calculate_load_score(
        self,
        active_cameras_count: int,
        cpu_percent: float,
        avg_latency_ms: float,
        gpu_utilization: float | None = None,
    ) -> float:
        """
        Calculates composite load score for worker capacity ranking.
        """
        gpu_penalty = (gpu_utilization * 0.5) if gpu_utilization is not None else 0.0
        score = (
            (active_cameras_count * 100.0)
            + cpu_percent
            + (0.5 * avg_latency_ms)
            + gpu_penalty
        )
        return round(float(score), 2)

    def select_best_worker(
        self,
        workers_info: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Filters eligible workers and selects worker with lowest load score.
        """
        eligible: list[tuple[dict[str, Any], float]] = []

        for w in workers_info:
            state = w.get("state", "OFFLINE").upper()
            if state in ("OFFLINE", "STOPPING", "DEGRADED"):
                continue

            active_cams = w.get("active_cameras_count", 0)
            if active_cams >= self.max_cameras_per_worker:
                continue

            cpu = w.get("cpu_percent", 0.0)
            if cpu >= self.max_cpu_percent:
                continue

            latency = w.get("avg_inference_latency_ms", 0.0)
            gpu_util = w.get("gpu_utilization_percent")

            score = self.calculate_load_score(active_cams, cpu, latency, gpu_util)
            eligible.append((w, score))

        if not eligible:
            logger.warning("No eligible workers available with capacity for camera assignment.")
            return None

        # Sort by load score ascending (lowest load score first)
        eligible.sort(key=lambda x: x[1])
        selected_worker = eligible[0][0]
        logger.info(
            f"Selected Worker '{selected_worker['worker_id']}' (Load Score: {eligible[0][1]}) "
            f"for camera placement."
        )
        return selected_worker
