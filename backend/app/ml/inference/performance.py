"""
Performance Measurement & Latency/FPS Tracker Module.
"""

import time

import numpy as np


class PerformanceTracker:
    """
    Tracks rolling latencies and FPS across video stream inference frames.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: list[float] = []
        self.detection_latencies: list[float] = []
        self.recognition_latencies: list[float] = []
        self.liveness_latencies: list[float] = []
        self.total_latencies: list[float] = []

    def record_frame(
        self,
        det_latency: float,
        rec_latency: float,
        live_latency: float,
        total_latency: float,
    ) -> float:
        """
        Records frame component latencies and returns current average FPS.
        """
        now = time.perf_counter()
        self.frame_times.append(now)

        self.detection_latencies.append(det_latency)
        self.recognition_latencies.append(rec_latency)
        self.liveness_latencies.append(live_latency)
        self.total_latencies.append(total_latency)

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            self.detection_latencies.pop(0)
            self.recognition_latencies.pop(0)
            self.liveness_latencies.pop(0)
            self.total_latencies.pop(0)

        if len(self.frame_times) >= 2:
            time_elapsed = self.frame_times[-1] - self.frame_times[0]
            fps = float((len(self.frame_times) - 1) / max(1e-5, time_elapsed))
        else:
            fps = float(1000.0 / max(1e-5, total_latency)) if total_latency > 0 else 0.0

        return round(fps, 2)

    def get_avg_latencies(self) -> dict[str, float]:
        return {
            "avg_det_ms": round(float(np.mean(self.detection_latencies or [0.0])), 2),
            "avg_rec_ms": round(float(np.mean(self.recognition_latencies or [0.0])), 2),
            "avg_live_ms": round(float(np.mean(self.liveness_latencies or [0.0])), 2),
            "avg_total_ms": round(float(np.mean(self.total_latencies or [0.0])), 2),
        }
