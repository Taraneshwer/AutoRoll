"""
Performance & System Metrics Tracker for AutoRoll ML Workers and Server.
"""

import time
from typing import Any


class MetricsTracker:
    """
    Lightweight sliding window latency and FPS counter.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies_ms: list[float] = []
        self.frame_timestamps: list[float] = []

    def record_frame(self, latency_ms: float) -> None:
        now = time.time()
        self.latencies_ms.append(latency_ms)
        self.frame_timestamps.append(now)

        if len(self.latencies_ms) > self.window_size:
            self.latencies_ms.pop(0)
            self.frame_timestamps.pop(0)

    def get_avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def get_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        time_elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if time_elapsed <= 0:
            return 0.0
        return (len(self.frame_timestamps) - 1) / time_elapsed

    def get_summary(self) -> dict[str, Any]:
        return {
            "avg_latency_ms": round(self.get_avg_latency_ms(), 2),
            "fps": round(self.get_fps(), 2),
            "sample_count": len(self.latencies_ms),
        }
