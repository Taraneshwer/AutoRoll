"""
Inference & Latency Benchmark Helper Script for AutoRoll ML Pipeline.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import time

from app.core.logger import get_logger
from app.core.metrics import MetricsTracker

logger = get_logger("bench_inference")


def main():
    logger.info("Initializing AutoRoll ML Pipeline Benchmark Suite...")
    tracker = MetricsTracker(window_size=50)

    # Dummy benchmarking simulation for Phase 1 skeleton
    for i in range(20):
        start = time.perf_counter()
        time.sleep(0.01)  # Simulate 10ms processing step
        latency_ms = (time.perf_counter() - start) * 1000.0
        tracker.record_frame(latency_ms)

    summary = tracker.get_summary()
    logger.info(
        f"Benchmark Results: Average Latency={summary['avg_latency_ms']}ms | FPS={summary['fps']}"
    )


if __name__ == "__main__":
    main()
