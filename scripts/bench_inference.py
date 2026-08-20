"""
Inference & Latency Benchmark Helper Script for AutoRoll ML Pipeline.
"""

import time

from autoroll.common.logger import get_logger
from autoroll.common.metrics import MetricsTracker

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
