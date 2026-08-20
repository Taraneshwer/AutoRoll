"""
Unit tests for WorkerCapacityCalculator.
"""

from app.services.scheduler.capacity import WorkerCapacityCalculator


def test_calculate_load_score():
    calc = WorkerCapacityCalculator()

    score1 = calc.calculate_load_score(
        active_cameras_count=1, cpu_percent=20.0, avg_latency_ms=10.0
    )
    score2 = calc.calculate_load_score(
        active_cameras_count=3, cpu_percent=40.0, avg_latency_ms=15.0
    )

    # Worker with fewer cameras should have lower load score
    assert score1 < score2


def test_select_best_worker():
    calc = WorkerCapacityCalculator(max_cameras_per_worker=4)

    workers = [
        {
            "worker_id": "worker_busy",
            "state": "READY",
            "active_cameras_count": 4,  # Full capacity
            "cpu_percent": 30.0,
            "avg_inference_latency_ms": 10.0,
        },
        {
            "worker_id": "worker_light",
            "state": "READY",
            "active_cameras_count": 1,
            "cpu_percent": 15.0,
            "avg_inference_latency_ms": 8.0,
        },
        {
            "worker_id": "worker_offline",
            "state": "OFFLINE",
            "active_cameras_count": 0,
            "cpu_percent": 0.0,
            "avg_inference_latency_ms": 0.0,
        },
    ]

    best = calc.select_best_worker(workers)
    assert best is not None
    assert best["worker_id"] == "worker_light"
