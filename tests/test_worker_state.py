"""
Unit tests for Worker State and Health Metrics module.
"""

from worker.state import WorkerHealthMetrics, WorkerState
from worker.system_info import SystemInfoMonitor


def test_worker_state_enum():
    assert WorkerState.STARTING == "STARTING"
    assert WorkerState.READY == "READY"
    assert WorkerState.BUSY == "BUSY"
    assert WorkerState.DEGRADED == "DEGRADED"
    assert WorkerState.OFFLINE == "OFFLINE"
    assert WorkerState.STOPPING == "STOPPING"


def test_system_info_metrics():
    sys_metrics = SystemInfoMonitor.get_cpu_ram_metrics()
    assert "cpu_percent" in sys_metrics
    assert "ram_used_mb" in sys_metrics
    assert "ram_percent" in sys_metrics

    gpu_metrics = SystemInfoMonitor.get_gpu_metrics()
    assert "gpu_available" in gpu_metrics


def test_health_metrics_schema():
    metrics = WorkerHealthMetrics(
        worker_id="test_worker_1",
        state=WorkerState.READY,
        cpu_percent=12.5,
        ram_used_mb=512.0,
        ram_percent=45.0,
        gpu_available=False,
    )

    assert metrics.worker_id == "test_worker_1"
    assert metrics.state == WorkerState.READY
    assert metrics.active_cameras_count == 0
