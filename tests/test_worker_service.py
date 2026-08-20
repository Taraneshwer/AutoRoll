"""
Unit tests for Worker Service Lifecycle and Camera Assignment.
"""

from worker.config import WorkerSettings
from worker.service import WorkerService
from worker.state import WorkerState


def test_worker_service_lifecycle():
    cfg = WorkerSettings(WORKER_ID="test_worker_srv", SERVER_URL="http://localhost:8000")
    service = WorkerService(config=cfg)

    assert service.state == WorkerState.READY
    assert service.worker_id == "test_worker_srv"

    metrics = service.collect_health_metrics()
    assert metrics.worker_id == "test_worker_srv"
    assert metrics.active_cameras_count == 0

    # Stop service cleanly
    service.stop()
    assert service.state == WorkerState.OFFLINE


def test_worker_camera_assignment():
    cfg = WorkerSettings(WORKER_ID="test_worker_cam")
    service = WorkerService(config=cfg)

    # Assign camera
    service.assign_camera("cam_01", "rtsp://localhost/test")
    assert "cam_01" in service.camera_clients

    metrics = service.collect_health_metrics()
    assert metrics.active_cameras_count == 1
    assert service.state == WorkerState.BUSY

    # Unassign camera
    service.unassign_camera("cam_01")
    assert "cam_01" not in service.camera_clients

    metrics_after = service.collect_health_metrics()
    assert metrics_after.active_cameras_count == 0
    assert service.state == WorkerState.READY

    service.stop()
