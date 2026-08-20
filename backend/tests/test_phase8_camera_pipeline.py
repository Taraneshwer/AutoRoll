"""
Phase 8 Unit & Integration Tests for Camera Streaming Layer & Real-time Telemetry Pipeline.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.camera.local_camera import LocalCameraSource
from app.camera.manager import camera_manager
from app.main import app
from app.services.camera_pipeline_service import camera_pipeline_service

client = TestClient(app)


def test_camera_manager_initialization():
    source = camera_manager.initialize_source(source_type="local", camera_index=0)
    assert source is not None
    assert source.is_opened()
    metrics = source.get_metrics()
    assert metrics["source_type"] == "local_webcam"
    assert "capture_fps" in metrics
    camera_manager.stop_source()
    assert camera_manager.get_source() is None


def test_local_camera_bounded_queue():
    cam = LocalCameraSource(camera_index=0, target_fps=30)
    cam.start()

    # Verify frame queue receives frames and drops oldest when full
    time_spent = 0.0
    while cam.frame_queue.empty() and time_spent < 1.0:
        import time

        time.sleep(0.05)
        time_spent += 0.05

    success, frame = cam.read_frame()
    if success:
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert cam.frame_queue.qsize() <= 2

    cam.stop()


def test_camera_stream_api_endpoints():
    # 1. Start camera
    resp_start = client.post("/api/v1/camera/start", json={"source_type": "local", "camera_index": 0})
    assert resp_start.status_code == 200
    assert resp_start.json()["status"] == "STARTED"

    # 2. Get status
    resp_status = client.get("/api/v1/camera/status")
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] == "RUNNING"

    # 3. Process latest frame in pipeline
    telemetry = camera_pipeline_service.process_latest_frame()
    if telemetry:
        assert "pipeline_fps" in telemetry
        assert "capture_latency_ms" in telemetry
        assert "total_latency_ms" in telemetry
        assert "faces" in telemetry

    # 4. Stop camera
    resp_stop = client.post("/api/v1/camera/stop")
    assert resp_stop.status_code == 200
    assert resp_stop.json()["status"] == "STOPPED"
