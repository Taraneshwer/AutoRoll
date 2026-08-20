"""
Phase 10 Automated Tests for Central Control Plane, Worker Registration, Load Balancing, Failover, and Deduplication.
"""

import time
import pytest
from app.services.worker_service import (
    CameraRegistrationRequest,
    CentralWorkerControlPlane,
    RecognitionEvent,
    WorkerRegistrationRequest,
)


def test_worker_registration_and_secret():
    cp = CentralWorkerControlPlane()
    req_good = WorkerRegistrationRequest(
        worker_id="worker-01",
        hostname="node-01",
        secret="autoroll_secret_2026",
        gpu_name="NVIDIA RTX 5060",
        model_id="autoroll_v1",
        model_version="autoroll_arcface_r50_epoch1",
        embedding_dimension=512,
        threshold=0.0540,
    )
    res = cp.register_worker(req_good)
    assert res.worker_id == "worker-01"
    assert res.status == "ONLINE"


    # Reject invalid secret
    req_bad_secret = req_good.model_copy(update={"secret": "wrong_secret"})
    with pytest.raises(ValueError, match="Unauthorized"):
        cp.register_worker(req_bad_secret)


def test_model_consistency_enforcement():
    cp = CentralWorkerControlPlane()
    req_bad_model = WorkerRegistrationRequest(
        worker_id="worker-bad",
        hostname="node-bad",
        secret="autoroll_secret_2026",
        gpu_name="CPU",
        model_id="incompatible_model",
        model_version="v2",
        embedding_dimension=256,
        threshold=0.80,
    )
    with pytest.raises(ValueError, match="Incompatible worker model configuration"):
        cp.register_worker(req_bad_model)


def test_load_aware_camera_assignment():
    cp = CentralWorkerControlPlane()
    secret = "autoroll_secret_2026"

    cp.register_worker(WorkerRegistrationRequest(
        worker_id="worker-01", hostname="node-1", secret=secret,
        model_id="autoroll_v1", model_version="v1", embedding_dimension=512, threshold=0.0540
    ))
    cp.register_worker(WorkerRegistrationRequest(
        worker_id="worker-02", hostname="node-2", secret=secret,
        model_id="autoroll_v1", model_version="v1", embedding_dimension=512, threshold=0.0540
    ))

    cp.register_camera(CameraRegistrationRequest(camera_id="cam-01", camera_name="Cam 1", stream_url="rtsp://1"))
    cp.register_camera(CameraRegistrationRequest(camera_id="cam-02", camera_name="Cam 2", stream_url="rtsp://2"))

    w1_assigned = cp.cameras["cam-01"]["worker_id"]
    w2_assigned = cp.cameras["cam-02"]["worker_id"]

    # Load-aware distribution should assign cam-01 to worker-01 and cam-02 to worker-02
    assert w1_assigned != w2_assigned
    assert w1_assigned in ["worker-01", "worker-02"]
    assert w2_assigned in ["worker-01", "worker-02"]

    assert len(cp.workers["worker-01"].assigned_cameras) == 1
    assert len(cp.workers["worker-02"].assigned_cameras) == 1


def test_worker_failure_and_camera_reassignment():
    cp = CentralWorkerControlPlane()
    secret = "autoroll_secret_2026"

    cp.register_worker(WorkerRegistrationRequest(
        worker_id="w-01", hostname="n-1", secret=secret,
        model_id="autoroll_v1", model_version="v1", embedding_dimension=512, threshold=0.0540
    ))
    cp.register_worker(WorkerRegistrationRequest(
        worker_id="w-02", hostname="n-2", secret=secret,
        model_id="autoroll_v1", model_version="v1", embedding_dimension=512, threshold=0.0540
    ))

    cp.register_camera(CameraRegistrationRequest(camera_id="cam-10", camera_name="Door Cam", stream_url="rtsp://1"))
    assigned_worker = cp.cameras["cam-10"]["worker_id"]
    assert assigned_worker in ["w-01", "w-02"]

    # Simulate worker-01 heartbeat timeout (> 15s)
    cp.workers[assigned_worker].last_heartbeat = time.time() - 20.0

    reassigned = cp.check_health_and_failover()
    assert "cam-10" in reassigned
    assert cp.workers[assigned_worker].status == "OFFLINE"
    new_worker = cp.cameras["cam-10"]["worker_id"]
    assert new_worker != assigned_worker
    assert cp.workers[new_worker].status == "ONLINE"


def test_duplicate_attendance_event_deduplication():
    cp = CentralWorkerControlPlane()
    event1 = RecognitionEvent(
        worker_id="w-01",
        camera_id="cam-01",
        timestamp=time.time(),
        student_id="STU2001",
        similarity=0.74,
        liveness_score=0.96,
        decision="PRESENT",
        processing_latency_ms=5.5,
    )
    res1 = cp.ingest_recognition_event(event1)
    assert res1["status"] == "RECORDED"

    # Immediate duplicate from second camera
    event2 = event1.model_copy(update={"camera_id": "cam-02", "timestamp": time.time() + 0.1})
    res2 = cp.ingest_recognition_event(event2)
    assert res2["status"] == "DUPLICATE_SUPPRESSED"
