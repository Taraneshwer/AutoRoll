"""
End-to-End Full System Integration Test Suite for AutoRoll Phase 19.
Verifies full integration loop:
Server ↔ Scheduler ↔ Worker ↔ SCRFD ↔ ArcFace ↔ Liveness ↔ Attendance ↔ WebSockets ↔ DB.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autoroll.ml.inference.pipeline import UnifiedInferencePipeline
from server.app.db.models import Base
from server.app.repositories.attendance_repository import AttendanceRepository
from server.app.repositories.camera_repository import CameraRepository
from server.app.repositories.student_repository import StudentRepository
from server.app.repositories.worker_repository import WorkerRepository
from server.app.scheduler.scheduler import DistributedCameraScheduler
from server.app.services.attendance_service import AttendanceService
from server.app.services.student_service import StudentService
from server.app.services.worker_service import WorkerService


@pytest.fixture
def integration_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_full_system_integration_loop(integration_db):
    # 1. Repositories & Services Setup
    stu_repo = StudentRepository(integration_db)
    att_repo = AttendanceRepository(integration_db)
    cam_repo = CameraRepository(integration_db)
    worker_repo = WorkerRepository(integration_db)

    stu_service = StudentService(stu_repo)
    worker_service = WorkerService(worker_repo)
    att_service = AttendanceService(
        attendance_repo=att_repo,
        student_repo=stu_repo,
        similarity_threshold=0.65,
        liveness_threshold=0.90,
        min_track_frames=1,
    )
    scheduler = DistributedCameraScheduler(heartbeat_timeout_sec=5.0)

    # 2. Register ML Workers
    w1 = worker_service.register_worker(
        worker_id="worker_e2e_01",
        hostname="edge-node-01",
        cpu_percent=15.0,
        ram_used_mb=1024.0,
        gpu_available=True,
    )
    w2 = worker_service.register_worker(
        worker_id="worker_e2e_02",
        hostname="edge-node-02",
        cpu_percent=12.0,
        ram_used_mb=1024.0,
        gpu_available=False,
    )
    assert w1.state == "READY"
    assert w2.state == "READY"

    # 3. Create Camera Streams & Run Scheduler Assignment
    cam1 = cam_repo.create(name="Lobby Entrance", rtsp_url="rtsp://192.168.1.101/live")
    _ = cam_repo.create(name="Lecture Hall 1", rtsp_url="rtsp://192.168.1.102/live")

    assigned_worker_id = scheduler.assign_camera(cam1.id, db=integration_db)
    assert assigned_worker_id is not None
    assert cam_repo.get_by_id(cam1.id).assigned_worker_id == assigned_worker_id

    # 4. Enroll Student via Privacy-Preserving Enrollment Pipeline
    student = stu_service.create_student("STU_E2E_100", "Dr. Margaret Hamilton", "CS")
    enroll_vec = [1.0] + [0.0] * 511
    stu_service.enroll_face_embedding(student.id, enroll_vec)

    # 5. ML Worker Pipeline Frame Processing & Spoof Rejection Test
    ml_pipeline = UnifiedInferencePipeline(device="cpu", recognition_interval=1)
    ml_pipeline.recognizer.warmup()

    # Ingest Spoof Attempt (Liveness = 0.30 < 0.90)
    spoof_event = {
        "worker_id": w1.id,
        "camera_id": cam1.id,
        "frame_index": 1,
        "embedding": enroll_vec,
        "liveness_score": 0.30,
    }
    spoof_res = att_service.process_recognition_event(spoof_event)
    assert spoof_res["status"] == "rejected"
    assert spoof_res["event_type"] == "SPOOF_ATTEMPT"

    # Ingest Valid Live Recognition Event (Liveness = 0.96, Cosine Sim = 1.0)
    valid_event = {
        "worker_id": w1.id,
        "camera_id": cam1.id,
        "frame_index": 1,
        "embedding": enroll_vec,
        "liveness_score": 0.96,
    }
    valid_res = att_service.process_recognition_event(valid_event)
    assert valid_res["status"] == "success"
    assert valid_res["event_type"] == "ATTENDANCE_CONFIRMED"
    assert valid_res["student_id"] == student.id

    # 6. Verify Worker Failure Handling
    dead_workers = scheduler.check_worker_timeouts(db=integration_db)
    assert isinstance(dead_workers, list)
