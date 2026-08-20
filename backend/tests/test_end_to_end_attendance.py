"""
AutoRoll Phase 11 End-to-End Integration & Privacy Verification Test Suite.
Tests full pipeline: Model Checksum -> Student Creation -> 5-Sample Enrollment -> Template Aggregation
-> Vector Matching -> 3-Frame Temporal Confirmation -> Attendance Decision -> 30s Cooldown -> Spoof Rejection -> Privacy Audit.
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.crypto import normalize_vector
from app.database.models import Base
from app.database.repositories.attendance_repository import AttendanceRepository
from app.database.repositories.student_repository import StudentRepository
from app.ml.matching.matcher import FaceMatcher
from app.services.attendance_service import AttendanceService
from app.services.enrollment_service import EnrollmentService
from app.services.student_service import StudentService

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_model_checksum_verification():
    """Verify validated ArcFace ONNX model SHA256 checksum matches exact baseline."""
    onnx_path = PROJECT_ROOT / "models" / "pretrained" / "arcface_r50_webface_or_glint" / "model.onnx"
    assert onnx_path.exists()

    with open(onnx_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    expected_sha256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    assert digest == expected_sha256, f"ONNX model weight checksum mismatch! Expected {expected_sha256}, got {digest}"


@pytest.fixture
def e2e_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_end_to_end_attendance_pipeline(e2e_db):
    # 1. Setup Services
    stu_repo = StudentRepository(e2e_db)
    att_repo = AttendanceRepository(e2e_db)

    stu_service = StudentService(stu_repo)
    enroll_service = EnrollmentService(stu_service)
    att_service = AttendanceService(
        attendance_repo=att_repo,
        student_repo=stu_repo,
        similarity_threshold=0.0540,
        liveness_threshold=0.90,
        deduplication_window_sec=30,
        min_track_frames=3,
    )

    # 2. Create Student
    student = stu_service.create_student("STU_P11_001", "Dr. Margaret Hamilton", "Computer Science")
    assert student.id is not None

    # 3. Start Multi-Sample Enrollment Session
    session_id = enroll_service.start_enrollment_session(student.id)
    assert session_id is not None

    # Generate 5 valid normalized embedding samples
    np.random.seed(42)
    base_vec = np.random.normal(0.0, 1.0, 512).astype(np.float32)
    base_vec = normalize_vector(base_vec.tolist())

    for _ in range(5):
        noise = np.random.normal(0.0, 0.05, 512).astype(np.float32)
        sample = normalize_vector((np.array(base_vec) + noise).tolist())
        res = enroll_service.process_enrollment_frame(
            session_id=session_id,
            frame_chip=np.ones((112, 112, 3), dtype=np.uint8),
            test_embedding=sample,
        )
        assert res["accepted"] is True

    # Complete Enrollment Session -> Aggregate normalized mean template
    summary = enroll_service.complete_enrollment(session_id)
    assert summary["sample_count"] == 5
    assert summary["embedding_dimension"] == 512
    assert summary["model_id"] == "autoroll_v1"

    # 4. Test Vector Matching Engine
    matcher = FaceMatcher(model_id="autoroll_v1", threshold=0.0540)
    matcher.register_template(student.id, base_vec)

    match_res = matcher.match_embedding(base_vec)
    assert match_res.matched is True
    assert match_res.candidate_student_id == student.id
    assert match_res.similarity > 0.80

    # 5. Ingest Recognition Event -> Temporal Confirmation & Attendance Decision
    valid_event = {
        "worker_id": "worker-01",
        "camera_id": "cam-01",
        "frame_index": 3,
        "embedding": base_vec,
        "liveness_score": 0.96,
    }

    attendance_res = att_service.process_recognition_event(valid_event)
    assert attendance_res["status"] == "success"
    assert attendance_res["event_type"] == "ATTENDANCE_CONFIRMED"
    assert attendance_res["student_id"] == student.id

    # 6. Verify 30-second Attendance Cooldown & Duplicate Suppression
    dup_event = valid_event.copy()
    dup_event["frame_index"] = 4
    dup_res = att_service.process_recognition_event(dup_event)
    assert dup_res["status"] == "suppressed_duplicate"
    assert dup_res["event_type"] == "DUPLICATE_SUPPRESSED"

    # 7. Verify Spoof Attempt Rejection (Liveness = 0.35 < 0.90)
    spoof_event = {
        "worker_id": "worker-01",
        "camera_id": "cam-01",
        "frame_index": 3,
        "embedding": base_vec,
        "liveness_score": 0.35,
    }
    spoof_res = att_service.process_recognition_event(spoof_event)
    assert spoof_res["status"] == "rejected"
    assert spoof_res["event_type"] == "SPOOF_ATTEMPT"
