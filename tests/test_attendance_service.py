"""
Unit tests for AttendanceService vector similarity and deduplication.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.app.db.models import Base
from server.app.repositories.attendance_repository import AttendanceRepository
from server.app.repositories.student_repository import StudentRepository
from server.app.services.attendance_service import AttendanceService
from server.app.services.student_service import StudentService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_attendance_vector_matching_and_deduplication(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    # Enroll student with unit vector
    student = stu_service.create_student("STU_101", "Bob Johnson")
    ref_vec = [1.0] + [0.0] * 511
    stu_service.enroll_face_embedding(student.id, ref_vec)

    att_service = AttendanceService(
        attendance_repo=att_repo,
        student_repo=stu_repo,
        similarity_threshold=0.65,
        liveness_threshold=0.90,
        deduplication_window_sec=300,
    )

    # 1. Matching recognition event
    event = {
        "worker_id": "worker_01",
        "camera_id": "cam_01",
        "embedding": ref_vec,  # Identical vector -> Cosine sim = 1.0
        "liveness_score": 0.95,
    }

    res1 = att_service.process_recognition_event(event)
    assert res1 is not None
    assert res1["event_type"] == "ATTENDANCE_CONFIRMED"
    assert res1["student_id"] == student.id

    # 2. Duplicate event within deduplication window -> Suppressed
    res2 = att_service.process_recognition_event(event)
    assert res2 is not None
    assert res2["status"] == "suppressed_duplicate"
