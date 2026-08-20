"""
Comprehensive Unit Tests for AutoRoll Phase 12 Attendance Engine.
Tests:
- Valid recognition
- Unknown person
- Spoof attempt
- Repeated detection (deduplication suppression)
- Multiple people in frame
- Worker reassignment handling
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


def test_valid_recognition(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    student = stu_service.create_student("STU_V1", "Valid Student")
    ref_vec = [1.0] + [0.0] * 511
    stu_service.enroll_face_embedding(student.id, ref_vec)

    att_service = AttendanceService(
        attendance_repo=att_repo,
        student_repo=stu_repo,
        similarity_threshold=0.65,
        liveness_threshold=0.90,
    )

    event = {
        "worker_id": "worker_w1",
        "camera_id": "cam_01",
        "frame_index": 5,  # Satisfies min_track_frames=3
        "embedding": ref_vec,
        "liveness_score": 0.95,
    }

    res = att_service.process_recognition_event(event)
    assert res["status"] == "success"
    assert res["event_type"] == "ATTENDANCE_CONFIRMED"
    assert res["student_id"] == student.id
    assert res["worker_id"] == "worker_w1"


def test_unknown_person(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    student = stu_service.create_student("STU_V2", "Enrolled Student")
    stu_service.enroll_face_embedding(student.id, [1.0] + [0.0] * 511)

    att_service = AttendanceService(attendance_repo=att_repo, student_repo=stu_repo)

    # Unknown person embedding (orthogonal vector)
    unknown_vec = [0.0, 1.0] + [0.0] * 510
    event = {
        "worker_id": "worker_w1",
        "camera_id": "cam_01",
        "frame_index": 5,
        "embedding": unknown_vec,
        "liveness_score": 0.95,
    }

    res = att_service.process_recognition_event(event)
    assert res["status"] == "rejected"
    assert res["event_type"] == "UNKNOWN_PERSON"


def test_spoof_attempt(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    student = stu_service.create_student("STU_V3", "Spoofed Student")
    ref_vec = [1.0] + [0.0] * 511
    stu_service.enroll_face_embedding(student.id, ref_vec)

    att_service = AttendanceService(
        attendance_repo=att_repo, student_repo=stu_repo, liveness_threshold=0.90
    )

    event = {
        "worker_id": "worker_w1",
        "camera_id": "cam_01",
        "frame_index": 5,
        "embedding": ref_vec,
        "liveness_score": 0.40,  # Fails liveness threshold (photo/screen replay)
    }

    res = att_service.process_recognition_event(event)
    assert res["status"] == "rejected"
    assert res["event_type"] == "SPOOF_ATTEMPT"


def test_repeated_detection(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    student = stu_service.create_student("STU_V4", "Repeated Student")
    ref_vec = [0.0, 0.0, 1.0] + [0.0] * 509
    stu_service.enroll_face_embedding(student.id, ref_vec)

    att_service = AttendanceService(
        attendance_repo=att_repo, student_repo=stu_repo, deduplication_window_sec=300
    )

    event = {
        "worker_id": "worker_w1",
        "camera_id": "cam_01",
        "frame_index": 5,
        "embedding": ref_vec,
        "liveness_score": 0.95,
    }

    res1 = att_service.process_recognition_event(event)
    assert res1["status"] == "success"

    res2 = att_service.process_recognition_event(event)
    assert res2["status"] == "suppressed_duplicate"
    assert res2["event_type"] == "DUPLICATE_SUPPRESSED"


def test_multiple_people_in_frame(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    s1 = stu_service.create_student("STU_M1", "Person One")
    s2 = stu_service.create_student("STU_M2", "Person Two")

    vec1 = [1.0] + [0.0] * 511
    vec2 = [0.0, 1.0] + [0.0] * 510

    stu_service.enroll_face_embedding(s1.id, vec1)
    stu_service.enroll_face_embedding(s2.id, vec2)

    att_service = AttendanceService(attendance_repo=att_repo, student_repo=stu_repo)

    res1 = att_service.process_recognition_event(
        {
            "worker_id": "w1",
            "camera_id": "cam_multi",
            "frame_index": 5,
            "embedding": vec1,
            "liveness_score": 0.95,
        }
    )
    res2 = att_service.process_recognition_event(
        {
            "worker_id": "w1",
            "camera_id": "cam_multi",
            "frame_index": 5,
            "embedding": vec2,
            "liveness_score": 0.95,
        }
    )

    assert res1["status"] == "success"
    assert res1["student_id"] == s1.id
    assert res2["status"] == "success"
    assert res2["student_id"] == s2.id


def test_worker_reassignment_handling(db_session):
    stu_repo = StudentRepository(db_session)
    att_repo = AttendanceRepository(db_session)
    stu_service = StudentService(stu_repo)

    student = stu_service.create_student("STU_REASSIGN", "Reassigned Student")
    vec = [0.0, 0.0, 0.0, 1.0] + [0.0] * 508
    stu_service.enroll_face_embedding(student.id, vec)

    att_service = AttendanceService(attendance_repo=att_repo, student_repo=stu_repo)

    # First event from worker_01
    res1 = att_service.process_recognition_event(
        {
            "worker_id": "worker_01",
            "camera_id": "cam_reassigned",
            "frame_index": 5,
            "embedding": vec,
            "liveness_score": 0.95,
        }
    )
    assert res1["status"] == "success"
    assert res1["worker_id"] == "worker_01"

    # Worker 1 dies, camera reassigned to worker_02 -> Event from worker_02 for another student
    student2 = stu_service.create_student("STU_REASSIGN_2", "Student Two")
    vec2 = [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 507
    stu_service.enroll_face_embedding(student2.id, vec2)

    res2 = att_service.process_recognition_event(
        {
            "worker_id": "worker_02",
            "camera_id": "cam_reassigned",
            "frame_index": 5,
            "embedding": vec2,
            "liveness_score": 0.95,
        }
    )
    assert res2["status"] == "success"
    assert res2["worker_id"] == "worker_02"
