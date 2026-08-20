"""
Unit tests for StudentService.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.app.db.models import Base
from server.app.repositories.student_repository import StudentRepository
from server.app.services.student_service import StudentService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_student_creation_and_enrollment(db_session):
    repo = StudentRepository(db_session)
    service = StudentService(repo)

    student = service.create_student(
        student_code="STU_999", full_name="Alice Smith", department="Computer Science"
    )
    assert student.id is not None
    assert student.student_code == "STU_999"

    # Enroll 512-d embedding
    dummy_vec = [0.1] * 512
    emb = service.enroll_face_embedding(student.id, dummy_vec)
    assert emb.id is not None
    assert emb.student_id == student.id
