"""
Unit tests for FastAPI Central Server API Endpoints.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.database.session import Base, engine
from server.main import app


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert "X-Request-ID" in res.headers


def test_readiness_endpoint(client):
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_student_and_attendance_api(client):
    code = f"STU_{uuid.uuid4().hex[:8]}"
    # Create student
    stu_res = client.post(
        "/api/v1/students",
        json={"student_code": code, "full_name": "Charlie Brown", "department": "EE"},
    )
    assert stu_res.status_code == 201
    stu_data = stu_res.json()
    assert stu_data["student_code"] == code

    vec = [0.0, 0.0, 1.0] + [0.0] * 509
    # Enroll face embedding
    enroll_res = client.post(
        f"/api/v1/students/{stu_data['id']}/enroll",
        json={"embedding": vec, "model_version": "iresnet50_arcface_v1"},
    )
    assert enroll_res.status_code == 201

    # Ingest recognition event
    event_res = client.post(
        "/api/v1/events/recognition",
        json={
            "worker_id": "worker_api_1",
            "camera_id": "cam_api_1",
            "embedding": vec,
            "liveness_score": 0.96,
        },
    )
    assert event_res.status_code == 201
    assert event_res.json()["event_type"] == "ATTENDANCE_CONFIRMED"
