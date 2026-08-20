"""
Phase 7 Integration Tests for FastAPI REST Endpoints.
Tests /api/v1/health, /api/v1/ml/status, /api/v1/students, /api/v1/enrollment/start, /api/v1/attendance.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_ml_status_endpoint():
    response = client.get("/api/v1/ml/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "active_model_id" in data
    assert "recognition_threshold" in data
    assert data["embedding_dimension"] == 512


def test_students_list_endpoint():
    response = client.get("/api/v1/students")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_enrollment_start_endpoint():
    payload = {
        "student_code": "TEST-CODE-999",
        "full_name": "Test User",
        "department": "Engineering",
    }
    response = client.post("/api/v1/enrollment/start", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "STARTED"


def test_attendance_list_endpoint():
    response = client.get("/api/v1/attendance")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
