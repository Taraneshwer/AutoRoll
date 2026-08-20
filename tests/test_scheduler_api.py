"""
Unit tests for FastAPI Workers, Cameras, and Scheduler API routes.
"""

import pytest
from fastapi.testclient import TestClient

from server.app.db.session import Base, engine
from server.main import app


@pytest.fixture
def client():
    Base.metadata.create_all(bind=engine)
    return TestClient(app)


def test_scheduler_status_endpoint(client):
    response = client.get("/api/v1/scheduler/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_workers" in data
    assert "total_cameras" in data


def test_worker_registration_and_heartbeat(client):
    reg_payload = {
        "worker_id": "api_test_worker_1",
        "state": "READY",
        "cpu_percent": 10.0,
        "ram_used_mb": 256.0,
        "ram_percent": 30.0,
        "gpu_available": False,
    }

    resp = client.post("/api/v1/workers/register", json=reg_payload)
    assert resp.status_code == 201
    assert resp.json()["worker_id"] == "api_test_worker_1"

    hb_resp = client.post("/api/v1/workers/heartbeat", json=reg_payload)
    assert hb_resp.status_code == 200
    assert hb_resp.json()["status"] == "acknowledged"
