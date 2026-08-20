"""
Unit & Integration Tests for AutoRoll Phase 14 Distributed GPU Worker Architecture.
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.workers.worker_registry import WorkerRegistry
from app.workers.worker_health import WorkerHealthMonitor
from app.workers.worker_scheduler import WorkerScheduler
from app.workers.load_balancer import WorkerLoadBalancer
from app.workers.models import (
    WorkerRegistrationRequest,
    WorkerHeartbeatRequest,
    WorkerStatus,
    WorkerNodeInfo,
)
from app.workers.worker_protocol import WorkerEventType, WorkerFailoverEvent

client = TestClient(app)


def test_worker_registration_success():
    req = WorkerRegistrationRequest(
        worker_id="test-worker-01",
        hostname="node-alpha",
        secret="autoroll_secret_2026",
        gpu_name="NVIDIA RTX 5060",
        gpu_memory_total=8151.0,
        model_id="autoroll_v1",
        model_version="autoroll_arcface_r50_epoch1",
        embedding_dimension=512,
    )
    res = client.post("/api/v1/workers/register", json=req.model_dump())
    assert res.status_code == 200
    data = res.json()
    assert data["worker_id"] == "test-worker-01"
    assert data["status"] == "ONLINE"


def test_worker_registration_unauthorized():
    req = WorkerRegistrationRequest(
        worker_id="unauth-worker",
        secret="bad_secret_token",
        model_id="autoroll_v1",
        embedding_dimension=512,
    )
    res = client.post("/api/v1/workers/register", json=req.model_dump())
    assert res.status_code == 401
    assert "Unauthorized" in res.json()["detail"]


def test_worker_registration_incompatible_model():
    req = WorkerRegistrationRequest(
        worker_id="incompatible-worker",
        secret="autoroll_secret_2026",
        model_id="wrong_model",
        embedding_dimension=128,
    )
    res = client.post("/api/v1/workers/register", json=req.model_dump())
    assert res.status_code == 400
    assert "Incompatible model" in res.json()["detail"]


def test_worker_heartbeat_and_metrics():
    # Register worker first
    reg = WorkerRegistrationRequest(
        worker_id="hb-worker-01",
        secret="autoroll_secret_2026",
        model_id="autoroll_v1",
        embedding_dimension=512,
    )
    client.post("/api/v1/workers/register", json=reg.model_dump())

    hb = WorkerHeartbeatRequest(
        worker_id="hb-worker-01",
        secret="autoroll_secret_2026",
        timestamp=time.time(),
        status=WorkerStatus.ONLINE,
        queue_depth=1,
        inference_fps=28.5,
        average_latency_ms=6.2,
        p95_latency_ms=9.1,
        gpu_utilization=45.0,
        gpu_memory_used=2100.0,
    )
    res = client.post("/api/v1/workers/hb-worker-01/heartbeat", json=hb.model_dump())
    assert res.status_code == 200
    assert res.json()["acknowledged"] is True

    # Fetch metrics
    metrics_res = client.get("/api/v1/workers/hb-worker-01/metrics")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["gpu_utilization"] == 45.0
    assert m_data["inference_fps"] == 28.5


def test_load_balancer_scoring():
    balancer = WorkerLoadBalancer(gpu_weight=0.35, queue_weight=0.25, camera_weight=0.20, latency_weight=0.20)
    w_low = WorkerNodeInfo(
        worker_id="w-low",
        hostname="node1",
        gpu_utilization=10.0,
        queue_depth=0,
        assigned_cameras=[],
        p95_latency_ms=5.0,
    )
    w_high = WorkerNodeInfo(
        worker_id="w-high",
        hostname="node2",
        gpu_utilization=90.0,
        queue_depth=4,
        assigned_cameras=["c1", "c2", "c3"],
        p95_latency_ms=50.0,
    )

    score_low = balancer.calculate_score(w_low)
    score_high = balancer.calculate_score(w_high)
    assert score_low < score_high

    workers = {"w-low": w_low, "w-high": w_high}
    best = balancer.select_best_worker(workers)
    assert best.worker_id == "w-low"


def test_worker_draining():
    req = WorkerRegistrationRequest(
        worker_id="drain-worker",
        secret="autoroll_secret_2026",
        model_id="autoroll_v1",
        embedding_dimension=512,
    )
    client.post("/api/v1/workers/register", json=req.model_dump())

    res = client.post("/api/v1/workers/drain-worker/drain")
    assert res.status_code == 200
    assert res.json()["status"] == "DRAINING"



def test_worker_unsupported_restart():
    res = client.post("/api/v1/workers/test-worker-01/restart")
    assert res.status_code == 501
    assert "not directly supported" in res.json()["detail"]


def test_automatic_failover():
    registry = WorkerRegistry()
    scheduler = WorkerScheduler(registry)

    w1 = registry.register(WorkerRegistrationRequest(worker_id="w1", secret="autoroll_secret_2026"))
    w2 = registry.register(WorkerRegistrationRequest(worker_id="w2", secret="autoroll_secret_2026"))

    scheduler.assign_camera("cam-101", worker_id="w1")
    failovers = scheduler.handle_failover("w1")

    assert len(failovers) == 1
    assert failovers[0].camera_id == "cam-101"
    assert failovers[0].old_worker_id == "w1"
    assert failovers[0].new_worker_id == "w2"
