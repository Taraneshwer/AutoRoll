"""
Unit tests for DistributedCameraScheduler logic, assignment, and failover.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.app.db.models import Base, Camera, WorkerNode
from server.app.scheduler.scheduler import DistributedCameraScheduler


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_camera_assignment_and_unassignment(db_session):
    scheduler = DistributedCameraScheduler()

    # Add camera
    cam = Camera(id="cam_test_1", name="Cam 1", rtsp_url="rtsp://localhost/stream")
    db_session.add(cam)
    db_session.commit()

    # Register worker
    scheduler.register_worker({"worker_id": "worker_w1", "state": "READY"}, db_session)

    # Assign camera
    assigned_id = scheduler.assign_camera("cam_test_1", worker_id="worker_w1", db=db_session)
    assert assigned_id == "worker_w1"

    status = scheduler.get_scheduler_status(db_session)
    assert status["total_cameras"] == 1
    assert status["unassigned_cameras"] == 0

    # Unassign camera
    success = scheduler.unassign_camera("cam_test_1", db_session)
    assert success is True

    status_after = scheduler.get_scheduler_status(db_session)
    assert status_after["unassigned_cameras"] == 1


def test_worker_failover(db_session):
    scheduler = DistributedCameraScheduler(heartbeat_timeout_sec=15.0)

    # Add 2 cameras
    c1 = Camera(id="c1", name="Cam 1", rtsp_url="rtsp://localhost/c1")
    c2 = Camera(id="c2", name="Cam 2", rtsp_url="rtsp://localhost/c2")
    db_session.add_all([c1, c2])
    db_session.commit()

    # Register 2 workers
    scheduler.register_worker({"worker_id": "w1", "state": "READY"}, db_session)
    scheduler.register_worker({"worker_id": "w2", "state": "READY"}, db_session)

    # Assign both cameras to w1
    scheduler.assign_camera("c1", "w1", db_session)
    scheduler.assign_camera("c2", "w1", db_session)

    # Simulate w1 timeout by setting last_heartbeat_at to 60 seconds ago
    w1_db = db_session.query(WorkerNode).filter(WorkerNode.id == "w1").first()
    if w1_db:
        w1_db.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        db_session.commit()

    dead = scheduler.check_worker_timeouts(db_session)
    assert "w1" in dead

    status = scheduler.get_scheduler_status(db_session)
    assert status["unassigned_cameras"] == 0
    w2_status = next(w for w in status["workers"] if w["worker_id"] == "w2")
    assert w2_status["assigned_cameras_count"] == 2
