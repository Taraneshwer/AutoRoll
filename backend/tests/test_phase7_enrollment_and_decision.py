"""
Phase 7 Unit Tests for Enrollment Workflow, Liveness Verification, and Temporal Decision Engine.
"""

import time
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.ml.inference.decision import TemporalConfirmationTracker, UnifiedDecisionEngine
from app.services.enrollment_service import EnrollmentService


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_temporal_confirmation_tracker():
    tracker = TemporalConfirmationTracker(required_observations=3, confirmation_window_ms=1500)
    now = time.time()

    # Obs 1
    confirmed, count = tracker.add_observation(track_id=1, student_id="stu_001", similarity=0.85, timestamp=now)
    assert not confirmed
    assert count == 1

    # Obs 2
    confirmed, count = tracker.add_observation(track_id=1, student_id="stu_001", similarity=0.87, timestamp=now + 0.2)
    assert not confirmed
    assert count == 2

    # Obs 3 (within 1500 ms) -> Confirmed!
    confirmed, count = tracker.add_observation(track_id=1, student_id="stu_001", similarity=0.88, timestamp=now + 0.5)
    assert confirmed
    assert count == 3

    # Stale observation test (>1500 ms later)
    confirmed, count = tracker.add_observation(track_id=1, student_id="stu_001", similarity=0.88, timestamp=now + 2.5)
    assert not confirmed
    assert count == 1


def test_unified_decision_engine_attendance_rules():
    engine = UnifiedDecisionEngine()
    now = time.time()

    # Rule 1: Detection confidence < 0.5
    d1 = engine.evaluate_attendance_decision(
        track_id=10, detection_confidence=0.3, is_quality_ok=True, is_live=True,
        liveness_score=0.95, student_id="stu_A", similarity_score=0.80, recognition_threshold=0.0540, model_id="autoroll_v1"
    )
    assert d1 == "LOW_DETECTION_CONFIDENCE"

    # Rule 2: Quality check failed
    d2 = engine.evaluate_attendance_decision(
        track_id=10, detection_confidence=0.9, is_quality_ok=False, is_live=True,
        liveness_score=0.95, student_id="stu_A", similarity_score=0.80, recognition_threshold=0.0540, model_id="autoroll_v1"
    )
    assert d2 == "LOW_QUALITY"

    # Rule 3: Spoof detected (Liveness failed)
    d3 = engine.evaluate_attendance_decision(
        track_id=10, detection_confidence=0.9, is_quality_ok=True, is_live=False,
        liveness_score=0.20, student_id="stu_A", similarity_score=0.80, recognition_threshold=0.0540, model_id="autoroll_v1"
    )
    assert d3 == "REJECTED_SPOOF"

    # Rule 4: Model template incompatibility guard
    d4 = engine.evaluate_attendance_decision(
        track_id=10, detection_confidence=0.9, is_quality_ok=True, is_live=True,
        liveness_score=0.95, student_id="stu_A", similarity_score=0.80, recognition_threshold=0.0540,
        model_id="autoroll_v1", template_model_id="pretrained"
    )
    assert d4 == "INCOMPATIBLE_MODEL_TEMPLATE"

    # Rule 5: Insufficient similarity
    d5 = engine.evaluate_attendance_decision(
        track_id=10, detection_confidence=0.9, is_quality_ok=True, is_live=True,
        liveness_score=0.95, student_id="stu_A", similarity_score=0.02, recognition_threshold=0.0540, model_id="autoroll_v1"
    )
    assert d5 == "INSUFFICIENT_CONFIDENCE"


def test_enrollment_service_lifecycle(in_memory_db):
    service = EnrollmentService()
    session_id = service.start_session("STU-101", "John Doe", "CS")
    assert session_id in service.active_sessions

    # Inject synthetic embeddings directly to test mean template aggregation & DB persistence
    session = service.active_sessions[session_id]
    v1 = np.random.randn(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(512).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    session.valid_embeddings.extend([v1.tolist(), v2.tolist()])

    res = service.complete_session(session_id, db=in_memory_db, model_id="autoroll_v1")
    assert res["status"] == "ENROLLED"
    assert res["student_code"] == "STU-101"
    assert res["model_id"] == "autoroll_v1"
    assert res["embedding_dimension"] == 512
    assert res["sample_count"] == 2
