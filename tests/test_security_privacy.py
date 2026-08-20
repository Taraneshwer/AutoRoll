"""
Unit tests for Phase 18 Security, Privacy, RBAC, and Log Sanitization.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autoroll.common.logger import sanitize_log_message
from server.app.auth.rbac import (
    UserRole,
    check_role_permission,
    validate_camera_rtsp_url,
)
from server.app.db.models import Base
from server.app.services.audit_service import AuditService


def test_log_sanitization_rtsp_credentials():
    raw_log = "Connecting to RTSP stream at rtsp://admin:Secret123@192.168.1.100:554/live"
    sanitized = sanitize_log_message(raw_log)

    assert "Secret123" not in sanitized
    assert "admin" not in sanitized
    assert "rtsp://***:***@192.168.1.100:554/live" in sanitized


def test_rbac_permission_hierarchy():
    assert check_role_permission("admin", UserRole.OPERATOR) is True
    assert check_role_permission("operator", UserRole.ADMIN) is False
    assert check_role_permission("operator", UserRole.VIEWER) is True
    assert check_role_permission("viewer", UserRole.OPERATOR) is False


def test_rtsp_url_validation():
    valid_url = "rtsp://192.168.1.101:554/stream1"
    assert validate_camera_rtsp_url(valid_url) == valid_url

    with pytest.raises(Exception):
        validate_camera_rtsp_url("http://invalid-scheme.com")


def test_audit_service():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    audit_service = AuditService(session)
    log_entry = audit_service.log_action(
        action="ASSIGN_CAMERA",
        resource_type="CAMERA",
        admin_user_id="user_admin_1",
        resource_id="cam_01",
        details="Assigned camera cam_01 to worker_01",
    )

    assert log_entry.action == "ASSIGN_CAMERA"
    assert log_entry.admin_user_id == "user_admin_1"

    logs = audit_service.list_recent_audit_logs()
    assert len(logs) == 1
    session.close()
