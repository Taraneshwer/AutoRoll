"""
Administrative Audit Logging Service.
Logs administrative actions (camera assignment, worker state change, student deletion, etc.).
"""


from sqlalchemy.orm import Session

from autoroll.common.logger import get_logger
from server.app.db.models import AuditLog

logger = get_logger("audit_service")


class AuditService:
    def __init__(self, db: Session):
        self.db = db

    def log_action(
        self,
        action: str,
        resource_type: str,
        admin_user_id: str | None = None,
        resource_id: str | None = None,
        details: str | None = None,
    ) -> AuditLog:
        """
        Records an administrative action in the audit log.
        """
        entry = AuditLog(
            admin_user_id=admin_user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)

        logger.info(
            f"AUDIT LOG: Admin '{admin_user_id or 'System'}' performed '{action}' "
            f"on {resource_type} '{resource_id or 'N/A'}'."
        )
        return entry

    def list_recent_audit_logs(self, limit: int = 50) -> list[AuditLog]:
        return (
            self.db.query(AuditLog)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
            .all()
        )
