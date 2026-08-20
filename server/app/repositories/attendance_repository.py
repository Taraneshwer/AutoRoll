"""
Attendance Record Repository with Deduplication Check.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from server.app.db.models import AnalyticsEvent, AttendanceRecord


class AttendanceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_recent_checkin(
        self, student_id: str, window_seconds: int = 300
    ) -> AttendanceRecord | None:
        """
        Deduplication query: checks if student has already checked in
        within the last window_seconds.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        return (
            self.db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.student_id == student_id,
                AttendanceRecord.timestamp >= cutoff,
            )
            .order_by(AttendanceRecord.timestamp.desc())
            .first()
        )

    def create(
        self,
        student_id: str,
        camera_id: str | None,
        similarity_score: float,
        liveness_score: float,
        model_version: str,
        verification_status: str = "CONFIRMED",
        worker_id: str | None = None,
    ) -> AttendanceRecord:
        record = AttendanceRecord(
            student_id=student_id,
            camera_id=camera_id,
            worker_id=worker_id,
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            model_version=model_version,
            verification_status=verification_status,
            timestamp=datetime.now(timezone.utc),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def log_analytics_event(
        self,
        event_type: str,
        similarity_score: float = 0.0,
        liveness_score: float = 0.0,
        student_id: str | None = None,
        camera_id: str | None = None,
        worker_id: str | None = None,
        model_version: str = "iresnet50_arcface_v1",
    ) -> AnalyticsEvent:
        """
        Logs analytics events (e.g., UNKNOWN_PERSON, SPOOF_ATTEMPT, DUPLICATE_SUPPRESSED)
        without storing raw face images.
        """
        event = AnalyticsEvent(
            event_type=event_type,
            student_id=student_id,
            camera_id=camera_id,
            worker_id=worker_id,
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            model_version=model_version,
            timestamp=datetime.now(timezone.utc),
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def list_recent(self, limit: int = 50) -> list[AttendanceRecord]:
        return (
            self.db.query(AttendanceRecord)
            .order_by(AttendanceRecord.timestamp.desc())
            .limit(limit)
            .all()
        )

    def count_today(self) -> int:
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return (
            self.db.query(AttendanceRecord)
            .filter(AttendanceRecord.timestamp >= start_of_day)
            .count()
        )
