"""
Dashboard Metrics Aggregator Service.
Exposes control plane operational metrics and system stats.
"""

from typing import Any

from sqlalchemy.orm import Session

from app.database.models import Camera, Student, StudentEmbedding, WorkerNode
from app.database.repositories.attendance_repository import AttendanceRepository


class MetricsService:
    def __init__(self, db: Session):
        self.db = db
        self.attendance_repo = AttendanceRepository(db)

    def get_dashboard_stats(self) -> dict[str, Any]:
        total_students = self.db.query(Student).filter(Student.is_active.is_(True)).count()
        total_embeddings = self.db.query(StudentEmbedding).count()
        total_cameras = self.db.query(Camera).filter(Camera.is_active.is_(True)).count()
        assigned_cameras = (
            self.db.query(Camera)
            .filter(Camera.is_active.is_(True), Camera.assigned_worker_id.isnot(None))
            .count()
        )
        total_workers = self.db.query(WorkerNode).count()
        online_workers = (
            self.db.query(WorkerNode).filter(WorkerNode.state != "OFFLINE").count()
        )
        today_attendance = self.attendance_repo.count_today()

        return {
            "students_total": total_students,
            "embeddings_enrolled": total_embeddings,
            "cameras_total": total_cameras,
            "cameras_active": assigned_cameras,
            "workers_total": total_workers,
            "workers_online": online_workers,
            "attendance_today": today_attendance,
        }
