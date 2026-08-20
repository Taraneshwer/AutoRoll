"""
Attendance Log and Recognition Events API Endpoints.
Supports recent attendance listing and today's attendance logs.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database.repositories.attendance_repository import AttendanceRepository
from app.database.repositories.student_repository import StudentRepository
from app.database.session import get_db
from app.schemas.attendance import AttendanceRecordResponse, RecognitionEventPayload
from app.services.attendance_service import AttendanceService

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post(
    "/events/recognition",
    status_code=status.HTTP_201_CREATED,
    summary="Worker Recognition Event Ingestion",
)
def process_recognition_event(
    payload: RecognitionEventPayload, db: Session = Depends(get_db)
):
    att_repo = AttendanceRepository(db)
    stu_repo = StudentRepository(db)
    att_service = AttendanceService(att_repo, stu_repo)

    res = att_service.process_recognition_event(payload.model_dump())
    if not res:
        return {"status": "unrecognized_or_below_threshold"}
    return res


@router.get(
    "",
    response_model=list[AttendanceRecordResponse],
    summary="List Recent Attendance Log Records",
)
@router.get(
    "/",
    response_model=list[AttendanceRecordResponse],
    summary="List Recent Attendance Log Records",
)
def list_recent_attendance(limit: int = 50, db: Session = Depends(get_db)):
    att_repo = AttendanceRepository(db)
    records = att_repo.list_recent(limit=limit)
    return [
        AttendanceRecordResponse(
            id=r.id,
            student_id=r.student_id,
            camera_id=r.camera_id,
            similarity_score=r.similarity_score,
            liveness_score=r.liveness_score,
            model_version=r.model_version,
            verification_status=r.verification_status,
            timestamp=r.timestamp.isoformat() if r.timestamp else datetime.now(timezone.utc).isoformat(),
        )
        for r in records
    ]


@router.get(
    "/today",
    response_model=list[AttendanceRecordResponse],
    summary="List Today's Attendance Records",
)
def list_todays_attendance(db: Session = Depends(get_db)):
    att_repo = AttendanceRepository(db)
    records = att_repo.list_recent(limit=100)
    today_str = datetime.now(timezone.utc).date().isoformat()

    todays_records = [
        r for r in records if r.timestamp and r.timestamp.date().isoformat() == today_str
    ]

    return [
        AttendanceRecordResponse(
            id=r.id,
            student_id=r.student_id,
            camera_id=r.camera_id,
            similarity_score=r.similarity_score,
            liveness_score=r.liveness_score,
            model_version=r.model_version,
            verification_status=r.verification_status,
            timestamp=r.timestamp.isoformat(),
        )
        for r in todays_records
    ]
