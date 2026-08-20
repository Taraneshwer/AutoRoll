"""
Attendance Log and Recognition Events API Endpoints.
"""


from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from server.app.db.session import get_db
from server.app.repositories.attendance_repository import AttendanceRepository
from server.app.repositories.student_repository import StudentRepository
from server.app.schemas.attendance import AttendanceRecordResponse, RecognitionEventPayload
from server.app.services.attendance_service import AttendanceService

router = APIRouter(tags=["Attendance"])


@router.post(
    "/api/v1/events/recognition",
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
    "/api/v1/attendance",
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
            timestamp=r.timestamp.isoformat(),
        )
        for r in records
    ]
