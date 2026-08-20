"""
Student Face Enrollment API Routes.
Provides session-based enrollment endpoints: start, frame capture, complete.
"""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.enrollment_service import EnrollmentService

router = APIRouter(prefix="/enrollment", tags=["Enrollment"])

# Global Enrollment Service Singleton
_enrollment_service = EnrollmentService()


class EnrollmentStartRequest(BaseModel):
    student_code: str
    full_name: str
    department: str | None = None


@router.post("/start", status_code=status.HTTP_201_CREATED)
def start_enrollment_session(payload: EnrollmentStartRequest):
    """
    Starts a new student enrollment session.
    """
    try:
        session_id = _enrollment_service.start_session(
            student_code=payload.student_code,
            full_name=payload.full_name,
            department=payload.department,
        )
        return {
            "session_id": session_id,
            "student_code": payload.student_code,
            "full_name": payload.full_name,
            "status": "STARTED",
            "required_samples": 5,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/frame")
async def add_enrollment_frame(
    session_id: str,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
):
    """
    Processes an enrollment frame image for session_id.
    Detects face, validates quality/liveness, aligns face, extracts normalized ArcFace embedding.
    Rejects bad frames with explicit reason.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

        res = _enrollment_service.add_frame(
            session_id=session_id, frame=frame, model_id=model_id
        )
        return res
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/complete")
def complete_enrollment_session(
    session_id: str,
    model_id: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Completes session by computing normalized mean embedding template from accepted samples and saving to DB.
    """
    try:
        res = _enrollment_service.complete_session(
            session_id=session_id, db=db, model_id=model_id
        )
        return res
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
