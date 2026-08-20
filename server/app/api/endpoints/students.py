"""
Students and Face Vector Enrollment API Endpoints.
"""


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from server.app.db.session import get_db
from server.app.repositories.student_repository import StudentRepository
from server.app.schemas.student import EnrollmentPayload, StudentCreate, StudentResponse
from server.app.services.student_service import StudentService

router = APIRouter(prefix="/students", tags=["Students"])


@router.post("", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    student_repo = StudentRepository(db)
    student_service = StudentService(student_repo)
    try:
        student = student_service.create_student(
            student_code=payload.student_code,
            full_name=payload.full_name,
            department=payload.department,
        )
        return StudentResponse(
            id=student.id,
            student_code=student.student_code,
            full_name=student.full_name,
            department=student.department,
            is_active=student.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[StudentResponse])
def list_students(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    student_repo = StudentRepository(db)
    student_service = StudentService(student_repo)
    students = student_service.list_students(skip=skip, limit=limit)
    return [
        StudentResponse(
            id=s.id,
            student_code=s.student_code,
            full_name=s.full_name,
            department=s.department,
            is_active=s.is_active,
        )
        for s in students
    ]


@router.post("/{student_id}/enroll", status_code=status.HTTP_201_CREATED)
def enroll_student_face(
    student_id: str, payload: EnrollmentPayload, db: Session = Depends(get_db)
):
    student_repo = StudentRepository(db)
    student_service = StudentService(student_repo)
    try:
        emb_obj = student_service.enroll_face_embedding(
            student_id=student_id,
            embedding_vector=payload.embedding,
            model_version=payload.model_version,
        )
        return {
            "status": "enrolled",
            "student_id": student_id,
            "embedding_id": emb_obj.id,
            "model_version": emb_obj.model_version,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
