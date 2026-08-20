"""
Students Management API Endpoints.
Supports student listing, single student retrieval, creation, and deletion.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.repositories.student_repository import StudentRepository
from app.database.session import get_db
from app.schemas.student import StudentCreate, StudentResponse
from app.services.student_service import StudentService

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("", response_model=list[StudentResponse])
@router.get("/", response_model=list[StudentResponse])
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


@router.get("/{student_id}", response_model=StudentResponse)
def get_student_by_id(student_id: str, db: Session = Depends(get_db)):
    student_repo = StudentRepository(db)
    student_service = StudentService(student_repo)
    student = student_service.get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")
    return StudentResponse(
        id=student.id,
        student_code=student.student_code,
        full_name=student.full_name,
        department=student.department,
        is_active=student.is_active,
    )


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


@router.delete("/{student_id}", status_code=status.HTTP_200_OK)
def delete_student(student_id: str, db: Session = Depends(get_db)):
    student_repo = StudentRepository(db)
    student_service = StudentService(student_repo)
    success = student_service.delete_student(student_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")
    return {"status": "deleted", "student_id": student_id}
