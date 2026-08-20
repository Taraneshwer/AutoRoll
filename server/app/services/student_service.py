"""
Student Registration and Face Vector Enrollment Service.
"""


import numpy as np

from autoroll.common.logger import get_logger
from server.app.db.models import Student, StudentEmbedding
from server.app.repositories.student_repository import StudentRepository

logger = get_logger("student_service")


class StudentService:
    def __init__(self, student_repo: StudentRepository):
        self.student_repo = student_repo

    def create_student(
        self, student_code: str, full_name: str, department: str | None = None
    ) -> Student:
        existing = self.student_repo.get_by_code(student_code)
        if existing:
            raise ValueError(f"Student with code '{student_code}' already exists.")

        student = self.student_repo.create(
            student_code=student_code, full_name=full_name, department=department
        )
        logger.info(f"Student created: '{full_name}' ({student_code}).")
        return student

    def enroll_face_embedding(
        self,
        student_id: str,
        embedding_vector: list[float],
        model_version: str = "iresnet50_arcface_v1",
    ) -> StudentEmbedding:
        student = self.student_repo.get_by_id(student_id)
        if not student:
            raise ValueError(f"Student '{student_id}' not found.")

        if len(embedding_vector) != 512:
            raise ValueError(f"Expected 512-dimensional embedding, got {len(embedding_vector)}.")

        vec_arr = np.array(embedding_vector, dtype=np.float32)
        raw_bytes = vec_arr.tobytes()

        emb_obj = self.student_repo.add_embedding(
            student_id=student_id,
            embedding_bytes=raw_bytes,
            model_version=model_version,
            is_primary=True,
        )
        logger.info(f"Face embedding enrolled for Student '{student.full_name}'.")
        return emb_obj

    def get_student(self, student_id: str) -> Student | None:
        return self.student_repo.get_by_id(student_id)

    def list_students(self, skip: int = 0, limit: int = 100) -> list[Student]:
        return self.student_repo.list_all(skip=skip, limit=limit)
