"""
Student and Student Embedding Repository.
"""


from sqlalchemy.orm import Session

from server.app.db.models import Student, StudentEmbedding


class StudentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, student_id: str) -> Student | None:
        return self.db.query(Student).filter(Student.id == student_id).first()

    def get_by_code(self, student_code: str) -> Student | None:
        return self.db.query(Student).filter(Student.student_code == student_code).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> list[Student]:
        return self.db.query(Student).offset(skip).limit(limit).all()

    def create(self, student_code: str, full_name: str, department: str | None = None) -> Student:
        student = Student(
            student_code=student_code,
            full_name=full_name,
            department=department,
            is_active=True,
        )
        self.db.add(student)
        self.db.commit()
        self.db.refresh(student)
        return student

    def add_embedding(
        self,
        student_id: str,
        embedding_bytes: bytes,
        model_version: str,
        is_primary: bool = True,
    ) -> StudentEmbedding:
        emb = StudentEmbedding(
            student_id=student_id,
            embedding_vector=embedding_bytes,
            model_version=model_version,
            is_primary=is_primary,
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def get_all_embeddings(self) -> list[StudentEmbedding]:
        return self.db.query(StudentEmbedding).all()
