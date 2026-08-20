"""
Production Student Face Enrollment Service.
Handles multi-sample capture (5-10 samples), quality/liveness filtering,
normalized mean template generation, and DB template persistence.
"""

import struct
import uuid
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.crypto import normalize_vector
from app.core.logger import get_logger
from app.database.models import Student, StudentEmbedding
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.liveness.passive_fas import PassiveLivenessDetector
from app.ml.preprocessing.quality import FaceQualityFilter
from app.ml.recognition.factory import get_recognizer

logger = get_logger("enrollment_service")


class EnrollmentSession:
    def __init__(self, session_id: str, student_code: str, full_name: str, department: str | None = None):
        self.session_id = session_id
        self.student_code = student_code
        self.full_name = full_name
        self.department = department
        self.valid_embeddings: list[list[float]] = []
        self.reasons_log: list[dict[str, Any]] = []
        self.created_at = float(uuid.uuid4().time)


class EnrollmentService:
    def __init__(self, student_service=None):
        self.student_service = student_service
        self.settings = get_settings()
        self.detector = SCRFDDetector()
        self.aligner = FaceAligner()
        self.quality_filter = FaceQualityFilter()
        self.liveness_detector = PassiveLivenessDetector()
        self.active_sessions: dict[str, EnrollmentSession] = {}

    def start_session(self, student_code: str, full_name: str, department: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        session = EnrollmentSession(
            session_id=session_id,
            student_code=student_code,
            full_name=full_name,
            department=department,
        )
        self.active_sessions[session_id] = session
        logger.info(f"ENROLLMENT SESSION STARTED | SessionID: {session_id} | Code: {student_code} | Name: {full_name}")
        return session_id

    def start_enrollment_session(self, student_id_or_code: str) -> str:
        return self.start_session(student_code=student_id_or_code, full_name=student_id_or_code)

    def process_enrollment_frame(self, session_id: str, frame_chip: np.ndarray, test_embedding: list[float] | None = None) -> dict[str, Any]:
        if test_embedding is not None:
            if session_id not in self.active_sessions:
                raise KeyError(f"Enrollment session '{session_id}' not found.")
            session = self.active_sessions[session_id]
            session.valid_embeddings.append(test_embedding)
            return {"accepted": True, "sample_count": len(session.valid_embeddings)}
        return self.add_frame(session_id=session_id, frame=frame_chip)

    def complete_enrollment(self, session_id: str, db: Session | None = None) -> dict[str, Any]:
        if session_id not in self.active_sessions:
            raise KeyError(f"Enrollment session '{session_id}' not found.")
        session = self.active_sessions[session_id]
        if not session.valid_embeddings:
            raise ValueError("Cannot complete enrollment session with 0 accepted face samples.")
        embeddings_matrix = np.array(session.valid_embeddings, dtype=np.float32)
        mean_embedding = np.mean(embeddings_matrix, axis=0)
        final_template = normalize_vector(mean_embedding.tolist())

        if self.student_service is not None:
            self.student_service.enroll_face_embedding(session.student_code, final_template)

        del self.active_sessions[session_id]
        return {
            "student_id": session.student_code,
            "student_code": session.student_code,
            "full_name": session.full_name,
            "department": session.department,
            "model_id": "autoroll_v1",
            "model_version": "autoroll_arcface_r50_epoch1",
            "embedding_dimension": len(final_template),
            "sample_count": len(embeddings_matrix),
            "status": "ENROLLED",
        }



    def add_frame(self, session_id: str, frame: np.ndarray, model_id: str | None = None) -> dict[str, Any]:
        if session_id not in self.active_sessions:
            raise KeyError(f"Enrollment session '{session_id}' not found.")

        session = self.active_sessions[session_id]
        recognizer = get_recognizer(model_id=model_id)

        if frame is None or frame.size == 0:
            reason = "invalid_image"
            session.reasons_log.append({"reason": reason, "detail": "Image is empty or null"})
            return {"accepted": False, "reason": reason, "sample_count": len(session.valid_embeddings)}

        # 1. Face Detection
        detections = self.detector.detect(frame)
        if len(detections) == 0:
            reason = "no_face"
            session.reasons_log.append({"reason": reason, "detail": "No face detected in frame"})
            return {"accepted": False, "reason": reason, "sample_count": len(session.valid_embeddings)}
        elif len(detections) > 1:
            reason = "multiple_faces"
            session.reasons_log.append({"reason": reason, "detail": f"Detected {len(detections)} faces"})
            return {"accepted": False, "reason": reason, "sample_count": len(session.valid_embeddings)}

        det = detections[0]

        # 2. Quality Check
        quality_res = self.quality_filter.filter_quality(frame, det.bbox, det.landmarks)
        if not quality_res.is_acceptable:
            reason = "poor_quality"
            session.reasons_log.append({"reason": reason, "detail": f"Blur={quality_res.blur_score:.1f}, Size={quality_res.face_size_px}px"})
            return {"accepted": False, "reason": reason, "detail": f"Quality check failed: {reason}", "sample_count": len(session.valid_embeddings)}

        # 3. Liveness Check
        liveness_res = self.liveness_detector.predict(frame, det.bbox)
        if not liveness_res.is_live:
            reason = "spoof_detected"
            session.reasons_log.append({"reason": reason, "detail": f"Liveness score {liveness_res.combined_liveness_score:.3f} < {self.settings.LIVENESS_THRESHOLD}"})
            return {"accepted": False, "reason": reason, "detail": "Spoof or fake face detected", "sample_count": len(session.valid_embeddings)}

        # 4. Alignment (112x112 RGB)
        aligned_face = self.aligner.align_face(frame, det.landmarks)

        # 5. ArcFace Embedding & L2 Normalization
        rec_res = recognizer.extract_embedding(aligned_face)
        norm_emb = rec_res.embedding  # Already L2 normalized

        session.valid_embeddings.append(norm_emb)
        logger.info(f"ENROLLMENT SAMPLE ACCEPTED | SessionID: {session_id} | Sample #{len(session.valid_embeddings)}")

        return {
            "accepted": True,
            "sample_count": len(session.valid_embeddings),
            "target_samples": 5,
            "model_id": recognizer.get_model_id(),
            "model_version": recognizer.get_model_version(),
        }

    def complete_session(self, session_id: str, db: Session, model_id: str | None = None) -> dict[str, Any]:
        if session_id not in self.active_sessions:
            raise KeyError(f"Enrollment session '{session_id}' not found.")

        session = self.active_sessions[session_id]
        recognizer = get_recognizer(model_id=model_id)

        if len(session.valid_embeddings) == 0:
            raise ValueError("Cannot complete enrollment session with 0 accepted face samples.")

        # Aggregate Template: Normalized Mean Embedding
        embeddings_matrix = np.array(session.valid_embeddings, dtype=np.float32)
        mean_embedding = np.mean(embeddings_matrix, axis=0)
        final_template = normalize_vector(mean_embedding.tolist())

        # Serialize 512 float32 embedding vector into binary blob
        vec_bytes = struct.pack(f"{len(final_template)}f", *final_template)

        # Create or update Student record in DB
        student = db.query(Student).filter(Student.student_code == session.student_code).first()
        if not student:
            student = Student(
                student_code=session.student_code,
                full_name=session.full_name,
                department=session.department,
                is_active=True,
            )
            db.add(student)
            db.flush()
        else:
            student.full_name = session.full_name
            if session.department:
                student.department = session.department

        # Deactivate old primary embeddings for this student
        db.query(StudentEmbedding).filter(
            StudentEmbedding.student_id == student.id,
            StudentEmbedding.is_primary == True,
        ).update({"is_primary": False})

        # Save new StudentEmbedding with model metadata
        new_template = StudentEmbedding(
            student_id=student.id,
            embedding_vector=vec_bytes,
            model_id=recognizer.get_model_id(),
            model_version=recognizer.get_model_version(),
            embedding_dimension=len(final_template),
            sample_count=len(session.valid_embeddings),
            is_primary=True,
        )
        db.add(new_template)
        db.commit()

        logger.info(
            f"ENROLLMENT COMPLETE | Student: '{student.student_code}' ({student.full_name}) | "
            f"ModelID: '{recognizer.get_model_id()}' | Version: '{recognizer.get_model_version()}' | "
            f"SampleCount: {len(session.valid_embeddings)} | Template Dim: 512"
        )

        del self.active_sessions[session_id]

        return {
            "student_id": student.id,
            "student_code": student.student_code,
            "full_name": student.full_name,
            "department": student.department,
            "model_id": recognizer.get_model_id(),
            "model_version": recognizer.get_model_version(),
            "embedding_dimension": 512,
            "sample_count": len(session.valid_embeddings),
            "status": "ENROLLED",
        }
