"""
Attendance Processing Engine.
Performs vector cosine similarity matching against enrolled student embeddings,
applies similarity & liveness thresholds, handles deduplication, and broadcasts events.
"""

from typing import Any

import numpy as np

from autoroll.common.logger import get_logger
from server.app.db.models import Student
from server.app.repositories.attendance_repository import AttendanceRepository
from server.app.repositories.student_repository import StudentRepository
from server.app.websockets.manager import ws_manager

logger = get_logger("attendance_service")


class AttendanceService:
    """
    Control Plane Attendance Verification Service.
    Performs fast float32 vector matching without running heavy CNN inference models.
    """

    def __init__(
        self,
        attendance_repo: AttendanceRepository,
        student_repo: StudentRepository,
        similarity_threshold: float = 0.65,
        liveness_threshold: float = 0.90,
        deduplication_window_sec: int = 300,
        min_track_frames: int = 1,
    ):
        self.attendance_repo = attendance_repo
        self.student_repo = student_repo
        self.similarity_threshold = similarity_threshold
        self.liveness_threshold = liveness_threshold
        self.deduplication_window_sec = deduplication_window_sec
        self.min_track_frames = min_track_frames

    def process_recognition_event(
        self, event_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Ingests recognition event from worker and evaluates complete Phase 12 attendance rules:
        1. Liveness check -> SPOOF_ATTEMPT if liveness < threshold
        2. Temporal consistency check -> TEMPORAL_INSUFFICIENT if frame_index < min_track_frames
        3. Vector matching -> UNKNOWN_PERSON if similarity < threshold
        4. Deduplication window check -> DUPLICATE_SUPPRESSED if checked in recently
        5. Attendance Confirmation -> Record & Broadcast
        """
        worker_id = event_payload.get("worker_id")
        camera_id = event_payload.get("camera_id")
        frame_index = event_payload.get("frame_index", 1)
        query_embedding = event_payload.get("embedding")
        liveness_score = float(event_payload.get("liveness_score", 0.0))
        min_track_frames = event_payload.get("min_track_frames", self.min_track_frames)

        if not query_embedding or len(query_embedding) != 512:
            logger.warning("Invalid or missing 512-d embedding in recognition event.")
            return {"status": "invalid_payload", "reason": "Missing or invalid 512-d embedding"}

        # Rule 1: Liveness Anti-Spoofing Check
        if liveness_score < self.liveness_threshold:
            logger.warning(
                f"SPOOF ATTEMPT detected on Camera '{camera_id}' (Worker '{worker_id}'): "
                f"Liveness {liveness_score:.3f} < {self.liveness_threshold}."
            )
            self.attendance_repo.log_analytics_event(
                event_type="SPOOF_ATTEMPT",
                similarity_score=0.0,
                liveness_score=liveness_score,
                camera_id=camera_id,
                worker_id=worker_id,
            )
            return {
                "status": "rejected",
                "event_type": "SPOOF_ATTEMPT",
                "reason": (
                    f"Liveness score {liveness_score:.2f} failed threshold "
                    f"({self.liveness_threshold})"
                ),
            }

        # Rule 2: Temporal Consistency Check
        if frame_index < min_track_frames:
            logger.info(
                f"Recognition event skipped: Temporal consistency insufficient "
                f"(Frame {frame_index} < {min_track_frames})."
            )
            return {
                "status": "skipped",
                "event_type": "TEMPORAL_INSUFFICIENT",
                "reason": f"Track frame index {frame_index} below minimum ({min_track_frames})",
            }

        # Rule 3: Recognition Vector Matching
        best_match = self._match_vector(query_embedding)
        if not best_match:
            logger.info(f"UNKNOWN PERSON detected on Camera '{camera_id}' (Worker '{worker_id}').")
            self.attendance_repo.log_analytics_event(
                event_type="UNKNOWN_PERSON",
                similarity_score=0.0,
                liveness_score=liveness_score,
                camera_id=camera_id,
                worker_id=worker_id,
            )
            return {
                "status": "rejected",
                "event_type": "UNKNOWN_PERSON",
                "reason": (
                    "No student vector matched above threshold "
                    f"({self.similarity_threshold})"
                ),
            }

        student, similarity_score = best_match

        # Rule 4: Deduplication Check
        recent_checkin = self.attendance_repo.get_recent_checkin(
            student_id=student.id, window_seconds=self.deduplication_window_sec
        )
        if recent_checkin:
            logger.info(
                f"Duplicate check-in suppressed for Student '{student.full_name}' "
                f"({self.deduplication_window_sec}s deduplication window)."
            )
            self.attendance_repo.log_analytics_event(
                event_type="DUPLICATE_SUPPRESSED",
                similarity_score=similarity_score,
                liveness_score=liveness_score,
                student_id=student.id,
                camera_id=camera_id,
                worker_id=worker_id,
            )
            return {
                "status": "suppressed_duplicate",
                "event_type": "DUPLICATE_SUPPRESSED",
                "student_id": student.id,
                "student_name": student.full_name,
            }

        # Rule 5: Save Attendance Record & Broadcast
        record = self.attendance_repo.create(
            student_id=student.id,
            camera_id=camera_id,
            worker_id=worker_id,
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            model_version="iresnet50_arcface_v1",
            verification_status="CONFIRMED",
        )

        self.attendance_repo.log_analytics_event(
            event_type="ATTENDANCE_CONFIRMED",
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            student_id=student.id,
            camera_id=camera_id,
            worker_id=worker_id,
        )

        event_data = {
            "status": "success",
            "event_type": "ATTENDANCE_CONFIRMED",
            "attendance_id": record.id,
            "student_id": student.id,
            "student_code": student.student_code,
            "full_name": student.full_name,
            "camera_id": camera_id,
            "worker_id": worker_id,
            "timestamp": record.timestamp.isoformat(),
            "similarity_score": round(similarity_score, 4),
            "liveness_score": round(liveness_score, 4),
        }

        ws_manager.broadcast_sync("ATTENDANCE_CONFIRMED", event_data)
        logger.info(
            f"Attendance CONFIRMED for Student '{student.full_name}' "
            f"(Similarity: {similarity_score:.4f}, Liveness: {liveness_score:.4f})"
        )
        return event_data

    def _match_vector(
        self, query_emb: list[float]
    ) -> tuple[Student, float] | None:
        q_vec = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-10

        all_embeddings = self.student_repo.get_all_embeddings()
        best_student: Student | None = None
        best_sim = -1.0

        for emb_obj in all_embeddings:
            try:
                ref_vec = np.frombuffer(emb_obj.embedding_vector, dtype=np.float32)
                if len(ref_vec) != 512:
                    continue

                r_norm = np.linalg.norm(ref_vec) + 1e-10
                sim = float(np.dot(q_vec, ref_vec) / (q_norm * r_norm))

                if sim > best_sim and sim >= self.similarity_threshold:
                    best_sim = sim
                    best_student = emb_obj.student
            except Exception as e:
                logger.warning(f"Error parsing student embedding bytes: {e}")

        if best_student:
            return best_student, best_sim
        return None
