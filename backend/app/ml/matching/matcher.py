"""
Face Matcher Vector Engine for AutoRoll.
Performs cosine similarity search against enrolled face templates.
Strictly returns identity candidate metadata without exposing raw embedding vectors.
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.config import get_settings
from app.core.crypto import normalize_vector
from app.core.logger import get_logger

logger = get_logger("face_matcher")
settings = get_settings()


class MatchResult:
    def __init__(
        self,
        candidate_student_id: Optional[str],
        similarity: float,
        matched: bool,
        model_id: str,
        model_version: str,
    ):
        self.candidate_student_id = candidate_student_id
        self.similarity = round(float(similarity), 4)
        self.matched = matched
        self.model_id = model_id
        self.model_version = model_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_student_id": self.candidate_student_id,
            "similarity": self.similarity,
            "matched": self.matched,
            "model_id": self.model_id,
            "model_version": self.model_version,
        }


class FaceMatcher:
    def __init__(
        self,
        model_id: str = "autoroll_v1",
        threshold: Optional[float] = None,
    ):
        self.model_id = model_id
        self.threshold = threshold if threshold is not None else getattr(settings, "AUTOROLL_RECOGNITION_THRESHOLD", 0.0540)
        self.enrolled_templates: Dict[str, np.ndarray] = {}  # student_id -> normalized vector
        self.model_version = "autoroll_arcface_r50_epoch1" if model_id == "autoroll_v1" else "arcface_r50_v1"

    def register_template(self, student_id: str, template_vector: List[float]) -> None:
        norm_vec = np.array(normalize_vector(template_vector), dtype=np.float32)
        self.enrolled_templates[student_id] = norm_vec
        logger.info(f"Registered template for student '{student_id}' (Dim: {len(template_vector)}).")

    def match_embedding(self, query_embedding: List[float]) -> MatchResult:
        if not self.enrolled_templates:
            return MatchResult(
                candidate_student_id=None,
                similarity=0.0,
                matched=False,
                model_id=self.model_id,
                model_version=self.model_version,
            )

        q_vec = np.array(normalize_vector(query_embedding), dtype=np.float32)
        best_student_id: Optional[str] = None
        best_similarity = -1.0

        for student_id, t_vec in self.enrolled_templates.items():
            sim = float(np.dot(q_vec, t_vec))
            if sim > best_similarity:
                best_similarity = sim
                best_student_id = student_id

        matched = best_similarity >= self.threshold
        candidate_id = best_student_id if matched else None

        return MatchResult(
            candidate_student_id=candidate_id,
            similarity=best_similarity,
            matched=matched,
            model_id=self.model_id,
            model_version=self.model_version,
        )
