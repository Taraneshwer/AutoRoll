"""
Unified Decision Engine & Temporal Confirmation Tracker.
Evaluates detection quality, anti-spoofing liveness, model thresholding,
and sliding-window temporal observation history for final attendance decisions.
"""

import time
from typing import Dict, List, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger
from app.ml.inference.result import TrackedFaceResult
from app.ml.inference.tracker import FaceTrack

logger = get_logger("decision_engine")


class TemporalConfirmationTracker:
    """
    Sliding-window temporal confirmation tracker.
    Requires same identity to be recognized consistently across multiple frames
    within a short time window (e.g. 3 observations within 1500 ms).
    """

    def __init__(
        self,
        required_observations: int = 3,
        confirmation_window_ms: int = 1500,
    ):
        self.required_observations = required_observations
        self.confirmation_window_sec = confirmation_window_ms / 1000.0
        # Storage: track_id -> list of (timestamp_sec, student_id, similarity)
        self.history: Dict[int, List[Tuple[float, str, float]]] = {}

    def add_observation(
        self, track_id: int, student_id: str, similarity: float, timestamp: float | None = None
    ) -> Tuple[bool, int]:
        """
        Adds an identity observation for a track and returns (is_confirmed, observation_count).
        """
        now = timestamp or time.time()
        if track_id not in self.history:
            self.history[track_id] = []

        # Add new observation
        self.history[track_id].append((now, student_id, similarity))

        # Prune stale observations outside the confirmation window
        cutoff = now - self.confirmation_window_sec
        self.history[track_id] = [
            obs for obs in self.history[track_id] if obs[0] >= cutoff
        ]

        # Count observations matching the target student_id within window
        matching_count = sum(
            1 for obs in self.history[track_id] if obs[1] == student_id
        )

        is_confirmed = matching_count >= self.required_observations
        return is_confirmed, matching_count

    def clear_track(self, track_id: int) -> None:
        self.history.pop(track_id, None)


class UnifiedDecisionEngine:
    """
    Combines quality, liveness, model thresholding, and temporal confirmation
    to generate definitive attendance decisions.
    """

    def __init__(self, temporal_tracker: TemporalConfirmationTracker | None = None):
        settings = get_settings()
        self.temporal_tracker = temporal_tracker or TemporalConfirmationTracker(
            required_observations=settings.TEMPORAL_REQUIRED_OBSERVATIONS,
            confirmation_window_ms=settings.TEMPORAL_CONFIRMATION_WINDOW_MS,
        )

    def evaluate_attendance_decision(
        self,
        track_id: int,
        detection_confidence: float,
        is_quality_ok: bool,
        is_live: bool,
        liveness_score: float,
        student_id: str | None,
        similarity_score: float | None,
        recognition_threshold: float,
        model_id: str,
        template_model_id: str | None = None,
        timestamp: float | None = None,
    ) -> str:
        """
        Evaluates production attendance rules:
        - Face detected (confidence >= 0.5)
        - Face quality acceptable
        - Liveness passed (is_live == True)
        - Model compatibility matched (template.model_id == recognizer.model_id)
        - Similarity >= model recognition threshold
        - Temporal confirmation passed (>= 3 observations in 1500 ms)
        """
        if detection_confidence < 0.5:
            return "LOW_DETECTION_CONFIDENCE"

        if not is_quality_ok:
            return "LOW_QUALITY"

        if not is_live or liveness_score < 0.50:
            return "REJECTED_SPOOF"

        if student_id is None or similarity_score is None:
            return "UNKNOWN"

        # Model Version Compatibility Guard
        if template_model_id and template_model_id != model_id:
            logger.warning(
                f"MODEL MISMATCH REJECTION | Active Model: '{model_id}' != Template Model: '{template_model_id}'"
            )
            return "INCOMPATIBLE_MODEL_TEMPLATE"

        if similarity_score < recognition_threshold:
            return "INSUFFICIENT_CONFIDENCE"

        # Temporal Confirmation Check
        is_confirmed, count = self.temporal_tracker.add_observation(
            track_id=track_id,
            student_id=student_id,
            similarity=similarity_score,
            timestamp=timestamp,
        )

        if is_confirmed:
            return "PRESENT"
        return "PENDING_TEMPORAL"

    @staticmethod
    def evaluate_track_decision(
        track: FaceTrack,
        is_live: bool,
        liveness_score: float,
        liveness_decision: str,
        embedding: list[float] | None = None,
        recognition_error: str | None = None,
    ) -> TrackedFaceResult:
        try:
            track.is_live = is_live
            track.liveness_score = liveness_score
            track.liveness_decision = liveness_decision

            if embedding is not None:
                track.embedding = embedding
                track.recognition_status = "RECOGNIZED" if is_live else "SPOOF_REJECTED"
            elif recognition_error:
                track.recognition_status = f"FAILED ({recognition_error})"
            elif track.embedding is not None:
                track.recognition_status = "RECOGNIZED_CACHED"

            return TrackedFaceResult(
                track_id=track.track_id,
                bbox=track.bbox,
                detection_confidence=round(track.confidence, 4),
                landmarks=track.landmarks,
                embedding=track.embedding,
                is_live=track.is_live,
                liveness_score=round(track.liveness_score, 4),
                liveness_decision=track.liveness_decision,
                recognition_status=track.recognition_status,
                frames_tracked=track.frames_tracked,
            )

        except Exception as e:
            logger.error(f"Error evaluating decision for track {track.track_id}: {e}")
            return TrackedFaceResult(
                track_id=track.track_id,
                bbox=track.bbox,
                detection_confidence=round(track.confidence, 4),
                landmarks=track.landmarks,
                embedding=None,
                is_live=False,
                liveness_score=0.0,
                liveness_decision="ERROR",
                recognition_status="PROCESSING_ERROR",
                frames_tracked=track.frames_tracked,
            )
