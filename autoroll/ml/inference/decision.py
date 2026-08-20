"""
Unified Decision Engine Module combining Anti-Spoofing Liveness & Recognition.
"""

from autoroll.common.logger import get_logger
from autoroll.ml.inference.result import TrackedFaceResult
from autoroll.ml.inference.tracker import FaceTrack

logger = get_logger("decision_engine")


class UnifiedDecisionEngine:
    """
    Evaluates anti-spoofing and feature extraction status for final decisions.
    Never lets a single face processing failure crash or block the entire pipeline.
    """

    @staticmethod
    def evaluate_track_decision(
        track: FaceTrack,
        is_live: bool,
        liveness_score: float,
        liveness_decision: str,
        embedding: list[float] | None = None,
        recognition_error: str | None = None,
    ) -> TrackedFaceResult:
        """
        Processes model outputs for a track into a TrackedFaceResult.
        """
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
                # Retain cached embedding from previous recognition step
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
            # Fallback error object to ensure stream is never blocked
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
