"""
Independent AutoRoll Anti-Spoofing Liveness Pipeline.
Coordinates: Face Crop -> Spatial FAS -> Temporal Aggregation -> Liveness Decision.
"""

import time

import numpy as np

from autoroll.common.config import get_settings
from autoroll.common.logger import get_logger
from autoroll.common.schemas import LivenessResult
from autoroll.ml.liveness.base import BaseLivenessDetector
from autoroll.ml.liveness.passive_fas import PassiveAntiSpoofingModel
from autoroll.ml.liveness.temporal_analyzer import TemporalLivenessAggregator

logger = get_logger("liveness_pipeline")
settings = get_settings()


class LivenessPipeline(BaseLivenessDetector):
    """
    Independent Presentation Attack Detection (PAD) Pipeline.
    Supports single-frame evaluation and sliding temporal sequence evaluation.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        liveness_threshold: float | None = None,
        temporal_window: int | None = None,
    ):
        pad_path = model_path or settings.PAD_MODEL_PATH
        self.threshold = (
            liveness_threshold
            if liveness_threshold is not None
            else settings.LIVENESS_THRESHOLD
        )
        window_size = (
            temporal_window
            if temporal_window is not None
            else settings.TEMPORAL_WINDOW_SIZE
        )

        self.spatial_model = PassiveAntiSpoofingModel(
            model_path=pad_path, device=device, model_version=settings.PAD_MODEL_VERSION
        )
        self.temporal_aggregator = TemporalLivenessAggregator(window_size=window_size)
        self.device = self.spatial_model.device

        logger.info(
            f"Liveness Anti-Spoofing Pipeline initialized on device '{self.device}' "
            f"(Threshold: {self.threshold}, Temporal Window: {window_size})"
        )

    def predict(self, face_chip: np.ndarray) -> LivenessResult:
        """
        Evaluates a single face crop for presentation attacks.
        """
        start_time = time.perf_counter()

        spatial_score = self.spatial_model.predict_spatial_score(face_chip)
        aggregated_score = self.temporal_aggregator.update(spatial_score, face_chip)

        is_live = aggregated_score >= self.threshold
        decision_str = "REAL" if is_live else "SPOOF"
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        ml_score = spatial_score if self.spatial_model.session is not None else 0.0
        heuristic_score = spatial_score if self.spatial_model.session is None else 0.0

        return LivenessResult(
            is_live=is_live,
            liveness_score=aggregated_score,
            ml_liveness_score=round(ml_score, 4),
            auxiliary_heuristic_score=round(heuristic_score, 4),
            combined_liveness_score=round(aggregated_score, 4),
            method="passive_mini_fas_temporal",
            details={
                "spatial_score": round(spatial_score, 4),
                "aggregated_temporal_score": round(aggregated_score, 4),
                "decision": decision_str,
                "latency_ms": round(latency_ms, 2),
                "ml_mode": "ML_ONNX_MODEL" if self.spatial_model.session else "AUXILIARY_HEURISTIC",
                "model_version": settings.PAD_MODEL_VERSION,
            },
        )

    def predict_sequence(self, face_chips: list[np.ndarray]) -> LivenessResult:
        """
        Evaluates a sequence of multi-frame face crops.
        """
        start_time = time.perf_counter()

        if not face_chips:
            raise ValueError("Cannot evaluate empty sequence of face chips.")

        spatial_scores = [
            self.spatial_model.predict_spatial_score(chip) for chip in face_chips
        ]
        aggregated_score = self.temporal_aggregator.aggregate_sequence(spatial_scores)

        is_live = aggregated_score >= self.threshold
        decision_str = "REAL" if is_live else "SPOOF"
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return LivenessResult(
            is_live=is_live,
            liveness_score=aggregated_score,
            method="passive_mini_fas_temporal_seq",
            details={
                "avg_spatial_score": round(float(np.mean(spatial_scores)), 4),
                "temporal_score": aggregated_score,
                "decision": decision_str,
                "latency_ms": round(latency_ms, 2),
                "frame_count": len(face_chips),
                "model_version": settings.PAD_MODEL_VERSION,
            },
        )
