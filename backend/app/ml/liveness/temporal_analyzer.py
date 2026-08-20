"""
Temporal Liveness Aggregator and Sliding Window Motion Analyzer.
"""

import numpy as np

from app.core.logger import get_logger

logger = get_logger("temporal_liveness")


class TemporalLivenessAggregator:
    """
    Sliding window temporal aggregator for multi-frame liveness decision.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.spatial_scores_buffer: list[float] = []
        self.frame_diffs_buffer: list[float] = []

    def reset(self) -> None:
        self.spatial_scores_buffer.clear()
        self.frame_diffs_buffer.clear()

    def update(self, spatial_score: float, face_chip: np.ndarray | None = None) -> float:
        """
        Updates sliding buffer with new frame spatial score and optional face chip.
        Returns aggregated temporal liveness probability (0.0 to 1.0).
        """
        self.spatial_scores_buffer.append(spatial_score)
        if len(self.spatial_scores_buffer) > self.window_size:
            self.spatial_scores_buffer.pop(0)

        # Calculate mean spatial score over temporal window
        avg_spatial = float(np.mean(self.spatial_scores_buffer))

        # Exponential recency weighting
        weights = np.exp(np.linspace(-1.0, 0.0, len(self.spatial_scores_buffer)))
        weights /= np.sum(weights)
        weighted_spatial = float(np.sum(np.array(self.spatial_scores_buffer) * weights))

        # Final aggregated score
        final_liveness_score = 0.5 * avg_spatial + 0.5 * weighted_spatial
        return round(float(np.clip(final_liveness_score, 0.0, 1.0)), 4)

    def aggregate_sequence(self, spatial_scores: list[float]) -> float:
        """
        Aggregates a complete pre-recorded sequence of spatial scores.
        """
        if not spatial_scores:
            return 0.0
        scores_arr = np.array(spatial_scores, dtype=np.float32)
        avg_score = float(np.mean(scores_arr))
        min_score = float(np.min(scores_arr))

        # If any frame in sequence is a blatant spoof, penalize total score
        aggregated = 0.7 * avg_score + 0.3 * min_score
        return round(float(np.clip(aggregated, 0.0, 1.0)), 4)
