"""
Face Verification Metrics Calculator.
Calculates ROC, FAR, FRR, EER, TAR@FAR, Accuracy, Precision, Recall, F1, and Latency.
"""

import numpy as np
from pydantic import BaseModel, Field

from app.core.logger import get_logger

logger = get_logger("verification_metrics")


class DistributionStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class TARAtFAR(BaseModel):
    far_target: float
    tar: float
    threshold: float


class VerificationMetrics(BaseModel):
    model_version: str
    total_pairs: int
    num_genuine: int
    num_impostor: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float
    frr: float
    eer: float
    eer_threshold: float
    tar_at_far: list[TARAtFAR] = Field(default_factory=list)
    genuine_stats: DistributionStats
    impostor_stats: DistributionStats
    avg_latency_ms: float


class VerificationMetricsCalculator:
    """
    Computes open-set face verification metrics from genuine and impostor similarity scores.
    """

    @staticmethod
    def compute_distribution_stats(scores: list[float]) -> DistributionStats:
        if not scores:
            return DistributionStats(mean=0.0, std=0.0, min=0.0, max=0.0)
        arr = np.array(scores, dtype=np.float32)
        return DistributionStats(
            mean=round(float(np.mean(arr)), 4),
            std=round(float(np.std(arr)), 4),
            min=round(float(np.min(arr)), 4),
            max=round(float(np.max(arr)), 4),
        )

    @staticmethod
    def compute_metrics(
        genuine_scores: list[float],
        impostor_scores: list[float],
        threshold: float,
        model_version: str = "arcface_v1",
        avg_latency_ms: float = 0.0,
    ) -> VerificationMetrics:
        gen_arr = np.array(genuine_scores, dtype=np.float32)
        imp_arr = np.array(impostor_scores, dtype=np.float32)

        n_gen = len(gen_arr)
        n_imp = len(imp_arr)
        n_total = n_gen + n_imp

        if n_gen == 0 or n_imp == 0:
            raise ValueError("Must provide non-empty genuine and impostor score lists.")

        # Scores above threshold -> predicted genuine (1), below threshold -> predicted impostor (0)
        tp = int(np.sum(gen_arr >= threshold))
        fn = int(np.sum(gen_arr < threshold))
        fp = int(np.sum(imp_arr >= threshold))
        tn = int(np.sum(imp_arr < threshold))

        accuracy = float((tp + tn) / max(1, n_total))
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float(2 * precision * recall / max(1e-7, precision + recall))

        far = float(fp / max(1, n_imp))  # False Accept Rate
        frr = float(fn / max(1, n_gen))  # False Reject Rate

        # EER Calculation over dynamic thresholds sweep
        eer, eer_thresh = VerificationMetricsCalculator.calculate_eer(gen_arr, imp_arr)

        # TAR@FAR calculation
        tar_far_list = VerificationMetricsCalculator.calculate_tar_at_far(
            gen_arr, imp_arr, [1e-1, 1e-2, 1e-3, 1e-4]
        )

        return VerificationMetrics(
            model_version=model_version,
            total_pairs=n_total,
            num_genuine=n_gen,
            num_impostor=n_imp,
            threshold=round(threshold, 4),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            eer=round(eer, 4),
            eer_threshold=round(eer_thresh, 4),
            tar_at_far=tar_far_list,
            genuine_stats=VerificationMetricsCalculator.compute_distribution_stats(genuine_scores),
            impostor_stats=VerificationMetricsCalculator.compute_distribution_stats(impostor_scores),
            avg_latency_ms=round(avg_latency_ms, 2),
        )

    @staticmethod
    def calculate_eer(
        genuine_scores: np.ndarray, impostor_scores: np.ndarray, num_steps: int = 1000
    ) -> tuple[float, float]:
        """
        Calculates Equal Error Rate (EER) where FAR == FRR.
        """
        min_score = min(float(np.min(genuine_scores)), float(np.min(impostor_scores)))
        max_score = max(float(np.max(genuine_scores)), float(np.max(impostor_scores)))

        thresholds = np.linspace(min_score, max_score, num_steps)
        best_eer = 1.0
        best_threshold = (min_score + max_score) / 2.0
        min_diff = 1.0

        for t in thresholds:
            far = np.sum(impostor_scores >= t) / len(impostor_scores)
            frr = np.sum(genuine_scores < t) / len(genuine_scores)
            diff = abs(far - frr)

            if diff < min_diff:
                min_diff = diff
                best_eer = (far + frr) / 2.0
                best_threshold = t

        return float(best_eer), float(best_threshold)

    @staticmethod
    def calculate_tar_at_far(
        genuine_scores: np.ndarray, impostor_scores: np.ndarray, far_targets: list[float]
    ) -> list[TARAtFAR]:
        """
        Calculates True Accept Rate (TAR = 1 - FRR) at specified False Accept Rate (FAR) targets.
        """
        results: list[TARAtFAR] = []
        sorted_imp = np.sort(impostor_scores)[::-1]  # Descending order

        for target in far_targets:
            # Find threshold that gives FAR <= target
            idx = int(target * len(sorted_imp))
            idx = min(max(0, idx), len(sorted_imp) - 1)
            t = float(sorted_imp[idx])

            tar = float(np.sum(genuine_scores >= t) / len(genuine_scores))
            results.append(
                TARAtFAR(far_target=target, tar=round(tar, 4), threshold=round(t, 4))
            )
        return results
