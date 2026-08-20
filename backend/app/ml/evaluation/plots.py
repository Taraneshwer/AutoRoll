"""
Evaluation Data Formatters and Distribution Histogram Binner.
Generates structured plot datasets for ROC curves and similarity distribution charts.
"""

import numpy as np
from pydantic import BaseModel


class ROCPlotData(BaseModel):
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]


class DistributionHistogram(BaseModel):
    bins: list[float]
    genuine_counts: list[int]
    impostor_counts: list[int]


class EvaluationPlotter:
    """
    Generates plottable JSON arrays for ROC curves and score distribution histograms.
    """

    @staticmethod
    def generate_roc_data(
        genuine_scores: list[float], impostor_scores: list[float], num_points: int = 100
    ) -> ROCPlotData:
        gen_arr = np.array(genuine_scores, dtype=np.float32)
        imp_arr = np.array(impostor_scores, dtype=np.float32)

        min_score = min(float(np.min(gen_arr)), float(np.min(imp_arr)))
        max_score = max(float(np.max(gen_arr)), float(np.max(imp_arr)))

        thresholds = np.linspace(min_score, max_score, num_points)[::-1]  # High to low
        fprs: list[float] = []
        tprs: list[float] = []

        for t in thresholds:
            fp = float(np.sum(imp_arr >= t))
            tp = float(np.sum(gen_arr >= t))

            fpr = fp / max(1, len(imp_arr))
            tpr = tp / max(1, len(gen_arr))

            fprs.append(round(fpr, 4))
            tprs.append(round(tpr, 4))

        return ROCPlotData(
            fpr=fprs,
            tpr=tprs,
            thresholds=[round(float(t), 4) for t in thresholds],
        )

    @staticmethod
    def generate_distribution_histogram(
        genuine_scores: list[float], impostor_scores: list[float], num_bins: int = 20
    ) -> DistributionHistogram:
        min_score = min(min(genuine_scores), min(impostor_scores))
        max_score = max(max(genuine_scores), max(impostor_scores))

        bins = np.linspace(min_score, max_score, num_bins + 1)
        gen_counts, _ = np.histogram(genuine_scores, bins=bins)
        imp_counts, _ = np.histogram(impostor_scores, bins=bins)

        return DistributionHistogram(
            bins=[round(float(b), 4) for b in bins],
            genuine_counts=[int(c) for c in gen_counts],
            impostor_counts=[int(c) for c in imp_counts],
        )
