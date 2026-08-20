"""
Automated Threshold Calibration Module.
Calibrates recognition similarity threshold from validation dataset.
"""

import os

import numpy as np
import yaml
from pydantic import BaseModel

from autoroll.common.logger import get_logger
from autoroll.ml.evaluation.metrics import VerificationMetricsCalculator

logger = get_logger("threshold_calibrator")


class CalibratedThreshold(BaseModel):
    model_version: str
    threshold: float
    criterion: str
    far_at_threshold: float
    frr_at_threshold: float
    f1_at_threshold: float
    calibrated_on_samples: int


class ThresholdCalibrator:
    """
    Calibrates recognition similarity threshold from validation genuine and impostor scores.
    """

    def __init__(self, criterion: str = "eer", target_far: float = 0.001):
        self.criterion = criterion.lower()
        self.target_far = target_far

    def calibrate(
        self,
        genuine_scores: list[float],
        impostor_scores: list[float],
        model_version: str = "arcface_iresnet50_v1",
    ) -> CalibratedThreshold:
        gen_arr = np.array(genuine_scores, dtype=np.float32)
        imp_arr = np.array(impostor_scores, dtype=np.float32)

        if len(gen_arr) == 0 or len(imp_arr) == 0:
            raise ValueError("Cannot calibrate threshold with empty genuine or impostor scores.")

        min_score = min(float(np.min(gen_arr)), float(np.min(imp_arr)))
        max_score = max(float(np.max(gen_arr)), float(np.max(imp_arr)))
        thresholds = np.linspace(min_score, max_score, 1000)

        best_thresh = 0.65
        best_far = 0.0
        best_frr = 0.0
        best_f1 = 0.0

        if self.criterion == "eer":
            eer, eer_thresh = VerificationMetricsCalculator.calculate_eer(gen_arr, imp_arr)
            best_thresh = eer_thresh
            best_far = float(np.sum(imp_arr >= best_thresh) / len(imp_arr))
            best_frr = float(np.sum(gen_arr < best_thresh) / len(gen_arr))
            tp = int(np.sum(gen_arr >= best_thresh))
            fp = int(np.sum(imp_arr >= best_thresh))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, len(gen_arr))
            best_f1 = 2 * precision * recall / max(1e-7, precision + recall)

        elif self.criterion == "max_f1":
            max_f1_val = -1.0
            for t in thresholds:
                tp = int(np.sum(gen_arr >= t))
                fp = int(np.sum(imp_arr >= t))
                fn = int(np.sum(gen_arr < t))

                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-7, prec + rec)

                if f1 > max_f1_val:
                    max_f1_val = f1
                    best_thresh = float(t)
                    best_far = float(fp / len(imp_arr))
                    best_frr = float(fn / len(gen_arr))
                    best_f1 = float(f1)

        elif self.criterion == "target_far":
            sorted_imp = np.sort(imp_arr)[::-1]
            idx = int(self.target_far * len(sorted_imp))
            idx = min(max(0, idx), len(sorted_imp) - 1)
            best_thresh = float(sorted_imp[idx])
            best_far = float(np.sum(imp_arr >= best_thresh) / len(imp_arr))
            best_frr = float(np.sum(gen_arr < best_thresh) / len(gen_arr))
            tp = int(np.sum(gen_arr >= best_thresh))
            fp = int(np.sum(imp_arr >= best_thresh))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, len(gen_arr))
            best_f1 = 2 * prec * rec / max(1e-7, prec + rec)

        calibrated = CalibratedThreshold(
            model_version=model_version,
            threshold=round(best_thresh, 4),
            criterion=self.criterion,
            far_at_threshold=round(best_far, 4),
            frr_at_threshold=round(best_frr, 4),
            f1_at_threshold=round(best_f1, 4),
            calibrated_on_samples=len(gen_arr) + len(imp_arr),
        )

        logger.info(
            f"Calibrated Threshold: {calibrated.threshold} (Criterion: '{calibrated.criterion}', "
            f"FAR: {calibrated.far_at_threshold}, FRR: {calibrated.frr_at_threshold}, "
            f"F1: {calibrated.f1_at_threshold})"
        )
        return calibrated

    def save_calibration_yaml(self, calibrated: CalibratedThreshold, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = calibrated.model_dump()
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Saved Calibrated Threshold Configuration to '{save_path}'")
        return save_path
