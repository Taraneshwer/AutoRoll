"""
Unit tests for Verification Metrics and Threshold Calibration modules.
"""

import os

from autoroll.ml.evaluation.metrics import VerificationMetricsCalculator
from autoroll.ml.evaluation.threshold import ThresholdCalibrator


def test_verification_metrics_calculation():
    # Synthetic genuine scores (high ~0.8) and impostor scores (low ~0.2)
    genuine_scores = [0.75, 0.82, 0.88, 0.91, 0.68, 0.79]
    impostor_scores = [0.12, 0.25, 0.18, 0.31, 0.22, 0.15]

    metrics = VerificationMetricsCalculator.compute_metrics(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold=0.50,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.far == 0.0
    assert metrics.frr == 0.0
    assert len(metrics.tar_at_far) > 0


def test_threshold_calibration(tmp_path):
    genuine_scores = [0.70, 0.80, 0.85, 0.90]
    impostor_scores = [0.10, 0.20, 0.30, 0.40]

    calibrator = ThresholdCalibrator(criterion="eer")
    calibrated = calibrator.calibrate(genuine_scores, impostor_scores)

    assert 0.40 <= calibrated.threshold <= 0.70
    assert calibrated.criterion == "eer"

    yaml_path = str(tmp_path / "threshold.yaml")
    saved_path = calibrator.save_calibration_yaml(calibrated, yaml_path)
    assert os.path.exists(saved_path)
