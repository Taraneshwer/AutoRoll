"""
Unit tests for PAD evaluation metrics (APCER, BPCER, ACER).
"""

from autoroll.ml.liveness.evaluation import PADEvaluationReport, PADEvaluator


def test_pad_evaluation_metrics():
    # 10 real faces (all high liveness >= 0.90)
    real_scores = [0.95, 0.92, 0.98, 0.94, 0.91, 0.96, 0.93, 0.97, 0.95, 0.99]

    # Attack categories
    attack_scores = {
        "printed_attack": [0.10, 0.15, 0.20, 0.12, 0.18],  # All correctly rejected as spoofs
        "photo_replay": [0.25, 0.30, 0.28, 0.22],          # All correctly rejected as spoofs
        "video_replay": [0.35, 0.40, 0.32, 0.38],          # All correctly rejected as spoofs
    }

    report = PADEvaluator.evaluate(real_scores, attack_scores, threshold=0.90)

    assert isinstance(report, PADEvaluationReport)
    assert report.bpcer == 0.0  # Zero false rejections of real faces
    assert report.apcer == 0.0  # Zero false acceptances of spoofs
    assert report.acer == 0.0   # ACER = (0 + 0) / 2
    assert len(report.category_breakdown) == 3
