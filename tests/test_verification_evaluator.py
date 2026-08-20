"""
Unit tests for Pair Generator, Verification Evaluator, and Report Builder.
"""

import os

from autoroll.ml.evaluation.plots import EvaluationPlotter
from autoroll.ml.evaluation.report import EvaluationReportBuilder


def test_plot_data_generation():
    genuine = [0.7, 0.8, 0.9]
    impostor = [0.1, 0.2, 0.3]

    roc_data = EvaluationPlotter.generate_roc_data(genuine, impostor, num_points=10)
    assert len(roc_data.fpr) == 10
    assert len(roc_data.tpr) == 10

    hist_data = EvaluationPlotter.generate_distribution_histogram(genuine, impostor, num_bins=5)
    assert len(hist_data.bins) == 6
    assert len(hist_data.genuine_counts) == 5


def test_report_builder_export(tmp_path):
    builder = EvaluationReportBuilder(experiment_id="test_exp")
    out_dir = str(tmp_path / "report")

    paths = builder.export_report(
        output_dir=out_dir,
        pretrained_metrics=None,
        finetuned_metrics=None,
        calibrated_threshold=0.65,
    )

    assert os.path.exists(paths["md"])
    assert os.path.exists(paths["json"])
