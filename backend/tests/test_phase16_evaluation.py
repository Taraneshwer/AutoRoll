"""
Unit & Integration Tests for AutoRoll Phase 16 Real-World Evaluation Framework.
"""

import json
import pytest
from pathlib import Path

from scripts.evaluation.prepare_real_world_eval import generate_eval_manifest, DATASET_ROOT
from scripts.evaluation.statistical_analysis import (
    compute_eer_and_auc,
    compute_fisher_d_prime,
    bootstrap_confidence_interval,
    compute_statistical_significance,
)
from scripts.evaluation.generate_phase16_report import (
    verify_model_checksum,
    MODEL_ONNX_PATH,
    EXPECTED_SHA256,
)


def test_eval_dataset_manifest_generation():
    manifest = generate_eval_manifest(100)
    assert manifest["total_participants"] == 100
    assert manifest["calibration_participants"] == 50
    assert manifest["test_participants"] == 50
    assert manifest["provenance"]["disjoint_from_training"] is True
    assert len(manifest["participants"]) == 100
    assert manifest["participants"][0]["split"] == "CALIBRATION"
    assert manifest["participants"][99]["split"] == "TEST"


def test_fisher_d_prime_calculation():
    genuine = [0.8, 0.85, 0.9, 0.78, 0.82]
    impostor = [0.1, 0.15, 0.2, 0.12, 0.18]
    d_prime = compute_fisher_d_prime(genuine, impostor)
    assert d_prime > 10.0


def test_eer_and_auc_calculation():
    genuine = [0.7, 0.75, 0.8, 0.85, 0.9]
    impostor = [0.1, 0.15, 0.2, 0.25, 0.3]
    eer, opt_t, auc = compute_eer_and_auc(genuine, impostor)
    assert eer == 0.0
    assert auc == 1.0


def test_bootstrap_confidence_interval():
    genuine = [0.7, 0.75, 0.8, 0.85, 0.9] * 10
    impostor = [0.1, 0.15, 0.2, 0.25, 0.3] * 10
    ci = bootstrap_confidence_interval(genuine, impostor, n_bootstraps=50)
    assert "eer_ci" in ci
    assert "auc_ci" in ci
    assert ci["auc_ci"][0] <= ci["auc_ci"][1]


def test_statistical_significance_calculation():
    scores_a = [0.5, 0.52, 0.48, 0.51, 0.49]
    scores_b = [0.7, 0.72, 0.68, 0.71, 0.69]
    sig = compute_statistical_significance(scores_a, scores_b)
    assert sig["statistically_significant"] is True
    assert sig["mean_difference"] > 0.15


def test_model_sha256_checksum():
    assert verify_model_checksum(MODEL_ONNX_PATH, EXPECTED_SHA256) is True
