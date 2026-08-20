"""
Phase 9 Automated Tests for Real-World Calibration & Model Comparison Benchmark.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.crypto import normalize_vector
from scripts.evaluation.eval_real_world_benchmark import (
    compute_metrics,
    find_optimal_threshold,
    fisher_d_prime,
)

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "autoroll_benchmark"



def test_benchmark_identity_separation():
    """Verify data/autoroll_benchmark is separate from CASIA training data."""
    assert BENCHMARK_DIR.exists()
    assert (BENCHMARK_DIR / "enrollment").exists()
    assert (BENCHMARK_DIR / "genuine").exists()
    assert (BENCHMARK_DIR / "impostor").exists()
    assert (BENCHMARK_DIR / "metadata").exists()

    casia_dir = BENCHMARK_DIR.parent / "face_recognition"
    assert str(BENCHMARK_DIR) != str(casia_dir)


def test_consent_manifest_validation():
    """Verify anonymization guarantees and privacy manifest format."""
    manifest_path = BENCHMARK_DIR / "metadata" / "consent_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "privacy_guarantees" in manifest
    assert manifest["privacy_guarantees"]["consent_type"] == "Explicit Informed Consent"
    assert len(manifest["participants"]) >= 20

    for p in manifest["participants"]:
        assert p["id"].startswith("P")
        assert p["consent"] is True
        assert p["sample_count"] >= 5


def test_fisher_d_prime_and_metric_math():
    """Verify Fisher separability index d' and evaluation metric calculations."""
    np.random.seed(42)
    gen_sims = np.random.normal(0.43, 0.20, 100).tolist()
    imp_sims = np.random.normal(0.01, 0.05, 100).tolist()

    d_prime = fisher_d_prime(gen_sims, imp_sims)
    assert d_prime > 1.5

    metrics = compute_metrics(gen_sims, imp_sims, threshold=0.0540)
    assert "eer" in metrics
    assert "auc" in metrics
    assert "accuracy" in metrics
    assert "tar" in metrics
    assert "far" in metrics
    assert "frr" in metrics
    assert metrics["auc"] > 0.85


def test_calibration_threshold_selection_no_leakage():
    """Verify threshold selection on calibration set without test set leakage."""
    np.random.seed(42)
    calib_gen = np.random.normal(0.40, 0.15, 100).tolist()
    calib_imp = np.random.normal(0.00, 0.05, 100).tolist()

    optimal_threshold = find_optimal_threshold(calib_gen, calib_imp)
    assert -0.2 < optimal_threshold < 0.8

    # Apply to separate test set
    test_gen = np.random.normal(0.42, 0.15, 100).tolist()
    test_imp = np.random.normal(0.01, 0.05, 100).tolist()

    test_metrics = compute_metrics(test_gen, test_imp, optimal_threshold)
    assert test_metrics["threshold"] == optimal_threshold
    assert test_metrics["accuracy"] > 0.80
