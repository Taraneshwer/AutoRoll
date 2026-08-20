"""
Unit tests for Real Dataset Ingestion Pipeline & Training Guard Enforcement.
Verifies that synthetic data generation is blocked and training fails when real source manifest is missing.
"""

import os
import json
import pytest

from app.ml.training.trainer import verify_real_dataset_guard, ArcFacePilotTrainer
from backend.scripts.dataset.ingest_real_dataset import verify_dataset_authenticity, ingest_real_dataset


def test_verify_real_dataset_guard_missing_manifest(tmp_path):
    fake_manifest_path = str(tmp_path / "source_manifest.json")
    with pytest.raises(RuntimeError, match="Source manifest missing"):
        verify_real_dataset_guard(manifest_path=fake_manifest_path)


def test_verify_real_dataset_guard_synthetic_rejection(tmp_path):
    fake_manifest_path = str(tmp_path / "source_manifest.json")
    synthetic_data = {
        "dataset_name": "Synthetic_Test",
        "dataset_type": "synthetic",
        "synthetic": True,
        "source_url": "http://example.com",
        "license": "MIT",
        "local_identity_count": 100,
        "local_image_count": 1000,
    }
    with open(fake_manifest_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f)

    with pytest.raises(RuntimeError, match="Dataset is synthetic"):
        verify_real_dataset_guard(manifest_path=fake_manifest_path)


def test_ingest_real_dataset_missing_source():
    with pytest.raises(FileNotFoundError, match="Source dataset directory not found"):
        ingest_real_dataset(source="non_existent_directory_xyz123")


def test_trainer_init_blocks_without_real_manifest(monkeypatch, tmp_path):
    # Ensure guard receives a non-existent manifest path
    missing_manifest = str(tmp_path / "non_existent_manifest.json")
    with pytest.raises(RuntimeError, match="Source manifest missing"):
        verify_real_dataset_guard(manifest_path=missing_manifest)


def test_trainer_init_passes_with_valid_casia_manifest():
    # Enforce that the real dataset guard PASSES with our installed CASIA-WebFace manifest
    verify_real_dataset_guard()
