"""
Unit tests for end-to-end dataset preprocessing pipeline.
"""

import os
import shutil

import pytest

from autoroll.ml.preprocessing.dataset_loader import DatasetConfig
from scripts.prepare_dataset import run_pipeline
from scripts.validate_dataset import validate_dataset


@pytest.fixture
def temp_dataset_dir(tmp_path):
    raw_dir = str(tmp_path / "raw")
    out_dir = str(tmp_path / "processed")
    yield raw_dir, out_dir
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir, ignore_errors=True)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)


def test_end_to_end_dataset_pipeline(temp_dataset_dir):
    raw_dir, out_dir = temp_dataset_dir

    config = DatasetConfig(
        dataset_name="test_synthetic_pipeline",
        raw_data_dir=raw_dir,
        output_dir=out_dir,
        min_face_size=20,
        min_blur_score=1.0,
        min_detection_confidence=0.4,
        resumable=True,
    )

    # Execute pipeline on synthetic test subset
    run_pipeline(config, use_synthetic=True)

    # Verify report output file exists
    report_path = os.path.join(out_dir, "dataset_report.json")
    assert os.path.exists(report_path)

    # Validate identity disjointness and resolution of aligned chips
    is_valid = validate_dataset(out_dir)
    assert is_valid is True
