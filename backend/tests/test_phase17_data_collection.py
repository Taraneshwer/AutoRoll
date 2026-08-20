"""
Pytest Test Suite for AutoRoll Phase 17.1 Real-World Data Collection System.
"""

import json
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.dataset.collect_real_world_evaluation import EvaluationDataCollector
from scripts.dataset.generate_evaluation_trial_pairs import generate_trial_pairs
from scripts.dataset.validate_real_world_eval_dataset import validate_real_world_eval_dataset



def make_dummy_image_bytes(color_var: float = 50.0) -> bytes:
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_data_collector_ingest_and_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        collector = EvaluationDataCollector(root_dir=tmp_path)

        img_bytes = make_dummy_image_bytes()
        record = collector.ingest_sample(
            image_bytes=img_bytes,
            participant_id="P001",
            sample_type="enrollment",
        )

        assert record["participant_id"] == "P001"
        assert record["sample_type"] == "enrollment"
        assert record["sha256"] in collector.seen_hashes

        # Test duplicate rejection
        with pytest.raises(ValueError, match="DUPLICATE IMAGE DETECTED"):
            collector.ingest_sample(
                image_bytes=img_bytes,
                participant_id="P001",
                sample_type="probe",
            )


def test_low_variance_image_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        collector = EvaluationDataCollector(root_dir=tmp_path)

        # Solid black image (variance = 0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        with pytest.raises(ValueError, match="color variance too low"):
            collector.ingest_sample(
                image_bytes=buf.tobytes(),
                participant_id="P001",
                sample_type="enrollment",
            )


def test_calibration_test_split_assignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        collector = EvaluationDataCollector(root_dir=tmp_path)

        # P001 -> CALIBRATION, P016 -> TEST
        collector.ingest_sample(make_dummy_image_bytes(), "P001", "enrollment")
        collector.ingest_sample(make_dummy_image_bytes(), "P016", "enrollment")

        manifest_file = tmp_path / "manifests" / "consent_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["P001"]["split"] == "CALIBRATION"
        assert manifest["P016"]["split"] == "TEST"


def test_validator_runs_cleanly():
    summary = validate_real_world_eval_dataset()
    assert "valid" in summary
    assert "total_image_files" in summary
