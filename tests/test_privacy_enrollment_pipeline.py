"""
Unit tests for PrivacyPreservingEnrollmentPipeline.
"""

import os

import cv2
import numpy as np

from autoroll.ml.enrollment.pipeline import PrivacyPreservingEnrollmentPipeline


def test_privacy_enrollment_pipeline(tmp_path):
    # Create 2 synthetic face images
    img1_path = str(tmp_path / "sample_1.jpg")
    img2_path = str(tmp_path / "sample_2.jpg")

    synthetic_frame = np.full((300, 300, 3), 120, dtype=np.uint8)
    cv2.rectangle(synthetic_frame, (80, 80), (220, 220), (200, 200, 200), -1)
    cv2.imwrite(img1_path, synthetic_frame)
    cv2.imwrite(img2_path, synthetic_frame)

    pipeline = PrivacyPreservingEnrollmentPipeline()

    result = pipeline.enroll(
        student_code="STU_TEST_001",
        full_name="Jane Test",
        sample_inputs=[img1_path, img2_path],
        delete_raw_images=True,
    )

    assert result.student_code == "STU_TEST_001"
    assert result.samples_processed == 2

    # Verify temp files deleted from disk for privacy
    assert not os.path.exists(img1_path)
    assert not os.path.exists(img2_path)
