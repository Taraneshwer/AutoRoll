"""
Dataset Face Detector Wrapper for Dataset Preprocessing Pipeline.
"""

import cv2
import numpy as np

from app.core.logger import get_logger
from app.schemas.common import DetectionResult
from app.ml.detectors.scrfd import SCRFDDetector

logger = get_logger("dataset_detector")


class DatasetFaceDetector:
    """
    Dataset face detector wrapper enforcing single face presence validation.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.detector = SCRFDDetector(device="auto", conf_threshold=min_confidence)

    def detect_face(
        self, image_path: str
    ) -> tuple[np.ndarray | None, DetectionResult | None, str | None]:
        """
        Reads image from image_path and returns (image_array, single_detection, error_reason).
        """
        if not cv2.os.path.exists(image_path):
            return None, None, "file_not_found"

        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return None, None, "corrupt_image_file"

        try:
            detections = self.detector.detect(image)
        except Exception as e:
            return image, None, f"detection_exception: {e}"

        if len(detections) == 0:
            return image, None, "no_face_detected"

        if len(detections) > 1:
            # Pick highest confidence face
            detections = sorted(detections, key=lambda d: d.det_confidence, reverse=True)

        return image, detections[0], None
