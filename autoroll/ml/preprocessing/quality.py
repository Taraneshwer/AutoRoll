"""
Face Quality Assessment and Filtering Module.
"""

import cv2
import numpy as np
from pydantic import BaseModel

from autoroll.common.logger import get_logger
from autoroll.common.schemas import DetectionResult

logger = get_logger("face_quality")


class QualityCheckResult(BaseModel):
    passed: bool
    reason: str
    blur_score: float
    face_width: float
    face_height: float


class FaceQualityFilter:
    """
    Evaluates detected face quality against resolution, blur, and detection confidence metrics.
    """

    def __init__(
        self,
        min_face_size: int = 30,
        min_blur_score: float = 20.0,
        min_confidence: float = 0.5,
    ):
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.min_confidence = min_confidence

    def evaluate(self, image: np.ndarray, detection: DetectionResult) -> QualityCheckResult:
        bbox = detection.bbox
        width = bbox.width
        height = bbox.height

        if width < self.min_face_size or height < self.min_face_size:
            return QualityCheckResult(
                passed=False,
                reason=(
                    f"Face resolution ({width:.1f}x{height:.1f}) below minimum "
                    f"({self.min_face_size}px)"
                ),
                blur_score=0.0,
                face_width=width,
                face_height=height,
            )

        if detection.det_confidence < self.min_confidence:
            return QualityCheckResult(
                passed=False,
                reason=(
                    f"Detection confidence ({detection.det_confidence:.2f}) below threshold "
                    f"({self.min_confidence:.2f})"
                ),
                blur_score=0.0,
                face_width=width,
                face_height=height,
            )

        # Crop face for blur estimation
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(image.shape[1], int(bbox.x2)), min(image.shape[0], int(bbox.y2))
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return QualityCheckResult(
                passed=False,
                reason="Invalid or 0-area face crop",
                blur_score=0.0,
                face_width=width,
                face_height=height,
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.min_blur_score:
            return QualityCheckResult(
                passed=False,
                reason=(
                    f"Blur score ({blur_score:.1f}) below minimum threshold "
                    f"({self.min_blur_score:.1f})"
                ),
                blur_score=round(blur_score, 2),
                face_width=width,
                face_height=height,
            )

        return QualityCheckResult(
            passed=True,
            reason="Quality criteria passed",
            blur_score=round(blur_score, 2),
            face_width=width,
            face_height=height,
        )
