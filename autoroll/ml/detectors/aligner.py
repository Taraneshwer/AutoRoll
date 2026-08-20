"""
Independent Face Alignment Module using 5-point facial landmark similarity transformation.
Standardized 112x112 affine alignment for ArcFace input.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from autoroll.common.logger import get_logger
from autoroll.common.schemas import FaceLandmarks

logger = get_logger("face_aligner")

# Standard ArcFace 112x112 reference landmarks (Umeyama similarity transformation target)
REFERENCE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class BaseFaceAligner(ABC):
    """
    Abstract contract for 5-point facial landmark similarity transformation alignment.
    """

    @abstractmethod
    def align(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        output_size: tuple[int, int] = (112, 112),
    ) -> np.ndarray:
        pass


class FaceAligner(BaseFaceAligner):
    """
    Deterministic 5-point similarity transformation face aligner.
    """

    def __init__(self, target_size: tuple[int, int] = (112, 112)):
        self.target_size = target_size
        self.ref_pts = (
            REFERENCE_5PTS_112
            if target_size == (112, 112)
            else REFERENCE_5PTS_112 * (target_size[0] / 112.0)
        )

    def align(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        output_size: tuple[int, int] = (112, 112),
    ) -> np.ndarray:
        """
        Aligns input BGR image using facial landmarks to produce output_size aligned face chip.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        if not landmarks.validate_5point():
            raise ValueError("FaceLandmarks must contain exactly 5 points.")

        src_pts = np.array(landmarks.points, dtype=np.float32)

        # Estimate partial affine similarity matrix
        tfm, _ = cv2.estimateAffinePartial2D(src_pts, self.ref_pts)

        if tfm is None:
            # Fallback to identity transform if matrix estimation fails
            tfm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        aligned = cv2.warpAffine(
            image,
            tfm,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned
