"""
Dataset Face Aligner and Persistence Manager with Resumability Support.
"""

import os

import cv2
import numpy as np

from app.core.logger import get_logger
from app.schemas.common import DetectionResult
from app.ml.detectors.aligner import FaceAligner

logger = get_logger("dataset_aligner")


class DatasetFaceAligner:
    """
    Aligns and saves processed face chips into target split structure.
    """

    def __init__(self, target_size: tuple[int, int] = (112, 112), resumable: bool = True):
        self.aligner = FaceAligner(target_size=target_size)
        self.resumable = resumable

    def process_and_save(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        output_dir: str,
        split_name: str,
        identity_id: str,
        image_name: str,
    ) -> tuple[str, bool]:
        """
        Aligns face chip and writes to disk at output_dir/split_name/identity_id/image_name.
        Returns (saved_path, is_skipped_due_to_resumability).
        """
        dest_dir = os.path.join(output_dir, split_name, identity_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, image_name)

        if self.resumable and os.path.exists(dest_path):
            logger.debug(f"Resumable check: File '{dest_path}' already exists. Skipping alignment.")
            return dest_path, True

        aligned_chip = self.aligner.align(image, detection.landmarks)
        success = cv2.imwrite(dest_path, aligned_chip)

        if not success:
            raise OSError(f"Failed to write aligned image to '{dest_path}'")

        return dest_path, False
