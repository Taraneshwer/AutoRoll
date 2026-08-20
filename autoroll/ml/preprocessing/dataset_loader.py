"""
Configurable Dataset Loader supporting directory scanning, YAML configuration,
and synthetic dataset generation.
"""

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import yaml
from pydantic import BaseModel

from autoroll.common.logger import get_logger

logger = get_logger("dataset_loader")


class DatasetConfig(BaseModel):
    dataset_name: str
    dataset_version: str = "1.0.0"
    preprocessing_version: str = "1.0.0"
    raw_data_dir: str
    output_dir: str
    download_url: str | None = None
    min_face_size: int = 30
    min_blur_score: float = 20.0
    min_detection_confidence: float = 0.5
    split_ratios: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    resumable: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DatasetConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset YAML config not found at '{yaml_path}'")
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class RawImageRecord(BaseModel):
    identity_id: str
    image_name: str
    full_path: str


class BaseDatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self) -> dict[str, list[RawImageRecord]]:
        """
        Scans dataset and returns mapping: {identity_id: [RawImageRecord]}
        """
        pass


class DirectoryDatasetLoader(BaseDatasetLoader):
    """
    Loads raw image dataset from directory formatted as:
    root_dir/
      identity_01/
        img1.jpg
        img2.jpg
      identity_02/
        ...
    """

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def load_dataset(self) -> dict[str, list[RawImageRecord]]:
        if not os.path.exists(self.raw_data_dir):
            logger.warning(f"Raw data directory '{self.raw_data_dir}' does not exist.")
            return {}

        identity_map: dict[str, list[RawImageRecord]] = {}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for item in sorted(os.listdir(self.raw_data_dir)):
            item_path = os.path.join(self.raw_data_dir, item)
            if os.path.isdir(item_path):
                identity_id = item
                image_records = []
                for fname in sorted(os.listdir(item_path)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in valid_extensions:
                        full_path = os.path.join(item_path, fname)
                        image_records.append(
                            RawImageRecord(
                                identity_id=identity_id,
                                image_name=fname,
                                full_path=full_path,
                            )
                        )
                if image_records:
                    identity_map[identity_id] = image_records

        logger.info(
            f"DirectoryDatasetLoader found {len(identity_map)} identities "
            f"and {sum(len(v) for v in identity_map.values())} total images."
        )
        return identity_map


class SyntheticDatasetLoader(BaseDatasetLoader):
    """
    Generates a small synthetic dataset on disk for test and pipeline verification.
    """

    def __init__(self, target_dir: str, num_identities: int = 5, images_per_id: int = 3):
        self.target_dir = target_dir
        self.num_identities = num_identities
        self.images_per_id = images_per_id

    def load_dataset(self) -> dict[str, list[RawImageRecord]]:
        os.makedirs(self.target_dir, exist_ok=True)
        identity_map: dict[str, list[RawImageRecord]] = {}

        for i in range(1, self.num_identities + 1):
            identity_id = f"student_id_{i:03d}"
            id_dir = os.path.join(self.target_dir, identity_id)
            os.makedirs(id_dir, exist_ok=True)

            image_records = []
            for j in range(1, self.images_per_id + 1):
                fname = f"sample_{j:02d}.jpg"
                full_path = os.path.join(id_dir, fname)

                # Generate realistic synthetic face with distinct skin tone, hair, eyes, nose, mouth
                skin_tones = [
                    (180, 200, 240), (140, 180, 220), (100, 140, 190), (60, 90, 140), (120, 160, 210),
                    (160, 190, 235), (90, 120, 170), (50, 70, 120), (130, 170, 220), (110, 140, 190)
                ]
                skin = skin_tones[(i - 1) % len(skin_tones)]
                w_face = 170 + (i * 5) % 30
                h_face = 220 + (i * 7) % 40
                cx = 320 + (j - 1) * 8
                cy = 240 + (j - 1) * 6

                img = np.full((480, 640, 3), (240, 235, 230), dtype=np.uint8)
                # Hair
                cv2.ellipse(img, (cx, cy - 25), (w_face + 15, h_face + 15), 0, 0, 360, (20, 15, 10), -1)
                # Face
                cv2.ellipse(img, (cx, cy), (w_face, h_face), 0, 0, 360, skin, -1)

                # Eyes
                eye_sep = 65 + (i * 4) % 25
                eye_y = cy - 50
                cv2.ellipse(img, (cx - eye_sep, eye_y), (22, 12), 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(img, (cx + eye_sep, eye_y), (22, 12), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (cx - eye_sep, eye_y), 9, (20, 10, 5), -1)
                cv2.circle(img, (cx + eye_sep, eye_y), 9, (20, 10, 5), -1)

                # Eyebrows
                cv2.line(img, (cx - eye_sep - 20, eye_y - 20), (cx - eye_sep + 20, eye_y - 20), (10, 10, 10), 6)
                cv2.line(img, (cx + eye_sep - 20, eye_y - 20), (cx + eye_sep + 20, eye_y - 20), (10, 10, 10), 6)

                # Nose
                cv2.line(img, (cx, eye_y + 10), (cx - 8, cy + 25), (skin[0]-40, skin[1]-40, skin[2]-40), 5)
                cv2.line(img, (cx - 8, cy + 25), (cx + 12, cy + 25), (skin[0]-40, skin[1]-40, skin[2]-40), 5)

                # Lips
                cv2.ellipse(img, (cx, cy + 75), (40 + (i * 4) % 25, 15), 0, 0, 360, (60, 40, 120), -1)
                cv2.imwrite(full_path, img)

                image_records.append(
                    RawImageRecord(
                        identity_id=identity_id,
                        image_name=fname,
                        full_path=full_path,
                    )
                )
            identity_map[identity_id] = image_records

        return identity_map
