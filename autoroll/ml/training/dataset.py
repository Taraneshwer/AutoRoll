"""
PyTorch Dataset Loader for Real Aligned Face Recognition Chips.
Supports online training augmentations (random horizontal flip, color jitter, blur)
and maps identities to contiguous class indices.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RealFaceDataset(Dataset):
    """
    PyTorch Dataset for loading 112x112 aligned face chips from split directory.
    """

    def __init__(self, split_dir: str, augment: bool = True):
        self.split_dir = split_dir
        self.augment = augment

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory '{split_dir}' does not exist.")

        identities = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        for idx, id_name in enumerate(identities):
            self.class_to_idx[id_name] = idx
            id_path = os.path.join(split_dir, id_name)
            for fname in sorted(os.listdir(id_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(id_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def _apply_augmentations(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1. Random Horizontal Flip (p = 0.5)
        if np.random.rand() > 0.5:
            img_bgr = cv2.flip(img_bgr, 1)

        # 2. Mild Color Jitter / Brightness variation (p = 0.3)
        if np.random.rand() > 0.7:
            alpha = 1.0 + np.random.uniform(-0.1, 0.1)
            beta = np.random.uniform(-10, 10)
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # 3. Mild Gaussian Blur (p = 0.2)
        if np.random.rand() > 0.8:
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

        return img_bgr

    def __getitem__(self, index: int):
        fpath = self.image_paths[index]
        label = self.labels[index]

        img_bgr = cv2.imread(fpath)
        if img_bgr is None or img_bgr.size == 0:
            # Fallback to black chip
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        if self.augment:
            img_bgr = self._apply_augmentations(img_bgr)

        # Standard ArcFace Preprocessing: BGR -> RGB -> Float32 -> (x - 127.5) / 127.5 -> NCHW
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))

        tensor_x = torch.from_numpy(blob).float()
        tensor_y = torch.tensor(label, dtype=torch.long)

        return tensor_x, tensor_y


def create_training_dataloaders(
    split_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    train_dir = os.path.join(split_root, "train")
    val_dir = os.path.join(split_root, "val")

    train_ds = RealFaceDataset(train_dir, augment=True)
    val_ds = RealFaceDataset(val_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, len(train_ds.class_to_idx)


# Alias for backward compatibility
FaceDataset = RealFaceDataset
