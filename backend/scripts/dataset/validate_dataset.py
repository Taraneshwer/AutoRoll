"""
AutoRoll Dataset Validation Utility Script.
Checks dataset integrity and verifies zero identity leakage across splits.
Usage: python scripts/validate_dataset.py [--dir data/processed_datasets/sample_subset]
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import argparse
import os

import cv2

from app.core.logger import get_logger

logger = get_logger("validate_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Dataset Validation Utility")
    parser.add_argument(
        "--dir",
        default="./data/processed_datasets/sample_subset",
        help="Path to processed dataset directory",
    )
    return parser.parse_args()


def validate_dataset(processed_dir: str) -> bool:
    logger.info(f"Validating dataset integrity at '{processed_dir}'...")

    if not os.path.exists(processed_dir):
        logger.error(f"Directory '{processed_dir}' does not exist.")
        return False

    split_identities = {}
    valid = True

    for split in ["train", "val", "test"]:
        split_path = os.path.join(processed_dir, split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory '{split_path}' missing.")
            split_identities[split] = set()
            continue

        id_dirs = [
            d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))
        ]
        split_identities[split] = set(id_dirs)

        # Validate image chip dimensions (112x112)
        for id_dir in id_dirs:
            full_id_dir = os.path.join(split_path, id_dir)
            for fname in os.listdir(full_id_dir):
                fpath = os.path.join(full_id_dir, fname)
                img = cv2.imread(fpath)
                if img is None or img.shape[:2] != (112, 112):
                    shape = img.shape if img is not None else None
                    logger.error(f"Invalid image chip found: {fpath} (shape: {shape})")
                    valid = False

    # Check for Identity Leakage between splits
    train_ids = split_identities.get("train", set())
    val_ids = split_identities.get("val", set())
    test_ids = split_identities.get("test", set())

    train_val_leak = train_ids.intersection(val_ids)
    train_test_leak = train_ids.intersection(test_ids)
    val_test_leak = val_ids.intersection(test_ids)

    if train_val_leak or train_test_leak or val_test_leak:
        logger.error("IDENTITY LEAKAGE DETECTED BETWEEN SPLITS!")
        if train_val_leak:
            logger.error(f"Train/Val Overlap: {train_val_leak}")
        if train_test_leak:
            logger.error(f"Train/Test Overlap: {train_test_leak}")
        if val_test_leak:
            logger.error(f"Val/Test Overlap: {val_test_leak}")
        valid = False
    else:
        logger.info("IDENTITY DISJOINTNESS VERIFIED: Zero identity leakage between splits.")

    if valid:
        print("\n" + "=" * 60)
        print("          DATASET VALIDATION VERDICT: PASSED          ")
        print("=" * 60)
        print(f"Train Identities : {len(train_ids)}")
        print(f"Val Identities   : {len(val_ids)}")
        print(f"Test Identities  : {len(test_ids)}")
        print("Identity Leakage : NONE (0% overlap)")
        print("Image Resolution : 112x112 (100% verified)")
        print("=" * 60 + "\n")

    return valid


def main():
    args = parse_args()
    success = validate_dataset(args.dir)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
