"""
AutoRoll Real Training Dataset Validation Suite.
Programmatically verifies source manifest authenticity, identity disjointness, image dimensions,
file integrity, and non-synthetic real face constraints before ArcFace fine-tuning.
"""

import os
import sys
import json
import cv2

# Ensure project root is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoroll.common.logger import get_logger

logger = get_logger("validate_training_dataset")

SPLITS_DIR = "data/face_recognition/splits"
SOURCE_MANIFEST_PATH = "data/face_recognition/metadata/source_manifest.json"


def validate_dataset():
    logger.info("Initializing AutoRoll Real Training Dataset Validation...")
    errors = []

    # 1. Source Manifest Verification
    if not os.path.exists(SOURCE_MANIFEST_PATH):
        errors.append(
            f"CRITICAL ERROR: Source manifest missing at '{SOURCE_MANIFEST_PATH}'. "
            "Real dataset ingestion is required before validation. Run 'scripts/ingest_real_dataset.py'."
        )
        logger.error(errors[-1])
        print("\n" + "=" * 80)
        print("           AUTOROLL REAL TRAINING DATASET VALIDATION REPORT           ")
        print("=" * 80)
        print("Validation Status : FAILED (source_manifest.json missing)")
        print("Reason            : REAL TRAINING DATASET REQUIRED")
        print("=" * 80 + "\n")
        sys.exit(1)

    with open(SOURCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 2. Authenticity & Synthetic Guard Checks
    if manifest.get("synthetic", True) is True or manifest.get("dataset_type") != "real":
        errors.append("CRITICAL REJECTION: Manifest indicates dataset is SYNTHETIC. Synthetic data is prohibited.")

    if not manifest.get("source_url") or not manifest.get("official_source"):
        errors.append("CRITICAL REJECTION: Source provenance metadata (source_url / official_source) missing.")

    # 3. Read Splits from Disk
    train_dir = os.path.join(SPLITS_DIR, "train")
    val_dir = os.path.join(SPLITS_DIR, "val")
    test_dir = os.path.join(SPLITS_DIR, "test")

    train_ids = set(os.listdir(train_dir)) if os.path.exists(train_dir) else set()
    val_ids = set(os.listdir(val_dir)) if os.path.exists(val_dir) else set()
    test_ids = set(os.listdir(test_dir)) if os.path.exists(test_dir) else set()

    if not train_ids:
        errors.append("No training identities found in TRAIN split.")

    # 4. Identity Leakage Verification (Disjointness)
    logger.info("Verifying Identity-Disjointness across splits...")
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap:
        errors.append(f"Identity Leakage detected between TRAIN and VAL splits: {train_val_overlap}")
    if train_test_overlap:
        errors.append(f"Identity Leakage detected between TRAIN and TEST splits: {train_test_overlap}")
    if val_test_overlap:
        errors.append(f"Identity Leakage detected between VAL and TEST splits: {val_test_overlap}")

    # 5. Aligned Face Shape & Integrity Verification
    logger.info("Validating image shapes and readability...")
    total_validated_images = 0
    sampled_image_paths = []

    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(SPLITS_DIR, split_name)
        if not os.path.exists(split_path):
            continue

        id_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        for id_folder in id_folders:
            id_path = os.path.join(split_path, id_folder)
            files = [f for f in os.listdir(id_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for fname in files:
                fpath = os.path.join(id_path, fname)
                if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                    errors.append(f"Empty/missing image file: '{fpath}'")
                total_validated_images += 1
                sampled_image_paths.append(fpath)

    # Subsample up to 5,000 images for full cv2 decode validation
    import random
    random.seed(42)
    sample_to_check = random.sample(sampled_image_paths, min(5000, len(sampled_image_paths))) if sampled_image_paths else []

    logger.info(f"Performing cv2 decode and shape check on representative sample of {len(sample_to_check)} images...")
    for fpath in sample_to_check:
        img = cv2.imread(fpath)
        if img is None or img.size == 0:
            errors.append(f"Corrupt/unreadable aligned image: '{fpath}'")
            continue
        if img.shape != (112, 112, 3):
            errors.append(f"Invalid image shape {img.shape} at '{fpath}'. Expected (112, 112, 3).")

    # 6. Summary Report
    print("\n" + "=" * 80)
    print("           AUTOROLL REAL TRAINING DATASET VALIDATION REPORT           ")
    print("=" * 80)
    print(f"Dataset Name               : {manifest.get('dataset_name', 'Unknown')}")
    print(f"Dataset Type               : {manifest.get('dataset_type', 'Unknown')} (Synthetic: {manifest.get('synthetic')})")
    print(f"Total Identities Validated : {len(train_ids) + len(val_ids) + len(test_ids)}")
    print(f"  |- Train Identities      : {len(train_ids)}")
    print(f"  |- Val Identities        : {len(val_ids)}")
    print(f"  |- Test Identities       : {len(test_ids)}")
    print(f"Total Images Validated     : {total_validated_images}")
    print(f"Identity Leakage Check     : {'PASSED (Zero Overlap)' if not (train_val_overlap or train_test_overlap or val_test_overlap) else 'FAILED'}")
    print(f"Validation Status          : {'PASSED' if not errors else 'FAILED'}")
    print("=" * 80 + "\n")

    if errors:
        logger.error("DATASET VALIDATION FAILED with the following errors:")
        for err in errors:
            logger.error(f"  |- {err}")
        sys.exit(1)
    else:
        logger.info("ALL DATASET VALIDATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    validate_dataset()
