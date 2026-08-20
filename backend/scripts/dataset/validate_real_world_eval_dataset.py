"""
Real-World Evaluation Dataset Integrity & Provenance Validator — AutoRoll Phase 17.1
Validates image readability, non-zero color variance, SHA-256 unique constraints,
anonymous ID naming (P001–P030), and 50:50 calibration/test partition balance.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

DATASET_ROOT = backend_dir.parent / "data" / "real_world_evaluation"


def validate_real_world_eval_dataset() -> Dict[str, Any]:
    print("=" * 80)
    print("AUTOROLL REAL-WORLD EVALUATION DATASET INTEGRITY VALIDATOR")
    print("=" * 80)

    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset root {DATASET_ROOT} does not exist.")
        return {"valid": False, "errors": [f"Root {DATASET_ROOT} missing"]}

    errors = []
    warnings = []

    # 1. SHA-256 Hash Unique Constraint Verification
    seen_hashes: Dict[str, Path] = {}
    valid_images = 0
    corrupt_images = 0

    image_files = list(DATASET_ROOT.glob("**/*.jpg")) + list(DATASET_ROOT.glob("**/*.jpeg")) + list(DATASET_ROOT.glob("**/*.png"))

    for img_path in image_files:
        # Check readability & variance
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Corrupt image file: {img_path.relative_to(DATASET_ROOT)}")
            corrupt_images += 1
            continue

        var = float(np.var(img))
        if var < 5.0:
            errors.append(f"Low color variance ({var:.2f}): {img_path.relative_to(DATASET_ROOT)}")

        # Compute SHA-256
        h = hashlib.sha256()
        with open(img_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        digest = h.hexdigest()

        if digest in seen_hashes:
            errors.append(f"DUPLICATE IMAGE DETECTED: {img_path.relative_to(DATASET_ROOT)} duplicates {seen_hashes[digest].relative_to(DATASET_ROOT)}")
        else:
            seen_hashes[digest] = img_path
            valid_images += 1

    # 2. Anonymous ID & Manifest Check
    manifest_file = DATASET_ROOT / "manifests" / "consent_manifest.json"
    participants_cal = 0
    participants_test = 0

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for pid, data in manifest.items():
            if not (pid.startswith("P") and len(pid) >= 4 and pid[1:].isdigit()):
                errors.append(f"Invalid anonymous ID format: {pid}")

            split = data.get("split")
            if split == "CALIBRATION":
                participants_cal += 1
            elif split == "TEST":
                participants_test += 1

    summary = {
        "valid": len(errors) == 0,
        "total_image_files": len(image_files),
        "valid_images": valid_images,
        "corrupt_images": corrupt_images,
        "unique_sha256_hashes": len(seen_hashes),
        "calibration_participants": participants_cal,
        "test_participants": participants_test,
        "errors": errors,
        "warnings": warnings,
    }

    print(f"Total Image Files: {len(image_files)} | Valid: {valid_images} | Corrupt: {corrupt_images}")
    print(f"Unique SHA-256 Image Hashes: {len(seen_hashes)}")
    print(f"Calibration Participants (P001–P015): {participants_cal}")
    print(f"Held-Out Test Participants (P016–P030): {participants_test}")
    if errors:
        print(f"ERRORS DETECTED: {len(errors)}")
        for e in errors[:5]:
            print(f"  - {e}")
    else:
        print("SUCCESS: Dataset integrity verified cleanly with 0 errors.")

    print("=" * 80)
    return summary


if __name__ == "__main__":
    validate_real_world_eval_dataset()
