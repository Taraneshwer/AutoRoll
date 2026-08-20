"""
AutoRoll Real Training Dataset Preparation Pipeline Script.
Processes real raw face images with SCRFD, 5-point alignment (112x112), quality filtering,
identity-disjoint splitting, and saves dataset manifest metadata.
"""

import os
import sys
import json
import time
import shutil
import cv2
import numpy as np

from autoroll.common.logger import get_logger
from autoroll.ml.detectors.scrfd import SCRFDDetector
from autoroll.ml.detectors.aligner import FaceAligner
from autoroll.ml.preprocessing.quality import FaceQualityFilter
from autoroll.ml.preprocessing.splitter import IdentityDisjointSplitter

logger = get_logger("prepare_real_training_dataset")

RAW_DIR = "data/face_recognition/raw"
DETECTED_DIR = "data/face_recognition/detected"
ALIGNED_DIR = "data/face_recognition/aligned"
METADATA_DIR = "data/face_recognition/metadata"
SPLITS_DIR = "data/face_recognition/splits"


def process_dataset():
    logger.info("Initializing Real Face Dataset Preparation Pipeline...")

    # Load SCRFD detector, Face aligner, and Quality filter
    detector = SCRFDDetector(conf_threshold=0.5)
    aligner = FaceAligner(target_size=(112, 112))
    quality_filter = FaceQualityFilter(min_face_size=30, min_blur_score=15.0, min_confidence=0.5)

    identities = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    if not identities:
        logger.error(f"No raw identity folders found in '{RAW_DIR}'. Run download_real_face_dataset.py first.")
        sys.exit(1)

    total_raw_images = 0
    aligned_identity_map = {}
    rejection_records = []

    start_time = time.time()
    processed_count = 0
    rejected_count = 0

    for id_name in identities:
        id_raw_dir = os.path.join(RAW_DIR, id_name)
        id_detected_dir = os.path.join(DETECTED_DIR, id_name)
        id_aligned_dir = os.path.join(ALIGNED_DIR, id_name)

        os.makedirs(id_detected_dir, exist_ok=True)
        os.makedirs(id_aligned_dir, exist_ok=True)
        aligned_identity_map[id_name] = []

        img_files = sorted([f for f in os.listdir(id_raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        total_raw_images += len(img_files)

        for img_name in img_files:
            img_path = os.path.join(id_raw_dir, img_name)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None or img_bgr.size == 0:
                rejection_records.append({"path": img_path, "identity": id_name, "reason": "unreadable_image_file"})
                rejected_count += 1
                continue

            # Step 1: SCRFD Face Detection
            dets = detector.detect(img_bgr, score_threshold=0.5)
            if not dets:
                rejection_records.append({"path": img_path, "identity": id_name, "reason": "no_face_detected"})
                rejected_count += 1
                continue

            if len(dets) > 1:
                # Flag multiple faces, select highest confidence face
                dets.sort(key=lambda d: d.det_confidence, reverse=True)

            best_det = dets[0]

            # Step 2: Quality Filtering
            q_res = quality_filter.evaluate(img_bgr, best_det)
            if not q_res.passed:
                rejection_records.append({"path": img_path, "identity": id_name, "reason": f"quality_failed: {q_res.reason}"})
                rejected_count += 1
                continue

            # Save detection overlay visualization in detected/
            overlay_img = img_bgr.copy()
            bbox = best_det.bbox.to_list()
            cv2.rectangle(
                overlay_img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )
            cv2.imwrite(os.path.join(id_detected_dir, img_name), overlay_img)

            # Step 3: 5-Point Similarity Transformation Alignment (112x112)
            aligned_chip = aligner.align(img_bgr, best_det.landmarks)
            aligned_path = os.path.join(id_aligned_dir, img_name)
            cv2.imwrite(aligned_path, aligned_chip)

            aligned_identity_map[id_name].append({
                "image_name": img_name,
                "aligned_path": aligned_path,
                "bbox": bbox,
                "confidence": float(best_det.det_confidence),
            })
            processed_count += 1

    elapsed_time = time.time() - start_time
    fps = processed_count / elapsed_time if elapsed_time > 0 else 0.0

    # Step 4: Identity-Disjoint Splitting
    # Clean map to only include identities with at least 1 aligned face crop
    valid_map = {k: v for k, v in aligned_identity_map.items() if len(v) > 0}
    splitter = IdentityDisjointSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    split_identities = splitter.split_identities(valid_map)

    # Populate splits/ directory hierarchy
    split_summary = {}
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(SPLITS_DIR, split_name)
        # Clear existing
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

        id_submap = split_identities.get(split_name, {})
        split_img_count = 0

        for id_name, records in id_submap.items():
            id_split_dir = os.path.join(split_dir, id_name)
            os.makedirs(id_split_dir, exist_ok=True)
            for r in records:
                shutil.copy2(r["aligned_path"], os.path.join(id_split_dir, r["image_name"]))
                split_img_count += 1

        split_summary[split_name] = {
            "identity_count": len(id_submap),
            "image_count": split_img_count,
            "identities": list(id_submap.keys())
        }

    # Step 5: Save Dataset Manifest metadata
    manifest = {
        "dataset_name": "AutoRoll_Real_Face_Recognition_Dataset",
        "dataset_version": "1.0.0",
        "preprocessing_version": "1.0.0_scrfd_112x112_umeyama",
        "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_raw_images": total_raw_images,
        "total_processed_aligned_images": processed_count,
        "total_rejected_images": rejected_count,
        "processing_throughput_fps": round(fps, 2),
        "processing_latency_per_img_ms": round(1000 / fps, 2) if fps > 0 else 0.0,
        "splits_summary": split_summary,
        "rejection_summary": {
            "total_rejections": len(rejection_records),
            "records": rejection_records
        }
    }

    manifest_path = os.path.join(METADATA_DIR, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 70)
    logger.info("       AUTOROLL REAL FACE DATASET PREPARATION SUMMARY       ")
    logger.info("=" * 70)
    logger.info(f"Total Raw Images Analyzed   : {total_raw_images}")
    logger.info(f"Successfully Aligned Faces : {processed_count}")
    logger.info(f"Rejected / Filtered Images  : {rejected_count}")
    logger.info(f"Processing Throughput       : {fps:.2f} images/sec")
    logger.info("-" * 70)
    logger.info(f"TRAIN Split Identities      : {split_summary['train']['identity_count']} ({split_summary['train']['image_count']} images)")
    logger.info(f"VAL Split Identities        : {split_summary['val']['identity_count']} ({split_summary['val']['image_count']} images)")
    logger.info(f"TEST Split Identities       : {split_summary['test']['identity_count']} ({split_summary['test']['image_count']} images)")
    logger.info("=" * 70)
    logger.info(f"Dataset Manifest saved to '{manifest_path}'.")


if __name__ == "__main__":
    process_dataset()
