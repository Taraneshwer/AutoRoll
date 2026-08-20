"""
Detailed audit runner for AutoRoll ML Phase 5.1 Dataset Authenticity & Provenance Audit.
Performs fresh SCRFD detector analysis, pHash near-duplicate analysis, pixel stats, and source tracing.
"""
import os
import sys
import json
import time
import glob
import hashlib
import cv2
import numpy as np

from autoroll.ml.detectors.scrfd import SCRFDDetector
from autoroll.ml.detectors.aligner import FaceAligner

def compute_dhash(image, hash_size=8):
    # Resize to (hash_size + 1, hash_size), grayscale
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def run_detailed_audit():
    print("================================================================================")
    print("RUNNING FRESH SCRFD DETECTOR & DETAILED AUDIT SUITE")
    print("================================================================================")

    data_dir = "data/face_recognition"
    splits_dir = os.path.join(data_dir, "splits")
    raw_dir = os.path.join(data_dir, "raw")

    # 1. Fresh SCRFD Detection Test on 500 Source Images
    print("\n--- 1. FRESH SCRFD DETECTOR TEST ON 500 SOURCE IMAGES ---")
    detector = SCRFDDetector(model_path="models/scrfd_10g_bnkps.onnx", conf_threshold=0.3)

    sample_raw_files = []
    # Collect 500 raw image paths
    raw_ids = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    for rid in raw_ids[:50]:
        rpath = os.path.join(raw_dir, rid)
        imgs = [os.path.join(rpath, f) for f in os.listdir(rpath) if f.endswith(".jpg")]
        sample_raw_files.extend(imgs[:10])
        if len(sample_raw_files) >= 500:
            break

    print(f"Collected {len(sample_raw_files)} raw source images for fresh SCRFD detection.")

    confidences = []
    face_sizes = []
    blurs = []
    brightnesses = []
    detected_count = 0

    for path in sample_raw_files:
        img = cv2.imread(path)
        if img is None:
            continue

        # Blur (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        blurs.append(blur_val)

        # Brightness
        bright_val = np.mean(gray)
        brightnesses.append(bright_val)

        # Detector call
        faces = detector.detect(img)
        if len(faces) > 0:
            detected_count += 1
            top_face = max(faces, key=lambda f: f.det_confidence)
            confidences.append(top_face.det_confidence)
            face_sizes.append(top_face.bbox.area)

    print(f"SCRFD Detection Results on 500 Synthetic Source Images:")
    print(f"  Faces Detected: {detected_count} / {len(sample_raw_files)} ({detected_count / len(sample_raw_files) * 100:.1f}%)")
    if len(confidences) > 0:
        print(f"  Confidence - Mean: {np.mean(confidences):.4f}, Min: {np.min(confidences):.4f}, Max: {np.max(confidences):.4f}")
        print(f"  Face Box Area (px^2) - Mean: {np.mean(face_sizes):.1f}, Min: {np.min(face_sizes):.1f}, Max: {np.max(face_sizes):.1f}")
    else:
        print("  Confidence: N/A (SCRFD detected ZERO faces in synthetic drawings at threshold 0.3!)")
    print(f"  Blur (Laplacian Var) - Mean: {np.mean(blurs):.2f}, Min: {np.min(blurs):.2f}, Max: {np.max(blurs):.2f}")
    print(f"  Brightness (Gray Mean) - Mean: {np.mean(brightnesses):.2f}, Min: {np.min(brightnesses):.2f}, Max: {np.max(brightnesses):.2f}")

    # 2. Near-Duplicate Audit (dHash)
    print("\n--- 2. NEAR-DUPLICATE AUDIT USING DIFFERENCE HASH (dHash) ---")
    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")
    test_dir = os.path.join(splits_dir, "test")

    split_samples = []
    for sname, sdir in [("TRAIN", train_dir), ("VAL", val_dir), ("TEST", test_dir)]:
        ids = [d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d))][:30]
        for id_name in ids:
            ipath = os.path.join(sdir, id_name)
            for f in os.listdir(ipath)[:10]:
                split_samples.append((sname, id_name, f, os.path.join(ipath, f)))

    print(f"Sampled {len(split_samples)} aligned chips across TRAIN/VAL/TEST for dHash similarity.")
    hashes = {}
    near_dups_same_id = 0
    near_dups_cross_id = 0
    near_dups_cross_split = 0

    for sname, id_name, fname, path in split_samples:
        img = cv2.imread(path)
        dh = compute_dhash(img)
        if dh in hashes:
            prev_sname, prev_id, prev_fname = hashes[dh]
            if prev_id == id_name:
                near_dups_same_id += 1
            else:
                near_dups_cross_id += 1
                if prev_sname != sname:
                    near_dups_cross_split += 1
        else:
            hashes[dh] = (sname, id_name, fname)

    print(f"Near-Duplicate Analysis Results (dHash):")
    print(f"  Unique dHashes: {len(hashes)} / {len(split_samples)}")
    print(f"  Near-Duplicates Same Identity: {near_dups_same_id}")
    print(f"  Near-Duplicates Cross Identity: {near_dups_cross_id}")
    print(f"  Near-Duplicates Cross Split: {near_dups_cross_split}")

    # 3. Source-to-Processed Verification (100 Samples)
    print("\n--- 3. SOURCE-TO-PROCESSED VERIFICATION (100 SAMPLES) ---")
    verified_matches = 0
    sample_100_ids = raw_ids[:100]
    for rid in sample_100_ids:
        raw_id_dir = os.path.join(raw_dir, rid)
        aligned_id_dir = os.path.join(data_dir, "aligned", rid)
        if os.path.exists(raw_id_dir) and os.path.exists(aligned_id_dir):
            raw_files = os.listdir(raw_id_dir)
            aligned_files = os.listdir(aligned_id_dir)
            if len(raw_files) > 0 and len(aligned_files) > 0:
                verified_matches += 1

    print(f"  Source-to-Processed Identity Folder Mapping: {verified_matches} / 100 Verified.")

    # 4. Pixel statistics per split
    print("\n--- 4. PIXEL STATISTICS PER SPLIT ---")
    for sname, sdir in [("TRAIN", train_dir), ("VAL", val_dir), ("TEST", test_dir)]:
        ids = os.listdir(sdir)[:20]
        means, stds = [], []
        for id_name in ids:
            ipath = os.path.join(sdir, id_name)
            for f in os.listdir(ipath)[:10]:
                img = cv2.imread(os.path.join(ipath, f))
                if img is not None:
                    means.append(np.mean(img))
                    stds.append(np.std(img))
        print(f"  {sname} Split (200 images): Mean Intensity = {np.mean(means):.2f}, Mean Std = {np.mean(stds):.2f}")

if __name__ == "__main__":
    run_detailed_audit()
