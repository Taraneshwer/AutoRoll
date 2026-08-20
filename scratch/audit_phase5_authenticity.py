"""
Scratch script to perform comprehensive ML Phase 5.1 dataset authenticity & provenance audit (ASCII clean).
"""
import os
import sys
import json
import glob
import hashlib
import cv2
import numpy as np

def run_audit():
    print("================================================================================")
    print("STARTING AUTOROLL ML PHASE 5.1 DATASET AUTHENTICITY & PROVENANCE AUDIT")
    print("================================================================================")

    data_dir = "data/face_recognition"
    splits_dir = os.path.join(data_dir, "splits")
    raw_dir = os.path.join(data_dir, "raw")
    aligned_dir = os.path.join(data_dir, "aligned")

    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")
    test_dir = os.path.join(splits_dir, "test")

    train_ids = os.listdir(train_dir) if os.path.exists(train_dir) else []
    val_ids = os.listdir(val_dir) if os.path.exists(val_dir) else []
    test_ids = os.listdir(test_dir) if os.path.exists(test_dir) else []

    train_imgs = sum(len(os.listdir(os.path.join(train_dir, id_name))) for id_name in train_ids)
    val_imgs = sum(len(os.listdir(os.path.join(val_dir, id_name))) for id_name in val_ids)
    test_imgs = sum(len(os.listdir(os.path.join(test_dir, id_name))) for id_name in test_ids)

    print(f"\n1. Filesystem Directory Scan:")
    print(f"  Train Dir: {len(train_ids)} IDs, {train_imgs} images")
    print(f"  Val Dir:   {len(val_ids)} IDs, {val_imgs} images")
    print(f"  Test Dir:  {len(test_ids)} IDs, {test_imgs} images")
    print(f"  Total:     {len(train_ids) + len(val_ids) + len(test_ids)} IDs, {train_imgs + val_imgs + test_imgs} images")

    print("\nSample Identity Names in Train:")
    print(train_ids[:20])
    print("\nSample Identity Names in Val:")
    print(val_ids[:10])
    print("\nSample Identity Names in Test:")
    print(test_ids[:10])

    # 2. Check if identity_0001, identity_0002... exist vs real named identities
    synth_pattern_count = sum(1 for x in train_ids + val_ids + test_ids if x.startswith("identity_"))
    real_named_count = sum(1 for x in train_ids + val_ids + test_ids if not x.startswith("identity_"))
    print(f"\nIdentity Naming Analysis:")
    print(f"  Synthetic naming pattern (identity_XXXX): {synth_pattern_count}")
    print(f"  Real named identities (e.g. angelina_jolie): {real_named_count}")

    # 3. Images per identity statistics BEFORE vs AFTER cap
    images_per_id = []
    all_id_dirs = []
    for s_dir, s_ids in [(train_dir, train_ids), (val_dir, val_ids), (test_dir, test_ids)]:
        for id_name in s_ids:
            id_path = os.path.join(s_dir, id_name)
            cnt = len(os.listdir(id_path))
            images_per_id.append(cnt)

    arr = np.array(images_per_id)
    print(f"\nImages Per Identity Distribution (Filesystem):")
    print(f"  Mean:   {np.mean(arr):.2f}")
    print(f"  Median: {np.median(arr):.2f}")
    print(f"  Std:    {np.std(arr):.2f}")
    print(f"  Min:    {np.min(arr)}")
    print(f"  Max:    {np.max(arr)}")
    print(f"  P10:    {np.percentile(arr, 10):.2f}")
    print(f"  P25:    {np.percentile(arr, 25):.2f}")
    print(f"  P50:    {np.percentile(arr, 50):.2f}")
    print(f"  P75:    {np.percentile(arr, 75):.2f}")
    print(f"  P90:    {np.percentile(arr, 90):.2f}")

    # 4. Identity Leakage
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    print("\nIdentity Leakage Check:")
    print(f"  Train vs Val overlap:  {len(train_set.intersection(val_set))}")
    print(f"  Train vs Test overlap: {len(train_set.intersection(test_set))}")
    print(f"  Val vs Test overlap:   {len(val_set.intersection(test_set))}")

    # 5. Image Authenticity & Visual Inspection (Real Photograph vs Synthetic Drawing)
    print("\nImage Authenticity & Color Histogram Analysis:")
    sample_images = []
    for split_name, s_dir, s_ids in [("train", train_dir, train_ids), ("val", val_dir, val_ids), ("test", test_dir, test_ids)]:
        for id_name in s_ids[:100]:
            id_path = os.path.join(s_dir, id_name)
            files = os.listdir(id_path)
            for f in files[:2]:
                img_path = os.path.join(id_path, f)
                sample_images.append((split_name, id_name, f, img_path))

    unique_color_counts = []
    means, stds, mins, maxs = [], [], [], []
    drawing_count = 0
    real_photo_count = 0

    for split_name, id_name, fname, path in sample_images:
        img = cv2.imread(path)
        if img is None:
            continue
        u_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        unique_color_counts.append(u_colors)
        means.append(np.mean(img))
        stds.append(np.std(img))
        mins.append(np.min(img))
        maxs.append(np.max(img))

        if u_colors < 1000:
            drawing_count += 1
        else:
            real_photo_count += 1

    print(f"  Total Sample Images Tested: {len(sample_images)}")
    print(f"  Real Photographs (>1000 unique colors): {real_photo_count}")
    print(f"  Synthetic Drawings (<1000 unique colors): {drawing_count}")
    print(f"  Mean Unique Colors per 112x112 image: {np.mean(unique_color_counts):.1f}")
    print(f"  Pixel Intensity - Mean: {np.mean(means):.2f}, Std: {np.mean(stds):.2f}, Min: {np.min(mins)}, Max: {np.max(maxs)}")

    # 6. Check underlying raw downloads directory
    print("\nUnderlying Raw Downloads Inspection:")
    if os.path.exists("data/face_recognition/raw"):
        raw_items = os.listdir("data/face_recognition/raw")
        print(f"  data/face_recognition/raw contains {len(raw_items)} items.")
        print(f"  First 10 items in raw: {raw_items[:10]}")

    # Check if there are other directories under data/ or data/face_recognition
    for root, dirs, files in os.walk(data_dir):
        if len(files) > 0 and root.replace("\\", "/").count("/") <= 4:
            print(f"  Subdir: {root} -> {len(files)} files")

if __name__ == "__main__":
    run_audit()
