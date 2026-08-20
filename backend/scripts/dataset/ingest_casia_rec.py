"""
AutoRoll ML Phase 5.4 — CASIA-WebFace .rec Ingestion Pipeline.

Ingests the genuine CASIA-WebFace dataset from MXNet RecordIO format (.rec + .idx + .lst)
WITHOUT requiring mxnet.

CASIA-WebFace Details:
  - 10,572 identities, ~494,149 images, ~46.7 images/identity
  - Source: Dong Yi et al., "Learning Face Representation from Scratch", arXiv:1411.7923
  - Original repository: https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace

IMPORTANT: The CASIA-WebFace .rec images are ALREADY pre-aligned 112x112 face chips produced
by the original InsightFace pipeline. We do NOT re-apply Umeyama alignment on them — the
.lst landmarks reference the original (larger) image coordinates. Re-applying alignment on
already-aligned chips would corrupt the images. We use the chips directly.

The .lst file is used only to map rec_id -> identity folder name (and as a record index).

STRICT RULE: No synthetic data generation. Dataset extraction ONLY from real .rec archives.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import os
import sys
import json
import struct
import time
import hashlib
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_casia_rec")

# ---------------------------------------------------------------------------
# Umeyama 5-point alignment constants (InsightFace standard 112x112 template)
# ---------------------------------------------------------------------------
ARCFACE_DST_5PT = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041],
], dtype=np.float32)

# Shift for 112x112 (InsightFace ArcFace template)
ARCFACE_DST_5PT[:, 0] += 8.0


def umeyama_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Compute Umeyama similarity transform between 5 corresponding 2D point pairs.
    Returns a 2x3 affine transformation matrix for cv2.warpAffine.
    """
    n, m = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / n
    d = np.ones((m,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[-1] = -1
    U, S, Vt = np.linalg.svd(A)
    T = np.eye(m + 1, dtype=np.float64)
    T[:m, :m] = U @ np.diag(d) @ Vt
    scale = (1.0 / src_demean.var(axis=0).sum()) * (S * d).sum()
    T[:m, m] = dst_mean - scale * T[:m, :m] @ src_mean
    T[:m, :m] *= scale
    return T[:2, :]


def align_face_5pt(img: np.ndarray, landmarks_5pt: np.ndarray) -> np.ndarray:
    """
    Applies 5-point similarity transform alignment to produce a 112x112 ArcFace chip.
    landmarks_5pt: shape (5, 2) float32 — (le, re, nose, lm, rm) x/y pairs
    """
    M = umeyama_transform(landmarks_5pt, ARCFACE_DST_5PT)
    aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned


# ---------------------------------------------------------------------------
# RecordIO Parser (mxnet-free)
# ---------------------------------------------------------------------------
RECORDIO_MAGIC = 0xCED7230A


def parse_idx_file(idx_path: str) -> dict:
    """
    Parse InsightFace .idx file (text format: <record_id>\\t<byte_offset>).
    Returns {record_id: byte_offset}.
    """
    idx_map = {}
    with open(idx_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    rec_id = int(parts[0])
                    offset = int(parts[1])
                    idx_map[rec_id] = offset
                except ValueError:
                    continue
    return idx_map


def read_record_at_offset(rec_file, offset: int) -> Optional[bytes]:
    """
    Read one RecordIO record from the given byte offset.
    Returns raw record bytes or None if invalid.
    """
    rec_file.seek(offset)
    header_buf = rec_file.read(8)
    if len(header_buf) < 8:
        return None
    magic, length_flag = struct.unpack("<II", header_buf)
    if magic != RECORDIO_MAGIC:
        return None
    length = length_flag & ((1 << 29) - 1)
    record_data = rec_file.read(length)
    if len(record_data) < length:
        return None
    return record_data


def extract_image_from_record(record_data: bytes) -> tuple[Optional[int], Optional[bytes]]:
    """
    Extract (class_label, jpeg_bytes) from a raw RecordIO record.
    InsightFace format: first 4 bytes = flag, next 4 bytes = label (float32),
    then id1 (int64), id2 (int64), then JPEG data.
    """
    if len(record_data) < 8:
        return None, None

    # Search for JPEG magic bytes
    jpg_idx = record_data.find(b"\xff\xd8")
    if jpg_idx == -1:
        return None, None

    # Extract label (class ID stored as float32)
    try:
        label = struct.unpack("<f", record_data[4:8])[0]
        class_id = int(round(label))
    except Exception:
        class_id = 0

    jpeg_bytes = record_data[jpg_idx:]
    return class_id, jpeg_bytes


# ---------------------------------------------------------------------------
# LST Parser — pre-computed 5-point landmarks
# ---------------------------------------------------------------------------
def parse_lst_file(lst_path: str) -> dict:
    """
    Parse InsightFace train.lst for pre-computed 5-point landmarks.
    Format: <label_idx>\\t<path>\\t<class_id>\\t<bbox_x1>\\t<bbox_y1>\\t<bbox_x2>\\t<bbox_y2>\\t<lm_x1>\\t<lm_y1>...

    IMPORTANT: The IDX record ID = LST line number + 1 (IDX is 1-indexed, records 1..N map to lines 0..N-1).
    The first column in LST is the class_id label (not the record index).
    The third column is the class_id (same value). Identity folder name is derived from path.

    Returns {rec_id: (class_id, identity_name, landmarks_5pt)}
    """
    logger.info(f"Parsing LST file: {lst_path}")
    records = {}
    with open(lst_path, "r") as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                # IDX record IDs are 1-indexed (LST line 0 -> rec_id 1)
                rec_id = line_num + 1
                # Column 2 (index 2) is the numeric class ID
                class_id = int(parts[2])
                # Identity folder name from path (e.g. /raid5data/.../0000045/001.jpg -> 0000045)
                path = parts[1]
                identity_name = os.path.basename(os.path.dirname(path))

                # Landmarks: 5 points × 2 coords = 10 floats, starting at column 7
                if len(parts) >= 17:
                    lm_vals = [float(v) for v in parts[7:17]]
                    landmarks = np.array(lm_vals, dtype=np.float32).reshape(5, 2)
                else:
                    landmarks = None

                records[rec_id] = (class_id, identity_name, landmarks)
            except (ValueError, IndexError):
                continue

    logger.info(f"Parsed {len(records)} records from LST file.")
    return records


# ---------------------------------------------------------------------------
# Quality Filter
# ---------------------------------------------------------------------------
def quality_filter(img: np.ndarray, min_blur: float = 15.0) -> tuple[bool, str]:
    """
    Returns (passed, reason). Checks blur and brightness on aligned 112x112 chip.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_val < min_blur:
        return False, "BLUR"
    brightness = float(np.mean(gray))
    if brightness < 20 or brightness > 240:
        return False, "EXTREME_BRIGHTNESS"
    return True, "OK"


# ---------------------------------------------------------------------------
# SHA256 Hashing (for provenance tracking — skip exact duplicate detection
# since .rec extraction is deterministic by construction)
# ---------------------------------------------------------------------------
def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main Ingestion
# ---------------------------------------------------------------------------
def ingest_casia_rec(
    rec_path: str,
    idx_path: str,
    lst_path: str,
    dest_dir: str = "data/face_recognition",
    dataset_name: str = "CASIA-WebFace",
    max_images_per_id: Optional[int] = None,
    target_identities: Optional[int] = None,
    min_blur: float = 15.0,
    num_workers: int = 4,
) -> dict:
    """
    Full CASIA-WebFace ingestion from .rec/.idx/.lst into AutoRoll aligned dataset format.

    Arguments:
        rec_path: Path to train.rec
        idx_path: Path to train.idx
        lst_path: Path to train.lst
        dest_dir: Output directory for aligned dataset
        dataset_name: Name tag for provenance
        max_images_per_id: Optional cap per identity
        target_identities: Optional limit on number of identities to process
        min_blur: Minimum Laplacian variance for sharpness filtering
        num_workers: Worker threads for image decode/filter (not for .rec reads)
    """
    logger.info("=" * 80)
    logger.info("AUTOROLL CASIA-WEBFACE .rec INGESTION PIPELINE")
    logger.info("=" * 80)

    t_start = time.time()

    # --- Validate sources ---
    for fpath, label in [(rec_path, "train.rec"), (idx_path, "train.idx"), (lst_path, "train.lst")]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required CASIA file not found: '{fpath}' ({label})")
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"  {label}: {size_mb:.1f} MB")

    # --- Setup output directories ---
    aligned_dir = os.path.join(dest_dir, "aligned")
    splits_dir = os.path.join(dest_dir, "splits")
    metadata_dir = os.path.join(dest_dir, "metadata")
    for d in [aligned_dir, splits_dir, metadata_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Parse LST ---
    lst_records = parse_lst_file(lst_path)

    # --- Parse IDX ---
    logger.info(f"Parsing IDX file: {idx_path}")
    idx_map = parse_idx_file(idx_path)
    logger.info(f"IDX map: {len(idx_map)} entries")

    # --- Build identity -> record_ids map ---
    identity_to_records: dict[str, list[int]] = {}
    class_id_to_identity: dict[int, str] = {}

    for rec_idx, (class_id, identity_name, landmarks) in lst_records.items():
        # Normalise identity name (CASIA folder names are zero-padded 7-digit numbers)
        identity_to_records.setdefault(identity_name, []).append(rec_idx)
        class_id_to_identity[class_id] = identity_name

    all_identities = sorted(identity_to_records.keys())
    total_source_ids = len(all_identities)
    logger.info(f"Total identities in LST: {total_source_ids}")
    logger.info(f"Total records in LST: {len(lst_records)}")

    # Apply identity cap if specified
    if target_identities and target_identities < total_source_ids:
        np.random.seed(42)
        selected_ids = np.random.choice(all_identities, size=target_identities, replace=False).tolist()
        logger.info(f"Sampling {target_identities} identities from {total_source_ids} total.")
    else:
        selected_ids = all_identities
        logger.info(f"Processing all {total_source_ids} identities.")

    # --- Process Records ---
    rejection_stats = {
        "MISSING_IDX": 0,
        "INVALID_RECORD": 0,
        "JPEG_DECODE_FAIL": 0,
        "BLUR": 0,
        "EXTREME_BRIGHTNESS": 0,
    }

    processed_aligned_by_id: dict[str, list[str]] = {}
    provenance_records = []
    processed_count = 0
    passed_count = 0

    logger.info(f"Opening .rec file: {rec_path}")
    with open(rec_path, "rb") as rec_file:
        total_ids = len(selected_ids)
        for id_idx, identity_name in enumerate(selected_ids):
            record_ids = identity_to_records[identity_name]
            if max_images_per_id:
                record_ids = record_ids[:max_images_per_id]

            id_aligned_dir = os.path.join(aligned_dir, identity_name)
            os.makedirs(id_aligned_dir, exist_ok=True)
            processed_aligned_by_id[identity_name] = []

            for chip_idx, rec_idx in enumerate(record_ids):
                processed_count += 1

                # Get byte offset
                if rec_idx not in idx_map:
                    rejection_stats["MISSING_IDX"] += 1
                    continue

                offset = idx_map[rec_idx]
                record_data = read_record_at_offset(rec_file, offset)
                if record_data is None:
                    rejection_stats["INVALID_RECORD"] += 1
                    continue

                # Extract JPEG bytes
                class_id, jpeg_bytes = extract_image_from_record(record_data)
                if jpeg_bytes is None:
                    rejection_stats["JPEG_DECODE_FAIL"] += 1
                    continue

                # Decode image — CASIA .rec images are already 112x112 pre-aligned chips
                img_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img is None:
                    rejection_stats["JPEG_DECODE_FAIL"] += 1
                    continue

                # Quality filter on the pre-aligned chip (no re-alignment needed)
                passed_qf, reason = quality_filter(img, min_blur=min_blur)
                if not passed_qf:
                    rejection_stats[reason] += 1
                    continue

                # Save chip as-is (already 112x112 ArcFace-aligned)
                chip_filename = f"{identity_name}_chip_{chip_idx:04d}.jpg"
                chip_path = os.path.join(id_aligned_dir, chip_filename)
                cv2.imwrite(chip_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                passed_count += 1
                processed_aligned_by_id[identity_name].append(chip_path)

                sha256 = compute_sha256_bytes(jpeg_bytes)
                provenance_records.append({
                    "source_dataset": dataset_name,
                    "source_identity": identity_name,
                    "rec_record_id": rec_idx,
                    "source_class_id": class_id,
                    "source_sha256": sha256,
                    "processed_path": chip_path,
                    "preprocessing_version": "2.1.0_casia_rec_prealigned_112x112",
                })

            # Progress log every 100 identities
            if (id_idx + 1) % 100 == 0 or (id_idx + 1) == total_ids:
                elapsed = time.time() - t_start
                rate = passed_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{id_idx + 1}/{total_ids}] identities | "
                    f"{passed_count} chips passed | {rate:.1f} img/s | "
                    f"{elapsed:.0f}s elapsed"
                )

    elapsed_time = time.time() - t_start
    throughput = passed_count / elapsed_time if elapsed_time > 0 else 0

    # --- Identity-Disjoint Splits (80/10/10) ---
    valid_ids = [iid for iid, chips in processed_aligned_by_id.items() if len(chips) >= 5]
    logger.info(f"Valid identities (≥5 chips): {len(valid_ids)} / {len(selected_ids)}")

    np.random.seed(42)
    shuffled_ids = np.array(valid_ids)
    np.random.shuffle(shuffled_ids)

    n_valid = len(shuffled_ids)
    n_train = int(n_valid * 0.80)
    n_val = int(n_valid * 0.10)

    train_ids = list(shuffled_ids[:n_train])
    val_ids = list(shuffled_ids[n_train : n_train + n_val])
    test_ids = list(shuffled_ids[n_train + n_val :])

    split_counts = {"train": 0, "val": 0, "test": 0}
    split_id_lists = [("train", train_ids), ("val", val_ids), ("test", test_ids)]

    for split_name, split_id_list in split_id_lists:
        split_dir = os.path.join(splits_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for iid in split_id_list:
            id_split_dir = os.path.join(split_dir, iid)
            os.makedirs(id_split_dir, exist_ok=True)
            for chip_path in processed_aligned_by_id[iid]:
                fname = os.path.basename(chip_path)
                dest_chip_path = os.path.join(id_split_dir, fname)
                # Hard-link if possible, else copy
                if not os.path.exists(dest_chip_path):
                    import shutil
                    shutil.copy2(chip_path, dest_chip_path)
                split_counts[split_name] += 1

    logger.info(
        f"Splits: train={split_counts['train']} imgs ({len(train_ids)} ids) | "
        f"val={split_counts['val']} imgs ({len(val_ids)} ids) | "
        f"test={split_counts['test']} imgs ({len(test_ids)} ids)"
    )

    # --- Source Manifest ---
    source_manifest = {
        "dataset_name": dataset_name,
        "dataset_version": "1.0.0",
        "dataset_type": "real",
        "synthetic": False,
        "source_url": "https://huggingface.co/datasets/Pijush22049/casia-webface",
        "official_source": "https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace",
        "citation": (
            "Dong Yi, Zhen Lei, Shengcai Liao, Stan Z. Li, "
            "'Learning Face Representation from Scratch', arXiv:1411.7923, 2014."
        ),
        "license": "Non-Commercial Research Use Only (CASIA-WebFace original terms)",
        "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_rec_path": rec_path,
        "source_idx_path": idx_path,
        "source_lst_path": lst_path,
        "source_image_count": len(lst_records),
        "source_identity_count": total_source_ids,
        "selected_identity_count": len(selected_ids),
        "local_image_count": passed_count,
        "local_identity_count": len(valid_ids),
        "train_identity_count": len(train_ids),
        "val_identity_count": len(val_ids),
        "test_identity_count": len(test_ids),
        "train_image_count": split_counts["train"],
        "val_image_count": split_counts["val"],
        "test_image_count": split_counts["test"],
        "preprocessing_version": "2.1.0_casia_rec_prealigned_112x112",
        "quality_statistics": {
            "total_processed": processed_count,
            "total_passed": passed_count,
            "pass_rate_pct": round(100.0 * passed_count / max(processed_count, 1), 2),
            "resolution": "112x112",
            "min_blur_threshold": min_blur,
            "landmark_source": "CASIA-WebFace pre-aligned .rec chips (no re-alignment applied)",
        },
        "rejection_statistics": rejection_stats,
        "performance": {
            "processing_time_sec": round(elapsed_time, 2),
            "images_per_sec": round(throughput, 2),
        },
    }

    manifest_path = os.path.join(metadata_dir, "source_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(source_manifest, f, indent=2)
    logger.info(f"Source manifest saved: {manifest_path}")

    provenance_path = os.path.join(metadata_dir, "provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)
    logger.info(f"Provenance log saved: {provenance_path}")

    logger.info("=" * 80)
    logger.info(
        f"INGESTION COMPLETE | {passed_count}/{processed_count} chips passed | "
        f"{len(valid_ids)} identities | {throughput:.1f} img/s | {elapsed_time:.0f}s"
    )
    logger.info("=" * 80)

    return source_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AutoRoll CASIA-WebFace .rec Ingestion Pipeline (mxnet-free)"
    )
    parser.add_argument(
        "--rec", type=str, default="data/tmp/casia_webface/train.rec",
        help="Path to train.rec"
    )
    parser.add_argument(
        "--idx", type=str, default="data/tmp/casia_webface/train.idx",
        help="Path to train.idx"
    )
    parser.add_argument(
        "--lst", type=str, default="data/tmp/casia_webface/train.lst",
        help="Path to train.lst"
    )
    parser.add_argument(
        "--dest", type=str, default="data/face_recognition",
        help="Output directory for aligned dataset"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="CASIA-WebFace",
        help="Dataset name for provenance"
    )
    parser.add_argument(
        "--max-images-per-id", type=int, default=None,
        help="Optional cap on images per identity"
    )
    parser.add_argument(
        "--target-identities", type=int, default=None,
        help="Optional limit: process only N randomly selected identities"
    )
    parser.add_argument(
        "--min-blur", type=float, default=15.0,
        help="Minimum Laplacian variance for blur filter"
    )
    args = parser.parse_args()

    ingest_casia_rec(
        rec_path=args.rec,
        idx_path=args.idx,
        lst_path=args.lst,
        dest_dir=args.dest,
        dataset_name=args.dataset_name,
        max_images_per_id=args.max_images_per_id,
        target_identities=args.target_identities,
        min_blur=args.min_blur,
    )


if __name__ == "__main__":
    main()
