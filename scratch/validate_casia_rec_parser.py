"""
Quick validation of the CASIA .rec parser against the partially downloaded train.rec.
Tests: IDX offset reading, JPEG extraction, 5-point landmark alignment.
"""
import os
import sys
import struct
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REC_PATH = "data/tmp/casia_webface/train.rec"
IDX_PATH = "data/tmp/casia_webface/train.idx"
LST_PATH = "data/tmp/casia_webface/train.lst"

ARCFACE_DST_5PT = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041],
], dtype=np.float32)
ARCFACE_DST_5PT[:, 0] += 8.0

RECORDIO_MAGIC = 0xCED7230A


def umeyama_transform(src, dst):
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


def parse_idx(idx_path):
    idx_map = {}
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    idx_map[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
    return idx_map


def read_record(rec_file, offset):
    rec_file.seek(offset)
    buf = rec_file.read(8)
    if len(buf) < 8:
        return None
    magic, length_flag = struct.unpack("<II", buf)
    if magic != RECORDIO_MAGIC:
        return None
    length = length_flag & ((1 << 29) - 1)
    data = rec_file.read(length)
    return data if len(data) == length else None


def extract_jpeg(record_data):
    if len(record_data) < 8:
        return None, None
    jpg_idx = record_data.find(b"\xff\xd8")
    if jpg_idx == -1:
        return None, None
    label = struct.unpack("<f", record_data[4:8])[0] if len(record_data) >= 8 else 0
    return int(round(label)), record_data[jpg_idx:]


def parse_lst_sample(lst_path, n=20):
    records = {}
    with open(lst_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 17:
                try:
                    rec_idx = int(parts[0])
                    class_id = int(parts[2])
                    identity = os.path.basename(os.path.dirname(parts[1]))
                    lm_vals = [float(v) for v in parts[7:17]]
                    landmarks = np.array(lm_vals, dtype=np.float32).reshape(5, 2)
                    records[rec_idx] = (class_id, identity, landmarks)
                    if len(records) >= n:
                        break
                except (ValueError, IndexError):
                    pass
    return records


def main():
    print("=" * 60)
    print("CASIA .rec Parser Validation")
    print("=" * 60)

    rec_size = os.path.getsize(REC_PATH)
    print(f"train.rec size (partial): {rec_size / (1024*1024):.1f} MB")

    idx_map = parse_idx(IDX_PATH)
    print(f"IDX entries: {len(idx_map)}")
    print(f"First 5 IDX entries: { {k: v for k, v in list(idx_map.items())[:5]} }")

    lst_sample = parse_lst_sample(LST_PATH, n=20)
    print(f"LST sample records: {len(lst_sample)}")

    # Test reading records that fall within the already-downloaded portion
    accessible_rec_ids = [
        rec_id for rec_id, offset in idx_map.items()
        if offset < rec_size - 8192 and rec_id in lst_sample
    ]
    print(f"Records accessible within {rec_size / (1024*1024):.1f} MB: {len(accessible_rec_ids)}")

    success = 0
    failures = 0
    chips_saved = []

    os.makedirs("scratch/download_test", exist_ok=True)

    with open(REC_PATH, "rb") as rec_file:
        for rec_id in accessible_rec_ids[:10]:
            offset = idx_map[rec_id]
            record_data = read_record(rec_file, offset)
            if record_data is None:
                print(f"  Record {rec_id} @ offset {offset}: INVALID RECORD")
                failures += 1
                continue

            class_id, jpeg_bytes = extract_jpeg(record_data)
            if jpeg_bytes is None:
                print(f"  Record {rec_id} @ offset {offset}: NO JPEG FOUND")
                failures += 1
                continue

            img_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  Record {rec_id}: JPEG DECODE FAIL")
                failures += 1
                continue

            _, identity, landmarks = lst_sample[rec_id]

            # 5-point alignment
            M = umeyama_transform(landmarks, ARCFACE_DST_5PT)
            aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)

            # Quality check
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = float(np.mean(gray))

            chip_path = f"scratch/download_test/chip_{rec_id}_{identity}.jpg"
            cv2.imwrite(chip_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            chips_saved.append(chip_path)

            print(
                f"  Record {rec_id} | class={class_id} | id={identity} | "
                f"src_shape={img.shape} | blur={blur_val:.1f} | brightness={brightness:.1f} | OK"
            )
            success += 1

    print(f"\nResults: {success} success, {failures} failures")
    print(f"Sample chips saved to: scratch/download_test/")
    if chips_saved:
        print(f"  Example: {chips_saved[0]}")

    print("\n[PASS] CASIA .rec parser validation complete." if success > 0 else "\n[FAIL] No records extracted.")


if __name__ == "__main__":
    main()
