"""
Dry-run of the CASIA .rec ingestion pipeline on the first 50 accessible records.
Tests the corrected no-re-alignment path.
"""
import struct, cv2, numpy as np, os

REC = 'data/tmp/casia_webface/train.rec'
IDX = 'data/tmp/casia_webface/train.idx'
LST = 'data/tmp/casia_webface/train.lst'
MAGIC = 0xCED7230A
JPEG = bytes([0xff, 0xd8])

# Parse IDX
idx_map = {}
with open(IDX, 'r') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            try: idx_map[int(p[0])] = int(p[1])
            except: pass

# Parse LST (line_num + 1 = rec_id)
lst_records = {}
with open(LST, 'r') as f:
    for ln, line in enumerate(f):
        p = line.strip().split('\t')
        if len(p) >= 3:
            try:
                class_id = int(p[2])
                identity = os.path.basename(os.path.dirname(p[1]))
                lst_records[ln + 1] = (class_id, identity)
            except: pass
        if ln >= 60000: break

rec_size = os.path.getsize(REC)
accessible = [rid for rid in range(1, 55000) if rid in idx_map and idx_map[rid] < rec_size - 50000]
print(f"Accessible records: {len(accessible)}")

os.makedirs('scratch/download_test', exist_ok=True)
passed = 0
rejected_blur = 0
rejected_bright = 0
rejected_decode = 0

with open(REC, 'rb') as f:
    for rid in accessible[:50]:
        offset = idx_map[rid]
        f.seek(offset)
        buf = f.read(8)
        magic, lf = struct.unpack('<II', buf)
        length = lf & ((1 << 29) - 1)
        data = f.read(length)
        ji = data.find(JPEG)
        if ji == -1:
            rejected_decode += 1
            continue
        arr = np.frombuffer(data[ji:], dtype=np.uint8)
        img = cv2.imdecode(arr, 1)
        if img is None:
            rejected_decode += 1
            continue

        # Quality filter (no re-alignment)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))

        if blur_val < 15.0:
            rejected_blur += 1
            continue
        if brightness < 20 or brightness > 240:
            rejected_bright += 1
            continue

        class_id, identity = lst_records.get(rid, (-1, 'unknown'))
        passed += 1

        if passed <= 5:
            print(f"  rec={rid} class={class_id} id={identity} shape={img.shape} blur={blur_val:.1f} bright={brightness:.1f}")
            chip_path = f"scratch/download_test/noalign_{rid}_{identity}.jpg"
            cv2.imwrite(chip_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

print(f"\nResults (first 50 accessible records):")
print(f"  PASSED: {passed}")
print(f"  BLUR rejected: {rejected_blur}")
print(f"  BRIGHTNESS rejected: {rejected_bright}")
print(f"  DECODE failed: {rejected_decode}")
print(f"  Pass rate: {100.0 * passed / 50:.1f}%")
if passed > 0:
    print(f"\n[PASS] Ingestion pipeline validated")
else:
    print(f"\n[FAIL] Zero records passed")
