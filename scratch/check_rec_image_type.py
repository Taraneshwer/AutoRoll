"""
Check if CASIA .rec images are already aligned 112x112 chips or full-resolution photos.
"""
import struct, cv2, numpy as np, os

REC = 'data/tmp/casia_webface/train.rec'
IDX = 'data/tmp/casia_webface/train.idx'

MAGIC = 0xCED7230A
JPEG = bytes([0xff, 0xd8])

idx_map = {}
with open(IDX, 'r') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            try: idx_map[int(p[0])] = int(p[1])
            except: pass

print("Checking first 20 records for image dimensions:")
with open(REC, 'rb') as f:
    for rec_id in range(1, 21):
        offset = idx_map.get(rec_id)
        if offset is None: continue
        f.seek(offset)
        buf = f.read(8)
        if len(buf) < 8: continue
        magic, lf = struct.unpack('<II', buf)
        length = lf & ((1 << 29) - 1)
        data = f.read(length)
        ji = data.find(JPEG)
        if ji == -1: continue
        arr = np.frombuffer(data[ji:], dtype=np.uint8)
        img = cv2.imdecode(arr, 1)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  rec={rec_id} shape={img.shape} mean_bright={gray.mean():.1f} max={gray.max()}")

# Now also look at the LST landmarks vs image size:
print("\nChecking LST landmarks for first 5 records:")
with open('data/tmp/casia_webface/train.lst', 'r') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        p = line.strip().split('\t')
        if len(p) >= 17:
            # bbox
            bbox = [float(v) for v in p[3:7]]
            lm = [float(v) for v in p[7:17]]
            print(f"  line={i} bbox={bbox} lm_range_x=[{min(lm[0::2]):.1f},{max(lm[0::2]):.1f}] lm_range_y=[{min(lm[1::2]):.1f},{max(lm[1::2]):.1f}]")
