import struct, os

idx_path = 'data/tmp/casia_webface/train.idx'
rec_path = 'data/tmp/casia_webface/train.rec'

idx_map = {}
with open(idx_path,'r') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            try: idx_map[int(p[0])] = int(p[1])
            except: pass

print(f"IDX[0]={idx_map.get(0)}")
print(f"IDX[1]={idx_map.get(1)}")
print(f"IDX[2]={idx_map.get(2)}")

MAGIC = 0xCED7230A
JPEG = bytes([0xff, 0xd8])

with open(rec_path, 'rb') as f:
    for rec_id in [1, 2, 3]:
        offset = idx_map[rec_id]
        f.seek(offset)
        buf = f.read(8)
        magic, lf = struct.unpack('<II', buf)
        length = lf & ((1 << 29) - 1)
        data = f.read(min(length, 50))
        label_float = struct.unpack('<f', data[4:8])[0] if len(data) >= 8 else -1
        ji = data.find(JPEG)
        cls = int(round(label_float))
        print(f"rec={rec_id} offset={offset} class_float={label_float:.1f} class_id={cls} jpeg_at={ji}")
