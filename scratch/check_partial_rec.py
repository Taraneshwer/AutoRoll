import os, struct

rec_path = 'data/tmp/casia_webface/train.rec'
idx_path = 'data/tmp/casia_webface/train.idx'

MAGIC = 0xCED7230A
JPEG_MAGIC = bytes([0xff, 0xd8])

idx_map = {}
with open(idx_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                idx_map[int(parts[0])] = int(parts[1])
            except ValueError:
                pass

rec_size = os.path.getsize(rec_path)
accessible = [(rid, off) for rid, off in idx_map.items() if off < rec_size - 50000]
print(f'Records with offsets < {rec_size/1024/1024:.1f} MB: {len(accessible)}')
if accessible:
    print('First 5:', accessible[:5])

with open(rec_path, 'rb') as f:
    f.seek(0)
    buf = f.read(8)
    magic, length_flag = struct.unpack('<II', buf)
    length = length_flag & ((1 << 29) - 1)
    print(f'First record: magic=0x{magic:08X} (expected 0x{MAGIC:08X}), length={length}')
    is_valid = (magic == MAGIC)
    print(f'Magic valid: {is_valid}')
    if is_valid:
        data = f.read(min(length, 200))
        print(f'First 32 bytes (hex): {data[:32].hex()}')
        jpg_idx = data.find(JPEG_MAGIC)
        print(f'JPEG marker found at byte: {jpg_idx}')
        if jpg_idx != -1:
            label = struct.unpack('<f', data[4:8])[0] if len(data) >= 8 else 0
            print(f'Label (class float): {label}')

# Test reading first accessible record
if accessible:
    rid, off = accessible[0]
    with open(rec_path, 'rb') as f:
        f.seek(off)
        buf = f.read(8)
        magic, length_flag = struct.unpack('<II', buf)
        length = length_flag & ((1 << 29) - 1)
        print(f'Record {rid} @ offset {off}: magic=0x{magic:08X} length={length}')
        if magic == MAGIC:
            data = f.read(min(length, 50))
            jpg_idx = data.find(JPEG_MAGIC)
            print(f'  JPEG at byte: {jpg_idx}')
