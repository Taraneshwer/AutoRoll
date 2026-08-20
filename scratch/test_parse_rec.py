"""
Parser for MXNet RecordIO (.rec) binary face datasets without requiring mxnet.
Extracted images and identity labels directly from InsightFace train.rec / train.idx.
"""
import struct
import io
import cv2
import numpy as np

def read_recordio_header(stream):
    header_fmt = "<II" # magic, length_flag
    buf = stream.read(8)
    if not buf or len(buf) < 8:
        return None
    magic, length_flag = struct.unpack(header_fmt, buf)
    if magic != 0xced7230a:
        # Not a valid RecordIO magic
        return None
    flag = length_flag >> 29
    length = length_flag & ((1 << 29) - 1)
    record_data = stream.read(length)
    # Pad to 4-byte alignment
    pad = (4 - (length % 4)) % 4
    if pad > 0:
        stream.read(pad)
    return record_data

def parse_header_and_image(record_data):
    """
    Parses MXNet RecordHeader (flag, label, id, id2) and JPEG image payload.
    """
    # MXNet RecordHeader2 format:
    # flag (int32), label (float32 or array), id (int64), id2 (int64)
    if len(record_data) < 8:
        return None, None
    flag, label_len = struct.unpack("<II", record_data[:8])
    
    # If flag > 0, it's an indexed header
    if flag > 0:
        # Header with metadata/labels
        header_size = 8 + label_len * 4 + 16
        # Find JPEG header magic \xff\xd8 inside record_data
        jpg_idx = record_data.find(b"\xff\xd8")
        if jpg_idx != -1:
            img_bytes = record_data[jpg_idx:]
            # Parse identity label from header
            label = struct.unpack("<f", record_data[8:12])[0] if label_len > 0 else 0
            return int(label), img_bytes
    else:
        # Simple record: 4-byte float label followed by JPEG
        jpg_idx = record_data.find(b"\xff\xd8")
        if jpg_idx != -1:
            label = struct.unpack("<f", record_data[4:8])[0] if len(record_data) >= 8 else 0
            img_bytes = record_data[jpg_idx:]
            return int(label), img_bytes

    return None, None

def test_parse_sample(rec_file_path):
    print(f"Testing RecordIO parser on '{rec_file_path}'...")
    count = 0
    with open(rec_file_path, "rb") as f:
        while count < 10:
            rec_data = read_recordio_header(f)
            if not rec_data:
                break
            label, img_bytes = parse_header_and_image(rec_data)
            if img_bytes:
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    count += 1
                    print(f"Record {count}: Label={label}, Shape={img.shape}, JPEG size={len(img_bytes)} bytes")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_parse_sample(sys.argv[1])
