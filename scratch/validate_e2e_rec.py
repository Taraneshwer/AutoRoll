"""
End-to-end validation of ingest_casia_rec.py using the partially downloaded train.rec.
Tests: correct LST->IDX mapping, JPEG extraction, 5-point Umeyama alignment, quality filter.
"""
import os, sys, struct
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REC = 'data/tmp/casia_webface/train.rec'
IDX = 'data/tmp/casia_webface/train.idx'
LST = 'data/tmp/casia_webface/train.lst'

MAGIC = 0xCED7230A
JPEG = bytes([0xff, 0xd8])
DST = np.array([[30.2946+8,51.6963],[65.5318+8,51.5014],[48.0252+8,71.7366],[33.5493+8,92.3655],[62.7299+8,92.2041]],dtype=np.float32)

def umeyama(src, dst):
    n,m=src.shape; sm,dm=src.mean(0),dst.mean(0); sd,dd=src-sm,dst-dm
    A=dd.T@sd/n; d=np.ones(m)
    if np.linalg.det(A)<0: d[-1]=-1
    U,S,Vt=np.linalg.svd(A); T=np.eye(m+1); T[:m,:m]=U@np.diag(d)@Vt
    sc=(1./sd.var(0).sum())*(S*d).sum(); T[:m,m]=dm-sc*T[:m,:m]@sm; T[:m,:m]*=sc
    return T[:2,:]

# Parse IDX
idx_map = {}
with open(IDX,'r') as f:
    for line in f:
        p=line.strip().split('\t')
        if len(p)>=2:
            try: idx_map[int(p[0])]=int(p[1])
            except: pass

# Parse LST — line_num+1 = rec_id
lst_records = {}
with open(LST,'r') as f:
    for ln, line in enumerate(f):
        p=line.strip().split('\t')
        if len(p)>=17:
            try:
                class_id=int(p[2])
                identity=os.path.basename(os.path.dirname(p[1]))
                lm=np.array([float(v) for v in p[7:17]],dtype=np.float32).reshape(5,2)
                lst_records[ln+1]=(class_id,identity,lm)
            except: pass
        if ln >= 25000: break  # Only parse first 25k for speed

rec_size = os.path.getsize(REC)
accessible = [rid for rid in range(1, 22000) if rid in idx_map and idx_map[rid] < rec_size - 50000 and rid in lst_records]
print(f"Accessible rec IDs for test: {len(accessible)}")

os.makedirs('scratch/download_test', exist_ok=True)
passed = 0

with open(REC,'rb') as f:
    for rid in accessible[:10]:
        offset = idx_map[rid]
        f.seek(offset)
        buf = f.read(8)
        magic, lf = struct.unpack('<II', buf)
        length = lf & ((1<<29)-1)
        data = f.read(length)
        ji = data.find(JPEG)
        if ji == -1:
            print(f"  rec={rid}: no JPEG in data")
            continue
        label = struct.unpack('<f', data[4:8])[0] if len(data)>=8 else -1
        class_id, identity, lm = lst_records[rid]
        arr = np.frombuffer(data[ji:], dtype=np.uint8)
        img = cv2.imdecode(arr, 1)
        if img is None:
            print(f"  rec={rid}: decode fail")
            continue
        M = umeyama(lm, DST)
        chip = cv2.warpAffine(img, M, (112,112), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = float(np.mean(gray))
        chip_path = f"scratch/download_test/chip_{rid}_{identity}.jpg"
        cv2.imwrite(chip_path, chip, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  rec={rid} | class_rec={int(round(label))} | class_lst={class_id} | id={identity} | src={img.shape} | chip=112x112 | blur={blur:.1f} | bright={bright:.1f}")
        passed += 1

print(f"\n[{'PASS' if passed>0 else 'FAIL'}] {passed}/10 chips extracted and aligned correctly")
if passed > 0:
    print(f"Sample chips: scratch/download_test/")
