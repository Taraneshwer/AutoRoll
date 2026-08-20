"""
Test PyTorch iresnet50 GPU embedding extraction speed on RTX 5060.
"""
import os
import sys
import time
import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoroll.ml.recognition.iresnet_torch import iresnet50
from scratch.convert_onnx_to_iresnet import convert_onnx_to_iresnet

def test_gpu():
    print("Testing PyTorch iresnet50 on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Load model onto GPU
    model, _ = convert_onnx_to_iresnet("models/pretrained/arcface_r50_webface_or_glint/model.onnx")
    model = model.to(device)
    model.eval()

    val_dir = "data/face_recognition/splits/val"
    sample_imgs = []
    for root, _, files in os.walk(val_dir):
        for f in files:
            sample_imgs.append(os.path.join(root, f))
            if len(sample_imgs) >= 3000:
                break
        if len(sample_imgs) >= 3000:
            break

    print(f"Loaded {len(sample_imgs)} image paths. Benchmark starting...")
    t0 = time.time()
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(sample_imgs), batch_size):
            chunk = sample_imgs[i:i+batch_size]
            blobs = []
            for p in chunk:
                img = cv2.imread(p)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                blob = (rgb - 127.5) / 127.5
                blobs.append(np.transpose(blob, (2, 0, 1)))
            batch_t = torch.from_numpy(np.array(blobs, dtype=np.float32)).float().to(device)
            out = model(batch_t)
            
    elapsed = time.time() - t0
    print(f"Extracted {len(sample_imgs)} embeddings in {elapsed:.2f}s ({len(sample_imgs)/elapsed:.1f} img/s) on GPU!")

if __name__ == "__main__":
    test_gpu()
