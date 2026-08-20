"""
Shape-aware converter from InsightFace ONNX to PyTorch iresnet50.
"""
import os
import sys
import onnx
import torch
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnx import numpy_helper
from autoroll.ml.recognition.iresnet_torch import iresnet50

def convert_onnx_to_iresnet(onnx_path="models/pretrained/arcface_r50_webface_or_glint/model.onnx"):
    m = onnx.load(onnx_path)
    inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

    model = iresnet50()
    sd = model.state_dict()

    # Collect ONNX initializers by operation type
    conv_inits = []
    prelu_inits = []
    
    for node in m.graph.node:
        if node.op_type == "Conv":
            w_name = node.input[1]
            if w_name in inits:
                conv_inits.append((w_name, inits[w_name]))
        elif node.op_type == "PRelu":
            w_name = node.input[1]
            if w_name in inits:
                prelu_inits.append((w_name, inits[w_name]))

    print(f"ONNX Conv count: {len(conv_inits)}, PReLU count: {len(prelu_inits)}")

    # We will match PyTorch state_dict parameters to ONNX initializers by exact shape matching
    pt_conv_items = [(k, v) for k, v in sd.items() if "conv" in k and "weight" in k]
    pt_prelu_items = [(k, v) for k, v in sd.items() if "prelu" in k and "weight" in k]

    print(f"PyTorch Conv params: {len(pt_conv_items)}, PReLU params: {len(pt_prelu_items)}")

    # Match convs sequentially with shape check
    onnx_conv_idx = 0
    new_sd = {}

    for pt_name, pt_tensor in pt_conv_items:
        target_shape = tuple(pt_tensor.shape)
        # Find next ONNX conv with matching shape
        matched = False
        while onnx_conv_idx < len(conv_inits):
            w_name, w_arr = conv_inits[onnx_conv_idx]
            onnx_conv_idx += 1
            if w_arr.shape == target_shape:
                new_sd[pt_name] = torch.from_numpy(w_arr.copy()).float()
                matched = True
                break
        if not matched:
            print(f"Failed to match PyTorch Conv '{pt_name}' (shape {target_shape})")

    # Match prelus sequentially
    onnx_prelu_idx = 0
    for pt_name, pt_tensor in pt_prelu_items:
        target_len = pt_tensor.shape[0]
        while onnx_prelu_idx < len(prelu_inits):
            w_name, w_arr = prelu_inits[onnx_prelu_idx]
            onnx_prelu_idx += 1
            if w_arr.size == target_len:
                new_sd[pt_name] = torch.from_numpy(w_arr.flatten().copy()).float()
                break

    # Copy named keys (BatchNorm, FC, features)
    for k in sd.keys():
        if k in inits:
            new_sd[k] = torch.from_numpy(inits[k].copy()).float()

    model.load_state_dict(new_sd, strict=True)
    model.eval()

    # Validate numerical parity against ONNXRuntime
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    np.random.seed(42)
    dummy_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = (rgb - 127.5) / 127.5
    blob_nchw = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)

    onnx_out = sess.run(None, {in_name: blob_nchw})[0][0]
    onnx_norm = onnx_out / np.linalg.norm(onnx_out)

    with torch.no_grad():
        t = torch.from_numpy(blob_nchw).float()
        pt_out = model(t).numpy()[0]

    cos_sim = float(np.dot(onnx_norm, pt_out) / (np.linalg.norm(onnx_norm) * np.linalg.norm(pt_out)))
    print(f"\n[VALIDATION] ONNX vs PyTorch embedding cosine similarity: {cos_sim:.8f}")
    return model, cos_sim

if __name__ == "__main__":
    convert_onnx_to_iresnet()
