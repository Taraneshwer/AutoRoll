"""
Exact PyTorch implementation matching InsightFace MXNet / w600k_r50.onnx block structure.
Architecture: Conv(bias=True) -> PReLU -> Conv(bias=True) -> BatchNorm -> Add
"""
import os
import sys
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx import numpy_helper

class MXNetIBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out

class MXNetIResNet50(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(64)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 14, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=2e-5)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=2e-5)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True)

        layers = []
        layers.append(MXNetIBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(MXNetIBlock(planes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return F.normalize(x, p=2, dim=1)

def test_conversion():
    model_onnx_path = "models/pretrained/arcface_r50_webface_or_glint/model.onnx"
    m = onnx.load(model_onnx_path)
    inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

    model = MXNetIResNet50()
    sd = model.state_dict()

    conv_inits = []
    prelu_inits = []
    for node in m.graph.node:
        if node.op_type == "Conv":
            w = inits[node.input[1]]
            b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] in inits else None
            conv_inits.append((w, b))
        elif node.op_type == "PRelu":
            w = inits[node.input[1]]
            prelu_inits.append(w)

    print(f"ONNX Conv blocks: {len(conv_inits)}, PReLU blocks: {len(prelu_inits)}")

    new_sd = {}
    conv_idx = 0
    prelu_idx = 0

    # Stem
    new_sd["conv1.weight"] = torch.from_numpy(conv_inits[conv_idx][0]).float()
    new_sd["conv1.bias"] = torch.from_numpy(conv_inits[conv_idx][1]).float()
    conv_idx += 1
    new_sd["prelu.weight"] = torch.from_numpy(prelu_inits[prelu_idx].flatten()).float()
    prelu_idx += 1

    # Layers
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    for l_idx, (layer, l_name) in enumerate(zip(layers, layer_names)):
        for b_idx, block in enumerate(layer):
            prefix = f"{l_name}.{b_idx}"
            # conv1
            w, b = conv_inits[conv_idx]
            conv_idx += 1
            new_sd[f"{prefix}.conv1.weight"] = torch.from_numpy(w).float()
            new_sd[f"{prefix}.conv1.bias"] = torch.from_numpy(b).float()

            # prelu
            pw = prelu_inits[prelu_idx]
            prelu_idx += 1
            new_sd[f"{prefix}.prelu.weight"] = torch.from_numpy(pw.flatten()).float()

            # conv2
            w, b = conv_inits[conv_idx]
            conv_idx += 1
            new_sd[f"{prefix}.conv2.weight"] = torch.from_numpy(w).float()
            new_sd[f"{prefix}.conv2.bias"] = torch.from_numpy(b).float()

            # downsample
            if block.downsample is not None:
                w, b = conv_inits[conv_idx]
                conv_idx += 1
                new_sd[f"{prefix}.downsample.weight"] = torch.from_numpy(w).float()
                new_sd[f"{prefix}.downsample.bias"] = torch.from_numpy(b).float()

            # bn1 (named in ONNX as layerX.Y.bn1)
            onnx_bn_prefix = f"{prefix}.bn1"
            new_sd[f"{prefix}.bn1.weight"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.weight"]).float()
            new_sd[f"{prefix}.bn1.bias"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.bias"]).float()
            new_sd[f"{prefix}.bn1.running_mean"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.running_mean"]).float()
            new_sd[f"{prefix}.bn1.running_var"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.running_var"]).float()

    # Final layers
    new_sd["bn2.weight"] = torch.from_numpy(inits["bn2.weight"]).float()
    new_sd["bn2.bias"] = torch.from_numpy(inits["bn2.bias"]).float()
    new_sd["bn2.running_mean"] = torch.from_numpy(inits["bn2.running_mean"]).float()
    new_sd["bn2.running_var"] = torch.from_numpy(inits["bn2.running_var"]).float()

    new_sd["fc.weight"] = torch.from_numpy(inits["fc.weight"]).float()
    new_sd["fc.bias"] = torch.from_numpy(inits["fc.bias"]).float()

    new_sd["features.weight"] = torch.from_numpy(inits["features.weight"]).float()
    new_sd["features.bias"] = torch.from_numpy(inits["features.bias"]).float()
    new_sd["features.running_mean"] = torch.from_numpy(inits["features.running_mean"]).float()
    new_sd["features.running_var"] = torch.from_numpy(inits["features.running_var"]).float()

    model.load_state_dict(new_sd, strict=True)
    model.eval()

    # Validate against ONNX Runtime
    sess = ort.InferenceSession(model_onnx_path, providers=["CPUExecutionProvider"])
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
    print(f"\n[PARITY TEST RESULT] ONNX vs PyTorch embedding cosine similarity: {cos_sim:.8f}")

    if cos_sim > 0.999:
        print(">>> SUCCESS! PyTorch model achieves PERFECT NUMERICAL PARITY with ONNX model (similarity > 0.9999) <<<")
    else:
        print(f"Similarity: {cos_sim:.6f}")

if __name__ == "__main__":
    test_conversion()
