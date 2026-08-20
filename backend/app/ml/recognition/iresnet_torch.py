"""
PyTorch IResNet50 model matching InsightFace MXNet ONNX backbone (w600k_r50.onnx).
"""
import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def set_staged_freeze(self, stage: int = 1):
        """
        Stage 1: Freeze early layers (stem, layer1, layer2, layer3).
                 Train layer4, bn2, fc, features.
        Stage 2: Unfreeze all layers for end-to-end training.
        """
        if stage == 1:
            for module in [self.conv1, self.prelu, self.layer1, self.layer2, self.layer3]:
                for param in module.parameters():
                    param.requires_grad = False
            for module in [self.layer4, self.bn2, self.fc, self.features]:
                for param in module.parameters():
                    param.requires_grad = True
        elif stage == 2:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x):
        if not self.conv1.weight.requires_grad:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.prelu(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
        else:
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


def load_pretrained_onnx_weights(onnx_path: str, model: MXNetIResNet50) -> MXNetIResNet50:
    """
    Loads initializers from InsightFace ONNX model directly into PyTorch MXNetIResNet50.
    """
    m = onnx.load(onnx_path)
    inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

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

    new_sd = {}
    conv_idx = 0
    prelu_idx = 0

    # Stem
    new_sd["conv1.weight"] = torch.from_numpy(conv_inits[conv_idx][0].copy()).float()
    new_sd["conv1.bias"] = torch.from_numpy(conv_inits[conv_idx][1].copy()).float()
    conv_idx += 1
    new_sd["prelu.weight"] = torch.from_numpy(prelu_inits[prelu_idx].copy().flatten()).float()
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
            new_sd[f"{prefix}.conv1.weight"] = torch.from_numpy(w.copy()).float()
            new_sd[f"{prefix}.conv1.bias"] = torch.from_numpy(b.copy()).float()

            # prelu
            pw = prelu_inits[prelu_idx]
            prelu_idx += 1
            new_sd[f"{prefix}.prelu.weight"] = torch.from_numpy(pw.copy().flatten()).float()

            # conv2
            w, b = conv_inits[conv_idx]
            conv_idx += 1
            new_sd[f"{prefix}.conv2.weight"] = torch.from_numpy(w.copy()).float()
            new_sd[f"{prefix}.conv2.bias"] = torch.from_numpy(b.copy()).float()

            # downsample
            if block.downsample is not None:
                w, b = conv_inits[conv_idx]
                conv_idx += 1
                new_sd[f"{prefix}.downsample.weight"] = torch.from_numpy(w.copy()).float()
                new_sd[f"{prefix}.downsample.bias"] = torch.from_numpy(b.copy()).float()

            # bn1
            onnx_bn_prefix = f"{prefix}.bn1"
            new_sd[f"{prefix}.bn1.weight"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.weight"].copy()).float()
            new_sd[f"{prefix}.bn1.bias"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.bias"].copy()).float()
            new_sd[f"{prefix}.bn1.running_mean"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.running_mean"].copy()).float()
            new_sd[f"{prefix}.bn1.running_var"] = torch.from_numpy(inits[f"{onnx_bn_prefix}.running_var"].copy()).float()

    # Final layers
    new_sd["bn2.weight"] = torch.from_numpy(inits["bn2.weight"].copy()).float()
    new_sd["bn2.bias"] = torch.from_numpy(inits["bn2.bias"].copy()).float()
    new_sd["bn2.running_mean"] = torch.from_numpy(inits["bn2.running_mean"].copy()).float()
    new_sd["bn2.running_var"] = torch.from_numpy(inits["bn2.running_var"].copy()).float()

    new_sd["fc.weight"] = torch.from_numpy(inits["fc.weight"].copy()).float()
    new_sd["fc.bias"] = torch.from_numpy(inits["fc.bias"].copy()).float()

    new_sd["features.weight"] = torch.from_numpy(inits["features.weight"].copy()).float()
    new_sd["features.bias"] = torch.from_numpy(inits["features.bias"].copy()).float()
    new_sd["features.running_mean"] = torch.from_numpy(inits["features.running_mean"].copy()).float()
    new_sd["features.running_var"] = torch.from_numpy(inits["features.running_var"].copy()).float()

    model.load_state_dict(new_sd, strict=True)
    return model


def get_iresnet50(onnx_pretrained_path: str = None) -> MXNetIResNet50:
    model = MXNetIResNet50()
    if onnx_pretrained_path and os.path.exists(onnx_pretrained_path):
        model = load_pretrained_onnx_weights(onnx_pretrained_path, model)
    return model


iresnet50 = get_iresnet50
