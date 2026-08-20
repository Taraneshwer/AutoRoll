"""
IResNet50 PyTorch Architecture with Staged Unfreezing Support.
Outputs 512-dimensional normalized embedding vectors.
"""

import torch
import torch.nn as nn

from app.core.logger import get_logger

logger = get_logger("iresnet_model")


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IResNet50(nn.Module):
    """
    IResNet50 Face Feature Extractor producing 512-dimensional embeddings.
    """

    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64, num_blocks=3, stride=2)
        self.layer2 = self._make_layer(128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=14, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=3, stride=2)

        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [IBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(IBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x

    def set_unfreezing_stage(self, stage: int) -> None:
        """
        Staged unfreezing to prevent catastrophic forgetting:
        Stage 1: Freeze stem, layer1, layer2, layer3 -> Train layer4, fc, features
        Stage 2: Freeze stem, layer1, layer2 -> Train layer3, layer4, fc, features
        Stage 3: Unfreeze all layers
        """
        logger.info(f"Setting IResNet50 Staged Unfreezing -> Stage {stage}")

        if stage == 1:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = True
            for p in self.fc.parameters():
                p.requires_grad = True
            for p in self.features.parameters():
                p.requires_grad = True
        elif stage == 2:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = True
            for p in self.layer4.parameters():
                p.requires_grad = True
            for p in self.fc.parameters():
                p.requires_grad = True
            for p in self.features.parameters():
                p.requires_grad = True
        else:  # Stage 3
            for p in self.parameters():
                p.requires_grad = True
