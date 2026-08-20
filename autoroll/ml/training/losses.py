"""
ArcFace Margin Product Loss Layer for PyTorch Fine-Tuning.
Ref: ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ArcMarginProduct(nn.Module):
    """
    ArcFace Margin Product layer.
    Applies additive angular margin m to target class cosine similarity scores.
    """

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 10,
        s: float = 64.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot vector
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
