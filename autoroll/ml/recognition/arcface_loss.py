"""
PyTorch Implementation of ArcFace Additive Angular Margin Loss.
Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace Additive Angular Margin Loss Head.
    Computes cosine similarity logits with additive angular margin:
        cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
    """

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 10,
        scale: float = 64.0,
        margin: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Trainable class weights normalized on sphere
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trigonometric terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ArcFace Loss.
        embeddings: Normalized 512-D feature vectors (batch_size, 512)
        label: Target class indices (batch_size,)
        Returns scalar cross-entropy loss with additive angular margin.
        """
        # Normalize weights & embeddings to unit hypersphere
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        
        # Clamp cosine for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot target encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        
        # Apply margin only to target class logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = F.cross_entropy(output, label)
        return loss
