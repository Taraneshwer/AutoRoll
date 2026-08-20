"""
Tiny Overfit Test for ArcFace Fine-Tuning Pipeline Verification.
Verifies that loss decreases monotonically towards zero on a tiny 2-class dataset.
"""

import sys

import torch
import torch.nn as nn

from autoroll.common.logger import get_logger
from autoroll.ml.training.iresnet import IResNet50
from autoroll.ml.training.losses import ArcMarginProduct

logger = get_logger("overfit_test")


def run_overfit_test():
    logger.info("Running Tiny Overfit Test for ArcFace Training Verification...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic batch of 4 images with 2 distinct classes
    x = torch.randn(4, 3, 112, 112, device=device)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

    model = IResNet50(embedding_size=512).to(device)
    head = ArcMarginProduct(in_features=512, out_features=2, s=64.0, m=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=0.005
    )

    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(1, 15):
        optimizer.zero_grad()
        embeddings = model(x)
        logits = head(embeddings, y)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if epoch == 1:
            initial_loss = current_loss
        final_loss = current_loss

        logger.info(f"Overfit Epoch {epoch:2d} | Loss: {current_loss:.4f}")

    assert final_loss < initial_loss, (
        f"Loss did not decrease! (Initial: {initial_loss:.4f}, Final: {final_loss:.4f})"
    )
    logger.info(
        f"Overfit Test PASSED: Initial Loss={initial_loss:.4f} -> Final Loss={final_loss:.4f}"
    )
    return True


if __name__ == "__main__":
    success = run_overfit_test()
    if not success:
        sys.exit(1)
