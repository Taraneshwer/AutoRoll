"""
Unit tests for ArcFace PyTorch fine-tuning modules.
"""

import os

import torch

from autoroll.ml.training.checkpoint import CheckpointManager
from autoroll.ml.training.iresnet import IResNet50
from autoroll.ml.training.losses import ArcMarginProduct


def test_iresnet_forward_and_unfreezing():
    model = IResNet50(embedding_size=512)
    x = torch.randn(2, 3, 112, 112)
    embeddings = model(x)

    assert embeddings.shape == (2, 512)

    # Test staged unfreezing
    model.set_unfreezing_stage(1)
    # Check that layer1 is frozen
    for p in model.layer1.parameters():
        assert p.requires_grad is False
    # Check that fc is trainable
    for p in model.fc.parameters():
        assert p.requires_grad is True

    model.set_unfreezing_stage(3)
    for p in model.parameters():
        assert p.requires_grad is True


def test_arc_margin_product_shape():
    head = ArcMarginProduct(in_features=512, out_features=5, s=64.0, m=0.5)
    embeddings = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = head(embeddings, labels)
    assert logits.shape == (4, 5)


def test_checkpoint_manager(tmp_path):
    save_dir = str(tmp_path / "models")
    mgr = CheckpointManager(save_dir=save_dir, experiment_id="exp_test_01")

    backbone_state = {"weight": torch.tensor([1.0])}
    head_state = {"bias": torch.tensor([0.0])}
    opt_state = {}

    latest_path = mgr.save_checkpoint(
        epoch=1,
        backbone_state=backbone_state,
        head_state=head_state,
        optimizer_state=opt_state,
        val_loss=0.5,
        is_best=True,
    )

    assert os.path.exists(latest_path)
    assert os.path.exists(os.path.join(save_dir, "exp_test_01", "best_checkpoint.pt"))

    loaded = mgr.load_checkpoint(latest_path)
    assert loaded["epoch"] == 1
    assert loaded["val_loss"] == 0.5
