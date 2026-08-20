"""
PyTorch ArcFace Pilot Fine-Tuning Engine with AMP Mixed Precision.
Executes controlled CUDA fine-tuning, tracks VRAM usage, loss convergence, and saves/restores checkpoints.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from autoroll.common.logger import get_logger
from autoroll.common.config import get_settings
from autoroll.ml.recognition.iresnet_torch import iresnet50
from autoroll.ml.recognition.arcface_loss import ArcFaceLoss

logger = get_logger("arcface_pilot_trainer")


def verify_real_dataset_guard(
    manifest_path: str = "data/face_recognition/metadata/source_manifest.json",
    min_identities: int = 10,
    min_images: int = 100,
):
    """
    STRICT TRAINING GUARD:
    Verifies that a valid real face dataset source_manifest.json exists and satisfies
    all authenticity and scale requirements before ArcFace training can execute.
    """
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"REAL TRAINING DATASET GUARD FAILED: Source manifest missing at '{manifest_path}'. "
            "Real dataset ingestion is mandatory before ArcFace fine-tuning. "
            "Run 'python scripts/ingest_real_dataset.py --source /path/to/real/dataset'."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("synthetic", True) is True or manifest.get("dataset_type") != "real":
        raise RuntimeError("REAL TRAINING DATASET GUARD FAILED: Dataset is synthetic. Synthetic training is prohibited.")

    if not manifest.get("source_url") or not manifest.get("license"):
        raise RuntimeError("REAL TRAINING DATASET GUARD FAILED: Missing source provenance metadata (source_url/license).")

    id_count = manifest.get("local_identity_count", 0)
    img_count = manifest.get("local_image_count", 0)

    if id_count < min_identities or img_count < min_images:
        raise RuntimeError(
            f"REAL TRAINING DATASET GUARD FAILED: Insufficient real dataset scale. "
            f"Found {id_count} IDs and {img_count} images (Minimum required: {min_identities} IDs, {min_images} images)."
        )

    logger.info(f"REAL TRAINING DATASET GUARD PASSED: Real dataset '{manifest.get('dataset_name')}' verified ({id_count} IDs, {img_count} imgs).")


class ArcFacePilotTrainer:
    """
    GPU-Accelerated ArcFace Pilot Trainer supporting PyTorch AMP and Staged Fine-Tuning.
    """

    def __init__(
        self,
        num_classes: int,
        device_type: str = "auto",
        batch_size: int = 32,
        lr: float = 1e-03,
        weight_decay: float = 5e-04,
        margin: float = 0.50,
        scale: float = 64.0,
        output_dir: str = "models/trained/autoroll_arcface_pilot_v1",
        use_amp: bool = True,
        skip_dataset_guard: bool = False,
    ):
        if not skip_dataset_guard:
            verify_real_dataset_guard()

        self.settings = get_settings()
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = max(num_classes, 2)

        os.makedirs(self.output_dir, exist_ok=True)

        # Device Resolution
        self.device = torch.device(self.settings.resolve_device())
        logger.info(f"ArcFacePilotTrainer initialized on device: '{self.device}'")

        if self.device.type != "cuda":
            logger.warning("CUDA is not active. Training will run in CPU mode.")

        # Initialize Backbone & Loss Head
        self.backbone = iresnet50(embedding_size=512)
        self.backbone.set_staged_freeze(stage=1)
        self.backbone.to(self.device)

        self.loss_head = ArcFaceLoss(
            in_features=512,
            out_features=self.num_classes,
            scale=scale,
            margin=margin,
        ).to(self.device)

        # Optimizer & Scheduler
        params = [
            {"params": filter(lambda p: p.requires_grad, self.backbone.parameters()), "lr": lr},
            {"params": self.loss_head.parameters(), "lr": lr},
        ]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-05)
        self.scaler = GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))

        self.start_epoch = 1
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader, epoch: int):
        self.backbone.train()
        self.loss_head.train()

        total_loss = 0.0
        total_samples = 0
        t0 = time.time()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                embeddings = self.backbone(images)
                loss = self.loss_head(embeddings, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        elapsed = time.time() - t0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        fps = total_samples / elapsed if elapsed > 0 else 0.0
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if self.device.type == "cuda" else 0.0

        logger.info(
            f"Epoch {epoch:02d} [TRAIN] | Loss: {avg_loss:.4f} | Throughput: {fps:.2f} img/s | "
            f"Peak VRAM: {peak_vram_mb:.2f} MB | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        return {
            "loss": avg_loss,
            "fps": fps,
            "peak_vram_mb": peak_vram_mb,
            "elapsed_sec": elapsed,
        }

    def evaluate(self, val_loader):
        self.backbone.eval()
        self.loss_head.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                    embeddings = self.backbone(images)
                    loss = self.loss_head(embeddings, labels)

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info(f"        [VAL]   | Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, filename: str = "latest.pt"):
        ckpt_path = os.path.join(self.output_dir, filename)
        state = {
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "loss_head_state": self.loss_head.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "val_loss": val_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "batch_size": self.batch_size,
        }
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint to '{ckpt_path}'.")

    def load_checkpoint(self, filename: str = "latest.pt") -> int:
        ckpt_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint file '{ckpt_path}' not found.")
            return 0

        state = torch.load(ckpt_path, map_location=self.device)
        self.backbone.load_state_dict(state["backbone_state"])
        self.loss_head.load_state_dict(state["loss_head_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])

        resumed_epoch = state.get("epoch", 0)
        self.start_epoch = resumed_epoch + 1
        logger.info(f"Restored checkpoint from '{ckpt_path}' (Resuming from epoch {self.start_epoch}).")
        return self.start_epoch


# Alias for backward compatibility
ArcFaceTrainer = ArcFacePilotTrainer
