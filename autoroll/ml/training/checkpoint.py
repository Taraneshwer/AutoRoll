"""
Checkpoint Management Module for ArcFace Training & Fine-Tuning.
Never overwrites original pretrained weights.
"""

import json
import os
from typing import Any

import torch
import yaml

from autoroll.common.logger import get_logger

logger = get_logger("checkpoint_manager")


class CheckpointManager:
    """
    Manages experiment directory creation, model saving, and checkpoint loading.
    Every experiment lives in its own dedicated directory: models/trained/<experiment_id>/
    """

    def __init__(self, save_dir: str, experiment_id: str):
        self.experiment_dir = os.path.join(save_dir, experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.best_loss = float("inf")

    def save_config(self, config_dict: dict[str, Any]) -> str:
        cfg_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return cfg_path

    def save_checkpoint(
        self,
        epoch: int,
        backbone_state: dict[str, Any],
        head_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        val_loss: float,
        is_best: bool = False,
    ) -> str:
        checkpoint_data = {
            "epoch": epoch,
            "backbone_state_dict": backbone_state,
            "head_state_dict": head_state,
            "optimizer_state_dict": optimizer_state,
            "val_loss": val_loss,
        }

        latest_path = os.path.join(self.experiment_dir, "latest_checkpoint.pt")
        torch.save(checkpoint_data, latest_path)

        if is_best:
            self.best_loss = val_loss
            best_path = os.path.join(self.experiment_dir, "best_checkpoint.pt")
            torch.save(checkpoint_data, best_path)
            logger.info(
                f"Saved new BEST checkpoint at epoch {epoch} "
                f"(Val Loss: {val_loss:.4f}) -> {best_path}"
            )

        return latest_path

    def load_checkpoint(
        self, checkpoint_path: str
    ) -> dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

        logger.info(f"Loading checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint

    def save_metadata(self, metadata: dict[str, Any]) -> str:
        meta_path = os.path.join(self.experiment_dir, "model_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return meta_path
