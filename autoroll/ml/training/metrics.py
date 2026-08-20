"""
Training Metrics & History Tracking Module.
"""

import json
import os

from pydantic import BaseModel, Field

from autoroll.common.logger import get_logger

logger = get_logger("training_metrics")


class EpochMetrics(BaseModel):
    epoch: int
    stage: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    learning_rate: float


class TrainingHistory(BaseModel):
    experiment_id: str
    epochs: list[EpochMetrics] = Field(default_factory=list)


class TrainingMetricsTracker:
    """
    Logs epoch metrics and saves training_history.json to experiment directory.
    """

    def __init__(self, experiment_id: str):
        self.history = TrainingHistory(experiment_id=experiment_id)

    def record_epoch(
        self,
        epoch: int,
        stage: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        metrics = EpochMetrics(
            epoch=epoch,
            stage=stage,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            train_acc=round(train_acc, 4),
            val_acc=round(val_acc, 4),
            learning_rate=lr,
        )
        self.history.epochs.append(metrics)
        logger.info(
            f"Epoch {epoch:2d} [Stage {stage}] | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%) | LR: {lr:.6f}"
        )

    def save_history(self, output_dir: str) -> str:
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history.model_dump(), f, indent=2)
        return history_path
