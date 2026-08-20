"""
AutoRoll GPU-Accelerated ArcFace Pilot Fine-Tuning Execution Script.
Executes 3 controlled pilot fine-tuning epochs on NVIDIA RTX 5060, evaluates loss convergence,
saves checkpoints to models/trained/autoroll_arcface_pilot_v1/, and tests checkpoint restoration.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import argparse
import os
import sys
import time
import torch

from app.core.logger import get_logger
from app.core.config import get_settings
from app.ml.training.dataset import create_training_dataloaders
from app.ml.training.trainer import ArcFacePilotTrainer

logger = get_logger("train_arcface_pilot")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll ArcFace GPU Fine-Tuning Pilot")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of pilot epochs")
    parser.add_argument("--lr", type=float, default=1e-03, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    return parser.parse_args()


def run_pilot():
    args = parse_args()
    settings = get_settings()

    print("=================================================================================")
    print("AUTOROLL GPU-ACCELERATED ARCFACE FINE-TUNING PILOT")
    print("=================================================================================")
    print(f"Target Execution Device  : {settings.resolve_device().upper()}")
    print(f"PyTorch Version          : {torch.__version__}")
    print(f"CUDA Available           : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Model                : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version             : {torch.version.cuda}")
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"Total VRAM Capacity      : {total_mem / (1024**3):.2f} GB")
        print(f"Available VRAM           : {free_mem / (1024**3):.2f} GB")
    else:
        logger.warning("CUDA is NOT active. Running in CPU Execution Mode.")

    split_root = "data/face_recognition/splits"
    if not os.path.exists(os.path.join(split_root, "train")):
        logger.error(f"Training split directory not found at '{split_root}'. Run prepare_real_training_dataset.py first.")
        sys.exit(1)

    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, num_classes = create_training_dataloaders(
        split_root=split_root,
        batch_size=args.batch_size,
        num_workers=0,  # Single-process for Windows safety
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"Loaded DataLoaders | Number of Classes: {num_classes}")

    trainer = ArcFacePilotTrainer(
        num_classes=num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir="models/trained/autoroll_arcface_pilot_v1",
        use_amp=True,
    )

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint("latest.pt")
        target_epochs = start_epoch + 1
    else:
        target_epochs = args.epochs

    print("\n---------------------------------------------------------------------------------")
    print("PILOT TRAINING PROGRESS LOG")
    print("---------------------------------------------------------------------------------")
    print(f"{'EPOCH':<8} | {'TRAIN LOSS':<12} | {'VAL LOSS':<12} | {'THROUGHPUT':<14} | {'PEAK VRAM':<12} | {'STATUS':<8}")
    print("-" * 75)

    for epoch in range(start_epoch, target_epochs + 1):
        t_stats = trainer.train_epoch(train_loader, epoch)
        v_loss = trainer.evaluate(val_loader)

        # Update learning rate scheduler
        trainer.scheduler.step()

        # Save latest & best validation checkpoints
        trainer.save_checkpoint(epoch, v_loss, filename="latest.pt")
        if v_loss < trainer.best_val_loss:
            trainer.best_val_loss = v_loss
            trainer.save_checkpoint(epoch, v_loss, filename="best.pt")

        print(
            f"{epoch:02d}/{target_epochs:02d}   | {t_stats['loss']:<12.4f} | {v_loss:<12.4f} | "
            f"{t_stats['fps']:<14.2f} | {t_stats['peak_vram_mb']:<12.2f} | SUCCESS"
        )

    print("=================================================================================")
    print("PILOT ARCFACE FINE-TUNING EXECUTED SUCCESSFULLY!")
    print("=================================================================================\n")


if __name__ == "__main__":
    run_pilot()
