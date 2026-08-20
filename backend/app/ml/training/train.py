"""
ArcFace Fine-Tuning Training Command CLI.
Usage: python -m app.ml.training.train --config configs/training.yaml
"""

import argparse
import os
import sys

import yaml

from app.core.logger import get_logger
from app.ml.training.trainer import ArcFaceTrainer

logger = get_logger("train_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll ArcFace Fine-Tuning CLI")
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to YAML training configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = ArcFaceTrainer(config)
    exp_dir = trainer.train()

    print("\n" + "=" * 60)
    print("        AUTOROLL ARCFACE FINE-TUNING COMPLETE        ")
    print("=" * 60)
    print(f"Experiment ID    : {config.get('experiment_id')}")
    print(f"Experiment Dir   : {exp_dir}")
    print("Artifacts Saved  : best_checkpoint.pt, config.yaml, training_history.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
