"""
AutoRoll Anti-Spoofing & Liveness Pipeline Test Command.
Usage: python scripts/test_liveness.py <path_to_media> [--window 10] [--device auto|cpu|cuda]
"""

import argparse
import os
import sys

import cv2
import numpy as np

from autoroll.common.logger import get_logger
from autoroll.ml.liveness.pipeline import LivenessPipeline

logger = get_logger("test_liveness")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Liveness & Anti-Spoofing Test Utility")
    parser.add_argument(
        "media_path",
        nargs="?",
        default=None,
        help="Path to input image or video file (optional: generates sample if omitted)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Sliding temporal window size (default: 10)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device selection (default: auto)",
    )
    return parser.parse_args()


def create_sample_face(is_spoof: bool = False) -> np.ndarray:
    img = np.zeros((112, 112, 3), dtype=np.uint8) + (100 if is_spoof else 180)
    cv2.ellipse(img, (56, 56), (35, 45), 0, 0, 360, (160, 140, 120), -1)
    cv2.circle(img, (42, 45), 5, (40, 40, 40), -1)
    cv2.circle(img, (70, 45), 5, (40, 40, 40), -1)
    if is_spoof:
        # Add artificial Moire lines / screen noise for spoof sample
        for y in range(0, 112, 4):
            cv2.line(img, (0, y), (112, y), (50, 50, 50), 1)
    return img


def main():
    args = parse_args()

    pipeline = LivenessPipeline(device=args.device, temporal_window=args.window)

    if args.media_path and os.path.exists(args.media_path):
        ext = os.path.splitext(args.media_path)[1].lower()
        if ext in {".mp4", ".avi", ".mov", ".mkv"}:
            logger.info(f"Processing video stream file '{args.media_path}'...")
            cap = cv2.VideoCapture(args.media_path)
            frame_count = 0
            latest_result = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                crop = cv2.resize(frame, (112, 112))
                latest_result = pipeline.predict(crop)

            cap.release()
            result = latest_result
            if result is None:
                logger.error("No valid frames read from video.")
                sys.exit(1)
        else:
            logger.info(f"Processing image file '{args.media_path}'...")
            img = cv2.imread(args.media_path)
            result = pipeline.predict(img)
    else:
        if args.media_path:
            logger.warning(f"File '{args.media_path}' not found. Generating sample test chip...")
        else:
            logger.info("No input path provided. Testing with synthetic genuine face chip...")
        sample_img = create_sample_face(is_spoof=False)
        result = pipeline.predict(sample_img)

    print("\n" + "=" * 60)
    print("         AUTOROLL ANTI-SPOOFING LIVENESS TEST RESULT        ")
    print("=" * 60)
    print(f"Liveness Decision  : {result.details.get('decision', 'N/A')}")
    print(f"Liveness Score     : {result.liveness_score:.4f} (Threshold: {pipeline.threshold})")
    print(f"Spatial Score      : {result.details.get('spatial_score', 'N/A')}")
    print(f"Processing Latency : {result.details.get('latency_ms', 'N/A')} ms")
    print(f"Evaluation Method  : {result.method}")
    print(f"PAD Model Version  : {result.details.get('model_version', 'N/A')}")
    print(f"Device Used        : {pipeline.device}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
