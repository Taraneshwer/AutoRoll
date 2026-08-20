"""
AutoRoll Baseline Face Recognition Testing Script.
Usage: python scripts/test_recognition.py <path_to_image> [--device auto|cpu|cuda]
"""

import argparse
import os
import sys

import cv2
import numpy as np

from autoroll.common.logger import get_logger
from autoroll.ml.pipeline import AutoRollMLPipeline

logger = get_logger("test_recognition")


def create_sample_face_image() -> np.ndarray:
    """
    Creates a synthetic test image with a face shape for testing when no image file is provided.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 200
    # Draw face oval
    cv2.ellipse(img, (320, 240), (100, 140), 0, 0, 360, (180, 160, 140), -1)
    # Draw eyes
    cv2.circle(img, (280, 200), 12, (50, 50, 50), -1)
    cv2.circle(img, (360, 200), 12, (50, 50, 50), -1)
    # Draw nose
    cv2.line(img, (320, 210), (320, 250), (100, 80, 60), 3)
    # Draw mouth
    cv2.ellipse(img, (320, 280), (35, 15), 0, 0, 180, (80, 40, 40), 3)
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoRoll Face Recognition Test Script"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image file (optional: generates sample image if omitted)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device selection (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image_path and os.path.exists(args.image_path):
        logger.info(f"Loading input image from '{args.image_path}'...")
        img = cv2.imread(args.image_path)
        if img is None:
            logger.error(f"Failed to read image at '{args.image_path}'")
            sys.exit(1)
    else:
        if args.image_path:
            logger.warning(
                f"Image file '{args.image_path}' not found. Generating synthetic test image..."
            )
        else:
            logger.info("No image path provided. Generating synthetic test face image...")
        img = create_sample_face_image()

    logger.info(f"Initializing AutoRoll ML Pipeline (Device: {args.device})...")
    pipeline = AutoRollMLPipeline(device=args.device)

    logger.info("Executing Face Detection, Alignment, and ArcFace Recognition...")
    result = pipeline.process_frame(img, camera_id="test_cli")

    num_faces = len(result.faces)

    print("\n" + "=" * 60)
    print("           AUTOROLL RECOGNITION TEST RESULTS           ")
    print("=" * 60)
    print(f"Device Used        : {pipeline.device}")
    print(f"Inference Latency  : {result.processing_time_ms} ms")
    print(f"Number of Faces    : {num_faces}")

    for idx, face in enumerate(result.faces):
        bbox = face.bbox
        emb = face.recognition.embedding if face.recognition else []
        model_ver = face.recognition.model_version if face.recognition else "N/A"

        print(f"\n--- Face #{idx + 1} ---")
        print(
            "  Bounding Box     : ["
            f"x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, x2={bbox.x2:.1f}, y2={bbox.y2:.1f}]"
        )
        print(f"  Confidence       : {bbox.confidence:.2f}")
        print(f"  Embedding Dim    : {len(emb)}")
        print(f"  Model Version    : {model_ver}")
        print(f"  Embedding Sample : [{', '.join(f'{v:.4f}' for v in emb[:5])}...]")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
