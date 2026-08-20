"""
AutoRoll Unified Inference Pipeline Command Test Utility.
Usage: python scripts/test_pipeline.py <video_path> [--device auto|cpu|cuda] [--rec-interval 10]
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
import time

import cv2
import numpy as np

from app.core.logger import get_logger
from app.ml.inference.pipeline import UnifiedInferencePipeline

logger = get_logger("test_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoRoll Real-Time Unified Inference Pipeline Test Utility"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path to video file or camera index (default: synthetic frames)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device selection (default: auto)",
    )
    parser.add_argument(
        "--rec-interval",
        type=int,
        default=10,
        help="Recognition re-trigger interval in frames (default: 10)",
    )
    return parser.parse_args()


def create_synthetic_multi_face_frame(frame_idx: int) -> np.ndarray:
    """
    Generates a synthetic frame containing 2 faces moving horizontally.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 200

    # Face 1 position (left side moving slightly)
    x1 = 120 + int(np.sin(frame_idx * 0.1) * 10)
    cv2.ellipse(img, (x1 + 56, 120 + 56), (40, 50), 0, 0, 360, (180, 160, 140), -1)
    cv2.circle(img, (x1 + 40, 150), 5, (30, 30, 30), -1)
    cv2.circle(img, (x1 + 72, 150), 5, (30, 30, 30), -1)

    # Face 2 position (right side)
    x2 = 380 + int(np.cos(frame_idx * 0.1) * 10)
    cv2.ellipse(img, (x2 + 56, 200 + 56), (40, 50), 0, 0, 360, (170, 150, 130), -1)
    cv2.circle(img, (x2 + 40, 230), 5, (30, 30, 30), -1)
    cv2.circle(img, (x2 + 72, 230), 5, (30, 30, 30), -1)

    return img


def main():
    args = parse_args()

    logger.info("Initializing Unified Inference Pipeline...")
    pipeline = UnifiedInferencePipeline(
        device=args.device, recognition_interval=args.rec_interval
    )

    if args.source and os.path.exists(args.source):
        logger.info(f"Opening video file '{args.source}'...")
        cap = cv2.VideoCapture(args.source)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            res = pipeline.process_frame(frame, frame_index=frame_idx)

            print(
                f"Frame {res.frame_index:4d} | Faces: {res.num_faces_detected:2d} | "
                f"Live: {res.num_faces_live:2d} | Total Latency: {res.total_latency_ms:6.2f} ms | "
                f"FPS: {res.fps:5.1f}"
            )
            time.sleep(0.01)

        cap.release()
    else:
        if args.source:
            logger.warning(f"Video file '{args.source}' not found. Using synthetic stream...")
        else:
            logger.info("No video path provided. Testing with 15 synthetic frames...")

        for f in range(1, 16):
            synth_frame = create_synthetic_multi_face_frame(f)
            res = pipeline.process_frame(synth_frame, frame_index=f)

            print(
                f"Frame {res.frame_index:2d} | Faces: {res.num_faces_detected:2d} | "
                f"Live: {res.num_faces_live:2d} | Det: {res.detection_latency_ms:5.2f} ms | "
                f"Rec: {res.recognition_latency_ms:5.2f} ms | "
                f"Total: {res.total_latency_ms:6.2f} ms | FPS: {res.fps:5.1f}"
            )
            for face in res.faces:
                print(
                    f"   |- Track #{face.track_id:02d} | Conf: {face.detection_confidence:.2f} | "
                    f"Live: {face.is_live} | Status: {face.recognition_status}"
                )

    print("\n" + "=" * 65)
    print("         AUTOROLL UNIFIED INFERENCE PIPELINE TEST COMPLETE        ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
