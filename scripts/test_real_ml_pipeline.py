"""
AutoRoll Real Pretrained ML Pipeline Test Tool.
Runs SCRFD -> Alignment -> ArcFace -> MiniFASNet -> Decision Engine on real test images.
Reports per-stage latencies, embedding dimensions, device, and execution providers.
"""

import argparse
import os
import time

import cv2
import numpy as np

from autoroll.common.config import get_settings
from autoroll.common.logger import get_logger
from autoroll.ml.inference.pipeline import UnifiedInferencePipeline

logger = get_logger("test_real_ml_pipeline")
settings = get_settings()


def run_pipeline_test(image_path: str | None = None):
    logger.info("Initializing AutoRoll Real Pretrained ML Pipeline Test...")

    # Load input image or create test sample face image
    os.makedirs("data/test_samples", exist_ok=True)
    sample_path = "data/test_samples/sample_face.jpg"

    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        source_info = f"File: {image_path}"
    else:
        # Create a sample face image if not exists
        if not os.path.exists(sample_path):
            img = np.full((480, 640, 3), 200, dtype=np.uint8)
            cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (180, 150, 130), -1)  # Face
            cv2.circle(img, (280, 200), 14, (60, 40, 30), -1)  # Left eye
            cv2.circle(img, (360, 200), 14, (60, 40, 30), -1)  # Right eye
            cv2.line(img, (320, 230), (320, 260), (120, 90, 80), 4)  # Nose
            cv2.ellipse(img, (320, 290), (40, 15), 0, 0, 180, (100, 50, 50), 6)  # Mouth
            cv2.imwrite(sample_path, img)

        frame = cv2.imread(sample_path)
        source_info = f"Sample Face Image ({sample_path})"

    # Initialize Real Inference Pipeline
    pipeline = UnifiedInferencePipeline(device=settings.DEVICE_TYPE, recognition_interval=1)
    pipeline.recognizer.warmup()

    start_t = time.perf_counter()
    result = pipeline.process_frame(frame, frame_index=1)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print("\n" + "=" * 90)
    print("AUTOROLL REAL PRETRAINED ML PIPELINE TEST REPORT")
    print("=" * 90)
    print(f"Input Source           : {source_info}")
    print(f"ML Operating Mode      : {settings.AUTOROLL_ML_MODE.upper()}")
    print(f"Device / Provider      : {pipeline.device} ({pipeline.recognizer.providers[0]})")
    print(f"Faces Detected         : {result.num_faces_detected}")
    print(f"Faces Verified Live    : {result.num_faces_live}")
    print("-" * 90)
    print(f"SCRFD Detection Latency: {result.detection_latency_ms:.2f} ms")
    print(f"ArcFace Recog Latency  : {result.recognition_latency_ms:.2f} ms")
    print(f"MiniFASNet Live Latency: {result.liveness_latency_ms:.2f} ms")
    tot_ms = result.total_latency_ms
    print(f"Total Pipeline Latency : {tot_ms:.2f} ms (Wall: {elapsed_ms:.2f} ms)")
    print("-" * 90)

    for idx, face in enumerate(result.faces):
        emb_dim = len(face.embedding) if face.embedding else 0
        sample_vec = [round(v, 4) for v in face.embedding[:4]] if face.embedding else []
        print(f"Face #{idx+1} [Track ID {face.track_id}]:")
        print(f"  |- Bounding Box      : {face.bbox.to_list()}")
        print(f"  |- Confidence        : {face.detection_confidence:.2f}")
        print(f"  |- Embedding Dim     : {emb_dim}-dimensional vector")
        print(f"  |- Embedding Sample  : {sample_vec}")
        print(
            f"  |- Liveness Score    : {face.liveness_score:.4f} "
            f"(Decision: {face.liveness_decision})"
        )
        print(f"  |- Status            : {face.recognition_status}")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AutoRoll Real Pretrained ML Pipeline Test Tool")
    parser.add_argument("--image", type=str, default=None, help="Path to input BGR test image")
    args = parser.parse_args()

    run_pipeline_test(args.image)


if __name__ == "__main__":
    main()
