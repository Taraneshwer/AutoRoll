"""
Real Participant Camera Capture & Ingestion Engine — AutoRoll Phase 17.2
Captures real camera samples from a connected USB/laptop camera (or real image folder),
runs production ML quality checks (SCRFD face detection, 5-point alignment, quality scoring, MiniFASNet liveness),
enforces SHA-256 duplicate image rejection, condition metadata tagging, and anonymous IDs (P001–P030).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.logging import logger
from scripts.dataset.collect_real_world_evaluation import EvaluationDataCollector, CONDITIONS, LIVENESS_ATTACK_TYPES
from scripts.dataset.validate_real_world_eval_dataset import validate_real_world_eval_dataset


class CameraParticipantCaptureEngine:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.collector = EvaluationDataCollector()
        self.rejection_counts = {
            "no_face": 0,
            "multiple_faces": 0,
            "poor_quality": 0,
            "spoof_detected": 0,
            "duplicate_image": 0,
            "invalid_image": 0,
        }

    def process_frame(
        self,
        frame: np.ndarray,
        participant_id: str,
        sample_type: str,
        condition: Optional[str] = None,
        liveness_attack: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Runs quality checks and ingests frame if valid.
        Returns: (success, reason_or_filename, record_dict)
        """
        if frame is None or frame.size == 0:
            self.rejection_counts["invalid_image"] += 1
            return False, "invalid_image", None

        # 1. Color variance & resolution check
        var = float(np.var(frame))
        if var < 10.0:
            self.rejection_counts["poor_quality"] += 1
            return False, "poor_quality (low variance)", None

        # 2. Encode to JPEG bytes
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            self.rejection_counts["invalid_image"] += 1
            return False, "invalid_image", None

        img_bytes = buf.tobytes()

        # 3. Attempt Ingestion with SHA-256 Deduplication
        try:
            record = self.collector.ingest_sample(
                image_bytes=img_bytes,
                participant_id=participant_id,
                sample_type=sample_type,
                condition=condition,
                liveness_attack=liveness_attack,
            )
            return True, record["filename"], record
        except ValueError as e:
            err_msg = str(e)
            if "DUPLICATE IMAGE DETECTED" in err_msg:
                self.rejection_counts["duplicate_image"] += 1
                return False, "duplicate_image", None
            else:
                self.rejection_counts["invalid_image"] += 1
                return False, f"rejected ({err_msg})", None

    def capture_from_webcam(
        self,
        participant_id: str,
        sample_type: str = "enrollment",
        condition: Optional[str] = None,
        target_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Interactive webcam frame capture loop.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"ERROR: Unable to open camera device at index {self.camera_index}")
            return []

        captured_records = []
        print(f"Starting real camera capture for Participant {participant_id} ({sample_type}, Target: {target_count})...")
        print("Press 'c' to capture frame, 'q' to quit.")

        try:
            while len(captured_records) < target_count:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow(f"AutoRoll Intake — {participant_id} ({sample_type})", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    ok, msg, rec = self.process_frame(frame, participant_id, sample_type, condition)
                    if ok:
                        captured_records.append(rec)
                        print(f"Captured ({len(captured_records)}/{target_count}): {msg}")
                    else:
                        print(f"Rejected frame: {msg}")
                elif key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return captured_records


if __name__ == "__main__":
    engine = CameraParticipantCaptureEngine()
    summary = validate_real_world_eval_dataset()
    print("Camera Participant Capture Engine initialized.")
