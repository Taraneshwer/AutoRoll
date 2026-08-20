"""
AutoRoll Real-World Evaluation Data Collection & Provenance Engine — Phase 17.1
Captures and ingests real-world participant enrollment samples, independent genuine probes,
condition metadata (15 categories), and physical liveness presentation attack recordings.
Enforces SHA-256 hash deduplication, anonymous IDs (P001–P030), and production pipeline quality checks.
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
if str(backend_dir / "backend") not in sys.path:
    sys.path.insert(0, str(backend_dir / "backend"))

try:
    from app.core.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger("collect_real_world_evaluation")



DATASET_ROOT = backend_dir.parent / "data" / "real_world_evaluation"

CONDITIONS = [
    "Normal Lighting",
    "Low Lighting",
    "Bright Lighting",
    "Indoor Artificial Lighting",
    "Backlighting",
    "Mild Head Yaw",
    "Moderate Head Yaw",
    "High Head Yaw",
    "Mild Pitch",
    "Moderate Pitch",
    "Glasses",
    "Mask",
    "Partial Occlusion",
    "Different Camera Distance",
    "Different Camera Height",
]

LIVENESS_ATTACK_TYPES = [
    "Printed Photograph",
    "Phone Replay Attack",
    "Tablet Replay Attack",
    "Video Replay Attack",
    "Bona Fide Live Face",
]


class EvaluationDataCollector:

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or DATASET_ROOT
        self.enrollment_dir = self.root_dir / "enrollment"
        self.probes_dir = self.root_dir / "probes"
        self.impostor_dir = self.root_dir / "impostor"
        self.liveness_dir = self.root_dir / "liveness"
        self.metadata_dir = self.root_dir / "metadata"
        self.manifests_dir = self.root_dir / "manifests"

        self._ensure_directories()
        self.seen_hashes: set[str] = self._load_existing_hashes()

    def _ensure_directories(self):
        for d in [
            self.enrollment_dir,
            self.probes_dir,
            self.impostor_dir,
            self.liveness_dir,
            self.metadata_dir,
            self.manifests_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_existing_hashes(self) -> set[str]:
        hashes = set()
        for img_path in self.root_dir.glob("**/*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    h = self.compute_sha256(img_path)
                    hashes.add(h)
                except Exception:
                    pass
        return hashes

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_bytes_sha256(img_bytes: bytes) -> str:
        return hashlib.sha256(img_bytes).hexdigest()

    def ingest_sample(
        self,
        image_bytes: bytes,
        participant_id: str,
        sample_type: str,
        condition: Optional[str] = None,
        liveness_attack: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a real face image sample into the evaluation dataset.
        sample_type: 'enrollment', 'probe', 'liveness'
        """
        img_hash = self.compute_bytes_sha256(image_bytes)
        if img_hash in self.seen_hashes:
            raise ValueError(f"DUPLICATE IMAGE DETECTED (SHA256: {img_hash[:12]}...). Image rejected.")

        # Decode image to verify validity and color variance
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Corrupt image bytes received.")

        var = float(np.var(img))
        if var < 5.0:
            raise ValueError("Image color variance too low (blank/solid image rejected).")

        h, w, c = img.shape
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if sample_type == "enrollment":
            target_folder = self.enrollment_dir / participant_id
            target_folder.mkdir(parents=True, exist_ok=True)
            filename = f"{participant_id}_enroll_{timestamp}_{img_hash[:8]}.jpg"
        elif sample_type == "probe":
            cond_clean = (condition or "normal").lower().replace(" ", "_")
            target_folder = self.probes_dir / participant_id / cond_clean
            target_folder.mkdir(parents=True, exist_ok=True)
            filename = f"{participant_id}_probe_{cond_clean}_{timestamp}_{img_hash[:8]}.jpg"
        elif sample_type == "liveness":
            attack_clean = (liveness_attack or "bona_fide").lower().replace(" ", "_")
            target_folder = self.liveness_dir / participant_id / attack_clean
            target_folder.mkdir(parents=True, exist_ok=True)
            filename = f"{participant_id}_liveness_{attack_clean}_{timestamp}_{img_hash[:8]}.jpg"

        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        dest_file = target_folder / filename
        with open(dest_file, "wb") as f:
            f.write(image_bytes)

        self.seen_hashes.add(img_hash)

        record = {
            "participant_id": participant_id,
            "sample_type": sample_type,
            "condition": condition or "Normal Lighting",
            "liveness_attack": liveness_attack or "Bona Fide Live Face",
            "sha256": img_hash,
            "filename": filename,
            "filepath": str(dest_file.relative_to(self.root_dir)),
            "width": w,
            "height": h,
            "color_variance": round(var, 2),
            "ingest_timestamp": timestamp,
        }

        self._update_consent_manifest(participant_id, record)
        return record

    def _update_consent_manifest(self, participant_id: str, record: Dict[str, Any]):
        manifest_file = self.manifests_dir / "consent_manifest.json"
        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        if participant_id not in manifest:
            # 50:50 Calibration (P001-P015) vs Held-Out Test (P016-P030)
            pid_num = int(participant_id.replace("P", "")) if participant_id.startswith("P") else 1
            split = "CALIBRATION" if pid_num <= 15 else "TEST"

            manifest[participant_id] = {
                "participant_id": participant_id,
                "split": split,
                "created_timestamp": record["ingest_timestamp"],
                "samples": [],
            }

        manifest[participant_id]["samples"].append(record)

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    collector = EvaluationDataCollector()
    print("=" * 80)
    print("AUTOROLL REAL-WORLD EVALUATION DATA COLLECTION SYSTEM (PHASE 17.1)")
    print("=" * 80)
    print(f"Dataset Root: {collector.root_dir}")
    print(f"Existing Verified Image Hashes on Disk: {len(collector.seen_hashes)}")
    print(f"Supported Conditions: {len(CONDITIONS)}")
    print(f"Supported Anti-Spoofing Attacks: {len(LIVENESS_ATTACK_TYPES)}")
    print("Ready for real human participant image collection (P001–P030).")
    print("=" * 80)
