"""
AutoRoll Phase 5.4 — Post-download launcher.

Waits for train.rec to be fully downloaded, then:
1. Verifies file integrity (size check)
2. Runs ingest_casia_rec.py on the full dataset
3. Runs validate_training_dataset.py

Run this after confirming train.rec download is complete.
"""
import os
import sys
import subprocess
import time

EXPECTED_REC_SIZE_MB = 2599  # ~2600 MB
REC_PATH = "data/tmp/casia_webface/train.rec"
IDX_PATH = "data/tmp/casia_webface/train.idx"
LST_PATH = "data/tmp/casia_webface/train.lst"
DEST_DIR = "data/face_recognition"


def check_rec_complete():
    if not os.path.exists(REC_PATH):
        return False, "train.rec not found"
    size_mb = os.path.getsize(REC_PATH) / (1024 * 1024)
    if size_mb < EXPECTED_REC_SIZE_MB:
        return False, f"train.rec only {size_mb:.1f} MB (expected ~{EXPECTED_REC_SIZE_MB} MB)"
    return True, f"train.rec: {size_mb:.1f} MB OK"


def main():
    print("=" * 70)
    print("AutoRoll Phase 5.4 — Post-Download Ingestion Launcher")
    print("=" * 70)

    # Check .rec is complete
    complete, msg = check_rec_complete()
    if not complete:
        print(f"[ERROR] {msg}")
        print("Download not yet complete. Run this script after download finishes.")
        sys.exit(1)

    print(f"[OK] {msg}")
    print(f"[OK] train.idx: {os.path.getsize(IDX_PATH) / (1024*1024):.1f} MB")
    print(f"[OK] train.lst: {os.path.getsize(LST_PATH) / (1024*1024):.1f} MB")

    # Step 1: Run ingestion pipeline
    print("\n" + "=" * 70)
    print("Step 1: Running CASIA .rec ingestion pipeline...")
    print("=" * 70)
    ingest_cmd = [
        sys.executable, "scripts/ingest_casia_rec.py",
        "--rec", REC_PATH,
        "--idx", IDX_PATH,
        "--lst", LST_PATH,
        "--dest", DEST_DIR,
        "--dataset-name", "CASIA-WebFace",
    ]
    print(f"Command: {' '.join(ingest_cmd)}\n")
    result = subprocess.run(ingest_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Ingestion failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print("\n[OK] Ingestion complete.")

    # Step 2: Validate training dataset
    print("\n" + "=" * 70)
    print("Step 2: Running dataset validation...")
    print("=" * 70)
    validate_cmd = [sys.executable, "scripts/validate_training_dataset.py"]
    result = subprocess.run(validate_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Validation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print("\n[OK] Validation complete.")

    print("\n" + "=" * 70)
    print("Phase 5.4 ingestion pipeline COMPLETE.")
    print(f"Dataset available at: {DEST_DIR}/")
    print(f"Manifest: {DEST_DIR}/metadata/source_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
