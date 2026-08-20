"""
[DEPRECATED — NOT FOR TRAINING]
This script has been moved to 'scripts/legacy/synthetic/build_large_face_dataset.py'.
Synthetic dataset generation is strictly prohibited for AutoRoll production training.
"""
import sys

def main():
    raise RuntimeError(
        "CRITICAL ERROR: 'scripts/build_large_face_dataset.py' has been DEPRECATED and QUARANTINED "
        "to 'scripts/legacy/synthetic/build_large_face_dataset.py'. "
        "Synthetic data must NEVER be used for ArcFace model fine-tuning. "
        "Use 'scripts/ingest_real_dataset.py' with a legitimate real human face dataset."
    )

if __name__ == "__main__":
    main()
