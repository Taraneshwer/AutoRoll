"""
[DEPRECATED — NOT FOR TRAINING]
AutoRoll ML Phase 5 — Legacy Synthetic Face Dataset Generator.

CRITICAL NOTICE:
THIS SCRIPT IS DEPRECATED AND QUARANTINED. IT GENERATES SYNTHETIC GEOMETRIC DRAWINGS
(CV2 ELLIPSES & CIRCLES) AND MUST NEVER BE USED FOR AUTOROLL ARCFACE PRODUCTION TRAINING.
EXPLICIT RUNTIME PROTECTION HAS BEEN ENGAGED TO PREVENT PRODUCTION INVOCATION.
"""

import os
import sys

def build_large_dataset(*args, **kwargs):
    raise RuntimeError(
        "CRITICAL ERROR: 'build_large_face_dataset.py' is DEPRECATED and QUARANTINED. "
        "Synthetic dataset generation is prohibited for AutoRoll ArcFace training. "
        "Production training requires a genuine real public face dataset ingested via "
        "'scripts/ingest_real_dataset.py'."
    )

if __name__ == "__main__":
    print("[DEPRECATED — NOT FOR TRAINING] Synthetic dataset generator is disabled.")
    raise RuntimeError(
        "CRITICAL ERROR: 'build_large_face_dataset.py' is DEPRECATED and QUARANTINED. "
        "Synthetic dataset generation is prohibited for AutoRoll ArcFace training."
    )
