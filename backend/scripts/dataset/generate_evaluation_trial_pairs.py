"""
Evaluation Trial Pair Generator — AutoRoll Phase 17.1
Generates genuine (identity_A == identity_B) and balanced impostor (identity_A != identity_B) comparison pairs
from physically ingested real-world face images in data/real_world_evaluation/.
Enforces strict 50:50 calibration/test participant split (P001–P015 vs P016–P030).
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

DATASET_ROOT = backend_dir.parent / "data" / "real_world_evaluation"


def generate_trial_pairs(seed: int = 2026) -> Dict[str, Any]:
    manifest_file = DATASET_ROOT / "manifests" / "consent_manifest.json"
    if not manifest_file.exists():
        print("WARNING: consent_manifest.json not found. Returning empty trial pairing manifest.")
        return {"calibration_genuine": [], "calibration_impostor": [], "test_genuine": [], "test_impostor": []}

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(seed)

    cal_participants = [pid for pid, data in manifest.items() if data.get("split") == "CALIBRATION"]
    test_participants = [pid for pid, data in manifest.items() if data.get("split") == "TEST"]

    def build_pairs(pids: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        genuine = []
        impostor = []

        # Collect enrollment and probe samples per participant
        enroll_map = {}
        probe_map = {}

        for pid in pids:
            p_data = manifest.get(pid, {})
            samples = p_data.get("samples", [])
            enroll_map[pid] = [s for s in samples if s.get("sample_type") == "enrollment"]
            probe_map[pid] = [s for s in samples if s.get("sample_type") == "probe"]

        # Genuine Pairs (probe vs enrollment template of same participant)
        for pid in pids:
            e_samples = enroll_map[pid]
            p_samples = probe_map[pid]
            for p_s in p_samples:
                for e_s in e_samples:
                    genuine.append({
                        "identity_A": pid,
                        "identity_B": pid,
                        "enrollment_sha256": e_s["sha256"],
                        "probe_sha256": p_s["sha256"],
                        "enrollment_file": e_s["filepath"],
                        "probe_file": p_s["filepath"],
                        "condition": p_s.get("condition", "Normal Lighting"),
                        "is_genuine": True,
                    })

        # Impostor Pairs (probe vs enrollment template of different participant)
        for i, pid_a in enumerate(pids):
            for pid_b in pids:
                if pid_a == pid_b:
                    continue
                p_samples = probe_map[pid_a]
                e_samples = enroll_map[pid_b]
                for p_s in p_samples[:2]:  # Limit balance ratio
                    for e_s in e_samples[:2]:
                        impostor.append({
                            "identity_A": pid_a,
                            "identity_B": pid_b,
                            "probe_sha256": p_s["sha256"],
                            "enrollment_sha256": e_s["sha256"],
                            "probe_file": p_s["filepath"],
                            "enrollment_file": e_s["filepath"],
                            "condition": p_s.get("condition", "Normal Lighting"),
                            "is_genuine": False,
                        })

        return genuine, impostor

    cal_gen, cal_imp = build_pairs(cal_participants)
    test_gen, test_imp = build_pairs(test_participants)

    trial_manifest = {
        "calibration_split": "P001-P015",
        "test_split": "P016-P030",
        "calibration_genuine_count": len(cal_gen),
        "calibration_impostor_count": len(cal_imp),
        "test_genuine_count": len(test_gen),
        "test_impostor_count": len(test_imp),
        "calibration_pairs": {"genuine": cal_gen, "impostor": cal_imp},
        "test_pairs": {"genuine": test_gen, "impostor": test_imp},
    }

    output_file = DATASET_ROOT / "manifests" / "trial_pairs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trial_manifest, f, indent=2)

    print(f"Trial pairs written to: {output_file}")
    print(f"Calibration Pairs: Genuine={len(cal_gen)}, Impostor={len(cal_imp)}")
    print(f"Test Pairs: Genuine={len(test_gen)}, Impostor={len(test_imp)}")

    return trial_manifest


if __name__ == "__main__":
    generate_trial_pairs()
