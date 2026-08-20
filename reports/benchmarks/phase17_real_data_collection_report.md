# AutoRoll Phase 17.2 — Real Participant Data Acquisition Report

## 1. Executive Summary & Acquisition Status

- **Status:** **`REAL DATA COLLECTION INCOMPLETE — 0/30 PARTICIPANTS`**
- **Actual Participants Captured:** **0 / 30 Participants** (Calibration: 0, Held-Out Test: 0)
- **Actual Physical Enrollment Images:** **0 Images**
- **Actual Physical Probe Images:** **0 Images**
- **Actual Physical Liveness Images:** **0 Images**
- **Total Physical Real Images on Disk:** **0 Images**
- **Duplicate Images Detected:** **0 Duplicates**
- **Manifest SHA-256 Checksum:** `NONE`

---

## 2. Strict Scientific Rules Enforced

- **Zero Synthetic Images:** Every image file has been inspected and verified against raw camera byte acquisition protocols.
- **Zero Synthetic Scores:** Recognition model benchmarking remains **strictly blocked** until physical image acquisition reaches target capacity (~900 real images).
- **Model Weight Immutability:** Pretrained ArcFace ONNX SHA-256 checksum verified (`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`).
- **Privacy Compliance:** Anonymous sequential IDs (`P001`–`P030`) only. No PII stored in workspace. Raw biometric files excluded from Git tracking via `.gitignore`.

---

## 3. Condition Taxonomy & Rejection Metrics

| Category | Count | Status |
| :--- | :---: | :--- |
| **Participants Intake (P001–P030)** | 0 / 30 | In Progress (Real Camera Intake Protocol) |
| **Enrollment Samples (Frontal/Pose)** | 0 | Physical Acquisition |
| **Probe Samples (15 Taxonomy Conditions)** | 0 | Physical Acquisition |
| **Liveness Presentation Attacks** | 0 | Physical Acquisition |
| **Duplicates Rejected (SHA-256)** | 0 | Verified (0 Cross-Session Overlap) |

---

## 4. Next Steps

Camera intake sessions via `capture_camera_participant.py` will continue for remaining human participants (`P001`–`P030`) before triggering Phase 17 final benchmark execution.
