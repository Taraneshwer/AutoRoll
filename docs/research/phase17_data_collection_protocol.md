# AutoRoll Phase 17.1 — Real-World Evaluation Data Collection Protocol

## 1. Objective
Establish a scientifically rigorous, privacy-compliant, zero-duplication data collection protocol for real human participants (`P001`–`P030`) under physical environment conditions.

---

## 2. Privacy & Anonymous ID Intake
- All participants are assigned anonymous sequential IDs: `P001`, `P002`, ..., `P030`.
- No raw participant names, email addresses, or personally identifiable strings are stored in git repositories or metadata manifests.
- Consent records link anonymous IDs (`PXXX`) to signed physical/digital consent forms outside the public codebase.

---

## 3. Directory Layout (`data/real_world_evaluation/`)

```
data/real_world_evaluation/
├── enrollment/
│   ├── P001/
│   └── P002/
├── probes/
│   ├── P001/
│   │   ├── normal_lighting/
│   │   ├── low_lighting/
│   │   ├── backlight/
│   │   └── mask/
├── liveness/
│   ├── printed_photograph/
│   ├── phone_replay_attack/
│   ├── tablet_replay_attack/
│   ├── video_replay_attack/
│   └── bona_fide_live_face/
├── metadata/
└── manifests/
    ├── consent_manifest.json
    ├── trial_pairs.json
    └── split_manifest.json
```

---

## 4. Session Separation & SHA-256 Hash Provenance
- **Enrollment Session (Session A):** 5–10 high-quality frontal face images per participant.
- **Evaluation Probes (Session B/C/D):** Collected in distinct sessions under various lighting, pose, and occlusion conditions.
- **Zero Image Reuse Constraint:** SHA-256 cryptographic hashes are computed for every captured image. Duplicate images across enrollment, probes, and liveness datasets are rejected automatically.

---

## 5. Condition Taxonomy (15 Categories)
1. Normal Lighting
2. Low Lighting
3. Bright Lighting
4. Indoor Artificial Lighting
5. Backlighting
6. Mild Head Yaw (15°)
7. Moderate Head Yaw (30°)
8. High Head Yaw (45°)
9. Mild Pitch (15°)
10. Moderate Pitch (30°)
11. Glasses
12. Mask
13. Partial Occlusion (Hand/Paper)
14. Different Camera Distance (1m vs 2.5m)
15. Different Camera Height (Desk vs Stand)

---

## 6. Calibration vs Test Partitioning (50:50 Split)
- **Calibration Split (`P001`–`P015`):** 50% of participants used exclusively to select the optimal decision threshold.
- **Held-Out Test Split (`P016`–`P030`):** 50% of participants evaluated using the frozen calibration threshold.
