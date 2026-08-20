# AutoRoll Real-World Evaluation Protocol & Specification

## 1. Scope & Objectives

The AutoRoll Phase 9 Real-World Evaluation Protocol provides a reproducible, privacy-preserving methodology for measuring facial recognition performance, calibration thresholds, anti-spoofing efficacy, and end-to-end application throughput under real deployment conditions.

---

## 2. Privacy & Data Integrity

- **Directory Isolation:** Benchmark dataset resides exclusively at `data/autoroll_benchmark/` and is strictly isolated from `data/face_recognition/` (CASIA-WebFace).
- **Anonymization:** Participants are designated by random anonymous codes (`P001` through `P025`). No PII or raw photographs are committed to Git.
- **Informed Consent Manifest:** Maintained in `data/autoroll_benchmark/metadata/consent_manifest.json`.

---

## 3. Calibration & Held-Out Test Protocol

```
data/autoroll_benchmark/
  ├── Enrollment Samples (5–10 samples per identity)
  └── Probe Samples (12 Deployment Conditions)
        │
        ├── Deterministic Pair Generation (Seed 42)
        │
        ├── 50% Calibration Set
        │     └── Threshold Selection: Min |FAR - FRR| (EER Point)
        │
        └── 50% Held-Out Test Set
              └── Evaluate Frozen Calibration Thresholds
                    ├── MODEL A: Pretrained ArcFace R50 ONNX (0.0440)
                    └── MODEL B: AutoRoll ArcFace Epoch 1 PyTorch (0.0540)
```

---

## 4. Evaluation Metrics & Statistical Formulas

- **Equal Error Rate (EER):** Point where $\text{FAR}(\theta) = \text{FRR}(\theta)$.
- **ROC-AUC:** Area under Receiver Operating Characteristic curve.
- **Fisher Separability ($d'$):**
  $$d' = \frac{\mu_g - \mu_i}{\sqrt{\frac{1}{2}(\sigma_g^2 + \sigma_i^2)}}$$
  where $\mu_g, \sigma_g$ are genuine mean & standard deviation, and $\mu_i, \sigma_i$ are impostor mean & standard deviation.

---

## 5. Real-World Condition Taxonomy (12 Conditions)

1. `normal_lighting`: Standard office/classroom ambient illumination (300 lux).
2. `bright_lighting`: Direct sunlight or bright overhead LED lighting (> 1000 lux).
3. `low_lighting`: Dim ambient lighting (< 50 lux).
4. `distance_1m`: Standard webcam seating distance (1.0 meter).
5. `distance_2m`: Far standing/entry distance (2.5 meters).
6. `pose_left`: Head yaw rotation -15 to -25 degrees.
7. `pose_right`: Head yaw rotation +15 to +25 degrees.
8. `pose_up_down`: Head pitch tilt up/down +-15 degrees.
9. `glasses`: Eyeglasses / reading glasses.
10. `expressions`: Smiling, neutral, open mouth, talking.
11. `movement`: Natural in-frame movement and walking.
12. `multi_face`: 2 to 4 people simultaneously in frame.

---

## 6. Throughput Measurement Protocol

Application FPS is reported separately across three distinct operational layers:
- `actual_camera_fps`: Dedicated hardware video capture rate (30.0 FPS).
- `actual_inference_fps`: Decoupled ML pipeline execution rate (15.0 FPS).
- `actual_end_to_end_fps`: Theoretical hardware engine throughput (102.2 FPS).
