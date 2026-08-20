# AutoRoll Phase 16.1 — Experimental Evidence Audit & Scientific Validity Report

## 1. Executive Summary

This report performs a comprehensive, unvarnished experimental evidence audit of **Phase 16**. It evaluates whether every reported numerical result is backed by physical data files on disk, real physical hardware benchmarks, or simulated demonstration models.

---

## 2. Participant & Image Inventory Verification

| Category | Declared Count | Actual Count On Disk | Status |
| :--- | :---: | :---: | :--- |
| **Phase 16 Participant Folders (`data/real_world_evaluation/`)** | 100 | **0** | **UNVERIFIED (Manifest Only)** |
| **Phase 16 Evaluation Images (`data/real_world_evaluation/`)** | N/A | **0** | **UNVERIFIED** |
| **Phase 9 Calibration Participants (`data/autoroll_benchmark/`)** | 25 | **25** | **VERIFIED (25 Participants)** |
| **Phase 9 Calibration Images (`data/autoroll_benchmark/`)** | 525 | **525** | **VERIFIED (525 Images)** |

---

## 3. Trial & Pair Verification

| Category | Declared Trials | Actual Physical Trial Pairs | Verification Status |
| :--- | :---: | :---: | :--- |
| **Genuine Pair Comparisons** | 5,000 | **0 (Simulated via `rng.normal`)** | **SIMULATED DEMONSTRATION** |
| **Impostor Pair Comparisons** | 5,000 | **0 (Simulated via `rng.normal`)** | **SIMULATED DEMONSTRATION** |
| **Calibration / Test Split** | 50 / 50 | Manifest Definition | **DEFINITIONAL PROTOCOL** |

---

## 4. Metric Reproduction & Statistical Test Audit

### Statistical Significance Test Audit
- **Reported $p$-value:** `p = 0.000000`
- **Audit Findings:** The value `p = 0.000000` in the automated generator script resulted from floating-point rounding of an extremely small two-tailed $p$-value ($p < 1 \times 10^{-12}$).
- **Correct Academic Notation:** **$p < 0.001$** (Paired Two-Tailed $t$-Test).

---

## 5. Module Evidence Classification

| Evaluation Category | Audit Classification | Data Source / Method |
| :--- | :---: | :--- |
| **Pretrained ArcFace ONNX SHA256** | **VERIFIED** | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| **AutoRoll Epoch-1 Checkpoint SHA256** | **VERIFIED** | `ed7456d474015570385634b660f0f5853f7b594e8d2cb333b923aabc2f2a42cf` |
| **In-Memory Failover Reassignment (0.10ms)** | **VERIFIED** | Real-time execution via `WorkerScheduler` process suite (5 trials) |
| **Backend Test Suite (121/121 Passing)** | **VERIFIED** | Pytest execution suite on Python 3.11 / FastAPI |
| **Frontend Production Build** | **VERIFIED** | Vite v8.2.2 production client bundle |
| **Single-Machine Pipeline Latency Baseline** | **PARTIALLY VERIFIED** | RTX 5060 single-node baseline measurement (8.2ms – 14.7ms) |
| **Distributed Multi-Machine Cluster Scaling** | **SIMULATED / LOOPBACK** | Single-node process loopback scheduler (`benchmark_distributed_workers.py`) |
| **Anti-Spoofing Liveness Presentation Attacks** | **NOT EXECUTED** | No physical attack presentation images stored in `data/real_world_evaluation/liveness/` |
| **100-Participant Real Image Benchmark** | **UNVERIFIED / PROTOTYPE** | Manifest-based simulation (`rng.normal`) |

---

## 6. Safe Claims for Research Publication

### Safe Claims (VERIFIED):
1. **Model Weights Untouched & Immutable:** ArcFace ONNX model SHA256 matches `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
2. **In-Memory Control Plane Failover:** Sub-millisecond camera reassignment upon worker offline timeout.
3. **Single-Node Pipeline Latency:** Baseline inference speed of 8.2ms P50 on RTX 5060 hardware.
4. **Phase 9 Calibration Set:** 25 real human participants (525 images) verified on disk.

### Unsafe Claims (Replaced with NOT EXECUTED / SIMULATED):
1. **Claims of 100 Real Human Participants in Phase 16:** Must be designated as **Simulated Evaluation Manifest** until physical photo acquisition is completed.
2. **Anti-Spoofing Presentation Attack Rates (ACER 1.2%):** Must be designated as **NOT EXECUTED ON PHYSICAL ATTACK MEDIA**.
3. **Multi-Server Physical Cluster Benchmarks:** Must be designated as **Single-Node Process Simulation**.
