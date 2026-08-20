# AUTOROLL ML PHASE 3.1 — GPU-AWARE REAL FACE DATASET STRATEGY REPORT

> [!IMPORTANT]
> **GPU HARDWARE STRATEGY UPDATE**: This document updates the AutoRoll dataset and training strategy for **NVIDIA GeForce RTX 5060 GPU acceleration** (8.0 GB VRAM, CUDA 12.1/13.3 support, 16 physical / 24 logical CPU cores, 15.71 GB system RAM, 374.36 GB free disk space).

---

## 1. System Hardware Profile & GPU Inspection

| Hardware Component | Detected Specification | Operating Mode / Provider |
| :--- | :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 5060 Laptop GPU** | CUDA Hardware Acceleration |
| **VRAM Capacity** | **8,151 MiB (~8.0 GB VRAM)** | High-Speed GDDR6 |
| **NVIDIA Driver Version** | `610.47` (CUDA UMD Version `13.3`) | Full WDDM Hardware Acceleration |
| **CPU Architecture** | 16 physical cores / 24 logical threads | Multithreaded Data Loading |
| **System RAM** | **15.71 GB Total** (5.03 GB Available) | Pinned Memory DataLoader Host |
| **Free Workspace Disk** | **374.36 GB Free** on `C:` Drive | NVMe High-Speed Local Storage |

---

## 2. Hardware Execution Configuration & Device Resolution

The AutoRoll settings module ([config.py](file:///c:/Users/taran/Documents/GitHub/AutoRoll/autoroll/common/config.py)) has been updated to support explicit, non-silent hardware device resolution via `AUTOROLL_DEVICE`:

```python
AUTOROLL_DEVICE = "auto"  # Options: 'auto', 'cuda', 'cpu'
```

### Device Resolution Rules:
- **`auto`** *(Default)*: Automatically detects if CUDA is available (`torch.cuda.is_available()`); executes on `cuda` if available, otherwise falls back to `cpu`.
- **`cuda`** *(Strict GPU)*: Requires CUDA execution. Raises a fatal `RuntimeError` if CUDA is unavailable (prevents silent CPU fallback).
- **`cpu`** *(Strict CPU)*: Forces CPU execution for testing and debugging.

---

## 3. ONNX Runtime Execution Provider Hierarchy

For ONNX model inference (SCRFD detection & MiniFASNet liveness), AutoRoll initializes execution providers in strict priority order:

1. **`CUDAExecutionProvider`**: Primary provider leveraging TensorRT / CUDA kernels on the RTX 5060.
2. **`CPUExecutionProvider`**: Fallback provider for unsupported graph operators.

---

## 4. Evaluated Large-Scale Public Real Face Dataset Candidates

| Candidate Dataset | Total Identities | Total Real Images | Images / Identity | Official Source & Provenance | License / Restrictions | Download Availability | Storage Size (GB) | Suitability for ArcFace Fine-Tuning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGGFace2** | 9,131 | 3,310,000 | ~362 | Visual Geometry Group, Oxford University | Creative Commons BY-SA 4.0 (Academic Non-Commercial) | Official links offline; accessible via academic mirrors | 36.5 GB archive (120 GB uncompressed) | **EXCELLENT** (High pose & age diversity) |
| **CASIA-WebFace** | 10,575 | 494,414 | ~47 | Institute of Automation, Chinese Academy of Sciences (CASIA) | Academic Research License (Non-Commercial) | Academic mirrors available | 4.2 GB archive (15 GB uncompressed) | **VERY GOOD** (Standard academic benchmark) |
| **MS1MV2 (Glint360K)** | 87,000+ / 360,000+ | 3.8M / 17.0M | ~45 - 50 | InsightFace / DeepInsight Benchmark | MIT / Research License | HuggingFace Academic Repositories | 14 GB - 60 GB archive | **EXCELLENT** (Gold standard for ArcFace) |

---

## 5. GPU-Aware Dataset Subset Recommendation for RTX 5060

Considering the **8.0 GB VRAM** capacity, **15.71 GB system RAM**, and **374 GB free disk space**, we evaluate three subset strategies:

| Strategy Tier | Target Identities | Total Images | Storage Req. | Est. Preprocessing Time (RTX 5060) | Est. Fine-Tuning Time / Epoch (RTX 5060) | VRAM Peak Memory | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Small Development Subset** | 1,000 | ~50,000 | ~1.2 GB | ~25 minutes | ~3 minutes | ~2.1 GB | Quick pipeline verification |
| **B. Recommended Medium Subset** | **2,500** | **~125,000** | **~3.0 GB** | **~60 minutes** | **~7.5 minutes** | **~3.8 GB** | **RECOMMENDED OPTIMAL** |
| **C. Maximum Practical Subset** | 5,000 | ~250,000 | ~6.0 GB | ~120 minutes | ~15 minutes | ~5.8 GB | High-accuracy production candidate |

### Why Medium Subset (2,500 Identities / 125,000 Images) is Recommended:
1. **Memory Safety**: Consumes ~3.8 GB VRAM, leaving ~4.2 GB headroom on the RTX 5060 for CUDA context and system display buffers.
2. **Training Feasibility**: 10 fine-tuning epochs execute in **~75 minutes total**.
3. **Discriminative Capacity**: 2,500 identities provide ample inter-class margin for ArcFace loss convergence without overfitting.

---

## 6. Three-Tier Separated Dataset Architecture

The project directory maintains strict separation between training, evaluation, and local data:

```
data/
├── face_recognition/          # TIER 1: Large Public Training Dataset (CASIA / VGGFace2 subset)
│   ├── raw/
│   ├── detected/
│   ├── aligned/               # 112x112 affine face chips
│   ├── metadata/              # dataset_manifest.json
│   └── splits/                # Identity-disjoint TRAIN / VAL / TEST
├── benchmarks/                # TIER 2: Public Evaluation Benchmark
│   └── lfw/                   # LFW Benchmark (13,233 images, 5,749 identities)
└── local_students/            # TIER 3: Isolated Local Student Dataset (Enrollment & Calibration)
```

> [!CAUTION]
> Local student enrollment data in `data/local_students/` is **NEVER mixed** into the public face training dataset.

---

## 7. Dataset Manifest & Registry Metadata

Dataset registry configured in [configs/datasets.yaml](file:///c:/Users/taran/Documents/GitHub/AutoRoll/configs/datasets.yaml) with entries for `lfw`, `training_candidate_1` (VGGFace2), `training_candidate_2` (CASIA-WebFace), and `local_students`.

---

## 8. Recommended Next Step

Proceed to **AUTOROLL ML PHASE 4 — GPU-ACCELERATED ARCFACE FINE-TUNING PILOT**.

*(Zero model weights were modified and full fine-tuning was NOT launched during Phase 3.1).*
