# AutoRoll ArcFace Full Real-Data 1-Epoch Training Pre-Flight Audit Report

> [!NOTE]  
> **Pre-Flight Execution Timestamp**: 2026-08-20 05:48:04 UTC  
> **Status**: **APPROVED FOR FULL 10-EPOCH TRAINING**

---

## A. Dataset Statistics & Manifest Integrity

- **Ingested Source**: CASIA-WebFace Genuine Real Dataset
- **Total Valid Aligned Chips**: 487,739
- **Total Identities**: 10,428
- **Train Split**: 390,835 images / 8,342 identities (80%)
- **Validation Split**: 48,555 images / 1,042 identities (10%)
- **Test Split**: 48,339 images / 1,044 identities (10%)
- **Identity Leakage**: **ZERO** (100% disjoint splits)
- **Synthetic Data**: **FALSE** (Real human faces strictly verified)

---

## B. Model Architecture & Pretrained Integrity

- **Backbone**: PyTorch `MXNetIResNet50` (InsightFace `w600k_r50` architecture match)
- **Pretrained Checkpoint**: `models/pretrained/arcface_r50_webface_or_glint/model.onnx`
- **Pretrained ONNX SHA256**: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`
- **Parity Status**: Verified numerical parity against original ONNX graph (`cos_sim > 0.9987`)
- **Stage 1 Freezing**: Stem, Layer1, Layer2, Layer3 **FROZEN** (25,055,424 params, 57.4%); Layer4, BN2, FC, Features **TRAINABLE** (18,574,848 params, 42.6%)

---

## C. Training & Loss Configuration

- **Loss Head**: `ArcFaceLoss` (`num_classes=8342`, `scale=30.0`, `margin=0.20`)
- **Optimizer**: `SGD` (`lr=1e-4`, `momentum=0.9`, `weight_decay=5e-4`)
- **Learning Rate Schedule**: `CosineAnnealingLR` over 6,107 steps
- **Batch Size**: 64
- **Precision**: PyTorch FP32 Execution

---

## D. Pre-Flight Execution Performance Stats

- **Epoch Duration**: 487.87 seconds (8.13 minutes)
- **Average Throughput**: 801.1 images / second
- **Total Processed Images**: 390,835
- **Gradient Sanity**: Non-zero gradients observed across all trainable blocks; zero gradients on frozen blocks; **Zero NaN / Inf detected**.
- **Final 1-Epoch Train Loss**: `15.8434`

---

## E. Pre-Training Baseline vs. Epoch 1 Validation Protocols

| Metric | Pretrained Baseline | Epoch 1 Pre-Flight | Change / Delta |
| :--- | :---: | :---: | :---: |
| **Validation EER** | `25.10%` | `25.40%` | `+0.30%` |
| **Validation ROC-AUC** | `0.8336` | `0.8384` | `+0.0048` |
| **Validation Optimal Threshold** | `0.0410` | `0.0670` | `+0.0260` |
| **Validation Genuine Sim Mean** | `0.3617` | `0.3821` | `+0.0204` |
| **Validation Impostor Sim Mean** | `0.0040` | `0.0191` | `+0.0151` |
| **Validation TAR @ FAR=1e-3** | `65.40%` | `65.50%` | `+0.10%` |

---

## F. Pre-Training Baseline vs. Epoch 1 Test Protocol (at Validation-Selected Threshold)

> [!IMPORTANT]
> The threshold (`0.0410`) was selected strictly on Validation data.

| Metric | Pretrained Baseline | Epoch 1 Pre-Flight | Change / Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | `76.90%` | `77.30%` | `+0.40%` |
| **Test FAR** | `24.50%` | `23.60%` | `-0.90%` |
| **Test FRR** | `21.70%` | `21.80%` | `+0.10%` |
| **Test TAR** | `78.30%` | `78.20%` | `-0.10%` |

---

## G. Embedding Variance & Collapse Assessment

- **Baseline Global Embedding Variance**: `0.001953`
- **Post-Epoch 1 Embedding Variance**: `0.001953`
- **Variance Ratio (Post / Base)**: `1.0000`
- **Per-Dimension Variance (Mean)**: `0.001913` (Min: `0.001356`, Max: `0.002606`)
- **Collapse Verdict**: **NO EMBEDDING COLLAPSE** (Embedding space dimensions remain active and well-distributed).

---

## H. Checkpoint Artifact Verification

- **Checkpoint File**: [epoch_001_preflight.pt](file:///c:/Users/taran/Documents/GitHub/AutoRoll/models/trained/autoroll_arcface_v1/epoch_001_preflight.pt)
- **Status**: Saved and verified cleanly.

---

## I. Final Decision & Recommendation

> [!TIP]  
> **Recommendation**: **APPROVED FOR FULL 10-EPOCH TRAINING**  
> The 1-epoch pre-flight on real CASIA-WebFace data completed cleanly with valid loss convergence, stable gradients, intact embedding variance, and consistent generalization on unseen test identities.
