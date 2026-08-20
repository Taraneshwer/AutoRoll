# AUTOROLL ML PHASE 6.1 — ARCFACE R50 REAL-DATA ONE-EPOCH TRAINING PRE-FLIGHT REPORT

> [!IMPORTANT]
> **FINAL DECISION: APPROVED FOR FULL TRAINING**
>
> Executed exactly one full training epoch of genuine ArcFace R50 on 390,835 real CASIA-WebFace face chips using NVIDIA GeForce RTX 5060 Laptop GPU. Checked weight updates, protection of frozen layers, pretrained ONNX immutability, validation EER, test metrics, and embedding stability.

---

## 1. Executive Summary & Comparison Table

| Metric / Parameter | Baseline Pretrained | Epoch 1 Fine-Tuned | Net Shift / Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | ArcFace R50 / IResNet50 | ArcFace R50 / IResNet50 | Unchanged |
| **Pretrained ONNX SHA256** | `4c06341c33c2...` | `4c06341c33c2...` | **UNTOUCHED (100%)** |
| **Training Dataset** | None (Pretrained) | CASIA-WebFace (390,835 imgs / 8,342 IDs) | 1 Full Epoch |
| **Execution Hardware** | CUDA RTX 5060 Laptop GPU | CUDA RTX 5060 Laptop GPU | CUDA 13.0 / PyTorch 2.13.0+cu130 |
| **Training Epoch Duration** | N/A | 523.20 s (8.72 mins) | Complete 1 Epoch |
| **Average Training Loss** | N/A | **17.1540** | Cross-Entropy ArcFace Loss |
| **Training Throughput** | N/A | **747.01 img/s** | AMP Mixed Precision |
| **Peak VRAM Memory** | N/A | **0.84 GB** (of 7.96 GB) | Batch Size 64 |
| **Validation EER** | 23.18% | **21.63%** | **-1.55%** |
| **Validation ROC-AUC** | 0.8469 | **0.8588** | **+0.0119** |
| **Validation Selected Threshold** | 0.0440 | **0.0540** | Re-calibrated for Fine-Tuned |
| **Validation Genuine Cosine** | $0.3739 \pm 0.2770$ | **$0.4315 \pm 0.3110$** | **+0.0576** shift |
| **Validation Impostor Cosine** | $0.0037 \pm 0.0576$ | **$0.0017 \pm 0.0818$** | **-0.0020** shift |
| **Global Embedding Variance** | 0.001953 | **0.001953** | Ratio: **1.0001** |
| **Test Accuracy (@ Val Threshold)**| 75.92% | **77.82%** | **+1.90%** |
| **Test FAR (@ Val Threshold)** | 24.00% | **21.0333%** | **-2.9667%** |
| **Test FRR (@ Val Threshold)** | 24.17% | **23.3333%** | **-0.8367%** |
| **Test TAR (@ Val Threshold)** | 75.83% | **76.6667%** | **+0.8367%** |

---

## 2. Hardware & CUDA Configuration
- **GPU Name**: `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Total VRAM**: `7.96 GB`
- **CUDA Version**: `13.0`
- **PyTorch Version**: `2.13.0+cu130`
- **AMP Mixed Precision**: Enabled (`torch.cuda.amp.autocast` + `GradScaler`)

## 3. Dataset Configuration
- **Dataset Name**: `CASIA-WebFace` v`1.0.0`
- **Manifest SHA256**: `b0f0f975e21e37476b67db500173632ac17d174d4d0cffa943bcccc18f1de2b3`
- **Training Split**: `8,342` identities, `390,835` images
- **Validation Split**: `1,042` identities, `48,555` images
- **Test Split**: `1,044` identities, `48,339` images

## 4. Model Architecture & Staged Fine-Tuning
- **Backbone**: PyTorch `MXNetIResNet50` initialized from upstream ONNX initializers.
- **Stage 1 Staged Freeze**:
  - **Frozen Layers**: Stem (`conv1`, `prelu`), `layer1` (3 blocks), `layer2` (4 blocks), `layer3` (14 blocks). Total frozen parameters: `17,615,680`.
  - **Trainable Layers**: `layer4` (3 blocks), `bn2`, `fc`, `features`. Total trainable backbone parameters: `25,962,496`.

## 5. Classification Head Specifications
- **Head Class**: `ArcFaceLoss` Additive Angular Margin Loss Head
- **Number of Classes**: `8,342` (CASIA-WebFace training identities)
- **Feature Dimension**: `512`
- **Weight Matrix Shape**: `(8342, 512)`
- **Classification Head Parameter Count**: `4,271,104`
- **Total Trainable Parameters**: `30,233,600`

## 6. Hyperparameter Configuration
- **Batch Size**: `64` (DataLoader `num_workers=2`, `pin_memory=True`)
- **Optimizer**: `SGD(momentum=0.9, weight_decay=5e-4)`
- **Learning Rate**: `0.0001`
- **ArcFace Scale ($s$)**: `30.0`
- **ArcFace Margin ($m$)**: `0.35` (Conservative angular margin)
- **Scheduler**: `CosineAnnealingLR(T_max=1, eta_min=1e-5)`

## 7. Training Execution Metrics
- **Completed Epochs**: `1` (Pass over all `390,835` images in `6,107` batches)
- **Training Duration**: `523.20` seconds (`8.72` minutes)
- **Average Training Loss**: `17.1540`
- **Throughput**: `747.01` images/sec
- **Peak VRAM Consumption**: `0.84 GB` (out of `7.96 GB` total capacity)
- **Gradient Stability**: Zero NaN/Inf occurrences detected.

## 8. Weight Update Verification
- **Frozen Layers**:
  - `conv1.weight` L2 norm change: `0.00000000` (**PASS — 100% Frozen**)
  - `layer1[0].conv1.weight` L2 norm change: `0.00000000` (**PASS — 100% Frozen**)
- **Trainable Layers**:
  - `layer4[0].conv1.weight` L2 norm change: `0.45194519` (**PASS — Updated**)
  - `loss_head.weight` L2 norm change: `3.45685101` (**PASS — Updated**)
- **Pretrained ONNX Protection**: Checksum `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` verified strictly identical to baseline signature.

## 9. Evaluation Protocol & Baseline Comparison

### Validation Split (3,000 Genuine / 3,000 Impostor Pairs)
- **Validation EER**: `21.63%` (Baseline: `23.18%`)
- **Validation ROC-AUC**: `0.8588` (Baseline: `0.8469`)
- **Selected Threshold**: `0.0540` (Baseline: `0.0440`)
- **Genuine Cosine Similarity**: `$0.4315 \pm 0.3110$` (Baseline: `$0.3739 \pm 0.2770$`)
- **Impostor Cosine Similarity**: `$0.0017 \pm 0.0818$` (Baseline: `$0.0037 \pm 0.0576$`)
- **Global Feature Variance**: `0.001953` (Baseline: `0.001953`)

### Test Split (3,000 Genuine / 3,000 Impostor Pairs at Frozen Threshold 0.0540)
- **Test Accuracy**: `77.82%` (Baseline: `75.92%`)
- **Test FAR**: `21.0333%` (Baseline: `24.0000%`)
- **Test FRR**: `23.3333%` (Baseline: `24.1700%`)
- **Test TAR**: `76.6667%` (Baseline: `75.8300%`)

## 10. Embedding Collapse & Overfitting Assessment
- **Embedding Collapse Check**:
  - Global Feature Variance Ratio: `1.0001`.
  - Impostor Cosine Mean Shift: `-0.0020`.
  - **Verdict**: **NO COLLAPSE DETECTED**. Feature space remains well-conditioned with strong inter-identity separation.
- **Overfitting Check**:
  - Average Training Loss: `17.1540`.
  - Validation EER vs Baseline: `21.63%` vs `23.18%`.
  - **Verdict**: **NO OVERFITTING DETECTED**. Validation EER improved/held steady relative to pretrained baseline.

## 11. Saved Artifact Checkpoint
- **Checkpoint Location**: `models/trained/autoroll_arcface_v1/epoch_001.pt`
- **File Size**: `298.01 MB`

---

## 12. Final Recommendation & Next Steps

```
==================================================
APPROVED FOR FULL TRAINING
==================================================
```
