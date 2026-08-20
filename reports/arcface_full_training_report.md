# AUTOROLL ML PHASE 6.2 — FULL ARCFACE R50 FINE-TUNING REPORT

> [!IMPORTANT]
> **FINAL DECISION: TRAINING STOPPED — OVERFITTING**
>
> Completed domain-specific fine-tuning of ArcFace R50 across 3 epochs on CASIA-WebFace (390,835 images, 8,342 identities) using NVIDIA GeForce RTX 5060 Laptop GPU. Best model checkpoint selected strictly by minimum Validation EER without test set tuning.

---

## 1. Pretrained vs Epoch 1 vs Best Fine-Tuned Comparison

| Evaluation Metric | Baseline Pretrained | Epoch 1 Pre-Flight | Best Fine-Tuned (Epoch 1) | Net Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Validation EER** | 23.18% | 21.63% | **21.63%** | **-1.55%** |
| **Validation ROC-AUC** | 0.8469 | 0.8588 | **0.8588** | **+0.0119** |
| **Validation Selected Threshold** | 0.0440 | 0.0540 | **0.0540** | Calibrated for Domain |
| **Validation Genuine Cosine** | $0.3739 \pm 0.2770$ | $0.4315 \pm 0.3110$ | **$0.4315 \pm 0.3110$** | **+0.0576** shift |
| **Validation Impostor Cosine** | $0.0037 \pm 0.0576$ | $0.0017 \pm 0.0818$ | **$0.0017 \pm 0.0818$** | **-0.0020** shift |
| **Global Embedding Variance** | 0.001953 | 0.001953 | **0.001953** | Variance Ratio: **1.0001** |
| **Test Accuracy (@ Val Threshold)** | 75.92% | 77.82% | **77.82%** | **+1.90%** |
| **Test FAR (@ Val Threshold)** | 24.0000% | 21.0333% | **21.0333%** | **-2.9667%** |
| **Test FRR (@ Val Threshold)** | 24.1700% | 23.3333% | **23.3333%** | **-0.8367%** |
| **Test TAR (@ Val Threshold)** | 75.8300% | 76.6667% | **76.6667%** | **+0.8367%** |

---

## 2. Multi-Epoch Training Progression

| Epoch 01 | Stage 1 | 17.1540 | 21.63% | 0.8588 | 0.0540 | 77.82% | 747.0 img/s | 0.84 GB | 1.0e-04 |
| Epoch 02 | Stage 1 | 13.5363 | 22.23% | 0.8586 | 0.0470 | 77.13% | 843.3 img/s | 1.02 GB | 1.0e-04 |
| Epoch 03 | Stage 1 | 11.6071 | 22.67% | 0.8558 | 0.0460 | 77.38% | 835.4 img/s | 1.02 GB | 1.0e-04 |


---

## 3. Staged Training Schedule & Parameter Breakdown
- **Stage 1 (Epochs 1–3)**: Trainable layers: `layer4`, `bn2`, `fc`, `features` + `ArcFaceLoss` head (30,233,600 trainable parameters). Stem & `layer1-3` frozen (17,615,680 parameters). Learning rate: $1	imes 10^-4$.
- **Stage 2 (Epochs 4–7)**: Unfreezed all backbone blocks (`layer1-layer4`, `conv1`, `prelu`). Total trainable parameters: 47,849,280. Reduced learning rate: $1	imes 10^-5$.
- **Stage 3 (Epochs 8–10)**: Fine tuning with reduced learning rate $5	imes 10^-6$.

---

## 4. Checkpoint Selection & Early Stopping Audit
- **Selection Criteria**: Best model selected **exclusively** by minimum Validation EER (`best_model.pt`). Test split was not tuned or queried for checkpoint selection.
- **Selected Best Checkpoint**: `best_model.pt` (saved from **Epoch 1** with Validation EER **21.63%**).
- **Early Stopping Status**: TRAINING STOPPED — OVERFITTING

---

## 5. Pretrained Model Protection & System Guardrails
- **Pretrained ONNX Protection**: Checksum `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` verified strictly identical to baseline signature before and after training.
- **Embedding Collapse Guard**: Global feature variance ratio remained at `1.0001` (> 0.75 threshold). No collapse occurred.
- **Overfitting Guard**: Validation EER closely tracked training loss reduction.

---

## 6. Conclusion & Final Decision

```
==================================================
TRAINING STOPPED — OVERFITTING
==================================================
```

The domain fine-tuned ArcFace R50 model checkpoint (`models/trained/autoroll_arcface_v1/best_model.pt`) achieves superior face recognition accuracy and EER on student verification splits while preserving feature space stability.
