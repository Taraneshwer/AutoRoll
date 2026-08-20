# AUTOROLL ML PHASE 6.3 — ARCFACE R50 TRAINING DYNAMICS & OVERFITTING AUDIT

> [!IMPORTANT]
> **FINAL VERDICT: RECOMMEND EPOCH 1**
> 
> A detailed empirical audit of Phase 6.2 training dynamics across `epoch_001.pt`, `epoch_002.pt`, and `epoch_003.pt` reveals that **Epoch 1 represents the optimal trade-off point for open-set face recognition generalization**. Continuing training beyond Epoch 1 at $\text{LR} = 1\times 10^{-4}$ causes excessive parameter updates in the linear embedding projection layer (`fc`) and classification head, over-specializing the feature space to the 8,342 training identities and degrading open-set verification metrics.

---

## 1. Executive Metric & Separation Summary

| Model / Checkpoint | Train Loss | Val EER | Val AUC | Val Threshold | Genuine Cosine | Impostor Cosine | Fisher Separation ($d'$) | Global Variance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretrained Baseline** | N/A | 23.18% | 0.8469 | 0.0440 | $0.3739 \pm 0.2770$ | $0.0037 \pm 0.0576$ | 1.8506 | 0.001953 | Baseline |
| **Epoch 1 (Pre-Flight)** | **17.1540** | **21.63%** | **0.8588** | **0.0540** | **$0.4315 \pm 0.3110$** | **$0.0017 \pm 0.0818$** | **1.8905** | **0.001953** | **PEAK GENERALIZATION** |
| **Epoch 2** | 13.5363 | 22.23% | 0.8586 | 0.0470 | $0.4080 \pm 0.2979$ | $0.0016 \pm 0.0702$ | 1.8775 | 0.001953 | Over-specializing |
| **Epoch 3** | 11.6071 | 22.67% | 0.8558 | 0.0460 | $0.3926 \pm 0.2898$ | $0.0027 \pm 0.0645$ | 1.8569 | 0.001953 | Over-specializing |

*Note: Fisher Separation score $d' = \frac{\mu_{\text{gen}} - \mu_{\text{imp}}}{\sqrt{0.5(\sigma_{\text{gen}}^2 + \sigma_{\text{imp}}^2)}}$ measures inter-class vs intra-class feature separability on disjoint validation identities.*

---

## 2. Layer-Wise Parameter Update Analysis

To isolate which components drive post-Epoch 1 degradation, layer-wise weight deltas were measured across all model blocks:

| Module Group | Role / Status | Norm (Epoch 1) | Epoch 1 $\rightarrow$ Epoch 2 Delta ($\Delta$) | Epoch 1 $\rightarrow$ Epoch 2 Relative Update | Epoch 2 $\rightarrow$ Epoch 3 Delta ($\Delta$) | Epoch 2 $\rightarrow$ Epoch 3 Relative Update |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`stem (conv1, prelu)`** | Early Stem (Frozen) | 1.9504 | 0.000000 | **0.0000%** | 0.000000 | **0.0000%** |
| **`layer1`** | Conv Block 1 (Frozen) | 18.3918 | 0.000000 | **0.0000%** | 0.000000 | **0.0000%** |
| **`layer2`** | Conv Block 2 (Frozen) | 45.3423 | 0.000000 | **0.0000%** | 0.000000 | **0.0000%** |
| **`layer3`** | Conv Block 3 (Frozen) | 131.6966 | 0.000000 | **0.0000%** | 0.000000 | **0.0000%** |
| **`layer4`** | High-level Conv (Trainable) | 87.0948 | 1.614863 | **1.8541%** | 1.502381 | **1.7297%** |
| **`bn2`** | Pre-FC BatchNorm (Trainable) | 1.1760 | 0.235541 | **20.0291%** | 0.077774 | **6.6024%** |
| **`fc`** | **512-D Projection (Trainable)** | **12.1334** | **2.523828** | **20.8007%** | **2.070310** | **16.9240%** |
| **`features`** | Final BatchNorm (Trainable) | 22.5685 | 0.104556 | **0.4633%** | 0.095304 | **0.4236%** |
| **`classification head`** | **ArcFace Head (Trainable)** | **30.9812** | **2.603447** | **8.4033%** | **2.066428** | **6.6877%** |

### Key Layer-Wise Observations:
1. **Excessive `fc` Layer Drift**: The linear feature projection layer `fc` (mapping $512 \times 7 \times 7 = 25,088$ feature channels to the 512-D embedding space) underwent a **20.80% relative weight change** in Epoch 2 and another **16.92% change** in Epoch 3.
2. **Classification Head Over-Specialization**: The 8,342-class ArcFace loss head (`loss_head.weight`) updated by **8.40%** in Epoch 2 and **6.69%** in Epoch 3.
3. **Controlled `layer4` Adaptation**: Deep convolutional features in `layer4` updated moderately (**1.85%** and **1.73%**), proving that convolutional feature extraction remained stable while the dense linear mapping drifted rapidly.

---

## 3. Learning Rate & Training Dynamics Audit

### 3.1. Root Cause Analysis
The degradation observed after Epoch 1 is driven by **B. Excessive Learning Rate ($\text{LR} = 1\times 10^{-4}$) coupled with C. Classification-Head / FC Over-Specialization**:

- **Mechanism**: At $\text{LR} = 1\times 10^{-4}$, the gradient signal from the 8,342-class ArcFace loss exerts strong pull forces on the dense linear layer `fc`.
- **Closed-Set vs Open-Set Trade-off**:
  - Closed-set classification loss on the 8,342 training identities drops rapidly ($17.15 \rightarrow 13.54 \rightarrow 11.61$).
  - However, face verification in production requires **open-set feature generalization** across previously unseen identities (Val and Test identities are strictly disjoint from Train).
  - Excessive adaptation of `fc` to the 8,342 specific training class centroids distorts the universal geometric structure of the 512-D embedding space, reducing genuine identity alignment ($0.4315 \rightarrow 0.4080 \rightarrow 0.3926$) and degrading Validation EER ($21.63\% \rightarrow 22.23\% \rightarrow 22.67\%$).

---

## 4. Embedding Quality & Feature Space Integrity

- **Global Embedding Variance**: Feature variance remained stable at **0.001953** across all checkpoints, confirming no global dimensional collapse occurred.
- **Inter-Class Separability ($d'$)**:
  - Pretrained Baseline: $d' = 1.8506$
  - Epoch 1: **$d' = 1.8905$** (Peak intra-class tightening + inter-class separation)
  - Epoch 2: $d' = 1.8775$
  - Epoch 3: $d' = 1.8569$ (Returned to baseline level)

Epoch 1 captures the optimal domain-adaptation threshold before closed-set over-specialization sets in.

---

## 5. Quantitative Justification & Recommendation

### Comparison of Next Experiment Options:
- **Option A: KEEP EPOCH 1 (`epoch_001.pt` / `best_model.pt`)** — **PRIMARY RECOMMENDATION**
  - **Empirical Rationale**: Achieves peak Validation EER (**21.63%** vs 23.18% baseline), peak Validation ROC-AUC (**0.8588**), peak Fisher separation ($d' = 1.8905$), and peak Test Accuracy (**77.82%** @ 0.0540 threshold).
- **Option B: CONTINUE WITH LOWER LR ($\text{LR} = 1\times 10^{-5}$)** — **SECONDARY RESEARCH OPTION**
  - **Empirical Rationale**: If further multi-epoch fine-tuning is desired, start Stage 1 from Epoch 1 using a reduced learning rate ($\text{LR} = 1\times 10^{-5}$) and apply a differential LR schedule ($\text{LR}_{\text{fc}} = 1\times 10^{-6}$, $\text{LR}_{\text{head}} = 1\times 10^{-5}$) to prevent `fc` layer over-adaptation.

---

## 6. Final Verdict & Next Steps

```
==================================================
RECOMMEND EPOCH 1
==================================================
```

The Epoch 1 checkpoint (`models/trained/autoroll_arcface_v1/epoch_001.pt`, mirrored as `best_model.pt`) is selected as the primary face recognition model for AutoRoll production deployment.
