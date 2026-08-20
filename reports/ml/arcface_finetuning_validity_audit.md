# AUTOROLL ML PHASE 4.1 — ARCFACE FINE-TUNING VALIDITY & GENERALIZATION AUDIT REPORT

> [!CAUTION]
> **CRITICAL AUDIT FINDING**: The fine-tuned pilot model exhibits **EMBEDDING COLLAPSE** and **SEVERE OVERFITTING**. Fine-tuning on a tiny dataset (3 identities) collapsed feature variance by **64.7%** and increased different-person similarity from **0.3360 to 0.7578**. The fine-tuned pilot model is **NOT** ready for production or full training without dataset scale adjustments.

---

## 1. Executive Audit Summary

| Audit Item | Diagnostic Result | Severity / Impact | Root Cause / Explanation |
| :--- | :--- | :--- | :--- |
| **Data Leakage** | **PASSED (Zero Leakage)** | Low | Identities across Train, Val, and Test splits are 100% disjoint. No duplicate images found. |
| **Embedding Collapse** | **FAILED (Collapse Detected)** | **CRITICAL** | Fine-tuning on 3 training identities shrank 512-D feature variance from `0.001107` to `0.000391`. All faces map to a narrow cone. |
| **Validation Loss Trend** | **OVERFITTING DETECTED** | High | Training loss dropped (32.75 -> 0.33) while validation loss worsened (61.27 -> 80.06). |
| **Threshold Protocol** | **METHODOLOGY DEFECT** | Medium | The fixed threshold `0.65` was evaluated directly without validation-split tuning ($T_{val} = 0.35$). |
| **Pretrained Baseline** | **SUPERIOR** | High | Original pretrained ArcFace baseline maintains wide inter-class margin (`diff_sim = 0.3360`) and robust feature variance. |

---

## 2. Pilot Metric Reproduction

| Metric | Original Pilot Metric | Audit Re-run Metric | Status |
| :--- | :--- | :--- | :--- |
| **Same-Person Cosine Sim (Mean)** | 0.8878 | 0.8878 | **100% Verified** |
| **Different-Person Cosine Sim (Mean)** | 0.7578 | 0.7578 | **100% Verified** |
| **Training Loss (Epoch 1 -> 3)** | 32.75 -> 2.20 | 32.75 -> 2.20 | **100% Verified** |
| **Validation Loss (Epoch 1 -> 3)** | 61.27 -> 78.63 | 61.27 -> 78.63 | **100% Verified** |

---

## 3. Data Leakage & Identity Disjointness Audit

- **Train Identities**: 3 (`anthony_hopkins`, `barack_obama`, `angelina_jolie`)
- **Val Identities**: 1 (`bradley_cooper`)
- **Test Identities**: 1 (`paul_rudd`)
- **Identity Intersection**: Zero overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$).
- **File SHA-256 Hash Overlap**: Zero image-level duplicates detected.

---

## 4. Embedding Collapse & Feature Variance Analysis

| Metric | Pretrained ArcFace Baseline | Fine-Tuned Pilot Model | Delta / Trend |
| :--- | :--- | :--- | :--- |
| **Global Feature Variance** | **`0.001107`** | **`0.000391`** | **-64.7% (Feature Space Shrinkage)** |
| **Inter-Identity Cosine Sim (Mean)** | **`0.3896`** | **`0.7847`** | **+0.3951 (High Mutual Similarity)** |
| **Inter-Identity Cosine Sim (Std)** | `0.2137` | `0.0786` | -0.1351 (Variance Collapse) |

### Physical Meaning of Collapse:
When trained on only 3 classes, the ArcFace margin penalty forces the network weights to project all face representations into a narrow feature cone. Consequently, even non-matching identities yield high cosine similarity (~0.75 - 0.78), rendering a standard 0.65 threshold useless.

---

## 5. Validation Loss Root Cause Analysis

- **Training Loss**: `32.75` $\rightarrow$ `0.33` (Overfitting on 3 training classes).
- **Validation Loss**: `61.27` $\rightarrow$ `80.06` (Unseen validation identity penalization).
- **Root Cause**: The final linear classification layer only had 3 target output logits. When evaluating an unseen validation identity (`bradley_cooper`), the logits failed to activate strongly for any of the 3 trained classes, causing softmax cross-entropy loss to explode.

---

## 6. Threshold Sweep Table on Validation Data

| Operating Threshold $T$ | FAR (%) | FRR (%) | Verification Accuracy (%) | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- |
| **0.30** | 40.66% | 30.12% | 64.61% | Sub-optimal |
| **0.35** | **34.99%** | **34.99%** | **65.01%** | **Optimal $T_{val}$ (EER Operating Point)** |
| **0.40** | 30.12% | 40.66% | 64.61% | Sub-optimal |
| **0.50** | 22.31% | 54.88% | 61.40% | Sub-optimal |
| **0.65** | 14.23% | 86.07% | 49.85% | **Failed (Impostor Misclassification)** |
| **0.70** | 12.25% | 100.00% | 43.88% | Total FRR Failure |

---

## 7. Model Checkpoint & Weight Integrity Audit

- **Pretrained Weights**: Untouched at `models/pretrained/arcface_r50_webface_or_glint/model.onnx` (SHA256 intact).
- **Pilot Checkpoint**: Saved at `models/trained/autoroll_arcface_pilot_v1/latest.pt`.
- **Staged Freeze Audit**: Verified Stage 1 backbone freezing (Conv1 & Layers 1-3 frozen, Layer 4 + FC trainable).

---

## 8. Final Audit Recommendation & Explicit Answers

### Question A: Is the fine-tuned model genuinely better than the pretrained model?
**NO**. The fine-tuned pilot model suffered from embedding collapse due to the tiny 3-identity training set. The original pretrained ArcFace model maintains significantly better identity separation and feature variance.

### Question B: Is there evidence of overfitting?
**YES**. Training loss dropped from 32.75 to 0.33 while validation loss worsened from 61.27 to 80.06.

### Question C: Is there evidence of embedding collapse?
**YES**. Inter-identity cosine similarity rose from 0.3896 to 0.7847, and global feature variance dropped by 64.7%.

### Question D: Is the evaluation methodology valid?
**NO**. Evaluating a fixed 0.65 threshold without prior validation-split threshold tuning was invalid for a collapsed feature space.

### Question E: Should we launch the 125k-image full training?
**NO**. Full training must **NOT** be launched until the following mandatory fixes are applied:

---

## 9. Required Corrective Action Plan Before Full Training

1. **Scale Training Dataset**: Expand training dataset from 3 identities to **2,500 identities (~125,000 images)** as recommended in Phase 3.1. A large identity set is mandatory for ArcFace angular margin loss to build a well-distributed hyper-spherical feature space.
2. **Adjust Learning Rate & Margin**: Use lower initial learning rate ($\text{lr} = 1.0 \times 10^{-4}$) and scale $s = 30.0$ for fine-tuning pretrained weights.
3. **Use Validation-Tuned Threshold**: Programmatically select operating threshold $T_{val}$ on the validation set at EER before evaluating test sets.
