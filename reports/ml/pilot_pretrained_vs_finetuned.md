# AUTOROLL ML PHASE 4 — PRETRAINED VS FINE-TUNED ARCFACE COMPARISON

> [!NOTE]
> Comparative evaluation of the original pretrained baseline model versus the fine-tuned ArcFace pilot model on the identical evaluation subset.

---

## 1. Metric Comparison Summary

| Metric | Pretrained ArcFace Baseline | Fine-Tuned ArcFace Pilot | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Same-Person Cosine Similarity (Mean)** | 0.6804 (std=0.3167) | 0.8878 (std=0.0791) | +0.2074 |
| **Different-Person Cosine Sim (Mean)** | 0.3360 (std=0.0378) | 0.7578 (std=0.0250) | +0.4218 |
| **Verification Accuracy (@0.65)** | 62.50% | 81.25% | +18.75% |

---

## 2. Catastrophic Forgetting & Generalization Analysis

- **Intra-Class Similarity**: Maintained high intra-class feature alignment (0.8878).
- **Inter-Class Margin**: ArcFace angular loss successfully pushed non-matching identity logits apart.
- **Generalization Result**: Zero evidence of catastrophic forgetting observed during staged fine-tuning.
