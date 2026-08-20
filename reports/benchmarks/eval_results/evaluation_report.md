# AutoRoll Face Verification Evaluation Report

**Experiment ID**: `eval_phase5`  
**Calibrated Operating Threshold**: `0.9393`

---

## Verification Metrics Summary

| Metric | Pretrained Model | Fine-Tuned Model | Improvement |
| :--- | :---: | :---: | :---: |
| Accuracy | 0.9048 | 0.9048 | +0.0000 |
| Equal Error Rate (EER) | 0.0000 | 0.0000 | +0.0000 |
| F1 Score | 0.9474 | 0.9474 | +0.0000 |
| FAR (False Accept Rate) | 0.6667 | 0.6667 | - |
| FRR (False Reject Rate) | 0.0000 | 0.0000 | - |
| Avg Latency (ms) | 0.4600 | 0.4600 | - |

## TAR at Target FAR Values (Fine-Tuned Model)

| Target FAR | True Accept Rate (TAR) | Operating Threshold |
| :--- | :---: | :---: |
| FAR = 0.1000 | TAR = 1.0000 | Threshold = 0.9892 |
| FAR = 0.0100 | TAR = 1.0000 | Threshold = 0.9892 |
| FAR = 0.0010 | TAR = 1.0000 | Threshold = 0.9892 |
| FAR = 0.0001 | TAR = 1.0000 | Threshold = 0.9892 |

## Cosine Similarity Score Distributions

- **Genuine Pairs**: Mean = `1.0000`, Std = `0.0000`
- **Impostor Pairs**: Mean = `0.9627`, Std = `0.0261`

---
*Report generated automatically by AutoRoll Evaluation Pipeline.*
