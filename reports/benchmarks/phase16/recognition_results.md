# Recognition Performance & Verification Metrics — Phase 16

| Metric | Model A (Pretrained ArcFace R50) | Model B (AutoRoll Epoch-1) |
| :--- | :---: | :---: |
| **Frozen Calibration Threshold** | 0.3003 | 0.3023 |
| **Held-Out Test EER** | **6.73%** | **1.24%** |
| **ROC-AUC Score** | 0.9668 | 0.9767 |
| **Fisher d' Separability** | 2.9138 | 4.5265 |
| **95% Bootstrap EER CI** | [6.24, 7.26]% | [1.01, 1.48]% |
| **95% Bootstrap AUC CI** | [0.9629, 0.97] | [0.973, 0.9808] |
| **Genuine Cosine Mean ± Std** | 0.4786 ± 0.1192 | 0.5517 ± 0.1084 |
| **Impostor Cosine Mean ± Std** | 0.1814 ± 0.0811 | 0.1396 ± 0.0695 |
