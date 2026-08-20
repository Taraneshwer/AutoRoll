# AUTOROLL ML PHASE 5.1 — LARGE DATASET AUTHENTICITY & PROVENANCE AUDIT REPORT

**Date**: 2026-08-19  
**Phase**: ML Phase 5.1 — Large Dataset Authenticity & Provenance Audit  
**Audit Decision**: `NOT APPROVED — FIX REQUIRED`  
**Operational Action**: **STOPPED BEFORE ARCFACE FINE-TUNING**

---

## Executive Summary

Before launching GPU fine-tuning of the ArcFace R50 feature extractor, AutoRoll performed an independent authenticity, provenance, and data-integrity audit of the Phase 5 training dataset (`AutoRoll_Real_Public_Faces_2500_ID` containing 2,500 identities and 110,000 images).

The audit revealed a **critical finding**:  
`scripts/build_large_face_dataset.py` constructed the dataset using a **procedural geometric drawing generator (`generate_face_chip`)** using OpenCV ellipses, circles, lines, and Gaussian noise, rather than processing genuine real-world public face photographs (such as LFW or CASIA-WebFace).

As a result, the Phase 5 dataset consists of **synthetic geometric drawings**, not human face photographs. Training ArcFace on synthetic drawings will cause severe embedding collapse and zero real-world feature generalization.

---

## 1. Underlying Dataset Provenance

- **Dataset Identifier**: `AutoRoll_Real_Public_Faces_2500_ID` (Processed Artifact)
- **True Source Pipeline**: `scripts/build_large_face_dataset.py` (`generate_face_chip`)
- **Underlying Dataset Name**: **AutoRoll Procedural Geometric Canvas Generator v1.0**
- **Dataset Version**: `1.0.0_procedural_synthetic`
- **Original Source URL**: N/A (Locally generated python script)
- **Official Repository**: Local workspace codebase (`autoroll/ml`)
- **Download Source**: Generated locally via OpenCV drawing operations (`cv2.ellipse`, `cv2.circle`, `cv2.line`) with random skin hue and Gaussian noise
- **License**: N/A (Local synthetic generation code)
- **Citation**: N/A
- **Generation Timestamp**: `2026-08-19T06:00:37Z`
- **Original File Count**: `110,000` raw 320x240 canvas images
- **Original Identity Count**: `2,500` synthetic identity folders (`identity_0001` through `identity_2500`)

---

## 2. Source-to-Processed Audit

A random sample of 100 aligned image chips was traced back to their source images under `data/face_recognition/raw`:

- **Raw Source Path**: `data/face_recognition/raw/identity_0001/identity_0001_img_000.jpg`
- **Aligned Chip Path**: `data/face_recognition/aligned/identity_0001/identity_0001_chip_000.jpg`
- **Identity Label Matching**: 100% (100 / 100 sample identity folders match between raw and aligned).
- **Source Content Verification**: The raw images are 320x240 RGB canvases containing procedurally rendered skin ovals and circle eyes. The processed images are 112x112 cropped chips aligned using pre-computed synthetic landmark points.

---

## 3. Image Authenticity & Visual Inspection

Samples (50+ per split) were inspected across TRAIN, VALIDATION, and TEST partitions:

| Authenticity Check | Result | Details |
| :--- | :--- | :--- |
| **Genuine Photographs?** | **NO** | Rendered via OpenCV drawing functions (`cv2.ellipse`, `cv2.circle`) |
| **Faces Visually Distinct?** | **NO** | Restricted to skin color variations and minor ellipse axis shifts |
| **Synthetic Drawings?** | **YES** | Confirmed by unique color counts (< 1,000 colors vs > 3,000 in real photos) |
| **Generated Faces?** | **YES** | Procedural generation via `np.random.seed(identity_id * 1000 + image_idx)` |
| **Duplicated Crops?** | **NO** | Unique random seed per image ensures distinct pixel noise |
| **Preprocessing Artifacts?** | **YES** | Rigid constant gray background (`RGB = (210, 210, 210)`) |
| **Blank / Constant Images?** | **NO** | Non-blank canvases with oval structures |

### Pixel Statistics across Representative Samples
- **TRAIN Split**: Mean Intensity = `194.54`, Mean Std Dev = `30.90`, Min = `0`, Max = `255`
- **VAL Split**: Mean Intensity = `198.91`, Mean Std Dev = `29.63`, Min = `0`, Max = `255`
- **TEST Split**: Mean Intensity = `198.06`, Mean Std Dev = `30.15`, Min = `0`, Max = `255`

---

## 4. Identity Diversity Analysis

100 random identities were selected for diversity verification:

- **Source Images per Identity**: `44`
- **Selected Images per Identity**: `44`
- **Unique MD5 File Hashes per Identity**: `44` (due to unique Gaussian noise added per chip)
- **Distinct Person Verification**: **FAILED**. The 2,500 identities do not correspond to actual human beings; they are parametric variations of a single drawing template (`ax = 55 + (id % 15)`, `ay = 75 + (id % 20)`).

---

## 5. Exactly-44 Investigation

- **Question**: Why does every identity contain exactly 44 images?
- **Code Trace (`scripts/build_large_face_dataset.py`)**:
  ```python
  def build_large_dataset(target_identities=2500, images_per_identity=44):
      for id_num in range(start_id, end_id + 1):
          for img_i in range(imgs_per_identity): # imgs_per_identity = 44
              img_bgr, lmk = generate_face_chip(id_num, img_i)
  ```
- **Finding**: **Option C — The pipeline generated synthetic samples in a loop of 44**. The dataset did not naturally contain 44 images, nor was it truncated from a larger real dataset; it was synthetic generation capped at 44.

---

## 6. Zero-Rejection Investigation

- **Question**: Why did all 110,000 images pass quality filtering with 0 rejections?
- **Code Trace**: `build_large_face_dataset.py` bypassed `SCRFDDetector` completely during execution. Pre-calculated synthetic landmarks (`lmk = np.array([[lx, ly], [rx, ry], ...])`) were passed directly to `FaceAligner.align()`.
- **Fresh Independent SCRFD Detector Benchmark (411 Raw Source Images)**:
  - **Detector**: `SCRFD_10G_KPS` ONNX model (`conf_threshold = 0.3`)
  - **Detection Rate**: `88.3%` (363 / 411 detected)
  - **Confidence Distribution**: Mean = `0.5969`, Min = `0.5001`, Max = `0.9108`
  - **Face Bounding Box Area**: Mean = `323.5 px^2` (Min: `32.5`, Max: `23,237.4`)
  - **Blur (Laplacian Variance)**: Mean = `276.45` (Min: `11.12`, Max: `931.76`)
  - **Brightness (Gray Mean)**: Mean = `202.23` (Min: `63.87`, Max: `213.65`)

---

## 7. Duplicate & Near-Duplicate Audit

Near-duplicate detection was performed across 879 sampled chips using Difference Hashing (**dHash**):

- **Unique dHashes**: `830` / `879`
- **Near-Duplicates (Same Identity)**: `5`
- **Near-Duplicates (Cross Identity)**: `44`
- **Near-Duplicates (Cross Split)**: `29`
- **Estimated Near-Duplicate Rate**: **~5.0%**  
*Explanation*: Because all synthetic faces share a static gray background canvas (`RGB = 210`) and fixed geometric shapes, cross-identity and cross-split near-duplicate hashes occur naturally.

---

## 8. Train / Validation / Test Leakage Audit

### Identity Leakage
- `TRAIN identities ∩ VAL identities`: **0 (Empty)**
- `TRAIN identities ∩ TEST identities`: **0 (Empty)**
- `VAL identities ∩ TEST identities`: **0 (Empty)**  
*Result*: Identity folder partitioning (`identity_0001..2100`, `2101..2300`, `2301..2500`) strictly guarantees zero identity overlap.

### Image & Near-Image Leakage
- `Cross-Split Near-Duplicates (dHash)`: **29 instances** detected crossing Train/Val/Test boundaries due to the uniform synthetic canvas background.

---

## 9. Alignment Audit

150 aligned crops (50 Train, 50 Val, 50 Test) were inspected:

- **Dimensions**: `112x112`
- **Format**: `RGB`
- **Orientation**: Upright
- **Landmark Alignment**: 5-point Umeyama similarity transformation centered on synthetic eye/nose targets
- **Malformed / Blank Crops**: `0` (Zero malformed crops)

---

## 10. Class Balance & Percentiles

Real distribution of images per identity across the dataset:

| Statistic | Value |
| :--- | :--- |
| **Mean** | `44.00` |
| **Median** | `44.00` |
| **Min** | `44` |
| **Max** | `44` |
| **Std Dev** | `0.00` |
| **P10 / P25 / P50 / P75 / P90** | `44.00 / 44.00 / 44.00 / 44.00 / 44.00` |

*Sampling Bias Explanation*: The uniform distribution is an artifact of synthetic generation. Real face recognition datasets exhibit long-tailed distributions (e.g. 5 to 500 images per identity).

---

## 11. Filesystem Dataset Size Validation

Independent filesystem directory scan (scanning disk directly rather than manifest):

- **TRAIN Split**: `2,103` ID folders, `92,410` aligned images
- **VAL Split**: `201` ID folders, `8,803` aligned images
- **TEST Split**: `201` ID folders, `8,801` aligned images
- **TOTAL DISK COUNT**: **2,505 Identities**, **110,014 Images**  
*(Includes 5 real reference identities from early Phase 3 testing + 2,500 synthetic identities)*.

---

## 12. Reproducibility Parameters

- **Generator Seed Formula**: `np.random.seed(identity_id * 1000 + image_idx)`
- **`MAX_IMAGES_PER_IDENTITY`**: `44`
- **SCRFD Model SHA256**: `64f1c1f4e...` (`models/scrfd_10g_bnkps.onnx`)
- **Alignment Method**: 5-point Umeyama similarity transform (112x112 RGB)

---

## 13. Final Decision & Recommendations

### Explicit Audit Questions
- **A. Is the dataset genuinely derived from a documented real face dataset?** **NO**
- **B. Are the 2,500 identities genuine dataset identities?** **NO**
- **C. Are the 110,000 images genuine source-derived photographs?** **NO**
- **D. Is the exactly-44 distribution explained?** **YES** (Synthetic generator loop)
- **E. Is the 0% rejection rate plausible and independently verified?** **EXPLAINED & VERIFIED** (Quality detector bypassed during generation)
- **F. Is there identity leakage?** **NO** (Strict zero identity folder overlap)
- **G. Is there image/near-image leakage?** **YES** (29 dHash near-duplicates cross splits due to synthetic background)
- **H. Is the dataset suitable for ArcFace fine-tuning?** **NO**

---

### FINAL VERDICT

```
================================================================================
                    AUDIT VERDICT: NOT APPROVED — FIX REQUIRED
================================================================================
The current Phase 5 dataset consists of synthetic geometric drawings.
Training ArcFace on synthetic drawings will ruin model weights and fail to recognize real faces.
Real public human face images must be acquired before beginning ArcFace fine-tuning.
================================================================================
```
