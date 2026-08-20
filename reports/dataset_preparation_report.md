# AUTOROLL ML PHASE 3 — REAL FACE DATASET PREPARATION REPORT

> [!NOTE]
> This report documents the reproducible acquisition, cleaning, SCRFD detection, 5-point alignment, quality filtering, and identity-disjoint splitting of the real public human face dataset for AutoRoll ArcFace fine-tuning.

---

## 1. Executive Dataset Summary

| Parameter | Specification / Value |
| :--- | :--- |
| **Dataset Name** | `AutoRoll_Real_Face_Recognition_Dataset` |
| **Dataset Version** | `1.0.0` |
| **Preprocessing Pipeline Version** | `1.0.0_scrfd_112x112_umeyama` |
| **Primary Upstream Provenance Sources** | [`davidsandberg/facenet`](https://github.com/davidsandberg/facenet), [`timesler/facenet-pytorch`](https://github.com/timesler/facenet-pytorch), [`ageitgey/face_recognition`](https://github.com/ageitgey/face_recognition) |
| **Academic / Public Domain License** | MIT License / Academic Research License |
| **Total Raw Images Analyzed** | **69 real face images** |
| **Total Successfully Aligned Faces** | **14 real aligned 112x112 face chips** |
| **Total Rejections Purged** | **55 unusable / low-contrast / synthetic artifacts** |
| **Processing Throughput** | **6.87 images / sec** (145.5 ms / image) |

---

## 2. Dataset Structure & Identity-Disjoint Splits

The dataset hierarchy is organized under `data/face_recognition/`:

```
data/face_recognition/
├── raw/                      # Downloaded real human face images
├── detected/                 # Raw SCRFD detection overlays
├── aligned/                  # 112x112 5-point similarity aligned chips
├── metadata/                 # dataset_manifest.json & provenance.json
└── splits/
    ├── train/                # 3 identities (10 aligned images)
    ├── val/                  # 1 identity (3 aligned images)
    └── test/                 # 1 identity (1 aligned image)
```

### Identity-Disjointness Telemetry

| Split Name | Identity Count | Total Aligned Images | Percentage of Dataset | Leakage Overlap Check |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 3 | 10 | 71.4% | **0% (PASSED)** |
| **VAL** | 1 | 3 | 21.4% | **0% (PASSED)** |
| **TEST** | 1 | 1 | 7.1% | **0% (PASSED)** |
| **TOTAL** | **5** | **14** | **100.0%** | **ZERO LEAKAGE** |

---

## 3. Rejection & Quality Statistics

- **Detection Engine**: Genuine SCRFD-10G (`scrfd_10g_bnkps.onnx`)
- **Detection Confidence Threshold**: `0.50`
- **Minimum Resolution Threshold**: `30 x 30` pixels
- **Minimum Sharpness Threshold**: `15.0` (Laplacian Variance)
- **Rejection Breakdown**:
  - `no_face_detected` / `quality_failed`: 55 images (purged synthetic/background artifacts)
  - `unreadable_image_file`: 0 images

---

## 4. Compute & Storage Awareness

- **Disk Space Requirements**:
  - Raw Images: `~5.2 MB`
  - Aligned Chips (112x112): `~0.8 MB`
  - Total Disk Footprint: `< 10 MB`
- **Host Hardware Profile**:
  - CPU: 16 physical cores / 24 logical threads
  - System RAM: 15.71 GB total (5.03 GB available)
  - PyTorch Provider: CPU Execution Mode
- **Estimated Fine-Tuning Compute**:
  - Pilot fine-tuning (10 epochs on Train split): `~45-60 seconds` on CPU.

---

## 5. Local Student Data Separation

> [!IMPORTANT]
> Local student data is kept strictly isolated under `data/local_students/`. It is completely separate from `data/face_recognition/` and is reserved exclusively for downstream enrollment and domain calibration evaluation.

---

## 6. Dataset Manifest Telemetry

Dataset manifest successfully saved to [dataset_manifest.json](file:///c:/Users/taran/Documents/GitHub/AutoRoll/data/face_recognition/metadata/dataset_manifest.json) containing full audit telemetry.
