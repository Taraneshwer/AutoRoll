# AUTOROLL ML PHASE 5.2 — REAL DATASET INGESTION PIPELINE & SYNTHETIC ELIMINATION REPORT

**Date**: 2026-08-19  
**Phase**: ML Phase 5.2 — Eliminate Synthetic Data & Install Real Ingestion Pipeline  
**Final Status**: `REAL TRAINING DATASET REQUIRED`  
**Operational Status**: **SYNTHETIC DATA ELIMINATED & QUARANTINED | PRODUCTION TRAINING GUARD ACTIVE | STOPPED BEFORE ARCFACE FINE-TUNING**

---

## Executive Summary

AutoRoll ML Phase 5.2 has successfully eliminated synthetic data from the training path, quarantined the Phase 5 synthetic dataset, installed a production real dataset ingestion engine (`scripts/ingest_real_dataset.py`), updated dataset validation (`scripts/validate_training_dataset.py`), and enforced strict runtime training guards (`verify_real_dataset_guard`) in `autoroll/ml/training/trainer.py`.

Because a legitimate 2,000–2,500 identity real human face dataset is not currently available on the local filesystem, no synthetic replacement data was fabricated. All training entry points will reject training until a genuine real public dataset is provided via `ingest_real_dataset.py`.

---

## 1. Synthetic Data Removal & Quarantine Actions

- **Deprecation of Generator Script**: `scripts/build_large_face_dataset.py` has been relocated to `scripts/legacy/synthetic/build_large_face_dataset.py` and marked `[DEPRECATED — NOT FOR TRAINING]`. The original entry point has been replaced with a stub that raises an explicit `RuntimeError` if invoked.
- **Quarantine Location**: The 110,000-image synthetic dataset previously located under `data/face_recognition/` has been moved to:  
  `data/quarantine/synthetic_phase5/` (containing `aligned/`, `raw/`, `splits/`, and `splits_dev/`).
- **Clean Root Structure**: `data/face_recognition/` has been reset as a clean root containing `raw/`, `aligned/`, `splits/`, and `metadata/`.

---

## 2. Ingestion Pipeline & Authenticity Architecture

### Real Dataset Ingestion Engine (`scripts/ingest_real_dataset.py`)
- **CLI Usage**:
  ```bash
  python scripts/ingest_real_dataset.py --source /path/to/real/dataset --dataset-name <name> [--max-images-per-id N]
  ```
- **Authenticity Audit**: Inspects color variance across sample images (`>800` unique colors per 112x112 chip required). Rejects synthetic geometric drawings or constant gray backgrounds automatically.
- **Duplicate Audit**: Computes SHA256 hashes for exact file duplicates and Difference Hashing (**dHash**) for near-duplicate detection before split generation.
- **SCRFD Inference & Alignment**: Executes real `SCRFD_10G_KPS` detector (with CUDA acceleration on RTX 5060) and 5-point Umeyama similarity transform alignment (112x112 RGB chips). Pre-computed synthetic landmarks are prohibited.
- **Quality Filtering**: Applies confidence ($\ge 0.50$), face size ($\ge 32\times 32$), blur ($\text{Laplacian Var} \ge 15.0$), and brightness bounds ($[20, 240]$).
- **Identity-Disjoint Splitting**: Splits valid identities into Train (80%), Validation (10%), and Test (10%) with zero identity overlap.

---

## 3. Strict Training Guard Enforcement (`autoroll/ml/training/trainer.py`)

The `ArcFacePilotTrainer` class in `autoroll/ml/training/trainer.py` now enforces `verify_real_dataset_guard()` prior to initialization.

```python
def verify_real_dataset_guard(manifest_path="data/face_recognition/metadata/source_manifest.json"):
    if not os.path.exists(manifest_path):
        raise RuntimeError("REAL TRAINING DATASET GUARD FAILED: Source manifest missing.")
    # Checks: synthetic == False, dataset_type == "real", source_url & license exist, scale requirements
```

---

## 4. Ingestion Report Breakdown (A–N)

| Section | Metric / Requirement | Current Status / Result |
| :--- | :--- | :--- |
| **A. Real Dataset Name** | Ingested Dataset Name | `None` (Awaiting local ingestion) |
| **B. Exact Source** | Local Source Path | `None` (Local dataset directory not specified) |
| **C. License** | License Identifier | `None` (To be recorded from source_manifest) |
| **D. Original Identity Count** | Source Identity Count | `0` |
| **E. Original Image Count** | Source Image Count | `0` |
| **F. Selected Identity Count**| Ingested Identities | `0` |
| **G. Selected Image Count**   | Ingested Images | `0` |
| **H. SCRFD Detections** | Successful Detections | `0` |
| **I. Rejection Statistics** | Rejection Breakdown | `NO_FACE: 0, LOW_CONF: 0, BLUR: 0` |
| **J. Duplicate Statistics** | Purged Duplicates | `Exact: 0, Near-Duplicates: 0` |
| **K. Train / Val / Test** | Partition Counts | `Train: 0 ID, Val: 0 ID, Test: 0 ID` |
| **L. Provenance Verification**| `source_manifest.json` | Missing (`REAL TRAINING DATASET REQUIRED`) |
| **M. GPU Performance** | RTX 5060 Benchmark | CUDA Provider Active (`0.0 img/s`) |
| **N. Final Approval Status** | Compliance Verdict | **REAL TRAINING DATASET REQUIRED** |

---

## 5. Automated Validation & Test Suite Results

1. **Validation Script (`python scripts/validate_training_dataset.py`)**:
   ```
   ================================================================================
              AUTOROLL REAL TRAINING DATASET VALIDATION REPORT           
   ================================================================================
   Validation Status : FAILED (source_manifest.json missing)
   Reason            : REAL TRAINING DATASET REQUIRED
   ================================================================================
   ```
   *Confirmed: Fails cleanly as expected when source_manifest.json is absent.*

2. **PyTest Suite (`pytest`)**:
   - Executed **82 unit & integration tests** (including `tests/test_dataset_authenticity_guard.py`).
   - **Result**: **`82 passed in 123.61s (100% Pass Rate)`**.

---

## 6. Benchmark & Directory Isolation Confirmations

- **LFW Benchmark**: Preserved exclusively under `data/benchmarks/lfw/`.
- **Local Student Data**: Preserved exclusively under `data/local_students/`.

---

## 7. FINAL VERDICT & OPERATIONAL STOP

```
================================================================================
                        REAL TRAINING DATASET REQUIRED
================================================================================
Synthetic dataset eliminated and quarantined.
Ingestion pipeline and strict training guard active.
To begin ArcFace training, provide a local real face dataset and run:

python scripts/ingest_real_dataset.py \
    --source /path/to/real/dataset \
    --dataset-name <Name>
================================================================================
```
