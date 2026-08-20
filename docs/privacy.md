# AutoRoll Privacy & Data Protection Architecture

**AutoRoll** is designed around a **Privacy-by-Design** architecture. Biometric integrity, zero-retention raw image policies, and strict access controls are built directly into the system core.

---

## 1. What Data Is Stored vs. What Is NOT Stored

### ✅ What IS Stored:
1. **Student Metadata**: Student code, full name, department, active status, creation timestamp.
2. **512-Dimensional Feature Vectors**: L2-normalized ArcFace mathematical floating-point embedding centroids derived from enrollment samples.
3. **Attendance Logs**: Student ID, camera ID, worker ID, verification status (`CONFIRMED`, `SPOOF_ATTEMPT`, `UNKNOWN_PERSON`), similarity score, liveness score, timestamp.
4. **Analytics & Audit Event Logs**: Operation event logs for cluster security and operational health metrics.

### 🚫 What Is NOT Stored (Zero Retention Guarantee):
1. **NO Raw Facial Photographs**: Uploaded or captured enrollment photos are deleted immediately after feature extraction (`delete_raw_images=True`).
2. **NO Cropped Face Chips**: Bounding box cropped face chips generated during inference are processed exclusively in RAM memory and are never persisted to disk or database.
3. **NO Video Stream Recording**: Raw RTSP camera video frames are decoded in RAM on edge worker nodes and discarded immediately after inference.
4. **NO Reconstructable Biometric Templates**: ArcFace embeddings are non-invertible, 512-dimensional feature space mappings ($\mathbb{R}^{512}$). Raw face images cannot be reverse-engineered or reconstructed from feature embeddings.

---

## 2. Enrollment Data Privacy Flow

```
Raw Face Photos
    ↓ (In-Memory Processing)
SCRFD Detection & Quality Filter
    ↓
ArcFace 512-d Embedding Extraction
    ↓
L2-Normalized Centroid Aggregation
    ├─────────────────────────────────┐
    ↓                                 ↓
Store 512-d Vector             Purge Raw Image Files
(Database)                     (Disk Cleaned Immediately)
```

---

## 3. Data Retention & Retention Policies

| Data Category | Retention Period | Deletion Trigger / Policy |
| :--- | :--- | :--- |
| **Raw Enrollment Photos** | 0 Seconds (Immediate) | Purged from disk immediately after vector aggregation |
| **Face Bounding Chips** | 0 Seconds (In-Memory) | Released from RAM memory immediately after feature extraction |
| **Attendance Check-in Logs** | 365 Days (Configurable) | Automated database archiving or retention purging |
| **Analytics & Audit Logs** | 180 Days | Purged automatically via background scheduler |
| **512-d Vector Embeddings** | Active Enrollment Duration | Deleted immediately when a student profile is removed |

---

## 4. Model Versioning & Biometric Migration

- Every enrolled vector and verified check-in record stores `model_version` (e.g., `arcface_iresnet50_v1`).
- When a recognition model upgrade occurs (e.g., upgrading from `v1` to `v2`), existing `v1` embeddings are retained for backwards compatibility or students are re-enrolled using the privacy-preserving enrollment pipeline without retraining the underlying model architecture.

---

## 5. Security Access Control & RBAC

- **Admin Role**: Full control plane management (student enrollment, worker power control, camera RTSP configuration, system threshold adjustments, audit log inspection).
- **Operator Role**: Day-to-day operations (attendance view, student enrollment, camera status view).
- **Viewer Role**: Read-only telemetry access (dashboard metrics and live activity feeds).
- **Sanitized Logging**: All RTSP stream URLs containing embedded HTTP/RTSP basic authentication credentials (`rtsp://user:pass@host/live`) are automatically sanitized and redacted (`rtsp://***:***@host/live`) prior to log output.
