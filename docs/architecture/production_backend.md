# Production Backend Architecture — AutoRoll Phase 11

## 1. Overview

The AutoRoll Production Backend combines local and distributed ML recognition, vector similarity search, temporal identity confirmation, attendance record persistence, and real-time event broadcasting.

```
                    Camera Streams (Webcam / RTSP)
                                │
                    Worker Node / Pipeline Service
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
            SCRFD Detector             MiniFASNet Liveness
            (10G_KPS)                  (Passive PAD)
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                       5-Point Alignment
                                │
                                ▼
                       ArcFace R50 Model
                       (512-dim Embedding)
                                │
                                ▼
                        FaceMatcher Engine
                    (Cosine Search vs DB Templates)
                                │
                                ▼
                  Temporal Confirmation (3 Frames)
                                │
                                ▼
                 Attendance Cooldown Guard (30 Sec)
                                │
                                ▼
                  SQLite/PostgreSQL DB Logging &
                  WebSocket Broadcast (/ws/events)
```

---

## 2. Component Details

- **Database Layer (`backend/app/database/models.py`):** Relational abstraction containing `Student`, `StudentEmbedding` (`FaceTemplate`), `AttendanceRecord`, `Camera`, `WorkerNode`, `AnalyticsEvent`, `AuditLog`.
- **Enrollment Service (`backend/app/services/enrollment_service.py`):** Captures 5–10 valid samples, filters non-live or poor-quality faces, and computes $\text{normalize}(\text{mean}(\text{embeddings}))$.
- **Vector Matching Engine (`backend/app/ml/matching/matcher.py`):** Performs nearest-neighbor cosine similarity matching against enrolled templates (`AUTOROLL_RECOGNITION_THRESHOLD=0.0540`).
- **Attendance & Cooldown Engine (`backend/app/services/attendance_service.py`):** Enforces 3-frame identity confirmation and 30-second deduplication cooldowns per student.
- **Event Bus (`backend/app/api/routes/event_bus.py`):** Broadcasts real-time events over `/ws/events`.
