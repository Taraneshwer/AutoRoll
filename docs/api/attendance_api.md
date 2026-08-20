# Attendance & System API Reference — AutoRoll Phase 11

## 1. Student Management & Enrollment Endpoints

- `GET /api/v1/students`: List enrolled students.
- `POST /api/v1/students`: Create new student.
- `GET /api/v1/students/{student_id}`: Retrieve student profile.
- `DELETE /api/v1/students/{student_id}`: Delete student profile and associated face template.
- `POST /api/v1/students/{student_id}/enrollment/start`: Start multi-sample enrollment session.
- `POST /api/v1/students/{student_id}/enrollment/sample`: Ingest camera frame chip.
- `POST /api/v1/students/{student_id}/enrollment/complete`: Aggregate 5–10 samples into normalized mean template.

---

## 2. Attendance & Verification Endpoints

- `GET /api/v1/attendance`: Retrieve attendance logs.
- `GET /api/v1/attendance/today`: Retrieve today's attendance events.
- `GET /api/v1/attendance/{student_id}`: Retrieve student attendance history.

---

## 3. Health & Observability Endpoints

- `GET /health` / `GET /ready` / `GET /api/v1/system/health`: Backend, DB, GPU, active model, worker count, and camera count telemetry.
- `GET /metrics` / `GET /api/v1/system/metrics`: Model thresholds, embedding dimension, confirmation frame window, and cooldown metrics.

---

## 4. Real-Time WebSockets

- `GET /ws/events`: Broadcasts system events (`ATTENDANCE_CONFIRMED`, `SPOOF_DETECTED`, `WORKER_ONLINE`, etc.).
- `GET /ws/monitoring`: Low-latency video frame metadata and detection bounding box overlays.
