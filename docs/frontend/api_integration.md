# Frontend API & WebSocket Integration Reference — AutoRoll Phase 13

## 1. REST API Integration (`frontend/src/services/api.ts`)

The `ApiService` class abstracts all REST HTTP calls to `/api/v1`:

| API Call | Endpoint | Description |
| :--- | :--- | :--- |
| `getDashboardMetrics()` | `GET /api/v1/metrics/dashboard` | Cluster status, active cameras, FPS, latency, check-ins |
| `getStudents()` | `GET /api/v1/students` | Enrolled student roster |
| `createStudent()` | `POST /api/v1/students` | Register new student profile |
| `deleteStudent()` | `DELETE /api/v1/students/{id}` | Permanently delete student profile and biometric template |
| `getAttendance()` | `GET /api/v1/attendance` | Attendance log audit records |
| `getCameras()` | `GET /api/v1/cameras` | Configured RTSP camera sources |
| `createCamera()` | `POST /api/v1/cameras` | Register new RTSP video stream source |
| `assignCamera()` | `POST /api/v1/cameras/{id}/assign` | Assign worker cluster node to camera |
| `getWorkers()` | `GET /api/v1/workers` | Active ML worker nodes, GPU utilization, load scores |
| `checkHealth()` | `GET /health` | Backend and database readiness probe |

---

## 2. WebSocket Event & Telemetry Integration (`frontend/src/services/websocket.ts`)

- **`/ws/events`:** Real-time system events (`ATTENDANCE_CONFIRMED`, `SPOOF_DETECTED`, `WORKER_ONLINE`, `CAMERA_ONLINE`).
- **`/ws/monitoring`:** Low-latency video frame detection metadata, bounding box coordinates, face counts, pipeline FPS, and per-stage latency breakdown.
- **Auto-Reconnect & Failover:** WebSockets attempt auto-reconnect every 3 seconds if connection drops. Zero mock numbers or synthetic fallback streams are generated.
