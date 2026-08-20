# Real-Time Camera Pipeline & Frontend Integration Specification

## 1. System Architecture

The AutoRoll Phase 8 real-time camera ingestion and decoupled inference pipeline bridges hardware video feeds (laptop/USB webcams and network RTSP streams) to a React/Vite frontend dashboard via MJPEG video streaming and WebSocket telemetry.

```
Camera (Laptop/USB Webcam)
  │
  ├── LocalCameraSource (Threaded capture loop @ 30 FPS, bounded queue maxsize=2)
  │
  ├── Decoupled Inference Loop (@ 15 FPS)
  │     ├── SCRFD Face Detection
  │     ├── Face Quality Filter (Blur & Min Size)
  │     ├── 112×112 Umeyama Alignment
  │     ├── MiniFASNet Liveness (ML + Texture/FFT Moire)
  │     ├── ArcFace Embedding (512-dim + L2 Normalization)
  │     ├── Cosine Template Matching vs. Enrolled Students
  │     └── Temporal Confirmation (3 observations / 1500 ms)
  │
  ├── MJPEG Video Stream Generator (`GET /api/v1/camera/mjpeg`)
  │
  ├── WebSocket Telemetry Broadcast (`WS /ws/monitoring`)
  │
  └── AutoRoll React Frontend Dashboard
        ├── Dashboard (Overview, quick stats, active model)
        ├── Live Monitor (MJPEG video feed, bounding box overlay, identity pills, FPS, GPU, latency metrics)
        ├── Students (Student list, details, delete with modal)
        ├── Enrollment (6-step wizard, sample progress bar, rejection reasons, template summary)
        ├── Attendance (Date/student/status filters, today's attendance)
        ├── Analytics (Trends, similarity distributions, liveness rejections)
        ├── System (Backend/DB health, GPU VRAM, model switching, threshold metadata)
        └── Settings (Camera index, target FPS)
```

---

## 2. Decoupled Frame Processing Loop

- **Camera Capture Loop (`LocalCameraSource`):**
  Capture thread runs continuously at `AUTOROLL_CAMERA_FPS` (30 FPS). Incoming frames are placed into a bounded queue buffer (`maxsize=2`). If the buffer is full, the oldest frame is dropped immediately, eliminating frame backlog and preferring the newest frame.

- **ML Inference Loop (`CameraPipelineService`):**
  Inference loop processes frames at `AUTOROLL_INFERENCE_FPS` (~15 FPS). Empirical latency metrics are calculated independently per stage:
  - `capture_latency_ms`
  - `detection_latency_ms`
  - `alignment_latency_ms`
  - `liveness_latency_ms`
  - `recognition_latency_ms`
  - `matching_latency_ms`
  - `total_latency_ms`

---

## 3. Real-Time Telemetry Payload (`/ws/monitoring`)

WebSockets emit sanitized JSON telemetry without raw embeddings:

```json
{
  "timestamp": 1724155000.12,
  "pipeline_fps": 178.5,
  "camera_fps": 30.0,
  "total_latency_ms": 5.6,
  "capture_latency_ms": 0.1,
  "detection_latency_ms": 1.4,
  "alignment_latency_ms": 0.3,
  "liveness_latency_ms": 1.1,
  "recognition_latency_ms": 2.5,
  "matching_latency_ms": 0.1,
  "face_count": 1,
  "faces": [
    {
      "bbox": [120.0, 80.0, 240.0, 220.0],
      "detection_confidence": 0.985,
      "is_live": true,
      "liveness_score": 0.96,
      "student_id": "STU2001",
      "similarity": 0.745,
      "decision": "PRESENT"
    }
  ],
  "gpu_name": "NVIDIA RTX 5060 Laptop GPU",
  "vram_used_mb": 420.5,
  "active_model_id": "autoroll_v1",
  "recognition_threshold": 0.0540,
  "queue_depth": 1
}
```

---

## 4. Enrollment & Student Management

- **6-Step Enrollment Wizard:**
  1. Student Information input
  2. Camera readiness check
  3. Face target box positioning
  4. Multi-sample collection (5–10 samples) with progress indicator & explicit rejection badges (`no_face`, `multiple_faces`, `poor_quality`, `spoof_detected`).
  5. Mean template computation & L2 normalization.
  6. Success summary displaying Student ID, Name, Samples Used, Model ID, Embedding Dim (512), and Timestamp — **no raw embeddings shown!**

- **Student Directory:**
  Enrolled student cards, search filter, student details, model version, and deletion with confirmation modal.
