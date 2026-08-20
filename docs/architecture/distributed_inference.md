# Distributed Multi-Camera Inference Architecture

## 1. Executive Overview

AutoRoll Phase 10 introduces a Control-Plane / Worker architecture for scaling real-time face recognition attendance across campus environments.

```
                    AutoRoll Control Server (Central Backend)
                           │
                 Load-Aware Balancer & Scheduler
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Worker 01        Worker 02        Worker 03
    (Webcam/RTSP)    (Webcam/RTSP)    (Webcam/RTSP)
          │                │                │
   [Local Inference] [Local Inference] [Local Inference]
          │                │                │
          └────────────────┼────────────────┘
                           ▼
              Sanitized Recognition Events (WS)
                           │
                           ▼
                Central Attendance Service
             (Model Validation & Deduplication)
                           │
                           ▼
                    Database Logging
```

---

## 2. Security & Privacy Protocol

- **Secret-Based Authentication:** Workers authenticate via `AUTOROLL_WORKER_SECRET`.
- **Zero Raw Embeddings over Network:** Workers extract embeddings locally, perform template matching, and send only sanitized event metadata over WebSocket.
- **Zero Image Storage:** Video frames remain in volatile worker RAM buffers.

---

## 3. Load Balancing & Failover Recovery

- **Deterministic Load Score Formula:**
  $$\text{Load Score} = (\text{Active Cameras} \times 2.0) + \text{Queue Depth} + \left(\frac{\text{Latency (ms)}}{10.0}\right)$$
- **Automatic Disconnect Detection:** Heartbeats checked every 5 seconds. If a worker misses heartbeats for > 15 seconds, it is marked `OFFLINE` and affected cameras are reassigned immediately.
