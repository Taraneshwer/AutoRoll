# Distributed GPU Worker Architecture — AutoRoll Phase 14

## 1. System Overview

AutoRoll Phase 14 introduces a clean distributed GPU worker architecture that decouples heavy ML video inference (SCRFD detection, 5-point alignment, MiniFASNet anti-spoofing, ArcFace embedding extraction, cosine vector matching) from the central web API control server.

```
                    React Frontend UI
                            │
                      REST / WebSockets
                            │
                    ┌───────v───────┐
                    │Central Control│
                    │ FastAPI Server│
                    └───────┬───────┘
                            │ REST / WS Protocol
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Worker 1     │   │ Worker 2     │   │ Worker N     │
  │ RTX 5060 GPU │   │ RTX 4090 GPU │   │ RTX GPU Node │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
      Camera A           Camera B           Camera N
```

---

## 2. Deployment Modes

1. **MODE A (Single Machine):** `AUTOROLL_WORKER_MODE=local` — Central server automatically initializes a default local GPU worker process (`local-worker-01`). Zero external setup required.
2. **MODE B (Central Server + 1 GPU Worker):** `AUTOROLL_WORKER_MODE=distributed` — Central control plane runs on primary server, single GPU worker process connects over network API.
3. **MODE C (Central Server + Multi-GPU Cluster):** Horizontal scaling with $N$ independent worker nodes. Cameras are dynamically assigned via load scores.

---

## 3. Load Balancing & Camera Scheduling

The central scheduler assigns cameras using a deterministic multi-factor load score formula:

$$\text{Load Score} = 0.35 \cdot \text{GPU} + 0.25 \cdot \text{Queue} + 0.20 \cdot \text{Cameras} + 0.20 \cdot \text{P95 Latency}$$

- **Camera Affinity:** Assigned cameras stay bound to their worker to avoid stream interruption.
- **Failover:** If worker heartbeat is missing for >15s, status transitions to `OFFLINE` and cameras migrate automatically to the lowest-load online worker.
