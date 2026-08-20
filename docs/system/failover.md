# Automatic Worker Failover Protocol — AutoRoll Phase 14

## 1. Failover Lifecycle Workflow

```
[ Worker Heartbeat Stream ]
           │
     Heartbeat Lost
           │
   Elapsed > 10.0s ───> Transition to DEGRADED
           │
   Elapsed > 15.0s ───> Transition to OFFLINE
           │
  Central Scheduler Triggers Failover
           │
  Unassign Cameras from Offline Worker
           │
  Calculate Load Scores for Online Workers
           │
  Reassign Camera Streams & Broadcast WORKER_FAILOVER Event
```

---

## 2. Failover Event Schema

```json
{
  "event_id": "93a18e24-...",
  "event_type": "WORKER_FAILOVER",
  "timestamp": 1776700000.0,
  "camera_id": "cam-004",
  "old_worker_id": "gpu-worker-01",
  "new_worker_id": "gpu-worker-02",
  "reason": "Worker 'gpu-worker-01' heartbeat timeout (OFFLINE)",
  "failover_latency_ms": 2.10
}
```

---

## 3. Duplicate Assignment Prevention

The Central Worker Scheduler maintains an explicit camera-to-worker map (`self.camera_assignments`). Before binding a camera stream to a new worker node, it clears any previous assignment state to guarantee zero duplicate processing across nodes.
