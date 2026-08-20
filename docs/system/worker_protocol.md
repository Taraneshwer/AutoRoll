# Worker Protocol & Control Plane Reference — AutoRoll Phase 14

## 1. REST API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/workers` | List all registered worker nodes and their status |
| `GET` | `/api/v1/workers/{id}` | Retrieve detailed worker state and camera assignments |
| `GET` | `/api/v1/workers/{id}/metrics` | Fetch real-time GPU utilization, VRAM, and latency |
| `POST` | `/api/v1/workers/register` | Register new GPU worker node with auth token verification |
| `POST` | `/api/v1/workers/{id}/heartbeat` | Send 5-second periodic heartbeat signal |
| `POST` | `/api/v1/workers/{id}/drain` | Gracefully drain worker (reassign active streams) |
| `POST` | `/api/v1/workers/{id}/restart` | Unsupported control response (501 Not Implemented) |
| `POST` | `/api/v1/workers/{id}/assign-camera` | Manually bind RTSP camera stream to worker node |
| `POST` | `/api/v1/workers/{id}/remove-camera` | Remove camera stream assignment |
| `POST` | `/api/v1/cameras/{id}/reassign` | Reassign camera stream to optimal worker |

---

## 2. Event Protocol & WebSocket Payloads

All worker lifecycle and telemetry events inherit from `BaseWorkerEvent` Pydantic schema:

```typescript
interface BaseWorkerEvent {
  event_id: string;
  event_type: WorkerEventType;
  timestamp: number;
  worker_id: string;
  camera_id?: string;
  payload: Record<string, any>;
}
```

### Event Types:
- `WORKER_REGISTERED`, `WORKER_ONLINE`, `WORKER_HEARTBEAT`, `WORKER_DEGRADED`, `WORKER_OFFLINE`, `WORKER_DRAINING`, `WORKER_FAILOVER`
- `CAMERA_ASSIGNED`, `CAMERA_UNASSIGNED`, `CAMERA_ONLINE`, `CAMERA_OFFLINE`
- `FACE_RECOGNIZED`, `FACE_UNKNOWN`, `SPOOF_DETECTED`, `LOW_QUALITY`, `ATTENDANCE_CONFIRMED`
