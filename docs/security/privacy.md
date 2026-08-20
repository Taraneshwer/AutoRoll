# Privacy & Security Protocol — AutoRoll Phase 11

## 1. Zero Image Storage Policy

- **No Frame Persistence:** Camera video streams are processed in volatile RAM buffers. No video frames, raw enrollment photographs, screenshots, or face crops are saved to disk or database tables.
- **Volatile Processing:** Once an embedding vector is extracted, the source image frame is immediately discarded from memory.

---

## 2. Protected Embedding Telemetry

- **No Embeddings in Public APIs:** API endpoints (`/api/v1/students`, `/api/v1/attendance`, `/ws/events`) exclude 512-dim embedding vectors.
- **Encrypted Binary Storage:** Templates are stored as binary byte blobs in database tables accessible only to internal services.

---

## 3. Worker Authentication & Secret Token

- Standalone recognition workers authenticate using secret token (`AUTOROLL_WORKER_SECRET`).
- Workers send identity matching metadata over TLS/WebSocket without sending raw embeddings over the network.
