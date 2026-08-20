# AutoRoll Production Deployment Guide

This guide details deployment options for **AutoRoll**: Single-Machine Docker Compose, Control-Plane Server, and Distributed Edge ML Workers across a Local Area Network (LAN).

---

## 1. Environment Configuration

Copy `.env.example` to `.env` and update values prior to deployment:

```bash
cp .env.example .env
```

| Environment Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `POSTGRES_USER` | PostgreSQL Database User | `autoroll` |
| `POSTGRES_PASSWORD` | PostgreSQL Database Password | `autoroll_secret` |
| `POSTGRES_DB` | PostgreSQL Database Name | `autoroll_db` |
| `JWT_SECRET_KEY` | Secret Key for Auth Tokens | `<random_hex_64>` |
| `AUTOROLL_SERVER_URL` | Server URL for Workers | `http://192.168.1.100:8000` |
| `AUTOROLL_WORKER_ID` | Unique ML Worker Node ID | `worker-node-01` |
| `AUTOROLL_DEVICE` | Compute device (`auto` / `cpu` / `cuda`) | `auto` |
| `AUTOROLL_MODEL_PATH` | Path to ArcFace model weights | `models/checkpoints/arcface.pth` |
| `AUTOROLL_CAMERA_CONFIG`| Path to worker camera streams config | `configs/camera_config.yaml` |

---

## 2. Deployment Topology Options

### Option A: Single Machine Deployment (Server + Postgres + Worker + Frontend)

Use `docker-compose.single.yml` when all control plane services and ML worker inference run on a single workstation or server.

```bash
docker-compose -f docker-compose.single.yml up -d --build
```

- **Frontend Web UI**: `http://localhost:80`
- **FastAPI Control Plane**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432`

---

### Option B: Central Control Plane Server Deployment (Server + Postgres + Frontend)

Use `docker-compose.server.yml` to deploy the central server on a master node or cloud host without local ML workers.

```bash
docker-compose -f docker-compose.server.yml up -d --build
```

---

### Option C: Distributed Edge ML Worker Node Deployment

Deploy independent ML worker containers on edge machines (e.g., Jetson, GPU workstations) located near RTSP cameras across the local network.

#### 1. CPU Worker Deployment:
```bash
AUTOROLL_SERVER_URL="http://192.168.1.100:8000" \
AUTOROLL_WORKER_ID="worker-edge-01" \
AUTOROLL_DEVICE="cpu" \
docker-compose -f docker-compose.worker.yml up -d --build
```

#### 2. GPU Worker Deployment (NVIDIA Container Toolkit):
Ensure `nvidia-container-toolkit` is installed on the worker host, then uncomment the GPU device section in `docker-compose.worker.yml`:

```bash
AUTOROLL_SERVER_URL="http://192.168.1.100:8000" \
AUTOROLL_WORKER_ID="worker-gpu-01" \
AUTOROLL_DEVICE="cuda" \
docker-compose -f docker-compose.worker.yml up -d --build
```

---

## 3. Health Monitoring & Graceful Shutdown

- **Liveness Probes**:
  - `GET /health` (Server Liveness)
  - `GET /ready` (Database Connectivity Readiness)
- **Graceful Shutdown**:
  - All services respond to `SIGTERM` / `SIGINT` signals. Workers unassign active RTSP camera sessions and send heartbeat shutdown signals before exiting cleanly.
