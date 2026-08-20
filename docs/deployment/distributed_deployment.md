# Distributed Deployment Guide

## 1. Environment Configurations

AutoRoll supports two deployment modes:

### Single Machine Mode
```bash
docker-compose -f deployment/docker/docker-compose.single.yml up -d
```

### Control Server Mode
```bash
docker-compose -f deployment/docker/docker-compose.server.yml up -d
```

### Distributed Worker Node Mode
```bash
export AUTOROLL_SERVER_URL="http://central-server-ip:8000"
export AUTOROLL_WORKER_ID="worker-node-building-a"
docker-compose -f deployment/docker/docker-compose.worker.yml up -d
```

---

## 2. Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AUTOROLL_SERVER_URL` | Central Control Plane URL | `http://localhost:8000` |
| `AUTOROLL_WORKER_SECRET` | Secret token for worker authentication | `autoroll_secret_2026` |
| `AUTOROLL_WORKER_ID` | Unique worker node identifier | `worker-node-01` |
| `AUTOROLL_RECOGNITION_MODEL` | Enforced recognition model ID | `autoroll_v1` |
