# AutoRoll 🚀

**Privacy-Preserving, AI-Powered Face Recognition Attendance System for Real-Time Edge & Distributed Scaling**

---

## 🌟 Overview

AutoRoll is a enterprise-grade attendance solution designed for horizontal scaling across edge GPU/CPU worker nodes and central server orchestration.

### Key Architecture Features
- **Privacy First**: Enrolled face images are converted into 512-dimensional vector embeddings and immediately purged. Zero raw image storage by default.
- **Edge RTSP Processing**: ML Workers connect directly to camera RTSP video feeds over LAN to minimize latency and prevent central server network bottlenecks.
- **Model Independence**: ArcFace IResNet50 feature extractor and SCRFD face detector wrapped inside a pure Python ML package (`autoroll.ml`), completely independent of web application logic.
- **Presentation Attack Detection (PAD)**: Built-in anti-spoofing module protecting against printed photos, mobile screen displays, and replayed video attacks.
- **Distributed Camera Scheduling**: Load-aware allocation and dynamic reassignment of camera streams on worker failures.

---

## 📁 Repository Structure

```
AutoRoll/
├── autoroll/              # Standalone Python ML & Common Package
│   ├── common/            # Shared schemas, DTOs, logger, & crypto math
│   └── ml/                # Face Detectors, Aligner, PAD, ArcFace Recognizer, & Pipeline
├── server/                # FastAPI Central Control Server
│   └── app/               # DB ORM models, REST API router, & WebSockets hub
├── worker/                # Standalone Edge ML Worker process
├── frontend/              # Modern React + Vite Dashboard
├── deploy/                # Docker & Docker Compose setup
├── docs/                  # Architecture, API specs, & Development guides
├── scripts/               # Benchmark suites & environment utilities
├── experiments/           # Training & anti-spoofing experiment trackers
└── tests/                 # Comprehensive Pytest test suite
```

---

## 🛠️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Configure environment
cp .env.example .env

# 3. Run Pytest test suite
pytest

# 4. Launch Central Server
python -m server.main

# 5. Launch Worker Node
python -m worker.main
```

---

## 🐳 Production Docker Deployment

See [docs/deployment.md](docs/deployment.md) for full deployment instructions.

### Single-Machine Full Stack (Server + Postgres + Worker + Frontend):
```bash
docker-compose -f docker-compose.single.yml up -d --build
```

### Central Control Plane Server (Server + Postgres + Frontend):
```bash
docker-compose -f docker-compose.server.yml up -d --build
```

### Independent Edge ML Worker:
```bash
AUTOROLL_SERVER_URL="http://192.168.1.100:8000" \
AUTOROLL_WORKER_ID="worker-node-01" \
AUTOROLL_DEVICE="auto" \
docker-compose -f docker-compose.worker.yml up -d --build
```

---

## 📄 License
MIT License - Copyright (c) 2026 AutoRoll Architecture Team.
