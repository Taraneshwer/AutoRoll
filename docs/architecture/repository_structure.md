# AutoRoll Repository Structure Documentation

## Overview

The AutoRoll codebase is structured as a production-grade, microservice-ready repository. All Python application core logic, database ORM models, API routes, and machine learning components reside in `backend/app/`, while operational scripts, tests, deployment assets, datasets, model checkpoints, and configuration files are organized in dedicated root-level folders.

---

## Directory Architecture Diagram

```
AutoRoll/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/              # FastAPI REST endpoints
│   │   │   └── websocket/           # Real-time WebSocket monitoring
│   │   ├── core/                    # System configuration, logging, security, crypto
│   │   ├── database/                # SQLAlchemy database models & repositories
│   │   ├── ml/                      # Machine Learning core modules
│   │   │   ├── alignment/           # Face alignment pipelines
│   │   │   ├── detectors/           # SCRFD face detection models
│   │   │   ├── enrollment/          # Multi-view enrollment aggregation
│   │   │   ├── evaluation/          # Verification metrics (EER, AUC, TAR@FAR)
│   │   │   ├── inference/           # Real-time tracking & decision engine
│   │   │   ├── liveness/            # Anti-spoofing & FAS liveness models
│   │   │   ├── preprocessing/       # Quality filtering & chip generation
│   │   │   ├── recognition/         # ArcFace IResNet50 feature extraction
│   │   │   ├── tracking/            # Multi-face SORT tracking
│   │   │   └── training/            # PyTorch ArcFace training pipeline
│   │   ├── schemas/                 # Pydantic data schemas
│   │   ├── services/                # Business logic (attendance, auth, students, scheduler)
│   │   ├── workers/                 # RTSP streaming worker node service
│   │   └── main.py                  # FastAPI application entry point
│   ├── scripts/                     # Operational execution scripts
│   │   ├── dataset/                 # Ingestion & dataset validation
│   │   ├── evaluation/              # Verification protocol evaluation scripts
│   │   ├── maintenance/             # Benchmark, profiling, & test scripts
│   │   └── training/                # PyTorch training & fine-tuning scripts
│   ├── tests/                       # Complete pytest suite (83 tests)
│   ├── requirements.txt             # Backend dependencies
│   ├── pyproject.toml               # Python package & pytest configuration
│   └── .env.example                 # Environment configuration template
├── frontend/                        # Web dashboard user interface
├── models/                          # Machine Learning weight binaries
│   ├── pretrained/                  # Pretrained ONNX models (ArcFace R50, SCRFD)
│   └── trained/                     # Fine-tuned PyTorch checkpoints (autoroll_arcface_v1)
├── data/                            # Datasets & benchmark evaluation splits
│   ├── face_recognition/            # CASIA-WebFace raw/detected/aligned chips
│   ├── autoroll_benchmark/          # Evaluation probes & enrollment sets
│   ├── local_students/              # Student face enrollment data
│   └── quarantine/                  # Low-quality image quarantine
├── deployment/                      # Deployment configuration assets
│   ├── docker/                      # Dockerfiles for server, worker, and frontend
│   ├── server/                      # Server docker-compose configurations
│   ├── distributed/                 # Distributed worker node docker-compose
│   ├── single/                      # Single-node monolithic docker-compose
│   └── nginx/                       # Nginx reverse proxy configuration
├── docs/                            # Documentation
│   ├── architecture/                # System architecture documentation
│   ├── api/                         # REST & WebSocket API specification
│   ├── ml/                          # Machine Learning design docs
│   ├── deployment/                  # Deployment & setup guides
│   └── research/                    # Privacy & system evaluation whitepapers
├── experiments/                     # Machine Learning experiment logs & configs
│   ├── baselines/                   # Pretrained baseline evaluation records
│   ├── finetuning/                  # Fine-tuning hyperparameter experiments
│   ├── benchmarks/                  # System scaling benchmark logs
│   └── ablation/                    # Loss function & margin ablation studies
├── reports/                         # Audit reports & benchmark evaluations
│   ├── ml/                          # Model validation & pre-flight audit reports
│   ├── benchmarks/                  # Scaling & throughput benchmark reports
│   ├── training/                    # Fine-tuning epoch evaluation reports
│   └── system/                      # Repository restructure reports
└── configs/                         # Environment & experiment configuration YAMLs
    ├── development/                 # Local dev threshold & dataset configs
    ├── production/                  # Production deployment configurations
    └── experiments/                 # Hyperparameter training YAMLs
```

---

## Python Import Conventions

When developing or executing code within the `backend/` subtree:
- Package root is `app.*` (e.g., `from app.core.config import settings`).
- All database models are imported via `from app.database.models import Student, AttendanceRecord`.
- ML feature extractors are imported via `from app.ml.recognition.iresnet_torch import MXNetIResNet50`.
- Execution command: `python -m app.main` from inside `backend/` or `python -m backend.app.main` from workspace root.
