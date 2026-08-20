# AutoRoll Architectural Overview

AutoRoll is a privacy-preserving, AI-powered face recognition attendance system designed for real-time edge processing and dynamic multi-machine scaling.

## Core Pillars

1. **Edge ML Processing**: Heavy RTSP camera video decoding, SCRFD face detection, MiniFASNet presentation attack detection (PAD), and ArcFace feature embedding occur locally on standalone Worker processes.
2. **Central Control Backend**: A FastAPI central server orchestrates dynamic camera assignment, worker health monitoring, database persistence, and WebSocket client broadcasts.
3. **Privacy First**: Raw face images are never permanently saved by default. Enrolled faces are immediately transformed into L2-normalized 512-dimensional vector embeddings tagged with `model_version`.
