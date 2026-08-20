"""
Real-time Face Recognition API Route.
Processes an uploaded image frame through the production pipeline:
SCRFD Detection -> Quality -> Alignment -> Liveness -> ArcFace Embedding -> Template Cosine Match.
"""

import struct
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.models import StudentEmbedding
from app.database.session import get_db
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.liveness.passive_fas import PassiveLivenessDetector
from app.ml.preprocessing.quality import FaceQualityFilter
from app.ml.recognition.factory import get_recognizer

router = APIRouter(prefix="/recognition", tags=["Recognition"])

# Cache initialized pipeline models
_detector = SCRFDDetector()
_aligner = FaceAligner()
_quality_filter = FaceQualityFilter()
_liveness_detector = PassiveLivenessDetector()


def cos_sim(v1: list[float], v2: list[float]) -> float:
    a1 = np.array(v1, dtype=np.float32)
    a2 = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(a1)
    n2 = np.linalg.norm(a2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a1, a2) / (n1 * n2))


@router.post("/frame")
async def recognize_frame(
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """
    Performs real-time face detection, liveness check, feature extraction,
    and template comparison against enrolled student face templates in DB.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    recognizer = get_recognizer(model_id=model_id)
    active_model_id = recognizer.get_model_id()
    model_threshold = recognizer.get_recognition_threshold()

    # 1. Detection
    detections = _detector.detect(frame)
    if len(detections) == 0:
        return {
            "face_found": False,
            "results": [],
            "total_latency_ms": (time.perf_counter() - start_time) * 1000.0,
        }

    # Fetch stored primary face templates for active model
    templates_db = (
        db.query(StudentEmbedding)
        .filter(
            StudentEmbedding.is_primary == True,
            StudentEmbedding.model_id == active_model_id,
        )
        .all()
    )

    loaded_templates = []
    for t in templates_db:
        # Unpack serialized float32 embedding vector
        num_floats = len(t.embedding_vector) // 4
        vec = list(struct.unpack(f"{num_floats}f", t.embedding_vector))
        loaded_templates.append((t.student_id, vec, t.model_id))

    recognition_results = []
    for det in detections:
        # 2. Quality Check
        quality_res = _quality_filter.filter_quality(frame, det.bbox, det.landmarks)

        # 3. Liveness Check
        liveness_res = _liveness_detector.predict(frame, det.bbox)

        # 4. Alignment & ArcFace Feature Extraction
        aligned_face = _aligner.align_face(frame, det.landmarks)
        rec_res = recognizer.extract_embedding(aligned_face)
        emb = rec_res.embedding

        # 5. Template Matching
        best_student_id = None
        best_sim = -1.0

        for student_id, tmpl_vec, tmpl_model_id in loaded_templates:
            # Model Compatibility Guard
            if tmpl_model_id != active_model_id:
                continue
            sim = cos_sim(emb, tmpl_vec)
            if sim > best_sim:
                best_sim = sim
                best_student_id = student_id

        is_recognized = (
            best_sim >= model_threshold and liveness_res.is_live and quality_res.is_acceptable
        )

        recognition_results.append(
            {
                "bbox": det.bbox.to_list(),
                "face_confidence": round(det.det_confidence, 4),
                "is_quality_acceptable": quality_res.is_acceptable,
                "is_live": liveness_res.is_live,
                "liveness_score": round(liveness_res.combined_liveness_score, 4),
                "ml_liveness_score": round(liveness_res.ml_liveness_score, 4),
                "auxiliary_liveness_score": round(liveness_res.auxiliary_heuristic_score, 4),
                "matched_student_id": best_student_id if is_recognized else None,
                "similarity_score": round(best_sim, 4) if best_sim >= 0 else 0.0,
                "threshold_applied": model_threshold,
                "model_id": active_model_id,
                "model_version": recognizer.get_model_version(),
                "inference_latency_ms": round(rec_res.inference_latency_ms, 2),
            }
        )

    total_latency = (time.perf_counter() - start_time) * 1000.0

    return {
        "face_found": True,
        "face_count": len(detections),
        "results": recognition_results,
        "model_id": active_model_id,
        "total_latency_ms": round(total_latency, 2),
    }
