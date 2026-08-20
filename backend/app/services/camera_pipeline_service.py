"""
Decoupled Camera Inference Pipeline Service.
Runs real-time ML inference loop, calculates empirical telemetry metrics,
streams sanitized telemetry over WebSocket (/ws/monitoring), and provides MJPEG stream encoding.
"""

import asyncio
import struct
import time
from typing import Any, Generator

import cv2
import numpy as np
import torch

from app.api.websocket.manager import ws_manager
from app.camera.manager import camera_manager
from app.core.config import get_settings
from app.core.logger import get_logger
from app.database.models import StudentEmbedding
from app.database.session import SessionLocal
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.inference.decision import UnifiedDecisionEngine
from app.ml.liveness.passive_fas import PassiveLivenessDetector
from app.ml.preprocessing.quality import FaceQualityFilter
from app.ml.recognition.factory import get_recognizer

logger = get_logger("camera_pipeline_service")


def cos_sim(v1: list[float], v2: list[float]) -> float:
    a1 = np.array(v1, dtype=np.float32)
    a2 = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(a1)
    n2 = np.linalg.norm(a2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a1, a2) / (n1 * n2))


class CameraPipelineService:
    def __init__(self):
        self.settings = get_settings()
        self.detector = SCRFDDetector()
        self.aligner = FaceAligner()
        self.quality_filter = FaceQualityFilter()
        self.liveness_detector = PassiveLivenessDetector()
        self.decision_engine = UnifiedDecisionEngine()

        self.is_running = False
        self.loop_task: asyncio.Task | None = None
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_telemetry: dict[str, Any] = {}

    def start_pipeline(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        logger.info("Starting Decoupled Camera Inference Pipeline Service...")

    def stop_pipeline(self) -> None:
        self.is_running = False
        logger.info("Decoupled Camera Inference Pipeline Service stopped.")

    def process_latest_frame(self) -> dict[str, Any] | None:
        """
        Executes real-time face attendance pipeline on the newest frame from camera queue.
        Calculates empirical latency metrics per stage.
        """
        source = camera_manager.get_source()
        if source is None or not source.is_opened():
            return None

        cap_start = time.perf_counter()
        ret, frame = source.read_frame()
        capture_latency = (time.perf_counter() - cap_start) * 1000.0

        if not ret or frame is None:
            return None

        start_total = time.perf_counter()

        # Resolve Active Recognizer & Threshold
        recognizer = get_recognizer()
        active_model_id = recognizer.get_model_id()
        threshold = recognizer.get_recognition_threshold()

        # 1. SCRFD Face Detection
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        det_latency = (time.perf_counter() - t0) * 1000.0

        # Load enrolled templates from DB for active model
        with SessionLocal() as db:
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
                num_floats = len(t.embedding_vector) // 4
                vec = list(struct.unpack(f"{num_floats}f", t.embedding_vector))
                loaded_templates.append((t.student_id, vec, t.model_id))

        faces_output = []
        annotated_frame = frame.copy()

        align_latencies = []
        liveness_latencies = []
        rec_latencies = []
        match_latencies = []

        for track_id, det in enumerate(detections, start=1):
            # 2. Quality Check
            quality_res = self.quality_filter.filter_quality(frame, det.bbox, det.landmarks)

            # 3. Alignment
            t_align = time.perf_counter()
            aligned_face = self.aligner.align_face(frame, det.landmarks)
            align_latencies.append((time.perf_counter() - t_align) * 1000.0)

            # 4. MiniFASNet Liveness
            t_live = time.perf_counter()
            liveness_res = self.liveness_detector.predict(frame, det.bbox)
            liveness_latencies.append((time.perf_counter() - t_live) * 1000.0)

            # 5. ArcFace Feature Extraction (512-dim embedding)
            t_rec = time.perf_counter()
            rec_res = recognizer.extract_embedding(aligned_face)
            emb = rec_res.embedding
            rec_latencies.append((time.perf_counter() - t_rec) * 1000.0)

            # 6. Template Cosine Matching
            t_match = time.perf_counter()
            best_student_id = None
            best_sim = -1.0
            matched_tmpl_model_id = None

            for student_id, tmpl_vec, tmpl_model_id in loaded_templates:
                sim = cos_sim(emb, tmpl_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_student_id = student_id
                    matched_tmpl_model_id = tmpl_model_id
            match_latencies.append((time.perf_counter() - t_match) * 1000.0)

            # 7. Temporal Confirmation & Attendance Decision
            decision = self.decision_engine.evaluate_attendance_decision(
                track_id=track_id,
                detection_confidence=det.det_confidence,
                is_quality_ok=quality_res.is_acceptable,
                is_live=liveness_res.is_live,
                liveness_score=liveness_res.combined_liveness_score,
                student_id=best_student_id,
                similarity_score=best_sim if best_sim >= 0 else 0.0,
                recognition_threshold=threshold,
                model_id=active_model_id,
                template_model_id=matched_tmpl_model_id,
                timestamp=start_total,
            )

            # Render Bounding Box and Decision Label on Annotated Frame
            x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
            color = (0, 255, 0) if decision == "PRESENT" else (0, 0, 255) if "SPOOF" in decision else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{decision} | Sim:{best_sim:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            faces_output.append(
                {
                    "bbox": det.bbox.to_list(),
                    "detection_confidence": round(det.det_confidence, 4),
                    "is_live": liveness_res.is_live,
                    "liveness_score": round(liveness_res.combined_liveness_score, 4),
                    "student_id": best_student_id if decision == "PRESENT" else None,
                    "similarity": round(best_sim, 4) if best_sim >= 0 else 0.0,
                    "decision": decision,
                }
            )

        total_latency = (time.perf_counter() - start_total) * 1000.0
        pipeline_fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        # GPU Utilization Telemetry
        vram_mb = 0.0
        gpu_name = "N/A"
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            gpu_name = torch.cuda.get_device_name(0)

        camera_metrics = source.get_metrics()

        telemetry = {
            "timestamp": time.time(),
            "pipeline_fps": round(pipeline_fps, 1),
            "camera_fps": camera_metrics.get("capture_fps", 0.0),
            "total_latency_ms": round(total_latency, 2),
            "capture_latency_ms": round(capture_latency, 2),
            "detection_latency_ms": round(det_latency, 2),
            "alignment_latency_ms": round(sum(align_latencies) / len(align_latencies) if align_latencies else 0.0, 2),
            "liveness_latency_ms": round(sum(liveness_latencies) / len(liveness_latencies) if liveness_latencies else 0.0, 2),
            "recognition_latency_ms": round(sum(rec_latencies) / len(rec_latencies) if rec_latencies else 0.0, 2),
            "matching_latency_ms": round(sum(match_latencies) / len(match_latencies) if match_latencies else 0.0, 2),
            "face_count": len(detections),
            "faces": faces_output,
            "gpu_name": gpu_name,
            "vram_used_mb": vram_mb,
            "active_model_id": active_model_id,
            "recognition_threshold": threshold,
            "queue_depth": camera_metrics.get("queue_depth", 0),
        }

        self.latest_annotated_frame = annotated_frame
        self.latest_telemetry = telemetry

        # Broadcast sanitized telemetry over WebSocket
        ws_manager.broadcast_sync("telemetry_update", telemetry)

        return telemetry

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yields multipart MJPEG JPEG frames for HTML <img> streaming in frontend.
        """
        while True:
            frame = self.latest_annotated_frame
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            time.sleep(0.03)  # ~30 FPS stream rate


camera_pipeline_service = CameraPipelineService()
