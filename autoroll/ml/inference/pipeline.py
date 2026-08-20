"""
AutoRoll Unified Multi-Face Real-Time Inference Pipeline.
Coordinates: SCRFD Detection -> Multi-Face Tracking -> Face Alignment ->
ArcFace Recognition & Liveness Anti-Spoofing -> Decision Engine.
"""

import time

import numpy as np

from autoroll.common.logger import get_logger
from autoroll.common.schemas import FaceLandmarks
from autoroll.ml.detectors.aligner import FaceAligner
from autoroll.ml.detectors.scrfd import SCRFDDetector
from autoroll.ml.inference.decision import UnifiedDecisionEngine
from autoroll.ml.inference.performance import PerformanceTracker
from autoroll.ml.inference.result import TrackedFaceResult, UnifiedFrameResult
from autoroll.ml.inference.tracker import MultiFaceTracker
from autoroll.ml.liveness.pipeline import LivenessPipeline
from autoroll.ml.recognition.arcface_iresnet import ArcFaceRecognizer

logger = get_logger("unified_inference_pipeline")


class UnifiedInferencePipeline:
    """
    Unified ML Inference Pipeline processing multi-face frames end-to-end.
    """

    def __init__(
        self,
        detector: SCRFDDetector | None = None,
        aligner: FaceAligner | None = None,
        recognizer: ArcFaceRecognizer | None = None,
        liveness: LivenessPipeline | None = None,
        device: str = "auto",
        recognition_interval: int = 10,
        liveness_threshold: float = 0.90,
    ):
        self.device = device
        self.recognition_interval = recognition_interval
        self.liveness_threshold = liveness_threshold

        # Initialize ML Subsystems
        self.detector = detector or SCRFDDetector(device=device)
        self.aligner = aligner or FaceAligner()
        self.recognizer = recognizer or ArcFaceRecognizer(device=device)
        self.liveness = liveness or LivenessPipeline(
            device=device, liveness_threshold=liveness_threshold
        )

        # Multi-Face Tracker, Decision Engine, and Performance Tracker
        self.tracker = MultiFaceTracker(
            iou_threshold=0.30,
            max_disappeared=5,
            recognition_interval=recognition_interval,
        )
        self.decision_engine = UnifiedDecisionEngine()
        self.perf_tracker = PerformanceTracker(window_size=30)

        logger.info(
            f"Unified Inference Pipeline initialized on device '{self.device}' "
            f"(Interval: every {recognition_interval} frames, "
            f"Liveness Threshold: {liveness_threshold})"
        )

    def process_frame(
        self, frame: np.ndarray, frame_index: int = 0
    ) -> UnifiedFrameResult:
        """
        Processes a single BGR video frame containing 0, 1, or multiple faces.
        Returns UnifiedFrameResult with latency and tracking details.
        """
        total_start = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid or empty image frame provided for inference.")

        # 1. SCRFD Face Detection
        det_start = time.perf_counter()
        detections = self.detector.detect(frame)
        det_latency = (time.perf_counter() - det_start) * 1000.0

        # 2. Multi-Face Tracking Association
        det_tuples = [(d.bbox, d.landmarks.points, d.det_confidence) for d in detections]
        updated_tracks = self.tracker.update(det_tuples, frame_index)

        # 3. Process Each Tracked Face (Recognition & Liveness)
        rec_latency_accum = 0.0
        live_latency_accum = 0.0
        face_results: list[TrackedFaceResult] = []

        for track, trigger_rec in updated_tracks:
            # Crop & Align Face Chip (112x112)
            try:
                lms_obj = (
                    FaceLandmarks(points=track.landmarks)
                    if isinstance(track.landmarks, list)
                    else track.landmarks
                )
                aligned_chip = self.aligner.align(frame, lms_obj)
            except Exception as e:
                logger.warning(
                    f"Failed to align face for track {track.track_id}: {e}. "
                    "Falling back to bbox crop."
                )
                b = track.bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
                x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
                crop = frame[y1:y2, x1:x2]
                aligned_chip = (
                    self.aligner.resize(crop, (112, 112))
                    if crop.size > 0
                    else np.zeros((112, 112, 3), dtype=np.uint8)
                )

            # A. Anti-Spoofing Liveness Evaluation
            live_start = time.perf_counter()
            try:
                live_res = self.liveness.predict(aligned_chip)
                is_live = live_res.is_live
                live_score = live_res.liveness_score
                live_dec = live_res.details.get("decision", "SPOOF")
            except Exception as e:
                logger.error(f"Liveness error on track {track.track_id}: {e}")
                is_live = False
                live_score = 0.0
                live_dec = "ERROR"
            live_latency_accum += (time.perf_counter() - live_start) * 1000.0

            # B. Selective ArcFace Recognition Embedding Extraction
            embedding = None
            rec_error = None
            if trigger_rec and is_live:
                rec_start = time.perf_counter()
                try:
                    rec_res = self.recognizer.extract_embedding(aligned_chip)
                    embedding = rec_res.embedding
                    track.last_recognition_frame = frame_index
                except Exception as e:
                    logger.error(f"Recognition error on track {track.track_id}: {e}")
                    rec_error = str(e)
                rec_latency_accum += (time.perf_counter() - rec_start) * 1000.0

            # C. Decision Engine Synthesis
            face_res = self.decision_engine.evaluate_track_decision(
                track=track,
                is_live=is_live,
                liveness_score=live_score,
                liveness_decision=live_dec,
                embedding=embedding,
                recognition_error=rec_error,
            )
            face_results.append(face_res)

        total_latency = (time.perf_counter() - total_start) * 1000.0
        fps = self.perf_tracker.record_frame(
            det_latency=det_latency,
            rec_latency=rec_latency_accum,
            live_latency=live_latency_accum,
            total_latency=total_latency,
        )

        num_live = sum(1 for f in face_results if f.is_live)

        return UnifiedFrameResult(
            frame_index=frame_index,
            timestamp_ms=round(time.time() * 1000.0, 2),
            num_faces_detected=len(face_results),
            num_faces_live=num_live,
            faces=face_results,
            detection_latency_ms=round(det_latency, 2),
            recognition_latency_ms=round(rec_latency_accum, 2),
            liveness_latency_ms=round(live_latency_accum, 2),
            total_latency_ms=round(total_latency, 2),
            fps=fps,
        )
