"""
Abstract Base Pipeline and Integrated AutoRoll ML Baseline Pipeline.
Coordinates: Image -> Detector -> Aligner -> Recognizer -> 512-dim Embedding.
"""

import time
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.common import FrameProcessingResult, TrackedFace
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.recognition.arcface_iresnet import ArcFaceRecognizer

logger = get_logger("autoroll_pipeline")
settings = get_settings()


class BasePipeline(ABC):
    """
    Abstract Coordinator combining Detector, Tracker, Liveness PAD, and Recognizer.
    """

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, camera_id: str = "local", frame_number: int = 0
    ) -> FrameProcessingResult:
        """
        Processes single raw BGR video frame and yields detected, tracked, and verified faces.
        """
        pass


class AutoRollMLPipeline(BasePipeline):
    """
    Standalone AutoRoll ML Baseline Pipeline.
    """

    def __init__(
        self,
        scrfd_model_path: str | None = None,
        arcface_model_path: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.5,
    ):
        scrfd_path = scrfd_model_path or settings.SCRFD_MODEL_PATH
        arcface_path = arcface_model_path or settings.ARCFACE_MODEL_PATH

        self.detector = SCRFDDetector(
            model_path=scrfd_path, device=device, conf_threshold=conf_threshold
        )
        self.aligner = FaceAligner(target_size=(112, 112))
        self.recognizer = ArcFaceRecognizer(
            model_path=arcface_path, device=device, model_version=settings.MODEL_VERSION
        )
        self.device = self.detector.device

        logger.info(
            f"AutoRoll ML Pipeline initialized on device '{self.device}' "
            f"(Model Version: {settings.MODEL_VERSION})"
        )

    def process_frame(
        self, frame: np.ndarray, camera_id: str = "local", frame_number: int = 0
    ) -> FrameProcessingResult:
        """
        Executes complete pipeline on input image:
        Detect faces -> Align landmarks -> Extract 512-d ArcFace embedding.
        Records detailed component latencies.
        """
        start_time = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input image frame is empty or invalid.")

        # Step 1: Detect Faces
        detections = self.detector.detect(frame)

        processed_faces: list[TrackedFace] = []

        # Step 2 & 3: Align & Extract Embeddings for each detected face
        for idx, det in enumerate(detections):
            # Align face chip to 112x112
            aligned_chip = self.aligner.align(frame, det.landmarks)

            # Extract 512-dim ArcFace embedding
            rec_result = self.recognizer.extract_embedding(aligned_chip)

            tracked_face = TrackedFace(
                track_id=idx + 1,
                bbox=det.bbox,
                landmarks=det.landmarks,
                recognition=rec_result,
            )
            processed_faces.append(tracked_face)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return FrameProcessingResult(
            camera_id=camera_id,
            timestamp=time.time(),
            frame_number=frame_number,
            processing_time_ms=round(total_latency_ms, 2),
            faces=processed_faces,
        )
