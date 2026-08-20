"""
Independent ArcFace / IResNet50 Face Feature Extractor Module
supporting ONNX Runtime (CPU/CUDA) and deterministic feature extraction.
"""

import os

import cv2
import numpy as np

from app.core.crypto import normalize_vector
from app.core.logger import get_logger
from app.schemas.common import RecognitionResult
from app.ml.recognition.base import BaseFaceRecognizer
from app.ml.utils import get_execution_device

logger = get_logger("arcface_recognizer")


class ArcFaceRecognizer(BaseFaceRecognizer):
    """
    ArcFace Feature Extractor with IResNet50 backbone.
    Outputs 512-dimensional L2-normalized embedding vectors.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        model_version: str = "arcface_iresnet50_v1",
    ):
        self.model_version = model_version
        self.model_path = model_path
        self.session = None
        import time
        from app.core.config import get_settings
        settings = get_settings()
        self.ml_mode = settings.AUTOROLL_ML_MODE.lower()
        self.model_id = settings.AUTOROLL_RECOGNITION_MODEL.lower()
        self.model_name = f"ArcFace_{self.model_id.upper()}"

        if not self.model_path:
            if self.model_id == "ms1mv2" and os.path.exists(settings.ARCFACE_MS1MV2_PATH):
                path_to_use = settings.ARCFACE_MS1MV2_PATH
            elif self.model_id in ("glint360k", "webface600k") and os.path.exists(settings.ARCFACE_GLINT_PATH):
                path_to_use = settings.ARCFACE_GLINT_PATH
            else:
                path_to_use = settings.ARCFACE_MODEL_PATH
        else:
            path_to_use = self.model_path

        self.model_path = path_to_use

        if os.path.exists(self.model_path):
            try:
                from app.ml.utils import create_onnx_session
                self.session, self.device, self.providers = create_onnx_session(
                    self.model_path, device_preference=device
                )
                logger.info(
                    f"ML MODEL LOADED | ID: '{self.model_id}' | Name: '{self.model_name}' | "
                    f"Version: '{self.model_version}' | Path: '{self.model_path}' | "
                    f"Backend: ONNXRuntime | Device: '{self.device}' | "
                    "Precision: FP32 | Status: READY"
                )
            except Exception as e:
                err_msg = (
                    f"PRODUCTION ML ERROR: Failed to load ArcFace ONNX model from "
                    f"'{self.model_path}': {e}"
                )
                if self.ml_mode == "production":
                    logger.error(err_msg)
                    raise RuntimeError(err_msg) from e
                logger.warning(f"{err_msg}. Operating in TEST fallback mode.")
        else:
            err_msg = (
                f"PRODUCTION ML ERROR: ArcFace model weights not found at "
                f"'{self.model_path}'."
            )
            if self.ml_mode == "production":
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            logger.info(f"{err_msg} Operating in TEST fallback mode.")

    def extract_embedding(self, aligned_face: np.ndarray) -> RecognitionResult:
        """
        Extracts 512-dimensional normalized embedding vector from 112x112 aligned face chip.
        """
        import time
        if (
            aligned_face is None
            or not isinstance(aligned_face, np.ndarray)
            or aligned_face.size == 0
        ):
            raise ValueError("Invalid or empty image provided for feature extraction.")

        # Ensure correct input size (112, 112)
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))

        start_t = time.perf_counter()

        # Standard ArcFace Preprocessing: BGR -> RGB, scale (x - 127.5) / 127.5, (1, 3, 112, 112)
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        blob = (rgb_face.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
            raw_embedding = outputs[0][0].tolist()
        else:
            raw_embedding = self._extract_fallback_embedding(blob)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Apply L2 Normalization
        normalized_emb = normalize_vector(raw_embedding)

        return RecognitionResult(
            embedding=normalized_emb,
            model_id=self.model_id,
            model_version=self.model_version,
            embedding_dimension=len(normalized_emb),
            backend="ONNXRuntime",
            device=self.device,
            inference_latency_ms=elapsed_ms,
        )

    def _extract_fallback_embedding(self, blob: np.ndarray) -> list[float]:
        """
        Deterministic pseudo feature extraction for unweighted/test environments.
        Produces a consistent 512-dimensional vector derived deterministically from the image blob.
        """
        # Reduce spatial dimensions into a deterministic 512 vector
        flat = blob.flatten()
        # Repeat or pool to exactly 512 elements
        if len(flat) >= 512:
            step = len(flat) // 512
            raw_vec = [float(flat[i * step]) for i in range(512)]
        else:
            padded = np.pad(flat, (0, 512 - len(flat)), mode="constant")
            raw_vec = padded.tolist()

        return raw_vec

    def warmup(self) -> None:
        """
        Warms up model inference session with a dummy tensor to eliminate cold-start latency.
        """
        dummy_chip = np.zeros((112, 112, 3), dtype=np.uint8)
        self.extract_embedding(dummy_chip)
        logger.info(f"ArcFaceRecognizer model session warmed up on device '{self.device}'.")

    def extract_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> list[RecognitionResult]:
        """
        Batched feature extraction across multiple aligned face chips.
        """
        if not aligned_faces:
            return []

        blobs = []
        for face in aligned_faces:
            if face.shape[:2] != (112, 112):
                face = cv2.resize(face, (112, 112))
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            b = (rgb.astype(np.float32) - 127.5) / 127.5
            blobs.append(np.transpose(b, (2, 0, 1)))

        batch_blob = np.array(blobs, dtype=np.float32)

        results: list[RecognitionResult] = []
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: batch_blob})
            for raw_emb in outputs[0]:
                norm_emb = normalize_vector(raw_emb.tolist())
                results.append(
                    RecognitionResult(
                        embedding=norm_emb, model_version=self.model_version
                    )
                )
        else:
            for b in batch_blob:
                raw_emb = self._extract_fallback_embedding(np.expand_dims(b, axis=0))
                norm_emb = normalize_vector(raw_emb)
                results.append(
                    RecognitionResult(
                        embedding=norm_emb, model_version=self.model_version
                    )
                )

        return results

    def get_model_id(self) -> str:
        return self.model_id

    def get_model_version(self) -> str:
        return self.model_version

    def get_recognition_threshold(self) -> float:
        from app.core.config import get_settings
        settings = get_settings()
        if self.model_id in ("autoroll_v1", "autoroll", "epoch1"):
            return settings.AUTOROLL_V1_THRESHOLD
        return settings.PRETRAINED_THRESHOLD

