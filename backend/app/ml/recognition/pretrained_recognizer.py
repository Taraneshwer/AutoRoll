"""
Pretrained ArcFace R50 (ONNX Runtime) Feature Extractor.
Model ID: 'pretrained' | Validated Threshold: 0.0440
"""

import os
import time

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.crypto import normalize_vector
from app.core.logger import get_logger
from app.ml.recognition.base import BaseFaceRecognizer
from app.schemas.common import RecognitionResult

logger = get_logger("pretrained_recognizer")


class PretrainedArcFaceRecognizer(BaseFaceRecognizer):
    """
    Genuine Pretrained ArcFace R50 ONNX model feature extractor.
    Outputs 512-dimensional L2-normalized embedding vectors.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
    ):
        settings = get_settings()
        self.model_id = "pretrained"
        self.model_version = "arcface_r50_v1"
        self.threshold = settings.PRETRAINED_THRESHOLD  # 0.0440
        self.model_path = model_path or settings.ARCFACE_GLINT_PATH
        self.ml_mode = settings.AUTOROLL_ML_MODE.lower()
        self.session = None
        self.device = "cpu"
        self.providers = []

        if os.path.exists(self.model_path):
            try:
                from app.ml.utils import create_onnx_session

                self.session, self.device, self.providers = create_onnx_session(
                    self.model_path, device_preference=device
                )
                logger.info(
                    f"RECOGNIZER INITIALIZED | ID: '{self.model_id}' | "
                    f"Version: '{self.model_version}' | Threshold: {self.threshold:.4f} | "
                    f"Path: '{self.model_path}' | Device: '{self.device}'"
                )
            except Exception as e:
                err = f"Failed to initialize Pretrained ONNX session from '{self.model_path}': {e}"
                if self.ml_mode == "production":
                    logger.error(err)
                    raise RuntimeError(err) from e
                logger.warning(f"{err}. Falling back to test mode.")
        else:
            err = f"Pretrained ONNX weights not found at '{self.model_path}'."
            if self.ml_mode == "production":
                logger.error(err)
                raise FileNotFoundError(err)
            logger.warning(f"{err}. Operating in test mode.")

    def extract_embedding(self, aligned_face: np.ndarray) -> RecognitionResult:
        if (
            aligned_face is None
            or not isinstance(aligned_face, np.ndarray)
            or aligned_face.size == 0
        ):
            raise ValueError("Invalid or empty image provided for feature extraction.")

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))

        start_t = time.perf_counter()
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
            raw_emb = outputs[0][0].tolist()
        else:
            raw_emb = self._fallback_embedding(blob)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        norm_emb = normalize_vector(raw_emb)

        return RecognitionResult(
            embedding=norm_emb,
            model_id=self.model_id,
            model_version=self.model_version,
            embedding_dimension=len(norm_emb),
            backend="ONNXRuntime",
            device=self.device,
            inference_latency_ms=elapsed_ms,
        )

    def extract_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> list[RecognitionResult]:
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
        results = []

        start_t = time.perf_counter()
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: batch_blob})
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / len(aligned_faces)
            for raw_emb in outputs[0]:
                norm_emb = normalize_vector(raw_emb.tolist())
                results.append(
                    RecognitionResult(
                        embedding=norm_emb,
                        model_id=self.model_id,
                        model_version=self.model_version,
                        embedding_dimension=len(norm_emb),
                        backend="ONNXRuntime",
                        device=self.device,
                        inference_latency_ms=elapsed_ms,
                    )
                )
        else:
            for b in batch_blob:
                raw_emb = self._fallback_embedding(np.expand_dims(b, axis=0))
                norm_emb = normalize_vector(raw_emb)
                results.append(
                    RecognitionResult(
                        embedding=norm_emb,
                        model_id=self.model_id,
                        model_version=self.model_version,
                        embedding_dimension=len(norm_emb),
                        backend="ONNXRuntime",
                        device=self.device,
                        inference_latency_ms=0.1,
                    )
                )

        return results

    def get_model_id(self) -> str:
        return self.model_id

    def get_model_version(self) -> str:
        return self.model_version

    def get_recognition_threshold(self) -> float:
        return self.threshold

    def _fallback_embedding(self, blob: np.ndarray) -> list[float]:
        flat = blob.flatten()
        if len(flat) >= 512:
            step = len(flat) // 512
            return [float(flat[i * step]) for i in range(512)]
        return np.pad(flat, (0, 512 - len(flat)), mode="constant").tolist()
