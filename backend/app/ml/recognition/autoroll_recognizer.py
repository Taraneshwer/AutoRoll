"""
AutoRoll Fine-Tuned ArcFace R50 (PyTorch Checkpoint) Feature Extractor.
Model ID: 'autoroll_v1' | Validated Threshold: 0.0540
"""

import os
import time

import cv2
import numpy as np
import torch

from app.core.config import get_settings
from app.core.crypto import normalize_vector
from app.core.logger import get_logger
from app.ml.recognition.base import BaseFaceRecognizer
from app.ml.recognition.iresnet_torch import get_iresnet50
from app.ml.utils import get_execution_device
from app.schemas.common import RecognitionResult

logger = get_logger("autoroll_recognizer")


class AutoRollArcFaceRecognizer(BaseFaceRecognizer):
    """
    Fine-tuned AutoRoll ArcFace R50 PyTorch model feature extractor (epoch_001.pt).
    Outputs 512-dimensional L2-normalized embedding vectors.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
    ):
        settings = get_settings()
        self.model_id = "autoroll_v1"
        self.model_version = "autoroll_arcface_r50_epoch1"
        self.threshold = settings.AUTOROLL_V1_THRESHOLD  # 0.0540
        self.model_path = model_path or settings.ARCFACE_AUTOROLL_V1_PATH
        self.ml_mode = settings.AUTOROLL_ML_MODE.lower()
        self.model = None

        # Hardware execution resolution
        self.device_str, _ = get_execution_device(device)
        self.device = torch.device("cuda" if self.device_str == "cuda" and torch.cuda.is_available() else "cpu")

        if os.path.exists(self.model_path):
            try:
                self.model = get_iresnet50().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state = checkpoint.get("backbone_state", checkpoint)
                self.model.load_state_dict(state)
                self.model.eval()

                logger.info(
                    f"RECOGNIZER INITIALIZED | ID: '{self.model_id}' | "
                    f"Version: '{self.model_version}' | Threshold: {self.threshold:.4f} | "
                    f"Path: '{self.model_path}' | Device: '{self.device}'"
                )
            except Exception as e:
                err = f"Failed to load AutoRoll PyTorch checkpoint from '{self.model_path}': {e}"
                if self.ml_mode == "production":
                    logger.error(err)
                    raise RuntimeError(err) from e
                logger.warning(f"{err}. Operating in test fallback mode.")
        else:
            err = f"AutoRoll PyTorch weights not found at '{self.model_path}'."
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
        blob_tensor = torch.from_numpy(np.transpose(blob, (2, 0, 1))).float().unsqueeze(0).to(self.device)

        if self.model is not None:
            with torch.no_grad():
                out_tensor = self.model(blob_tensor)
                raw_emb = out_tensor.squeeze(0).cpu().numpy().tolist()
        else:
            raw_emb = self._fallback_embedding(blob_tensor.cpu().numpy())

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        norm_emb = normalize_vector(raw_emb)

        return RecognitionResult(
            embedding=norm_emb,
            model_id=self.model_id,
            model_version=self.model_version,
            embedding_dimension=len(norm_emb),
            backend="PyTorch",
            device=str(self.device),
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

        batch_tensor = torch.from_numpy(np.array(blobs, dtype=np.float32)).to(self.device)
        results = []

        start_t = time.perf_counter()
        if self.model is not None:
            with torch.no_grad():
                outs = self.model(batch_tensor).cpu().numpy()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / len(aligned_faces)
            for raw_emb in outs:
                norm_emb = normalize_vector(raw_emb.tolist())
                results.append(
                    RecognitionResult(
                        embedding=norm_emb,
                        model_id=self.model_id,
                        model_version=self.model_version,
                        embedding_dimension=len(norm_emb),
                        backend="PyTorch",
                        device=str(self.device),
                        inference_latency_ms=elapsed_ms,
                    )
                )
        else:
            for b in blobs:
                raw_emb = self._fallback_embedding(b)
                norm_emb = normalize_vector(raw_emb)
                results.append(
                    RecognitionResult(
                        embedding=norm_emb,
                        model_id=self.model_id,
                        model_version=self.model_version,
                        embedding_dimension=len(norm_emb),
                        backend="PyTorch",
                        device=str(self.device),
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
