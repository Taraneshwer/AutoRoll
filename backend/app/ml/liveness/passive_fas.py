"""
Passive Anti-Spoofing Model Module combining Deep Feature Representation
and Frequency Spectrum Moire Pattern Analysis.
"""

import os
from typing import Any

import cv2
import numpy as np

from app.core.logger import get_logger
from app.ml.utils import get_execution_device

logger = get_logger("passive_fas")


class PassiveAntiSpoofingModel:
    """
    Spatial Passive Anti-Spoofing Model.
    Analyzes high-frequency texture noise, Moire patterns, and deep features.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        model_version: str = "minifasnet_v1",
    ):
        self.model_version = model_version
        self.model_path = model_path
        self.session = None
        from app.core.config import get_settings
        settings = get_settings()
        self.ml_mode = settings.AUTOROLL_ML_MODE.lower()
        self.model_name = "MiniFASNet_PassivePAD"

        path_to_use = self.model_path or settings.PAD_MODEL_PATH
        self.model_path = path_to_use

        if os.path.exists(self.model_path):
            try:
                from app.ml.utils import create_onnx_session
                self.session, self.device, self.providers = create_onnx_session(
                    self.model_path, device_preference=device
                )
                logger.info(
                    f"ML MODEL LOADED | Name: '{self.model_name}' | "
                    f"Version: '{self.model_version}' | Path: '{self.model_path}' | "
                    f"Backend: ONNXRuntime | Device: '{self.device}' | "
                    "Precision: FP32 | Status: READY"
                )
            except Exception as e:
                err_msg = (
                    f"PRODUCTION ML ERROR: Failed to load MiniFASNet ONNX model from "
                    f"'{self.model_path}': {e}"
                )
                if self.ml_mode == "production":
                    logger.error(err_msg)
                    raise RuntimeError(err_msg) from e
                logger.warning(f"{err_msg}. Operating in TEST fallback mode.")
        else:
            err_msg = (
                f"PRODUCTION ML ERROR: MiniFASNet model weights not found at "
                f"'{self.model_path}'."
            )
            if self.ml_mode == "production":
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            logger.info(f"{err_msg} Operating in TEST fallback mode.")

    def predict_spatial_score(self, face_chip: np.ndarray) -> float:
        """
        Evaluates 112x112 or scaled face chip for spatial liveness score (0.0=Spoof, 1.0=Real).
        """
        if face_chip is None or not isinstance(face_chip, np.ndarray) or face_chip.size == 0:
            raise ValueError("Invalid or empty face chip provided for anti-spoofing evaluation.")

        if self.session is not None:
            return self._predict_onnx(face_chip)
        else:
            return self._predict_analytical_fallback(face_chip)

    def _predict_onnx(self, face_chip: np.ndarray) -> float:
        """
        ONNX Runtime inference for MiniFASNet.
        """
        res = self.evaluate_liveness_detailed(face_chip)
        return res["ml_real_prob"]

    def evaluate_liveness_detailed(self, face_chip: np.ndarray) -> dict[str, Any]:
        """
        Detailed liveness evaluation keeping ML model predictions and auxiliary heuristics separate.
        Returns dictionary containing raw_logits, ml_real_prob, aux_heuristic_score, and combined_score.
        """
        resized = cv2.resize(face_chip, (80, 80))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            # Class 1 is real face, Class 0 is spoof in Silent-Face-Anti-Spoofing
            ml_real_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            raw_logits = logits.tolist()
        else:
            ml_real_prob = self._predict_analytical_fallback(face_chip)
            raw_logits = [1.0 - ml_real_prob, ml_real_prob]

        aux_heuristic = self._predict_analytical_fallback(face_chip)
        combined = 0.85 * ml_real_prob + 0.15 * aux_heuristic

        return {
            "raw_logits": raw_logits,
            "ml_real_prob": ml_real_prob,
            "aux_heuristic_score": aux_heuristic,
            "combined_score": float(np.clip(combined, 0.0, 1.0)),
        }

    def _predict_analytical_fallback(self, face_chip: np.ndarray) -> float:
        """
        Analytical texture and frequency spectrum fallback estimator for testing.
        Analyzes Laplacian variance and 2D Discrete Fourier Transform (FFT) high-frequency ratio.
        Screen and paper printouts show Moire patterns or low high-frequency variance.
        """
        gray = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY) if face_chip.ndim == 3 else face_chip

        # 1. High-frequency texture score (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        texture_score = min(1.0, lap_var / 300.0)

        # 2. Fourier Spectrum Moire Analysis
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)

        # Ratio of high-frequency outer energy to central low-frequency energy
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(cy, cx) // 4
        low_freq_energy = np.mean(magnitude_spectrum[cy - r : cy + r, cx - r : cx + r])
        total_energy = np.mean(magnitude_spectrum)
        hf_ratio = min(1.0, float(total_energy / max(1e-5, low_freq_energy)))

        # Combined spatial score
        combined_score = 0.6 * texture_score + 0.4 * hf_ratio
        return float(np.clip(combined_score, 0.0, 1.0))

    def predict(self, frame: np.ndarray, bbox: Any = None) -> Any:
        """
        Convenience predictor extracting face chip from bbox and returning LivenessResult schema object.
        """
        from app.schemas.common import LivenessResult
        from app.core.config import get_settings

        settings = get_settings()

        if bbox is not None and hasattr(bbox, "x1"):
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = int(max(0, bbox.x1)), int(max(0, bbox.y1)), int(min(w, bbox.x2)), int(min(h, bbox.y2))
            if x2 > x1 and y2 > y1:
                chip = frame[y1:y2, x1:x2]
            else:
                chip = frame
        else:
            chip = frame

        res = self.evaluate_liveness_detailed(chip)
        ml_score = res["ml_real_prob"]
        aux_score = res["aux_heuristic_score"]
        combined_score = res["combined_score"]
        is_live = combined_score >= settings.LIVENESS_THRESHOLD

        return LivenessResult(
            is_live=is_live,
            liveness_score=combined_score,
            ml_liveness_score=ml_score,
            auxiliary_heuristic_score=aux_score,
            combined_liveness_score=combined_score,
            method="passive_fas",
        )


PassiveLivenessDetector = PassiveAntiSpoofingModel

