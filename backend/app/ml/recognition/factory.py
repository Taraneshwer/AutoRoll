"""
Factory module for instantiating FaceRecognitionModel based on configuration.
Supports dynamic switching between 'pretrained' and 'autoroll_v1'.
"""

from app.core.config import get_settings
from app.core.logger import get_logger
from app.ml.recognition.autoroll_recognizer import AutoRollArcFaceRecognizer
from app.ml.recognition.base import FaceRecognitionModel
from app.ml.recognition.pretrained_recognizer import PretrainedArcFaceRecognizer

logger = get_logger("recognition_factory")

_MODEL_CACHE: dict[str, FaceRecognitionModel] = {}


def get_recognizer(
    model_id: str | None = None, device: str = "auto", force_reload: bool = False
) -> FaceRecognitionModel:
    """
    Returns an instance of FaceRecognitionModel for the given model_id.
    If model_id is None, reads AUTOROLL_RECOGNITION_MODEL from application settings.
    """
    settings = get_settings()
    target_id = (model_id or settings.AUTOROLL_RECOGNITION_MODEL).lower().strip()

    cache_key = f"{target_id}_{device}"
    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if target_id in ("pretrained", "baseline", "onnx"):
        recognizer = PretrainedArcFaceRecognizer(device=device)
    elif target_id in ("autoroll_v1", "autoroll", "epoch1", "pytorch"):
        recognizer = AutoRollArcFaceRecognizer(device=device)
    else:
        logger.warning(
            f"Unknown AUTOROLL_RECOGNITION_MODEL='{target_id}'. Defaulting to 'autoroll_v1'."
        )
        recognizer = AutoRollArcFaceRecognizer(device=device)

    _MODEL_CACHE[cache_key] = recognizer
    return recognizer
