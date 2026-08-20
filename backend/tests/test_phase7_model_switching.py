"""
Phase 7 Unit Tests for FaceRecognitionModel Abstraction and Model Switching.
Tests loading 'pretrained' vs 'autoroll_v1' recognizers, verifying threshold binding,
embedding dimensions, and model metadata tags.
"""

import os
import numpy as np
import pytest

from app.core.config import get_settings
from app.ml.recognition.autoroll_recognizer import AutoRollArcFaceRecognizer
from app.ml.recognition.base import BaseFaceRecognizer, FaceRecognitionModel
from app.ml.recognition.factory import get_recognizer
from app.ml.recognition.pretrained_recognizer import PretrainedArcFaceRecognizer


def test_pretrained_recognizer_properties():
    recognizer = PretrainedArcFaceRecognizer(device="cpu")
    assert isinstance(recognizer, BaseFaceRecognizer)
    assert recognizer.get_model_id() == "pretrained"
    assert recognizer.get_model_version() == "arcface_r50_v1"
    assert abs(recognizer.get_recognition_threshold() - 0.0440) < 1e-4

    # Dummy 112x112 chip test
    chip = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    res = recognizer.extract_embedding(chip)
    assert len(res.embedding) == 512
    norm = np.linalg.norm(res.embedding)
    assert abs(norm - 1.0) < 1e-5
    assert res.model_id == "pretrained"


def test_autoroll_v1_recognizer_properties():
    recognizer = AutoRollArcFaceRecognizer(device="cpu")
    assert isinstance(recognizer, BaseFaceRecognizer)
    assert recognizer.get_model_id() == "autoroll_v1"
    assert recognizer.get_model_version() == "autoroll_arcface_r50_epoch1"
    assert abs(recognizer.get_recognition_threshold() - 0.0540) < 1e-4

    # Dummy 112x112 chip test
    chip = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    res = recognizer.extract_embedding(chip)
    assert len(res.embedding) == 512
    norm = np.linalg.norm(res.embedding)
    assert abs(norm - 1.0) < 1e-5
    assert res.model_id == "autoroll_v1"


def test_factory_model_switching():
    r_pretrained = get_recognizer("pretrained", device="cpu", force_reload=True)
    assert r_pretrained.get_model_id() == "pretrained"
    assert abs(r_pretrained.get_recognition_threshold() - 0.0440) < 1e-4

    r_autoroll = get_recognizer("autoroll_v1", device="cpu", force_reload=True)
    assert r_autoroll.get_model_id() == "autoroll_v1"
    assert abs(r_autoroll.get_recognition_threshold() - 0.0540) < 1e-4


def test_model_threshold_uniqueness():
    r_pretrained = PretrainedArcFaceRecognizer(device="cpu")
    r_autoroll = AutoRollArcFaceRecognizer(device="cpu")
    assert r_pretrained.get_recognition_threshold() != r_autoroll.get_recognition_threshold()
