"""
Unit tests for logger and crypto utilities.
"""

import logging

from autoroll.common.crypto import compute_cosine_similarity, hash_string, normalize_vector
from autoroll.common.logger import get_logger


def test_logger_instance():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_vector_normalization_and_similarity():
    v1 = [3.0, 4.0]
    norm_v1 = normalize_vector(v1)
    # Norm of [3, 4] is 5, so normalized is [0.6, 0.8]
    assert abs(norm_v1[0] - 0.6) < 1e-5
    assert abs(norm_v1[1] - 0.8) < 1e-5

    v2 = [0.6, 0.8]
    sim = compute_cosine_similarity(norm_v1, v2)
    assert abs(sim - 1.0) < 1e-5


def test_hash_string():
    h1 = hash_string("autoroll")
    assert len(h1) == 64
