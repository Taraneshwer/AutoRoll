"""
Unit tests for EmbeddingAggregator.
"""

import numpy as np
import pytest

from app.ml.enrollment.aggregator import EmbeddingAggregator


def test_embedding_aggregator_normalization():
    vec1 = [1.0] + [0.0] * 511
    vec2 = [0.0, 1.0] + [0.0] * 510

    centroid = EmbeddingAggregator.aggregate([vec1, vec2])

    assert len(centroid) == 512
    # Verify L2 normalization: norm == 1.0
    norm = np.linalg.norm(np.array(centroid))
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_invalid_embedding_dimensions():
    with pytest.raises(ValueError):
        EmbeddingAggregator.aggregate([[1.0, 2.0]])
