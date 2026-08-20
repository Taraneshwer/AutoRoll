"""
Crypto and Math Utilities for Vector Normalization & Cosine Similarity.
"""

import hashlib

import numpy as np


def normalize_vector(embedding: list[float]) -> list[float]:
    """
    L2 Normalizes a 1D vector (List[float]).
    """
    arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return embedding
    normalized = arr / norm
    return normalized.tolist()


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Computes cosine similarity between two normalized 1D vectors.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)

    # Assuming vectors are already L2 normalized: dot product equals cosine similarity
    similarity = np.dot(v1, v2)
    return float(similarity)


def hash_string(input_str: str) -> str:
    """
    Returns SHA-256 hash of string.
    """
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()
