"""
Embedding Vector Aggregator for Multi-Sample Face Enrollment.
Computes L2-normalized mean feature embedding across face samples.
"""


import numpy as np

from app.core.logger import get_logger

logger = get_logger("embedding_aggregator")


class EmbeddingAggregator:
    """
    Combines multiple 512-d ArcFace embeddings into a single normalized representative vector.
    """

    @staticmethod
    def aggregate(embeddings: list[list[float]]) -> list[float]:
        """
        Computes L2-normalized centroid vector across input embedding samples.
        """
        if not embeddings:
            raise ValueError("Cannot aggregate empty list of embeddings.")

        arrs = [np.array(e, dtype=np.float32) for e in embeddings]
        for idx, a in enumerate(arrs):
            if a.shape != (512,):
                raise ValueError(f"Sample #{idx} has invalid shape {a.shape}, expected (512,)")

        # Element-wise mean
        mean_vec = np.mean(arrs, axis=0)

        # L2 Normalization
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-10:
            norm_vec = mean_vec / norm
        else:
            norm_vec = mean_vec

        logger.info(f"Aggregated {len(embeddings)} face embeddings into normalized 512-d centroid.")
        return norm_vec.astype(np.float32).tolist()
