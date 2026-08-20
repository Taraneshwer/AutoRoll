"""
Face Pair Generator and Verification Evaluator.
Generates genuine and impostor face pairs and measures cosine similarity & latency.
"""

import os
import random
import time

import cv2
from pydantic import BaseModel

from autoroll.common.crypto import compute_cosine_similarity
from autoroll.common.logger import get_logger
from autoroll.ml.recognition.base import BaseFaceRecognizer

logger = get_logger("verification_evaluator")


class FacePair(BaseModel):
    img_a_path: str
    img_b_path: str
    is_same: bool
    identity_a: str
    identity_b: str


class PairEvaluationResult(BaseModel):
    pair: FacePair
    similarity_score: float
    latency_ms: float


class VerificationEvaluator:
    """
    Generates genuine & impostor pairs from dataset and evaluates model recognition.
    """

    def __init__(self, recognizer: BaseFaceRecognizer, max_pairs: int = 200, seed: int = 42):
        self.recognizer = recognizer
        self.max_pairs = max_pairs
        self.seed = seed

    def generate_pairs(self, dataset_split_dir: str) -> list[FacePair]:
        """
        Scans split directory (split_dir/identity_id/img.jpg) and constructs balanced pairs.
        """
        if not os.path.exists(dataset_split_dir):
            logger.warning(f"Dataset split directory '{dataset_split_dir}' does not exist.")
            return []

        identity_dirs = sorted(
            [
                d
                for d in os.listdir(dataset_split_dir)
                if os.path.isdir(os.path.join(dataset_split_dir, d))
            ]
        )

        id_map: dict[str, list[str]] = {}
        for identity_id in identity_dirs:
            id_dir = os.path.join(dataset_split_dir, identity_id)
            imgs = [
                os.path.join(id_dir, f)
                for f in os.listdir(id_dir)
                if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            if len(imgs) >= 1:
                id_map[identity_id] = imgs

        identities = list(id_map.keys())
        if len(identities) < 2:
            logger.warning("Need at least 2 identities to generate verification pairs.")
            return []

        rng = random.Random(self.seed)
        genuine_pairs: list[FacePair] = []
        impostor_pairs: list[FacePair] = []

        # Genuine Pairs (Same identity)
        for identity_id, img_list in id_map.items():
            if len(img_list) >= 2:
                for i in range(len(img_list)):
                    for j in range(i + 1, len(img_list)):
                        genuine_pairs.append(
                            FacePair(
                                img_a_path=img_list[i],
                                img_b_path=img_list[j],
                                is_same=True,
                                identity_a=identity_id,
                                identity_b=identity_id,
                            )
                        )

        # Impostor Pairs (Different identities)
        for i in range(len(identities)):
            for j in range(i + 1, len(identities)):
                id1, id2 = identities[i], identities[j]
                img1 = rng.choice(id_map[id1])
                img2 = rng.choice(id_map[id2])
                impostor_pairs.append(
                    FacePair(
                        img_a_path=img1,
                        img_b_path=img2,
                        is_same=False,
                        identity_a=id1,
                        identity_b=id2,
                    )
                )

        rng.shuffle(genuine_pairs)
        rng.shuffle(impostor_pairs)

        n_per_class = self.max_pairs // 2
        selected_pairs = genuine_pairs[:n_per_class] + impostor_pairs[:n_per_class]
        rng.shuffle(selected_pairs)

        n_gen_sel = min(len(genuine_pairs), n_per_class)
        n_imp_sel = min(len(impostor_pairs), n_per_class)
        logger.info(
            f"Generated {len(selected_pairs)} verification pairs "
            f"({n_gen_sel} genuine, {n_imp_sel} impostor)."
        )
        return selected_pairs

    def evaluate_pairs(
        self, pairs: list[FacePair]
    ) -> tuple[list[float], list[float], float]:
        """
        Evaluates face pairs and returns (genuine_scores, impostor_scores, avg_pair_latency_ms).
        """
        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        embedding_cache: dict[str, list[float]] = {}

        start_time = time.perf_counter()

        def get_emb(fpath: str) -> list[float]:
            if fpath not in embedding_cache:
                img = cv2.imread(fpath)
                if img is None:
                    raise OSError(f"Could not load image at '{fpath}'")
                res = self.recognizer.extract_embedding(img)
                embedding_cache[fpath] = res.embedding
            return embedding_cache[fpath]

        for pair in pairs:
            emb_a = get_emb(pair.img_a_path)
            emb_b = get_emb(pair.img_b_path)
            sim = compute_cosine_similarity(emb_a, emb_b)

            if pair.is_same:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = total_time_ms / max(1, len(pairs))

        return genuine_scores, impostor_scores, avg_latency_ms
