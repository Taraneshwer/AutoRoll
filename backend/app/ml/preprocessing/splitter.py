"""
Identity-Disjoint Train/Validation/Test Splitter.
Guarantees zero identity leakage between splits.
"""

import random
from typing import Any

from app.core.logger import get_logger

logger = get_logger("identity_splitter")


class IdentityDisjointSplitter:
    """
    Splits identities into disjoint sets for training, validation, and testing.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.2f})")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_identities(
        self, identity_map: dict[str, list[Any]]
    ) -> dict[str, dict[str, list[Any]]]:
        """
        Splits identity keys into disjoint sets and returns split datasets.
        """
        identities = sorted(list(identity_map.keys()))
        if not identities:
            return {"train": {}, "val": {}, "test": {}}

        rng = random.Random(self.seed)
        shuffled = identities.copy()
        rng.shuffle(shuffled)

        n_identities = len(shuffled)
        n_val = max(1, int(n_identities * self.val_ratio))
        n_test = max(1, int(n_identities * self.test_ratio))
        if n_identities >= 6:
            n_val = max(2, n_val)
            n_test = max(2, n_test)
        n_train = max(1, n_identities - n_val - n_test)

        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train : n_train + n_val])
        test_ids = set(shuffled[n_train + n_val :])

        # Assert identity disjointness
        assert train_ids.isdisjoint(val_ids), "Identity leakage detected between train and val!"
        assert train_ids.isdisjoint(test_ids), "Identity leakage detected between train and test!"
        assert val_ids.isdisjoint(test_ids), "Identity leakage detected between val and test!"

        split_result: dict[str, dict[str, list[Any]]] = {
            "train": {identity_id: identity_map[identity_id] for identity_id in train_ids},
            "val": {identity_id: identity_map[identity_id] for identity_id in val_ids},
            "test": {identity_id: identity_map[identity_id] for identity_id in test_ids},
        }

        logger.info(
            f"Identity-Disjoint Split Complete: Total Identities={n_identities} | "
            f"Train={len(train_ids)} | Val={len(val_ids)} | Test={len(test_ids)}"
        )

        return split_result
