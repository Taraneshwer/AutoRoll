"""
Unit tests for Identity-Disjoint Splitter module.
Verifies zero identity leakage across splits.
"""

from app.ml.preprocessing.splitter import IdentityDisjointSplitter


def test_identity_disjoint_splitting():
    splitter = IdentityDisjointSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # 10 distinct identities
    identity_map = {f"student_{i:02d}": [f"img_{j}.jpg" for j in range(3)] for i in range(10)}

    splits = splitter.split_identities(identity_map)

    train_ids = set(splits["train"].keys())
    val_ids = set(splits["val"].keys())
    test_ids = set(splits["test"].keys())

    # Verify complete partition
    all_assigned = train_ids | val_ids | test_ids
    assert len(all_assigned) == 10

    # Verify identity disjointness (Zero leakage!)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
