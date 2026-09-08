"""Run-disjoint splits shared by training and inference export."""
import numpy as np


def split_cached_segments(labels, groups=None, test_ratio=0.2, val_ratio=0.15, rng=None):
    """Split at run level when cache groups exist, otherwise by segment."""
    rng = np.random if rng is None else rng
    labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        n_test = max(1, int(len(unique_groups) * test_ratio))
        remaining = unique_groups[n_test:]
        n_val = max(1, int(len(remaining) * val_ratio))
        test_groups = unique_groups[:n_test]
        val_groups = remaining[:n_val]
        train_groups = remaining[n_val:]
        return (
            np.flatnonzero(np.isin(groups, train_groups)),
            np.flatnonzero(np.isin(groups, val_groups)),
            np.flatnonzero(np.isin(groups, test_groups)),
        )

    train_idx, val_idx, test_idx = [], [], []

    for cls in (0, 1):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_ratio)) if len(cls_idx) > 2 else 0
        remaining = cls_idx[n_test:]
        n_val = max(1, int(len(remaining) * val_ratio)) if len(remaining) > 2 else 0
        test_idx.extend(cls_idx[:n_test])
        val_idx.extend(remaining[:n_val])
        train_idx.extend(remaining[n_val:])

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )

