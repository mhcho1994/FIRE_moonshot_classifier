"""One-time OOD calibration of an existing checkpoint on a SITL feature cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fire_moonshot_classifier.datamanager.splits import split_cached_segments
from fire_moonshot_classifier.processor.sequence_windows import windows_from_sequence
from .bundle import save_bundle, sha256_file
from .model import DiversifyNetwork, inference_state, knn_distances


@torch.inference_mode()
def export_checkpoint(checkpoint, sitl_cache, output, *, feature_names=None, seed=42,
                      knn_k=5, percentile=95.0, min_valid=3, min_fraction=0.15,
                      batch_size=128):
    """Recalibrate frozen weights; use train groups for bank and held-out groups for threshold.

    This supports the cached turn-feature pipeline, not legacy feat7 raw-log models.
    Existing checkpoints do not record their split: seed must match the training run.
    The resulting bundle records that calibration was reconstructed from a cache.
    """
    checkpoint, sitl_cache, output = map(Path, (checkpoint, sitl_cache, output))
    if output.resolve() in (checkpoint.resolve(), sitl_cache.resolve()):
        raise ValueError("Export output must differ from the input checkpoint and cache")
    if type(knn_k) is not int or knn_k < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("knn_k and batch_size must be positive integers")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100, exclusive")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "format_version" in state:
        raise ValueError("Checkpoint is already an inference bundle")
    model_config, weights = inference_state(state)
    with np.load(sitl_cache, allow_pickle=False) as cache:
        sequences, labels = cache["X_seq"], cache["y"]
        groups = cache["runs"] if "runs" in cache else None
        stored_names = [str(name) for name in cache["feature_names"]] if "feature_names" in cache else None
        if feature_names is not None and stored_names is not None and list(feature_names) != stored_names:
            raise ValueError("Explicit feature order conflicts with cache metadata")
        feature_names = stored_names if stored_names is not None else feature_names
        if feature_names is None:
            raise ValueError("Cache has no feature_names; specify --target-features in training channel order")
        if (sequences.ndim != 3 or labels.shape != (len(sequences),)
                or (groups is not None and groups.shape != labels.shape)
                or sequences.shape[2] != model_config["n_feat"]
                or len(feature_names) != model_config["n_feat"]):
            raise ValueError("Checkpoint channels and cache X_seq/y/runs/feature_names do not match")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("SITL calibration cache must contain only PX4=0 and ArduPilot=1")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Calibration requires both PX4 and ArduPilot")
        train_idx, val_idx, test_idx = split_cached_segments(
            labels, groups=groups, rng=np.random.RandomState(seed),
        )
        model = DiversifyNetwork(**model_config).eval()
        model.load_state_dict(weights, strict=True)

        def features_for(indices):
            features, window_labels, pending = [], [], []
            for index in indices:
                for window in windows_from_sequence(sequences[index]):
                    pending.append(window)
                    window_labels.append(int(labels[index]))
                    if len(pending) == batch_size:
                        features.append(model(torch.stack(pending))[1])
                        pending = []
            if pending:
                features.append(model(torch.stack(pending))[1])
            if not features or set(window_labels) != {0, 1}:
                raise ValueError("Train and validation splits must each produce windows from both classes")
            return torch.cat(features)

        bank = features_for(train_idx)
        if len(bank) < knn_k:
            raise ValueError("Not enough training windows for the requested knn_k")
        validation = features_for(val_idx)
    distances = torch.cat([
        knn_distances(batch, bank, knn_k) for batch in validation.split(batch_size)
    ]).numpy()
    threshold = float(np.percentile(distances, percentile))
    return save_bundle(
        output, state_dict=weights, feature_names=feature_names, bank_l1=bank,
        threshold=threshold, knn_k=knn_k, min_valid=min_valid, min_fraction=min_fraction,
        provenance={
            "calibration": "reconstructed_from_cache", "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "sitl_cache_sha256": sha256_file(sitl_cache), "seed": seed,
            "test_ratio": 0.2, "validation_ratio_of_remaining": 0.15,
            "split_level": "run" if groups is not None else "segment",
            "n_train_segments": len(train_idx), "n_validation_segments": len(val_idx),
            "n_test_segments": len(test_idx), "n_bank_windows": len(bank),
            "n_validation_windows": len(validation), "ood_percentile": percentile,
        },
    )
