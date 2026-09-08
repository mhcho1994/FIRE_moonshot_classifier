"""Versioned, tensor-only model bundle with its calibrated OOD reference."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from fire_moonshot_classifier.datamanager import config
from .model import inference_state


PREPROCESSING_V1 = {
    "version": "turn-kinematics-v1",
    "kinematic_hz": 50,
    "segment_hz": 20,
    "segment_window_sec": 2.0,
    "segment_overlap_sec": 1.0,
    "smooth_window_len": 200,
    "smooth_poly_order": 3,
    "normalization": "per-window-zscore-v1",
}
CLASS_NAMES = ["ArduPilot", "PX4"]


def preprocessing_config(feature_names, win_len=100, hop_len=50, min_win=30):
    return {
        **PREPROCESSING_V1,
        "feature_names": list(feature_names),
        "win_len": win_len, "hop_len": hop_len, "min_win": min_win,
        "hmm_pi": config.HMM_PI.tolist(), "hmm_a": config.HMM_A.tolist(),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Expected a version-1 inference bundle; export raw weights with 'fireclassify export'")
    try:
        inferred, _ = inference_state(bundle["model_state_dict"])
        if inferred != bundle["model_config"] or bundle["class_names"] != CLASS_NAMES:
            raise ValueError("Model configuration or class order does not match the weights")
        prep = bundle["preprocessing"]
        if any(prep.get(key) != value for key, value in PREPROCESSING_V1.items()):
            raise ValueError("Unsupported preprocessing recipe")
        names = prep["feature_names"]
        if (not isinstance(names, list) or len(names) != inferred["n_feat"]
                or any(name not in config.FEATURE_MAP for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Feature names must match the model's input channels in training order")
        for key in ("win_len", "hop_len", "min_win"):
            if type(prep[key]) is not int or prep[key] < 1:
                raise ValueError(f"Invalid preprocessing {key}")
        if prep["win_len"] < 4 or prep["min_win"] > prep["win_len"]:
            raise ValueError("Expected 1 <= min_win <= win_len and win_len >= 4")
        for key, shape in (("hmm_pi", (6,)), ("hmm_a", (6, 6))):
            values = np.asarray(prep[key], dtype=float)
            if (values.shape != shape or not np.isfinite(values).all()
                    or np.any(values < 0) or not np.allclose(values.sum(axis=-1), 1.0)):
                raise ValueError(f"Invalid {key} probabilities")
        decision = bundle["decision"]
        k = decision["knn_k"]
        if type(k) is not int or k < 1:
            raise ValueError("knn_k must be a positive integer")
        if decision["ood_level"] != "block1_mean" or decision["tie_break"] != "ArduPilot":
            raise ValueError("Unsupported OOD scoring or voting recipe")
        if type(decision["min_valid"]) is not int or decision["min_valid"] < 1:
            raise ValueError("min_valid must be a positive integer")
        if not 0 <= decision["min_fraction"] <= 1:
            raise ValueError("min_fraction must be between 0 and 1")
        threshold = bundle["ood_threshold"]
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("OOD threshold must be finite and nonnegative")
        bank = bundle["ood_bank_l1"]
        if (not isinstance(bank, torch.Tensor) or bank.ndim != 2
                or bank.shape[1] != 32 or len(bank) < k or not torch.isfinite(bank).all()):
            raise ValueError("OOD bank must contain at least knn_k finite 32-dimensional vectors")
        # Metadata must also stay loadable with weights_only=True and JSON serializable.
        json.dumps({k: bundle[k] for k in ("model_config", "preprocessing", "decision", "provenance")}, allow_nan=False)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Incomplete or invalid inference bundle: {exc}") from exc


def save_bundle(path, *, state_dict, feature_names, bank_l1, threshold,
                knn_k=5, min_valid=3, min_fraction=0.15,
                win_len=100, hop_len=50, min_win=30, provenance=None):
    """Export after calibration; no training datasets are included in the file."""
    model_config, weights = inference_state(state_dict)
    bundle = {
        "format_version": 1, "model_config": model_config,
        "model_state_dict": weights, "class_names": CLASS_NAMES.copy(),
        "preprocessing": preprocessing_config(feature_names, win_len, hop_len, min_win),
        "ood_bank_l1": bank_l1.detach().cpu().float().clone(),
        "ood_threshold": float(threshold),
        "decision": {"ood_level": "block1_mean", "knn_k": knn_k,
                     "min_valid": min_valid, "min_fraction": min_fraction,
                     "tie_break": "ArduPilot"},
        "provenance": provenance or {},
    }
    validate_bundle(bundle)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(bundle, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return path


def load_bundle(path, expected_sha256=None):
    path = Path(path).expanduser()
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Model bundle SHA-256 does not match the expected digest")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(bundle)
    return bundle, digest
