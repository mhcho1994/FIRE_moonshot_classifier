"""Unlabeled trajectory CSV to a JSON-serializable autopilot prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fire_moonshot_classifier.datamanager.dataset_manager import extract_turn_sequences
from fire_moonshot_classifier.processor.sequence_windows import windows_from_sequence
from .bundle import CLASS_NAMES, load_bundle
from .model import DiversifyNetwork, knn_distances


def read_trajectory(csv_path, measurement_type="vision"):
    """Read seconds/meters, drop non-finite rows and deduplicate before differentiation."""
    if measurement_type not in ("vision", "mocap"):
        raise ValueError("measurement_type must be vision or mocap")
    csv_path = Path(csv_path).expanduser()
    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8-sig"))
    names = data.dtype.names or ()
    time_key = next((key for key in ("time_s", "timestamp") if key in names), None)
    candidates = (("x_smooth", "y_smooth", "z_smooth"), ("xsmooth", "ysmooth", "zsmooth"))
    if measurement_type == "mocap":
        candidates = (("gt_x", "gt_y", "gt_z"), ("gtx", "gty", "gtz"))
    xyz = next((keys for keys in candidates if all(key in names for key in keys)), None)
    if time_key is None or xyz is None:
        raise ValueError(f"CSV requires time_s/timestamp and {measurement_type} columns {candidates}")
    values = np.column_stack([data[key] for key in (time_key, *xyz)])
    finite = np.isfinite(values).all(axis=1)
    valid = values[finite]
    _, indices = np.unique(valid[:, 0], return_index=True)
    clean = valid[indices]  # np.unique orders timestamps, keeping the first valid row.
    info = {"n_rows": len(values), "n_valid_rows": len(clean),
            "n_nonfinite_rows": int((~finite).sum()),
            "n_duplicate_timestamps": len(valid) - len(clean)}
    if len(clean) < 10:
        return None, info
    t, x, y, z = clean.T
    return {"t": t, "x": x, "y": y, "z": z,
            "vx": np.gradient(x, t), "vy": np.gradient(y, t), "vz": np.gradient(z, t)}, info


class TrajectoryPredictor:
    """Load once per server worker, then call predict_trajectory for each CSV.

    Inference uses frozen BatchNorm statistics and never modifies model state.
    CPU is the default so classification does not compete with SAM3 for GPU memory.
    """
    def __init__(self, model_path, *, device="cpu", batch_size=128, expected_sha256=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        bundle, self.model_sha256 = load_bundle(model_path, expected_sha256)
        self.preprocessing = bundle["preprocessing"]
        self.decision = bundle["decision"]
        self.threshold = float(bundle["ood_threshold"])
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = DiversifyNetwork(**bundle["model_config"])
        self.model.load_state_dict(bundle["model_state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.bank = bundle["ood_bank_l1"].cpu().float()

    @torch.inference_mode()
    def predict_trajectory(self, csv_path, *, measurement_type="vision"):
        """Return PX4/ArduPilot/Unknown and window counts; no ground truth is read.

        Missing/malformed files raise exceptions. Valid but insufficient data
        returns Unknown with a reason, not a fabricated binary prediction.
        """
        raw, quality = read_trajectory(csv_path, measurement_type)
        prep = self.preprocessing
        sequences, _ = extract_turn_sequences(
            raw, prep["feature_names"], hmm_pi=prep["hmm_pi"], hmm_a=prep["hmm_a"],
        )
        windows = [window for seq in sequences for window in windows_from_sequence(
            seq, prep["win_len"], prep["hop_len"], prep["min_win"],
        )]
        n_total = len(windows)
        result = {
            "file": str(Path(csv_path)), "prediction": "Unknown", "reason": None,
            "model_sha256": self.model_sha256, "measurement_type": measurement_type,
            "n_segments": len(sequences), "n_windows": n_total,
            "n_accepted": 0, "n_rejected": 0, "n_px4": 0, "n_ardu": 0,
            "reject_rate": None, "knn_dist": None, "ood_threshold": self.threshold,
            "vote_fraction": None, "input_quality": quality,
        }
        if not windows:
            result["reason"] = ("insufficient_samples" if raw is None else
                                "no_turn_segments" if not sequences else "no_valid_windows")
            return result
        votes = torch.zeros(2, dtype=torch.int64)
        distance_sum = 0.0
        for start in range(0, n_total, self.batch_size):
            x = torch.stack(windows[start:start + self.batch_size]).to(self.device)
            logits, features = self.model(x)
            if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
                raise ValueError("Model produced non-finite outputs")
            distances = knn_distances(features, self.bank, self.decision["knn_k"])
            accepted = distances <= self.threshold
            result["n_accepted"] += int(accepted.sum())
            distance_sum += float(distances.double().sum())
            votes += torch.bincount(logits.cpu()[accepted].argmax(dim=1), minlength=2)
        n_valid = result["n_accepted"]
        result.update(n_rejected=n_total - n_valid, reject_rate=(n_total - n_valid) / n_total,
                      knn_dist=distance_sum / n_total)
        if n_valid < self.decision["min_valid"] or n_valid / n_total < self.decision["min_fraction"]:
            result["reason"] = "insufficient_in_distribution_windows"
            return result
        label = int(votes.argmax())  # torch.mode in training resolves ties to class 0.
        result.update(prediction=CLASS_NAMES[label], reason="classified",
                      n_ardu=int(votes[0]), n_px4=int(votes[1]),
                      vote_fraction=int(votes[label]) / n_valid)
        return result


def predict_trajectory(csv_path, model_path, *, measurement_type="vision", device="cpu",
                       batch_size=128, expected_sha256=None):
    """One-shot API. For a web server, reuse TrajectoryPredictor instead."""
    predictor = TrajectoryPredictor(model_path, device=device, batch_size=batch_size,
                                    expected_sha256=expected_sha256)
    return predictor.predict_trajectory(csv_path, measurement_type=measurement_type)
