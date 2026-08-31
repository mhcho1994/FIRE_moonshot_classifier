from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from fire_moonshot_classifier.datamanager import config, dataset_manager
from fire_moonshot_classifier.datamanager.data_extractor import parse_real_csv
from fire_moonshot_classifier.processor.feature_builder import (
    compute_dwt_statistics,
    pad_sequences,
)


LABEL_MAP = {
    "px4": 0,
    "ardu": 1,
    "ardupilot": 1,
    "cogni": 2,
    "cognipilot": 2,
}


def normalize_feature_names(values: Sequence[str] | None) -> list[str]:
    """Normalize comma- or space-separated feature names to configured IDs."""
    if not values:
        return list(config.TARGET_FEATURES)

    requested = [part.strip() for value in values for part in value.split(",") if part.strip()]
    canonical = {name.casefold(): name for name in config.FEATURE_MAP}
    unknown = [name for name in requested if name.casefold() not in canonical]
    if unknown:
        available = ", ".join(config.FEATURE_MAP)
        raise ValueError(
            f"Unknown target feature(s): {', '.join(unknown)}. Available: {available}"
        )
    return [canonical[name.casefold()] for name in requested]


def parse_labeled_trajectory(value: str) -> tuple[Path, int]:
    """Parse LABEL=PATH used for direct FireTrack trajectory integration."""
    try:
        raw_label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid trajectory specification {value!r}; expected LABEL=PATH"
        ) from exc

    label_name = raw_label.strip().casefold()
    if label_name not in LABEL_MAP:
        raise ValueError(
            f"Unknown trajectory label {raw_label!r}; expected px4, ardupilot, or cogni"
        )
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    return path, LABEL_MAP[label_name]


def _cache_path(cache_dir: Path, dataset_name: str) -> Path:
    suffix = "" if dataset_name.endswith("_features") else "_features"
    return cache_dir / f"{dataset_name}{suffix}.npz"


def _save_feature_cache(
    sequences: Sequence[np.ndarray],
    labels: Sequence[int] | np.ndarray,
    runs: Sequence[str],
    *,
    cache_dir: Path,
    dataset_name: str,
    filter_invalid: bool,
    feature_names: Sequence[str],
) -> Path | None:
    if not sequences:
        print(f"[Warning] No valid features extracted for {dataset_name}.")
        return None

    dwt = np.asarray(
        [
            compute_dwt_statistics(sequence, config.WAVELET_NAME, config.WAVELET_LEVEL)
            for sequence in sequences
        ]
    )
    padded = pad_sequences(sequences)
    labels_array = np.asarray(labels)
    runs_array = np.asarray(runs)

    if filter_invalid:
        valid = np.isfinite(dwt).all(axis=1)
        removed = int(len(dwt) - valid.sum())
        if removed:
            print(f"[Warning] Removed {removed} corrupted segments containing non-finite values.")
        dwt = dwt[valid]
        padded = padded[valid]
        labels_array = labels_array[valid]
        runs_array = runs_array[valid]

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = _cache_path(cache_dir, dataset_name)
    np.savez(
        output,
        X=dwt,
        X_seq=padded,
        y=labels_array,
        runs=runs_array,
        feature_names=np.asarray(feature_names),
    )
    print(
        f"[Success] Cached {dataset_name}: {output} "
        f"(X={dwt.shape}, X_seq={padded.shape})"
    )
    return output


def _infer_sitl(path: Path) -> bool:
    if "sitl" in path.name.casefold():
        return True
    if any(path.rglob("*.csv")):
        return False
    return any(path.rglob("*.ulg")) or any(path.rglob("*.bin"))


def build_dataset_cache(
    logs: Path | str,
    *,
    source: str = "auto",
    measurement_type: str = "mocap",
    target_features: Sequence[str] | None = None,
    max_runs: int | None = None,
    cache_dir: Path = config.CACHE_DIR,
) -> Path | None:
    """Build one cache from the repository's existing run-folder layout."""
    logs_path = Path(logs).expanduser()
    if not logs_path.exists():
        candidate = Path("data") / logs_path
        if candidate.exists():
            logs_path = candidate
        else:
            raise FileNotFoundError(f"Log dataset not found: {logs}")

    is_sitl = _infer_sitl(logs_path) if source == "auto" else source == "sitl"
    selected = normalize_feature_names(target_features)
    print(
        f"\n[Info] Building {logs_path.name} "
        f"(source={'sitl' if is_sitl else 'real'}, features={selected})"
    )
    sequences, _, labels, runs = dataset_manager.process_dataset_folder(
        logs_path,
        is_sitl=is_sitl,
        measurement_type=measurement_type,
        max_runs=max_runs,
        target_features=selected,
    )
    return _save_feature_cache(
        sequences,
        labels,
        runs,
        cache_dir=Path(cache_dir),
        dataset_name=logs_path.name,
        filter_invalid=not is_sitl,
        feature_names=selected,
    )


def build_default_caches(
    *,
    target_features: Sequence[str] | None = None,
    max_runs: int | None = None,
    cache_dir: Path = config.CACHE_DIR,
    measurement_type: str = "mocap",
) -> list[Path]:
    """Preserve the original build_features.py behavior."""
    outputs = []
    sitl = build_dataset_cache(
        config.SITL_FOLDER,
        source="sitl",
        target_features=target_features,
        max_runs=max_runs,
        cache_dir=cache_dir,
    )
    if sitl is not None:
        outputs.append(sitl)

    for folder in config.REAL_FLIGHT_FOLDERS:
        try:
            output = build_dataset_cache(
                folder,
                source="real",
                measurement_type=measurement_type,
                target_features=target_features,
                max_runs=max_runs,
                cache_dir=cache_dir,
            )
        except FileNotFoundError as exc:
            print(f"[Warning] {exc}")
            continue
        if output is not None:
            outputs.append(output)
    return outputs


def build_labeled_trajectory_cache(
    trajectories: Iterable[tuple[Path, int]],
    *,
    dataset_name: str,
    measurement_type: str = "vision",
    target_features: Sequence[str] | None = None,
    cache_dir: Path = config.CACHE_DIR,
) -> Path | None:
    """Build a cache directly from labeled FireTrack trajectory.csv outputs."""
    selected = normalize_feature_names(target_features)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[str] = []

    for csv_path, label in trajectories:
        print(f"\n[Info] Processing trajectory: {csv_path}")
        raw_data = parse_real_csv(str(csv_path), measurement_type=measurement_type)
        run_name = f"{dataset_name}/{csv_path.parent.name}"
        X_run, _, y_run, runs_run = dataset_manager.process_raw_trajectory(
            raw_data,
            label,
            run_name,
            target_features=selected,
        )
        print(f"    - {len(X_run)} turn segments extracted.")
        sequences.extend(X_run)
        labels.extend(y_run.tolist())
        runs.extend(runs_run)

    return _save_feature_cache(
        sequences,
        labels,
        runs,
        cache_dir=Path(cache_dir),
        dataset_name=dataset_name,
        filter_invalid=True,
        feature_names=selected,
    )
