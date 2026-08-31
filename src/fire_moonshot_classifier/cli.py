"""Command-line interface for the trajectory-based autopilot classifier."""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence


def _package_version() -> str:
    try:
        return version("fire-moonshot-classifier")
    except PackageNotFoundError:
        return "0.1.0"


def _add_cache_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Feature-cache directory (default: existing package cache directory).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fireclassify",
        description="Build trajectory features and train the FIRE autopilot classifiers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    feature_build = commands.add_parser(
        "feature-build",
        help="Build DWT and sequence feature caches from flight trajectories.",
    )
    inputs = feature_build.add_mutually_exclusive_group()
    inputs.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        help="Existing classifier dataset root(s), such as data/260615_sitl_logs.",
    )
    inputs.add_argument(
        "--trajectory",
        action="append",
        default=None,
        metavar="LABEL=CSV",
        help=(
            "Labeled FireTrack trajectory.csv; repeat for multiple runs "
            "(labels: px4, ardupilot, cogni)."
        ),
    )
    feature_build.add_argument(
        "--dataset-name",
        default="firetrack",
        help="Output cache name for --trajectory inputs (default: firetrack).",
    )
    feature_build.add_argument(
        "--source",
        choices=("auto", "sitl", "real"),
        default="auto",
        help="Dataset source for --logs (default: infer from path/layout).",
    )
    feature_build.add_argument(
        "--measurement-type",
        choices=("mocap", "vision"),
        default=None,
        help=(
            "Position columns for real CSVs. Defaults to mocap for dataset roots "
            "and vision for FireTrack trajectory.csv."
        ),
    )
    feature_build.add_argument(
        "--target-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        help=(
            "Selected kinematic features, space- or comma-separated "
            "(default: XY-Accel XY-Jerk Curvature)."
        ),
    )
    feature_build.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit run folders for a smoke test.",
    )
    _add_cache_dir(feature_build)
    feature_build.set_defaults(handler=_run_feature_build)

    train = commands.add_parser(
        "train",
        help="Train an SVM or DIVERSIFY model from feature caches.",
    )
    models = train.add_subparsers(dest="model", required=True)

    svm = models.add_parser("svm", help="Train the DWT + RBF-SVM baseline.")
    svm.add_argument("--sitl-folder", default=None, help="SITL cache dataset name.")
    svm.add_argument("--real-folders", nargs="+", default=None)
    svm.add_argument("--no-real", action="store_true", help="Skip real-flight evaluation.")
    _add_cache_dir(svm)
    svm.set_defaults(handler=_run_svm)

    diversify = models.add_parser("diversify", help="Train the cached-sequence DIVERSIFY model.")
    diversify.add_argument("--sitl-folder", default=None, help="SITL cache dataset name.")
    diversify.add_argument("--real-folders", nargs="+", default=None)
    diversify.add_argument("--epochs", type=int, default=None)
    diversify.add_argument("--local-epochs", type=int, default=None)
    diversify.add_argument("--batch-size", type=int, default=None)
    diversify.add_argument("--lr", type=float, default=None)
    diversify.add_argument("--no-wandb", action="store_true")
    _add_cache_dir(diversify)
    diversify.set_defaults(handler=_run_diversify)

    return parser


def _run_feature_build(args: argparse.Namespace) -> int:
    from fire_moonshot_classifier.datamanager import config
    from fire_moonshot_classifier.workflows.feature_build import (
        build_dataset_cache,
        build_default_caches,
        build_labeled_trajectory_cache,
        normalize_feature_names,
        parse_labeled_trajectory,
    )

    cache_dir = args.cache_dir or config.CACHE_DIR
    features = normalize_feature_names(args.target_features)

    if args.trajectory:
        trajectories = [parse_labeled_trajectory(value) for value in args.trajectory]
        build_labeled_trajectory_cache(
            trajectories,
            dataset_name=args.dataset_name,
            measurement_type=args.measurement_type or "vision",
            target_features=features,
            cache_dir=cache_dir,
        )
        return 0

    if args.logs:
        for logs in args.logs:
            build_dataset_cache(
                logs,
                source=args.source,
                measurement_type=args.measurement_type or "mocap",
                target_features=features,
                max_runs=args.max_runs,
                cache_dir=cache_dir,
            )
        return 0

    build_default_caches(
        target_features=features,
        max_runs=args.max_runs,
        cache_dir=cache_dir,
        measurement_type=args.measurement_type or "mocap",
    )
    return 0


def _run_svm(args: argparse.Namespace) -> int:
    from fire_moonshot_classifier.datamanager import config
    from fire_moonshot_classifier.training.train_svm import train_svm

    return train_svm(
        sitl_folder=args.sitl_folder or config.SITL_FOLDER,
        cache_dir=args.cache_dir or config.CACHE_DIR,
        real_folders=args.real_folders or config.REAL_FLIGHT_FOLDERS,
        evaluate_real=not args.no_real,
    )


def _run_diversify(args: argparse.Namespace) -> int:
    from fire_moonshot_classifier.training import train_diversify

    forwarded: list[str] = []
    for option, value in (
        ("--sitl-folder", args.sitl_folder),
        ("--cache-dir", args.cache_dir),
        ("--epochs", args.epochs),
        ("--local-epochs", args.local_epochs),
        ("--batch-size", args.batch_size),
        ("--lr", args.lr),
    ):
        if value is not None:
            forwarded.extend((option, str(value)))
    if args.real_folders:
        forwarded.append("--real-folders")
        forwarded.extend(args.real_folders)
    if args.no_wandb:
        forwarded.append("--no-wandb")
    result = train_diversify.main(forwarded)
    return int(result or 0)


def _normalize_train_alias(argv: Sequence[str]) -> list[str]:
    """Support the requested ``train --svm`` and ``train --diversify`` aliases."""
    normalized = list(argv)
    if not normalized or normalized[0] != "train":
        return normalized

    aliases = [flag for flag in ("--svm", "--diversify") if flag in normalized]
    if len(aliases) > 1:
        raise ValueError("Choose exactly one of --svm or --diversify.")
    if aliases:
        flag = aliases[0]
        normalized.remove(flag)
        normalized.insert(1, flag.removeprefix("--"))
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        normalized = _normalize_train_alias(raw_argv)
        args = parser.parse_args(normalized)
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
