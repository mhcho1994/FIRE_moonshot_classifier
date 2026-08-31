from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fire_moonshot_classifier.datamanager import config
from fire_moonshot_classifier.training.evaluation_utils import (
    plot_confusion_matrix,
    plot_pca_2d_projection,
    print_detailed_prediction_map,
)


def save_real_flight_results(
    y_true,
    y_pred,
    runs,
    output_dir: Path = Path("results/svm_statistics"),
) -> Path:
    """Save real-flight prediction statistics and per-sample results as text."""
    label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_flight_classification.txt"

    correct_count = int(np.sum(y_pred == y_true))
    lines = [
        "SVM Real Flight Classification Results",
        "=" * 85,
        f"Accuracy: {(correct_count / len(y_true)) * 100:.2f}% "
        f"({correct_count}/{len(y_true)})",
        "",
        "Classification Report",
        "-" * 85,
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["PX4", "ArduPilot"],
            zero_division=0,
        ).rstrip(),
        "",
        "Detailed Prediction Map",
        "-" * 85,
        f"{'No.':<4} | {'Run Folder':<30} | {'True Label':<12} | "
        f"{'Prediction':<12} | {'Status'}",
        "-" * 85,
    ]

    for i, (true_label, pred_label, run_name) in enumerate(
        zip(y_true, y_pred, runs), start=1
    ):
        true_name = label_map.get(int(true_label), "Unknown")
        pred_name = label_map.get(int(pred_label), "Unknown")
        status = "Match" if true_label == pred_label else "Fail"
        lines.append(
            f"{i:<4} | {str(run_name):<30} | {true_name:<12} | "
            f"{pred_name:<12} | {status}"
        )

    lines.append("-" * 85)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Info] Saved real flight classification results to '{output_path}'")
    return output_path


def train_svm(
    *,
    sitl_folder: str = config.SITL_FOLDER,
    cache_dir: Path = config.CACHE_DIR,
    real_folders: Sequence[str] = config.REAL_FLIGHT_FOLDERS,
    evaluate_real: bool = True,
) -> int:
    """Run the existing DWT + RBF-SVM training and evaluation workflow."""
    cache_dir = Path(cache_dir)
    sitl_cache = cache_dir / f"{sitl_folder}_features.npz"
    if not sitl_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {sitl_cache}. Run `fireclassify feature-build` first."
        )

    print("\n[Info] Loading cached DWT features...")
    with np.load(sitl_cache) as sitl_data:
        X_sitl = sitl_data["X"].copy()
        y_sitl = sitl_data["y"].copy()
    print(f"  -> SITL Features Loaded (Shape: {X_sitl.shape})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sitl, y_sitl, test_size=0.1, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[Info] Training SVM model...")
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    target_names = ["PX4", "ArduPilot"]

    print("\n================ SITL Classification Results ================")
    print(f"Accuracy on SITL Test Data: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )

    if not evaluate_real:
        print("\n[Info] Skipping real flight evaluation as requested.")
        return 0

    X_real_list, y_real_list, runs_real_list = [], [], []
    print("\n[Info] Loading cached Real Flight features...")
    for folder in real_folders:
        real_cache = cache_dir / f"{folder}_features.npz"
        if not real_cache.exists():
            continue
        with np.load(real_cache) as real_data:
            X_real_list.append(real_data["X"].copy())
            y_real_list.append(real_data["y"].copy())
            runs_real_list.append(real_data["runs"].copy())
        print(f"  -> Loaded '{folder}' features.")

    if not X_real_list:
        print("\n[Warning] No real flight cache found. Skipping real flight evaluation.")
        plot_confusion_matrix(y_test, y_pred, target_names=target_names, model_name="SVM")
        plot_pca_2d_projection(
            X_train_scaled, X_test_scaled, y_train, y_test, model_name="SVM"
        )
        return 0

    X_real = np.vstack(X_real_list)
    y_real = np.hstack(y_real_list)
    runs_real = np.hstack(runs_real_list)
    X_real_scaled = scaler.transform(X_real)
    y_real_pred = model.predict(X_real_scaled)

    correct_count = int(np.sum(y_real_pred == y_real))
    print(
        f"Final Real Data Accuracy: {(correct_count / len(y_real)) * 100:.2f}% "
        f"({correct_count}/{len(y_real)})"
    )
    print_detailed_prediction_map(y_real, y_real_pred, runs_real)
    print("\n[Real Data Classification Report]")
    print(
        classification_report(
            y_real,
            y_real_pred,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )
    save_real_flight_results(y_real, y_real_pred, runs_real)
    plot_confusion_matrix(y_real, y_real_pred, target_names=target_names, model_name="SVM")
    plot_pca_2d_projection(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_real_scaled,
        y_real,
        y_real_pred,
        model_name="SVM",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train SVM on SITL DWT features and evaluate on real-flight caches."
    )
    parser.add_argument("--sitl-folder", default=config.SITL_FOLDER)
    parser.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR)
    parser.add_argument("--real-folders", nargs="+", default=config.REAL_FLIGHT_FOLDERS)
    parser.add_argument("--no-real", action="store_true")
    args = parser.parse_args(argv)
    return train_svm(
        sitl_folder=args.sitl_folder,
        cache_dir=args.cache_dir,
        real_folders=args.real_folders,
        evaluate_real=not args.no_real,
    )


if __name__ == "__main__":
    raise SystemExit(main())
