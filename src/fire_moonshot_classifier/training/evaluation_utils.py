import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

def plot_confusion_matrix(y_true, y_pred, target_names, model_name="SVM", save_dir="results/svm_figs"):
    """
    Visualizes the model's prediction results as a Confusion Matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"{model_name} Classification Confusion Matrix", fontweight='bold')
    
    plt.tight_layout()
    file_path = f"{save_dir}/{model_name.lower()}_confusion_matrix.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"[Info] Saved Confusion Matrix plot to '{file_path}'")

def plot_pca_2d_projection(X_train, X_test, y_train, y_test, X_real=None, y_real=None, y_real_pred=None, model_name="SVM", save_dir="results/svm_figs"):
    """
    Visualizes the distribution of the DWT feature space and actual prediction results 
    by projecting them onto a 2D PCA plane.

    PCA signs are aligned so that the ArduPilot centroid lies in the positive
    direction from the PX4 centroid. A 10th–90th percentile view and a
    complete-range view are saved as separate images.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_colors = {0: 'tab:green', 1: 'tab:orange', 2: 'tab:blue'}
    label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # A PCA axis has an arbitrary sign. Align both axes to make plots from
    # different runs visually comparable.
    class_delta = (
        X_train_pca[np.asarray(y_train) == 1].mean(axis=0)
        - X_train_pca[np.asarray(y_train) == 0].mean(axis=0)
    )
    axis_signs = np.where(class_delta < 0, -1.0, 1.0)
    X_train_pca *= axis_signs
    X_test_pca *= axis_signs

    X_real_pca = None
    if X_real is not None and len(X_real) > 0 and y_real is not None and y_real_pred is not None:
        X_real_pca = pca.transform(X_real) * axis_signs

    projected_sets = [X_train_pca, X_test_pca]
    if X_real_pca is not None:
        projected_sets.append(X_real_pca)
    all_projected = np.vstack(projected_sets)

    def scatter_projection(ax, show_labels):
        # 1. SITL Data (Train / Test)
        for cls in [0, 1]:
            mask_train = (y_train == cls)
            train_label = f'SITL {label_map[cls]} (Train)' if show_labels else None
            ax.scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1],
                       c=class_colors[cls], alpha=0.15, marker='o', label=train_label)

            mask_test = (y_test == cls)
            test_label = f'SITL {label_map[cls]} (Test)' if show_labels else None
            ax.scatter(X_test_pca[mask_test, 0], X_test_pca[mask_test, 1],
                       c=class_colors[cls], alpha=0.6, marker='o', s=80,
                       edgecolor='k', label=test_label)

        # 2. Real Flight Data (Matches / Fails)
        if X_real_pca is None:
            return

        y_real_array = np.asarray(y_real)
        y_real_pred_array = np.asarray(y_real_pred)
        for cls in [0, 1, 2]:
            cls_mask = (y_real_array == cls)
            if not np.any(cls_mask):
                continue

            correct_mask = cls_mask & (y_real_array == y_real_pred_array)
            if np.any(correct_mask):
                match_label = f'Real {label_map[cls]} (Match)' if show_labels else None
                ax.scatter(X_real_pca[correct_mask, 0], X_real_pca[correct_mask, 1],
                           c=class_colors[cls], marker='*', s=350, edgecolor='k', linewidths=0.8,
                           zorder=5, label=match_label)

            incorrect_mask = cls_mask & (y_real_array != y_real_pred_array)
            if np.any(incorrect_mask):
                fail_label = f'Real {label_map[cls]} (Fail)' if show_labels else None
                ax.scatter(X_real_pca[incorrect_mask, 0], X_real_pca[incorrect_mask, 1],
                           c=class_colors[cls], marker='X', s=200, edgecolor='red', linewidths=1.5,
                           zorder=6, label=fail_label)

    # Focus on the central 80%. Add padding so boundary points are not cut in
    # half. A second image below retains every outlier.
    robust_low = np.percentile(all_projected, 10, axis=0)
    robust_high = np.percentile(all_projected, 90, axis=0)
    robust_padding = np.maximum((robust_high - robust_low) * 0.08, 1e-6)

    xlabel = f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    ylabel = f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"

    plot_specs = [
        {
            "suffix": "percentile_10_90",
            "title": f"PCA 2D Projection ({model_name}) — 10th–90th Percentile",
            "xlim": (
                robust_low[0] - robust_padding[0],
                robust_high[0] + robust_padding[0],
            ),
            "ylim": (
                robust_low[1] - robust_padding[1],
                robust_high[1] + robust_padding[1],
            ),
        },
        {
            "suffix": "full_range",
            "title": f"PCA 2D Projection ({model_name}) — Full Range",
            "xlim": None,
            "ylim": None,
        },
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 7))
        scatter_projection(ax, show_labels=True)
        if spec["xlim"] is not None:
            ax.set_xlim(*spec["xlim"])
            ax.set_ylim(*spec["ylim"])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(spec["title"], fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, frameon=True)
        plt.tight_layout()

        file_path = f"{save_dir}/{model_name.lower()}_pca_scatter_{spec['suffix']}.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Info] Saved PCA Scatter plot to '{file_path}'")

def print_detailed_prediction_map(y_real, y_real_pred, runs_real):
    """
    Outputs the detailed prediction results of real flight data in a console table format.
    """
    label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}
    
    print("\n[Detailed Prediction Map]")
    print("-" * 85)
    print(f"{'No.':<4} | {'Run Folder':<30} | {'True Label':<12} | {'Prediction':<12} | {'Status'}")
    print("-" * 85)
    
    for i in range(len(y_real)):
        run_name = runs_real[i]
        true_name = label_map.get(y_real[i], "Unknown")
        pred_name = label_map.get(y_real_pred[i], "Unknown")
        status = "Match" if true_name == pred_name else "Fail"
        print(f"{i+1:<4} | {run_name:<30} | {true_name:<12} | {pred_name:<12} | {status}")
    print("-" * 85)
