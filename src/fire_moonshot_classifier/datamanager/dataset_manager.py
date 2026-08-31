import os
from pathlib import Path

import fire_moonshot_classifier.datamanager.config as config
import fire_moonshot_classifier.processor.flight_segmenter as flight_segmenter
import fire_moonshot_classifier.processor.kinematic_processor as kinematic_processor
import numpy as np
# Import our modular pipeline components
from fire_moonshot_classifier.datamanager.data_extractor import parse_ardu_bin, parse_px4_ulog, parse_real_csv


def process_raw_trajectory(raw_data, class_label, run_name, target_features=None):
    """Convert one parsed trajectory into classified turn-feature sequences."""
    X_ts, T_ts, y, runs = [], [], [], []
    if raw_data is None:
        return X_ts, T_ts, np.asarray(y), runs

    selected_features = target_features or config.TARGET_FEATURES
    target_indices = [config.FEATURE_MAP[name] for name in selected_features]

    kinematic_features = kinematic_processor.compute_kinematics_pca(raw_data)
    if kinematic_features is None:
        return X_ts, T_ts, np.asarray(y), runs
    _, spans = flight_segmenter.extract_segments(kinematic_features)

    t_full, feat_full = kinematic_processor.compute_kinematics_diff(raw_data)
    if t_full is None or feat_full is None:
        return X_ts, T_ts, np.asarray(y), runs

    turn_spans = spans.get('turn_left', []) + spans.get('turn_right', [])
    for span in turn_spans:
        indices = np.where((t_full >= span[0]) & (t_full <= span[1]))[0]
        if len(indices) == 0:
            continue

        X_ts.append(feat_full[indices][:, target_indices])
        T_ts.append(t_full[indices] - t_full[indices][0])
        y.append(class_label)
        runs.append(str(run_name))

    return X_ts, T_ts, np.asarray(y), runs


def process_dataset_folder(
    base_folder,
    is_sitl=True,
    measurement_type='mocap',
    max_runs=None,
    target_features=None,
    data_root=Path("data"),
):
    """
    Crawls folders, runs the ETL pipeline (Extract -> Kinematics -> Segment), 
    and returns extracted Turn segments.
    """
    supplied_path = Path(base_folder).expanduser()
    base_dir = supplied_path if supplied_path.exists() else Path(data_root) / supplied_path
    dataset_name = base_dir.name
    X_ts, T_ts, y, runs = [], [], [], []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_ts, T_ts, np.array(y), runs

    run_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("run_") and (base_dir / f).is_dir()])
    
    if max_runs is not None:
        print(f"[Info] Limiting processing to {max_runs} runs for {base_folder} (out of {len(run_folders)}).")
        run_folders = run_folders[:max_runs]
    
    for run_folder in run_folders:
        run_dir = base_dir / run_folder
        
        for fw_config in config.get_fw_configs(is_sitl):
            # Skip Cogni if it's SITL data
            if is_sitl and fw_config.name == 'cogni': continue
            
            fw_dir = config.find_fw_dir(run_dir, fw_config.name, fw_config.sub_paths)
            if not fw_dir:
                continue
                
            for file in os.listdir(fw_dir):
                if file.lower().endswith(fw_config.ext):
                    file_path = str(fw_dir / file)
                    
                    # [Step 1: Extract Raw Data]
                    if is_sitl and fw_config.name == 'px4': raw_data = parse_px4_ulog(file_path)
                    elif is_sitl and fw_config.name == 'ardu': raw_data = parse_ardu_bin(file_path)
                    else: raw_data = parse_real_csv(file_path, measurement_type)
                    
                    if raw_data is None: continue
                    
                    # [Step 2-4: Kinematics -> Segment -> selected turn features]
                    X_run, T_run, y_run, runs_run = process_raw_trajectory(
                        raw_data,
                        fw_config.class_label,
                        f"{dataset_name}/{run_folder}",
                        target_features=target_features,
                    )
                    X_ts.extend(X_run)
                    T_ts.extend(T_run)
                    y.extend(y_run.tolist())
                    runs.extend(runs_run)
                    count_in_run = len(X_run)
                            
                    if count_in_run > 0:
                        print(f"    - {run_folder} [{fw_config.name.upper()}]: {count_in_run} turn segments extracted.")
                    break # Process only the first valid file per firmware

    return X_ts, T_ts, np.array(y), runs
