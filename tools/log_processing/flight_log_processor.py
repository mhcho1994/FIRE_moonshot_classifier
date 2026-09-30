#!/usr/bin/env python3
"""Organize flight_logs/source into run_XXX/{ardu,px4,cogni}_logs.

See tools/log_processing/README.md for source precedence, output schemas,
metadata matching, optional raw-log dependencies, and CLI examples.
The legacy --ardu-logs-dir ROS-bag-to-CSV interface remains available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml



def _find_rosbag2_dirs(raw_dir: Path) -> list[Path]:
    """
    Find rosbag2 bag directories under a raw log directory.

    A rosbag2 bag directory is assumed to contain:
    - metadata.yaml
    - one or more *.mcap files

    Parameters
    ----------
    raw_dir : Path
        Directory such as /data/flight_logs/ardu_logs/raw

    Returns
    -------
    list[Path]
        Sorted list of bag directories.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []

    bag_dirs: list[Path] = []
    for metadata_file in raw_dir.rglob("metadata.yaml"):
        bag_dir = metadata_file.parent
        if any(bag_dir.glob("*.mcap")):
            bag_dirs.append(bag_dir)

    return sorted(set(bag_dirs))


def _topic_name_to_filename(topic_name: str) -> str:
    """
    Convert a ROS topic name into a safe CSV filename suffix.

    Example
    -------
    /camera/odom -> camera_odom
    """
    return topic_name.strip("/").replace("/", "_")


def _read_rosbag2_topic_rows(
    bag_dir: Path,
    topic_name: str,
) -> list[dict[str, float]]:
    """
    Read one Odometry topic from a rosbag2 bag directory.

    Parameters
    ----------
    bag_dir : Path
        Path to rosbag2 bag directory containing metadata.yaml.
    topic_name : str
        Topic name to extract, e.g. /camera/odom.

    Returns
    -------
    list[dict[str, float]]
        Extracted rows. Each row contains:
        - bag_time_ns
        - header_stamp_ns
        - x, y, z
        - qx, qy, qz, qw
        - vx, vy, vz
        - wx, wy, wz
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="mcap",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic_name not in type_map:
        return []

    msg_type = get_message(type_map[topic_name])

    rows: list[dict[str, float]] = []

    while reader.has_next():
        current_topic, data, bag_time_ns = reader.read_next()
        if current_topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        header_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

        rows.append(
            {
                "bag_time_ns": int(bag_time_ns),
                "header_stamp_ns": header_stamp_ns,
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "z": float(msg.pose.pose.position.z),
                "qx": float(msg.pose.pose.orientation.x),
                "qy": float(msg.pose.pose.orientation.y),
                "qz": float(msg.pose.pose.orientation.z),
                "qw": float(msg.pose.pose.orientation.w),
                "vx": float(msg.twist.twist.linear.x),
                "vy": float(msg.twist.twist.linear.y),
                "vz": float(msg.twist.twist.linear.z),
                "wx": float(msg.twist.twist.angular.x),
                "wy": float(msg.twist.twist.angular.y),
                "wz": float(msg.twist.twist.angular.z),
            }
        )

    return rows


def _write_rows_to_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    """
    Write extracted rows to CSV.

    Parameters
    ----------
    csv_path : Path
        Output CSV file path.
    rows : list[dict[str, float]]
        Rows to write.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Create an empty file with header for consistency.
        fieldnames = [
            "bag_time_ns",
            "header_stamp_ns",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_rosbag2_mocap_to_csv(
    ardu_logs_dir: Path,
    topic_names: Iterable[str] = ("/camera/odom", "/ardupilot/odom"),
    skip_existing: bool = False,
) -> list[Path]:
    """
    Extract mocap/odometry topics from rosbag2 MCAP bags into CSV files.

    Directory convention
    --------------------
    Input:
        <ardu_logs_dir>/raw/<bag_dir>/metadata.yaml
        <ardu_logs_dir>/raw/<bag_dir>/*.mcap

    Output:
        <ardu_logs_dir>/processed/<bag_dir>__<topic>.csv

    Parameters
    ----------
    ardu_logs_dir : Path
        Path such as /data/flight_logs/ardu_logs
    topic_names : Iterable[str], optional
        Topic names to extract.
    skip_existing : bool, optional
        Skip writing if all expected CSV files already exist.

    Returns
    -------
    list[Path]
        Paths of generated CSV files.
    """
    ardu_logs_dir = Path(ardu_logs_dir)
    raw_dir = ardu_logs_dir / "raw"
    processed_dir = ardu_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bag_dirs = _find_rosbag2_dirs(raw_dir)
    if not bag_dirs:
        print(f"[WARN] No rosbag2 MCAP directories found in: {raw_dir}")
        return []

    generated: list[Path] = []

    for bag_dir in bag_dirs:
        bag_name = bag_dir.name
        print(f"[INFO] Processing rosbag2 bag: {bag_dir}")

        expected_paths = [
            processed_dir / f"{bag_name}__{_topic_name_to_filename(topic)}.csv"
            for topic in topic_names
        ]

        if skip_existing and all(path.exists() for path in expected_paths):
            print(f"[SKIP] Existing mocap CSVs found for: {bag_name}")
            generated.extend(expected_paths)
            continue

        for topic_name in topic_names:
            rows = _read_rosbag2_topic_rows(bag_dir=bag_dir, topic_name=topic_name)
            out_csv = processed_dir / f"{bag_name}__{_topic_name_to_filename(topic_name)}.csv"

            if not rows:
                print(f"[WARN] Topic not found or empty: {topic_name} in {bag_dir}")
                continue

            _write_rows_to_csv(out_csv, rows)
            print(f"[INFO] Wrote {len(rows)} rows -> {out_csv}")
            generated.append(out_csv)

    return generated


# Dataset organization -------------------------------------------------------

SOURCE_ORDER = (
    'vision/SAM3_pipeline', 'vision/CalibOpt_pipeline', 'raw/fc_logs',
    'raw/videos_and_mocap_logs', 'raw/mocap_logs',
)
EXTENSIONS = {'.npz', '.csv', '.ulg', '.bin', '.tsv', '.mcap'}
RUN_COLUMNS = ('run', 'run_id', 'run_number', 'sortie_name')
TIME_COLUMNS = ('times_s', 'time_s', 'mocap_time_s', 'timestamp', 't')
POSITION_COLUMNS = {
    'trajectory_raw': (('x_raw', 'y_raw', 'z_raw'), ('x', 'y', 'z')),
    'trajectory_smooth': (('x_smooth', 'y_smooth', 'z_smooth'), ('xsmooth', 'ysmooth', 'zsmooth')),
    'gt_drone': (('gt_x', 'gt_y', 'gt_z'), ('gtx', 'gty', 'gtz')),
}


def warn(message):
    print(f'[WARN] {message}', file=sys.stderr)


def normalized(value):
    value = str(value).lower().replace('arduipilot', 'ardu').replace('ardupilot', 'ardu')
    value = value.replace('cognipilot', 'cogni')
    return re.sub(r'[^a-z0-9]+', '_', value).strip('_')


def infer_autopilot(value):
    value = normalized(value)
    matches = [name for name in ('ardu', 'px4', 'cogni') if name in value]
    return matches[0] if len(matches) == 1 else None


def sortie_key(value):
    value = normalized(value)
    # Remove known export suffixes, never arbitrary numeric suffixes.
    return re.sub(r'(?:_(?:trajectory|mocap|data|6d|logs?))+$', '', value)


@dataclass(frozen=True)
class Sortie:
    id: int
    name: str
    autopilot: str
    source_path: str | None = None
    topic: str | None = None
    body: str | None = None


@dataclass(frozen=True)
class Candidate:
    path: Path
    root: Path
    priority: int
    run: str | None = None

    @property
    def parts(self):
        rel = self.path.relative_to(self.root)
        return [*rel.parts[:-1], rel.stem]

    @property
    def autopilot(self):
        return infer_autopilot('/'.join(self.parts))

    @property
    def name(self):
        if self.run is not None:
            name = sortie_key(self.run)
            if not infer_autopilot(name) and self.autopilot:
                name = f'{self.autopilot}_{name}'
            return name
        for part in reversed(self.parts):
            name = sortie_key(part)
            if re.search(r'(?:run|test|traj|points)_?\d+', name) or infer_autopilot(name):
                if not infer_autopilot(name) and self.autopilot:
                    name = f'{self.autopilot}_{name}'
                return name
        return sortie_key(self.path.stem)


def read_metadata(path):
    if not path.exists():
        warn(f'No metadata: {path}; inferring sorties and autopilots from paths.')
        return None, None
    data = yaml.safe_load(path.read_text()) or {}
    sorties = []
    for entry in data.get('trajectory_sets', []):
        legacy = next((k for k in ('ardupilot_run', 'px4_run', 'cognipilot_run') if k in entry), None)
        name = entry.get('sortie_name') or (entry[legacy] if legacy else None)
        autopilot = infer_autopilot(entry.get('autopilot') or legacy or name or '')
        ident = entry.get('id')
        if not name or not autopilot or type(ident) is not int or ident < 0:
            raise ValueError(f'Invalid trajectory_sets entry (need id, sortie_name, autopilot): {entry}')
        sorties.append(Sortie(ident, str(name), autopilot, entry.get('source_path'), entry.get('topic'), entry.get('body')))
    if len({s.id for s in sorties}) != len(sorties):
        raise ValueError('Duplicate trajectory_sets ids would overwrite run directories.')
    if len({(s.autopilot, sortie_key(s.name)) for s in sorties}) != len(sorties):
        raise ValueError('Duplicate sortie names for the same autopilot.')
    total = data.get('runs', {}).get('total')
    if total is not None and (type(total) is not int or total < 0):
        raise ValueError('runs.total must be a nonnegative integer.')
    if total is not None and total != len(sorties):
        warn(f'runs.total={total}, but metadata defines {len(sorties)} sorties.')
    return sorties, total


def read_table(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate CSV column names')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty CSV')
    numeric_fields = set(TIME_COLUMNS) | {'frame', 'mocap_frame', 'n_views', 'reproj_px', 'reproj_errors_px'}
    numeric_fields.update(c for groups in POSITION_COLUMNS.values() for group in groups for c in group)
    result = {}
    for field in fields:
        values = [row.get(field, '') or '' for row in rows]
        if field in RUN_COLUMNS or field == 'used_cameras':
            result[field] = np.asarray(values, dtype=str)
        else:
            try:
                result[field] = np.asarray([float(v) if v.strip() else np.nan for v in values])
            except ValueError as exc:
                if field in numeric_fields:
                    raise ValueError(f'Invalid numeric column {field!r}: {exc}') from exc
                result[field] = np.asarray(values, dtype=str)
    return result


def read_arrays(path):
    if path.suffix.lower() == '.csv':
        return read_table(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def canonical_arrays(arrays):
    arrays = dict(arrays)
    time_key = next((k for k in TIME_COLUMNS if k in arrays), None)
    if time_key is None:
        raise ValueError('No supported time column (times_s/time_s/mocap_time_s/timestamp/t)')
    times = np.asarray(arrays[time_key], dtype=float)
    if times.ndim != 1 or not len(times):
        raise ValueError('Time must be a nonempty 1D array')
    arrays['times_s'] = times
    for key in (*RUN_COLUMNS, 'frame', 'mocap_frame', 'n_views', 'reproj_px', 'reproj_errors_px', 'used_cameras'):
        if key in arrays and arrays[key].shape != times.shape:
            raise ValueError(f'{key} must have the same 1D shape as time')
    usable = False
    for target, alternatives in POSITION_COLUMNS.items():
        if target not in arrays:
            columns = next((cols for cols in alternatives if all(c in arrays for c in cols)), None)
            if columns:
                arrays[target] = np.column_stack([arrays[c] for c in columns]).astype(float)
        if target in arrays:
            pos = np.asarray(arrays[target], dtype=float)
            if pos.shape != (len(times), 3):
                raise ValueError(f'{target} must have shape ({len(times)}, 3), got {pos.shape}')
            usable |= bool(np.any(np.isfinite(times) & np.isfinite(pos).all(axis=1)))
    if not usable:
        raise ValueError('No finite time/position samples')
    if 'reproj_px' in arrays:
        arrays['reproj_errors_px'] = arrays['reproj_px']
    return arrays


def split_arrays(arrays, run):
    if run is None:
        return arrays
    key = next((k for k in RUN_COLUMNS if k in arrays), None)
    if key is None:
        raise ValueError('all_runs needs a run/run_id/run_number/sortie_name column')
    labels = np.asarray(arrays[key]).astype(str)
    mask = labels == run
    return {k: v[mask] if v.ndim and v.shape[0] == len(mask) else v for k, v in arrays.items()}


def discover(source):
    candidates = []
    for priority, subdir in enumerate(SOURCE_ORDER):
        root = source / subdir
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if priority < 2 and path.suffix.lower() not in ('.csv', '.npz'):
                continue
            if path.suffix.lower() in ('.csv', '.npz'):
                try:
                    arrays = read_arrays(path)
                    # Ignore phone GPS/IMU/calibration tables with no trajectory schema.
                    canonical_arrays(arrays)
                    key = next((k for k in RUN_COLUMNS if k in arrays), None)
                    if key:
                        labels = np.asarray(arrays[key]).astype(str)
                        if labels.ndim != 1:
                            raise ValueError('Run labels must be a 1D array')
                        for label in sorted(set(labels)):
                            if not label or label.lower() == 'nan':
                                warn(f'Ignoring empty run label in {path}')
                                continue
                            candidates.append(Candidate(path, root, priority, label))
                        continue
                    if 'all_runs' in path.stem.lower():
                        raise ValueError('all_runs file has no run identifier column')
                except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
                    warn(f'Ignoring {path}: {exc}')
                    continue
            candidates.append(Candidate(path, root, priority))
    return candidates


def match_score(sortie, candidate, source):
    if sortie.source_path:
        explicit = (source / sortie.source_path).resolve()
        if candidate.path.resolve() == explicit or explicit in candidate.path.resolve().parents:
            if candidate.run is None or sortie_key(candidate.name) == sortie_key(sortie.name):
                return 100
    if candidate.autopilot and candidate.autopilot != sortie.autopilot:
        return 0
    name = sortie_key(sortie.name)
    if candidate.run is not None:
        group = sortie_key(candidate.run)
        if group == name or sortie_key(candidate.name) == name:
            return 90
        # A group named run1 (or 1) belongs to the firmware named in all_runs.
        number = re.fullmatch(r'(?:run_?)?(\d+)(?:\.0)?', group)
        expected = re.fullmatch(r'(?:ardu|px4|cogni)_run_?(\d+)', name)
        if number and expected and candidate.autopilot == sortie.autopilot:
            return 80 if int(number[1]) == int(expected[1]) else 0
        return 0
    parts = [sortie_key(p) for p in candidate.parts]
    if name in parts:
        return 90
    # Nested layouts such as px4_runs_1_3/run1/run1_6D.tsv.
    if candidate.autopilot == sortie.autopilot:
        if any(f'{sortie.autopilot}_{part}' == name for part in parts):
            return 80
    return 0


def candidate_order(candidate):
    ext = candidate.path.suffix.lower()
    return (candidate.priority, {'.npz': 0, '.csv': 1, '.ulg': 2, '.bin': 2, '.tsv': 3, '.mcap': 4}[ext],
            0 if '6d' in candidate.path.stem.lower() else 1, str(candidate.path), candidate.run or '')


def raw_arrays(times, position, kind, frame, **extra):
    return canonical_arrays({
        'times_s': np.asarray(times, dtype=float),
        'trajectory_raw': np.asarray(position, dtype=float).reshape(-1, 3),
        'measurement_type': np.asarray(kind), 'coordinate_frame': np.asarray(frame),
        'position_unit': np.asarray('m'), **extra,
    })


def read_ulog(path):
    from pyulog import ULog
    data = ULog(str(path), message_name_filter_list=['vehicle_local_position']).get_dataset('vehicle_local_position').data
    position = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(float)
    for flag, axes in (('xy_valid', (0, 1)), ('z_valid', (2,))):
        if flag in data:
            position[np.ix_(~np.asarray(data[flag], dtype=bool), axes)] = np.nan
    return raw_arrays(data['timestamp'] / 1e6, position, 'fc', 'NED')


def read_bin(path, core=0):
    from pymavlink import mavutil
    log = mavutil.mavlink_connection(str(path))
    streams = {'XKF1': [], 'NKF1': []}
    try:
        while True:
            msg = log.recv_match(type=list(streams), blocking=False)
            if msg is None:
                break
            if getattr(msg, 'C', core) != core:
                continue
            streams[msg.get_type()].append((msg.TimeUS / 1e6, msg.PN, msg.PE, msg.PD))
    finally:
        log.close()
    rows = streams['XKF1'] or streams['NKF1']
    if not rows:
        raise ValueError(f'No XKF1/NKF1 positions for EKF core {core}')
    data = np.asarray(rows)
    return raw_arrays(data[:, 0], data[:, 1:], 'fc', 'NED', ekf_core=np.asarray(core))


def read_tsv(path, body='drone'):
    with path.open(encoding='utf-8-sig') as stream:
        lines = list(csv.reader(stream, delimiter='\t'))
    fields = {row[0]: row[1:] for row in lines if row}
    if 'FREQUENCY' not in fields:
        raise ValueError('Expected Qualisys TSV with FREQUENCY and body position headers')
    frequency = float(fields['FREQUENCY'][0])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('Invalid mocap frequency')
    # Select the named rigid body, never the first camera or an arbitrary marker.
    header = next(((i, row) for i, row in enumerate(lines)
                   if any(cell.strip().lower() == f'{body} x'.lower() for cell in row)), None)
    if header is None:
        raise ValueError(f'No rigid body {body!r}; use a 6D TSV or --mocap-body')
    index, names = header
    col = next(i for i, name in enumerate(names) if name.strip().lower() == f'{body} x'.lower())
    values = []
    for row in lines[index + 1:]:
        if not row or not any(cell.strip() for cell in row):
            continue
        block = [float(v) if v.strip() else np.nan for v in row[col:col + 6]]
        if len(block) < 3:
            raise ValueError('Truncated mocap body data')
        xyz = np.asarray(block[:3]) / 1000.0
        if np.all(np.asarray(block) == 0):
            xyz[:] = np.nan
        values.append(xyz)
    return raw_arrays(np.arange(len(values)) / frequency, values, 'mocap', 'Qualisys global',
                      gt_drone=np.asarray(values).reshape(-1, 3), body=np.asarray(body))


def read_mcap(path, autopilot, topic=None):
    """Read a standalone ROS 2 MCAP without depending on adjacent bag metadata."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    supported = {'nav_msgs/msg/Odometry', 'geometry_msgs/msg/PoseStamped', 'geometry_msgs/msg/PoseWithCovarianceStamped'}
    topics = {t.name: t.type for t in reader.get_all_topics_and_types() if t.type in supported}
    if topic is None:
        preferred = {'ardu': '/ardupilot/odom', 'px4': '/px4/odom', 'cogni': '/cognipilot/odom'}[autopilot]
        drone_topics = [t for t in topics if 'camera' not in t.lower() and 'cam/' not in t.lower()]
        if preferred in topics:
            topic = preferred
        elif len(drone_topics) == 1:
            topic = drone_topics[0]
        else:
            raise ValueError(f'Ambiguous/missing drone pose topic: {list(topics)}; specify --mcap-topic or metadata topic')
    if topic not in topics:
        raise ValueError(f'Pose topic {topic!r} not found; available: {list(topics)}')
    msg_type = get_message(topics[topic])
    positions, stamps, bag_stamps, frames = [], [], [], set()
    while reader.has_next():
        name, raw, bag_ns = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(raw, msg_type)
        pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
        positions.append((pose.position.x, pose.position.y, pose.position.z))
        stamps.append(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        bag_stamps.append(bag_ns)
        frames.add(msg.header.frame_id)
    if not positions:
        raise ValueError(f'Empty pose topic {topic}')
    if len(frames) > 1:
        raise ValueError(f'Pose topic changes coordinate frame: {frames}')
    use_header = all(t > 0 for t in stamps)
    stamp_ns = np.asarray(stamps if use_header else bag_stamps, dtype=np.int64)
    return raw_arrays((stamp_ns - stamp_ns[0]) / 1e9, positions, 'odometry', next(iter(frames)),
                      epoch_times_s=stamp_ns / 1e9, topic=np.asarray(topic),
                      timestamp_source=np.asarray('header' if use_header else 'bag'))


def load_candidate(candidate, sortie, *, mocap_body='drone', mcap_topic=None, ekf_core=0):
    ext = candidate.path.suffix.lower()
    if ext in ('.csv', '.npz'):
        return canonical_arrays(split_arrays(read_arrays(candidate.path), candidate.run))
    if ext == '.ulg':
        return read_ulog(candidate.path)
    if ext == '.bin':
        return read_bin(candidate.path, ekf_core)
    if ext == '.tsv':
        return read_tsv(candidate.path, sortie.body or mocap_body)
    if ext == '.mcap':
        return read_mcap(candidate.path, sortie.autopilot, sortie.topic or mcap_topic)
    raise ValueError(f'Unsupported extension {ext}')


def write_array_csv(path, arrays):
    columns = {'frame': arrays.get('frame', arrays.get('mocap_frame', np.arange(len(arrays['times_s'])))),
               'time_s': arrays['times_s']}
    for key in ('n_views', 'used_cameras'):
        if key in arrays:
            columns[key] = arrays[key]
    if 'reproj_errors_px' in arrays:
        columns['reproj_px'] = arrays['reproj_errors_px']
    for key, names in (('trajectory_raw', ('x_raw', 'y_raw', 'z_raw')),
                       ('trajectory_smooth', ('x_smooth', 'y_smooth', 'z_smooth')),
                       ('gt_drone', ('gt_x', 'gt_y', 'gt_z'))):
        if key in arrays:
            for axis, name in enumerate(names):
                columns[name] = arrays[key][:, axis]
    n = len(arrays['times_s'])
    # Preserve additional per-sample fields when splitting a merged file.
    for key, value in arrays.items():
        if value.ndim == 1 and len(value) == n and key not in columns and key != 'times_s':
            columns[key] = value
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def save_trajectory(destination, candidate, arrays, sortie):
    destination.mkdir(parents=True, exist_ok=True)
    # Stage both formats before publishing either; sources are always read-only.
    with tempfile.TemporaryDirectory(prefix='.trajectory-', dir=destination) as tmp:
        stage = Path(tmp)
        if candidate.path.suffix.lower() == '.npz' and candidate.run is None:
            shutil.copy2(candidate.path, stage / 'trajectory.npz')
        else:
            np.savez_compressed(stage / 'trajectory.npz', **arrays)
        if candidate.path.suffix.lower() == '.csv' and candidate.run is None:
            shutil.copy2(candidate.path, stage / 'trajectory.csv')
        else:
            write_array_csv(stage / 'trajectory.csv', arrays)
        provenance = {'source': str(candidate.path.resolve()), 'source_priority': SOURCE_ORDER[candidate.priority],
                      'source_run': candidate.run, 'sortie_name': sortie.name, 'id': sortie.id,
                      'autopilot': sortie.autopilot, 'samples': len(arrays['times_s'])}
        for key in ('measurement_type', 'coordinate_frame', 'position_unit', 'topic', 'body', 'ekf_core', 'timestamp_source'):
            if key in arrays and arrays[key].ndim == 0:
                provenance[key] = arrays[key].item()
        (stage / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        for path in stage.iterdir():
            path.replace(destination / path.name)


def process_flight_logs(flight_logs, *, output_root=None, dry_run=False, overwrite=False,
                        mocap_body='drone', mcap_topic=None, ekf_core=0):
    flight_logs = Path(flight_logs).resolve()
    source = flight_logs / 'source'
    if not source.is_dir():
        raise FileNotFoundError(f'No source directory: {source}')
    output_root = Path(output_root).resolve() if output_root else flight_logs
    if output_root == source or source in output_root.parents:
        raise ValueError('Output must be outside source to avoid changing source data')
    sorties, total = read_metadata(source / 'metadata.yaml')
    candidates = discover(source)
    if sorties is None:
        inferred = {}
        for candidate in candidates:
            if candidate.autopilot:
                inferred[(candidate.autopilot, candidate.name)] = None
            else:
                warn(f'Cannot infer autopilot; skipping {candidate.path}')
        sorties = [Sortie(i, name, ap) for i, (ap, name) in enumerate(sorted(inferred))]
    assigned = {s.id: [] for s in sorties}
    for candidate in candidates:
        scores = [(match_score(s, candidate, source), s) for s in sorties]
        best = max((score for score, _ in scores), default=0)
        matches = [s for score, s in scores if score == best and score > 0]
        if len(matches) == 1:
            assigned[matches[0].id].append(candidate)
        elif matches:
            warn(f'Ambiguous mapping: {candidate.path} run={candidate.run}; matches {[s.name for s in matches]}')
        else:
            warn(f'No metadata match: {candidate.path} run={candidate.run}')
    vision_files = {c.path for c in candidates if c.priority < 2}
    vision_sorties = sum(any(c.priority < 2 for c in assigned[s.id]) for s in sorties)
    print(f'[INFO] Vision: {len(vision_files)} files, {vision_sorties} matched sorties (CSV/NPZ pairs count once).')
    if total is not None and vision_sorties != total:
        warn(f'Vision contains {vision_sorties} matched sorties ({len(vision_files)} files), runs.total={total}; trying lower-priority sources for missing sorties.')
    results = []
    for sortie in sorted(sorties, key=lambda s: s.id):
        destination = output_root / f'run_{sortie.id:03d}' / f'{sortie.autopilot}_logs'
        if not overwrite and any((destination / name).exists() for name in ('trajectory.npz', 'trajectory.csv')):
            warn(f'Existing output left unchanged: {destination}; use --overwrite to rebuild.')
            if all((destination / name).exists() for name in ('trajectory.npz', 'trajectory.csv')):
                results.append(destination / 'trajectory.npz')
            continue
        options = sorted(assigned[sortie.id], key=candidate_order)
        for index, candidate in enumerate(options):
            try:
                arrays = load_candidate(candidate, sortie, mocap_body=mocap_body, mcap_topic=mcap_topic, ekf_core=ekf_core)
                same_priority = [c for c in options[index + 1:] if c.priority == candidate.priority
                                 and (c.path.with_suffix(''), c.run) != (candidate.path.with_suffix(''), candidate.run)]
                if same_priority:
                    warn(f'Multiple sources for {sortie.name} at the same priority; selected {candidate.path}')
                if not dry_run:
                    save_trajectory(destination, candidate, arrays, sortie)
                verb = 'WOULD WRITE' if dry_run else 'WROTE'
                print(f'[{verb}] {sortie.name}: {candidate.path}'
                      f'{" [" + candidate.run + "]" if candidate.run is not None else ""}'
                      f' -> {destination / "trajectory.npz"} ({len(arrays["times_s"])} samples)')
                results.append(destination / 'trajectory.npz')
                break
            except (ImportError, ValueError, OSError, RuntimeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                warn(f'Cannot process {candidate.path}: {exc}; trying next source.')
        else:
            warn(f'No usable trajectory for id={sortie.id}, sortie={sortie.name}, autopilot={sortie.autopilot}')
    expected = total if total is not None else len(sorties)
    if len(results) != expected:
        warn(f'Output count {len(results)} does not match runs.total/expected={expected}.')
    print(f'[INFO] {len(results)}/{len(sorties)} sorties {"available" if dry_run else "written or already present"}.')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='Organize one flight_logs/source dataset into run_XXX/{ardu,px4,cogni}_logs/trajectory.npz and .csv.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--flight-logs-dir', type=Path, help='Dataset directory under data/flight_test')
    parser.add_argument('--output-root', type=Path, help='Output root (default: beside source)')
    parser.add_argument('--dry-run', action='store_true', help='Validate and show selections without writing')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing generated trajectory files')
    parser.add_argument('--mocap-body', default='drone', help='Qualisys rigid body name (default: drone)')
    parser.add_argument('--mcap-topic', help='Explicit ROS pose topic; otherwise select a drone topic')
    parser.add_argument('--ekf-core', type=int, default=0, help='ArduPilot EKF core (default: 0)')
    inputs.add_argument('--ardu-logs-dir', type=Path, help='Legacy mode: convert raw ROS bags to processed CSVs')
    parser.add_argument('--skip-existing', action='store_true', help='Legacy mode: skip existing CSVs')
    args = parser.parse_args(argv)
    try:
        if args.ardu_logs_dir:
            extract_rosbag2_mocap_to_csv(args.ardu_logs_dir, skip_existing=args.skip_existing)
            return 0
        results = process_flight_logs(args.flight_logs_dir, output_root=args.output_root,
                                     dry_run=args.dry_run, overwrite=args.overwrite, mocap_body=args.mocap_body,
                                     mcap_topic=args.mcap_topic, ekf_core=args.ekf_core)
        return 0 if results else 1
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
