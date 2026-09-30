# Flight-log organization

Process one dataset (the argument is the directory containing `source`):

```bash
python tools/log_processing/flight_log_processor.py \
  --flight-logs-dir data/flight_test/260527_flight_logs

# Validate source selection and decode inputs without creating outputs:
python tools/log_processing/flight_log_processor.py \
  --flight-logs-dir data/flight_test/260827_flight_logs --dry-run
```

Specify the dataset with `--flight-logs-dir PATH`; positional paths are not accepted.
`--output-root PATH` writes to a separate directory for inspection.
Existing trajectory files are left unchanged unless `--overwrite` is supplied.
Original files under `source` are never modified.

Output for metadata entry `id: 0`, `autopilot: ArduPilot`:

```text
260527_flight_logs/
├── source/
└── run_000/
    └── ardu_logs/
        ├── trajectory.npz
        ├── trajectory.csv
        └── provenance.json
```

Autopilots map to `ardu`, `px4`, or `cogni`, ignoring capitalization.
Run numbers come directly from `trajectory_sets[].id`, including nonconsecutive IDs.

## Source selection

Selection happens independently for each sortie, in this order:

1. `source/vision/SAM3_pipeline`
2. `source/vision/CalibOpt_pipeline`
3. `source/raw/fc_logs`
4. `source/raw/videos_and_mocap_logs`
5. `source/raw/mocap_logs`

Files are searched recursively. A sortie name can be a directory name or a
filename; known suffixes such as `_trajectory`, `_data`, `_mocap`, and `_6D`
are recognized. Vision CSV/NPZ files without `trajectory` in their names are
also accepted when they contain time and position arrays (e.g. 260424).
Names such as `ardu_run1`, `ardu_run10`, and `ardu_run5_comp2` stay distinct.
ArduPilot/CogniPilot spelling variants map to the same firmware aliases.

Merged files with a `run`, `run_id`, `run_number`, or `sortie_name` column/array
are split by that field. For `ardupilot_all_runs` and `cognipilot_all_runs`,
`run1` refers to that firmware's first sortie, not metadata ID 1. Frame and time
values are preserved, including time restarting between runs. An `all_runs`
file without a run identifier is rejected rather than guessed.

At the same source priority, NPZ is preferred over CSV, and Qualisys 6D TSV
is preferred over other TSVs. Multiple distinct candidates generate a warning.
An unreadable/unsupported candidate is reported and the next candidate is tried.
A non-split selected NPZ or CSV is copied unchanged; the other format is
created from its data. Split exports contain only the selected run's rows.

Without metadata, firmware and sortie identities are inferred from path names,
then assigned deterministic IDs starting at zero. Unknown firmware names are
reported and skipped. When metadata exists, unmatched inputs are reported;
they do not silently create extra runs.

Optional metadata fields resolve unusual layouts:

```yaml
trajectory_sets:
  - id: 0
    sortie_name: test
    autopilot: ArduPilot
    source_path: raw/mocap_logs/unusual_name.mcap  # Relative to source; file or directory
    topic: /ardupilot/odom                       # Optional MCAP pose topic
    body: drone                                 # Optional Qualisys rigid body
```

`source_path` provides an additional explicit match; the standard source
priority still applies. Metadata IDs and names must be unambiguous.

The processor reports physical vision-file counts and matched sortie counts.
CSV/NPZ copies of the same sortie are counted once; an `all_runs` file counts
as its individual sorties. Warnings compare those counts and final outputs
with `runs.total`, and identify missing/unmatched sorties. Partial datasets can
still produce usable outputs; inspect warnings (the CLI returns nonzero if no
trajectory can be produced or if configuration is invalid).

## Output arrays and raw formats

CSV conversion preserves original columns and adds these NPZ arrays when
applicable:

| NPZ array | CSV input |
| --- | --- |
| `times_s` | `time_s`, `mocap_time_s`, `timestamp`, or `t` |
| `trajectory_raw` (N × 3) | `x_raw,y_raw,z_raw` or `x,y,z` |
| `trajectory_smooth` (N × 3) | `x_smooth,y_smooth,z_smooth` or `xsmooth,ysmooth,zsmooth` |
| `gt_drone` (N × 3) | `gt_x,gt_y,gt_z` or `gtx,gty,gtz` |
| `reproj_errors_px` | `reproj_px` |

CSV `timestamp` is assumed to already be seconds. Raw-format timestamp units
are handled by their respective decoders. Copied NPZ files keep their original
schema. Missing values remain NaN; no interpolation or smoothing is performed.

Raw conversion writes `times_s` and `trajectory_raw`, plus scalar
`measurement_type`, `coordinate_frame`, and `position_unit` metadata:

- **ULG:** PX4 `vehicle_local_position`, timestamps from microseconds to seconds,
  positions in meters and NED. Invalid XY/Z flags mask the corresponding axes.
- **BIN:** ArduPilot `XKF1` (fallback `NKF1`) position, in meters and NED.
  Only one EKF core is selected (`--ekf-core 0` by default); estimator cores
  and XKF1/NKF1 streams are not interleaved.
- **TSV:** Qualisys rigid-body XYZ, selecting `drone` by default
  (`--mocap-body NAME`, or metadata `body`). Positions convert mm to meters;
  time uses the file's `FREQUENCY`. Untracked zero poses become NaN. The body
  position is also exported as `gt_drone`. Camera bodies and anonymous marker
  clouds are not substituted for drone position.
- **MCAP:** ROS 2 Odometry, PoseStamped, or PoseWithCovarianceStamped. Prefer
  the firmware's `/ardupilot/odom`, `/px4/odom`, or `/cognipilot/odom`; otherwise
  require an unambiguous non-camera pose topic. Use `--mcap-topic TOPIC` or
  metadata `topic` to select explicitly. Preserve the topic's coordinate
  frame; use header timestamps (bag timestamps if any header stamp is zero),
  with `times_s` relative to the first sample. `epoch_times_s` preserves the
  source timestamps in seconds; ROS simulated clocks are not necessarily UTC.

FC/ROS positions are not automatically relabeled as ground truth or smoothed
vision output. Consequently, downstream loaders restricted to vision smooth
columns need a separate explicit choice of how to consume these raw outputs.
Coordinate frames are not aligned across measurement systems by this tool.
`provenance.json` records the selected file, split run, firmware, sample count,
and available raw-source metadata.

The organizer requires NumPy and PyYAML (already project dependencies).
BIN and ULG use the project's `pymavlink` and `pyulog` dependencies.
MCAP requires a sourced ROS 2 Python environment (`rosbag2_py`, `rclpy`,
`rosidl_runtime_py`) and the rosbag2 MCAP storage plugin. These imports are
lazy: processing vision data does not require ROS.

The existing `--ardu-logs-dir PATH [--skip-existing]` interface remains
available for the legacy `raw/<bag>/` to `processed/*.csv` conversion.

Tests:

```bash
python -m pytest tests/test_flight_log_processor.py -q
```
