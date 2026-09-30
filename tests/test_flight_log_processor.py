"""Behavioral tests for dataset mapping, precedence, splitting, and raw decoders."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/log_processing/flight_log_processor.py'
SPEC = importlib.util.spec_from_file_location('flight_log_processor', SCRIPT)
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / 'source').mkdir()
    return tmp_path


def metadata(root, entries, total=None):
    (root / 'source/metadata.yaml').write_text(yaml.safe_dump({
        'runs': {'total': len(entries) if total is None else total}, 'trajectory_sets': entries}))


def sortie(ident=0, name='ardu_run1', autopilot='ArduPilot', **extra):
    return dict(id=ident, sortie_name=name, autopilot=autopilot, **extra)


def csv_file(root, relative, value=1):
    path = root / 'source' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('frame,time_s,x_smooth,y_smooth,z_smooth,gt_x,gt_y,gt_z\n'
                    f'0,0,{value},2,3,4,5,6\n1,0.1,{value},2,3,4,5,6\n')
    return path


def npz_file(root, relative, value=1, **extra):
    path = root / 'source' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, times_s=np.array([0., .1]), trajectory_smooth=np.full((2, 3), value), **extra)
    return path


def test_priority_is_per_sortie_and_ids_are_preserved(dataset):
    metadata(dataset, [sortie(4), sortie(8, 'px4_run2', 'PX4')])
    preferred = csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1/trajectory.csv', 11)
    csv_file(dataset, 'vision/CalibOpt_pipeline/ardu_run1_trajectory.csv', 22)
    csv_file(dataset, 'vision/CalibOpt_pipeline/px4_run2_trajectory.csv', 33)
    result = p.process_flight_logs(dataset)
    assert len(result) == 2
    with np.load(dataset / 'run_004/ardu_logs/trajectory.npz') as data:
        assert data['trajectory_smooth'][0, 0] == 11
        assert data['gt_drone'][0, 0] == 4
    assert (dataset / 'run_004/ardu_logs/trajectory.csv').read_bytes() == preferred.read_bytes()
    with np.load(dataset / 'run_008/px4_logs/trajectory.npz') as data:
        assert data['trajectory_smooth'][0, 0] == 33


def test_npz_copied_byte_for_byte_and_pair_counted_once(dataset, capsys):
    metadata(dataset, [sortie()])
    source = npz_file(dataset, 'vision/SAM3_pipeline/ardu_run1_trajectory.npz', 9)
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1_trajectory.csv', 2)
    output = p.process_flight_logs(dataset)[0]
    assert output.read_bytes() == source.read_bytes()
    assert 'runs.total=' not in capsys.readouterr().err
    assert output.with_suffix('.csv').is_file()


def test_split_all_runs_both_firmwares_and_noncontiguous_rows(dataset):
    metadata(dataset, [sortie(7), sortie(2, 'cognipilot_run1', 'CogniPilot'), sortie(9, 'ardu_run2')])
    for fw in ('ardupilot', 'cognipilot'):
        path = csv_file(dataset, f'vision/SAM3_pipeline/{fw}_all_runs_trajectory.csv')
        path.write_text('run,frame,time_s,x_smooth,y_smooth,z_smooth\n'
                        'run1,0,5,1,2,3\nrun2,0,8,4,5,6\nrun1,1,5.1,7,8,9\n')
    assert len(p.process_flight_logs(dataset)) == 3
    with np.load(dataset / 'run_007/ardu_logs/trajectory.npz') as data:
        np.testing.assert_allclose(data['times_s'], [5, 5.1])
        np.testing.assert_allclose(data['trajectory_smooth'][:, 0], [1, 7])
        assert set(data['run']) == {'run1'}
    with np.load(dataset / 'run_002/cogni_logs/trajectory.npz') as data:
        assert data['trajectory_smooth'].shape == (2, 3)
    with np.load(dataset / 'run_009/ardu_logs/trajectory.npz') as data:
        assert data['times_s'].tolist() == [8]


def test_npz_all_runs_numeric_ids(dataset):
    metadata(dataset, [sortie(5), sortie(6, 'ardu_run2')])
    npz_file(dataset, 'vision/SAM3_pipeline/ardu_all_runs.npz', run_id=np.array([1, 2]))
    assert len(p.process_flight_logs(dataset)) == 2
    with np.load(dataset / 'run_006/ardu_logs/trajectory.npz') as data:
        assert data['trajectory_smooth'].shape == (1, 3)


def test_no_metadata_inference_and_no_run1_run10_collision(dataset):
    csv_file(dataset, 'vision/SAM3_pipeline/ardupilot_run1_trajectory.csv', 1)
    csv_file(dataset, 'vision/SAM3_pipeline/px4_run10/trajectory.csv', 10)
    csv_file(dataset, 'vision/SAM3_pipeline/cogni/run2/trajectory.csv', 2)
    csv_file(dataset, 'vision/CalibOpt_pipeline/ardu_run1_trajectory.csv', 99)
    assert len(p.process_flight_logs(dataset)) == 3
    manifests = [json.loads(f.read_text()) for f in dataset.glob('run_*/*/provenance.json')]
    assert {(m['sortie_name'], m['autopilot']) for m in manifests} == {
        ('ardu_run1', 'ardu'), ('px4_run10', 'px4'), ('cogni_run2', 'cogni')}


def test_exact_matching_comp_and_comp2(dataset):
    metadata(dataset, [sortie(0, 'ardu_run5_comp'), sortie(1, 'ardu_run5_comp2')])
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_run5_comp/trajectory.csv', 1)
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_run5_comp2/trajectory.csv', 2)
    assert len(p.process_flight_logs(dataset)) == 2
    with np.load(dataset / 'run_001/ardu_logs/trajectory.npz') as data:
        assert data['trajectory_smooth'][0, 0] == 2


def test_bad_preferred_source_falls_back(dataset, capsys):
    metadata(dataset, [sortie()])
    bad = csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1_trajectory.csv')
    bad.write_text('time_s,x_smooth,y_smooth,z_smooth\n0,47...,2,3\n')
    csv_file(dataset, 'vision/CalibOpt_pipeline/ardu_run1.csv', 42)
    result = p.process_flight_logs(dataset)
    with np.load(result[0]) as data:
        assert data['trajectory_smooth'][0, 0] == 42
    assert 'Ignoring' in capsys.readouterr().err


def test_missing_sortie_and_count_warn_without_inventing_output(dataset, capsys):
    metadata(dataset, [sortie(), sortie(1, 'ardu_run10')], total=3)
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1.csv')
    assert len(p.process_flight_logs(dataset)) == 1
    assert not (dataset / 'run_001').exists()
    err = capsys.readouterr().err
    assert 'metadata defines 2' in err
    assert 'No usable trajectory' in err
    assert 'Output count 1' in err


def test_dry_run_and_existing_output(dataset):
    metadata(dataset, [sortie()])
    source = csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1.csv', 1)
    assert len(p.process_flight_logs(dataset, dry_run=True)) == 1
    assert not list(dataset.glob('run_*'))
    output = p.process_flight_logs(dataset)[0]
    before = output.read_bytes()
    source.write_text(source.read_text().replace('0,0,1,', '0,0,9,'))
    p.process_flight_logs(dataset)
    assert output.read_bytes() == before
    p.process_flight_logs(dataset, overwrite=True)
    with np.load(output) as data:
        assert data['trajectory_smooth'][0, 0] == 9


def test_mocap_time_alias_and_gt_missing(dataset):
    metadata(dataset, [sortie()])
    source = csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1.csv')
    source.write_text('mocap_frame,mocap_time_s,x_raw,y_raw,z_raw,gt_x,gt_y,gt_z\n0,0.2,1,2,3,,,\n')
    output = p.process_flight_logs(dataset)[0]
    with np.load(output) as data:
        assert data['times_s'][0] == .2
        assert np.isnan(data['gt_drone']).all()


def test_no_pickle_and_all_runs_requires_labels(dataset, capsys):
    metadata(dataset, [sortie()])
    npz_file(dataset, 'vision/SAM3_pipeline/ardu_run1.npz', unsafe=np.array([{}], dtype=object))
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_all_runs.csv')
    assert p.process_flight_logs(dataset) == []
    assert 'run identifier' in capsys.readouterr().err


def test_duplicate_metadata_ids_rejected(dataset):
    metadata(dataset, [sortie(), sortie(0, 'px4_run1', 'PX4')])
    with pytest.raises(ValueError, match='Duplicate'):
        p.process_flight_logs(dataset)


def test_explicit_source_path(dataset):
    metadata(dataset, [sortie(source_path='vision/SAM3_pipeline/arbitrary/output.csv')])
    csv_file(dataset, 'vision/SAM3_pipeline/arbitrary/output.csv')
    assert len(p.process_flight_logs(dataset)) == 1


def test_qualisys_body_units_and_missing_tracking(dataset):
    metadata(dataset, [sortie()])
    path = dataset / 'source/raw/videos_and_mocap_logs/ardu_run1/run1_6D.tsv'
    path.parent.mkdir(parents=True)
    path.write_text('FREQUENCY\t100\nBODY_NAMES\tcamera\tdrone\n'
                    'camera X\tY\tZ\tRoll\tPitch\tYaw\t\tdrone X\tY\tZ\tRoll\tPitch\tYaw\n'
                    '9000\t8000\t7000\t0\t0\t0\t\t1000\t2000\t3000\t0\t0\t0\n'
                    '9000\t8000\t7000\t0\t0\t0\t\t0\t0\t0\t0\t0\t0\n')
    output = p.process_flight_logs(dataset)[0]
    with np.load(output) as data:
        np.testing.assert_allclose(data['trajectory_raw'][0], [1, 2, 3])
        np.testing.assert_allclose(data['times_s'], [0, .01])
        assert np.isnan(data['trajectory_raw'][1]).all()
        assert 'trajectory_smooth' not in data
        assert data['measurement_type'] == 'mocap'


def test_bin_keeps_one_ekf_stream_core_and_ned():
    pytest.importorskip('pymavlink')
    def msg(kind, core, t, x):
        return SimpleNamespace(C=core, TimeUS=t, PN=x, PE=2, PD=3, get_type=lambda: kind)
    messages = iter([msg('NKF1', 0, 1_000_000, 100), msg('XKF1', 1, 1_000_000, 200),
                     msg('XKF1', 0, 1_000_000, 1), msg('XKF1', 0, 2_000_000, 4), None])
    connection = SimpleNamespace(recv_match=lambda **_: next(messages), close=lambda: None)
    with patch('pymavlink.mavutil.mavlink_connection', return_value=connection):
        arrays = p.read_bin(Path('unused.BIN'))
    np.testing.assert_allclose(arrays['trajectory_raw'], [[1, 2, 3], [4, 2, 3]])
    np.testing.assert_allclose(arrays['times_s'], [1, 2])
    assert arrays['coordinate_frame'] == 'NED'


def test_ulog_extracts_local_position_and_masks_invalid():
    pytest.importorskip('pyulog')
    data = dict(timestamp=np.array([1e6, 2e6]), x=np.array([1., 4.]), y=np.array([2., 5.]),
                z=np.array([3., 6.]), xy_valid=np.array([1, 0]), z_valid=np.array([1, 1]))
    with patch('pyulog.ULog') as ulog:
        ulog.return_value.get_dataset.return_value.data = data
        arrays = p.read_ulog(Path('unused.ulg'))
    np.testing.assert_allclose(arrays['trajectory_raw'][0], [1, 2, 3])
    assert np.isnan(arrays['trajectory_raw'][1, :2]).all()
    assert arrays['trajectory_raw'][1, 2] == 6


def test_mcap_excludes_camera_and_uses_header_time():
    pytest.importorskip('rosbag2_py')
    pytest.importorskip('rclpy')
    pytest.importorskip('rosidl_runtime_py')
    def message(x, sec):
        return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=2, z=3))),
                               header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0), frame_id='map'))
    messages = iter([('/camera/odom', message(99, 10), 100), ('/ardupilot/odom', message(1, 10), 101),
                     ('/ardupilot/odom', message(4, 11), 102)])
    items = list(messages)
    reader = SimpleNamespace(open=lambda *args: None,
        get_all_topics_and_types=lambda: [SimpleNamespace(name=t, type='nav_msgs/msg/Odometry')
                                         for t in ('/camera/odom', '/ardupilot/odom')],
        has_next=lambda: bool(items), read_next=lambda: items.pop(0))
    with patch('rosbag2_py.SequentialReader', return_value=reader), \
         patch('rosidl_runtime_py.utilities.get_message', return_value=object), \
         patch('rclpy.serialization.deserialize_message', side_effect=lambda raw, _: raw):
        arrays = p.read_mcap(Path('unused.mcap'), 'ardu')
    np.testing.assert_allclose(arrays['trajectory_raw'][:, 0], [1, 4])
    np.testing.assert_allclose(arrays['times_s'], [0, 1])
    assert arrays['topic'] == '/ardupilot/odom'


def test_output_cannot_be_inside_source(dataset):
    with pytest.raises(ValueError, match='outside source'):
        p.process_flight_logs(dataset, output_root=dataset / 'source/output')


def test_cli_requires_named_input(dataset):
    metadata(dataset, [sortie()])
    csv_file(dataset, 'vision/SAM3_pipeline/ardu_run1.csv')
    assert p.main(['--flight-logs-dir', str(dataset), '--dry-run']) == 0
    for arguments in ([], [str(dataset), '--dry-run'],
                      ['--flight-logs-dir', str(dataset), '--ardu-logs-dir', str(dataset)]):
        with pytest.raises(SystemExit) as error:
            p.main(arguments)
        assert error.value.code == 2


def test_all_five_source_priorities(dataset):
    metadata(dataset, [sortie()])
    sources = [csv_file(dataset, subdir + '/ardu_run1.csv', i + 1)
               for i, subdir in enumerate(p.SOURCE_ORDER)]
    for i, source in enumerate(sources):
        result = p.process_flight_logs(dataset, overwrite=True)
        with np.load(result[0]) as arrays:
            assert arrays['trajectory_smooth'][0, 0] == i + 1
        source.unlink()


def test_malformed_quality_array_falls_back(dataset, capsys):
    metadata(dataset, [sortie()])
    npz_file(dataset, 'vision/SAM3_pipeline/ardu_run1.npz', n_views=np.array([2]))
    csv_file(dataset, 'vision/CalibOpt_pipeline/ardu_run1.csv')
    assert len(p.process_flight_logs(dataset)) == 1
    assert 'same 1D shape' in capsys.readouterr().err
