import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fire_moonshot_classifier.cli import main
from fire_moonshot_classifier.datamanager.dataset_manager import extract_turn_sequences
from fire_moonshot_classifier.inference import TrajectoryPredictor, predict_trajectory
from fire_moonshot_classifier.inference.bundle import load_bundle, save_bundle
from fire_moonshot_classifier.inference.export import export_checkpoint
from fire_moonshot_classifier.inference.model import DiversifyNetwork, knn_distances
from fire_moonshot_classifier.inference.predictor import read_trajectory
from fire_moonshot_classifier.processor.sequence_windows import windows_from_sequence


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "unlabeled.csv"
        t = np.arange(0, 20, 0.02)
        xyz = np.column_stack((2 * np.cos(t), 2 * np.sin(t), np.zeros_like(t)))
        self.write_csv(t, xyz)
        self.model = DiversifyNetwork(3, 128, 32, 64).eval()
        self.path = self.root / "model.pt"
        self.names = ["XY-Accel", "XY-Jerk", "Curvature"]
        self.bank = torch.zeros(5, 32)
        self.save_model()

    def write_csv(self, t, xyz):
        np.savetxt(self.csv, np.column_stack((t, xyz)), delimiter=",",
                   header="time_s,x_smooth,y_smooth,z_smooth", comments="")

    def save_model(self, **kwargs):
        return save_bundle(self.path, state_dict=self.model.state_dict(), feature_names=self.names,
                           bank_l1=self.bank, threshold=kwargs.pop("threshold", 1e6), **kwargs)

    def test_unlabeled_csv_matches_direct_window_evaluation(self):
        # Independent aggregation of the same physical trajectory: gates and votes
        # must be invariant to the inference batch size and filename.
        raw, _ = read_trajectory(self.csv)
        sequences, _ = extract_turn_sequences(raw, self.names)
        windows = [w for seq in sequences for w in windows_from_sequence(seq)]
        self.assertGreaterEqual(len(windows), 3)
        with torch.inference_mode():
            logits, features = self.model(torch.stack(windows))
        reference_votes = logits.argmax(dim=1)
        distances = torch.cdist(features, self.bank).topk(5, largest=False).values[:, -1]
        result = predict_trajectory(self.csv, self.path, batch_size=3)
        expected = ["ArduPilot", "PX4"][int(torch.mode(reference_votes).values)]
        self.assertEqual(result["prediction"], expected)
        self.assertEqual(result["n_windows"], len(windows))
        self.assertEqual(result["n_accepted"], len(windows))
        self.assertAlmostEqual(result["knn_dist"], float(distances.mean()), places=5)
        self.assertNotIn("ground_truth", result)
        self.assertNotIn("correct", result)
        json.dumps(result, allow_nan=False)
        renamed = self.csv.with_name("px4_ardu_cogni.csv")
        self.csv.rename(renamed)
        self.assertEqual(predict_trajectory(renamed, self.path)["prediction"], expected)

    def test_unknown_for_no_turns_short_data_and_ood(self):
        t = np.arange(0, 10, 0.02)
        self.write_csv(t, np.column_stack((t, t * 0, t * 0)))
        result = predict_trajectory(self.csv, self.path)
        self.assertEqual(result["prediction"], "Unknown")
        self.assertEqual(result["reason"], "no_turn_segments")
        self.assertIsNone(result["knn_dist"])
        self.write_csv(t[:3], np.zeros((3, 3)))
        self.assertEqual(predict_trajectory(self.csv, self.path)["reason"], "insufficient_samples")
        self.write_csv(t, np.column_stack((np.cos(t), np.sin(t), t * 0)))
        self.bank = torch.full((5, 32), 1e6)
        self.save_model(threshold=0.0)
        result = predict_trajectory(self.csv, self.path)
        self.assertEqual(result["prediction"], "Unknown")
        self.assertEqual(result["reason"], "insufficient_in_distribution_windows")
        self.assertEqual(result["n_rejected"], result["n_windows"])
        self.assertEqual(result["reject_rate"], 1.0)

    def test_duplicate_unsorted_nonfinite_samples_are_cleaned_before_gradient(self):
        t = np.arange(20, dtype=float)
        xyz = np.column_stack((t, t * 2, t * 3))
        self.write_csv(np.r_[t[::-1], 5, np.nan], np.vstack((xyz[::-1], xyz[5], xyz[0])))
        raw, quality = read_trajectory(self.csv)
        np.testing.assert_array_equal(raw["t"], t)
        np.testing.assert_allclose(raw["vx"], 1)
        self.assertEqual(quality["n_duplicate_timestamps"], 1)
        self.assertEqual(quality["n_nonfinite_rows"], 1)

    def test_wrong_schema_missing_file_and_raw_weights_fail_clearly(self):
        self.csv.write_text("time_s,x,y,z\n0,1,2,3\n")
        with self.assertRaisesRegex(ValueError, "CSV requires"):
            predict_trajectory(self.csv, self.path)
        with self.assertRaises(FileNotFoundError):
            predict_trajectory(self.root / "missing.csv", self.path)
        torch.save(self.model.state_dict(), self.path)
        with self.assertRaisesRegex(ValueError, "fireclassify export"):
            TrajectoryPredictor(self.path)

    def test_bundle_validation_checksum_and_frozen_batchnorm(self):
        predictor = TrajectoryPredictor(self.path)
        before = {k: v.clone() for k, v in predictor.model.state_dict().items()}
        predictor.predict_trajectory(self.csv)
        predictor.predict_trajectory(self.csv)
        for key, value in before.items():
            torch.testing.assert_close(value, predictor.model.state_dict()[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            TrajectoryPredictor(self.path, expected_sha256="0" * 64)
        bundle, _ = load_bundle(self.path)
        bundle["class_names"].reverse()
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(ValueError, "class order"):
            TrajectoryPredictor(self.path)

    def test_edge_padding_zscore_and_reference_bank_chunking(self):
        seq = np.arange(40 * 3).reshape(40, 3).astype(float)
        padded = np.pad(seq, ((0, 20), (0, 0)))
        actual = windows_from_sequence(padded)
        expected = np.pad(seq.astype(np.float32), ((0, 60), (0, 0)), mode="edge").T.copy()
        expected = (expected - expected.mean(axis=1, keepdims=True)) / (expected.std(axis=1, keepdims=True) + 1e-8)
        np.testing.assert_array_equal(actual[0].numpy(), expected)
        self.assertEqual(windows_from_sequence(seq[:29]), [])
        bank, query = torch.randn(107, 32), torch.randn(31, 32)
        expected = torch.cdist(query, bank).topk(5, largest=False).values[:, -1]
        torch.testing.assert_close(knn_distances(query, bank, 5, bank_batch_size=37), expected)

    def test_export_uses_disjoint_groups_and_is_deterministic(self):
        checkpoint = self.root / "weights.pt"
        torch.save(self.model.state_dict(), checkpoint)
        cache = self.root / "sitl.npz"
        rng = np.random.RandomState(42)
        # Both classes per run, just as in the real SITL dataset layout.
        np.savez(cache, X_seq=rng.randn(40, 120, 3), y=np.tile([0, 1], 20),
                 runs=np.repeat(np.arange(20), 2), feature_names=self.names)
        output = self.root / "exported.pt"
        export_checkpoint(checkpoint, cache, output)
        first, _ = load_bundle(output)
        export_checkpoint(checkpoint, cache, output)
        second, _ = load_bundle(output)
        torch.testing.assert_close(first["ood_bank_l1"], second["ood_bank_l1"], rtol=0, atol=0)
        self.assertEqual(first["ood_threshold"], second["ood_threshold"])
        self.assertEqual(first["provenance"]["split_level"], "run")
        self.assertEqual(len(first["ood_bank_l1"]), 28)
        self.assertEqual(first["provenance"]["n_validation_windows"], 4)
        self.assertEqual(first["provenance"]["n_test_segments"], 8)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            export_checkpoint(checkpoint, cache, output, feature_names=self.names[::-1])
        # Removing training inputs must not affect deployment.
        checkpoint.unlink()
        cache.unlink()
        self.assertIn(predict_trajectory(self.csv, output)["prediction"], ("PX4", "ArduPilot", "Unknown"))

    def test_mil_absolute_fractional_quorum_and_vote_tie(self):
        sequences = [np.random.RandomState(0).randn(550, 3)]  # ten windows
        self.save_model(threshold=1.0, min_valid=3, min_fraction=0.5)
        predictor = TrajectoryPredictor(self.path)
        logits = torch.tensor([[1., 0.], [0., 1.]] * 5)
        with patch("fire_moonshot_classifier.inference.predictor.extract_turn_sequences",
                   return_value=(sequences, [])), \
             patch.object(predictor.model, "forward", return_value=(logits, torch.zeros(10, 32))):
            # Four votes pass the absolute floor but fail the fraction requirement.
            with patch("fire_moonshot_classifier.inference.predictor.knn_distances",
                       return_value=torch.tensor([0.] * 4 + [2.] * 6)):
                result = predictor.predict_trajectory(self.csv)
                self.assertEqual(result["prediction"], "Unknown")
                self.assertEqual(result["n_accepted"], 4)
            # Six accepted windows, equal class votes, tie resolves to class 0.
            with patch("fire_moonshot_classifier.inference.predictor.knn_distances",
                       return_value=torch.tensor([0.] * 6 + [2.] * 4)):
                result = predictor.predict_trajectory(self.csv)
                self.assertEqual(result["prediction"], "ArduPilot")
                self.assertEqual(result["vote_fraction"], 0.5)
                self.assertEqual((result["n_ardu"], result["n_px4"]), (3, 3))
            predictor.decision["min_fraction"] = 0.0
            with patch("fire_moonshot_classifier.inference.predictor.knn_distances",
                       return_value=torch.tensor([0.] * 2 + [2.] * 8)):
                self.assertEqual(predictor.predict_trajectory(self.csv)["prediction"], "Unknown")

    def test_cli_produces_only_json_and_saved_result(self):
        output = self.root / "results" / "prediction.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["predict", "--trajectory", str(self.csv), "--model", str(self.path),
                           "--output", str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))

    def test_inference_does_not_import_training_or_log_parsers(self):
        script = """
import sys
from fire_moonshot_classifier.inference import predict_trajectory
predict_trajectory(sys.argv[1], sys.argv[2])
assert not any(k.startswith(('wandb', 'pymavlink', 'pyulog', 'matplotlib',
    'fire_moonshot_classifier.training')) for k in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.csv), str(self.path)],
                                capture_output=True, text=True,
                                env={**os.environ, "OMP_NUM_THREADS": "2"})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
