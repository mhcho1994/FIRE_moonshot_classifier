import unittest
from pathlib import Path

from fire_moonshot_classifier.cli import _normalize_train_alias, build_parser


class CliParserTests(unittest.TestCase):
    def test_feature_build_dataset_arguments(self):
        args = build_parser().parse_args(
            [
                "feature-build",
                "--logs",
                "data/260615_sitl_logs",
                "--source",
                "sitl",
                "--target-features",
                "XY-Accel,",
                "XY-Jerk,",
                "Curvature",
            ]
        )

        self.assertEqual(args.command, "feature-build")
        self.assertEqual(args.logs, [Path("data/260615_sitl_logs")])
        self.assertEqual(args.source, "sitl")
        self.assertEqual(args.target_features[-1], "Curvature")

    def test_firetrack_trajectory_arguments(self):
        args = build_parser().parse_args(
            [
                "feature-build",
                "--trajectory",
                "px4=/work/triangulation/run_001/trajectory.csv",
                "--dataset-name",
                "firetrack_eval",
            ]
        )

        self.assertEqual(
            args.trajectory,
            ["px4=/work/triangulation/run_001/trajectory.csv"],
        )
        self.assertEqual(args.dataset_name, "firetrack_eval")

    def test_requested_train_flags_map_to_model_subcommands(self):
        self.assertEqual(
            _normalize_train_alias(["train", "--svm", "--no-real"]),
            ["train", "svm", "--no-real"],
        )
        self.assertEqual(
            _normalize_train_alias(["train", "--diversify", "--epochs", "3"]),
            ["train", "diversify", "--epochs", "3"],
        )

    def test_train_model_subcommands(self):
        args = build_parser().parse_args(["train", "diversify", "--no-wandb"])
        self.assertEqual(args.model, "diversify")
        self.assertTrue(args.no_wandb)


if __name__ == "__main__":
    unittest.main()
