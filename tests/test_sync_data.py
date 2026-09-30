"""CLI behavior for optional rsync and rclone exclusions without transferring files."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "data" / "sync_data.py"
SPEC = importlib.util.spec_from_file_location("sync_data", SCRIPT)
sync_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_data)


class SyncDataExcludeTests(unittest.TestCase):
    def test_all_commands_accept_all_exclude_modes(self):
        custom = "CalibOpt_pipeline/"
        modes = (
            ([], []),
            (["--default-excludes"], None),
            (["--exclude", custom], [custom]),
            (["--default-excludes", "--exclude", custom], None),
        )

        with tempfile.TemporaryDirectory() as tmp:
            windows = Path(tmp) / "windows"
            wsl = Path(tmp) / "wsl"
            windows.mkdir()
            wsl.mkdir()

            for direction in ("pull", "push", "download", "upload"):
                backend = "rsync" if direction in ("pull", "push") else "rclone"
                defaults = (
                    sync_data.DEFAULT_RSYNC_EXCLUDES
                    if backend == "rsync"
                    else sync_data.DEFAULT_RCLONE_EXCLUDES
                )
                for options, explicit in modes:
                    with self.subTest(direction=direction, options=options):
                        args = [
                            "sync_data.py",
                            direction,
                            "--windows-dir", str(windows),
                            "--wsl-dir", str(wsl),
                            *options,
                        ]
                        if backend == "rclone":
                            args.extend(["--onedrive-dir", "remote:FlightTest"])

                        output = io.StringIO()
                        with (
                            patch.object(sys, "argv", args),
                            patch.object(sync_data, "check_command"),
                            patch.object(sync_data.subprocess, "run") as run,
                            contextlib.redirect_stdout(output),
                        ):
                            sync_data.main()

                        command = run.call_args.args[0]
                        expected = list(defaults) if "--default-excludes" in options else []
                        if explicit is not None:
                            expected.extend(explicit)
                        elif "--exclude" in options:
                            expected.append(custom)
                        actual = [
                            command[index + 1]
                            for index, value in enumerate(command[:-1])
                            if value == "--exclude"
                        ]
                        self.assertEqual(command[0], backend)
                        self.assertEqual(actual, expected)
                        self.assertIn("--dry-run", command)
                        if not expected:
                            self.assertIn("Excluded: none", output.getvalue())

    def test_local_onedrive_paths_rejected_before_any_transfer_or_mkdir(self):
        invalid = (
            "FIRE_moonshot_classifier/DARPA_FIRE/FIRE_Moonshot/FlightTest",
            "/tmp/FlightTest", "./local:folder", ":onedrive:folder",
            "https://example.com/FlightTest", "C:/FlightTest",
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "not-created"
            for direction in ("upload", "download"):
                for path in invalid:
                    with (
                        self.subTest(direction=direction, path=path),
                        patch.object(sync_data, "check_command") as check,
                        patch.object(sync_data.subprocess, "run") as run,
                    ):
                        with self.assertRaisesRegex(ValueError, "REMOTE:path"):
                            sync_data.sync_onedrive_wsl(
                                direction, path, destination, dry_run=False, delete=False,
                            )
                        check.assert_not_called()
                        run.assert_not_called()
                        self.assertFalse(destination.exists())

    def test_named_remote_paths_accepted(self):
        for path in (
            "FIRE_moonshot_classifier:DARPA_FIRE/FIRE_Moonshot/FlightTest",
            "remote:", "remote:/FlightTest", "remote:folder with spaces",
        ):
            with self.subTest(path=path):
                sync_data.validate_onedrive_path(path)

    def test_execute_and_delete_preserve_backend_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            for direction in ("push", "upload"):
                args = [
                    "sync_data.py", direction,
                    "--windows-dir", str(source),
                    "--wsl-dir", str(source),
                    "--execute", "--delete",
                ]
                if direction == "upload":
                    args.extend(["--onedrive-dir", "remote:FlightTest"])
                with (
                    self.subTest(direction=direction),
                    patch.object(sys, "argv", args),
                    patch.object(sync_data, "check_command"),
                    patch.object(sync_data.subprocess, "run") as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    sync_data.main()
                command = run.call_args.args[0]
                self.assertNotIn("--dry-run", command)
                if direction == "push":
                    self.assertEqual(command[0], "rsync")
                    self.assertIn("--delete", command)
                else:
                    self.assertEqual(command[:2], ["rclone", "sync"])


if __name__ == "__main__":
    unittest.main()
