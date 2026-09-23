"""Check optional media-tool wiring without loading SGLang or touching GPUs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "media_tools_env.sh"


class MediaToolsEnvTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.original = self.directory / "original-bin"
        self.media = self.directory / "media tools' env" / "bin"
        self.log = self.directory / "commands.jsonl"
        self.original.mkdir()
        self.media.mkdir(parents=True)
        self.env = dict(os.environ)
        for key in ("H3_MEDIA_TOOLS_DIR", "PYTHON", "BASH_ENV", "ENV"):
            self.env.pop(key, None)
        self.env.update(
            PATH=f"{self.original}:/usr/bin:/bin:/usr/sbin:/sbin",
            TMPDIR=str(self.directory),
            MOCK_LOG=str(self.log),
            LD_LIBRARY_PATH="/existing/sglang/lib",
            CONDA_PREFIX="/existing/sglang",
        )
        for name in ("python3", "sglang"):
            self.make_command(self.original / name, "original/" + name)
        # These must never replace the active SGLang environment's commands.
        for name in ("python3", "sglang", "ffmpeg", "ffprobe"):
            self.make_command(self.media / name, "media/" + name)

    def make_command(self, path, label):
        code = (
            "import json, os, shutil, sys\n"
            f"record = {{'command': {label!r}, 'args': sys.argv[1:]}}\n"
            "record.update({key: os.environ.get(key) for key in "
            "('PATH', 'LD_LIBRARY_PATH', 'CONDA_PREFIX', 'CUDA_VISIBLE_DEVICES')})\n"
            "record['resolved'] = {name: shutil.which(name) for name in "
            "('python3', 'sglang', 'ffmpeg', 'ffprobe')}\n"
            "with open(os.environ['MOCK_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(record) + '\\n')\n"
        )
        path.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(code)} \"$@\"\n"
        )
        path.chmod(0o755)

    def run_shell(self, script, *args, env=None):
        return subprocess.run(
            ["/bin/bash", "-c", "set -euo pipefail; " + script, "test", str(HELPER), *args],
            env=self.env if env is None else env,
            text=True, capture_output=True, timeout=10,
        )

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_unconfigured_helper_does_not_change_path_or_require_tools(self):
        for value in (None, ""):
            with self.subTest(value=value):
                env = dict(self.env, PATH=str(self.original))
                if value is not None:
                    env["H3_MEDIA_TOOLS_DIR"] = value
                result = self.run_shell('source "$1"; printf "%s" "$PATH"', env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, env["PATH"])
                self.assertFalse(self.log.exists())

    def test_only_media_commands_are_exposed_without_running_prechecks(self):
        self.env["H3_MEDIA_TOOLS_DIR"] = str(self.media)
        result = self.run_shell('source "$1"; python3 inspect')
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.commands()
        self.assertEqual(len(records), 1, "sourcing must not execute tool prechecks")
        record = records[0]
        self.assertEqual(record["command"], "original/python3")
        self.assertEqual(record["LD_LIBRARY_PATH"], self.env["LD_LIBRARY_PATH"])
        self.assertEqual(record["CONDA_PREFIX"], self.env["CONDA_PREFIX"])
        for name in ("python3", "sglang"):
            self.assertEqual(record["resolved"][name], str(self.original / name))
        shim, rest = record["PATH"].split(os.pathsep, 1)
        self.assertEqual(rest, self.env["PATH"])
        self.assertNotEqual(Path(shim), self.media)
        self.assertEqual({entry.name for entry in Path(shim).iterdir()}, {"ffmpeg", "ffprobe"})
        for name in ("ffmpeg", "ffprobe"):
            self.assertEqual(record["resolved"][name], str(Path(shim) / name))
            self.assertTrue(os.access(Path(shim) / name, os.X_OK))

    def test_media_wrappers_preserve_arguments_and_environment(self):
        self.env["H3_MEDIA_TOOLS_DIR"] = str(self.media)
        arguments = ["-i", "cat's video.mp4", "", "line\nbreak", "$literal `text`", "-version"]
        result = self.run_shell('source "$1"; shift; ffmpeg "$@"; ffprobe "$@"', *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.commands()
        self.assertEqual([item["command"] for item in records], ["media/ffmpeg", "media/ffprobe"])
        for record in records:
            self.assertEqual(record["args"], arguments)
            self.assertEqual(record["LD_LIBRARY_PATH"], self.env["LD_LIBRARY_PATH"])

    def test_invalid_explicit_directory_or_missing_executable_fails(self):
        bad = self.directory / "missing"
        cases = [(bad, "H3_MEDIA_TOOLS_DIR")]
        for name in ("ffmpeg", "ffprobe"):
            directory = self.directory / ("missing-" + name)
            directory.mkdir()
            other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
            self.make_command(directory / other, "unused")
            cases.append((directory, name))
        for directory, expected in cases:
            with self.subTest(directory=directory):
                env = dict(self.env, H3_MEDIA_TOOLS_DIR=str(directory))
                result = self.run_shell('source "$1"; printf "unexpected success"', env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("unexpected success", result.stdout)
                self.assertFalse(self.log.exists())

    def test_entrypoints_keep_active_python_sglang_and_launch_arguments(self):
        self.env["H3_MEDIA_TOOLS_DIR"] = str(self.media)
        request_args = ["--server-urls", "http://127.0.0.1:30010"]
        for entry in ("launch_minimax_h3_sglang.sh", "infer_ref2va_8gpu.sh", "serve_ref2va_8gpu.sh"):
            with self.subTest(entry=entry):
                self.log.unlink(missing_ok=True)
                extra = [] if entry.startswith("launch_") else request_args
                result = subprocess.run(
                    ["/bin/bash", str(ROOT / entry), *extra], env=self.env,
                    text=True, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                records = self.commands()
                self.assertEqual(len(records), 1, "entrypoint must not execute media prechecks")
                record = records[0]
                self.assertEqual(record["LD_LIBRARY_PATH"], self.env["LD_LIBRARY_PATH"])
                self.assertNotEqual(Path(record["resolved"]["ffprobe"]).parent, self.media)
                self.assertTrue(record["PATH"].endswith(os.pathsep + self.env["PATH"]))
                if entry.startswith("launch_"):
                    self.assertEqual(record["command"], "original/sglang")
                    self.assertEqual(record["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
                    self.assertEqual(record["args"][0], "serve")
                    for flag, value in (("--port", "30010"), ("--num-gpus", "4"),
                                        ("--tp-size", "2"), ("--ulysses-degree", "2"),
                                        ("--model-variant", "ref2va")):
                        self.assertEqual(record["args"][record["args"].index(flag) + 1], value)
                else:
                    self.assertEqual(record["command"], "original/python3")
                    expected = [str(ROOT / "scripts" / "infer_ref2va_8gpu.py")]
                    if entry.startswith("serve_"):
                        expected.append("--serve-only")
                    self.assertEqual(record["args"], expected + request_args)


if __name__ == "__main__":
    unittest.main()
