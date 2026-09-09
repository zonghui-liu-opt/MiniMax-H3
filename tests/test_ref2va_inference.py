from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import batch_fl2va_sglang as base
import batch_ref2va_sglang as ref
import infer_ref2va_8gpu as entry
import infer_smoke_8gpu as runner
from test_infer_smoke_8gpu import Handler, servers


PROMPT = "\n\n".join(f"{section}:\n" + (
    "<Subject 1> is the cat in <Picture 1>, following motion from <Video 1>."
    if section == "subject_definitions" else "[Shot 1] The cat moves naturally.")
    for section in ref.SECTIONS)


@unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"), "needs ffmpeg/ffprobe")
class RefInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_temp = tempfile.TemporaryDirectory()
        cls.media = Path(cls.media_temp.name)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=orange:s=48x80:r=24", "-frames:v", "1",
                        str(cls.media / "cat.png")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=48x80:r=24", "-t", "4", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(cls.media / "motion.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(cls.media / "motion.mp4"),
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                        "-c:v", "copy", "-c:a", "aac", "-shortest",
                        str(cls.media / "source.mp4")], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.media_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "exp_Ref2VA"
        self.output.mkdir()
        shutil.copyfile(self.media / "cat.png", self.root / "猫.png")
        shutil.copyfile(self.media / "motion.mp4", self.output / "motion.mp4")
        self.metadata = self.output / "metadata.csv"
        self.rows = []
        for i, seed in enumerate((0, 10000, 20000, 30000)):
            self.rows.append({"id": f"{i:03d}", "input_image": "../猫.png",
                    "reference_video": "motion.mp4", "prompt": PROMPT,
                    "cat_id": "00-orange-cat", "motion_id": "02-pull-left-ear",
                    "motion_slug": "drag_ear", "seed": str(seed),
                    "output_name": f"{i:03d}_00-orange-cat_02-pull-left-ear_seed-{seed}",
                    "aspect_ratio": "9:16", "duration_seconds": "4", "fps": "24"})
        self.write_rows()

    def write_rows(self):
        with self.metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def args(self, *extra):
        return entry.build_parser().parse_args([
            "--metadata", str(self.metadata), "--output-dir", str(self.output), *extra])

    def run_on(self, instances, *extra):
        argv = ["--metadata", str(self.metadata), "--output-dir", str(self.output),
                "--server-urls", *[f"http://127.0.0.1:{s.server_port}" for s in instances],
                "--poll-interval", ".001", *extra]
        return entry.main(argv)

    def use_compact_metadata(self):
        prompt = self.output / "motion.v1.txt"
        prompt.write_text(PROMPT)
        shutil.copyfile(self.media / "source.mp4", self.output / "source.mp4")
        for row in self.rows:
            row.pop("prompt")
            row.update(prompt_file="motion.v1.txt",
                       prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                       source_reference_video="source.mp4",
                       source_reference_sha256=ref.media_digest(self.output / "source.mp4"),
                       reference_video="prepared/source.motion.mp4", reference_duration_seconds="4")
        self.write_rows()

    def test_fresh_checkout_dry_run_writes_nothing_and_previews_are_opt_in(self):
        self.use_compact_metadata()
        before = {p.relative_to(self.root) for p in self.root.rglob("*")}
        argv = ["--metadata", str(self.metadata), "--output-dir", str(self.output), "--dry-run"]
        self.assertEqual(entry.main(argv), 0)
        self.assertEqual(before, {p.relative_to(self.root) for p in self.root.rglob("*")})
        self.assertEqual(entry.main([*argv, "--write-requests", "--limit", "1"]), 0)
        self.assertEqual(len(list((self.output / "requests/drag_ear").glob("*.json"))), 1)
        self.assertFalse((self.output / "prepared").exists())

    def test_fresh_checkout_builds_silent_cache_and_rebuild_preserves_resume(self):
        self.use_compact_metadata()
        with servers() as instances:
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 0)
            cache = self.output / "prepared/source.motion.mp4"
            streams = ref.probe(cache)["streams"]
            self.assertFalse(any(s["codec_type"] == "audio" for s in streams))
            self.assertFalse((self.output / "requests").exists())
            self.assertEqual(instances[0].submitted[0]["prompt"], PROMPT)
            shutil.rmtree(cache.parent)
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 0)
            self.assertTrue(cache.is_file())
            self.assertEqual(len(instances[0].submitted), 1)
        (self.output / "motion.v1.txt").write_text(PROMPT + "changed")
        with self.assertRaisesRegex(base.BatchError, "提示词文件已更新"):
            ref.load_cases(self.args("--dry-run"))

    def test_payload_names_unicode_and_explicit_seed(self):
        cases = ref.load_cases(self.args())
        request = ref.build_request(cases[0], self.args())
        self.assertEqual(request["task"], "ref2va")
        self.assertEqual(request["target"], {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 4})
        self.assertEqual([c["role"] for c in request["conditions"]], ["reference", "reference"])
        self.assertEqual([c["type"] for c in request["conditions"]], ["image", "video"])
        self.assertTrue(all("frame_index" not in c for c in request["conditions"]))
        self.assertTrue(request["conditions"][1]["uri"].startswith("data:video/mp4;base64,"))
        self.assertEqual(cases[1].seed, 10000)
        self.assertEqual(ref.case_paths(self.output, cases[1]).video,
                         self.output / "videos/drag_ear" / (self.rows[1]["output_name"] + ".mp4"))
        file_request = ref.build_request(cases[0], self.args("--file-uri"))
        self.assertTrue(file_request["conditions"][0]["uri"].startswith("file:///"))
        preview = ref.build_request(cases[0], self.args(), preview=True)
        self.assertIn("<omitted>", preview["conditions"][1]["uri"])
        self.assertNotIn("source_path", request["conditions"][0])

    def test_first_frame_lock_is_explicit_and_changes_resume_fingerprint(self):
        case = ref.load_cases(self.args("--limit", "1"))[0]
        args = self.args("--limit", "1", "--lock-first-frame")
        locked = ref.load_cases(args)[0]
        request = ref.build_request(locked, args)
        self.assertEqual([c["role"] for c in request["conditions"]],
                         ["keyframe", "reference", "reference"])
        self.assertEqual([c.get("frame_index") for c in request["conditions"]], [0, None, None])
        self.assertEqual(request["conditions"][0]["uri"], request["conditions"][1]["uri"])
        self.assertNotEqual(case.request_fingerprint, locked.request_fingerprint)
        paths = ref.case_paths(self.output, case)
        base.atomic_write_json(paths.state, ref.state_payload(case, paths, status="completed"))
        with self.assertRaisesRegex(base.BatchError, "参数不同"):
            ref.load_cases(args)

    def test_reference_only_server_downloads_default_and_rejects_lock_without_fallback(self):
        attempts = []

        def submit_reference_only(handler):
            request = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
            attempts.append(request)
            if any(c["role"] == "keyframe" for c in request["conditions"]):
                handler.reply({"detail": "Ref2VA does not support keyframe conditions"}, 400)
                return
            handler.server.submitted.append(request)
            handler.reply({"id": f"job-{len(handler.server.submitted)}", "status": "queued"})

        with mock.patch.object(Handler, "do_POST", submit_reference_only), servers() as instances:
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 0)
            files = list((self.output / "videos/drag_ear").glob("*.mp4"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].stem, self.rows[0]["output_name"])
            self.assertEqual(self.run_on(instances[:1], "--limit", "1", "--lock-first-frame",
                                        "--output-dir", str(self.output / "locked")), 1)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(len(instances[0].submitted), 1)
            self.assertNotIn("/server_info", instances[0].reads)
            state = base.read_json(next((self.output / "locked/states/drag_ear").glob("*.json")))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["server_response"]["status"], "rejected")

    def test_two_replicas_download_and_resume_without_resubmit(self):
        with servers(barrier=threading.Barrier(2)) as instances:
            self.assertEqual(self.run_on(instances), 0)
            self.assertTrue(all(s.submitted for s in instances))
            self.assertEqual(sum(len(s.submitted) for s in instances), 4)
            files = sorted((self.output / "videos/drag_ear").glob("*.mp4"))
            self.assertEqual({p.stem for p in files}, {r["output_name"] for r in self.rows})
            self.assertEqual(self.run_on(instances), 0)
            self.assertEqual(sum(len(s.submitted) for s in instances), 4)
            state = json.loads(next((self.output / "states/drag_ear").glob("*.json")).read_text())
            self.assertEqual(len(state["request_fingerprint"]), 64)
            summary = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual(summary["completed"], 4)
            self.assertEqual(summary["manifests"], [str(self.output.resolve() / "metadata.manifest.json")])

    def test_changed_prompt_and_material_refuse_stale_results(self):
        case = ref.load_cases(self.args())[0]
        paths = ref.case_paths(self.output, case)
        base.atomic_write_json(paths.state, ref.state_payload(case, paths, status="completed"))
        self.rows[0]["prompt"] += "\nMore motion."
        self.write_rows()
        with self.assertRaisesRegex(base.BatchError, "参数不同"):
            ref.load_cases(self.args())
        ref.load_cases(self.args("--force"))
        self.rows[0]["prompt"] = PROMPT
        self.write_rows()
        with (self.root / "猫.png").open("ab") as handle:
            handle.write(b"changed-content")
        with self.assertRaisesRegex(base.BatchError, "参数不同"):
            ref.load_cases(self.args())

    def test_inflight_owned_by_original_server_and_missing_id_resubmits(self):
        with servers() as instances:
            cases = ref.load_cases(self.args("--limit", "1"))
            case = cases[0]
            owner = f"http://127.0.0.1:{instances[1].server_port}"
            paths = ref.case_paths(self.output, case)
            base.atomic_write_json(paths.state, ref.state_payload(case, paths, status="queued",
                                   server_url=owner, video_id="missing"))
            self.assertEqual(self.run_on(instances, "--limit", "1"), 0)
            self.assertFalse(instances[0].submitted)
            self.assertEqual(len(instances[1].submitted), 1)

    def test_failed_job_requires_retry_flag_and_force_archives(self):
        with servers() as instances:
            instances[0].failure_prompt = PROMPT
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 1)
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 1)
            self.assertEqual(len(instances[0].submitted), 1)
            instances[0].failure_prompt = None
            self.assertEqual(self.run_on(instances[:1], "--limit", "1", "--retry-failed"), 0)
            self.assertEqual(len(instances[0].submitted), 2)
            self.assertEqual(self.run_on(instances[:1], "--limit", "1", "--force"), 0)
            self.assertEqual(len(instances[0].submitted), 3)
            self.assertEqual(len(list((self.output / "history").rglob("*.mp4"))), 1)

    def test_unknown_post_is_not_retried(self):
        case = ref.load_cases(self.args("--limit", "1"))[0]
        with servers() as instances:
            with mock.patch.object(base.SGLangClient, "submit", side_effect=base.ApiError("lost response")) as submit:
                self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 1)
                self.assertEqual(submit.call_count, 1)
            state = base.read_json(ref.case_paths(self.output, case).state)
            self.assertEqual(state["status"], "submission_unknown")
            self.assertEqual(self.run_on(instances[:1], "--limit", "1"), 2)
            self.assertFalse(instances[0].submitted)

    def test_force_recovers_corrupt_state_by_archiving(self):
        case = ref.load_cases(self.args("--limit", "1"))[0]
        paths = ref.case_paths(self.output, case)
        paths.state.parent.mkdir(parents=True)
        paths.state.write_text("{partial state")
        with self.assertRaises(base.BatchError):
            ref.load_cases(self.args("--limit", "1"))
        with servers() as instances:
            self.assertEqual(self.run_on(instances[:1], "--limit", "1", "--force"), 0)
        archived = list((self.output / "history").rglob(paths.state.name))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), "{partial state")

    def test_reordered_rows_keep_names_and_invalid_ratio_is_caught(self):
        original = {c.name for c in ref.load_cases(self.args())}
        self.rows.reverse()
        self.write_rows()
        self.assertEqual({c.name for c in ref.load_cases(self.args())}, original)
        self.rows[0]["aspect_ratio"] = "15:26"
        self.write_rows()
        with self.assertRaisesRegex(base.BatchError, "aspect_ratio"):
            ref.load_cases(self.args())

    def test_server_commands_select_ref2va_and_unique_ports(self):
        args = self.args("--start-servers")
        commands, ports = runner.server_commands(args)
        self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual([runner.replica_ports(args, n)["HTTP"] for n in range(2)], [30110, 30112])
        for _, command in commands:
            self.assertEqual(command[command.index("--model-variant") + 1], "ref2va")
            self.assertNotIn("--served-model-name", command)

    def test_serve_only_launches_without_metadata_or_runtime_probes(self):
        args = self.args("--serve-only", "--metadata", str(self.root / "missing.csv"),
                         "--service-dir", str(self.root / "services"))
        stop = threading.Event()
        processes = [mock.Mock(pid=100 + i, **{"poll.return_value": None}) for i in range(2)]

        def stop_after_ready(message):
            if message.startswith("全部服务就绪"):
                stop.set()

        with mock.patch.object(runner.subprocess, "Popen", side_effect=processes) as launch, \
                mock.patch.object(runner.subprocess, "run") as runtime_probe, \
                mock.patch.object(runner.shutil, "which", return_value="/test/sglang") as which, \
                mock.patch.object(runner, "check_ports_available"), \
                mock.patch.object(runner, "stop_servers") as cleanup, \
                mock.patch.object(base.SGLangClient, "check_server") as health, \
                mock.patch.object(base, "log", side_effect=stop_after_ready), \
                mock.patch.object(ref, "load_cases", side_effect=AssertionError("read CSV")):
            self.assertEqual(runner.run(args, stop, backend=ref), 0)
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(health.call_count, 2)
            which.assert_called_once_with("sglang")
            runtime_probe.assert_not_called()
            cleanup.assert_called_once_with(processes)


if __name__ == "__main__":
    unittest.main()
