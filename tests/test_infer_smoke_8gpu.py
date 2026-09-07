from __future__ import annotations

import contextlib
import argparse
import csv
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import infer_smoke_8gpu as smoke


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, status=200):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.submitted.append(request)
        video_id = f"job-{len(self.server.submitted)}"
        if request["prompt"] == getattr(self.server, "failure_prompt", None):
            self.server.failed_ids.add(video_id)
        if len(self.server.submitted) == 1 and self.server.barrier:
            self.server.barrier.wait(timeout=5)
        self.reply({"id": video_id, "status": "queued"})

    def do_GET(self):
        self.server.reads.append(self.path)
        if self.path == "/models":
            self.reply({"model_path": "mock"})
        elif self.path.endswith("/content"):
            self.reply(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isom")
        elif self.path.startswith("/v1/videos/"):
            time.sleep(self.server.delay)
            video_id = self.path.split("/")[-1]
            if video_id == "missing":
                self.reply({"detail": "not found"}, 404)
            elif video_id in self.server.failed_ids:
                self.reply({"id": video_id, "status": "failed", "error": "mock OOM"})
            else:
                self.reply({"id": video_id, "status": "completed"})
        else:
            self.reply({}, 404)


@contextlib.contextmanager
def servers(*, barrier=None):
    instances = []
    for delay in (.12, .005):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.submitted, server.reads = [], []
        server.failed_ids = set()
        server.barrier, server.delay = barrier, delay
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        instances.append((server, thread))
    try:
        yield [server for server, _ in instances]
    finally:
        for server, thread in instances:
            server.shutdown()
            server.server_close()
            thread.join()


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "cat.png").write_bytes(b"test-image")
        self.metadata = []
        for version in ("v2", "v1"):
            path = self.root / f"{version}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["input_image", "prompt"])
                for i in range(4):
                    writer.writerow(["cat.png", f"{version}: prompt {i}\nsecond line"])
            self.metadata.append(path)
        self.output = self.root / "out"

    def argv(self, instances, *extra):
        return ["--metadata", str(self.metadata[0]), "--metadata-v1", str(self.metadata[1]),
                "--output-dir", str(self.output), "--poll-interval", ".001",
                "--server-urls", *[f"http://127.0.0.1:{s.server_port}" for s in instances],
                *extra]

    def test_two_replicas_balance_and_resume_completed(self):
        # Both first POSTs must be in flight together or the barrier fails.
        with servers(barrier=threading.Barrier(2)) as instances:
            self.assertEqual(smoke.main(self.argv(instances)), 0)
            counts = [len(s.submitted) for s in instances]
            self.assertEqual(sum(counts), 8)
            self.assertGreater(counts[1], counts[0])
            self.assertEqual(len(list(self.output.glob("*/videos/*.mp4"))), 8)
            for label in ("metadata_smoke_v2", "metadata_smoke_v1"):
                manifest = json.loads((self.output / label / "manifest.json").read_text())
                self.assertEqual(manifest["completed"], 4)
                self.assertEqual([r["seed"] for r in manifest["results"]], list(range(4)))
            self.assertEqual(smoke.main(self.argv(instances)), 0)
            self.assertEqual([len(s.submitted) for s in instances], counts)
            summary = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual(summary["generated"], 0)

    def seed_state(self, args, owner, video_id="old-id", status="running"):
        datasets, _, _ = smoke.prepare(args, args.server_urls)
        options, cases = datasets[0]
        paths = smoke.batch.case_paths(options.output_dir, cases[0])
        smoke.batch.atomic_write_json(paths.state, {
            "video_id": video_id, "server_url": owner, "status": status,
            "server_response": {"status": status},
        })

    def test_resume_keeps_server_affinity_when_url_order_changes(self):
        with servers() as instances:
            args = smoke.build_parser().parse_args(self.argv(instances, "--limit", "1"))
            self.seed_state(args, args.server_urls[1])
            # Swap order: persisted owner must win over order or load balancing.
            self.assertEqual(smoke.main(self.argv(instances[::-1], "--limit", "1")), 0)
            self.assertIn("/v1/videos/old-id", instances[1].reads)
            self.assertNotIn("/v1/videos/old-id", instances[0].reads)
            self.assertEqual(sum(len(s.submitted) for s in instances), 1)

    def test_restart_404_resubmits_and_explicit_failed_retry(self):
        for video_id, status, extra in (("missing", "running", []),
                                        ("failed-id", "failed", ["--retry-failed"])):
            with self.subTest(video_id=video_id), servers() as instances:
                self.output = self.root / video_id
                argv = self.argv(instances, "--limit", "1", *extra)
                args = smoke.build_parser().parse_args(argv)
                self.seed_state(args, args.server_urls[0], video_id, status)
                self.assertEqual(smoke.main(argv), 0)
                self.assertEqual(sum(len(s.submitted) for s in instances), 2)

    def test_missing_owner_rejected_before_any_submission(self):
        with servers() as instances:
            argv = self.argv(instances, "--limit", "1")
            args = smoke.build_parser().parse_args(argv)
            self.seed_state(args, "http://127.0.0.1:9")
            self.assertEqual(smoke.main(argv), 2)
            self.assertFalse(any(s.submitted for s in instances))

    def test_failed_job_does_not_stop_queue_and_can_be_retried(self):
        with servers() as instances:
            for server in instances:
                server.failure_prompt = "v2: prompt 0\nsecond line"
            argv = self.argv(instances, "--limit", "2")
            self.assertEqual(smoke.main(argv), 1)
            manifest = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual((manifest["completed"], manifest["failed"]), (3, 1))
            self.assertEqual(sum(len(s.submitted) for s in instances), 4)
            for server in instances:
                server.failure_prompt = None
            self.assertEqual(smoke.main([*argv, "--retry-failed"]), 0)
            self.assertEqual(sum(len(s.submitted) for s in instances), 5)

    def test_output_lock_blocks_second_writer(self):
        self.output.mkdir()
        with (self.output / ".inference.lock").open("a") as handle:
            smoke.fcntl.flock(handle, smoke.fcntl.LOCK_EX | smoke.fcntl.LOCK_NB)
            self.assertEqual(smoke.main([
                "--output-dir", str(self.output), "--dry-run"
            ]), 2)

    def test_invalid_gpu_groups(self):
        args = smoke.build_parser().parse_args(["--gpu-groups", "0,1,2,3", "3,4,5,6"])
        with self.assertRaises(smoke.batch.BatchError):
            smoke.server_commands(args)

    def test_default_commands_work_with_cli_without_nccl_port(self):
        # Reproduce Diffusion's CLI, which exposes master/scheduler ports
        # but rejects --nccl-port.
        legacy_cli = argparse.ArgumentParser()
        for flag in ("--model-path", "--num-gpus", "--tp-size", "--ulysses-degree",
                     "--performance-mode", "--host", "--port", "--model-variant",
                     "--master-port", "--scheduler-port"):
            legacy_cli.add_argument(flag)
        args = smoke.build_parser().parse_args([])
        commands, ports = smoke.server_commands(args)
        parsed = [legacy_cli.parse_args(command[2:]) for _, command in commands]
        self.assertEqual([item.port for item in parsed], ["30010", "30011"])
        self.assertEqual([item.master_port for item in parsed], ["31010", "31011"])
        self.assertEqual([item.scheduler_port for item in parsed], ["32010", "32011"])
        self.assertEqual(len(set(ports)), 6)
        self.assertEqual([group for group, _ in commands], ["0,1,2,3", "4,5,6,7"])

    def test_legacy_nccl_option_maps_to_master_ports(self):
        args = smoke.build_parser().parse_args(["--base-nccl-port", "31010"])
        commands, ports = smoke.server_commands(args)
        self.assertEqual([command[command.index("--master-port") + 1]
                          for _, command in commands], ["31010", "31011"])
        self.assertFalse(any("--nccl-port" in command for _, command in commands))
        self.assertEqual(set(ports), {30010, 30011, 31010, 31011, 32010, 32011})
        args.base_master_port = 30011
        with self.assertRaises(smoke.batch.BatchError):
            smoke.server_commands(args)

    def test_shared_port_override_is_rejected(self):
        for token in ("--master-port=30005", "--scheduler-port", "--port=30010", "--nccl-port"):
            args = smoke.build_parser().parse_args([f"--server-arg={token}"])
            with self.subTest(token=token), self.assertRaises(smoke.batch.BatchError):
                smoke.server_commands(args)

    def test_live_master_listener_blocks_launch(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with mock.patch.object(smoke.shutil, "which", return_value="sglang"), \
                    mock.patch.object(smoke.subprocess, "Popen") as launch:
                self.assertEqual(smoke.main([
                    "--metadata", str(self.metadata[0]), "--metadata-v1", str(self.metadata[1]),
                    "--output-dir", str(self.output), "--base-master-port", str(port)
                ]), 2)
                launch.assert_not_called()

    def test_launch_two_processes_with_distinct_listeners_and_separate_run_logs(self):
        # Real child processes bind all three ports, so using a shared default
        # reproduces EADDRINUSE. No GPU or installed SGLang is needed.
        stub = self.root / "sglang_stub.py"
        stub.write_text(f'''import argparse, socket, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_infer_smoke_8gpu import Handler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument("command", choices=["serve"])
for flag in ("--model-path", "--num-gpus", "--tp-size", "--ulysses-degree",
             "--performance-mode", "--host", "--port", "--model-variant"):
    p.add_argument(flag)
p.add_argument("--master-port", type=int, default=30005)
p.add_argument("--scheduler-port", type=int, default=5555)
a = p.parse_args()
listeners = []
for port in (a.master_port, a.scheduler_port):
    listener = socket.socket()
    listener.bind(("127.0.0.1", port))
    listener.listen()
    listeners.append(listener)
s = ThreadingHTTPServer((a.host, int(a.port)), Handler)
s.submitted, s.reads, s.failed_ids = [], [], set()
s.barrier, s.delay = None, 0
s.serve_forever()
''')
        # Find a currently unused six-port block, reserving all of it while
        # checking; release it immediately before the launcher preflight.
        for _ in range(100):
            with contextlib.ExitStack() as stack:
                first = stack.enter_context(socket.socket())
                first.bind(("127.0.0.1", 0))
                base = first.getsockname()[1]
                if base > 65530:
                    continue
                try:
                    for port in range(base + 1, base + 6):
                        stack.enter_context(socket.socket()).bind(("127.0.0.1", port))
                except OSError:
                    continue
                break
        else:
            self.fail("Could not allocate six free test ports")
        argv = ["--metadata", str(self.metadata[0]), "--metadata-v1", str(self.metadata[1]),
                "--output-dir", str(self.output), "--limit", "1", "--startup-timeout", "15",
                "--base-port", str(base), "--base-master-port", str(base + 2),
                "--base-scheduler-port", str(base + 4)]
        real_popen = smoke.subprocess.Popen
        children = []

        def launch(command, **kwargs):
            process = real_popen([sys.executable, str(stub), *command[1:]], **kwargs)
            children.append(process)
            return process

        with mock.patch.object(smoke.shutil, "which", return_value=str(stub)), \
                mock.patch.object(smoke.subprocess, "Popen", side_effect=launch):
            self.assertEqual(smoke.main(argv), 0)
            original_logs = {path: path.read_bytes() for path in self.output.glob("logs/*/*.log")}
            self.assertEqual(len(original_logs), 2)
            self.assertEqual(smoke.main(argv), 0)
        self.assertEqual(len(list(self.output.glob("logs/*/*.log"))), 4)
        for path, contents in original_logs.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_cancel_interrupts_poll_wait(self):
        event = threading.Event()
        event.set()
        args = smoke.build_parser().parse_args([])
        client = smoke.make_client("http://127.0.0.1:9", args, event)
        with self.assertRaises(smoke.batch.BatchError):
            client.pause(100)


if __name__ == "__main__":
    unittest.main()
