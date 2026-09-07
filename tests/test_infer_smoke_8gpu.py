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

    def launch_argv(self):
        # Reserve the full eight-port footprint while finding an unused block.
        for _ in range(100):
            with contextlib.ExitStack() as stack:
                first = stack.enter_context(socket.socket())
                first.bind(("127.0.0.1", 0))
                base = first.getsockname()[1]
                if base > 65528:
                    continue
                try:
                    for port in range(base + 1, base + 8):
                        stack.enter_context(socket.socket()).bind(("127.0.0.1", port))
                except OSError:
                    continue
                break
        else:
            self.fail("Could not allocate eight free test ports")
        return ["--metadata", str(self.metadata[0]), "--metadata-v1", str(self.metadata[1]),
                "--output-dir", str(self.output), "--limit", "1", "--startup-timeout", "15",
                "--base-port", str(base), "--base-master-port", str(base + 4),
                "--base-scheduler-port", str(base + 6)]

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
        self.assertEqual([item.port for item in parsed], ["30010", "30012"])
        self.assertEqual([item.master_port for item in parsed], ["31010", "31011"])
        self.assertEqual([item.scheduler_port for item in parsed], ["32010", "32011"])
        self.assertEqual(set(ports), {30010, 30011, 30012, 30013,
                                     31010, 31011, 32010, 32011})
        self.assertEqual([group for group, _ in commands], ["0,1,2,3", "4,5,6,7"])

    def test_legacy_nccl_option_maps_to_master_ports(self):
        args = smoke.build_parser().parse_args(["--base-nccl-port", "31010"])
        commands, ports = smoke.server_commands(args)
        self.assertEqual([command[command.index("--master-port") + 1]
                          for _, command in commands], ["31010", "31011"])
        self.assertFalse(any("--nccl-port" in command for _, command in commands))
        self.assertEqual(set(ports), {30010, 30011, 30012, 30013,
                                     31010, 31011, 32010, 32011})
        args.base_master_port = 30011
        with self.assertRaises(smoke.batch.BatchError):
            smoke.server_commands(args)

    def test_shared_port_override_is_rejected(self):
        for token in ("--master-port=30005", "--scheduler-port", "--port=30010", "--nccl-port",
                      "--broker-port=30011", "--broker-port", "--host=0.0.0.0"):
            args = smoke.build_parser().parse_args([f"--server-arg={token}"])
            with self.subTest(token=token), self.assertRaises(smoke.batch.BatchError):
                smoke.server_commands(args)

    def test_broker_cannot_overlap_master_or_scheduler(self):
        for flag, port in (("--base-master-port", 30011), ("--base-scheduler-port", 30013),
                           ("--base-master-port", 30009)):
            args = smoke.build_parser().parse_args([flag, str(port)])
            with self.subTest(flag=flag, port=port), self.assertRaisesRegex(
                    smoke.batch.BatchError, "端口 .*冲突"):
                smoke.server_commands(args)

    def test_port_range_includes_last_broker(self):
        args = smoke.build_parser().parse_args(["--base-port", "65532"])
        _, ports = smoke.server_commands(args)
        self.assertIn(65535, ports)
        for flag, port in (("--base-port", 65533), ("--base-port", 0),
                           ("--base-master-port", 65535), ("--base-scheduler-port", 65535)):
            args = smoke.build_parser().parse_args([flag, str(port)])
            with self.subTest(flag=flag), self.assertRaisesRegex(
                    smoke.batch.BatchError, "超出 1–65535"):
                smoke.server_commands(args)

    def test_live_listener_on_any_port_blocks_all_launches(self):
        argv = self.launch_argv()
        args = smoke.build_parser().parse_args(argv)
        for index in range(2):
            for role, port in smoke.replica_ports(args, index).items():
                for host in ("127.0.0.1", "0.0.0.0"):
                    with self.subTest(index=index, role=role, host=host), socket.socket() as listener:
                        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        listener.bind((host, port))
                        listener.listen()
                        with mock.patch.object(smoke.shutil, "which", return_value="sglang"), \
                                mock.patch.object(smoke.subprocess, "Popen",
                                                  side_effect=AssertionError("端口占用时启动了模型")), \
                                mock.patch.object(smoke.batch, "log") as log:
                            self.assertEqual(smoke.main(argv), 2)
                            self.assertIn(f"{role} 端口 {port} 不可用", log.call_args.args[0])

    def write_server_stub(self):
        # Match all FOUR listeners, including SGLang's implicit HTTP+1 broker.
        # The previous three-port stub could not reproduce the production bug.
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
for port in (a.master_port, a.scheduler_port, int(a.port) + 1):
    listener = socket.socket()
    listener.bind(("127.0.0.1", port))
    listener.listen()
    listeners.append(listener)
s = ThreadingHTTPServer((a.host, int(a.port)), Handler)
s.submitted, s.reads, s.failed_ids = [], [], set()
s.barrier, s.delay = None, 0
print("All four listeners bound", flush=True)
s.serve_forever()
''')
        return stub

    def test_launch_two_processes_with_distinct_listeners_and_separate_run_logs(self):
        stub = self.write_server_stub()
        argv = self.launch_argv()
        base = int(argv[argv.index("--base-port") + 1])
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
            self.assertEqual({p.name for p in original_logs},
                             {f"server_{base}.log", f"server_{base + 2}.log"})
            self.assertTrue(all(b"All four listeners bound" in value
                                for value in original_logs.values()))
            summary = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual(summary["server_urls"], [f"http://127.0.0.1:{base}",
                                                     f"http://127.0.0.1:{base + 2}"])
            self.assertEqual(smoke.main(argv), 0)
        self.assertEqual(len(list(self.output.glob("logs/*/*.log"))), 4)
        for path, contents in original_logs.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_old_adjacent_http_ports_fail_and_print_diagnostics_and_cleanup(self):
        stub = self.write_server_stub()
        argv = self.launch_argv()
        base = int(argv[argv.index("--base-port") + 1])
        real_popen = smoke.subprocess.Popen
        children = []

        def launch(command, **kwargs):
            command = list(command)
            if children:
                # Inject the old layout AFTER preflight to reproduce the
                # log's HTTP/Broker bind failure in real child processes.
                command[command.index("--port") + 1] = str(base + 1)
            process = real_popen([sys.executable, str(stub), *command[1:]], **kwargs)
            children.append(process)
            return process

        with mock.patch.object(smoke.shutil, "which", return_value=str(stub)), \
                mock.patch.object(smoke.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(smoke.batch, "log") as log, \
                mock.patch.object(smoke, "run_work") as work:
            self.assertEqual(smoke.main(argv), 2)
            work.assert_not_called()
        messages = "\n".join(call.args[0] for call in log.call_args_list)
        self.assertIn("Address already in use", messages)
        self.assertIn("服务日志末尾:", messages)
        self.assertIn("已退出", messages)
        self.assertTrue(all(child.poll() is not None for child in children))
        args = smoke.build_parser().parse_args(argv)
        smoke.check_ports_available(args, smoke.server_commands(args)[0])

    def test_startup_timeout_reports_log_tails_and_cleans_up(self):
        stub = self.write_server_stub()
        argv = self.launch_argv() + ["--startup-timeout", ".1"]
        real_popen = smoke.subprocess.Popen
        children = []

        def launch(command, **kwargs):
            process = real_popen([sys.executable, str(stub), *command[1:]], **kwargs)
            children.append(process)
            return process

        with mock.patch.object(smoke.shutil, "which", return_value=str(stub)), \
                mock.patch.object(smoke.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(smoke.batch.SGLangClient, "check_server",
                                  side_effect=smoke.batch.ApiError("still loading")), \
                mock.patch.object(smoke.batch, "log") as log:
            self.assertEqual(smoke.main(argv), 2)
        messages = "\n".join(call.args[0] for call in log.call_args_list)
        self.assertIn("模型启动超时", messages)
        self.assertIn("All four listeners bound", messages)
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
