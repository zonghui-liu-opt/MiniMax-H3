from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_fl2va_sglang.py"
SPEC = importlib.util.spec_from_file_location("batch_fl2va_sglang", SCRIPT_PATH)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class MockVideoHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, format, *args):
        return

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/models":
            self._json({"model_path": "MiniMaxAI/MiniMax-H3"})
        elif self.path == "/v1/videos/mock-video":
            self._json({"id": "mock-video", "status": "completed"})
        elif self.path == "/v1/videos/mock-video/content":
            body = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isom"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"detail": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/v1/videos":
            self._json({"detail": "not found"}, status=404)
            return
        length = int(self.headers["Content-Length"])
        self.__class__.requests.append(json.loads(self.rfile.read(length)))
        self._json({"id": "mock-video", "status": "queued"})


class BatchFl2VATest(unittest.TestCase):
    def setUp(self):
        MockVideoHandler.requests = []

    def parse_args(self, *extra):
        return batch.build_parser().parse_args(list(extra))

    def test_repository_metadata_builds_loop_requests(self):
        args = self.parse_args(
            "--metadata",
            str(REPO_ROOT / "testsets" / "metadata_6cases_480x832.csv"),
            "--limit",
            "1",
        )
        batch.validate_args(args)
        cases = batch.load_cases(args)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].last_image, cases[0].first_image)
        request = batch.build_request(cases[0], args)
        self.assertEqual(
            [item["frame_index"] for item in request["conditions"]], [0, -1]
        )
        self.assertTrue(
            request["conditions"][0]["uri"].startswith("data:image/png;base64,")
        )
        self.assertEqual(request["target"]["short_edge"], 768)
        self.assertEqual(request["target"]["aspect_ratio"], "auto")

    def test_single_frame_mode(self):
        args = self.parse_args(
            "--metadata",
            str(REPO_ROOT / "testsets" / "metadata_6cases_480x832.csv"),
            "--limit",
            "1",
            "--single-frame",
        )
        case = batch.load_cases(args)[0]
        request = batch.build_request(case, args)
        self.assertEqual([item["frame_index"] for item in request["conditions"]], [0])

    def test_legacy_metadata_exposes_actual_names_and_csv_seed_source(self):
        args = self.parse_args(
            "--metadata", str(REPO_ROOT / "data_h3/metadata_smoke_v4_multiseed.csv"),
            "--limit", "3", "--seed", "888")
        cases = batch.load_cases(args)
        self.assertEqual([case.name for case in cases],
                         ["000_00_01_0", "001_00_02_1", "002_00_03_2"])
        with mock.patch.object(batch, "log") as log:
            batch.log_case_configuration(args, cases)
        messages = "\n".join(call.args[0] for call in log.call_args_list)
        self.assertIn(str(args.metadata.resolve()), messages)
        self.assertIn("3 条读取 CSV 的 seed 列", messages)
        self.assertIn("旧版纯编号输出名", messages)
        self.assertIn("id=001, seed=1 -> 001_00_02_1.mp4", messages)

    def test_end_to_end_with_mock_server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockVideoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "output"
                result = batch.main(
                    [
                        "--metadata",
                        str(REPO_ROOT / "testsets" / "metadata_6cases_480x832.csv"),
                        "--output-dir",
                        str(output_dir),
                        "--server-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--limit",
                        "1",
                        "--poll-interval",
                        "0.01",
                        "--retry-backoff",
                        "0.01",
                    ]
                )
                self.assertEqual(result, 0)
                videos = list((output_dir / "videos").glob("*.mp4"))
                self.assertEqual(len(videos), 1)
                self.assertTrue(batch.looks_like_mp4(videos[0]))
                self.assertEqual(MockVideoHandler.requests[0]["task"], "fl2va")
                self.assertEqual(
                    [
                        item["frame_index"]
                        for item in MockVideoHandler.requests[0]["conditions"]
                    ],
                    [0, -1],
                )
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["completed"], 1)
                self.assertEqual(manifest["failed"], 0)

                second_result = batch.main(
                    [
                        "--metadata",
                        str(REPO_ROOT / "testsets" / "metadata_6cases_480x832.csv"),
                        "--output-dir",
                        str(output_dir),
                        "--server-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--limit",
                        "1",
                        "--poll-interval",
                        "0.01",
                        "--retry-backoff",
                        "0.01",
                    ]
                )
                self.assertEqual(second_result, 0)
                self.assertEqual(len(MockVideoHandler.requests), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
