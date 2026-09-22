"""Exercise real local HTTP traffic with an intentionally broken enterprise proxy."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import batch_fl2va_sglang as batch


VIDEO = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isom"
PROXY_ERROR = b"<!doctype html><title>HIS Proxy Notification</title>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, status=200):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.requests.append(("GET", self.path))
        if self.server.is_proxy:
            self.reply(PROXY_ERROR, 504)
        elif self.path == "/models":
            self.reply({"model_path": "mock-ref2va"}, self.server.models_status)
        elif self.path == "/health":
            self.reply(b"ok")
        elif self.path == "/v1/videos/job-1":
            self.reply({"id": "job-1", "status": "completed"})
        elif self.path == "/v1/videos/job-1/content":
            self.reply(VIDEO)
        else:
            self.reply({}, 404)

    def do_POST(self):
        self.server.requests.append(("POST", self.path))
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.server.is_proxy:
            self.reply(PROXY_ERROR, 504)
        elif self.path == "/v1/videos":
            self.server.submitted.append(json.loads(body))
            self.reply({"id": "job-1", "status": "queued"})
        else:
            self.reply({}, 404)


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


@contextlib.contextmanager
def local_server(*, is_proxy=False, ipv6=False):
    server_class = IPv6Server if ipv6 else ThreadingHTTPServer
    server = server_class(("::1" if ipv6 else "127.0.0.1", 0), Handler)
    server.is_proxy, server.models_status = is_proxy, 200
    server.requests, server.submitted = [], []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextlib.contextmanager
def force_proxy(proxy):
    url = f"http://127.0.0.1:{proxy.server_port}"
    environment = {key: url for key in (
        "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
    )}
    environment.update(no_proxy="", NO_PROXY="")
    # Restore the caller's environment and global opener when this test exits.
    with mock.patch.dict(os.environ, environment, clear=True), \
            mock.patch.object(urllib.request, "proxy_bypass", return_value=False), \
            mock.patch.object(urllib.request, "_opener", None):
        yield


def client(url, *, retries=0):
    return batch.SGLangClient(server_url=url, api_key=None, request_timeout=2,
                              retries=retries, retry_backoff=0)


class ProxyRegressionTest(unittest.TestCase):
    def assert_generation_direct(self, instance, url):
        before = len(instance.requests)
        api = client(url)
        self.assertEqual(api.check_server(), {"model_path": "mock-ref2va"})
        request = {"task": "ref2va", "prompt": "cat identity and motion"}
        self.assertEqual(api.submit(request)["id"], "job-1")
        self.assertEqual(api.retrieve("job-1")["status"], "completed")
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            api.download("job-1", video)
            self.assertEqual(video.read_bytes(), VIDEO)
        self.assertEqual(instance.submitted[-1], request)
        self.assertEqual(instance.requests[before:], [
            ("GET", "/models"), ("POST", "/v1/videos"),
            ("GET", "/v1/videos/job-1"), ("GET", "/v1/videos/job-1/content"),
        ])
        return api

    def test_full_generation_bypasses_proxy_for_loopback_spellings(self):
        with local_server() as service, local_server(is_proxy=True) as proxy, force_proxy(proxy):
            url = f"http://127.0.0.1:{service.server_port}"
            # Reproduce the reported failure before exercising the fixed client.
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url + "/models", timeout=2)
            self.assertEqual(caught.exception.code, 504)
            self.assertEqual(caught.exception.read(), PROXY_ERROR)
            caught.exception.close()
            self.assertEqual(proxy.requests, [("GET", url + "/models")])
            self.assertEqual(service.requests, [])
            for host in ("127.0.0.1", "127.0.01", "127.1", "localhost"):
                for suffix in ("", "/v1/"):
                    with self.subTest(host=host, suffix=suffix):
                        api = self.assert_generation_direct(
                            service, f"http://{host}:{service.server_port}{suffix}")
                        normalized_host = "localhost" if host == "localhost" else "127.0.0.1"
                        self.assertEqual(api.server_url,
                                         f"http://{normalized_host}:{service.server_port}")
                        self.assertEqual(len(proxy.requests), 1)

    def test_models_404_health_fallback_remains_direct(self):
        with local_server() as service, local_server(is_proxy=True) as proxy, force_proxy(proxy):
            service.models_status = 404
            api = client(f"http://127.0.01:{service.server_port}/v1")
            self.assertEqual(api.check_server(), {"status": "ok"})
            self.assertEqual(service.requests, [("GET", "/models"), ("GET", "/health")])
            self.assertEqual(proxy.requests, [])

    def test_failed_direct_connection_never_retries_through_proxy(self):
        with socket.socket() as unavailable, local_server(is_proxy=True) as proxy, force_proxy(proxy):
            # Reserve a port without listening so refusal is deterministic.
            unavailable.bind(("127.0.0.1", 0))
            api = client(f"http://127.0.0.1:{unavailable.getsockname()[1]}", retries=1)
            with self.assertRaises(batch.ApiError) as caught:
                api.check_server()
            self.assertIsNone(caught.exception.status_code)
            self.assertIn("请求失败", str(caught.exception))
            self.assertEqual(proxy.requests, [])

    def test_non_loopback_keeps_existing_proxy_policy(self):
        with local_server(is_proxy=True) as proxy, force_proxy(proxy):
            url = "http://sglang-regression.invalid:30010"
            with self.assertRaises(batch.ApiError) as caught:
                client(url).check_server()
            self.assertEqual(caught.exception.status_code, 504)
            self.assertIn("HIS Proxy Notification", caught.exception.body)
            self.assertEqual(proxy.requests, [("GET", url + "/models")])

    def test_ipv6_loopback_generation_remains_direct(self):
        with contextlib.ExitStack() as stack:
            try:
                service = stack.enter_context(local_server(ipv6=True))
            except OSError as exc:
                self.skipTest(f"IPv6 loopback unavailable: {exc}")
            proxy = stack.enter_context(local_server(is_proxy=True))
            stack.enter_context(force_proxy(proxy))
            self.assert_generation_direct(service, f"http://[::1]:{service.server_port}/v1")
            self.assertEqual(proxy.requests, [])


if __name__ == "__main__":
    unittest.main()
