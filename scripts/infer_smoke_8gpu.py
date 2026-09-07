#!/usr/bin/env python3
"""Run both smoke CSVs on independent SGLang replicas, using the proven client."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import os
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import batch_fl2va_sglang as batch


ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = batch.build_parser()
    parser.description = "8 张 H100：两个 4 卡 SGLang 副本，动态分配两份 smoke CSV"
    parser.set_defaults(
        metadata=ROOT / "data_h3/metadata_smoke_v2.csv",
        output_dir=ROOT / "results/smoke",
    )
    parser.add_argument("--metadata-v1", type=Path,
                        default=ROOT / "data_h3/metadata_smoke_v1.csv")
    parser.add_argument("--server-urls", nargs="+",
                        help="连接已有服务，不启动/停止服务；每个 URL 一个独立副本")
    parser.add_argument("--model-path", default=os.environ.get(
        "MODEL_PATH", "/srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3"))
    parser.add_argument("--gpu-groups", nargs="+", default=["0,1,2,3", "4,5,6,7"],
                        help="每个副本的物理 GPU ID；默认两个 4 卡组")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=30010)
    parser.add_argument("--base-nccl-port", type=int,
                        help="可选：仅当所用 SGLang 支持 --nccl-port 时设置；默认沿用服务自身的通信端口配置")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--server-arg", action="append", default=[],
                        help="附加 serve 参数，每个 token 用 --server-arg=TOKEN 传入")
    parser.add_argument("--retry-failed", action="store_true",
                        help="重提服务端明确失败的任务；不重提仍在运行或仅 HTTP 失败的任务")
    parser.epilog = ("--max-concurrency 是每个服务的在途任务数，默认 1；"
                     "--metadata 指定 v2 CSV，--limit 对每份 CSV 分别生效。"
                     "自动启动的服务会在完成或中断时停止。")
    return parser


def make_client(url, args, stop_event, *, probe=False):
    return batch.SGLangClient(
        server_url=url, api_key=args.api_key,
        request_timeout=min(args.request_timeout, 2) if probe else args.request_timeout,
        retries=0 if probe else args.retries,
        retry_backoff=args.retry_backoff, stop_event=stop_event,
    )


def server_commands(args):
    seen = set()
    commands = []
    ports = []
    for index, group in enumerate(args.gpu_groups):
        ids = group.split(",")
        if any(not value.isdigit() for value in ids) or len(set(ids)) != len(ids):
            raise batch.BatchError(f"无效的 GPU 分组: {group}")
        if seen.intersection(ids):
            raise batch.BatchError("副本之间不能共享 GPU")
        seen.update(ids)
        if len(ids) != args.tp_size * args.ulysses_degree:
            raise batch.BatchError(f"GPU 组 {group} 的卡数必须等于 TP × Ulysses")
        port = args.base_port + index
        ports.append(port)
        communication_args = []
        # The known-working deployment does not pass --nccl-port. Some
        # SGLang diffusion releases do not expose it in their CLI at all.
        if args.base_nccl_port is not None:
            nccl = args.base_nccl_port + index
            ports.append(nccl)
            communication_args = ["--nccl-port", str(nccl)]
        commands.append((group, [
            "sglang", "serve", "--model-path", args.model_path,
            "--num-gpus", str(len(ids)), "--tp-size", str(args.tp_size),
            "--ulysses-degree", str(args.ulysses_degree),
            "--performance-mode", "speed", "--host", "127.0.0.1",
            "--port", str(port), *communication_args,
            "--model-variant", "fl2va", *args.server_arg,
        ]))
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise batch.BatchError("HTTP/NCCL 端口必须各不相同且在 1–65535 之间")
    return commands, ports


def prepare(args, urls):
    """Validate every row and route unfinished IDs to their original server."""
    batch.validate_args(args)
    datasets = []
    shared = queue.Queue()
    pinned = {url: queue.Queue() for url in urls}
    for label, metadata in (("metadata_smoke_v2", args.metadata),
                            ("metadata_smoke_v1", args.metadata_v1)):
        options = argparse.Namespace(**vars(args))
        options.metadata = metadata
        options.output_dir = args.output_dir / label
        cases = batch.load_cases(options)
        datasets.append((options, cases))
        for case in cases:
            paths = batch.case_paths(options.output_dir, case)
            previous = {} if args.force else (batch.read_json(paths.state) or {})
            job = (options, case)
            if not args.force and previous.get("video_id") and not batch.looks_like_mp4(paths.video):
                # Legacy single-server state files have no owner: the original
                # --server-url is their only safe default, never a random replica.
                owner = previous.get("server_url") or batch.SGLangClient(
                    server_url=args.server_url, api_key=None, request_timeout=1,
                    retries=0, retry_backoff=1).server_url
                if owner not in pinned:
                    raise batch.BatchError(
                        f"{paths.state}: 未完成任务属于 {owner}；请把该地址加入 "
                        "--server-urls，或用 --force 明确重新生成")
                pinned[owner].put(job)
            else:
                shared.put(job)
    return datasets, shared, pinned


def run_work(args, clients, datasets, shared, pinned, stop_event):
    started = time.monotonic()
    results = {str(options.metadata): [] for options, _ in datasets}
    lock = threading.Lock()

    def worker(client):
        while not stop_event.is_set():
            try:
                job = pinned[client.server_url].get_nowait()
            except queue.Empty:
                try:
                    job = shared.get_nowait()
                except queue.Empty:
                    return
            options, case = job
            paths = batch.case_paths(options.output_dir, case)
            case_started = time.monotonic()
            previous = {} if options.force else (batch.read_json(paths.state) or {})
            previous_id = previous.get("video_id")
            # Only a terminal server failure authorizes retry-failed. A local
            # polling timeout must retain its ID to avoid duplicate GPU work.
            terminal_status = str(previous.get("server_response", {}).get("status", "")).lower()
            if args.retry_failed and terminal_status in batch.FAILED_STATUSES:
                options = argparse.Namespace(**vars(options))
                options.force = True
            try:
                result = batch.run_case(case, options, client)
            except Exception as exc:
                previous = batch.read_json(paths.state) or previous
                result = {
                    **previous,
                    **batch.state_payload(
                        case, paths, status="interrupted" if stop_event.is_set() else "failed",
                        video_id=previous.get("video_id"), server_url=client.server_url,
                        error=str(exc)),
                }
                batch.log(f"[{options.metadata.stem}/{case.name}] 失败: {exc}")
            result["elapsed_seconds"] = round(time.monotonic() - case_started, 3)
            result["resumed_existing"] = bool(
                previous_id and result.get("video_id") == previous_id and not options.force
            )
            result.setdefault("server_url", client.server_url)
            batch.atomic_write_json(paths.state, result)
            with lock:
                results[str(options.metadata)].append(result)
                done = sum(len(items) for items in results.values())
                batch.log(f"总进度 {done}/{sum(len(cases) for _, cases in datasets)}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(clients) * args.max_concurrency
    ) as pool:
        futures = [pool.submit(worker, client) for client in clients
                   for _ in range(args.max_concurrency)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    manifests = []
    for options, cases in datasets:
        items = sorted(results[str(options.metadata)], key=lambda item: item["case_index"])
        failed = [item for item in items if item["status"] != "completed"]
        manifest = {
            "metadata": str(options.metadata.resolve()), "total": len(cases),
            "completed": len(items) - len(failed), "failed": len(failed),
            "pending": len(cases) - len(items), "results": items,
            "failures": [{"name": item["name"], "error": item.get("error")}
                         for item in failed],
        }
        batch.atomic_write_json(options.output_dir / "manifest.json", manifest)
        manifests.append(manifest)
    elapsed = time.monotonic() - started
    generated = sum(item["status"] == "completed" and not item.get("skipped_existing")
                    and not item.get("resumed_existing")
                    for items in results.values() for item in items)
    summary = {
        "server_urls": [client.server_url for client in clients],
        "concurrency_per_server": args.max_concurrency,
        "elapsed_seconds": round(elapsed, 3), "generated": generated,
        "generated_videos_per_hour": round(generated * 3600 / max(elapsed, .001), 2),
        "total": sum(m["total"] for m in manifests),
        "completed": sum(m["completed"] for m in manifests),
        "failed": sum(m["failed"] for m in manifests),
        "pending": sum(m["pending"] for m in manifests),
        "manifests": [str(options.output_dir / "manifest.json") for options, _ in datasets],
    }
    batch.atomic_write_json(args.output_dir / "manifest.json", summary)
    batch.log(f"推理结束: {summary}")
    return 130 if stop_event.is_set() else int(bool(summary["failed"] or summary["pending"]))


def stop_servers(processes):
    # Only signal process groups created by this invocation, never existing servers.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(p.poll() is None for p in processes):
        time.sleep(.2)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(args, stop_event):
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.startup_timeout <= 0 or args.tp_size <= 0 or args.ulysses_degree <= 0:
        raise batch.BatchError("startup-timeout、tp-size 和 ulysses-degree 必须大于 0")
    commands, ports = ([], []) if args.server_urls else server_commands(args)
    urls = args.server_urls or [f"http://127.0.0.1:{args.base_port + i}"
                               for i in range(len(commands))]
    clients = [make_client(url, args, stop_event) for url in urls]
    urls = [client.server_url for client in clients]
    if len(set(urls)) != len(urls):
        raise batch.BatchError("--server-urls 不能重复")
    datasets, shared, pinned = prepare(args, urls)
    batch.log(f"{sum(len(cases) for _, cases in datasets)} 个 case，{len(clients)} 个副本，"
              f"每副本 {args.max_concurrency} 个在途请求；输出: {args.output_dir}")
    for group, command in commands:
        batch.log(f"CUDA_VISIBLE_DEVICES={group} {shlex.join(command)}")
    if args.dry_run:
        for options, _ in datasets:
            batch.run_batch(options)
        return 0

    processes = []
    try:
        if commands:
            if not shutil.which("sglang"):
                raise batch.BatchError("找不到 sglang；请在已跑通推理的 Python 环境执行")
            for port in ports:
                with socket.socket() as sock:
                    try:
                        sock.bind(("127.0.0.1", port))
                    except OSError as exc:
                        raise batch.BatchError(
                            f"端口 {port} 不可用；可用 --server-urls 连接已有服务，"
                            "或修改 --base-port/--base-nccl-port") from exc
            log_dir = args.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            for index, (group, command) in enumerate(commands):
                log_path = log_dir / f"server_{args.base_port + index}.log"
                with log_path.open("a") as handle:
                    process = subprocess.Popen(
                        command, env={**os.environ, "CUDA_VISIBLE_DEVICES": group,
                                      "PYTHONUNBUFFERED": "1"},
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                processes.append(process)
                batch.log(f"启动 pid={process.pid}, GPU={group}, 日志={log_path}")
            deadline, last_log = time.monotonic() + args.startup_timeout, 0.0
            pending = set(urls)
            while pending:
                if stop_event.is_set():
                    return 130
                for index, process in enumerate(processes):
                    if process.poll() is not None:
                        raise batch.BatchError(f"服务 {urls[index]} 已退出 ({process.returncode})；请检查日志")
                for url in list(pending):
                    try:
                        make_client(url, args, stop_event, probe=True).check_server()
                    except batch.ApiError:
                        continue
                    pending.remove(url)
                    batch.log(f"服务就绪: {url}")
                if time.monotonic() >= deadline and pending:
                    raise batch.BatchError(f"模型启动超时: {sorted(pending)}；请检查日志")
                if time.monotonic() - last_log >= 30 and pending:
                    batch.log(f"等待模型加载: {sorted(pending)}")
                    last_log = time.monotonic()
                if pending:
                    stop_event.wait(2)
        elif not args.skip_server_check:
            for client in clients:
                client.check_server()
        return run_work(args, clients, datasets, shared, pinned, stop_event)
    finally:
        stop_servers(processes)


def main(argv=None):
    args = build_parser().parse_args(argv)
    stop_event = threading.Event()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop_event.set())
    try:
        # An OS lock releases automatically on crash; stale lock files are safe.
        args.output_dir = args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / ".inference.lock").open("a") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise batch.BatchError("已有推理进程正在写入该输出目录") from exc
            return run(args, stop_event)
    except (batch.BatchError, OSError) as exc:
        batch.log(f"错误: {exc}")
        return 130 if stop_event.is_set() else 2
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
