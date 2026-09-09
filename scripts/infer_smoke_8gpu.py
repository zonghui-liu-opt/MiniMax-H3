#!/usr/bin/env python3
"""Run smoke V5 multi-seed cases against reusable SGLang replicas."""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
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
    parser.description = "8 张 H100：两个 4 卡 SGLang 副本，默认对 Smoke V5 进行首帧引导多 seed 推理"
    parser.set_defaults(
        metadata=ROOT / "data_h3/metadata_smoke_v5_multiseed.csv",
        output_dir=ROOT / "results/v5_multiseed_smoke",
        single_frame=True,
    )
    parser.add_argument("--metadata-v1", type=Path,
                        help="兼容旧入口：显式指定时才追加该 CSV；默认只推理 --metadata")
    service_mode = parser.add_mutually_exclusive_group()
    service_mode.add_argument("--server-urls", nargs="+",
                             help="连接已有服务；默认连接本机 base-port 起的两个副本")
    service_mode.add_argument("--serve-only", action="store_true",
                             help="只启动并保持 SGLang 服务，不读取 CSV；Ctrl-C 停止服务")
    service_mode.add_argument("--start-servers", action="store_true",
                             help="旧的一体化模式：启动服务并推理，结束或中断后停止服务")
    parser.add_argument("--service-dir", type=Path, default=ROOT / "results/sglang_services",
                        help="常驻服务的日志与锁目录，独立于推理 output-dir")
    parser.add_argument("--model-path", default=os.environ.get(
        "MODEL_PATH", "/srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3"))
    parser.add_argument("--gpu-groups", nargs="+", default=["0,1,2,3", "4,5,6,7"],
                        help="每个副本的物理 GPU ID；默认两个 4 卡组")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=30010,
                        help="HTTP 端口起点，副本间隔 2；每个 HTTP+1 留给 ZMQ Broker")
    parser.add_argument("--base-master-port", "--base-nccl-port", dest="base_master_port",
                        type=int, default=31010,
                        help="分布式初始化端口起点；旧名称 --base-nccl-port 是兼容别名，实际传入 --master-port")
    parser.add_argument("--base-scheduler-port", type=int, default=32010,
                        help="调度器端口起点，每个副本使用独立端口")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--server-arg", action="append", default=[],
                        help="附加 serve 参数，每个 token 用 --server-arg=TOKEN 传入")
    parser.add_argument("--retry-failed", action="store_true",
                        help="重提服务端明确失败的任务；不重提仍在运行或仅 HTTP 失败的任务")
    parser.epilog = ("--max-concurrency 是每个服务的在途任务数，默认 1；"
                     "默认只读取 metadata_smoke_v5_multiseed.csv（126 条），--limit 限制该 CSV 的条数。"
                     "仅显式传入 --metadata-v1 时追加第二份 CSV，--limit 对每份分别生效。"
                     "默认复用已有服务，推理结束后服务保留；先用 serve_smoke_8gpu.sh 启动服务。")
    return parser


def make_client(url, args, stop_event, *, probe=False):
    return batch.SGLangClient(
        server_url=url, api_key=args.api_key,
        request_timeout=min(args.request_timeout, 2) if probe else args.request_timeout,
        retries=0 if probe else args.retries,
        retry_backoff=args.retry_backoff, stop_event=stop_event,
    )


def replica_ports(args, index):
    # SGLang Diffusion also binds a ZMQ broker at HTTP+1. Adjacent HTTP
    # ports collide only AFTER expensive model loading has completed.
    http = args.base_port + 2 * index
    return {"HTTP": http, "ZMQ Broker (HTTP+1)": http + 1,
            "master": args.base_master_port + index,
            "scheduler": args.base_scheduler_port + index}


def server_commands(args):
    for token in args.server_arg:
        if token.split("=", 1)[0] in {
            "--port", "--broker-port", "--master-port", "--scheduler-port", "--nccl-port",
        }:
            raise batch.BatchError("请使用 --base-port/--base-master-port/--base-scheduler-port 设置端口，"
                                   "不能通过 --server-arg 给所有副本覆盖为同一个端口")
        if token.split("=", 1)[0] == "--host":
            raise batch.BatchError("自动启动固定使用 127.0.0.1，不能通过 --server-arg 覆盖 --host")
    seen = set()
    commands = []
    port_owners = {}
    for index, group in enumerate(args.gpu_groups):
        ids = group.split(",")
        if any(not value.isdigit() for value in ids) or len(set(ids)) != len(ids):
            raise batch.BatchError(f"无效的 GPU 分组: {group}")
        if seen.intersection(ids):
            raise batch.BatchError("副本之间不能共享 GPU")
        seen.update(ids)
        if len(ids) != args.tp_size * args.ulysses_degree:
            raise batch.BatchError(f"GPU 组 {group} 的卡数必须等于 TP × Ulysses")
        ports = replica_ports(args, index)
        for role, port in ports.items():
            owner = f"副本 {index + 1} (GPU={group}) {role}"
            if not 1 <= port <= 65535:
                raise batch.BatchError(f"{owner} 端口 {port} 超出 1–65535 范围")
            if port in port_owners:
                raise batch.BatchError(f"端口 {port} 冲突：{port_owners[port]} 与 {owner}；"
                                       "每个 HTTP+1 必须留给 ZMQ Broker")
            port_owners[port] = owner
        # Diffusion's distributed rendezvous uses master_port. Checking a
        # shared default before launch is racy: both replicas can choose it
        # before either worker binds. Allocate distinct ports explicitly.
        commands.append((group, [
            "sglang", "serve", "--model-path", args.model_path,
            "--num-gpus", str(len(ids)), "--tp-size", str(args.tp_size),
            "--ulysses-degree", str(args.ulysses_degree),
            "--performance-mode", "speed", "--host", "127.0.0.1",
            "--port", str(ports["HTTP"]), "--master-port", str(ports["master"]),
            "--scheduler-port", str(ports["scheduler"]),
            "--model-variant", getattr(args, "model_variant", "fl2va"), *args.server_arg,
        ]))
    return commands, list(port_owners)


def check_ports_available(args, commands):
    # Check all listeners, including the implicit broker, before loading GPUs.
    for index, (group, _) in enumerate(commands):
        for role, port in replica_ports(args, index).items():
            try:
                # On some OSes SO_REUSEADDR allows a loopback bind alongside
                # a live wildcard listener. Probe it before the restart-safe
                # bind check so it cannot be mistaken for an available port.
                with socket.socket() as probe:
                    probe.settimeout(.2)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        raise OSError(errno.EADDRINUSE, "address already in use")
                with socket.socket() as sock:
                    # TIME_WAIT from the previous run is harmless; a live
                    # listener (including a wildcard bind) must block launch.
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", port))
                    sock.listen(1)
            except OSError as exc:
                raise batch.BatchError(
                    f"副本 {index + 1} (GPU={group}) {role} 端口 {port} 不可用: {exc}；"
                    "尚未启动模型。请停止占用该端口的服务，或修改 "
                    "--base-port/--base-master-port/--base-scheduler-port；"
                    "连接已有服务请使用 --server-urls") from exc


def print_server_log_tails(log_paths):
    for path in log_paths:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 8192))
                tail = "\n".join(handle.read().decode("utf-8", "replace").splitlines()[-40:])
            batch.log(f"服务日志末尾: {path}\n{tail}")
        except OSError as exc:
            batch.log(f"无法读取服务日志 {path}: {exc}")


def prepare(args, urls, *, backend=batch):
    """Validate every row and route unfinished IDs to their original server."""
    backend.validate_args(args)
    datasets = []
    shared = queue.Queue()
    pinned = {url: queue.Queue() for url in urls}
    metadata_paths = [args.metadata]
    if args.metadata_v1 is not None:
        metadata_paths.append(args.metadata_v1)
    labels = [metadata.stem for metadata in metadata_paths]
    if len(set(labels)) != len(labels):
        raise batch.BatchError("CSV 文件名（不含扩展名）不能重复，以免覆盖同一输出子目录")
    for metadata in metadata_paths:
        options = argparse.Namespace(**vars(args))
        options.metadata = metadata
        options.output_dir = (args.output_dir if getattr(args, "flat_output", False)
                              else args.output_dir / metadata.stem)
        cases = backend.load_cases(options)
        if not args.dry_run:
            backend.log_case_configuration(options, cases)
        datasets.append((options, cases))
        for case in cases:
            paths = backend.case_paths(options.output_dir, case)
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


def dataset_manifest_path(options):
    name = (f"{options.metadata.stem}.manifest.json" if getattr(options, "flat_output", False)
            else "manifest.json")
    return options.output_dir / name


def run_work(args, clients, datasets, shared, pinned, stop_event, *, backend=batch):
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
            paths = backend.case_paths(options.output_dir, case)
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
                result = backend.run_case(case, options, client)
            except Exception as exc:
                previous = batch.read_json(paths.state) or previous
                result = {
                    **previous,
                    **backend.state_payload(
                        case, paths, status=("submission_unknown" if previous.get("status") == "submission_unknown"
                                            else "interrupted" if stop_event.is_set() else "failed"),
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
        batch.atomic_write_json(dataset_manifest_path(options), manifest)
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
        "manifests": [str(dataset_manifest_path(options)) for options, _ in datasets],
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


def run(args, stop_event, *, backend=batch):
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.startup_timeout <= 0 or args.tp_size <= 0 or args.ulysses_degree <= 0:
        raise batch.BatchError("startup-timeout、tp-size 和 ulysses-degree 必须大于 0")
    configured_commands, _ = ([], []) if args.server_urls else server_commands(args)
    http_ports = [replica_ports(args, i)["HTTP"] for i in range(len(configured_commands))]
    commands = configured_commands if args.serve_only or args.start_servers else []
    urls = args.server_urls or [f"http://127.0.0.1:{port}" for port in http_ports]
    clients = [make_client(url, args, stop_event) for url in urls]
    urls = [client.server_url for client in clients]
    if len(set(urls)) != len(urls):
        raise batch.BatchError("--server-urls 不能重复")
    if args.serve_only:
        batch.log(f"服务模式：启动 {len(clients)} 个副本，不读取 CSV、不提交推理任务")
    else:
        datasets, shared, pinned = prepare(args, urls, backend=backend)
        batch.log(f"{sum(len(cases) for _, cases in datasets)} 个 case，{len(clients)} 个副本，"
                  f"每副本 {args.max_concurrency} 个在途请求；输出: {args.output_dir}")
        if not commands:
            batch.log(f"复用已有服务: {urls}；本次推理不会启动或停止 SGLang")
    for index, (group, command) in enumerate(commands):
        batch.log(f"副本 {index + 1} 端口: {replica_ports(args, index)}")
        batch.log(f"CUDA_VISIBLE_DEVICES={group} {shlex.join(command)}")
    if args.dry_run:
        if not args.serve_only:
            for options, _ in datasets:
                backend.run_batch(options)
        return 0

    processes = []
    log_paths = []
    try:
        if commands:
            if not shutil.which("sglang"):
                raise batch.BatchError("找不到 sglang；请在已跑通推理的 Python 环境执行")
            check_ports_available(args, commands)
            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}"
            log_root = args.service_dir.expanduser().resolve() if args.serve_only else args.output_dir
            log_dir = log_root / "logs" / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            for index, (group, command) in enumerate(commands):
                log_path = log_dir / f"server_{http_ports[index]}.log"
                log_paths.append(log_path)
                with log_path.open("w") as handle:
                    handle.write(f"CUDA_VISIBLE_DEVICES={group} {shlex.join(command)}\n")
                    handle.flush()
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
                        raise batch.BatchError(f"服务 {urls[index]} 已退出 ({process.returncode})；"
                                               f"日志: {log_paths[index]}")
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
                try:
                    make_client(client.server_url, args, stop_event, probe=True).check_server()
                except batch.ApiError as exc:
                    raise batch.BatchError(
                        f"服务不可用: {client.server_url}；请先运行 bash "
                        f"{getattr(args, 'serve_entrypoint', 'serve_smoke_8gpu.sh')} "
                        "并等待服务就绪，或用 --server-urls 指定已有服务。"
                        f"本次未启动模型。详情: {exc}") from exc
        if args.serve_only:
            batch.log(f"全部服务就绪: {urls}；可在另一终端运行 bash "
                      f"{getattr(args, 'infer_entrypoint', 'infer_smoke_8gpu.sh')}。"
                      "服务持续运行；在本服务终端按 Ctrl-C 停止。")
            while not stop_event.wait(.5):
                for index, process in enumerate(processes):
                    if process.poll() is not None:
                        raise batch.BatchError(f"服务 {urls[index]} 已退出 ({process.returncode})；"
                                               f"日志: {log_paths[index]}")
            return 0
        return run_work(args, clients, datasets, shared, pinned, stop_event, backend=backend)
    except (batch.BatchError, OSError):
        print_server_log_tails(log_paths)
        raise
    finally:
        stop_servers(processes)


def main(argv=None, *, parser=None, backend=batch):
    args = (parser or build_parser()).parse_args(argv)
    stop_event = threading.Event()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop_event.set())
    try:
        # An OS lock releases automatically on crash; stale lock files are safe.
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.dry_run and not getattr(args, "write_requests", True):
            return run(args, stop_event, backend=backend)
        lock_dir = args.service_dir.expanduser().resolve() if args.serve_only else args.output_dir
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_name = ".service.lock" if args.serve_only else ".inference.lock"
        with (lock_dir / lock_name).open("a") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                message = ("已有服务管理进程使用该 service-dir" if args.serve_only
                           else "已有推理进程正在写入该输出目录")
                raise batch.BatchError(message) from exc
            return run(args, stop_event, backend=backend)
    except (batch.BatchError, OSError) as exc:
        batch.log(f"错误: {exc}")
        return 130 if stop_event.is_set() else 2
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
