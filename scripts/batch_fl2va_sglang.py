#!/usr/bin/env python3
"""Batch MiniMax-H3 FL2VA inference through SGLang's /v1/videos API.

The input CSV must contain ``input_image`` and ``prompt`` columns.  By default,
the same image is anchored at frame indexes 0 and -1, which matches looping
prompts that return to their initial composition.  Add a ``last_image`` column
to use a different final frame, or pass ``--single-frame`` to condition only on
the first frame.

The script has no third-party Python dependencies.  It stores one state file
per case, so rerunning the same command resumes submitted jobs and skips videos
that have already downloaded successfully.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import mimetypes
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


MIN_DURATION_SECONDS = 4.0
MAX_DURATION_SECONDS = 15.0
MINIMAX_H3_SHORT_EDGE = 768
MAX_SIGNED_SEED = (1 << 63) - 1
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
FAILED_STATUSES = {"failed", "cancelled", "canceled", "expired", "rejected"}
SUCCESS_STATUSES = {"completed", "succeeded", "success"}
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
PRINT_LOCK = threading.Lock()


class BatchError(RuntimeError):
    """Base exception for actionable batch-inference failures."""


class ApiError(BatchError):
    """HTTP API error with the response status and body preserved."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class Case:
    index: int
    row_number: int
    name: str
    prompt: str
    first_image: Path
    last_image: Path | None
    duration_seconds: float
    seed: int
    source_row: dict[str, str]


@dataclass(frozen=True)
class CasePaths:
    video: Path
    state: Path
    request_preview: Path


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchError(f"无法读取状态文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchError(f"状态文件必须是 JSON object: {path}")
    return value


def slugify(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value).strip("._-")
    return slug[:100] or "case"


def resolve_media_path(value: str, metadata_dir: Path, *, field: str, row: int) -> Path:
    raw = value.strip()
    if not raw:
        raise BatchError(f"CSV 第 {row} 行的 {field} 为空")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = metadata_dir / path
    path = path.resolve()
    if not path.is_file():
        raise BatchError(f"CSV 第 {row} 行的 {field} 文件不存在: {path}")
    if path.suffix.lower() not in IMAGE_MIME_TYPES:
        raise BatchError(
            f"CSV 第 {row} 行的 {field} 必须是 PNG/JPEG/WebP，实际为: {path}"
        )
    return path


def parse_float(value: str, *, field: str, row: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise BatchError(f"CSV 第 {row} 行的 {field} 不是数字: {value!r}") from exc
    return parsed


def parse_int(value: str, *, field: str, row: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BatchError(f"CSV 第 {row} 行的 {field} 不是整数: {value!r}") from exc
    return parsed


def validate_duration(value: float, *, context: str) -> float:
    if not MIN_DURATION_SECONDS <= value <= MAX_DURATION_SECONDS:
        raise BatchError(
            f"{context} 必须在 {MIN_DURATION_SECONDS:g}–{MAX_DURATION_SECONDS:g} 秒之间，"
            f"实际为 {value:g}"
        )
    return value


def validate_seed(value: int, *, context: str) -> int:
    if not 0 <= value <= MAX_SIGNED_SEED:
        raise BatchError(f"{context} 必须在 0–{MAX_SIGNED_SEED} 之间，实际为 {value}")
    return value


def load_cases(args: argparse.Namespace) -> list[Case]:
    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise BatchError(f"metadata CSV 不存在: {metadata}")

    cases: list[Case] = []
    names: set[str] = set()
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {args.input_column, args.prompt_column}
        missing = sorted(required - columns)
        if missing:
            raise BatchError(f"metadata CSV 缺少必需列: {', '.join(missing)}")

        for index, row in enumerate(reader):
            row_number = index + 2
            first_image = resolve_media_path(
                row.get(args.input_column, ""),
                metadata.parent,
                field=args.input_column,
                row=row_number,
            )
            prompt = row.get(args.prompt_column, "")
            if not prompt.strip():
                raise BatchError(f"CSV 第 {row_number} 行的 {args.prompt_column} 为空")

            explicit_last = row.get(args.last_image_column, "").strip()
            if args.single_frame:
                last_image = None
            elif explicit_last:
                last_image = resolve_media_path(
                    explicit_last,
                    metadata.parent,
                    field=args.last_image_column,
                    row=row_number,
                )
            else:
                last_image = first_image

            duration_raw = row.get(args.duration_column, "").strip()
            duration = (
                parse_float(
                    duration_raw,
                    field=args.duration_column,
                    row=row_number,
                )
                if duration_raw
                else args.duration_seconds
            )
            validate_duration(duration, context=f"CSV 第 {row_number} 行的 duration")

            seed_raw = row.get(args.seed_column, "").strip()
            seed = (
                parse_int(seed_raw, field=args.seed_column, row=row_number)
                if seed_raw
                else args.seed + index * args.seed_stride
            )
            validate_seed(seed, context=f"CSV 第 {row_number} 行的 seed")

            requested_name = row.get(args.output_name_column, "").strip()
            base_name = requested_name or first_image.stem
            name = f"{index:03d}_{slugify(base_name)}"
            if name in names:
                raise BatchError(f"CSV 第 {row_number} 行产生了重复输出名: {name}")
            names.add(name)

            cases.append(
                Case(
                    index=index,
                    row_number=row_number,
                    name=name,
                    prompt=prompt,
                    first_image=first_image,
                    last_image=last_image,
                    duration_seconds=duration,
                    seed=seed,
                    source_row={key: value or "" for key, value in row.items()},
                )
            )

    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise BatchError("没有可执行的 case；请检查 CSV 内容或 --limit")
    return cases


def image_to_uri(path: Path, *, use_file_uri: bool) -> str:
    if use_file_uri:
        return path.as_uri()
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        guessed, _ = mimetypes.guess_type(path.name)
        mime_type = guessed or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_request(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    first_uri = image_to_uri(case.first_image, use_file_uri=args.file_uri)
    conditions: list[dict[str, Any]] = [
        {
            "type": "image",
            "uri": first_uri,
            "role": "keyframe",
            "frame_index": 0,
        }
    ]
    if case.last_image is not None:
        last_uri = (
            first_uri
            if case.last_image == case.first_image
            else image_to_uri(case.last_image, use_file_uri=args.file_uri)
        )
        conditions.append(
            {
                "type": "image",
                "uri": last_uri,
                "role": "keyframe",
                "frame_index": -1,
            }
        )

    request: dict[str, Any] = {
        "prompt": case.prompt,
        "seconds": case.duration_seconds,
        "task": "fl2va",
        "conditions": conditions,
        "target": {
            "short_edge": MINIMAX_H3_SHORT_EDGE,
            "aspect_ratio": "auto",
            "duration_seconds": case.duration_seconds,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "seed": case.seed,
    }
    if args.model:
        request["model"] = args.model
    return request


def redacted_request(request: dict[str, Any], case: Case) -> dict[str, Any]:
    preview = json.loads(json.dumps(request, ensure_ascii=False))
    source_paths = [case.first_image]
    if case.last_image is not None:
        source_paths.append(case.last_image)
    for condition, source_path in zip(preview["conditions"], source_paths):
        uri = condition["uri"]
        if uri.startswith("data:"):
            condition["uri"] = f"data:{source_path.suffix.lower()};base64,<omitted>"
        condition["source_path"] = str(source_path)
    return preview


def case_paths(output_dir: Path, case: Case) -> CasePaths:
    return CasePaths(
        video=output_dir / "videos" / f"{case.name}.mp4",
        state=output_dir / "states" / f"{case.name}.json",
        request_preview=output_dir / "requests" / f"{case.name}.json",
    )


def looks_like_mp4(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with path.open("rb") as handle:
        header = handle.read(12)
    return header[4:8] == b"ftyp"


class SGLangClient:
    def __init__(
        self,
        *,
        server_url: str,
        api_key: str | None,
        request_timeout: float,
        retries: int,
        retry_backoff: float,
    ) -> None:
        normalized = server_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        self.server_url = normalized
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.retries = retries
        self.retry_backoff = retry_backoff

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _open(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ):
        url = f"{self.server_url}{path}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        attempts = self.retries + 1
        for attempt in range(attempts):
            request = urllib.request.Request(
                url,
                data=data,
                headers=self._headers(json_body=json_body is not None),
                method=method,
            )
            try:
                return urllib.request.urlopen(request, timeout=self.request_timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code in RETRYABLE_HTTP_CODES and attempt + 1 < attempts:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue
                raise ApiError(
                    f"{method} {url} 返回 HTTP {exc.code}: {body[:1000]}",
                    status_code=exc.code,
                    body=body,
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue
                raise ApiError(f"{method} {url} 请求失败: {exc}") from exc
        raise AssertionError("unreachable")

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._open(method, path, json_body=json_body) as response:
            raw = response.read()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            text = raw.decode("utf-8", "replace")
            raise ApiError(f"{method} {path} 返回的不是 JSON: {text[:1000]}") from exc
        if not isinstance(value, dict):
            raise ApiError(f"{method} {path} 返回的 JSON 不是 object")
        return value

    def check_server(self) -> dict[str, Any]:
        try:
            return self.request_json("GET", "/models")
        except ApiError as exc:
            if exc.status_code != 404:
                raise
        # Older SGLang diffusion builds may expose only the generic health
        # endpoint.  A successful response is enough for the startup check;
        # the first video request will still surface any model mismatch.
        with self._open("GET", "/health") as response:
            status = response.read().decode("utf-8", "replace").strip()
        return {"status": status or "ok"}

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.request_json("POST", "/v1/videos", json_body=request)
        if not response.get("id"):
            raise ApiError(f"提交响应缺少 video id: {response}")
        return response

    def retrieve(self, video_id: str) -> dict[str, Any]:
        return self.request_json("GET", f"/v1/videos/{video_id}")

    def download(self, video_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with self._open("GET", f"/v1/videos/{video_id}/content") as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if not looks_like_mp4(temporary):
                raise ApiError(
                    f"下载内容不是有效的 MP4（文件过小或缺少 ftyp header）: {temporary}"
                )
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def extract_failure(response: dict[str, Any]) -> str:
    for key in ("error", "last_error", "failure_reason", "detail", "message"):
        value = response.get(key)
        if value:
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)
    return "服务未返回失败详情"


def state_payload(
    case: Case,
    paths: CasePaths,
    *,
    status: str,
    video_id: str | None = None,
    server_response: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "case_index": case.index,
        "csv_row": case.row_number,
        "name": case.name,
        "status": status,
        "video_id": video_id,
        "seed": case.seed,
        "duration_seconds": case.duration_seconds,
        "first_image": str(case.first_image),
        "last_image": str(case.last_image) if case.last_image else None,
        "output_video": str(paths.video),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if server_response is not None:
        value["server_response"] = server_response
    if error is not None:
        value["error"] = error
    return value


def wait_for_terminal_status(
    client: SGLangClient,
    *,
    video_id: str,
    case: Case,
    paths: CasePaths,
    poll_interval: float,
    job_timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + job_timeout
    previous_status = ""
    while True:
        response = client.retrieve(video_id)
        status = str(response.get("status", "")).strip().lower()
        if not status:
            raise ApiError(f"查询响应缺少 status: {response}")
        if status != previous_status:
            log(f"[{case.name}] server status: {status}")
            atomic_write_json(
                paths.state,
                state_payload(
                    case,
                    paths,
                    status=status,
                    video_id=video_id,
                    server_response=response,
                ),
            )
            previous_status = status
        if status in SUCCESS_STATUSES:
            return response
        if status in FAILED_STATUSES:
            raise BatchError(f"video {video_id} 生成失败: {extract_failure(response)}")
        if time.monotonic() >= deadline:
            raise BatchError(f"video {video_id} 等待超过 {job_timeout:g} 秒")
        time.sleep(poll_interval)


def run_case(
    case: Case, args: argparse.Namespace, client: SGLangClient
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    paths = case_paths(output_dir, case)
    request = build_request(case, args)
    atomic_write_json(paths.request_preview, redacted_request(request, case))

    if not args.force and looks_like_mp4(paths.video):
        log(f"[{case.name}] 已存在，跳过: {paths.video}")
        previous = read_json(paths.state) or {}
        return {
            **previous,
            **state_payload(
                case,
                paths,
                status="completed",
                video_id=previous.get("video_id"),
            ),
            "skipped_existing": True,
        }

    previous = None if args.force else read_json(paths.state)
    video_id = (
        str(previous.get("video_id")) if previous and previous.get("video_id") else None
    )

    if video_id:
        log(f"[{case.name}] 恢复 video id: {video_id}")
        try:
            terminal = wait_for_terminal_status(
                client,
                video_id=video_id,
                case=case,
                paths=paths,
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
            )
        except ApiError as exc:
            if exc.status_code != 404:
                raise
            log(f"[{case.name}] 服务端已找不到旧任务，重新提交")
            video_id = None
        else:
            client.download(video_id, paths.video)
            result = state_payload(
                case,
                paths,
                status="completed",
                video_id=video_id,
                server_response=terminal,
            )
            atomic_write_json(paths.state, result)
            log(f"[{case.name}] 完成: {paths.video}")
            return result

    log(f"[{case.name}] 提交（seed={case.seed}, duration={case.duration_seconds:g}s）")
    submitted = client.submit(request)
    video_id = str(submitted["id"])
    atomic_write_json(
        paths.state,
        state_payload(
            case,
            paths,
            status=str(submitted.get("status") or "submitted").lower(),
            video_id=video_id,
            server_response=submitted,
        ),
    )
    terminal = wait_for_terminal_status(
        client,
        video_id=video_id,
        case=case,
        paths=paths,
        poll_interval=args.poll_interval,
        job_timeout=args.job_timeout,
    )
    client.download(video_id, paths.video)
    result = state_payload(
        case,
        paths,
        status="completed",
        video_id=video_id,
        server_response=terminal,
    )
    atomic_write_json(paths.state, result)
    log(f"[{case.name}] 完成: {paths.video}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通过已启动的 SGLang 服务批量运行 MiniMax-H3 FL2VA 推理。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("testsets/metadata_6cases_480x832.csv"),
        help="输入 metadata CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/fl2va"), help="输出目录"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("SGLANG_DEPLOYMENT_URL", "http://127.0.0.1:30010"),
        help="SGLang 服务根 URL（不要重复添加 /v1）",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SGLANG_API_KEY"),
        help="可选 Bearer token，也可通过 SGLANG_API_KEY 提供",
    )
    parser.add_argument("--model", help="可选 model 字段；本地 H3 服务通常无需设置")
    parser.add_argument(
        "--duration-seconds", type=float, default=4.0, help="CSV 未提供时的生成时长"
    )
    parser.add_argument("--seed", type=int, default=0, help="第一个 case 的基础 seed")
    parser.add_argument(
        "--seed-stride", type=int, default=1, help="相邻 CSV case 的 seed 增量"
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument(
        "--max-concurrency", type=int, default=1, help="同时提交并等待的任务数"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=3.0, help="状态轮询间隔（秒）"
    )
    parser.add_argument(
        "--job-timeout", type=float, default=7200.0, help="单个任务最长等待时间（秒）"
    )
    parser.add_argument(
        "--request-timeout", type=float, default=120.0, help="单次 HTTP 请求超时（秒）"
    )
    parser.add_argument("--retries", type=int, default=3, help="HTTP 请求重试次数")
    parser.add_argument(
        "--retry-backoff", type=float, default=1.0, help="HTTP 指数退避基数（秒）"
    )
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="只将 input_image 作为首帧；默认缺少 last_image 时同图首尾锚定",
    )
    parser.add_argument(
        "--file-uri",
        action="store_true",
        help="使用 file:// URI，要求服务端能访问相同绝对路径；默认使用 data URI",
    )
    parser.add_argument(
        "--force", action="store_true", help="忽略已有状态和 MP4，重新生成"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只校验 CSV 并写请求预览，不访问服务"
    )
    parser.add_argument(
        "--skip-server-check", action="store_true", help="跳过启动时的 GET /models 检查"
    )
    parser.add_argument("--limit", type=int, help="只处理 CSV 开头的 N 条数据")
    parser.add_argument("--input-column", default="input_image")
    parser.add_argument("--last-image-column", default="last_image")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--duration-column", default="duration_seconds")
    parser.add_argument("--seed-column", default="seed")
    parser.add_argument("--output-name-column", default="output_name")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_duration(args.duration_seconds, context="--duration-seconds")
    validate_seed(args.seed, context="--seed")
    if args.seed_stride < 0:
        raise BatchError("--seed-stride 不能小于 0")
    for field in ("num_inference_steps", "max_concurrency"):
        if getattr(args, field) <= 0:
            raise BatchError(f"--{field.replace('_', '-')} 必须大于 0")
    for field in (
        "poll_interval",
        "job_timeout",
        "request_timeout",
        "retry_backoff",
    ):
        if getattr(args, field) <= 0:
            raise BatchError(f"--{field.replace('_', '-')} 必须大于 0")
    if args.retries < 0:
        raise BatchError("--retries 不能小于 0")
    if args.limit is not None and args.limit <= 0:
        raise BatchError("--limit 必须大于 0")


def run_batch(args: argparse.Namespace) -> int:
    validate_args(args)
    cases = load_cases(args)
    output_dir = args.output_dir.expanduser().resolve()

    if args.dry_run:
        for case in cases:
            paths = case_paths(output_dir, case)
            request = build_request(case, args)
            atomic_write_json(paths.request_preview, redacted_request(request, case))
        log(
            f"dry-run 完成：已校验 {len(cases)} 个 case，请求预览位于 {output_dir / 'requests'}"
        )
        return 0

    client = SGLangClient(
        server_url=args.server_url,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )
    if not args.skip_server_check:
        model_info = client.check_server()
        model_label = (
            model_info.get("model_path") or model_info.get("model") or "unknown"
        )
        log(f"SGLang 服务可用：{client.server_url} (model={model_label})")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def execute(case: Case) -> dict[str, Any]:
        try:
            return run_case(case, args, client)
        except Exception as exc:
            paths = case_paths(output_dir, case)
            failed = state_payload(
                case,
                paths,
                status="failed",
                video_id=(read_json(paths.state) or {}).get("video_id"),
                error=str(exc),
            )
            atomic_write_json(paths.state, failed)
            log(f"[{case.name}] 失败: {exc}")
            raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrency
    ) as pool:
        future_to_case = {pool.submit(execute, case): case for case in cases}
        for future in concurrent.futures.as_completed(future_to_case):
            case = future_to_case[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"name": case.name, "error": str(exc)})

    manifest = {
        "metadata": str(args.metadata.expanduser().resolve()),
        "server_url": client.server_url,
        "total": len(cases),
        "completed": len(results),
        "failed": len(failures),
        "results": sorted(results, key=lambda item: item.get("case_index", -1)),
        "failures": failures,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    log(f"批量任务结束：completed={len(results)}, failed={len(failures)}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except (BatchError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
