#!/usr/bin/env python3
"""Ref2VA backend for the shared, resumable two-replica H3 runner.

Only the Python standard library and ffprobe are needed by this client.
Keyframes do not receive Picture labels in H3's reference text encoder, so the
first image is also supplied as an identity reference, before the motion video.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import batch_fl2va_sglang as base
import prepare_ref2va_metadata as metadata_tools


RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
SECTIONS = ("subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music")


@dataclass(frozen=True)
class Case(base.Case):
    reference_video: Path
    cat_id: str
    motion_id: str
    motion_slug: str
    aspect_ratio: str
    request_fingerprint: str


@lru_cache(maxsize=512)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def media_digest(path):
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=512)
def _probe(path: str, size: int, mtime_ns: int):
    if not shutil.which("ffprobe"):
        raise base.BatchError("缺少 ffprobe；请先按 exp_Ref2VA/README.md 备齐离线环境")
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format",
               "-of", "json", path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise base.BatchError(f"无法解码素材 {path}: {result.stderr[-1500:]}")
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise base.BatchError(f"ffprobe 检查失败 {path}: {exc}") from exc


def probe(path):
    stat = path.stat()
    return _probe(str(path), stat.st_size, stat.st_mtime_ns)


def video_stream(path):
    info = probe(path)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise base.BatchError(f"素材没有可解码的图像/视频轨道: {path}")
    return streams[0]


def validate_args(args):
    base.validate_args(args)
    if not args.single_frame:
        raise base.BatchError("本工程使用首帧 + 动作参考；不支持 --first-last-frame 自动同图闭环")
    if args.metadata_v1 is not None:
        raise base.BatchError("Ref2VA 请在同一 metadata 中配置多个动作，不使用 --metadata-v1")
    for field in ("flow_shift", "audio_flow_shift"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            raise base.BatchError(f"{field} 必须是有限正数")


def validate_prompt(prompt, row_number):
    found = re.findall(r"(?m)^([a-z_]+):[ \t]*", prompt)
    if tuple(found) != SECTIONS:
        raise base.BatchError(f"CSV 第 {row_number} 行必须使用官方 Ref2VA 六段提示词，顺序为 {SECTIONS}")
    for label in ("<Picture 1>", "<Video 1>", "<Subject 1>"):
        if label not in prompt:
            raise base.BatchError(f"CSV 第 {row_number} 行缺少参考标签 {label}")


def resolve_reference(row, metadata_dir, *, dry_run):
    """Keep only source media in Git; derive silent motion locally when running."""
    raw = row["reference_video"].strip()
    ref = Path(raw).expanduser()
    ref = (ref if ref.is_absolute() else metadata_dir / ref).resolve()
    if not raw or ref.suffix.lower() != ".mp4":
        raise base.BatchError(f"reference_video 必须是 MP4 路径: {ref}")
    if row.get("source_reference_video"):
        source = Path(row["source_reference_video"]).expanduser()
        source = (source if source.is_absolute() else metadata_dir / source).resolve()
        if not source.is_file():
            raise base.BatchError(f"原动作视频不存在: {source}")
        source_hash = media_digest(source)
        if row.get("source_reference_sha256") and row["source_reference_sha256"] != source_hash:
            raise base.BatchError(f"原动作视频已变化: {source}；请重新运行 prepare_ref2va_metadata.py")
        if ref == source or ref.name != f"{source.stem}.motion.mp4":
            raise base.BatchError("reference_video 应是 prepared/<原视频名>.motion.mp4 缓存，不能覆盖原视频")
        motion = {"source_reference": source, "fps": int(row.get("fps", "24"))}
        if row.get("reference_duration_seconds"):
            motion["reference_duration_seconds"] = float(row["reference_duration_seconds"])
        try:
            if dry_run:
                metadata_tools.validate_reference(motion)
            else:
                prepared, _ = metadata_tools.prepare_reference(motion, ref.parent)
                if prepared.resolve() != ref:
                    raise base.BatchError("静音参考缓存路径与CSV不符")
        except metadata_tools.MetadataError as exc:
            raise base.BatchError(str(exc)) from exc
        return ref, source_hash
    # Compatibility with existing CSVs that point directly at a silent video.
    if not ref.is_file():
        raise base.BatchError(f"动作参考不存在: {ref}")
    info = probe(ref)
    stream = video_stream(ref)
    duration = float(info.get("format", {}).get("duration", stream.get("duration", 0)))
    if not 2 <= duration <= 15.05:
        raise base.BatchError(f"参考视频须为 2–15 秒: {ref} ({duration:g}s)")
    if any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise base.BatchError("动作参考含音轨；请在CSV中配置source_reference_video以自动去除音轨")
    return ref, media_digest(ref)


def load_cases(args):
    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise base.BatchError(f"metadata 不存在: {metadata}；运行 python3 scripts/prepare_ref2va_metadata.py")
    cases, names, references, prompts = [], set(), {}, {}
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", args.input_column, "reference_video", "cat_id",
                    "motion_id", "motion_slug", args.seed_column, "aspect_ratio",
                    args.duration_column, args.output_name_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise base.BatchError(f"Ref2VA CSV 缺少列: {', '.join(sorted(missing))}")
        if not {"prompt_file", args.prompt_column}.intersection(reader.fieldnames or []):
            raise base.BatchError("Ref2VA CSV 需要 prompt_file（或旧版内联prompt）")
        for index, row in enumerate(reader):
            number = index + 2
            if None in row or any(value is None for value in row.values()):
                raise base.BatchError(f"CSV 第 {number} 行的列数与表头不一致")
            image = base.resolve_media_path(row[args.input_column], metadata.parent,
                                            field=args.input_column, row=number)
            reference_key = tuple(row.get(k, "") for k in (
                "reference_video", "source_reference_video", "source_reference_sha256",
                "reference_duration_seconds", "fps"))
            if reference_key not in references:
                references[reference_key] = resolve_reference(row, metadata.parent, dry_run=args.dry_run)
            ref, reference_hash = references[reference_key]
            image_stream = video_stream(image)
            if not image_stream.get("width") or not image_stream.get("height"):
                raise base.BatchError(f"无法确定首帧尺寸: {image}")
            for field in ("cat_id", "motion_id"):
                if not re.fullmatch(r"[0-9]+-[a-z][a-z0-9-]*", row[field]):
                    raise base.BatchError(f"CSV 第 {number} 行 {field} 必须是编号加英文名")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", row["motion_slug"]):
                raise base.BatchError(f"CSV 第 {number} 行 motion_slug 不合法")
            if not re.fullmatch(r"[0-9]+", row["id"]):
                raise base.BatchError(f"CSV 第 {number} 行 id 必须是固定数字编号")
            seed = base.validate_seed(base.parse_int(row[args.seed_column], field="seed", row=number), context="seed")
            name = row[args.output_name_column]
            expected = f"{row['id']}_{row['cat_id']}_{row['motion_id']}_seed-{seed}"
            if name != expected or len(name) > 240:
                raise base.BatchError(f"CSV 第 {number} 行 output_name 应为 {expected}（不带.mp4）")
            if name in names:
                raise base.BatchError(f"CSV 输出名重复: {name}")
            names.add(name)
            ratio = row["aspect_ratio"]
            if ratio not in RATIOS:
                raise base.BatchError(f"Ref2VA aspect_ratio 必须是 {RATIOS} 之一，实际 {ratio!r}")
            prompt = row.get(args.prompt_column, "")
            if row.get("prompt_file"):
                path = Path(row["prompt_file"]).expanduser()
                path = (path if path.is_absolute() else metadata.parent / path).resolve()
                if path not in prompts:
                    prompts[path] = path.read_text(encoding="utf-8").strip()
                prompt = prompts[path]
                if row.get("prompt_sha256") and hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
                    raise base.BatchError(f"提示词文件已更新: {path}；请重新生成metadata")
            validate_prompt(prompt, number)
            seconds = base.validate_duration(base.parse_float(row[args.duration_column], field="duration_seconds", row=number), context="duration_seconds")
            if row.get("fps", "24") != "24":
                raise base.BatchError("H3 当前输出固定 24fps；metadata.fps 应为24")
            case = Case(index, number, name, prompt, image, None, seconds, seed, row,
                        ref, row["cat_id"], row["motion_id"], row["motion_slug"], ratio, "")
            # Hash content and sampling settings, excluding deployment paths and transport.
            identity = build_request(case, args, preview=True)
            for condition in identity["conditions"]:
                condition.pop("uri")
                condition.pop("source_path", None)
            identity["first_image_sha256"] = media_digest(image)
            identity["reference_video_sha256"] = reference_hash
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            case = replace(case, request_fingerprint=fingerprint)
            if args.limit is None or len(cases) < args.limit:
                paths = case_paths(args.output_dir, case)
                old = {} if args.force else (base.read_json(paths.state) or {})
                if not args.force and (old or paths.video.exists()):
                    if old.get("request_fingerprint") != fingerprint:
                        raise base.BatchError(f"{name}: 现有结果/任务与当前素材、提示词或采样参数不同；"
                                              "使用新的 --output-dir，或 --force 显式重跑（旧结果会归档）")
                    if old.get("status") == "submission_unknown":
                        raise base.BatchError(f"{name}: 上次 POST 未获得确认，可能已经占用GPU；"
                                              "先检查服务任务/日志，再决定 --force，避免重复提交")
                cases.append(case)
    if not cases:
        raise base.BatchError("metadata 中没有待处理数据")
    return cases


def build_request(case, args, *, preview=False):
    def uri(path, mime):
        if preview and not args.file_uri:
            return f"data:{mime};base64,<omitted>"
        return base.image_to_uri(path, use_file_uri=args.file_uri)

    first_uri = uri(case.first_image, base.IMAGE_MIME_TYPES[case.first_image.suffix.lower()])
    conditions = [
        {"type": "image", "uri": first_uri, "role": "keyframe", "frame_index": 0},
        {"type": "image", "uri": first_uri, "role": "reference"},
        {"type": "video", "uri": uri(case.reference_video, "video/mp4"), "role": "reference"},
    ]
    if preview:
        for item, source in zip(conditions, (case.first_image, case.first_image, case.reference_video)):
            item["source_path"] = str(source)
    result = {"prompt": case.prompt, "task": "ref2va", "seconds": case.duration_seconds,
              "conditions": conditions,
              "target": {"short_edge": 768, "aspect_ratio": case.aspect_ratio,
                         "duration_seconds": case.duration_seconds},
              "num_outputs_per_prompt": 1, "num_inference_steps": args.num_inference_steps,
              "flow_shift": args.flow_shift, "audio_flow_shift": args.audio_flow_shift,
              "seed": case.seed}
    if args.model:
        result["model"] = args.model
    return result


def case_paths(output_dir, case):
    output_dir = Path(output_dir)
    return base.CasePaths(output_dir / "videos" / case.motion_slug / f"{case.name}.mp4",
                          output_dir / "states" / case.motion_slug / f"{case.name}.json",
                          output_dir / "requests" / case.motion_slug / f"{case.name}.json")


def state_payload(case, paths, **kwargs):
    return {**base.state_payload(case, paths, **kwargs), "task": "ref2va",
            "reference_video": str(case.reference_video), "cat_id": case.cat_id,
            "motion_id": case.motion_id, "motion_slug": case.motion_slug,
            "request_fingerprint": case.request_fingerprint,
            "prompt_version": case.source_row.get("prompt_version"),
            "prompt_status": case.source_row.get("prompt_status")}


def log_case_configuration(args, cases):
    base.log(f"Ref2VA metadata: {args.metadata.resolve()}；{len(cases)} 条，seed逐行读取CSV")
    base.log(f"首帧keyframe + 同图身份reference + 静音video reference；50步默认；无尾帧")
    for case in cases[:3]:
        base.log(f"输出: {case_paths(args.output_dir, case).video}")


def run_batch(args):
    # The shared runner calls this only for the preview pass.
    if not args.dry_run:
        raise base.BatchError("请通过 infer_ref2va_8gpu.sh 运行批量任务")
    cases = load_cases(args)
    log_case_configuration(args, cases)
    if args.write_requests:
        for case in cases:
            base.atomic_write_json(case_paths(args.output_dir, case).request_preview,
                                   build_request(case, args, preview=True))
    base.log(f"dry-run完成：{len(cases)} 条请求已校验；"
             + (f"预览保存在 {args.output_dir / 'requests'}" if args.write_requests else "未写入缓存或请求预览"))
    return 0


def wait_for_result(client, case, paths, args, video_id):
    deadline = time.monotonic() + args.job_timeout
    last_status = None
    while True:
        response = client.retrieve(video_id)
        status = str(response.get("status", "")).lower()
        if not status:
            raise base.ApiError(f"查询响应缺少status: {response}")
        if status != last_status:
            base.atomic_write_json(paths.state, state_payload(case, paths, status=status,
                                   video_id=video_id, server_url=client.server_url, server_response=response))
            base.log(f"[{case.name}] {status}")
            last_status = status
        if status in base.SUCCESS_STATUSES:
            return response
        if status in base.FAILED_STATUSES:
            raise base.BatchError(f"video {video_id} 生成失败: {base.extract_failure(response)}")
        if time.monotonic() >= deadline:
            raise base.BatchError(f"video {video_id} 等待超时；保留ID供恢复")
        client.pause(args.poll_interval)


def run_case(case, args, client):
    paths = case_paths(args.output_dir, case)
    previous = {} if args.force else (base.read_json(paths.state) or {})
    if args.force:
        archive = args.output_dir / "history" / f"{time.time_ns()}_{case.name}"
        for path in (paths.video, paths.state, paths.request_preview):
            if path.exists():
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive / path.name))
        previous = {}
    if str(previous.get("server_response", {}).get("status", "")).lower() in base.FAILED_STATUSES:
        raise base.BatchError("上次服务端任务明确失败；使用 --retry-failed 重试，或检查 states 中的错误")
    if args.write_requests:
        base.atomic_write_json(paths.request_preview, build_request(case, args, preview=True))
    if base.looks_like_mp4(paths.video):
        base.log(f"[{case.name}] 已存在且参数一致，跳过")
        return {**previous, **state_payload(case, paths, status="completed",
                video_id=previous.get("video_id"), server_url=previous.get("server_url")),
                "skipped_existing": True}
    video_id = previous.get("video_id")
    if video_id:
        if previous.get("server_url") != client.server_url:
            raise base.BatchError("未完成任务必须返回原服务恢复")
        try:
            terminal = wait_for_result(client, case, paths, args, video_id)
        except base.ApiError as exc:
            if exc.status_code != 404:
                raise
            video_id = None
            base.log(f"[{case.name}] 旧ID已失效，重新提交")
    if not video_id:
        # A lost POST response must not cause blind automatic resubmission.
        submitter = base.SGLangClient(server_url=client.server_url, api_key=client.api_key,
                    request_timeout=client.request_timeout, retries=0,
                    retry_backoff=client.retry_backoff, stop_event=client.stop_event)
        request = build_request(case, args)
        base.atomic_write_json(paths.state, state_payload(case, paths, status="submission_unknown",
                               server_url=client.server_url))
        try:
            response = submitter.submit(request)
        except base.ApiError as exc:
            if exc.status_code is not None and 400 <= exc.status_code < 500:
                base.atomic_write_json(paths.state, state_payload(case, paths, status="rejected",
                                       server_url=client.server_url, error=str(exc),
                                       server_response={"status": "rejected"}))
            raise
        video_id = str(response["id"])
        base.atomic_write_json(paths.state, state_payload(case, paths, status="submitted",
                               video_id=video_id, server_url=client.server_url, server_response=response))
        terminal = wait_for_result(client, case, paths, args, video_id)
    client.download(video_id, paths.video)
    result = state_payload(case, paths, status="completed", video_id=video_id,
                           server_url=client.server_url, server_response=terminal)
    base.atomic_write_json(paths.state, result)
    base.log(f"[{case.name}] 完成: {paths.video}")
    return result


def validate_runtime(args):
    import ref2va_preflight
    report = ref2va_preflight.preflight(args)
    ref2va_preflight.print_report(report)
    base.atomic_write_json(args.output_dir / "preflight.json", report)
    if not report["ok"]:
        raise base.BatchError("Ref2VA环境预检未通过；全部缺项见上述报告，尚未启动GPU模型")


def validate_servers(args, clients):
    for client in clients:
        info = client.check_server()
        # Server versions expose different model metadata. Reject an explicit
        # FL2VA selection; absence is recorded and the documented request remains
        # the authoritative check, rather than inventing a capabilities endpoint.
        variant = info.get("model_variant") or info.get("variant")
        if variant and str(variant).lower() not in ("ref2va", "hybrid"):
            raise base.BatchError(f"{client.server_url} 运行的是 {variant}，请启动Ref2VA服务")
        try:
            server_info = client.request_json("GET", "/server_info")
        except base.ApiError as exc:
            if exc.status_code != 404:
                raise
            server_info = {}
        name = server_info.get("served_model_name")
        if name != "minimax-h3-ref2va" and not args.allow_unmarked_server:
            raise base.BatchError(f"{client.server_url} 未返回本工程Ref2VA服务标识；"
                                  "请用 serve_ref2va_8gpu.sh 启动，或在确认其Ref2VA分区后"
                                  "显式 --allow-unmarked-server。此检查不提交推理任务")
        base.log(f"Ref2VA服务: {client.server_url}；model_variant={variant or '未由 /models 暴露'}")
