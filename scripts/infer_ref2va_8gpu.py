#!/usr/bin/env python3
"""Batch cat identity + reference motion inference, sharing the 8-GPU runner."""
from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path

import batch_ref2va_sglang as ref
import infer_smoke_8gpu as smoke

ROOT = Path(__file__).resolve().parents[1]


def enable_custom_short_edge(package_root: Path | None = None) -> list[Path]:
    """Relax only the two known legacy guards; leave unfamiliar installations alone."""
    if package_root is None:
        spec = find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            return []
        package_root = Path(next(iter(spec.submodule_search_locations)))
    directory = package_root / "multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3"
    old_request = '''    if short_edge != 768:
        raise ValueError(
            f"{path}.short_edge must be 768 for minimax_h3, got {short_edge}"
        )'''
    new_request = old_request.replace("short_edge != 768", "short_edge <= 0").replace(
        "must be 768 for minimax_h3", "must be positive for minimax_h3")
    old_shape = '''    except (TypeError, ValueError) as exc:
        raise ValueError("target.short_edge must be 768") from exc
    if short_edge != MINIMAX_H3_BASE_SHORT_EDGE or value != short_edge:
        raise ValueError(
            f"target.short_edge must be 768 for MiniMax H3 shape policy v2, got {value!r}"
        )'''
    new_shape = old_shape.replace(
        "short_edge != MINIMAX_H3_BASE_SHORT_EDGE or value != short_edge",
        "short_edge <= 0 or value != short_edge",
    ).replace("must be 768", "must be a positive integer")
    replacements = (
        (directory / "request_validation.py", old_request, new_request),
        (directory / "resolved_plan.py", old_shape, new_shape),
    )
    prepared = []
    for path, old, new in replacements:
        if not path.is_file():
            return []
        try:
            source = path.read_text(encoding="utf-8")
            mode = path.stat().st_mode & 0o7777
        except (OSError, UnicodeError):
            return []
        if source.count(old) == 1:
            updated = source.replace(old, new, 1)
            prepared.append((path, source, updated, mode))
        elif source.count(new) != 1:
            return []
    # Validate both source files before changing either; no backups or source copies.
    try:
        for path, _, updated, _ in prepared:
            compile(updated, str(path), "exec")
    except SyntaxError:
        return []
    written = []
    try:
        for path, original, updated, mode in prepared:
            ref.metadata_tools.atomic_text(path, updated)
            written.append((path, original, mode))
            path.chmod(mode)
    except OSError as exc:
        rollback_errors = []
        for path, original, mode in reversed(written):
            try:
                ref.metadata_tools.atomic_text(path, original)
                path.chmod(mode)
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            raise OSError(f"{exc}；回滚未完成：{'；'.join(rollback_errors)}") from exc
        raise
    return [path for path, _, _, _ in prepared]


def build_parser():
    parser = smoke.build_parser()
    parser.description = "H3 Ref2VA：80猫 × 动作视频 × 显式seed，两个四卡副本批量推理"
    parser.set_defaults(metadata=ROOT / "exp_Ref2VA/metadata.csv",
                        output_dir=ROOT / "exp_Ref2VA", single_frame=True,
                        model_variant="ref2va", flat_output=True,
                        base_port=30110, base_master_port=31110,
                        base_scheduler_port=32110,
                        service_dir=ROOT / "exp_Ref2VA/services",
                        serve_entrypoint="serve_ref2va_8gpu.sh",
                        infer_entrypoint="infer_ref2va_8gpu.sh")
    parser.add_argument("--lock-first-frame", action="store_true",
                        help="额外提交 frame_index=0 的 keyframe；仅用于支持该条件的Ref2VA服务")
    parser.add_argument("--resolution", type=ref.parse_resolution, metavar="WIDTHxHEIGHT",
                        help="原生生成尺寸，例如480x832；省略时沿用768短边，不做生成后缩放")
    parser.add_argument("--write-requests", action="store_true",
                        help="显式保存调试请求预览；默认不生成这些文件")
    parser.epilog = ("默认读取 exp_Ref2VA/metadata.csv，复用30110/30112的Ref2VA服务；"
                     "--start-servers 一条命令启动、推理并清理服务。"
                     "输出 videos/<motion_slug>/<output_name>.mp4；"
                     "--dry-run 不加载模型、不请求网络。")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for token in args.server_arg:
        if token.split("=", 1)[0] in ("--model-variant", "--model-path"):
            parser.error("Ref2VA入口固定 --model-variant ref2va；模型根目录使用 --model-path")
    # All generation is local; do not try to download missing Hub assets on the
    # isolated inference host.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.resolution and (args.serve_only or args.start_servers) and not args.dry_run:
        try:
            changed = enable_custom_short_edge()
        except (OSError, SyntaxError, ValueError) as exc:
            print(f"错误: 无法启用原生自定义分辨率：{exc}", file=sys.stderr)
            return 2
        if changed:
            print("已放开 SGLang H3 的自定义短边；本次启动的新服务进程生效，已有服务需重启。")
    return smoke.main(argv, parser=parser, backend=ref)


if __name__ == "__main__":
    sys.exit(main())
