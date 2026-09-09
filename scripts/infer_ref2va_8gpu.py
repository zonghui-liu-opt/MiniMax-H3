#!/usr/bin/env python3
"""Batch cat identity + reference motion inference, sharing the 8-GPU runner."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import batch_ref2va_sglang as ref
import infer_smoke_8gpu as smoke

ROOT = Path(__file__).resolve().parents[1]


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
    return smoke.main(argv, parser=parser, backend=ref)


if __name__ == "__main__":
    sys.exit(main())
