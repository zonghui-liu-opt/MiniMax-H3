#!/usr/bin/env python3
"""Batch cat identity + reference motion inference, sharing the 8-GPU runner."""
from __future__ import annotations

import os
import sys
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path

import batch_ref2va_sglang as ref
import infer_smoke_8gpu as smoke

ROOT = Path(__file__).resolve().parents[1]


_H3_STAGE_PREFIX = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3."
)
_RESOLUTION_MODULES = {
    _H3_STAGE_PREFIX + "request_validation",
    _H3_STAGE_PREFIX + "resolved_plan",
}
_RESOLUTION_ENV = "MINIMAX_H3_CUSTOM_RESOLUTION"


def _resolution_source(fullname, source):
    """Relax known legacy guards in memory; retain all other source unchanged."""
    if fullname == _H3_STAGE_PREFIX + "request_validation":
        return source.replace("if short_edge != 768:", "if short_edge <= 0:").replace(
            "short_edge must be 768 for minimax_h3", "short_edge must be positive for minimax_h3")
    if fullname == _H3_STAGE_PREFIX + "resolved_plan":
        return source.replace(
            "short_edge != MINIMAX_H3_BASE_SHORT_EDGE or value != short_edge",
            "short_edge <= 0 or value != short_edge",
        ).replace("target.short_edge must be 768", "target.short_edge must be a positive integer")
    return source


class _ResolutionLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Bypass the bytecode cache: no installation files are written, even
        # when a read-only environment contains an existing .pyc.
        source = _resolution_source(fullname, self.get_source(fullname))
        return compile(source, self.path, "exec", dont_inherit=True)


class _ResolutionFinder:
    _minimax_h3_resolution_override = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _RESOLUTION_MODULES:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _ResolutionLoader(fullname, spec.origin)
            return spec
        return None


def install_resolution_override():
    """Install before SGLang imports, both in the server and spawned workers."""
    if not any(getattr(finder, "_minimax_h3_resolution_override", False)
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _ResolutionFinder())


def run_sglang_server():
    os.environ[_RESOLUTION_ENV] = "1"
    install_resolution_override()
    # Keep this file as __main__, so multiprocessing spawn replays the hook
    # before unpickling worker targets. Calling the CLI directly preserves it.
    del sys.argv[1]
    print("Ref2VA自定义分辨率：仅在本次服务及其工作进程内生效，不修改SGLang安装文件。", flush=True)
    from sglang.cli.main import main as sglang_main
    return sglang_main()


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
    if args.resolution and (args.serve_only or args.start_servers):
        parser.set_defaults(server_command_prefix=[
            sys.executable, "-B", str(Path(__file__).resolve()), "--sglang-server"])
    return smoke.main(argv, parser=parser, backend=ref)


if __name__ == "__mp_main__" and os.environ.get(_RESOLUTION_ENV) == "1":
    install_resolution_override()
elif __name__ == "__main__":
    if sys.argv[1:2] == ["--sglang-server"]:
        sys.exit(run_sglang_server())
    sys.exit(main())
