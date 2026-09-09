#!/usr/bin/env python3
"""Offline, CPU-only checks for a first-frame + video-reference H3 deployment.

No model imports, downloads, installs, or GPU allocation happen here.  Source
inspection deliberately checks capabilities, because a version number alone
does not identify early Ref2VA builds that reject hybrid conditioning.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
STAGES = Path("multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3")
COMMON_COMPONENTS = ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae")


class PreflightError(RuntimeError):
    pass


def _record(report, category, ok, detail, action=None):
    report["checks"].append({"category": category, "ok": bool(ok), "detail": detail})
    if not ok:
        report["errors"].append(detail)
    if action and action not in report["actions"]:
        report["actions"].append(action)


def _check_file(path):
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256)
        if not prefix:
            return "文件为空"
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return "只有 Git LFS 指针，未下载实际文件"
    except OSError as exc:
        return str(exc)
    return None


def _json_file(path, report, category="model"):
    problem = _check_file(path)
    if problem:
        _record(report, category, False, f"{path}: {problem}")
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("JSON 顶层必须是对象")
        return result
    except (OSError, ValueError) as exc:
        _record(report, category, False, f"{path}: 无法解析 JSON: {exc}")
        return None


def check_safetensors(path):
    """Read only the tensor header; detect pointers and truncated transfers."""
    problem = _check_file(path)
    if problem:
        return problem
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return "safetensors 文件头不完整"
            header_size = struct.unpack("<Q", raw_length)[0]
            if not 2 <= header_size <= min(128 * 1024 * 1024, size - 8):
                return "不是有效 safetensors 文件，或文件传输被截断"
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            return "safetensors header 不是对象"
        ends = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or not all(isinstance(v, int) for v in offsets)
                    or not 0 <= offsets[0] <= offsets[1]):
                return f"tensor {name} 的 data_offsets 无效"
            ends.append(offsets[1])
        if not ends or max(ends) != size - 8 - header_size:
            return "safetensors 数据长度与文件头不一致（可能未完整传输）"
    except (OSError, ValueError, struct.error) as exc:
        return f"safetensors 检查失败: {exc}"
    return None


def resolve_model_partition(model_path):
    root = Path(model_path).expanduser().resolve()
    return root / "Ref2VA" if (root / "Ref2VA").is_dir() else root


def check_model(model_path, report):
    root = Path(model_path).expanduser().resolve()
    if not (root / "Ref2VA").is_dir():
        _record(report, "model", False,
                f"--model-path 必须是含 Ref2VA/ 子目录的模型根目录：{root}；"
                "入口固定传 --model-variant ref2va，不能直接传 Ref2VA 子目录或根 transformer_ref。")
    model = resolve_model_partition(model_path)
    report["model_partition"] = str(model)
    index = _json_file(model / "model_index.json", report)
    if index is not None:
        partition = index.get("_minimax_h3", {}).get("partition")
        _record(report, "model", partition == "ref2va",
                f"model_index.json partition={partition!r}；要求 'ref2va'（不能使用 FL2VA transformer）")
    weights = set()
    for component in ("transformer", *COMMON_COMPONENTS):
        directory = model / component
        if component in ("tokenizer", "processor"):
            filenames = ("tokenizer_config.json", "tokenizer.json")
            if component == "processor":
                filenames += ("preprocessor_config.json", "video_preprocessor_config.json")
            for filename in filenames:
                _json_file(directory / filename, report)
            continue
        config = _json_file(directory / "config.json", report)
        if config is None:
            continue
        shard_indexes = list(directory.glob("*.safetensors.index.json"))
        if shard_indexes:
            for shard_index in shard_indexes:
                shard_data = _json_file(shard_index, report)
                mapping = shard_data.get("weight_map") if shard_data else None
                if not isinstance(mapping, dict) or not mapping:
                    _record(report, "model", False, f"{shard_index}: 缺少有效 weight_map")
                    continue
                for name in set(mapping.values()):
                    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                        _record(report, "model", False, f"{shard_index}: 非法分片路径 {name!r}")
                    else:
                        weights.add(directory / name)
        elif config.get("source_safetensors_path"):
            source = directory / config.get("source_path", "")
            weights.add(source / config["source_safetensors_path"])
            if config.get("source_path"):
                _json_file(source / "config.json", report)
        else:
            single_weights = list(directory.glob("*.safetensors"))
            if single_weights:
                weights.update(single_weights)
            else:
                _record(report, "model", False, f"{directory}: 缺少 safetensors 权重和分片索引")
        for field in ("source_config_path", "source_metadata_path"):
            if config.get(field):
                dependency = directory / config[field]
                problem = _check_file(dependency)
                if problem:
                    report["warnings"].append(
                        f"{dependency}: {problem}；模型辅助文件缺失，SGLang native VAE 不依赖此文件。")
        for target in config.get("auto_map", {}).values():
            if isinstance(target, str):
                dependency = directory / (target.rsplit(".", 1)[0].replace(".", "/") + ".py")
                problem = _check_file(dependency)
                if problem:
                    report["warnings"].append(
                        f"{dependency}: {problem}；auto_map wrapper 缺失，SGLang native loader 不依赖此 wrapper。")
    for weight in sorted(weights):
        problem = check_safetensors(weight)
        _record(report, "weights", problem is None,
                f"{weight}: {problem or '文件头和数据长度完整'}")
    report["weight_file_count"] = len(weights)
    if any(c["category"] in ("model", "weights") and not c["ok"] for c in report["checks"]):
        destination = model.parent if model.name == "Ref2VA" else root
        report["actions"].append(
            "模型根目录需要完整的官方 Ref2VA 分区，可在联网机器补齐后同步：\n"
            f"  hf download MiniMaxAI/MiniMax-H3 --revision {MODEL_REVISION} --include 'Ref2VA/**' --local-dir {shlex.quote(str(destination))}\n"
            "已有 FL2VA 的公共 text_encoder/tokenizer/processor/video_vae/audio_vae 仅在同一模型 revision、"
            "文件哈希一致时可复用；Ref2VA/model_index.json 和 Ref2VA/transformer 必须来自 Ref2VA。"
            "脚本不会自动复制、下载或用 FL2VA 权重替代。")


def locate_sglang_source():
    """Find the installed package used by this Python, not a vendored snapshot."""
    try:
        spec = importlib.util.find_spec("sglang")
        if spec and spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations)))
    except (ImportError, ValueError):
        pass
    return None


def _literal(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    return None


def inspect_hybrid_source(source):
    """Inspect the active Ref2VA profile using AST, without executing source."""
    profile_path = source / STAGES / "task_profiles.py"
    validation_path = source / STAGES / "request_validation.py"
    try:
        tree = ast.parse(profile_path.read_text(encoding="utf-8"))
        validation = ast.parse(validation_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return False, str(exc)
    names = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            for name in statement.targets:
                if isinstance(name, ast.Name):
                    names[name.id] = statement.value.value
    profile = None
    for statement in tree.body:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "MINIMAX_H3_TASK_PROFILES" for t in targets):
            value = statement.value
            if isinstance(value, ast.Dict):
                profile = next((v for k, v in zip(value.keys, value.values)
                                if _literal(k, names) == "ref2va"), None)
    if not isinstance(profile, ast.Call):
        return False, "未找到可静态验证的 Ref2VA task profile"
    kwargs = {item.arg: item.value for item in profile.keywords}
    rules = kwargs.get("condition_rules")
    admitted = set()
    for rule in rules.elts if isinstance(rules, (ast.Tuple, ast.List)) else []:
        if isinstance(rule, ast.Call):
            options = {kw.arg: _literal(kw.value, names) for kw in rule.keywords}
            admitted.add((options.get("role"), options.get("condition_type")))
    if ("keyframe", "image") not in admitted or ("reference", "video") not in admitted:
        return False, "Ref2VA profile 未同时允许 keyframe/image 和 reference/video"
    if _literal(kwargs.get("video_reference_supported"), names) is not True:
        return False, "Ref2VA profile 未显式启用 video_reference_supported=True"
    for node in ast.walk(validation):
        if not isinstance(node, ast.If):
            continue
        tests_ref = any(isinstance(child, ast.Name) and child.id == "MINIMAX_H3_TASK_REF2VA"
                        for child in ast.walk(node.test))
        calls_keyframe = any(isinstance(child, ast.Call)
                             and isinstance(child.func, ast.Name)
                             and child.func.id == "_validate_keyframe_conditions"
                             for statement in node.body for child in ast.walk(statement))
        if tests_ref and calls_keyframe:
            return True, f"{source}: Ref2VA profile 和请求校验均支持首帧 + 参考视频"
    return False, "request_validation.py 未找到 Ref2VA hybrid keyframe 验证分支"


def check_server_encoders(report):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return  # The common client checks already report the missing binary.
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], text=True,
                                capture_output=True, timeout=10, check=False)
        if result.returncode:
            _record(report, "encoder", False, "ffmpeg -encoders 失败: " + result.stderr.strip())
            return
        names = {parts[1] for line in result.stdout.splitlines()
                 if len(parts := line.split()) >= 2}
        for encoder in ("libx264", "aac"):
            _record(report, "encoder", encoder in names,
                    f"服务端 ffmpeg 编码器 {encoder}: {'可用' if encoder in names else '缺失'}",
                    None if encoder in names else "准备同时启用 libx264 和 AAC 编码器的 Linux ffmpeg；SGLang 需要将生成视频和音频封装为 MP4。")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _record(report, "encoder", False, f"ffmpeg 编码器检查失败: {exc}")


def check_runtime(args, report):
    check_server_encoders(report)
    source = locate_sglang_source()
    report["sglang_source"] = str(source) if source else None
    ok, detail = inspect_hybrid_source(source) if source else (False, "未找到 SGLang MiniMax-H3 源码")
    _record(report, "sglang", ok, detail)
    for distribution in ("sglang", "torch", "transformers", "diffusers", "safetensors", "numpy", "av", "soundfile"):
        try:
            version = importlib.metadata.version(distribution)
            _record(report, "dependency", True, f"{distribution}=={version}")
        except importlib.metadata.PackageNotFoundError:
            _record(report, "dependency", False, f"当前 Python 环境缺少 {distribution}")
    _record(report, "dependency", shutil.which("sglang") is not None, "PATH 中的 sglang 命令")
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        _record(report, "gpu", False, "PATH 中缺少 nvidia-smi；完整服务预检需要在 Linux GPU 主机运行")
    else:
        try:
            result = subprocess.run([nvidia, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                                    text=True, capture_output=True, timeout=10, check=False)
            _record(report, "gpu", result.returncode == 0 and bool(result.stdout.strip()),
                    "nvidia-smi: " + (result.stdout.strip() or result.stderr.strip()))
        except (OSError, subprocess.TimeoutExpired) as exc:
            _record(report, "gpu", False, f"nvidia-smi 检查失败: {exc}")
    if not ok or any(c["category"] == "dependency" and not c["ok"] for c in report["checks"]):
        report["actions"].append(
            "当前 Python 环境需要支持首帧 + 参考视频的 SGLang；本工程基于 0.5.19 接口。"
            "联网时在与推理机相同的 Linux/Python/CUDA/架构环境准备依赖：\n"
            "  python -m pip download 'sglang[diffusion]==0.5.19' --dest ref2va-wheelhouse\n"
            "推理机安装：\n"
            "  python -m pip install --no-index --find-links=ref2va-wheelhouse 'sglang[diffusion]==0.5.19'\n"
            "使用独立环境以保留现有 FL2VA 环境，并用启动服务的同一 Python 重新执行预检。"
            "依赖文件无需提交到 Git；macOS wheels 不能用于 Linux。")


def preflight(args):
    """Return a serializable report; never print, install, or raise on gaps.

    Runner namespaces check server assets only for serve_only/start_servers.
    Set check_server=True to audit a deployment without starting it.
    """
    report = {"ok": False, "errors": [], "warnings": [], "checks": [], "actions": []}
    _record(report, "python", sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}（要求 >=3.10）")
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        _record(report, "client", bool(path), f"{executable}: {path or 'PATH 中缺失'}",
                None if path else "在推理机 PATH 中安装 ffmpeg + ffprobe，确保动态库齐全。")
    server = not getattr(args, "client_only", False) and (
        getattr(args, "check_server", False) or getattr(args, "serve_only", False) or getattr(args, "start_servers", False))
    report["mode"] = "server" if server else "client"
    if server:
        check_model(args.model_path, report)
        check_runtime(args, report)
    else:
        report["warnings"].append("复用服务时仅核查客户端；远端权重和 hybrid 支持请在服务主机运行 ref2va_preflight.py。")
    report["warnings"].append("权重检查验证文件头/长度，不能替代发布方 SHA256；本预检不保证 CUDA 扩展 ABI 或实际推理效果。")
    report["ok"] = not report["errors"]
    return report


def print_report(report):
    print(f"Ref2VA {report['mode']} 预检：{'通过' if report['ok'] else '未通过'}")
    for label, key in (("缺项", "errors"), ("提示", "warnings"), ("操作", "actions")):
        for item in report[key]:
            print(f"[{label}] {item}")
    if report["ok"]:
        print(f"已完成 {len(report['checks'])} 项静态检查。")


def assert_preflight(args):
    report = preflight(args)
    if not report["ok"]:
        raise PreflightError("Ref2VA 环境预检未通过，尚未启动模型：\n" +
                             "\n".join(report["errors"] + report["actions"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3"))
    parser.add_argument("--client-only", action="store_true", help="只核查客户端，连接已有服务时使用")
    parser.add_argument("--json", type=Path, help="保存完整可审计报告")
    args = parser.parse_args(argv)
    args.check_server = True
    report = preflight(args)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
