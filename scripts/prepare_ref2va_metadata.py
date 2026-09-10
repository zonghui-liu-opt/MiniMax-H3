#!/usr/bin/env python3
"""Build compact, portable Ref2VA metadata; optionally prepare silent video caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [0]
PROMPT_SECTIONS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
)
SUPPORTED_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
FIELDS = [
    "id", "source_id", "input_image", "source_reference_video", "reference_video",
    "prompt_file", "cat_idx", "cat_id", "cat_name_zh", "motion_idx", "motion_id",
    "motion_slug", "motion_name_zh", "seed", "output_name", "duration_seconds", "reference_duration_seconds",
    "fps", "width", "height", "source_aspect_ratio", "aspect_ratio",
    "prompt_version", "prompt_status", "prompt_sha256", "source_reference_sha256",
]
NAMED_ID = re.compile(r"[0-9]+-[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class MetadataError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def resolve_file(value: str, parent: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"{field} 必须是非空本地文件路径")
    if "://" in value:
        raise MetadataError(f"{field} 必须是本地文件，不能依赖网络地址: {value}")
    path = Path(value).expanduser()
    path = (parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise MetadataError(f"{field} 文件缺失或为空: {path}")
    return path


def matches(value: str, selectors: list[str] | None, slug: str = "") -> bool:
    return selectors is None or any(
        selection == "*" or selection in (value, slug)
        or (selection.isascii() and selection.isdigit()
            and int(selection) == int(value.split("-", 1)[0]))
        for selection in selectors
    )


def read_catalog(path: Path, cat_dir: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"cat_idx", "cat_id", "cat_name_zh", "image_filename"}.issubset(
                reader.fieldnames or []):
            raise MetadataError("cat_catalog.csv 缺少 cat_idx/cat_id/cat_name_zh/image_filename")
        rows = list(reader)
    used_ids, used_images, cats = set(), set(), []
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise MetadataError("cat_catalog.csv 中有列数错误")
        cat_idx, cat_id = row["cat_idx"], row["cat_id"]
        if not cat_idx.isascii() or not cat_idx.isdigit() or not NAMED_ID.fullmatch(cat_id):
            raise MetadataError(f"无效的猫编号: {cat_idx}/{cat_id}")
        if int(cat_idx) != int(cat_id.split("-", 1)[0]) or int(cat_idx) in used_ids:
            raise MetadataError(f"猫编号不匹配或重复: {cat_id}")
        filename = row["image_filename"]
        if Path(filename).name != filename or not filename.startswith(f"{cat_idx}_"):
            raise MetadataError(f"image_filename 必须为当前猫编号开头的纯文件名: {filename}")
        image = resolve_file(filename, cat_dir, "input_image")
        if image in used_images:
            raise MetadataError(f"重复的猫咪图: {image}")
        used_ids.add(int(cat_idx))
        used_images.add(image)
        cats.append({**row, "image": image})
    discovered = {path.resolve() for path in cat_dir.iterdir()
                  if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}}
    unknown = discovered - used_images
    if unknown:
        raise MetadataError("首帧尚未登记 cat_catalog.csv: " + ", ".join(sorted(p.name for p in unknown)))
    if not cats:
        raise MetadataError("猫咪目录为空")
    return sorted(cats, key=lambda row: int(row["cat_idx"]))


def validate_prompt(prompt: str, path: Path) -> None:
    sections = re.findall(r"(?m)^([a-z_]+):\s*", prompt)
    if sections != list(PROMPT_SECTIONS):
        raise MetadataError(f"{path} 必须按官方 Ref2VA 格式包含六段: {', '.join(PROMPT_SECTIONS)}")
    if not all(label in prompt for label in ("<Picture 1>", "<Video 1>", "<Subject 1>", "[Shot 1]")):
        raise MetadataError(f"{path} 缺少首帧、动作视频、猫咪身份或镜头标签")
    if re.search(r"[\u3400-\u9fff]", prompt):
        raise MetadataError(f"{path} 的六段提示词必须使用英文")


def read_identity_prompts(path: Path) -> dict:
    """Load one versioned, image-bound identity catalog shared by all motions."""
    config = json.loads(path.read_text(encoding="utf-8"))
    template_keys = {"subject_definition", "retention", "opening", "continuity", "closing"}
    if (not isinstance(config, dict) or config.get("schema_version") != 1
            or not isinstance(config.get("cats"), dict) or not config["cats"]
            or not isinstance(config.get("templates"), dict)
            or set(config["templates"]) != template_keys):
        raise MetadataError(f"身份提示词格式无效: {path}")
    for field in ("prompt_version", "prompt_status"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise MetadataError(f"身份提示词缺少 {field}: {path}")
    for cat_id, identity in config["cats"].items():
        if (not NAMED_ID.fullmatch(cat_id) or not isinstance(identity, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("image_sha256", "")))):
            raise MetadataError(f"身份提示词 cat_id/image_sha256 无效: {cat_id}")
        for field in ("description", "anchor"):
            value = identity.get(field)
            if (not isinstance(value, str) or not value.strip()
                    or "\n" in value or re.search(r"[\u3400-\u9fff]", value)):
                raise MetadataError(f"{cat_id} 的 {field} 必须是单行英文身份描述")
    for name, template in config["templates"].items():
        if not isinstance(template, str) or not template.strip() or "\n" in template:
            raise MetadataError(f"身份提示词模板无效: {name}")
        try:
            template.format(description="description", anchor="anchor")
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            raise MetadataError(f"身份提示词模板只能使用 description/anchor: {name}") from exc
    return config


def compose_identity_prompt(prompt: str, config: dict, cat_id: str, image_hash: str) -> str:
    """Specialize identity while retaining the versioned motion's action paragraphs.

    Both metadata preparation and inference use this exact renderer. The CSV
    pins its final text, so changes cannot silently reuse old generation states.
    """
    identity = config["cats"].get(cat_id)
    if identity is None:
        raise MetadataError(f"身份提示词未登记 cat_id: {cat_id}")
    if identity["image_sha256"] != image_hash:
        raise MetadataError(f"{cat_id} 首帧图与身份描述的 image_sha256 不符；请重新审阅猫图和身份描述")
    templates = {key: value.format(description=identity["description"], anchor=identity["anchor"])
                 for key, value in config["templates"].items()}
    validate_prompt(prompt, Path("motion prompt"))
    parts = re.split(r"(?m)^([a-z_]+):[ \t]*\n", prompt)
    sections = dict(zip(parts[1::2], (value.strip() for value in parts[2::2])))
    for section, template in (("subject_definitions", "subject_definition"),
                              ("retention_analysis", "retention")):
        sections[section], count = re.subn(r"(?m)^<Subject 1>[^\n]*$",
                                          lambda _: templates[template], sections[section])
        if count != 1:
            raise MetadataError(f"动作提示词 {section} 必须有一行独立的 <Subject 1>，无法组合身份")
    paragraphs = sections["detailed_description"].splitlines()
    openings = [i for i, paragraph in enumerate(paragraphs) if paragraph.startswith("[Shot 1]")]
    if len(openings) != 1:
        raise MetadataError("动作提示词必须有一行独立的 [Shot 1] 开场，无法组合身份")
    opening = openings[0]
    paragraphs[opening] = templates["opening"]
    for i in range(opening + 1, len(paragraphs)):
        if paragraphs[i].startswith("Keep the action continuous through the final frame"):
            # Replace shared generic ending prose; keep the actual ending pose
            # in the preceding action paragraph and the scene/sound constraints.
            tail = paragraphs[i].partition("No person,")[2]
            paragraphs[i] = templates["closing"] + (" No person," + tail if tail else "")
            break
    else:
        paragraphs[-1] += " " + templates["closing"]
    middle = opening + max(1, (len(paragraphs) - opening - 1) // 2)
    paragraphs[middle] += " " + templates["continuity"]
    sections["detailed_description"] = "\n".join(paragraphs)
    result = "\n\n".join(f"{section}:\n{sections[section]}" for section in PROMPT_SECTIONS)
    validate_prompt(result, Path(config["prompt_version"]))
    return result


def read_motions(path: Path, selectors: list[str] | None) -> list[dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("motions"), list):
        raise MetadataError("motions.json 必须包含 schema_version=1 和 motions 数组")
    ids, slugs, references, motions = set(), set(), set(), []
    for source in config["motions"]:
        row = dict(source)
        motion_id, slug = row.get("motion_id", ""), row.get("motion_slug", "")
        if not NAMED_ID.fullmatch(motion_id) or not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
            raise MetadataError(f"动作 ID 或目录名称无效: {motion_id}/{slug}")
        index = int(motion_id.split("-", 1)[0])
        if index in ids or slug in slugs:
            raise MetadataError(f"动作编号或目录重复: {motion_id}/{slug}")
        ids.add(index)
        slugs.add(slug)
        reference = resolve_file(row.get("reference_video"), path.parent, "reference_video")
        references.add(reference)
        row["source_reference"] = reference
        duration, fps = row.get("duration_seconds"), row.get("fps")
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 4 <= duration <= 15:
            raise MetadataError(f"{motion_id} duration_seconds 必须在 [4, 15] 内")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps != 24:
            raise MetadataError(f"{motion_id} 当前 H3 Ref2VA 工作流要求 fps=24")
        prompt_file = resolve_file(row.get("prompt_file"), path.parent, "prompt_file")
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        validate_prompt(prompt, prompt_file)
        for field in ("prompt_version", "prompt_status"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise MetadataError(f"{motion_id} 缺少 {field}")
        motions.append({**row, "prompt": prompt, "prompt_file_path": prompt_file})
    if not motions:
        raise MetadataError("motions.json 没有动作")
    unknown = {p.resolve() for p in path.parent.glob("*.mp4")} - references
    if unknown and selectors is None:
        raise MetadataError("新增参考视频尚未配置独立提示词，请先登记 motions.json: "
                            + ", ".join(sorted(p.name for p in unknown))
                            + "；只运行全部已登记动作可显式指定 --motion-ids '*'")
    for selector in selectors or []:
        if not any(matches(row["motion_id"], [selector], row["motion_slug"]) for row in motions):
            raise MetadataError(f"找不到 motion_id: {selector}")
    return sorted(motions, key=lambda row: int(row["motion_id"].split("-", 1)[0]))


def run_media(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise MetadataError(f"缺少本机工具 {command[0]}；请在搬运前将 ffmpeg/ffprobe 一起备好") from exc
    except subprocess.CalledProcessError as exc:
        raise MetadataError(f"{command[0]} 失败: {(exc.stderr or str(exc))[-3000:]}") from exc
    return result.stdout


def probe(path: Path) -> dict:
    return json.loads(run_media(["ffprobe", "-v", "error", "-show_streams",
                                 "-show_format", "-of", "json", str(path)]))


def image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
    else:
        streams = probe(path).get("streams", [])
        stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        width, height = stream.get("width", 0), stream.get("height", 0)
    if width <= 0 or height <= 0:
        raise MetadataError(f"无法读取首帧宽高: {path}")
    return width, height


def aspect_ratios(width: int, height: int) -> tuple[str, str]:
    divisor = math.gcd(width, height)
    source = f"{width // divisor}:{height // divisor}"
    def ratio_distance(value: str) -> float:
        numerator, denominator = map(int, value.split(":"))
        return abs(math.log((width / height) / (numerator / denominator)))
    return source, min(SUPPORTED_ASPECT_RATIOS, key=ratio_distance)


def validate_reference(motion: dict) -> tuple[dict, float]:
    """Validate the source video without creating derived files."""
    source = motion["source_reference"]
    source_info = probe(source)
    video = next((s for s in source_info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MetadataError(f"参考素材没有视频轨: {source}")
    duration = float(video.get("duration") or source_info.get("format", {}).get("duration", 0))
    if not 2 <= duration <= 15:
        raise MetadataError(f"{source.name} 参考时长 {duration:g}s 必须在 [2, 15] 内")
    expected_duration = motion.get("reference_duration_seconds")
    if expected_duration is not None and not math.isclose(
            duration, float(expected_duration), abs_tol=1 / motion["fps"]):
        raise MetadataError(f"{source.name} 时长 {duration:g}s 与已审阅参考时长 "
                            f"{expected_duration}s 不符；请先匹配视频、配置和动作时间线")
    return video, duration


def prepare_reference(motion: dict, prepared_dir: Path) -> tuple[Path, str]:
    """Create or repair an ignored, machine-local silent video cache."""
    source = motion["source_reference"]
    source_hash = sha256(source)
    target = prepared_dir / f"{source.stem}.motion.mp4"
    sidecar = target.with_suffix(".json")
    video, duration = validate_reference(motion)
    cached = {}
    if sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    if target.is_file() and cached.get("source_sha256") == source_hash:
        if cached.get("prepared_sha256") == sha256(target) and cached.get("audio_removed") is True:
            return target, cached["prepared_sha256"]
    prepared_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".mp4", dir=prepared_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        run_media(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                   "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(temporary)])
        info = probe(temporary)
        if not temporary.stat().st_size or any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
            raise MetadataError(f"静音派生素材校验失败: {source}")
        prepared_hash = sha256(temporary)
        os.replace(temporary, target)
        atomic_text(sidecar, json.dumps({
            "source_file": Path(os.path.relpath(source, prepared_dir)).as_posix(),
            "source_sha256": source_hash, "prepared_sha256": prepared_hash,
            "audio_removed": True, "video_stream_copy": True,
            "duration_seconds": duration,
            "source_width": video.get("width"), "source_height": video.get("height"),
        }, ensure_ascii=False, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return target, prepared_hash


def prepare_metadata(motions_path: Path, cat_catalog: Path, cat_dir: Path,
                     output: Path, seeds: list[int], cat_ids: list[str] | None = None,
                     motion_ids: list[str] | None = None, *,
                     prepare_media: bool = False,
                     identity_prompts: Path | None = None) -> list[dict[str, str]]:
    motions_path, cat_catalog, cat_dir, output = (
        p.expanduser().resolve() for p in (motions_path, cat_catalog, cat_dir, output))
    if output in (motions_path, cat_catalog):
        raise MetadataError("输出 metadata 不能覆盖 motions 配置或 cat catalog")
    if not seeds or len(set(seeds)) != len(seeds):
        raise MetadataError("seed 列表必须非空且不能重复")
    if any(type(seed) is not int or not 0 <= seed < (1 << 63) for seed in seeds):
        raise MetadataError("每个 seed 必须是 [0, 2^63) 范围的整数")
    cats = read_catalog(cat_catalog, cat_dir)
    motions = read_motions(motions_path, motion_ids)
    inputs = {cat["image"] for cat in cats} | {
        motion[field] for motion in motions for field in ("source_reference", "prompt_file_path")}
    identities = None
    if identity_prompts is not None:
        identity_prompts = identity_prompts.expanduser().resolve()
        inputs.add(identity_prompts)
        identities = read_identity_prompts(identity_prompts)
    if output in inputs:
        raise MetadataError("输出 metadata 不能覆盖首帧、参考视频或提示词")
    for selector in cat_ids or []:
        if not any(matches(cat["cat_id"], [selector]) for cat in cats):
            raise MetadataError(f"找不到 cat_id: {selector}")
    selected_motions = [m for m in motions if matches(m["motion_id"], motion_ids, m["motion_slug"])]
    references = {}
    for motion in selected_motions:
        source = motion["source_reference"]
        prepared_dir = motions_path.parent / "prepared"
        _, reference_duration = validate_reference(motion)
        if prepare_media:
            reference, _ = prepare_reference(motion, prepared_dir)
        else:
            reference = prepared_dir / f"{source.stem}.motion.mp4"
        # CSVs pin the original input, since stream-copy muxing may produce
        # different bytes across ffmpeg versions. Cache hashes live in sidecars.
        references[motion["motion_id"]] = reference, sha256(source), reference_duration
    rows = []
    for cat_index, cat in enumerate(cats):
        if not matches(cat["cat_id"], cat_ids):
            continue
        width, height = image_dimensions(cat["image"])
        image_hash = sha256(cat["image"]) if identities else ""
        source_ratio, target_ratio = aspect_ratios(width, height)
        for motion_index, motion in enumerate(motions):
            if not matches(motion["motion_id"], motion_ids, motion["motion_slug"]):
                continue
            reference, source_reference_hash, reference_duration = references[motion["motion_id"]]
            source_id = cat_index * len(motions) + motion_index
            prompt = (compose_identity_prompt(motion["prompt"], identities, cat["cat_id"], image_hash)
                      if identities else motion["prompt"])
            for seed_index, seed in enumerate(seeds):
                case_id = f"{source_id * len(seeds) + seed_index:03d}"
                output_name = f"{case_id}_{cat['cat_id']}_{motion['motion_id']}_seed-{seed}"
                if len(output_name) > 240:
                    raise MetadataError(f"输出名称过长: {output_name}")
                row = {
                    "id": case_id, "source_id": source_id,
                    "input_image": Path(os.path.relpath(cat["image"], output.parent)).as_posix(),
                    "source_reference_video": Path(os.path.relpath(motion["source_reference"], output.parent)).as_posix(),
                    "reference_video": Path(os.path.relpath(reference, output.parent)).as_posix(),
                    "prompt_file": Path(os.path.relpath(motion["prompt_file_path"], output.parent)).as_posix(),
                    "cat_idx": cat["cat_idx"], "cat_id": cat["cat_id"],
                    "cat_name_zh": cat["cat_name_zh"], "motion_idx": motion["motion_id"].split("-", 1)[0],
                    "motion_id": motion["motion_id"], "motion_slug": motion["motion_slug"],
                    "motion_name_zh": motion.get("motion_name_zh", ""), "seed": seed,
                    "output_name": output_name, "duration_seconds": motion["duration_seconds"],
                    "reference_duration_seconds": reference_duration,
                    "fps": motion["fps"], "width": width, "height": height,
                    "source_aspect_ratio": source_ratio, "aspect_ratio": target_ratio,
                    "prompt_version": motion["prompt_version"], "prompt_status": motion["prompt_status"],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "source_reference_sha256": source_reference_hash,
                }
                if identities:
                    row["identity_prompt_file"] = Path(os.path.relpath(identity_prompts, output.parent)).as_posix()
                    row["prompt_version"] += "+" + identities["prompt_version"]
                    row["prompt_status"] = identities["prompt_status"]
                rows.append({field: str(value) for field, value in row.items()})
    if not rows:
        raise MetadataError("筛选后没有猫咪/动作组合")
    write_metadata(output, rows)
    return rows


def write_metadata(output: Path, rows: list[dict[str, str]], *,
                   relative_to: Path | None = None) -> None:
    """Write a subset without renumbering tasks; rebase paths for its location."""
    output = output.expanduser().resolve()
    content = io.StringIO(newline="")
    fields = FIELDS + (["identity_prompt_file"] if rows and "identity_prompt_file" in rows[0] else [])
    writer = csv.DictWriter(content, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in rows:
        row = dict(source)
        if relative_to is not None:
            for field in ("input_image", "source_reference_video", "reference_video", "prompt_file",
                          "identity_prompt_file"):
                if not row.get(field):
                    continue
                target = (relative_to / row[field]).resolve()
                row[field] = Path(os.path.relpath(target, output.parent)).as_posix()
        writer.writerow(row)
    atomic_text(output, content.getvalue())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, default=ROOT / "exp_Ref2VA/motions.json")
    parser.add_argument("--cat-catalog", type=Path, default=ROOT / "exp_Ref2VA/cat_catalog.csv")
    parser.add_argument("--cat-dir", type=Path, default=ROOT / "data_h3/cat_ids")
    parser.add_argument("--output", type=Path,
                        help="默认metadata_all.csv；启用身份提示词时默认metadata_identity_v1.csv，保留旧版")
    parser.add_argument("--identity-prompts", type=Path,
                        help="可选：按cat_id组合身份描述的版本化JSON；省略则沿用原动作提示词")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="每个组合的实际seed，默认仅 0：每个参考视频80只猫，不叠加行号")
    parser.add_argument("--cat-ids", nargs="+", help="猫编号或完整 ID，例如 00 38-peterbald")
    parser.add_argument("--motion-ids", nargs="+", help="动作编号、完整 ID 或目录名；'*'显式选择全部已登记动作")
    parser.add_argument("--prepare-media", action="store_true",
                        help="同时生成静音参考缓存；默认由推理入口在运行时自动准备")
    parser.add_argument("--smoke-output", type=Path,
                        help="同时输出单猫覆盖全部所选动作的CSV，复用全量任务编号和seed")
    parser.add_argument("--smoke-cat-id", default="00",
                        help="预览使用的猫编号或完整ID，默认00；须在本次所选猫中")
    parser.add_argument("--per-motion-dir", type=Path,
                        help="同时按动作拆分CSV到此目录，文件名为<motion_slug>.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is None:
        args.output = ROOT / "exp_Ref2VA" / (
            "metadata_identity_v1.csv" if args.identity_prompts else "metadata_all.csv")
    try:
        # Validate extra destinations before writing any metadata. The motion
        # registry also supplies the exact filenames used for split outputs.
        destinations = [args.output.expanduser().resolve()]
        motions = read_motions(args.motions.expanduser().resolve(), args.motion_ids)
        selected = [m for m in motions if matches(m["motion_id"], args.motion_ids, m["motion_slug"])]
        if args.smoke_output:
            cats = read_catalog(args.cat_catalog.expanduser().resolve(), args.cat_dir.expanduser().resolve())
            if not any(matches(c["cat_id"], [args.smoke_cat_id]) and matches(c["cat_id"], args.cat_ids)
                       for c in cats):
                raise MetadataError(f"预览猫 {args.smoke_cat_id} 不在本次所选猫中")
            destinations.append(args.smoke_output.expanduser().resolve())
        if args.per_motion_dir:
            destinations.extend((args.per_motion_dir / f"{m['motion_slug']}.csv").expanduser().resolve()
                                for m in selected)
        protected = {args.motions.expanduser().resolve(), args.cat_catalog.expanduser().resolve()}
        if args.identity_prompts:
            protected.add(args.identity_prompts.expanduser().resolve())
        protected.update(m[k] for m in motions for k in ("source_reference", "prompt_file_path"))
        protected.update(p.resolve() for p in args.cat_dir.expanduser().iterdir() if p.is_file())
        if len(set(destinations)) != len(destinations) or set(destinations) & protected:
            raise MetadataError("metadata 输出路径重复或会覆盖输入文件")
        rows = prepare_metadata(args.motions, args.cat_catalog, args.cat_dir, args.output,
                                args.seeds, args.cat_ids, args.motion_ids,
                                prepare_media=args.prepare_media, identity_prompts=args.identity_prompts)
        if args.smoke_output:
            smoke = [r for r in rows if matches(r["cat_id"], [args.smoke_cat_id])]
            write_metadata(args.smoke_output.expanduser().resolve(), smoke,
                           relative_to=args.output.expanduser().resolve().parent)
            print(f"单猫预览 {len(smoke)} 行: {args.smoke_output}")
        if args.per_motion_dir:
            for motion in selected:
                subset = [r for r in rows if r["motion_slug"] == motion["motion_slug"]]
                write_metadata((args.per_motion_dir / f"{motion['motion_slug']}.csv").expanduser().resolve(),
                               subset, relative_to=args.output.expanduser().resolve().parent)
            print(f"已按动作拆分 {len(selected)} 份CSV: {args.per_motion_dir}")
    except (MetadataError, OSError, csv.Error, ValueError, TypeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已保存 {len(rows)} 行 Ref2VA metadata: {args.output}")
    print(f"猫咪 {len({r['cat_id'] for r in rows})} × 动作 {len({r['motion_id'] for r in rows})}"
          f" × seed {len(args.seeds)}；提示词状态：待模型验证")
    print(f"首个输出: videos/{rows[0]['motion_slug']}/{rows[0]['output_name']}.mp4")
    print("prompt_file 引用单份提示词；reference_video 为运行时自动准备的静音缓存路径")
    if args.identity_prompts:
        print("identity_prompt_file 按cat_id组合专属身份；prompt_sha256 校验组合后的完整提示词")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
