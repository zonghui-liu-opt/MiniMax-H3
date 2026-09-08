#!/usr/bin/env python3
"""Select named cat/motion cases and expand explicit seeds for the H3 client."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path

import batch_fl2va_sglang as batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [0, 10000, 20000]


def matches(value: str, selectors: list[str] | None) -> bool:
    """Accept either a numeric index (00 or 0) or the complete readable ID."""
    return selectors is None or any(
        selector == value
        or (selector.isascii() and selector.isdigit()
            and int(selector) == int(value.split("-", 1)[0]))
        for selector in selectors
    )


def prepare_metadata(metadata: Path, output: Path, seeds: list[int],
                     cat_ids: list[str] | None = None,
                     motion_ids: list[str] | None = None) -> list[dict[str, str]]:
    metadata = metadata.expanduser().resolve()
    output = output.expanduser().resolve()
    if metadata == output:
        raise batch.BatchError("输出文件不能覆盖输入 metadata；请指定新的 CSV 路径")
    if not seeds or len(set(seeds)) != len(seeds):
        raise batch.BatchError("seed 列表必须非空且不能重复")
    for seed in seeds:
        batch.validate_seed(seed, context="--seeds")

    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"input_image", "prompt", "cat_id", "motion_id"}
        if not required.issubset(fields):
            raise batch.BatchError("输入必须包含 input_image、prompt、cat_id、motion_id 列")
        source = list(reader)

    pairs = set()
    for index, row in enumerate(source):
        if None in row or any(value is None for value in row.values()):
            raise batch.BatchError(f"CSV 第 {index + 2} 行的列数与表头不一致")
        for field in ("cat_id", "motion_id"):
            if not re.fullmatch(r"[0-9]+-[a-z][a-z0-9]*(?:-[a-z0-9]+)*", row[field]):
                raise batch.BatchError(f"{field} 必须使用编号加英文名称，如 13-small-jump")
        pair = (row["cat_id"], row["motion_id"])
        if pair in pairs:
            raise batch.BatchError("输入猫咪/动作组合重复；请使用未扩展 seed 的基础 metadata")
        pairs.add(pair)
    for field, selectors in (("cat_id", cat_ids), ("motion_id", motion_ids)):
        for selector in selectors or []:
            if not any(matches(row[field], [selector]) for row in source):
                raise batch.BatchError(f"找不到 {field}: {selector}")

    rows = []
    for source_id, row in enumerate(source):
        if not matches(row["cat_id"], cat_ids) or not matches(row["motion_id"], motion_ids):
            continue
        if not row["prompt"].strip():
            raise batch.BatchError(f"CSV 第 {source_id + 2} 行的 prompt 为空")
        copied = dict(row)
        for field in ("input_image", "last_image"):
            if copied.get(field):
                media = batch.resolve_media_path(
                    copied[field], metadata.parent, field=field, row=source_id + 2)
                copied[field] = Path(os.path.relpath(media, output.parent)).as_posix()
        for seed in seeds:
            name = f"{row['cat_id']}_{row['motion_id']}_seed-{seed}"
            # The existing inference client applies slugify and a 100-character limit.
            # Check the actual contract so no readable label or seed gets truncated.
            if batch.slugify(name) != name:
                raise batch.BatchError(f"输出名过长或含不支持字符: {name}")
            rows.append({**copied, "id": f"{len(rows):03d}",
                         "source_id": str(source_id), "seed": str(seed),
                         "output_name": name})
    if not rows:
        raise batch.BatchError("筛选后没有猫咪/动作组合")

    output_fields = ["id", "source_id"] + [
        field for field in fields if field not in {"id", "source_id", "seed", "output_name"}
    ] + ["seed", "output_name"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output.parent,
                prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=ROOT / "data_h3/metadata_smoke_v5.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="每个选中猫咪/动作组合使用的实际 seed；默认 0 10000 20000")
    parser.add_argument("--cat-ids", nargs="+", help="猫编号或完整名称，如 00 或 38-peterbald")
    parser.add_argument("--motion-ids", nargs="+", help="动作编号或完整名称，如 13 14")
    args = parser.parse_args(argv)
    try:
        rows = prepare_metadata(args.metadata, args.output, args.seeds,
                                args.cat_ids, args.motion_ids)
    except (OSError, csv.Error, batch.BatchError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已保存 {len(rows)} 行（{len(rows) // len(args.seeds)} 个组合 × "
          f"{len(args.seeds)} 个 seed）：{args.output}")
    print(f"首个输出文件名：{rows[0]['id']}_{rows[0]['output_name']}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
