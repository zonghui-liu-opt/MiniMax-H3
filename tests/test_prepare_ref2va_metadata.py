from __future__ import annotations

import csv
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_ref2va_metadata as prepare


class PrepareRef2VATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment = self.root / "exp_Ref2VA"
        self.experiment.mkdir()
        self.cat_dir = self.root / "data_h3/cat_ids"
        self.cat_dir.mkdir(parents=True)
        self.catalog = self.experiment / "cat_catalog.csv"
        with self.catalog.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cat_idx", "cat_id", "cat_name_zh", "image_filename"])
            writer.writeheader()
            for index, name in [("00", "orange-shorthair-mackerel-tabby"), ("38", "peterbald")]:
                filename = f"{index}_猫咪.png"
                # The preparation code reads PNG dimensions, while decoding is
                # independently validated by the inference client's preflight.
                (self.cat_dir / filename).write_bytes(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 480, 832))
                writer.writerow({"cat_idx": index, "cat_id": f"{index}-{name}",
                                 "cat_name_zh": "猫咪", "image_filename": filename})
        self.source = self.experiment / "Ref_drag_ear.mp4"
        self.source.write_bytes(b"source placeholder")
        self.prompt = self.experiment / "drag_ear.v1.en.txt"
        self.prompt.write_text((ROOT / "exp_Ref2VA/prompts/drag_ear.v1.en.txt").read_text(), encoding="utf-8")
        self.motions = self.experiment / "motions.json"
        self.config = {"schema_version": 1, "motions": [{
            "motion_id": "02-pull-left-ear", "motion_slug": "drag_ear",
            "motion_name_zh": "左前爪擦耳", "reference_video": self.source.name,
            "prompt_file": self.prompt.name, "duration_seconds": 4, "reference_duration_seconds": 4, "fps": 24,
            "prompt_version": "drag_ear.v1", "prompt_status": "pending_model_validation"}]}
        self.write_config()
        self.output = self.experiment / "metadata.csv"
        self.prepared = self.experiment / "prepared/Ref_drag_ear.motion.mp4"

    def write_config(self):
        self.motions.write_text(json.dumps(self.config), encoding="utf-8")

    def run_prepare(self, output=None, seeds=None, cats=None, motions=None):
        with patch.object(prepare, "validate_reference", return_value=({}, 4.0)):
            return prepare.prepare_metadata(self.motions, self.catalog, self.cat_dir,
                                            output or self.output,
                                            seeds if seeds is not None else [0, 10000, 20000],
                                            cats, motions)

    def test_explicit_seeds_stable_names_and_portable_paths(self):
        rows = self.run_prepare()
        self.assertEqual(len(rows), 6)
        self.assertEqual([row["seed"] for row in rows], ["0", "10000", "20000"] * 2)
        self.assertEqual(rows[0]["output_name"],
                         "000_00-orange-shorthair-mackerel-tabby_02-pull-left-ear_seed-0")
        self.assertEqual(rows[-1]["output_name"], "005_38-peterbald_02-pull-left-ear_seed-20000")
        self.assertEqual(rows[0]["aspect_ratio"], "9:16")
        self.assertEqual(rows[0]["source_aspect_ratio"], "15:26")
        self.assertEqual(rows[0]["duration_seconds"], "4")
        self.assertEqual(rows[0]["reference_duration_seconds"], "4.0")
        self.assertNotIn("prompt", rows[0])
        self.assertNotIn("reference_sha256", rows[0])
        self.assertEqual(rows[0]["prompt_sha256"],
                         hashlib.sha256(self.prompt.read_text().strip().encode()).hexdigest())
        self.assertEqual(rows[0]["source_reference_sha256"], prepare.sha256(self.source))
        self.assertFalse(self.prepared.parent.exists())
        selected = self.run_prepare(self.experiment / "nested/selected.csv", cats=["38"], motions=["drag_ear"])
        self.assertEqual([r["output_name"] for r in selected], [r["output_name"] for r in rows[3:]])
        for field, target in (("input_image", self.cat_dir / "38_猫咪.png"),
                              ("source_reference_video", self.source),
                              ("reference_video", self.prepared),
                              ("prompt_file", self.prompt)):
            self.assertFalse(Path(selected[0][field]).is_absolute())
            self.assertEqual((self.experiment / "nested" / selected[0][field]).resolve(), target.resolve())
        with self.output.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), rows)
        moved = self.root / "moved"
        shutil.copytree(self.experiment, moved / "exp_Ref2VA")
        shutil.copytree(self.root / "data_h3", moved / "data_h3")
        for row in rows:
            for field in ("input_image", "source_reference_video", "prompt_file"):
                self.assertTrue((moved / "exp_Ref2VA" / row[field]).is_file())
            self.assertFalse((moved / "exp_Ref2VA" / row["reference_video"]).exists())

    def test_metadata_prepares_media_only_when_explicitly_requested(self):
        with patch.object(prepare, "validate_reference", return_value=({}, 4.0)), \
             patch.object(prepare, "prepare_reference", return_value=(self.prepared.resolve(), "cache hash")) as media:
            rows = prepare.prepare_metadata(self.motions, self.catalog, self.cat_dir,
                                            self.output, [0], prepare_media=True)
            media.assert_called_once()
        self.assertEqual(rows[0]["source_reference_sha256"], prepare.sha256(self.source))
        self.assertEqual(rows[0]["reference_video"], "prepared/Ref_drag_ear.motion.mp4")

    def test_invalid_source_video_preserves_metadata_and_creates_no_cache(self):
        self.output.write_text("existing metadata")
        with patch.object(prepare, "probe", return_value={"streams": []}):
            with self.assertRaisesRegex(prepare.MetadataError, "没有视频轨"):
                prepare.prepare_metadata(self.motions, self.catalog, self.cat_dir, self.output, [0])
        self.assertEqual(self.output.read_text(), "existing metadata")
        self.assertFalse(self.prepared.parent.exists())

    def test_unknown_video_requires_its_own_motion_prompt(self):
        (self.experiment / "Ref_jump.mp4").write_bytes(b"different motion")
        self.output.write_text("existing metadata")
        with self.assertRaisesRegex(prepare.MetadataError, "尚未配置独立提示词"):
            self.run_prepare()
        self.assertEqual(self.output.read_text(), "existing metadata")
        self.assertEqual(len(self.run_prepare(motions=["drag_ear"])), 6)

    def test_unprefixed_and_space_prefixed_references_are_not_silently_skipped(self):
        for filename in ("walk_forward_to_screen.mp4", " Ref_curl_up_and_lie_down.mp4"):
            with self.subTest(filename=filename):
                path = self.experiment / filename
                path.write_bytes(b"new reference")
                with self.assertRaisesRegex(prepare.MetadataError, "尚未配置独立提示词"):
                    self.run_prepare()
                path.unlink()

    def cli_args(self, *extra):
        return ["--motions", str(self.motions), "--cat-catalog", str(self.catalog),
                "--cat-dir", str(self.cat_dir), "--output", str(self.output), *extra]

    def test_cli_single_seed_smoke_and_split_share_full_task_names_and_rebased_paths(self):
        motion = dict(self.config["motions"][0])
        motion.update(motion_id="03-pull-right-ear", motion_slug="drag_ear_mirror")
        self.config["motions"].append(motion)
        self.write_config()
        smoke_path = self.experiment / "preview/smoke.csv"
        split_dir = self.experiment / "metadata"
        with patch.object(prepare, "validate_reference", return_value=({}, 4.0)):
            self.assertEqual(prepare.main(self.cli_args(
                "--smoke-output", str(smoke_path), "--smoke-cat-id", "38",
                "--per-motion-dir", str(split_dir))), 0)
        def read(path):
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        full, smoke = read(self.output), read(smoke_path)
        self.assertEqual(len(full), 4)
        self.assertEqual({r["seed"] for r in full}, {"0"})
        self.assertEqual({r["cat_idx"] for r in smoke}, {"38"})
        self.assertEqual({r["output_name"] for r in smoke},
                         {r["output_name"] for r in full if r["cat_idx"] == "38"})
        splits = [(path, read(path)) for path in split_dir.glob("*.csv")]
        self.assertEqual(len(splits), 2)
        self.assertTrue(all(len(rows) == 2 for _, rows in splits))
        self.assertEqual({r["output_name"] for _, rows in splits for r in rows},
                         {r["output_name"] for r in full})
        by_name = {r["output_name"]: r for r in full}
        for path, subset in [(smoke_path, smoke), *splits]:
            for row in subset:
                original = by_name[row["output_name"]]
                self.assertEqual(row["id"], original["id"])
                self.assertEqual(row["prompt_sha256"], original["prompt_sha256"])
                for field in ("input_image", "source_reference_video", "reference_video", "prompt_file"):
                    self.assertEqual((path.parent / row[field]).resolve(),
                                     (self.output.parent / original[field]).resolve())

    def test_invalid_smoke_selection_or_output_collision_preserves_existing_metadata(self):
        self.output.write_text("existing metadata")
        for options in (("--smoke-output", str(self.output)),
                        ("--smoke-output", str(self.source)),
                        ("--smoke-output", str(self.experiment / "smoke.csv"), "--smoke-cat-id", "99"),
                        ("--smoke-output", str(self.experiment / "smoke.csv"), "--cat-ids", "38")):
            with self.subTest(options=options):
                self.assertEqual(prepare.main(self.cli_args(*options)), 2)
                self.assertEqual(self.output.read_text(), "existing metadata")

    def test_invalid_seeds_selection_and_prompt_preserve_metadata(self):
        self.output.write_text("existing metadata")
        for seeds, cats, motions in [([], None, None), ([0, 0], None, None),
                                     ([-1], None, None), ([1 << 63], None, None),
                                     ([0], ["99"], None), ([0], None, ["jump"])]:
            with self.subTest(seeds=seeds, cats=cats, motions=motions):
                with self.assertRaises(prepare.MetadataError):
                    self.run_prepare(seeds=seeds, cats=cats, motions=motions)
                self.assertEqual(self.output.read_text(), "existing metadata")
        self.prompt.write_text("integrated_multimodal_description: wrong mode", encoding="utf-8")
        with self.assertRaisesRegex(prepare.MetadataError, "官方 Ref2VA"):
            self.run_prepare()
        self.assertEqual(self.output.read_text(), "existing metadata")

    def test_duplicate_or_unregistered_cat_and_input_overwrite_fail(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(prepare.MetadataError, "不能覆盖"):
            self.run_prepare(output=self.source)
        self.assertEqual(self.source.read_bytes(), original)
        (self.cat_dir / "39_新猫.png").write_bytes(b"image")
        with self.assertRaisesRegex(prepare.MetadataError, "尚未登记"):
            self.run_prepare()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires local ffmpeg/ffprobe")
    def test_reference_audio_removed_source_preserved_and_cache_repaired(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=gray:s=32x48:r=24:d=4", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(self.source)], check=True, capture_output=True)
        source_hash = prepare.sha256(self.source)
        motion = prepare.read_motions(self.motions, None)[0]
        target, target_hash = prepare.prepare_reference(motion, self.prepared.parent)
        info = prepare.probe(target)
        self.assertEqual([s["codec_type"] for s in info["streams"]], ["video"])
        self.assertEqual(prepare.sha256(self.source), source_hash)
        with patch.object(prepare, "run_media", wraps=prepare.run_media) as run:
            self.assertEqual(prepare.prepare_reference(motion, self.prepared.parent), (target, target_hash))
            self.assertFalse(any(call.args[0][0] == "ffmpeg" for call in run.call_args_list))
        target.write_bytes(b"interrupted or damaged derivative")
        with patch.object(prepare, "run_media", wraps=prepare.run_media) as run:
            restored, restored_hash = prepare.prepare_reference(motion, self.prepared.parent)
            self.assertEqual(restored, target)
            self.assertEqual(prepare.sha256(restored), restored_hash)
            self.assertTrue(any(call.args[0][0] == "ffmpeg" for call in run.call_args_list))
        motion["reference_duration_seconds"] = 3
        with self.assertRaisesRegex(prepare.MetadataError, "时长"):
            prepare.prepare_reference(motion, self.prepared.parent)


if __name__ == "__main__":
    unittest.main()
