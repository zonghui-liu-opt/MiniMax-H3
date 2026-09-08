from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_multiseed_metadata as prepare


class PrepareMultiseedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "cat.png"
        self.image.write_bytes(b"test image")
        self.metadata = self.root / "base.csv"
        with self.metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "input_image", "prompt", "cat_id", "motion_id", "seed", "output_name"])
            writer.writeheader()
            for cat in ["00-orange-shorthair", "38-peterbald"]:
                for motion in ["13-small-jump", "14-play-teaser-wand"]:
                    writer.writerow({"input_image": "cat.png", "prompt": f"{cat}: {motion}\nexact prompt",
                                     "cat_id": cat, "motion_id": motion,
                                     "seed": "99", "output_name": "old-name"})

    def test_selected_seeds_reach_client_with_readable_unique_names(self):
        output = self.root / "nested" / "selected.csv"
        rows = prepare.prepare_metadata(self.metadata, output, [0, 42, 10000], ["0"], ["13"])
        self.assertEqual([row["seed"] for row in rows], ["0", "42", "10000"])
        self.assertEqual(len({row["prompt"] for row in rows}), 1)
        args = prepare.batch.build_parser().parse_args([
            "--metadata", str(output), "--single-frame", "--seed", "888"])
        cases = prepare.batch.load_cases(args)
        self.assertEqual([case.seed for case in cases], [0, 42, 10000])
        self.assertEqual([case.name for case in cases], [
            "000_00-orange-shorthair_13-small-jump_seed-0",
            "001_00-orange-shorthair_13-small-jump_seed-42",
            "002_00-orange-shorthair_13-small-jump_seed-10000"])
        for case in cases:
            self.assertEqual(case.first_image, self.image.resolve())
            self.assertIsNone(case.last_image)
            request = prepare.batch.build_request(case, args)
            self.assertEqual(request["seed"], case.seed)
            self.assertEqual(request["prompt"], rows[0]["prompt"])

    def test_full_expansion_and_named_selectors(self):
        rows = prepare.prepare_metadata(self.metadata, self.root / "all.csv", [1, 2])
        self.assertEqual(len(rows), 8)
        selected = prepare.prepare_metadata(self.metadata, self.root / "selected.csv", [7],
                                            ["38-peterbald"], ["14-play-teaser-wand"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_id"], "3")
        self.assertEqual(selected[0]["cat_id"], "38-peterbald")

    def test_invalid_selection_and_seeds_do_not_overwrite_output(self):
        output = self.root / "out.csv"
        output.write_text("existing output")
        for seeds, cats in [([0, 0], None), ([-1], None), ([1 << 63], None),
                            ([0], ["99"]), ([0], ["00", "unknown"])]:
            with self.subTest(seeds=seeds, cats=cats):
                with self.assertRaises(prepare.batch.BatchError):
                    prepare.prepare_metadata(self.metadata, output, seeds, cats)
                self.assertEqual(output.read_text(), "existing output")

    def test_rejects_nested_expansion_and_input_overwrite(self):
        expanded = self.root / "expanded.csv"
        prepare.prepare_metadata(self.metadata, expanded, [0, 1])
        with self.assertRaises(prepare.batch.BatchError):
            prepare.prepare_metadata(expanded, self.root / "nested.csv", [2])
        before = self.metadata.read_bytes()
        with self.assertRaises(prepare.batch.BatchError):
            prepare.prepare_metadata(self.metadata, self.metadata, [0])
        self.assertEqual(self.metadata.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
