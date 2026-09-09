from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ref2va_preflight as pre


def tensor_file(path, data=b"\0\0\0\0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def report():
    return {"errors": [], "warnings": [], "checks": [], "actions": []}


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self):
        partition = self.root / "Ref2VA"
        json_file(partition / "model_index.json", {"_minimax_h3": {"partition": "ref2va"}})
        for component in ("transformer", "text_encoder", "video_vae", "audio_vae"):
            json_file(partition / component / "config.json", {})
            json_file(partition / component / "model.safetensors.index.json",
                      {"weight_map": {"x": "model-00001-of-00001.safetensors"}})
            tensor_file(partition / component / "model-00001-of-00001.safetensors")
        for component in ("processor", "tokenizer"):
            for name in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json"):
                json_file(partition / component / name, {})
        return partition

    def source(self, *, keyframe=True, video=True, validation=True):
        directory = self.root / pre.STAGES
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "task_profiles.py").write_text("\n".join([
            'MINIMAX_H3_TASK_REF2VA = "ref2va"',
            'MINIMAX_H3_CONDITION_ROLE_KEYFRAME = "keyframe"',
            'MINIMAX_H3_CONDITION_ROLE_REFERENCE = "reference"',
            'MINIMAX_H3_TASK_PROFILES: dict = {MINIMAX_H3_TASK_REF2VA: Profile(',
            f'video_reference_supported={video!r}, condition_rules=(',
            'Rule(role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME, condition_type="image"),' if keyframe else '',
            'Rule(role=MINIMAX_H3_CONDITION_ROLE_REFERENCE, condition_type="video"),',
            '))}',
        ]))
        (directory / "request_validation.py").write_text(
            "if profile.task == MINIMAX_H3_TASK_REF2VA:\n    _validate_keyframe_conditions(conditions)\n"
            if validation else "if profile.task == MINIMAX_H3_TASK_FL2VA:\n    _validate_keyframe_conditions(conditions)\n")
        return self.root

    def test_complete_partition_and_full_shard_inventory(self):
        self.model()
        result = report()
        pre.check_model(self.root, result)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["weight_file_count"], 4)

    def test_reports_all_missing_shards_and_lfs_pointers(self):
        partition = self.model()
        (partition / "text_encoder/model-00001-of-00001.safetensors").unlink()
        (partition / "transformer/model-00001-of-00001.safetensors").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 9999\n")
        result = report()
        pre.check_model(self.root, result)
        self.assertEqual(len(result["errors"]), 2)
        self.assertTrue(any("Git LFS" in value for value in result["errors"]))
        self.assertIn("hf download", result["actions"][0])
        self.assertIn("Ref2VA/**", result["actions"][0])

    def test_rejects_fl2va_transformer_partition_and_direct_subdirectory(self):
        partition = self.model()
        json_file(partition / "model_index.json", {"_minimax_h3": {"partition": "fl2va"}})
        result = report()
        pre.check_model(partition, result)
        self.assertTrue(any("--model-path" in value for value in result["errors"]))
        self.assertTrue(any("partition='fl2va'" in value for value in result["errors"]))

    def test_truncated_tensor_rejected(self):
        path = self.root / "truncated.safetensors"
        tensor_file(path, b"\0")
        self.assertIn("长度", pre.check_safetensors(path))

    def test_native_loader_does_not_require_optional_vae_wrapper(self):
        partition = self.model()
        json_file(partition / "audio_vae/config.json", {
            "source_config_path": "missing-config.yaml",
            "source_metadata_path": "missing-metadata.json",
            "auto_map": {"AutoModel": "missing_wrapper.MiniMaxH3AudioVAE"},
        })
        result = report()
        pre.check_model(self.root, result)
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["warnings"]), 3)

    def test_server_requires_both_mp4_encoders(self):
        result = report()
        command = argparse.Namespace(returncode=0, stdout=" V....D libx264 H.264\n A..... aac AAC\n", stderr="")
        with mock.patch.object(pre.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                mock.patch.object(pre.subprocess, "run", return_value=command):
            pre.check_server_encoders(result)
        self.assertEqual(result["errors"], [])
        command.stdout = " A..... aac AAC\n"
        result = report()
        with mock.patch.object(pre.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                mock.patch.object(pre.subprocess, "run", return_value=command):
            pre.check_server_encoders(result)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("libx264", result["errors"][0])

    def test_current_hybrid_source_detected_without_importing_it(self):
        self.assertTrue(pre.inspect_hybrid_source(self.source())[0])

    def test_early_ref_profile_rejected_even_when_validation_mentions_ref2va(self):
        self.assertFalse(pre.inspect_hybrid_source(self.source(keyframe=False))[0])
        self.assertFalse(pre.inspect_hybrid_source(self.source(video=False))[0])
        self.assertFalse(pre.inspect_hybrid_source(self.source(validation=False))[0])

    def test_reused_server_only_checks_client_no_model_or_torch_loading(self):
        args = argparse.Namespace(model_path="/not-mounted", server_urls=["http://server:30010"],
                                  serve_only=False, start_servers=False)
        with mock.patch.object(pre.shutil, "which", return_value="/usr/bin/tool"), \
                mock.patch.object(pre, "check_model") as model, \
                mock.patch.object(pre, "check_runtime") as runtime:
            result = pre.preflight(args)
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "client")
        model.assert_not_called()
        runtime.assert_not_called()

    def test_missing_client_tools_are_aggregated(self):
        with mock.patch.object(pre.shutil, "which", return_value=None):
            result = pre.preflight(argparse.Namespace(client_only=True))
        self.assertEqual(len(result["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
