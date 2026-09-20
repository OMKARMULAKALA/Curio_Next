from __future__ import annotations

import unittest

from src.utils.config import load_yaml


class ConfigLoadingTests(unittest.TestCase):
    def test_dataset_config_loads_and_has_required_keys(self) -> None:
        cfg = load_yaml("configs/dataset.yaml")
        for key in ("dataset_root", "train_manifest", "validation_manifest", "test_manifest", "schema"):
            self.assertIn(key, cfg)

    def test_training_config_preserves_100_epochs(self) -> None:
        cfg = load_yaml("configs/training.yaml")
        self.assertEqual(int(cfg["num_train_epochs"]), 100)
        self.assertIn("lora", cfg)
        self.assertIn("module_scope", cfg["lora"])

    def test_model_config_targets_qwen2_audio(self) -> None:
        cfg = load_yaml("configs/model.yaml")
        self.assertEqual(cfg["model_name"], "Qwen/Qwen2-Audio-7B-Instruct")
        self.assertEqual(int(cfg["sampling_rate"]), 16000)

    def test_non_mapping_yaml_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(bad)


if __name__ == "__main__":
    unittest.main()
