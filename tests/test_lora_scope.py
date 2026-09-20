"""Verifies the P0-2 fix: LoRA target-module resolution/validation must be
scoped to the language_model subtree and must actually reject a
misconfigured target, rather than silently matching audio_tower.*.

Uses a fake "model" object exposing named_modules() with the same submodule
names verified (by direct source inspection) in the installed transformers
4.45.2 Qwen2AudioForConditionalGeneration:
  - audio_tower.*.self_attn.{q_proj,k_proj,v_proj,out_proj}
  - language_model.*.self_attn.{q_proj,k_proj,v_proj,o_proj}
  - language_model.*.mlp.{gate_proj,up_proj,down_proj}
This avoids downloading the real 7B model just to test the scoping logic.
"""

from __future__ import annotations

import re
import unittest

from src.models.lora import (
    DEFAULT_MODULE_SCOPE_PREFIX,
    _scoped_regex,
    resolve_lora_target_modules,
    validate_target_modules,
)


class FakeModel:
    """Minimal stand-in exposing named_modules() like a real Qwen2Audio tree."""

    NAMES = [
        "",
        "audio_tower",
        "audio_tower.layers.0",
        "audio_tower.layers.0.self_attn",
        "audio_tower.layers.0.self_attn.q_proj",
        "audio_tower.layers.0.self_attn.k_proj",
        "audio_tower.layers.0.self_attn.v_proj",
        "audio_tower.layers.0.self_attn.out_proj",
        "multi_modal_projector",
        "multi_modal_projector.linear",
        "language_model",
        "language_model.model",
        "language_model.model.layers.0",
        "language_model.model.layers.0.self_attn",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.k_proj",
        "language_model.model.layers.0.self_attn.v_proj",
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.mlp",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
    ]

    def named_modules(self):
        return [(name, None) for name in self.NAMES]


class LoraScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()

    def test_default_scope_resolves_language_model_targets_only(self) -> None:
        resolved = resolve_lora_target_modules(self.model)
        self.assertEqual(
            set(resolved),
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        )

    def test_validate_accepts_correct_scoped_targets(self) -> None:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        report = validate_target_modules(self.model, targets, scope_prefix=DEFAULT_MODULE_SCOPE_PREFIX)
        for target in targets:
            self.assertGreaterEqual(report[target]["in_scope_matches"], 1)

    def test_validate_rejects_target_that_only_exists_in_audio_tower(self) -> None:
        # out_proj exists on audio_tower's attention but has no equivalent
        # under language_model (which uses o_proj) -- this must fail loudly,
        # not silently resolve to nothing or to the wrong submodel.
        with self.assertRaises(ValueError):
            validate_target_modules(self.model, ["out_proj"], scope_prefix=DEFAULT_MODULE_SCOPE_PREFIX)

    def test_validate_rejects_target_that_matches_nothing_at_all(self) -> None:
        with self.assertRaises(ValueError):
            validate_target_modules(self.model, ["totally_made_up_proj"], scope_prefix=DEFAULT_MODULE_SCOPE_PREFIX)

    def test_scoped_regex_matches_language_model_not_audio_tower(self) -> None:
        pattern = _scoped_regex(DEFAULT_MODULE_SCOPE_PREFIX, ["q_proj", "k_proj", "v_proj", "o_proj"])
        matched = [name for name, _ in self.model.named_modules() if re.fullmatch(pattern, name)]
        self.assertIn("language_model.model.layers.0.self_attn.q_proj", matched)
        self.assertIn("language_model.model.layers.0.self_attn.o_proj", matched)
        # The whole point of the fix: audio_tower's q/k/v_proj must NOT match.
        self.assertNotIn("audio_tower.layers.0.self_attn.q_proj", matched)
        self.assertNotIn("audio_tower.layers.0.self_attn.k_proj", matched)
        self.assertNotIn("audio_tower.layers.0.self_attn.v_proj", matched)

    def test_unscoped_all_mode_would_include_audio_tower(self) -> None:
        # Sanity check that resolve_lora_target_modules(scope_prefix=None)
        # (the explicit opt-in "all" mode) DOES see the audio tower, proving
        # the default scoping above is doing real work rather than the
        # candidates simply never existing on audio_tower.
        resolved = resolve_lora_target_modules(self.model, candidates=["out_proj"], scope_prefix=None)
        self.assertEqual(resolved, ["out_proj"])


if __name__ == "__main__":
    unittest.main()
