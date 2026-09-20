from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_TARGET_CANDIDATES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Qwen2AudioForConditionalGeneration composes two submodels:
#   - `audio_tower`: a Whisper-style encoder whose attention projections are
#     named q_proj/k_proj/v_proj/out_proj (verified against
#     transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioAttention).
#   - `language_model`: a Qwen2 decoder whose attention projections are named
#     q_proj/k_proj/v_proj/o_proj and MLP gate_proj/up_proj/down_proj
#     (verified against transformers.models.qwen2.modeling_qwen2).
# q_proj/k_proj/v_proj exist under those exact leaf names on BOTH submodels,
# so naive suffix matching (hand-rolled, or PEFT's own list-based matching)
# silently attaches LoRA to the audio tower's attention too. Matches are
# scoped to the language_model subtree by default; widening that requires an
# explicit opt-in via config (module_scope: all).
DEFAULT_MODULE_SCOPE_PREFIX = "language_model."


@dataclass
class ParameterReport:
    total_parameters: int
    trainable_parameters: int
    trainable_percent: float
    lora_parameters: int = 0


def _in_scope(module_name: str, scope_prefix: str | None) -> bool:
    if not scope_prefix:
        return True
    prefix = scope_prefix.rstrip(".")
    return module_name == prefix or module_name.startswith(scope_prefix)


def resolve_lora_target_modules(
    model,
    candidates: list[str] | None = None,
    scope_prefix: str | None = DEFAULT_MODULE_SCOPE_PREFIX,
) -> list[str]:
    """Resolve which candidate leaf module names actually exist in ``model``,
    restricted to submodules whose full dotted path starts with
    ``scope_prefix`` (default: the language-model decoder only).
    """
    candidates = candidates or DEFAULT_TARGET_CANDIDATES
    named = list(model.named_modules())
    in_scope_names = {name.split(".")[-1] for name, _ in named if _in_scope(name, scope_prefix)}
    resolved = [name for name in candidates if name in in_scope_names]
    if not resolved:
        all_names = sorted({name.split(".")[-1] for name, _ in named})
        raise ValueError(
            f"No LoRA target modules matched within scope_prefix={scope_prefix!r}. "
            f"Candidates={candidates}. Leaf module names actually present in the "
            f"loaded model: {all_names[:40]}{' ... (truncated)' if len(all_names) > 40 else ''}"
        )
    return resolved


def validate_target_modules(model, target_modules: list[str], scope_prefix: str | None) -> dict:
    """Validate explicitly configured target modules against the REAL loaded
    model. Raises ValueError with the exact reason if a configured target
    does not resolve to any in-scope submodule -- this is the check that
    must actually run whenever target_modules is non-empty, not only when it
    is omitted.
    """
    named = list(model.named_modules())
    report: dict[str, dict] = {}
    for target in target_modules:
        in_scope_matches = [name for name, _ in named if _in_scope(name, scope_prefix) and name.split(".")[-1] == target]
        out_of_scope_matches = [name for name, _ in named if not _in_scope(name, scope_prefix) and name.split(".")[-1] == target]
        report[target] = {
            "in_scope_matches": len(in_scope_matches),
            "out_of_scope_matches": len(out_of_scope_matches),
            "sample_in_scope": in_scope_matches[:3],
            "sample_out_of_scope": out_of_scope_matches[:3],
        }
        if not in_scope_matches and not out_of_scope_matches:
            raise ValueError(
                f"Configured LoRA target module '{target}' does not match any submodule "
                "of the loaded model at all. Inspect model.named_modules() before training."
            )
        if not in_scope_matches and out_of_scope_matches:
            raise ValueError(
                f"Configured LoRA target module '{target}' only matches modules OUTSIDE "
                f"scope_prefix={scope_prefix!r} (e.g. {out_of_scope_matches[:3]}). "
                "If you intend to adapt those modules too, set training.yaml "
                "lora.module_scope: all explicitly. Otherwise this is very likely a bug."
            )
    return report


def _scoped_regex(scope_prefix: str, target_modules: list[str]) -> str:
    alternation = "|".join(re.escape(t) for t in target_modules)
    return rf"^{re.escape(scope_prefix)}.*\.(?:{alternation})$"


def apply_lora(model, config: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    scope = config.get("module_scope", "language_model")
    scope_prefix = None if scope == "all" else config.get("module_scope_prefix", DEFAULT_MODULE_SCOPE_PREFIX)

    configured_targets = config.get("target_modules")
    if not configured_targets or configured_targets == "auto":
        target_modules = resolve_lora_target_modules(model, scope_prefix=scope_prefix)
    else:
        target_modules = list(configured_targets)
        validate_target_modules(model, target_modules, scope_prefix=scope_prefix)

    if scope_prefix:
        # Belt-and-braces beyond validate_target_modules: PEFT's own
        # list-based target_modules matching is suffix-only and has no
        # concept of scope, so a bare list would still reach into
        # audio_tower.*.{q,k,v}_proj. Passing PEFT a single regex string
        # (matched with re.fullmatch against the full dotted module path)
        # anchors the match to the scope prefix.
        peft_target_modules: list[str] | str = _scoped_regex(scope_prefix, target_modules)
    else:
        peft_target_modules = target_modules

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=peft_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def save_lora_adapters(model, output_dir: str) -> None:
    model.save_pretrained(output_dir)


def load_lora_adapters(model, adapter_dir: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir)


def parameter_report(model) -> ParameterReport:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    is_peft_model = hasattr(model, "peft_config")
    lora_params = trainable if is_peft_model else 0
    return ParameterReport(total, trainable, pct, lora_params)
