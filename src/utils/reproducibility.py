from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_environment() -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            env["gpu"] = {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
            }
        else:
            env["gpu"] = None
    except Exception as exc:
        env["cuda_probe_error"] = str(exc)
    return env


def require_cuda(min_vram_gb: float = 0.0) -> dict:
    """Fail fast with a clear message if no CUDA GPU is available, instead of
    letting training either crash deep inside a forward pass with an opaque
    error or silently (and uselessly) attempt to run a 7B-parameter model on
    CPU. Returns the runtime_environment() dict on success.
    """
    env = runtime_environment()
    if not env.get("cuda_available"):
        raise RuntimeError(
            "No CUDA-capable GPU was detected on this machine "
            f"(platform={env.get('platform')}, torch={env.get('torch')}). "
            "Qwen2-Audio-7B-Instruct LoRA fine-tuning requires a CUDA GPU "
            "(~24GB+ VRAM recommended). Refusing to start training on CPU. "
            "Provision a CUDA machine, or pass allow_cpu=True to this "
            "function only for non-training smoke tests that do not load "
            "the full model."
        )
    gpu = env.get("gpu") or {}
    total_memory_gb = gpu.get("total_memory_gb", 0.0)
    if min_vram_gb and total_memory_gb < min_vram_gb:
        raise RuntimeError(
            f"Detected GPU '{gpu.get('name')}' has {total_memory_gb:.1f}GB VRAM, "
            f"below the configured minimum of {min_vram_gb}GB for this experiment. "
            "Refusing to start training; lower the requirement explicitly if intentional."
        )
    return env


def experiment_record(
    experiment_config: dict,
    dataset_config: dict,
    model_config: dict,
    training_config: dict,
) -> dict:
    return {
        "experiment_id": experiment_config.get("experiment_id"),
        "runtime": runtime_environment(),
        "dataset": dataset_config,
        "model": model_config,
        "training": training_config,
    }

