from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def audio_info(path: str | Path) -> dict:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "samplerate": info.samplerate,
        "frames": info.frames,
        "duration_sec": info.frames / info.samplerate if info.samplerate else None,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def load_audio(path: str | Path):
    import librosa

    return librosa.load(path, sr=None, mono=True)


def load_audio_for_model(
    path: str | Path,
    target_sampling_rate: int,
    mono: bool = True,
    max_seconds: float | None = None,
) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=target_sampling_rate, mono=mono)
    if max_seconds is not None and max_seconds > 0:
        max_samples = int(target_sampling_rate * max_seconds)
        audio = audio[:max_samples]
    if audio.size == 0:
        raise ValueError(f"Audio file decoded to zero samples: {path}")
    return audio.astype(np.float32, copy=False)
