from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import resolve_project_path


@dataclass
class AudioIntegrityResult:
    path: str
    exists: bool
    decodable: bool
    container: str | None = None
    format: str | None = None
    subtype: str | None = None
    samplerate: int | None = None
    channels: int | None = None
    duration_sec: float | None = None
    extension_mismatch: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def sniff_container(path: Path) -> str:
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head[:3] == b"ID3":
        return "mp3_id3"
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3_frame_sync"
    if head[:4] == b"RIFF":
        return "riff"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    return f"unknown:{head.hex()}"


def check_audio_file(path: str | Path, decode_check: bool = True, decode_sr: int = 16000) -> AudioIntegrityResult:
    """Validate one audio file using the actual runtime audio stack
    (soundfile/libsndfile, with an optional full librosa decode), not
    Python's stdlib `wave` module. `wave` only understands plain integer-PCM
    RIFF/WAVE and misreports both float-PCM WAV (valid, just unsupported by
    `wave`) and MPEG/MP3 data saved with a .wav extension (which libsndfile
    >= 1.2.0, used by soundfile >= 0.12, decodes natively) as "corrupted".
    """
    import soundfile as sf

    resolved = resolve_project_path(path)
    if not resolved.exists():
        return AudioIntegrityResult(path=str(path), exists=False, decodable=False, error="file does not exist")

    container = sniff_container(resolved)
    extension_mismatch = container in ("mp3_id3", "mp3_frame_sync") and resolved.suffix.lower() == ".wav"

    try:
        info = sf.info(str(resolved))
    except Exception as exc:
        return AudioIntegrityResult(
            path=str(path),
            exists=True,
            decodable=False,
            container=container,
            extension_mismatch=extension_mismatch,
            error=f"{type(exc).__name__}: {exc}",
        )

    result = AudioIntegrityResult(
        path=str(path),
        exists=True,
        decodable=True,
        container=container,
        format=info.format,
        subtype=info.subtype,
        samplerate=info.samplerate,
        channels=info.channels,
        duration_sec=(info.frames / info.samplerate) if info.samplerate else None,
        extension_mismatch=extension_mismatch,
    )

    if decode_check:
        try:
            import librosa

            audio, _ = librosa.load(str(resolved), sr=decode_sr, mono=True)
            if audio.size == 0:
                raise ValueError("decoded to zero samples")
        except Exception as exc:
            result.decodable = False
            result.error = f"librosa decode failed: {type(exc).__name__}: {exc}"

    return result
