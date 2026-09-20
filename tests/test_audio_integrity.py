"""Verifies P0-6: the real audio stack (soundfile/librosa) must decode both
plain PCM WAV and the MPEG/MP3-as-.wav files found in the test split, using
actual project data. Skipped automatically if dataset/ (gitignored, not
part of the git repo) is not present in the environment running the tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data.audio_integrity import check_audio_file, sniff_container
from src.utils.paths import project_root

DATASET_ROOT = project_root() / "dataset" / "dcase_part1"
_HAS_DATASET = DATASET_ROOT.exists()


@unittest.skipUnless(_HAS_DATASET, "dataset/dcase_part1 is not present in this environment (gitignored raw data)")
class RealAudioDecodeTests(unittest.TestCase):
    def test_plain_wav_from_train_decodes(self) -> None:
        candidates = list((DATASET_ROOT / "audio" / "train").glob("fold1-a-*.wav"))
        self.assertTrue(candidates, "expected at least one fold1-a-*.wav under audio/train")
        result = check_audio_file(candidates[0], decode_check=True)
        self.assertTrue(result.decodable)
        self.assertFalse(result.extension_mismatch)
        self.assertEqual(result.container, "riff")

    def test_mp3_labeled_as_wav_still_decodes(self) -> None:
        target = DATASET_ROOT / "audio" / "dev" / "audio_00001.wav"
        if not target.exists():
            self.skipTest("audio_00001.wav not present")
        result = check_audio_file(target, decode_check=True)
        self.assertTrue(result.decodable, "requires soundfile>=0.12.1 / libsndfile>=1.2.0 per requirements.txt")
        self.assertTrue(result.extension_mismatch)
        self.assertEqual(result.container, "mp3_frame_sync")

    def test_missing_file_reports_not_decodable(self) -> None:
        result = check_audio_file(DATASET_ROOT / "audio" / "train" / "does_not_exist.wav")
        self.assertFalse(result.exists)
        self.assertFalse(result.decodable)


class ContainerSniffTests(unittest.TestCase):
    def test_sniff_recognizes_mp3_frame_sync_header(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"\xff\xfb\x94\x64" + b"\x00" * 32)
            path = Path(handle.name)
        try:
            self.assertEqual(sniff_container(path), "mp3_frame_sync")
        finally:
            path.unlink(missing_ok=True)

    def test_sniff_recognizes_riff_header(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"RIFF" + b"\x00" * 32)
            path = Path(handle.name)
        try:
            self.assertEqual(sniff_container(path), "riff")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
