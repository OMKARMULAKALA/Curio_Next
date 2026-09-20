"""Verifies P0-4: training must fail fast with a clear message when no CUDA
GPU is available, rather than silently attempting (and hanging/crashing on)
CPU with a 7B-parameter model.
"""

from __future__ import annotations

import unittest

from src.utils.reproducibility import require_cuda, runtime_environment


class HardwareGateTests(unittest.TestCase):
    def test_runtime_environment_reports_cuda_availability(self) -> None:
        env = runtime_environment()
        self.assertIn("cuda_available", env)
        self.assertIn("torch", env)

    def test_require_cuda_raises_clearly_when_unavailable(self) -> None:
        # This test suite runs on a CPU-only machine (no CUDA GPU present),
        # so this exercises the real, unmocked code path.
        env = runtime_environment()
        if env.get("cuda_available"):
            self.skipTest("a CUDA GPU is actually available in this environment")
        with self.assertRaises(RuntimeError) as ctx:
            require_cuda()
        message = str(ctx.exception)
        self.assertIn("CUDA", message)
        self.assertIn("Qwen2-Audio", message)


if __name__ == "__main__":
    unittest.main()
