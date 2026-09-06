"""2 GB train-step preflight: refuse L=32 / 25M before allocating."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.mlx.env import PROCESS_BUDGET_BYTES, MemoryBudgetError
from training.memory_preflight import assert_train_fits_budget, estimate_train_step_bytes


class TrainMemoryPreflightTests(unittest.TestCase):
    def test_story_sub1m_fits(self):
        estimated = estimate_train_step_bytes(
            n_params=830_000,
            batch_size=8,
            max_len=128,
            embedding_dim=128,
            num_heads=8,
            num_layers=4,
            grad_accum=2,
        )
        self.assertLess(estimated, PROCESS_BUDGET_BYTES)
        assert_train_fits_budget(
            n_params=830_000,
            batch_size=8,
            max_len=128,
            embedding_dim=128,
            num_heads=8,
            num_layers=4,
            grad_accum=2,
        )

    def test_c256_l4_t256_batch4_fits(self):
        estimated = estimate_train_step_bytes(
            n_params=3_900_000,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=4,
            grad_accum=4,
        )
        self.assertLess(estimated, PROCESS_BUDGET_BYTES)

    def test_l32_c256_t256_batch8_refused(self):
        with self.assertRaises(MemoryBudgetError) as ctx:
            assert_train_fits_budget(
                n_params=25_362_847,
                batch_size=8,
                max_len=256,
                embedding_dim=256,
                num_heads=16,
                num_layers=32,
                grad_accum=2,
            )
        msg = str(ctx.exception)
        self.assertIn("L=32", msg)
        self.assertIn("2 GB", msg)


if __name__ == "__main__":
    unittest.main()
