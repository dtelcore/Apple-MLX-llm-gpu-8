"""Router: cabinet wins over calc; search miss falls through."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import CabinetIndex
from training.router import MISS_HINT, looks_like_fact_question, route


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.index = CabinetIndex()
        self.index.add("What is the capital of France?", "The capital of France is Paris.")
        self.index.add("What is 0 factorial?", "0 factorial equals 1.")
        self.index.add("Tell me about Neonics.", "Neonics were banned in the EU.")

    def test_cabinet_france_uses_stored_prompt(self):
        d = route("what is the capital of france", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.text, "User: What is the capital of France? Assistant:")
        self.assertEqual(d.fact.assistant, "The capital of France is Paris.")

    def test_cabinet_wins_over_arithmetic_looking_question(self):
        d = route("What is 0 factorial?", self.index, search_fn=lambda q: "SHOULD NOT SEARCH")
        self.assertEqual(d.kind, "cabinet")
        self.assertIn("factorial", d.fact.assistant)

    def test_calc_two_plus_two(self):
        d = route("2+2", self.index, search_enabled=False)
        self.assertEqual(d.kind, "calc")
        self.assertEqual(d.text, "4")

    def test_unobtanium_search_success(self):
        d = route(
            "What is unobtanium",
            self.index,
            search_enabled=True,
            search_fn=lambda q: "A fictional metal.",
        )
        self.assertEqual(d.kind, "search")
        self.assertEqual(d.text, "A fictional metal.")

    def test_unobtanium_search_failure_is_miss(self):
        d = route(
            "What is unobtanium",
            self.index,
            search_enabled=True,
            search_fn=lambda q: None,
        )
        self.assertEqual(d.kind, "miss")
        self.assertEqual(d.text, MISS_HINT)

    def test_no_search_flag_skips_wikipedia(self):
        d = route("What is unobtanium", self.index, search_enabled=False)
        self.assertEqual(d.kind, "miss")

    def test_looks_like_fact_question(self):
        self.assertTrue(looks_like_fact_question("What is unobtanium"))
        self.assertTrue(looks_like_fact_question("User: Tell me about widget"))
        self.assertFalse(looks_like_fact_question("2+2"))
        self.assertFalse(looks_like_fact_question("once upon a time"))
