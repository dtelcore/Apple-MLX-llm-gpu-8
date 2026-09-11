"""Normalized exact lookup for the chat fact cabinet."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import (
    CabinetIndex,
    load_cabinet,
    normalize_question,
)
from training.chat_format import USER_PREFIX


def _write_jsonl(path: Path, pairs) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            handle.write(
                json.dumps({"query": {"user": user}, "response": {"assistant": assistant}})
                + "\n"
            )


class NormalizeQuestionTests(unittest.TestCase):
    def test_strip_and_collapse_whitespace(self):
        self.assertEqual(
            normalize_question("  What   is the   capital of France?  "),
            normalize_question("What is the capital of France?"),
        )

    def test_casefold(self):
        self.assertEqual(
            normalize_question("WHAT IS THE CAPITAL OF FRANCE?"),
            normalize_question("what is the capital of france?"),
        )

    def test_optional_trailing_question_mark(self):
        self.assertEqual(
            normalize_question("What is the capital of France"),
            normalize_question("What is the capital of France?"),
        )

    def test_strip_leading_user_prefix(self):
        self.assertEqual(
            normalize_question("User: What is the capital of France?"),
            normalize_question("What is the capital of France?"),
        )
        self.assertEqual(
            normalize_question("user: What is the capital of France?"),
            normalize_question("What is the capital of France?"),
        )

    def test_strip_pasted_assistant_tail(self):
        self.assertEqual(
            normalize_question(
                "User: What is the capital of France? Assistant: The capital of France is Paris."
            ),
            normalize_question("What is the capital of France?"),
        )


class CabinetIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = CabinetIndex()
        self.index.add("What is the capital of France?", "The capital of France is Paris.")
        self.index.add("Tell me about Neonics.", "Neonics were banned in the EU.")
        self.index.add("What is 0 factorial?", "0 factorial equals 1.")

    def test_france_hit_without_question_mark(self):
        hit = self.index.lookup("what is the capital of france")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.user, "What is the capital of France?")
        self.assertEqual(hit.generate_prompt, "User: What is the capital of France? Assistant:")

    def test_typed_user_prefix_still_hits(self):
        hit = self.index.lookup("User: What is the capital of France?")
        self.assertIsNotNone(hit)
        self.assertTrue(hit.generate_prompt.startswith(USER_PREFIX))
        self.assertIn("Assistant:", hit.generate_prompt)

    def test_unobtanium_miss_is_not_fuzzy(self):
        self.assertIsNone(self.index.lookup("What is unobtanium"))
        self.assertIsNone(self.index.lookup("What is sequential layer streaming?"))

    def test_neonics_hit(self):
        hit = self.index.lookup("Tell me about Neonics.")
        self.assertIsNotNone(hit)
        self.assertIn("Neonics", hit.assistant)

    def test_conflicting_answers_not_indexed(self):
        idx = CabinetIndex()
        first = idx.add("What is the capital of Kashmir?", "Srinagar.")
        second = idx.add("What is the capital of Kashmir?", "Jammu.")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(idx.lookup("What is the capital of Kashmir?").assistant, "Srinagar.")

    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonl"
            _write_jsonl(
                path,
                [
                    ("What is the capital of France?", "The capital of France is Paris."),
                    ("What is the capital of France?", "The capital of France is Paris."),
                ],
            )
            loaded = load_cabinet(path)
            self.assertEqual(len(loaded), 1)
            self.assertIsNotNone(loaded.lookup("What is the capital of France?"))


class LiveMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = _ROOT / "data" / "chat_facts.jsonl"
        cls.mix = load_cabinet(path) if path.is_file() else None

    def test_live_mix_france_neonics_unobtanium(self):
        if self.mix is None or len(self.mix) == 0:
            self.skipTest("data/chat_facts.jsonl missing")
        self.assertIsNotNone(self.mix.lookup("What is the capital of France?"))
        self.assertIsNotNone(self.mix.lookup("Tell me about Neonics."))
        self.assertIsNone(self.mix.lookup("What is unobtanium"))


if __name__ == "__main__":
    unittest.main()
