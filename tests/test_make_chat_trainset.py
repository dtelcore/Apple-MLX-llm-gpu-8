"""Unit tests for tools/make_chat_trainset.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.make_chat_trainset import (
    make_user_query,
    qa_record,
    topic_from_fact,
    wrap_braces,
    wrap_native,
    write_corpus,
)
from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX


class MakeChatTrainsetTests(unittest.TestCase):
    def test_topic_strips_article(self):
        topic = topic_from_fact(
            "The Erie Canal stretched 363 miles across New York State."
        )
        self.assertTrue(topic.lower().startswith("erie"))
        self.assertNotIn("\n", topic)

    def test_native_line_is_user_then_assistant(self):
        user = make_user_query("Cats are small carnivorous mammals.", 0)
        line = wrap_native(user, "Cats are small carnivorous mammals.")
        self.assertTrue(line.startswith(USER_PREFIX))
        self.assertIn(ASSISTANT_PREFIX, line)
        self.assertNotIn("\n", line)
        self.assertIn("Cats are small", line)

    def test_braces_and_json_schema(self):
        user = "What is Salmonella?"
        assistant = "Salmonella is a genus of bacteria."
        braces = wrap_braces(user, assistant)
        self.assertIn("query{user}", braces)
        self.assertIn("response{assistant}", braces)
        rec = qa_record(user, assistant)
        self.assertEqual(rec["query"]["user"], user)
        self.assertEqual(rec["response"]["assistant"], assistant)

    def test_write_corpus_native_and_jsonl(self):
        facts = [
            "The moon orbits the Earth once every 27 days.",
            "Too short",
            "Honey bees pollinate many flowering plants in temperate regions.",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.txt"
            jsonl = Path(tmp) / "chat.jsonl"
            n_out, n_chat = write_corpus(
                [f for f in facts if len(f) >= 40],
                out,
                markers="native",
                jsonl_path=jsonl,
            )
            self.assertEqual(n_out, 2)
            self.assertEqual(n_chat, 2)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(line.startswith("User:") for line in lines))
            recs = [json.loads(row) for row in jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 2)
            self.assertIn("user", recs[0]["query"])
            self.assertIn("assistant", recs[0]["response"])


if __name__ == "__main__":
    unittest.main()
