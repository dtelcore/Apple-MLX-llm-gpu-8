"""Tests for fact-overfit mix and explicit dataset path loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from setup.config_loader import resolve_dataset_corpus
from tools.make_fact_mix import (
    _NEONICS_USER,
    build_pairs,
    load_user_facts,
    neonics_pair,
    write_outputs,
)


class FactMixTests(unittest.TestCase):
    def test_load_user_facts_skips_hash_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_facts.txt"
            path.write_text(
                "# ignore this User: Tell me about X. Assistant: skip\n"
                "User: What is the current version? Assistant: Version 0.0.5.\n",
                encoding="utf-8",
            )
            pairs = load_user_facts([path])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "What is the current version?")

    def test_neonics_repeated_and_written(self):
        pairs = build_pairs(user_facts=[], wiki_core=[], user_repeat=3, wiki_repeat=1)
        self.assertGreaterEqual(len(pairs), 3)
        self.assertTrue(all(p[0] == _NEONICS_USER for p in pairs[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / "fact_overfit.txt"
            jsonl = Path(tmp) / "fact_overfit_train.jsonl"
            n = write_outputs(pairs, txt, jsonl)
            self.assertEqual(n, len(pairs))
            text = txt.read_text(encoding="utf-8")
            self.assertIn("Tell me about Neonics.", text)
            self.assertIn("banning of Neonics", text)
            rec = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["query"]["user"], _NEONICS_USER)

    def test_resolve_path_jsonl_not_whole_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "fact_overfit_train.jsonl"
            user, assistant = neonics_pair()
            jsonl.write_text(
                json.dumps({"query": {"user": user}, "response": {"assistant": assistant}}) + "\n",
                encoding="utf-8",
            )
            (Path(tmp) / "train.txt").write_text("this should not be loaded\n" * 5, encoding="utf-8")
            corpus = resolve_dataset_corpus(
                {"name": "ignored", "combine": True, "path": str(jsonl)},
                data_dir=tmp,
            )
            self.assertEqual(len(corpus), 1)
            self.assertIn("Neonics", corpus[0])
            self.assertNotIn("this should not be loaded", corpus[0])
            alias = resolve_dataset_corpus(
                {"name": "ignored", "combine": True, "dataset_path": str(jsonl)},
                data_dir=tmp,
            )
            self.assertEqual(alias, corpus)


if __name__ == "__main__":
    unittest.main()
