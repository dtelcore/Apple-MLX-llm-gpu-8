"""Wikipedia helper: mocked urllib only."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import wiki_search


def _http_json(payload) -> mock.MagicMock:
    raw = json.dumps(payload).encode("utf-8")
    ctx = mock.MagicMock()
    ctx.read.return_value = raw
    ctx.__enter__.return_value = ctx
    ctx.__exit__.return_value = False
    return ctx


class WikiSearchTests(unittest.TestCase):
    def test_success_extract(self):
        open_search = ["unobtanium", ["Unobtainium"], ["desc"], ["http://example"]]
        summary = {"title": "Unobtainium", "extract": "A rare fictional metal."}

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("What is unobtanium")
        self.assertEqual(text, "A rare fictional metal.")

    def test_network_error_returns_none(self):
        with mock.patch(
            "tools.wiki_search.urllib.request.urlopen",
            side_effect=URLError("offline"),
        ):
            self.assertIsNone(wiki_search.wiki_summary("What is unobtanium"))

    def test_skips_disambiguation_for_next_title(self):
        open_search = ["python", ["Python", "Python (programming language)"], ["", ""], ["", ""]]
        dab = {"title": "Python", "type": "disambiguation", "extract": "Python may refer to:"}
        real = {
            "title": "Python (programming language)",
            "type": "standard",
            "extract": "Python is a high-level programming language.",
        }

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            if "Python_%28programming_language%29" in url:
                return _http_json(real)
            return _http_json(dab)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("python")
        self.assertEqual(text, "Python is a high-level programming language.")

    def test_empty_extract_returns_none(self):
        open_search = ["x", ["X"], [""], ["http://example"]]
        summary = {"title": "X", "extract": "   "}

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertIsNone(wiki_search.wiki_summary("X"))
