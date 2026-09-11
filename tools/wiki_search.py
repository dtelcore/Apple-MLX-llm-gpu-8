"""Wikipedia OpenSearch + REST summary via stdlib urllib. No extra deps."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from version import __version__

OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = (
    f"Apple-MLX-llm-gpu-8/{__version__} "
    "(local cabinet router; https://github.com/dtelcore/Apple-MLX-llm-gpu-8)"
)
DEFAULT_TIMEOUT_S = 8.0


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[object]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _opensearch_title(query: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
    params = urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": query,
            "limit": "1",
            "namespace": "0",
            "format": "json",
        }
    )
    data = _get_json(f"{OPENSEARCH_URL}?{params}", timeout=timeout)
    if not isinstance(data, list) or len(data) < 2:
        return None
    titles = data[1]
    if not isinstance(titles, list) or not titles:
        return None
    title = titles[0]
    if not isinstance(title, str) or not title.strip():
        return None
    return title.strip()


def _summary_extract(title: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
    quoted = urllib.parse.quote(title.replace(" ", "_"), safe="")
    data = _get_json(SUMMARY_URL + quoted, timeout=timeout)
    if not isinstance(data, dict):
        return None
    extract = data.get("extract") or data.get("description")
    if not isinstance(extract, str):
        return None
    text = " ".join(extract.split())
    return text or None


def wiki_summary(query: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
    """Return a short Wikipedia extract, or None on any failure/empty result."""
    q = " ".join((query or "").split())
    if not q:
        return None
    title = _opensearch_title(q, timeout=timeout)
    if not title:
        return None
    return _summary_extract(title, timeout=timeout)
