"""Normalized exact lookup for the chat fact cabinet.

The index is a dictionary of unique User questions. Hits return the
*trained* ``User: … Assistant:`` generate prompt, not the REPL typing.
No fuzzy or substring matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX, format_conversation

DEFAULT_FACTS_PATH = Path("data/chat_facts.jsonl")


@dataclass(frozen=True)
class CabinetFact:
    """One unique cabinet Q&A plus the canonical generate prompt."""

    user: str
    assistant: str
    generate_prompt: str
    key: str
    source: str = "trained"


def normalize_question(text: str) -> str:
    """Canonical key for exact cabinet lookup.

    - Strip leading/trailing whitespace and collapse internal whitespace
    - Strip a leading ``User:`` if the user typed it
    - Drop a trailing ``Assistant: …`` paste if present
    - Drop an optional trailing ``?``
    - Casefold
    """
    s = " ".join((text or "").split())
    if not s:
        return ""
    lower = s.casefold()
    if lower.startswith("user:"):
        s = s.split(":", 1)[1].strip()
        s = " ".join(s.split())
        lower = s.casefold()
    marker = ASSISTANT_PREFIX.rstrip().casefold()
    idx = lower.find(marker)
    if idx >= 0:
        s = s[:idx].rstrip()
        s = " ".join(s.split())
    if s.endswith("?"):
        s = s[:-1].rstrip()
        s = " ".join(s.split())
    return s.casefold()


def generate_prompt_for(user: str) -> str:
    """Stored generate string: ``User: {trained question} Assistant:``."""
    return format_conversation([], pending_user=user, open_assistant=True)


def _pairs_from_jsonl(path: Path) -> Iterator[tuple]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            user = (rec.get("query") or {}).get("user") or rec.get("user")
            assistant = (rec.get("response") or {}).get("assistant") or rec.get("assistant")
            if user and assistant:
                yield str(user).strip(), str(assistant).strip()


def _pairs_from_txt(path: Path) -> Iterator[tuple]:
    prefix_u = USER_PREFIX.rstrip()
    prefix_a = ASSISTANT_PREFIX.rstrip()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = " ".join(raw.split())
            if prefix_u not in text or prefix_a not in text:
                continue
            after = text.split(USER_PREFIX, 1)[-1] if USER_PREFIX in text else text.split("User:", 1)[-1]
            if ASSISTANT_PREFIX not in after and "Assistant:" not in after:
                continue
            if ASSISTANT_PREFIX in after:
                user, assistant = after.split(ASSISTANT_PREFIX, 1)
            else:
                user, assistant = after.split("Assistant:", 1)
            user, assistant = user.strip(), assistant.strip()
            if user and assistant:
                yield user, assistant


class CabinetIndex:
    """Exact-normalized map of trained questions → CabinetFact."""

    def __init__(self, facts: Optional[Dict[str, CabinetFact]] = None):
        self._facts: Dict[str, CabinetFact] = dict(facts or {})

    def __len__(self) -> int:
        return len(self._facts)

    def lookup(self, text: str) -> Optional[CabinetFact]:
        key = normalize_question(text)
        if not key:
            return None
        return self._facts.get(key)

    def add(self, user: str, assistant: str, *, source: str = "trained") -> Optional[CabinetFact]:
        key = normalize_question(user)
        if not key:
            return None
        existing = self._facts.get(key)
        if existing is not None:
            if existing.assistant != assistant.strip():
                return None
            return existing
        fact = CabinetFact(
            user=user.strip(),
            assistant=assistant.strip(),
            generate_prompt=generate_prompt_for(user.strip()),
            key=key,
            source=source,
        )
        self._facts[key] = fact
        return fact


def _pairs_from_path(path: Path) -> Iterator[tuple]:
    if path.suffix.lower() == ".jsonl":
        return _pairs_from_jsonl(path)
    return _pairs_from_txt(path)


def _jsonl_line(user: str, assistant: str) -> str:
    return json.dumps(
        {"query": {"user": user}, "response": {"assistant": assistant}},
        ensure_ascii=False,
    )


def load_cabinet(path: Union[str, Path], *, source: str = "trained") -> CabinetIndex:
    """Load unique facts from JSONL (or native ``User:/Assistant:`` txt)."""
    src = Path(path)
    index = CabinetIndex()
    if not src.is_file():
        raise FileNotFoundError(f"Cabinet facts not found: {src}")
    for user, assistant in _pairs_from_path(src):
        index.add(user, assistant, source=source)
    return index


def merge_cabinet(
    index: CabinetIndex,
    path: Union[str, Path],
    *,
    source: str = "learned",
) -> int:
    """Load extra facts into an existing index. Returns how many keys were new."""
    src = Path(path)
    if not src.is_file():
        return 0
    before = len(index)
    for user, assistant in _pairs_from_path(src):
        index.add(user, assistant, source=source)
    return len(index) - before


def remember(
    index: CabinetIndex,
    path: Union[str, Path],
    user: str,
    assistant: str,
    *,
    source: str = "learned",
) -> Optional[CabinetFact]:
    """Index a new Q&A and append JSONL. Duplicates and conflicts do not write."""
    if not (user or "").strip() or not (assistant or "").strip():
        return None
    existing = index.lookup(user)
    if existing is not None:
        return existing
    fact = index.add(user, assistant, source=source)
    if fact is None:
        return None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_jsonl_line(fact.user, fact.assistant) + "\n")
    return fact
