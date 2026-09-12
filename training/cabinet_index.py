"""Normalized exact lookup for the chat fact cabinet.

The index is a dictionary of unique User questions. Hits return the
*trained* ``User: … Assistant:`` generate prompt, not the REPL typing.
No fuzzy or substring matching.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Union

from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX, format_conversation

DEFAULT_FACTS_PATH = Path("data/chat_facts.jsonl")
RELATED_LIMIT = 5

_CAPITAL_USER = re.compile(r"^what is the capital of (.+)$")
_CAPITAL_ASST = re.compile(r"^the capital of (.+) is (.+)$")
_WHERE_CITY_ASST = re.compile(r"^(.+) is the capital of (.+)$")
_INVENTOR_USER = re.compile(r"^who invented (.+)$")
_INVENTOR_ASST = re.compile(r"^(.+) is credited with inventing (.+)$")
_ATOMIC_USER = re.compile(r"^what is the atomic number of (.+)$")
_ATOMIC_ASST = re.compile(r"^the atomic number of (.+) is (\d+)$")
_BIRTH_USER = re.compile(r"^when was (.+) born$")
_TELL_USER = re.compile(r"^tell me about (.+)$")
_POSSESS_USER = re.compile(r"^what is (.+)'s capital$")
_COUNTRY_OF_USER = re.compile(r"^what country is (.+) in$")
_WHERE_USER = re.compile(r"^where is (.+)$")


def _bare_entity(text: str) -> str:
    s = " ".join((text or "").split()).strip(" .")
    return s.casefold()


def extract_entities(user: str, assistant: str) -> FrozenSet[str]:
    """Closed-world slots from known trained templates. Empty if unrecognized."""
    u = normalize_question(user)
    a = _bare_entity(assistant)
    found = set()

    m = _CAPITAL_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _CAPITAL_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))
    m = _WHERE_CITY_ASST.match(a)
    if m and "capital of" in a:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _INVENTOR_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _INVENTOR_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _ATOMIC_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _ATOMIC_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _BIRTH_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _TELL_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _POSSESS_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _COUNTRY_OF_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _WHERE_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))

    return frozenset(e for e in found if e)


def template_keys_for(entity: str) -> List[str]:
    """Normalized questions that count as a known follow-up for one entity."""
    e = _bare_entity(entity)
    if not e:
        return []
    return [
        normalize_question(f"where is {e}"),
        normalize_question(f"what country is {e} in"),
        normalize_question(f"what is the capital of {e}"),
        normalize_question(f"what is {e}'s capital"),
        normalize_question(f"who invented {e}"),
        normalize_question(f"who is credited with inventing {e}"),
        normalize_question(f"what did {e} invent"),
        normalize_question(f"what is the atomic number of {e}"),
        normalize_question(f"which element has atomic number {e}"),
        normalize_question(f"tell me about {e}"),
        normalize_question(f"when was {e} born"),
        normalize_question(f"what is {e}"),
    ]


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


_JSONL_DECODER = json.JSONDecoder()


def _records_from_jsonl_line(line: str) -> Iterator[dict]:
    """Yield JSON objects on a line. Glued records (missing newline) still parse."""
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            return
        try:
            rec, end = _JSONL_DECODER.raw_decode(line, i)
        except json.JSONDecodeError:
            return
        if isinstance(rec, dict):
            yield rec
        i = end


def _pairs_from_jsonl(path: Path) -> Iterator[tuple]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            for rec in _records_from_jsonl_line(line):
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
        self._aliases: Dict[str, CabinetFact] = {}
        self._entity_facts: Dict[str, List[CabinetFact]] = defaultdict(list)
        self._fact_entities: Dict[str, FrozenSet[str]] = {}
        for fact in self._facts.values():
            if fact.source == "trained":
                self._index_entities(fact)

    def __len__(self) -> int:
        return len(self._facts)

    def lookup(self, text: str) -> Optional[CabinetFact]:
        key = normalize_question(text)
        if not key:
            return None
        return self._facts.get(key) or self._aliases.get(key)

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
        if source == "trained":
            self._index_entities(fact)
        return fact

    def add_alias(self, text: str, fact: CabinetFact) -> bool:
        """In-memory lookup alias. Not a second stored question."""
        key = normalize_question(text)
        if not key or key in self._facts or key in self._aliases:
            return False
        self._aliases[key] = fact
        return True

    def unique_facts(self) -> Iterator[CabinetFact]:
        yield from self._facts.values()

    def _index_entities(self, fact: CabinetFact) -> None:
        ents = extract_entities(fact.user, fact.assistant)
        self._fact_entities[fact.key] = ents
        for ent in ents:
            bucket = self._entity_facts[ent]
            if fact not in bucket:
                bucket.append(fact)

    def entities_of(self, fact: CabinetFact) -> FrozenSet[str]:
        return self._fact_entities.get(fact.key, frozenset())

    def known_entities(self) -> FrozenSet[str]:
        return frozenset(self._entity_facts)

    def entities_mentioned(self, text: str) -> FrozenSet[str]:
        """Entities from the closed set that appear as whole tokens in text."""
        key = normalize_question(text)
        if not key:
            return frozenset()
        tokens = f" {key} "
        found = set()
        for ent in self._entity_facts:
            pad = f" {ent} "
            if pad in tokens or key == ent:
                found.add(ent)
        return frozenset(found)

    def facts_for_entities(self, entities: Iterable[str]) -> List[CabinetFact]:
        seen: Dict[str, CabinetFact] = {}
        for ent in entities:
            for fact in self._entity_facts.get(_bare_entity(ent), []):
                if fact.source == "trained":
                    seen[fact.key] = fact
        return list(seen.values())

    def related_prompts(
        self,
        entities: Iterable[str],
        *,
        limit: int = RELATED_LIMIT,
        exclude_key: Optional[str] = None,
    ) -> List[str]:
        out: List[str] = []
        for fact in self.facts_for_entities(entities):
            if exclude_key and fact.key == exclude_key:
                continue
            out.append(fact.user)
            if len(out) >= limit:
                break
        return out

    def related_template_hits(self, text: str, extra_entities: Iterable[str] = ()) -> List[CabinetFact]:
        """Trained facts whose user is a known template for a mentioned/session entity."""
        typed = normalize_question(text)
        if not typed:
            return []
        ents = set(self.entities_mentioned(text))
        ents.update(_bare_entity(e) for e in extra_entities if e)
        seen: Dict[str, CabinetFact] = {}
        for ent in ents:
            if typed not in template_keys_for(ent):
                continue
            fact = self.lookup(text)
            if fact is not None and fact.source == "trained":
                seen[fact.key] = fact
        return list(seen.values())


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
    if dest.is_file() and dest.stat().st_size:
        with dest.open("rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    with dest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_jsonl_line(fact.user, fact.assistant) + "\n")
    return fact
