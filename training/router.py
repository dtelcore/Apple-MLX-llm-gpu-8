"""Python router: cabinet generate, then calc, then Wikipedia, then miss.

Never loads a second checkpoint. Search snippets are the answer; they are
not fed back into the GPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from tools.calc import try_calc
from training.cabinet_index import CabinetFact, CabinetIndex, remember
from training.chat_format import is_chat_model_name

SearchFn = Callable[[str], Optional[str]]

MISS_HINT = (
    "That is not in the local fact cabinet. For a short encyclopedia summary, "
    "ask a What/Who/Tell-me-about question (Wikipedia). For open English, start "
    "a story checkpoint: python interactive.py --checkpoint <story> --no-router"
)

_FACT_PREFIXES = (
    "what is",
    "what are",
    "what's",
    "whats",
    "who is",
    "who was",
    "who were",
    "who's",
    "where is",
    "where was",
    "when is",
    "when was",
    "tell me about",
    "how many",
    "how much",
)

# Strip these before retrying calc so "what is 1 + 1" is arithmetic, not Wikipedia.
_CALC_WRAPPERS = (
    "what is the value of",
    "what is",
    "what's",
    "whats",
    "calculate",
    "compute",
)
_CALC_OPS = set("+-*/^%")


@dataclass(frozen=True)
class RouteDecision:
    kind: str  # cabinet | calc | search | miss
    text: str
    fact: Optional[CabinetFact] = None
    detail: str = ""


def router_enabled(explicit: Optional[bool], model_name: str) -> bool:
    """Chat checkpoints default to the router; story checkpoints do not."""
    if explicit is not None:
        return bool(explicit)
    return is_chat_model_name(model_name)


def _bare_query(text: str) -> str:
    s = " ".join((text or "").split())
    if s.casefold().startswith("user:"):
        s = s.split(":", 1)[1].strip()
        s = " ".join(s.split())
    return s


def _calc_remainder(text: str) -> Optional[str]:
    s = _bare_query(text)
    low = s.casefold()
    for prefix in sorted(_CALC_WRAPPERS, key=len, reverse=True):
        if low.startswith(prefix):
            rest = s[len(prefix):].strip(" :?")
            rest = " ".join(rest.split())
            return rest or None
    return None


def try_calc_query(text: str) -> Optional[str]:
    """Calc on the raw text, or on a 'what is …' remainder that is an expression."""
    hit = try_calc(text)
    if hit is not None:
        return hit
    rest = _calc_remainder(text)
    if not rest:
        return None
    if not any(c.isdigit() for c in rest):
        return None
    if not any(c in _CALC_OPS for c in rest):
        return None
    return try_calc(rest)


def looks_like_fact_question(text: str) -> bool:
    s = _bare_query(text).casefold()
    if not s:
        return False
    return any(s.startswith(p) for p in _FACT_PREFIXES)


def search_topic(text: str) -> str:
    """Noun/topic for Wikipedia: drop 'what is' / 'tell me about' / 'a' / '?'."""
    s = _bare_query(text)
    if s.endswith("?"):
        s = s[:-1].rstrip()
        s = " ".join(s.split())
    low = s.casefold()
    for prefix in sorted(_FACT_PREFIXES, key=len, reverse=True):
        if low.startswith(prefix):
            s = s[len(prefix):].strip(" :?")
            s = " ".join(s.split())
            break
    parts = s.split()
    if parts and parts[0].casefold() in ("a", "an", "the"):
        s = " ".join(parts[1:])
    return s


def remember_search_hit(
    index: Optional[CabinetIndex],
    path,
    typed: str,
    extract: str,
) -> Optional[CabinetFact]:
    """Persist a Wikipedia extract under the typed question and its search topic."""
    if index is None or not extract:
        return None
    fact = remember(index, path, typed, extract)
    topic = search_topic(typed)
    if topic:
        remember(index, path, topic, extract)
    return fact


def route(
    text: str,
    index: Optional[CabinetIndex],
    *,
    search_enabled: bool = True,
    search_fn: Optional[SearchFn] = None,
) -> RouteDecision:
    """Cabinet first, then calc, then Wikipedia, then polite miss."""
    raw = text or ""
    if index is not None:
        fact = index.lookup(raw)
        if fact is None:
            topic = search_topic(raw)
            if topic:
                fact = index.lookup(topic)
        if fact is not None:
            learned = fact.source == "learned"
            return RouteDecision(
                kind="cabinet",
                text=fact.assistant if learned else fact.generate_prompt,
                fact=fact,
                detail="cabinet learned" if learned else "cabinet exact",
            )

    calc = try_calc_query(raw)
    if calc is not None:
        return RouteDecision(kind="calc", text=calc, detail="calc")

    if search_enabled and looks_like_fact_question(raw):
        topic = search_topic(raw)
        fn = search_fn
        extract = fn(topic) if fn is not None and topic else None
        if extract:
            return RouteDecision(kind="search", text=extract, detail="wikipedia")
        return RouteDecision(kind="miss", text=MISS_HINT, detail="search_failed")

    return RouteDecision(kind="miss", text=MISS_HINT, detail="no_route")
