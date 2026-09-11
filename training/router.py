"""Python router: cabinet generate, then calc, then Wikipedia, then miss.

Never loads a second checkpoint. Search snippets are the answer; they are
not fed back into the GPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from tools.calc import try_calc
from training.cabinet_index import CabinetFact, CabinetIndex
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


def looks_like_fact_question(text: str) -> bool:
    s = " ".join((text or "").split()).casefold()
    if s.startswith("user:"):
        s = s.split(":", 1)[1].strip()
    if not s:
        return False
    return any(s.startswith(p) for p in _FACT_PREFIXES)


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
        if fact is not None:
            return RouteDecision(
                kind="cabinet",
                text=fact.generate_prompt,
                fact=fact,
                detail="cabinet exact",
            )

    calc = try_calc(raw)
    if calc is not None:
        return RouteDecision(kind="calc", text=calc, detail="calc")

    if search_enabled and looks_like_fact_question(raw):
        fn = search_fn
        extract = fn(raw) if fn is not None else None
        if extract:
            return RouteDecision(kind="search", text=extract, detail="wikipedia")
        return RouteDecision(kind="miss", text=MISS_HINT, detail="search_failed")

    return RouteDecision(kind="miss", text=MISS_HINT, detail="no_route")
