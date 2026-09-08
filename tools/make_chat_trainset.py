"""
tools/make_chat_trainset.py

Turn one-fact-per-line prose (wiki ``data/train.txt``) into chat documents.

Training lines (default ``native``) match ``training/chat_format.py``:

    User: Tell me about the Erie Canal. Assistant: It stretched 363 miles ...

JSONL (``--jsonl``) uses the query{user} / response{assistant} schema:

    {"query": {"user": "..."}, "response": {"assistant": "..."}}

``--markers braces`` writes the same pair as a single text line:

    query{user} ... response{assistant} ...

Prefer ``native`` for a fine-tune of run8+16: interactive.py and quality
probes look for ``User:`` / ``Assistant:``.

Usage:
    python tools/make_chat_trainset.py
    python tools/make_chat_trainset.py --input data/train.txt --output data/chat_train.txt --jsonl data/chat_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.chat_format import ASSISTANT_ROLE, USER_ROLE, format_conversation

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_LEAD_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_QUESTION_TEMPLATES = (
    "What is {topic}?",
    "Tell me about {topic}.",
    "Can you explain {topic}?",
    "What do we know about {topic}?",
    "Summarize {topic}.",
    "Give a short answer about {topic}.",
)


def topic_from_fact(text: str) -> str:
    """Short noun-ish span from the start of a fact line."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "this"
    head = re.split(r"[.;:!?]", cleaned, maxsplit=1)[0].strip()
    words = head.split()
    n = min(8, max(3, min(len(words), 6)))
    topic = " ".join(words[:n]).rstrip(" ,;:-")
    topic = _LEAD_ARTICLE.sub("", topic).strip() or "this"
    if len(topic) > 80:
        topic = topic[:80].rsplit(" ", 1)[0] or topic[:80]
    return topic


def make_user_query(fact: str, template_index: int = 0) -> str:
    topic = topic_from_fact(fact)
    return _QUESTION_TEMPLATES[template_index % len(_QUESTION_TEMPLATES)].format(topic=topic)


def wrap_native(user: str, assistant: str, *, system: Optional[str] = None) -> str:
    """One training document: User query + Assistant response."""
    return format_conversation(
        [(USER_ROLE, user), (ASSISTANT_ROLE, assistant)],
        system=system,
        open_assistant=False,
    )


def wrap_braces(user: str, assistant: str, *, system: Optional[str] = None) -> str:
    """Literal query{user} / response{assistant} markers on one line."""
    sys_part = ""
    if system:
        sys_part = " ".join(system.split()) + " "
    u = " ".join(user.split())
    a = " ".join(assistant.split())
    return f"{sys_part}query{{user}} {u} response{{assistant}} {a}".strip()


def qa_record(user: str, assistant: str) -> dict:
    return {"query": {"user": user}, "response": {"assistant": assistant}}


def iter_source_lines(path: Path, *, min_chars: int, max_docs: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = " ".join(raw.split())
            if len(line) < min_chars:
                continue
            yield line
            n += 1
            if max_docs is not None and n >= int(max_docs):
                return


def wrap_line(
    fact: str,
    index: int,
    *,
    markers: str,
    system: Optional[str],
) -> Tuple[str, str, str]:
    user = make_user_query(fact, template_index=index)
    assistant = fact
    if markers == "braces":
        text = wrap_braces(user, assistant, system=system)
    else:
        text = wrap_native(user, assistant, system=system)
    return text, user, assistant


def write_corpus(
    facts: Iterable[str],
    out_path: Path,
    *,
    markers: str = "native",
    system: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
    keep_raw: float = 0.0,
    seed: int = 42,
) -> Tuple[int, int]:
    import random

    keep_raw = min(1.0, max(0.0, float(keep_raw)))
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = open(jsonl_path, "w", encoding="utf-8")
    n_out = 0
    n_chat = 0
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            for i, fact in enumerate(facts):
                if keep_raw > 0.0 and rng.random() < keep_raw:
                    handle.write(fact + "\n")
                    n_out += 1
                    continue
                text, user, assistant = wrap_line(
                    fact, i, markers=markers, system=system,
                )
                handle.write(text + "\n")
                n_out += 1
                n_chat += 1
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(qa_record(user, assistant), ensure_ascii=False) + "\n")
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
    return n_out, n_chat


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a User/Assistant (query/response) chat train set from fact lines",
    )
    parser.add_argument(
        "--input", type=str, default=str(ROOT / "data" / "train.txt"),
        help="Source facts (one document per line). Default: data/train.txt",
    )
    parser.add_argument(
        "--output", type=str, default=str(ROOT / "data" / "chat_train.txt"),
        help="Destination chat corpus (one document per line)",
    )
    parser.add_argument(
        "--jsonl", type=str, default=None,
        help="Optional JSONL with query.user / response.assistant objects",
    )
    parser.add_argument(
        "--markers", choices=("native", "braces"), default="native",
        help="native = User:/Assistant: (train this). braces = query{user} ... response{assistant} ...",
    )
    parser.add_argument(
        "--system", type=str, default="",
        help="Optional system prefix on each native/braces line",
    )
    parser.add_argument(
        "--min-chars", type=int, default=40,
        help="Skip source lines shorter than this (default 40)",
    )
    parser.add_argument(
        "--keep-raw", type=float, default=0.0,
        help="Fraction of source lines kept as plain prose (default 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Optional cap on kept source lines",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        return 1
    system = args.system.strip() or None
    facts = iter_source_lines(src, min_chars=int(args.min_chars), max_docs=args.max_docs)
    jsonl = Path(args.jsonl) if args.jsonl else None
    n_out, n_chat = write_corpus(
        facts,
        Path(args.output),
        markers=args.markers,
        system=system,
        jsonl_path=jsonl,
        keep_raw=args.keep_raw,
        seed=args.seed,
    )
    extra = f", jsonl={args.jsonl}" if jsonl else ""
    print(
        f"Wrote {n_out:,} documents ({n_chat:,} chat) to {args.output} "
        f"markers={args.markers}{extra}"
    )
    if args.markers == "braces":
        print(
            "Note: braces lines are not the live train format. "
            "Use --markers native to fine-tune run8+16.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
