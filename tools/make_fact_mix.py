"""
tools/make_fact_mix.py

Build a tiny repeated User/Assistant corpus for weight-side memorization.

Writes:
  data/chat_facts.txt                (train this)
  data/chat_facts.jsonl              (same pairs, query/response)

Also loads data/user_facts.txt and data/facts/*.txt (e.g. Wikidata export).
Does not concatenate data/train.txt or the full chat_train dump.
Same User: question with two different answers is dropped entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_chat_trainset import qa_record, topic_from_fact, wrap_native
from tools.wikidata_to_facts import drop_conflicts
from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX

_NEONICS_ASSISTANT = (
    "This has led to the recent banning of Neonics in the EU, however the US "
    "and Canada are still using this chemical pesticide."
)
_NEONICS_USER = "Tell me about Neonics."


def _split_native(line: str) -> Optional[Tuple[str, str]]:
    text = " ".join((line or "").split())
    if USER_PREFIX not in text or ASSISTANT_PREFIX not in text:
        return None
    after_user = text.split(USER_PREFIX, 1)[1]
    if ASSISTANT_PREFIX not in after_user:
        return None
    user, assistant = after_user.split(ASSISTANT_PREFIX, 1)
    user = user.strip()
    assistant = assistant.strip()
    if not user or not assistant:
        return None
    return user, assistant


def load_user_facts(paths: Sequence[Path]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                pair = _split_native(stripped)
                if pair:
                    out.append(pair)
    return drop_conflicts(out)


def load_wiki_core(chat_train: Path, *, max_facts: int) -> List[Tuple[str, str]]:
    seen = set()
    core: List[Tuple[str, str]] = []
    if not chat_train.is_file():
        return core
    with chat_train.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            pair = _split_native(raw)
            if not pair:
                continue
            user, assistant = pair
            if not user.lower().startswith("tell me about"):
                continue
            topic = topic_from_fact(assistant) or user
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            core.append((f"Tell me about {topic}.", assistant))
            if len(core) >= max_facts:
                break
    return core


def neonics_pair() -> Tuple[str, str]:
    return _NEONICS_USER, _NEONICS_ASSISTANT


def build_pairs(
    *,
    user_facts: Sequence[Tuple[str, str]],
    wiki_core: Sequence[Tuple[str, str]],
    user_repeat: int,
    wiki_repeat: int,
) -> List[Tuple[str, str]]:
    seed = [neonics_pair()]
    # Dedupe neonics if user already supplied it.
    extra_user = [p for p in user_facts if p[0].strip().lower() != _NEONICS_USER.lower()]
    wiki = [p for p in wiki_core if "neonics" not in p[0].lower()]
    out: List[Tuple[str, str]] = []
    for _ in range(max(1, int(user_repeat))):
        out.extend(seed)
        out.extend(extra_user)
    for _ in range(max(1, int(wiki_repeat))):
        out.extend(wiki)
    return out


def write_outputs(pairs: Iterable[Tuple[str, str]], txt_path: Path, jsonl_path: Path) -> int:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with txt_path.open("w", encoding="utf-8", newline="\n") as txt, \
            jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl:
        for user, assistant in pairs:
            txt.write(wrap_native(user, assistant) + "\n")
            jsonl.write(json.dumps(qa_record(user, assistant), ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repeated chat fact-overfit corpus")
    parser.add_argument("--chat-train", type=str, default=str(ROOT / "data" / "chat_train.txt"))
    parser.add_argument("--user-facts", type=str, default=str(ROOT / "data" / "user_facts.txt"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data" / "chat_facts.txt"))
    parser.add_argument(
        "--jsonl", type=str, default=str(ROOT / "data" / "chat_facts.jsonl"),
    )
    parser.add_argument("--max-wiki-facts", type=int, default=0)
    parser.add_argument("--user-repeat", type=int, default=300)
    parser.add_argument("--wiki-repeat", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    user = load_user_facts([Path(args.user_facts), *(Path(ROOT / "data" / "facts").glob("*.txt") if (ROOT / "data" / "facts").is_dir() else [])])
    wiki = load_wiki_core(Path(args.chat_train), max_facts=int(args.max_wiki_facts))
    pairs = build_pairs(
        user_facts=user,
        wiki_core=wiki,
        user_repeat=int(args.user_repeat),
        wiki_repeat=int(args.wiki_repeat),
    )
    n = write_outputs(pairs, Path(args.output), Path(args.jsonl))
    print(
        f"Wrote {n:,} documents to {args.output} "
        f"(user_facts={len(user)} wiki_core={len(wiki)} "
        f"user_repeat={args.user_repeat} wiki_repeat={args.wiki_repeat})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
