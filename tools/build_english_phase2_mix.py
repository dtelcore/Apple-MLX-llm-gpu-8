#!/usr/bin/env python3
"""Host-only Phase 2 mix: unique v9 facts, four distinct frames, interleaved into train.txt.

Does not touch Metal. Zero identical fact-frame duplication (no 300× loops).
The prose file stays the architectural anchor; facts are a light inject.

Usage:
  python tools/build_english_phase2_mix.py
  python tools/build_english_phase2_mix.py --v9 data/chat_facts_v9.jsonl \\
      --train data/train.txt --output data/english_phase2_mix.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.cabinet_index import _pairs_from_jsonl
from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX

DEFAULT_V9 = ROOT / "data" / "chat_facts_v9.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "train.txt"
DEFAULT_OUT = ROOT / "data" / "english_phase2_mix.txt"


def unique_pairs_from_jsonl(path: Path) -> Dict[str, str]:
    """Last assistant wins per user. v9 is 300×; this collapses to unique keys."""
    pairs: Dict[str, str] = {}
    for user, assistant in _pairs_from_jsonl(path):
        user = " ".join((user or "").split())
        assistant = " ".join((assistant or "").split())
        if user and assistant:
            pairs[user] = assistant
    return pairs


def _bare(text: str, prefix: str) -> str:
    s = " ".join((text or "").split())
    if s.casefold().startswith(prefix.casefold()):
        s = s[len(prefix) :].strip()
    return s


def generate_frames(user: str, assistant: str) -> List[str]:
    """Four distinct linguistic mappings. dict.fromkeys drops accidental collisions."""
    user = " ".join((user or "").split())
    assistant = " ".join((assistant or "").split())
    clean_u = _bare(user, USER_PREFIX).replace("?", "").strip()
    clean_a = _bare(assistant, ASSISTANT_PREFIX).strip()
    frames = [
        f"{USER_PREFIX}{user} {ASSISTANT_PREFIX}{assistant}".rstrip(),
        f"It is known that {clean_u} is resolved by {clean_a}.",
        f"Regarding {clean_u}, the corresponding detail is {clean_a}.",
        f"Question: {clean_u}? Response: {clean_a}.",
    ]
    return list(dict.fromkeys(frame for frame in frames if frame.strip()))


def all_fact_frames(pairs: Dict[str, str]) -> List[str]:
    frames: List[str] = []
    for user, assistant in pairs.items():
        frames.extend(generate_frames(user, assistant))
    return frames


def max_identical_count(lines: Sequence[str]) -> int:
    if not lines:
        return 0
    return max(Counter(lines).values())


def interleave_stride(prose_lines: Sequence[str], fact_frames: Sequence[str]) -> List[str]:
    """Insert one fact frame on a uniform stride so facts are not a prefix clump."""
    total_prose = len(prose_lines)
    total_facts = len(fact_frames)
    if total_facts == 0:
        return list(prose_lines)
    stride = max(1, total_prose // total_facts) if total_prose else 1
    mixed: List[str] = []
    fact_idx = 0
    for i, line in enumerate(prose_lines):
        mixed.append(line)
        if i % stride == 0 and fact_idx < total_facts:
            mixed.append(fact_frames[fact_idx])
            fact_idx += 1
    while fact_idx < total_facts:
        mixed.append(fact_frames[fact_idx])
        fact_idx += 1
    return mixed


def load_prose_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def write_mix(path: Path, lines: Iterable[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
            n += 1
    return n


def build_mix(
    *,
    pairs: Dict[str, str],
    prose_lines: Sequence[str],
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    frames = all_fact_frames(pairs)
    dup = max_identical_count(frames)
    if dup > 1:
        raise ValueError(
            f"Fact frames are not unique (max identical count={dup}). "
            "Refuse to write a recitation pack."
        )
    rng = __import__("random").Random(int(seed))
    shuffled = list(frames)
    rng.shuffle(shuffled)
    mixed = interleave_stride(prose_lines, shuffled)
    return mixed, frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 2 English inject mix (host-only)")
    parser.add_argument("--v9", type=str, default=str(DEFAULT_V9), help="v9 cabinet JSONL")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN), help="Phase 1 prose anchor")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT), help="Interleaved mix path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v9_path = Path(args.v9)
    train_path = Path(args.train)
    out_path = Path(args.output)
    if not v9_path.is_file():
        print(f"ERROR: v9 cabinet not found: {v9_path}", file=sys.stderr)
        return 2
    if not train_path.is_file():
        print(f"ERROR: prose file not found: {train_path}", file=sys.stderr)
        return 2

    print(f"Loading unique facts from {v9_path}...")
    pairs = unique_pairs_from_jsonl(v9_path)
    print(f"Streaming and interleaving with {train_path}...")
    prose_lines = load_prose_lines(train_path)
    mixed, frames = build_mix(pairs=pairs, prose_lines=prose_lines, seed=int(args.seed))
    write_mix(out_path, mixed)
    fact_dup = max_identical_count(frames)
    mix_dup = max_identical_count(mixed)
    print("")
    print("--- Uniqueness & Dataset Mix Stats ---")
    print(f"Unique Facts Loaded: {len(pairs)}")
    print(f"Total Fact Frames Generated: {len(frames)}")
    print(f"Total Combined Output Lines: {len(mixed)}")
    print(f"Maximum Identical Fact-Frame Lines: {fact_dup}")
    print(f"Maximum Identical Lines In Mix: {mix_dup}")
    print(f"Dataset Mix written to {out_path} successfully.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
