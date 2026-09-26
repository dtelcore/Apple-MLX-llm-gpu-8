#!/usr/bin/env python3
"""Confirm the c512 run uses the existing TinyStories shards. Does not rebuild them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "tinystories" / "manifest.json"
EXPECTED_SHA = "a486e56b3cbafbac6ca685039a5bf7402af663b3f2a05e943eb1ab24bc9f0fda"
EXPECTED_VOCAB = 6102
EXPECTED_TRAIN = 771869493
EXPECTED_VALID = 7790044


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST}", file=sys.stderr)
        print("Build it once with: python tools/prepare_tinystories.py", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = MANIFEST.parent
    problems: list[str] = []
    if manifest.get("vocab_sha256") != EXPECTED_SHA:
        problems.append("vocab sha does not match the c256 TinyStories BPE")
    if int(manifest.get("vocab_size", 0)) != EXPECTED_VOCAB:
        problems.append(f"vocab_size {manifest.get('vocab_size')} != {EXPECTED_VOCAB}")
    if int(manifest.get("train_tokens", 0)) != EXPECTED_TRAIN:
        problems.append(f"train_tokens {manifest.get('train_tokens')} != {EXPECTED_TRAIN}")
    if int(manifest.get("valid_tokens", 0)) != EXPECTED_VALID:
        problems.append(f"valid_tokens {manifest.get('valid_tokens')} != {EXPECTED_VALID}")
    for key in ("train_shards", "valid_shards"):
        for rel in manifest.get(key) or []:
            path = root / rel
            if not path.is_file():
                problems.append(f"missing shard {path}")
    vocab = root / manifest.get("vocab_path", "vocab.json")
    if not vocab.is_file():
        problems.append(f"missing vocab {vocab}")
    if problems:
        print("data check failed:", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("selected: full TinyStories train + valid")
    print(f"vocab: {EXPECTED_VOCAB} tokens, sha {EXPECTED_SHA[:12]}")
    print(f"train_tokens: {EXPECTED_TRAIN}")
    print(f"valid_tokens: {EXPECTED_VALID}")
    print("no new BPE, no length filter, no second corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
