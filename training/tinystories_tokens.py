"""Load prebuilt TinyStories int32 token shards as a concatenated memmap.

Cabinet recipes omit dataset.vocab_path / dataset.token_dir and never enter here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from logging_config import logger
from paths import PROJECT_ROOT


class ConcatenatedMemmap:
    """Read-only concatenation of memmaps. Slices stay off the full-copy path."""

    keep_memmap = True

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        if not arrays:
            raise ValueError("ConcatenatedMemmap needs at least one array")
        self._arrays = list(arrays)
        lengths = [int(a.shape[0]) for a in self._arrays]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        self._offsets = offsets
        self._n = offsets[-1]
        self.dtype = np.dtype(self._arrays[0].dtype)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n)
            if start >= stop:
                return np.empty(0, dtype=self.dtype)
            if step != 1:
                return np.asarray([self[i] for i in range(start, stop, step)], dtype=self.dtype)
            pieces: List[np.ndarray] = []
            idx = 0
            while idx < len(self._arrays) and self._offsets[idx + 1] <= start:
                idx += 1
            pos = start
            while pos < stop and idx < len(self._arrays):
                arr_start = self._offsets[idx]
                local = pos - arr_start
                take = min(stop - pos, int(self._arrays[idx].shape[0]) - local)
                pieces.append(self._arrays[idx][local : local + take])
                pos += take
                idx += 1
            if len(pieces) == 1:
                return pieces[0]
            return np.concatenate(pieces)
        if key < 0:
            key += self._n
        if key < 0 or key >= self._n:
            raise IndexError(key)
        idx = 0
        while idx < len(self._arrays) and self._offsets[idx + 1] <= key:
            idx += 1
        return self._arrays[idx][key - self._offsets[idx]]


def dataset_uses_prebuilt_tokens(dataset_cfg: Optional[dict]) -> bool:
    if not dataset_cfg:
        return False
    return bool(dataset_cfg.get("vocab_path") and dataset_cfg.get("token_dir"))


def resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_manifest(token_dir: Path) -> dict[str, Any]:
    path = token_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"TinyStories manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _open_shards(token_dir: Path, rels: Sequence[str]) -> ConcatenatedMemmap:
    arrays = []
    for rel in rels:
        path = token_dir / rel
        if not path.is_file():
            # allow "tokens/foo.npy" when token_dir is the family root
            alt = token_dir / Path(rel).name
            path = alt if alt.is_file() else path
        if not path.is_file():
            raise FileNotFoundError(f"Token shard missing: {path}")
        arrays.append(np.load(str(path), mmap_mode="r"))
    return ConcatenatedMemmap(arrays)


def load_split_tokens(token_dir: str | Path, split: str) -> ConcatenatedMemmap:
    root = resolve_under_root(token_dir)
    manifest = load_manifest(root)
    key = "train_shards" if split == "train" else "valid_shards"
    rels = list(manifest.get(key) or [])
    if not rels:
        raise FileNotFoundError(f"manifest {root / 'manifest.json'} has no {key}")
    tokens = _open_shards(root, rels)
    logger.info(
        "Loaded TinyStories %s tokens from %s shards (%s tokens, dtype=%s)",
        split, len(rels), len(tokens), tokens.dtype,
    )
    return tokens


def resolve_vocab_path(dataset_cfg: dict) -> Path:
    path = resolve_under_root(dataset_cfg["vocab_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Prebuilt vocab missing: {path}. Run python tools/prepare_tinystories.py first."
        )
    return path
