"""
tokenizer/bpe.py

Whitespace-aware BPE over character symbols. Default tokenizer for new training
runs (see tokenizer/factory.py). Char remains available via --tokenizer char.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from logging_config import logger

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable

        class _Dummy:
            def update(self, n=1):
                pass

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Dummy()

# Per-process state for spawn workers (set by _init_encode_worker).
_WORKER_MERGES: List[Tuple[str, str]] = []
_WORKER_TOKEN_TO_ID: Dict[str, int] = {}


def _resolve_workers(workers: Optional[int]) -> int:
    if workers is not None:
        return max(1, int(workers))
    raw = os.environ.get("LLM_BPE_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, min(os.cpu_count() or 1, 8))


def _apply_merges_seq(symbols: List[str], merges: List[Tuple[str, str]]) -> List[str]:
    seq = list(symbols)
    for a, b in merges:
        merged = a + b
        new_seq: List[str] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
    return seq


def _ids_from_pieces(pieces: List[str], token_to_id: Dict[str, int]) -> List[int]:
    ids: List[int] = []
    for p in pieces:
        tid = token_to_id.get(p)
        if tid is not None:
            ids.append(tid)
            continue
        for ch in p:
            cid = token_to_id.get(ch)
            if cid is not None:
                ids.append(cid)
    return ids


def _init_encode_worker(merges: List[Tuple[str, str]], token_to_id: Dict[str, int]) -> None:
    global _WORKER_MERGES, _WORKER_TOKEN_TO_ID
    _WORKER_MERGES = merges
    _WORKER_TOKEN_TO_ID = token_to_id


def _worker_encode_words(words: List[str]) -> List[List[int]]:
    merges = _WORKER_MERGES
    table = _WORKER_TOKEN_TO_ID
    out: List[List[int]] = []
    for word in words:
        out.append(_ids_from_pieces(_apply_merges_seq(list(word), merges), table))
    return out


def _word_tokens(text: str) -> List[str]:
    """Split on whitespace; keep spaces as separate tokens so decode round-trips."""
    parts: List[str] = []
    buf = []
    for ch in text:
        if ch.isspace():
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _merge_symbols(seq: List[str], a: str, b: str) -> List[str]:
    merged = a + b
    new_seq: List[str] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            new_seq.append(merged)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


def _add_pairs(
    pair_freq: Counter,
    pair_index: Dict[Tuple[str, str], Set[int]],
    word_i: int,
    seq: List[str],
    freq: int,
    sign: int,
) -> None:
    for j in range(len(seq) - 1):
        pair = (seq[j], seq[j + 1])
        pair_freq[pair] += sign * freq
        if sign > 0:
            pair_index[pair].add(word_i)
            continue
        if pair_freq[pair] <= 0:
            del pair_freq[pair]
            pair_index.pop(pair, None)
        else:
            pair_index[pair].discard(word_i)


class BPETokenizer:
    """Whitespace-aware BPE; API-compatible with CharacterGPTTokenizer for train/trace."""

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self._encode_cache: Dict[str, List[int]] = {}

    @staticmethod
    def _sample_text(corpus: Iterable[str], max_chars: Optional[int]) -> str:
        """Join corpus docs with spaces, stopping once max_chars is reached.

        Avoids materializing the full TinyStories join (~100M+ chars) when only
        a BPE training sample is needed.
        """
        if max_chars is None:
            return " ".join(corpus)
        limit = int(max_chars)
        if limit <= 0:
            return ""
        parts: List[str] = []
        n = 0
        for doc in corpus:
            if n >= limit:
                break
            if n > 0:
                parts.append(" ")
                n += 1
                if n >= limit:
                    break
            take = limit - n
            if len(doc) <= take:
                parts.append(doc)
                n += len(doc)
            else:
                parts.append(doc[:take])
                n = limit
                break
        return "".join(parts)

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[str],
        num_merges: int = 200,
        max_chars: Optional[int] = 200_000,
    ) -> "BPETokenizer":
        text = cls._sample_text(corpus, max_chars)
        logger.info(
            "BPE training sample: %s chars (cap=%s), merges=%s",
            len(text),
            max_chars if max_chars is not None else "none",
            num_merges,
        )
        inst = cls()
        inst.train(text, num_merges=num_merges)
        return inst

    def train(self, text: str, num_merges: int = 200) -> None:
        """Learn merges from unique words with counts; update only affected words."""
        chars = sorted(set(text))
        counts = Counter(_word_tokens(text))
        words = list(counts.keys())
        freqs = [counts[w] for w in words]
        seqs = [list(w) for w in words]

        pair_freq: Counter = Counter()
        pair_index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        for i, seq in enumerate(seqs):
            _add_pairs(pair_freq, pair_index, i, seq, freqs[i], 1)

        merges: List[Tuple[str, str]] = []
        vocab_set = set(chars)
        total = int(num_merges)
        logger.info(
            "BPE train: %s unique words (%s occurrences), %s pair types",
            len(words),
            sum(freqs),
            len(pair_freq),
        )
        for _step in tqdm(range(total), desc="BPE merges", unit="merge", dynamic_ncols=True):
            if not pair_freq:
                break
            pair, count = pair_freq.most_common(1)[0]
            if count < 2:
                break
            merges.append(pair)
            vocab_set.add(pair[0] + pair[1])
            a, b = pair
            affected = list(pair_index.pop(pair, ()))
            for i in affected:
                seq = seqs[i]
                freq = freqs[i]
                _add_pairs(pair_freq, pair_index, i, seq, freq, -1)
                new_seq = _merge_symbols(seq, a, b)
                seqs[i] = new_seq
                _add_pairs(pair_freq, pair_index, i, new_seq, freq, 1)

        self.merges = merges
        self.vocab = sorted(vocab_set, key=lambda s: (len(s), s))
        self._token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self._id_to_token = {i: t for t, i in self._token_to_id.items()}
        self.vocab_size = len(self.vocab)
        logger.info("BPE vocab ready: %s tokens after %s merges", self.vocab_size, len(self.merges))

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        return _apply_merges_seq(symbols, self.merges)

    def _encode_word(self, word: str) -> List[int]:
        """Encode one whitespace-delimited piece; cached (TinyStories has heavy reuse)."""
        cache = self._encode_cache
        cached = cache.get(word)
        if cached is not None:
            return cached
        ids = _ids_from_pieces(self._apply_merges(list(word)), self._token_to_id)
        cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in _word_tokens(text):
            ids.extend(self._encode_word(word))
        return ids

    def _fill_cache_parallel(self, words: List[str], workers: int) -> None:
        import multiprocessing as mp

        chunk = max(64, min(512, max(1, len(words) // (workers * 8))))
        chunks = [words[i : i + chunk] for i in range(0, len(words), chunk)]
        logger.info(
            "BPE encode words: %s unique, %s workers, chunk=%s",
            len(words),
            workers,
            chunk,
        )
        ctx = mp.get_context("spawn")
        encoded: List[List[int]] = []
        bar = tqdm(total=len(words), desc="BPE encode words", unit="word", dynamic_ncols=True)
        try:
            with ctx.Pool(
                processes=workers,
                initializer=_init_encode_worker,
                initargs=(self.merges, self._token_to_id),
            ) as pool:
                for part in pool.imap(_worker_encode_words, chunks, chunksize=1):
                    encoded.extend(part)
                    bar.update(len(part))
        finally:
            bar.close()
        for word, ids in zip(words, encoded):
            self._encode_cache[word] = ids

    def encode_corpus(
        self,
        corpus: List[str],
        progress_every: int = 25_000,
        workers: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode docs joined by single spaces (same token stream as ' '.join + encode)."""
        import numpy as np

        del progress_every  # tqdm bars replace the old every-N-docs log
        n = len(corpus)
        n_workers = _resolve_workers(workers)
        logger.info("BPE-encoding corpus: %s documents (workers=%s)...", n, n_workers)

        pending: List[str] = []
        seen = set(self._encode_cache)
        for doc in tqdm(corpus, desc="BPE scan docs", unit="doc", dynamic_ncols=True):
            for word in _word_tokens(doc):
                if word not in seen:
                    seen.add(word)
                    pending.append(word)

        if pending:
            # Auto-picked workers skip the pool on tiny unique sets (spawn cost).
            # An explicit workers= argument always uses the pool so tests can
            # pin the parallel path.
            use_pool = n_workers > 1 and (len(pending) >= 512 or workers is not None)
            if use_pool:
                self._fill_cache_parallel(pending, n_workers)
            else:
                for word in tqdm(pending, desc="BPE encode words", unit="word", dynamic_ncols=True):
                    self._encode_word(word)

        space_ids = self._encode_word(" ")
        chunks: List[List[int]] = []
        buf: List[int] = []
        cache = self._encode_cache
        for i, doc in enumerate(tqdm(corpus, desc="BPE stitch", unit="doc", dynamic_ncols=True)):
            if i:
                buf.extend(space_ids)
            for word in _word_tokens(doc):
                buf.extend(cache[word] if word in cache else self._encode_word(word))
            if len(buf) >= 1_000_000:
                chunks.append(buf)
                buf = []
        if buf:
            chunks.append(buf)
        if not chunks:
            return np.array([], dtype=np.int64)
        tokens = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
        logger.info(
            "BPE encode done: %s tokens from %s docs (unique-word cache=%s)",
            len(tokens), n, len(self._encode_cache),
        )
        return tokens

    def decode(self, ids: List[int]) -> str:
        return "".join(self._id_to_token.get(i, "") for i in ids)

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, -1)

    def save_vocab(self, filepath: Union[str, Path]) -> None:
        self.save(filepath)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("BPE vocab saved to %s (%s tokens)", path, self.vocab_size)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BPETokenizer":
        state = json.loads(Path(filepath).read_text(encoding="utf-8"))
        inst = cls()
        inst.vocab = state["vocab"]
        inst.merges = [tuple(p) for p in state["merges"]]
        inst._token_to_id = {str(k): int(v) for k, v in state["token_to_id"].items()}
        inst._id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        inst.vocab_size = len(inst.vocab)
        return inst

    def coverage_stats(self, text: str, context_tokens: int) -> Dict[str, float]:
        """Compare char vs BPE semantic span inside a fixed token window."""
        char_len = len(text)
        ids = self.encode(text)
        window = ids[:context_tokens]
        decoded = self.decode(window)
        return {
            "char_len": float(char_len),
            "bpe_token_count": float(len(ids)),
            "chars_per_token": float(char_len) / max(1, len(ids)),
            "context_token_window": float(context_tokens),
            "chars_covered_in_window": float(len(decoded)),
            "compression_vs_chars": float(char_len) / max(1, len(ids)),
        }
