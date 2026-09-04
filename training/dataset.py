"""
training/dataset.py

Turns the corpus (list of sentence strings) into a contiguous token stream
and yields (X, Y) sliding-window batches, Y being X shifted by one position
(next-character prediction).
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from logging_config import logger


class WindowedDataset:
    def __init__(
        self,
        corpus: List[str],
        tokenizer,
        max_len: int,
        batch_size: int,
        window_stride: int = 1,
        tokens: Optional[np.ndarray] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.window_stride = max(1, int(window_stride))

        # Prefer a resume cache (tokens.npy) so BPE encode is not repeated.
        # Else streaming encode (BPE): avoids " ".join of entire TinyStories
        # (~100M chars) and reuses a per-word encode cache across docs.
        if tokens is not None:
            self.tokens = np.asarray(tokens, dtype=np.int64)
            logger.info("Using pre-encoded token stream (%s tokens)", len(self.tokens))
        elif hasattr(tokenizer, "encode_corpus"):
            self.tokens = np.asarray(tokenizer.encode_corpus(corpus), dtype=np.int64)
        else:
            logger.info("Encoding corpus to tokens (%s documents)...", len(corpus))
            joined_text = " ".join(corpus)
            self.tokens = np.array(tokenizer.encode(joined_text), dtype=np.int64)
            logger.info("Encoded %s tokens from %s documents", len(self.tokens), len(corpus))
        self.total_tokens = len(self.tokens)

        if self.total_tokens < max_len + 1:
            raise ValueError(
                f"Corpus has only {self.total_tokens} tokens, need at least "
                f"{max_len + 1} for a single window (max_len={max_len}). "
                f"Use a larger dataset (e.g. tiny_stories) or lower max_len / "
                f"context window to <= {max(1, self.total_tokens - 1)}."
            )

    def num_dense_windows(self) -> int:
        """Sliding starts with stride=1 (upper bound on unique windows)."""
        return self.total_tokens - self.max_len

    def num_windows(self) -> int:
        dense = self.num_dense_windows()
        if dense <= 0:
            return 0
        return int(np.arange(0, dense, self.window_stride).size)

    def num_batches(self) -> int:
        return max(1, self.num_windows() // self.batch_size)

    def iter_batches(self, shuffle: bool = True, rng: np.random.Generator = None) -> Iterator[List[Tuple[np.ndarray, np.ndarray]]]:
        """Yields lists of (x, y) sequence pairs, one list per mini-batch."""
        rng = rng or np.random.default_rng()
        starts = np.arange(0, self.num_dense_windows(), self.window_stride)
        if shuffle:
            rng.shuffle(starts)

        for b in range(self.num_batches()):
            batch_starts = starts[b * self.batch_size : (b + 1) * self.batch_size]
            if len(batch_starts) == 0:
                continue
            pairs = []
            for start in batch_starts:
                x = self.tokens[start : start + self.max_len]
                y = self.tokens[start + 1 : start + self.max_len + 1]
                pairs.append((x, y))
            yield pairs
