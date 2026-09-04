"""
model/mlx/graph.py

CUDA Graph analog. v1 stubs with a pointer to ``mx.compile`` later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from logging_config import logger


@dataclass
class GraphStatus:
    supported: bool = False
    captured: bool = False
    mode: str = "fallback"  # eager | graph | fallback
    reason: str = ""
    capture_ms: float = 0.0
    replay_ms: float = 0.0
    eager_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": self.supported,
            "captured": self.captured,
            "mode": self.mode,
            "reason": self.reason,
            "capture_ms": self.capture_ms,
            "replay_ms": self.replay_ms,
            "eager_ms": self.eager_ms,
            "details": self.details,
        }


_STUB_REASON = "CUDA Graph dropped on MLX; use mx.compile on the closed step later"


def probe_cuda_graphs() -> GraphStatus:
    return GraphStatus(supported=False, captured=False, mode="fallback", reason=_STUB_REASON)


def capture_gpu_callable(fn: Callable[[Any], None], repeats: int = 20) -> GraphStatus:
    logger.info("mx.compile of the train step is out of scope for v1; %s", _STUB_REASON)
    return probe_cuda_graphs()


def try_capture_decode(decode_fn: Callable[[], None]) -> GraphStatus:
    st, _ = try_capture_decode_replayable(decode_fn)
    return st


def try_capture_decode_replayable(
    decode_fn: Callable[[], None],
) -> tuple:
    st = probe_cuda_graphs()
    return st, None
