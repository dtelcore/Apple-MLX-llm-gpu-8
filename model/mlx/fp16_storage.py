"""
model/mlx/fp16_storage.py

Optional FP16 activation storage. Compute math stays FP32.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from model.mlx import ops as cuda_ops
from model.mlx.array import DeviceArray

DEFAULT_FP16_KEYS = (
    "ln1_out_d", "ln1_xhat_d", "ln2_out_d", "ln2_xhat_d",
    "attn_concat_d", "probs_d", "q_d", "k_d", "v_d",
    "hidden_d", "act_d", "h_final_d",
)

_ENABLED = False


def set_fp16_activation_storage(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def fp16_storage_enabled() -> bool:
    return _ENABLED


def _is_device(x: Any) -> bool:
    return isinstance(x, DeviceArray) or hasattr(x, "gpudata")


def to_fp16_storage(arr: DeviceArray) -> DeviceArray:
    return cuda_ops.float_to_half(arr)


def to_fp32_compute(arr: DeviceArray) -> DeviceArray:
    return cuda_ops.half_to_float(arr)


def compress_cache_fp16(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> Dict[str, int]:
    if not _ENABLED:
        return {}
    keyset = set(keys)
    saved: Dict[str, int] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                child = f"{path}.{k}" if path else str(k)
                if k in keyset and _is_device(v) and v.dtype == np.float32:
                    before = int(v.nbytes)
                    node[k] = to_fp16_storage(v)
                    saved[child] = before - int(node[k].nbytes)
                else:
                    visit(v, child)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{path}[{i}]")

    visit(cache, "")
    cache["_fp16_storage"] = True
    return saved


def expand_cache_fp32(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> None:
    if not cache.get("_fp16_storage"):
        return
    keyset = set(keys)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k in keyset and _is_device(v) and v.dtype == np.float16:
                    node[k] = to_fp32_compute(v)
                else:
                    visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(cache)
    cache["_fp16_storage"] = False


def estimate_savings_bytes(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> int:
    keyset = set(keys)
    total = 0

    def visit(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keyset and _is_device(v) and v.dtype == np.float32:
                    total += int(v.nbytes) // 2
                else:
                    visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(cache)
    return total
