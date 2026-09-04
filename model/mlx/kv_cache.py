"""
model/mlx/kv_cache.py

Static device KV arenas sized from GPTConfig.max_len.
Prefill packs K/V into fixed [B*H, max_len, hd] buffers; decode appends rows.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from model.mlx import ops as cuda_ops
from model.mlx.array import empty


def alloc_layer_arena(batch_heads: int, max_len: int, head_dim: int) -> Dict:
    shape = (int(batch_heads), int(max_len), int(head_dim))
    return {
        "k_d": empty(shape, dtype=np.float32),
        "v_d": empty(shape, dtype=np.float32),
    }


def pack_prefill_into_arena(
    k_src,
    v_src,
    arena: Dict,
    *,
    batch_heads: int,
    seq_len: int,
    max_len: int,
    head_dim: int,
) -> None:
    cuda_ops.kv_pack_prefill(
        k_src, arena["k_d"],
        batch_heads=batch_heads, seq_len=seq_len, max_len=max_len, head_dim=head_dim,
    )
    cuda_ops.kv_pack_prefill(
        v_src, arena["v_d"],
        batch_heads=batch_heads, seq_len=seq_len, max_len=max_len, head_dim=head_dim,
    )


def append_kv_row(
    k_new,
    v_new,
    arena: Dict,
    *,
    batch_heads: int,
    t: int,
    max_len: int,
    head_dim: int,
) -> None:
    cuda_ops.kv_append_row(
        k_new, arena["k_d"],
        batch_heads=batch_heads, t=t, max_len=max_len, head_dim=head_dim,
    )
    cuda_ops.kv_append_row(
        v_new, arena["v_d"],
        batch_heads=batch_heads, t=t, max_len=max_len, head_dim=head_dim,
    )


def arena_nbytes(kv_state: Dict) -> int:
    total = 0
    for layer in kv_state.get("layers", []):
        if "k_d" in layer:
            total += int(layer["k_d"].nbytes) + int(layer["v_d"].nbytes)
        else:
            total += int(layer["k"].nbytes) + int(layer["v"].nbytes)
    return total


def build_device_kv_state(
    cache: Dict,
    *,
    max_len: int,
    num_heads: int,
    head_dim: int,
) -> Dict:
    B = int(cache["B"])
    T = int(cache["T"])
    H = int(num_heads)
    hd = int(head_dim)
    BH = B * H
    layers: List[Dict] = []
    for layer_cache in cache["layers"]:
        attn = layer_cache["attn"]
        arena = alloc_layer_arena(BH, max_len, hd)
        if attn.get("gpu") and "k_d" in attn:
            pack_prefill_into_arena(
                attn["k_d"], attn["v_d"], arena,
                batch_heads=BH, seq_len=T, max_len=max_len, head_dim=hd,
            )
        else:
            k_h = attn["k_h"].reshape(BH, T, hd).astype(np.float32, copy=False)
            v_h = attn["v_h"].reshape(BH, T, hd).astype(np.float32, copy=False)
            k_tmp = cuda_ops.to_device(np.ascontiguousarray(k_h))
            v_tmp = cuda_ops.to_device(np.ascontiguousarray(v_h))
            pack_prefill_into_arena(
                k_tmp, v_tmp, arena,
                batch_heads=BH, seq_len=T, max_len=max_len, head_dim=hd,
            )
        layers.append(arena)
    return {
        "layers": layers,
        "T": T,
        "B": B,
        "device": True,
        "max_len": int(max_len),
        "num_heads": H,
        "head_dim": hd,
    }


def clone_device_kv_state(kv_state: Dict) -> Dict:
    if not kv_state.get("device"):
        return {
            "B": kv_state["B"],
            "T": kv_state["T"],
            "layers": [
                {"k": ly["k"].copy(), "v": ly["v"].copy()}
                for ly in kv_state["layers"]
            ],
        }
    layers = []
    for ly in kv_state["layers"]:
        layers.append({
            "k_d": cuda_ops.to_device(cuda_ops.to_host(ly["k_d"])),
            "v_d": cuda_ops.to_device(cuda_ops.to_host(ly["v_d"])),
        })
    out = dict(kv_state)
    out["layers"] = layers
    return out
