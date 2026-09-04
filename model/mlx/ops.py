"""
model/mlx/ops.py

MLX primitives matching Kepler ``model/cuda/ops.py`` names and shapes.
Forward = composed ``mx`` ops (no ``mx.fast.*`` on the train path).
Backward = explicit VJPs. No autograd.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from logging_config import logger
from model.mlx import env as _env
from model.mlx.array import DeviceArray, as_mx, empty, wrap, _to_mx_dtype

_env.configure()

logger.info("MLX ops ready (explicit VJPs, no autograd)")

MAX_THREADS_PER_BLOCK = 1024  # Kepler leftover; unused on Metal
TILE_SIZE = 16
GELU_K = 0.79788456
GELU_C = 0.044715
EPS = 1e-5


class ScratchPool:
    """Reusable named device buffers (pool lifetime, not tensor free timing)."""

    def __init__(self):
        self._buffers = {}

    def get(self, shape, dtype=np.float32, zero=False, name=None) -> DeviceArray:
        from tools.tracing.runtime_metrics import memory_timeline

        key = (tuple(int(s) for s in shape), np.dtype(dtype).str, name)
        buf = self._buffers.get(key)
        shape_t = tuple(int(s) for s in shape)
        if buf is None or buf.shape != shape_t:
            buf = empty(shape_t, dtype=dtype)
            self._buffers[key] = buf
            if memory_timeline.enabled:
                mem = get_memory_usage()
                memory_timeline.record_alloc(
                    name, shape_t, int(buf.nbytes), key,
                    driver_free=int(mem["driver_free_bytes"]),
                    driver_total=int(mem["driver_total_bytes"]),
                )
        else:
            if memory_timeline.enabled:
                memory_timeline.record_reuse(name, shape_t, int(buf.nbytes))
        if zero:
            buf.replace(mx.zeros(shape_t, dtype=_to_mx_dtype(dtype)))
        return buf

    def clear(self):
        from tools.tracing.runtime_metrics import memory_timeline

        self._buffers.clear()
        if memory_timeline.enabled:
            memory_timeline.record_clear()

    def resident_bytes(self) -> int:
        return int(sum(int(b.nbytes) for b in self._buffers.values()))


scratch_pool = ScratchPool()


def eval_for_host(*arrays) -> None:
    """Force evaluation before a host read (Kepler to_host / scalar sync)."""
    from tools.tracing.runtime_metrics import runtime_metrics

    mxs = [as_mx(a) for a in arrays if a is not None]
    if not mxs:
        return
    if runtime_metrics.enabled:
        with runtime_metrics.measure("eval"):
            mx.eval(*mxs)
    else:
        mx.eval(*mxs)


def get_memory_info():
    usage = _env.check_memory(where="get_memory_info")
    return usage["driver_free_bytes"], usage["driver_total_bytes"]


def reset_memory_baseline():
    _env.reset_peak_memory()
    return _env.get_active_memory()


def get_memory_usage():
    """Process-attributed unified memory plus capacity.

    ``process_used_bytes`` is ``mx.get_active_memory()`` (may lag). Peak is
    also reported. 2 GB abort + 5.5 GB soft guard live in ``env.check_memory``.
    """
    return _env.check_memory(where="get_memory_usage")


def process_used_mb():
    return get_memory_usage()["process_used_bytes"] / (1024.0 ** 2)


def next_pow2(n: int, cap: int = MAX_THREADS_PER_BLOCK) -> int:
    p = 32
    while p < n and p < cap:
        p *= 2
    return min(p, cap)


def to_device(arr: np.ndarray) -> DeviceArray:
    from tools.tracing.runtime_metrics import runtime_metrics

    host = np.ascontiguousarray(arr, dtype=np.float32)
    if not runtime_metrics.enabled:
        return DeviceArray(mx.array(host))
    with runtime_metrics.measure("to_device"):
        out = DeviceArray(mx.array(host))
        eval_for_host(out)
    return out


def to_host(arr: DeviceArray) -> np.ndarray:
    from tools.tracing.runtime_metrics import runtime_metrics

    if not runtime_metrics.enabled:
        eval_for_host(arr)
        return np.asarray(as_mx(arr))
    with runtime_metrics.measure("to_host"):
        eval_for_host(arr)
        return np.asarray(as_mx(arr))


def to_device_int64(arr: np.ndarray) -> DeviceArray:
    host = np.ascontiguousarray(arr, dtype=np.int64)
    return DeviceArray(mx.array(host))


def float_to_half(arr: DeviceArray) -> DeviceArray:
    if arr.dtype == np.float16:
        return arr
    return DeviceArray(as_mx(arr).astype(mx.float16))


def half_to_float(arr: DeviceArray) -> DeviceArray:
    if arr.dtype == np.float32:
        return arr
    return DeviceArray(as_mx(arr).astype(mx.float32))


def to_device_ptrs(gpudata_list) -> DeviceArray:
    """Unused on MLX (no pointer-table AdamW). Kept for API compatibility."""
    ptrs = np.array([int(p) for p in gpudata_list], dtype=np.uint64)
    return DeviceArray(mx.array(ptrs))


def take_row(arr: DeviceArray, index: int, keepdims: bool = False) -> DeviceArray:
    """Copy row ``index`` of a 2-D array (replaces Kepler memcpy_dtod slice)."""
    x = as_mx(arr)
    row = x[int(index)]
    if keepdims:
        row = mx.reshape(row, (1, -1))
    return DeviceArray(row)


def add_arrays(a: DeviceArray, b: DeviceArray) -> DeviceArray:
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return DeviceArray(as_mx(a) + as_mx(b))


def add_into(a: DeviceArray, b: DeviceArray) -> DeviceArray:
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return a.replace(as_mx(a) + as_mx(b))


def matmul(A: DeviceArray, B: DeviceArray, tracer=None, name: str = "gemm") -> DeviceArray:
    assert A.ndim == 2 and B.ndim == 2, f"matmul expects 2D, got {A.shape} @ {B.shape}"
    assert A.shape[1] == B.shape[0], f"Shape mismatch: {A.shape} vs {B.shape}"
    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        M, K = A.shape
        N = B.shape[1]
        tracer.log_vectorization(name, (M, K), (K, N), (M, N), (1, 1, 1), (1, 1, 1))
    return DeviceArray(mx.matmul(as_mx(A), as_mx(B)))


def transpose_2d(x: DeviceArray, name: str = "transpose_2d") -> DeviceArray:
    assert x.ndim == 2
    out = scratch_pool.get((x.shape[1], x.shape[0]), name=name)
    return out.replace(mx.transpose(as_mx(x)))


def matmul_bias(
    A: DeviceArray, B: DeviceArray, bias: DeviceArray,
    tracer=None, name: str = "gemm_bias",
) -> DeviceArray:
    from tools.tracing.runtime_metrics import kernel_timeline

    M, K = int(A.shape[0]), int(A.shape[1])
    assert B.shape[0] == K and int(bias.size) == int(B.shape[1])
    N = int(B.shape[1])
    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        tracer.log_vectorization(name, (M, K), (K, N), (M, N), (1, 1, 1), (1, 1, 1))
    with kernel_timeline.measure(name, category="gemm"):
        out = mx.matmul(as_mx(A), as_mx(B)) + as_mx(bias)
    return DeviceArray(out)


def _gelu_mx(x: mx.array) -> mx.array:
    inner = mx.array(GELU_K) * (x + mx.array(GELU_C) * x * x * x)
    return mx.array(0.5) * x * (mx.array(1.0) + mx.tanh(inner))


def _gelu_grad_mx(x: mx.array) -> mx.array:
    inner = mx.array(GELU_K) * (x + mx.array(GELU_C) * x * x * x)
    tanh_val = mx.tanh(inner)
    sech2 = mx.array(1.0) - tanh_val * tanh_val
    return (
        mx.array(0.5) * (mx.array(1.0) + tanh_val)
        + mx.array(0.5) * x * sech2 * mx.array(GELU_K) * (mx.array(1.0) + mx.array(3.0) * mx.array(GELU_C) * x * x)
    )


def gelu(x: DeviceArray) -> DeviceArray:
    return DeviceArray(_gelu_mx(as_mx(x)))


def gelu_backward(x: DeviceArray, d_out: DeviceArray) -> DeviceArray:
    return DeviceArray(_gelu_grad_mx(as_mx(x)) * as_mx(d_out))


def add_bias(a: DeviceArray, bias: DeviceArray) -> DeviceArray:
    return DeviceArray(as_mx(a) + as_mx(bias))


def matmul_bias_gelu(
    A: DeviceArray, B: DeviceArray, bias: DeviceArray,
    tracer=None, name: str = "gemm_bias_gelu",
):
    hidden = matmul_bias(A, B, bias, tracer=tracer, name=name)
    act = gelu(hidden)
    return hidden, act


def fused_mlp_row(
    x: DeviceArray, w1: DeviceArray, b1: DeviceArray,
    w2: DeviceArray, b2: DeviceArray,
) -> DeviceArray:
    _hidden, act = matmul_bias_gelu(x, w1, b1)
    return matmul_bias(act, w2, b2)


def softmax(logits: DeviceArray) -> DeviceArray:
    x = as_mx(logits)
    shifted = x - mx.max(x, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return DeviceArray(exp / mx.sum(exp, axis=-1, keepdims=True))


def reduce_sum_axis0(x: DeviceArray) -> DeviceArray:
    assert x.ndim == 2
    return DeviceArray(mx.sum(as_mx(x), axis=0))


def linear_backward(dout: DeviceArray, x: DeviceArray, weight: DeviceArray):
    """y = x @ weight + b. weight is [in, out]. Returns (d_x, d_weight, d_bias)."""
    d_weight = matmul(x.T, dout)
    d_bias = reduce_sum_axis0(dout)
    d_x = matmul(dout, weight.T)
    return d_x, d_weight, d_bias


def linear_qkv_split(
    A: DeviceArray, W_qkv: DeviceArray, bias: DeviceArray,
    tracer=None, name: str = "qkv_split",
):
    """(A @ W_qkv + bias) split into Q,K,V without a live fused [M, 3C] buffer.

    Three column-slices of W are applied separately so parity tests exercise
    the split path rather than a naïve fused 3C matmul stand-in.
    """
    from model.mlx.allocator import lifetime_allocator

    assert int(W_qkv.shape[1]) % 3 == 0, f"qkv_split expects N = 3*C, got {W_qkv.shape[1]}"
    c = int(W_qkv.shape[1]) // 3
    w = as_mx(W_qkv)
    b = as_mx(bias)
    a = as_mx(A)
    M = int(A.shape[0])
    wq, wk, wv = w[:, :c], w[:, c:2 * c], w[:, 2 * c:]
    bq, bk, bv = b[:c], b[c:2 * c], b[2 * c:]
    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        tracer.log_vectorization(name, A.shape, W_qkv.shape, (M, c), (1, 1, 1), (1, 1, 1))
    Q = lifetime_allocator.empty((M, c), dtype=np.float32, lifetime="qkv_split")
    K_out = lifetime_allocator.empty((M, c), dtype=np.float32, lifetime="qkv_split")
    V = lifetime_allocator.empty((M, c), dtype=np.float32, lifetime="qkv_split")
    Q.replace(mx.matmul(a, wq) + bq)
    K_out.replace(mx.matmul(a, wk) + bk)
    V.replace(mx.matmul(a, wv) + bv)
    return Q, K_out, V


def split_qkv(qkv: DeviceArray, hidden_dim: int):
    c = int(hidden_dim)
    x = as_mx(qkv)
    return DeviceArray(x[:, :c]), DeviceArray(x[:, c:2 * c]), DeviceArray(x[:, 2 * c:])


def _to_heads_mx(x: mx.array, batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> mx.array:
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    y = mx.reshape(x, (B, T, NH, HD))
    y = mx.transpose(y, (0, 2, 1, 3))
    return mx.reshape(y, (B * NH, T, HD))


def interleaved_to_heads(
    x: DeviceArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
    name: str = None,
) -> DeviceArray:
    """[B*T, NH*HD] -> [B*NH, T, HD]. Fresh buffer unless a scratch-pool ``name`` is set."""
    y = _to_heads_mx(as_mx(x), batch_size, seq_len, num_heads, head_dim)
    if name:
        B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
        return scratch_pool.get((B * NH, T, HD), name=name).replace(y)
    return DeviceArray(y)


def merge_heads(
    heads: DeviceArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
    name: str = None,
) -> DeviceArray:
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    shape = (B * T, NH * HD)
    y = mx.reshape(as_mx(heads), (B, NH, T, HD))
    y = mx.transpose(y, (0, 2, 1, 3))
    y = mx.reshape(y, shape)
    if name:
        return scratch_pool.get(shape, name=name).replace(y)
    return DeviceArray(y)


def split_heads_from_qkv(
    qkv: DeviceArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
):
    q, k, v = split_qkv(qkv, num_heads * head_dim)
    return (
        interleaved_to_heads(q, batch_size, seq_len, num_heads, head_dim, name="split_q"),
        interleaved_to_heads(k, batch_size, seq_len, num_heads, head_dim, name="split_k"),
        interleaved_to_heads(v, batch_size, seq_len, num_heads, head_dim, name="split_v"),
    )


def pack_qkv(q: DeviceArray, k: DeviceArray, v: DeviceArray) -> DeviceArray:
    return DeviceArray(mx.concatenate([as_mx(q), as_mx(k), as_mx(v)], axis=-1))


def pack_qkv_from_heads(
    d_q: DeviceArray, d_k: DeviceArray, d_v: DeviceArray,
    batch_size: int, seq_len: int, num_heads: int, head_dim: int,
) -> DeviceArray:
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    q = merge_heads(d_q, B, T, NH, HD)
    k = merge_heads(d_k, B, T, NH, HD)
    v = merge_heads(d_v, B, T, NH, HD)
    return pack_qkv(q, k, v)


def _causal_mask(t: int) -> mx.array:
    # True where j > i (future positions)
    i = mx.arange(t)[:, None]
    j = mx.arange(t)[None, :]
    return j > i


def _softmax_last(x: mx.array) -> mx.array:
    shifted = x - mx.max(x, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return exp / mx.sum(exp, axis=-1, keepdims=True)


def _attn_heads(q_h: mx.array, k_h: mx.array, v_h: mx.array, scale: float):
    """q/k/v [BH, T, hd] → (out [BH, T, hd], probs [BH, T, T])."""
    scores = mx.matmul(q_h, mx.transpose(k_h, (0, 2, 1))) * mx.array(float(scale))
    t = int(q_h.shape[1])
    scores = mx.where(_causal_mask(t), mx.array(-1e9, dtype=scores.dtype), scores)
    probs = _softmax_last(scores)
    out = mx.matmul(probs, v_h)
    return out, probs


def causal_self_attention(
    q: DeviceArray,
    k: DeviceArray,
    v: DeviceArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
):
    """Causal MHA on interleaved Q/K/V [B*T, C]. Returns (attn_concat, probs flat)."""
    from tools.tracing.runtime_metrics import kernel_timeline

    B, T, H, hd = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    with kernel_timeline.measure("causal_mha", category="attention"):
        q_h = _to_heads_mx(as_mx(q), B, T, H, hd)
        k_h = _to_heads_mx(as_mx(k), B, T, H, hd)
        v_h = _to_heads_mx(as_mx(v), B, T, H, hd)
        out_h, probs = _attn_heads(q_h, k_h, v_h, scale)
        attn = merge_heads(DeviceArray(out_h), B, T, H, hd)
        probs_flat = DeviceArray(mx.reshape(probs, (B * H * T * T,)))
    return attn, probs_flat


def fused_causal_attention_from_qkv(
    ln1_out_d: DeviceArray,
    w_qkv: DeviceArray,
    bias_qkv: DeviceArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    tracer=None,
    name: str = "qkv",
    rope_base: float = None,
    pos_offset: int = 0,
):
    """QKV-split projection + causal attention on heads layout (never a live 3C GEMM)."""
    from model.mlx.allocator import lifetime_allocator

    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    H = B * NH
    q, k, v = linear_qkv_split(ln1_out_d, w_qkv, bias_qkv, tracer=tracer, name=name)
    # Fresh arrays (not ScratchPool): these are stored in the forward cache / KV arenas.
    q_h = DeviceArray(_to_heads_mx(as_mx(q), B, T, NH, HD))
    k_h = DeviceArray(_to_heads_mx(as_mx(k), B, T, NH, HD))
    v_h = DeviceArray(_to_heads_mx(as_mx(v), B, T, NH, HD))
    lifetime_allocator.release(q)
    lifetime_allocator.release(k)
    lifetime_allocator.release(v)
    if rope_base is not None:
        rope_apply_inplace(
            q_h, batch_heads=H, seq_len=T, head_dim=HD,
            base=float(rope_base), pos_offset=int(pos_offset),
        )
        rope_apply_inplace(
            k_h, batch_heads=H, seq_len=T, head_dim=HD,
            base=float(rope_base), pos_offset=int(pos_offset),
        )
    out_h, probs = _attn_heads(as_mx(q_h), as_mx(k_h), as_mx(v_h), scale)
    attn_concat = merge_heads(DeviceArray(out_h), B, T, NH, HD)
    return attn_concat, DeviceArray(mx.reshape(probs, (H * T * T,))), q_h, k_h, v_h


def softmax_backward(
    probs: DeviceArray,
    d_probs: DeviceArray,
    scale: float = 1.0,
) -> DeviceArray:
    p = as_mx(probs)
    dp = as_mx(d_probs)
    sum_term = mx.sum(p * dp, axis=-1, keepdims=True)
    return DeviceArray(mx.array(float(scale)) * p * (dp - sum_term))


def attention_backward_heads(
    d_attn_concat: DeviceArray,
    q: DeviceArray,
    k: DeviceArray,
    v: DeviceArray,
    probs: DeviceArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    heads_layout: bool = False,
):
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    H = B * NH
    M, D = T, HD
    if heads_layout:
        q_h, k_h, v_h = as_mx(q), as_mx(k), as_mx(v)
    else:
        q_h = as_mx(interleaved_to_heads(q, B, T, NH, HD, name="attn_bwd_q"))
        k_h = as_mx(interleaved_to_heads(k, B, T, NH, HD, name="attn_bwd_k"))
        v_h = as_mx(interleaved_to_heads(v, B, T, NH, HD, name="attn_bwd_v"))
    d_out_h = as_mx(interleaved_to_heads(d_attn_concat, B, T, NH, HD, name="attn_bwd_dout"))
    probs_h = mx.reshape(as_mx(probs), (H, M, M))

    d_v_h = mx.matmul(mx.transpose(probs_h, (0, 2, 1)), d_out_h)
    d_probs = mx.matmul(d_out_h, mx.transpose(v_h, (0, 2, 1)))
    d_raw = as_mx(softmax_backward(DeviceArray(probs_h), DeviceArray(d_probs), scale=scale))
    d_q_h = mx.matmul(d_raw, k_h)
    d_k_h = mx.matmul(mx.transpose(d_raw, (0, 2, 1)), q_h)

    dq = scratch_pool.get((H, M, D), name="attn_d_q").replace(d_q_h)
    dk = scratch_pool.get((H, M, D), name="attn_d_k").replace(d_k_h)
    dv = scratch_pool.get((H, M, D), name="attn_d_v").replace(d_v_h)
    if heads_layout:
        return dq, dk, dv
    return (
        merge_heads(dq, B, T, NH, HD),
        merge_heads(dk, B, T, NH, HD),
        merge_heads(dv, B, T, NH, HD),
    )


def add_block(
    acc: DeviceArray,
    block: DeviceArray,
    row0: int,
    col_start: int,
    C: int,
    hd: int,
) -> None:
    """Accumulate a [block_rows, hd] tile into acc[row0:, col_start:]."""
    host = to_host(acc)
    blk = to_host(block)
    block_rows = int(blk.shape[0])
    host[int(row0):int(row0) + block_rows, int(col_start):int(col_start) + int(hd)] += blk
    acc.replace(mx.array(host))


def add_inplace(acc: DeviceArray, block: DeviceArray) -> None:
    assert acc.shape == block.shape
    acc.replace(as_mx(acc) + as_mx(block))


def scal_mul(arr: DeviceArray, scale: float) -> DeviceArray:
    return arr.replace(as_mx(arr) * mx.array(float(scale)))


def grad_global_norm_sq(grads) -> float:
    total = mx.array(0.0)
    for g in grads.values():
        x = as_mx(g)
        total = total + mx.sum(x * x)
    eval_for_host(DeviceArray(total))
    return float(np.asarray(total))


def param_global_norm(device_tensors) -> float:
    total_sq = 0.0
    for arr in device_tensors:
        if arr is None:
            continue
        total_sq += float(grad_global_norm_sq({"_": arr}))
    return float(np.sqrt(total_sq))


def _rows_hidden(x: mx.array):
    hidden = int(x.shape[-1])
    rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    return mx.reshape(x, (rows, hidden)), rows, hidden


def layernorm(x: DeviceArray, gamma: DeviceArray, beta: DeviceArray, eps: float = 1e-5) -> DeviceArray:
    y, _, _ = layernorm_with_cache(x, gamma, beta, eps=eps)
    return y


def layernorm_with_cache(
    x: DeviceArray, gamma: DeviceArray, beta: DeviceArray, eps: float = 1e-5,
):
    flat, rows, hidden = _rows_hidden(as_mx(x))
    mean = mx.mean(flat, axis=-1, keepdims=True)
    var = mx.mean((flat - mean) ** 2, axis=-1, keepdims=True)
    invstd = 1.0 / mx.sqrt(var + mx.array(float(eps)))
    xhat = (flat - mean) * invstd
    y = xhat * as_mx(gamma) + as_mx(beta)
    return (
        DeviceArray(mx.reshape(y, x.shape)),
        DeviceArray(mx.reshape(xhat, x.shape)),
        DeviceArray(mx.reshape(invstd, (rows,))),
    )


def rmsnorm_with_cache(
    x: DeviceArray, gamma: DeviceArray, eps: float = 1e-5,
):
    flat, rows, hidden = _rows_hidden(as_mx(x))
    mean_sq = mx.mean(flat * flat, axis=-1, keepdims=True)
    inv_rms = 1.0 / mx.sqrt(mean_sq + mx.array(float(eps)))
    xhat = flat * inv_rms
    y = xhat * as_mx(gamma)
    return (
        DeviceArray(mx.reshape(y, x.shape)),
        DeviceArray(mx.reshape(xhat, x.shape)),
        DeviceArray(mx.reshape(inv_rms, (rows,))),
    )


def residual_layernorm_with_cache(
    x: DeviceArray,
    residual: DeviceArray,
    gamma: DeviceArray,
    beta: DeviceArray,
    eps: float = 1e-5,
):
    x_out = DeviceArray(as_mx(x) + as_mx(residual))
    y, xhat, inv = layernorm_with_cache(x_out, gamma, beta, eps=eps)
    return x_out, y, xhat, inv


def residual_rmsnorm_with_cache(
    x: DeviceArray,
    residual: DeviceArray,
    gamma: DeviceArray,
    eps: float = 1e-5,
):
    x_out = DeviceArray(as_mx(x) + as_mx(residual))
    y, xhat, inv = rmsnorm_with_cache(x_out, gamma, eps=eps)
    return x_out, y, xhat, inv


def _zeros_gpu(shape, dtype=np.float32) -> DeviceArray:
    return empty(shape, dtype=dtype)


def layernorm_backward(
    dout: DeviceArray,
    xhat: DeviceArray,
    invstd_row: DeviceArray,
    gamma: DeviceArray,
):
    xh = as_mx(xhat)
    do = as_mx(dout)
    hidden = int(xh.shape[-1])
    rows = int(np.prod(xh.shape[:-1])) if xh.ndim > 1 else 1
    xh_f = mx.reshape(xh, (rows, hidden))
    do_f = mx.reshape(do, (rows, hidden))
    inv = mx.reshape(as_mx(invstd_row), (rows, 1))
    g = as_mx(gamma)
    dgamma = mx.sum(do_f * xh_f, axis=0)
    dbeta = mx.sum(do_f, axis=0)
    dxhat = do_f * g
    sum_dxhat = mx.sum(dxhat, axis=-1, keepdims=True)
    sum_dxhat_xhat = mx.sum(dxhat * xh_f, axis=-1, keepdims=True)
    n = mx.array(float(hidden))
    dx = (inv / n) * (n * dxhat - sum_dxhat - xh_f * sum_dxhat_xhat)
    return (
        DeviceArray(mx.reshape(dx, xhat.shape)),
        DeviceArray(dgamma),
        DeviceArray(dbeta),
    )


def rmsnorm_backward(
    dout: DeviceArray,
    xhat: DeviceArray,
    invrms_row: DeviceArray,
    gamma: DeviceArray,
):
    xh = as_mx(xhat)
    do = as_mx(dout)
    hidden = int(xh.shape[-1])
    rows = int(np.prod(xh.shape[:-1])) if xh.ndim > 1 else 1
    xh_f = mx.reshape(xh, (rows, hidden))
    do_f = mx.reshape(do, (rows, hidden))
    inv = mx.reshape(as_mx(invrms_row), (rows, 1))
    g = as_mx(gamma)
    dxhat = do_f * g
    mean_dxhat_xhat = mx.mean(dxhat * xh_f, axis=-1, keepdims=True)
    dx = inv * (dxhat - xh_f * mean_dxhat_xhat)
    dgamma = mx.sum(do_f * xh_f, axis=0)
    return DeviceArray(mx.reshape(dx, xhat.shape)), DeviceArray(dgamma)


def cross_entropy(logits: DeviceArray, targets: np.ndarray):
    """Mean CE. logits [rows, V], targets [rows] int. Returns (loss, dlogits)."""
    x = as_mx(logits)
    rows = int(x.shape[0])
    targets_mx = mx.array(np.ascontiguousarray(targets, dtype=np.int32).reshape(-1))
    shifted = x - mx.max(x, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    probs = exp / mx.sum(exp, axis=-1, keepdims=True)
    idx = mx.arange(rows)
    correct = probs[idx, targets_mx]
    loss = -mx.mean(mx.log(mx.clip(correct, 1e-12, None)))
    onehot = mx.zeros_like(probs)
    onehot = onehot.at[idx, targets_mx].add(1.0)
    d_logits = (probs - onehot) / mx.array(float(rows))
    eval_for_host(DeviceArray(loss), DeviceArray(d_logits))
    return float(np.asarray(loss)), DeviceArray(d_logits)


def adamw_update(
    w: DeviceArray,
    g: DeviceArray,
    m: DeviceArray,
    v: DeviceArray,
    lr: float,
    wd: float,
    b1: float,
    b2: float,
    eps: float,
    bc1: float,
    bc2: float,
) -> None:
    gg = as_mx(g)
    mm = mx.array(float(b1)) * as_mx(m) + mx.array(1.0 - float(b1)) * gg
    vv = mx.array(float(b2)) * as_mx(v) + mx.array(1.0 - float(b2)) * gg * gg
    m.replace(mm)
    v.replace(vv)
    m_hat = mm / mx.array(float(bc1))
    v_hat = vv / mx.array(float(bc2))
    ww = as_mx(w) - mx.array(float(lr)) * (m_hat / (mx.sqrt(v_hat) + mx.array(float(eps))))
    if wd > 0.0:
        ww = ww - mx.array(float(lr) * float(wd)) * ww
    w.replace(ww)


def adamw_update_batched(
    offsets_d, w_ptrs_d, g_ptrs_d, m_ptrs_d, v_ptrs_d,
    ntensors, total_n, lr, wd, b1, b2, eps, bc1, bc2,
) -> None:
    raise RuntimeError("adamw_update_batched is CUDA pointer-table only; use adamw_update")


def embedding_lookup(ids: np.ndarray, emb: DeviceArray, pos_emb: DeviceArray, T: int) -> DeviceArray:
    ids = np.asarray(ids, dtype=np.int32)
    B = int(ids.shape[0])
    tok = as_mx(emb)[mx.array(ids.reshape(-1))]
    pos_idx = np.tile(np.arange(int(T), dtype=np.int32), B)
    out = tok + as_mx(pos_emb)[mx.array(pos_idx)]
    return DeviceArray(out)


def embedding_lookup_tokens(ids: np.ndarray, emb: DeviceArray, T: int) -> DeviceArray:
    ids = np.asarray(ids, dtype=np.int32)
    tok = as_mx(emb)[mx.array(ids.reshape(-1))]
    return DeviceArray(tok)


def rope_apply_inplace(
    x: DeviceArray,
    *,
    batch_heads: int,
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    pos_offset: int = 0,
    backward: bool = False,
) -> DeviceArray:
    assert x.ndim == 3 and int(x.shape[2]) == head_dim
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    BH, T, HD = int(batch_heads), int(seq_len), int(head_dim)
    pairs = HD // 2
    sign = -1.0 if backward else 1.0
    freqs = base ** (-2.0 * np.arange(pairs, dtype=np.float64) / HD)
    pos = (np.arange(T, dtype=np.float64) + pos_offset)[:, None]
    angles = (sign * pos * freqs[None, :]).astype(np.float32)
    c = mx.array(np.cos(angles).astype(np.float32))
    s = mx.array(np.sin(angles).astype(np.float32))
    xx = as_mx(x)
    x0 = xx[..., 0::2]
    x1 = xx[..., 1::2]
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    y = mx.reshape(mx.stack([y0, y1], axis=-1), (BH, T, HD))
    return x.replace(y)


def embed_backward(
    ids: np.ndarray, d_h: DeviceArray, vocab_size: int, embed_dim: int,
    *, with_position: bool = True,
):
    B, T = ids.shape
    C = embed_dim
    ids_flat = mx.array(np.ascontiguousarray(ids, dtype=np.int32).reshape(-1))
    dh = mx.reshape(as_mx(d_h), (B * T, C))
    d_tok = mx.zeros((int(vocab_size), C), dtype=mx.float32)
    d_tok = d_tok.at[ids_flat].add(dh)
    if not with_position:
        return DeviceArray(d_tok), None
    d_pos = mx.sum(mx.reshape(dh, (B, T, C)), axis=0)
    return DeviceArray(d_tok), DeviceArray(d_pos)


def sync_to_host(device_arr: DeviceArray, host_arr: np.ndarray) -> None:
    from tools.tracing.runtime_metrics import runtime_metrics

    if runtime_metrics.enabled:
        with runtime_metrics.measure("to_host"):
            host_arr[:] = to_host(device_arr).reshape(host_arr.shape)
        return
    host_arr[:] = to_host(device_arr).reshape(host_arr.shape)


def kv_pack_prefill(
    src: DeviceArray,
    dst: DeviceArray,
    *,
    batch_heads: int,
    seq_len: int,
    max_len: int,
    head_dim: int,
    stream=None,
) -> None:
    T = int(seq_len)
    packed = mx.concatenate([as_mx(src), as_mx(dst)[:, T:, :]], axis=1)
    mx.eval(packed)
    dst.replace(packed)


def kv_append_row(
    src: DeviceArray,
    dst: DeviceArray,
    *,
    batch_heads: int,
    t: int,
    max_len: int,
    head_dim: int,
    stream=None,
) -> None:
    BH, hd = int(batch_heads), int(head_dim)
    t = int(t)
    mid = mx.reshape(as_mx(src), (BH, 1, hd))
    packed = mx.concatenate([as_mx(dst)[:, :t, :], mid, as_mx(dst)[:, t + 1:, :]], axis=1)
    mx.eval(packed)
    dst.replace(packed)


def causal_mha_decode(
    q: DeviceArray,
    k_arena: DeviceArray,
    v_arena: DeviceArray,
    *,
    batch_heads: int,
    max_len: int,
    valid_len: int,
    head_dim: int,
    scale: float,
    out: DeviceArray = None,
    stream=None,
) -> DeviceArray:
    from tools.tracing.runtime_metrics import kernel_timeline

    BH, hd = int(batch_heads), int(head_dim)
    T_valid = int(valid_len)
    qh = mx.reshape(as_mx(q), (BH, 1, hd))
    k = as_mx(k_arena)[:, :T_valid, :]
    v = as_mx(v_arena)[:, :T_valid, :]
    with kernel_timeline.measure("causal_mha_decode", category="attention"):
        scores = mx.matmul(qh, mx.transpose(k, (0, 2, 1))) * mx.array(float(scale))
        probs = _softmax_last(scores)
        y = mx.reshape(mx.matmul(probs, v), (BH, hd))
    if out is None:
        return DeviceArray(y)
    return out.replace(y)


def argmax_1d(logits: DeviceArray, out_idx: DeviceArray = None, stream=None) -> DeviceArray:
    idx = mx.argmax(as_mx(logits))
    idx = mx.reshape(idx.astype(mx.int32), (1,))
    if out_idx is None:
        return DeviceArray(idx)
    return out_idx.replace(idx)


def topk_mask_inplace(logits: DeviceArray, k: int) -> DeviceArray:
    x = as_mx(logits)
    n = int(x.size)
    kk = max(1, min(int(k), n))
    host = np.asarray(x)
    mx.eval(x)
    host = np.asarray(x).reshape(-1).copy()
    kth = np.partition(host, -kk)[-kk]
    host[host < kth] = -1e30
    return logits.replace(mx.array(host.reshape(logits.shape).astype(np.float32)))


def sample_logits_device(
    logits: DeviceArray,
    *,
    temperature: float = 1.0,
    top_k: int = None,
    out_idx: DeviceArray = None,
) -> DeviceArray:
    n = int(logits.size)
    work = scratch_pool.get((n,), name="sample_logits_work")
    work.replace(mx.reshape(as_mx(logits), (n,)))
    temp = max(float(temperature), 1e-6)
    if abs(temp - 1.0) > 1e-8:
        scal_mul(work, 1.0 / temp)
    if top_k is not None and int(top_k) > 0:
        topk_mask_inplace(work, int(top_k))
    return argmax_1d(work, out_idx=out_idx)
