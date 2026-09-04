"""Parity: causal attention forward (host reference vs GPU)."""

from __future__ import annotations

import numpy as np

from model.gpt import _batched_attention_host
from tests.parity._common import B, C, H, HD, CudaTestCase, T, assert_close


class TestAttentionParity(CudaTestCase):
    def test_causal_attention_forward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(4)
        scale = 1.0 / np.sqrt(HD)
        # causal_self_attention expects interleaved Q/K/V as [B*T, C]
        q = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        k = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        v = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        qkv = np.concatenate([q, k, v], axis=-1)

        attn_ref, probs_ref, _, _, _ = _batched_attention_host(qkv, B, T, H, HD, scale)

        qd = ops.to_device(q)
        kd = ops.to_device(k)
        vd = ops.to_device(v)
        attn_d, probs_d = ops.causal_self_attention(qd, kd, vd, B, T, H, HD, scale)
        assert_close("attention.out", ops.to_host(attn_d), attn_ref, rtol=2e-4, atol=2e-5)

        probs_h = ops.to_host(probs_d).reshape(B, H, T, T)
        assert_close("attention.probs", probs_h, probs_ref, rtol=2e-4, atol=2e-5)

    def test_linear_qkv_split(self) -> None:
        """QKV projection must split into Q,K,V without a fused [B·T, 3C] stand-in."""
        ops = self.cuda_ops
        rng = np.random.default_rng(9)
        M = B * T
        A = rng.standard_normal((M, C), dtype=np.float32)
        W = rng.standard_normal((C, 3 * C), dtype=np.float32) * 0.1
        bias = rng.standard_normal((3 * C,), dtype=np.float32) * 0.01
        y = A @ W + bias
        q_ref, k_ref, v_ref = y[:, :C], y[:, C:2 * C], y[:, 2 * C:]

        q_d, k_d, v_d = ops.linear_qkv_split(ops.to_device(A), ops.to_device(W), ops.to_device(bias))
        assert q_d.shape == (M, C)
        assert k_d.shape == (M, C)
        assert v_d.shape == (M, C)
        assert_close("qkv_split.q", ops.to_host(q_d), q_ref)
        assert_close("qkv_split.k", ops.to_host(k_d), k_ref)
        assert_close("qkv_split.v", ops.to_host(v_d), v_ref)

        scale = 1.0 / np.sqrt(HD)
        attn_ref, probs_ref, _, _, _ = _batched_attention_host(y, B, T, H, HD, scale)
        attn_d, probs_d, q_h, k_h, v_h = ops.fused_causal_attention_from_qkv(
            ops.to_device(A), ops.to_device(W), ops.to_device(bias),
            B, T, H, HD, scale,
        )
        assert q_h.shape == (B * H, T, HD)
        assert_close("qkv_split.attn", ops.to_host(attn_d), attn_ref, rtol=2e-4, atol=2e-5)
        assert_close(
            "qkv_split.probs",
            ops.to_host(probs_d).reshape(B, H, T, T),
            probs_ref,
            rtol=2e-4,
            atol=2e-5,
        )
