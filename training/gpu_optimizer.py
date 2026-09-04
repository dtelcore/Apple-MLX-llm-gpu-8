"""
training/gpu_optimizer.py

AdamW on MLX-resident weight mirrors. Host NumPy copies sync only at checkpoint.
"""

import math
from typing import Dict, Iterable, Optional

import numpy as np

from model.mlx import ops as cuda_ops
from model.mlx.array import DeviceArray
from model.weights import ModelParameters


class AdamWGPU:
    """AdamW on device weight mirrors (ModelParameters.device_weights/biases)."""

    def __init__(
        self,
        params: ModelParameters,
        learning_rate: float,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        gradient_clip: float = 1.0,
        total_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.params = params
        self.base_lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.warmup_steps = max(0, warmup_steps)
        self.gradient_clip = gradient_clip
        self.total_steps = max(0, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.t = 0

        all_keys = list(params.trainable_param_names())
        self.m: Dict[str, DeviceArray] = {}
        self.v: Dict[str, DeviceArray] = {}
        for key in all_keys:
            if key in params.device_weights:
                arr = params.device_weights[key]
            elif key in params.device_biases:
                arr = params.device_biases[key]
            else:
                continue
            z = np.zeros(arr.shape, dtype=np.float32)
            self.m[key] = cuda_ops.to_device(z)
            self.v[key] = cuda_ops.to_device(z)

        self._batch_keys = [k for k in all_keys if k in self.m]

    def current_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return self.base_lr
        min_lr = self.base_lr * self.min_lr_ratio
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (self.t - self.warmup_steps) / denom))
        return min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def clip_grads_(self, grads: Dict[str, DeviceArray]) -> float:
        total_sq = cuda_ops.grad_global_norm_sq(grads)
        global_norm = float(np.sqrt(total_sq))
        if self.gradient_clip and global_norm > self.gradient_clip:
            scale = self.gradient_clip / (global_norm + 1e-6)
            for key in grads:
                cuda_ops.scal_mul(grads[key], scale)
        return global_norm

    def _get_weight(self, key: str) -> DeviceArray:
        if key in self.params.device_weights:
            return self.params.device_weights[key]
        return self.params.device_biases[key]

    def step(self, grads: Dict[str, DeviceArray]) -> None:
        self.t += 1
        lr = self.current_lr()
        b1, b2, eps = self.beta1, self.beta2, self.epsilon
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        updated = []
        for key in self._batch_keys:
            if key not in grads:
                continue
            w = self._get_weight(key)
            cuda_ops.adamw_update(
                w, grads[key], self.m[key], self.v[key],
                lr, self.weight_decay, b1, b2, eps, bc1, bc2,
            )
            updated.append(w.mx)
        if updated:
            import mlx.core as mx
            mx.eval(*updated)
        if self.params.tie_embeddings and "token_embedding" in self.params.device_weights:
            self.params.device_weights["lm_head"] = self.params.device_weights["token_embedding"].T

    def sync_host_weights(self, names: Optional[Iterable[str]] = None) -> None:
        """Pull device mirrors back to host NumPy dicts (checkpoint save only)."""
        if names is not None:
            keys = list(names)
        else:
            keys = list(self.params.trainable_param_names())
        for key in keys:
            if self.params.tie_embeddings and key == "lm_head":
                continue
            if key in self.params.device_weights:
                host = self.params.weights[key]
                if not host.flags.c_contiguous:
                    host = np.ascontiguousarray(host)
                    self.params.weights[key] = host
                cuda_ops.sync_to_host(self.params.device_weights[key], host)
            elif key in self.params.device_biases:
                cuda_ops.sync_to_host(self.params.device_biases[key], self.params.biases[key])
        if self.params.tie_embeddings:
            self.params.weights["lm_head"] = self.params.weights["token_embedding"].T
            self.params.device_weights["lm_head"] = self.params.device_weights["token_embedding"].T
