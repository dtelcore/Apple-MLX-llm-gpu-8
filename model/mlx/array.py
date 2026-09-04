"""Mutable wrapper around mx.array so in-place Kepler ops keep object identity."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import mlx.core as mx


class _Flags:
    __slots__ = ("c_contiguous",)

    def __init__(self, c_contiguous: bool = True) -> None:
        self.c_contiguous = bool(c_contiguous)


def _to_mx_dtype(dtype) -> Any:
    dt = np.dtype(dtype)
    if dt == np.float32:
        return mx.float32
    if dt == np.float16:
        return mx.float16
    if dt == np.int32:
        return mx.int32
    if dt == np.int64:
        return mx.int64
    if dt == np.uint64:
        return mx.uint64
    if dt == np.bool_ or dt == bool:
        return mx.bool_
    return mx.float32


def _np_dtype_from_mx(arr: mx.array) -> np.dtype:
    mapping = {
        mx.float32: np.float32,
        mx.float16: np.float16,
        mx.int32: np.int32,
        mx.int64: np.int64,
        mx.uint32: np.uint32,
        mx.bool_: np.bool_,
    }
    return np.dtype(mapping.get(arr.dtype, np.float32))


class DeviceArray:
    """gpuarray-shaped handle: ``.mx`` is the live ``mlx.core.array``."""

    __slots__ = ("_x", "_flags")

    def __init__(self, x, *, c_contiguous: bool = True) -> None:
        if isinstance(x, DeviceArray):
            self._x = x._x
        elif isinstance(x, mx.array):
            self._x = x
        else:
            host = np.ascontiguousarray(x)
            self._x = mx.array(host)
        self._flags = _Flags(c_contiguous)

    def replace(self, new) -> "DeviceArray":
        self._x = new._x if isinstance(new, DeviceArray) else new
        return self

    @property
    def mx(self) -> mx.array:
        return self._x

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self._x.shape)

    @property
    def dtype(self):
        return _np_dtype_from_mx(self._x)

    @property
    def ndim(self) -> int:
        return int(self._x.ndim)

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return int(n)

    @property
    def nbytes(self) -> int:
        return int(self.size * np.dtype(self.dtype).itemsize)

    @property
    def flags(self) -> _Flags:
        return self._flags

    @property
    def gpudata(self) -> int:
        """Dummy attribute so Kepler ``hasattr(..., 'gpudata')`` checks pass."""
        return id(self)

    @property
    def T(self) -> "DeviceArray":
        return DeviceArray(mx.transpose(self._x), c_contiguous=False)

    def reshape(self, *shape) -> "DeviceArray":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return DeviceArray(mx.reshape(self._x, tuple(int(s) for s in shape)))

    def ravel(self) -> "DeviceArray":
        return DeviceArray(mx.reshape(self._x, (-1,)))

    def astype(self, dtype) -> "DeviceArray":
        return DeviceArray(self._x.astype(_to_mx_dtype(dtype)))

    def get(self) -> np.ndarray:
        mx.eval(self._x)
        return np.asarray(self._x)

    def __getitem__(self, key) -> "DeviceArray":
        return DeviceArray(self._x[key])

    def __array__(self, dtype=None) -> np.ndarray:
        mx.eval(self._x)
        out = np.asarray(self._x)
        if dtype is not None:
            return out.astype(dtype, copy=False)
        return out


def empty(shape, dtype=np.float32) -> DeviceArray:
    return DeviceArray(mx.zeros(tuple(int(s) for s in shape), dtype=_to_mx_dtype(dtype)))


def zeros(shape, dtype=np.float32) -> DeviceArray:
    return empty(shape, dtype=dtype)


def as_mx(x) -> mx.array:
    if isinstance(x, DeviceArray):
        return x.mx
    if isinstance(x, mx.array):
        return x
    return mx.array(np.ascontiguousarray(x))


def wrap(x, *, c_contiguous: bool = True) -> DeviceArray:
    if isinstance(x, DeviceArray):
        return x
    return DeviceArray(x, c_contiguous=c_contiguous)
