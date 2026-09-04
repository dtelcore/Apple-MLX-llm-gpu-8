"""
model/mlx/env.py

Metal / unified-memory bootstrap for MacBook Air M3 (8 GB).

Process budget: never more than 2 GB of unified memory (weights + activations
+ ScratchPool + KV + optimizer). Soft machine guard at 5.5 GB if counters lag.
"""

from __future__ import annotations

from logging_config import logger

PROCESS_BUDGET_BYTES = 2 * 1024 ** 3
SOFT_MACHINE_BYTES = int(5.5 * 1024 ** 3)

_configured = False


class MemoryBudgetError(RuntimeError):
    """Raised when the 2 GB process budget or 5.5 GB machine guard is exceeded."""


def configure() -> None:
    """Idempotent Metal init + 2 GB MLX memory limit."""
    global _configured
    if _configured:
        return
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal backend is not available on this machine")

    # Ask MLX not to grow past the process budget. The train loop still aborts
    # if reported use exceeds 2 GB (limit can lag / under-count).
    if hasattr(mx, "set_memory_limit"):
        mx.set_memory_limit(PROCESS_BUDGET_BYTES)
    elif hasattr(mx.metal, "set_memory_limit"):
        mx.metal.set_memory_limit(PROCESS_BUDGET_BYTES)

    info = device_info()
    logger.info(
        "MLX Metal configured: arch=%s memory_size_gb=%.2f process_budget_gb=2.0",
        info.get("architecture") or info.get("device_name") or "unknown",
        float(info.get("memory_size", 0)) / (1024 ** 3),
    )
    _configured = True


def device_info() -> dict:
    import mlx.core as mx

    if hasattr(mx, "device_info"):
        return dict(mx.device_info())
    return dict(mx.metal.device_info())


def metal_available() -> bool:
    import mlx.core as mx

    return bool(mx.metal.is_available())


def _call_mem(name: str, *args):
    import mlx.core as mx

    fn = getattr(mx, name, None)
    if fn is not None:
        return fn(*args)
    metal_fn = getattr(mx.metal, name, None)
    if metal_fn is None:
        raise AttributeError(f"mlx has no {name} (tried mx.{name} and mx.metal.{name})")
    return metal_fn(*args)


def get_active_memory() -> int:
    return int(_call_mem("get_active_memory"))


def get_peak_memory() -> int:
    return int(_call_mem("get_peak_memory"))


def reset_peak_memory() -> None:
    _call_mem("reset_peak_memory")


def get_cache_memory() -> int:
    try:
        return int(_call_mem("get_cache_memory"))
    except AttributeError:
        return 0


def check_memory(where: str = "") -> dict:
    """Read MLX memory counters; abort if over the 2 GB / 5.5 GB guards.

    Returns a usage dict. ``mx.get_active_memory`` / ``get_peak_memory`` can
    lag or under-count — the 5.5 GB check is a last-resort machine backstop.
    """
    active = get_active_memory()
    peak = get_peak_memory()
    cache = get_cache_memory()
    loc = f" ({where})" if where else ""
    if active > PROCESS_BUDGET_BYTES or peak > PROCESS_BUDGET_BYTES:
        raise MemoryBudgetError(
            f"process unified memory exceeded 2 GB budget{loc}: "
            f"active={active / (1024 ** 2):.1f} MB peak={peak / (1024 ** 2):.1f} MB"
        )
    if active > SOFT_MACHINE_BYTES or peak > SOFT_MACHINE_BYTES:
        raise MemoryBudgetError(
            f"machine unified memory exceeded 5.5 GB soft guard{loc}: "
            f"active={active / (1024 ** 2):.1f} MB peak={peak / (1024 ** 2):.1f} MB"
        )
    return {
        "process_used_bytes": int(active),
        "peak_bytes": int(peak),
        "cache_bytes": int(cache),
        "driver_free_bytes": max(0, PROCESS_BUDGET_BYTES - int(active)),
        "driver_total_bytes": int(PROCESS_BUDGET_BYTES),
        "driver_used_bytes": int(active),
        "source": "mlx",
    }
