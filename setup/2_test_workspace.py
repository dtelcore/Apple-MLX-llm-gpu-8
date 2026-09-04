"""Metal + MLX workspace smoke: device, one matmul, memory APIs, 2 GB helper."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=" * 60)
print("RUNNING MLX / METAL WORKSPACE VERIFICATION")
print("=" * 60)

try:
    import mlx.core as mx
    import numpy as np

    if not mx.metal.is_available():
        raise RuntimeError("Metal backend is not available")

    from model.mlx import env

    env.configure()
    info = env.device_info()
    print("[SUCCESS] MLX module bindings: operational")
    print(f"[DEVICE]  architecture: {info.get('architecture') or info.get('device_name')}")
    mem_bytes = int(info.get("memory_size") or 0)
    print(f"[DEVICE]  unified memory (device_info): {mem_bytes / (1024 ** 3):.2f} GB")
    print(f"[BUDGET]  process hard cap: {env.PROCESS_BUDGET_BYTES / (1024 ** 3):.1f} GB")
    print(f"[BUDGET]  machine soft guard: {env.SOFT_MACHINE_BYTES / (1024 ** 3):.1f} GB")
except Exception as e:
    print("[FAILURE] Stage A (Metal / MLX init) failed.")
    print(f"ERROR: {e}")
    sys.exit(1)

print("-" * 60)

try:
    print("Attempting float32 matmul on Metal...")
    a = mx.array(np.array([[1, 2], [3, 4]], dtype=np.float32))
    b = mx.array(np.array([[5, 6], [7, 8]], dtype=np.float32))
    c = mx.matmul(a, b)
    mx.eval(c)
    result = np.asarray(c)
    print(f"Matmul result:\n{result}")
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.allclose(result, expected), f"mismatch: {result}"

    env.reset_peak_memory()
    usage = env.check_memory(where="smoke")
    print(
        f"[MEMORY] active={usage['process_used_bytes'] / (1024 ** 2):.2f} MB "
        f"peak={usage['peak_bytes'] / (1024 ** 2):.2f} MB source={usage['source']}"
    )
    print("-" * 60)
    print("[SUCCESS] MLX matmul + memory APIs passed.")
    print("[SUCCESS] MacBook Air M3 Metal path is ready (2 GB process budget).")
    print("=" * 60)
except Exception as e:
    print("[FAILURE] Stage B (matmul / memory) failed.")
    print(f"ERROR: {e}")
    sys.exit(1)
