# Apple MLX GPT (M3 Air)

From-scratch inspectable GPT on MacBook Air M3 (8 GB unified memory). Host-side
CLI / tokenizer / NumPy reference come from [llm-gpu-8](https://github.com/dtelcore/llm-gpu-8);
the device layer is MLX ops + explicit VJPs (no autograd).

**v0.0.4** — process memory controller. Process budget: **2 GB** of unified
memory (hardcoded, not a CLI). Soft machine guard: **5.5 GB**.

```bash
# Python 3.11 or 3.12
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python setup/2_test_workspace.py          # Metal + matmul + memory APIs
python -m tests.parity.run_parity
python auto_train.py --config setup/story_c256_l6_config.json --steps 20 --no-prompt
```

Stable English recipe on this Air: `setup/story_c256_l6_config.json`
(C=256, L=6, T=256, batch 4, accum 4, GPT-2 residual scale). Smaller smoke:
`setup/story_sub1m_config.json` (C=128, L=4, T=128, batch 8).

Keep the lid open on this fanless Air; long GEMMs will thermal-throttle.

Live path: `model/mlx/ops.py` (forward primitives + hand VJPs). Do not use
`mx.value_and_grad`, `mlx.nn`, or `mlx.optimizers` on the train loop.

## Memory controller (2 GB)

`training/memory_controller.py` sizes **batch, context, activations, and KV**
to the 2 GB cap for any C/L/H/T. It never changes width or depth.

On train start it logs a plan, for example:

```text
[memory] budget=2048MB usable=1741MB estimate=517MB B=4 accum=4 T=256 ckpt=0 fp16=0 eval_per_layer=0 | no changes (already fits)
```

Knobs, cheapest quality impact first:

1. Realize the MLX graph after each layer (`eval_per_layer`)
2. Gradient checkpointing
3. FP16 storage for kept activations (compute stays FP32)
4. Smaller micro-batch, more `--grad-accum` (same tokens/step)
5. Shorter context `T` (last resort)

```bash
# default: autoscale inside 2 GB
python auto_train.py --config setup/story_c256_l6_config.json --no-prompt

# refuse instead of shrinking (old preflight)
python auto_train.py --config ... --no-autoscale

# more compile slack; still cannot exceed 2 GB
python auto_train.py --config ... --memory-headroom 0.20
```

`--no-grad-checkpoint` blocks checkpointing; the controller shrinks batch or
context instead. Generate clips `--max-new-tokens` to the remaining window.

If **weights + Adam** alone do not fit the usable budget, it still aborts.
That needs layer-weight streaming (not in 0.0.4).

## Layer parallel / pipeline (prep, not shipped)

`eval_per_layer` isolates each transformer block’s MLX graph so unused
intermediates can free. That is the sequential seam a later pipeline would
use: run layer *i*, drop its working set, optionally page its weights, run
*i+1*.

Not in this release:

- Swapping idle layer weights to host/disk
- True pipeline-parallel across processes
- Changing L or C to “fit” the cap

See `CHANGELOG.md` and CLI catalog `py_calls.md`.
