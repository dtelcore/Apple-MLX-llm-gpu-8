# Changelog

## 0.0.5

Sequential layer streaming so L=32–48 at C=256 can train under the hardcoded **2 GB** Metal cap.

- `layer_strategy=stream` keeps embeddings + final LN resident and pages one transformer
  block (weights + Adam m/v + VJP working set) onto Metal at a time. Idle layer Adam
  moments stay on host NumPy. Residual-stream checkpoints `h_in` stay on device (~48 MB
  at L=48, B=4, T=256, C=256); host offload of `h` is not in this release.
- Train: stream-forward → loss → recompute backward → global clip on the host grad dict
  → stream Adam (`t += 1` once, then `step_layer`). Generate prefill/decode load/unload
  the same way and pack KV into arenas before dropping a block. CUDA-graph decode is
  disabled while streaming.
- Autoscale enables stream + `eval_per_layer` **before** shrinking context `T`.
  `--layer-stream` forces it (L=6 bring-up). `--no-layer-stream` keeps the old refuse.
  Generate estimates no longer subtract Adam (that underflowed to negative MB on stream).
- Expect **2–4×** lower tok/s versus a fully resident stack, plus fanless thermal
  throttle on long GEMMs. GPT-2 residual scale `1/√(2L)` stays mandatory for deep stacks.

## 0.0.4

Process memory controller and a sequential-layer seam for later pipeline / layer parallel.

- Autoscale batch, context, gradient checkpointing, FP16 activation storage, and
  per-layer MLX `eval` to the hardcoded **2 GB** process cap (`model/mlx/env.py`).
  Architecture (C, L, H) is never changed. `--no-autoscale` keeps refuse-if-over.
  `--memory-headroom` (default 0.15) reserves compile/scratch inside that cap.
- `eval_per_layer` realizes the residual stream after each block so unused
  intermediates can free. That is the hook for future layer-pipeline / weight
  streaming; swapping idle layer weights is not implemented yet. If
  weights+Adam alone exceed the usable budget, training still refuses.
- Train and generate both go through `training/memory_controller.py`. The plan
  is logged as `[memory] ...` and stored on the run config as `memory_plan`.

## 0.0.3

Faster tokenizer path and safer M3-Air training since v0.0.2.

- Incremental BPE merge training on unique-word counts (4000 merges on 200k chars in ~1s).
- Parallel unique-word BPE encode (`LLM_BPE_WORKERS`) with tqdm bars for merges, scan, encode, and stitch.
- Memory preflight before weight allocation; first-run config/corpus seeding; training-log plotter fixes.

## 0.0.2

Apple MLX port: explicit VJPs, Metal train/generate, 2 GB process budget.

## 0.0.1

Planning.
