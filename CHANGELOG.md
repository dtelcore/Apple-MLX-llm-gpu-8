# Changelog

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
