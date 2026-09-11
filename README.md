# Apple MLX GPT (M3 Air)

From-scratch inspectable GPT on MacBook Air M3 (8 GB unified memory). Host-side
CLI / tokenizer / NumPy reference come from [llm-gpu-8](https://github.com/dtelcore/llm-gpu-8);
the device layer is MLX ops + explicit VJPs (no autograd).

**v0.0.7** — chat **router** (cabinet → calc → Wikipedia) under the **2 GB**
process budget (hardcoded, not a CLI). Soft machine guard: **5.5 GB**. Fact
pipelines are from 0.0.6; layer streaming is from 0.0.5.

```bash
# Python 3.11 or 3.12
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python setup/2_test_workspace.py          # Metal + matmul + memory APIs
python -m tests.parity.run_parity
python auto_train.py --config setup/story_c256_l6_config.json --steps 20 --no-prompt
python generate_config.py          # write a setup/*.json recipe (C/H/L/T/B)
python tools/wikidata_to_facts.py  # optional: SPARQL → data/facts/wikidata_facts.txt
python tools/wikidata_to_facts2.py # optional: tech/health/maths → data/facts/*_facts.txt
python tools/make_fact_mix.py      # repeat facts → data/chat_facts.jsonl
python auto_train.py --config setup/chat_facts_config.json --checkpoint output/checkpoints/chat_facts_v5 --steps 1000 --no-prompt
python interactive.py --checkpoint output/checkpoints/chat_facts_v4 --chat
```

Chat checkpoints default to a **Python router**: exact cabinet hit → generate the
stored `User: … Assistant:` line; `2+2` → Decimal calc; unknown `What is …` →
Wikipedia summary; otherwise a miss plus a hint to a **separate** story REPL
(`--no-router`). Never load two checkpoints in one process. `--no-search` skips
the network. Do not `combine` `data/*.txt`.

Stable English recipe on this Air: `setup/story_c256_l6_config.json`
(C=256, L=6, T=256, batch 4, accum 4, GPT-2 residual scale). Smaller smoke:
`setup/story_sub1m_config.json` (C=128, L=4, T=128, batch 8).

Chat cabinet (memorize your Q&A, not Wikipedia): native lines in
`data/user_facts.txt` and `data/facts/*.txt`, mix with `tools/make_fact_mix.py`,
train `setup/chat_facts_config.json` on `data/chat_facts.jsonl` only. Do not
`combine` `data/*.txt`. Probe with the **exact** `User:` wording and `--stop User:`.
A few dozen facts at ~300 repeats stick; hundreds of unique facts at 20 repeats do not.

Keep the lid open on this fanless Air; long GEMMs will thermal-throttle.

Live path: `model/mlx/ops.py` (forward primitives + hand VJPs). Do not use
`mx.value_and_grad`, `mlx.nn`, or `mlx.optimizers` on the train loop.

## Memory controller (2 GB)

`training/memory_controller.py` sizes **batch, context, activations, and KV**
to the 2 GB cap for any C/L/H/T. It never changes width or depth.

On train start it logs a plan, for example:

```text
[memory] budget=2048MB usable=1741MB estimate=517MB B=4 accum=4 T=256 ckpt=0 fp16=0 eval_per_layer=0 layers=resident | no changes (already fits)
```

Knobs, cheapest quality impact first:

1. Realize the MLX graph after each layer (`eval_per_layer`)
2. Gradient checkpointing
3. FP16 storage for kept activations (compute stays FP32)
4. Sequential **layer streaming** (one block + Adam on Metal)
5. Smaller micro-batch, more `--grad-accum` (same tokens/step)
6. Shorter context `T` (last resort)

```bash
# default: autoscale inside 2 GB (may enable stream before shrinking T)
python auto_train.py --config setup/story_c256_l6_config.json --no-prompt

# force stream (L=6 bring-up / deep stacks)
python auto_train.py --config ... --layer-stream --no-prompt

# never stream; shrink T or refuse
python auto_train.py --config ... --no-layer-stream --no-prompt

# refuse instead of shrinking (old preflight)
python auto_train.py --config ... --no-autoscale

# more compile slack; still cannot exceed 2 GB
python auto_train.py --config ... --memory-headroom 0.20
```

`--no-grad-checkpoint` blocks checkpointing; the controller shrinks batch or
enables stream before cutting context. Generate clips `--max-new-tokens` to the
remaining window.

Streaming drops **Metal** copies of idle layers. Host NumPy still holds every
weight and idle Adam `m/v` (about 300 MB at L=32 C=256); that is fine. Peak
Metal is O(1) in depth. Residual checkpoints stay on device; moving `h_in` to
host is a later fallback if someone combines extreme L with large B.

Expect **2–4×** lower tok/s than a resident L=6 stack, and keep the lid open —
fanless Airs thermal-throttle on long GEMMs. GPT-2 residual scale `1/√(2L)` is
mandatory for deep stacks.

## Layer streaming (Phase 1)

`layer_strategy=stream` loads one transformer block, runs it, then
`unload_layer` (including `scratch_pool.clear()`). The same unfused block is
used for resident and stream so logits/grads match. Generate packs K/V into
per-layer arenas before unload so L=32 decode does not upload the full stack.

Not in this release:

- Prefetch / keeping 2–4 hot layers
- `np.memmap` of params or Adam
- Activation offload of residual `h`
- True pipeline-parallel across processes
- Changing L or C to “fit” the cap

See `CHANGELOG.md` and CLI catalog `py_calls.md`.
