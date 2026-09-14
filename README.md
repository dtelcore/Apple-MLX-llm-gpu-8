# Apple MLX GPT (M3 Air)

From-scratch inspectable GPT on MacBook Air M3 (8 GB unified memory). Host-side
CLI / tokenizer / NumPy reference come from [llm-gpu-8](https://github.com/dtelcore/llm-gpu-8);
the device layer is MLX ops + explicit VJPs (no autograd).

**v0.0.9** — unguided trainer + autotrainer daemon. **v0.0.8** — **App.py** (chat + weight viewer + model selector), related
cabinet follow-ups, pick-a-neuron. Router is from 0.0.7 (cabinet → calc →
Wikipedia). **2 GB** process budget (hardcoded). Soft machine guard: **5.5 GB**.
Fact pipelines are from 0.0.6; layer streaming is from 0.0.5.


## Where this sits vs GPT-2

Hand VJPs, a hardcoded **2 GB** Metal budget (layer streaming), and a Python router
(cabinet → calc → Wikipedia) make this an **instrumented research GPT**, not a
black-box mini ChatGPT. Forward cost scales roughly like `L * T * C^2`;
precise TFLOPs depend on stream vs resident and are not quoted here.

| | **chat_facts_v4** | **chat_facts_v7** | **GPT-2 small** |
|---|---|---|---|
| Width / depth | C=512 · **L=6** · H=8 | C=512 · **L=16** · H=8 | C=768 · L=12 · H=12 |
| Context `T` | 256 | 512 | 1024 |
| Vocab | ~1587 (mix BPE) | 4086 | 50257 |
| Params (counted) | **20.5M** | **54.6M** | ~124M |
| Weights on disk | ~82 MB | ~219 MB | (release artifacts) |
| Pos / norm | RoPE · RMSNorm | RoPE · RMSNorm | learned pos · LayerNorm |
| Gradients | explicit VJPs | explicit VJPs | framework autograd |
| Layer strategy | stream | stream | resident |
| Train steps / loss | 400 · ~0.033 | 1000 · ~0.039 | web-text LM (not comparable) |
| Objective | closed fact cabinet | linked cabinet + related UX | open web text |
| Best for | clean recitation (France→Paris) | product checkpoint + App | general English |

**v7 ≈ half of GPT-2 small in params**, half the context, ~1/12 the vocab — in the
“serious small transformer” band — but trained for **recitation**, not open-ended LM.
**v4** remains the cleanest small-cabinet baseline; do not `--resume` v4/v6/v7 across
vocab or mix changes. Do not train more v7 or rebuild its mix for alias fixes.


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
python App.py
python App.py --checkpoint output/checkpoints/chat_facts_v6 --chat
python webui.py --checkpoint output/checkpoints/chat_facts_v5 --chat
python npzviewer.py --open output/checkpoints/chat_facts_v6/weights.npz
python trainmon.py
python unguided_trainer.py --config setup/chat_facts_v7_config.json --policy setup/unguided_v7_policy.json --dry-run
python unguided_prober.py --name Unguarded-Initialv7-Run-2 --dry-run
```

Chat checkpoints default to a **Python router**: exact cabinet hit → generate the
stored `User: … Assistant:` line; `2+2` → Decimal calc; after a generate turn,
unknown follow-ups list **related trained questions** (UI chips / `:related`)
instead of auto-Wikipedia; `:search` or a cold-start `What is …` still hits
Wikipedia. Wikipedia hits are appended to `output/cabinet_learned.jsonl`
and replayed later; they are not trained into the checkpoint. Central UI:
`python App.py` (http://127.0.0.1:7860) — model selector loads one checkpoint
into chat (Metal) and npzviewer (mmap). The **Train** tab reads logs only.
Switching models restarts the process. Standalone `webui.py` / `npzviewer.py`
/ `trainmon.py` (7862) still work; do not run chat UIs next to `App.py` or
`interactive.py`. `trainmon.py` is the safe watcher beside an unguided train
(App must not Load a checkpoint onto Metal then). Never load two checkpoints
in one process.
`--no-search` skips the network. Do not `combine` `data/*.txt`. Do not
`--resume` v6 into v7.

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


## Progress (0.0.1 → 0.0.9)

Software (see `CHANGELOG.md` for detail):

| Ver | Leap |
|---|---|
| 0.0.1–0.0.2 | Plan → MLX port (explicit VJPs, 2 GB budget) |
| 0.0.3 | Fast BPE + train preflight |
| 0.0.4 | Memory controller (never changes C/L/H) |
| 0.0.5 | Sequential layer streaming |
| 0.0.6 | Fact pipelines + Wikidata packs |
| 0.0.7 | Router (cabinet → calc → Wikipedia) + web UI |
| 0.0.8 | `App.py` (chat + pick-a-neuron + selector), related chips, topic/`the` aliases |
| 0.0.9 | Unguided trainer kernel + autotrainer daemon + train monitor |

Cabinet checkpoints under `output/checkpoints/`:

| CKPT | Shape | Params | Notes |
|---|---|---:|---|
| v1 | C512 L6 T256 | — | first cabinet |
| v2 | C512 L6 · small V | — | recitation emerging |
| v3 | C512 L12 | — | deeper on dirty multi-answer mix → worse |
| **v4** | C512 L6 | **20.5M** | fair mix; best clean France→Paris |
| v5 | C512 L6 | — | broader 20× topics |
| v6 | C512 L16 T512 | — | linked/learned fold-in |
| **v7** | C512 L16 T512 | **54.6M** | current product; do not train more |
| `user_facts_tiny` | C16 L2 T16 | 0.02M | weight inspector only |

Lessons: dirty multi-answer mixes fail; fair one-answer mixes stick; router aliases
fix **lookup**, not binding; never `--resume` across vocab / arch / mix changes.

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
