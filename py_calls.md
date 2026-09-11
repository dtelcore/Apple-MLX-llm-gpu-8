# py_calls.md — runnable entry points

**Apple MLX (this tree, v0.0.7):** MacBook Air M3, 2 GB process cap. Start with
[`README.md`](README.md). Activate `venv/` then `python setup/2_test_workspace.py`.

Kepler GT 730 host-CLI notes remain below; device path is MLX, not PyCUDA.

Shared flag groups live in [`cli_common.py`](cli_common.py) and are referenced below as **(shared: …)**.

---

## Shared flag groups (`cli_common.py`)

### Tracing `(shared: trace)`

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--verbose` | flag | off | Token + top-k logit traces |
| `--trace-logits` | flag | off | Dump top-k logits |
| `--trace-tokens` | flag | off | Dump token ↔ id |
| `--trace-neurons` | flag | off | Per-layer activation stats |
| `--trace-vectorization` | flag | off | GEMM shapes / CUDA grid |
| `--trace-every` | int | `None` | Every N steps (train default ≈ 10% of steps; generate/interactive default every step) |

### Runtime observability `(shared: obs)` — Stage 3.1, off by default

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--runtime-metrics` | flag | off | Extra `[train]` fields; meter host↔device sync |
| `--memory-timeline` | flag | off | ScratchPool JSONL under `output/logs/`; implies metrics |

### Config / seed / checkpoint

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--config` | str | `output/configs/training_config.json` | `(shared: config)` |
| `--checkpoint` | str | `output/checkpoints/run1` | `(shared: checkpoint)` |
| `--seed` | int | `42` | `(shared: seed)` |

### Training length `(shared: length)`

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--learning-rate` | float | `None` | Prompted if omitted (unless `--no-prompt`) |
| `--epochs` | int | `None` | Ignored if `--steps` set |
| `--steps` | int | `None` | Total steps across all epochs |
| `--log-every` | int | `100` | Progress line cadence |
| `--checkpoint-every` | int | `None` | Save every N steps |
| `--min-lr-ratio` | float | `None` | Cosine LR floor vs base after warmup (default **0.1** in train wiring) |
| `--window-stride` | int | `None` | Stride between sliding windows (preset **64**; else hyperparams or **1**) |
| `--val-every` | int | `0` | Val loss every N steps without quarterly I/O (**0**=off) |
| `--no-prompt` | flag | off | No interactive prompts |

### Model hyperparameters `(shared: model)`

| Flag | Type | Default |
|------|------|---------|
| `--embedding-dim` | int | `None` |
| `--num-heads` | int | `None` |
| `--num-layers` | int | `None` |
| `--max-len` | int | `None` |
| `--dropout` | float | `None` |
| `--batch-size` | int | `None` |
| `--weight-decay` | float | `None` |
| `--warmup-steps` | int | `None` |
| `--gradient-clip` | float | `None` |
| `--grad-accum` / `--gradient-accumulation-steps` | int | `None` (config or 1) |
| `--tie-embeddings` | flag | off (presets default tied) |
| `--no-tie-embeddings` | flag | off |
| `--run-budget` | int | `None` (absolute quarterly budget) |
| `--norm-type` | `layernorm`\|`rmsnorm` | `None` (presets → rmsnorm) |
| `--pos-encoding` | `learned`\|`rope` | `None` (presets → rope) |
| `--grad-checkpoint` | flag | off (controller may enable under 2 GB) |
| `--no-grad-checkpoint` | flag | off (then controller shrinks B/T instead) |
| `--no-autoscale` | flag | off (refuse if 2 GB estimate overflows) |
| `--memory-headroom` | float | `0.15` (compile slack; does not raise the 2 GB cap) |
| `--layer-stream` | flag | off (force one-block Metal streaming) |
| `--no-layer-stream` | flag | off (never autoscale into stream; shrink T or refuse) |

v0.0.5+: `training/memory_controller.py` autoscales batch/context/activations and may enable sequential layer streaming **before** shrinking `T`. Architecture is never changed. See [`README.md`](README.md). Residual checkpoints stay on device; expect 2–4× tok/s vs resident L=6.

v0.0.6: chat facts train `setup/chat_facts_config.json` → `data/chat_facts.jsonl` only. `combine: true` concatenates every `data/*.txt` and overwrites the cabinet.

v0.0.7: `interactive.py` router (cabinet → calc → Wikipedia). Chat checkpoints default `--router` on. Do not load a second checkpoint in that process.

### Generate probes `(shared: probe)`

| Flag | Type | Default |
|------|------|---------|
| `--no-generate-probe` | flag | off |
| `--generate-probe-prompt` | str | `once upon a` |
| `--generate-probe-tokens` | int | `256` |

### Quality trial `(shared: quality)`

| Flag | Type | Default |
|------|------|---------|
| `--quality-trial` | flag | off |
| `--no-quality-trial` | flag | off |
| `--quality-prompt` | str | `None` |
| `--quality-weights` | str | `None` |
| `--quality-mode` | `story`\|`chat`\|`both` | `None` (`chat_5m` → chat) |
| `--compare-quarters` | flag | off |
| `--set-best` | str | `None` (e.g. `quarter_50`) |

### Tokenizer `(shared: tokenizer)`

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--tokenizer` | `bpe`\|`char` | `bpe` (new runs) | Resume loads checkpoint `vocab.json` type |
| `--bpe-merges` | int | `200` | Only for new BPE builds |

### `--menu` flag-group picker

`train.py --menu`, `auto_train.py --menu`, and `generate.py --menu` call
`cli_common.prompt_run_flag_menu`. Enter = keep defaults for unlisted groups;
selected groups prompt each flag with Enter-to-accept (booleans as `y/N` / `Y/n`).

| Entry | Groups |
|-------|--------|
| train | Tokenizer*, Length, Model*, Obs, Trace, Probe, Quality, Plot |
| auto_train | same + Smoke generate + Decode (no Plot) |
| generate | Sampling, Decode (KV / cuda-graph), Trace |

\* Tokenizer / Model only on fresh runs (not resume).

---

## Training & generation

### `train.py`

```text
python train.py [flags]
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| *(shared: config, checkpoint, seed, length, model, trace, obs, probe, quality, tokenizer)* | | | |
| `--menu` | flag | off | Wizard + flag-group picker |
| `--data-dir` | str | `data` | Datasets for `--menu` |
| `--models-dir` | str | `output/checkpoints` | Checkpoint scan for menu / `--generate` |
| `--generate` | flag | off | Skip train; generation REPL |
| `--resume` | flag | off | Resume from `--checkpoint` |
| `--temperature` | float | probe default | Probes / `--generate` |
| `--top-k` | int | probe default | |
| `--top-p` | float | probe default | |
| `--plot` | flag | off | Post-train log / landscape plots |

**Examples**

```powershell
# Recommended Fast Stories (<1M) run (wizard preset 2), TinyStories (preset 3), or Chat 5M (preset 4) — see guide.md
python train.py --menu

python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 --steps 5000 --no-prompt
python train.py --resume --checkpoint output\checkpoints\ts_run --val-every 500 --steps 2000 --no-prompt
python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 --runtime-metrics --memory-timeline --no-prompt --steps 200
python train.py --tokenizer char --menu   # opt into character tokenizer
python train.py --generate --models-dir output\checkpoints
python train.py --compare-quarters --checkpoint output\checkpoints\BiggerTest256256
```

---

### `auto_train.py`

Train then smoke-generate.

```text
python auto_train.py [flags]
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| *(shared: config, checkpoint, seed, length, model, probe, quality, trace, obs, tokenizer)* | | | |
| `--resume` | flag | off | |
| `--prompt` | str | `the` | Smoke sample seed |
| `--max-new-tokens` | int | `80` | Smoke sample length |
| `--temperature` | float | probe default | **0.6** |
| `--top-k` | int | probe default | **10** |
| `--top-p` | float | probe default | **0.9** |
| `--no-kv-cache` | flag | off | Smoke generate: full recompute each token |
| `--cuda-graph` | flag | off | Smoke generate: KV kernel-chain graph |
| `--menu` | flag | off | Wizard + flag groups (incl. smoke generate / decode) |
| `--data-dir` | str | `data` | |
| `--models-dir` | str | `output/checkpoints` | |

---

### `generate_config.py`

Interactive (or `--no-prompt`) writer for `setup/*.json` recipes. Pick C / H / L / T / B / accum, residual scale, layer stream, and dataset; prints a 2 GB train estimate. Does not start training. New C/L/H needs a **new** checkpoint (do not `--resume` `chat8b`).

```text
python generate_config.py
python generate_config.py --from setup/chat_c256_l6_config.json --embedding-dim 384 --num-layers 8 --no-prompt --output setup/chat_c384_l8_config.json
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--from` | str | `setup/chat_c256_l6_config.json` when `--no-prompt` | Base recipe |
| `--output` | str | `setup/{kind}_c{C}_l{L}_config.json` | Destination |
| `--embedding-dim` | int | from base | **C** |
| `--num-heads` | int | from base | **H** (must divide C) |
| `--num-layers` | int | from base | **L** (L≥6 requires residual scale) |
| `--max-len` | int | from base | **T** |
| `--batch-size` | int | from base | **B** |
| `--grad-accum` | int | from base | |
| `--residual-scale` | `on`\|`off` | from base / on if L≥6 | |
| `--layer-strategy` | `resident`\|`stream` | from base | |
| `--dataset` | str | from base | e.g. `chat_train`, `data_dir` |
| `--combine` / `--no-combine` | flag | from base | `data/*.txt` concat |
| `--no-prompt` | flag | off | Flags only |
| `--force` | flag | off | Overwrite existing JSON |

### `tools/make_fact_mix.py`

Repeat native `User:` / `Assistant:` lines into the chat cabinet. Loads
`data/user_facts.txt` and `data/facts/*.txt`. Default wiki slice from
`chat_train.txt` is **off** (`--max-wiki-facts 0`). Writes `data/chat_facts.txt`
and `data/chat_facts.jsonl`. Train those with `setup/chat_facts_config.json`,
not `chat_c256_l6_config.json`.

```text
python tools/make_fact_mix.py
python tools/make_fact_mix.py --user-repeat 300 --wiki-repeat 10 --max-wiki-facts 0
python tools/make_fact_mix.py --learned output/cabinet_learned.jsonl --user-repeat 20 --max-wiki-facts 0 \
  --output data/chat_facts_v6.txt --jsonl data/chat_facts_v6.jsonl
```

| Flag | Type | Default |
|------|------|---------|
| `--user-facts` | str | `data/user_facts.txt` |
| `--chat-train` | str | `data/chat_train.txt` |
| `--output` | str | `data/chat_facts.txt` |
| `--jsonl` | str | `data/chat_facts.jsonl` |
| `--learned` | str | (off) |
| `--max-learned-chars` | int | `512` |
| `--max-wiki-facts` | int | `0` |
| `--user-repeat` | int | `300` |
| `--wiki-repeat` | int | `10` |

v6 (cabinet packs + unique learned extras, new BPE, do not `--resume` v5):

```text
python auto_train.py --config setup/chat_facts_v6_config.json \
  --checkpoint output/checkpoints/chat_facts_v6 \
  --steps 1000 --run-budget 16000 --no-prompt --log-every 1 \
  --prompt "User: What is the capital of France? Assistant:" \
  --stop "User:"
```

### `tools/wikidata_to_facts.py`

SPARQL groups → native chat lines under `data/facts/` (picked up by make_fact_mix).
Keeps LIMIT modest. Drops Q-id labels and same-question conflicts. Uses stdlib
urllib; `SPARQLWrapper` if installed.

```text
python tools/wikidata_to_facts.py
python tools/wikidata_to_facts.py --queries capitals elements --limit 40
python tools/wikidata_to_facts.py --output data/facts/my_wikidata.txt
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--queries` | str+ | `capitals elements` | `capitals`, `inventors`, `birth_years`, `elements` |
| `--output` | str | `data/facts/wikidata_facts.txt` | |
| `--sleep` | float | `1.0` | Pause between groups |
| `--limit` | int | query default | Override SPARQL LIMIT |

### `tools/wikidata_to_facts2.py`

Second SPARQL pack (`wikidatafetch2`): technology, health, and maths into
**separate** files under `data/facts/` (`tech_facts.txt`, `health_facts.txt`,
`maths_facts.txt`). Distinct question templates per kind. Drops any User: line
that has two different answers. Health lines are encyclopedic only (vitamins,
organs, pathogens, amino acids — no dosing or treatment advice). Modest LIMITs.
Same network stack as the first tool.

```text
python tools/wikidata_to_facts2.py
python tools/wikidata_to_facts2.py --domains technology maths --limit 25
python tools/wikidata_to_facts2.py --output-dir data/facts --sleep 1
python tools/wikidata_to_facts2.py --queries lang_designers si_units --tech-output data/facts/tech_facts.txt
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--domains` | str+ | `technology health maths` | Which packs to write |
| `--queries` | str+ | all kinds in those domains | Optional SPARQL subset |
| `--output-dir` | str | `data/facts` | Writes `tech_facts.txt` / `health_facts.txt` / `maths_facts.txt` |
| `--tech-output` | str | `<output-dir>/tech_facts.txt` | |
| `--health-output` | str | `<output-dir>/health_facts.txt` | |
| `--maths-output` | str | `<output-dir>/maths_facts.txt` | |
| `--sleep` | float | `1.0` | Pause between groups |
| `--limit` | int | query default | Override SPARQL LIMIT |

---

### `generate.py`

```text
python generate.py [flags]
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| *(shared: checkpoint, seed, trace)* | | | |
| `--prompt` | str | `the` | Story prompts like `once upon a` / `lilly was` read better |
| `--max-new-tokens` | int | `80` | Use 256 to match probes |
| `--temperature` | float | `0.8` | **0.6** reads better on these story ckpts |
| `--top-k` | int | `None` | **10** (with top-p) |
| `--top-p` | float | `None` | **0.9** (with top-k) |
| `--stop` | str (repeatable) | `None` | Stop when the string appears in new text |
| `--no-kv-cache` | flag | off | Full recompute each token |
| `--cuda-graph` | flag | off | Stage 4: capture KV kernel-chain graph; full decode stays eager GPU |
| `--menu` | flag | off | Pick checkpoint + Sampling / Decode / Trace groups |
| `--models-dir` | str | `output/checkpoints` | For `--menu` |
| `--no-prompt` | flag | off | Skip flag prompts when used with `--menu` |

`--menu` Sampling group: Enter keeps CLI defaults (`0.8`, no top-k/p). For readable stories set **temp 0.6 / top-k 10 / top-p 0.9** (same as train probes). Each run writes `output/logs/generate_<checkpoint>_<timestamp>.log`.

```powershell
python generate.py --menu
python generate.py --checkpoint output\checkpoints\BiggerTest256256 --prompt "once upon a" --max-new-tokens 256 --temperature 0.6 --top-k 10 --top-p 0.9
python generate.py --checkpoint output\checkpoints\BiggerTest256256 --cuda-graph --max-new-tokens 128
```

---

### `interactive.py`

REPL; session commands: `:temp`, `:tokens`, `:topk`, `:topp`, `:trace on|off`, `:quit`. Chat mode adds `:clear`, `:system`. Router adds `:search`, `:calc`, `:route`.

Chat checkpoints default **`--router`**: cabinet exact hit → generate stored `User: … Assistant:` (temp 0.2); else calc; else Wikipedia; else miss. Wikipedia hits append to `output/cabinet_learned.jsonl` and replay verbatim next time (not trained into the net). Story checkpoints default generate-every-turn. One checkpoint only (2 GB). `--no-search` skips the network.

```text
python interactive.py --checkpoint output/checkpoints/chat_facts_v4 --chat
python interactive.py --checkpoint output/checkpoints/chat_facts_v4 --no-router
python interactive.py --checkpoint output/checkpoints/<story> --no-router
```

| Flag | Type | Default |
|------|------|---------|
| *(shared: checkpoint, seed, trace)* | | |
| `--chat` | flag | off (auto-on for chat-named checkpoints) |
| `--no-chat` | flag | off |
| `--system` | str | simple-assistant line in chat mode |
| `--stop` | str (repeatable) | `User:` markers in chat mode |
| `--temperature` | float | `0.8` (chat: **0.7**; cabinet route uses **0.2**) |
| `--max-new-tokens` | int | `80` |
| `--top-k` | int | `None` (chat: **32**) |
| `--top-p` | float | `None` (chat: **0.9**) |
| `--router` / `--no-router` | flags | chat name → on; story → off |
| `--facts` | str | `data/chat_facts.jsonl` |
| `--learned` | str | `output/cabinet_learned.jsonl` |
| `--no-search` | flag | off |
| `--no-kv-cache` / `--cuda-graph` | flags | KV on; graph off |

---

### `setup/training_setup.py`

Interactive training setup wizard (also reachable via `train.py --menu`).

```text
python setup/training_setup.py [--data-dir DIR]
```

| Flag | Type | Default |
|------|------|---------|
| `--data-dir` | str | `data` |

---

## Benchmarks (mostly fixed hyperparameters)

### `bench_step.py`

No CLI flags. Benches `minimal` + `tiny_english` with hardcoded small model; requires metrics **disabled**.

```text
python bench_step.py
```

### `bench_profile.py`

No CLI flags. Profiles forward / backward / optimizer / sync split.

```text
python bench_profile.py
```

### `bench_mlp_fusion.py`

No CLI flags. Compares matmul+GELU path vs `fused_mlp_row`.

```text
python bench_mlp_fusion.py
```

### `tools/bench_generate.py`

Stage 3.2 generate latency (KV on vs off).

```text
python tools/bench_generate.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--checkpoint` | str | `None` (tiny random model if omitted) |
| `--prompt` | str | `once upon a time` |
| `--max-new-tokens` | int | `256` |
| `--seed` | int | `42` |
| `--out` | str | `output/baselines/stage32_kv_generate.json` |

```powershell
python tools\bench_generate.py --checkpoint output\checkpoints\BiggerTest256256 --max-new-tokens 256
```

### `tools/make_story_chat.py`

Wrap TinyStories lines as `User:` / `Assistant:` documents for mixed `chat_5m` training.

```text
python tools/make_story_chat.py [--input PATH] [--output PATH] [--keep-raw 0.2] [--max-docs N]
```

```powershell
python tools\make_story_chat.py --input data\tiny_stories.txt --output data\story_chat.txt
```

---

## Plotters & logs

### `training_log_plotter.py`

```text
python training_log_plotter.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--logs` | paths… | |
| `--log-dir` | str | `output/logs` |
| `--multi` | flag | off |
| `--all-runs` | flag | off |
| `--keep-short` | flag | off |
| `--min-points` | int | (module default) |
| `--max-step-gap` | int | (module default) |
| `--tail-lines` | int | (module default; `0` = entire file) |
| `--metric` | str | `tok/s` |
| `--smooth-window` | int | `21` |
| `--ema-alpha` | float | `0.08` |
| `--raw-alpha` | float | `0.10` |
| `--forecast-window` | int | `40` |
| `--no-forecast` | flag | off |
| `--forecast-raw` | flag | off |
| `--show-raw-loss` | flag | off |
| `--show-ema-loss` | flag | off |
| `--hide-raw-metric` | flag | off |
| `--select` | flag | off |
| `--live` | flag | off |
| `--refresh-seconds` | float | `1.0` |
| `--save` | path | |
| `--show` | flag | off |

### `loss_landscape_plotter.py`

```text
python loss_landscape_plotter.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--log-dir` | str | `output` |
| `--all-runs` | flag | off |
| `--keep-short` | flag | off |
| `--min-points` | int | (module default) |
| `--max-step-gap` | int | (module default) |
| `--out` | str | `output/logs/loss_landscape_latest.png` |
| `--show` | flag | off |
| `--volatility-window` | int | `25` |

---

## Stage 3 tools

### `tools/tracing/memory_timeline.py`

```text
python -m tools.tracing.memory_timeline -i PATH [--plot] [--plot-out PATH]
# or
python tools/tracing/memory_timeline.py -i PATH ...
```

| Flag | Type | Default |
|------|------|---------|
| `--input` / `-i` | str | **required** |
| `--plot` | flag | off |
| `--plot-out` | str | `output/logs/memory_timeline.png` |

### `tools/tracing/activation_account.py`

Stage 3.4 VRAM attribution.

```text
python tools/tracing/activation_account.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--batch-size` | int | `4` |
| `--context` | int | `256` |
| `--embed` | int | `256` |
| `--layers` | int | `4` |
| `--heads` | int | `8` |
| `--vocab` | int | `110` |
| `--out` | str | `output/baselines/stage34_activation_account.json` |

### `tools/bpe_protocol.py`

Stage 3.3 char vs BPE experiment (does not change BiggerTest default).

```text
python tools/bpe_protocol.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--dataset` | str | `tiny_english` |
| `--num-merges` | int | `150` |
| `--context` | int | `64` |
| `--embed` | int | `64` |
| `--layers` | int | `2` |
| `--heads` | int | `4` |
| `--steps` | int | `3` |
| `--out` | str | `output/baselines/stage33_bpe_protocol.json` |

### `tools/stage3_milestones.py`

Runs Stage 3.4–3.7 and 3.11 measurement artifacts.

```text
python tools/stage3_milestones.py [--stages 34,35,36,37,311]
```

| Flag | Type | Default |
|------|------|---------|
| `--stages` | str | `34,35,36,37,311` |

Writes under `output/baselines/` including `stage311_cuda_graph.json`.

### `tools/reports/evolution_report.py`

```text
python tools/reports/evolution_report.py [--out PATH]
```

| Flag | Type | Default |
|------|------|---------|
| `--out` | str | `output/reports/evolution.html` |

### `tools/releases/make_snapshot.py`

Known-good release snapshot (parity gate included).

```text
python tools/releases/make_snapshot.py [--tag v0.1.2]
```

| Flag | Type | Default |
|------|------|---------|
| `--tag` | str | `v0.1.2` |

Writes `output/releases/<tag>/{runtime,quality,generation,memory}.json`, `parity.txt`, `evolution.html`.

---

## Tests & workspace

### `tests/parity/run_parity.py`

```text
.\venv\Scripts\python.exe -m tests.parity.run_parity
```

No flags. Discovers `tests/parity/test_*.py`. Release gate: **10/10**.

Individual modules (also `unittest`-runnable):

```text
python -m unittest tests.parity.test_kv_cache -v
python -m unittest tests.parity.test_attention tests.parity.test_step -v
```

### `setup/2_test_workspace.py`

No flags. Verifies CUDA 10.1 / PyCUDA / sm_35 toolchain.

```text
python setup/2_test_workspace.py
```

---

## Not standalone CLIs (library / import only)

These are imported by the entry points above; they have no project-facing argparse CLI:

| Path | Role |
|------|------|
| `model/*`, `model/cuda/*` | GPT + kernels / ops / allocator / FP16 storage / graph |
| `training/*` | dataset, loss, checkpoint, optimizer, probe, quality, eval, **memory_controller** (2 GB autoscale) |
| `tokenizer/tokenizer.py`, `tokenizer/bpe.py` | Char + experimental BPE |
| `setup/config_loader.py`, `dataset_setup.py`, `model_config.py`, `training_presets.py`, `weight_init.py` | Setup helpers |
| `tools/tracing/runtime_metrics.py` | SyncMeter / MemoryTimeline / KernelTimeline (enabled via train flags) |
| `cli_common.py`, `paths.py`, `logging_config.py`, `version.py` | Shared utilities |

---

## Quick index

| Command | Purpose |
|---------|---------|
| `train.py` | Train / resume / generate menu / quality |
| `auto_train.py` | Train + smoke generate |
| `generate_config.py` | Interactive `setup/*.json` recipe writer (C/H/L/T/B, 2 GB estimate) |
| `tools/make_fact_mix.py` | Repeat user/Wikidata facts → `data/chat_facts.jsonl` |
| `tools/wikidata_to_facts.py` | Wikidata SPARQL → `data/facts/wikidata_facts.txt` |
| `tools/wikidata_to_facts2.py` | Wikidata SPARQL → `data/facts/tech_facts.txt`, `health_facts.txt`, `maths_facts.txt` |
| `generate.py` | One-shot sample (KV on by default) |
| `interactive.py` | Generation REPL (chat checkpoints default `--router`) |
| `tools/calc.py` | Safe AST+Decimal arithmetic (used by the router) |
| `tools/wiki_search.py` | Wikipedia OpenSearch + summary (used by the router) |
| `bench_step.py` | Train-step microbench |
| `bench_profile.py` | Fwd/bwd/opt split |
| `bench_mlp_fusion.py` | MLP fusion A/B |
| `tools/bench_generate.py` | Generate KV bench |
| `tools/make_story_chat.py` | TinyStories → chat lines |
| `training_log_plotter.py` | Loss / tok/s charts |
| `loss_landscape_plotter.py` | 3D loss trajectory |
| `tools/tracing/memory_timeline.py` | Summarize ScratchPool JSONL |
| `tools/tracing/activation_account.py` | Activation VRAM buckets |
| `tools/bpe_protocol.py` | Char vs BPE protocol |
| `tools/stage3_milestones.py` | Stages 3.4–3.7 + 3.11 batch |
| `tools/reports/evolution_report.py` | HTML evolution report |
| `tools/releases/make_snapshot.py` | `v0.1.2` release snapshot |
| `python -m tests.parity.run_parity` | Correctness gate (prefer `.\venv\Scripts\python.exe -m tests.parity.run_parity`) |
| [`guide.md`](guide.md) | GT 730 TinyStories fast start |
| `setup/training_setup.py` | Setup wizard |
| `setup/2_test_workspace.py` | CUDA workspace check |
