# Changelog

## 0.0.7

Python **router** around one chat checkpoint: cabinet lookup, then calc, then Wikipedia.

- `interactive.py` defaults `--router` on for chat checkpoints (`--no-router` for story).
  Hits feed the stored `User: … Assistant:` prompt into generate (temp 0.2). `2+2` is
  AST+Decimal (`tools/calc.py`, no `eval`). Unknown `What is …` questions use
  Wikipedia via stdlib urllib (`tools/wiki_search.py`). Search uses the stripped topic
  (`tell me about ford` → `ford`), not the wrapper phrase. A Wikipedia hit is appended
  to `output/cabinet_learned.jsonl` and replayed on later exact/topic hits (weights
  unchanged). Miss hints and calc are not stored. Network/empty extract falls
  through to a polite miss; snippets are not fed back into the GPT.
- Exact normalized cabinet index: [`training/cabinet_index.py`](training/cabinet_index.py).
  No fuzzy match (unobtanium must not hit layer-streaming). Cabinet wins over calc
  (`What is 0 factorial?`). One checkpoint only — do not load a story net in the
  same process (2 GB cap).
- Do not `combine: true`. Do not `--resume` v4 into a new mix. Keep
  `output/checkpoints/chat_facts_v4` as the 105×300 recitation baseline.

## 0.0.6

Chat **fact data pipelines** and the learning rules that actually stick on this 2 GB Air.

- Facts live as exact `User: … Assistant: …` lines. `tools/make_fact_mix.py` repeats
  `data/user_facts.txt` plus `data/facts/*.txt` and writes `data/chat_facts.jsonl`.
  `setup/chat_facts_config.json` trains **only** that file (`combine: false`, explicit
  `path`). `combine: true` or `chat_c256_l6_config.json` / `chat_train.txt` wash the
  cabinet out with ~1M unique wraps.
- What stuck in training: C=512 L=6 T=256 stream, ~20M params. **~300 repeats** of a
  few dozen short Q&As recites (v2). **20 repeats** of a 300-topic wiki slice in 300
  steps does not. New words need a **new BPE and a new checkpoint**, not `--resume`.
- `tools/wikidata_to_facts.py` pulls modest SPARQL groups (capitals, elements,
  inventors, birth years) into `data/facts/wikidata_facts.txt`. Drops Q-id labels.
  Uses stdlib urllib (optional SPARQLWrapper). Keep LIMITs small. Same-question
  conflicts are dropped entirely.
- `tools/wikidata_to_facts2.py` (`wikidatafetch2`) writes separate tech / health /
  maths packs (`tech_facts.txt`, `health_facts.txt`, `maths_facts.txt`). Distinct
  templates per kind; health stays encyclopedic (no dosing/treatment).
- Dataset `path` / `dataset_path` always wins over combine. JSONL query/response
  records load as native chat lines. `--stop` on generate/auto_train cuts at `User:`.

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
