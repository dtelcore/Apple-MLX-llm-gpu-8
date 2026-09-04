---
name: From-scratch English then tools
overview: "After the v0.0.2 MLX port, learn English on the 8 GB M3 Air (2 GB process budget) before chat format, tool-call format, an orchestrator, or self-reflection. Do not put search/tools/memory on a model that cannot yet model English."
todos:
  - id: p0-smoke
    content: "Phase 0: drop a tiny English corpus in data/, run auto_train.py --steps 20 --no-prompt, confirm checkpoint + generate do not crash under the 2 GB abort"
    status: pending
  - id: p0-baseline
    content: "Phase 0: freeze output/baselines/m3_air_8gb_story_sub1m.json (or toy smoke JSON) with step_ms, tok/s, device_used_mb, peak, loss; rewrite guide.md for M3/2 GB"
    status: pending
  - id: p1-corpus
    content: "Phase 1: land a real English corpus under data/ (TinyStories-class + simple Wikipedia/instruction mix); keep BPE; document license and token count"
    status: pending
  - id: p1-long-run
    content: "Phase 1: story_sub1m (C=128, L=4, T=128) for thousands of steps with val_every + fixed-prompt samples; only then consider tiny_stories (~3M) if peak stays under 2 GB"
    status: pending
  - id: p1-gate
    content: "Phase 1 gate: loss falling, val loss not exploding, samples are grammatical English on 3 fixed prompts — otherwise do not start chat format"
    status: pending
  - id: p2-chat
    content: "Phase 2: chat-formatted corpus (User:/Assistant:) + chat_5m or continued story_sub1m; train until the model stays in Assistant role; reuse training/chat_format.py and quality_mode=chat"
    status: pending
  - id: p3-tool-format
    content: "Phase 3: synthesize tool-call traces (<tool>web_search{...}</tool> + fake result + answer); train format only; orchestrator stays a stub"
    status: pending
  - id: p4-orchestrator
    content: "Phase 4: Python loop detect tool call → search/browse → paste result → continue; SQLite/JSON memory; demo on our weights"
    status: pending
  - id: p5-reflect
    content: "Phase 5 (only after 1–4): critique → memory write and/or small reinforce on good traces; adapter or replay-buffer fine-tune on this stack"
    status: pending
  - id: p6-decision
    content: "Phase 6: keep scaling the from-scratch GPT (A) vs keep this stack as the lab and put a 3B–8B 4-bit MLX instruct model underneath (B)"
    status: pending
isProject: false
---

# From-scratch English, then tools (M3 Air 8 GB)

Stack: **Apple-MLX-llm-gpu-8 v0.0.2**. Device path is MLX + explicit VJPs (no autograd). Machine: MacBook Air M3, 8 GB unified, fanless. **Process budget: 2 GB** hard abort; 5.5 GB soft machine guard.

```text
You cannot jump to web search + tool calling + self-reflection
on a model that does not yet model English.
Those features sit on top of language competence.
```

## Realistic starting point

| What you have | What that means |
|---|---|
| v0.0.2 trainable GPT, MLX, explicit VJPs, parity green | The lab works. It is not an assistant. |
| Presets ~0.8–5 M params (`story_sub1m`, `tiny_stories`, `chat_5m`) | Can learn local statistics of text |
| Story/chat CLI, checkpoints, quality trial, KV generate | Data pipeline and train loop exist |
| `data/` empty | No English competence yet — expected |

At 1–5 M params this stays a **toy language model** (early char-RNN / tiny GPT-2), not ChatGPT. That is the necessary foundation.

## Honest constraint

On this Air, with a 2 GB process budget and a from-scratch codebase:

- You **can** get a small model that speaks basic English and follows a chat/tool format.
- You **cannot** get a strong general assistant purely by training from scratch here in a reasonable time.
- Web search and tools will work; the bottleneck will be the model’s ability to decide when to use them and how to use the results.

Do not add 100M+ presets. Lid open, low brightness; long GEMMs thermal-throttle.

## What already exists in this repo (do not rebuild)

- Train/generate: `train.py`, `auto_train.py`, `generate.py`, `interactive.py --chat`
- Presets: `toy` (~7k), `story_sub1m` (~0.83M, C=128 L=4 T=128 batch 8 accum 2), `tiny_stories` (~3M, C=256 L=4), `chat_5m` (~5M, C=256 L=6)
- Chat surface: `training/chat_format.py` (`User:` / `Assistant:`, stop strings, quality_mode=chat)
- Observability: `--runtime-metrics`, `--memory-timeline`, quarterly checkpoints, val holdout, generate probe
- Live math: `model/mlx/ops.py` — still no `mx.value_and_grad` / `mlx.nn` / `mlx.optimizers` on the train path

## Phase 0 — Prove the port (this week)

Parity and Metal smoke already passed in v0.0.2. Remaining gate is a **real train→checkpoint→generate** under the 2 GB abort.

```bash
source venv/bin/activate
python setup/2_test_workspace.py
python -m tests.parity.run_parity
# need at least one data/*.txt (toy can use builtin "minimal" via --menu)
python auto_train.py --steps 20 --prompt "once upon a" --no-prompt
```

**Goal:** MLX path stable, checkpoint writes `weights.npz`, generate does not crash, `device_used_mb` stays under 2048.

Also close the leftover port item: freeze `output/baselines/m3_air_8gb_story_sub1m.json` (tok/s, step_ms, active/peak MB, loss) and rewrite `guide.md` for this machine / 2 GB budget (not GT 730).

Exit Phase 0 before a multi-hour run.

## Phase 1 — Actually learn English (weeks)

Stay inside this stack. Scale only as far as 8 GB + **2 GB process** allow.

**Data (required):** real English, not only the 20-step toy corpus. Prefer a mix:

- TinyStories-class narrative
- Simple Wikipedia / WikiText subset (short articles)
- Optional short instructional lines (`Q: … A: …`) — still language, not chat roleplay yet

Put files in `data/*.txt` (combined dir is already the story_sub1m default). Keep BPE. Log license + token count. Do not commit huge corpora.

**Train:** thousands–tens of thousands of steps, not 20.

1. Default: `story_sub1m` (safest under 2 GB). `warmup_steps=500`, val holdout, `--val-every`, fixed prompts every N steps.
2. If peak memory has headroom: `tiny_stories` (~3M). Do **not** jump to `chat_5m` here — that is Phase 2 architecture + format.
3. Fanless: expect throttle; judge by loss/samples, not GT 730 tok/s.

**Track:** train loss, val loss, three frozen prompts (e.g. `once upon a`, `the cat sat`, `in the morning`). Save samples next to the quarter checkpoint.

**Gate (must pass before Phase 2):**

- Loss falls and stays finite; val loss does not explode
- Samples are grammatical English and complete simple prompts coherently
- Still a toy LM — that is success

## Phase 2 — Chat format (still this model)

Switch **data** to simple multi-turn chat (already specified in `training/chat_format.py`):

```text
User: ... Assistant: ...
```

Optional system line. `tools/make_story_chat.py` exists for wrapping stories; extend rather than invent a second format.

Train (`chat_5m` or continued smaller model — **do not resume L=4 into L=6**) until it reliably stays in the Assistant role. Use `quality_mode=chat` and `CHAT_STOP_STRINGS`.

**Goal:** multi-turn dialogue that stays on format, even if knowledge is weak.

## Phase 3 — Tool format only (no real tools)

Synthesize traces, e.g.

```text
Assistant: I need to search.
<tool>web_search{"query":"..."}</tool>
```

then a fake result and a final answer. Train the small model to **emit the format**. The orchestrator stays a stub.

**Goal:** the model learns when and how to write a tool call. Do not wire DuckDuckGo/browser until this format is stable.

## Phase 4 — Orchestrator + real tools

Python chat loop: detect tool call → run search/browse → paste result → continue generation (`interactive.py` or a thin new REPL). Memory store (SQLite / JSON) for facts you want to persist.

**Goal:** end-to-end “chat + search + memory” demo **on our weights**, however limited the intelligence is.

## Phase 5 — Self-reflection + additive updates

Only after Phases 1–4 work.

- Critique → memory write, and/or a small reinforce term on good trajectories
- Persistent adapter or full fine-tune on a replay buffer — using the training machinery already in this repo (manual backward, AdamW, checkpoints)

## Phase 6 — Decision point

Either:

- **A.** Keep scaling the from-scratch GPT (more layers/width, better data, longer train) and accept it will lag open 3B–8B models, or
- **B.** Keep this stack as the lab for learning rules / tools / memory / reflection, and put a real MLX instruct model (3B–8B 4-bit) underneath as the product brain.

A stays pure. B gets a usable assistant sooner while still using everything built here. Do not spend Phase 1–4 effort assuming B; the format/orchestrator transfers.

## Literal next actions

1. Finish Phase 0: a few English `.txt` files in `data/`, `auto_train.py --steps 20`, confirm checkpoint + generate.
2. Pick a real English corpus and a longer `story_sub1m` run (thousands of steps).
3. Measure: does loss fall, do samples become English-like?
4. Only then move to chat-formatted data.

This is the correct first step for a from-scratch line that started on a GT 730. Assistant features come **after** the model has language, not before.
