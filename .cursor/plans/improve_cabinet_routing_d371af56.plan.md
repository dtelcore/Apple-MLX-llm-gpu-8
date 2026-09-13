---
name: Improve Cabinet Routing
overview: Improve messy-question handling without fuzzy matching, keep every trained-cabinet response model-generated, and emit structured diagnostics that distinguish router failures from neural target-binding failures. Also quarantine known-bad facts so routing improvements do not make incorrect source data more authoritative.
todos:
  - id: safe-routing
    content: Add collision-safe template rewriting, explicit trained aliases, and trained-before-learned resolution
    status: completed
  - id: mismatch-reporting
    content: Emit route provenance, compare generated replies with stored targets, and persist/display retraining flags
    status: completed
  - id: binding-probe
    content: Add an offline teacher-forced target-token rank probe for weight-level diagnosis
    status: completed
  - id: quarantine-data
    content: Exclude confirmed bad facts from runtime routing and future mixes
    status: completed
  - id: regression-tests
    content: Add routing/session/UI tests, fix stale fixture assumptions, and replay the v7 prompt set
    status: completed
isProject: false
---

# Improve cabinet routing and generation diagnostics

## 1. Resolve messy questions safely
- Extend the existing [`training/cabinet_index.py`](/Users/it/dev/Apple%20MLX/training/cabinet_index.py) API with source-aware trained lookup and collision-safe canonical candidates; preserve its no-argument/dictionary constructor and current JSONL loaders rather than replacing them with path-bound initialization.
- Keep `normalize_question()` as the single key normalizer; do not introduce a second normalizer that adds `?` or strips apostrophes. Add deterministic article handling only inside recognized template slots, so `Who invented the plough?` can resolve to the unique trained `Who invented Plough?` while unrelated text such as `What is unobtanium?` remains a miss. Collect all trained candidates and accept only exactly one rather than returning the first variant.
- Add a small explicit alias file, [`data/cabinet_aliases.json`](/Users/it/dev/Apple%20MLX/data/cabinet_aliases.json), mapping intentional semantic shorthand such as `what is layer streaming?` to `What is sequential layer streaming?`. Reject aliases whose target is absent or ambiguous.
- Update the existing functional `route()` flow in [`training/router.py`](/Users/it/dev/Apple%20MLX/training/router.py)—there is no `Router.resolve()` or `RouteResult` class—to resolve in this order: trained exact → safe/explicit trained alias → learned exact/topic → calculator/search/miss. This lets the curated layer-streaming alias override the already-learned incorrect Howdy result without deleting history.
- Replace the ineffective `related_template_hits()` retry path with actual candidate resolution; canonical trained prompts remain the prompts sent to the model.
- Extend `RouteDecision` with deterministic provenance: normalized input, matched canonical question/key, source, and `match_type` (`trained_exact`, `trained_alias`, `trained_article`, `learned_exact`, `learned_topic`, or `none`). Do not fabricate `route_logits`: this router is ordered rules and dictionary lookup, not a learned classifier.

## 2. Return generation and emit structured diagnostics
- Keep the generated reply as the user-visible answer, as requested; do not silently replay or substitute the stored cabinet answer.
- In the existing `_router_turn()` path in [`training/chat_session.py`](/Users/it/dev/Apple%20MLX/training/chat_session.py), normalize whitespace/case/trailing punctuation and compare each trained-cabinet generation with `CabinetFact.assistant`.
- On disagreement, set `generate_target_mismatch` (a target-string mismatch, not a claim that semantic truth was evaluated), emit a warning containing typed question, canonical question, expected answer, generated answer, and checkpoint, and append a JSONL event to `output/cabinet_retrain.jsonl`. Matching generations remain `generate`.
- Emit one structured turn record to `output/cabinet_diagnostics.jsonl` with event ID/time, checkpoint, raw/normalized query, route provenance, canonical prompt, stored target, generated text, and classification. Use `ROUTER_FALSE_MISS`/`ROUTER_FALSE_HIT` only in labeled evaluation runs; live traffic without an intent label records observed route facts rather than guessing ground truth.
- Classify `BINDING_ENTITY_SWAP` only when a generated response matches another known cabinet target in the same recognized relation family. Otherwise use `TARGET_MISMATCH`; derive `TEMPLATE_COLLAPSE` only as an aggregate repeated-output pattern, not from one turn.
- Expose the mismatch flag through [`app/webui/__init__.py`](/Users/it/dev/Apple%20MLX/app/webui/__init__.py) and [`app/webui/index.html`](/Users/it/dev/Apple%20MLX/app/webui/index.html), while displaying the generated response unchanged. The API already forwards `detail`; extend only the additional diagnostic fields needed by the UI. Add CLI paths in [`App.py`](/Users/it/dev/Apple%20MLX/App.py) and shared session arguments, defaulting under `output/`.

## 3. Add an offline binding probe
- Add [`tools/eval_cabinet_bindings.py`](/Users/it/dev/Apple%20MLX/tools/eval_cabinet_bindings.py) and a small labeled regression fixture containing the observed layer-streaming, Cubitt, Paris, and Plough cases.
- Separate router evaluation (precision/recall over labeled intents) from model evaluation. For each canonical trained prompt, run teacher-forced scoring of the expected answer tokens and record expected-token rank, probability, top-k alternatives, exact generated-answer match, and recognized entity match. This is more meaningful than inspecting only the first sampled answer token, especially for multi-token names.
- Reuse the existing host logits available during generation/forward rather than retaining full-vocabulary logits for every token. Keep detailed top-k capture opt-in so normal App traffic and the 2 GB Metal budget are unaffected.
- Defer Q/K cross-prompt cosine and attention-template ratios to the separate v8 experiment. A query vector and “distractor key” from different prompts are not jointly attended, raw cosine thresholds such as `0.85` are not evidence of collapse, and the current KV generation path does not retain all attention probabilities. A later probe must define layer/head aggregation, entity token spans, and a baseline before assigning thresholds.

## 4. Quarantine bad source facts
- Add a normalized-key quarantine file for the actual three linked rows: `Who invented Plough?`, `Who is credited with inventing Plough?`, and `What did Ernesto Schiaparelli invent?`. Do not add aliases such as `plow` as quarantine substitutes.
- Apply the quarantine both when loading the runtime cabinet and in [`tools/make_fact_mix.py`](/Users/it/dev/Apple%20MLX/tools/make_fact_mix.py), preventing current routing and future fresh mixes from treating those rows as ground truth.
- Do not alter or resume the v7 checkpoint. Quarantined current-v7 questions will proceed to the normal search/miss path; mismatch records will identify valid facts that need stronger binding in a future fresh train.

## 5. Verify behavior
- Add focused coverage in [`tests/test_cabinet_index.py`](/Users/it/dev/Apple%20MLX/tests/test_cabinet_index.py), [`tests/test_router.py`](/Users/it/dev/Apple%20MLX/tests/test_router.py), [`tests/test_chat_session.py`](/Users/it/dev/Apple%20MLX/tests/test_chat_session.py), and [`tests/test_app.py`](/Users/it/dev/Apple%20MLX/tests/test_app.py) for article rewriting, alias priority over learned poison, non-fuzzy misses, generated-response preservation, mismatch persistence, and UI/API flags.
- Establish measured baselines before setting pass/fail targets; do not hard-code proposed 98%/95%/92% or cosine/entropy thresholds without benchmark results.
- Repair the stale live-mix test that assumes France exists in `data/chat_facts.jsonl`; v7-specific assertions should use `data/chat_facts_v7.jsonl`.
- Run non-Metal tests first. After stopping the currently running App, restart the same process and replay the logged prompts against v7; do not launch a second Metal process.