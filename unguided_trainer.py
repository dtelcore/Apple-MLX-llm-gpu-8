#!/usr/bin/env python3
"""0.0.9 Unguided (non-interactive) trainer kernel.

Never calls input(). Remix aborts with NEXT_MIX.json + ABORT_REASON.
Does not --resume across a BPE change.

Usage:
  python unguided_trainer.py --config setup/chat_facts_v7_config.json --policy setup/unguided_v7_policy.json
  python unguided_trainer.py --config ... --policy ... --dry-run
  python unguided_trainer.py --config ... --policy ... --unguarded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("llm_gpu.unguided")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _append_decision(run_dir: Path, record: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "decisions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_abort(run_dir: Path, reason: str, next_mix: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ABORT_REASON").write_text(reason + "\n", encoding="utf-8")
    if next_mix is not None:
        (run_dir / "NEXT_MIX.json").write_text(json.dumps(next_mix, indent=2) + "\n", encoding="utf-8")


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dry_run_param_estimate(recipe: dict) -> dict:
    from setup.model_config import estimate_vram_footprint

    model = dict(recipe.get("model") or {})
    dataset = recipe.get("dataset") or {}
    vocab = model.get("vocab_size")
    if not vocab:
        vocab = int(dataset.get("bpe_merges") or 4000) + 256
    model["vocab_size"] = int(vocab)
    try:
        return estimate_vram_footprint(model)
    except Exception:
        return {"total_params": None, "vocab_size": vocab}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0.0.9 Unguided trainer (no input())")
    parser.add_argument("--config", required=True, help="Recipe JSON (setup/*.json)")
    parser.add_argument("--policy", required=True, help="Policy JSON (setup/unguided_*_policy.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit 0, no Metal")
    parser.add_argument(
        "--unguarded",
        action="store_true",
        help="Relax data hygiene only; hard limits still enforced",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override policy max_steps")
    parser.add_argument("--eval-every", type=int, default=None, help="Override policy eval_every")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    policy_path = Path(args.policy)

    if not config_path.is_file():
        logger.error("Recipe not found: %s", config_path)
        return 2
    if not policy_path.is_file():
        logger.error("Policy not found: %s", policy_path)
        return 2

    recipe = _load_json(config_path)
    policy = _load_json(policy_path)
    if args.max_steps is not None:
        policy["max_steps"] = args.max_steps
    if args.eval_every is not None:
        policy["eval_every"] = args.eval_every
    if args.unguarded:
        policy["unguarded"] = True

    run_name = policy.get("run_name", "unguided_v7")
    run_dir = Path("output/runs") / run_name
    max_steps = int(policy.get("max_steps", 500))
    eval_every = int(policy.get("eval_every", 50))
    max_wall_s = float(policy.get("max_wall_s", 7200))
    checkpoint_dir = Path(policy.get("checkpoint_dir") or f"output/checkpoints/{run_name}")

    if args.dry_run:
        estimate = _dry_run_param_estimate(recipe)
        model = recipe.get("model") or {}
        print("=== UNGUIDED DRY-RUN ===")
        print(f"recipe:        {config_path}")
        print(f"policy:        {policy_path}")
        print(f"run_name:      {run_name}")
        print(f"checkpoint:    {checkpoint_dir}")
        print(f"dataset:       {(recipe.get('dataset') or {}).get('path')}")
        print(
            f"arch:          C={model.get('embedding_dim')} L={model.get('num_layers')} "
            f"T={model.get('max_len')} H={model.get('num_heads')}"
        )
        print(f"param_est:     {estimate.get('total_params')} (vocab placeholder until BPE)")
        print(f"max_steps:     {max_steps}")
        print(f"eval_every:    {eval_every}")
        print(f"max_wall_s:    {max_wall_s}")
        print(f"early_stop:    patience={policy.get('early_stop_patience')}")
        print(f"loss_spike:    ratio={policy.get('loss_spike_ratio')}")
        print(f"remix_if:      {policy.get('remix_if')}")
        print(f"unguarded:     {bool(args.unguarded)}")
        print("No Metal init. Exiting 0.")
        return 0

    from training.checkpoint import promote_best
    from training.unguided.decide import Decision, DecideContext, decide
    from training.unguided.eval_suite import run_eval_suite
    from training.unguided.loop import train_segment
    from training.unguided.session import build_train_session, make_train_args

    train_args = make_train_args(
        config_path, checkpoint_dir, policy, unguarded=bool(args.unguarded),
    )
    try:
        session = build_train_session(train_args, policy)
    except RuntimeError as exc:
        logger.error("Session build refused: %s", exc)
        _write_abort(run_dir, str(exc))
        return 1

    logger.info(
        "Unguided session ready | run=%s | vocab_fp=%s | ckpt=%s",
        run_name, session.vocab_fingerprint, session.checkpoint_dir,
    )

    history: list[float] = []
    start = time.time()
    last_eval = None

    while True:
        remaining = max_steps - session.step
        if remaining <= 0:
            result_action = Decision.STOP_LIMIT
            result_reason = f"max_steps={max_steps}"
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "step": session.step,
                "action": result_action.value,
                "reason": result_reason,
                "promote": False,
            }
            _append_decision(run_dir, rec)
            _write_abort(run_dir, result_reason)
            summary = {
                "final_step": session.step,
                "best_val_loss": session.best_val_loss,
                "action": result_action.value,
                "reason": result_reason,
            }
            if last_eval is not None:
                summary.update(last_eval)
            _write_summary(run_dir, summary)
            return 0

        metrics = train_segment(session, min(eval_every, remaining))
        logger.info(
            "segment done | step=%d loss=%.4f nan=%s",
            session.step, metrics.avg_loss, metrics.nan_detected,
        )

        eval_res = run_eval_suite(session, policy)
        if eval_res.val_loss is not None:
            history.append(eval_res.val_loss)
            window = int(policy.get("loss_median_window", 5))
            session.recent_val_losses = history[-window:]

        ctx = DecideContext(
            step=session.step,
            max_steps=max_steps,
            wall_s=time.time() - start,
            max_wall_s=max_wall_s,
            val_loss=eval_res.val_loss,
            best_val_loss=session.best_val_loss,
            recent_val_losses=session.recent_val_losses,
            nan_detected=metrics.nan_detected or eval_res.nan_detected,
            cabinet_exact_match=eval_res.cabinet_exact_match,
            policy=policy,
            no_improvement_count=session.no_improvement_count,
        )
        result = decide(ctx)

        if eval_res.val_loss is not None:
            if session.best_val_loss is None or eval_res.val_loss < session.best_val_loss:
                session.best_val_loss = eval_res.val_loss
                session.no_improvement_count = 0
            else:
                session.no_improvement_count += 1

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": session.step,
            "action": result.action.value,
            "reason": result.reason,
            "val_loss": eval_res.val_loss,
            "cabinet_exact_match": eval_res.cabinet_exact_match,
            "promote": result.promote,
        }
        _append_decision(run_dir, rec)
        logger.info("decision=%s reason=%s", result.action.value, result.reason)

        last_eval = {
            "val_loss": eval_res.val_loss,
            "val_ppl": eval_res.val_ppl,
            "cabinet_exact_match": eval_res.cabinet_exact_match,
            "router_precision": eval_res.router_precision,
            "router_recall": eval_res.router_recall,
            "anchor_hits": eval_res.anchor_hits,
            "harvested_entity_swaps": eval_res.harvested_entity_swaps,
            "harvested_exact": eval_res.harvested_exact,
            "best_val_loss": session.best_val_loss,
            "step": session.step,
        }
        _write_summary(run_dir, {**last_eval, "action": result.action.value, "reason": result.reason})

        if result.promote:
            try:
                promote_best(
                    session.run_dir,
                    session.run_dir,
                    meta={
                        "metric": "val_loss",
                        "value": session.best_val_loss,
                        "step": session.step,
                    },
                )
            except Exception as exc:
                logger.warning("promote_best failed: %s", exc)

        if result.action in (Decision.CONTINUE, Decision.PROMOTE):
            continue

        if result.action == Decision.ABORT_REMIX:
            _write_abort(run_dir, result.reason, result.next_mix_recipe)
            logger.warning("ABORT_REMIX: %s", result.reason)
            return 1

        if result.action == Decision.ABORT_SPIKE:
            _write_abort(run_dir, result.reason)
            logger.error("ABORT_SPIKE: %s", result.reason)
            return 1

        if result.action in (Decision.EARLY_STOP, Decision.STOP_LIMIT):
            _write_abort(run_dir, result.reason)
            logger.info("Stopping: %s", result.reason)
            _write_summary(
                run_dir,
                {
                    **(last_eval or {}),
                    "final_step": session.step,
                    "best_val_loss": session.best_val_loss,
                    "action": result.action.value,
                    "reason": result.reason,
                },
            )
            return 0

        _write_abort(run_dir, f"unknown_action:{result.action}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
