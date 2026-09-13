"""Pure decision engine for the unguided trainer. Fully unit-tested."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class Decision(str, Enum):
    CONTINUE = "continue"
    PROMOTE = "promote"
    EARLY_STOP = "early_stop"
    ABORT_SPIKE = "abort_spike"
    ABORT_REMIX = "abort_remix"
    STOP_LIMIT = "stop_limit"


@dataclass
class DecideContext:
    """Snapshot passed to decide()."""
    step: int
    max_steps: int
    wall_s: float
    max_wall_s: float
    val_loss: float | None
    best_val_loss: float | None
    recent_val_losses: Sequence[float]  # oldest → newest, length ≤ window
    nan_detected: bool
    cabinet_exact_match: float | None
    policy: dict[str, Any]
    no_improvement_count: int = 0


@dataclass
class DecideResult:
    action: Decision
    reason: str
    promote: bool = False
    next_mix_recipe: dict[str, Any] | None = None


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("inf")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def decide(ctx: DecideContext) -> DecideResult:
    """
    Deterministic policy. No I/O, no randomness, no input().
    Order of checks is intentional and stable.
    """
    p = ctx.policy
    spike_ratio = float(p.get("loss_spike_ratio", 2.0))
    patience = int(p.get("early_stop_patience", 4))
    remix_cfg = p.get("remix_if") or {}
    remix_threshold = remix_cfg.get("cabinet_exact_match_below")
    remix_after = int(remix_cfg.get("after_steps", 10**9))

    if ctx.nan_detected or (ctx.val_loss is not None and not _finite(ctx.val_loss)):
        return DecideResult(
            action=Decision.ABORT_SPIKE,
            reason="non_finite_loss",
        )

    if ctx.val_loss is not None and ctx.recent_val_losses:
        med = _median(ctx.recent_val_losses)
        if med > 0 and ctx.val_loss > spike_ratio * med:
            return DecideResult(
                action=Decision.ABORT_SPIKE,
                reason=f"loss_spike val={ctx.val_loss:.4f} median={med:.4f} ratio={spike_ratio}",
            )

    if ctx.step >= ctx.max_steps:
        return DecideResult(
            action=Decision.STOP_LIMIT,
            reason=f"max_steps={ctx.max_steps}",
        )
    if ctx.wall_s >= ctx.max_wall_s:
        return DecideResult(
            action=Decision.STOP_LIMIT,
            reason=f"max_wall_s={ctx.max_wall_s}",
        )

    if (
        remix_threshold is not None
        and ctx.cabinet_exact_match is not None
        and ctx.step >= remix_after
        and ctx.cabinet_exact_match < float(remix_threshold)
    ):
        next_mix = {
            "action": "abort_remix",
            "reason": "cabinet_exact_match_below_threshold",
            "cabinet_exact_match": ctx.cabinet_exact_match,
            "threshold": remix_threshold,
            "after_steps": remix_after,
            "suggested_cmd": [
                "python",
                "tools/make_fact_mix.py",
                "--config",
                p.get("recipe", "setup/chat_facts_v7_config.json"),
            ],
        }
        return DecideResult(
            action=Decision.ABORT_REMIX,
            reason=f"cabinet_exact_match={ctx.cabinet_exact_match:.3f} < {remix_threshold}",
            next_mix_recipe=next_mix,
        )

    promote = False
    if ctx.val_loss is not None:
        if ctx.best_val_loss is None or ctx.val_loss < ctx.best_val_loss:
            promote = True

    if ctx.no_improvement_count >= patience:
        return DecideResult(
            action=Decision.EARLY_STOP,
            reason=f"no_improvement_for_{patience}_evals",
            promote=promote,
        )

    return DecideResult(
        action=Decision.CONTINUE if not promote else Decision.PROMOTE,
        reason="improved" if promote else "ok",
        promote=promote,
    )
