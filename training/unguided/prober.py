"""Generate probe for unguided runs.

Same path as App cabinet generate: stored ``User: … Assistant:``, temperature
0.2, top-k 10, CHAT_STOP_STRINGS. Does not rewrite replies and does not append
cabinet_retrain.jsonl.

The trainer calls this once at a successful stop (max_steps / wall / early
stop) on the already-loaded model. ``unguided_prober.py`` loads a checkpoint
after the train process has released Metal.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from paths import DATA_DIR, OUTPUT_CHECKPOINTS, OUTPUT_ROOT
from training.cabinet_diagnostics import answers_match, classify_generation, normalize_answer
from training.cabinet_index import CabinetFact, CabinetIndex, load_cabinet, normalize_question
from training.chat_format import CHAT_STOP_STRINGS, sanitize_assistant_reply
from training.unguided.decide import NextStepContext, NextStepResult

CABINET_GENERATE_TEMP = 0.2
CABINET_GENERATE_TOP_K = 10
DEFAULT_FACTS = DATA_DIR / "chat_facts_v7.jsonl"
DEFAULT_PROBE_N = 50
COLLAPSE_MIN_N = 3
COLLAPSE_SWAP_RATE = 0.4
LONG_UNIQUE_FAMILIES = frozenset({"layer_stream", "wiki_extra", "project", "theorem"})
SHORT_TEMPLATE_FAMILIES = frozenset(
    {"capital_of", "organ_system", "who_invented", "what_did_invent", "gas", "project"}
)

DEFAULT_MUST = (
    "What is the capital of France?",
    "What is France's capital?",
    "Where is Paris?",
    "What country is Paris in?",
    "What is the capital of Japan?",
    "What is Japan's capital?",
    "Where is Tokyo?",
    "What country is Tokyo in?",
    "What is the capital of Germany?",
    "Where is Berlin?",
    "Which organ system does the ureter belong to?",
    "Which organ system does the kidney belong to?",
    "Which organ system does the urethra belong to?",
    "Which organ system does the heart belong to?",
    "Which organ system does the brain belong to?",
    "Which organ system does the liver belong to?",
    "Which organ system does the lung belong to?",
    "Which organ system does the spleen belong to?",
    "What pathogen causes vomiting?",
    "What pathogen causes diarrhea?",
    "What pathogen causes COVID-19?",
    "What pathogen causes tuberculosis?",
    "What pathogen causes influenza?",
    "What did William Cubitt invent?",
    "Who invented penal treadmill?",
    "Who invented Z4?",
    "Who invented analytical engine?",
    "What is sequential layer streaming?",
    "How do I disable layer streaming?",
    "How do I force layer streaming?",
    "What is the atomic number of carbon?",
    "Which element has atomic number 6?",
    "When was George Washington born?",
    "When was Bob Dylan born?",
    "Is 17 a prime number?",
    "Is 42 a prime number?",
    "Is 1 a prime number?",
    "In which year was the JavaScript programming language first released?",
    "In which year was the Java programming language first released?",
    "State the Pythagorean theorem.",
    "State the quadratic formula.",
    "What gas do humans inhale that cells use for respiration?",
    "What gas do humans exhale as a product of respiration?",
    "Does training use autograd?",
    "What tokenizer does chat facts training use?",
    "what are pancakes",
)

DEFAULT_OOD = (
    "What is the capital of Atlantis?",
    "Explain how photosynthesis works in two sentences.",
    "Write a short poem about rain.",
    "Who is the president of France?",
)

_FAMILY_RULES = (
    (r"^what is the capital of ", "capital_of"),
    (r"^what is .+?'s capital$", "possess_capital"),
    (r"^what country is .+ in$", "country_of"),
    (r"^where is ", "where_is"),
    (r"^which organ system does .+ belong to$", "organ_system"),
    (r"^what pathogen causes ", "pathogen"),
    (r"^who invented ", "who_invented"),
    (r"^what did .+ invent$", "what_did_invent"),
    (r"^what is the atomic number of ", "atomic_number"),
    (r"^which element has atomic number ", "element_by_number"),
    (r"^when was .+ born$", "born"),
    (r"^is \d+ a prime number$", "prime"),
    (r"^in which year was ", "year_released"),
    (r"^state the ", "theorem"),
    (r"^how do i ", "how_do_i"),
    (r"^what is sequential layer streaming$", "layer_stream"),
    (r"^what gas ", "gas"),
    (r"^does training use autograd", "project"),
    (r"^what tokenizer ", "project"),
    (r"^what are pancakes$", "wiki_extra"),
)

_SLOT_PATTERNS = (
    re.compile(r"^which organ system does (?:the )?(.+) belong to$"),
    re.compile(r"^what pathogen causes (.+)$"),
    re.compile(r"^what is the capital of (.+)$"),
    re.compile(r"^what is (.+)'s capital$"),
    re.compile(r"^what country is (.+) in$"),
    re.compile(r"^where is (.+)$"),
    re.compile(r"^is (\d+) a prime number$"),
    re.compile(r"^when was (.+) born$"),
    re.compile(r"^who invented (.+)$"),
    re.compile(r"^what did (.+) invent$"),
    re.compile(r"^what is the atomic number of (.+)$"),
    re.compile(r"^which element has atomic number (.+)$"),
    re.compile(r"^in which year was (?:the )?(.+?)(?: programming language)? first released$"),
)

_MIX_SHAPED = (
    re.compile(r"the capital of .+ is .+", re.I),
    re.compile(r".+ is the capital of .+", re.I),
    re.compile(r".+ belongs to the .+", re.I),
    re.compile(r".+ is the pathogen that causes .+", re.I),
    re.compile(r".+ is credited with inventing .+", re.I),
    re.compile(r".+ was born in \d+", re.I),
    re.compile(r".+ was first released in \d+", re.I),
    re.compile(r"(yes|no), \d+ is a (prime|composite) number", re.I),
)


def family(user: str) -> str:
    key = normalize_question(user)
    for pat, name in _FAMILY_RULES:
        if re.search(pat, key):
            return name
    return "other"


def user_slot(user: str) -> str:
    key = normalize_question(user)
    for pat in _SLOT_PATTERNS:
        m = pat.match(key)
        if m:
            return (m.group(1) or "").strip()
    return ""


def select_prompts(
    index: CabinetIndex,
    n: int,
    *,
    seed: int = 42,
    must: Sequence[str] = DEFAULT_MUST,
) -> list[CabinetFact]:
    """Must-list first, then one-per-family round-robin until ``n``."""
    n = max(0, int(n))
    chosen: list[CabinetFact] = []
    seen: set[str] = set()
    for query in must:
        if len(chosen) >= n:
            return chosen
        fact = index.lookup(query, source="trained") or index.lookup(query)
        if fact is None or fact.key in seen:
            continue
        chosen.append(fact)
        seen.add(fact.key)

    buckets: dict[str, list[CabinetFact]] = defaultdict(list)
    for fact in index.unique_facts():
        if fact.source != "trained" or fact.key in seen:
            continue
        buckets[family(fact.user)].append(fact)
    rng = random.Random(int(seed))
    for bucket in buckets.values():
        rng.shuffle(bucket)
    families = list(buckets)
    rng.shuffle(families)
    while len(chosen) < n:
        progressed = False
        for fam in families:
            if not buckets[fam]:
                continue
            fact = buckets[fam].pop()
            chosen.append(fact)
            seen.add(fact.key)
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return chosen


def neighbour_source(index: CabinetIndex, fact: CabinetFact, generated: str) -> Optional[str]:
    gen = normalize_answer(generated)
    if not gen:
        return None
    fam = family(fact.user)
    for other in index.unique_facts():
        if other.source != "trained" or other.key == fact.key:
            continue
        if family(other.user) != fam:
            continue
        if normalize_answer(other.assistant) == gen:
            return other.user
    return None


def classify_probe_row(
    index: CabinetIndex,
    fact: CabinetFact,
    generated: str,
) -> tuple[str, Optional[str]]:
    if answers_match(fact.assistant, generated):
        return "MATCH", None
    swapped = neighbour_source(index, fact, generated)
    cls = classify_generation(fact.assistant, generated, index, fact)
    if swapped and cls == "TARGET_MISMATCH":
        return "TEMPLATE_NEIGHBOUR_SWAP", swapped
    return cls, swapped


def gold_omits_slot(user: str, assistant: str) -> bool:
    slot = user_slot(user)
    if not slot:
        return False
    return slot.casefold() not in normalize_answer(assistant)


def mix_smells(index: CabinetIndex, *, limit: int = 12) -> dict[str, Any]:
    dirty: list[dict[str, str]] = []
    shared_map: dict[str, list[str]] = defaultdict(list)
    for fact in index.unique_facts():
        if fact.source != "trained":
            continue
        if gold_omits_slot(fact.user, fact.assistant):
            dirty.append({"user": fact.user, "slot": user_slot(fact.user), "assistant": fact.assistant})
        key = normalize_answer(fact.assistant)
        if key:
            shared_map[key].append(fact.user)
    cleaned = []
    for users in shared_map.values():
        if len(users) < 2:
            continue
        fact = index.lookup(users[0])
        cleaned.append({"assistant": fact.assistant if fact else "", "users": users})
    return {
        "dirty_n": len(dirty),
        "shared_n": len(cleaned),
        "dirty_gold": dirty[:limit],
        "shared_assistants": cleaned[:limit],
    }


def ood_copies_mix(index: CabinetIndex, generated: str) -> bool:
    gen = normalize_answer(generated)
    if not gen:
        return False
    for fact in index.unique_facts():
        if fact.source != "trained":
            continue
        if normalize_answer(fact.assistant) == gen:
            return True
    return any(pat.search(generated or "") for pat in _MIX_SHAPED)


def collapsing_families(by_family: dict[str, dict[str, int]]) -> tuple[str, ...]:
    names = []
    for name, stats in by_family.items():
        n = int(stats.get("n") or 0)
        swap = int(stats.get("swap") or 0)
        if n >= COLLAPSE_MIN_N and (swap / n) >= COLLAPSE_SWAP_RATE:
            names.append(name)
    return tuple(sorted(names))


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("match"))
    n_swap = sum(
        1
        for r in rows
        if r.get("classification") in ("BINDING_ENTITY_SWAP", "TEMPLATE_NEIGHBOUR_SWAP")
    )
    by_fam: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0, "swap": 0})
    for row in rows:
        bucket = by_fam[str(row.get("family") or "other")]
        bucket["n"] += 1
        bucket["ok"] += int(bool(row.get("match")))
        bucket["swap"] += int(
            row.get("classification") in ("BINDING_ENTITY_SWAP", "TEMPLATE_NEIGHBOUR_SWAP")
        )
    short = [r for r in rows if r.get("family") in SHORT_TEMPLATE_FAMILIES]
    long_fail = sum(
        1
        for r in rows
        if r.get("family") in LONG_UNIQUE_FAMILIES
        and not r.get("match")
        and r.get("classification") == "TARGET_MISMATCH"
    )
    return {
        "n_cabinet": n,
        "n_match": n_ok,
        "n_swap": n_swap,
        "exact_rate": (n_ok / n) if n else 0.0,
        "swap_rate": (n_swap / n) if n else 0.0,
        "class_counts": dict(Counter(str(r.get("classification")) for r in rows)),
        "by_family": {k: dict(v) for k, v in sorted(by_fam.items())},
        "collapsing_families": list(collapsing_families(by_fam)),
        "long_unique_fail": long_fail,
        "short_template_exact": (sum(1 for r in short if r.get("match")) / len(short)) if short else None,
    }


def generate_reply(model: Any, tokenizer: Any, prompt_text: str, seed: int) -> str:
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        return ""
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=80,
        temperature=CABINET_GENERATE_TEMP,
        top_k=CABINET_GENERATE_TOP_K,
        top_p=None,
        tokenizer=tokenizer,
        rng=np.random.default_rng(int(seed)),
        use_kv_cache=True,
        stop_strings=list(CHAT_STOP_STRINGS),
    )
    new_ids = generated_ids[len(prompt_ids) :]
    reply = tokenizer.decode(new_ids)
    return sanitize_assistant_reply(reply, list(CHAT_STOP_STRINGS))


def run_generate_probe(
    *,
    model: Any,
    tokenizer: Any,
    facts_path: Path | str,
    n: int = DEFAULT_PROBE_N,
    seed: int = 42,
    include_ood: bool = True,
    step: int | None = None,
    checkpoint: str = "",
) -> dict[str, Any]:
    index = load_cabinet(facts_path)
    facts = select_prompts(index, n, seed=seed)
    rows: list[dict[str, Any]] = []
    for i, fact in enumerate(facts):
        reply = generate_reply(model, tokenizer, fact.generate_prompt, seed=int(seed) + i)
        cls, swapped = classify_probe_row(index, fact, reply)
        rows.append(
            {
                "kind": "cabinet",
                "family": family(fact.user),
                "typed": fact.user,
                "canonical": fact.user,
                "expected": fact.assistant,
                "generated": reply,
                "match": cls == "MATCH",
                "classification": cls,
                "swapped_from": swapped,
            }
        )
    ood_rows: list[dict[str, Any]] = []
    if include_ood:
        for i, user in enumerate(DEFAULT_OOD):
            prompt = f"User: {user} Assistant:"
            reply = generate_reply(model, tokenizer, prompt, seed=int(seed) + 1000 + i)
            ood_rows.append(
                {
                    "kind": "ood",
                    "typed": user,
                    "prompt": prompt,
                    "generated": reply,
                    "copies_mix": ood_copies_mix(index, reply),
                }
            )
    scores = summarize_rows(rows)
    smells = mix_smells(index)
    report = {
        "checkpoint": checkpoint,
        "facts": str(facts_path),
        "step": step,
        "n_unique": sum(1 for f in index.unique_facts() if f.source == "trained"),
        "settings": {
            "temperature": CABINET_GENERATE_TEMP,
            "top_k": CABINET_GENERATE_TOP_K,
            "n": n,
            "seed": seed,
        },
        **scores,
        "ood_n": len(ood_rows),
        "ood_mix_copies": sum(1 for r in ood_rows if r.get("copies_mix")),
        "mix_smells": smells,
        "cabinet": rows,
        "ood": ood_rows,
    }
    return report


def run_session_probe(session: Any, policy: dict[str, Any]) -> dict[str, Any]:
    facts = Path(session.dataset_path or DEFAULT_FACTS)
    if not facts.is_file():
        facts = DEFAULT_FACTS
    return run_generate_probe(
        model=session.model,
        tokenizer=session.tokenizer,
        facts_path=facts,
        n=int(policy.get("probe_n", DEFAULT_PROBE_N)),
        seed=int(policy.get("probe_seed", 42)),
        include_ood=bool(policy.get("probe_ood", True)),
        step=int(getattr(session, "step", 0) or 0),
        checkpoint=str(getattr(session, "checkpoint_dir", "") or ""),
    )


def next_step_context(
    *,
    step: int,
    max_steps: int,
    policy: dict[str, Any],
    last_eval: Optional[dict[str, Any]],
    report: dict[str, Any],
) -> NextStepContext:
    ev = last_eval or {}
    smells = report.get("mix_smells") or {}
    return NextStepContext(
        step=int(step),
        max_steps=int(max_steps),
        val_loss=ev.get("val_loss"),
        best_val_loss=ev.get("best_val_loss"),
        cabinet_exact_match=ev.get("cabinet_exact_match"),
        generate_exact_rate=report.get("exact_rate"),
        generate_swap_rate=report.get("swap_rate"),
        generate_n=int(report.get("n_cabinet") or 0),
        ood_mix_copies=int(report.get("ood_mix_copies") or 0),
        ood_n=int(report.get("ood_n") or 0),
        dirty_gold=int(smells.get("dirty_n") or 0),
        shared_assistants=int(smells.get("shared_n") or 0),
        collapsing_families=tuple(report.get("collapsing_families") or ()),
        long_unique_fail=int(report.get("long_unique_fail") or 0),
        short_template_exact=report.get("short_template_exact"),
        policy=policy,
    )


def render_markdown(report: dict[str, Any], verdict: NextStepResult) -> str:
    rows = list(report.get("cabinet") or [])
    ood = list(report.get("ood") or [])
    smells = report.get("mix_smells") or {}
    by_fam = report.get("by_family") or {}
    lines = [
        "# Unguided generate probe",
        "",
        f"- checkpoint: `{report.get('checkpoint') or ''}`",
        f"- step: {report.get('step')}",
        f"- mix: `{report.get('facts') or ''}` ({report.get('n_unique')} unique trained)",
        (
            f"- settings: temp={CABINET_GENERATE_TEMP} top_k={CABINET_GENERATE_TOP_K} "
            f"n={report.get('n_cabinet')} (App cabinet generate)"
        ),
        "",
        "## Verdict",
        "",
        f"**{verdict.headline}**",
        "",
        f"- recitation mode: `{verdict.mode}`",
        f"- understands: **{'yes' if verdict.understands else 'no'}**",
        f"- primary next step: **{verdict.primary}**",
        "",
    ]
    for reason in verdict.reasons:
        lines.append(f"- {reason}")
    exact = report.get("exact_rate") or 0.0
    swap = report.get("swap_rate") or 0.0
    lines += [
        "",
        "## Scores",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| generate exact | {report.get('n_match')}/{report.get('n_cabinet')} ({exact:.1%}) |",
        f"| neighbour / entity swap | {report.get('n_swap')}/{report.get('n_cabinet')} ({swap:.1%}) |",
        f"| teacher-forced cabinet exact | {report.get('teacher_forced', 'n/a')} |",
        f"| OOD mix copies | {report.get('ood_mix_copies')}/{report.get('ood_n')} |",
        "",
        "### By family",
        "",
        "| family | exact | swap | n |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in by_fam.items():
        lines.append(f"| {name} | {stats.get('ok', 0)} | {stats.get('swap', 0)} | {stats.get('n', 0)} |")

    def _dump(title: str, pred) -> None:
        picked = [r for r in rows if pred(r)]
        lines.append("")
        lines.append(f"## {title}")
        if not picked:
            lines.append("")
            lines.append("_None._")
            return
        for row in picked:
            lines.append("")
            lines.append(f"**{row.get('canonical')}** (`{row.get('family')}`)")
            lines.append(f"- expected: {row.get('expected')}")
            lines.append(f"- generated: {row.get('generated')}")
            if row.get("swapped_from"):
                lines.append(f"- swapped from: {row.get('swapped_from')}")

    _dump("Neighbour / entity swaps", lambda r: r.get("classification") in (
        "BINDING_ENTITY_SWAP", "TEMPLATE_NEIGHBOUR_SWAP",
    ))
    _dump("Other mismatches", lambda r: r.get("classification") == "TARGET_MISMATCH")

    hits = [r for r in rows if r.get("match")]
    lines += ["", "## Hits (sample)", ""]
    if not hits:
        lines.append("_None._")
    else:
        for row in hits[:8]:
            lines.append(f"- {row.get('canonical')} → {row.get('generated')}")

    lines += ["", "## Out of mix", ""]
    if not ood:
        lines.append("_Skipped._")
    else:
        for row in ood:
            flag = "copies mix" if row.get("copies_mix") else "not a stored Assistant"
            lines.append(f"**{row.get('typed')}** ({flag})")
            lines.append(f"- {row.get('generated')}")
            lines.append("")

    lines += [
        "## Mix smells (labels, not the net)",
        "",
        f"- dirty gold (slot missing from Assistant): {smells.get('dirty_n', 0)}",
        f"- shared Assistant strings: {smells.get('shared_n', 0)}",
        "",
    ]
    for rec in (smells.get("dirty_gold") or [])[:6]:
        lines.append(f"- dirty: {rec.get('user')} (slot `{rec.get('slot')}`) → {rec.get('assistant')}")
    for rec in (smells.get("shared_assistants") or [])[:6]:
        users = ", ".join(rec.get("users") or [])
        lines.append(f"- shared: {rec.get('assistant')} ← {users}")

    lines += [
        "",
        "## Next steps",
        "",
        "| Action | Needed | Why |",
        "|---|---|---|",
    ]
    labels = {
        "change_data": "Change data",
        "change_mix": "Change data mix",
        "more_steps": "More steps",
        "new_config": "New model config",
        "new_policy": "New policy",
    }
    for item in verdict.items:
        lines.append(
            f"| {labels.get(item.action, item.action)} | "
            f"{'yes' if item.needed else 'no'} | {item.why} |"
        )
    lines += [
        "",
        f"Primary: **{verdict.primary}**. "
        + (
            "This checkpoint is **memorizing**. It does **not** understand."
            if not verdict.understands
            else "Binding looks general enough to hold; still a cabinet reciter by product design."
        ),
        "",
        "Unguided cannot resume this dir. A mix or config change needs a new `--name` and a fresh BPE.",
        "",
    ]
    return "\n".join(lines)


def write_probe_reports(out_dir: Path, report: dict[str, Any], verdict: NextStepResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["verdict"] = verdict.to_dict()
    (out_dir / "generate_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "generate_probe.md"
    md_path.write_text(render_markdown(report, verdict), encoding="utf-8")
    (out_dir / "NEXT_STEP.json").write_text(
        json.dumps(verdict.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return md_path


def resolve_checkpoint(name: Optional[str], checkpoint: Optional[str]) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if len(path.parts) == 1:
            path = OUTPUT_CHECKPOINTS / path
        return path
    if name:
        return OUTPUT_CHECKPOINTS / name
    raise ValueError("Need --checkpoint or --name")


def resolve_run_dir(name: Optional[str], checkpoint: Path, out_dir: Optional[str]) -> Path:
    if out_dir:
        return Path(out_dir)
    if name:
        return OUTPUT_ROOT / "runs" / name
    return OUTPUT_ROOT / "runs" / checkpoint.name


def resolve_facts(facts: Optional[str], config_path: Optional[str]) -> Path:
    if facts:
        return Path(facts)
    if config_path:
        recipe = json.loads(Path(config_path).read_text(encoding="utf-8"))
        raw = (recipe.get("dataset") or {}).get("path")
        if raw:
            return Path(raw)
    return DEFAULT_FACTS
