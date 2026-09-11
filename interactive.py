"""
interactive.py

Interactive REPL for sampling from a trained checkpoint. Each prompt you
type continues from the model; trace flags apply to every generation
you run in the session.

Usage:
    python interactive.py --checkpoint output/checkpoints/run1
    python interactive.py --checkpoint output/checkpoints/chat_facts_v4 --chat
    python interactive.py --checkpoint output/checkpoints/run1 --trace-tokens --trace-logits --trace-every 1

Chat checkpoints default to --router (cabinet → calc → Wikipedia → miss).
Story checkpoints stay generate-every-turn. Never loads a second net.

Session commands:
    :temp <value>       set sampling temperature
    :tokens <n>         set max new tokens per turn
    :topk <n>|none      set top-k
    :topp <n>|none      set top-p
    :clear              reset chat history (chat mode)
    :system <text>      set / replace the system prefix (chat mode)
    :search <q>         Wikipedia summary (router; saved to learned JSONL)
    :calc <expr>        safe Decimal arithmetic (router)
    :route              print last router decision
    :trace on|off       toggle all tracing for subsequent turns
    :quit / :exit       leave the REPL
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

import cli_common
from logging_config import logger, setup_generate_run_logging
from model.gpt import GPTModel
from paths import DATA_DIR, OUTPUT_ROOT, ensure_output_dirs
from training.router import (
    MISS_HINT,
    RouteDecision,
    alias_learned_topics,
    remember_search_hit,
    route,
    router_enabled,
    search_topic,
    try_calc_query,
)
from tools.wiki_search import wiki_summary
from training.cabinet_index import CabinetIndex, load_cabinet, merge_cabinet
from training.checkpoint import load_checkpoint
from training.chat_format import (
    ASSISTANT_ROLE,
    CHAT_STOP_STRINGS,
    DEFAULT_CHAT_SYSTEM,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_CHAT_TOP_P,
    USER_ROLE,
    build_chat_prompt_ids,
    is_chat_model_name,
    sanitize_assistant_reply,
)

CABINET_GENERATE_TEMP = 0.2
CABINET_GENERATE_TOP_K = 10
DEFAULT_FACTS = DATA_DIR / "chat_facts.jsonl"
DEFAULT_LEARNED = OUTPUT_ROOT / "cabinet_learned.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive REPL for a trained checkpoint")
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--chat", action="store_true", help="Multi-turn User/Assistant history")
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Force single-prompt REPL even if the checkpoint name looks like chat_5m",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Optional system prefix for --chat (default: a short simple-assistant line)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Initial sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Initial tokens generated per turn")
    parser.add_argument("--top-k", type=int, default=None, help="Only sample from top K tokens")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument(
        "--stop", action="append", default=None,
        help="Stop string (repeatable). Chat mode defaults to User: role markers.",
    )
    parser.add_argument(
        "--router", dest="router", action="store_true",
        help="Force cabinet/calc/search router (default on for chat checkpoints)",
    )
    parser.add_argument(
        "--no-router", dest="router", action="store_false",
        help="Generate every turn (story REPL; default for non-chat checkpoints)",
    )
    parser.set_defaults(router=None)
    parser.add_argument(
        "--facts", type=str, default=str(DEFAULT_FACTS),
        help="Cabinet JSONL/txt for --router (default: data/chat_facts.jsonl)",
    )
    parser.add_argument(
        "--learned", type=str, default=str(DEFAULT_LEARNED),
        help="JSONL overlay for Wikipedia hits (default: output/cabinet_learned.jsonl)",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="With --router, skip Wikipedia (fact-shaped misses go to the polite hint)",
    )
    cli_common.add_generate_decode_args(parser)
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def _resolve_chat_mode(args: argparse.Namespace, model_name: str) -> bool:
    if getattr(args, "no_chat", False):
        return False
    if getattr(args, "chat", False):
        return True
    return is_chat_model_name(model_name)


def _load_index(facts_path: str, learned_path: str) -> CabinetIndex:
    index = CabinetIndex()
    src = facts_path
    try:
        index = load_cabinet(facts_path)
    except FileNotFoundError:
        logger.warning("cabinet facts missing at %s; starting empty trained index", facts_path)
        print(f"[router] facts not found: {facts_path} (trained cabinet empty)")
        src = facts_path
    except (OSError, ValueError) as exc:
        logger.warning("cabinet facts failed to load from %s: %s", facts_path, exc)
        print(f"[router] could not load facts: {exc}")
    n_trained = len(index)
    n_learned = merge_cabinet(index, learned_path, source="learned")
    alias_learned_topics(index)
    print(
        f"[router] cabinet index: {len(index)} unique "
        f"({n_trained} trained from {src}, {n_learned} learned from {learned_path})"
    )
    return index


def run_repl(args: argparse.Namespace, *, configure_logging: bool = True) -> None:
    """Runs the interactive generation REPL for the checkpoint in `args.checkpoint`.
    Reusable by other CLIs (e.g. train.py --generate) that build their own args."""
    ensure_output_dirs()
    log_path = None
    if configure_logging:
        log_path = setup_generate_run_logging(args.checkpoint)
        source = getattr(args, "_entry", None) or "interactive"
        logger.info(
            "interactive generation | source=%s | checkpoint=%s | log=%s",
            source, args.checkpoint, log_path,
        )

    gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
    model = GPTModel(gpt_config, params)
    tracer = cli_common.build_tracer(args, default_trace_every=1)
    rng = np.random.default_rng(args.seed)

    model_name = getattr(gpt_config, "name", "") or ""
    chat_mode = _resolve_chat_mode(args, model_name)
    router_on = router_enabled(getattr(args, "router", None), model_name)
    search_enabled = router_on and not bool(getattr(args, "no_search", False))
    learned_path = str(getattr(args, "learned", DEFAULT_LEARNED))
    cabinet: Optional[CabinetIndex] = None
    if router_on:
        cabinet = _load_index(str(getattr(args, "facts", DEFAULT_FACTS)), learned_path)

    if chat_mode:
        temperature = args.temperature if args.temperature is not None else DEFAULT_CHAT_TEMPERATURE
        top_k = args.top_k if args.top_k is not None else DEFAULT_CHAT_TOP_K
        top_p = args.top_p if args.top_p is not None else DEFAULT_CHAT_TOP_P
        system = args.system if args.system is not None else DEFAULT_CHAT_SYSTEM
        stop_strings = list(args.stop) if args.stop else list(CHAT_STOP_STRINGS)
    else:
        temperature = args.temperature if args.temperature is not None else 0.8
        top_k = getattr(args, "top_k", None)
        top_p = getattr(args, "top_p", None)
        system = args.system
        stop_strings = list(args.stop) if args.stop else None
    max_new_tokens = args.max_new_tokens
    trace_enabled = tracer.any_enabled
    history: List[tuple] = []
    use_kv_cache = not getattr(args, "no_kv_cache", False)
    use_cuda_graph = bool(getattr(args, "cuda_graph", False))
    last_route: Optional[RouteDecision] = None

    def _generate(prompt_ids, prompt_text, *, temp, k, p) -> str:
        active_tracer = tracer if trace_enabled else None
        if active_tracer is not None:
            active_tracer.dump_tokens(prompt_ids, tokenizer, label="prompt")
        need_tokenizer = active_tracer is not None or bool(stop_strings)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=k,
            top_p=p,
            tracer=active_tracer,
            tokenizer=tokenizer if need_tokenizer else None,
            rng=rng,
            use_kv_cache=use_kv_cache,
            use_cuda_graph=use_cuda_graph,
            stop_strings=stop_strings,
        )
        new_ids = generated_ids[len(prompt_ids):]
        reply = tokenizer.decode(new_ids)
        if chat_mode:
            reply = sanitize_assistant_reply(reply, stop_strings)
        else:
            reply = reply.strip()
        full_text = tokenizer.decode(generated_ids)
        logger.info("prompt=%r generated_text:\n%s", prompt_text, full_text)
        return reply if chat_mode else full_text

    def _save_search(typed: str, extract: str) -> None:
        if cabinet is None or not extract:
            return
        before = len(cabinet)
        fact = remember_search_hit(cabinet, learned_path, typed, extract)
        if fact is None:
            return
        added = len(cabinet) - before
        if added:
            logger.info("cabinet learned +%s user=%r path=%s", added, typed, learned_path)
            print(f"[cabinet] saved {added} → {learned_path}")

    def _emit_user_reply(user_text: str, reply: str) -> None:
        print(reply)
        if chat_mode:
            history.append((USER_ROLE, user_text))
            history.append((ASSISTANT_ROLE, reply))

    print("=" * 70)
    mode_label = "chat" if chat_mode else "prompt"
    if router_on:
        mode_label = f"{mode_label}+router"
    print(f"INTERACTIVE GENERATION -- checkpoint: {args.checkpoint} ({mode_label})")
    print(f"Model: {gpt_config.name} | vocab={gpt_config.vocab_size} | max_len={gpt_config.max_len}")
    if log_path is not None:
        print(f"Log: {log_path}")
    cmds = ":temp N  :tokens N  :topk N  :topp N  :trace on|off  :quit"
    if chat_mode:
        cmds = ":clear  :system TEXT  " + cmds
        print(f"Chat format: User: … Assistant: …  (history truncated from the front)")
        if system:
            print(f"System: {system}")
    if router_on:
        cmds = ":search Q  :calc EXPR  :route  " + cmds
        search_note = "on" if search_enabled else "off"
        print(f"Router: cabinet → calc → Wikipedia({search_note}) → miss  (one checkpoint)")
        print(f"Learned KB: Wikipedia hits append to {learned_path} (replayed, not trained)")
    print(f"Type a prompt and press Enter. Commands: {cmds}")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt in (":quit", ":exit"):
            break
        if prompt.startswith(":temp "):
            temperature = float(prompt.split(maxsplit=1)[1])
            print(f"[temperature -> {temperature}]")
            continue
        if prompt.startswith(":tokens "):
            max_new_tokens = int(prompt.split(maxsplit=1)[1])
            print(f"[max_new_tokens -> {max_new_tokens}]")
            continue
        if prompt.startswith(":topk "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_k = None if raw in ("none", "off", "0") else int(raw)
            print(f"[top_k -> {top_k}]")
            continue
        if prompt.startswith(":topp "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_p = None if raw in ("none", "off", "1", "1.0") else float(raw)
            print(f"[top_p -> {top_p}]")
            continue
        if prompt.startswith(":trace "):
            trace_enabled = prompt.split(maxsplit=1)[1].strip().lower() == "on"
            print(f"[tracing -> {'on' if trace_enabled else 'off'}]")
            continue
        if chat_mode and prompt == ":clear":
            history = []
            print("[history cleared]")
            continue
        if chat_mode and prompt.startswith(":system"):
            rest = prompt[len(":system"):].strip()
            system = rest or None
            print(f"[system -> {system!r}]")
            continue
        if router_on and prompt == ":route":
            if last_route is None:
                print("[route] none yet")
            else:
                print(f"[route] kind={last_route.kind} detail={last_route.detail}")
            continue
        if router_on and prompt.startswith(":search"):
            raw_q = prompt[len(":search"):].strip()
            query = search_topic(raw_q) or raw_q
            extract = wiki_summary(query) if search_enabled and query else None
            if extract:
                last_route = RouteDecision(kind="search", text=extract, detail="wikipedia_forced")
                _save_search(raw_q or query, extract)
                _emit_user_reply(query or prompt, extract)
            else:
                last_route = RouteDecision(kind="miss", text=MISS_HINT, detail="search_failed")
                _emit_user_reply(query or prompt, MISS_HINT)
            continue
        if router_on and prompt.startswith(":calc"):
            expr = prompt[len(":calc"):].strip()
            value = try_calc_query(expr) if expr else None
            if value is not None:
                last_route = RouteDecision(kind="calc", text=value, detail="calc_forced")
                _emit_user_reply(expr, value)
            else:
                last_route = RouteDecision(kind="miss", text=MISS_HINT, detail="calc_failed")
                print("[calc] not a safe arithmetic expression")
            continue

        if router_on:
            decision = route(
                prompt,
                cabinet,
                search_enabled=search_enabled,
                search_fn=wiki_summary if search_enabled else None,
            )
            last_route = decision
            logger.info("route kind=%s detail=%s prompt=%r", decision.kind, decision.detail, prompt)
            if decision.kind == "cabinet" and decision.fact is not None:
                if decision.fact.source == "learned":
                    _emit_user_reply(prompt, decision.fact.assistant)
                    continue
                prompt_text = decision.fact.generate_prompt
                prompt_ids = tokenizer.encode(prompt_text)
                if not prompt_ids:
                    print("[No recognized characters in stored cabinet prompt]")
                    continue
                reply = _generate(
                    prompt_ids,
                    prompt_text,
                    temp=CABINET_GENERATE_TEMP,
                    k=CABINET_GENERATE_TOP_K,
                    p=None,
                )
                _emit_user_reply(prompt, reply)
                continue
            if decision.kind == "search":
                _save_search(prompt, decision.text)
                _emit_user_reply(prompt, decision.text)
                continue
            if decision.kind in ("calc", "miss"):
                _emit_user_reply(prompt, decision.text)
                continue

        if chat_mode:
            budget = max(1, int(gpt_config.max_len) - int(max_new_tokens))
            prompt_ids, prompt_text = build_chat_prompt_ids(
                tokenizer,
                history,
                prompt,
                system=system,
                max_prompt_tokens=budget,
            )
        else:
            prompt_text = prompt
            prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print("[No recognized characters in prompt for this vocabulary; try different text]")
            continue

        reply = _generate(prompt_ids, prompt_text, temp=temperature, k=top_k, p=top_p)
        if chat_mode:
            print(reply)
            history.append((USER_ROLE, prompt))
            history.append((ASSISTANT_ROLE, reply))
        else:
            print(reply)


def main() -> None:
    args = parse_args()
    run_repl(args)


if __name__ == "__main__":
    main()
