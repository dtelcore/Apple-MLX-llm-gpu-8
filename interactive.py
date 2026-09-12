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
    :related            reprint trained follow-up questions
    :trace on|off       toggle all tracing for subsequent turns
    :quit / :exit       leave the REPL
"""

from __future__ import annotations

import argparse

from training.chat_session import ChatSession, add_session_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive REPL for a trained checkpoint")
    add_session_args(parser)
    return parser.parse_args()


def run_repl(args: argparse.Namespace, *, configure_logging: bool = True) -> None:
    """Runs the interactive generation REPL for the checkpoint in `args.checkpoint`.
    Reusable by other CLIs (e.g. train.py --generate) that build their own args."""
    args._entry = getattr(args, "_entry", None) or "interactive"
    session = ChatSession.from_args(args, configure_logging=configure_logging)
    gpt_config = session.gpt_config
    log_path = getattr(session, "log_path", None)

    print("=" * 70)
    mode_label = "chat" if session.chat_mode else "prompt"
    if session.router_on:
        mode_label = f"{mode_label}+router"
    print(f"INTERACTIVE GENERATION -- checkpoint: {args.checkpoint} ({mode_label})")
    print(f"Model: {gpt_config.name} | vocab={gpt_config.vocab_size} | max_len={gpt_config.max_len}")
    if log_path is not None:
        print(f"Log: {log_path}")
    cmds = ":temp N  :tokens N  :topk N  :topp N  :trace on|off  :quit"
    if session.chat_mode:
        cmds = ":clear  :system TEXT  " + cmds
        print("Chat format: User: … Assistant: …  (history truncated from the front)")
        if session.system:
            print(f"System: {session.system}")
    if session.router_on:
        cmds = ":search Q  :calc EXPR  :route  :related  " + cmds
        search_note = "on" if session.search_enabled else "off"
        print(f"Router: cabinet → calc → Wikipedia({search_note}) → miss  (one checkpoint)")
        print(f"Learned KB: Wikipedia hits append to {session.learned_path} (replayed, not trained)")
    print(f"Type a prompt and press Enter. Commands: {cmds}")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        result = session.turn(prompt)
        if result.quit:
            break
        if not result.text:
            continue
        if result.learned_added:
            print(f"[cabinet] saved {result.learned_added} → {session.learned_path}")
        print(result.text)
        if result.related:
            print("[related]")
            for q in result.related:
                print(f"  - {q}")


def main() -> None:
    args = parse_args()
    run_repl(args)


if __name__ == "__main__":
    main()
