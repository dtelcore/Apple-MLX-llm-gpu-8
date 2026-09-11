"""Flask chat UI around ChatSession (same path as interactive.py).

One checkpoint, 2 GB cap. Do not run this and interactive.py at the same time.

Usage:
    python webui.py --checkpoint output/checkpoints/chat_facts_v5 --chat
    python webui.py --checkpoint output/checkpoints/chat_facts_v6 --chat --facts data/chat_facts_v6.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from training.chat_session import ChatSession, add_session_args

_WEBUI_DIR = Path(__file__).resolve().parent / "webui"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask chat UI for the interactive session")
    add_session_args(parser)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default loopback)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    return parser.parse_args()


def create_app(session: ChatSession) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_WEBUI_DIR / "templates"),
        static_folder=None,
    )
    app.config["SESSION"] = session

    @app.get("/")
    def index():
        return render_template("index.html", status=session.status())

    @app.get("/api/status")
    def api_status():
        return jsonify(session.status())

    @app.post("/api/chat")
    def api_chat():
        body = request.get_json(silent=True) or {}
        message = str(body.get("message") or body.get("prompt") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        result = session.turn(message)
        if result.quit:
            return jsonify({"error": "quit is a CLI command"}), 400
        return jsonify(
            {
                "reply": result.text,
                "kind": result.kind,
                "detail": result.detail,
                "learned_added": result.learned_added,
            }
        )

    @app.post("/api/clear")
    def api_clear():
        session.clear()
        return jsonify({"ok": True})

    @app.post("/v1/chat/completions")
    def openai_chat():
        """Minimal OpenAI-style endpoint so other UIs can point here."""
        body = request.get_json(silent=True) or {}
        messages = body.get("messages") or []
        content = ""
        if messages:
            content = str((messages[-1] or {}).get("content") or "")
        elif body.get("prompt"):
            content = str(body.get("prompt"))
        content = content.strip()
        if not content:
            return jsonify({"error": {"message": "empty messages"}}), 400
        result = session.turn(content)
        return jsonify(
            {
                "id": "chatcmpl-mlx",
                "object": "chat.completion",
                "model": session.status()["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return app


def main() -> None:
    args = parse_args()
    args._entry = "webui"
    session = ChatSession.from_args(args, configure_logging=True)
    app = create_app(session)
    st = session.status()
    print("=" * 70)
    print(f"WEBUI -- checkpoint: {st['checkpoint']}  model: {st['model']}")
    print(f"Open http://{args.host}:{args.port}  (one checkpoint, 2 GB; quit interactive.py first)")
    print("=" * 70)
    app.run(host=args.host, port=int(args.port), debug=False, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
