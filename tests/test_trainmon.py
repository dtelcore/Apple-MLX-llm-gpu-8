"""Host-only train monitor: parse plotter-style logs, no Metal."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import create_app
from app.trainmon import create_app as create_trainmon, series_payload


def _write_train_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2026-09-13 15:00:00 | INFO | llm_gpu | [train] step=10/500 epoch=1 loss=3.2100 ppl=24.78 tok_s=120 lr=0.0003",
                "2026-09-13 15:00:05 | INFO | llm_gpu | [train] step=20/500 epoch=1 loss=2.9800 ppl=19.69 tok_s=118 lr=0.0003",
                "2026-09-13 15:00:10 | INFO | llm_gpu | [train] step=30/500 epoch=1 loss=2.7400 ppl=15.49 tok_s=121 lr=0.0003",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class SeriesPayloadTests(unittest.TestCase):
    def test_same_step_val_line_keeps_train_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "training_demo.log"
            log.write_text(
                "\n".join(
                    [
                        "[train] step=10/20 loss=2.1000 ppl=8.2 tok_s=99",
                        "[train] step=20/20 loss=2.0000 ppl=7.4 tok_s=100",
                        "[train] step=20/20 val_loss=2.5000 val_ppl=12.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = series_payload(log, keep_short=True)
            self.assertEqual(payload["last"]["loss"], 2.0)
            self.assertEqual(payload["last"]["tok_s"], 100.0)
            self.assertEqual(payload["last"]["val_loss"], 2.5)

    def test_plotter_lines_and_volatility(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "unguided_demo.log"
            _write_train_log(log)
            payload = series_payload(log, keep_short=True)
            self.assertEqual(payload["source"], "log")
            self.assertEqual(payload["steps"], [10, 20, 30])
            self.assertEqual(payload["loss"][0], 3.21)
            self.assertEqual(payload["last"]["step"], 30)
            self.assertEqual(payload["last"]["total"], 500)
            self.assertEqual(len(payload["volatility"]), 3)
            self.assertIsNotNone(payload["volatility"][-1])

    def test_decisions_fallback_when_log_has_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "logs" / "unguided_demo.log"
            log.parent.mkdir()
            log.write_text("session ready\n", encoding="utf-8")
            decisions = root / "runs" / "demo" / "decisions.jsonl"
            decisions.parent.mkdir(parents=True)
            decisions.write_text(
                json.dumps({"step": 50, "action": "continue", "reason": "ok", "val_loss": 2.4})
                + "\n"
                + json.dumps({"step": 100, "action": "promote", "reason": "best", "val_loss": 2.1})
                + "\n",
                encoding="utf-8",
            )
            payload = series_payload(log, keep_short=True, runs_root=root / "runs")
            self.assertEqual(payload["source"], "decisions")
            self.assertEqual(payload["steps"], [50, 100])
            self.assertEqual(payload["val_loss"][-1], 2.1)
            self.assertEqual(payload["decisions"][-1]["action"], "promote")


class TrainmonAppTests(unittest.TestCase):
    def test_logs_and_series_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "training_demo.log")
            app = create_trainmon(log_dir=logs)
            client = app.test_client()
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b"Train monitor", home.data)
            listing = client.get("/api/logs")
            self.assertEqual(listing.status_code, 200)
            names = [row["name"] for row in listing.get_json()["logs"]]
            self.assertEqual(names, ["training_demo.log"])
            series = client.get("/api/series", query_string={"log": "training_demo.log"})
            self.assertEqual(series.status_code, 200)
            body = series.get_json()
            self.assertEqual(body["steps"], [10, 20, 30])
            self.assertAlmostEqual(body["tok_s"][0], 120.0)

    def test_rejects_path_outside_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "ok.log")
            app = create_trainmon(log_dir=logs)
            client = app.test_client()
            escaped = client.get("/api/series", query_string={"log": "../ok.log"})
            # basename is still ok.log inside the log dir
            self.assertEqual(escaped.status_code, 200)
            missing = client.get("/api/series", query_string={"log": "nope.log"})
            self.assertEqual(missing.status_code, 404)


class AppTrainTabTests(unittest.TestCase):
    def test_shell_exposes_train_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "live.log")
            app = create_app(root=Path(tmp), models_root=Path(tmp), log_dir=logs)
            client = app.test_client()
            home = client.get("/?view=train")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b'id="tab-train"', home.data)
            pane = client.get("/train/")
            self.assertEqual(pane.status_code, 200)
            self.assertIn(b"Train monitor", pane.data)
            series = client.get("/train/api/series", query_string={"log": "live.log"})
            self.assertEqual(series.status_code, 200)
            self.assertEqual(series.get_json()["last"]["step"], 30)


if __name__ == "__main__":
    unittest.main()
