"""Project version. 0.1.1 trains story-packed TinyStories. 0.1.0 is the hybrid learner."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.1.0"


__version__ = _read_version()
