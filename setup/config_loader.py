"""Load training config and resolve dataset corpus without embedding it in JSON."""

import shutil
from pathlib import Path
from typing import Dict, List

from logging_config import logger
from paths import BUNDLED_SMOKE_CORPUS, DATA_DIR
from setup.dataset_setup import COMBINED_DATASET_NAME, DatasetLoader


def is_combined_dataset(dataset_cfg: Dict) -> bool:
    """True when config asks to concatenate every data/*.txt file."""
    if dataset_cfg.get("combine"):
        return True
    return dataset_cfg.get("name") == COMBINED_DATASET_NAME


def resolve_dataset_corpus(dataset_cfg: Dict, data_dir: str = None) -> List[str]:
    """Return corpus text from inline config, combined data dir, or dataset name.

    Combined-directory mode (`combine: true` or `name: data_dir`) loads every
    non-empty line from sorted `*.txt` files under `data_dir`. Inline `corpus`
    still wins so tests and baked-in configs keep working. Single-file stems
    and built-in names are unchanged.
    """
    corpus = dataset_cfg.get("corpus")
    if corpus:
        return corpus

    resolved_dir = data_dir or str(DATA_DIR)
    loader = DatasetLoader(data_dir=resolved_dir)
    if is_combined_dataset(dataset_cfg):
        try:
            return loader.load_combined_directory(resolved_dir)
        except FileNotFoundError:
            if _try_seed_bundled_smoke_corpus(resolved_dir):
                return loader.load_combined_directory(resolved_dir)
            raise

    name = dataset_cfg.get("name", "minimal")
    return loader.load_by_name(name)


def _try_seed_bundled_smoke_corpus(resolved_dir: str) -> bool:
    """Copy setup/smoke_english.txt into the default data/ dir when it is empty.

    Only seeds the project data/ directory so tests that use a temp folder still
    see FileNotFoundError. Returns True if a seed file was written.
    """
    try:
        if Path(resolved_dir).resolve() != DATA_DIR.resolve():
            return False
    except OSError:
        return False
    if not BUNDLED_SMOKE_CORPUS.is_file():
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if any(DATA_DIR.glob("*.txt")):
        return False
    dest = DATA_DIR / BUNDLED_SMOKE_CORPUS.name
    shutil.copy2(BUNDLED_SMOKE_CORPUS, dest)
    logger.warning(
        "No data/*.txt found under %s; seeded bundled smoke corpus %s. "
        "Replace this file with a real English corpus before a long run.",
        DATA_DIR, dest,
    )
    return dest.is_file() and dest.stat().st_size > 0
