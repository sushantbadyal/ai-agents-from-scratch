"""
================================================================================
config.py  —  Central configuration + shared infrastructure
================================================================================

WHAT THIS FILE DOES
-------------------
Every other file in this package imports its knobs from here instead of
hard-coding them, so there is one place to retune the model, dataset, paths,
and defaults:

  * One place to change the model, the dataset, the index location, `k`, etc.
  * All values come from the "sparse_retrieval_configs" map in the shared app
    config JSON (see utils.constants.APP_CONFIG_PATH), so the same file is the
    single source of truth on a laptop, in CI, and on a GPU box.
  * A single, consistently-formatted `logging` setup for diagnostics (instead of
    scattering bare `print()` calls).
  * A tiny `import_step()` helper so the numbered, digit-prefixed modules
    (`03_build_inverted_index.py`, ...) can import each other cleanly — a plain
    `import 03_...` is a Python SyntaxError.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    (standard library: logging, sys, importlib, pathlib)
    (project: configs.app_config)
================================================================================
"""

from __future__ import annotations                            # Allow forward-referenced type hints on older Pythons

import logging                                                # Standard library: structured logging
import sys                                                    # Standard library: extend the import path to reach src/
import importlib                                               # Standard library: import modules by string name
from pathlib import Path                                      # Standard library: object-oriented filesystem paths
from types import ModuleType                                  # Standard library: type hint for imported modules

# This file lives at src/rag/sparse_retrieval/config.py and runs as a standalone
# script's neighbour (not as part of an installed package), so `src/` isn't on
# sys.path by default. Add it so `configs.app_config` can be imported below.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from configs.app_config import config as _app_config          # Project: single shared config, loaded from JSON

_CFG = _app_config.sparse_retrieval_configs                    # This package's slice of the app config

# ── Generator model ───────────────────────────────────────────────────────────
MODEL_ID: str = _CFG.MODEL_ID

# ── Corpus ─────────────────────────────────────────────────────────────────────
WIKI_DATASET: str = _CFG.WIKI_DATASET                          # HF dataset id

# ── Retrieval / generation defaults ─────────────────────────────────────────────
DEFAULT_K: int = _CFG.DEFAULT_K                                 # How many passages to retrieve per query
DEFAULT_MAX_NEW_TOKENS: int = _CFG.DEFAULT_MAX_NEW_TOKENS       # Cap on generated answer length

# ── Index storage ────────────────────────────────────────────────────────────
INDEX_ROOT: Path = Path(_CFG.INDEX_ROOT)                       # Where Terrier writes its on-disk indexes
WIKI_INDEX_NAME: str = _CFG.WIKI_INDEX_NAME                     # Sub-folder name for the wikipedia index
SYNTHETIC_INDEX_NAME: str = _CFG.SYNTHETIC_INDEX_NAME           # Sub-folder name for the synthetic index
FORCE_REBUILD_INDEX: bool = _CFG.FORCE_REBUILD_INDEX            # Default: reuse a cached index instead of rebuilding

# ── Visualization output ────────────────────────────────────────────────────────
VIZ_OUTPUT_DIR: Path = Path(_CFG.VIZ_OUTPUT_DIR)                # Where plots (e.g. TF-IDF heatmaps) are saved

# ── Logging ────────────────────────────────────────────────────────────────────
_LOGGING_CONFIGURED = False                                   # Module flag so we configure logging exactly once


def configure_logging() -> None:
    """Set up root logging once, honouring the sparse_retrieval_configs.LOG_LEVEL."""
    global _LOGGING_CONFIGURED                                # We flip the module-level guard
    if _LOGGING_CONFIGURED:                                   # If already configured, do nothing (idempotent)
        return                                                # Early return keeps repeated calls cheap
    logging.basicConfig(                                      # Configure the root logger
        level=_CFG.LOG_LEVEL,                                 # Verbosity, from the shared config
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",  # Consistent, parseable log lines
        datefmt="%H:%M:%S",                                   # Compact timestamps
    )
    _LOGGING_CONFIGURED = True                                # Remember that we've configured logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Every module wants to report progress and errors, but bare `print()`
        can't be filtered by importance or silenced in tests. Handing each file a
        logger named after itself gives uniform, level-controlled output where you
        can always tell which step a message came from — and logging is set up
        automatically the first time any file asks for a logger.
    """
    configure_logging()                                       # Ensure handlers/format are set up
    return logging.getLogger(name)                            # Hand back a namespaced logger


def import_step(module_name: str) -> ModuleType:
    """Import a digit-prefixed step module (e.g. '03_build_inverted_index') by name.

    WHY THIS EXISTS / WHAT IT SOLVES:
        We prefix files with 01_, 02_, ... to make the reading/running ORDER
        obvious, but that same prefix makes them un-importable the normal way
        (`import 03_...` is a SyntaxError — identifiers can't start with a digit).
        This helper lets later steps reuse earlier ones by name so we get both the
        numbered ordering AND normal code reuse, with the workaround in one place.
    """
    return importlib.import_module(module_name)               # Resolve + import the module object


def index_path(index_name: str) -> Path:
    """Return the absolute path where a named index lives, creating parents if needed.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Both indexes (wiki, synthetic) need a consistent, writable home on disk.
        Centralising the path logic here means no file hard-codes a directory, and
        the parent folder is guaranteed to exist before Terrier tries to write to it.
    """
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)             # Ensure the index root directory exists
    return INDEX_ROOT / index_name                            # Full path to this specific index folder


def start_terrier() -> None:
    """Start Terrier's JVM backend if this PyTerrier version needs an explicit init.

    WHY THIS EXISTS / WHAT IT SOLVES:
        PyTerrier is only a thin Python wrapper — the real search engine is Java,
        so a JVM has to be running before we can index or search. Newer PyTerrier
        boots it automatically, older versions don't, and this smooths over that
        difference so callers never have to care. The import is deliberately lazy
        so files that never touch Terrier (01, 02, 05) don't pay JVM startup cost.
    """
    import pyterrier as pt                                    # Lazy import: only when Terrier is actually needed
    if hasattr(pt, "started") and not pt.started():           # Older API exposes .started(); check if JVM is off
        pt.init()                                             # Explicitly boot the JVM + download Terrier jars
