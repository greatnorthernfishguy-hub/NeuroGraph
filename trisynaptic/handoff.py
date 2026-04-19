# ---- Changelog ----
# [2026-04-19] Claude Code (Opus 4.7, 1M) — TriSyn Phase 1 initial
# What: Atomic handoff-file helpers for manager↔worker communication.
# Why: Manager writes backlog + config snapshot; worker reads. Atomicity
#   prevents partial-read during concurrent spawn/read window.
# How: tempfile + rename pattern. JSON default handles numpy arrays
#   (forest_embedding) via .tolist() coercion at serialization time.
# -------------------
"""TriSyn handoff file helpers — atomic write/read between manager and worker."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


HANDOFF_DIR = Path("/tmp")
HANDOFF_PREFIX = "trisynaptic_handoff_"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to path atomically via tmp-file + rename."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, default=_json_default)
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_handoff(path: Path) -> Dict[str, Any]:
    """Read and parse handoff JSON."""
    with open(path, "r") as fh:
        return json.load(fh)


def list_orphan_handoffs() -> List[Path]:
    """Return leftover handoff files (worker crash or NG restart orphans)."""
    return sorted(HANDOFF_DIR.glob(f"{HANDOFF_PREFIX}*.json"))
