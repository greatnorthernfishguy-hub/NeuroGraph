# ---- Changelog ----
# [2026-04-19] Claude Code (Opus 4.7, 1M) — TriSyn Phase 1 initial
# What: Subprocess entry point. Reads handoff file, processes each backlog
#   entry via NGEmbed.dual_record_outcome (the canonical ecosystem concept
#   extraction path), records run-narrative outcome at exit.
# Why: Offloads blocking TID calls from NG's main process into a
#   systemd-run-isolated subprocess. MemoryMax / CPUQuota constrained.
# How: ThreadPoolExecutor parallelizes TID calls within the worker.
#   Module_id="neurograph" — deposits land in the same tract stream as
#   the in-process concept pulse used to write to. No new consumer wiring.
#   Advisor fix #2: reuses dual_record_outcome rather than ad-hoc prompt.
# -------------------
"""TriSyn concept-extraction worker — subprocess entry point.

Invoked via systemd-run --user --scope with MemoryMax=1G and CPUQuota=50%.
Reads TRISYN_HANDOFF env var for the handoff JSON path. Exits cleanly when
backlog exhausted, TID fail threshold hit, or max runtime reached.

Exit codes:
    0 = normal (backlog exhausted)
    1 = usage / setup error
    2 = TID failure threshold exceeded
    3 = max runtime exceeded
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict


# Make NeuroGraph importable when invoked from systemd-run (no cwd guarantee)
_NG_ROOT = Path(__file__).resolve().parent.parent
if str(_NG_ROOT) not in sys.path:
    sys.path.insert(0, str(_NG_ROOT))

from ng_embed import NGEmbed  # noqa: E402
from ng_tract_bridge import NGTractBridge  # noqa: E402
from trisynaptic.handoff import read_handoff  # noqa: E402


logger = logging.getLogger("trisyn.worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [trisyn.worker] %(message)s",
)


class _PeerBridgeEco:
    """Minimal ecosystem adapter for NGEmbed.dual_record_outcome.

    Mirrors wire_absorption._PeerBridgeEco (private in that module, so
    reimplemented here to avoid depending on private API).
    """

    def __init__(self, bridge: NGTractBridge, module_id: str = "neurograph"):
        self._bridge = bridge
        self._module_id = module_id

    def record_outcome(self, embedding, target_id, success,
                       strength: float = 1.0, metadata=None):
        meta = dict(metadata or {})
        if strength != 1.0:
            meta["_strength"] = strength
        return self._bridge.record_outcome(
            embedding, target_id, success, self._module_id, meta,
        )


def _process_entry(embedder: NGEmbed, eco: _PeerBridgeEco,
                   entry: Dict[str, Any]) -> Dict[str, Any]:
    """Run dual_record_outcome on one backlog entry. Returns summary dict."""
    import numpy as np

    event_node_id = entry["event_node_id"]
    content = entry.get("content_preview", "")
    source = entry.get("source", "unknown")

    forest_emb = np.asarray(entry["forest_embedding"], dtype=np.float32)

    result = embedder.dual_record_outcome(
        ecosystem=eco,
        content=content,
        embedding=forest_emb,
        target_id=event_node_id,
        success=True,
        strength=1.0,
        metadata={"source": source, "source_type": "wire"},
    )
    return {
        "event_node_id": event_node_id,
        "trees": len(result.get("tree_ids", [])),
        "concepts": len(result.get("concepts", [])),
        "pass2_attempted": result.get("pass2_attempted", False),
    }


def main() -> int:
    handoff_path_str = os.environ.get("TRISYN_HANDOFF")
    if not handoff_path_str:
        logger.error("TRISYN_HANDOFF env var not set")
        return 1
    handoff_path = Path(handoff_path_str)
    if not handoff_path.exists():
        logger.error("Handoff file not found: %s", handoff_path)
        return 1

    payload = read_handoff(handoff_path)
    entries = payload.get("entries", [])
    config = payload.get("config", {})
    mode = payload.get("spawn_mode", "unspecified")

    max_workers = int(config.get("trisyn_max_workers", 1))
    tid_fail_limit = int(config.get("trisyn_tid_fail_exit", 5))
    max_runtime_s = float(config.get("trisyn_max_runtime_s", 1800))

    logger.info(
        "Starting run: mode=%s, entries=%d, max_workers=%d, "
        "tid_fail_exit=%d, max_runtime=%.0fs",
        mode, len(entries), max_workers, tid_fail_limit, max_runtime_s,
    )

    embedder = NGEmbed.get_instance()
    bridge = NGTractBridge(module_id="neurograph")
    eco = _PeerBridgeEco(bridge, module_id="neurograph")

    started = time.time()
    tid_failures = 0
    trees_total = 0
    entries_processed = 0
    exit_reason = "complete"
    exit_code = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_entry, embedder, eco, e): e
                   for e in entries}
        for fut in as_completed(futures):
            if time.time() - started > max_runtime_s:
                exit_reason = "max_runtime"
                exit_code = 3
                break
            try:
                res = fut.result()
                entries_processed += 1
                trees_total += res["trees"]
                # Treat pass2_attempted=True with 0 concepts as TID failure
                # (extraction ran but got nothing back — rate limit or model
                # unreachable).
                if res.get("pass2_attempted") and res["concepts"] == 0:
                    tid_failures += 1
                    if tid_failures >= tid_fail_limit:
                        exit_reason = "tid_unhealthy"
                        exit_code = 2
                        break
            except Exception as exc:
                tid_failures += 1
                logger.warning("Entry failed: %s", exc)
                if tid_failures >= tid_fail_limit:
                    exit_reason = "tid_unhealthy"
                    exit_code = 2
                    break

    wall_time = time.time() - started

    narrative = (
        f"TriSyn run {int(started)}: mode={mode}, processed "
        f"{entries_processed}/{len(entries)} entries over "
        f"{wall_time:.1f}s, {tid_failures} TID failures, "
        f"{trees_total} trees extracted, exit_reason={exit_reason}."
    )
    logger.info(narrative)

    try:
        narrative_emb = embedder.embed(narrative)
        bridge.record_outcome(
            narrative_emb,
            f"trisyn:run:{int(started)}",
            exit_code == 0,
            "neurograph",
            {
                "narrative": narrative,
                "mode": mode,
                "entries_total": len(entries),
                "entries_processed": entries_processed,
                "trees_total": trees_total,
                "tid_failures": tid_failures,
                "wall_time_s": wall_time,
                "exit_reason": exit_reason,
            },
        )
    except Exception as exc:
        logger.warning("Narrative deposit failed: %s", exc)

    # Only delete handoff on clean exit; manager triages non-zero exits
    if exit_code == 0:
        try:
            handoff_path.unlink()
        except OSError:
            pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
