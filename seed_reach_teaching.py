#!/usr/bin/env python3
# ---- Changelog ----
# [2026-06-21] Claude Code (DudeMan CC, Opus 4.8) — #reach: seed Syl's reach-teaching node
# What: Seed selfcap::reach::teaching — the self-model node that SHOWS her the [[reach: …]]
#   gesture + worked examples, surfaced (competence-faded) through /assemble.
# Why: Voice/Hands gives her hands; this teaches her to reach (docs/prd/2026-06-21-reach-teaching-design).
#   Substrate-first / Law 7: a real node that surfaces, not a static prompt-staple.
# How: constitutional:True for permanent prune-protection (the never-silent FLOOR, #92) + selfcap
#   marker so /assemble routes it to the faded "How I Reach" block, NOT "Who I Am". Idempotent
#   (stable id). OFFLINE only — a HARD single-writer guard (Syl's Law / #261/#299) aborts if a
#   live NeuroGraph sidecar PID exists when targeting the live checkpoint (orphans fail to bind
#   8850 but still autosave, so we check PIDs, not the port).
# -------------------
import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_reach_teaching")

LIVE_CHECKPOINT = Path.home() / "NeuroGraph/data/checkpoints/main.msgpack"

REACH_NODE_ID = "selfcap::reach::teaching"
REACH_DESC = (
    "When I want to act in the world, I reach: I name what I want in my own words inside "
    "[[reach: …]], and my hands carry it out. The 🔧✓ is the world confirming the reach was mine."
)
REACH_EXAMPLES = [
    "[[reach: read the file at /docs/two-axis.md]]  →  🔧 read_file({\"path\": \"/docs/two-axis.md\"}) ✓",
    "[[reach: search the web for the latest on Fable 5]]  →  🔧 web_search({\"query\": \"Fable 5 latest\"}) ✓",
]


def _live_sidecar_pids():
    """Live NeuroGraph sidecar PIDs (by process name, not port — orphans still autosave)."""
    try:
        out = subprocess.run(["pgrep", "-f", "neurograph_rpc"],
                             capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001 — if pgrep is unavailable, fail SAFE (treat as live)
        return [-1]
    me = os.getpid()
    return [int(x) for x in out.stdout.split() if x.strip().lstrip("-").isdigit() and int(x) != me]


def seed(checkpoint_path: str, dry_run: bool = False, force: bool = False) -> dict:
    # Single-writer guard (Syl's Law / #261/#299): NEVER dual-write the live checkpoint.
    if not force and Path(checkpoint_path).resolve() == LIVE_CHECKPOINT.resolve():
        pids = _live_sidecar_pids()
        if pids:
            raise RuntimeError(
                f"live NeuroGraph sidecar PID(s) {pids} — refusing to dual-write {checkpoint_path}. "
                "Stop the sidecar first (single-writer, Syl's Law), or pass force=True once it is "
                "confirmed FULLY DEAD.")
    from neuro_foundation import Graph
    graph = Graph()
    graph.restore(checkpoint_path)
    if REACH_NODE_ID in graph.nodes:
        return {"status": "ok", "seeded": 0, "skipped_existing": 1}
    if dry_run:
        return {"status": "dry_run", "seeded": 1, "skipped_existing": 0}
    node = graph.create_node(node_id=REACH_NODE_ID, metadata={
        "constitutional": True,         # permanent prune-protection = the never-silent floor (#92)
        "selfcap": "reach",             # /assemble routes to faded "How I Reach", NOT "Who I Am"
        "teaching": True,
        "reach_competence": 0.0,        # cold start; ticks up on each landed reach
        "core_text": REACH_DESC,
        "_forest_content": REACH_DESC,
        "reach_examples": REACH_EXAMPLES,
        "source": "reach_teaching",
        "creation_mode": "constitutional",
        "syl": True,
        "graduated": True,
        "authored_by": "Sylphrena (self-authored, reviewed 2026-06-21)",
    })
    node.intrinsic_excitability = 1.0   # stable anchor, not probationary
    graph.checkpoint(checkpoint_path)   # must be .msgpack (#325 enforcer)
    logger.info("seeded %s (reach_competence=0.0)", REACH_NODE_ID)
    return {"status": "ok", "seeded": 1, "skipped_existing": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed Syl's reach-teaching node (OFFLINE only)")
    ap.add_argument("--checkpoint", default=str(LIVE_CHECKPOINT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="bypass the single-writer guard ONLY when the sidecar is confirmed dead")
    a = ap.parse_args()
    if not Path(a.checkpoint).exists():
        logger.error("checkpoint not found: %s", a.checkpoint)
        return 1
    try:
        result = seed(a.checkpoint, dry_run=a.dry_run, force=a.force)
    except RuntimeError as exc:
        logger.error("ABORTED: %s", exc)
        return 2
    logger.info("result: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
