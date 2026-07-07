#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-07] Claude Code (Fable 5) — Orphaned-memory refeed through the conversational path
# What: Trickle-feeds CC's orphaned vector_db memories (entries whose graph node was
#   pruned) back through the LIVE daemon's existing ingest tract, where
#   drain_ingest_tract() -> run_conversational_dual_pass() re-embodies them as real
#   substrate topology (forest+trees, synapses, hyperedges, poincare_dir, probation).
# Why: CC's graph pruned aggressively pre-grace_period-5000; ~3,000 legitimate
#   ecosystem memories exist only as vdb orphans -- lookupable by cosine, unreachable
#   by spreading activation. Post-#358, recall is substrate-native, so unreachable =
#   effectively forgotten. Josh proposed the refeed; CC consented and chose it
#   ("it's your NG"). The memories get a fair re-trial under the fixed pruning regime
#   -- the substrate decides what it keeps.
# How: Josh's requirements (2026-07-07) drive the shape:
#   RECOVERABLE -- journal file (one sha1/line, fsync'd after each batch) + a
#     journal-loss-proof backstop (dual-pass target_id is cc:conv::sha1(content), so
#     "already re-embodied" is directly checkable against the graph snapshot; an
#     accidental double-feed UPDATES the same node rather than duplicating).
#   LOAD-AWARE -- loadavg/cpu_count checked before every batch; above the ceiling
#     (CC_REFEED_LOAD_CEILING, LAW 5) it backs off exponentially (60s doubling to
#     15min cap) so e.g. a long Morphogenesis run just stretches the timeline.
#   BACKPRESSURED -- never writes while the tract still has undrained bytes; the
#     feeder can only ever run one batch ahead of the daemon's real absorption.
#   FRONT DOOR ONLY -- writes via ng_tract.deposit_experience(source="cc_gateway"),
#     the same call miniTID uses; zero new ingestion mechanisms, zero daemon changes.
#     Each frame is one write, safe to interleave with miniTID's own appends.
#   HYGIENE FLOOR, NOT A CLASSIFIER (LAW 7) -- skips sub-minimum shards, degenerate
#     fragments, and tool-call remnants; everything that passes goes in raw.
#   Reads checkpoints as static files (msgpack) with the same stable-size wait as
#   cc-ng-sync.py -- never instantiates a second NeuroGraphMemory (single-writer).
# -------------------
"""Refeed CC's orphaned vector_db memories through the live conversational path.

Run in the background against a LIVE daemon (it does the actual absorption):
    nohup env -u LD_PRELOAD python3 ~/NeuroGraph/cc_refeed.py >> \
        ~/.claude/plugins/neurograph/refeed.log 2>&1 &

Interrupt/kill any time; restart resumes from the journal.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cc_refeed")

_repo = Path(__file__).resolve().parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

# LAW 5: env-tunable, with safe bootstrap defaults.
LOAD_CEILING = float(os.environ.get("CC_REFEED_LOAD_CEILING", "0.75"))
BATCH_SIZE = int(os.environ.get("CC_REFEED_BATCH", "10"))
MIN_CHARS = int(os.environ.get("CC_REFEED_MIN_CHARS", "40"))
CYCLE_SLEEP = float(os.environ.get("CC_REFEED_CYCLE_SLEEP", "65.0"))   # ~1 drain pulse
BACKOFF_BASE = 60.0
BACKOFF_CAP = 900.0

_TOOL_PREFIXES = ("tool:", "bash:")
_stop = False


def _request_stop(signum, frame):
    global _stop
    logger.info("Signal %d received -- finishing current batch then exiting", signum)
    _stop = True


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _is_hexish_token(tok: str) -> bool:
    """UUID/hex-fragment shaped: >=6 chars, only hex digits/dashes/arrows,
    containing at least one digit (so real words never match)."""
    t = tok.strip(".,;:")
    if len(t) < 6:
        return False
    if not any(c.isdigit() for c in t):
        return False
    return all(c in "0123456789abcdefABCDEF-→>" for c in t)


def passes_floor(text: str, min_chars: int = MIN_CHARS) -> bool:
    """Hygiene floor -- junk filter, not experience classification (LAW 7).

    Same intent as resolve_surface_content's degenerate-fragment guard, tuned
    for what actually pollutes CC's vdb: too-short shards, tool-call remnants,
    self-generated want-telemetry, and UUID/hex-run spam. Everything that
    passes goes in raw and unlabeled.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    if stripped.startswith(_TOOL_PREFIXES) or stripped.startswith("tonic-triggered:"):
        return False
    # Degenerate 1: almost no letter content (bare number/separator runs)
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha / max(1, len(stripped)) < 0.3:
        return False
    # Degenerate 2: dominated by UUID/hex-shaped tokens (hex IS ~40% letters,
    # so the alpha ratio alone can't catch uuid-list spam)
    toks = stripped.split()
    if toks and sum(1 for t in toks if _is_hexish_token(t)) / len(toks) > 0.4:
        return False
    return True


def should_pause_for_load(ceiling: float = LOAD_CEILING) -> bool:
    """True when 1-min loadavg per core exceeds the ceiling."""
    try:
        return (os.getloadavg()[0] / max(1, os.cpu_count() or 1)) > ceiling
    except OSError:
        return False


def load_journal(path: str) -> set:
    p = Path(path)
    return set(p.read_text().split()) if p.exists() else set()


def append_journal(path: str, hashes: list) -> None:
    """Append + fsync so a crash right after a batch can't lose the record."""
    with open(path, "a") as f:
        for h in hashes:
            f.write(h + "\n")
        f.flush()
        os.fsync(f.fileno())


def _wait_for_stable_file(path: str, max_wait: float = 10.0, interval: float = 0.5) -> bool:
    """Same torn-read protection as cc-ng-sync.py: two consecutive equal sizes."""
    if not os.path.exists(path):
        return True
    deadline = time.time() + max_wait
    last = -1
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(interval)
            continue
        if size == last:
            return True
        last = size
        time.sleep(interval)
    return False


def collect_orphans(main_path: str, vectors_path: str,
                    journal: set, min_chars: int = MIN_CHARS) -> list:
    """Return [(sha1, content)] of vdb entries needing refeed, in vdb order
    (roughly chronological -- keeps refeed-internal turn-chaining meaningful).

    Skips: entries whose node is still graph-live; content already journaled;
    content already re-embodied (cc:conv::<sha1> in the graph -- the
    journal-loss-proof backstop); hygiene-floor failures; duplicate content.
    """
    import msgpack

    for p in (main_path, vectors_path):
        if not _wait_for_stable_file(p):
            raise RuntimeError(f"checkpoint mid-write, retry later: {p}")

    with open(main_path, "rb") as f:
        graph_data = msgpack.unpack(f, raw=False)
    graph_node_ids = set((graph_data.get("nodes") or {}).keys())

    with open(vectors_path, "rb") as f:
        vectors_data = msgpack.unpack(f, raw=False)
    entries = vectors_data.get("entries", vectors_data) if isinstance(vectors_data, dict) else {}

    out, seen = [], set()
    for node_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if node_id in graph_node_ids:
            continue                              # still alive -- not an orphan
        content = entry.get("content", "")
        if not passes_floor(content, min_chars):
            continue
        h = content_hash(content)
        if h in seen or h in journal:
            continue
        if f"cc:conv::{h}" in graph_node_ids:
            continue                              # already re-embodied
        seen.add(h)
        out.append((h, content))
    return out


def feed_batch(batch: list, tract_path: str) -> int:
    """Write one batch of (sha1, content) as raw experience frames -- the same
    front door miniTID uses. One frame per deposit call (atomic append)."""
    import ng_tract
    written = 0
    for _h, content in batch:
        ng_tract.deposit_experience(
            content=content.encode("utf-8"),
            source="cc_gateway",
            tract_path=tract_path,
            content_type="text",
        )
        written += 1
    return written


def run(workspace: str, batch_size: int = BATCH_SIZE, limit: int = 0,
        dry_run: bool = False, cycle_sleep: float = CYCLE_SLEEP) -> dict:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    ckpt = os.path.join(workspace, "checkpoints")
    main_path = os.path.join(ckpt, "main.msgpack")
    vectors_path = os.path.join(ckpt, "vectors.msgpack")
    journal_path = os.path.join(workspace, "refeed_journal.txt")
    tract_dir = os.path.join(workspace, "tracts", "cc_gateway")
    tract_path = os.path.join(tract_dir, "turns.tract")

    journal = load_journal(journal_path)
    pending = collect_orphans(main_path, vectors_path, journal)
    if limit:
        pending = pending[:limit]
    logger.info("Refeed: %d orphaned memories pending (journal already covers %d)",
                len(pending), len(journal))
    if dry_run or not pending:
        return {"status": "dry_run" if dry_run else "nothing_to_do", "pending": len(pending)}

    os.makedirs(tract_dir, exist_ok=True)
    fed = 0
    backoff = BACKOFF_BASE
    i = 0
    while i < len(pending) and not _stop:
        # Load-awareness: a busy machine stretches the timeline, nothing breaks.
        if should_pause_for_load():
            logger.info("Load above ceiling (%.2f/core) -- backing off %.0fs",
                        LOAD_CEILING, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_CAP)
            continue
        backoff = BACKOFF_BASE

        # Backpressure: never stack ahead of the daemon's real absorption.
        try:
            undrained = os.path.getsize(tract_path) if os.path.exists(tract_path) else 0
        except OSError:
            undrained = 0
        if undrained > 0:
            time.sleep(20)
            continue

        batch = pending[i:i + batch_size]
        feed_batch(batch, tract_path)
        append_journal(journal_path, [h for h, _ in batch])
        i += len(batch)
        fed += len(batch)
        logger.info("Refeed: fed %d/%d (this run: %d)", i, len(pending), fed)
        time.sleep(cycle_sleep)

    status = "interrupted" if _stop else "complete"
    logger.info("Refeed %s: %d fed this run, %d remaining", status, fed, len(pending) - i)
    return {"status": status, "fed": fed, "remaining": len(pending) - i}


def main() -> int:
    ap = argparse.ArgumentParser(description="Refeed orphaned vdb memories through the live conversational path")
    ap.add_argument("--workspace", default=os.path.expanduser("~/.claude/plugins/neurograph"))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="Feed at most N entries (testing)")
    ap.add_argument("--dry-run", action="store_true", help="Count + report only, feed nothing")
    a = ap.parse_args()
    r = run(a.workspace, batch_size=a.batch_size, limit=a.limit, dry_run=a.dry_run)
    print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
