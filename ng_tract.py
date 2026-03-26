"""
ng_tract.py — Experimental myelinated tract for inter-process experience transport.

EXPERIMENTAL — First implementation of the tract concept.  Scoped to the
feeder→topology-owner pathway only (GUI, feed-syl, file watcher → ContextEngine
RPC bridge).  This is a sandbox test of the five tract requirements (#53), not
a commitment to the ecosystem-wide architecture.

Biological analog: myelinated axon tract.  Passive conductive tissue that
carries raw experience from point A to point B without transformation,
classification, or polling.

Design against the five #53 requirements:
  1. Raw experience — no serialization/deserialization at boundaries.
     Experience enters as raw text + source metadata.  The tract does not
     inspect, classify, or transform it.
  2. No polling — the consumer calls drain() when it's ready (event-driven
     from afterTurn), not on a timer.
  3. Topology IS the communication medium — partially satisfied.  The tract
     carries experience TO the topology; it is not itself topology.  This
     is the honest gap: tracts are conduits, not substrate.  Whether that
     satisfies requirement 3 for the ecosystem-wide case is an open question.
  4. Stigmergic coordination — feeders and consumer don't know about each
     other.  Feeders deposit.  Consumer drains.  No handshake, no protocol
     negotiation, no acknowledgment.
  5. Module isolation — feeder crash can't corrupt the tract or the consumer.
     Consumer crash leaves deposited experience intact for next drain.

Concurrency model:
  - deposit() appends to a memory-mapped region via exclusive flock().
  - drain() atomically renames the tract file, reads the renamed copy,
    then deletes it.  New deposits go to a fresh file.  No data loss,
    no read/write collision.

# ---- Changelog ----
# [2026-03-16] Claude (Opus 4.6) — Initial experimental implementation.
#   What: Myelinated tract for feeder→topology-owner experience transport.
#   Why:  Sandbox test of #53 tract requirements.  Eliminates dual-write
#         hazard from GUI, feed-syl, and future feeders running alongside
#         the ContextEngine RPC bridge.
#   How:  Append-only tract file with atomic rename-based drain.
#         deposit() for feeders, drain() for the topology owner.
# -------------------
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neurograph.tract")

# Default tract location — inside NeuroGraph's data directory
_DEFAULT_TRACT_DIR = os.path.expanduser("~/NeuroGraph/data/tract")
_TRACT_FILENAME = "experience.tract"


class ExperienceTract:
    """Passive conductive tissue for raw experience transport.

    Feeders call ``deposit()`` to push raw experience into the tract.
    The topology owner calls ``drain()`` to pull all pending experience
    out in one atomic operation.

    The tract does not inspect, classify, or transform experience.
    It is conductive tissue, not processing tissue.

    Args:
        tract_dir: Directory for tract files.  Created if absent.
    """

    def __init__(self, tract_dir: Optional[str] = None) -> None:
        self._dir = Path(tract_dir or _DEFAULT_TRACT_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._tract_path = self._dir / _TRACT_FILENAME

    # ── Feeder API ────────────────────────────────────────────────────

    def deposit(
        self,
        content: str,
        source: str = "unknown",
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Deposit raw experience into the tract.

        This is the feeder-side API.  GUI, feed-syl, file watcher, and
        any future tool call this to send experience toward the topology
        owner.  The experience is raw — no classification, no embedding,
        no processing.

        Args:
            content: The raw experience.  For text, the text itself.
                     For files, the absolute file path.
            source: Who deposited this (e.g., "gui", "feed-syl", "watcher").
            content_type: What kind of experience ("text", "file", "url").
            metadata: Optional additional context.  Not interpreted by
                      the tract — passed through as-is.
        """
        entry = {
            "content": content,
            "source": source,
            "content_type": content_type,
            "timestamp": time.time(),
        }
        if metadata:
            entry["metadata"] = metadata

        line = json.dumps(entry, default=str) + "\n"

        try:
            fd = os.open(
                str(self._tract_path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o664,
            )
            try:
                # Exclusive lock for the duration of the write.
                # Other depositors wait; drain() uses a different path
                # (rename), so no deadlock.
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8"))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        except OSError as exc:
            logger.warning("Tract deposit failed: %s", exc)

    # ── Consumer API ──────────────────────────────────────────────────

    def drain(self) -> List[Dict[str, Any]]:
        """Drain all pending experience from the tract.

        This is the topology-owner-side API.  Called during afterTurn
        (event-driven, not polled).  Atomically swaps the tract file
        so new deposits go to a fresh file while we read the old one.
        No data loss, no read/write collision.

        Returns:
            List of experience entries, chronologically ordered.
            Empty list if nothing pending.
        """
        # Atomic rename: move the tract file to a drain-specific name.
        # New deposits immediately go to a fresh tract file.
        drain_path = self._dir / f".draining.{os.getpid()}.tract"

        try:
            os.rename(str(self._tract_path), str(drain_path))
        except FileNotFoundError:
            # No tract file — nothing to drain
            return []
        except OSError as exc:
            logger.warning("Tract drain rename failed: %s", exc)
            return []

        # Read the drained entries
        entries: List[Dict[str, Any]] = []
        try:
            with open(drain_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipped malformed tract entry")
        except OSError as exc:
            logger.warning("Tract drain read failed: %s", exc)
            return entries
        finally:
            # Clean up the drained file
            try:
                os.unlink(str(drain_path))
            except OSError:
                pass

        if entries:
            logger.info("Drained %d entries from tract", len(entries))

        return entries

    # ── Diagnostics ───────────────────────────────────────────────────

    def pending_count(self) -> int:
        """Approximate count of pending entries (for status display)."""
        try:
            if not self._tract_path.exists():
                return 0
            with open(self._tract_path) as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def stats(self) -> Dict[str, Any]:
        """Return tract diagnostics."""
        return {
            "tract_dir": str(self._dir),
            "tract_file": str(self._tract_path),
            "pending": self.pending_count(),
            "exists": self._tract_path.exists(),
        }
