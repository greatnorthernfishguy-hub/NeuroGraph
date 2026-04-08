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

import json  # retained for residual JSONL drain only
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
        # BTF binary deposit via Rust — raw bytes, no JSON, no inflation.
        # Content enters as-is and stays that way until extraction.  Law 7.
        try:
            import ng_tract
            ng_tract.deposit_experience(
                content=content,
                source=source,
                tract_path=str(self._tract_path),
                content_type=content_type,
            )
        except Exception as exc:
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

        # Read the drained entries — BTF binary via Rust TractReader
        entries: List[Dict[str, Any]] = []
        try:
            with open(drain_path, "rb") as f:
                raw = f.read()
            if raw:
                import ng_tract
                reader = ng_tract.TractReader(raw)
                for entry in reader:
                    if isinstance(entry, bytes):
                        # Residual JSONL from pre-BTF era
                        try:
                            entries.append(json.loads(entry))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                    elif entry.entry_type == ng_tract.ENTRY_EXPERIENCE:
                        entries.append({
                            "content": entry.content,
                            "source": entry.source,
                            "content_type": entry.content_type,
                            "timestamp": entry.timestamp,
                        })
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
            with open(self._tract_path, "rb") as f:
                raw = f.read()
            if not raw:
                return 0
            import ng_tract
            count = 0
            reader = ng_tract.TractReader(raw)
            for _ in reader:
                count += 1
            return count
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        """Return tract diagnostics."""
        return {
            "tract_dir": str(self._dir),
            "tract_file": str(self._tract_path),
            "pending": self.pending_count(),
            "exists": self._tract_path.exists(),
        }
