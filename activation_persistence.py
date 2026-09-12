"""
CES Activation Persistence — Save and restore node activation state across sessions.

Writes a JSON sidecar file alongside the main msgpack checkpoint that
captures each node's voltage, last-fired step, and excitability.  On
restore, a temporal decay is applied based on elapsed wall-clock time so
that stale activations fade naturally.

Usage::

    from activation_persistence import ActivationPersistence
    ap = ActivationPersistence(ces_config)
    ap.save(graph, "/path/to/checkpoint.msgpack")
    # ... later ...
    ap.restore(graph, "/path/to/checkpoint.msgpack")

# ---- Changelog ----
# [2026-02-22] Claude (Opus 4.6) — Initial implementation.
#   What: ActivationPersistence with capture/save/restore, exponential
#         temporal decay, max_entries bounding, and auto-save timer.
#   Why:  The main msgpack checkpoint resets all voltages to resting
#         potential.  This sidecar preserves "warm" activations across
#         sessions so the SNN resumes where it left off rather than
#         starting cold.
# -------------------
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ces_config import CESConfig

logger = logging.getLogger("neurograph.ces.persistence")


class ActivationPersistence:
    """Manages activation state sidecar files for cross-session persistence.

    Args:
        ces_config: ``CESConfig`` with persistence parameters.
    """

    def __init__(self, ces_config: CESConfig) -> None:
        self._cfg = ces_config.persistence
        self._last_save_time: Optional[float] = None
        self._last_decay_applied: float = 0.0
        self._entries_saved: int = 0
        self._sidecar_path: Optional[str] = None

        # Auto-save timer state
        self._auto_save_timer: Optional[threading.Timer] = None
        self._auto_save_graph: Any = None
        self._auto_save_checkpoint_path: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────

    def capture(self, graph: Any) -> Dict[str, Dict[str, Any]]:
        """Capture current activation state from the graph.

        Returns a dict keyed by node_id with voltage, last_spike_time,
        intrinsic_excitability, and a wall-clock timestamp.

        Only includes nodes with non-zero voltage to keep the sidecar
        compact.  Bounded to ``max_entries`` by voltage magnitude.

        Every captured value is a plain float, so the result is inherently
        detached from live node objects — no aliasing to break (#423).
        """
        now = time.time()
        entries: Dict[str, Dict[str, Any]] = {}

        # list() the items first: node creation by a concurrent writer would
        # otherwise raise "dictionary changed size during iteration" mid-capture.
        for node_id, node in list(graph.nodes.items()):
            if node.voltage == 0.0 or node.voltage == node.resting_potential:
                continue
            entries[node_id] = {
                "voltage": node.voltage,
                "last_spike_time": node.last_spike_time,
                "excitability": node.intrinsic_excitability,
                "timestamp": now,
            }

        # Bound to max_entries by descending absolute voltage
        max_entries = self._cfg.max_entries
        if len(entries) > max_entries:
            sorted_ids = sorted(
                entries.keys(),
                key=lambda nid: abs(entries[nid]["voltage"]),
                reverse=True,
            )
            entries = {nid: entries[nid] for nid in sorted_ids[:max_entries]}

        return entries

    # ---- Changelog ----
    # [2026-09-11] Claude (Sonnet 5) — #423: capture/write split + surfaced errors
    # What: save() factored into capture_state(graph) -> envelope dict and
    #       write_state(checkpoint_path, captured) -> path. write_state RAISES on write
    #       failure instead of logging a warning and returning the path anyway; save()
    #       propagates that. capture() now iterates a list() snapshot of graph.nodes.
    # Why:  (a) A coherent checkpoint needs capture at a coordination boundary and disk
    #       I/O after live mutation resumes. (b) The old except-and-return made a failed
    #       sidecar indistinguishable from a written one at the call site — the hook had
    #       to infer "writer swallowed its error" from the absent file. A partial capture
    #       must never be able to produce an accepted receipt.
    # How:  Envelope construction in capture_state; json.dump in write_state with the
    #       success bookkeeping only after the write returns, and the exception logged
    #       then re-raised. _auto_save_tick already guards with try/except, so the
    #       timer chain is unaffected.
    # [2026-09-11] Codex — explicit optional write receipt; consumers no longer
    # infer successful publication by inspecting private timer bookkeeping.
    # -------------------
    def capture_state(self, graph: Any) -> Dict[str, Any]:
        """Capture the full activation sidecar payload in RAM — NO disk I/O.

        Args:
            graph: The ``Graph`` instance to capture from.

        Returns:
            The sidecar envelope (version, saved_at, timestep, entries), suitable
            for :meth:`write_state`.
        """
        entries = self.capture(graph)
        return {
            "version": "1.0",
            "saved_at": time.time(),
            "timestep": graph.timestep,
            "entries": entries,
        }

    def write_state(self, checkpoint_path: str, captured: Dict[str, Any],
                    *, with_receipt: bool = False) -> str | Dict[str, Any]:
        """Write a previously captured activation payload next to the checkpoint.

        Pure I/O: reads no live graph state, so it may run after mutation resumes.

        Args:
            checkpoint_path: Path to the main checkpoint file.
            captured: Envelope returned by :meth:`capture_state`.

        Returns:
            Path to the written sidecar file, or with ``with_receipt=True`` a
            mapping containing that path and the captured ``saved_at`` token.
            The receipt is returned only after the write closes successfully.

        Raises:
            Exception: Whatever the write failed with. The error is surfaced, NOT
                swallowed — a caller must never treat a failed sidecar as saved
                (#423). Save-time bookkeeping (``get_stats``) is updated only after
                the write succeeds.
        """
        sidecar_path = self._sidecar_path_for(checkpoint_path)
        entries = captured.get("entries", {})

        try:
            with open(sidecar_path, "w") as f:
                json.dump(captured, f)
        except Exception as exc:
            logger.error(
                "Failed to save activation sidecar to %s: %s", sidecar_path, exc,
            )
            raise

        self._last_save_time = captured.get("saved_at")
        self._entries_saved = len(entries)
        self._sidecar_path = sidecar_path
        logger.info(
            "Activation sidecar saved: %d entries to %s",
            len(entries),
            sidecar_path,
        )
        if with_receipt:
            return {"path": sidecar_path, "saved_at": captured["saved_at"]}
        return sidecar_path

    def save(self, graph: Any, checkpoint_path: str) -> str:
        """Capture and write activation sidecar next to the checkpoint.

        Capture-then-write, fused: equivalent to
        ``write_state(checkpoint_path, capture_state(graph))``. Use the two
        primitives directly when the capture must complete at a coordination
        boundary and the write must happen after live mutation resumes (#423).

        Args:
            graph: The ``Graph`` instance to capture from.
            checkpoint_path: Path to the main checkpoint file.

        Returns:
            Path to the written sidecar file.

        Raises:
            Exception: Propagated from :meth:`write_state` if the write fails.
                (Before #423 this was swallowed and the path returned regardless.)
        """
        return self.write_state(checkpoint_path, self.capture_state(graph))

    def restore(self, graph: Any, checkpoint_path: str) -> int:
        """Restore activation state from sidecar with temporal decay.

        Applies exponential decay based on elapsed wall-clock time:
        ``activation *= (1 - decay_per_hour) ^ elapsed_hours``

        Entries below ``min_activation`` after decay are discarded.
        Entries for nodes no longer in the graph are skipped.

        Args:
            graph: The ``Graph`` instance to restore into.
            checkpoint_path: Path to the main checkpoint file.

        Returns:
            Number of nodes restored.
        """
        sidecar_path = self._sidecar_path_for(checkpoint_path)

        if not Path(sidecar_path).exists():
            logger.debug("No activation sidecar at %s", sidecar_path)
            return 0

        try:
            with open(sidecar_path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read activation sidecar: %s", exc)
            return 0

        entries = data.get("entries", {})
        saved_at = data.get("saved_at", time.time())

        # Apply temporal decay
        elapsed_hours = (time.time() - saved_at) / 3600.0
        if elapsed_hours < 0:
            elapsed_hours = 0.0
        self._last_decay_applied = elapsed_hours

        entries = self._apply_decay(entries, elapsed_hours)

        # Inject into graph
        restored = 0
        for node_id, state in entries.items():
            node = graph.nodes.get(node_id)
            if node is None:
                continue

            node.voltage = state["voltage"]
            if "excitability" in state:
                node.intrinsic_excitability = state["excitability"]
            restored += 1

        self._sidecar_path = sidecar_path
        logger.info(
            "Restored %d/%d activation entries (%.1fh decay applied)",
            restored,
            len(entries),
            elapsed_hours,
        )
        return restored

    def start_auto_save(self, graph: Any, checkpoint_path: str) -> None:
        """Start periodic auto-saving of activation state."""
        self._auto_save_graph = graph
        self._auto_save_checkpoint_path = checkpoint_path
        self._schedule_auto_save()

    def stop_auto_save(self) -> None:
        """Stop the auto-save timer."""
        if self._auto_save_timer is not None:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None
        self._auto_save_graph = None
        self._auto_save_checkpoint_path = None

    def get_stats(self) -> Dict[str, Any]:
        """Return persistence statistics."""
        return {
            "entries_saved": self._entries_saved,
            "last_save_time": self._last_save_time,
            "last_decay_applied": round(self._last_decay_applied, 2),
            "sidecar_path": self._sidecar_path,
        }

    # ── Internal ───────────────────────────────────────────────────────

    def _sidecar_path_for(self, checkpoint_path: str) -> str:
        """Compute the sidecar file path for a given checkpoint path."""
        return checkpoint_path + self._cfg.sidecar_suffix

    def _apply_decay(
        self,
        entries: Dict[str, Dict[str, Any]],
        elapsed_hours: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Apply exponential decay and prune sub-threshold entries."""
        if elapsed_hours <= 0:
            return entries

        decay_factor = (1.0 - self._cfg.decay_per_hour) ** elapsed_hours
        min_act = self._cfg.min_activation

        result: Dict[str, Dict[str, Any]] = {}
        for node_id, state in entries.items():
            decayed_voltage = state["voltage"] * decay_factor
            if abs(decayed_voltage) < min_act:
                continue
            result[node_id] = {
                **state,
                "voltage": decayed_voltage,
            }

        return result

    def _schedule_auto_save(self) -> None:
        """Schedule the next auto-save tick."""
        if self._auto_save_graph is None:
            return
        self._auto_save_timer = threading.Timer(
            self._cfg.auto_save_interval,
            self._auto_save_tick,
        )
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _auto_save_tick(self) -> None:
        """Auto-save callback."""
        # Check if we've been stopped before doing anything — prevents orphan
        # timer chains that keep rescheduling after stop_auto_save().
        if self._auto_save_graph is None or self._auto_save_checkpoint_path is None:
            return
        try:
            self.save(self._auto_save_graph, self._auto_save_checkpoint_path)
        except Exception as exc:
            logger.warning("Auto-save failed: %s", exc)
        self._schedule_auto_save()
