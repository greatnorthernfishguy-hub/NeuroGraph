"""
CES Surfacing Monitor — Surface relevant knowledge from the SNN for prompt injection.

Monitors node activations after each ``graph.step()``, maintains a
bounded priority queue of "surfaced" concepts that score above threshold,
and formats them as a context block for prompt injection.

The queue decays each step so stale concepts fade out and fresh firings
take priority.

Usage::

    from surfacing import SurfacingMonitor
    monitor = SurfacingMonitor(graph, vector_db, ces_config)
    step_result = graph.step()
    monitor.after_step(step_result)
    context = monitor.format_context()

# ---- Changelog ----
# [2026-07-05] CC (laptop) — Hyperedge-scan race fix; REVERTED an incorrect score-cap change
# What: _score_node()'s hyperedge-membership scan now snapshots graph.hyperedges under
#   _step_lock before iterating, instead of scanning the live dict directly. A separate
#   change to the voltage_norm/excitability caps (2.0 -> 1.0) was tried and reverted --
#   see _score_node()'s docstring for why.
# Why:  I initially misdiagnosed the >100% "confidence" values seen in CC's own surfaced-
#   knowledge hook output (178%, 176%...) as a normalization bug (weights summing to 1.0
#   implying [0,1]-normalized components) and capped both at 1.0. That's wrong: the design
#   range for voltage_norm/excitability is [1.0, 2.0], not [0,1] (test_ces.py's own comment
#   documents a 0.8 baseline for a bare-minimum fired node), and capping at 1.0 makes
#   voltage_norm CONSTANT for every fired node -- confirmed by test_higher_voltage_higher_score
#   failing outright under the change. Reverted that part. The unguarded hyperedges.values()
#   scan is a real, separate issue -- same class of race already found and fixed in
#   lenia/graph_substrate.py's _hyperedge_similarity (2026-07-05) -- not yet observed
#   crashing here, but the same unprotected pattern, called on every graph.step().
# How:  See surfacing.py's _score_node() docstring for the full reasoning.
# [2026-04-08] Claude (Opus 4.6) — Punchlist #55: Read attention params from substrate
#   What: _decay_queue() reads surfacing_decay_rate and surfacing_min_confidence
#         from graph.config (substrate) instead of frozen CES dataclass.
#   Why:  CES attention params are now in TUNABLE_PARAMS (ng_lite.py). Elmer
#         tunes them via update_tunable(). SurfacingMonitor must read the live
#         substrate value, not the bootstrap default. CES config is fallback only.
#   How:  getattr(self._graph, 'config', {}).get() with self._cfg fallback.
# [2026-04-07] Claude (Opus 4.6) — Fix temporal stuttering: decay before read
#   What: get_surfaced() now calls _decay_queue() before returning items.
#   Why:  Lifecycle ordering bug — assemble() reads the surfacing queue before
#         afterTurn() applies decay, so stale concepts persisted at inflated
#         scores across multiple turns. A concept firing at score 1.0 would
#         remain above min_confidence (0.3) for ~16 turns. With decay-before-read,
#         each read applies one decay step, roughly halving the persistence window.
#   How:  Single _decay_queue() call at the top of get_surfaced(). Safe because
#         _decay_queue() is idempotent within a step — double-decay in the same
#         step is bounded by the 0.95 multiplier and won't drop items incorrectly.
# [2026-03-25] Claude (Opus 4.6) — Salience weights from config (SVG Phase 3)
#   What: _score_node() reads weight_voltage/weight_excitability/weight_he_membership
#         from CES SurfacingConfig instead of hardcoded 0.5/0.3/0.2.
#   Why:  Static Value Graduation — salience weights are the substrate's concern.
#         Named config values are tunable by Elmer's TuningSocket. Bootstrap
#         defaults preserved (0.5/0.3/0.2).
#   How:  Weights read from self._cfg at scoring time. ces_config.py updated
#         with weight_voltage, weight_excitability, weight_he_membership fields.
# [2026-02-22] Claude (Opus 4.6) — Initial implementation.
#   What: SurfacingMonitor with bounded heap, composite scoring,
#         per-step decay, and context block formatting.
#   Why:  Surfaces knowledge the SNN considers relevant based on
#         activation dynamics — not explicit search, but associative
#         "remembering".
# -------------------
"""

from __future__ import annotations

import heapq
import logging
from typing import Any, Dict, List, Optional

from ces_config import CESConfig

logger = logging.getLogger("neurograph.ces.surfacing")


class _SurfacedItem:
    """Internal queue entry with negated score for max-heap via heapq."""

    __slots__ = ("node_id", "score", "content", "metadata", "age")

    def __init__(
        self,
        node_id: str,
        score: float,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        age: int = 0,
    ) -> None:
        self.node_id = node_id
        self.score = score
        self.content = content
        self.metadata = metadata or {}
        self.age = age

    def __lt__(self, other: _SurfacedItem) -> bool:
        # heapq is a min-heap; negate score for max-heap behavior
        return self.score > other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SurfacedItem):
            return NotImplemented
        return self.node_id == other.node_id


class SurfacingMonitor:
    """Monitors SNN firings and surfaces relevant knowledge.

    Args:
        graph: The NeuroGraph ``Graph`` instance.
        vector_db: ``SimpleVectorDB`` for content retrieval.
        ces_config: ``CESConfig`` with surfacing parameters.
    """

    def __init__(
        self,
        graph: Any,
        vector_db: Any,
        ces_config: CESConfig,
    ) -> None:
        self._graph = graph
        self._vector_db = vector_db
        self._cfg = ces_config.surfacing

        # Bounded priority queue (max-heap via __lt__ inversion)
        self._queue: List[_SurfacedItem] = []
        self._node_ids_in_queue: set = set()

        # Decay tracking — prevents double-decay when get_surfaced() and
        # after_step() are both called in the same logical step.
        self._last_decay_step: int = -1

        # Stats
        self._total_surfaced: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    def after_step(self, step_result: Any) -> None:
        """Process a step result and update the surfacing queue.

        Should be called after each ``graph.step()``.  Scans fired nodes,
        scores them, and inserts into the bounded queue.  Also applies
        decay to existing entries.

        Note: fired nodes have already had their voltage reset to resting
        potential by ``graph.step()`` before this method is called.  We do
        NOT filter by current voltage — being in ``fired_node_ids`` already
        means the node exceeded its firing threshold during the step.

        Args:
            step_result: ``StepResult`` from ``graph.step()``.
        """
        # Decay existing entries
        self._decay_queue()

        # Process newly fired nodes
        for node_id in step_result.fired_node_ids:
            node = self._graph.nodes.get(node_id)
            if node is None:
                continue

            score = self._score_node(node_id, node)
            if score < self._cfg.min_confidence:
                continue

            # Fetch content from vector DB
            db_entry = self._vector_db.get(node_id)
            content = ""
            metadata: Dict[str, Any] = {}
            if db_entry is not None:
                content = db_entry.get("content", "")
                metadata = db_entry.get("metadata", {})

            if not content:
                continue  # Skip nodes without retrievable content

            item = _SurfacedItem(
                node_id=node_id,
                score=score,
                content=content,
                metadata=metadata,
            )

            self._insert_item(item)
            self._total_surfaced += 1

    def get_surfaced(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the top surfaced items sorted by score.

        Applies decay before reading so that scores reflect elapsed steps
        since the last ``after_step()`` call.  This prevents stale concepts
        from being injected into the prompt at their pre-decay score when
        ``assemble()`` runs before the next ``after_step()``.

        Args:
            max_items: Maximum items to return.  Defaults to
                ``ces_config.surfacing.max_surfaced``.

        Returns:
            List of dicts with node_id, content, metadata, score.
        """
        # Decay before read — fixes lifecycle ordering where assemble()
        # reads the queue before afterTurn() applies decay, causing
        # stale concepts to persist at inflated scores.
        self._decay_queue()

        if max_items is None:
            max_items = self._cfg.max_surfaced

        # Sort by score descending
        sorted_items = sorted(self._queue, key=lambda x: x.score, reverse=True)

        results = []
        for item in sorted_items[:max_items]:
            entry: Dict[str, Any] = {
                "node_id": item.node_id,
                "content": item.content,
                "score": round(item.score, 4),
            }
            if self._cfg.include_metadata:
                entry["metadata"] = item.metadata
            results.append(entry)

        return results

    def format_context(
        self, surfaced_items: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Format surfaced items as a context block for prompt injection.

        Args:
            surfaced_items: Items to format.  If ``None``, calls
                ``get_surfaced()`` automatically.

        Returns:
            Formatted context block string, or empty string if nothing
            surfaced.
        """
        if surfaced_items is None:
            surfaced_items = self.get_surfaced()

        if not surfaced_items:
            return ""

        lines = ["[NeuroGraph Surfaced Knowledge]"]
        for item in surfaced_items:
            content = item.get("content", "")
            score = item.get("score", 0.0)
            # Truncate long content for context blocks
            if len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"- {content} (confidence: {score:.0%})")

        return "\n".join(lines)

    def clear(self) -> None:
        """Empty the surfacing queue."""
        self._queue.clear()
        self._node_ids_in_queue.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return surfacing statistics."""
        scores = [item.score for item in self._queue]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_surfaced": self._total_surfaced,
            "queue_depth": len(self._queue),
            "avg_score": round(avg_score, 4),
        }

    # ── Internal ───────────────────────────────────────────────────────

    def _score_node(self, node_id: str, node: Any) -> float:
        """Compute a composite relevance score for a fired node.

        Weights read from CES SurfacingConfig (bootstrap defaults: 0.5/0.3/0.2).
        These are tunable by Elmer's TuningSocket as the substrate learns which
        signals matter for surfacing.

        Components:
        - voltage_normalized = effective_voltage / node.threshold (capped at 2.0)
        - excitability = node.intrinsic_excitability (capped at 2.0)
        - he_membership = number of hyperedges containing this node / 10 (capped at 1.0)

        Note: ``graph.step()`` resets fired nodes' voltage to resting potential
        before this method is called, so we use ``max(voltage, threshold)`` as
        the effective voltage — fired nodes reached at least their threshold.
        This floors voltage_norm at 1.0 for EVERY fired node (effective_voltage
        is never below threshold), so the design range for voltage_norm and
        excitability is [1.0, 2.0], not [0, 1.0] -- a bare-minimum fired node
        with default (1.0) excitability and no hyperedge membership scores
        0.5*1.0 + 0.3*1.0 + 0.2*0 = 0.8 (see test_ces.py's own comment), and
        the ceiling for a strongly-firing, highly-excitable, densely-connected
        node is 0.5*2.0 + 0.3*2.0 + 0.2*1.0 = 1.8. [2026-07-05] CC: I initially
        misread this as a bug (weights summing to 1.0 implying [0,1]-normalized
        inputs) and capped both at 1.0 -- that broke test_higher_voltage_higher_score
        outright (voltage_norm becomes CONSTANT 1.0 for every fired node, since
        effective_voltage/threshold >= 1.0 always) and is not what's happening
        here: the >100% "confidence" values are the intended salience range,
        not an arithmetic bug. Reverted. The real issue, if any, is that
        format_context() labels this a "confidence" and formats it as a percent
        (implying a bounded probability) when it's actually an unbounded-above-
        0.8 salience score -- a display/labeling question, not a scoring one.
        """
        effective_voltage = max(node.voltage, node.threshold)
        voltage_norm = min(effective_voltage / max(node.threshold, 0.01), 2.0)
        excitability = min(node.intrinsic_excitability, 2.0)

        # Count hyperedge memberships. Snapshot under the Graph's step lock so a
        # concurrent graph.step() (prune/sprout mutates graph.hyperedges under
        # _step_lock) can't change the dict mid-iteration -- same race as
        # lenia/graph_substrate.py's _hyperedge_similarity (fixed 2026-07-05),
        # just not yet observed crashing here since after_step() runs on every
        # single graph.step(), a much shorter window per call than Lenia's
        # multi-hour populate(), but the same unguarded iteration.
        _lock = getattr(self._graph, '_step_lock', None)
        if _lock is not None:
            with _lock:
                _hes = list(self._graph.hyperedges.values())
        else:
            _hes = list(self._graph.hyperedges.values())
        he_count = 0
        for he in _hes:
            if node_id in getattr(he, 'member_nodes', getattr(he, 'member_node_ids', [])):
                he_count += 1
        he_norm = min(he_count / 10.0, 1.0)

        w_v = self._cfg.weight_voltage
        w_e = self._cfg.weight_excitability
        w_h = self._cfg.weight_he_membership
        return w_v * voltage_norm + w_e * excitability + w_h * he_norm

    def _insert_item(self, item: _SurfacedItem) -> None:
        """Insert or update an item in the bounded queue."""
        # If already in queue, update if higher score
        if item.node_id in self._node_ids_in_queue:
            for i, existing in enumerate(self._queue):
                if existing.node_id == item.node_id:
                    if item.score > existing.score:
                        self._queue[i] = item
                        heapq.heapify(self._queue)
                    return

        # If queue at capacity, replace weakest if new item is stronger
        capacity = self._cfg.queue_capacity
        if len(self._queue) >= capacity:
            # heapq[0] is the "smallest" — but our __lt__ inverts, so
            # heapq[0] is actually the highest score.  We want the lowest.
            weakest_idx = max(range(len(self._queue)), key=lambda i: -self._queue[i].score)
            if item.score > self._queue[weakest_idx].score:
                removed = self._queue[weakest_idx]
                self._node_ids_in_queue.discard(removed.node_id)
                self._queue[weakest_idx] = item
                self._node_ids_in_queue.add(item.node_id)
                heapq.heapify(self._queue)
            return

        heapq.heappush(self._queue, item)
        self._node_ids_in_queue.add(item.node_id)

    def _decay_queue(self) -> None:
        """Apply decay to all items in the queue, removing those below threshold.

        Idempotent within a single graph timestep — tracks the last step at
        which decay was applied and skips if already done this step.  This
        allows both ``get_surfaced()`` and ``after_step()`` to call decay
        without double-penalising queue items.
        """
        current_step = getattr(self._graph, 'timestep', -2)
        if current_step == self._last_decay_step:
            return  # Already decayed this step
        self._last_decay_step = current_step

        # Read from substrate config (tunable via Elmer) with CES config fallback
        graph_config = getattr(self._graph, 'config', {})
        decay_rate = graph_config.get("surfacing_decay_rate", self._cfg.decay_rate)
        min_confidence = graph_config.get("surfacing_min_confidence", self._cfg.min_confidence)

        surviving: List[_SurfacedItem] = []
        for item in self._queue:
            item.score *= decay_rate
            item.age += 1
            if item.score >= min_confidence:
                surviving.append(item)
            else:
                self._node_ids_in_queue.discard(item.node_id)

        self._queue = surviving
        heapq.heapify(self._queue)
