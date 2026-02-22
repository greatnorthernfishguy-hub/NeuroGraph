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

        # Stats
        self._total_surfaced: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    def after_step(self, step_result: Any) -> None:
        """Process a step result and update the surfacing queue.

        Should be called after each ``graph.step()``.  Scans fired nodes,
        filters by voltage threshold, scores, and inserts into the
        bounded queue.  Also applies decay to existing entries.

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

            # Filter by voltage threshold
            if node.voltage < self._cfg.voltage_threshold:
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

        Args:
            max_items: Maximum items to return.  Defaults to
                ``ces_config.surfacing.max_surfaced``.

        Returns:
            List of dicts with node_id, content, metadata, score.
        """
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

        Score = 0.5 * voltage_normalized + 0.3 * excitability + 0.2 * he_membership

        Where:
        - voltage_normalized = node.voltage / node.threshold (capped at 2.0)
        - excitability = node.intrinsic_excitability (capped at 2.0)
        - he_membership = number of hyperedges containing this node / 10 (capped at 1.0)
        """
        voltage_norm = min(node.voltage / max(node.threshold, 0.01), 2.0)
        excitability = min(node.intrinsic_excitability, 2.0)

        # Count hyperedge memberships
        he_count = 0
        for he in self._graph.hyperedges.values():
            if node_id in he.member_node_ids:
                he_count += 1
        he_norm = min(he_count / 10.0, 1.0)

        return 0.5 * voltage_norm + 0.3 * excitability + 0.2 * he_norm

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
        """Apply decay to all items in the queue, removing those below threshold."""
        decay_rate = self._cfg.decay_rate
        min_confidence = self._cfg.min_confidence

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
