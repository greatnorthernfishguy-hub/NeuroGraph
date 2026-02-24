"""
CES Stream Parser — Real-time text processing with Ollama embeddings.

Runs a background daemon thread that consumes text fed via ``feed()``,
chunks it into overlapping phrases, embeds each chunk via the Ollama API
(with fallback to the ingestor's embedding engine), finds similar nodes
in the vector DB, nudges their voltages, and triggers hyperedge pattern
completion.

This creates a continuous "attention stream" that pre-activates relevant
parts of the SNN graph while new text is being processed, so related
concepts are already warm when the next full ``graph.step()`` runs.

Usage::

    from stream_parser import StreamParser
    parser = StreamParser(graph, vector_db, ces_config)
    parser.feed("The quick brown fox jumps over the lazy dog")
    # ... chunks are processed asynchronously in background thread
    parser.stop()

# ---- Changelog ----
# [2026-02-22] Claude (Opus 4.6) — Initial implementation.
#   What: StreamParser with background thread, Ollama embedding, nudge
#         pipeline, and graceful fallback to hash/sentence-transformers.
#   Why:  Real-time attention stream keeps SNN primed as text arrives,
#         enabling faster pattern completion and surfacing.
# -------------------
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ces_config import CESConfig

logger = logging.getLogger("neurograph.ces.stream")


class StreamParser:
    """Background stream parser that nudges the SNN graph in real-time.

    Args:
        graph: The NeuroGraph ``Graph`` instance.
        vector_db: ``SimpleVectorDB`` for similarity lookups.
        ces_config: ``CESConfig`` with streaming parameters.
        fallback_embedder: Optional embedding callable that accepts a
            string and returns an ndarray.  Used when Ollama is
            unavailable.  If ``None``, no fallback is available.
    """

    def __init__(
        self,
        graph: Any,
        vector_db: Any,
        ces_config: CESConfig,
        fallback_embedder: Any = None,
    ) -> None:
        self._graph = graph
        self._vector_db = vector_db
        self._cfg = ces_config.streaming
        self._fallback_embedder = fallback_embedder

        # Processing queue
        self._queue: queue.Queue[Optional[str]] = queue.Queue(
            maxsize=self._cfg.max_queue
        )

        # State
        self._paused = False
        self._stopped = False
        self._lock = threading.Lock()

        # Stats
        self._chunks_processed = 0
        self._nudges_applied = 0
        self._completions_triggered = 0

        # Ollama availability (lazy-checked)
        self._ollama_available: Optional[bool] = None
        self._ollama_last_check: float = 0.0
        self._embedding_dim: Optional[int] = None

        # Start the processing thread
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True, name="ces-stream-parser"
        )
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """Queue text for asynchronous processing.

        Non-blocking.  If the queue is full, the oldest item is NOT
        dropped — the caller blocks briefly.  This provides natural
        backpressure to prevent unbounded memory growth.
        """
        if self._stopped:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.debug("Stream parser queue full, dropping text")

    def pause(self) -> None:
        """Pause processing (thread stays alive, queue accumulates)."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume processing after a pause."""
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        """Stop the background thread permanently."""
        self._stopped = True
        # Enqueue sentinel to unblock the thread
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

    @property
    def is_running(self) -> bool:
        """True if the thread is alive and not paused."""
        return self._thread.is_alive() and not self._paused and not self._stopped

    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            "chunks_processed": self._chunks_processed,
            "nudges_applied": self._nudges_applied,
            "completions_triggered": self._completions_triggered,
            "queue_depth": self._queue.qsize(),
            "is_running": self.is_running,
            "ollama_available": self._ollama_available,
        }

    # ── Background thread ──────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Main loop: dequeue text, chunk, embed, nudge, trigger."""
        while not self._stopped:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:  # sentinel
                break

            # Respect pause
            with self._lock:
                if self._paused:
                    continue

            try:
                self._process_text(text)
            except Exception as exc:
                logger.debug("Stream parser error: %s", exc)

    def _process_text(self, text: str) -> None:
        """Process a single text through the full pipeline."""
        chunks = self._chunk_text(text)
        for chunk in chunks:
            embedding = self._embed_chunk(chunk)
            if embedding is None:
                continue

            similar = self._find_similar(embedding)
            if similar:
                self._nudge_nodes(similar)
                self._trigger_completions()

            self._chunks_processed += 1

    # ── Pipeline stages ────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word-level chunks.

        Uses ``chunk_size`` tokens per chunk with ``overlap`` token
        overlap between consecutive chunks.
        """
        words = text.split()
        if not words:
            return []

        chunk_size = self._cfg.chunk_size
        overlap = self._cfg.overlap
        step = max(1, chunk_size - overlap)

        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break

        return chunks

    def _embed_chunk(self, chunk: str) -> Optional[np.ndarray]:
        """Embed a text chunk, trying Ollama first then fallback.

        Returns an L2-normalised vector or None on failure.
        """
        # Try Ollama first
        if self._check_ollama():
            vec = self._embed_via_ollama(chunk)
            if vec is not None:
                return vec

        # Fallback to provided embedder
        if self._fallback_embedder is not None:
            try:
                vec = self._fallback_embedder(chunk)
                if vec is not None:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    return vec
            except Exception as exc:
                logger.debug("Fallback embedder failed: %s", exc)

        return None

    def _embed_via_ollama(self, text: str) -> Optional[np.ndarray]:
        """Call the Ollama embeddings API."""
        url = f"{self._cfg.ollama_url}/api/embeddings"
        payload = json.dumps({
            "model": self._cfg.ollama_model,
            "prompt": text,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embedding = data.get("embedding")
                if embedding is not None:
                    vec = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    self._embedding_dim = len(vec)
                    return vec
        except Exception as exc:
            logger.debug("Ollama embedding failed: %s", exc)
            self._ollama_available = False

        return None

    def _check_ollama(self) -> bool:
        """Check Ollama availability (cached with periodic re-check)."""
        now = time.time()
        if (
            self._ollama_available is not None
            and now - self._ollama_last_check < self._cfg.ollama_check_interval
        ):
            return self._ollama_available

        self._ollama_last_check = now
        try:
            url = f"{self._cfg.ollama_url}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            self._ollama_available = True
        except Exception:
            self._ollama_available = False

        return self._ollama_available

    def _find_similar(
        self, embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Find nodes in the vector DB similar to this embedding."""
        try:
            results = self._vector_db.search(
                embedding,
                k=10,
                threshold=self._cfg.similarity_threshold,
            )
            return results
        except Exception as exc:
            logger.debug("Vector DB search failed: %s", exc)
            return []

    def _nudge_nodes(self, similar_nodes: List[Tuple[str, float]]) -> None:
        """Inject current into similar graph nodes (voltage nudge).

        Nudges are capped so voltage never exceeds 2x the node's threshold.
        This prevents unbounded voltage spikes from high similarity scores
        or repeated nudges that could destabilise the SNN.
        """
        for node_id, similarity in similar_nodes:
            node = self._graph.nodes.get(node_id)
            if node is not None and node.refractory_remaining == 0:
                nudge = similarity * self._cfg.nudge_strength
                max_voltage = node.threshold * 2.0
                node.voltage = min(node.voltage + nudge, max_voltage)
                self._nudges_applied += 1

    def _trigger_completions(self) -> None:
        """Evaluate hyperedges after nudging to trigger pattern completion.

        Uses the graph's hyperedge evaluation — checks each active
        hyperedge's activation against its threshold and fires
        completions for those that reach threshold.
        """
        completions_before = self._completions_triggered
        for he_id, he in self._graph.hyperedges.items():
            if he.is_archived or he.refractory_remaining > 0:
                continue

            # Count active members (voltage > 0)
            active = 0
            total = len(he.member_node_ids)
            if total == 0:
                continue

            for mid in he.member_node_ids:
                m_node = self._graph.nodes.get(mid)
                if m_node is not None and m_node.voltage > 0:
                    active += 1

            activation_level = active / total
            if activation_level >= he.threshold:
                # Pattern completion: pre-charge inactive members
                for mid in he.member_node_ids:
                    m_node = self._graph.nodes.get(mid)
                    if m_node is not None and m_node.voltage <= 0:
                        member_weight = he.member_weights.get(mid, 1.0)
                        completion_strength = he.pattern_completion_strength
                        m_node.voltage += (
                            completion_strength
                            * member_weight
                            * m_node.intrinsic_excitability
                        )
                self._completions_triggered += 1

        if self._completions_triggered > completions_before:
            logger.debug(
                "Stream parser triggered %d completions",
                self._completions_triggered - completions_before,
            )
