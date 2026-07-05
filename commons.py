"""
The Commons — shared substrate medium for peer-module communication.

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — #332 persist/restore wired + #330 error:* retention
#   What: (1) restore() now drops autonomic:* synapses post-load — arousal fresh-assesses on
#         restart (#328 Decision #2), everything else persists normally. Wired into
#         neurograph_rpc.py bootstrap (restore) + auto-save + shutdown (persist) — see that
#         file's changelog. (2) deposit() now windows error:* the same way as metrics:* —
#         time-series telemetry, not memory, recency-bounded per source:type so error volume
#         never pressures the weight-based max_synapses bound (#330 operational-logger).
#   Why:  #332 — persist()/restore() were defined with zero callers; Commons was wiped every
#         gateway restart, losing accumulated experience/topology/metrics/repair knowledge for
#         no reason. #330 — error:* is the operational-logger's new namespace (signal_error() on
#         ng_commons_eco.py, vendored change, separate sign-off) and needs the same recency
#         retention metrics:* already has, or a busy module's errors would crowd out memory.
#   How:  _drop_autonomic_synapses() mirrors _evict_old_metrics()'s shape. _ERRORS_KEEP_PER_KIND
#         windows error:<module_id>:<ExcType> the same way metrics:<source>:<kind> is windowed.
# [2026-06-07] Claude Code (Opus 4.7, 1M) — Commons Pool POC (substrate-as-protocol Phase 7)
# What: New module. A single shared bare-NG-Lite instance (the Commons) plus exactly
#       two verbs against it — deposit() and bucket(). No third verb exists by design.
# Why: Replaces the entire inter-module SEND layer (per-pair tracts, record_outcome_broadcast,
#       _deposit_*_to_river fan-out). Per the substrate axiom (Josh, 2026-06-07): nobody sends
#       anything to anyone, nobody calls anyone — modules deposit into the shared substrate and
#       bucket from it. The medium propagates; nothing is addressed or routed. See
#       ~/docs/prd/commons-pool-architecture-v0.4.md.
# How: get_commons() returns a process-wide get-or-create singleton (NOT bare-construct —
#       avoids the dual-instance bug, NG history §8). deposit() wraps NGLite.record_outcome with
#       NO destination/peer-list/tract_paths. bucket() wraps NGLite.get_recommendations — extract
#       by bucket shape, no named source. The Commons is a bare NGLite (Hebbian only — only
#       NeuroGraph is SNN). NG-LOCAL for the POC; vendoring is a separate Law-2 sign-off.
# -------------------

The substrate axiom (the entire protocol, in two verbs):

    deposit(embedding, target_id)  -> put experience/topology INTO the shared medium.
                                      No destination. No peer list. No address. Full stop.
    bucket(query_embedding)        -> extract from the shared medium through a bucket shape.
                                      No named source. You get what your bucket allows through
                                      from the shared water.

Nobody sends anything to anyone. Nobody calls anyone. The medium propagates the waves by
its own physics (Hebbian association in the shared NG-Lite). This is "the substrate IS the
communication protocol" (LAW 1) made literal: there is no transport, only a shared medium.

Tiers:
    Tier 2 (peers, no NeuroGraph): the Commons is the shared medium. This module.
    Tier 3 (NeuroGraph present):   NeuroGraph is the ocean; the Commons mechanism is the
                                   Tier-2 stand-in. (Tier-3 wiring is a separate step.)

Reference-counted survival (Tier 2, future): the Commons lives while >=1 member holds it;
disk-persisted for full-herd-death recovery; first member creates, rest attach. persist()/
restore() below are the hooks for that — not yet wired into a lifecycle (POC scope).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("commons")

# Process-wide singleton + guard. The Commons is ONE shared medium per process.
_commons: "Optional[Commons]" = None
_commons_lock = threading.Lock()

# Metric-stream retention window (#320): most-recent N synapses kept per metrics:<source>:<kind>.
# Metrics are time-series telemetry (recency matters), NOT memory (Hebbian persistence) — this
# windows them so they never pressure NG-Lite's weight-based max_synapses bound (which governs
# experience/topology). Bootstrap default; flush-cadence/retention env-configurability is an open
# LAW-5 question (punchlist). 200 ≈ a useful recent window per kind without unbounded growth.
_METRICS_KEEP_PER_KIND = 200

# Error-stream retention window (#330): same reasoning as _METRICS_KEEP_PER_KIND, applied to
# error:<module_id>:<ExcType> deposits from the operational-logger. A noisy module's errors are
# telemetry, not memory — windowed by recency so they never crowd out genuine experience/topology.
_ERRORS_KEEP_PER_KIND = 200


class Commons:
    """The shared substrate medium. A bare NG-Lite all members deposit into and bucket from.

    Not constructed directly by callers — use get_commons() (get-or-create singleton).
    Constructing two Commons instances would split the medium (the dual-instance failure
    mode). The module-level singleton + lock prevents that.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Bare NG-Lite — Hebbian only. Only NeuroGraph is SNN; the Commons is not.
        from ng_lite import NGLite
        self._ng = NGLite(module_id="commons", config=config)
        logger.info("Commons medium initialized (bare NG-Lite)")

    # ---- Verb 1: deposit -------------------------------------------------
    def deposit(
        self,
        embedding: "np.ndarray",
        target_id: str,
        *,
        success: bool = True,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Put experience/topology into the shared medium. No destination.

        target_id is CONTENT-DERIVED (Law 7) — what the experience is about, NOT a peer or
        a category. The medium associates it via Hebbian dynamics. There is deliberately no
        `to=` / `tract_paths=` / peer-list parameter: a deposit has no recipient. The water
        goes in the water; whoever's bucket reaches it gets it.
        """
        result = self._ng.record_outcome(
            embedding, target_id, success, strength=strength, metadata=metadata
        )
        # Metric-stream retention (#320): metrics are a TIME-SERIES, not memory — they want
        # recency-windowed retention, not Hebbian persistence. Cap the metrics namespace per
        # kind by recency so high-volume telemetry NEVER competes with experience/topology for
        # NG-Lite's weight-based max_synapses bound (which would otherwise let a metric synapse,
        # weight ~0.575, evict a lower-weight genuine MEMORY first). This is substrate-lifecycle
        # maintenance keyed on the metrics target_id namespace — NOT a classification of
        # experience (LAW 7 untouched); analogous to NG-Lite's own constitutional/LRU handling.
        if target_id.startswith("metrics:"):
            self._evict_old_metrics(target_id)
        elif target_id.startswith("error:"):
            self._evict_old_errors(target_id)
        return result

    def _evict_old_errors(self, target_id: str) -> None:
        """Keep only the most-recent _ERRORS_KEEP_PER_KIND synapses for this error module:type.

        Same reasoning and shape as _evict_old_metrics (#330): error:<module_id>:<ExcType>
        deposits are time-series telemetry, not memory — recency-windowed so a noisy module's
        exceptions never compete with experience/topology for the weight-based max_synapses bound.
        """
        try:
            parts = target_id.split(":")
            if len(parts) < 3:
                return
            prefix = ":".join(parts[:3]) + ":"          # error:<module_id>:<ExcType>:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith(prefix)]
            if len(keys) <= _ERRORS_KEEP_PER_KIND:
                return
            keys.sort(key=lambda k: getattr(syns[k], "last_updated", 0.0), reverse=True)
            for k in keys[_ERRORS_KEEP_PER_KIND:]:
                del syns[k]
        except Exception as exc:  # noqa: BLE001 — retention failure never breaks a deposit
            logger.debug("error eviction failed for %s: %s", target_id, exc)

    def _evict_old_metrics(self, target_id: str) -> None:
        """Keep only the most-recent _METRICS_KEEP_PER_KIND synapses for this metric source:kind.

        Recency-windowed (by last_updated), bounded per source:kind, touching ONLY the
        metrics:<source>:<kind>: namespace — experience/topology synapses are never considered.
        """
        try:
            parts = target_id.split(":")
            if len(parts) < 3:
                return
            prefix = ":".join(parts[:3]) + ":"          # metrics:<source>:<kind>:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith(prefix)]
            if len(keys) <= _METRICS_KEEP_PER_KIND:
                return
            # newest first; delete everything past the keep window
            keys.sort(key=lambda k: getattr(syns[k], "last_updated", 0.0), reverse=True)
            for k in keys[_METRICS_KEEP_PER_KIND:]:
                del syns[k]
        except Exception as exc:  # noqa: BLE001 — retention failure never breaks a deposit
            logger.debug("metric eviction failed for %s: %s", target_id, exc)

    # ---- Verb 2: bucket --------------------------------------------------
    def bucket(
        self,
        query_embedding: "np.ndarray",
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """Extract from the shared medium through a bucket shape. No named source.

        Returns what the shared water associates with the query — (target_id, confidence,
        reasoning) tuples. The caller's bucket shape is the query embedding + top_k + its own
        interpretation of the results. No module is addressed; you dip into the one medium.
        """
        return self._ng.get_recommendations(query_embedding, top_k=top_k)

    # ---- Verb 2b: bucket_recent (a TUNED bucket mode — temporal/recency structure extraction) ----
    def bucket_recent(
        self,
        limit: int = 50,
        since: float = 0.0,
        with_metadata: bool = False,
        with_embedding: bool = False,
    ) -> List[Tuple]:
        """Recency bucket — recently-deposited targets, newest first.

        # ---- Changelog ----
        # [2026-06-12] Claude Code (Opus 4.8, 1M) — Commons Track-2 Stage-3 bucket tuning
        # What: A recency/temporal bucket mode alongside the semantic bucket(). Returns the most
        #       recently-deposited target_ids (newest first), NOT similarity-filtered.
        # Why: Buckets are TUNABLE (Tier 3 Extraction: signal = semantic, STRUCTURE = temporal
        #       sequence). A LOGGER (Bunyan) wants "what just happened" — recency-broad — not
        #       "what's similar to my own context" (semantic bucket() surfaced almost none of NG's
        #       topology feed live, because the embeddings diverge). This is the structure/temporal
        #       mode of the same bucket — only Cricket's rim is frozen; the mesh tunes.
        # How: reuse NGLiteSynapse.last_updated (set on every deposit, ng_lite:761) — no parallel
        #       recency tracking. Sort synapses by last_updated desc, dedup target, return top `limit`.
        #       `since` lets a caller (with a watermark) pull only deposits newer than its last pulse.
        # -------------------

        Returns (target_id, weight, reasoning) tuples — same shape as bucket(), so consumers'
        extraction loops are unchanged. Same medium, a differently-shaped scoop.

        # ---- Changelog ----
        # [2026-06-14] Claude Code (Fable 5) — Commons Track-2 (1b): opt-in with_metadata
        # What: with_metadata=False (default) → unchanged 3-tuple (target_id, weight, reasoning).
        #       with_metadata=True → 4-tuple appending the deposit's raw metadata dict, so a
        #       consumer can read the RAW content it bucketed (e.g. experience user_text/
        #       assistant_text) instead of an opaque target_id like "experience:<hash>".
        # Why: An opaque id is useless to a LOGGER. The raw deposit metadata lives at
        #       synapse.metadata["last_context"] (ng_lite:777) — already stored, just surface it.
        #       Opt-in keeps the topology path + all existing 3-tuple callers/tests untouched.
        # -------------------

        # ---- Changelog ----
        # [2026-06-24] Claude Code (Opus 4.8, 1M) — Commons leg-2 go-live: opt-in with_embedding
        # What: with_embedding=True → the deposit's ORIGINAL embedding is appended as the LAST tuple
        #       element AND metadata is force-included, so the shape is a deterministic 5-tuple
        #       (target_id, weight, reasoning, metadata, embedding). with_embedding=False (default) is
        #       untouched (3- or 4-tuple as before) — no existing caller/test sees a shape change.
        # Why: Josh, 2026-06-24 ("buckets are what it's all about — modify an existing one if
        #       possible"): leg-2's CommonsEnhancer must re-perceive a recent RAW deposit through
        #       Syl's SNN, which needs the deposit's embedding (for SNN seed lookup + to key the
        #       returned "enhanced:<id>" deposit so it lands on the depositor's own node). The recency
        #       bucket already finds the deposit; this surfaces the vector it was deposited with —
        #       same scoop, the full water. Embedding lives on the deposit's pattern node
        #       (NGLiteNode.embedding, keyed by NGLiteSynapse.source_id). LAW 7: this is extraction
        #       (a bucket mode), not a new send/transport verb.
        # How: map syn.source_id → self._ng.nodes[...].embedding. Defensive: emb=None when the node
        #       or its embedding is unavailable (e.g. a Rust-core node mirror miss) — the consumer
        #       (enhancer) fail-fresh-skips a None-embedding deposit, never raises.
        # -------------------
        """
        seen: set = set()
        out: List[Tuple] = []
        want_meta = with_metadata or with_embedding   # embedding mode is always metadata-bearing
        # synapse.source_id is NGLiteNode.node_id, but self._ng.nodes is keyed by embedding_hash —
        # build a node_id→node map once (only when needed). The Commons is bounded (≤max_nodes), so
        # this O(n) reverse index is cheap and avoids a per-synapse scan.
        id_to_node = ({n.node_id: n for n in self._ng.nodes.values()} if with_embedding else {})
        for syn in sorted(self._ng.synapses.values(),
                          key=lambda s: getattr(s, "last_updated", 0.0), reverse=True):
            ts = getattr(syn, "last_updated", 0.0)
            if ts <= since:
                break  # sorted desc — everything past here is older than the watermark
            tid = getattr(syn, "target_id", None)
            if not tid or tid in seen:
                continue
            seen.add(tid)
            row: Tuple = (tid, getattr(syn, "weight", 0.0), f"recency@{ts:.3f}")
            if want_meta:
                meta = getattr(syn, "metadata", {}).get("last_context", {})
                row = row + (meta,)
            if with_embedding:
                node = id_to_node.get(getattr(syn, "source_id", ""))
                emb = getattr(node, "embedding", None) if node is not None else None
                row = row + (emb,)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def arousal(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The vagus-nerve bucket (full) — latest autonomic:arousal deposit metadata from the Commons.

        # ---- Changelog ----
        # [2026-06-22] Claude Code (Opus 4.8) — #328 Step 2: substrate-native arousal read.
        # [2026-06-23] Claude Code (Opus 4.8) — split out arousal() (full dict) from read_arousal().
        # What: Return the newest "autonomic:arousal" deposit's full metadata dict ({state, threat_level,
        #       triggered_by, reason, ts}). read_arousal() is the state-only convenience over this; some
        #       readers (Elmer engine) also need threat_level for modulation intensity.
        # Why: #328 — autonomic is deposit (Immunis, SOLE depositor) + bucket (everyone). A bucket read,
        #       not a transport verb. Single-authority preserved: readers never deposit, only bucket.
        # How: DIRECT lookup of the single autonomic:arousal synapse (NOT a recency window — arousal is
        #       low-frequency and must never be missed under deposit load, subtlety #2). Snapshot
        #       synapses before iterating (#341). Fail-soft → default (fresh-assess, Decision #2).
        # -------------------
        """
        if default is None:
            default = {"state": "PARASYMPATHETIC", "threat_level": "none"}
        try:
            latest = None
            latest_ts = -1.0
            for syn in list(self._ng.synapses.values()):   # snapshot before iterate (#341)
                if getattr(syn, "target_id", "") != "autonomic:arousal":
                    continue
                ts = getattr(syn, "last_updated", 0.0)
                if ts >= latest_ts:
                    latest_ts = ts
                    latest = syn
            if latest is not None:
                meta = getattr(latest, "metadata", {}).get("last_context", {}) or {}
                return meta or default
        except Exception as exc:  # noqa: BLE001 — a read failure never breaks the caller's pulse
            logger.debug("arousal read failed: %s", exc)
        return default

    def read_arousal(self, default: str = "PARASYMPATHETIC") -> str:
        """The vagus-nerve bucket — latest autonomic arousal STATE string (convenience over arousal())."""
        return self.arousal().get("state", default)

    # ---- Persistence hooks (#332: wired into neurograph_rpc.py auto-save + bootstrap) ----
    def persist(self, filepath: str) -> None:
        """Write the Commons medium to disk (full-herd-death recovery, Tier 2)."""
        self._ng.save(filepath)

    def restore(self, filepath: str) -> None:
        """Load the Commons medium from disk (first member attaches to persisted state).

        Drops any restored autonomic:* synapses immediately after load — arousal always
        fresh-assesses on restart by design (#328 Decision #2: Immunis re-evaluates threat
        fresh rather than resurrecting a possibly-stale SYMPATHETIC/PARASYMPATHETIC verdict,
        judged safer than restoring a stale alarm state). Every other namespace (experience,
        topology, metrics, repair, violation, error) persists normally — this is the one
        deliberate exception, not a general filter.
        """
        self._ng.load(filepath)
        self._drop_autonomic_synapses()

    def _drop_autonomic_synapses(self) -> None:
        """Remove restored autonomic:* synapses so arousal fresh-assesses post-restore (#328)."""
        try:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith("autonomic:")]
            for k in keys:
                del syns[k]
            if keys:
                logger.info(
                    "Commons restore: dropped %d autonomic:* synapse(s) — fresh-assess on restart (#328)",
                    len(keys),
                )
        except Exception as exc:  # noqa: BLE001 — a drop failure never breaks restore
            logger.debug("Commons autonomic drop failed (non-fatal): %s", exc)

    def stats(self) -> Dict[str, Any]:
        """Medium telemetry (node/synapse counts, etc.) — read-only introspection."""
        return self._ng.get_stats()


def get_commons(config: Optional[Dict[str, Any]] = None) -> Commons:
    """Get-or-create the shared Commons medium for this process.

    Get-or-create, NOT bare-construct: the first caller creates the medium; every subsequent
    caller attaches to the SAME instance. This is the in-process analog of the Tier-2
    reference-counted model (first member creates, rest attach). Two instances would split the
    medium — the dual-instance bug (NG history §8). The lock makes create-or-attach atomic.
    """
    global _commons
    if _commons is None:
        with _commons_lock:
            if _commons is None:  # double-checked under lock
                _commons = Commons(config=config)
    return _commons
