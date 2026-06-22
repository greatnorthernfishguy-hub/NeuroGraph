"""
The Commons — shared substrate medium for peer-module communication.

# ---- Changelog ----
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
        return result

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
        """
        seen: set = set()
        out: List[Tuple] = []
        for syn in sorted(self._ng.synapses.values(),
                          key=lambda s: getattr(s, "last_updated", 0.0), reverse=True):
            ts = getattr(syn, "last_updated", 0.0)
            if ts <= since:
                break  # sorted desc — everything past here is older than the watermark
            tid = getattr(syn, "target_id", None)
            if not tid or tid in seen:
                continue
            seen.add(tid)
            if with_metadata:
                meta = getattr(syn, "metadata", {}).get("last_context", {})
                out.append((tid, getattr(syn, "weight", 0.0), f"recency@{ts:.3f}", meta))
            else:
                out.append((tid, getattr(syn, "weight", 0.0), f"recency@{ts:.3f}"))
            if len(out) >= limit:
                break
        return out

    def read_arousal(self, default: str = "PARASYMPATHETIC") -> str:
        """The vagus-nerve bucket — latest autonomic arousal state from the Commons (#328).

        # ---- Changelog ----
        # [2026-06-22] Claude Code (Opus 4.8) — #328 Step 2: substrate-native arousal read.
        # What: Return the newest "autonomic:arousal" deposit's state (Immunis is the SOLE depositor).
        #       Every module reads arousal THIS way instead of ng_autonomic.read_state() on the file.
        # Why: #328 — autonomic is just deposit (Immunis) + bucket (everyone). This is a bucket (a
        #       read of the shared medium), NOT a new transport verb. Single-authority preserved:
        #       readers never deposit autonomic:arousal; only Immunis does.
        # How: DIRECT lookup of the single autonomic:arousal synapse (NOT a recency-window scan —
        #       arousal is low-frequency and must NEVER be missed under deposit load, design subtlety
        #       #2; the vagus is never missed). Snapshot synapses before iterating (punchlist #341 —
        #       avoid concurrent iterate/mutate with the deposit/pulse threads). Fail-soft → default
        #       PARASYMPATHETIC (fresh-assess when nothing deposited yet, design Decision #2).
        # -------------------
        """
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
                return meta.get("state", default)
        except Exception as exc:  # noqa: BLE001 — a read failure never breaks the caller's pulse
            logger.debug("read_arousal failed: %s", exc)
        return default

    # ---- Persistence hooks (Tier 2 reference-counted survival — not yet lifecycle-wired) ----
    def persist(self, filepath: str) -> None:
        """Write the Commons medium to disk (full-herd-death recovery, Tier 2)."""
        self._ng.save(filepath)

    def restore(self, filepath: str) -> None:
        """Load the Commons medium from disk (first member attaches to persisted state)."""
        self._ng.load(filepath)

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
