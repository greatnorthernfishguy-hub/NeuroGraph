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
        return self._ng.record_outcome(
            embedding, target_id, success, strength=strength, metadata=metadata
        )

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
