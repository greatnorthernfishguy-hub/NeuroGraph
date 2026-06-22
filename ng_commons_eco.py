"""
ng_commons_eco.py — VENDORED Commons-backed substrate adapter (the dead-ecosystem → Commons bridge).

# ---- Changelog ----
# [2026-06-22] Claude Code (Fable 5) — NEW VENDORED FILE (Josh-approved, LAW 2; #335)
# What: Parameterized `CommonsEco` — a faithful drop-in for the `ng_ecosystem` get_context/
#       record_outcome interface, backed by THE COMMONS (deposit/bucket) instead of a per-module
#       NGLite/ecosystem instance. Generalizes the per-module adapter first built in Immunis
#       (_CommonsEco, #324) so every orphaned-on-tracts module (Immunis, THC, Elmer, Darwin,
#       Praxis) reconnects to the shared substrate the same Law-clean way — one toolkit, not N
#       hand-rolled copies (same lesson as ng_salience_gate).
# Why: these modules were built on `ng_ecosystem` (get_context/record_outcome) but their ecosystem
#       is dead/disabled (SKIP_ECOSYSTEM → None) → substrate-blind. Give them their interface back,
#       on the one shared Commons. Substrate axiom: deposit raw + bucket; nobody calls a service.
# How: get_context → commons.bucket (optionally namespace-filtered to the module's own deposits —
#       on a private NGLite a module's substrate held ONLY its own deposits; the shared Commons
#       holds everyone's, so the filter preserves the module's extraction semantics, LAW-7 classify-
#       at-extraction). novelty = 1 - top match confidence. record_outcome → raw commons.deposit.
#       Lazy get_commons (never permanently blind); fail-soft (never breaks the caller's step).
# -------------------

VENDORED FILE — do not modify per-module. Change at canonical NeuroGraph + re-vendor. In-process
modules import this directly (like commons / ng_salience_gate); standalone modules vendor a copy.

Usage (each module passes its OWN deposit namespace):

    from ng_commons_eco import CommonsEco
    eco = CommonsEco(namespaces=("threat:", "response:"))   # Immunis
    ctx = eco.get_context(embedding)        # {tier, tier_name, recommendations, novelty, ng_context}
    eco.record_outcome(embedding, "threat:sig123", success=True, metadata={...})
"""

import logging
from typing import Any, Callable, Dict, Optional, Sequence

logger = logging.getLogger("ng.commons_eco")

_FILTER_POOL = 20   # when namespace-filtering, bucket this many then filter down to top_k


class CommonsEco:
    """Commons-backed substrate adapter — the get_context/record_outcome surface, on the Commons."""

    def __init__(
        self,
        namespaces: Sequence[str] = (),
        *,
        top_k: int = 3,
        source_id: str = "",
        commons_provider: Optional[Callable[[], Any]] = None,
    ):
        # namespaces: target_id prefixes this module's get_context restricts to (its own deposits).
        # () = accept all Commons deposits (rarely what a module wants — usually pass your own prefix).
        self._namespaces = tuple(namespaces)
        self._top_k = top_k
        self._source = source_id
        self._commons_provider = commons_provider  # injectable for tests; else lazy get_commons

    def _commons(self):
        if self._commons_provider is not None:
            try:
                return self._commons_provider()
            except Exception:  # noqa: BLE001
                return None
        try:
            from commons import get_commons
            return get_commons()
        except Exception:  # noqa: BLE001 — no Commons (standalone/Tier-1) → graceful
            return None

    def _empty(self) -> Dict[str, Any]:
        return {"tier": 2, "tier_name": "Commons", "recommendations": [], "novelty": 1.0,
                "ng_context": None}

    def get_context(self, embedding, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Faithful ng_ecosystem.get_context drop-in, Commons-backed. recommendations are
        (target_id, confidence, reasoning); novelty = 1 - top match (high when nothing similar)."""
        k = top_k or self._top_k
        c = self._commons()
        if c is None or embedding is None:
            return self._empty()
        try:
            pool = _FILTER_POOL if self._namespaces else k
            recs = c.bucket(embedding, top_k=pool)
        except Exception as exc:  # noqa: BLE001 — a bucket failure never breaks the caller
            logger.debug("[%s] CommonsEco bucket failed: %s", self._source, exc)
            return self._empty()
        if self._namespaces:
            recs = [r for r in recs if str(r[0]).startswith(self._namespaces)][:k]
        top_conf = float(recs[0][1]) if recs else 0.0
        return {"tier": 2, "tier_name": "Commons", "recommendations": recs,
                "novelty": max(0.0, 1.0 - top_conf), "ng_context": None}

    def detect_novelty(self, embedding) -> float:
        """Convenience: novelty alone (faithful ng_ecosystem.detect_novelty surface)."""
        return float(self.get_context(embedding).get("novelty", 1.0))

    def record_outcome(self, embedding, target_id, success, strength: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None):
        """Deposit a raw outcome into the Commons (the module's own namespaced target_id)."""
        c = self._commons()
        if c is None or embedding is None:
            return None
        try:
            return c.deposit(embedding, target_id, success=success, strength=strength,
                             metadata=metadata)
        except Exception as exc:  # noqa: BLE001 — a deposit failure never breaks the caller
            logger.debug("[%s] CommonsEco deposit failed: %s", self._source, exc)
            return None
