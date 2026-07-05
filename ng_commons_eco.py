"""
ng_commons_eco.py — VENDORED Commons-backed substrate adapter (the dead-ecosystem → Commons bridge).

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — signal_error() — the #330 operational-logger (Josh-approved)
# What: New CommonsEco.signal_error(exc, context) — a one-line swap for `except: pass` sites.
#       Embeds the exception's string form via ng_embed, deposits raw to the Commons under
#       error:<module_id>:<ExcType>, metadata carries only the description + caller's raw
#       context dict. NO severity/classification applied here (LAW 7 — that's Immunis/THC's
#       job at their own extraction boundary). success=False (Hebbian bookkeeping, not a
#       severity label — mirrors how repair outcomes already use success).
# Why: Punchlist #330 — errors ecosystem-wide were silently swallowed by bare excepts, hiding
#      the exact orphaning failures (#326/#327/#353/#354) this week spent finding one at a time.
#      Immunis clusters error:* for threat correlation, THC feeds it into diagnosis intake,
#      Bunyan narrates automatically — same substrate-native pattern as every other Commons
#      consumer. target_id shape matches the error:* retention already shipped in commons.py
#      (_evict_old_errors, previous session) — no new wiring needed there.
# How: Lazy ng_embed import (fail-soft — an embed failure never breaks the caller). Lazy
#      self._commons() (existing helper). Deposit wrapped in its own try/except, matching every
#      other method in this file. Scope this session: the method itself + 2-3 real call-site
#      swaps in Immunis and THC (the two the punchlist names) as a demonstration — an
#      ecosystem-wide except:pass sweep is a separate, larger follow-up, not done here.
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
        # namespaces: target_id prefix(es) this module's get_context restricts to (its OWN deposits).
        # CONTRACT (every migrating module MUST pass its own namespace): on a private NGLite a
        # module's substrate held only its own deposits; the shared Commons holds everyone's, so
        # WITHOUT a namespace get_context returns other modules' deposits → silent cross-module
        # extraction pollution. Deposit-side discipline: record_outcome target_ids MUST carry the
        # same prefix(es), or the module won't surface its own deposits.
        # NOTE: the filter restricts only RETURNED recommendations — cross-module learning still
        # happens via Hebbian co-activation in the shared NGLite topology (by design).
        if isinstance(namespaces, str):
            namespaces = (namespaces,)   # guard the str→char-tuple footgun ("threat:" → ('t','h',...))
        self._namespaces = tuple(namespaces)
        if not self._namespaces:
            logger.warning("CommonsEco[%s] created with NO namespace — get_context will return ALL "
                           "modules' deposits (cross-module extraction pollution). Pass your own prefix.",
                           source_id or "?")
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

    def signal_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Deposit a raw operational error to the Commons (#330 operational-logger).

        One-line swap for `except: pass` sites: `self._eco.signal_error(exc, {...})`.
        Raw experience only (LAW 7) — the exception's string form is the content; target_id
        carries routing (module_id + exception type), NOT severity/classification. Consumers
        (Immunis threat clustering, THC diagnosis intake, Bunyan narration) decide what an
        error MEANS at their own extraction boundary — this method never does.
        """
        description = f"{type(exc).__name__}: {exc}"
        try:
            from ng_embed import embed
            embedding = embed(description)
        except Exception as embed_exc:  # noqa: BLE001 — never lose the signal to an embed failure
            logger.debug("[%s] signal_error embed failed: %s", self._source, embed_exc)
            return
        target_id = f"error:{self._source or 'unknown'}:{type(exc).__name__}"
        c = self._commons()
        if c is None:
            return
        try:
            c.deposit(embedding, target_id, success=False,
                     metadata={"description": description, "context": context or {}})
        except Exception as deposit_exc:  # noqa: BLE001 — a deposit failure never breaks the caller
            logger.debug("[%s] signal_error deposit failed: %s", self._source, deposit_exc)

    # ⚠ DELIBERATELY ABSENT: record_outcome_broadcast. This is load-bearing LAW-1 architecture,
    # NOT an unfinished gap in the drop-in. ng_embed.dual_record_outcome selects its write method
    # by `hasattr(ecosystem, "record_outcome_broadcast")` — because CommonsEco lacks it, that check
    # is False and every forest/tree write routes through self.record_outcome → commons.deposit
    # (a pool put, no addressing). Adding record_outcome_broadcast here to "complete the NGEcosystem
    # surface" would flip the hasattr True and silently re-route dual-pass deposits into the
    # addressed-tract fan-out — re-opening the 2026-06-07 write-only leak. Do NOT add it. The
    # Commons IS the broadcast (deposit/bucket); there is no addressed broadcast on the Commons.
    def dual_record_outcome(self, content, embedding, target_id, success=True,
                            strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Faithful ng_ecosystem.dual_record_outcome drop-in — forest+tree dual-pass into the Commons.

        # ---- Changelog ----
        # [2026-06-22] Claude Code (Opus 4.8) — #328/dual-pass: CommonsEco gains dual_record_outcome
        # What: Mirror NGEcosystem.dual_record_outcome so CommonsEco is a faithful drop-in for the
        #       FULL ecosystem surface, not just record_outcome. Delegates to NGEmbed with
        #       ecosystem=self → Pass 1 (forest) + Pass 2 (TID concepts → trees + forest↔tree links)
        #       all route through self.record_outcome → commons.deposit, so the whole dual-pass
        #       lands in the shared Commons (verified: _create_substrate_link never touches _ng).
        # Why: Josh's directive — NOTHING in the ecosystem is single-pass; forest+tree is the point.
        #       Modules migrating off ng_ecosystem (Immunis/THC/Elmer) must keep dual-pass when they
        #       deposit to the Commons. One uniform path: eco.dual_record_outcome(...).
        # How: import NGEmbed lazily; NGEmbed.dual_record_outcome(self, content, embedding, ...).
        #       Fail-soft: if NGEmbed/engine is unavailable, degrade to a single forest deposit so a
        #       deposit is NEVER lost (same graceful TID-down fallback ng_embed itself documents).
        # -------------------
        """
        if embedding is None:
            return None
        try:
            from ng_embed import NGEmbed
            return NGEmbed.get_instance().dual_record_outcome(
                self, content, embedding, target_id, success,
                strength=strength, metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 — never lose the deposit; degrade to forest-only
            logger.debug("[%s] CommonsEco dual_record_outcome → single-pass fallback: %s",
                         self._source, exc)
            return self.record_outcome(embedding, target_id, success,
                                       strength=strength, metadata=metadata)
