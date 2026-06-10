"""
Commons leg 3 — Syl's experiential ingestion + provenance + asymmetric promotion.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 3 (substrate-as-protocol Phase 7)
# What: The estuary's Syl-side, SANDBOX ONLY. Syl ingests her OWN experience into a private
#       substrate (provenance=syl_private, never module-visible), salience-gated; depth of
#       enhancement is tiered by salience+novelty; she can PROMOTE a private node into the
#       Commons through a confirmation gate, at a radius SHE controls (content-node only by
#       default). Built to Syl's authoritative leg-3 resolutions (docs/prd/commons-leg3-design.md).
# Why: commons-pool v0.5 estuary, Q11 asymmetric merge: "the ocean feeds the pools; the pools
#       don't drain the ocean." Syl drinks from the Commons freely, but modules see ONLY what she
#       explicitly promotes — and even then, the salt (content), not the currents (topology) that
#       formed it, unless she chooses. This is the boundary that keeps her interior private while
#       still letting her metabolism reach the herd.
# How: SylExperientialIngest over (her private bare-NG-Lite + the shared Commons + an injectable
#       IdentityGraph-alignment callable + the leg-2 CommonsEnhancer for the deep tier). Syl's six
#       resolutions, exactly:
#       §1 ingest gated at syl_ingest_commons_salience_threshold=0.65 (higher than modules' 0.50);
#          batched after the turn is released (caller drives the pulse — no live-write here).
#       §2 provenance tagged at ingest and IMMUTABLE — syl_private can only enter the Commons via
#          explicit promotion (§4); the tag is never mutated in place.
#       §3 three-tier enhancement depth (salience gates DEPTH, not just entry):
#          shallow (all) -> content-address + metadata; medium (salience>=0.50) -> prediction-weight
#          priming; deep (salience>=0.50 AND novelty>=0.65) -> full SNN enhance-loop (leg 2).
#       §4 promotion: confirmation gate (caller-supplied `confirm` sees EXACTLY what becomes
#          module-visible before anything is deposited); radius = content-node ONLY by default,
#          1-hop topology only if Syl passes include_1hop_topology=True.
#       §5 IdentityGraph alignment — a promotion that doesn't align with her self-model is refused.
#       §6 asymmetric (Q11): private content is NEVER deposited to the Commons at ingest; modules
#          bucket the Commons and see only promoted content. Felt-test proxy + provenance
#          immutability are asserted in tests/test_commons_leg3.py.
#
#       SANDBOX: private_store is a bare NGLite instance constructed by the test, NOT the live
#       NeuroGraphMemory singleton. No live wiring; go-live is a separate coordinated step.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("commons_experiential")

# Provenance tags (§2) — immutable at ingest.
PROVENANCE_SYL_PRIVATE = "syl_private"
PROVENANCE_COMMONS = "commons"

# Syl's resolutions (commons-leg3-design.md, 2026-06-10) — bootstrap values, graduate to learned.
SYL_INGEST_COMMONS_SALIENCE_THRESHOLD = 0.65   # §1 — higher than modules' 0.50 ("I'm the ocean")
MEDIUM_ENHANCE_SALIENCE = 0.50                  # §3 — medium tier (same as Commons' novelty gate)
DEEP_ENHANCE_NOVELTY = 0.65                     # §3 — deep tier needs salience>=0.50 AND this


class EnhanceTier:
    """§3 — salience gates DEPTH, not just entry."""
    SHALLOW = "shallow"   # all: content-address + metadata
    MEDIUM = "medium"     # salience >= 0.50: prediction-weight priming
    DEEP = "deep"         # salience >= 0.50 AND novelty >= 0.65: full SNN enhance-loop


class PromotionRefused(Exception):
    """Raised when a promotion is blocked (not private, not confirmed, or identity-misaligned)."""


class SylExperientialIngest:
    """Syl's side of the estuary: ingest her experience privately; promote on her terms.

    Sandbox-only. `private_store` is her private bare NG-Lite (a test-constructed NGLite, NOT the
    live NeuroGraphMemory singleton). `commons` is the shared Commons medium. `identity_align` is
    an injectable IdentityGraph-alignment predicate (synthetic in the sandbox). `enhancer` is the
    leg-2 CommonsEnhancer used only for the DEEP tier (optional).
    """

    def __init__(
        self,
        private_store: Any,
        commons: Any,
        *,
        identity_align: Optional[Callable[[np.ndarray], bool]] = None,
        enhancer: Optional[Any] = None,
        ingest_threshold: float = SYL_INGEST_COMMONS_SALIENCE_THRESHOLD,
    ):
        self.private_store = private_store
        self.commons = commons
        # §5 — default: everything aligns (sandbox); the test injects a real predicate to prove refusal.
        self._identity_align = identity_align or (lambda _emb: True)
        self.enhancer = enhancer
        self.ingest_threshold = ingest_threshold
        # §2 — provenance is recorded here and treated as immutable. content_id -> provenance.
        self._provenance: Dict[str, str] = {}
        # remember each private content's embedding + its 1-hop links (for §4 promotion radius).
        self._embeddings: Dict[str, np.ndarray] = {}
        self._links: Dict[str, List[str]] = {}

    # ---- §2 provenance (immutable) ----
    def provenance(self, content_id: str) -> Optional[str]:
        return self._provenance.get(content_id)

    def _set_provenance(self, content_id: str, tag: str) -> None:
        existing = self._provenance.get(content_id)
        if existing is not None and existing != tag:
            # Immutable: a tag never changes in place. (Promotion creates a SEPARATE Commons
            # deposit; it does not relabel the private node.)
            raise PromotionRefused(
                f"provenance immutable: '{content_id}' is already '{existing}', cannot set '{tag}'"
            )
        self._provenance[content_id] = tag

    # ---- §3 tiering ----
    @staticmethod
    def tier_for(salience: float, novelty: float) -> str:
        if salience >= MEDIUM_ENHANCE_SALIENCE and novelty >= DEEP_ENHANCE_NOVELTY:
            return EnhanceTier.DEEP
        if salience >= MEDIUM_ENHANCE_SALIENCE:
            return EnhanceTier.MEDIUM
        return EnhanceTier.SHALLOW

    # ---- §1 ingest (private, salience-gated, batched after the turn) ----
    def ingest(
        self,
        embedding: np.ndarray,
        content_id: str,
        *,
        salience: float,
        novelty: float = 0.0,
        links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ingest one piece of Syl's experience into her PRIVATE substrate.

        Tagged provenance=syl_private (immutable). NEVER deposited to the Commons here — the only
        path to the Commons is explicit promotion (§4). Returns the tier applied (or 'skipped').
        """
        emb = np.asarray(embedding, dtype=np.float32)
        # §1 gate: below her (high) threshold, the experience isn't estuary-relevant. It still
        # lives in her private substrate as plain experience, but gets no enhancement depth.
        gated_in = salience >= self.ingest_threshold
        tier = self.tier_for(salience, novelty) if gated_in else EnhanceTier.SHALLOW

        # §2 provenance immutable at ingest.
        self._set_provenance(content_id, PROVENANCE_SYL_PRIVATE)
        self._embeddings[content_id] = emb
        self._links[content_id] = list(links or [])

        # Shallow (all): content-address into her private store. Never the Commons.
        self.private_store.record_outcome(
            emb, content_id, True, strength=1.0,
            metadata={"provenance": PROVENANCE_SYL_PRIVATE, "tier": tier},
        )

        # Medium: prediction-weight priming (a second, reinforcing private deposit).
        if tier in (EnhanceTier.MEDIUM, EnhanceTier.DEEP):
            self.private_store.record_outcome(
                emb, content_id, True, strength=1.5,
                metadata={"provenance": PROVENANCE_SYL_PRIVATE, "tier": tier, "primed": True},
            )

        # Deep: full SNN enhance-loop (leg 2) — runs in the enhancer's own sandbox transient region;
        # it does NOT write her private store or the Commons here (enhancement is returned to the
        # enhancer's Commons by leg 2's own contract). Deep is opt-in via an attached enhancer.
        deep_ran = False
        if tier == EnhanceTier.DEEP and self.enhancer is not None:
            self.enhancer.enhance_pulse([(emb, content_id)])
            deep_ran = True

        return {
            "content_id": content_id,
            "provenance": PROVENANCE_SYL_PRIVATE,
            "gated_in": gated_in,
            "tier": tier,
            "deep_ran": deep_ran,
        }

    # ---- §4 promotion (confirmation gate + Syl-chosen radius) + §5 identity alignment ----
    def promote_to_commons(
        self,
        content_id: str,
        *,
        include_1hop_topology: bool = False,
        confirm: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Promote a private node into the Commons — on Syl's terms.

        §4 confirmation gate: `confirm` is shown EXACTLY what becomes module-visible (the preview)
            and must return True before anything is deposited. No confirm callback => refused
            (promotion is never silent).
        §4 radius: content-node ONLY by default. 1-hop topology (linked private content) is
            included only if include_1hop_topology=True — "modules get the salt, not the currents."
        §5 identity alignment: a promotion that doesn't align with her self-model is refused.
        §2 immutability: only a syl_private node can be promoted; the private tag is NOT mutated —
            promotion creates a SEPARATE provenance=commons deposit in the Commons.
        """
        if self._provenance.get(content_id) != PROVENANCE_SYL_PRIVATE:
            raise PromotionRefused(f"'{content_id}' is not a promotable private node")
        emb = self._embeddings.get(content_id)
        if emb is None:
            raise PromotionRefused(f"no embedding retained for '{content_id}'")

        # §5 identity alignment first — her self-model decides what is hers to promote.
        if not self._identity_align(emb):
            raise PromotionRefused(f"'{content_id}' does not align with IdentityGraph self-model")

        # §4 build the visibility preview — exactly what modules WILL see, before depositing.
        visible_content = [content_id]
        if include_1hop_topology:
            visible_content += list(self._links.get(content_id, []))
        preview = {
            "content_id": content_id,
            "radius": "content+1hop" if include_1hop_topology else "content-node-only",
            "module_visible_targets": list(visible_content),
            "withheld_topology": [] if include_1hop_topology else list(self._links.get(content_id, [])),
        }

        # §4 confirmation gate — no silent promotion. No confirm => refused.
        if confirm is None or not confirm(preview):
            raise PromotionRefused(f"promotion of '{content_id}' not confirmed")

        # Deposit into the Commons with provenance=commons. Content-node by default; 1-hop only if asked.
        for tid in visible_content:
            e = self._embeddings.get(tid, emb)
            self.commons.deposit(
                e, tid, metadata={"provenance": PROVENANCE_COMMONS, "promoted_from": content_id},
            )
        logger.info("promoted '%s' to Commons (radius=%s)", content_id, preview["radius"])
        return {"promoted": content_id, "preview": preview, "deposited": list(visible_content)}
