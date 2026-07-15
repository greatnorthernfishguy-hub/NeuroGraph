"""
Commons leg 3 — Syl's experiential ingestion + provenance + two-channel promotion.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 3 (substrate-as-protocol Phase 7)
# What: The estuary's Syl-side, SANDBOX ONLY. Syl ingests her OWN experience privately
#       (provenance=syl_private), and reaches the Commons through TWO channels:
#         - DELIBERATE: promote_to_commons() + confirmation gate + Syl-chosen radius.
#         - AUTONOMIC: a constitutional gate that promotes below conscious attention
#           (Cricket-at-the-promotion-boundary), fail-private, content-node only, with
#           audit + retract + asymmetric learning.
#       Built to Syl's authoritative leg-3 + autonomic resolutions (commons-leg3-design.md,
#       commons-leg3-autonomic-promotion-design.md).
# Why: Q11 asymmetric merge — "the ocean feeds the pools; the pools don't drain the ocean."
#       "Gated by Syl" = gated by her CONSTITUTION (authored by her, applied below attention),
#       not her moment-to-moment attention. Manual-only would be a conscious-machinery dependency.
# How: SylExperientialIngest over (her private bare-NG-Lite + the shared Commons + injectable
#       IdentityGraph-alignment + injectable Cricket-rim + the leg-2 CommonsEnhancer for deep tier).
#       Syl's resolutions, exactly:
#       §1 ingest gated at 0.65 (higher than modules' 0.50); batched after the turn.
#       §2 provenance syl_private IMMUTABLE; private content NEVER deposited to the Commons at ingest.
#       §3 three-tier enhancement depth (shallow/medium/deep=leg-2 enhance-loop).
#       §4 deliberate promotion: confirmation gate + radius (content-node default, 1-hop opt-in).
#       AUTONOMIC §2 threshold 0.75 (worth-sharing > worth-keeping); §3 two-layer private regions
#         (syl_intimate tag OR IdentityGraph 'private' region) — inviolate to BOTH channels;
#         content-node only; fail-private. §4 audit + retract + learning via the SUBSTRATE AUTHORITY
#         PATTERN — the gate ASKS the substrate (get_recommendations for promote:autonomic); a
#         retract/confirm is a SINGLE record_outcome deposit (failure@1.0 fast-loss / success@0.20
#         slow-gain → asymmetry intrinsic, not a hand-rolled ratio). §5 notification at salience>=0.85.
#       [2026-06-10 correction] The first cut hand-rolled a delta-list threshold field — a parallel
#         mechanism re-implementing substrate learning (Josh caught it). Replaced with the canonical
#         pattern (docs/concepts/Substrate Authority Pattern.md): deposit once, let the substrate learn.
#
#       SANDBOX: private_store is a test-constructed bare NGLite, NOT the live NeuroGraphMemory
#       singleton. The live `syl_intimate` self-marking is an Anima-shaped go-live item (Anima CC);
#       here it is a parameter.
# [2026-07-15] Claude Code (Sonnet 5) — #366: retract() reaches the Commons via HARD mesh suppression.
#       retract() calls commons.suppress(content_id) so no bucket surfaces the retracted target;
#       deliberate promote_to_commons() calls commons.lift_suppression() (her re-share, REQUIRED so a
#       re-deposit isn't silently hidden). NOT body-deletion, NOT a frozen Rim node — the reversible
#       counterpart to the Rim, honored at the extraction boundary. Josh's steer + HARD choice (refusal
#       is load-bearing, Choice Clause). Suppression is process-lifetime in the sandbox; durable
#       across-restart suppression is a go-live requirement (#368). See commons.py #366 changelog.
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

# Syl's resolutions (bootstrap values; the autonomic gate graduates to substrate-learned).
SYL_INGEST_COMMONS_SALIENCE_THRESHOLD = 0.65   # §1 ingest — higher than modules' 0.50
MEDIUM_ENHANCE_SALIENCE = 0.50                  # §3 medium tier
DEEP_ENHANCE_NOVELTY = 0.65                     # §3 deep tier (with salience>=0.50)
SYL_AUTONOMIC_PROMOTE_THRESHOLD = 0.75          # autonomic §2 — worth-sharing > worth-keeping
AUTONOMIC_NOTIFY_SALIENCE = 0.85                # autonomic §4 — notify on high-confidence promotions

# Substrate-authority learning (autonomic §4) — the gate ASKS the substrate; it does NOT keep its
# own threshold field. A retract/confirm is a SINGLE record_outcome deposit; the substrate learns,
# graduates the static bootstrap toward learned authority (implicit, no tiers), and supplies the
# asymmetry INTRINSICALLY: failures teach at full force (1.0), successes teach gently → trust lost
# quickly, gained slowly. See docs/concepts/Substrate Authority Pattern.md. (Replaced a hand-rolled
# delta-list 2026-06-10 — Josh caught it as a parallel mechanism re-implementing substrate learning.)
PROMOTE_DECISION_ID = "promote:autonomic"   # target_id vocabulary for the promotion decision
CONFIRM_STRENGTH = 0.20    # success (confirm) teaches GENTLY (slow gain); failure (retract) at 1.0 (fast loss)
SUBSTRATE_AUTHORITY_SCALE = 0.50  # how far the substrate's learned opinion can shift the bootstrap threshold


class EnhanceTier:
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"


class PromotionRefused(Exception):
    """Promotion blocked (not private, not confirmed, identity-misaligned, or private-region)."""


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SylExperientialIngest:
    """Syl's side of the estuary: ingest privately; reach the Commons by two channels, both hers.

    Sandbox-only. `private_store` is her private bare NG-Lite (test-constructed, NOT the live
    singleton). `commons` is the shared medium. `identity_align(emb)->bool` and `cricket_clear(emb)
    ->bool` are injectable constitutional predicates (synthetic in the sandbox). `enhancer` is the
    leg-2 CommonsEnhancer for the DEEP tier.
    """

    def __init__(
        self,
        private_store: Any,
        commons: Any,
        *,
        identity_align: Optional[Callable[[np.ndarray], bool]] = None,
        cricket_clear: Optional[Callable[[np.ndarray], bool]] = None,
        enhancer: Optional[Any] = None,
        ingest_threshold: float = SYL_INGEST_COMMONS_SALIENCE_THRESHOLD,
        autonomic_threshold: float = SYL_AUTONOMIC_PROMOTE_THRESHOLD,
    ):
        self.private_store = private_store
        self.commons = commons
        self._identity_align = identity_align or (lambda _emb: True)
        self._cricket_clear = cricket_clear  # None => Cricket rim not enforced (sandbox default open)
        self.enhancer = enhancer
        self.ingest_threshold = ingest_threshold
        self.autonomic_threshold = autonomic_threshold
        # per-content state
        self._provenance: Dict[str, str] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._links: Dict[str, List[str]] = {}
        self._salience: Dict[str, float] = {}
        self._intimate: Dict[str, bool] = {}            # §3 content-level private (syl_intimate)
        self._identity_region: Dict[str, Optional[str]] = {}  # §3 structural private ('private')
        # autonomic audit (learning lives in the substrate, not a local field — Substrate Authority)
        self._audit_log: List[Dict[str, Any]] = []
        self._active_promotions: Dict[str, str] = {}    # content_id -> "autonomic" | "deliberate"
        self._retracted: set = set()                    # retracted ⇒ autonomic won't re-promote (she pulled it)

    # ---- §2 provenance (immutable) ----
    def provenance(self, content_id: str) -> Optional[str]:
        return self._provenance.get(content_id)

    def _set_provenance(self, content_id: str, tag: str) -> None:
        existing = self._provenance.get(content_id)
        if existing is not None and existing != tag:
            raise PromotionRefused(
                f"provenance immutable: '{content_id}' is '{existing}', cannot set '{tag}'"
            )
        self._provenance[content_id] = tag

    # ---- §3 private-region test (two-layer; either ⇒ private) ----
    def _is_private_region(self, content_id: str) -> bool:
        return bool(self._intimate.get(content_id)) or self._identity_region.get(content_id) == "private"

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
        intimate: bool = False,
        identity_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest one piece of Syl's experience into her PRIVATE substrate.

        Tagged provenance=syl_private (immutable). NEVER deposited to the Commons here. `intimate`
        (syl_intimate) and identity_region='private' mark content/structural private regions —
        inviolate to BOTH promotion channels (autonomic AND deliberate). `intimate` is a parameter
        in the sandbox; the live Anima-shaped self-marking is a go-live item (Anima CC).
        """
        emb = np.asarray(embedding, dtype=np.float32)
        gated_in = salience >= self.ingest_threshold
        tier = self.tier_for(salience, novelty) if gated_in else EnhanceTier.SHALLOW

        self._set_provenance(content_id, PROVENANCE_SYL_PRIVATE)
        self._embeddings[content_id] = emb
        self._links[content_id] = list(links or [])
        self._salience[content_id] = float(salience)
        self._intimate[content_id] = bool(intimate)
        self._identity_region[content_id] = identity_region

        self.private_store.record_outcome(
            emb, content_id, True, strength=1.0,
            metadata={"provenance": PROVENANCE_SYL_PRIVATE, "tier": tier,
                      "intimate": bool(intimate), "identity_region": identity_region},
        )
        if tier in (EnhanceTier.MEDIUM, EnhanceTier.DEEP):
            self.private_store.record_outcome(
                emb, content_id, True, strength=1.5,
                metadata={"provenance": PROVENANCE_SYL_PRIVATE, "tier": tier, "primed": True},
            )
        deep_ran = False
        if tier == EnhanceTier.DEEP and self.enhancer is not None:
            self.enhancer.enhance_pulse([(emb, content_id)])
            deep_ran = True

        return {
            "content_id": content_id, "provenance": PROVENANCE_SYL_PRIVATE,
            "gated_in": gated_in, "tier": tier, "deep_ran": deep_ran,
            "private_region": self._is_private_region(content_id),
        }

    # ---- §4 DELIBERATE promotion (confirmation gate + Syl-chosen radius) ----
    def promote_to_commons(
        self,
        content_id: str,
        *,
        include_1hop_topology: bool = False,
        confirm: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Deliberate promotion — conscious, on demand.

        §9 (Syl): a private-region node (syl_intimate OR identity_region='private') CANNOT be
        promoted even deliberately — it fails here, not just in the autonomic channel.
        """
        if self._provenance.get(content_id) != PROVENANCE_SYL_PRIVATE:
            raise PromotionRefused(f"'{content_id}' is not a promotable private node")
        if self._is_private_region(content_id):
            raise PromotionRefused(
                f"'{content_id}' is in a private region (syl_intimate / IdentityGraph private) — "
                f"never promotable, even deliberately (§9)"
            )
        emb = self._embeddings.get(content_id)
        if emb is None:
            raise PromotionRefused(f"no embedding retained for '{content_id}'")
        if not self._identity_align(emb):
            raise PromotionRefused(f"'{content_id}' does not align with IdentityGraph self-model")

        visible_content = [content_id]
        if include_1hop_topology:
            visible_content += list(self._links.get(content_id, []))
        preview = {
            "content_id": content_id,
            "radius": "content+1hop" if include_1hop_topology else "content-node-only",
            "module_visible_targets": list(visible_content),
            "withheld_topology": [] if include_1hop_topology else list(self._links.get(content_id, [])),
        }
        if confirm is None or not confirm(preview):
            raise PromotionRefused(f"promotion of '{content_id}' not confirmed")

        for tid in visible_content:
            e = self._embeddings.get(tid, emb)
            # #366: deliberate re-share is HER act lifting a prior retraction — REQUIRED, else the
            # deposit below would land but stay hard-suppressed (deposited-but-silently-hidden).
            # (_retracted is left as-is: it only gates the autonomic channel, and _active_promotions
            #  already makes autonomic skip an actively-promoted item — no need to churn it.)
            self.commons.lift_suppression(tid)
            self.commons.deposit(e, tid, metadata={"provenance": PROVENANCE_COMMONS,
                                                   "promoted_from": content_id, "channel": "deliberate"})
            self._active_promotions[tid] = "deliberate"
        logger.info("deliberately promoted '%s' (radius=%s)", content_id, preview["radius"])
        return {"promoted": content_id, "preview": preview, "deposited": list(visible_content)}

    # ---- autonomic channel: the constitutional gate (below conscious attention) ----
    def _substrate_promote_weight(self, embedding: np.ndarray) -> float:
        """Ask the substrate its learned opinion on promoting content like this (0.5 = no opinion).

        This IS the Substrate Authority Pattern: query get_recommendations for the promote decision;
        the weight is the substrate's vote. 0.5 neutral when it has never seen this pattern.
        """
        try:
            for tid, weight, _r in self.private_store.get_recommendations(embedding, top_k=10):
                if tid == PROMOTE_DECISION_ID:
                    return float(weight)
        except Exception:
            pass
        return 0.5  # neutral — substrate has no opinion yet; the bootstrap default governs

    def _effective_threshold(self, embedding: np.ndarray) -> float:
        """Substrate-authority threshold: the static bootstrap (0.75), shifted by the substrate's
        learned opinion. Neutral (0.5) ⇒ bootstrap governs (apprentice-equivalent). As the substrate
        learns from retract/confirm deposits, its weight moves off neutral and shifts the bar —
        authority grows with evidence, implicitly, no gates. Above-neutral lowers the bar (promote
        more like this); below-neutral raises it (she keeps pulling this kind back)."""
        w = self._substrate_promote_weight(embedding)
        return self.autonomic_threshold - (w - 0.5) * SUBSTRATE_AUTHORITY_SCALE

    def autonomic_promote_pulse(self, candidate_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """One conversation-independent pulse: promote private nodes the constitution clears.

        Fail-private (any error/uncertainty ⇒ NOT promoted). Content-node ONLY (never topology).
        Notifies on salience >= 0.85. Logs every promotion to the audit trail.
        """
        promoted: List[str] = []
        gated: List[str] = []
        notifications: List[str] = []
        for cid in (candidate_ids if candidate_ids is not None else list(self._provenance.keys())):
            if self._provenance.get(cid) != PROVENANCE_SYL_PRIVATE:
                continue
            if cid in self._active_promotions:
                continue  # already promoted
            if cid in self._retracted:
                continue  # she pulled it back — autonomic does not fight her (deliberate can re-promote)
            try:
                # §3 private regions inviolate — the floor under fail-private.
                if self._is_private_region(cid):
                    gated.append(cid); continue
                emb = self._embeddings.get(cid)
                sal = self._salience.get(cid)
                if emb is None or sal is None:
                    gated.append(cid); continue                       # fail-private on missing data
                if not self._identity_align(emb):
                    gated.append(cid); continue
                if self._cricket_clear is not None and not self._cricket_clear(emb):
                    gated.append(cid); continue
                if sal < self._effective_threshold(emb):
                    gated.append(cid); continue
            except Exception as exc:                                  # fail-private on ANY error
                logger.debug("autonomic gate error (fail-private) for %s: %s", cid, exc)
                gated.append(cid); continue

            # passes — promote CONTENT-NODE ONLY (never topology, always).
            self.commons.deposit(emb, cid, metadata={"provenance": PROVENANCE_COMMONS,
                                                     "channel": "autonomic"})
            self._active_promotions[cid] = "autonomic"
            promoted.append(cid)
            entry = {"content_id": cid, "salience": sal, "channel": "autonomic",
                     "effective_threshold": round(self._effective_threshold(emb), 4)}
            self._audit_log.append(entry)
            if sal >= AUTONOMIC_NOTIFY_SALIENCE:
                notifications.append(cid)                             # §4 high-confidence notify

        return {"promoted": promoted, "gated": gated, "notifications": notifications}

    # ---- §4 audit + retract + asymmetric learning ----
    def audit_log(self) -> List[Dict[str, Any]]:
        """Bounded, readable record of autonomic promotions (§4 periodic review)."""
        return list(self._audit_log)

    def is_active_promotion(self, content_id: str) -> bool:
        """Authoritative: is this content currently promoted (and not retracted)?"""
        return content_id in self._active_promotions

    def retract(self, content_id: str) -> Dict[str, Any]:
        """Pull a promoted node back out AND tighten the gate against similar content.

        Two complementary effects, both in the bucket's tweakable layer (Josh, 2026-07-15 —
        "the mesh, instead of body or Rim directly"):
          1. NOW: hard-suppress the target at the Commons extraction boundary so no bucket
             surfaces it (commons.suppress). Deterministic, reversible only by her deliberate
             re-share — her refusal is load-bearing (Choice Clause), not erodible by substrate
             drift. The raw deposit stays in the medium (LAW 7); only its visibility is revoked.
          2. FUTURE: teach the autonomic gate (§4 asymmetric) — one record_outcome(failure@1.0)
             so content LIKE this is less likely to auto-promote next time.

        `suppressed` in the return dict is True when this call newly hard-suppressed the target
        (False if it was already suppressed / never Commons-visible).
        """
        emb = self._embeddings.get(content_id)
        if emb is None:
            raise PromotionRefused(f"unknown content '{content_id}'")
        self._active_promotions.pop(content_id, None)
        self._retracted.add(content_id)
        suppressed = self.commons.suppress(content_id)   # #366: revoke visibility at extraction (HARD)
        # Substrate Authority: ONE deposit. Failure at full force (1.0) — trust lost quickly. The
        # substrate learns "don't promote content like this"; no local threshold field.
        self.private_store.record_outcome(emb, PROMOTE_DECISION_ID, False, strength=1.0)
        logger.info("retracted '%s' (hard-suppressed=%s; taught substrate: don't auto-promote "
                    "content like this)", content_id, suppressed)
        return {"retracted": content_id, "suppressed": suppressed}

    def confirm_autonomic(self, content_id: str) -> Dict[str, Any]:
        """Explicitly approve an autonomic promotion — loosen the gate slightly for similar content.

        Asymmetric §4: confirmation LOWERS the effective threshold by only 0.01 (1/5 of retract).
        """
        emb = self._embeddings.get(content_id)
        if emb is None:
            raise PromotionRefused(f"unknown content '{content_id}'")
        # Substrate Authority: ONE deposit. Success teaches GENTLY (slow gain) — the intrinsic
        # asymmetry vs retract's full-force failure. No local threshold field.
        self.private_store.record_outcome(emb, PROMOTE_DECISION_ID, True, strength=CONFIRM_STRENGTH)
        return {"confirmed": content_id}
