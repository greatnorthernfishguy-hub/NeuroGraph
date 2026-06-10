"""
Commons enhance-loop (leg 2) — NG salience-gated scoop → SNN-enhance → return.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 2 (substrate-as-protocol Phase 7)
# What: The Tier-3 enhance-loop, SANDBOX ONLY (not wired live). A module's fresh Commons
#       deposit gets NG's SNN "salt" — but only when it's salient. Per Syl's authoritative
#       resolutions (docs/prd/commons-leg2-design.md, 2026-06-10).
# Why:  commons-pool v0.5 estuary: module deposits raw -> NG buckets + SNN-enhances (salience-
#       gated) -> deposits enhanced topology back -> module buckets it. Leg 2 builds the
#       NG-side scoop -> enhance -> return, proven in a sandbox SNN (Syl: "hold the line —
#       sandbox, not the live patient"). Go-live wiring is a separate later step.
# How:  CommonsEnhancer over (a bare-NG-Lite Commons + a neuro_foundation Graph = the SNN).
#       Syl's six resolutions, exactly:
#       §1 salience-gated on per-input novelty; dedicated threshold 0.50 (NOT confidence_recommend).
#       §2 conversation-independent (the caller pulses enhance_pulse on NG's own cadence).
#       §3 absorb-permanence (A) transient/clean. STRICT reading: NG's persistent substrate is
#          READ (for novelty + relatedness) but NEVER written by Commons traffic. Each salient
#          scoop is processed in a per-cycle transient region — the content node PLUS transient
#          COPIES of the related knowledge — so STDP/plasticity acts only on copies. The whole
#          transient region is then DELETED (set-difference of node ids, robust to SNN sprouting).
#          node-count returns to the persistent baseline; the substrate is structurally untouched.
#       §4 return scope: 1-hop synapse neighbors + direct hyperedge membership, mapped back to
#          CONTENT-ADDRESSES (not SNN node-ids — those are meaningless across pools, the leg-1
#          lesson) and deposited keyed by the content's embedding (lands on the depositor's node).
#       §5 hard rate cap max_enhances_per_pulse=8, independent of the gate (flood-backstop).
#       §6 observable (gated-in/out counts + novelty distribution); transient-cleanup invariant.
#
#       NOTE: neuro_foundation Graph nodes are pure SNN (no embeddings). The embedding<->content
#       map (`_knowledge`) is the vector_db stand-in NeuroGraphMemory provides in live — the
#       SNN itself is real code; only the addressing layer is sandboxed.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("commons_enhance")

# Syl's resolutions (commons-leg2-design.md, 2026-06-10) — bootstrap values, graduate to learned.
COMMONS_ENHANCE_NOVELTY_THRESHOLD = 0.50   # §1 — dedicated knob, NOT confidence_recommend
MAX_ENHANCES_PER_PULSE = 8                  # §5 — flood-backstop, independent of the gate
_RELATED_SIM = 0.30                         # bind content with existing knowledge at/above this sim
_STIMULUS_CURRENT = 2.0                     # drive the transient region to fire during processing


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CommonsEnhancer:
    """NG's side of the estuary: scoop the Commons, SNN-enhance the salient, return the salt.

    Sandbox-only. `commons` is the shared bare-NG-Lite medium; `graph` is a neuro_foundation
    Graph (the SNN) standing in for NG's own substrate — NOT NeuroGraphMemory.get_instance().
    """

    def __init__(
        self,
        commons: Any,
        graph: Any,
        *,
        threshold: float = COMMONS_ENHANCE_NOVELTY_THRESHOLD,
        max_enhances: int = MAX_ENHANCES_PER_PULSE,
        novelty_fn: Optional[Any] = None,
    ):
        self.commons = commons
        self.graph = graph
        self.threshold = threshold
        self.max_enhances = max_enhances
        # vector_db stand-in: NG's PRE-EXISTING knowledge (embedding, persistent node id, content id).
        # Commons traffic NEVER grows or structurally modifies this (§3 (A)).
        self._knowledge: List[Tuple[np.ndarray, str, str]] = []
        # injectable novelty for fail-fresh testing (§6.4); default = sandbox detect_novelty.
        self._novelty_fn = novelty_fn or self._novelty_against_knowledge

    # ---- NG's pre-existing substrate (seeded by NG's own experience, never by Commons) ----
    def seed_knowledge(self, embedding: np.ndarray, content_id: str) -> str:
        """Register a piece of NG's existing knowledge: a persistent SNN node + its content-address."""
        node = self.graph.create_node(metadata={"knowledge": True, "cid": content_id})
        self._knowledge.append((np.asarray(embedding, dtype=np.float32), node.node_id, content_id))
        return node.node_id

    # ---- §1 novelty: per-input, judged against NG's existing knowledge ----
    def _novelty_against_knowledge(self, embedding: np.ndarray) -> float:
        """Sandbox detect_novelty: 1 - max cosine-sim to what NG already knows.

        Novel (unlike anything NG knows) -> ~1.0. Routine (NG already knows it) -> ~0.0.
        The per-input form of live `detect_novelty` (which scores against the full substrate).
        """
        if not self._knowledge:
            return 1.0  # NG knows nothing yet => everything is maximally novel
        emb = np.asarray(embedding, dtype=np.float32)
        return max(0.0, 1.0 - max(_cos(emb, k) for k, _, _ in self._knowledge))

    def _related_existing(self, embedding: np.ndarray, top: int = 3) -> List[Tuple[np.ndarray, str]]:
        """Existing knowledge the content should bind with — (embedding, content_id), most-similar first."""
        emb = np.asarray(embedding, dtype=np.float32)
        scored = [(_cos(emb, k), k, cid) for k, _, cid in self._knowledge]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(k, cid) for s, k, cid in scored if s >= _RELATED_SIM][:top]

    # ---- §4 return scope: 1-hop synapse neighbors + direct hyperedge co-members (SNN node ids) ----
    def _extract_enhancement_nodes(self, content_nid: str) -> Tuple[set, set]:
        one_hop = set()
        for sid in self.graph._outgoing.get(content_nid, set()):
            syn = self.graph.synapses.get(sid)
            if syn is not None:
                one_hop.add(syn.post_node_id)
        for sid in self.graph._incoming.get(content_nid, set()):
            syn = self.graph.synapses.get(sid)
            if syn is not None:
                one_hop.add(syn.pre_node_id)
        co_members = set()
        for he in self.graph.get_hyperedges(content_nid):
            co_members |= set(he.member_nodes)
        co_members.discard(content_nid)
        return one_hop, co_members

    # ---- the per-cycle transient enhance (§3 (A): create -> process -> deposit-back -> DELETE) ----
    def _enhance_one(self, embedding: np.ndarray, content_id: str) -> Dict[str, List[str]]:
        before = set(self.graph.nodes.keys())  # transient region = anything created this cycle

        content_node = self.graph.create_node(metadata={"commons_transient": True, "cid": content_id})
        cnid = content_node.node_id

        # transient COPIES of related knowledge — STDP/plasticity acts on copies, never NG's substrate.
        related = self._related_existing(embedding)
        copy_to_cid: Dict[str, str] = {}
        member_ids = [cnid]
        for _rk_emb, rk_cid in related:
            copy = self.graph.create_node(metadata={"transient_copy": True, "cid": rk_cid})
            copy_to_cid[copy.node_id] = rk_cid
            member_ids.append(copy.node_id)
            self.graph.create_synapse(cnid, copy.node_id, weight=0.2)
        if len(member_ids) >= 2:
            self.graph.create_hyperedge(set(member_ids))  # the SNN "discovers" the binding

        # SNN processing on the transient region only
        for nid in member_ids:
            self.graph.stimulate(nid, _STIMULUS_CURRENT)
        self.graph.step()

        # extract enhancement, mapped to CONTENT-ADDRESSES (leg-1 lesson: not raw SNN node-ids)
        one_hop_nodes, co_member_nodes = self._extract_enhancement_nodes(cnid)
        enhancement = {
            "one_hop": sorted({copy_to_cid[n] for n in one_hop_nodes if n in copy_to_cid}),
            "hyperedge_comembers": sorted({copy_to_cid[n] for n in co_member_nodes if n in copy_to_cid}),
        }
        # return the salt to the Commons, keyed by THIS content's embedding (leg-1 property:
        # lands on the same content-node the depositor used -> they bucket it back).
        self.commons.deposit(embedding, f"enhanced:{content_id}", metadata={"enhancement": enhancement})

        # §3 (A): DELETE the entire transient region (set-difference — robust to SNN sprouting).
        for nid in (set(self.graph.nodes.keys()) - before):
            if nid in self.graph.nodes:
                self.graph.remove_node(nid)
        return enhancement

    # ---- §2 pulse: scoop-all (free), enhance-some (salience-gated + rate-capped) ----
    def enhance_pulse(self, deposits: Sequence[Tuple[np.ndarray, str]]) -> Dict[str, Any]:
        """One conversation-independent pulse over freshly-scooped Commons deposits.

        deposits: [(embedding, content_id), ...] — what was bucketed from the Commons.
        Returns observable stats (§6). Asserts the transient-cleanup invariant.
        """
        baseline = len(self.graph.nodes)
        enhanced = gated_fresh = gated_cap = gated_error = 0
        novelties: List[float] = []
        enhancements: Dict[str, Dict[str, List[str]]] = {}

        for embedding, content_id in deposits:
            # §5 fail-FRESH: any gate error -> do NOT enhance (the recoverable side)
            try:
                novelty = float(self._novelty_fn(embedding))
            except Exception as exc:
                gated_error += 1
                logger.debug("novelty gate error (fail-fresh) for %s: %s", content_id, exc)
                continue
            novelties.append(novelty)

            if novelty < self.threshold:
                gated_fresh += 1                 # fresh water stays fresh (Hebbian-adequate)
                continue
            if enhanced >= self.max_enhances:
                gated_cap += 1                   # §5 hard cap — flood impossible even if mistuned
                continue

            enhancements[content_id] = self._enhance_one(embedding, content_id)
            enhanced += 1

        # §6 (Syl's addition): transient cleanup — no Commons traffic leaked into the substrate
        leaked = len(self.graph.nodes) - baseline
        assert leaked == 0, f"transient region leaked {leaked} node(s) into NG's substrate"

        stats = {
            "enhanced": enhanced,
            "gated_fresh": gated_fresh,
            "gated_cap": gated_cap,
            "gated_error": gated_error,
            "novelty_distribution": novelties,
            "baseline_nodes": baseline,
            "final_nodes": len(self.graph.nodes),
            "enhancements": enhancements,
        }
        logger.info("enhance_pulse: enhanced=%d fresh=%d cap=%d err=%d (baseline=%d intact)",
                    enhanced, gated_fresh, gated_cap, gated_error, baseline)
        return stats
