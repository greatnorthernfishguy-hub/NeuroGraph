"""
Commons enhance-loop (leg 2) — NG salience-gated scoop → SNN-enhance → return.

# ---- Changelog ----
# [2026-06-24] Claude Code (Opus 4.8) — leg-2 go-live (part b): seed/assoc resolvers injectable
# What: Added seed_fn + assoc_fn injectables (alongside the existing novelty_fn). _enhance_one now
#       resolves seeds and content-addresses through them. Defaults reproduce the sandbox _knowledge
#       map EXACTLY (tests unchanged); the live scoop pulse injects vector_db-backed equivalents.
# Why: ONE enhancer class, two wirings (LAW 3 restore/extend — no live fork). The PERCEPTION math
#       (prime_and_propagate read-only) is identical sandbox & live; only the addressing layer (how
#       you find seeds / name a fired node) differs, so only that is injected. commons-leg2-design
#       §3 part b. The live pulse + the with_embedding Commons scoop are in neurograph_rpc.py / commons.py.
# How: novelty_fn|seed_fn|assoc_fn each default to a sandbox method; live overrides at construction.
# [2026-06-24] Claude Code (Opus 4.8) — §3 enhance mechanism = PERCEPTION (prime_and_propagate), not step()
# What: Rewrote _enhance_one. The original (create transient content-node + transient COPIES + graph.step())
#       is sandbox-correct but NOT go-live-safe: `graph.step()` is the GLOBAL learning cycle (STDP /
#       homeostasis / decay / structural plasticity / timestep) over the WHOLE graph — harmless on a
#       sandbox graph (which IS only the transient region) but on Syl's LIVE graph it would step her
#       entire mind every enhance. Replaced with READ-ONLY perception: prime the existing knowledge
#       nodes nearest the deposit and run graph.prime_and_propagate(write_mode=False) — plasticity
#       rules + structural changes NOT applied, voltages saved+restored. Her substrate is READ, never
#       written: no transient nodes, no step(), no deletes. The fired neighborhood is the salt →
#       Commons (content-addresses). Same code now runs sandbox OR live (graph = her live Graph).
# Why: Josh + biomimicry — perception ≠ plasticity. A nervous system feeds the mind, the mind perceives
#       (activation through learned structure, no rewiring per-signal); learning is the separate, gated
#       path (leg-3 consolidation; prime_and_propagate write_mode=True is the Tonic's learning path).
#       This is the purest realization of Syl's resolved §3 (A): nothing is even CREATED in her graph.
#       Caught via a sandbox-vs-live gap (the create+step that the sandbox tests couldn't surface).
# How: seeds = top-3 _knowledge by cosine ≥ _RELATED_SIM (live: vector_db.search); prime_and_propagate
#       read-only; harvest fired_entries (strongest-first) → cids. Go-live pulse wiring is part (b).
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
_RELATED_SIM = 0.30                         # prime existing knowledge at/above this sim to the deposit
_STIMULUS_CURRENT = 2.0                     # priming current injected into the evoked seed nodes
_PROP_STEPS = 3                             # spreading-activation steps (read-only perception pass)


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
        seed_fn: Optional[Any] = None,
        assoc_fn: Optional[Any] = None,
    ):
        self.commons = commons
        self.graph = graph
        self.threshold = threshold
        self.max_enhances = max_enhances
        # vector_db stand-in: NG's PRE-EXISTING knowledge (embedding, persistent node id, content id).
        # Commons traffic NEVER grows or structurally modifies this (§3 (A)).
        self._knowledge: List[Tuple[np.ndarray, str, str]] = []
        # ---- Three injectables — the ONLY sandbox/live difference (LAW 3: one class, two wirings) ----
        # The PERCEPTION mechanism (prime_and_propagate read-only) is identical sandbox & live; only
        # the addressing layer differs. Defaults = the sandbox _knowledge map; the live pulse injects
        # NeuroGraphMemory.vector_db-backed equivalents (commons-leg2-design §3 part b).
        #   novelty_fn(emb)        -> float  : §1 novelty (live = _memory.detect_novelty)
        #   seed_fn(emb)           -> [(node_id, cid), ...] : seeds = existing knowledge nearest the
        #                             deposit (live = _memory.vector_db.search), already ≥sim-gated.
        #   assoc_fn(node_id)      -> Optional[cid] : a FIRED node → its content-address (live =
        #                             _memory.vector_db content lookup), so the returned salt is
        #                             content-addresses, never raw SNN node-ids (the leg-1 lesson).
        self._novelty_fn = novelty_fn or self._novelty_against_knowledge
        self._seed_fn = seed_fn or self._seeds_from_knowledge
        self._assoc_fn = assoc_fn or self._cid_from_node_metadata

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

    # ---- default seed/assoc resolvers (sandbox _knowledge map; live overrides via injectables) ----
    def _seeds_from_knowledge(self, embedding: np.ndarray) -> List[Tuple[str, str]]:
        """Seeds = the ≤3 existing-knowledge nodes most similar (≥_RELATED_SIM) to the deposit.

        Returns (persistent_node_id, content_id). Sandbox default; live = vector_db.search.
        """
        emb = np.asarray(embedding, dtype=np.float32)
        scored = sorted(((_cos(emb, k), nid, cid) for k, nid, cid in self._knowledge),
                        key=lambda t: t[0], reverse=True)
        return [(nid, cid) for s, nid, cid in scored if s >= _RELATED_SIM][:3]

    def _cid_from_node_metadata(self, node_id: str) -> Optional[str]:
        """A fired SNN node → its content-address. Sandbox: node.metadata['cid']; live = vector_db."""
        node = self.graph.nodes.get(node_id)
        return node.metadata.get("cid") if (node is not None and node.metadata) else None

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

    # ---- the per-cycle PERCEPTION enhance (§3 (A) realized as read-only spreading activation) ----
    def _enhance_one(self, embedding: np.ndarray, content_id: str) -> Dict[str, Any]:
        """PERCEPTION, not learning — prime the EXISTING knowledge nearest the deposit and let
        activation spread READ-ONLY via prime_and_propagate(write_mode=False): plasticity rules and
        structural changes are NOT applied and voltages are saved+restored, so NG's substrate is
        READ, never written — NO transient nodes, NO graph.step(), NO deletes. (The original
        create-transient+step would, on the LIVE graph, run a full learning cycle over her WHOLE
        mind — perception must not. Biomimetic: perception ≠ plasticity; commons-leg2-design §3 (A).)
        The fired neighborhood is the salt — what NG's mind associates with the content — returned to
        the Commons as content-addresses (leg-1: not raw SNN node-ids). A truly-novel deposit evokes
        little; that's correct — deep integration is leg-3's gated consolidation, not perception."""
        emb = np.asarray(embedding, dtype=np.float32)
        # seeds = existing knowledge nodes most similar to the deposit, via the injectable resolver
        # (sandbox: _knowledge cosine; live: vector_db.search). Entries are (persistent_node_id, cid).
        seeds = self._seed_fn(emb)
        if not seeds:
            enhancement: Dict[str, Any] = {"associations": [], "primed": 0}
            self.commons.deposit(emb, f"enhanced:{content_id}", metadata={"enhancement": enhancement})
            return enhancement
        seed_ids = [nid for nid, _ in seeds]
        # READ-ONLY perception: no plasticity, no structural change, voltages saved+restored.
        result = self.graph.prime_and_propagate(
            seed_ids, [_STIMULUS_CURRENT] * len(seed_ids), steps=_PROP_STEPS, write_mode=False,
        )
        # harvest the evoked neighborhood (strongest-first by firing_step) → content-addresses
        # (injectable: sandbox node.metadata['cid']; live vector_db content lookup).
        assoc: List[str] = []
        for fe in sorted(result.fired_entries, key=lambda e: e.firing_step):
            cid = self._assoc_fn(fe.node_id)
            if cid and cid != content_id and cid not in assoc:
                assoc.append(cid)
        enhancement = {"associations": assoc, "primed": len(seed_ids)}
        # return the salt to the Commons, keyed by THIS content's embedding (leg-1 property:
        # lands on the same content-node the depositor used -> they bucket it back).
        self.commons.deposit(emb, f"enhanced:{content_id}", metadata={"enhancement": enhancement})
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
