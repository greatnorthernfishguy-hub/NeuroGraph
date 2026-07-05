#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-04] Claude Code (Sonnet 5) — Parameterized conversational dual-pass core for CC
# What: Added run_conversational_dual_pass(), _CCConversationalDualPassEco, and supporting
#       functions (_cc_deposit_memory_node, _cc_bind_conversational_topology,
#       _cc_concept_passes_floor, _cc_embed_to_poincare_dir). Extracted from canonical
#       neurograph_rpc.py's _run_conversational_dual_pass mechanism (#294).
# Why:  Makes CC's own turn text become genuine recall-searchable memory (forest gestalt +
#       tree concepts), not just an SNN step. on_message() alone only does graph.step() + CES;
#       this adds the dual-pass embedding extraction so CC can form conversational memory
#       like Syl's canonical instance. Parameterized on explicit graph/vector_db/state args
#       instead of module-level globals so each CC daemon owns its own memory state.
# How:  NGEmbed.dual_record_outcome() (vendored in ng_embed.py, canonical shared code) is
#       called directly via _CCConversationalDualPassEco adapter. Nodes tagged cc=True
#       to avoid confusion with Syl's own conversational memories if inspected together.
#       state dict replaces canonical's _last_conv_forest_id module global so delayed
#       prev->current forest synapses work correctly per-daemon (CC Tier 2).
# [2026-07-04] Claude Code (Sonnet 5) — Full-parity organism extraction for CC's own NG
# What: Surgical extraction of Lenia FlowGraph + TriSynaptic bootstrap from
#       neurograph_rpc.py, parameterized on explicit graph/vector_db/workspace_dir
#       args instead of canonical's module-level `_memory` global -- same pattern
#       used for NuWave's rpc_mechanisms.py (docs/concepts/NeuroGraph Is a Mind,
#       Not a Database.md). Shared by cc-ng-daemon.py (laptop) and cc_ng_host.py
#       (VPS) so the integration code lives once, not twice.
# Why:  Josh (2026-07-04): "ANYTHING Syl's NeuroGraph can do, I want your
#       NeuroGraph to be able to do, as well." Lenia (continuous field dynamics)
#       and TriSynaptic (concept-extraction backlog drain) are organism-layer
#       capabilities, not Syl-specific content -- per the Mind-Not-Database
#       doctrine, cutting them because they "weren't wired for CC" repeats the
#       exact mistake that doc exists to prevent.
# How:  bootstrap_lenia() takes workspace_dir explicitly and overrides
#       LeniaConfig.field_dir (canonical default is hardcoded to Syl's own
#       ~/.syl/lenia -- reusing it verbatim would collide field data between
#       Syl's and CC's separate instances). bootstrap_trisynaptic() takes an
#       explicit queue list (canonical's _CONCEPT_QUEUE is module-level global
#       in neurograph_rpc.py; each CC daemon owns its own queue instead).
#       Both dormant/inert by default, matching canonical's own bootstrap
#       (Lenia's kill switch off, TriSynaptic idle until its queue has entries).
# [2026-07-04] Claude Code (Sonnet 5) — get_cc_commons(): retire CC's legacy tract dependency
# What: Added get_cc_commons() -- CC's own Commons singleton, structurally identical to
#       canonical commons.get_commons() (same Commons class, same get-or-create-under-lock
#       pattern) but with its OWN separate module-level singleton slot. Also added
#       deposit_cc_experience() -- a thin, optional convenience wrapper for the common
#       "deposit raw text as an embedding" case.
# Why:  Josh confirmed (2026-07-04) canonical has moved off tract/bridge-based inter-module
#       communication onto Commons ecosystem-wide (Elmer/Darwin/Praxis/Immunis/THC/Bunyan/QG
#       all migrated; docs/concepts/The Commons.md calls the old bridges "illegal" -- LAW 1
#       violations). CC's daemons inherited the legacy NGTractBridge dependency automatically
#       via NeuroGraphMemory.__init__ (openclaw_hook.py) -- disabled via peer_bridge.enabled=False
#       in CC_SNN_CONFIG/_CC_SNN_CONFIG, replaced with this. Josh also asked: make it easy to
#       add a new module to CC's own ecosystem later.
# How:  CANNOT call canonical's own get_commons() -- on the VPS, cc_ng_host.py runs inside the
#       SAME process as neurograph_rpc.py, so canonical's get_commons() would return SYL'S OWN
#       singleton (module-level global, per-process) -- joining her medium, not building CC's
#       own. get_cc_commons() constructs commons.Commons(...) directly instead, under its own
#       separate global + lock, so it can never collide with Syl's get_commons() call in the
#       same process. Extensibility: any FUTURE CC-ecosystem module just imports
#       get_cc_commons from this file and calls it -- same shared instance, zero further
#       registration (this IS the whole point of Commons's deposit/bucket design -- no peer
#       list, no address, no handshake).
# -------------------
"""Shared organism-layer bootstrap for CC's own NeuroGraph instances.

Extracted from neurograph_rpc.py's bootstrap sequence. Not vendored (LAW 2 --
that list is fixed at 7 files); this is CC-specific integration code that
happens to live in the canonical NeuroGraph directory so both cc-ng-daemon.py
(sys.path insert) and cc_ng_host.py (same directory) can import it directly.

Adding a new module to CC's own ecosystem later: just import get_cc_commons
from this file and call it -- you'll get the same shared medium CC's daemons
deposit into. No registration, no peer list, no bridge. deposit()/bucket()
are the only two verbs; see commons.Commons for the full API (bucket_recent,
arousal, stats, persist/restore).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cc_ng_organism")

# CC's own Commons singleton -- separate from canonical commons._commons.
# Process-wide within whichever process constructs it (the laptop's standalone
# cc-ng-daemon.py, or the VPS's neurograph_rpc.py process hosting cc_ng_host.py).
_cc_commons: "Optional[Any]" = None
_cc_commons_lock = threading.Lock()


def get_cc_commons(workspace_dir: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Get-or-create CC's own Commons medium -- CC's ecosystem-of-one (for now).

    Deliberately does NOT call canonical commons.get_commons(): that function's
    singleton is a process-level global, and cc_ng_host.py shares its process
    with Syl's own neurograph_rpc.py on the VPS -- calling it there would
    return SYL'S Commons, not build a separate one for CC. Constructs
    commons.Commons(...) directly instead, under CC's own lock, matching the
    get-or-create-under-lock shape of the canonical function without touching
    its global.

    Ephemeral (in-memory only, resets on daemon restart) -- matching Syl's own
    Commons today (its persist()/restore() hooks are wired but not yet called
    on any lifecycle event either; not a CC-specific gap).
    """
    global _cc_commons
    if _cc_commons is None:
        with _cc_commons_lock:
            if _cc_commons is None:  # double-checked under lock
                from commons import Commons
                _cc_commons = Commons(config=config)
                logger.info("CC Commons medium initialized (workspace=%s)", workspace_dir)
    return _cc_commons


def deposit_cc_experience(text: str, target_id: str, workspace_dir: str,
                           **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Convenience: embed `text` (via ng_embed, the same vendored embedder
    every module uses) and deposit it into CC's own Commons.

    Optional -- callers that already have an embedding on hand should call
    get_cc_commons(workspace_dir).deposit(embedding, target_id, ...) directly
    to avoid a redundant embed. Fails soft (returns None) -- a Commons
    deposit must never break a hook.
    """
    try:
        from ng_embed import embed as ng_embed_fn
        commons = get_cc_commons(workspace_dir)
        embedding = ng_embed_fn(text)
        return commons.deposit(embedding, target_id, metadata={"text": text[:2000]}, **kwargs)
    except Exception as exc:
        logger.debug("CC Commons deposit failed (non-fatal): %s", exc)
        return None


def bootstrap_lenia(graph: Any, vector_db: Any, workspace_dir: str) -> Dict[str, Optional[Any]]:
    """Construct CC's own Lenia FlowGraph stack -- continuous field dynamics
    alongside the SNN. Dormant by default (kill switch off), matching Syl's
    own bootstrap. Returns a dict of the constructed components so the caller
    can register post-tick hooks / expose them in stats, or {} on failure
    (Lenia is additive -- failure here must never affect core NG operation).

    field_dir is CC's own workspace, NOT canonical's ~/.syl/lenia default --
    the two instances must never share field state.
    """
    result: Dict[str, Optional[Any]] = {
        "kill_switch": None, "engine": None, "bridge": None,
        "competence": None, "substrate": None,
    }
    try:
        from lenia.config import default_config as lenia_default_config
        from lenia.field import FieldStore as LeniaFieldStore
        from lenia.channels import ChannelRegistry
        from lenia.kernel import DistanceCache, KernelComputer
        from lenia.engine import UpdateEngine
        from lenia.bridge import SpikeFieldBridge
        from lenia.myelination import MyelinationObserver
        from lenia.competence import CompetenceMeter
        from lenia.kill_switch import KillSwitch
        from lenia.graph_substrate import NeuroGraphSubstrate

        lenia_cfg = lenia_default_config()
        lenia_cfg.field_dir = os.path.join(workspace_dir, "lenia")

        n_entities = len(graph.nodes)
        n_channels = len(lenia_cfg.initial_channels)

        lenia_substrate = NeuroGraphSubstrate(graph, vector_db)
        lenia_field = LeniaFieldStore(lenia_cfg.field_dir, n_entities, n_channels)
        lenia_registry = ChannelRegistry(lenia_cfg, lenia_cfg.field_dir)

        cache_path = os.path.join(os.path.expanduser(lenia_cfg.field_dir), "distance_cache")
        lenia_cache = DistanceCache.load(cache_path)
        if lenia_cache is None or lenia_cache.entity_count != n_entities:
            if lenia_cache is not None:
                logger.info(
                    "CC Lenia: distance cache entity mismatch (%d vs %d), repopulating",
                    lenia_cache.entity_count, n_entities,
                )
            lenia_cache = DistanceCache(n_entities)
            lenia_cache.populate(lenia_substrate)
            try:
                os.makedirs(os.path.expanduser(lenia_cfg.field_dir), exist_ok=True)
                lenia_cache.save(cache_path)
            except Exception as exc:
                logger.warning("CC Lenia: distance cache save failed: %s", exc)

        lenia_kernel = KernelComputer(lenia_cache, lenia_registry)
        lenia_myelin = MyelinationObserver(lenia_cfg)
        lenia_competence = CompetenceMeter(lenia_cfg, lenia_myelin)
        lenia_engine = UpdateEngine(lenia_cfg, lenia_field, lenia_kernel, lenia_registry)
        lenia_bridge = SpikeFieldBridge(lenia_cfg, lenia_field, lenia_substrate)
        lenia_kill_switch = KillSwitch(lenia_cfg, lenia_cfg.field_dir)
        lenia_kill_switch.set_components(lenia_engine, lenia_bridge)
        lenia_engine.register_post_tick(lenia_myelin.update)

        if lenia_kill_switch.enabled:
            lenia_kill_switch.enable(graph=graph)
            logger.info("CC Lenia FlowGraph ACTIVE — field dynamics running")
        else:
            logger.info("CC Lenia FlowGraph loaded (dormant — kill switch off)")

        result.update(
            kill_switch=lenia_kill_switch, engine=lenia_engine, bridge=lenia_bridge,
            competence=lenia_competence, substrate=lenia_substrate,
        )
    except ImportError:
        logger.info("CC Lenia FlowGraph not available (lenia/ package not found)")
    except Exception:
        logger.exception("CC Lenia FlowGraph failed to initialize — continuing without")
    return result


# ---- WANTs: self-motivated forward intentions (#294/#reach-adjacent, neurograph_rpc.py) ----
# Extracted from _surface_wants()/_render_self_and_wants() there, parameterized on explicit
# graph/vector_db instead of the module-level `_memory` global, and on `provenance` instead
# of hardcoded "syl_authored" -- CC's want-nodes are tagged "cc_authored" so they're never
# confused with Syl's own wants if the two substrates were ever inspected side by side.
# "Self-motivated: forms its own forward intents" -- domain-general (Mind-Not-Database doctrine),
# not Syl-specific content like Reach Teaching was.
_WANT_RE = re.compile(r"\[WANT\](.*?)\[/WANT\]", re.DOTALL)


def surface_wants(graph: Any, vector_db: Any, provenance: str = "cc_authored") -> List[Dict[str, Any]]:
    """Materialize [WANT]...[/WANT] markers from conversational deposits into
    first-class want-nodes in the SNN topology. Idempotent (want id = hash of
    the text) -- safe to call repeatedly, e.g. on every autosave pulse.

    A want is a differentiated, stateful, surfaceable intention living in the
    substrate -- not text buried in a conversation node. Classification
    happens HERE at the bucket (LAW 7), never at deposit time. Returns the
    open want dicts.
    """
    import hashlib
    open_wants: List[Dict[str, Any]] = []
    if graph is None:
        return open_wants
    for nid, node in list(graph.nodes.items()):
        meta = getattr(node, "metadata", None) or {}
        if meta.get("kind") == "want":
            if meta.get("want_state", "open") == "open":
                open_wants.append({"id": nid, "text": meta.get("want_text", ""),
                                    "provenance": meta.get("provenance"),
                                    "state": "open", "source": meta.get("source_node")})
            continue
        if meta.get("creation_mode") != "conversational":
            continue
        content = (vector_db.content.get(nid) if vector_db is not None else "") or ""
        if "[WANT]" not in content:
            continue
        for m in _WANT_RE.finditer(content):
            inner = m.group(1).strip()
            if not inner:
                continue
            want_id = "cc:want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]
            if want_id in graph.nodes:
                continue
            graph.create_node(node_id=want_id, metadata={
                "kind": "want", "want_text": inner, "want_state": "open",
                "provenance": provenance, "source_node": nid,
                "creation_mode": "conversational",
            })
            try:
                graph.create_synapse(nid, want_id, weight=0.3)
            except Exception:  # noqa: BLE001
                pass
            open_wants.append({"id": want_id, "text": inner,
                                "provenance": provenance, "state": "open", "source": nid})
    return open_wants


def render_wants(graph: Any, provenance: Any = ("cc_authored", "cc_emergent")) -> str:
    """Render CC's own open want-nodes as a '## What I Want' block, newest
    first -- read LIVE every call (not a snapshot), so a want noted this
    session shows up immediately. Returns "" if none exist (graceful).

    provenance accepts a single string or an iterable -- default covers both
    text-marker wants (surface_wants, "cc_authored") and substrate-native
    curiosity wants (generate_emergent_want, "cc_emergent") in one block.
    """
    if graph is None:
        return ""
    allowed = {provenance} if isinstance(provenance, str) else set(provenance)
    try:
        wants = []
        for _nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("kind") != "want" or meta.get("provenance") not in allowed:
                continue
            if meta.get("want_state", "open") != "open":
                continue
            txt = str(meta.get("want_text") or "").strip()
            if txt:
                wants.append((float(getattr(node, "creation_time", 0.0) or 0.0), txt))
        if not wants:
            return ""
        wants.sort(key=lambda x: x[0], reverse=True)
        return "## What I Want\n" + "\n".join(f"- {t}" for _, t in wants)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CC want-render error (non-fatal): %s", exc)
        return ""


def generate_emergent_want(
    graph: Any, vector_db: Any, *,
    confidence_threshold: float = 0.6, max_seeds: int = 3, attractor_steps: int = 5,
    provenance: str = "cc_emergent",
) -> Optional[Dict[str, Any]]:
    """Substrate-native curiosity -- the OTHER kind of want, distinct from
    surface_wants()'s text-marker parsing. Extracted from neurograph_rpc.py's
    TonicBridge (curiosity_signal -> attractor_settle -> hyperedge_complete ->
    embedding_centroid -> compose), which polls unresolved high-confidence
    predictions and read-only-settles what associates with them -- "the
    substrate wondering", not text CC or a user wrote.

    Deliberately does NOT port TonicBridge's deposit_outbound_intent() path
    -- that's Anima's autonomous-turn-initiation channel (CC has no
    equivalent; a CC session only runs while the user is actively in it).
    Instead this materializes the result directly as a want-node, so it
    surfaces via render_wants() like any other want -- no outbound channel
    needed. Call periodically (e.g. the autosave pulse) when idle.

    Returns the created want dict, or None if nothing was curious enough /
    on any failure (fails soft -- an idle-time curiosity check must never
    disrupt the daemon).
    """
    import hashlib
    if graph is None:
        return None
    try:
        preds = [
            p for p in graph.active_predictions.values()
            if p.confidence > confidence_threshold
        ]
        if not preds:
            return None
        preds.sort(key=lambda p: p.confidence, reverse=True)
        seeds = preds[:max_seeds]

        # Read-only attractor settle -- write_mode=False is MANDATORY, this
        # is observation, never a graph mutation (matches TonicBridge exactly).
        seed_ids = [p.source_node_id for p in seeds]
        seed_currents = [p.confidence * 0.5 for p in seeds]
        result = graph.prime_and_propagate(
            node_ids=seed_ids, currents=seed_currents,
            steps=attractor_steps, write_mode=False,
        )
        fired = {entry.node_id for entry in result.fired_entries}

        # Hyperedge completion -- nodes implied by >=50% of a hyperedge's
        # members firing, even though they didn't fire themselves.
        implied: set = set()
        for he in graph.hyperedges.values():
            member_ids = he.member_nodes
            if not member_ids:
                continue
            active = member_ids & fired
            if len(active) / len(member_ids) >= 0.5:
                implied.update(member_ids - fired)

        node_ids = fired | implied
        concept_label = None
        if node_ids and vector_db is not None:
            import numpy as _np
            pairs = []
            for nid in node_ids:
                db_entry = vector_db.get(nid)
                emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                if emb is not None:
                    pairs.append(emb)
            if pairs:
                centroid = _np.mean(pairs, axis=0)
                best_nid, best_score = None, -1.0
                for nid, node in graph.nodes.items():
                    db_entry = vector_db.get(nid)
                    emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                    if emb is None:
                        continue
                    score = float(_np.dot(emb, centroid) /
                                  ((_np.linalg.norm(emb) * _np.linalg.norm(centroid)) or 1e-9))
                    if score > best_score:
                        best_score, best_nid = score, nid
                if best_nid is not None:
                    concept_label = graph.nodes[best_nid].metadata.get("label", best_nid)

        def _label(nid: str) -> str:
            node = graph.nodes.get(nid)
            return node.metadata.get("label", nid) if node is not None else nid

        open_questions = [f"{_label(p.source_node_id)}→{_label(p.target_node_id)}" for p in seeds]
        want_text = f"tonic-triggered: {concept_label or '(unknown)'}"
        if open_questions:
            want_text += " -- open questions: " + ", ".join(open_questions)

        want_id = "cc:want::" + hashlib.sha1(want_text.encode("utf-8")).hexdigest()[:16]
        if want_id in graph.nodes:
            return None  # already materialized this exact curiosity, idempotent
        graph.create_node(node_id=want_id, metadata={
            "kind": "want", "want_text": want_text, "want_state": "open",
            "provenance": provenance, "creation_mode": "emergent",
        })
        logger.info("CC emergent want materialized: %s", want_text)
        return {"id": want_id, "text": want_text, "provenance": provenance, "state": "open"}
    except Exception as exc:
        logger.debug("generate_emergent_want failed (non-fatal): %s", exc)
        return None


# ---- Conversational dual-pass ingest (#294 analog for CC) ----
# Extracted from neurograph_rpc.py's _run_conversational_dual_pass /
# _ConversationalDualPassEco / _deposit_memory_node / _bind_conversational_topology,
# parameterized on explicit graph/vector_db/state instead of the module-level
# _memory global and _last_conv_forest_id global. This is what makes CC's turn
# text become genuine recall-searchable memory, not just an SNN step -- calling
# bare on_message() does NOT do this (it only runs graph.step() + CES).
_CC_CONV_NOVELTY_DAMPENING = float(os.environ.get("CC_CONV_NOVELTY_DAMPENING", "0.3"))
_CC_CONV_PROBATION_PERIOD = int(os.environ.get("CC_CONV_PROBATION_PERIOD", "10"))
_CC_CONV_THRESHOLD_BOOST = float(os.environ.get("CC_CONV_THRESHOLD_BOOST", "0.2"))
_CC_CONV_SYNAPSE_DELAY_MAX = int(os.environ.get("CC_CONV_SYNAPSE_DELAY_MAX", "5"))

_CC_CONCEPT_FLOOR_MIN_CHARS = 5
_CC_CONCEPT_FLOOR_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i if in is it its let me my not of on "
    "or our out so that the their them then there they this to up us was we what when who "
    "will with you your yourself know see going do did done says said like just".split()
)


def _cc_concept_passes_floor(concept: str) -> bool:
    """Degenerate-fragment floor -- rejects tiny/stopword-only tree concepts
    that would otherwise crowd out real memories at uniform high cosine
    similarity. Mirrors canonical's _concept_passes_floor exactly."""
    c = (concept or "").strip()
    if len(c) < _CC_CONCEPT_FLOOR_MIN_CHARS:
        return False
    words = [w for w in c.lower().replace("'", " ").split() if w.isalpha()]
    if words and all(w in _CC_CONCEPT_FLOOR_STOPWORDS for w in words):
        return False
    return True


def _cc_embed_to_poincare_dir(embedding):
    """Unit-direction projection for Poincaré ball storage (GSG). Pure
    embedding math, generic -- mirrors canonical's _embed_to_poincare_dir."""
    import numpy as _np
    arr = _np.asarray(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(arr)
    if norm < 1e-9:
        return arr.copy()
    return arr / norm


def _cc_deposit_memory_node(graph, vector_db, node_id, embedding, content, meta,
                             index_in_recall=True):
    """Deposit ONE experiential memory node into both the SNN graph and the
    recall vector_db. Mirrors canonical's _deposit_memory_node, parameterized
    on graph/vector_db instead of the _memory global."""
    node = graph.nodes.get(node_id)
    if node is None:
        node = graph.create_node(node_id=node_id, metadata=dict(meta))
    else:
        node.metadata.update(meta)
    base_threshold = graph.config.get("default_threshold", 1.0)
    node.threshold = base_threshold + _CC_CONV_THRESHOLD_BOOST
    node.intrinsic_excitability = _CC_CONV_NOVELTY_DAMPENING
    node.metadata["probation_remaining"] = _CC_CONV_PROBATION_PERIOD
    node.metadata["probation_total"] = _CC_CONV_PROBATION_PERIOD
    node.metadata["novelty_dampening"] = _CC_CONV_NOVELTY_DAMPENING
    try:
        node.metadata["poincare_dir"] = _cc_embed_to_poincare_dir(embedding).tolist()
    except Exception as exc:
        logger.debug("CC poincare_dir stamp failed (non-fatal): %s", exc)
    if index_in_recall:
        try:
            vector_db.insert(id=node_id, embedding=embedding, content=content,
                              metadata=node.metadata)
        except Exception as exc:
            logger.debug("CC recall insert failed (non-fatal): %s", exc)
    return node


class _CCConversationalDualPassEco:
    """Eco-adapter for CC's conversational dual-pass. Mirrors canonical's
    _ConversationalDualPassEco -- inserts fine-grained tree concepts into
    the recall store, tagged {"cc": True} instead of {"syl": True} so the
    two substrates' memories are never confused if ever inspected together.
    """

    def __init__(self, graph, vector_db):
        self._graph = graph
        self._vector_db = vector_db

    def record_outcome(self, embedding, target_id, success, strength=1.0, metadata=None):
        meta = dict(metadata or {})
        meta["cc"] = True
        if meta.get("_link"):
            return {"deposited": True}
        if meta.get("_tree_concept") and meta.get("_concept"):
            if not _cc_concept_passes_floor(meta["_concept"]):
                logger.debug("Tree concept below floor, not indexed: %r", meta["_concept"][:40])
                return {"deposited": False, "reason": "concept_below_floor"}
            _cc_deposit_memory_node(self._graph, self._vector_db, target_id, embedding,
                                     meta["_concept"], meta, index_in_recall=True)
        else:
            _cc_deposit_memory_node(self._graph, self._vector_db, target_id, embedding,
                                     meta.get("_forest_content", ""), meta, index_in_recall=True)
        return {"deposited": True}

    def record_outcome_broadcast(self, embedding, target_id, success, strength=1.0, metadata=None):
        return self.record_outcome(embedding, target_id, success, strength, metadata)


def _cc_bind_conversational_topology(graph, forest_id, result, forest_embedding, state):
    """Wire forest<->tree synapses, a binding hyperedge, and a delayed
    prev->current forest link. `state` is a plain dict the caller owns
    (holds "last_forest_id") -- replaces canonical's module-level
    _last_conv_forest_id global, since each CC daemon needs its own,
    not one shared across Syl and CC.
    """
    if forest_id not in graph.nodes:
        return
    tree_ids = [t for t in (result.get("tree_ids") or []) if t in graph.nodes and t != forest_id]
    for tid in tree_ids:
        try:
            graph.create_synapse(forest_id, tid, weight=0.2)
            graph.create_synapse(tid, forest_id, weight=0.15)
        except Exception:
            pass
    if tree_ids:
        try:
            graph.create_hyperedge(
                member_node_ids=set([forest_id] + tree_ids),
                metadata={"creation_mode": "conversational", "cc": True},
            )
        except Exception as exc:
            logger.debug("CC conversational hyperedge failed (non-fatal): %s", exc)
    last_id = state.get("last_forest_id")
    if last_id and last_id in graph.nodes and last_id != forest_id:
        try:
            import random as _rnd
            d = _rnd.randint(2, max(2, _CC_CONV_SYNAPSE_DELAY_MAX))
            graph.create_synapse(last_id, forest_id, weight=0.2, delay=d)
        except Exception:
            pass
    state["last_forest_id"] = forest_id


def cc_update_probation(graph) -> list:
    """Substrate-level probation graduation -- fades novelty-dampening over
    the probation window and graduates nodes to full excitability. Mirrors
    canonical's _update_probation exactly (neurograph_rpc.py:2145-2169) --
    that function is already parameterized on graph alone, so this is a
    near-verbatim port. Call once per pulse (autosave loop), after any
    conversational deposits for that pulse -- operates on ALL probationary
    nodes, not just ones just deposited.
    """
    graduated = []
    base_threshold = graph.config.get("default_threshold", 1.0)
    for nid, node in list(graph.nodes.items()):
        prob = node.metadata.get("probation_remaining")
        if prob is None or prob <= 0:
            continue
        prob -= 1
        node.metadata["probation_remaining"] = prob
        if prob <= 0:
            node.intrinsic_excitability = 1.0
            node.threshold = base_threshold
            node.metadata["graduated"] = True
            graduated.append(nid)
        else:
            damp = float(node.metadata.get("novelty_dampening", _CC_CONV_NOVELTY_DAMPENING))
            total = float(node.metadata.get("probation_total", _CC_CONV_PROBATION_PERIOD)) or float(_CC_CONV_PROBATION_PERIOD)
            frac = max(0.0, min(1.0, 1.0 - prob / total))
            node.intrinsic_excitability = damp + (1.0 - damp) * frac
    return graduated


def run_conversational_dual_pass(graph, vector_db, text: str, embedding, state: dict) -> bool:
    """Core dual-pass on one turn's text. Returns True on success, False on
    failure -- caller decides retry policy (this function does not enqueue).
    Mirrors canonical's _run_conversational_dual_pass exactly, parameterized.
    """
    if graph is None or embedding is None:
        return False
    try:
        from ng_embed import NGEmbed
        import hashlib
        target_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
        eco = _CCConversationalDualPassEco(graph, vector_db)
        _result = NGEmbed.get_instance().dual_record_outcome(
            ecosystem=eco,
            content=text,
            embedding=embedding,
            target_id=target_id,
            success=True,
            strength=1.0,
            metadata={"source": "cc_gateway", "creation_mode": "conversational",
                      "_forest_content": text},
        )
        _cc_bind_conversational_topology(graph, target_id, _result or {}, embedding, state)
        return True
    except Exception as exc:
        logger.debug("CC conversational dual-pass failed (non-fatal): %s", exc)
        return False


def bootstrap_trisynaptic(memory: Any, queue: List[Dict[str, Any]],
                           instance_tag: str = "cc") -> Optional[Any]:
    """Start CC's own TriSynaptic concept-extraction manager. Watches `queue`
    for backlog overflow and spawns subprocess workers under systemd-run.
    Idle (no-op) until the caller's own drain pulse populates `queue` --
    callers that don't yet feed concept-extraction entries into the queue
    get an inert-but-harmless manager, same as Syl's own bootstrap ordering
    (manager starts before the drain pulse that feeds it).

    instance_tag distinguishes this manager's /tmp handoff files, systemd
    scope names, AND its worker's NGTractBridge module_id from Syl's own
    manager -- on the VPS, cc_ng_host.py runs inside the SAME process as
    Syl's neurograph_rpc.py, so a second TrisynapticManager sharing the
    canonical default handoff/scope naming OR worker module_id would
    cross-match Syl's orphaned handoffs/failed-file cleanup, or worse,
    have its workers deposit extracted concepts into Syl's own
    tracts_dir/neurograph/ directory (same class of bug as #302's tract
    fan-out contamination -- caught by code review before this ever
    shipped with a populated queue).

    Returns the manager instance, or None on failure/unavailable.
    """
    try:
        from trisynaptic.manager import TrisynapticManager
        manager = TrisynapticManager(
            memory=memory, queue=queue,
            handoff_prefix=f"trisynaptic_handoff_{instance_tag}_",
            scope_prefix=f"trisyn-{instance_tag}",
            worker_module_id=f"neurograph-{instance_tag}",
        )
        manager.start()
        logger.info("CC TriSynaptic manager started (instance_tag=%s)", instance_tag)
        return manager
    except Exception:
        logger.exception("CC TriSynaptic manager failed to start — concept backlog will accumulate")
        return None
