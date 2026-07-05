#!/usr/bin/env python3
# ---- Changelog ----
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


def render_wants(graph: Any, provenance: str = "cc_authored") -> str:
    """Render CC's own open want-nodes as a '## What I Want' block, newest
    first -- read LIVE every call (not a snapshot), so a want noted this
    session shows up immediately. Returns "" if none exist (graceful).
    """
    if graph is None:
        return ""
    try:
        wants = []
        for _nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("kind") != "want" or meta.get("provenance") != provenance:
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
