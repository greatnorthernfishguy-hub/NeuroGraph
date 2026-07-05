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
# -------------------
"""Shared organism-layer bootstrap for CC's own NeuroGraph instances.

Extracted from neurograph_rpc.py's bootstrap sequence. Not vendored (LAW 2 --
that list is fixed at 7 files); this is CC-specific integration code that
happens to live in the canonical NeuroGraph directory so both cc-ng-daemon.py
(sys.path insert) and cc_ng_host.py (same directory) can import it directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cc_ng_organism")


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
