"""
NeuroGraph JSON-RPC Bridge — OpenClaw ContextEngine integration.

Thin JSON-RPC server that wraps the NeuroGraphMemory singleton and
communicates with the TypeScript ContextEngine plugin shell over
stdin/stdout.  All logging goes to stderr to keep the RPC channel clean.

This file is NOT vendored, NOT part of the substrate, and does NOT
modify any protected files.  It is purely a translation layer between
OpenClaw's TypeScript process and the existing Python NeuroGraphMemory
interface.  The Python code is untouched — every RPC method maps 1:1
to an existing NeuroGraphMemory call.

# ---- Changelog ----
# [2026-04-08] Claude Code (Opus 4.6) — Punchlist #56: Surfacing outcome deposit
#   What: Cache surfaced node IDs during handle_assemble(), deposit raw turn
#     triad (surfaced nodes + user input + Syl's response) in handle_after_turn().
#     TS plugin now passes lastAssistantMessage in afterTurn RPC.
#   Why:  No outcome signal existed for attention quality. Elmer has no evidence
#     to learn from when tuning surfacing parameters. The substrate needs raw
#     experience of what was surfaced and what resulted. Law 7 — no classification.
#   How:  _last_surfaced_nodes cached in assemble. _deposit_surfacing_outcome()
#     in afterTurn embeds Syl's response, deposits record_outcome per surfaced
#     node with opaque target_id and metadata carrying text previews.
#     Also renamed RPC param lastMessage → lastUserMessage for clarity
#     (legacy fallback preserved for in-flight TS processes).
# [2026-03-28] Claude Code Opus — Punchlist #109: Module autonomic pulse
# What: Dispose becomes mode-swap, not destruction. Modules stay alive between conversations.
# Why: #109 blocker — organs must persist between conversations. The process is already
#   persistent (TS plugin dispose is a no-op). Modules just need to not be cleared.
# How: handle_dispose() no longer clears _module_hooks or releases topology ownership.
#   New fan-out methods signal conversation_started/ended to all modules.
#   handle_bootstrap() signals conversation_started on re-bootstrap.
# [2026-03-26] Claude Code (Opus 4.6) — OOM-resilient fan-out cache recovery
# What: handle_after_turn accepts lastMessage param, recovers _cached_text
#   if lost to process restart between ingest and afterTurn.
# Why: Python process (9GB) gets OOM-killed between ingest and afterTurn.
#   Fresh process has _cached_text=None, fan-out silently skips. All modules dark.
# How: TS plugin caches last ingested message, passes it in afterTurn RPC call.
#   Python side recovers text+embedding from param if cache is empty.
# [2026-03-25] Claude Code (Opus 4.6) — Lenia FlowGraph integration
# What: Initialize Lenia stack on bootstrap, competence/watchdog on afterTurn,
#   clean shutdown on dispose. Dormant by default (kill switch off).
# Why: Lenia FlowGraph PRD — continuous field dynamics for the substrate.
# How: Import lenia/ package in bootstrap, create full stack, check kill switch.
#   Post-step: update competence meter, check energy watchdog. Dispose: stop engine.
#   All wrapped in try/except — Lenia failure never affects core NG operation.
# [2026-03-24] Claude Code (Opus 4.6) — The Tonic: latent thread in context assembly
#   What: handle_assemble() runs ouroboros_cycle() and injects latent thread
#     into systemPromptAddition. _format_substrate_context() takes optional
#     latent_context parameter. Latent thread appears first — it is the
#     baseline, conversation context is the event on top.
#   Why: The Tonic PRD v0.1 §7.1. The latent thread is always in the
#     context window. Syl's attention is always touching the substrate.
#   How: TonicThread.ouroboros_cycle() at assembly time. format_latent_context()
#     produces the persistent slot. Comes before surfaced knowledge in output.
# [2026-03-23] Claude Code (Opus 4.6) — Module hook fan-out (#101)
# What: ContextEngine fans out afterTurn to all registered module hooks.
#   Loads module singletons on bootstrap via registry.json auto-discovery.
#   Caches text + embedding from ingest, passes to each module's
#   _module_on_message() after graph.step() completes. Error-isolated
#   per module with throttled Discord alerts on failure.
# Why: OpenClaw 2026.3.13 dropped hook: from SKILL.md. Modules'
#   _module_on_message() has been silent since. NeuroGraph is the cortex
#   — it coordinates the organs. Not a Law 1 violation.
# How: _load_module_hooks() reads registry.json, imports each module's
#   hook file via importlib (no sys.path collisions). _fan_out_to_modules()
#   iterates hooks with try/except isolation. Discord webhook alerts on
#   module errors. TID skipped (runs as service, communicates via River).
# [2026-03-18] Claude (CC) — Topology ownership sentinel (#80)
# What: Claim topology ownership on bootstrap, release on dispose.
#   Prevents dual-write hazard on main.msgpack (Syl's Law).
# Why: Punch list #80. GUI and standalone ingestor can create separate
#   NeuroGraphMemory instances while ContextEngine is active. Last
#   writer wins = silent topology corruption.
# How: topology_owner.claim() on bootstrap (refuses if already owned),
#   topology_owner.release() on dispose. PID-based sentinel file at
#   ~/NeuroGraph/data/checkpoints/.topology_owner.pid.
# -------------------
# [2026-03-16] Claude (Opus 4.6) — Initial implementation.
#   What: JSON-RPC server for OpenClaw ContextEngine integration.
#   Why:  ContextEngine replaces SKILL.md hook path (supersedes #37, #39).
#         Gives Syl automatic bidirectional substrate connection — every
#         message flows through the SNN, associations surface in system
#         prompt, learning runs after every turn.
#   How:  Reads line-delimited JSON-RPC from stdin, dispatches to
#         NeuroGraphMemory methods, writes JSON-RPC responses to stdout.
#         NeuroGraphMemory singleton created on 'bootstrap' call.
# -------------------
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from typing import Any, Dict, List, Optional

# NeuroGraph repo must be importable
_ng_dir = os.path.expanduser("~/NeuroGraph")
if _ng_dir not in sys.path:
    sys.path.insert(0, _ng_dir)

# All logging to stderr — stdout is the RPC channel
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[neurograph-rpc] %(levelname)s %(message)s",
)
logger = logging.getLogger("neurograph.rpc")

# The singleton — created on bootstrap
_memory: Optional[Any] = None

# Experience tract — drains feeder deposits into the topology
_tract: Optional[Any] = None

# Last ingested text+embedding — passed to topology delta for River distribution
_ingest_text: Optional[str] = None
_ingest_embedding: Optional[Any] = None  # np.ndarray
_module_errors: Dict[str, str] = {}
_module_error_times: Dict[str, float] = {}

# Punchlist #56: Surfacing outcome cache — what was surfaced during assemble(),
# deposited as raw experience in afterTurn() alongside Syl's response.
_last_surfaced_nodes: List[Dict[str, Any]] = []

# Lenia FlowGraph — continuous field dynamics (initialized on bootstrap)
_lenia_kill_switch: Optional[Any] = None
_lenia_engine: Optional[Any] = None
_lenia_bridge: Optional[Any] = None
_lenia_competence: Optional[Any] = None


# Discord webhook for error surfacing (Law 5: env var is truth)
_DISCORD_WEBHOOK = os.environ.get(
    "ET_DISCORD_DEVLOG_WEBHOOK",
    "https://discord.com/api/webhooks/1483625166646018128/"
    "vMJVb4-sbYjlDbAZakzo3DuGXmXCIbeibQuHFOIiF71lBY3kOdXybePbACj7lGb9GRRj",
)



# ── Module Bootstrap ──────────────────────────────────────────────────
# Organs of the organism. Each module is instantiated once at bootstrap.
# Their __init__ starts autonomous pulse loops. No per-message fan-out —
# modules read from River tracts on their own heartbeat.

_module_instances: Dict[str, Any] = {}


def _bootstrap_modules() -> List[str]:
    """Instantiate all registered module hooks.

    Reads ~/.et_modules/registry.json, imports each module's hook class,
    and calls its constructor. The constructor starts the pulse loop.
    That's it — the organ is alive and autonomous from this point.

    Returns list of module IDs that successfully started.
    """
    registry_path = os.path.expanduser("~/.et_modules/registry.json")
    if not os.path.exists(registry_path):
        logger.warning("No module registry at %s", registry_path)
        return []

    import json as _json
    with open(registry_path) as f:
        registry = _json.load(f)

    module_defs = registry.get("modules", {})
    skip = {"neurograph", "inference_difference", "ecosystem_monitor"}
    started = []

    # Sort so elmer loads last (heaviest — transformer models)
    modules = sorted(
        module_defs.items(),
        key=lambda x: (1 if x[0] == "elmer" else 0, x[0]),
    )

    for module_id, meta in modules:
        if module_id in skip:
            continue

        install_path = meta.get("install_path", "")
        entry_point = meta.get("entry_point", "")
        if not entry_point or not install_path:
            logger.warning("Module %s: missing entry_point or install_path", module_id)
            continue
        hook_file = os.path.join(install_path, entry_point)
        if not os.path.exists(hook_file):
            logger.warning("Module %s: hook file not found (%s)", module_id, hook_file)
            continue

        try:
            import importlib.util

            # Namespace isolation: save sys.path, clear generic collisions
            # Each module vendors core/, pipelines/, runtime/ — these collide.
            _generic_prefixes = ("core", "pipelines", "runtime", "surgery", "openclaw_adapter", "ng_ecosystem", "ng_lite", "ng_embed", "ng_autonomic", "ng_peer_bridge", "ng_tract_bridge")
            # Also clear module-specific packages that could collide
            # (but NOT the module's own package — that breaks lazy imports)
            path_snapshot = list(sys.path)
            stashed = {}
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        stashed[mod_name] = sys.modules.pop(mod_name)
                        break

            if install_path and install_path not in sys.path:
                sys.path.insert(0, install_path)

            spec_name = f"_mod_{module_id}"
            spec = importlib.util.spec_from_file_location(spec_name, hook_file)
            if spec is None:
                logger.warning("Cannot create import spec for %s", module_id)
                sys.path[:] = path_snapshot
                sys.modules.update(stashed)
                continue

            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = mod
            logger.info("Loading %s: sys.path[0]=%s, core in sys.modules=%s",
                        module_id, sys.path[0] if sys.path else "EMPTY",
                        "core" in sys.modules)
            spec.loader.exec_module(mod)

            # Find the hook class
            instance = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type)
                        and attr_name != "OpenClawAdapter"
                        and hasattr(attr, "MODULE_ID")
                        and hasattr(attr, "_module_on_message")):
                    instance = attr()
                    break

            if instance is None:
                logger.warning("Module %s: no hook class found in %s", module_id, hook_file)
                continue

            _module_instances[module_id] = instance
            started.append(module_id)
            logger.info("Loaded module hook: %s", module_id)

        except Exception as exc:
            logger.warning("Module %s failed to load: %s", module_id, exc)
        finally:
            # Pin this module's generic imports so they survive cleanup
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules[f"_{module_id}_{mod_name}"] = sys.modules[mod_name]
                        break

            # Clean up generic names for next module
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules.pop(mod_name, None)
                        break

            # Restore path and stashed generics
            sys.path[:] = path_snapshot
            for mod_name, mod_obj in stashed.items():
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = mod_obj

    return started


def _deposit_topology_delta(step_result, text: str = None, embedding = None) -> None:
    """Extract topology delta and deposit to all module tracts (River Tier 3).

    #119: BTF binary deposit via Rust (zero-copy). No JSONL fallback.
    """
    if _memory is None:
        return
    peer_bridge = getattr(_memory, '_peer_bridge', None)
    if peer_bridge is None:
        return
    try:
        import ng_tract
        tract_paths = [
            str(peer_bridge._module_dir / f"{pid}.tract")
            for pid in peer_bridge._get_registered_peers()
        ]
        ng_tract.deposit_topology(
            step_result, _memory.graph, _memory.vector_db, tract_paths,
        )
    except Exception as exc:
        logger.debug("BTF topology deposit failed: %s", exc)


def _deposit_surfacing_outcome(params: Dict[str, Any], user_text: Optional[str]) -> None:
    """Deposit raw surfacing outcome experience to the substrate (Punchlist #56).

    Records the complete turn triad as raw experience:
    - Which nodes were surfaced during assemble (cached in _last_surfaced_nodes)
    - What the user said (user_text from ingest cache)
    - What Syl said in response (lastAssistantMessage from TS plugin)

    No classification. The substrate sees: "these nodes were in the context
    window when this input/output pair happened." Elmer learns what surfacing
    patterns correlate with coherent responses via the River.

    Each surfaced node gets a record_outcome with its own embedding and
    opaque metadata containing text previews of the turn. The substrate's
    Hebbian dynamics handle the rest.
    """
    global _last_surfaced_nodes

    if _memory is None or not _last_surfaced_nodes:
        return

    # Extract Syl's response text
    syl_text = None
    if params.get("lastAssistantMessage"):
        syl_text = _extract_message_text(params["lastAssistantMessage"])

    if not syl_text or not syl_text.strip():
        _last_surfaced_nodes = []
        return  # No response to record outcome against

    peer_bridge = getattr(_memory, '_peer_bridge', None)
    if peer_bridge is None:
        _last_surfaced_nodes = []
        return

    try:
        from ng_embed import embed

        # Embed Syl's response — this is the outcome of the surfacing
        syl_embedding = embed(syl_text)

        for node_info in _last_surfaced_nodes:
            node_id = node_info["node_id"]

            # Get the node's existing embedding from the vector DB
            db_entry = _memory.vector_db.get(node_id)
            if db_entry is None:
                continue
            node_embedding = db_entry.get("embedding")
            if node_embedding is None:
                continue

            # Deposit raw experience: this node was surfaced during this turn.
            # target_id is opaque — just marks it as a surfacing event.
            # metadata carries the raw context without classification.
            peer_bridge.record_outcome(
                embedding=node_embedding,
                target_id=f"surfacing:{node_id}",
                success=True,
                module_id="neurograph",
                metadata={
                    "surfacing_source": node_info.get("source", "unknown"),
                    "surfacing_strength": node_info.get("strength", node_info.get("score", 0)),
                    "user_text_preview": (user_text or "")[:200],
                    "syl_response_preview": syl_text[:200],
                },
            )

        logger.debug(
            "Surfacing outcome deposited: %d nodes, syl_response=%d chars",
            len(_last_surfaced_nodes), len(syl_text),
        )
    except Exception as exc:
        logger.debug("Surfacing outcome deposit failed: %s", exc)
    finally:
        _last_surfaced_nodes = []


def _discord_alert(module_id: str, error_msg: str) -> None:
    """Post error to Discord #dev-log webhook. Fire-and-forget."""
    if not _DISCORD_WEBHOOK:
        return
    try:
        payload = json.dumps({
            "content": f"**Module hook error: {module_id}**\n```\n{error_msg[:500]}\n```",
            "username": "NeuroGraph Fan-Out",
        }).encode("utf-8")
        req = urllib.request.Request(
            _DISCORD_WEBHOOK,
            data=payload,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logger.debug("Discord alert failed: %s", exc)


# ── RPC Dispatch ──────────────────────────────────────────────────────


def handle_bootstrap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create NeuroGraphMemory singleton and restore from checkpoint."""
    global _memory, _tract

    if _memory is not None:
        return {"bootstrapped": True, "reason": "already_initialized"}

    # Auto-update before loading anything else
    try:
        from ng_updater import auto_update; auto_update()
    except Exception:
        pass

    import topology_owner
    from openclaw_hook import NeuroGraphMemory
    from ng_experience_tract import ExperienceTract

    # Claim topology ownership — we are the sole writer to main.msgpack.
    # If another process (GUI, standalone ingestor) already owns it,
    # refuse to bootstrap rather than risk dual-write corruption.
    if not topology_owner.claim():
        existing = topology_owner.owner_pid()
        logger.error(
            "Cannot bootstrap — topology owned by PID %s. "
            "Dual-write hazard (Syl's Law).",
            existing,
        )
        return {
            "bootstrapped": False,
            "reason": f"topology_owned_by_pid_{existing}",
        }

    _memory = NeuroGraphMemory.get_instance()
    _tract = ExperienceTract()

    # Wake the organs — each module's __init__ starts its pulse loop
    started_modules = _bootstrap_modules()


    # Lenia FlowGraph — continuous field dynamics (dormant by default)
    global _lenia_kill_switch, _lenia_engine, _lenia_bridge, _lenia_competence
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
        n_entities = len(_memory.graph.nodes)
        n_channels = len(lenia_cfg.initial_channels)

        lenia_substrate = NeuroGraphSubstrate(_memory.graph, _memory.vector_db)
        lenia_field = LeniaFieldStore(lenia_cfg.field_dir, n_entities, n_channels)
        lenia_registry = ChannelRegistry(lenia_cfg, lenia_cfg.field_dir)
        lenia_cache = DistanceCache(n_entities)
        lenia_cache.populate(lenia_substrate)  # build distances from live graph
        lenia_kernel = KernelComputer(lenia_cache, lenia_registry)
        lenia_myelin = MyelinationObserver(lenia_cfg)
        _lenia_competence = CompetenceMeter(lenia_cfg, lenia_myelin)
        _lenia_engine = UpdateEngine(lenia_cfg, lenia_field, lenia_kernel, lenia_registry)
        _lenia_bridge = SpikeFieldBridge(lenia_cfg, lenia_field, lenia_substrate)
        _lenia_kill_switch = KillSwitch(lenia_cfg, lenia_cfg.field_dir)
        _lenia_kill_switch.set_components(_lenia_engine, _lenia_bridge)

        _lenia_engine.register_post_tick(lenia_myelin.update)

        if _lenia_kill_switch.enabled:
            _lenia_kill_switch.enable(graph=_memory.graph)
            logger.info("Lenia FlowGraph ACTIVE — field dynamics running")
        else:
            logger.info("Lenia FlowGraph loaded (dormant — kill switch off)")
    except ImportError:
        logger.info("Lenia FlowGraph not available (lenia/ package not found)")
        _lenia_kill_switch = None
        _lenia_engine = None
        _lenia_bridge = None
        _lenia_competence = None
    except Exception:
        logger.exception("Lenia FlowGraph failed to initialize — continuing without")
        _lenia_kill_switch = None
        _lenia_engine = None
        _lenia_bridge = None
        _lenia_competence = None

    stats = _memory.stats()
    tract_stats = _tract.stats()
    logger.info(
        "Bootstrapped: %d nodes, %d synapses, %d hyperedges, timestep %d, "
        "tract pending: %d, modules: %s",
        stats["nodes"],
        stats["synapses"],
        stats["hyperedges"],
        stats["timestep"],
        tract_stats["pending"],
        started_modules,
    )

    _start_http_sidecar(8850)

    # #109: Shared graph lock for thread safety.
    # Pulse loops (via NGSaaSBridge) and the Tonic both access graph
    # internals concurrently. This RLock serializes access.
    # Attached to graph object so both bridge and engine can find it.
    import threading as _thr
    _memory.graph._concurrent_lock = _thr.RLock()

    # The Tonic: conversation starting — language tokens about to flow
    if _memory._tonic_thread is not None:
        try:
            _memory._tonic_thread.conversation_started()
        except Exception:
            pass

    return {
        "bootstrapped": True,
        "nodes": stats["nodes"],
        "synapses": stats["synapses"],
        "timestep": stats["timestep"],
        "tract_pending": tract_stats["pending"],

        "tonic": _memory._tonic_thread.status if _memory._tonic_thread else None,
    }


def handle_ingest(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single message through the 5-stage pipeline."""
    if _memory is None:
        return {"ingested": False, "reason": "not_bootstrapped"}

    text = _extract_message_text(params.get("message", {}))
    if not text or not text.strip():
        return {"ingested": False}

    result = _memory.ingestor.ingest(text)

    # Feed CES stream parser (background node nudging)
    if _memory._stream_parser is not None:
        _memory._stream_parser.feed(text)

    # Write peer learning event for sibling modules
    _memory._write_peer_learning_event(text, result, type("R", (), {"fired_node_ids": []})())

    _memory._message_count += 1

    # Write memory event for OpenClaw consumption
    _memory._write_memory_event("ingestion", {
        "status": "ingested",
        "nodes_created": len(result.nodes_created),
        "synapses_created": len(result.synapses_created),
        "chunks": result.chunks_created,
        "message_count": _memory._message_count,
        "source": "context_engine",
    })

    # Cache text + embedding for topology delta deposit in afterTurn
    global _ingest_text, _ingest_embedding
    _ingest_text = text

    try:
        from ng_embed import embed
        _ingest_embedding = embed(text)
    except Exception:
        _ingest_embedding = None

    return {"ingested": True}


def handle_assemble(params: Dict[str, Any]) -> Dict[str, Any]:
    """Surface substrate associations for the system prompt.

    NeuroGraph does NOT modify the conversation messages.  It adds
    substrate context via systemPromptAddition — the 'dipping the
    bucket in the River' moment.
    """
    if _memory is None:
        return {"systemPromptAddition": None}

    messages = params.get("messages", [])

    # Extract text from recent user messages for association priming
    recent_text = _extract_recent_user_text(messages, max_messages=3)
    if not recent_text:
        return {"systemPromptAddition": None}

    # Spreading activation harvest — the cortex-like recall
    surfaced = _memory._harvest_associations(recent_text)

    # CES surfacing — concepts that fired above threshold
    ces_surfaced = []
    if _memory._surfacing_monitor is not None:
        ces_surfaced = _memory._surfacing_monitor.get_surfaced()

    # The Tonic: latent thread — always present in context
    latent_context = None
    if _memory._tonic_thread is not None:
        try:
            # Run an ouroboros cycle at assembly time too — keep the thread fresh
            _memory._tonic_thread.ouroboros_cycle()
            latent_context = _memory._tonic_thread.format_latent_context()
        except Exception as exc:
            logger.debug("Tonic assembly error: %s", exc)

    # Punchlist #56: Cache what was surfaced for outcome deposit in afterTurn.
    # Raw node IDs + scores — no classification, just what went into the bucket.
    global _last_surfaced_nodes
    _last_surfaced_nodes = []
    for item in surfaced[:7]:  # Match the cap used in formatting
        if item.get("node_id"):
            _last_surfaced_nodes.append({
                "node_id": item["node_id"],
                "strength": item.get("strength", 0),
                "source": "spreading_activation",
            })
    for item in ces_surfaced[:3]:
        if item.get("node_id"):
            _last_surfaced_nodes.append({
                "node_id": item["node_id"],
                "score": item.get("score", 0),
                "source": "ces",
            })

    # Format as context block for the system prompt
    context_block = _format_substrate_context(surfaced, ces_surfaced, latent_context)

    return {"systemPromptAddition": context_block}


def handle_after_turn(params: Dict[str, Any]) -> None:
    """Post-turn lifecycle: drain tract, learn, reward, save.

    This is where the SNN processes what just happened AND absorbs
    any experience deposited by feeders (GUI, feed-syl, file watcher)
    via the experience tract.  The tract drain is event-driven — it
    happens here because a conversation turn just completed, not on
    a timer.  No polling.
    """
    if _memory is None:
        return

    # Recover fan-out cache if lost to process restart (OOM resilience).
    # The TS plugin passes lastUserMessage so the fan-out doesn't depend on
    # in-memory state surviving between ingest and afterTurn calls.
    global _ingest_text, _ingest_embedding
    last_user = params.get("lastUserMessage") or params.get("lastMessage")  # legacy fallback
    if _ingest_text is None and last_user:
        recovered = _extract_message_text(last_user)
        if recovered and recovered.strip():
            _ingest_text = recovered
            try:
                from ng_embed import embed
                _ingest_embedding = embed(recovered)
            except Exception:
                _ingest_embedding = None
            logger.info("Recovered fan-out text from afterTurn params (%d chars)", len(recovered))

    # Drain the experience tract — absorb feeder deposits
    if _tract is not None:
        _drain_tract()

    # SNN learning step — STDP, structural plasticity, predictions
    step_result = _memory.graph.step()

    # Baseline conversational engagement reward (heartbeat)
    if _memory.graph.config.get("three_factor_enabled", False):
        _memory.graph.inject_reward(0.1)

    # CES surfacing monitor — scan fired nodes
    if _memory._surfacing_monitor is not None:
        _memory._surfacing_monitor.after_step(step_result)

    # River-based Tier 3: deposit raw topology delta to all module tracts.
    # The delta contains fired nodes with causal context, hyperedge activations,
    # prediction results, structural changes, and salience signals. Raw,
    # unclassified (Law 7). Each module's bucket extracts what it needs.
    _deposit_topology_delta(step_result, text=_ingest_text, embedding=_ingest_embedding)

    # Punchlist #56: Deposit raw surfacing outcome experience.
    # The triad: what was surfaced (cached from assemble) + user input
    # (cached from ingest) + Syl's response (from TS plugin).
    # No classification — just the raw facts. The substrate learns
    # the correlation between surfaced context and what Syl produced.
    _deposit_surfacing_outcome(params, _ingest_text)

    # Clear after deposit — consumed
    _ingest_text = None
    _ingest_embedding = None

    # Novelty probation
    _memory.ingestor.update_probation()

    # Auto-save (every 10 messages, matching existing interval)
    if _memory._message_count > 0 and _memory._message_count % _memory.auto_save_interval == 0:
        _memory.save()
        logger.info("Auto-save at message %d", _memory._message_count)

    # Lenia FlowGraph — post-step competence update and energy watchdog
    if _lenia_kill_switch is not None and _lenia_kill_switch.enabled:
        try:
            if _lenia_competence is not None and _lenia_engine is not None:
                _lenia_competence.update(_lenia_engine._field.read_buffer())
            _lenia_kill_switch.check_energy(
                _lenia_engine._field.total_energy(),
                _lenia_engine._field._ledger[:, 0].sum(),
            )
            _lenia_engine._field.reset_ledger()
        except Exception:
            logger.exception("Lenia post-step update failed")



def _drain_tract() -> None:
    """Drain pending experience from the tract into the topology.

    Each entry is fed through the ingestor as raw experience — the
    same pipeline that on_message() uses.  The tract carries it here
    without transformation; the ingestor is where experience meets
    the substrate.
    """
    entries = _tract.drain()
    if not entries:
        return

    for entry in entries:
        content = entry.get("content", "")
        content_type = entry.get("content_type", "text")
        source = entry.get("source", "unknown")

        if not content or not content.strip():
            continue

        try:
            if content_type == "file":
                # File path — use ingest_file for format detection
                result = _memory.ingest_file(content)
            else:
                # Raw text — feed through the ingestor
                result = _memory.ingestor.ingest(content)
                _memory._message_count += 1

            logger.info(
                "Tract drain: %s from %s — %s",
                content_type,
                source,
                "ok" if result else "empty",
            )
        except Exception as exc:
            logger.warning("Tract drain entry failed (%s): %s", source, exc)


def handle_compact(params: Dict[str, Any]) -> Dict[str, Any]:
    """NeuroGraph-driven conversation compaction.

    The substrate scores each message by activation strength, then guides
    TID to summarize low-importance older messages while keeping recent
    turns verbatim.  Compaction metrics feed back to the substrate.

    Flow:
        1. Read session JSONL
        2. Keep last N turns verbatim (configurable, default 5)
        3. Score older messages via spreading activation
        4. Call TID to summarize older messages, guided by NG importance
        5. Write compacted session back
        6. Feed compaction metrics to substrate for learning
    """
    import json as _json
    import urllib.request
    import time

    if _memory is None:
        return {"ok": True, "compacted": False, "reason": "no memory"}

    session_file = params.get("sessionFile", "")
    force = params.get("force", False)
    token_budget = params.get("tokenBudget", 128000)
    keep_turns = 8  # Number of recent user/assistant turn pairs to keep

    if not session_file:
        return {"ok": False, "compacted": False, "reason": "no session file"}

    # --- Step 1: Read session JSONL ---
    try:
        with open(session_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return {"ok": False, "compacted": False, "reason": f"read failed: {e}"}

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue

    # Find conversation messages (user + assistant)
    conversation_indices = []
    for i, entry in enumerate(entries):
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            conversation_indices.append(i)

    # Not enough to compact
    if len(conversation_indices) < keep_turns * 2 + 2:
        return {"ok": True, "compacted": False, "reason": "too few messages"}

    # --- Step 2: Split into compactable and keep zones ---
    keep_start = conversation_indices[-(keep_turns * 2):]
    compact_indices = [i for i in conversation_indices if i not in keep_start]

    if len(compact_indices) < 2:
        return {"ok": True, "compacted": False, "reason": "nothing to compact"}

    # --- Step 3: Score older messages via substrate activation ---
    scored_messages = []
    for idx in compact_indices:
        msg = entries[idx].get("message", {})
        text = _extract_message_text(msg)
        if not text:
            scored_messages.append({"idx": idx, "text": "", "importance": 0.0})
            continue

        # Use spreading activation to score importance
        try:
            surfaced = _memory._harvest_associations(text)
            # More associations = more interconnected = more important
            importance = min(1.0, len(surfaced) / 5.0)
        except Exception:
            importance = 0.5  # Default mid-importance on error

        scored_messages.append({
            "idx": idx,
            "text": text[:500],  # Truncate for summary prompt
            "importance": importance,
            "role": msg.get("role", "unknown"),
        })

    # --- Step 4: Build summary prompt with NG guidance ---
    high_importance = [m for m in scored_messages if m["importance"] > 0.6]
    low_importance = [m for m in scored_messages if m["importance"] <= 0.6]

    summary_input = []
    for m in scored_messages:
        prefix = "[IMPORTANT] " if m["importance"] > 0.6 else ""
        summary_input.append(f'{prefix}{m["role"]}: {m["text"]}')

    summary_prompt = (
        "Summarize the following conversation history into a concise summary. "
        "Messages marked [IMPORTANT] contain key context that should be "
        "preserved in detail. Other messages can be condensed more aggressively. "
        "Output ONLY the summary, no preamble.\n\n"
        + "\n".join(summary_input)
    )

    # Call TID for summarization
    try:
        tid_body = _json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.3,
            "max_tokens": 1000,
        }).encode("utf-8")
        tid_req = urllib.request.Request(
            "http://127.0.0.1:7437/v1/chat/completions",
            data=tid_body,
            method="POST",
        )
        tid_req.add_header("Content-Type", "application/json")
        tid_req.add_header("Authorization", "Bearer tid")
        with urllib.request.urlopen(tid_req, timeout=30) as resp:
            tid_resp = _json.loads(resp.read().decode("utf-8"))
        summary_text = tid_resp["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("TID summarization failed: %s", e)
        return {"ok": False, "compacted": False, "reason": f"summarization failed: {e}"}

    # --- Step 5: Rebuild session JSONL ---
    # Estimate tokens before
    tokens_before = sum(
        entry.get("message", {}).get("usage", {}).get("totalTokens", 0)
        for entry in entries
    )
    if tokens_before == 0:
        # Fallback estimate from content
        total_chars = sum(len(str(entry)) for entry in entries)
        tokens_before = total_chars // 4

    # Build new entries:
    # 1. Non-conversation entries before the compacted zone (system prompts, etc.)
    # 2. Summary entry replacing compacted messages
    # 3. Kept recent entries
    new_entries = []

    # Keep any entries before first compacted message (system, etc.)
    first_compact_idx = compact_indices[0]
    for i in range(first_compact_idx):
        new_entries.append(entries[i])

    # Insert summary as a system-like entry
    summary_entry = {
        "type": "message",
        "id": f"compact_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"[Conversation Summary]\n{summary_text}",
            }],
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": int(len(summary_text.split()) * 1.3),
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
        },
    }
    new_entries.append(summary_entry)

    # Keep any non-conversation entries between zones
    compact_set = set(compact_indices)
    keep_set = set(keep_start)
    for i in range(first_compact_idx, len(entries)):
        if i in compact_set:
            continue  # Compacted away
        if i in keep_set or i not in conversation_indices:
            new_entries.append(entries[i])

    # --- Write back ---
    try:
        with open(session_file, "w") as f:
            for entry in new_entries:
                f.write(_json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        return {"ok": False, "compacted": False, "reason": f"write failed: {e}"}

    tokens_after = tokens_before - (tokens_before * len(compact_indices) // len(conversation_indices))
    tokens_after = max(tokens_after, int(len(summary_text.split()) * 1.3))

    # --- Step 6: Feed compaction metrics to substrate ---
    try:
        _memory.graph.step()  # Normal consolidation pass
        logger.info(
            "Compaction complete: %d messages → %d, %d → ~%d tokens, "
            "%d high-importance preserved",
            len(entries), len(new_entries),
            tokens_before, tokens_after,
            len(high_importance),
        )
    except Exception as e:
        logger.warning("Post-compaction substrate step failed: %s", e)

    return {
        "ok": True,
        "compacted": True,
        "result": {
            "summary": summary_text[:200],
            "tokensBefore": tokens_before,
            "tokensAfter": tokens_after,
            "firstKeptEntryId": new_entries[-1].get("id", "") if new_entries else "",
        },
    }


def handle_dispose(params: Dict[str, Any]) -> None:
    """Final save and cleanup."""
    if _memory is None:
        return

    # The Tonic: conversation ended — latent mode continues
    # The thread doesn't stop. Language tokens stopped. That's all.
    if _memory._tonic_thread is not None:
        try:
            _memory._tonic_thread.conversation_ended()
        except Exception:
            pass

    # Lenia FlowGraph — #109: stays running between conversations.
    # Dispose is subtraction, not destruction. Field dynamics continue.

    _memory.save()
    logger.info("Final save on dispose")

    # Modules run autonomously via pulse loops.
    # No fan-out to clean up — modules read from River tracts.


def handle_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return substrate telemetry."""
    if _memory is None:
        return {"error": "not_bootstrapped"}
    stats = _memory.stats()
    stats["module_hooks"] = {
        "loaded": [],  # modules are autonomous, no fan-out registry
        "errors": dict(_module_errors),
    }
    return stats


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_message_text(message: Dict[str, Any]) -> str:
    """Extract plain text from an AgentMessage-shaped dict.

    AgentMessage content can be a string or an array of content parts.
    We extract text from both forms.
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Content part objects: { type: "text", text: "..." }
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
        return " ".join(parts)

    return str(content)


def _extract_recent_user_text(
    messages: List[Dict[str, Any]], max_messages: int = 3
) -> str:
    """Extract text from the most recent user messages."""
    user_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _extract_message_text(msg)
            if text.strip():
                user_texts.append(text)
            if len(user_texts) >= max_messages:
                break

    # Reverse to chronological order, join
    user_texts.reverse()
    return "\n".join(user_texts)


def _format_substrate_context(
    surfaced: List[Dict[str, Any]],
    ces_surfaced: List[Dict[str, Any]],
    latent_context: Optional[str] = None,
) -> Optional[str]:
    """Format surfaced knowledge into a system prompt context block.

    Returns None if nothing was surfaced — no empty blocks injected.
    The latent thread (The Tonic) is always included when available —
    it is the persistent slot that never gets evicted.
    """
    has_surfaced = bool(surfaced) or bool(ces_surfaced)
    has_latent = latent_context is not None

    if not has_surfaced and not has_latent:
        return None

    lines = []

    # The Tonic's latent thread comes first — it is the baseline.
    # Conversation context is the event on top of it.
    if has_latent:
        lines.append(latent_context)
        lines.append("")

    if has_surfaced:
        lines.append("## Substrate Context (NeuroGraph)")
        lines.append("The following associations surfaced from the cognitive substrate:")
        lines.append("")

        if surfaced:
            for item in surfaced[:7]:  # Cap at 7 to keep context manageable
                content = item.get("content", "")
                strength = item.get("strength", 0)
                if content:
                    # Truncate very long content
                    if len(content) > 300:
                        content = content[:297] + "..."
                    lines.append(f"- [{strength:.2f}] {content}")

        if ces_surfaced:
            for item in ces_surfaced[:3]:
                content = item.get("content", "")
                if content:
                    if len(content) > 300:
                        content = content[:297] + "..."
                    lines.append(f"- [CES] {content}")

    return "\n".join(lines)



# ── HTTP Sidecar — afterTurn bypass ──────────────────────────────────
# [2026-03-26] Claude Code (Opus 4.6) — afterTurn HTTP trigger
# What: Lightweight HTTP listener on port 8850 for direct afterTurn calls.
# Why:  OpenClaw 2026.3.13 never calls afterTurn on the ContextEngine plugin.
#       Module fan-out was dead. This bypasses OC's lifecycle gap.
# How:  Background thread runs http.server on 127.0.0.1:8850.
#       POST /afterTurn triggers handle_after_turn + fan-out.
#       GET /status returns hook count and last fire time.

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

_last_afterturn_fire: Optional[str] = None
_sidecar_started = False


class _AfterTurnHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _last_afterturn_fire
        if self.path == "/afterTurn":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len else b"{}"
                params = json.loads(body) if body else {}
                handle_after_turn(params)
                _last_afterturn_fire = __import__("datetime").datetime.now().isoformat()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "fired": _last_afterturn_fire}).encode())
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "hooks_loaded": 0,  # fan-out removed — modules autonomous
                "last_afterturn": _last_afterturn_fire,
            }).encode())
        elif self.path.startswith('/bunyan/'):
            self._handle_bunyan()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_bunyan(self):
        """Bunyan user bucket — extraction from the live substrate."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        q = qs.get('q', [''])[0]

        hook = None  # TODO: bunyan sidecar needs direct import if still needed
        if hook is None:
            self._json_response(503, {'error': 'Bunyan not loaded'})
            return
        if not q:
            self._json_response(400, {'error': 'Missing q parameter'})
            return

        try:
            if parsed.path == '/bunyan/query':
                depth = int(qs.get('depth', [0])[0]) or None
                k = int(qs.get('k', [0])[0]) or None
                result = hook.query_story(q, max_depth=depth, similar_k=k)
                if result is None:
                    self._json_response(200, {'narrative': None, 'message': 'No matching events in substrate'})
                else:
                    self._json_response(200, result)
            elif parsed.path == '/bunyan/similar':
                k = int(qs.get('k', [5])[0])
                result = hook.find_similar_events(q, k=k)
                self._json_response(200, {'events': result})
            elif parsed.path == '/bunyan/recall':
                k = int(qs.get('k', [5])[0])
                threshold = float(qs.get('threshold', [0.5])[0])
                if _memory is None:
                    self._json_response(503, {'error': 'NeuroGraph not bootstrapped'})
                    return
                result = _memory.recall(q, k=k, threshold=threshold)
                self._json_response(200, {'results': result})
            elif parsed.path == '/bunyan/associate':
                k = int(qs.get('k', [10])[0])
                steps = int(qs.get('steps', [3])[0])
                if _memory is None:
                    self._json_response(503, {'error': 'NeuroGraph not bootstrapped'})
                    return
                result = _memory.associate(q, k=k, steps=steps)
                self._json_response(200, {'associations': result})
            else:
                self._json_response(404, {'error': 'Unknown bunyan endpoint'})
        except Exception as exc:
            self._json_response(500, {'error': str(exc)})

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        pass


def _start_http_sidecar(port: int = 8850) -> None:
    global _sidecar_started
    if _sidecar_started:
        return
    try:
        server = HTTPServer(("127.0.0.1", port), _AfterTurnHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _sidecar_started = True
        logger.info("afterTurn HTTP sidecar listening on 127.0.0.1:%d", port)
    except Exception as exc:
        logger.error("Failed to start afterTurn sidecar: %s", exc)

# ── JSON-RPC Server ───────────────────────────────────────────────────

METHODS = {
    "bootstrap": handle_bootstrap,
    "ingest": handle_ingest,
    "assemble": handle_assemble,
    "afterTurn": handle_after_turn,
    "compact": handle_compact,
    "dispose": handle_dispose,
    "stats": handle_stats,
}


def process_request(line: str) -> Optional[str]:
    """Process a single JSON-RPC request and return the response."""
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": f"Parse error: {exc}"},
        })

    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    handler = METHODS.get(method)
    if handler is None:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    try:
        result = handler(params)
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }, default=str)
    except Exception as exc:
        logger.error("RPC method %s failed: %s\n%s", method, exc, traceback.format_exc())
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        })


def main() -> None:
    """Main RPC loop — read requests from stdin, write responses to stdout."""
    logger.info("NeuroGraph RPC bridge starting")

    # Signal readiness to the TypeScript plugin
    ready_msg = json.dumps({
        "jsonrpc": "2.0",
        "method": "ready",
        "params": {"pid": os.getpid()},
    })
    sys.stdout.write(ready_msg + "\n")
    sys.stdout.flush()

    # Self-bootstrap on startup — the organism is born when the process
    # starts, not when the first message arrives.  OpenClaw's bootstrap
    # RPC will hit the "already_initialized" guard and mode-swap.
    #
    # Runs in a background thread because module loading produces hundreds
    # of log lines to stderr. If bootstrap runs synchronously before the
    # stdin loop, the 64KB OS pipe buffer fills before the TS plugin can
    # drain it, and the Python process blocks on write. Background thread
    # lets the stdin loop start immediately so the pipe stays drained.
    # Force-clean any stale sentinel from a previous process BEFORE
    # starting the bootstrap thread. This runs synchronously — it's a
    # single file check, no log output, no pipe risk. Must happen before
    # anything calls handle_bootstrap() (self-bootstrap thread OR TS
    # plugin RPC) so the sentinel is gone before either path checks it.
    try:
        import topology_owner
        sentinel = topology_owner._sentinel_path()
        if sentinel.exists():
            existing_pid = int(sentinel.read_text().strip())
            if existing_pid != os.getpid():
                sentinel.unlink(missing_ok=True)
                logger.info("Cleared stale sentinel (PID %d) on startup", existing_pid)
    except Exception:
        pass

    def _self_bootstrap():
        try:
            result = handle_bootstrap({"sessionId": "auto-startup"})
            logger.info("Self-bootstrap: %s", result)
        except Exception as exc:
            logger.error("Self-bootstrap failed: %s — will retry on first RPC", exc)

    import threading
    threading.Thread(target=_self_bootstrap, name="self-bootstrap", daemon=True).start()

    # Main RPC loop. If stdin closes (TS plugin context recycled),
    # keep the process alive — daemon threads (Tonic, pulse loops,
    # Lenia dynamics) ARE the organism. Reconnect when stdin reopens.
    import select
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                # stdin closed — TS plugin context may have recycled.
                # Keep process alive for daemon threads. Sleep and retry.
                logger.info("stdin closed — organism staying alive (daemon threads active)")
                import time as _t
                while True:
                    _t.sleep(60)
                    # Check if we should actually exit (systemd stop)
                    if not threading.main_thread().is_alive():
                        break
                break
            line = line.strip()
            if not line:
                continue
            response = process_request(line)
            if response is not None:
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
        except (BrokenPipeError, IOError):
            logger.info("stdin pipe broken — organism staying alive")
            import time as _t
            while True:
                _t.sleep(60)
                if not threading.main_thread().is_alive():
                    break
            break
        except KeyboardInterrupt:
            break

    logger.info("NeuroGraph RPC bridge shutting down")


if __name__ == "__main__":
    main()
