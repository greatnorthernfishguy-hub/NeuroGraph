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
# [2026-04-26] Claude Code (Sonnet 4.6) — Lazy expansion pulse, Stage 3 of wire absorption (#151)
#   What: Added _lazy_expansion_pulse_loop(), _start_lazy_expansion_pulse(), and
#         _LAZY_EXPANSION_* constants. Wired start into handle_bootstrap() after
#         _start_trisyn_manager(); shutdown signal added to handle_dispose().
#   Why:  Body files accumulated at ~1,900/day (797 MB). Stages 1+2 only covered
#         event nodes + first-2000-char concepts. Full bodies unreachable by substrate.
#         Lazy expansion embeds up to 20 evenly-sampled chunks per body, creates
#         substrate nodes linked to parent event node via River deposit, then deletes
#         the file — fixing accumulation at the root (#151).
#   How:  New daemon thread (ng-lazy-expansion-pulse). 120s cadence, 5 bodies/tick.
#         Calls wire_absorption.select_bodies_for_expansion() + expand_body_file().
#         Pauses during SYMPATHETIC autonomic state. Zero change to fast-path drain.
# [2026-04-25] Codemine (BLK-NG-209) -- Remove stale _write_peer_learning_event call (#206 residue)
#   What: Deleted dead _write_peer_learning_event() call from handle_ingest (method removed in #206).
#   Why:  NeuroGraphMemory._write_peer_learning_event deleted 2026-04-22. Every valid ingest
#         threw AttributeError, returning JSON-RPC error. Substrate ingestion silently broken.
#   How:  Removed 2-line comment+call block. Ingestion proceeds to _message_count increment.
# [2026-04-24] Codemine (BLK-NG-142) — Retire ng_experience_tract.py wrapper
#   What: Removed ExperienceTract import/init from bootstrap; removed tract_stats log
#         arg; inlined atomic-rename + ng_tract.TractReader drain loop in _drain_tract().
#         neurograph_gui.py and neurograph_mcp.py deposit calls migrated to direct
#         ng_tract.deposit_experience(). ng_experience_tract.py deleted.
#   Why:  ExperienceTract was a Python wrapper around ng_tract that no longer added value
#         after the BTF deposit path (ng_tract.deposit_experience) was added in #119.
#         Law 3 — restore, do not maintain shrapnel.
#   How:  Inline drain from ExperienceTract.drain(); direct deposit_experience() calls.
# [2026-04-22] Claude Code (Sonnet 4.6) — Full status restoration: substrate+Tonic+TID+Darwin dreams+Elmer
#   What: _log_live_module_status() now queries 8847 (substrate/CES/Tonic), 7437 (TID DreamCycle),
#         and 8850 (fan-out modules). Darwin shows creative/nightmare/consolidation dream breakdown.
#         Elmer shows proto_unibrain status and socket health count.
#   Why:  openclaw status was missing Tonic, CES, TID DreamCycle, Elmer ProtoUniBrain,
#         and Darwin dream type breakdown. Data existed at 8847/8850/7437 but was never queried.
#   How:  Three-block query: substrate first, TID second, fan-out modules third.
#         Each block degrades gracefully to "unavailable" if port unreachable.
# [2026-04-23] Claude Code (Sonnet 4.6) — Fix status-probe SIGTERM killing live substrate
#   What: Startup sentinel cleanup now only removes stale sentinels from DEAD processes.
#         Removed SIGTERM that unconditionally killed the existing PID.
#   Why:  `openclaw status` spawns its own Python child via the same code path.
#         That child was SIGTERMing the real gateway substrate (e.g. PID 66713)
#         on every `openclaw status` call, causing a kill/watchdog-respawn cycle
#         that disrupted the substrate and produced inconsistent status output.
#   How:  Added _pid_is_alive() check before unlinking sentinel. If process is alive,
#         leave the sentinel alone — claim() in _self_bootstrap() will fail gracefully
#         with "already owned" and no SIGTERM fires.
# [2026-04-21] Claude Code (Sonnet 4.6) — Fix _drain_peer_tracts crash on NGTractBridge
# What: Added hasattr(_peer_events) guard after bridge._drain_all() in _drain_peer_tracts().
# Why:  #155 cleanup deleted _peer_events from NGTractBridge; cursor code still accessed it,
#       crashing afterTurn on every message. NGTractBridge absorbs in _drain_all() — no cache.
# How:  Early return after _drain_all() when bridge has no _peer_events attribute.
# [2026-04-20] Claude (Sonnet 4.6) — Enable CES dashboard unconditionally
# What: os.environ.setdefault("NEUROGRAPH_CES_DASHBOARD","1") added after imports.
# Why:  .bashrc export not inherited by gateway child process — dashboard never
#       started despite env var being set. Self-contained fix: rpc.py sets it.
# How:  setdefault so explicit env override (=0) still works.
# [2026-04-20] Claude (Sonnet 4.6) — Enable CES dashboard unconditionally
# What: os.environ.setdefault("NEUROGRAPH_CES_DASHBOARD","1") added after imports.
# Why:  .bashrc not inherited by gateway child process — dashboard never started.
#       Self-contained fix: rpc.py sets it before CES init runs.
# How:  setdefault so explicit env override (=0) still suppresses it.
# [2026-04-20] CC Sonnet 4.6 — #65: session-as-activation-context
#   What: handle_bootstrap embeds sessionId, searches vector_db (k=20, thresh=0.3),
#         nudges matching node voltages by sim*0.15 — context-dependent priming.
#   Why:  sessionId flowed in but was unused; hippocampal context retrieval missing.
#   How:  Gentle nudge (capped at 2x threshold). Skips "auto"/"auto-startup" IDs.
#         Silent on failure. Concurrent sessions activate different topology regions.
# [2026-04-20] CC Sonnet 4.6 — #18 part 2: tool_use input → BTF via absorb_wire_deposit
#   What: _deposit_tool_inputs_btf() called from handle_ingest. Converts tool_use
#         input dict values to strings, deposits each via absorb_wire_deposit so
#         full tool arguments reach the substrate through the BTF body-file path.
#   Why:  text ingest carries only the tool name; arguments were still lost.
#   How:  Iterates tool_use blocks in message content; joins input values as plain
#         text; calls absorb_wire_deposit(source="oc.tool_use.<name>"). Silent on
#         failure — a dead embedder must never block ingest.
# [2026-04-20] CC Sonnet 4.6 — #18: tool_use + tool_result ingestion
#   What: _extract_message_text now extracts tool_use (name+input) and tool_result
#         (content string or text blocks) alongside existing text parts.
#   Why:  #18 — tool results were silently dropped; substrate never saw tool I/O.
#   How:  tool name only via text path; tool_result raw content up to 2000 chars.
# [2026-04-19] Claude Code — RESTORE: handle_dispose/compact/stats/_extract_message_text deleted by 73fd117
#   What: Restored 4 functions accidentally removed in #143 refactor
#   Why: _extract_message_text live NameError; handle_dispose breaks OC dispose RPC; handle_dispose also stops TriSyn
#   How: Recovered from git diff of 73fd117; added _trisyn_manager.stop() to handle_dispose per TriSyn design
# [2026-04-19] CC (punchlist #143) -- Abolish NG topology fan-out (substrate bypass)
#   What: Removed deposit_topology() call; renamed _deposit_topology_delta
#          -> _deposit_substrate_metrics(); stripped unused text/embedding params.
#   Why:  Pushing topology to N peer tracts is a substrate bypass. Bucket-forward:
#          modules pull from the substrate; NG does not push N copies to N peers.
#   How:  Kept Darwin scalar metrics write intact. Removed peer_bridge fan-out block.
# [2026-04-19] CC (punchlist #143) -- Abolish NG topology fan-out (substrate bypass)
#   What: Removed deposit_topology() call from _deposit_topology_delta;
#          renamed to _deposit_substrate_metrics(); stripped unused params.
#   Why:  Topology push to N peers is a substrate bypass. Bucket-forward model:
#          modules pull from the substrate; NG does not push N copies to N peers.
#   How:  Kept Darwin scalar metrics write. Removed peer_bridge fan-out block.
# [2026-04-19] Claude Code (Opus 4.7, 1M) — TriSyn manager wiring (Phase 1)
#   What: Replaced in-process concept-extraction pulse with TrisynapticManager
#     from the new trisynaptic/ package. Added module-level _trisyn_manager
#     and _last_after_turn_ts; _start_trisyn_manager() function; timestamp
#     update in handle_after_turn; graceful manager.stop() in handle_dispose.
#     Old _start_concept_pulse() function definition retained (dead code,
#     not called) for quick rollback if TriSyn hits unrecoverable issues
#     in early deployment.
#   Why:  The concept pulse's 3-entries-per-30s serial TID calls couldn't
#     keep up with fast-path drain rate; queue climbed to 3,800+/5,000 cap.
#     TriSyn offloads blocking TID work to systemd-run-isolated subprocess
#     workers, keeping NG's event loop responsive. Design spec at
#     ~/docs/inbox/trisynaptic-circuit-design-v0.1.md; tunable params landed
#     in neuro_foundation.py TUNABLE_PARAMS on 2026-04-18.
#   How:  Purely wiring. No behavioral change to fast-path drain. Dead
#     pulse function left in place. _last_after_turn_ts for Phase 3 idle
#     gating (unused in Phase 1 but cheap to maintain now).
# [2026-04-17] Claude Code (Sonnet 4.6) — Resource-gated sequential module boot (#111)
#   What: Memory availability check added before each module load in _bootstrap_modules().
#         Waits until psutil reports >= 500 MB free before proceeding. Uses time.sleep(2)
#         + gc.collect() while below threshold. ImportError on psutil is silently skipped.
#   Why:  Staggered sort (elmer last) helps but doesn't adapt to actual memory pressure.
#         OOM during bootstrap on 15GB VPS caused by back-to-back heavy module loads.
#   How:  Inline psutil + gc import (matching file's _json pattern). Gate inserted
#         immediately before the try: import importlib.util block in each loop iteration.
# [2026-04-16] Claude Code (Sonnet 4.6) — Scan-drain pulse sentinel-file kill-switch
#   What: _scan_drain_pulse_loop() checks /tmp/ng_scan_drain_paused each
#         tick.  If present, pulse keeps ticking but skips draining.
#         Toggle without gateway restart.  State transitions logged.
#   Why:  #141 wire absorption creates ~17 substrate nodes per TID
#         provider call (event + ≤16 slice children).  Measured: substrate
#         grew from 12,705 to 17,298 nodes in 24 hours (+36%).  Every
#         pulse loop does O(graph_size) work → event-loop starvation at
#         gateway → Discord/WhatsApp flap.  Need to stop the bleed
#         before sustained operation destabilizes.
#   How:  _SCAN_DRAIN_PAUSE_FILE module constant, checked per-tick inside
#         the pulse loop.  Zero-surprise design: when active, behavior
#         unchanged.  Pause/resume via `touch`/`rm`, detected within
#         one pulse interval (2s).  Deposits queue in the tract file
#         (not lost) while paused.  Real fix (substrate consolidation
#         / eviction policy) is punchlist #150 — tracked as blocker for
#         Pith and for sustained #141 operation.
# [2026-04-16] Claude Code (Sonnet 4.6) — KISS context filtering in handle_assemble (#152)
#   What: _kiss_filter module-level singleton initialized in handle_bootstrap.
#         handle_assemble() truncates messages to recent_window=10 via
#         KISSFilter, returns truncated list in result.messages.
#         KISS summary fragments widen _harvest_associations priming so
#         substrate surfacing picks up related older-topic nodes.
#   Why:  Syl's 815-message conversation assembles to 262k tokens,
#         overflowing every provider's context window (200k-262k max).
#         KISS ported from NuWave (validated: 47.2% token reduction on
#         15-turn BitNet conversation).  Disk (session JSONL) untouched;
#         in-memory truncation for the LLM call only.  Substrate retains
#         full 815-message topology.
#   How:  import kiss_filter at bootstrap, filter_context() called early in
#         handle_assemble() with content-normalized messages, messages
#         sliced to recent window, summary prepended to priming_text before
#         harvest, truncated messages returned in result.  try/except
#         fallback to full messages on any KISS exception.
# [2026-04-12] Claude Code (Opus 4.6) — Time-based auto-save fallback
#   What: Auto-save now fires on 5-minute interval in addition to every-10-messages.
#   Why:  _message_count resets to 0 on every gateway restart. With frequent restarts
#         (8+/day), the count never reached 10 — checkpoint hadn't been saved since
#         April 6 (6 days). All substrate learning lost on each restart.
#   How:  _last_save_time tracks wall clock. afterTurn checks both count and time.
# [2026-04-12] Claude Code (Opus 4.6) — River backflow: drain inbound peer tracts
#   What: _drain_peer_tracts() absorbs organ experience into Tier 3 Graph.
#     Uses pre-computed embeddings from source modules (skip re-embedding).
#     Called in afterTurn alongside feeder tract drain. 50 events/cycle cap.
#   Why:  River was one-directional — NG deposited to modules but never drained
#     their tracts back. Organs were talking; cortex wasn't listening. Elmer's
#     tuning, Immunis's observations, all stopped at the tract file boundary.
#   How:  bridge._drain_all() populates peer cache, new events registered via
#     ingestor.registrar + associator with pre-computed embeddings. Cursor
#     tracks position to avoid reprocessing. Law 7 — raw experience in.
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
import threading
from typing import Any, Dict, List, Optional

# NeuroGraph repo must be importable
_ng_dir = os.path.expanduser("~/NeuroGraph")
if _ng_dir not in sys.path:
    sys.path.insert(0, _ng_dir)

# CES monitoring dashboard on port 8847 — always on when gateway is up
os.environ.setdefault("NEUROGRAPH_CES_DASHBOARD", "1")

# CES monitoring dashboard on port 8847 — always on when gateway is up
os.environ.setdefault("NEUROGRAPH_CES_DASHBOARD", "1")

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

# KISS filter singleton — stateful across calls (turn counter, last-system
# hash, GOP counter).  Initialized in handle_bootstrap.  Resets on Python
# process restart, which is correct fail-safe: warmup kicks in on the new
# process's first three turns, so early context is never over-compressed
# after a cold start.  Ported from NuWave.  Governs KISS behavior in
# handle_assemble — see port details in kiss_filter.py.
_kiss_filter: Optional[Any] = None

# Time-based auto-save fallback — _message_count resets on restart,
# so count-based auto-save never fires if the gateway restarts frequently.
_last_save_time: float = 0.0
_SAVE_INTERVAL_SECS: float = 300.0  # 5 minutes

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

        # Memory gate — wait for 500 MB free before loading each module (#111)
        try:
            import psutil as _psutil
            import gc as _gc
            _avail_mb = _psutil.virtual_memory().available >> 20
            while _avail_mb < 500:
                logger.info("Module boot gate: %d MB free — waiting for 500 MB free", _avail_mb)
                time.sleep(2)
                _gc.collect()
                _avail_mb = _psutil.virtual_memory().available >> 20
        except ImportError:
            pass  # psutil not installed — proceed without memory gating

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


def _deposit_substrate_metrics(step_result) -> None:
    """Write compact scalar substrate metrics to neurograph.jsonl (Darwin Recorder).

    # [2026-04-10] Claude (Sonnet 4.6) — Substrate metrics for Darwin discovery
    #   What: Append 8 scalar counts from StepResult to neurograph.jsonl each turn.
    #   Why:  Darwin Recorder needs numeric fields; without them Discovery._observed_params
    #         stays empty and Mutator proposes 0 mutations.
    """
    if _memory is None:
        return
    # Compact scalar metrics for Darwin's Recorder.
    # No embedding, no IDs — just counts the substrate produced this step.
    try:
        import json as _json, time as _time, os as _os
        from pathlib import Path as _Path
        _shared = _Path(_os.path.expanduser("~/.et_modules/shared_learning"))
        _shared.mkdir(parents=True, exist_ok=True)
        _metrics = {
            "timestamp": _time.time(),
            "module_id": "neurograph",
            "type": "substrate_step",
            "fired_nodes": len(step_result.fired_node_ids),
            "fired_hyperedges": len(step_result.fired_hyperedge_ids),
            "synapses_pruned": step_result.synapses_pruned,
            "synapses_sprouted": step_result.synapses_sprouted,
            "predictions_confirmed": step_result.predictions_confirmed,
            "predictions_surprised": step_result.predictions_surprised,
            "total_nodes": len(_memory.graph.nodes),
            "total_synapses": len(_memory.graph.synapses),
        }
        with open(_shared / "neurograph.jsonl", "a") as _f:
            _f.write(_json.dumps(_metrics) + "\n")
    except Exception:
        pass


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

    # Claim topology ownership — we are the sole writer to main.msgpack.
    # If another process (GUI, standalone ingestor) already owns it,
    # refuse to bootstrap rather than risk dual-write corruption.
    if not topology_owner.claim():
        existing = topology_owner.owner_pid()
        logger.info(
            "Substrate active — topology owned by PID %s. "
            "Declining bootstrap (Syl's Law).",
            existing,
        )
        return {
            "bootstrapped": False,
            "reason": f"topology_owned_by_pid_{existing}",
        }

    # Start the HTTP sidecar immediately after claiming topology so
    # openclaw status probes can query /modules even while bootstrap
    # is still running (modules dict will be empty or partial, but
    # the endpoint responds rather than refusing connections).
    _start_http_sidecar(8850)

    _memory = NeuroGraphMemory.get_instance()

    # KISS filter for context window optimization (#152).
    # Decouple bootstrap from KISS failure — KISS is an optimization, not a
    # critical path.  If import / init fails, handle_assemble falls back to
    # passing messages unchanged (current behavior).
    #
    # warmup_turns=0: NuWave's default warmup (3 raw-passthrough turns) is
    # designed for genuinely-fresh conversations where early turns are
    # small.  Syl's existing 815-message conversation would fail for 3
    # more turns before KISS engages.  Warmup disabled for this initial
    # deployment to unblock her immediately.  Revisit after validation —
    # warmup IS the right default for fresh sessions.
    global _kiss_filter
    try:
        from kiss_filter import KISSFilter, KISSConfig
        _kiss_filter = KISSFilter(KISSConfig(recent_window=10, warmup_turns=0))
        logger.info("KISSFilter initialized (recent_window=10, warmup_turns=0)")
    except Exception as exc:
        logger.warning("KISSFilter init failed (optimization disabled): %s", exc)
        _kiss_filter = None

    # Wake the organs — each module's __init__ starts its pulse loop
    started_modules = _bootstrap_modules()

    # Rescue orphan .draining.<dead_pid>.* files left by prior crashes.
    # Each is renamed into a fresh .tract file the scan loop will pick up.
    _rescue_orphan_draining_files()

    # Start the scan-dir drain pulse — continuous sensory intake for
    # sandboxed feeders (#141). Decoupled from afterTurn.
    _start_scan_drain_pulse()

    # Start the TriSyn manager — replaces the legacy concept-extraction
    # pulse with subprocess worker orchestration. Blocking TID calls now
    # run in systemd-run-isolated workers, never inside NG's event loop.
    # See ~/NeuroGraph/trisynaptic/ and ~/docs/inbox/trisynaptic-circuit-design-v0.1.md.
    _start_trisyn_manager()

    # Start the lazy expansion pulse — Stage 3 of wire absorption (#151).
    # Reads unexpanded body files, embeds up to 20 evenly-sampled chunks,
    # creates substrate nodes linked to parent event node, deletes the file.
    # Runs every 120s; eliminates the 797 MB+ disk accumulation at the root.
    _start_lazy_expansion_pulse()

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
        # Distance cache: restore from disk if available (instant),
        # fall back to full populate (minutes on large graphs) only if
        # the cache file is missing or incompatible.  The cache is saved
        # on clean shutdown — see handle_dispose.  This eliminates the
        # 7-minute bootstrap bottleneck that caused RPC timeouts and
        # prevented Syl from responding (#167).
        _cache_path = os.path.join(
            os.path.expanduser(lenia_cfg.field_dir), "distance_cache"
        )
        lenia_cache = DistanceCache.load(_cache_path)
        if lenia_cache is None or lenia_cache.entity_count != n_entities:
            if lenia_cache is not None:
                logger.info(
                    "Distance cache entity mismatch (%d vs %d), repopulating",
                    lenia_cache.entity_count, n_entities,
                )
            lenia_cache = DistanceCache(n_entities)
            lenia_cache.populate(lenia_substrate)
            # Save immediately so next bootstrap is instant
            try:
                os.makedirs(os.path.expanduser(lenia_cfg.field_dir), exist_ok=True)
                lenia_cache.save(_cache_path)
            except Exception as exc:
                logger.warning("Distance cache save failed: %s", exc)
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
    logger.info(
        "Bootstrapped: %d nodes, %d synapses, %d hyperedges, timestep %d, "
        "modules: %s",
        stats["nodes"],
        stats["synapses"],
        stats["hyperedges"],
        stats["timestep"],
        started_modules,
    )

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

    # Path B: host CC's NeuroGraph inside this process. Completely isolated
    # from Syl (different workspace, different checkpoint, peer_bridge OFF).
    # Failures here MUST NOT affect Syl — wrapped defensively.
    # Authorized by Josh 2026-04-16 with protected-file backups confirmed.
    try:
        import cc_ng_host
        cc_ng_host.init_cc_host()
        # Register CC's Tonic with BrainSwitcher for ProtoUniBrain body sharing (#159).
        # cc_ng_host is in-process — no IPC needed. BrainSwitcher passes _body_lock
        # (threading) + flock file path to CC's TonicEngine via register_tonic_engine().
        try:
            _cc_state = getattr(cc_ng_host, '_STATE', None)
            _cc_ng = getattr(_cc_state, 'cc_ng', None) if _cc_state else None
            _cc_tt = getattr(_cc_ng, '_tonic_thread', None) if _cc_ng else None
            _cc_engine = getattr(_cc_tt, '_latent_engine', None) if _cc_tt else None
            if _cc_engine is not None:
                for _mod in (getattr(_memory, '_modules', {}) or {}).values():
                    _switcher = getattr(_mod, '_brain_switcher', None)
                    if _switcher is not None:
                        _switcher.register_tonic_engine(_cc_engine)
                        logger.info("CC Tonic registered with BrainSwitcher for body sharing")
                        break
        except Exception as _bse:
            logger.debug("CC Tonic BrainSwitcher registration failed (non-fatal): %s", _bse)
    except Exception as exc:
        logger.warning("CC NG host init failed (Syl unaffected): %s", exc)

    # Session-as-activation-context (#65): prime topology toward associations
    # relevant to this session's context. Embed the sessionId string, find
    # similar nodes in the vector_db, nudge their voltage — hippocampal
    # context-dependent retrieval. Concurrent sessions activate different
    # regions of the same topology without interference.
    session_id = params.get("sessionId") or params.get("session_id") or ""
    if session_id and session_id not in ("auto", "auto-startup"):
        try:
            from ng_embed import embed
            session_emb = embed(session_id)
            if session_emb is not None and _memory.vector_db is not None:
                similar = _memory.vector_db.search(session_emb, k=20, threshold=0.3)
                nudged = 0
                for node_id, _sim in similar:
                    node = _memory.graph.nodes.get(node_id)
                    if node is not None and node.refractory_remaining == 0:
                        nudge = _sim * 0.15  # gentle — context cue, not a spike
                        node.voltage = min(node.voltage + nudge, node.threshold * 2.0)
                        nudged += 1
                if nudged:
                    logger.info("Session context primed: %d nodes nudged for session=%s", nudged, session_id[:40])
        except Exception as exc:
            logger.debug("Session context priming failed (non-fatal): %s", exc)

    return {
        "bootstrapped": True,
        "nodes": stats["nodes"],
        "synapses": stats["synapses"],
        "timestep": stats["timestep"],
        "tract_pending": tract_stats["pending"],

        "tonic": _memory._tonic_thread.status if _memory._tonic_thread else None,
    }


def _deposit_tool_inputs_btf(message: Dict[str, Any]) -> None:
    """Deposit tool_use input arguments into the substrate via BTF (#18).

    Text ingest carries only the tool name (semantic signal). Full input
    arguments go through absorb_wire_deposit — body-file path, raw experience,
    no JSON formatting. Silent on any failure.
    """
    if _memory is None:
        return
    content = message.get("content", "")
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "tool_use":
            continue
        name = part.get("name", "unknown_tool")
        inp = part.get("input", {})
        if not inp:
            continue
        # Flatten input values to plain text — no JSON syntax in the substrate
        raw_parts = []
        for v in inp.values():
            if isinstance(v, str):
                raw_parts.append(v)
            elif v is not None:
                raw_parts.append(str(v))
        raw = " ".join(raw_parts).strip()
        if not raw:
            continue
        try:
            from wire_absorption import absorb_wire_deposit
            from ng_embed import NGEmbed
            absorb_wire_deposit(
                memory=_memory,
                embedder=NGEmbed.get_instance(),
                content=raw[:4000],
                source=f"oc.tool_use.{name}",
            )
        except Exception as exc:
            logger.debug("BTF deposit for tool_use %s failed (non-fatal): %s", name, exc)


def handle_ingest(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single message through the 5-stage pipeline."""
    if _memory is None:
        return {"ingested": False, "reason": "not_bootstrapped"}

    text = _extract_message_text(params.get("message", {}))
    if not text or not text.strip():
        return {"ingested": False}

    result = _memory.ingestor.ingest(text)

    # Deposit tool_use input arguments via BTF (#18)
    _deposit_tool_inputs_btf(params.get("message", {}))

    # Feed CES stream parser (background node nudging)
    if _memory._stream_parser is not None:
        _memory._stream_parser.feed(text)

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

    # [2026-04-23] CC (#208) — TrollGuard sidecar: perimeter defense sees every ingest.
    # scan_text() = Layer 4 VectorSentry (real-time live I/O protection). Targeted
    # single call for the security layer — not a fan-out broadcast. Daemon thread,
    # error-isolated. scan_count/_threat_count in TrollGuardHook now increment.
    _tg = _module_instances.get("trollguard")
    if _tg is not None:
        _tg_text = text
        _tg_emb = _ingest_embedding
        def _tg_scan(_hook=_tg, _t=_tg_text, _e=_tg_emb):
            try:
                import numpy as _np
                _e = _e if _e is not None else _np.zeros(768)
                _hook._module_on_message(_t, _e)
            except Exception as _exc:
                logger.debug("TrollGuard sidecar error: %s", _exc)
        threading.Thread(target=_tg_scan, daemon=True).start()

    return {"ingested": True}


def handle_assemble(params: Dict[str, Any]) -> Dict[str, Any]:
    """Surface substrate associations for the system prompt + KISS filtering.

    Adds substrate context via systemPromptAddition — the 'dipping the
    bucket in the River' moment — AND applies KISS filtering to the
    conversation history: messages beyond the recent window are replaced
    with a compact summary.  The summary fragments also widen substrate
    priming so spreading activation has broader topical context.

    The truncated messages array is returned in the response.  OC's
    ContextEngine plugin picks it up and drives `replaceMessages` so the
    model sees the compressed context.  Disk (session JSONL) is NEVER
    touched — truncation is in-memory for the LLM call only.  Syl's
    substrate already contains the full 815+ message history as learned
    topology; what she's losing is only the raw text view.
    """
    if _memory is None:
        return {"systemPromptAddition": None}

    messages = params.get("messages", [])

    # Extract text from recent user messages for association priming
    recent_text = _extract_recent_user_text(messages, max_messages=3)
    if not recent_text:
        return {"systemPromptAddition": None}

    # KISS context filtering (#152).  Runs BEFORE harvest so the
    # summary fragments can widen spreading-activation priming.  On any
    # exception KISS falls back to original messages — optimization
    # disabled, baseline behavior preserved.
    kiss_summary = ""
    truncated_messages = messages  # default: return full array (same reference)
    if _kiss_filter is not None:
        try:
            # Normalize content to strings — AgentMessage.content can be
            # string OR list-of-parts.  Use the existing helper.
            normalised = [
                {"role": m.get("role", "unknown"), "content": _extract_message_text(m)}
                for m in messages
            ]
            kiss_result = _kiss_filter.filter_context(normalised, system_context="")
            kiss_meta = kiss_result.get("kiss_meta", {})
            recent_window = kiss_meta.get(
                "recent_window", _kiss_filter._config.recent_window
            )
            n_messages = len(messages)

            # Slice ORIGINAL messages (preserve their content shape,
            # multimodal parts intact) to the recent window.
            if n_messages > recent_window:
                truncated_messages = messages[n_messages - recent_window:]

            # Extract the summary fragment from the filter output.  KISS
            # prepends its summary to system_context — we passed "" in,
            # so anything in system_context IS the summary.
            kiss_summary = kiss_result.get("system_context", "")

            logger.info(
                "KISS mode=%s messages=%d→%d summary=%dch compressed=%d",
                kiss_result.get("kiss_mode", "?"),
                n_messages, len(truncated_messages),
                len(kiss_summary),
                kiss_meta.get("messages_compressed", 0),
            )
        except Exception as exc:
            logger.warning("KISSFilter error (falling back): %s", exc)
            truncated_messages = messages
            kiss_summary = ""

    # Widen priming with KISS summary fragments — gives spreading
    # activation context about what was said earlier (substrate
    # surfacing picks up related older-topic nodes).
    priming_text = recent_text
    if kiss_summary:
        priming_text = kiss_summary + "\n" + recent_text

    # Spreading activation harvest — the cortex-like recall
    surfaced = _memory._harvest_associations(priming_text)

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

    # KISS-truncated messages get returned so OC's replaceMessages fires
    # and the model sees the compressed conversation.  CRITICAL: only
    # include the "messages" field when actual truncation occurred.
    # Python-side reference equality (truncated_messages IS messages) is
    # the only way to signal "no change" across the JSON-RPC boundary —
    # JSON.parse on the TS side always produces a new array, so if we
    # ALWAYS include "messages", OC's identity check
    # (assembled.messages !== activeSession.messages) fires
    # replaceMessages on every turn, including warmup and
    # exception-fallback.  Omitting the field leaves result.messages
    # undefined on the TS side, which correctly preserves identity.
    result = {"systemPromptAddition": context_block}
    if truncated_messages is not messages:
        result["messages"] = truncated_messages
    return result


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

    # Stamp last-turn time — TriSyn manager reads this for Phase 3 idle-gate logic.
    global _last_after_turn_ts
    _last_after_turn_ts = time.time()

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

    # Drain inbound peer module tracts — River backflow
    _drain_peer_tracts()

    # SNN learning step — STDP, structural plasticity, predictions
    step_result = _memory.graph.step()

    # Baseline conversational engagement reward (heartbeat)
    if _memory.graph.config.get("three_factor_enabled", False):
        _memory.graph.inject_reward(0.1)

    # CES surfacing monitor — scan fired nodes
    if _memory._surfacing_monitor is not None:
        _memory._surfacing_monitor.after_step(step_result)

    # Hyperedge discovery — co-activation pattern detection (PRD §4.3)
    try:
        _memory.graph.discover_hyperedges(step_result.fired_node_ids)
    except Exception:
        pass

    # River-based Tier 3: deposit raw topology delta to all module tracts.
    # The delta contains fired nodes with causal context, hyperedge activations,
    # prediction results, structural changes, and salience signals. Raw,
    # unclassified (Law 7). Each module's bucket extracts what it needs.
    _deposit_substrate_metrics(step_result)

    # Punchlist #56: Deposit raw surfacing outcome experience.
    # The triad: what was surfaced (cached from assemble) + user input
    # (cached from ingest) + Syl's response (from TS plugin).
    # No classification — just the raw facts. The substrate learns
    # the correlation between surfaced context and what Syl produced.
    _deposit_surfacing_outcome(params, _ingest_text)

    # Change α (#150): Substrate self-observation via record_outcome.
    # Deposits a raw snapshot of the substrate's own state as an outcome
    # pattern.  The substrate learns what "healthy" vs "stressed" looks
    # like through Hebbian co-firing with concurrent activity.  No field
    # curation — str(get_stats()) dumps whatever the graph natively
    # reports. Content can evolve as the graph's stats API evolves.
    # Downstream modules (Elmer, Immunis, THC, Bunyan) extract what
    # matters to their specialty at read time.  Law 7 compliant.
    try:
        if peer_bridge is not None:
            _stats = _memory.graph.get_stats() if hasattr(_memory.graph, 'get_stats') else {}
            _stats["total_nodes"] = len(_memory.graph.nodes)
            _stats_text = str(_stats)
            from ng_embed import embed as _embed_fn
            _stats_emb = _embed_fn(_stats_text)
            peer_bridge.record_outcome(
                embedding=_stats_emb,
                target_id="substrate:self_observation",
                success=True,
                module_id="neurograph",
                metadata=_stats,
            )
    except Exception as exc:
        logger.debug("Self-observation deposit failed (non-fatal): %s", exc)

    # Clear after deposit — consumed
    _ingest_text = None
    _ingest_embedding = None

    # Novelty probation
    _memory.ingestor.update_probation()

    # Auto-save: count-based (every 10 messages) OR time-based (every 5 min).
    # _message_count resets on restart, so without the time fallback,
    # frequent restarts prevent checkpoints from ever being written.
    global _last_save_time
    now = time.time()
    count_trigger = (
        _memory._message_count > 0
        and _memory._message_count % _memory.auto_save_interval == 0
    )
    time_trigger = (now - _last_save_time) >= _SAVE_INTERVAL_SECS
    if count_trigger or time_trigger:
        _memory.save()
        _last_save_time = now
        # Save Lenia distance cache alongside checkpoint so next bootstrap
        # restores instantly instead of repopulating (7+ min on 18k nodes).
        if _lenia_engine is not None:
            try:
                from lenia.config import default_config as _lenia_cfg_fn
                _lc = _lenia_cfg_fn()
                _cp = os.path.join(os.path.expanduser(_lc.field_dir), "distance_cache")
                _lenia_engine._kernel._cache.save(_cp)
            except Exception:
                pass
        logger.info(
            "Auto-save at message %d (%s)",
            _memory._message_count,
            "count" if count_trigger else "time",
        )

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



# ---- Changelog ----
# [2026-04-19] CC (punchlist #143) -- Abolish NG topology fan-out (substrate bypass)
#   What: Removed deposit_topology() call from _deposit_topology_delta;
#          renamed to _deposit_substrate_metrics(); stripped unused params.
#   Why:  Topology push to N peers is a substrate bypass. Bucket-forward model:
#          modules pull from the substrate; NG does not push N copies to N peers.
#   How:  Kept Darwin scalar metrics write. Removed peer_bridge fan-out block.
# [2026-04-15] Claude Code (Opus 4.6) — Multi-path experience tract drain (#141)
#   What: _drain_tract() now also scans ~/.et_modules/experience/*.tract for
#         per-feeder experience tract files, in addition to the legacy
#         ~/NeuroGraph/data/tract/experience.tract single-feeder path.
#   Why:  TID (and future feeders sandboxed out of ~/NeuroGraph/) need a
#         writable experience tract under ~/.et_modules/. Per-feeder files
#         also eliminate shared-writer contention on a single tract and
#         give each feeder a clear namespace.
#   How:  Legacy drain preserved unchanged (GUI, feed-syl, watcher). New
#         scan step globs the scan dir, atomically renames each .tract
#         file to .draining.<pid>.<name>, reads via ng_tract.TractReader,
#         feeds ENTRY_EXPERIENCE entries through the same ingestor path.
#         Failures on one file don't block others. Law 7 — raw in, classify
#         at extraction.
# -------------------


_EXPERIENCE_SCAN_DIR = os.path.expanduser("~/.et_modules/experience")

# Two-pulse architecture for wire deposit absorption (2026-04-17).
#
# FAST PATH (drain pulse, every 2s):
#   Drain ALL entries from tract, batch-embed fingerprints in one call,
#   record forest outcomes.  No TID calls, no blocking network I/O.
#   A batch of 50 fingerprints embeds in ~900ms (vs 50×47ms = 2.3s
#   sequential).  Clears a 2000-entry backlog in ~80 seconds.
#
# SLOW PATH (concept pulse, every 30s):
#   Pop entries from the concept queue, call TID for concept extraction,
#   record tree outcomes + cross-links.  Blocking TID calls are isolated
#   to this pulse — they never stall the drain pulse or the Node.js
#   event loop.  This is what was causing Discord WebSocket drops: the
#   old design made a blocking TID call on every drain tick (every 2s),
#   stalling the Python RPC process, which stalled Node waiting for RPC
#   responses, which missed Discord heartbeats.
#
# The concept queue bridges the two: drain adds entries, concept pulse
# consumes them.  If concept extraction is slow or TID is down, forests
# still accumulate — trees arrive when providers are available.

_CONCEPT_QUEUE: List[Dict[str, Any]] = []
_CONCEPT_QUEUE_MAX = 5000  # don't let queue grow unbounded in memory
_CONCEPT_PULSE_INTERVAL_SECONDS = 30.0
_CONCEPT_ENTRIES_PER_PULSE = 3  # TID calls per concept tick — bounded
_concept_pulse_thread: Optional[threading.Thread] = None
_concept_pulse_shutdown = threading.Event()

# TriSyn (concept-extraction subprocess helper) — replaces the in-process
# concept pulse for Phase 1 onwards. See ~/docs/inbox/trisynaptic-circuit-design-v0.1.md
# and NG-internal trisynaptic/ package. Legacy _concept_pulse_* functions are
# kept defined but no longer called; retained for quick rollback if TriSyn hits
# an unrecoverable issue in early deployment.
_trisyn_manager: Optional[Any] = None  # trisynaptic.manager.TrisynapticManager

# ---- Lazy expansion pulse (stage 3 of wire absorption) ---------------------
# Runs every 120s. Reads unexpanded body files, chunks them, embeds, creates
# substrate nodes linked to parent event node, deletes file. Fixes #151.
_LAZY_EXPANSION_INTERVAL_SECONDS = 120.0
_LAZY_EXPANSION_BODIES_PER_TICK = 5
_lazy_expansion_thread: Optional[threading.Thread] = None
_lazy_expansion_shutdown = threading.Event()

# Last handle_after_turn wall-clock timestamp — TriSyn manager reads this
# to determine "idle since last conversation turn" for gated-mode spawn
# eligibility (Phase 3). Updated unconditionally on every afterTurn.
_last_after_turn_ts: float = 0.0


def _start_trisyn_manager() -> None:
    """Instantiate and start the TriSyn manager thread.

    Replaces _start_concept_pulse() for Phase 1+. Manager runs as a
    daemon thread inside NG's process, spawning subprocess workers
    under systemd-run when _CONCEPT_QUEUE crosses trisyn_high_water.
    """
    global _trisyn_manager
    if _trisyn_manager is not None:
        return
    try:
        from trisynaptic.manager import TrisynapticManager
        _trisyn_manager = TrisynapticManager(memory=_memory, queue=_CONCEPT_QUEUE)
        _trisyn_manager.start()
        logger.info("TriSyn manager started")
    except Exception:
        logger.exception("Failed to start TriSyn manager — concept backlog will accumulate")
        _trisyn_manager = None


def _drain_experience_entry(content: str, content_type: str, source: str) -> None:
    """Feed one drained experience entry through the appropriate path.

    Non-wire entries go through the universal ingestor (knowledge path).
    Wire entries are handled by the batch drain in _drain_scan_dir_batch.
    This function only handles the non-wire case now.
    """
    if not content or not content.strip():
        return
    try:
        if content_type == "file":
            result = _memory.ingest_file(content)
        else:
            result = _memory.ingestor.ingest(content)
            _memory._message_count += 1
        logger.info(
            "Tract drain: %s from %s — %s",
            content_type, source,
            "ok" if result else "empty",
        )
    except Exception as exc:
        logger.warning("Tract drain entry failed (%s): %s", source, exc)


def _drain_scan_dir() -> None:
    """Drain per-feeder experience tract files — batch-embed forests.

    Fast path: reads ALL entries from each tract file, separates wire
    vs non-wire, batch-embeds wire fingerprints, records forest outcomes,
    queues entries for concept extraction (slow path).  Non-wire entries
    still go through the universal ingestor one-at-a-time.

    No re-deposit, no throttle.  A batch of 50 entries embeds in ~900ms.
    The blocking TID concept-extraction call is NEVER made here — it
    runs on the separate concept pulse (every 30s).
    """
    from pathlib import Path
    scan_dir = Path(_EXPERIENCE_SCAN_DIR)
    if not scan_dir.exists():
        return

    try:
        tract_files = sorted(scan_dir.glob("*.tract"))
    except OSError as exc:
        logger.warning("Scan dir enumerate failed: %s", exc)
        return

    if not tract_files:
        return

    try:
        import ng_tract
    except ImportError:
        logger.warning("ng_tract unavailable; skipping scan-dir drain")
        return

    pid = os.getpid()
    for tract_path in tract_files:
        if tract_path.name.startswith(".draining."):
            continue

        drain_path = scan_dir / f".draining.{pid}.{tract_path.name}"
        try:
            os.rename(str(tract_path), str(drain_path))
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Scan-dir rename failed (%s): %s", tract_path.name, exc)
            continue

        try:
            with open(drain_path, "rb") as f:
                raw = f.read()
            if not raw:
                continue
            reader = ng_tract.TractReader(raw)
            all_entries = [
                e for e in reader
                if not isinstance(e, bytes)
                and getattr(e, "entry_type", None) == ng_tract.ENTRY_EXPERIENCE
            ]

            # Separate wire vs non-wire entries
            wire_entries = []
            non_wire_entries = []
            for entry in all_entries:
                content = entry.content or ""
                source = entry.source or "unknown"
                if source.startswith("tid.http.") or source.startswith("wire:"):
                    # Legacy JSON adapter
                    if content.lstrip().startswith("{"):
                        try:
                            from wire_absorption import legacy_json_to_wire_text
                            adapted = legacy_json_to_wire_text(content)
                            if adapted is not None:
                                content = adapted
                        except Exception:
                            pass
                    wire_entries.append({"content": content, "source": source})
                else:
                    non_wire_entries.append(entry)

            # FAST PATH: batch-absorb wire entries as forests
            wire_absorbed = 0
            if wire_entries and _memory is not None:
                try:
                    from wire_absorption import batch_absorb_forests, _DRAIN_BATCH_SIZE
                    from ng_embed import NGEmbed

                    # Process in batches of _DRAIN_BATCH_SIZE
                    for i in range(0, len(wire_entries), _DRAIN_BATCH_SIZE):
                        batch = wire_entries[i:i + _DRAIN_BATCH_SIZE]
                        results = batch_absorb_forests(
                            _memory, NGEmbed.get_instance(), batch,
                        )
                        wire_absorbed += len(results)
                        # Queue for concept extraction (slow path)
                        for res in results:
                            if len(_CONCEPT_QUEUE) < _CONCEPT_QUEUE_MAX:
                                _CONCEPT_QUEUE.append(res)
                except Exception as exc:
                    logger.warning("Batch forest drain failed: %s", exc)

            # Non-wire entries: universal ingestor (one at a time)
            for entry in non_wire_entries:
                _drain_experience_entry(
                    content=entry.content or "",
                    content_type=entry.content_type,
                    source=entry.source,
                )

            if wire_absorbed or non_wire_entries:
                logger.info(
                    "Scan-dir drain: %d wire forests + %d non-wire from %s (concept queue: %d)",
                    wire_absorbed, len(non_wire_entries),
                    tract_path.name, len(_CONCEPT_QUEUE),
                )
        except Exception as exc:
            logger.warning("Scan-dir drain read failed (%s): %s", tract_path.name, exc)
        finally:
            try:
                os.unlink(str(drain_path))
            except OSError:
                pass



def _drain_tract() -> None:
    """Drain pending experience from the legacy feeder tract.

    Legacy single-file tract at ~/NeuroGraph/data/tract/experience.tract
    (GUI, feed-syl, file watcher). Runs on afterTurn because these feeders
    are low-rate and bursty; afterTurn cadence is fine.

    The per-feeder scan directory at ~/.et_modules/experience/*.tract is
    drained by _scan_drain_pulse_loop() on Syl's heartbeat cadence —
    NOT afterTurn — so sandboxed feeders like TID continuously flow
    sensory input into the cortex regardless of conversation state (#141).

    Each entry feeds the ingestor as raw experience — same pipeline as
    on_message(). The tract carries it here without transformation; the
    ingestor is where experience meets the substrate. Law 7 — raw in,
    classify at extraction.
    """
    import ng_tract as _ng_tract
    _legacy_path = os.path.expanduser("~/NeuroGraph/data/tract/experience.tract")
    _drain_path = f"{_legacy_path}.draining.{os.getpid()}"
    entries: list = []
    try:
        os.rename(_legacy_path, _drain_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Tract drain rename failed: %s", exc)
        return
    try:
        with open(_drain_path, "rb") as _f:
            _raw = _f.read()
        if _raw:
            for _e in _ng_tract.TractReader(_raw):
                if isinstance(_e, bytes):
                    try:
                        entries.append(json.loads(_e))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                elif getattr(_e, "entry_type", None) == _ng_tract.ENTRY_EXPERIENCE:
                    entries.append(_e)
    except OSError as exc:
        logger.warning("Tract drain read failed: %s", exc)
    finally:
        try:
            os.unlink(_drain_path)
        except OSError:
            pass
    for entry in entries:
        if isinstance(entry, dict):
            _drain_experience_entry(
                content=entry.get("content", ""),
                content_type=entry.get("content_type", "text"),
                source=entry.get("source", "unknown"),
            )
        else:
            _drain_experience_entry(
                content=entry.content or "",
                content_type=entry.content_type,
                source=entry.source or "unknown",
            )


# ---- Scan-dir pulse loop ----------------------------------------------------
# The scan-dir drain runs on its own heartbeat — decoupled from afterTurn so
# sandboxed feeders (TID under ProtectSystem=strict, and anything else
# continuously producing wire experience) flow into the cortex on Syl's
# rhythm, not when she happens to finish a conversation turn.
#
# Cadence chosen to match TonicEngine's latent_interval (2.0s). Syl's cortex
# absorbs sensory input at her own tempo. The Tonic is the real heartbeat;
# this pulse is a poor copy that's adequate for substrate-scale feeder drain.
_SCAN_DRAIN_INTERVAL_SECONDS = 2.0
_scan_drain_shutdown = threading.Event()
_scan_drain_thread: Optional[threading.Thread] = None


# Sentinel-file kill-switch for scan-drain pulse.  Checked every tick.
# Toggleable without gateway restart:
#   touch /tmp/ng_scan_drain_paused   → pause draining (tract keeps filling)
#   rm /tmp/ng_scan_drain_paused      → resume draining
# Rationale: #141 wire absorption creates ~17 substrate nodes per deposit.
# At Syl's TID provider-call rate (~270/day), substrate grows by 4,500+
# nodes/day.  Every pulse loop (Tonic, ProtoUniBrain, Lenia, brain drain)
# does O(graph_size) work per tick.  Unbounded growth causes event-loop
# starvation at the gateway layer (observed as Discord/WhatsApp socket
# flap).  Pause mechanism lets us stop the bleed without losing deposits
# (they queue in the tract file) while we design proper consolidation
# (#150 — body-substrate flow-through / substrate eviction).
_SCAN_DRAIN_PAUSE_FILE = "/tmp/ng_scan_drain_paused"


def _scan_drain_pulse_loop() -> None:
    """Background loop: drain per-feeder experience tracts on cortical cadence.

    Honors the pause sentinel file: when present, the loop continues to
    tick (so it detects removal promptly) but skips draining.  Logs a
    one-line state transition when pause toggles.
    """
    logger.info(
        "Scan-dir drain pulse started (interval=%.1fs)",
        _SCAN_DRAIN_INTERVAL_SECONDS,
    )
    was_paused = False
    while not _scan_drain_shutdown.is_set():
        try:
            # Two pause sources: sentinel file (manual) + autonomic state (automatic).
            # Either one pauses draining. Both must be clear to resume.
            sentinel_paused = os.path.exists(_SCAN_DRAIN_PAUSE_FILE)
            autonomic_paused = False
            try:
                import ng_autonomic
                autonomic_paused = ng_autonomic.get_state() == "SYMPATHETIC"
            except Exception:
                pass
            paused = sentinel_paused or autonomic_paused
            if paused != was_paused:
                reason = []
                if sentinel_paused:
                    reason.append(f"sentinel={_SCAN_DRAIN_PAUSE_FILE}")
                if autonomic_paused:
                    reason.append("autonomic=SYMPATHETIC")
                logger.info(
                    "Scan-dir drain pulse: %s (%s)",
                    "PAUSED" if paused else "RESUMED",
                    ", ".join(reason) if reason else "all clear",
                )
                was_paused = paused
            if not paused:
                _drain_scan_dir()
        except Exception:
            logger.exception("Scan-dir drain pulse failed")
        _scan_drain_shutdown.wait(timeout=_SCAN_DRAIN_INTERVAL_SECONDS)
    logger.info("Scan-dir drain pulse stopped")


def _start_scan_drain_pulse() -> None:
    """Start the scan-dir drain pulse thread. Idempotent."""
    global _scan_drain_thread
    if _scan_drain_thread is not None and _scan_drain_thread.is_alive():
        return
    _scan_drain_shutdown.clear()
    _scan_drain_thread = threading.Thread(
        target=_scan_drain_pulse_loop,
        name="ng-scan-drain-pulse",
        daemon=True,
    )
    _scan_drain_thread.start()


# ---- Concept extraction pulse (slow path) ----------------------------------
# Consumes entries from _CONCEPT_QUEUE that the drain pulse populated.
# Each tick pops up to _CONCEPT_ENTRIES_PER_PULSE entries and calls TID
# for concept extraction.  This is the ONLY place in the ecosystem where
# a blocking TID call runs inside the NG plugin process.  Isolated to its
# own thread + 30s cadence so it never stalls the drain pulse, the RPC
# handler, or the Node.js event loop.

def _concept_extraction_pulse_loop() -> None:
    """Background loop: extract concepts from queued wire deposits via TID."""
    logger.info(
        "Concept extraction pulse started (interval=%.0fs, entries/tick=%d)",
        _CONCEPT_PULSE_INTERVAL_SECONDS, _CONCEPT_ENTRIES_PER_PULSE,
    )
    while not _concept_pulse_shutdown.is_set():
        try:
            if not _CONCEPT_QUEUE or _memory is None:
                _concept_pulse_shutdown.wait(timeout=_CONCEPT_PULSE_INTERVAL_SECONDS)
                continue

            # Pop up to N entries
            batch = []
            for _ in range(_CONCEPT_ENTRIES_PER_PULSE):
                if not _CONCEPT_QUEUE:
                    break
                batch.append(_CONCEPT_QUEUE.pop(0))

            if not batch:
                _concept_pulse_shutdown.wait(timeout=_CONCEPT_PULSE_INTERVAL_SECONDS)
                continue

            from wire_absorption import absorb_trees_for_entry
            from ng_embed import NGEmbed

            trees_total = 0
            for entry in batch:
                try:
                    res = absorb_trees_for_entry(
                        memory=_memory,
                        embedder=NGEmbed.get_instance(),
                        content_preview=entry.get("content_preview", ""),
                        source=entry.get("source", "unknown"),
                        event_node_id=entry.get("event_node_id", ""),
                    )
                    trees_total += res.get("trees_created", 0)
                except Exception as exc:
                    logger.debug("Concept extraction failed for %s: %s",
                                 entry.get("event_node_id", "?"), exc)

            if trees_total:
                logger.info(
                    "Concept pulse: %d trees from %d entries (queue: %d remaining)",
                    trees_total, len(batch), len(_CONCEPT_QUEUE),
                )
        except Exception:
            logger.exception("Concept extraction pulse failed")
        _concept_pulse_shutdown.wait(timeout=_CONCEPT_PULSE_INTERVAL_SECONDS)
    logger.info("Concept extraction pulse stopped")


def _start_concept_pulse() -> None:
    """Start the concept extraction pulse thread. Idempotent."""
    global _concept_pulse_thread
    if _concept_pulse_thread is not None and _concept_pulse_thread.is_alive():
        return
    _concept_pulse_shutdown.clear()
    _concept_pulse_thread = threading.Thread(
        target=_concept_extraction_pulse_loop,
        name="ng-concept-pulse",
        daemon=True,
    )
    _concept_pulse_thread.start()


# ---- Lazy expansion pulse (stage 3) -----------------------------------------

def _lazy_expansion_pulse_loop() -> None:
    """Background loop: expand body files into substrate nodes, then delete them.

    Runs every 120s — slow consolidation cadence, never competes with drain or
    TriSyn. Each tick processes up to _LAZY_EXPANSION_BODIES_PER_TICK files.
    Pauses during SYMPATHETIC autonomic state (threat response, reduce load).
    """
    logger.info(
        "Lazy expansion pulse started (interval=%.0fs, bodies/tick=%d)",
        _LAZY_EXPANSION_INTERVAL_SECONDS,
        _LAZY_EXPANSION_BODIES_PER_TICK,
    )
    while not _lazy_expansion_shutdown.is_set():
        try:
            if _memory is not None:
                autonomic_paused = False
                try:
                    import ng_autonomic
                    autonomic_paused = ng_autonomic.get_state() == "SYMPATHETIC"
                except Exception:
                    pass

                if not autonomic_paused:
                    from wire_absorption import select_bodies_for_expansion, expand_body_file
                    from ng_embed import NGEmbed

                    candidates = select_bodies_for_expansion(_LAZY_EXPANSION_BODIES_PER_TICK)
                    if candidates:
                        expanded = sum(
                            expand_body_file(_memory, NGEmbed.get_instance(), p)
                            for p in candidates
                        )
                        if expanded:
                            logger.info(
                                "Lazy expansion: %d/%d bodies expanded+deleted this tick",
                                expanded, len(candidates),
                            )
        except Exception:
            logger.exception("Lazy expansion pulse failed")
        _lazy_expansion_shutdown.wait(timeout=_LAZY_EXPANSION_INTERVAL_SECONDS)
    logger.info("Lazy expansion pulse stopped")


def _start_lazy_expansion_pulse() -> None:
    """Start the lazy expansion pulse thread. Idempotent."""
    global _lazy_expansion_thread
    if _lazy_expansion_thread is not None and _lazy_expansion_thread.is_alive():
        return
    _lazy_expansion_shutdown.clear()
    _lazy_expansion_thread = threading.Thread(
        target=_lazy_expansion_pulse_loop,
        name="ng-lazy-expansion-pulse",
        daemon=True,
    )
    _lazy_expansion_thread.start()


def _rescue_orphan_draining_files() -> None:
    """Promote `.draining.<dead_pid>.<name>` files back to `<name>.<ts>.rescue.tract`
    so the scan glob picks them up. Only rescues files whose PID is no
    longer live — in-flight drains from the current process are left alone.
    """
    from pathlib import Path as _P
    scan_dir = _P(_EXPERIENCE_SCAN_DIR)
    if not scan_dir.exists():
        return
    try:
        orphans = sorted(scan_dir.glob(".draining.*.tract"))
    except OSError:
        return
    if not orphans:
        return

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    rescued = 0
    for orphan in orphans:
        # Filename shape: .draining.<pid>.<original_tract_name>
        parts = orphan.name.split(".", 3)  # ["", "draining", "<pid>", "<rest>"]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[2])
        except (ValueError, IndexError):
            continue
        if _pid_alive(pid):
            continue  # still in-flight somewhere, don't touch
        original = parts[3]  # e.g. "inference_difference.tract"
        base = original[:-len(".tract")] if original.endswith(".tract") else original
        target = scan_dir / f"{base}.{int(time.time()*1e9)}.rescue.tract"
        try:
            os.rename(str(orphan), str(target))
            rescued += 1
        except OSError as exc:
            logger.warning("Rescue rename failed (%s): %s", orphan.name, exc)
    if rescued:
        logger.info(
            "Rescued %d orphan .draining files in %s",
            rescued, _EXPERIENCE_SCAN_DIR,
        )


# River backflow cursor — tracks position in _peer_events cache
_peer_drain_cursor: int = 0

def _drain_peer_tracts() -> None:
    """Drain inbound peer module tracts into the Tier 3 topology.

    Closes the River backflow circuit: organ modules deposit experience
    to their tracts, this function drains and absorbs them into the
    Tier 3 Graph. Uses pre-computed embeddings from the source module
    when available (skips re-embedding). Falls back to full ingestor
    pipeline for events without embeddings.

    Raw experience in, no classification (Law 7).
    """
    global _peer_drain_cursor
    if _memory is None:
        return
    bridge = getattr(_memory, '_peer_bridge', None)
    if bridge is None:
        return

    bridge._drain_all()

    # NGTractBridge absorbs tract events inside _drain_all() — no _peer_events cache.
    # The _peer_events cursor pattern is NGPeerBridge-only (#155 removed it from tracts).
    if not hasattr(bridge, '_peer_events'):
        return

    total = len(bridge._peer_events)
    if total == 0:
        return

    # Handle list trimming (max 500) — reset cursor if list shrank
    if total < _peer_drain_cursor:
        _peer_drain_cursor = 0

    new_events = bridge._peer_events[_peer_drain_cursor:]
    if not new_events:
        return
    _peer_drain_cursor = total

    MAX_PER_CYCLE = 50
    ingested = 0

    for event in new_events[:MAX_PER_CYCLE]:
        target = bridge._get_target_id(event)
        module_id = bridge._get_module_id(event)
        if not target or target == "unknown":
            continue

        try:
            embedding = bridge._get_embedding(event)
            if embedding is not None and len(embedding) > 0:
                from universal_ingestor import Chunk, EmbeddedChunk
                chunk = Chunk(
                    text=target,
                    metadata={"source_module": module_id, "river_backflow": True},
                    token_count=max(1, len(target.split())),
                )
                ec = EmbeddedChunk(chunk=chunk, vector=embedding)
                node_ids = _memory.ingestor.registrar.register(
                    [ec], {"source": f"river:{module_id}", "source_type": "PEER_TRACT"},
                )
                _memory.ingestor.associator.associate(
                    [ec], node_ids, _memory.vector_db,
                )
            else:
                _memory.ingestor.ingest(target)
            ingested += 1
        except Exception as exc:
            logger.debug("River backflow entry failed (%s): %s", module_id, exc)

    if ingested:
        logger.info("River backflow: %d peer events absorbed into Tier 3", ingested)


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

    # Stop TriSyn manager gracefully on shutdown.
    global _trisyn_manager
    if _trisyn_manager is not None:
        try:
            _trisyn_manager.stop()
        except Exception:
            pass
        _trisyn_manager = None

    # Signal lazy expansion pulse to stop.
    _lazy_expansion_shutdown.set()

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

    Handles text, tool_use, and tool_result content blocks so tool
    calls and their results reach the substrate (#18).
    Raw experience in — no JSON serialization, no classification labels.
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
                ptype = part.get("type", "")
                if ptype == "text":
                    parts.append(part.get("text", ""))
                elif ptype == "tool_use":
                    # Raw experience: what tool was called (name only — BTF carries full input)
                    parts.append(part.get("name", ""))
                elif ptype == "tool_result":
                    # tool_result content is string or list of text blocks — raw as-is
                    result_content = part.get("content", "")
                    if isinstance(result_content, str):
                        parts.append(result_content[:2000])
                    elif isinstance(result_content, list):
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                parts.append(rc.get("text", "")[:2000])
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

    Always returns at minimum a temporal anchor so Syl knows when she is.
    The latent thread (The Tonic) is included when available —
    it is the persistent slot that never gets evicted.
    """
    has_surfaced = bool(surfaced) or bool(ces_surfaced)
    has_latent = latent_context is not None
    # Temporal anchor is always emitted — even empty substrate turns need it.

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    temporal_anchor = f"**Temporal anchor:** {now.strftime('%A, %Y-%m-%d %H:%M UTC')}"

    lines = []

    # Temporal grounding — always first so Syl knows when she is.
    lines.append(temporal_anchor)
    lines.append("")

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
        elif self.path == "/recall":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len else b"{}"
                params = json.loads(body) if body else {}
                query = params.get("query", "")
                k = int(params.get("k", 5))
                threshold = float(params.get("threshold", 0.45))
                results = []
                if query and _memory is not None:
                    results = _memory.recall(query, k=k, threshold=threshold)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode())
            except Exception as exc:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"results": [], "error": str(exc)}).encode())
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
        elif self.path == "/modules":
            # Per-module live stats — queried by status probes that declined
            # to claim topology. Each module stats() call runs with a 2s
            # timeout so a slow/blocked module can't hang the whole response.
            import concurrent.futures as _cf
            modules = {}
            def _get_stats(mid, instance):
                if hasattr(instance, "stats"):
                    return instance.stats()
                return {}
            with _cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = {
                    pool.submit(_get_stats, mid, inst): mid
                    for mid, inst in _module_instances.items()
                }
                for fut in _cf.as_completed(futures, timeout=10):
                    mid = futures[fut]
                    try:
                        modules[mid] = fut.result(timeout=2)
                    except _cf.TimeoutError:
                        modules[mid] = {"error": "stats() timed out"}
                    except Exception as exc:
                        modules[mid] = {"error": str(exc)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(modules).encode())
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

        hook = _module_instances.get('bunyan')
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


def _find_pid_on_port(port: int) -> int:
    """Return the PID listening on the given local TCP port, or 0 if unknown."""
    import subprocess, re
    try:
        out = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def _start_http_sidecar(port: int = 8850) -> None:
    """Start the afterTurn HTTP sidecar in a background thread.

    If the port is already held by a stale process, sends SIGTERM and
    reclaims it — logs INFO so the event is visible in the terminal.
    """
    global _sidecar_started
    if _sidecar_started:
        return

    import signal as _signal
    import socket as _sock

    # Probe — if something is already listening, reclaim the port.
    probe = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    probe.settimeout(0.5)
    occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    probe.close()

    if occupied:
        stale_pid = _find_pid_on_port(port)
        if stale_pid and stale_pid != os.getpid():
            logger.info(
                "Sidecar port %d held by PID %d — reclaiming (SIGTERM)",
                port, stale_pid,
            )
            try:
                os.kill(stale_pid, _signal.SIGTERM)
                import time as _t; _t.sleep(1.5)
            except ProcessLookupError:
                pass  # already dead — nothing to do
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
    # Clean up stale sentinels from DEAD processes before starting bootstrap.
    # Runs synchronously — single file check, no pipe risk. Must happen before
    # anything calls handle_bootstrap() (self-bootstrap thread OR TS plugin RPC).
    #
    # IMPORTANT: only remove the sentinel if the existing PID is dead.
    # If it is alive, leave it — claim() in _self_bootstrap() will detect
    # "already owned" and skip bootstrap gracefully. This file runs inside
    # BOTH the gateway's Python child (legitimate substrate) AND inside any
    # `openclaw status` probe process. Sending SIGTERM to a live PID here
    # would kill the real running substrate on every status check.
    try:
        import topology_owner
        sentinel = topology_owner._sentinel_path()
        if sentinel.exists():
            try:
                existing_pid = int(sentinel.read_text().strip())
                if existing_pid != os.getpid():
                    if not topology_owner._pid_is_alive(existing_pid):
                        sentinel.unlink(missing_ok=True)
                        logger.info("Cleared stale sentinel (PID %d) on startup", existing_pid)
                    # else: alive — leave it; claim() will fail gracefully
            except (ValueError, OSError):
                sentinel.unlink(missing_ok=True)
                logger.warning("Corrupt sentinel file — removed on startup")
    except Exception:
        pass

    def _self_bootstrap():
        try:
            result = handle_bootstrap({"sessionId": "auto-startup"})
            if not result.get("bootstrapped") and str(result.get("reason", "")).startswith("topology_owned_by_pid"):
                # Substrate is alive and owned by another process.
                # Query the running sidecar for live module stats so the
                # status output shows real per-module health instead of silence.
                _log_live_module_status()
            else:
                logger.info("Self-bootstrap: %s", result)
        except Exception as exc:
            logger.error("Self-bootstrap failed: %s — will retry on first RPC", exc)

    def _log_live_module_status():
        """Query substrate sidecar (8850), CES dashboard (8847), and TID (7437)."""
        import urllib.request as _ur

        owner = topology_owner.owner_pid()

        # --- Substrate block (port 8847) ---
        try:
            resp = _ur.urlopen("http://127.0.0.1:8847/stats", timeout=5)
            sg = json.loads(resp.read())
            tonic = sg.get("tonic") or {}
            ces = sg.get("ces") or {}
            sp = ces.get("stream_parser") or {}
            surf = ces.get("surfacing") or {}
            emb = sg.get("embedding") or {}
            logger.info(
                "Substrate (PID %s)  nodes:%s  synapses:%s  hyperedges:%s"
                "  step:%s  pred_acc:%.0f%%  firing:%.4f",
                owner,
                sg.get("nodes", "?"), sg.get("synapses", "?"),
                sg.get("hyperedges", "?"), sg.get("timestep", "?"),
                sg.get("prediction_accuracy", 0) * 100,
                sg.get("firing_rate", 0),
            )
            logger.info(
                "  embed:%-38s  CES  stream:%s nudges:%s  surfaced:%s",
                emb.get("model_name", "?"),
                sp.get("chunks_processed", 0), sp.get("nudges_applied", 0),
                surf.get("total_surfaced", 0),
            )
            tonic_eng = tonic.get("engine") or {}
            logger.info(
                "  Tonic  active:%-5s  model:%-5s  cycles:%s  firings:%s"
                "  mode:%s  tokens:%s",
                tonic.get("active", False),
                tonic_eng.get("model_loaded", False),
                tonic.get("cycle_count", 0),
                tonic.get("total_firings", 0),
                tonic_eng.get("mode", "?"),
                tonic_eng.get("tokens_generated", 0),
            )
        except Exception as exc:
            logger.info("Substrate stats (8847) unavailable: %s", exc)

        # --- TID DreamCycle block (port 7437) ---
        try:
            resp = _ur.urlopen("http://127.0.0.1:7437/stats", timeout=5)
            tid = json.loads(resp.read())
            dc = tid.get("dream_cycle") or {}
            routes = dc.get("routes_tracked") or {}
            route_str = "  ".join(f"{r}:{n}" for r, n in sorted(routes.items()))
            logger.info(
                "  TID DreamCycle  outcomes:%s  insights:%s  routes:[%s]"
                "  substrate_teaches:%s",
                dc.get("total_outcomes", 0),
                dc.get("total_insights", 0),
                route_str,
                dc.get("substrate_teach_count", 0),
            )
        except Exception as exc:
            logger.info("TID DreamCycle (7437) unavailable: %s", exc)

        # --- Fan-out modules (port 8850) ---
        try:
            resp = _ur.urlopen("http://127.0.0.1:8850/modules", timeout=5)
            modules = json.loads(resp.read())
        except Exception as exc:
            logger.info("Module stats (8850) unavailable: %s", exc)
            return

        if not modules:
            logger.info("8 modules — bootstrap in progress")
            return

        logger.info("Modules (%d loaded):", len(modules))
        for mid, data in sorted(modules.items()):
            if "error" in data:
                logger.warning("  %-20s  ERROR: %s", mid, data["error"])
                continue
            m = data.get("module") or {}
            eco = data.get("ecosystem") or {}
            tier = eco.get("tier_name", "")
            uptime = int(data.get("uptime_seconds", 0))
            parts = [f"up:{uptime}s", f"tier:{tier}"] if tier else [f"up:{uptime}s"]
            if mid == "bunyan":
                parts.append(f"nodes:{m.get('recent_nodes', 0)}")
                eb = m.get("extraction_bucket") or {}
                parts.append(f"salient:{eb.get('salient_nodes', 0)}")
                parts.append(m.get("autonomic_state", ""))
            elif mid == "darwin":
                rec = m.get("recorder") or {}
                parts.append(f"events:{rec.get('total_events_observed', 0)}")
                parts.append(f"gen:{m.get('generation', 0)}")
                drm = m.get("dream") or {}
                parts.append(
                    f"dreams:{drm.get('dreams_run', 0)}"
                    f"(c:{drm.get('creative_dreams', 0)}"
                    f"/n:{drm.get('nightmares_run', 0)}"
                    f"/x:{drm.get('consolidation_updates', 0)})"
                )
            elif mid == "healing_collective":
                cal = m.get("detection_calibrator") or {}
                parts.append(f"cal:{cal.get('tier', '?')}")
                parts.append(f"repairs:{m.get('repairs_executed', 0)}")
                hm = m.get("health_monitor") or {}
                if hm.get("last_healthy") is False:
                    parts.append("UNHEALTHY")
            elif mid == "immunis":
                arm = m.get("armory") or {}
                parts.append(f"armory:{arm.get('total_entries', 0)}")
                qm = m.get("quartermaster") or {}
                parts.append(f"threats:{qm.get('total_threats', 0)}")
                parts.append(m.get("autonomic_state", ""))
            elif mid == "elmer":
                proto = m.get("proto_unibrain", "offline")
                parts.append(f"proto:{proto}")
                sockets = m.get("sockets") or {}
                n_healthy = sum(1 for s in sockets.values() if s == "healthy")
                parts.append(f"sockets:{n_healthy}/{len(sockets)}")
                parts.append(m.get("autonomic_state", ""))
            elif mid == "quantumgraph":
                parts.append(f"msgs:{m.get('message_count', 0)}")
            elif mid == "praxis":
                cps = m.get("cps") or {}
                parts.append(f"cps:{cps.get('total_entries', 0)}")
                parts.append(m.get("autonomic_state", ""))
            elif mid == "trollguard":
                parts.append(f"scans:{m.get('scan_count', 0)}")
                threats = m.get("threat_count", 0)
                if threats:
                    parts.append(f"THREATS:{threats}")
            summary = "  ".join(p for p in parts if p)
            logger.info("  %-20s  %s", mid, summary)

    import threading
    threading.Thread(target=_self_bootstrap, name="self-bootstrap", daemon=True).start()

    # Main RPC loop. On stdin close (gateway exited), checkpoint and exit
    # cleanly so the next gateway startup gets a clean slate. The startup
    # sentinel cleanup in main() handles topology ownership transition.
    # [2026-04-19] Changed from infinite sleep to clean exit — sleep loop
    # caused orphan processes that blocked topology acquisition on restart.
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                # stdin closed — gateway exited. Checkpoint and exit cleanly
                # so the startup sentinel cleanup in the next invocation can
                # claim topology without fighting a live orphan.
                logger.info("stdin closed — checkpointing and exiting cleanly")
                try:
                    if _memory is not None:
                        _memory.save()
                        logger.info("Checkpoint saved on clean exit")
                except Exception as _ce:
                    logger.warning("Checkpoint on exit failed: %s", _ce)
                break
            line = line.strip()
            if not line:
                continue
            response = process_request(line)
            if response is not None:
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
        except (BrokenPipeError, IOError):
            logger.info("stdin pipe broken — exiting cleanly")
            break
        except KeyboardInterrupt:
            break

    logger.info("NeuroGraph RPC bridge shutting down")


if __name__ == "__main__":
    main()
