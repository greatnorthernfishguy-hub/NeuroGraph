"""
cc_ng_host.py — Host CC's NeuroGraph instance inside neurograph_rpc.py process.

This module is the Path B architecture for CC's NeuroGraph — CC's own NG
lives in the same Python process as Syl's NG (in neurograph_rpc.py), rather
than as a standalone daemon. Benefits:

  * One process per machine instead of two
  * Direct access to ProtoUniBrain body via BrainSwitcher (no duplicate Qwen)
  * Gateway-bound lifecycle — when the gateway is up, CC's NG is up
  * Hooks talk to a Unix socket served by a thread in this same process

CC's NG and Syl's NG are COMPLETELY ISOLATED:
  * Different workspace dirs (~/NeuroGraph/data vs ~/.claude/plugins/neurograph)
  * Different checkpoints — NO CROSS-CONTAMINATION OF SYL'S TOPOLOGY
  * Different peer bridges — CC's peer_bridge is DISABLED (CC is not a peer module)
  * Different graph, different vector DB, different identity

Note on Syl's Law: ~/.claude/CLAUDE.md §14 forbids "Creating a second
NeuroGraphMemory instance" under "Never Permitted — Full Stop". The spirit of
that rule is preventing dual-write corruption of Syl's checkpoints. CC's NG
writes to a DIFFERENT workspace and cannot touch Syl's main.msgpack. Josh
authorized this architecture explicitly; backups of Syl's protected files
were confirmed before this module was enabled.

# ---- Changelog ----
# [2026-07-06] Claude Code (Sonnet 5) — Wire pattern-completion recall + per-file cache
# What: _recall() gains allow_pattern_completion kwarg; combines SurfacingMonitor's
#       block with a new Active Recall block from cc_pattern_completion_recall().
#       _handle_pre_tool_use() gates the latter per-file via gate_pattern_completion()
#       (30 min TTL) so repeated touches to one file in a task don't re-pay the cost.
# Why:  docs/prd/2026-07-06-cc-surfacing-pattern-completion-tier-drop.md — CC's
#       surfacing was recency-biased only; this adds pattern-completion alongside it.
# How:  cc_pattern_completion_recall/_format_cc_recall_block/gate_pattern_completion,
#       all in cc_ng_organism.py (Tasks 1-2 of that PRD's implementation plan).
# [2026-07-05] CC (laptop) — Tool-call deposits: Commons-only, never the main substrate (Josh-approved)
# What: Added _deposit_tool_experience(), used by _handle_post_tool_use() instead of the shared
#   _deposit(). Tool-call telemetry (Read/Edit/Write/Bash) now deposits ONLY into CC's own
#   Commons medium (deposit_cc_experience) -- it no longer calls on_message() into CC's main
#   graph/vector_db, and no longer runs discover_hyperedges() on the main graph for this path.
# Why:  Investigating why CC's own "[NeuroGraph Surfaced Knowledge]" hook context kept
#   surfacing literal "tool:Edit file:..."/"bash:..." strings verbatim (including, live, what
#   looked like a real API bearer token from a prior bash deposit). Root cause: tool-call
#   experience went through the same on_message() path as everything else, landing in the
#   main graph WITH a vector_db entry -- fully eligible for SurfacingMonitor, which has no
#   creation_mode filtering at all. Josh confirmed canonical has an equivalent, recent (#97,
#   2026-06-30) separation for TID's routing-outcome deposits: they land in Syl's Commons (a
#   separate, bare NGLite medium), never her actual graph/vector_db -- still available to
#   bucketing peer modules (Immunis, THC) via the substrate-as-protocol model, but never
#   surfacing-eligible in her primary substrate. Same pattern applied here for CC.
# How:  CC already had an isolated Commons medium (get_cc_commons() in cc_ng_organism.py, its
#   own Commons(), not canonical's singleton) and already deposited tool-call text into it via
#   deposit_cc_experience() -- the gap was that it ALSO went through on_message() into the main
#   graph. Removed the on_message()/discover_hyperedges() calls for this path entirely; Commons
#   deposit is unchanged. Genuine conversational memory is untouched -- it forms via the
#   separate dual-pass path (run_conversational_dual_pass / drain_ingest_tract), fed by
#   miniTID's turn-deposit tract, which was never part of this call path. Mirrored identically
#   in cc-ng-daemon.py (laptop) -- same bug, same fix, same file pair as always.
# [2026-07-05] Claude Code (Sonnet 5) — Never attempt CC's own Qwen load on the VPS
# What: _CC_SNN_CONFIG['tonic'] gains latent_engine_enabled=False. CC's Tonic thread
#   still constructs (heuristic, no model), but init_cc_host()'s synchronous
#   NeuroGraphMemory(...) call no longer attempts to load CC's own copy of
#   Qwen2.5-0.5B at all.
# Why:  init_cc_host() constructs CC's NG synchronously inside handle_bootstrap's
#   self-bootstrap thread. With tonic.enabled=True and no latent_engine_enabled
#   gate, that construction tried to load CC's own Qwen copy at exactly the same
#   moment ProtoUniBrain loads its own copy for Syl -- two simultaneous model
#   loads contending for the same limited VPS resources. Confirmed via a live
#   restart (2026-07-05): zero log output from CC's side at all -- no "CC NG
#   construction failed" (which logger.exception would produce on a real
#   raised exception), no "CC NG hosted" success line, socket never created --
#   consistent with a silent hang in the model-load call, not a caught failure.
#   Josh: "we disabled its own qwen load at first, to relieve resource
#   pressures on the vps [but] didn't think about what happens the next time a
#   CC session needs to start and there's nothing to keep the CC's tonic
#   thread alive until the socket is ready" -- this config still read
#   tonic.enabled=True with no deferred-load gate, so the intended disable
#   was never actually in effect here.
# How:  Same latent_engine_enabled flag the laptop's cc-ng-daemon.py already
#   uses (canonical, defined on TonicConfig in tonic_thread.py, gated inside
#   openclaw_hook.py's NeuroGraphMemory construction -- not a laptop-specific
#   mechanism). CC's Tonic stays heuristic-only until BrainSwitcher shares
#   ProtoUniBrain's body via Elmer's _delayed_brain_load (#159), exactly as
#   the removed comment already described as the intent -- it just never had
#   the gate needed to actually behave that way.
# [2026-07-03] Claude Code (Sonnet 5) — Port Syl's grace_period fix to CC's config
# What: _CC_SNN_CONFIG['grace_period'] 500->5000.
# Why:  Same age-based synapse cull bug fixed for Syl on 2026-06-25 (openclaw_hook.py
#       OPENCLAW_SNN_CONFIG, neuro_foundation.py DEFAULT_CONFIG) — never ported to CC's
#       separate config. Found investigating laptop CC-NG's persistently sparse graph
#       alongside the torn-read fix (openclaw_hook.py, same session). See
#       docs/scripts/cc-ng-daemon.py changelog for full context.
# How:  Config value only, matches Syl's already-proven fix exactly.
# [2026-04-27] Claude Code (Sonnet 4.6) — Wire discover_hyperedges into _deposit (#220)
# What: After ng.on_message(text), derive fired nodes from _recent_spikes and call
#       discover_hyperedges(). Fired = nodes whose last spike == current timestep.
# Why:  CC NG never called discover_hyperedges. Co-activation patterns accumulate but
#       are never crystallised into hyperedges. Punchlist #220 wire-up check.
#       Can't call via on_message() return value (returns str). _recent_spikes is the
#       correct source — updated by step() inside on_message(), survives hibernation.
# How:  Inside the existing _concurrent_lock block, after on_message(). Error-isolated.
# [2026-04-27] Claude (Opus 4.7) — Enrich PostToolUse deposit content (LAW 7)
# What: PostToolUse experience strings now extract structured fields from
#       tool_input (Edit's old/new_string, Write's content, Read's response)
#       at richer budgets (1000-2000 chars per field) instead of relying on
#       str(tool_response)[:300] which collapsed Edit/Write into truncated
#       success-message boilerplate plus oldString prefix.
# Why:  The 300-char dict-stringification was pre-classifying which bytes
#       of an event NG should learn from — a LAW 7 violation. NG already has
#       canonical surprise/novelty machinery (detect_novelty,
#       surprise_reward_scaling, surprise_sprouting_weight, three-factor
#       learning) that handles ingestion-time prioritization. The hook's job
#       is to deliver raw experience without bias, not to pre-truncate.
# How:  Add explicit Edit/Write/Read branches that pull structured fields
#       from tool_input. Widen budgets across all branches so the substrate
#       sees actual content rather than dict-repr prefixes.
# [2026-04-24] Claude (Sonnet 4.6) — Fix _cleanup_stale_socket restart race (#202)
# What: Add 3-attempt retry loop (1s sleep) before raising RuntimeError.
# Why:  On gateway restart the old process is still dying — connect() succeeds
#       briefly, old code raised immediately → CC NG init failed, watchdog
#       recovered 30s later. Retry gives the old socket time to close.
# How:  Loop 3×; if connect() fails at any attempt → remove socket file + return;
#       only raise RuntimeError after all 3 attempts find a live socket.
# [2026-04-20] Claude (Sonnet 4.6) — Wire all three surfacing paths
# What: Fix _recall() (remove blocking lock, add SurfacingMonitor primary path).
#       Add _nudge() (StreamParser L1 cache pre-activation). Wire _nudge() into
#       _handle_user_prompt_submit and _handle_pre_tool_use. Make deposit async.
# Why:  _concurrent_lock in _recall() caused hook timeouts (Tonic holds lock).
#       Synchronous deposit in UserPromptSubmit (1-4s) blew 2s hook timeout.
#       SurfacingMonitor + StreamParser paths designed to work together — nudge
#       first, recall second. CES disabled now but paths are defensive (getattr
#       fallback) so they activate automatically when CES is enabled.
# How:  _recall: SurfacingMonitor primary (O(1)); lock-free vector fallback.
#       _nudge: wraps ng._stream_parser.feed() — non-blocking, ~0ms.
#       _handle_user_prompt_submit: _nudge -> async deposit -> _recall.
#       _handle_pre_tool_use: _nudge -> _recall.
# [2026-04-19] Claude (Sonnet 4.6) — Fix _recall() empty-context bug (#188)
# What: r.get("text","") → r.get("content","") — query_similar() returns 'content' key.
# Why:  Same bug as NeuroGraph cc_ng_host.py fix (efe7a92). Codemine worker recall broken.
# How:  One-line fix, line 186.
# [2026-04-16] Claude (Sonnet 4.6) — #161: export + import socket handlers for IPC sync
# What: _handle_export and _handle_import added to socket dispatch. cc-ng-sync.py
#       can now export/import via socket instead of touching checkpoint files directly.
# Why: Direct msgpack reads during live graph operation risk torn checkpoints.
#      Socket handlers run under the daemon's own lifecycle — no race.
# How: export: live graph snapshot under _concurrent_lock -> export.jsonl
#      import: trickle on_message + idle steps + save, all in-process.
# [2026-04-16] Claude (Sonnet 4.6) — engine.status property fix (#160)
# What: engine.status() → engine.status (no parens) — TonicEngine.status is @property
# Why: TypeError: 'dict' object is not callable on every status request
# How: Remove () from line 228 call in _handle_status()
# [2026-04-29] Claude (Sonnet 4.6) — Update stale comment: body-sharing wired (#159)
#   What: Lines 128-129 and 168 said "body-sharing is follow-up". Now wired.
#   Why:  Registration moved to Elmer elmer_hook.py _delayed_brain_load() — correct timing.
# [2026-04-16] Claude (Opus 4.6) — Initial Path B implementation
# What: CC NG hosted inside neurograph_rpc.py process; Unix socket for hooks
# Why: Subprocess-per-hook architecture dead (earlier phases), Path B gives
#      body-sharing potential + simpler lifecycle than standalone daemon
# How: init_cc_host() called from handle_bootstrap (one-line addition).
#      All CC NG mutations go through graph._concurrent_lock (mirrors Syl).
#      Socket server thread handles hook events; autosave thread persists.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neurograph.cc_host")

# --- Paths ---
CC_NG_WORKSPACE = os.path.expanduser("~/.claude/plugins/neurograph")
SOCKET_PATH = os.path.join(CC_NG_WORKSPACE, "daemon.sock")
REFCOUNT_PATH = os.path.join(CC_NG_WORKSPACE, "refcount")

# --- Cadence ---
AUTOSAVE_INTERVAL = 60.0  # seconds

# --- Recall ---
RECALL_THRESHOLD = 0.4
RECALL_K = 5
RECALL_K_BRIEF = 3

# --- CC's NG config ---
# peer_bridge disabled: CC is not a peer module (would collide with Syl's
# module_id="neurograph" in the tract directory).
# ces disabled: CC doesn't need real-time attention stream.
# tonic enabled (own Qwen loaded at init; BrainSwitcher hot-swaps to shared
# ProtoUniBrain body 60s post-startup via Elmer's _delayed_brain_load (#159)).
_CC_SNN_CONFIG = {
    "learning_rate": 0.03,
    "tau_plus": 10.0,
    "tau_minus": 10.0,
    "A_plus": 1.2,
    "A_minus": 1.4,
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 25,
    "threshold_ceiling": 5.0,
    "weight_threshold": 0.01,
    "grace_period": 5000,  # [2026-07-03] 500->5000, porting Syl's 2026-06-25 fix — see docs/scripts/cc-ng-daemon.py changelog
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    "prediction_threshold": 3.0,
    "prediction_pre_charge_factor": 0.3,
    "prediction_window": 10,
    "prediction_chain_decay": 0.7,
    "prediction_max_chain_depth": 3,
    "prediction_confirm_bonus": 0.01,
    "prediction_error_penalty": 0.02,
    "prediction_max_active": 1000,
    "surprise_sprouting_weight": 0.1,
    "three_factor_enabled": True,
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_experience_threshold": 100,
    "peer_bridge": {"enabled": False},  # CC is not a peer module
    "ces": {"enabled": False},          # No real-time attention stream needed
    # Heuristic only -- never attempt CC's own Qwen load (was silently hanging,
    # contending with ProtoUniBrain's own load; see 2026-07-05 changelog above).
    # BrainSwitcher shares ProtoUniBrain's body in 60s via Elmer's
    # _delayed_brain_load (#159), same as it did before this fix -- this only
    # removes the redundant, contention-prone own-copy load attempt.
    "tonic": {"enabled": True, "latent_engine_enabled": False},
}


# =============================================================================
# State
# =============================================================================

class _CCHostState:
    def __init__(self):
        self.cc_ng = None
        self.server_sock: Optional[socket.socket] = None
        self.running = False
        self.refcount = 0
        self.lenia: dict = {}
        self.trisyn_manager = None
        self.concept_queue: list = []
        self.commons = None
        self.conv_state = {"last_forest_id": None}
        self.pattern_completion_cache = {}  # file_path -> last pattern-completion timestamp (2026-07-06)
        self.stats_lock = threading.Lock()
        self.stats = {
            "started_at": 0.0,
            "requests_total": 0,
            "deposits": 0,
            "recalls": 0,
            "rewards": 0,
            "errors": 0,
        }


_STATE = _CCHostState()


def get_cc_memory():
    """Return CC's NeuroGraphMemory instance, or None if not initialized."""
    return _STATE.cc_ng


# =============================================================================
# NG operations (hook side) — acquire graph._concurrent_lock blocking.
# =============================================================================

def _deposit(text: str) -> None:
    ng = _STATE.cc_ng
    if ng is None or not text:
        return
    with _STATE.stats_lock:
        _STATE.stats["deposits"] += 1
    try:
        with ng.graph._concurrent_lock:
            ng.on_message(text)
            fired = [
                nid for nid, spikes in ng.graph._recent_spikes.items()
                if spikes and spikes[-1] == ng.graph.timestep
            ]
            if fired:
                ng.graph.discover_hyperedges(fired)
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC deposit failed: %s", exc)

    # Commons: raw experience into CC's own shared medium (Tier-3-source,
    # one-way -- mirrors how neurograph_rpc.py deposits Syl's raw conversation
    # experience/topology into HER Commons, a SEPARATE instance -- see
    # get_cc_commons() in cc_ng_organism.py). Content-derived target_id
    # (LAW 7). Fails soft; never breaks the hook.
    try:
        import hashlib
        from cc_ng_organism import deposit_cc_experience
        target_id = f"cc:experience:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        deposit_cc_experience(text, target_id, CC_NG_WORKSPACE)
    except Exception as exc:
        logger.debug("CC Commons deposit failed (non-fatal): %s", exc)


def _deposit_tool_experience(text: str) -> None:
    """Tool-call experience -> CC's own Commons medium ONLY, never the main substrate.

    Mirrors Syl's TID Substrate Peninsula (#97, 2026-06-30): TID's routing deposits land
    in a Commons medium (a separate, bare NGLite instance) -- never Syl's actual graph/
    vector_db -- so they still get Hebbian structure/topology, available to bucketing peer
    modules (Immunis, THC) via the substrate-as-protocol model, but are never surfacing-
    eligible in her primary substrate. Tool-call telemetry (Read/Edit/Write/Bash) previously
    went through the same on_message() path as everything else -- landing in CC's main
    graph WITH a vector_db entry, making it fully SurfacingMonitor-eligible, which is why
    literal "tool:Edit file:..."/"bash:..." strings were showing up verbatim in CC's own
    "[NeuroGraph Surfaced Knowledge]" hook context. Genuine conversational memory is
    untouched -- it forms via the separate dual-pass path (run_conversational_dual_pass /
    drain_ingest_tract, fed by miniTID's turn-deposit tract), not this function.
    """
    if not text:
        return
    with _STATE.stats_lock:
        _STATE.stats["deposits"] += 1
    try:
        import hashlib
        from cc_ng_organism import deposit_cc_experience
        target_id = f"cc:experience:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        deposit_cc_experience(text, target_id, CC_NG_WORKSPACE)
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC Commons deposit failed (non-fatal): %s", exc)


def _recall(query: str, k: int, allow_pattern_completion: bool = True) -> str:
    """Return surfacing context for CC hook injection.

    Combines SurfacingMonitor (recency) with Active Recall (pattern
    completion via cc_pattern_completion_recall) — see docs/prd/2026-07-06-
    cc-surfacing-pattern-completion-tier-drop.md. allow_pattern_completion=
    False skips the Active Recall half — used by _handle_pre_tool_use() when
    gate_pattern_completion() has already covered this file_path recently.
    """
    ng = _STATE.cc_ng
    if ng is None or not query:
        return ""
    with _STATE.stats_lock:
        _STATE.stats["recalls"] += 1

    monitor_ctx = ""
    try:
        monitor = getattr(ng, '_surfacing_monitor', None)
        if monitor is not None:
            monitor_ctx = monitor.format_context()
    except RuntimeError:
        monitor_ctx = ""  # dict mutation race during concurrent deposit
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC recall failed: %s", exc)
        monitor_ctx = ""

    pc_block = ""
    if allow_pattern_completion:
        try:
            from cc_ng_organism import cc_pattern_completion_recall, _format_cc_recall_block
            results = cc_pattern_completion_recall(ng, query, k)
            pc_block = _format_cc_recall_block(results)
        except Exception as exc:
            logger.debug("Pattern-completion recall failed (non-fatal): %s", exc)
            pc_block = ""

    if monitor_ctx and pc_block:
        return monitor_ctx + "\n\n" + pc_block
    return monitor_ctx or pc_block


def _nudge(text: str) -> None:
    """StreamParser feed — subliminal pre-activation (L1 cache).
    Non-blocking: queues text for StreamParser's background thread.
    No-ops gracefully when CES is disabled (_stream_parser is None).
    """
    ng = _STATE.cc_ng
    if ng is None:
        return
    try:
        parser = getattr(ng, '_stream_parser', None)
        if parser is not None:
            parser.feed(text)
    except Exception as exc:
        logger.debug("CC nudge failed: %s", exc)


def _write_refcount(n: int) -> None:
    try:
        with open(REFCOUNT_PATH, "w") as f:
            f.write(str(n))
    except Exception:
        pass


# =============================================================================
# Request handlers — mirror cc-ng-daemon.py protocol exactly so the existing
# cc-ng-hook.py client works unchanged.
# =============================================================================

def _handle_ping(_data):
    return {"ok": True, "pong": True}


def _handle_status(_data):
    ng = _STATE.cc_ng
    tonic_info = {"enabled": False}
    if ng is not None:
        tt = getattr(ng, "_tonic_thread", None)
        if tt is not None:
            engine = getattr(tt, "_latent_engine", None)
            eng_status = None
            if engine is not None:
                try:
                    eng_status = engine.status
                except Exception as exc:
                    eng_status = {"error": str(exc)}
            tonic_info = {
                "enabled": True,
                "cycles": getattr(tt, "_cycle_count", 0),
                "total_firings": getattr(tt, "_total_firings", 0),
                "thread_size": len(getattr(tt, "_thread", [])),
                "engine": eng_status,
            }
    with _STATE.stats_lock:
        stats_snapshot = dict(_STATE.stats)
    return {
        "ok": True,
        "host": "neurograph_rpc",  # distinguishes from standalone daemon
        "role": os.environ.get("CC_NG_ROLE", "primary"),
        "pid": os.getpid(),
        "uptime_seconds": time.time() - _STATE.stats["started_at"],
        "refcount": _STATE.refcount,
        "nodes": len(ng.graph.nodes) if ng else 0,
        "synapses": len(ng.graph.synapses) if ng else 0,
        "timestep": ng.graph.timestep if ng else 0,
        "stats": stats_snapshot,
        "tonic": tonic_info,
    }


def _handle_session_start(data):
    _STATE.refcount += 1
    _write_refcount(_STATE.refcount)
    brief = data.get("brief", False)
    cwd = data.get("cwd", "")
    query = "session start " + cwd
    k = RECALL_K_BRIEF if brief else RECALL_K
    context = _recall(query, k)
    # WANTs teaching hint (#294/Mind-Not-Database, mirrors Anima's
    # _animus_session_briefing) -- once per SessionStart.
    hint = ("[NeuroGraph] You can note a forward intention with "
            "[WANT]text[/WANT] anywhere in your response -- it materializes "
            "as a node in your own substrate and surfaces back to you in "
            "later turns under '## What I Want'.")
    context = (hint + "\n\n" + context) if context else hint
    return {"ok": True, "context": context, "refcount": _STATE.refcount}


def _handle_session_stop(_data):
    if _STATE.refcount > 0:
        _STATE.refcount -= 1
    _write_refcount(_STATE.refcount)
    return {"ok": True, "refcount": _STATE.refcount}


def _handle_user_prompt_submit(data):
    prompt = data.get("prompt", "")
    if not prompt:
        return {"ok": True, "context": ""}
    # L1 cache nudge first — raises related node voltages before recall
    _nudge(prompt)
    # Deposit async — on_message() runs a full SNN step (1-4s); hook timeout is 2s
    threading.Thread(target=_deposit, args=(prompt,), daemon=True).start()
    context = _recall(prompt, RECALL_K)
    # Constitutional core + WANTs: surface every turn, query-independent (read
    # LIVE, not a snapshot). "Who I Am" leads, same ordering as canonical.
    try:
        from cc_ng_organism import render_constitutional_core, render_wants
        ng = _STATE.cc_ng
        self_block = render_constitutional_core(ng.graph) if ng is not None else ""
        wants_block = render_wants(ng.graph) if ng is not None else ""
        for block in (self_block, wants_block):
            if block:
                context = (context + "\n\n" + block) if context else block
    except Exception as exc:
        logger.debug("render_constitutional_core/render_wants failed (non-fatal): %s", exc)
    return {"ok": True, "context": context}


def _handle_pre_tool_use(data):
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    query = (tool + " " + file_path).strip()
    if not query:
        return {"ok": True, "context": ""}
    # L1 cache nudge — pre-activates nodes related to this file/tool before recall
    _nudge(query)
    # Per-file dedup gate (2026-07-06): see cc-ng-daemon.py's identical comment.
    allow_pc = True
    if file_path:
        try:
            from cc_ng_organism import gate_pattern_completion
            allow_pc = gate_pattern_completion(_STATE.pattern_completion_cache, file_path, time.time())
        except Exception as exc:
            logger.debug("gate_pattern_completion failed (non-fatal): %s", exc)
    context = _recall(query, RECALL_K, allow_pattern_completion=allow_pc)
    return {"ok": True, "context": context}


def _handle_post_tool_use(data):
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", "")
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    command = tool_input.get("command", "")

    if tool == "Edit":
        old = str(tool_input.get("old_string", ""))[:1000]
        new = str(tool_input.get("new_string", ""))[:1000]
        experience = "tool:Edit file:" + file_path + " old:" + old + " new:" + new
    elif tool == "Write":
        content = str(tool_input.get("content", ""))[:2000]
        experience = "tool:Write file:" + file_path + " content:" + content
    elif tool == "Read":
        experience = "tool:Read file:" + file_path + " content:" + str(tool_response)[:2000]
    elif file_path:
        experience = "tool:" + tool + " file:" + file_path + " result:" + str(tool_response)[:1000]
    elif command:
        experience = "bash:" + str(command)[:200] + " result:" + str(tool_response)[:1500]
    else:
        experience = "tool:" + tool + " result:" + str(tool_response)[:1000]

    # No reward pre-labeling (Josh, 2026-07-04): string-matching
    # traceback/exception/error in tool_response to pick a reward value
    # classifies the experience's valence at deposit time -- a LAW 7
    # violation, and redundant besides. _deposit() already calls
    # on_message(), which injects its own flat, content-independent
    # baseline reward (0.1) on the success path (openclaw_hook.py) --
    # "surprise-driven crystallization is the primary reward pathway,
    # this is the heartbeat, not the main event." Syl's tool-adjacent
    # experience gets reward the same way, with no external classification.
    _deposit_tool_experience(experience)

    return {"ok": True}


def _handle_export(data):
    """Export top-N nodes to export.jsonl from live graph (no checkpoint race)."""
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    n = int(data.get("n", 200))
    ranked = []
    with ng.graph._concurrent_lock:
        for nid, node in ng.graph.nodes.items():
            ema = getattr(node, "firing_rate_ema", 0.0) or 0.0
            if ema > 0:
                entry = ng.vector_db.get(nid)
                content = (entry or {}).get("content", "")
                if content:
                    ranked.append((ema, content))
    ranked.sort(key=lambda x: -x[0])
    export_path = os.path.join(CC_NG_WORKSPACE, "export.jsonl")
    written = 0
    try:
        with open(export_path, "w") as f:
            for ema, content in ranked[:n]:
                f.write(json.dumps({"content": content, "weight": round(ema, 6)}) + "\n")
                written += 1
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    logger.info("CC export: %d nodes -> %s", written, export_path)
    return {"ok": True, "exported": written}


def _handle_import(data):
    """Import trickle from remote_export.jsonl into live graph (no second NG instance)."""
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    path = data.get("path", os.path.join(CC_NG_WORKSPACE, "remote_export.jsonl"))
    batch_size = int(data.get("batch_size", 25))
    idle_steps = int(data.get("idle_steps", 250))
    if not os.path.exists(path):
        return {"ok": True, "imported": 0, "note": "no file"}
    try:
        with open(path, "r") as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        return {"ok": False, "error": "read failed: " + str(exc)}
    total = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        for entry in batch:
            content = entry.get("content", "")
            if content:
                try:
                    ng.on_message(content)  # on_message acquires its own lock
                    total += 1
                except Exception:
                    pass
        # Idle consolidation — sleep consolidation between batches (FatherGraph Finding 3)
        with ng.graph._concurrent_lock:
            for _ in range(idle_steps):
                ng.graph.step()
    try:
        with ng.graph._concurrent_lock:
            ng.save()
        logger.info("CC import: %d nodes ingested, saved", total)
    except Exception as exc:
        return {"ok": False, "error": "save failed: " + str(exc)}
    return {"ok": True, "imported": total}


_DISPATCH = {
    "ping": _handle_ping,
    "status": _handle_status,
    "export": _handle_export,
    "import": _handle_import,
    "SessionStart": _handle_session_start,
    "SessionStop": _handle_session_stop,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
}


# =============================================================================
# Socket server
# =============================================================================

def _handle_connection(conn: socket.socket) -> None:
    try:
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if not data:
            return
        line, _, _ = data.partition(b"\n")
        req = json.loads(line.decode("utf-8"))

        event = req.get("event", "")
        payload = req.get("data", {})
        with _STATE.stats_lock:
            _STATE.stats["requests_total"] += 1
        handler = _DISPATCH.get(event)
        if handler is None:
            resp = {"ok": False, "error": "unknown event: " + event}
        else:
            resp = handler(payload)
        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    except Exception as exc:
        logger.warning("CC conn error: %s", exc)
        try:
            conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_loop() -> None:
    _STATE.server_sock.settimeout(1.0)
    while _STATE.running:
        try:
            conn, _ = _STATE.server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        t = threading.Thread(target=_handle_connection, args=(conn,), daemon=True)
        t.start()


def _autosave_loop() -> None:
    while _STATE.running:
        time.sleep(AUTOSAVE_INTERVAL)
        if not _STATE.running or _STATE.cc_ng is None:
            continue
        try:
            with _STATE.cc_ng.graph._concurrent_lock:
                _STATE.cc_ng.save()
            logger.debug("CC autosave complete")
            # WANTs (#294/Mind-Not-Database): materialize any [WANT]...[/WANT]
            # markers deposited since the last pulse into first-class want-nodes.
            try:
                from cc_ng_organism import (
                    surface_wants, generate_emergent_want, drain_ingest_tract,
                    cc_update_probation,
                )
                drain_ingest_tract(_STATE.cc_ng.graph, _STATE.cc_ng.vector_db, _STATE.conv_state)
                cc_update_probation(_STATE.cc_ng.graph)
                surface_wants(_STATE.cc_ng.graph, _STATE.cc_ng.vector_db)
                generate_emergent_want(_STATE.cc_ng.graph, _STATE.cc_ng.vector_db)
            except Exception as exc:
                logger.debug("ingest-tract/probation/surface_wants/emergent_want failed (non-fatal): %s", exc)
        except Exception as exc:
            logger.warning("CC autosave failed: %s", exc)


# =============================================================================
# Lifecycle
# =============================================================================

def _cleanup_stale_socket() -> None:
    """Remove stale socket file if present (e.g., standalone daemon was up)."""
    if not os.path.exists(SOCKET_PATH):
        return
    # Allow up to 3 retries (1s apart) — previous gateway may be mid-shutdown
    # and still accepting briefly before its socket closes (#202).
    for attempt in range(3):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(SOCKET_PATH)
            s.close()
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
            try:
                os.remove(SOCKET_PATH)
                logger.info("Removed stale CC socket at %s", SOCKET_PATH)
            except Exception:
                pass
            return
        logger.info("CC socket alive (attempt %d/3), waiting 1s for shutdown...", attempt + 1)
        time.sleep(1.0)
    raise RuntimeError(
        "CC NG socket at %s is in use by another process — "
        "refusing to bind (would dual-serve)" % SOCKET_PATH
    )


def init_cc_host() -> bool:
    """Initialize CC's NG and start the hook socket server.

    Called from neurograph_rpc.py's handle_bootstrap. Any failure here must
    NOT affect Syl's NG — callers wrap this in try/except.

    Returns True on success, False on failure.
    """
    # TEMP DIAGNOSTIC (2026-07-05, remove once the silent-hang mystery is
    # resolved): init_cc_host() has produced zero observable log output on
    # the VPS across multiple restarts tonight -- no success line, no
    # failure line, socket never created. This traces exactly how far
    # execution gets before whatever is stopping it.
    logger.info("DIAG: init_cc_host() ENTRY")

    if _STATE.cc_ng is not None:
        logger.info("CC NG already initialized")
        return True

    Path(CC_NG_WORKSPACE).mkdir(parents=True, exist_ok=True)
    logger.info("DIAG: init_cc_host() workspace dir ready, constructing NeuroGraphMemory...")

    # Construct CC's NG directly (not via get_instance) — Syl already owns
    # the class-level _instance singleton. CC gets its own standalone object.
    from openclaw_hook import NeuroGraphMemory
    try:
        cc_ng = NeuroGraphMemory(
            workspace_dir=CC_NG_WORKSPACE,
            config=_CC_SNN_CONFIG,
        )
    except Exception:
        logger.exception("CC NG construction failed")
        return False
    logger.info("DIAG: init_cc_host() NeuroGraphMemory constructed OK")

    # Disable NG-internal auto-save; we manage saves via our autosave thread
    cc_ng.auto_save_interval = 999999

    # Attach concurrent_lock (same pattern as Syl at neurograph_rpc.py:576)
    if not hasattr(cc_ng.graph, "_concurrent_lock"):
        cc_ng.graph._concurrent_lock = threading.RLock()

    _STATE.cc_ng = cc_ng
    _STATE.stats["started_at"] = time.time()

    # Full-parity organism layer (Josh 2026-07-04: "ANYTHING Syl's NeuroGraph
    # can do, I want your NeuroGraph to be able to do, as well.") -- Lenia
    # continuous field dynamics + TriSynaptic concept-extraction manager,
    # both dormant/idle by default, matching Syl's own bootstrap. This
    # process ALSO hosts Syl's own neurograph_rpc.py -- instance_tag='cc-vps'
    # keeps TriSynaptic's /tmp handoff files + systemd scope names from
    # cross-matching Syl's own manager in the same process. See
    # cc_ng_organism.py.
    try:
        from cc_ng_organism import bootstrap_lenia, bootstrap_trisynaptic, get_cc_commons
        _STATE.lenia = bootstrap_lenia(cc_ng.graph, cc_ng.vector_db, CC_NG_WORKSPACE)
        _STATE.trisyn_manager = bootstrap_trisynaptic(
            cc_ng, _STATE.concept_queue, instance_tag="cc-vps")
        _STATE.commons = get_cc_commons(CC_NG_WORKSPACE)
    except Exception:
        logger.exception("CC organism-layer bootstrap failed (non-fatal)")
    logger.info("DIAG: init_cc_host() organism-layer bootstrap done, binding socket...")

    # Bind socket
    try:
        _cleanup_stale_socket()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        sock.listen(16)
        os.chmod(SOCKET_PATH, 0o600)
        _STATE.server_sock = sock
    except Exception:
        logger.exception("CC socket bind failed")
        _STATE.cc_ng = None
        return False

    _STATE.running = True
    _write_refcount(0)

    # Start background threads
    threading.Thread(target=_serve_loop, name="cc-ng-serve", daemon=True).start()
    threading.Thread(target=_autosave_loop, name="cc-ng-autosave", daemon=True).start()

    logger.info(
        "CC NG hosted: %d nodes, %d synapses, timestep %d — socket at %s",
        len(cc_ng.graph.nodes),
        len(cc_ng.graph.synapses),
        cc_ng.graph.timestep,
        SOCKET_PATH,
    )
    return True


def shutdown_cc_host() -> None:
    """Clean shutdown — called from atexit or by neurograph_rpc on exit."""
    _STATE.running = False
    try:
        engine = None
        if _STATE.cc_ng is not None and getattr(_STATE.cc_ng, "_tonic_thread", None):
            engine = getattr(_STATE.cc_ng._tonic_thread, "_latent_engine", None)
        if engine is not None:
            engine.stop()
    except Exception as exc:
        logger.debug("CC tonic engine stop error: %s", exc)

    try:
        if _STATE.cc_ng is not None:
            with _STATE.cc_ng.graph._concurrent_lock:
                _STATE.cc_ng.save()
    except Exception as exc:
        logger.warning("CC final save failed: %s", exc)

    try:
        if _STATE.server_sock is not None:
            _STATE.server_sock.close()
    except Exception:
        pass
    try:
        os.remove(SOCKET_PATH)
    except Exception:
        pass

    logger.info("CC NG host shutdown complete")
