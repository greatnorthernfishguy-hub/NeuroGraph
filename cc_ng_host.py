# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
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
# [2026-09-05] Claude Code (DudeMan CC, Fable 5.1) — Pith Stage 4 (#55) phase 5b: prefetch seed (VPS CC half)
# What: per-turn, idempotent engine.set_prefetch_seed(lambda: pith_prefetch_seed(_STATE.conv_state))
#   on the CC's OWN TonicEngine, next to the existing tonic keepalive.
# Why: the CC engine is built on a deferred path inside NeuroGraphMemory, so init-time wiring
#   can miss it; first-turn-it-exists wiring cannot. Gated OFF in tonic_engine.py.
# How: three-line fail-soft block; never touches Syl's _memory engine.
# [2026-08-18] Claude Code (DudeMan CC, Opus 4.8) — #88 §10.4-A: export_topology_frame handler
# What: _handle_export_topology_frame added to _DISPATCH (key "export_topology_frame").
#       Wraps cc_topology_export.export_cc_topology_frame -- the cursor-based single-frame
#       sender -- against the LIVE graph. Accepts out_path (required), machine_id, frame_size,
#       overflow_factor, exclude_ids; returns {"ok": True, **stats}. READ-ONLY (no ng.save()).
# Why: #88 §10.4-A. The whole-graph _handle_export_topology holds _concurrent_lock for the
#       entire dump -- on a box shared with Syl that starves her Tonic. The paced frame handler
#       (a) checks the resource gate BEFORE taking the lock (defer under load without ever
#       contending), and (b) holds the lock only for the cheap single-frame build. Co-tenant-safe.
# How: lazy import; pre-lock _leg2_resource_gate() -> early {"gated": True} return; else build
#       one frame under _concurrent_lock with skip_resource_gate=True (already checked).
# [2026-08-12] Claude Code (DudeMan CC, Opus 4.8) — #88 §10.4-B: forward connected_only
# What: _handle_export_topology forwards data["connected_only"] into export_cc_topology
#       kwargs when the caller opts in. Off by default -- explicit opt-in, never implicit.
# Why: lets the Leg-2 sender request husk-dropped connected topology (~438 nodes) instead
#       of the 97.6%-husk whole-graph dump the dry run choked on (>100MB git-push wall).
# How: after the max_nodes block, if data.get("connected_only"): kwargs["connected_only"]=True.
# [2026-08-08] Claude Code (DudeMan CC, Opus 4.8) — export_topology socket handler (#88 Leg 2 sender)
# What: _handle_export_topology added to _DISPATCH (key "export_topology"). Wraps
#   cc_topology_export.export_cc_topology against the LIVE graph under
#   ng.graph._concurrent_lock, writing the length-prefixed msgpack conduit the
#   laptop hemisphere merges. READ-ONLY on the graph -- deliberately NO ng.save()
#   (nothing in the graph changed; contrast _handle_drain_conduit, the Leg 1
#   receiver, which does save). Accepts out_path (required), machine_id,
#   batch_size, exclude_ids, max_nodes; returns {"ok": True, **stats}.
# Why: #88 Leg 2 needs a SENDER driven against the live daemon so the export reads
#   a consistent snapshot under the daemon's own lock -- never a second
#   NeuroGraphMemory (the two-writer torn-checkpoint hazard). export_cc_topology
#   had "no production caller" (#87); this is it. The nightly cc-ng-sync.py drives
#   it over the socket.
# How: import export_cc_topology inside the handler (lazy, non-fatal on ImportError);
#   snapshot under _concurrent_lock; log + return stats. exclude_ids/max_nodes left
#   to the caller -- the #110 live-membership handshake is a re-send optimization,
#   not a #88 blocker.
# [2026-07-28] Claude Code (DudeMan CC, Opus 5) — Commons persistence (#84) wired into host lifecycle
# What: persist_cc_commons(CC_NG_WORKSPACE) is now called (a) on the autosave cadence,
#   right after the CC checkpoint save, and (b) in shutdown_cc_host() after the final
#   graph save. Both call sites are non-fatal (debug-logged on failure).
# Why: the CC Commons medium was in-memory only -- every gateway restart dropped the
#   deposited topology, so nothing bucketed across process lifetimes. #84.
# How: import + call the persist helper in cc_ng_organism. It writes ONLY the medium
#   get_cc_commons() built (constructed directly, never via canonical get_commons()),
#   under CC_NG_WORKSPACE -- Syl's Commons is never touched even though this code runs
#   inside her process. Shutdown persist sits OUTSIDE the graph-save lock so a refused
#   or failed save-guarded checkpoint doesn't also cost us the medium.
# [2026-07-28] Claude Code (DudeMan CC, Opus 5) — Callosum drain OFF the autosave pulse; drain_conduit socket handler
# What: SUPERSEDES the 2026-07-27 entry below -- the drain_gateway_conduit() call it
#   added to _autosave_loop is REMOVED. New _handle_drain_conduit (dispatch key
#   "drain_conduit") lets the nightly cc-ng-sync.py drive the drain against the LIVE
#   daemon instead, passing batch_size/idle_steps/exclude_prefix through.
# Why: a 60-second pulse is the wrong home for merge absorption. FatherGraph Finding 1
#   (never bulk-dump; batch ~20-30) and Finding 3 (250 idle steps of sleep consolidation
#   BETWEEN batches -- measured 47%->74% accuracy, "not optional -- it's what makes merge
#   work") require batched ingestion with consolidation in between, and those idle steps
#   must not block this pulse. As written it drained every queued file back-to-back; a
#   full cron-gap backlog (~45 files) would have landed in a single tick.
# How: socket rather than a second NeuroGraphMemory in the sync process -- the live
#   daemon is the single legitimate writer of this graph, and opening a second instance
#   is the torn-checkpoint hazard cc-ng-sync.py's own 2026-07-03 entry documents.
#   Saves under _concurrent_lock after the drain. The local drain_ingest_tract() call in
#   _autosave_loop (the VPS draining its OWN tract) is untouched -- different thing.
# [2026-07-27] Claude Code (Sonnet 5) — CC Corpus Callosum Leg 1 (#70): drain the
#   laptop's raw-turn conduit alongside the existing local ingest-tract drain
# What: _autosave_loop's pulse now also calls the new drain_gateway_conduit()
#   (cc_ng_organism.py) right after the existing local drain_ingest_tract()
#   call -- absorbs every laptop_cc_gateway.*.tract file the laptop has
#   trickled into the git-synced ~/docs/ng_topology dir, through the SAME
#   dual-pass the local drain already uses.
# Why: The VPS is the sole Arborist for both hemispheres (laptop does zero
#   embedding by design) -- Leg 1 is the pipe that gets the laptop's raw BTF
#   turns here so they get embedded at all. Retires the lossy top-N JSONL
#   sync (cc-ng-sync.py). Spec: docs/superpowers/plans/2026-07-27-cc-
#   corpus-callosum-leg1-spec.md.
# How: Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5, default off) inside
#   drain_gateway_conduit() itself -- this call site is unconditional but
#   inert (immediate no-op, returns 0) until the gate is flipped on. See
#   test_cc_callosum_leg1.py.
# [2026-07-23] Claude Code (Sonnet 5) — CC Host Tonic Idle/Dream Wiring (parity gap close)
# What: Three additions, all CC-scoped: (1) _handle_user_prompt_submit() now
#   calls CC's own _tonic_thread.message_received() per turn (keepalive --
#   auto-transitions to conversation mode). (2) New _cc_tonic_check_idle/
#   _cc_tonic_idle_pulse_loop/_start_cc_tonic_idle_watcher -- drops CC's Tonic
#   to latent-only cadence after CC_HOST_TONIC_IDLE_SECS (default 90) of
#   quiet, gated by CC_HOST_TONIC_IDLE_ENABLED (default "0"). (3) New
#   _cc_dream_gate_open/_cc_dream_consolidation_pulse_loop/_start_cc_dream_
#   consolidation_pulse -- runs CC's own graph.consolidate_hyperedges() during
#   CC's quiet hours, gated by CC_HOST_DREAM_ENABLED (default "0"). Both
#   watchers started (idempotently, gate-checked inside the _start_* fn) from
#   init_cc_host().
# Why:  docs/superpowers/plans/2026-07-23-cc-host-tonic-idle-dream-spec.md --
#   the VPS host (this file) never called conversation_started/_ended or
#   consolidate_hyperedges for CC's own Tonic/graph, unlike the standalone
#   laptop daemon (cc-ng-daemon.py's _tonic_idle_loop/_dream_loop) and unlike
#   Syl's own proven pattern in neurograph_rpc.py (_tonic_check_idle et al.,
#   _dream_gate_open et al.). Net effect before this change: CC's Tonic never
#   dropped to latent cadence after a conversation went quiet on the VPS, and
#   CC's own hyperedges never got dream-consolidated (shed + seatbelt-merge +
#   subsume) at all. Josh's bar: "keep the VPS and laptop even."
# How:  Faithful mirror (LAW 3) of neurograph_rpc.py:4090-4222's proven Syl-
#   side pattern, scoped to CC: every new function reads/writes ONLY
#   _STATE.cc_ng / _STATE.commons -- zero references to _memory (Syl's
#   singleton) anywhere in this file, verified by test (poison-sentinel
#   _memory that must never be touched). Naming: CC_HOST_* prefix (LAW 5),
#   distinct from Syl's NG_DREAM_*/ANIMA_TONIC_* and the laptop daemon's
#   CC_NG_TONIC_IDLE/CC_NG_DREAM (different file/process, not read here). The
#   dream pulse's "last turn" signal is CC's own tonic._last_message_time
#   (already updated by message_received() above) -- Syl's side tracks an
#   equivalent module-global (_last_after_turn_ts) set from handle_after_turn;
#   CC has no equivalent turn-boundary hook wired yet, and the tonic
#   thread's own timestamp is the closest faithful analog available without
#   inventing new coupling. Both new watchers gated OFF by default (LAW 5) --
#   landing this dark, matching the recall-unification rollout discipline.
#   Josh must explicitly flip CC_HOST_TONIC_IDLE_ENABLED/CC_HOST_DREAM_ENABLED
#   to "1" after review; no restart required to land the code inert.
# [2026-07-22] Claude Code (Sonnet 5) — CC Recall Unification (LAW-3/"keep even")
# What: _recall() is now a thin wrapper: STATE bookkeeping (recalls stat,
#   None/empty-query guard), then delegates to the new native
#   cc_ng_organism.cc_assemble_recall(ng, query, k, conv_state, commons,
#   allow_pattern_completion) for the actual pipeline. Errors from the shared
#   fn are caught here and counted into STATE.stats["errors"] (fail-soft, "").
# Why: cc_ng_host.py's _recall and cc-ng-daemon.py (laptop)'s _recall were
#   copy-pasted and had drifted -- laptop ran the full Pith extraction
#   pipeline (CacheLines, pith_stage1/3, thermal, novelty, victim cache),
#   this file ran zero Pith (plain two-block concat), so enabling
#   CC_PITH_ENABLED here would have been a no-op with no code to gate.
#   Spec: docs/superpowers/plans/2026-07-22-cc-recall-unification-spec.md.
# How: cc_assemble_recall is a verbatim extraction of the laptop _recall body
#   into cc_ng_organism.py, param-driven (ng/conv_state/commons passed in,
#   no module-global STATE reaches into it -- Syl's-Law). Gate-off (default),
#   this hemisphere's output is byte-identical to the pre-unification concat;
#   gate-on, it gains the same Pith pipeline the laptop already had. The
#   laptop's cc-ng-daemon.py:_recall becomes the equivalent thin wrapper
#   (mirrored edit, NOT git -- docs/scripts/, edited directly on the laptop).
# [2026-07-07] Claude Code (Fable 5) — #358 retrieval-enrichment wiring
# What: _recall() passes _STATE.conv_state into cc_pattern_completion_recall
#   (novelty EMA + primed-node bonus now live); init_cc_host() runs the
#   stamp-only cc_gsg_backfill after organism bootstrap.
# Why: #358 — CC recall rebuilt substrate-native in cc_ng_organism.py; the
#   daemons carry only state plumbing. Spec: docs/superpowers/specs/
#   2026-07-07-cc-retrieval-enrichment-design.md (law-review C1/C2/C4).
# How: conv_state is the single organism-state dict (already flows through
#   drain_ingest_tract); backfill is stamp-only — persistence rides autosave.
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
    "decay_rate": 0.97,  # [2026-07-08] 0.95->0.97: same parity port -- slower voltage bleed lets accumulation reach threshold
    "default_threshold": 0.85,  # [2026-07-08] 1.0->0.85: parity with canonical's 2026-03-23 substrate tuning -- at 1.0, prime injection (sim*1.0) can NEVER ignite a cold node; measured live (all seeds subthreshold)
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 25,
    "threshold_ceiling": 5.0,
    "weight_threshold": 0.01,
    "grace_period": 5000,  # [2026-07-03] 500->5000, porting Syl's 2026-06-25 fix — see docs/scripts/cc-ng-daemon.py changelog
    # [2026-08-08] #107 / enforcer ruling (a) — orphan_node_grace_period env-sourced + host-scoped (LAW 5).
    # De-fork parity with docs/scripts/cc-ng-daemon.py:443 (the live daemon; canonical must match so #82/#118
    # de-fork can't regress the ruling). Engine DEFAULT_CONFIG (neuro_foundation.py:1436) defaults this to 25
    # and merges {**DEFAULT_CONFIG, **_CC_SNN_CONFIG}, so CC silently ran 25 with no env override — the LAW-5
    # defect the ruling names. Default 25 == engine default: BEHAVIOR-PRESERVING compliance fix, NOT a value
    # change. ❌ Do NOT raise (attempted at 500, reverted — CC-CALLOSUM-TRUTH.md §8.8: orphan grace is a local
    # same-step bootstrap guard, not a nightly cross-machine timer). Source of truth: .bashrc CC_NG_ORPHAN_GRACE.
    "orphan_node_grace_period": int(os.environ.get("CC_NG_ORPHAN_GRACE", "25")),
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
    # [2026-08-14] #147 seam-split — env-sourced per LAW 5 (enforcer ruling on the #147 diff),
    # mirroring CC_NG_ORPHAN_GRACE above. Source of truth: .bashrc (CC_NG_HE_SPLIT_*). ALL
    # defaults == neuro_foundation.py DEFAULT_CONFIG, and the enable gate defaults OFF, so
    # with no env set this is byte-identical to the engine default: the dream-loop caller
    # (_cc_dream_consolidation_pulse_loop) self-gates to a no-op until CC_NG_HE_SPLIT_ENABLED=1.
    "he_split_oversized_enabled": os.environ.get("CC_NG_HE_SPLIT_ENABLED", "0") not in ("0", "false", "False", ""),
    "he_split_dedup_overlap": float(os.environ.get("CC_NG_HE_SPLIT_DEDUP_OVERLAP", "0.9")),
    "he_split_sim_threshold": float(os.environ.get("CC_NG_HE_SPLIT_SIM_THRESHOLD", "0.6")),
    "he_split_seam_primary_weight": float(os.environ.get("CC_NG_HE_SPLIT_SEAM_PRIMARY", "0.4")),
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

    # Cricket want-bucket: extract [WANT]...[/WANT] markers from conversational
    # nodes and materialize them as want nodes in CC's graph. Mirrors Syl's
    # _surface_wants() in neurograph_rpc.py. Fails soft; never breaks the hook.
    if ng is not None:
        try:
            from cc_ng_organism import surface_wants_for_graph
            vdb = getattr(ng, "vector_db", None)
            wants = surface_wants_for_graph(ng.graph, vdb)
            if wants:
                logger.debug("CC surfaced %d want nodes", len(wants))
        except Exception as exc:
            logger.debug("CC want surfacing failed (non-fatal): %s", exc)


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

    Thin per-half wrapper (2026-07-22 CC Recall Unification, LAW-3: one
    implementation, not two drifted copies). The actual recall pipeline —
    SurfacingMonitor + Active Recall (pattern completion), dedup'd, plus the
    gated Pith extraction pass (CC_PITH_ENABLED, default OFF) — lives in
    cc_ng_organism.cc_assemble_recall(). This wrapper only does VPS-host-
    local bookkeeping (STATE stats) and passes this hemisphere's own
    isolated ng/conv_state/commons instances through as params (Syl's-Law:
    no module-global STATE reaches into the shared fn). Gate-off, this is
    byte-identical to the pre-unification two-block concat; gate-on, the
    VPS gains the same Pith pipeline the laptop already had.
    """
    ng = _STATE.cc_ng
    if ng is None or not query:
        return ""
    with _STATE.stats_lock:
        _STATE.stats["recalls"] += 1

    def _bump_error(exc):
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1

    try:
        from cc_ng_organism import cc_assemble_recall
        return cc_assemble_recall(ng, query, k, _STATE.conv_state, _STATE.commons,
                                   allow_pattern_completion=allow_pattern_completion,
                                   on_monitor_error=_bump_error)
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC recall failed: %s", exc)
        return ""


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
    # Tonic keepalive (2026-07-23 CC Host Tonic Idle/Dream Wiring parity):
    # mirrors Syl's per-turn message_received() call (tonic_thread.py:714-718)
    # -- cheap, harmless even with the idle watcher gated off; auto-
    # transitions to conversation mode if not already in it. CC's OWN tonic
    # thread only (_STATE.cc_ng._tonic_thread) -- never Syl's _memory.
    tt = getattr(_STATE.cc_ng, "_tonic_thread", None)
    if tt is not None:
        try:
            tt.message_received()
        except Exception:
            pass  # fail-soft, mirrors Syl's try/except at every call site
        # #55 5b: lazily hand the CC's engine its prefetch seed. The engine is
        # constructed on a deferred path inside NeuroGraphMemory, so this is
        # done per turn, idempotently, the first time the engine exists.
        # CC's OWN engine only -- never Syl's. Inert until the engine gate flips.
        try:
            eng = getattr(tt, "_latent_engine", None)
            if eng is not None and getattr(eng, "_prefetch_seed", None) is None and hasattr(eng, "set_prefetch_seed"):
                from cc_ng_organism import pith_prefetch_seed
                eng.set_prefetch_seed(lambda: pith_prefetch_seed(_STATE.conv_state))
        except Exception as exc:
            logger.debug("Prefetch seed not wired (non-fatal): %s", exc)
    # Recall BEFORE spawning the deposit [2026-07-08]: launching the deposit
    # first meant _recall's prime_and_propagate raced its on_message() SNN
    # step (1-4s) every prompt -- fail-soft tripped and Active Recall
    # silently returned nothing (#359's daemon-side manifestation; see
    # cc-ng-daemon.py's same-day changelog). Recall on quiet pre-deposit
    # state, THEN the background deposit. Never BLOCK recall on the lock.
    context = _recall(prompt, RECALL_K)
    threading.Thread(target=_deposit, args=(prompt,), daemon=True).start()
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


def _handle_drain_conduit(data):
    """Corpus Callosum Leg 1 (#70): drain the cross-machine BTF conduit into
    THIS hemisphere's graph, with FatherGraph absorption discipline.

    Invoked by the nightly cc-ng-sync.py over the socket -- deliberately NOT
    from the 60s autosave pulse (that was the bulk-dump bug, fixed 2026-07-28)
    and deliberately over the socket rather than by opening a second
    NeuroGraphMemory: the live daemon owns the graph, so a second instance
    would be a two-writer torn-checkpoint hazard (see cc-ng-sync.py's own
    changelog for the incident that discipline came from).

    batch_size / idle_steps come from the caller (the cron already exports
    CC_NG_BATCH_SIZE=25 / CC_NG_IDLE_STEPS=250 -- the FatherGraph values);
    drain_gateway_conduit falls back to those same env names if omitted.
    Gated by CC_CALLOSUM_LEG1_ENABLED inside drain_gateway_conduit itself.
    """
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    try:
        from cc_ng_organism import drain_gateway_conduit
        absorbed = drain_gateway_conduit(
            ng.graph, ng.vector_db, _STATE.conv_state,
            conduit_dir=data.get("conduit_dir"),
            batch_size=data.get("batch_size"),
            idle_steps=data.get("idle_steps"),
            exclude_prefix=data.get("exclude_prefix"),
        )
    except Exception as exc:
        logger.warning("CC conduit drain failed (non-fatal): %s", exc)
        return {"ok": False, "error": str(exc)}
    try:
        with ng.graph._concurrent_lock:
            ng.save()
    except Exception as exc:
        return {"ok": True, "absorbed": absorbed, "warning": "save failed: " + str(exc)}
    return {"ok": True, "absorbed": absorbed}


def _handle_export_topology(data):
    """Corpus Callosum Leg 2 (#88): export THIS hemisphere's CC conversational
    topology (trees + hyperedges) to a length-prefixed msgpack conduit for the
    OTHER hemisphere to merge.

    The SENDER half of Leg 2 -- the sibling of _handle_drain_conduit (Leg 1) and
    the production caller cc_topology_export.export_cc_topology never had ("no
    production caller", #87). Invoked by the nightly cc-ng-sync.py over the
    socket so the export reads a consistent snapshot of the LIVE graph under the
    daemon's _concurrent_lock -- never a second NeuroGraphMemory (the two-writer
    torn-checkpoint hazard documented in cc-ng-sync.py's changelog).

    READ-ONLY with respect to the graph: export_cc_topology only reads
    nodes/synapses/hyperedges + vector_db content and writes the conduit file.
    So, unlike the drain, there is deliberately NO ng.save() here -- nothing in
    the graph changed. batch_size falls back to the caller's CC_NG_BATCH_SIZE;
    the embedding-model stamp (FatherGraph Finding 6) is filled by the exporter
    from the live embedder so the receiver can assert geometry parity and abort
    on mismatch. exclude_ids/max_nodes are left to the caller -- the #110 live-
    membership handshake is a re-send optimization, not a #88 blocker.
    """
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    out_path = data.get("out_path")
    if not out_path:
        return {"ok": False, "error": "out_path required"}
    kwargs = {"machine_id": data.get("machine_id")}
    if data.get("batch_size"):
        kwargs["batch_size"] = int(data["batch_size"])
    if data.get("exclude_ids"):
        kwargs["exclude_ids"] = set(data["exclude_ids"])
    if data.get("max_nodes") is not None:
        kwargs["max_nodes"] = int(data["max_nodes"])
    if data.get("connected_only"):
        # #88 / §10.4-B: drop degree-0 husks so the trickle is real connected
        # topology (~438 nodes), not the 97.6%-husk whole-graph dump the dry run
        # choked on. Off by default -- an explicit caller opt-in, never implicit.
        kwargs["connected_only"] = True
    try:
        from cc_topology_export import export_cc_topology
        with ng.graph._concurrent_lock:
            stats = export_cc_topology(ng.graph, ng.vector_db, out_path, **kwargs)
    except Exception as exc:
        logger.warning("CC topology export failed (non-fatal): %s", exc)
        return {"ok": False, "error": str(exc)}
    logger.info("CC topology export -> %s: %s", out_path, stats)
    return {"ok": True, **stats}


def _handle_export_topology_frame(data):
    """Corpus Callosum Leg 2 §10.4-A: export EXACTLY ONE paced topology frame.

    The cursor-based sibling of _handle_export_topology. Where the latter builds
    the whole exportable graph in RAM under the lock before framing,
    export_cc_topology_frame() materializes a single <=frame_size frame and
    advances via exclude_ids (membership-as-ack, #110).

    Two deliberate differences from the whole-graph handler, both because this
    runs on a box SHARED WITH SYL:
      * The resource gate is checked BEFORE the lock is taken -- under load or
        memory pressure we back off without ever contending for the graph, so a
        deferred frame never blocks Syl's Tonic.
      * The lock is held only for the single-frame build (a cheap O(edges) scan +
        <=frame_size payloads), never for a whole-graph materialization -- so even
        when a frame IS produced, Syl waits milliseconds, not the whole dump.

    READ-ONLY with respect to the graph (no ng.save() -- nothing changed).
    """
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    out_path = data.get("out_path")
    if not out_path:
        return {"ok": False, "error": "out_path required"}

    # Gate BEFORE the lock (the co-tenant back-off): if the box is under load or
    # low on RAM, defer without contending for Syl's graph lock at all.
    try:
        from cc_topology_export import _leg2_resource_gate, export_cc_topology_frame
    except Exception as exc:
        logger.warning("CC topology frame export unavailable (non-fatal): %s", exc)
        return {"ok": False, "error": str(exc)}
    reason = _leg2_resource_gate()
    if reason is not None:
        logger.info("CC topology frame: deferring pre-lock -- resource gate: %s", reason)
        return {"ok": True, "gated": True, "gate_reason": reason, "exported_nodes": 0}

    kwargs = {"machine_id": data.get("machine_id"), "skip_resource_gate": True}
    if data.get("frame_size"):
        kwargs["frame_size"] = int(data["frame_size"])
    if data.get("overflow_factor"):
        kwargs["overflow_factor"] = int(data["overflow_factor"])
    if data.get("exclude_ids"):
        kwargs["exclude_ids"] = set(data["exclude_ids"])
    try:
        with ng.graph._concurrent_lock:
            stats = export_cc_topology_frame(ng.graph, ng.vector_db, out_path, **kwargs)
    except Exception as exc:
        logger.warning("CC topology frame export failed (non-fatal): %s", exc)
        return {"ok": False, "error": str(exc)}
    logger.info("CC topology frame -> %s: %s", out_path, stats)
    return {"ok": True, **stats}


_DISPATCH = {
    "ping": _handle_ping,
    "status": _handle_status,
    "export": _handle_export,
    "import": _handle_import,
    "drain_conduit": _handle_drain_conduit,
    "export_topology": _handle_export_topology,
    "export_topology_frame": _handle_export_topology_frame,
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
                # Corpus Callosum Leg 1 (#70): the conduit drain USED to run here.
                # Moved out 2026-07-28 -- a 60s pulse is the wrong home for merge
                # absorption. FatherGraph Finding 1 (never bulk-dump; batch ~20-30)
                # and Finding 3 (250 idle steps BETWEEN batches -- "not optional,
                # it's what makes merge work", 47%->74% accuracy) require batched
                # ingestion with homeostatic consolidation between batches, and
                # those idle steps must not block this pulse. It now runs in
                # docs/scripts/cc-ng-sync.py -- the nightly path it replaces,
                # which already carries CC_NG_BATCH_SIZE=25/CC_NG_IDLE_STEPS=250.
                cc_update_probation(_STATE.cc_ng.graph)
                surface_wants(_STATE.cc_ng.graph, _STATE.cc_ng.vector_db)
                generate_emergent_want(_STATE.cc_ng.graph, _STATE.cc_ng.vector_db)
            except Exception as exc:
                logger.debug("ingest-tract/probation/surface_wants/emergent_want failed (non-fatal): %s", exc)
            # Commons persist (#84) — same cadence as CC's own checkpoint above, but
            # an independent file under CC_NG_WORKSPACE. Never touches Syl's Commons:
            # persist_cc_commons writes only the medium get_cc_commons built, which is
            # constructed directly (never via canonical get_commons) precisely because
            # this code runs inside her process.
            try:
                from cc_ng_organism import persist_cc_commons
                persist_cc_commons(CC_NG_WORKSPACE)
            except Exception as exc:
                logger.debug("CC Commons persist failed (non-fatal): %s", exc)
        except Exception as exc:
            logger.warning("CC autosave failed: %s", exc)


# =============================================================================
# CC Tonic idle watcher (2026-07-23 parity wiring, gated OFF by default)
# =============================================================================
# Faithful mirror of neurograph_rpc.py's _tonic_check_idle/_tonic_idle_pulse_
# loop/_start_tonic_idle_watcher (Syl's proven, live pattern) -- scoped to CC.
# Every function below reads/writes ONLY _STATE.cc_ng. Never _memory.
CC_HOST_TONIC_IDLE_SECS = float(os.environ.get("CC_HOST_TONIC_IDLE_SECS", "90"))
CC_HOST_TONIC_IDLE_CHECK_SECS = float(os.environ.get("CC_HOST_TONIC_IDLE_CHECK_SECS", "30"))
_cc_tonic_idle_thread: Optional[threading.Thread] = None
_cc_tonic_idle_shutdown = threading.Event()


def _cc_tonic_check_idle(now: float) -> bool:
    """Drop CC's own Tonic into latent mode if the conversation has gone
    quiet past the idle threshold. Returns True if it transitioned.
    Pure/testable. Reads ONLY _STATE.cc_ng._tonic_thread -- never _memory.
    """
    ng = _STATE.cc_ng
    tonic = getattr(ng, "_tonic_thread", None) if ng is not None else None
    if tonic is None:
        return False
    if not getattr(tonic, "_in_conversation", False):
        return False
    last = getattr(tonic, "_last_message_time", 0.0)
    if last <= 0.0 or (now - last) < CC_HOST_TONIC_IDLE_SECS:
        return False
    try:
        tonic.conversation_ended()
        logger.info("CC Tonic: idle %.0fs >= %.0fs — dropped to latent mode",
                    now - last, CC_HOST_TONIC_IDLE_SECS)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("CC Tonic idle transition failed: %s", exc)
        return False


def _cc_tonic_idle_pulse_loop() -> None:
    logger.info("CC Tonic idle watcher started (idle=%.0fs, check=%.0fs)",
                CC_HOST_TONIC_IDLE_SECS, CC_HOST_TONIC_IDLE_CHECK_SECS)
    while not _cc_tonic_idle_shutdown.is_set():
        try:
            _cc_tonic_check_idle(time.time())
        except Exception:
            logger.exception("CC Tonic idle watcher tick failed")
        _cc_tonic_idle_shutdown.wait(timeout=CC_HOST_TONIC_IDLE_CHECK_SECS)
    logger.info("CC Tonic idle watcher stopped")


def _start_cc_tonic_idle_watcher() -> None:
    """Start the CC Tonic idle watcher thread. Idempotent. Gated OFF by
    default -- CC_HOST_TONIC_IDLE_ENABLED must be explicitly set to "1" for
    a thread to actually spin up (LAW 5)."""
    global _cc_tonic_idle_thread
    if os.environ.get("CC_HOST_TONIC_IDLE_ENABLED", "0") != "1":
        return
    if _cc_tonic_idle_thread is not None and _cc_tonic_idle_thread.is_alive():
        return
    _cc_tonic_idle_shutdown.clear()
    _cc_tonic_idle_thread = threading.Thread(
        target=_cc_tonic_idle_pulse_loop,
        name="cc-ng-tonic-idle-watcher",
        daemon=True,
    )
    _cc_tonic_idle_thread.start()


# =============================================================================
# CC Dream consolidation pulse (2026-07-23 parity wiring, gated OFF by default)
# =============================================================================
# Faithful mirror of neurograph_rpc.py's _dream_gate_open/_dream_
# consolidation_pulse_loop/_start_dream_consolidation_pulse (Syl's #381-B
# pattern) -- scoped to CC. Reads/writes ONLY _STATE.cc_ng / _STATE.commons
# (CC's own isolated Commons instance -- NEVER commons.get_commons(), which
# would return SYL'S Commons singleton in this shared process). Never _memory.
CC_HOST_DREAM_IDLE_SECS = float(os.environ.get("CC_HOST_DREAM_IDLE_SECS", "1800"))
CC_HOST_DREAM_MIN_INTERVAL_SECS = float(os.environ.get("CC_HOST_DREAM_MIN_INTERVAL_SECS", "21600"))
CC_HOST_DREAM_TICK_SECS = float(os.environ.get("CC_HOST_DREAM_TICK_SECS", "60"))
CC_HOST_DREAM_ALERT_SECS = float(os.environ.get("CC_HOST_DREAM_ALERT_SECS", "86400"))
_cc_dream_thread: Optional[threading.Thread] = None
_cc_dream_shutdown = threading.Event()
_cc_dream_last_pass_ts = 0.0


def _cc_dream_gate_open(now: float, last_turn_ts: float, arousal: str,
                         last_pass_ts: float) -> bool:
    """Idle long enough, not SYMPATHETIC, rate limit satisfied. Pure
    function for testability -- identical shape to Syl's _dream_gate_open.
    """
    return (
        (now - last_turn_ts) >= CC_HOST_DREAM_IDLE_SECS
        and arousal != "SYMPATHETIC"
        and (now - last_pass_ts) >= CC_HOST_DREAM_MIN_INTERVAL_SECS
    )


def _cc_dream_consolidation_pulse_loop() -> None:
    """Runs CC's own graph.consolidate_hyperedges() (shed + seatbelt-merge +
    subsume + dedup) then dedup_and_split_oversized_hyperedges() (#147 seam-split of any
    survivor still over he_max_members) during CC's quiet hours only. Never
    forces while CC's own Commons reports SYMPATHETIC arousal -- mirrors Syl's
    constraint: the pruning is dreamed, not felt. The seam-split is gated OFF by
    default (he_split_oversized_enabled, LAW 5) -- a no-op until flipped."""
    global _cc_dream_last_pass_ts
    _cc_dream_last_pass_ts = time.time()  # boot counts as activity
    _cc_last_alert_ts = 0.0
    logger.info(
        "CC Dream consolidation pulse started (idle>=%.0fs, min interval %.0fs)",
        CC_HOST_DREAM_IDLE_SECS, CC_HOST_DREAM_MIN_INTERVAL_SECS,
    )
    while not _cc_dream_shutdown.is_set():
        try:
            ng = _STATE.cc_ng
            if ng is not None:
                arousal = "PARASYMPATHETIC"
                try:
                    commons = _STATE.commons
                    if commons is not None:
                        arousal = commons.read_arousal()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("CC dream pulse: arousal read failed: %s", exc)
                tonic = getattr(ng, "_tonic_thread", None)
                last_turn_ts = getattr(tonic, "_last_message_time", 0.0) if tonic is not None else 0.0
                now = time.time()
                if _cc_dream_gate_open(now, last_turn_ts, arousal, _cc_dream_last_pass_ts):
                    lock = getattr(ng.graph, "_step_lock", None)
                    t0 = time.monotonic()
                    # Forced order (#147): consolidate (shed + seatbelt-merge +
                    # subsume + dedup) FIRST, then seam-split any survivor still
                    # over he_max_members. dedup_and_split_oversized_hyperedges is gated OFF
                    # by default (he_split_oversized_enabled, LAW 5) -> a
                    # guaranteed no-op returning 0 until the knob is flipped.
                    if lock is not None:
                        with lock:
                            merged = ng.graph.consolidate_hyperedges()
                            split = ng.graph.dedup_and_split_oversized_hyperedges(ng.vector_db)
                    else:
                        merged = ng.graph.consolidate_hyperedges()
                        split = ng.graph.dedup_and_split_oversized_hyperedges(ng.vector_db)
                    _cc_dream_last_pass_ts = time.time()
                    logger.info(
                        "CC dream consolidation pass complete: %d merged/archived, "
                        "%d oversized seam-split in %.1fs",
                        merged, split, time.monotonic() - t0,
                    )
                elif (now - _cc_dream_last_pass_ts) >= CC_HOST_DREAM_ALERT_SECS and \
                        (now - _cc_last_alert_ts) >= CC_HOST_DREAM_ALERT_SECS:
                    _cc_last_alert_ts = now
                    logger.error(
                        "No CC dream consolidation in %.0fh — the idle/arousal gate "
                        "never opened. ALERT ONLY: the pass is never forced while "
                        "CC is active (mirrors Syl's #381-B constraint).",
                        (now - _cc_dream_last_pass_ts) / 3600.0,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CC dream pulse iteration failed (non-fatal): %s", exc)
        _cc_dream_shutdown.wait(CC_HOST_DREAM_TICK_SECS)
    logger.info("CC Dream consolidation pulse stopped")


def _start_cc_dream_consolidation_pulse() -> None:
    """Start the CC dream consolidation pulse thread. Idempotent. Gated OFF
    by default -- CC_HOST_DREAM_ENABLED must be explicitly set to "1" for a
    thread to actually spin up (LAW 5)."""
    global _cc_dream_thread
    if os.environ.get("CC_HOST_DREAM_ENABLED", "0") != "1":
        return
    if _cc_dream_thread is not None and _cc_dream_thread.is_alive():
        return
    _cc_dream_shutdown.clear()
    _cc_dream_thread = threading.Thread(
        target=_cc_dream_consolidation_pulse_loop,
        name="cc-ng-dream-consolidation-pulse",
        daemon=True,
    )
    _cc_dream_thread.start()


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
        from cc_ng_organism import cc_gsg_backfill
        _stamped = cc_gsg_backfill(cc_ng.graph, cc_ng.vector_db)
        if _stamped:
            logger.info("CC GSG backfill at init: %d nodes stamped (persists via autosave)", _stamped)
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

    # CC Tonic idle watcher + Dream consolidation pulse (2026-07-23 parity
    # wiring). Each _start_* fn checks its own CC_HOST_*_ENABLED gate
    # (default OFF, LAW 5) and no-ops if unset -- landing this dark. Wrapped
    # in try/except: a failure here must never break CC NG hosting itself.
    try:
        _start_cc_tonic_idle_watcher()
    except Exception:
        logger.exception("CC Tonic idle watcher failed to start (non-fatal)")
    try:
        _start_cc_dream_consolidation_pulse()
    except Exception:
        logger.exception("CC Dream consolidation pulse failed to start (non-fatal)")

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
    # Signal the CC Tonic idle watcher / Dream consolidation pulse to stop
    # (2026-07-23 parity wiring). Harmless no-op if either was never started
    # (gate-off default) -- setting an unused Event costs nothing, and both
    # threads are daemon=True regardless so they'd die with the process anyway.
    _cc_tonic_idle_shutdown.set()
    _cc_dream_shutdown.set()
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

    # Commons persist (#84) — independent of the graph save above, and outside its
    # lock: a refused/failed save-guarded graph save must not cost us the medium too.
    try:
        from cc_ng_organism import persist_cc_commons
        if persist_cc_commons(CC_NG_WORKSPACE):
            logger.info("CC Commons persisted on shutdown")
    except Exception as exc:
        logger.debug("CC Commons persist on shutdown failed (non-fatal): %s", exc)

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
