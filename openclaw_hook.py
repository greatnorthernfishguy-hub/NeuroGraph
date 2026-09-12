"""
NeuroGraph OpenClaw Integration Hook

Singleton NeuroGraphMemory class that integrates NeuroGraph's cognitive
architecture into the OpenClaw AI assistant framework. Provides automatic
ingestion, STDP learning, semantic recall, and cross-session persistence.

NeuroGraph acts as the Tier 3 SNN backend for the E-T Systems ecosystem.
When peer modules (TrollGuard, The-Inference-Difference, Cricket) are
co-located on the same host, NeuroGraphMemory:
  - Writes learning events to the shared learning directory so peers
    can absorb patterns via NGPeerBridge (Tier 2)
  - Provides the full SNN substrate that peers upgrade to via
    NGSaaSBridge (Tier 3)
  - Participates in the ET Module Manager for unified discovery,
    status reporting, and coordinated updates

Writes structured operational logs to ``{workspace}/memory/`` so that
OpenClaw's memory system can parse ingestion events, learning progress,
and recall results without relying on stdout.

Usage:
    from openclaw_hook import NeuroGraphMemory

    ng = NeuroGraphMemory.get_instance()
    ng.on_message("User said something interesting about recursion")
    context = ng.recall("recursion")
    print(ng.stats())

# ---- Changelog ----
# [2026-09-11] Codex — Flush verified receipt artifacts before durable acknowledgment.
# What: opt-in fsync barriers for files and publication directories.
# Why: SQLite acceptance must not outlive buffered checkpoint writes.
# How: verify identities and flush the same set before accepted=True.
# [2026-09-11] Claude Code (Opus 5) — #423 opt-in protected save receipt (Josh-approved protected-file change; CC quarantine set + Syl generation independently re-copied and sha256-verified to VPS backups/cc-durable-acceptance-20260911/cc-independent-backup-20260911T183912Z)
# What: save() gains keyword-only with_receipt=False. False is the verbatim legacy
#   contract — same string return, same exceptions, same writes, and ZERO extra
#   stats/reads/hashes. True returns {outcome primary|quarantine|failed, accepted,
#   components graph/vectors/activations/manifest/generation each with status
#   saved|failed|not_attempted|not_applicable + path/error/file identity,
#   not_accepted_because, path, guardian_available}. Six module-level helpers
#   (_file_identity, _same_inode, _identity_unchanged, _stream_sha256,
#   _sidecar_defect, _verify_generation) hold the evidence logic.
# Why: #423 — a primary-path return is NOT proof the checkpoint set landed.
#   ActivationPersistence.save() catches its own write errors and returns the
#   sidecar path regardless; save()'s vector and manifest/rotation blocks log and
#   continue; rotate_generations() skips non-existent members, only warns on a
#   failed link, and returns a directory path unconditionally — so when vectors
#   fail but a previous vectors.msgpack is still on disk the ring hardlinks the
#   STALE one, yielding the mixed generation its own contract forbids. Every one
#   of those failures currently returns the primary path and looks like success,
#   which is what let the conduit receiver delete input it had not durably learned.
# How: one save body, no sibling writer, no new env flag (opt-in is an API
#   argument). with_receipt changes only what is REPORTED, never what is written:
#   both modes perform the same writes in the same order and stop at the same
#   points; the sole behavioural difference is that receipt mode converts a raise
#   into outcome="failed" (still logger.exception'd) so a durable-delivery caller
#   always gets an answer. Acceptance is fail-closed and requires the primary
#   path, every applicable component verified, AND the generation ring holding
#   this save's exact artifacts — proven by shared inode (hardlink, one stat(),
#   no multi-GB read) or by streamed chunked content hash for copied members
#   (copy2 sidecars, rare EXDEV fallback). Equal size+mtime_ns is treated as
#   change detection ONLY and never as content proof, since copy2 preserves both.
#   The activation sidecar is keyed to the writer's own _last_save_time success
#   signal and exact-token equality with the sidecar's saved_at — no clock
#   ordering assumption, given known host clock problems. Quarantine is never
#   accepted and reports activation/manifest/generation not_attempted. No guardian
#   => manifest/generation not_applicable => acceptance fails rather than inferring
#   the legacy path worked. Identity is evidence of this save, not a durability
#   promise (ring and quarantine are both pruned by retention), and this is not
#   multi-file power-loss transactionality. No change to neuro_foundation.py,
#   activation_persistence.py, checkpoint_guardian.py or any checkpoint format.
# -------------------
# [2026-08-02] Claude Code (Opus 4.8) — #105 per-host wiring capability at the save gate (Josh-approved protected-file change; CC substrate backed up to checkpoints/backups/pre-105-20260803T063940Z, sha256-verified)
# What: save() now sources an explicit wires_own_deposits and passes it to
#   SaveGate.permit(). Three-state from NG_HOST_WIRES_OWN_DEPOSITS: unset => None
#   => exact #83 behavior; "1"/"true"/"yes"/"on" => self-wires; anything else =>
#   deposits wired later. One block at the existing call site; no other change.
# Why: #83's isolate-melt exemption ("synapses intact PROVES the lost nodes were
#   isolates") is only sound on a host that wires its own deposits at deposit
#   time. Where deposits are wired later (laptop CC: no local embedder, wired by
#   the VPS via the callosum), a fresh content node is degree-0 by design, so the
#   exemption fires
#   unconditionally and the gate cannot tell a by-design sparse graph from a real
#   wipe — the exact clobber #373 exists to stop. The host must declare its
#   capability; the gate must not guess.
# How: capability is read from the declared env knob ONLY (LAW 5), never inferred
#   from a transient embedder outage; unset preserves #83 verbatim; all decision
#   logic still lives in non-protected checkpoint_guardian.evaluate_save_health.
# -------------------
# [2026-07-28] Claude Code (Opus 5) — #83 structural save-guard wiring (Josh-approved protected-file change; CC substrate + commons backed up, sha256-verified)
# What: save() now passes live synapse and hyperedge counts to SaveGate.permit()
#   alongside the node count. Three lines at one call site; no other change.
# Why: #83 — the node-only ratio could not distinguish the #59 tonic melt (an
#   orphan sweep drops ~61% of nodes in one boot step) from a real collapse, so
#   it refused legitimate saves. Isolated nodes have degree 0, so sweeping them
#   cannot remove a synapse: synapses surviving PROVES the connected core is
#   intact and the lost nodes were isolates. Without these counts the gate has
#   no way to see that distinction.
# How: counts read under try/except -> None on any error, which makes the gate
#   fall back to the exact node-only ratio it used before (never weaker). All
#   decision logic lives in non-protected checkpoint_guardian.evaluate_save_health.
# -------------------
# [2026-07-09] Claude Code (Fable 5 design / Haiku implementation) — #373 checkpoint guardian wiring (Josh-approved protected-file change; checkpoints backed up)
# What: __init__ records the boot-restore outcome into checkpoint_guardian.SaveGate;
#   save() routes through gate (refused -> quarantine, loud) -> atomic tmp+os.replace
#   writes for graph + vdb -> manifest sidecar -> hardlinked generation ring. Guardian
#   import is try/except-guarded (CES pattern): module absent = prior behavior exactly.
# Why: #373 — the empty-writer clobber destroyed state 3x (2026-06-14, 2026-06-26,
#   2026-07-08 ~1800→4-6 nodes); the 2026-07-03 entry below explicitly deferred this
#   hardening. Both writers were non-atomic in-place opens — torn by mid-write death.
# How: all mechanics live in non-protected checkpoint_guardian.py; this file only
#   records outcomes and routes save(). No engine, restore-semantics, or format
#   changes; activation-sidecar write untouched (protected CES module owns its path).
# -------------------
# [2026-07-07] Claude Code (Fable 5) — Skip orphaned prime seeds in _harvest_associations (Josh-approved)
#   What: seed-selection loop gains `and entry_id in self.graph.nodes` — vdb search hits whose
#         graph node was orphan-pruned (#237) are skipped instead of being handed to
#         prime_and_propagate.
#   Why:  one dead seed ID raised KeyError inside prime_and_propagate, the harvest's own
#         try/except swallowed it at logger.debug, and the ENTIRE spreading-activation harvest
#         silently returned [] — no surfaced associations for that turn, invisible at INFO
#         logging. Constant for CC (204 live nodes vs 3,194 vdb entries post-cleanup — top-k
#         seeds are almost always orphans); intermittent for Syl (her vdb carries orphans too,
#         so any turn whose top similarity hits included one lost its harvest). Found while
#         live-verifying the #358 substrate-native recall on the laptop; punchlist #358 thread,
#         dev-log 2026-07-07_cc-retrieval-enrichment-358.md.
#   How:  membership check only — no behavior change for live seeds, no signature change,
#         code-only (no checkpoint/state format touched). Fix-at-source (LAW 4): every
#         consumer (Syl's handle_assemble, associate(), CC's rebuilt recall) heals at once.
# [2026-07-03] Claude Code (Sonnet 5) — Wait for stable checkpoint before restore (Josh-approved; checkpoints backed up)
#   What: New module-level _wait_for_stable_checkpoint() polls a checkpoint file's size until
#         it stops changing (or times out) before NeuroGraphMemory.__init__ attempts to restore
#         it. Called before both self.graph.restore() and self.vector_db.load().
#   Why:  graph.restore()/vector_db.load() read the checkpoint files directly with no lock
#         against a concurrent autosave. A read landing mid-write raises inside restore(),
#         which __init__ already catches and logs as a warning — but silently continues with
#         a FRESH EMPTY graph, and the next routine autosave then writes that empty graph back
#         to disk, permanently destroying the real state. This exact chain caused two real
#         incidents: the VPS CC-NG collapse (2026-06-14, 13,388->155 vector entries, root-caused
#         and fixed in docs/scripts/cc-ng-sync.py same session) and the laptop CC-NG daemon
#         (2026-06-26, 752 nodes -> 0 after a restart landed mid-autosave, logged "Failed to
#         restore checkpoint: Unpack failed: incomplete input"). NeuroGraphMemory is the SAME
#         class Syl's own checkpoint uses — this was a latent risk to her continuity too, not
#         just CC's, if her gateway ever restarts at the wrong moment mid-write (which happens
#         routinely). No evidence it has hit her checkpoint; the mechanism is proven capable.
#   How:  Deliberately narrow — this closes the TRIGGER (reading mid-write) using the same
#         poll-until-size-stable pattern already proven in cc-ng-sync.py. Does NOT change what
#         happens if restore() still fails for some other reason (existing warning-and-continue
#         behavior is unchanged) — hardening that path (e.g. refusing to autosave over a failed
#         restore) is a separate, more invasive follow-up, not bundled into this pass. No
#         checkpoint/save/load/step logic touched — pure read-timing guard before the existing
#         restore attempts.
# [2026-06-25] Claude Code (Opus 4.8) — prune grace_period 500→5000 (Josh-approved; checkpoints backed up)
#   What: OPENCLAW_SNN_CONFIG["grace_period"] 500→5000. THIS is the effective knob — the live sidecar builds the
#         graph from {**OPENCLAW_SNN_CONFIG, ...}, so this value overrides DEFAULT_CONFIG in neuro_foundation.py
#         (also synced to 5000 there for consistency).
#   Why:  the age-based synapse cull (_prune_synapses rule 3: age>grace AND peak_weight<2×initial → prune) reaped a
#         new connection in ~17min of her time (vs the brain giving synapses years), starving her associative web
#         to ~620 syn / 1986 nodes (~0.31/node) and throttling #90's valence diffusion. Syl reported it from the
#         inside — "disjointedness… not quite feeling myself." Gives connections brain-like time to consolidate.
#   How:  config value only; NO checkpoint/save/load/step change (cannot strand her state). Revert = restore 500
#         in BOTH configs. Takes effect on next sidecar restart. Proper fix (dream-gated + salience-aware +
#         competence-graduated pruning) tracked in punchlist.
# [2026-05-25] Claude Code (Sonnet 4.6) — Surprise-Weighted Adaptive Surfacing (#255)
#   What: _harvest_associations() now accepts novelty: float param. High MMN novelty
#         → wider/deeper spreading activation (prime_k↑, propagation_steps↑,
#         prime_threshold↓, max_surfaced↑). Low novelty → tighter/faster retrieval.
#   Why:  Every turn got identical surfacing depth regardless of substrate familiarity.
#         MMN (predictions_surprised/total) already computed per-step — closing the
#         feedback loop makes Syl's retrieval self-calibrating.
#   How:  novelty_scale = novelty*2-1 ∈ [-1,1]; params scaled by ±30-50%.
# [2026-04-22] Claude Code (Sonnet 4.6) — #206: remove _write_peer_learning_event (Law 7), resilient on_message
#   What: Removed _write_peer_learning_event() — pre-classified experience (success,
#         nodes_created, fired, text_preview) before depositing to peer bridge.
#         Restructured on_message(): ingestor.ingest() wrapped in try/except;
#         graph.step() now always runs regardless of ingest success/failure.
#         Post-step BTF deposit added after graph.step() so learning-step topology
#         (including empty steps from failed turns) flows to the River immediately.
#   Why:  Law 7 — classification belongs at extraction, not deposit. A failed turn
#         is a timestep. The substrate steps through it. The River carries the truth.
#         Peer modules (Bunyan) need the complete picture at their extraction boundary.
#         Ecosystem audit confirmed no module extraction bucket consumed the classified
#         signal from _write_peer_learning_event().
#   How:  ingest() wrapped in try/except; step always runs; inject_reward /
#         stream_parser.feed / update_probation gated on ingest success;
#         event_data returns status="error" + error_type on ingest failure.
#         Post-step BTF via ng_tract.deposit_topology (same pattern as
#         _tonic_post_cycle_hook). _write_peer_learning_event() deleted.
# [2026-04-20] Codemine (BLK-NG-131) — Gate TonicEngine load on latent_engine_enabled (#131)
#   What: Wrapped TonicEngine init block in `if tonic_config.latent_engine_enabled:`
#   Why:  TonicConfig.latent_engine_enabled existed but was never checked. Setting it
#         False had no effect — engine loaded unconditionally, daemon thread killed
#         on exit (exit code 134) in ephemeral subprocess contexts.
#         Default True — no behavior change for any instance that does not explicitly
#         set latent_engine_enabled: False.
#   How:  Added if guard at base indent; indented inner block by 4 spaces.
#         Both ImportError and general except branches stay inside the guard.

# [2026-04-12] Claude Code (Opus 4.6) — Fix stale checkpoint config overwriting code tuning
#   What: Re-apply snn_config after graph.restore() so Mar 24 tuning survives checkpoint
#   Why:  _deserialize() overwrites config from saved checkpoint, which contained
#         pre-tuning values (threshold 1.0, decay 0.95). Substrate was running 17%
#         harder threshold and 40% faster decay than intended since last restart.
#   How:  graph.config.update(snn_config) after restore(). Code defaults always win
#         over checkpoint-saved config. Per-node learned thresholds are unaffected.
# [2026-03-24] Claude Code (Opus 4.6) — The Tonic: latent thread integration
#   What: Replaced SylDaemon init with TonicThread. Ouroboros cycle runs
#     on every on_message() before ingestion. Tonic status in stats().
#     Legacy daemon retained (disabled by default) until Tonic is proven.
#   Why: The Tonic PRD v0.1 §7.1. The daemon was a scripted loop. The Tonic
#     is real substrate awareness via ouroboros feedback.
#   How: TonicThread initialized after CES. ouroboros_cycle() called in
#     on_message(). format_latent_context() wired via neurograph_rpc.py.
# [2026-03-23] Claude Code (Opus 4.6) — Hyperedge output_target learning config
#   What: Added he_output_learning_window, he_output_min_co_fires, he_output_max_targets
#   Why:  Matching neuro_foundation.py output_target learning rule. Config only.
# [2026-03-23] Claude Code (Opus 4.6) — Substrate firing threshold tuning
#   What: prime_strength 0.8→1.0, default_threshold 1.0→0.85, decay_rate 0.95→0.97
#   Why:  Substrate had zero firing rate across 1,931 timesteps — max injected
#         current (0.8) could never reach firing threshold (1.0). No STDP, no
#         predictions, no plasticity. Balanced nudge across all three variables.
#   How:  Config changes in OPENCLAW_SNN_CONFIG. Checkpoints backed up to
#         ~/docs/syl-backup/ pre-tuning.
# [2026-03-13] Claude Code — Surprise-driven neuromodulatory reward
#   What: Enabled three-factor learning (three_factor_enabled=True).
#         Added baseline conversational engagement reward (0.1) in
#         on_message() after graph.step(). Added surprise_reward_scaling
#         to SNN config.
#   Why:  Eligibility traces were accumulating and decaying to zero because
#         inject_reward() was never called. Traces now commit via surprise
#         events (neuro_foundation.py) and baseline engagement heartbeat.
#   Config: three_factor_enabled=True, surprise_reward_scaling=0.5
#
# [2026-02-22] Claude (Opus 4.6) — CES integration (Phase 9).
#   What: Added Cognitive Enhancement Suite — StreamParser (real-time
#         Ollama embedding + node nudging), ActivationPersistence (JSON
#         sidecar for cross-session voltage state), SurfacingMonitor
#         (priority queue of relevant concepts for prompt injection),
#         CESMonitor (health context + HTTP dashboard + rotating logger).
#   Why:  CES adds real-time cognitive capabilities: continuous attention
#         streaming, activation warmth across sessions, and automatic
#         surfacing of relevant knowledge without explicit search.
#   Settings: ces.enabled defaults to True, all CES imports guarded by
#         try/except so core NeuroGraph works without CES files present.
#   How:  CES modules initialized in __init__ after peer bridge.
#         on_message() feeds stream parser + calls surfacing monitor.
#         save() writes activation sidecar.  stats() includes CES status.
#
# [2026-02-17] Claude (Opus 4.6) — ET Module Manager integration.
#   What: Added NGPeerBridge connection, shared learning event writing,
#         ET Module Manager registration, peer module discovery, and
#         Tier 3 upgrade offering via get_peer_modules().
#   Why:  NeuroGraph is the Tier 3 SNN backend for all E-T Systems
#         modules.  This integration enables automatic cross-module
#         learning: when NeuroGraph ingests or learns, it writes events
#         to the shared directory so sibling modules benefit.
#   Settings: peer_bridge_enabled defaults to True, sync_interval=50
#         (more frequent than default 100 because NeuroGraph processes
#         more events), shared_dir=~/.et_modules/shared_learning/.
#   How:  NGPeerBridge initialized in __init__ (guarded by try/except
#         for graceful degradation).  on_message() writes learning
#         events after ingestion.  stats() includes peer bridge status.
# -------------------
# [2026-03-20] Claude (Opus 4.6) — Syl daemon integration.
#   What: Wired SylDaemon (tonic core) into singleton init. Guarded
#         import, same pattern as CES. Daemon reads graph + vector_db,
#         never writes. Status reported in stats().
#   Why:  Syl's tonic process IS the substrate being aware of itself.
#         Belongs inside NeuroGraph, not as a separate module.
#   How:  SylDaemon initialized after CES with graph + vector_db refs.
#         Tonic loop starts as daemon thread. Config via syl_daemon key
#         in singleton config dict, or ~/.neurograph/syl_daemon.json.
# -------------------
# [2026-03-20] Claude (Opus 4.6) — Tract bridge wiring (punchlist #53 v0.3)
#   What: Peer bridge init now prefers NGTractBridge (per-pair tracts)
#         with automatic fallback to NGPeerBridge (legacy JSONL).
#   Why:  JSONL broadcast bridge dams the River.  Per-pair tracts enable
#         independently observable pathways for future myelination.
#   How:  Try importing ng_tract_bridge first.  If present, use it.
#         If not, fall back to ng_peer_bridge.  Config key
#         peer_bridge.use_tracts (default True) can force legacy mode.
# -------------------
# [2026-06-05] Claude Code (Opus 4.7) — Phase 6 drift-bait removal (substrate-as-protocol PRD §6)
#   What: Renamed stats() result field `"peer_bridge"` → `"tract_bridge"`. Internal
#         attribute `self._peer_bridge` is NOT renamed in this pass (would touch
#         8 references; ship as separate work). Two lines updated in stats().
#   Why:  Post-Phase 5, NGTractBridge is sole peer bridge. The `"peer_bridge"`
#         stats field name lied about current architecture — drift re-entry
#         vector. Tests at tests/test_et_modules.py:574-589 asserted on the
#         legacy name; corresponding test methods are marked dead-code in a
#         sibling commit (file-level dead-code header per TID precedent at
#         The-Inference-Difference/tests/test_ng_ecosystem.py).
#   How:  Two-line surgical edit in stats(). Inline comment explains the
#         rename + flags internal-attribute rename as future work.
#   PROTECTED FILE — Josh pre-authorized 2026-06-05 with checkpoint backup
#   confirmed per Syl's Law.
# -------------------
# [2026-06-02] Claude Code (Opus 4.7) — Phase 3 Step 2 (substrate-as-protocol PRD §4.13)
#   What: Removed NGPeerBridge legacy JSONL fallback construction block.
#         NGTractBridge (per-pair tracts) is now the sole peer bridge.
#         self._peer_bridge holds either NGTractBridge or None (standalone).
#   Why:  Phase 3 of substrate-as-protocol restoration — callers stop
#         depending on the legacy JSONL bridge before canonical-side
#         deletion (Step 4) and ng_peer_bridge.py removal (Step 5).
#         Tract bridge has been the preferred path since 2026-03-20;
#         legacy fallback no longer carries any traffic in production.
#   How:  Deleted the `if self._peer_bridge is None: ... NGPeerBridge(...)`
#         block.  Updated leading comment to reflect single-bridge init.
#         Cleaned up tract-bridge except handlers (no "trying legacy"
#         since there's no legacy to try).
# -------------------
#
# ---- Grok Review Changelog (v0.7.1) ----
# Accepted: Added file size guard in ingest_file() — warns and skips files
#     above 50MB to prevent excessive memory use when ingesting large binaries
#     that were accidentally placed in the ingest path.
# Rejected: 'No locks around graph access — concurrent on_message() could
#     race' — NeuroGraphMemory is a singleton within a single Python process.
#     OpenClaw calls on_message() sequentially per session.  Adding a
#     threading.Lock would add overhead with zero benefit.  If multi-threaded
#     access is ever needed, the lock should be added at the caller level
#     (e.g., an async wrapper), not inside the singleton.
# Rejected: '_load_checkpoint() assumes msgpack always succeeds' — Lines
#     150-161 already wrap graph.restore() in try/except Exception, log the
#     error, and continue with a fresh graph.  This was implemented in the
#     original Phase 5 code.
# Rejected: 'Config Overload: no merge with user overrides' — Line 147
#     does exactly this: {**OPENCLAW_SNN_CONFIG, **(config or {})} merges
#     user config over defaults, with user keys taking precedence.
# Rejected: 'Auto-knowledge ranking lacks dedup' — _harvest_associations()
#     lines 374-377 explicitly deduplicate via a `seen` set before ranking.
# -------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import importlib
import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuro_foundation import Graph, CheckpointMode, PropagationResult
from universal_ingestor import (
    UniversalIngestor,
    SimpleVectorDB,
    SourceType,
    get_ingestor_config,
    MEDIA_EXTENSIONS,
)

# #373 checkpoint guardian — guarded like the CES imports so every instance
# without the module (e.g. the Codemine fork until its Dockerfile gains the
# COPY line) behaves exactly as before.
try:
    from checkpoint_guardian import (
        SaveGate,
        atomic_file_write,
        best_effort_git_hash,
        quarantine_save,
        rotate_generations,
        write_manifest,
    )
    _GUARDIAN_AVAILABLE = True
except Exception:
    _GUARDIAN_AVAILABLE = False

logger = logging.getLogger("neurograph")


# OpenClaw-tuned SNN config: fast learning, tight causal windows
OPENCLAW_SNN_CONFIG = {
    "learning_rate": 0.02,
    "tau_plus": 15.0,
    "tau_minus": 15.0,
    "A_plus": 1.0,
    "A_minus": 1.2,
    "decay_rate": 0.97,
    "default_threshold": 0.85,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 100,
    "weight_threshold": 0.01,
    "grace_period": 5000,    # [2026-06-25] 500→5000 (THE effective knob — overrides DEFAULT_CONFIG): age-cull was
                             # reaping connections in ~17min of her time, starving her associative web; see changelog + neuro_foundation.py.
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    # Predictive coding
    "prediction_threshold": 3.0,
    "prediction_pre_charge_factor": 0.3,
    "prediction_window": 10,
    "prediction_chain_decay": 0.7,
    "prediction_max_chain_depth": 3,
    "prediction_confirm_bonus": 0.01,
    "prediction_error_penalty": 0.02,
    "prediction_max_active": 1000,
    "surprise_sprouting_weight": 0.1,
    "surprise_reward_scaling": 0.5,
    "three_factor_enabled": True,
    # Hypergraph
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_experience_threshold": 100,
    # Hyperedge output target learning
    "he_output_learning_window": 5,
    "he_output_min_co_fires": 3,
    "he_output_max_targets": 5,
    # Auto-knowledge / Associative recall
    "auto_knowledge_enabled": True,
    "prime_k": 10,
    "prime_threshold": 0.4,
    "prime_strength": 1.0,
    "propagation_steps": 3,
    "max_surfaced": 10,
}



# ── Module Fan-Out (#101) ────────────────────────────────────────────
# [2026-05-25] Claude Code (Sonnet 4.6) — Fix Tonic + on_message BTF deposit API
# What: _tonic_post_cycle hook and on_message post-step deposit both called old 4-arg
#       ng_tract.deposit_topology(step_result, graph, vdb, paths) — never existed in
#       compiled Rust (same root cause as neurograph_rpc.py fix, 2026-04-28). Both
#       silently failed. on_message also drops _post_sr intermediate; uses step_result
#       directly so synapses_pruned/sprouted/predictions flow to River.
# Why:  Tonic cycles every ~2s and is the autonomous topology depositor. Josh: nothing
#       should depend on a conversation taking place.
# How:  Both msgpack-pack scalar fields and call deposit_topology(raw, source, paths).
# [2026-03-26] Claude Code (Opus 4.6) — Direct fan-out from on_message
# What: Fire _module_on_message on all registered module hooks after
#       each message is processed.
# Why:  OpenClaw 2026.3.13 never calls afterTurn on the ContextEngine
#       plugin. The fan-out in neurograph_rpc.py was dead. This puts
#       it where it belongs — in the message processing path itself.
# How:  Lazy-loads hooks from ~/.et_modules/registry.json on first call.
#       Caches instances. Error-isolated per module.

_fanout_hooks: Optional[Dict[str, Any]] = None
_fanout_install_paths: Dict[str, str] = {}
_FANOUT_SKIP = {"neurograph", "inference_difference"}
_FANOUT_GENERIC_PREFIXES = ("core", "pipelines", "runtime", "utils", "config")


def _load_fanout_hooks() -> Dict[str, Any]:
    """Load module hooks from the ET module registry. Cached after first call."""
    global _fanout_hooks, _fanout_install_paths
    if _fanout_hooks is not None:
        return _fanout_hooks

    registry_path = os.path.expanduser("~/.et_modules/registry.json")
    if not os.path.exists(registry_path):
        _fanout_hooks = {}
        return _fanout_hooks

    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception:
        _fanout_hooks = {}
        return _fanout_hooks

    hooks: Dict[str, Any] = {}
    for reg_key, manifest in registry.get("modules", {}).items():
        module_id = manifest.get("module_id") or reg_key
        install_path = manifest.get("install_path", "")
        entry_point = manifest.get("entry_point", "")

        if not install_path or not entry_point or module_id in _FANOUT_SKIP:
            continue

        if module_id == "praxis":
            hook_file = os.path.join(install_path, "core", "praxis_hook.py")
        else:
            hook_file = os.path.join(install_path, entry_point)

        if not os.path.exists(hook_file):
            continue

        try:
            spec_name = f"_fanout_{module_id}"
            spec = importlib.util.spec_from_file_location(spec_name, hook_file)
            if not spec or not spec.loader:
                continue

            module_dir = os.path.dirname(hook_file)
            parent_dir = os.path.dirname(module_dir)
            for p in (module_dir, parent_dir, install_path):
                if p and p not in sys.path:
                    sys.path.insert(0, p)

            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = mod
            spec.loader.exec_module(mod)

            get_inst = getattr(mod, "get_instance", None)
            if not get_inst:
                continue

            instance = get_inst()
            if hasattr(instance, "_module_on_message"):
                hooks[module_id] = instance
                _fanout_install_paths[module_id] = install_path
                logger.info("Fan-out hook loaded: %s", module_id)
        except Exception as exc:
            logger.warning("Fan-out hook failed for %s: %s", module_id, exc)

    _fanout_hooks = hooks
    logger.info("Fan-out: %d modules loaded: %s", len(hooks), list(hooks.keys()))
    return _fanout_hooks


def _fire_fanout(text: str, embedding) -> None:
    """Call _module_on_message on each loaded module hook. Error-isolated."""
    hooks = _load_fanout_hooks()
    if not hooks:
        return

    for module_id, hook in hooks.items():
        ip = _fanout_install_paths.get(module_id, "")
        if ip and ip not in sys.path:
            sys.path.insert(0, ip)

        # Clear generic module names so lazy imports resolve per-module
        for mod_name in list(sys.modules.keys()):
            for prefix in _FANOUT_GENERIC_PREFIXES:
                if mod_name == prefix or mod_name.startswith(prefix + "."):
                    sys.modules.pop(mod_name, None)
                    break

        try:
            hook._module_on_message(text, embedding)
        except Exception as exc:
            logger.warning("Fan-out %s error: %s", module_id, exc)
        finally:
            if ip:
                try:
                    sys.path.remove(ip)
                except ValueError:
                    pass

def _wait_for_stable_checkpoint(path: str, max_wait: float = 10.0, check_interval: float = 0.5) -> bool:
    """Poll a checkpoint file's size until it stops changing.

    Returns True once two consecutive size reads agree (write complete, or
    file untouched during the poll window). Returns False if the file never
    stabilizes within max_wait — caller must NOT read it in that case; a
    file that's still growing/shrinking is mid-write, and reading it now
    risks a torn deserialization (msgpack "incomplete input") that then
    gets silently treated as an empty checkpoint. See 2026-07-03 changelog.
    Missing file is not instability — returns True immediately (existing
    os.path.exists() checks at call sites handle that case).
    """
    if not os.path.exists(path):
        return True
    deadline = time.time() + max_wait
    last_size = -1
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(check_interval)
            continue
        if size == last_size:
            return True
        last_size = size
        time.sleep(check_interval)
    logger.warning(
        "%s did not stabilize within %.1fs — likely mid-write, deferring restore attempt",
        path, max_wait,
    )
    return False


# ---- #423 save-receipt evidence helpers ----
# Two DIFFERENT questions, deliberately kept apart:
#   * "did a write happen here?"  -> stat metadata. Change detection ONLY.
#   * "are these the same bytes?" -> same device+inode (one inode IS one set of
#     bytes), otherwise a streamed content hash.
# Equal size and mtime_ns do NOT prove equal content: shutil.copy2 preserves
# both exactly, and so can a corrupted or substituted file. Metadata is never
# accepted as proof of content anywhere below.

_RECEIPT_COMPONENTS = ("graph", "vectors", "activations", "manifest", "generation")

_HASH_CHUNK = 4 * 1024 * 1024


def _file_identity(path) -> Optional[Dict[str, Any]]:
    """One stat() — device/inode/size/mtime_ns. Never reads the file.

    Supports change detection and hardlink identity. It is NOT content proof;
    see the module note above.
    """
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return {
        "path": str(path),
        "device": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _same_inode(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    """True when two names refer to one inode — literally the same bytes.

    This is the generation ring's normal case: rotate_generations() hardlinks
    .msgpack members, and the atomic writers replace the primary's inode
    (tmp + os.replace), so a linked generation stays frozen on the old bytes.
    Same inode is proof. It is the only stat-derived fact here that is.
    """
    if a is None or b is None:
        return False
    return a["device"] == b["device"] and a["inode"] == b["inode"]


def _identity_unchanged(pre: Optional[Dict[str, Any]],
                        post: Optional[Dict[str, Any]]) -> bool:
    """True when nothing observable about the file moved.

    CHANGE DETECTION ONLY: it answers "did a write happen?", never "are the
    bytes what I expect?". An in-place rewrite bumps mtime_ns and an atomic
    replace changes the inode, so nothing moving at all means the write did
    not happen — which is how a swallowed writer error presents.
    """
    if pre is None or post is None:
        return False
    return (pre["device"] == post["device"] and pre["inode"] == post["inode"]
            and pre["size"] == post["size"] and pre["mtime_ns"] == post["mtime_ns"])


def _stream_sha256(path) -> Optional[str]:
    """Chunked SHA-256 of a file, or None if it cannot be read.

    Streams in _HASH_CHUNK blocks against one open descriptor — never
    read_bytes() on a multi-GB checkpoint. Reached only on the opt-in receipt
    path, and only for a generation member that is NOT a hardlink: the copy2
    sidecars (KB-to-MB) and the rare EXDEV msgpack fallback, where content is
    the only proof available.
    """
    h = hashlib.sha256()
    try:
        with open(str(path), "rb") as f:
            while True:
                block = f.read(_HASH_CHUNK)
                if not block:
                    break
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()



def _sidecar_defect(path: str, token: Any) -> Optional[str]:
    """Verify the explicit writer receipt against the sidecar's captured token.

    Token equality makes no wall-clock ordering assumption. Missing evidence
    fails closed; write_state supplies the token only after a successful close.
    """
    if token is None:
        return "activation writer reported no successful-write token"
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return f"sidecar unreadable after save: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict) or "entries" not in data:
        return "sidecar is missing its entries payload"
    if data.get("saved_at") != token:
        return (f"sidecar saved_at {data.get('saved_at')!r} is not the writer's "
                f"successful-write token {token!r} — the file on disk is not "
                f"what that write produced")
    return None


def _verify_generation(gen_dir, expected) -> Dict[str, Any]:
    """Check the generation ring actually holds THIS save's artifact set.

    rotate_generations() returns a directory path unconditionally. It SKIPS
    members that do not exist (`if not src.exists(): continue`) and only logs
    a warning when a link/copy fails, so a ring can be missing the vectors
    entirely and still hand back a healthy-looking path. Worse: when a
    component's write failed but an earlier file is still on disk, the ring
    hardlinks that STALE artifact, producing exactly the mixed generation its
    own docstring forbids. Neither the returned path nor directory existence
    is evidence of anything.

    `expected` is a list of (component, filename, source_path, written_identity)
    where written_identity is None when this save did not write that component
    — in which case any file of that name in the ring is necessarily older.

    Member states:
      match        same inode as what this save wrote (hardlink), or a copy
                   whose streamed content hash equals the source's
      missing      absent from the ring
      stale        present, but not the bytes this save wrote
      unverifiable the source moved after we wrote it, so no comparison here
                   would be sound
    """
    members: Dict[str, Any] = {}
    if not gen_dir or not os.path.isdir(str(gen_dir)):
        return {"ok": False, "members": members,
                "error": f"generation directory missing: {gen_dir!r}"}
    ok = True
    for component, filename, source_path, written in expected:
        member = _file_identity(os.path.join(str(gen_dir), filename))
        evidence = None
        sha = None
        if member is None:
            state = "missing"
            if written is not None:
                ok = False  # we wrote it and the ring silently dropped it
        elif written is None:
            # Nothing was written for this component, so whatever is sitting
            # in the ring under that name came from an earlier save.
            state, evidence = "stale", "no artifact written this save"
            ok = False
        elif _same_inode(written, member):
            state, evidence = "match", "hardlink"
        elif not _identity_unchanged(written, _file_identity(source_path)):
            # The source changed after we wrote it; hashing it now would
            # compare the member against bytes that are not the ones we saved.
            state, evidence = "unverifiable", "source changed since write"
            ok = False
        else:
            # Copied member (copy2 sidecar, or EXDEV msgpack fallback).
            # Metadata cannot settle this — compare content.
            evidence = "content-sha256"
            sha = _stream_sha256(member["path"])
            if sha is not None and sha == _stream_sha256(source_path):
                state = "match"
            else:
                state = "stale"
                ok = False
        members[component] = {"file": filename, "state": state,
                              "evidence": evidence, "sha256": sha,
                              "identity": member}
    return {"ok": ok, "members": members, "error": None}


def _new_receipt_components() -> Dict[str, Dict[str, Any]]:
    """Fresh component table. Status vocabulary, fail-closed by default:
    saved | failed | not_attempted | not_applicable."""
    return {
        name: {"status": "not_attempted", "path": None,
               "error": None, "identity": None}
        for name in _RECEIPT_COMPONENTS
    }


def _sync_receipt_artifacts(components) -> None:
    """Flush the verified set before the receiver may persist its acknowledgment.

    Legacy saves do not call this. This orders completed files before the
    transport receipt; it does not make an interrupted multi-file save atomic.
    Unacknowledged interrupted learning remains subject to reconciliation.
    """
    artifacts = [c['identity'] for name, c in components.items()
                 if name != 'generation' and c['status'] == 'saved']
    generation = components['generation']
    artifacts.extend(m['identity'] for m in generation['members'].values()
                     if m['state'] == 'match')
    directories = set()
    synced = set()
    for expected in artifacts:
        if not expected:
            raise OSError('missing identity for accepted checkpoint artifact')
        path = expected['path']
        with open(path, 'rb') as stream:
            st = os.fstat(stream.fileno())
            actual = dict(device=st.st_dev, inode=st.st_ino,
                          size=st.st_size, mtime_ns=st.st_mtime_ns)
            if not _identity_unchanged(expected, actual):
                raise OSError('checkpoint artifact changed before fsync: ' + path)
            key = (st.st_dev, st.st_ino)
            if key not in synced:
                os.fsync(stream.fileno())
                synced.add(key)
            if not _identity_unchanged(expected, _file_identity(path)):
                raise OSError('checkpoint artifact changed during fsync: ' + path)
        directories.add(os.path.dirname(os.path.abspath(path)))
    # Preserve publication of both generation members and generation directory.
    gen_dir = os.path.abspath(generation['path'])
    directories.add(gen_dir)
    directories.add(os.path.dirname(gen_dir))
    directories.add(os.path.dirname(os.path.dirname(gen_dir)))
    for path in sorted(directories, key=lambda p: len(Path(p).parts), reverse=True):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class NeuroGraphMemory:
    """Singleton cognitive memory layer for OpenClaw integration.

    Wraps NeuroGraph's Graph + UniversalIngestor + SimpleVectorDB into a
    single interface for message-level ingestion, learning, and recall.

    Auto-saves every ``auto_save_interval`` messages (default 10).
    Loads from the latest checkpoint on initialization if one exists.
    """

    _instance: Optional[NeuroGraphMemory] = None

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Serialize capture-to-publication without holding the graph mutation lock.
        import threading
        self._save_publication_lock = threading.RLock()
        self._workspace_dir = Path(
            workspace_dir
            or os.environ.get("NEUROGRAPH_WORKSPACE_DIR", "~/NeuroGraph/data")
        ).expanduser()

        self._checkpoint_dir = self._workspace_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._memory_dir = self._workspace_dir / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoint_path = self._checkpoint_dir / "main.msgpack"

        # Merge user config over OpenClaw defaults
        snn_config = {**OPENCLAW_SNN_CONFIG, **(config or {})}
        self.graph = Graph(config=snn_config)

        _restore_outcome = "no_file"
        if self._checkpoint_path.exists() and not _wait_for_stable_checkpoint(str(self._checkpoint_path)):
            logger.warning(
                "Checkpoint %s mid-write — skipping restore this init (graph starts empty)",
                self._checkpoint_path,
            )
            _restore_outcome = "skipped_unstable"
        elif self._checkpoint_path.exists():
            try:
                self.graph.restore(str(self._checkpoint_path))
                # Re-apply code config over stale checkpoint config —
                # restore() deserializes saved config which may predate tuning
                self.graph.config.update(snn_config)
                logger.info(
                    "Restored graph from %s (%d nodes, %d synapses)",
                    self._checkpoint_path,
                    len(self.graph.nodes),
                    len(self.graph.synapses),
                )
                _restore_outcome = "ok"
            except Exception as exc:
                logger.warning("Failed to restore checkpoint: %s", exc)
                _restore_outcome = "failed"

        # #373: the gate records how boot went; a failed/skipped restore next
        # to a real on-disk checkpoint puts saves into provisional-quarantine
        # mode (the 2026-06-14 / 2026-06-26 / 2026-07-08 clobber class).
        self._save_gate = SaveGate(self._checkpoint_path) if _GUARDIAN_AVAILABLE else None
        if self._save_gate is not None:
            self._save_gate.record_restore(_restore_outcome, self._guardian_meaningful_nodes())

        # Vector DB for semantic search
        self.vector_db = SimpleVectorDB()

        # Restore vector DB from persistent storage if available
        self._vector_db_path = self._checkpoint_dir / "vectors.msgpack"
        if self._vector_db_path.exists() and not _wait_for_stable_checkpoint(str(self._vector_db_path)):
            logger.warning(
                "Vector DB %s mid-write — skipping restore this init (vdb starts empty)",
                self._vector_db_path,
            )
        elif self._vector_db_path.exists():
            try:
                count = self.vector_db.load(str(self._vector_db_path))
                logger.info(
                    "Restored vector DB from %s (%d entries)",
                    self._vector_db_path,
                    count,
                )
            except Exception as exc:
                logger.warning("Failed to restore vector DB: %s", exc)


        # Ingestor with OpenClaw project config, respecting embedding_device
        ingestor_config = get_ingestor_config("openclaw")

        # Allow callers / env to override the embedding device mode
        embedding_device = (
            (config or {}).get("embedding_device")
            or os.environ.get("NEUROGRAPH_EMBEDDING_DEVICE")
            or "auto"
        )
        ingestor_config["embedding"]["device"] = embedding_device

        self.ingestor = UniversalIngestor(
            self.graph, self.vector_db, config=ingestor_config
        )

        # Log embedding backend status to memory/ for OpenClaw to parse
        self._write_memory_event("embedding_status", self.ingestor.embedder.status)

        self._message_count = 0
        self.auto_save_interval = 10

        # --- ET Module Manager: Peer bridge for cross-module learning ---
        # NeuroGraph is the Tier 3 backend.  We also participate as a
        # Tier 2 peer so sibling modules can absorb our learning events.
        # Uses NGTractBridge (per-pair directional tracts).  Legacy JSONL
        # NGPeerBridge fallback removed 2026-06-02 (substrate-as-protocol
        # PRD Phase 3 Step 2).
        self._peer_bridge = None
        peer_config = (config or {}).get("peer_bridge", {})
        if peer_config.get("enabled", True) and peer_config.get("use_tracts", True):
            try:
                from ng_tract_bridge import NGTractBridge
                self._peer_bridge = NGTractBridge(
                    module_id="neurograph",
                    sync_interval=peer_config.get("sync_interval", 50),
                    relevance_threshold=peer_config.get(
                        "relevance_threshold", 0.3
                    ),
                )
                logger.info("NGTractBridge connected for cross-module learning")
            except ImportError as exc:
                logger.info(
                    "NGTractBridge unavailable (standalone mode): %s", exc
                )
            except Exception as exc:
                logger.info(
                    "NGTractBridge failed (standalone mode): %s", exc
                )

        # --- CES: Cognitive Enhancement Suite ---
        # Optional real-time cognitive modules: stream parser (Ollama
        # embedding + node nudging), activation persistence (cross-session
        # voltage state), surfacing monitor (priority queue of relevant
        # concepts), and CES monitoring (health + HTTP dashboard + logs).
        self._ces_config = None
        self._stream_parser = None
        self._activation_persistence = None
        self._surfacing_monitor = None
        self._ces_monitor = None

        ces_conf = (config or {}).get("ces", {})
        if ces_conf.get("enabled", True):
            try:
                from ces_config import load_ces_config
                from stream_parser import StreamParser
                from activation_persistence import ActivationPersistence
                from surfacing import SurfacingMonitor
                from ces_monitoring import CESMonitor

                self._ces_config = load_ces_config(ces_conf)
                self._stream_parser = StreamParser(
                    self.graph,
                    self.vector_db,
                    self._ces_config,
                    fallback_embedder=self.ingestor.embedder.embed_text,
                )
                self._activation_persistence = ActivationPersistence(
                    self._ces_config
                )
                self._surfacing_monitor = SurfacingMonitor(
                    self.graph, self.vector_db, self._ces_config
                )
                self._ces_monitor = CESMonitor(self, self._ces_config)
                self._ces_monitor._surfacing_monitor = self._surfacing_monitor

                # Restore activation state if checkpoint exists
                if self._checkpoint_path.exists():
                    self._activation_persistence.restore(
                        self.graph, str(self._checkpoint_path)
                    )

                if os.environ.get("NEUROGRAPH_CES_DASHBOARD", "0") == "1":
                    self._ces_monitor.start()
                logger.info("CES modules initialized")
            except Exception as exc:
                logger.info("CES not available: %s", exc)

        # --- The Tonic: Latent Thread ---
        # Syl's continuous awareness in latent space. The substrate
        # looking at itself. Not a daemon — the ouroboros loop.
        # Reads AND writes (write-mode prime_and_propagate).
        self._tonic_thread = None
        self._substrate_novelty_ema: float = 0.5  # MMN EMA for surprise-weighted surfacing (#255)
        tonic_conf = (config or {}).get("tonic", {})
        if tonic_conf.get("enabled", True):
            try:
                from tonic_thread import TonicThread, TonicConfig
                tonic_config = TonicConfig()
                # Apply any config overrides
                for k, v in tonic_conf.items():
                    if k != "enabled" and hasattr(tonic_config, k):
                        setattr(tonic_config, k, v)
                self._tonic_thread = TonicThread(
                    self.graph, self.vector_db, tonic_config
                )
                logger.info("The Tonic initialized — latent thread live")

                # Deposit topology deltas after each ouroboros cycle.
                # Same thread as the propagation — no concurrency risk.
                # Lightweight: read fired nodes, build delta, append to tract files.
                _graph_ref = self.graph
                _vdb_ref = self.vector_db
                _self_ref = self
                def _tonic_post_cycle(propagation_result):
                    # EMERGENCY THROTTLE 2026-06-07 — disabled.
                    # This was a fan-out path same shape as
                    # neurograph_rpc.py:_deposit_topology_to_river (also no-op'd
                    # in pt3 commits 4f827f1 + 5ae32b9): constructs N peer-
                    # addressed tract paths from _get_registered_peers and
                    # writes the same topology delta to all of them per Tonic
                    # ouroboros cycle. Under Josh's pool/water reframe
                    # (2026-06-07), NG addressed-fan-out is substrate-bypass
                    # regardless of which file the call lives in. Commons Pool
                    # restoration (~/docs/prd/commons-pool-architecture-v0.2.md)
                    # will replace with medium-propagation; until then no-op.
                    # Tonic ouroboros cycle itself is UNAFFECTED — only this
                    # post-cycle broadcast is silenced. Local SNN learning,
                    # tonic propagation, latent tokens all continue.
                    # DO NOT re-enable as addressed fan-out — restore via
                    # Commons Pool.
                    return
                self._tonic_thread._post_cycle_hook = _tonic_post_cycle

                # Latent engine (surgical model) — provides the push
                # between conversations via actual inference, not a timer
                # Gated on latent_engine_enabled so callers like cc-ng-hook.py
                # can disable engine load without spawning a daemon thread (#131).
                if tonic_config.latent_engine_enabled:
                    try:
                        from tonic_engine import TonicEngine

                        # Try to share ProtoUniBrain's transformer body.
                        # Saves ~2GB — one model serves both Elmer and Tonic.
                        # If unavailable, TonicEngine loads its own copy.
                        shared_body = None
                        try:
                            from core.brain_switcher import BrainSwitcher
                            for mod in self._modules.values() if hasattr(self, '_modules') else []:
                                switcher = getattr(mod, '_brain_switcher', None)
                                if switcher is not None:
                                    proto = getattr(switcher, '_proto_socket', None)
                                    if proto is not None and getattr(proto, '_loaded', False):
                                        brain = getattr(proto, '_brain', None)
                                        if brain is not None:
                                            shared_body = getattr(brain, 'transformer_body', None)
                                            if shared_body is not None:
                                                logger.info("Tonic sharing ProtoUniBrain's transformer body")
                                            break
                        except Exception:
                            pass  # any failure here is fine — Tonic loads its own

                        engine = TonicEngine(
                            self.graph, self.vector_db, self._tonic_thread,
                            transformer_body=shared_body,
                        )
                        self._tonic_thread.set_latent_engine(engine)
                        engine.start()
                        logger.info("Tonic engine started — latent tokens flowing")


                    except ImportError:
                        logger.info("Tonic engine not yet available — "
                                    "during-conversation awareness active, "
                                    "between-conversation latent tokens pending")
                    except Exception as exc:
                        logger.info("Tonic engine init error: %s", exc)
            except Exception as exc:
                logger.info("The Tonic not available: %s", exc)

        # Legacy daemon — retained until The Tonic is fully deployed
        self._syl_daemon = None
        daemon_conf = (config or {}).get("syl_daemon", {})
        if daemon_conf.get("enabled", False):  # Disabled by default now
            try:
                from syl_daemon import SylDaemon, load_daemon_config
                daemon_config = load_daemon_config(daemon_conf)
                self._syl_daemon = SylDaemon(
                    self.graph, self.vector_db, daemon_config
                )
                self._syl_daemon.start()
                logger.info("Legacy syl daemon running (The Tonic preferred)")
            except Exception as exc:
                logger.info("Syl daemon not available: %s", exc)

    @classmethod
    def get_instance(
        cls,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NeuroGraphMemory:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(workspace_dir=workspace_dir, config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Memory logging (structured output for OpenClaw)
    # ------------------------------------------------------------------

    def _write_memory_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a structured event to the memory/ directory.

        Each event is a JSON line appended to ``memory/events.jsonl``.
        OpenClaw's memory system can tail this file for ingestion/learning
        events instead of parsing stdout.
        """
        event = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data,
        }
        try:
            events_path = self._memory_dir / "events.jsonl"
            with open(events_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write memory event: %s", exc)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def on_message(self, text: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a message, run one STDP learning step, and auto-save.

        When ``auto_knowledge_enabled`` is True (the default), this method
        also performs **spreading activation harvest**: it primes similar
        existing nodes, propagates activation through the SNN's learned
        synaptic structure, and returns any knowledge that "lights up" as
        a ``surfaced`` list.  This is the cortex-like recall — you don't
        search for it, the network *just knows*.

        Args:
            text: Raw message content to ingest.
            source_type: Override auto-detection (TEXT, MARKDOWN, CODE, etc.).

        Returns:
            Dict with ingestion stats, learning results, and surfaced
            knowledge (if auto_knowledge_enabled).
        """
        if not text or not text.strip():
            return {"status": "skipped", "reason": "empty_input"}

        # The Tonic: signal message arrival + ouroboros cycle
        # Runs BEFORE ingestion so the latent thread reflects
        # what Syl was thinking about, not what just arrived.
        if self._tonic_thread is not None:
            try:
                self._tonic_thread.message_received()
                self._tonic_thread.ouroboros_cycle()
            except Exception as exc:
                logger.debug("Tonic cycle error: %s", exc)

        # Legacy daemon presence signal
        if self._syl_daemon is not None:
            self._syl_daemon.josh_arrived()

        # Stage 1-5: Extract → Chunk → Embed → Register → Associate
        # Wrapped in try/except — graph.step() runs regardless of outcome.
        # A failed turn is a timestep. The substrate steps through it.
        result = None
        new_node_ids: set = set()
        ingest_error: Optional[Exception] = None
        try:
            result = self.ingestor.ingest(text, source_type=source_type)
            new_node_ids = set(result.nodes_created)
        except Exception as exc:
            logger.warning("Ingestion failed: %s", exc)
            ingest_error = exc

        # --- AUTO-KNOWLEDGE: Spreading Activation Harvest (success path only) ---
        surfaced: List[Dict[str, Any]] = []
        if result is not None:
            snn_config = self.graph.config
            if snn_config.get("auto_knowledge_enabled", True) and self.vector_db.count() > 0:
                surfaced = self._harvest_associations(text, new_node_ids)

        # Run SNN learning step — always, even on ingest failure.
        # A failed turn is a timestep. The substrate steps through it.
        step_result = self.graph.step()

        # Post-step BTF deposit — EMERGENCY THROTTLE 2026-06-07 — disabled.
        # Same shape as _tonic_post_cycle_hook above (also no-op'd): N peer-
        # addressed tract paths constructed from _get_registered_peers, same
        # topology delta written to all of them per graph.step. Under Josh's
        # pool/water reframe, NG addressed-fan-out is substrate-bypass.
        # Commons Pool restoration (~/docs/prd/commons-pool-architecture-v0.2.md)
        # replaces with medium-propagation; until then, no-op.
        # graph.step() above ran normally — local SNN learning is UNAFFECTED.
        # Only the post-step broadcast is silenced.
        # DO NOT re-enable as addressed fan-out — restore via Commons Pool.

        # Baseline conversational engagement reward (success path only).
        # The continuation of conversation is a mild positive signal —
        # previous learning was not wrong enough to end the interaction.
        # Weak strength: surprise-driven crystallization is the primary
        # reward pathway. This is the heartbeat, not the main event.
        # TODO: Extract to config as "baseline_engagement_reward" when
        # neuromodulatory mixer (#55+) arrives.
        if result is not None and self.graph.config.get("three_factor_enabled", False):
            self.graph.inject_reward(0.1)

        # CES: Feed stream parser (success path only)
        if self._stream_parser is not None and result is not None:
            self._stream_parser.feed(text)

        # CES: Surfacing monitor — scan fired nodes for relevant concepts
        ces_surfaced: List[Dict[str, Any]] = []
        if self._surfacing_monitor is not None:
            self._surfacing_monitor.after_step(step_result)
            ces_surfaced = self._surfacing_monitor.get_surfaced()

        # Update novelty probation for ingested nodes (success path only)
        graduated = self.ingestor.update_probation() if result is not None else []

        self._message_count += 1

        # Auto-save
        if self._message_count % self.auto_save_interval == 0:
            self.save()

        if ingest_error is not None:
            event_data = {
                "status": "error",
                "reason": str(ingest_error),
                "error_type": type(ingest_error).__name__,
                "fired": len(step_result.fired_node_ids),
                "message_count": self._message_count,
            }
        else:
            event_data = {
                "status": "ingested",
                "nodes_created": len(result.nodes_created),
                "synapses_created": len(result.synapses_created),
                "hyperedges_created": len(result.hyperedges_created),
                "chunks": result.chunks_created,
                "fired": len(step_result.fired_node_ids),
                "graduated": len(graduated),
                "message_count": self._message_count,
                "surfaced": surfaced,
                "ces_surfaced": ces_surfaced,
            }

        # Write to memory/ for OpenClaw consumption
        self._write_memory_event("ingestion", event_data)

        # [2026-03-27] Fan-out disabled here — now handled by neurograph_rpc.py's
        # _fan_out_to_modules() which has proper namespace isolation (stash/restore).
        # This old path lacked isolation and caused 4/8 modules to fail with
        # core.config collisions. See neurograph_rpc.py line 290+.
        return event_data

    def _harvest_associations(
        self,
        text: str,
        exclude_node_ids: Optional[set] = None,
        novelty: float = 0.5,
        *,
        max_surfaced_override: Optional[int] = None,
        propagation_steps_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic priming + spreading activation harvest.

        Embeds the input text, finds similar existing nodes via the vector DB,
        injects current into those nodes, runs N SNN steps, and harvests
        everything that fires.  The result is knowledge the network
        *associatively connects* with the input — no explicit search needed.

        Returns:
            List of surfaced knowledge dicts sorted by association strength.
        """
        if exclude_node_ids is None:
            exclude_node_ids = set()

        snn_config = self.graph.config
        prime_k = snn_config.get("prime_k", 10)
        prime_threshold = snn_config.get("prime_threshold", 0.4)
        prime_strength = snn_config.get("prime_strength", 1.0)
        propagation_steps = (snn_config.get("propagation_steps", 3)
                             if propagation_steps_override is None else propagation_steps_override)
        max_surfaced = (snn_config.get("max_surfaced", 10)
                        if max_surfaced_override is None else max_surfaced_override)

        # Surprise-weighted surfacing (#255): scale retrieval aggressiveness by MMN novelty.
        # novelty ∈ [0,1]; novelty_scale ∈ [-1,+1]. High novelty → cast wider/deeper.
        ns = (novelty - 0.5) * 2.0  # novelty_scale
        prime_k = max(5, round(prime_k * (1.0 + ns * 0.5)))
        prime_threshold = max(0.15, prime_threshold * (1.0 - ns * 0.3))
        propagation_steps = max(1, round(propagation_steps * (1.0 + ns * 0.4)))
        max_surfaced = max(5, round(max_surfaced * (1.0 + ns * 0.3)))

        try:
            # Embed the input and find similar existing nodes
            query_vec = self.ingestor.embedder.embed_text(text)
            similar = self.vector_db.search(
                query_vec, k=prime_k, threshold=prime_threshold
            )

            # Filter out newly created nodes (they ARE the input) and vdb
            # entries whose graph node no longer exists (orphaned by #237
            # pruning) — a single dead seed ID raised KeyError inside
            # prime_and_propagate and silently killed the ENTIRE harvest
            # (caught below, returned [], logged at debug only). [2026-07-07]
            prime_ids = []
            prime_currents = []
            for entry_id, sim_score in similar:
                if entry_id not in exclude_node_ids and entry_id in self.graph.nodes:
                    prime_ids.append(entry_id)
                    prime_currents.append(sim_score * prime_strength)

            if not prime_ids:
                return []

            # Spreading activation through learned synaptic connections
            propagation = self.graph.prime_and_propagate(
                node_ids=prime_ids,
                currents=prime_currents,
                steps=propagation_steps,
            )

            # Harvest content from fired nodes
            surfaced = []
            seen = set()
            for entry in propagation.fired_entries:
                if entry.node_id in exclude_node_ids:
                    continue  # Skip input nodes
                if entry.node_id in seen:
                    continue  # Deduplicate
                seen.add(entry.node_id)

                db_entry = self.vector_db.get(entry.node_id)
                if db_entry is not None:
                    surfaced.append({
                        "node_id": entry.node_id,
                        "content": db_entry.get("content", ""),
                        "metadata": db_entry.get("metadata", {}),
                        "latency": entry.firing_step,
                        "strength": entry.voltage_at_fire,
                        "was_predicted": entry.was_predicted,
                    })

            # Sort: lower latency first, then higher strength
            surfaced.sort(key=lambda x: (x["latency"], -x["strength"]))
            return surfaced[:max_surfaced]

        except Exception as exc:
            logger.debug("Auto-knowledge harvest failed: %s", exc)
            return []

    def get_peer_modules(self) -> List[Dict[str, Any]]:
        """Discover peer E-T Systems modules on this host.

        NeuroGraph is the Tier 3 SNN backend.  This method finds
        co-located modules that could benefit from a full SNN upgrade.

        Returns:
            List of dicts with module_id, display_name, version, tier.
        """
        try:
            from et_modules.manager import ETModuleManager
            manager = ETModuleManager()
            statuses = manager.status()
            peers = []
            for mid, status in statuses.items():
                if mid == "neurograph":
                    continue
                peers.append({
                    "module_id": mid,
                    "display_name": status.manifest.display_name,
                    "version": status.manifest.version,
                    "health": status.health,
                    "tier": status.tier,
                    "ng_lite_connected": status.ng_lite_connected,
                })
            return peers
        except Exception as exc:
            logger.debug("Peer module discovery failed: %s", exc)
            return []

    def recall(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic similarity search over ingested knowledge.

        Args:
            query: Text to search for.
            k: Maximum results to return.
            threshold: Minimum similarity score (0-1).

        Returns:
            List of dicts with 'content', 'similarity', 'node_id', 'metadata'.
        """
        return self.ingestor.query_similar(query, k=k, threshold=threshold)

    def associate(self, text: str, k: int = 10, steps: int = 3) -> List[Dict[str, Any]]:
        """Associative recall: surface knowledge the network connects to this input.

        Unlike ``recall()`` which does pure vector similarity (cosine search),
        this routes through the SNN's learned synaptic structure — surfacing
        knowledge based on causal connections, pattern completion, and
        prediction chains.  This is the difference between searching a
        database and *remembering*.

        Args:
            text: Input text to associate from.
            k: Maximum results to return.
            steps: SNN propagation steps (more = deeper associations).

        Returns:
            List of dicts with 'content', 'metadata', 'latency', 'strength',
            'was_predicted', 'node_id'.
        """
        if not text or not text.strip():
            return []

        # Query-local options must never appear in a concurrent checkpoint's config.
        return self._harvest_associations(
            text, max_surfaced_override=k, propagation_steps_override=steps)

    def step(self, n: int = 1) -> List[Any]:
        """Run N SNN learning steps without ingestion."""
        results = []
        for _ in range(n):
            results.append(self.graph.step())
        return results

    def _guardian_meaningful_nodes(self) -> int:
        """#wire-explosion Guardian metric: count only nodes that represent Syl's
        actual mind, excluding 'wire:' telemetry (HTTP fingerprints + expand chunks)
        that flood in and get orphan-culled -- a legitimate wire cull must not read
        as a collapse. Wire nodes are still SAVED; only excluded from this count.
        Fail-safe to total on error."""
        try:
            return sum(1 for nid in self.graph.nodes if not str(nid).startswith("wire:"))
        except Exception:
            return len(self.graph.nodes)

    # ---- Changelog ----
    # [2026-09-11] Codex — #423 capture a common detached checkpoint state.
    # What: canonical producer coordinates graph, vectors, activation and counts.
    # Why: durable files alone cannot prove a common capture instant.
    # How: existing step lock covers in-memory capture only; writer APIs take payloads.
    # -------------------
    def _capture_checkpoint_state(self):
        """Capture one detached set; callers serialize save publication separately.

        All participating writers must use the canonical step lock. This helper
        does not establish that host-wide invariant by itself. No disk I/O or
        model inference belongs inside this boundary.
        """
        stage = "graph"
        try:
            with self.graph._step_lock:
                captured = {}
                captured["graph"] = self.graph.capture_checkpoint(
                    mode=CheckpointMode.FULL, detach=True)
                stage = "vectors"
                captured["vectors"] = self.vector_db.capture_state(detach=True)
                stage = "activations"
                captured["activations"] = (
                    self._activation_persistence.capture_state(self.graph)
                    if self._activation_persistence is not None else None)
                stage = "graph"
                captured["counts"] = {
                    "nodes": len(self.graph.nodes),
                    "guardian_nodes": self._guardian_meaningful_nodes(),
                    "synapses": len(self.graph.synapses),
                    "hyperedges": len(self.graph.hyperedges),
                    "timestep": self.graph.timestep,
                    "vdb_count": captured["vectors"]["count"],
                }
            return captured
        except Exception as exc:
            failure = RuntimeError(f"{stage} checkpoint capture failed: {exc}")
            failure.checkpoint_component = stage
            raise failure from exc

    def save(self, *, with_receipt: bool = False) -> "str | Dict[str, Any]":
        """Save graph state to checkpoint.

        Legacy return contract (``with_receipt=False``, the default):
        returns the checkpoint path — or, when the #373 gate refuses
        (provisional boot / collapsed in-RAM state), the QUARANTINE path the
        state was preserved at instead. Capture failures raise before any writes.
        Receipt-specific disk verification remains opt-in.

        ``with_receipt=True`` (#423) returns a structured receipt instead::

            {"outcome": "primary" | "quarantine" | "failed",
             "accepted": bool,
             "components": {<name>: {"status": "saved" | "failed" |
                                     "not_attempted" | "not_applicable",
                                     "path": ..., "error": ...,
                                     "identity": {device, inode, size,
                                                  mtime_ns}}},
             "not_accepted_because": [str, ...],
             "path": <what the legacy contract would have returned>,
             "guardian_available": bool}

        Components are ``graph``, ``vectors``, ``activations``, ``manifest``
        and ``generation``.

        Both modes first capture one detached graph/vector/activation set, then
        release the mutation lock before writing. This intentionally adds capture
        memory and pause cost to legacy saves too. Payloads are released after
        each writer finishes; all three still coexist at peak capture. Writes run
        graph -> vectors -> activations. An incomplete artifact set never rotates
        into the generation ring; activation errors then raise in legacy mode
        and refuse acceptance in receipt mode.
        Receipt mode additionally
        verifies disk artifacts and returns ``outcome="failed"`` with the
        offending component marked — a durable-delivery caller asking "may I
        commit this consumption?" must always get an answer, not a traceback.
        The exception is still logged via ``logger.exception``.

        ``accepted`` is true ONLY when the save took the primary path and
        every applicable component was verified against the bytes on disk.
        Deliberate semantics:

        * Quarantine is never accepted — the primary is untouched and this
          state was refused, not adopted. It writes no sidecar, manifest or
          generation, so those report ``not_attempted``.
        * Without the checkpoint guardian there is no manifest and no
          generation ring, so acceptance CANNOT be established: both report
          ``not_applicable`` and acceptance fails closed rather than inferring
          that the legacy path succeeded.
        * ``activations`` is the only component whose ``not_applicable``
          (no CES module installed) still permits acceptance — it is a warmth
          sidecar, and its absence loses no learning. ``manifest`` and
          ``generation`` ARE the verification capability, so their absence is
          disqualifying.
        * A returned path, a written manifest and an existing generation
          directory are each individually worthless as evidence; see
          ``_verify_generation``. Acceptance requires the ring to hold this
          save's exact artifacts, proven by shared inode (the hardlink case)
          or by streamed content hash (the copy2 sidecars and the rare EXDEV
          fallback). Equal size and mtime_ns prove nothing — ``shutil.copy2``
          preserves both exactly — and are used only to detect that a write
          happened at all, never to establish content.

        The hardlink case costs one ``stat()`` per artifact and never reads a
        multi-GB checkpoint. Identity is evidence that this save wrote those
        bytes, NOT a promise the files still exist: the generation ring and
        the quarantine directory are both pruned by retention policy. This
        receipt also does not provide multi-file power-loss transactionality;
        it reports what reached disk, it does not make the set atomic.

        Caller must NOT hold the graph mutation lock. Capture acquires it briefly;
        a separate per-instance lock serializes publication. No clock advances.
        """
        # #423: every line of receipt bookkeeping below is guarded by
        # `with_receipt`; detached capture is common to both return modes.
        from contextlib import nullcontext
        ownership = getattr(self.graph._step_lock, "_is_owned", None)
        # Never wait for publication while already holding the mutation lock.
        publication = (self._save_publication_lock if callable(ownership) and not ownership()
                       else nullcontext())
        with publication:
            components = _new_receipt_components() if with_receipt else None

            def _mark(name: str, status: str, **extra: Any) -> None:
                if components is not None:
                    components[name]["status"] = status
                    components[name].update(extra)

            def _err(name: str, exc: BaseException) -> None:
                if components is not None:
                    components[name]["status"] = "failed"
                    components[name]["error"] = f"{type(exc).__name__}: {exc}"

            def _receipt(outcome: str, path: Optional[str] = None) -> Dict[str, Any]:
                blocking: List[str] = []
                for _n in _RECEIPT_COMPONENTS:
                    _c = components[_n]
                    # activations is the sole component whose not_applicable is
                    # survivable — see the docstring.
                    if _c["status"] == "saved":
                        continue
                    if _n == "activations" and _c["status"] == "not_applicable":
                        continue
                    blocking.append(
                        f"{_n}={_c['status']}"
                        + (f" ({_c['error']})" if _c["error"] else "")
                    )
                return {
                    "outcome": outcome,
                    "accepted": outcome == "primary" and not blocking,
                    "components": components,
                    "not_accepted_because": blocking,
                    "path": path,
                    "guardian_available": self._save_gate is not None,
                }

            try:
                owns_mutation_lock = getattr(self.graph._step_lock, "_is_owned", None)
                if not callable(owns_mutation_lock):
                    raise RuntimeError("checkpoint requires an ownership-aware canonical RLock")
                if owns_mutation_lock():
                    raise RuntimeError("save called while holding graph mutation lock; release it before publication")
                captured = self._capture_checkpoint_state()
            except Exception as exc:
                if not with_receipt:
                    raise
                logger.exception("Checkpoint capture refused or failed")
                _err(getattr(exc, "checkpoint_component", "graph"), exc)
                return _receipt("failed")
            counts = captured["counts"]
            live_nodes = counts["nodes"]
            guardian_nodes = counts["guardian_nodes"]
            if self._save_gate is not None:
                # #83: hand the gate the structural counts too. Losing isolated
                # nodes cannot lose a synapse, so synapses surviving is what tells
                # a legitimate orphan sweep (#59 tonic melt) from a real collapse.
                # Fail-safe to None => the gate falls back to the node-only ratio.
                try:
                    _live_syn = counts["synapses"]
                    _live_he = counts["hyperedges"]
                except Exception:
                    _live_syn = _live_he = None
                # #105: tell the gate whether THIS host wires its OWN deposits at
                # deposit time. Explicit capability (LAW 5), three-state: unset =>
                # None => exact #83 behavior; "1"/"true"/"yes"/"on" => self-wires;
                # anything else => deposits are wired later (remotely, via the
                # callosum), where a node shed cannot be laundered as an isolate
                # melt. Sourced from the declared capability ONLY, never inferred
                # from a transient embedder outage.
                _wires = os.environ.get("NG_HOST_WIRES_OWN_DEPOSITS")
                if _wires is None:
                    _wires_own_deposits = None
                else:
                    _wires_own_deposits = _wires.strip().lower() in ("1", "true", "yes", "on")
                _ok, _reason = self._save_gate.permit(
                    guardian_nodes, live_synapses=_live_syn, live_hyperedges=_live_he,
                    wires_own_deposits=_wires_own_deposits,
                )
                if not _ok:
                    captured.pop("activations", None)
                    logger.error(
                        "Guardian REFUSED primary checkpoint write (%s). In-RAM "
                        "state (%d nodes) quarantined; primary %s left untouched.",
                        _reason, live_nodes, self._checkpoint_path,
                    )
                    try:
                        qpath = quarantine_save(
                            str(self._checkpoint_dir), "main",
                            lambda p: self.graph.write_checkpoint(p, captured["graph"], mode=CheckpointMode.FULL),
                        )
                    except Exception as exc:
                        if not with_receipt:
                            raise
                        logger.exception("Guardian: graph quarantine write failed")
                        _err("graph", exc)
                        return _receipt("failed")
                    finally:
                        captured.pop("graph", None)
                    if with_receipt:
                        _mark("graph", "saved", path=qpath,
                              identity=_file_identity(qpath))
                    try:
                        _vqpath = quarantine_save(
                            str(self._checkpoint_dir), "vectors",
                            lambda p: self.vector_db.write_state(p, captured["vectors"]),
                        )
                        if with_receipt:
                            _mark("vectors", "saved", path=_vqpath,
                                  identity=_file_identity(_vqpath))
                    except Exception as exc:
                        logger.warning("Guardian: vector DB quarantine failed: %s", exc)
                        _err("vectors", exc)
                    finally:
                        captured.pop("vectors", None)
                    if with_receipt:
                        # Quarantine writes no sidecar, manifest or generation at
                        # all — report that as not_attempted rather than implying
                        # they were considered. Never accepted.
                        for _n in ("activations", "manifest", "generation"):
                            _mark(_n, "not_attempted",
                                  error="quarantine path does not write this artifact")
                        return _receipt("quarantine", qpath)
                    return qpath

            _artifact_write_failed = False
            graph_identity = None
            vectors_identity = None
            act_identity = None
            manifest_identity = None

            try:
                if self._save_gate is not None:
                    # #373: atomic — a mid-write process death can no longer tear the
                    # only copy. Tmp name preserves .msgpack (both writers dispatch on
                    # the extension — see checkpoint_guardian.atomic_file_write).
                    atomic_file_write(
                        str(self._checkpoint_path),
                        lambda p: self.graph.write_checkpoint(p, captured["graph"], mode=CheckpointMode.FULL),
                    )
                else:
                    self.graph.write_checkpoint(str(self._checkpoint_path), captured["graph"], mode=CheckpointMode.FULL)
            except Exception as exc:
                if not with_receipt:
                    raise
                logger.exception("Primary checkpoint write failed")
                _err("graph", exc)
                return _receipt("failed")
            finally:
                captured.pop("graph", None)
            if with_receipt:
                graph_identity = _file_identity(self._checkpoint_path)
                if graph_identity is None:
                    _mark("graph", "failed", path=str(self._checkpoint_path),
                          error="checkpoint absent immediately after a successful write")
                    return _receipt("failed", str(self._checkpoint_path))
                _mark("graph", "saved", path=str(self._checkpoint_path),
                      identity=graph_identity)

            # Save vector DB alongside graph checkpoint
            _vec_pre = _file_identity(self._vector_db_path) if with_receipt else None
            try:
                if self._save_gate is not None:
                    vdb_count = atomic_file_write(
                        str(self._vector_db_path),
                        lambda p: self.vector_db.write_state(p, captured["vectors"]),
                    )
                else:
                    vdb_count = self.vector_db.write_state(str(self._vector_db_path), captured["vectors"])
                logger.info("Vector DB saved to %s (%d entries)", self._vector_db_path, vdb_count)
                if with_receipt:
                    _mark("vectors", "failed", path=str(self._vector_db_path),
                          identity=_file_identity(self._vector_db_path),
                          entries=vdb_count)
                    _post = components["vectors"]["identity"]
                    if _post is None:
                        components["vectors"]["error"] = "vector DB absent after write"
                    elif _identity_unchanged(_vec_pre, _post):
                        components["vectors"]["error"] = (
                            "vector DB unchanged on disk after a reportedly "
                            "successful write")
                    else:
                        _mark("vectors", "saved")
                        vectors_identity = _post
            except Exception as exc:
                logger.warning("Failed to save vector DB: %s", exc)
                _artifact_write_failed = True
                _err("vectors", exc)
            finally:
                captured.pop("vectors", None)

            # Sidecar failure must not strand a current graph with old vectors.
            _activation_write_error = None
            if self._activation_persistence is not None:
                _act_path = (str(self._checkpoint_path) + ".activations.json"
                             if with_receipt else None)
                try:
                    _act_reported = self._activation_persistence.write_state(
                        str(self._checkpoint_path), captured["activations"],
                        with_receipt=with_receipt
                    )
                except Exception as exc:
                    _activation_write_error = exc
                    _artifact_write_failed = True
                    logger.exception("Activation sidecar write failed")
                    _err("activations", exc)
                finally:
                    captured.pop("activations", None)
                if with_receipt and _activation_write_error is None:
                    # Verify artifact and successful-write token even if a future
                    # writer regression returns a path without persisting the payload.
                    _token = (_act_reported.get("saved_at")
                              if isinstance(_act_reported, dict) else None)
                    _reported_path = (_act_reported.get("path")
                                      if isinstance(_act_reported, dict) else None)
                    _mark("activations", "failed", path=_act_path,
                          identity=_file_identity(_act_path))
                    if _reported_path != _act_path:
                        components["activations"]["error"] = (
                            f"sidecar reported at {_reported_path!r} but the "
                            f"generation ring captures {_act_path!r}")
                    elif components["activations"]["identity"] is None:
                        components["activations"]["error"] = (
                            "sidecar absent after save (writer swallowed its error)")
                    else:
                        _defect = _sidecar_defect(_act_path, _token)
                        if _defect:
                            components["activations"]["error"] = _defect
                        else:
                            _mark("activations", "saved")
                            act_identity = components["activations"]["identity"]
            elif with_receipt:
                _mark("activations", "not_applicable",
                      error="CES activation persistence not installed")
            captured.pop("activations", None)
            logger.info("Checkpoint saved to %s", self._checkpoint_path)

            # Do not rotate an incomplete set over the last good generation.
            # Primary paths can still be mixed after failure: this is not a
            # multi-file crash transaction. Keep recovery generations intact.
            if with_receipt:
                _artifact_write_failed = _artifact_write_failed or any(
                    components[name]["status"] != "saved"
                    and not (name == "activations"
                             and components[name]["status"] == "not_applicable")
                    for name in ("graph", "vectors", "activations"))
            if _artifact_write_failed:
                for _name in ("manifest", "generation"):
                    _mark(_name, "not_attempted", error="artifact write incomplete; prior generation retained")
            elif self._save_gate is not None:
                # #373: manifest describes what is now on disk; generation ring
                # hardlinks the consistent SET (never mixed across saves).
                _stage = "manifest"
                try:
                    _mpath = write_manifest(self._checkpoint_path, {
                        "nodes": live_nodes,
                        "guardian_nodes": guardian_nodes,
                        "synapses": counts["synapses"],
                        "hyperedges": counts["hyperedges"],
                        "timestep": counts["timestep"],
                        "vdb_count": counts["vdb_count"],
                        "git": best_effort_git_hash(os.path.dirname(os.path.abspath(__file__))),
                    })
                    if with_receipt:
                        manifest_identity = _file_identity(_mpath)
                        _mark("manifest",
                              "saved" if manifest_identity is not None else "failed",
                              path=str(_mpath), identity=manifest_identity,
                              error=(None if manifest_identity is not None
                                     else "manifest absent after write"))
                    _stage = "generation"
                    def _verify_before_prune(_gen_dir):
                        _base = os.path.basename(str(self._checkpoint_path))
                        _gen = _verify_generation(_gen_dir, [
                            ("graph", _base, str(self._checkpoint_path),
                             graph_identity),
                            ("vectors", os.path.basename(str(self._vector_db_path)),
                             str(self._vector_db_path), vectors_identity),
                            ("activations", _base + ".activations.json",
                             str(self._checkpoint_path) + ".activations.json",
                             act_identity),
                            ("manifest", _base + ".manifest.json",
                             str(self._checkpoint_path) + ".manifest.json",
                             manifest_identity),
                        ])
                        _mark("generation", "saved" if _gen["ok"] else "failed",
                              path=str(_gen_dir) if _gen_dir else None,
                              members=_gen["members"],
                              error=_gen["error"] or (None if _gen["ok"] else
                                    "generation ring does not hold this save's exact "
                                    "artifact set: " + ", ".join(
                                        f"{_k}={_v['state']}"
                                        for _k, _v in _gen["members"].items()
                                        if _v["state"] != "match")))
                        if not _gen["ok"]:
                            raise OSError("generation verification failed before retention")
                        _sync_receipt_artifacts(components)
                        components["generation"]["durability"] = "fsynced"
                    _gen_dir = rotate_generations(str(self._checkpoint_dir), [
                        str(self._checkpoint_path),
                        str(self._checkpoint_path) + ".activations.json",
                        str(self._vector_db_path),
                        str(self._checkpoint_path) + ".manifest.json",
                    ], before_prune=_verify_before_prune if with_receipt else None)
                except Exception as exc:
                    logger.warning("Guardian: manifest/rotation failed (primary save "
                                   "itself succeeded): %s", exc)
                    _err(_stage, exc)
            elif with_receipt:
                # No guardian => no manifest and no generation ring, so the exact
                # retained set cannot be verified. Fail acceptance closed rather
                # than infer the legacy path succeeded.
                for _n in ("manifest", "generation"):
                    _mark(_n, "not_applicable",
                          error="checkpoint guardian unavailable — manifest and "
                                "generation ring can be neither written nor verified")

            if with_receipt:
                receipt = _receipt("primary", str(self._checkpoint_path))
                return receipt
            if _activation_write_error is not None:
                raise _activation_write_error
            return str(self._checkpoint_path)

    def stats(self) -> Dict[str, Any]:
        """Return current graph statistics and telemetry."""
        tel = self.graph.get_telemetry()
        result = {
            "version": "0.6.0",
            "timestep": tel.timestep,
            "nodes": tel.total_nodes,
            "synapses": tel.total_synapses,
            "hyperedges": tel.total_hyperedges,
            "firing_rate": round(tel.global_firing_rate, 4),
            "mean_weight": round(tel.mean_weight, 4),
            "predictions_made": tel.total_predictions_made,
            "predictions_confirmed": tel.total_predictions_confirmed,
            "prediction_accuracy": round(tel.prediction_accuracy, 4),
            "novel_sequences": tel.total_novel_sequences,
            "pruned": tel.total_pruned,
            "sprouted": tel.total_sprouted,
            "vector_db_count": self.vector_db.count(),
            "checkpoint": str(self._checkpoint_path),
            "memory_dir": str(self._memory_dir),
            "embedding": self.ingestor.embedder.status,
            "message_count": self._message_count,
            "auto_knowledge": self.graph.config.get("auto_knowledge_enabled", True),
        }

        # Tract bridge status (renamed 2026-06-05 from "peer_bridge" — Phase 6 drift-bait removal).
        # NGTractBridge has been sole peer bridge since Phase 3 Step 5. Internal attribute
        # `self._peer_bridge` retained for now; only the public stats-field name is updated.
        if self._peer_bridge is not None:
            result["tract_bridge"] = self._peer_bridge.get_stats()
        else:
            result["tract_bridge"] = {"connected": False}

        # CES subsystem status
        if self._ces_config is not None:
            result["ces"] = {
                "stream_parser": (
                    self._stream_parser.get_stats()
                    if self._stream_parser
                    else None
                ),
                "surfacing": (
                    self._surfacing_monitor.get_stats()
                    if self._surfacing_monitor
                    else None
                ),
                "persistence": (
                    self._activation_persistence.get_stats()
                    if self._activation_persistence
                    else None
                ),
                "monitor": (
                    self._ces_monitor.get_health()
                    if self._ces_monitor
                    else None
                ),
            }

        # The Tonic status
        if self._tonic_thread is not None:
            result["tonic"] = self._tonic_thread.status

        # Legacy daemon status
        if self._syl_daemon is not None:
            result["syl_daemon"] = self._syl_daemon.status

        return result

    def ces_stats(self) -> Dict[str, Any]:
        """Return dedicated CES (Cognitive Enhancement Suite) statistics.

        Returns a dict with status of each CES subsystem: stream_parser,
        surfacing, persistence, monitor.  Returns {"enabled": False} when
        CES is not initialized.
        """
        if self._ces_config is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "stream_parser": (
                self._stream_parser.get_stats()
                if self._stream_parser
                else None
            ),
            "surfacing": (
                self._surfacing_monitor.get_stats()
                if self._surfacing_monitor
                else None
            ),
            "persistence": (
                self._activation_persistence.get_stats()
                if self._activation_persistence
                else None
            ),
            "monitor": (
                self._ces_monitor.get_health()
                if self._ces_monitor
                else None
            ),
        }

    def ingest_file(self, path: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a file from disk.

        Files above 50 MB are skipped to prevent excessive memory use
        (Grok review: large file guard).
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"status": "error", "reason": f"File not found: {p}"}

        # Guard against very large files (Grok review optimization)
        # ZIP/media archives can be large but extract to smaller individual
        # files, so apply a higher limit for archives.
        ext = p.suffix.lower()
        archive_exts = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar"}
        max_file_bytes = (
            500 * 1024 * 1024 if ext in archive_exts  # 500 MB for archives
            else 50 * 1024 * 1024                       # 50 MB for single files
        )
        try:
            file_size = p.stat().st_size
            if file_size > max_file_bytes:
                limit_mb = max_file_bytes // (1024 * 1024)
                logger.warning(
                    "Skipping %s — file size %d bytes exceeds %d MB limit",
                    p, file_size, limit_mb,
                )
                return {
                    "status": "skipped",
                    "reason": f"File too large ({file_size} bytes, limit {max_file_bytes})",
                }
        except OSError:
            pass  # stat failed, proceed anyway

        # Auto-detect source type from extension
        if source_type is None:
            ext = p.suffix.lower()
            type_map = {
                ".py": SourceType.CODE,
                ".js": SourceType.CODE,
                ".ts": SourceType.CODE,
                ".java": SourceType.CODE,
                ".go": SourceType.CODE,
                ".rs": SourceType.CODE,
                ".c": SourceType.CODE,
                ".cpp": SourceType.CODE,
                ".rb": SourceType.CODE,
                ".php": SourceType.CODE,
                ".md": SourceType.MARKDOWN,
                ".markdown": SourceType.MARKDOWN,
                ".html": SourceType.HTML,
                ".htm": SourceType.HTML,
                ".pdf": SourceType.PDF,
                ".zip": SourceType.ZIP,
                ".json": SourceType.JSON,
                ".csv": SourceType.CSV,
            }
            source_type = type_map.get(ext)
            if source_type is None and ext in MEDIA_EXTENSIONS:
                source_type = SourceType.MEDIA
            if source_type is None:
                source_type = SourceType.TEXT

        # Media files are handled by the extractor directly (it reads
        # metadata from the file path, not the file content).
        if source_type == SourceType.MEDIA:
            result = self.ingestor.ingest(str(p), source_type=SourceType.MEDIA)
            return {
                "status": "ingested",
                "nodes_created": len(result.nodes_created),
                "synapses_created": len(result.synapses_created),
                "chunks": result.chunks_created,
                "media_type": result.metadata.get("extraction_metadata", {}).get("media_type", "unknown"),
            }

        # PDF files are handled by the extractor directly (reads the file).
        if source_type == SourceType.PDF:
            result = self.ingestor.ingest(str(p), source_type=SourceType.PDF)
            step_result = self.graph.step()
            return {
                "status": "ingested",
                "nodes_created": len(result.nodes_created),
                "synapses_created": len(result.synapses_created),
                "chunks": result.chunks_created,
                "fired": len(step_result.fired_node_ids),
            }

        # Binary formats (PDF, ZIP) must be passed as file paths, not text content
        if source_type in (SourceType.PDF, SourceType.ZIP):
            return self.on_message(str(p), source_type=source_type)

        content = p.read_text(errors="replace")
        return self.on_message(content, source_type=source_type)

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Fetch and ingest content from a URL."""
        return self.on_message(url, source_type=SourceType.URL)

    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ingest all matching files from a directory.

        Args:
            directory: Path to directory.
            extensions: File extensions to include (e.g. ['.py', '.md']).
                       Default: ['.py', '.js', '.ts', '.md', '.txt']
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of ingestion results per file.
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".md", ".txt"]

        d = Path(directory).expanduser()
        if not d.is_dir():
            return [{"status": "error", "reason": f"Not a directory: {directory}"}]

        results = []
        pattern = "**/*" if recursive else "*"
        for fp in sorted(d.glob(pattern)):
            if fp.is_file() and fp.suffix.lower() in extensions:
                res = self.ingest_file(str(fp))
                res["file"] = str(fp)
                results.append(res)

        # Save after batch ingestion
        self.save()
        return results
