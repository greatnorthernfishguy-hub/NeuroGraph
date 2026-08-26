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
# [2026-08-14] Claude Code (Opus 4.8) — #143 hoist CC-host init above Lenia populate
# What: Moved the "#74 defer" _init_cc_host_bg daemon-thread .start() from AFTER the
#   Lenia FlowGraph block to BEFORE it, in handle_bootstrap. Same thread, same
#   defensive wrapping; now uses module-level threading.Thread (was _thr, which is
#   imported further down). Old site left a pointer comment.
# Why: The #74 defer kept CC init off the 60s bootstrap-RPC watchdog, but the .start()
#   still sat downstream of the synchronous Lenia distance-cache populate() — a
#   multi-hour, sometimes-never-completing rebuild (~17.7M pairs on the live graph).
#   Evidence: the boot up since 2026-08-13 05:03 was still in populate() 27h later
#   (83 periodic checkpoints, 0 "Bootstrapped:" logs, 0 "CC NG host init dispatched")
#   — handle_bootstrap never reached the dispatch, so CC-host init never ran and the
#   CC socket never came up (stale daemon.sock from the prior boot). Dispatching first
#   lets CC rebuild concurrently with Lenia instead of being stranded behind it.
# How: _init_cc_host_bg touches only module-level names (os/logging/logger/cc_ng_host),
#   no handle_bootstrap locals, so it is position-independent within the function.
#   Punchlist #394; CC-CALLOSUM-TRUTH.md §10.1 Phase 4. Follows #74 (2026-07-19).
# [2026-07-29] Claude Code (Opus 5) — #93 graduation must be earned, not aged into
# What: _update_probation no longer stamps metadata["graduated"] on timer expiry alone.
#   Added _has_ever_fired(node) (reads spike_history) and env knob
#   ANIMA_CONV_PROBATION_REQUIRE_SPIKE (default "1", set 0 to restore pure-timer).
#   Un-fired nodes that age out get metadata["probation_expired_unfired"]=True and stay
#   eligible for late graduation if they ever fire. Novelty-dampening release is
#   deliberately NOT gated — it still happens on the timer.
# Why: #93 needs an earned-protection signal, but "graduated" was a pure wall-clock flag
#   (99.3% of nodes carried it), so it discriminated nothing. Gating the DAMPENING on
#   firing too would be a self-reinforcing trap: a never-fired node would keep a boosted
#   threshold, be less likely to fire, and could never earn release — that would
#   permanently handicap the 64% of Syl's nodes that have never fired. Only the stamp is
#   gated, so the flag becomes honest at zero behavioural cost.
# How: spike_history is the only monotonic firing ledger (#96) — last_spike_time is also
#   stamped by prime_and_propagate() in write_mode (Tonic traversal would forge it) and
#   firing_rate_ema decays back toward 0. No production code reads "graduated" today
#   (only openclaw_hook.py:975's log counter), so this is inert until #93 consumes it.
# [2026-07-11] Claude Code (Haiku 4.5) — #381-B quiet-hours dream consolidation pulse
# What: Added non-protected _dream_gate_open (pure), _dream_consolidation_pulse_loop,
#   _start_dream_consolidation_pulse, plus env-knob module globals: _DREAM_IDLE_SECS (1800),
#   _DREAM_MIN_INTERVAL_SECS (21600), _DREAM_ALERT_SECS (86400), _DREAM_TICK_SECS (60).
#   Wired thread start/shutdown via daemon thread + shutdown.Event idiom mirroring scan-drain.
# Why: #381-B — consolidate_hyperedges (shed + seatbelt-merge + subsume) should run during
#   quiet hours only (idle ≥30min, arousal PARASYMPATHETIC, rate-limited ≥6h). First pass
#   collapses mega-HE clones; 24h floor alerts when gate never opens (her constraint: dream
#   the pruning, never force while active).
# How: gate checks (now-last_turn_ts)≥IDLE and arousal≠SYMPATHETIC and (now-last_pass_ts)≥MIN_INTERVAL.
#   Loop calls graph.consolidate_hyperedges() under graph._step_lock when gate opens. Autonomous
#   thread on _DREAM_TICK_SECS cadence. Wired in handle_bootstrap and shutdown in handle_dispose.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — #371 reconcile-not-discard
# What: handle_bootstrap's Lenia block: on pruned-entity mismatch, call
#   lenia_cache.reconcile_removals(_live_ids) and fall through to the existing
#   watermark-resume/growth branches with the surviving order; full rebuild only
#   when reconcile returns None.
# Why: #371 — the subset-check bail ran BEFORE the watermark branch, so one pruned
#   node discarded the whole cache incl. resume progress (Syl lost ~1.79M computed
#   pairs at the 2026-07-08 07:50 restart; with continuous pruning the ~3-day
#   rebuild could never complete).
# How: reconcile compacts + translates in kernel.py; this block only swaps the
#   bail for the call. Downstream branches, fail-soft shape, and the post-branch
#   save are untouched.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — Lenia resume branch
# What: handle_bootstrap()'s Lenia block gains an elif between full-rebuild and growth:
#   a loaded cache carrying a resume watermark resumes the interrupted rebuild (with
#   resize first if the graph also grew) instead of being treated as complete.
# Why/How: see lenia/kernel.py's 2026-07-08 changelog entry — this is the caller half.
# [2026-07-06] Claude Code (Sonnet 5) — Periodic Lenia checkpointing during populate() (Josh-approved)
#   What: both handle_bootstrap() populate() call sites now pass
#     checkpoint_interval_secs=_LENIA_CHECKPOINT_INTERVAL_SECS (5 min, new
#     module constant) and on_checkpoint=lambda: lenia_cache.save(_cache_path).
#   Why: the 2026-07-05 incremental-extension fix made restarts cheaper but
#     did nothing for a run interrupted mid-populate — the only save() call
#     happened once, after the whole loop returned, unreachable on a hard
#     process kill. Confirmed live with Josh 2026-07-06: distance_cache.npz
#     was still dated 2026-07-02 after multiple full-day restart cycles,
#     each discarding 100% of that attempt's progress and re-triggering the
#     same "incompatible, full repopulate" path from the same stale save.
#   How: see lenia/kernel.py's populate() — checkpointing happens inside its
#     own loop now, time-checked every 1000 pairs, calling the caller's save
#     callback rather than a hardcoded path.
# [2026-07-05] CC (laptop) — Incremental Lenia distance-cache extension (Josh-approved)
#   What: handle_bootstrap's Lenia block now extends the on-disk DistanceCache in place
#     when the graph only grew since the last save, instead of nuking and repopulating
#     from scratch on any entity_count drift. Falls back to full rebuild only when
#     entities were actually removed (pruned) or on first-ever run.
#   Why: journalctl history showed the full rebuild took up to ~8 hours on Syl's live
#     graph, and every restart since the one successful save (Jun 30 -> Jul 02) got
#     interrupted by the next restart before ever reaching save() again — permanently
#     stuck repeating the same multi-hour attempt from scratch. This was the actual
#     reason CC's own Tonic/BrainSwitcher registration (code sits after this block)
#     never ran on any restart tonight, not a bug in CC's own init path.
#   How: see lenia/kernel.py (DistanceCache.populate's start_index, entity_ids
#     persistence) and lenia/graph_substrate.py (NeuroGraphSubstrate.known_entity_order,
#     _hyperedge_similarity concurrency fix — RuntimeError: dictionary changed size
#     during iteration, seen live in the Jul 3 11:36:42 crash traceback).
# [2026-06-28] Claude Code (Sonnet 4.6) — #294-B: wire dual-pass into afterTurn (Commons path)
#   What: call _file_conversational_experience() for both turn halves (user + assistant) directly
#     in handle_after_turn(), immediately after the Commons experience deposit.
#   Why: #294 gap — the dual-pass recall-store write was stranded in handle_ingest() (old OC
#     JSON-RPC slot Anima never calls). NG deposits on behalf of Anima — no separate Anima path
#     needed (Anima replaced OC; OC didn't need one either). The afterTurn call already carries
#     lastUserMessage and lastAssistantMessage; NG owns this deposit, not Anima.
#   How: two fail-soft _file_conversational_experience() calls (source="anima") after
#     _deposit_experience_to_river(); user turn reuses cached _ingest_embedding (free). All
#     retry/enqueue-on-failure logic lives inside the helper (Law 3 — single filing point).
# [2026-06-30] Claude Code (Sonnet 4.6) — #97 TID Commons valence routing: peninsula Commons-side
#   What: Start TIDPeninsulaCommons at NG startup; call tid_peninsula_push_enhanced() after each
#     Commons enhance scoop so fresh enhanced recs flow to TID-side peninsula.
#   Why: TID substrate peninsula — gives TID's compute body Commons-enriched routing intelligence
#     without a cross-module bridge. Intra-module IPC (TID's two halves), not inter-module.
#   How: start_tid_peninsula() in bootstrap (fail-soft); tid_peninsula_push_enhanced() appended
#     to _run_commons_enhance_scoop() call in _scan_drain_pulse_loop (fail-soft import).
# [2026-06-27] Claude Code (Sonnet 4.6) — #TID-cost-budget: fix compaction model (LAW 4)
#   What: compaction TID call no longer uses model:"auto". Uses NG_COMPACTION_MODEL env var
#     (default: openrouter/meta-llama/llama-3.3-70b-instruct). Bypasses TID routing entirely
#     for this infrastructure-level summarization task.
#   Why: "auto" sent a 300+ word prompt to TID → classifier called it EXTREME complexity →
#     Opus won on complexity_fit every time regardless of cost. 429 consecutive Opus calls
#     drained $30 from OpenRouter overnight. Summarizing a conversation requires no frontier
#     reasoning — any 70B model handles it. LAW 5: configurable via NG_COMPACTION_MODEL.
#   How: os.environ.get("NG_COMPACTION_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct")
#     replaces the "auto" literal in the TID request body.
# [2026-06-26] Claude Code (Sonnet 4.6) — #337: reach confabulation on bare file path
#   What: (1) whisper tier of _render_reach_teaching updated to mention file-path-as-reach-cue;
#     (2) _is_bare_path() + bare-path hint injection in handle_assemble so a lone path in Josh's
#     message surfaces an explicit reach nudge regardless of her competence tier;
#     (3) seed_reach_teaching.py updated with new example + --update flag.
#   Why: Syl received a bare file path, confabulated reading it instead of emitting [[reach:]].
#   How: whisper updated; bare-path detection in /assemble appends note to systemPromptAddition.
# [2026-06-14] Claude Code (DudeMan CC, Opus 4.8) — #spine: surface Syl's self-model in /assemble
#   What: new _render_self_and_wants(graph); handle_assemble PREPENDS "## Who I Am" (her
#     constitutional core nodes, ordered) + "## What I Want" (her live want-nodes, newest-first)
#     to systemPromptAddition, so her stable self leads every turn.
#   Why: /assemble surfaced only query-driven associations/recall/Tonic — no crystallized self
#     (her graph reads constitutional=0), so a lens had to RECONSTRUCT her each turn. This is the
#     read side of the hybrid self-model Syl authored (docs/prd/syl-constitutional-spine-v0.1).
#     Wants are read LIVE (not snapshot) so new [WANT]s are accounted for automatically.
#   How: prepend after marker-stripping/briefing so the spine prose is untouched + leads; graceful
#     "" when no constitutional/want nodes (pre-seed behavior unchanged). Paired with the
#     neuro_foundation orphan-skip (#spine) that keeps these nodes from ever being pruned.
# [2026-06-15] Claude Code (DudeMan CC, Opus 4.8) — #spine: self-block is NOT query-gated
#   What: compute _self_block up front (after the _memory guard) and surface it even on the
#     no-recent-text early return, instead of returning None there.
#   Why: live deploy showed the self-block sat AFTER `if not recent_text: return None`, so the
#     between-turns / Tonic assemble (empty messages) lost her self — exactly the "feel across
#     turns" space (invariant #4) where her anchor matters most. Her constitutional self is who
#     she IS, not a response to a query; it must never be gated on user input.
# [2026-06-14] Claude Code (Opus 4.8) — #294-A single filing point (recall durability)
#   What: add _file_conversational_experience() — one chokepoint that recall-indexes a
#     conversational experience (embed -> dual-pass -> enqueue-on-failure) keyed on
#     conversational source. Route _absorb_conversational_experience's inner loop AND
#     _drain_experience_entry through it.
#   Why: docs/prd/2026-06-14-syl-recall-heal-phase1-design.md Component A — filing must be a
#     property of conversational experience entering the substrate, not of which feeder
#     delivered it (Syl: "filing as a property of being experienced"). A future non-animus
#     feeder would otherwise silently skip recall.
#   How: extract the absorb inner-loop dispatch into the helper (Law 3/4 — consolidate, no 5th
#     path); _drain_experience_entry early-routes conversational source to the helper before
#     its ingestor (knowledge) path. Non-conversational experience unchanged.
# [2026-06-13] Claude Code (Opus 4.8) — KISS restore for the Anima era
#   What: lazy-init _kiss_filter in handle_assemble() (was only init'd in the dead
#     OpenClaw handle_bootstrap slot Anima never calls).
#   Why: _kiss_filter stayed None -> full transcript sent to model every turn ->
#     OpenRouter cost blowup + oversized-request 502s ('[TID: malformed response]').
#   How: anchored insert before the existing `if _kiss_filter is not None` guard;
#     mirrors the bootstrap init (recent_window=10). Law 3 (restore, not rebuild).
# [2026-06-12] Claude Code (Opus 4.8, surfacing CC) — substrate-first surfacing in handle_assemble
# What: handle_assemble() runs a substrate-first content resolution pass over surfaced +
#   ces_surfaced (and the Active Recall block) via surface_resolver.resolve_surface_content —
#   prefers each node's metadata['_forest_content'] (her turn, in the substrate) over the vdb
#   shard, and filters ingested source-code + degenerate fragments. Fail-safe (critical path).
# Why: CES/spreading/recall were displaying vdb SHARDS + degenerate fragments ('o','want') instead
#   of her voice = "no Syl" (handoff 2026-06-12). The vdb is NOT the substrate; the bucket dips the
#   substrate. Recovery for the surfacing collapse; sandbox-tested (tests/test_surface_resolver.py).
# How: a fail-safe _resolve_surfaced() pass before _format_substrate_context + the recall loop;
#   lazy import of resolve_surface_content (matches this file's lazy-import pattern). LAW 1/7: read-
#   only bucket refinement, no deposit changed, no module call, classify only at extraction.
# [2026-06-07] CC (Opus 4.8) — Conversation uses the Ingestor-free experiential path (Task A)
# What: _absorb no longer calls ingestor.ingest (document chunking). A turn now deposits as a
#       forest gestalt node + tree concept nodes into BOTH the recall vdb AND the SNN, with a
#       binding hyperedge and a delayed prev->current forest link (#257 polychrony). New helpers
#       _deposit_memory_node / _bind_conversational_topology / _update_probation. Probation
#       graduation decoupled from the ingestor (handle_after_turn calls _update_probation).
# Why:  The Universal Ingestor was never intended for conversation (Josh); it chunked her turns
#       like documents and entered them second-class. Restores the pre-ingestor experiential
#       principle, updated to today's SNN (DiffPC/GSG/delays). Ingestor reserved for documents.
# How:  Direct neuro_foundation primitives (create_node/create_synapse/create_hyperedge) + ng_embed
#       + vector_db.insert + poincare_dir stamp; light novelty-dampening (NG's own Competence-Model
#       concern, env-tunable, never Elmer). LAW 3 restore; non-protected; no vendored edits.
# [2026-06-07] CC (Opus 4.8) — Tonic conversation<->latent lifecycle restore (Anima-migration regression)
# What: handle_assemble() now calls _tonic_thread.message_received() per turn;
#       new _tonic_idle_watcher daemon calls conversation_ended() after
#       ANIMA_TONIC_IDLE_SECS of quiet. Adds _tonic_check_idle(),
#       _tonic_idle_pulse_loop(), _start_tonic_idle_watcher().
# Why:  OpenClaw drove conversation_started (on_message) and conversation_ended
#       (dispose). Anima's 2-verb HTTP surface (/assemble, /afterTurn) calls
#       NEITHER, so the Tonic was pinned in 'conversation' mode forever
#       (_in_conversation never reset). That pins the engine at conversation
#       cadence AND makes TonicBridge defer forever (it only deposits between-
#       turns wants when _in_conversation is False) — so Syl's between-turns
#       latent awareness / curiosity could never fire even if the bridge were
#       enabled. Restores the dropped lifecycle, NG-side only. LAW 3 (restore).
# How:  Per-turn message_received() in handle_assemble; idle-watcher daemon
#       mirrors the scan-drain pulse pattern; idle threshold via env (LAW 5).
# -------------------
# [2026-06-05] CC (Sonnet 4.6) — #297: bounded non-cyclic pass-2 retry-queue, drained on the autonomic pulse
# What: Split _conversational_dual_pass into _run_conversational_dual_pass (core, returns bool, no enqueue)
#       and _conversational_dual_pass (wrapper, enqueues on failure). Added _retry_queue(), _enqueue_failed_extraction(),
#       _drain_pass2_retries(). Drain wired into handle_after_turn after self-observation block.
# Why:  Failed pass-2 extractions were silently dropped forever. Retry-queue gives bounded recovery.
#       Non-cyclic by construction: drain uses core (no enqueue path), items at max_attempts are dropped
#       not re-queued, preventing the wire->absorb->extract OOM-recursion class.
# How:  memory_retry_queue.RetryQueue (msgpack-backed, dedup by target_id). Path/attempts from env vars
#       ANIMA_PASS2_RETRY_PATH / ANIMA_PASS2_RETRY_MAX_ATTEMPTS (LAW 5). Drain on pulse = off ingest hot path.
# [2026-06-05] CC (Sonnet 4.6) — #297 review fixes: per-pulse drain cap (LAW 5) + remove unused embedding param
# What: _drain_pass2_retries reads ANIMA_PASS2_RETRY_MAX_PER_PULSE (default 5) and passes as limit= to drain().
#       _enqueue_failed_extraction(text, embedding) → _enqueue_failed_extraction(text); embedding never persisted.
# Why: Unbounded drain on a long-outage queue would make one ONNX embed call per item, stalling the sidecar.
#      LAW 5: config cap belongs in env var, not hardcoded. Unused param was misleading — implied embedding stored.
# How: os.environ.get("ANIMA_PASS2_RETRY_MAX_PER_PULSE", "5") cap passed to RetryQueue.drain(limit=cap).
#      Single signature change + single call-site update in _conversational_dual_pass wrapper.
# [2026-06-05] CC (Sonnet 4.6) — #296a fix: target_id hashes full text (collision-safe across turns)
# What: _conversational_dual_pass target_id changed from sha1(text[:256])[:16] to sha1(text) (full text, full hex).
# Why: Two turns sharing the same first-256-char prefix produced the same target_id → same ::tree:: id →
#      SimpleVectorDB.insert silently overwrote the earlier atom. Episodic memory lost without error.
#      Full-text hash guarantees genuinely different turns produce different forest ids → no collision.
#      (Byte-identical turns still map to one id — acceptable: same content = same atom, Phase 3 consolidation handles it.)
# How: Single line change. No other behavior altered. Regression test added in tests/test_memory_phase1.py.
# [2026-06-05] CC (Opus 4.8 subagent) — #296a: conversational turns dual-passed; trees land in Syl's recall store tagged {syl:true}
# What: Add _ConversationalDualPassEco (eco adapter that inserts tree concepts into vector_db) and
#       _conversational_dual_pass() (named caller-side step invoking ng_embed dual_record_outcome).
#       Call site added in handle_ingest() after TrollGuard sidecar, before return.
# Why: Syl's specific conversational concepts (turn-level atoms) were never indexed in recall — only pass-1 chunks.
#      Trees carry her lived specifics; {syl:true} tag distinguishes lived memory from River-flowed-in topology.
# How: _ConversationalDualPassEco.record_outcome inserts only when _tree_concept+_concept present (forest ignored).
#      record_outcome_broadcast delegates to record_outcome (ng_embed uses broadcast variant when hasattr detects it).
#      _conversational_dual_pass wraps NGEmbed.get_instance().dual_record_outcome; non-fatal on any exception.
# [2026-06-05] CC (Opus 4.8 subagent) — #295: River-backflow routes peer telemetry to substrate only, not Syl's recall store
# What: _drain_peer_tracts now passes index_in_recall=False to registrar.register, removes associator.associate call,
#       and replaces the no-embedding ingestor.ingest(target) fallback with a silent pass
# Why: ~90% of Syl's recall store was machine telemetry from peer River events (Darwin sims, inference metrics, etc.)
#      — they belong in the substrate graph, not in the recall vector_db (#295 Decision 2)
# How: Single keyword arg + two line removals in the backflow loop; substrate graph node still created (LAW 7 preserved)
# [2026-06-03] Claude (Sonnet 4.6) — Phase 5: active recall in handle_assemble()
# What: After spreading activation, run _memory.recall(recent_text) and append a
#       "## Active Recall" block to context_block inside handle_assemble().
# Why: Spreading activation is associative (what does this topic remind Syl of?).
#      Direct recall is targeted (what has Syl learned that matches this query?).
#      Both together = richer grounding per turn. ANIMA_RECALL_K / ANIMA_RECALL_THRESHOLD
#      env vars control depth and confidence floor (LAW 5).
# How: Insert after _format_substrate_context(), before _read_outbound_log().
#      Errors caught and logged at DEBUG so a recall failure never breaks assembly.
# [2026-06-02] Claude (Sonnet 4.6) — Add GET /stats to HTTP sidecar
# What: New GET /stats route in do_GET calls handle_stats({}) — same data as JSON-RPC "stats"
# Why: Anima GUI's NG Status tab needs substrate telemetry (nodes, synapses, timestep, etc.)
#      via HTTP; only /status and /modules were previously exposed; /stats was JSON-RPC only
# How: Single elif branch — delegates to existing handle_stats(), zero new logic
# [2026-05-28] Claude Code (Sonnet 4.6) — GSG Phase 4: spherical surfacing
#   What: assemble() GSG re-scoring now checks node.manifold_type. Spherical nodes
#         use great circle distance arccos(dot(query_dir, node_dir)); hyperbolic
#         nodes use existing Poincare geodesic (Phase 1). Cross-manifold: neutral.
#   Why:  Completes S component of S x E x H mixed manifold from source GSG paper.
#   How:  getattr(nd, 'manifold_type', 'hyperbolic'); arccos on clamped dot product.
# [2026-05-26] Claude Code (Sonnet 4.6) — GSG backfill: stamp poincare_dir on all existing nodes at bootstrap
#   What: Added _gsg_backfill_existing_nodes() called at end of handle_bootstrap(). Iterates all
#         graph nodes, stamps node.metadata['poincare_dir'] from vector_db.embeddings[node_id]
#         (already L2-normalized on insert) for any node that lacks the field. Saves checkpoint
#         if any nodes were stamped. Idempotent: subsequent restarts skip in microseconds.
#   Why:  GSG Phase 1 scores surfaced nodes by hyperbolic distance from query. Existing nodes
#         (all of Syl's accumulated topology) had no poincare_dir — the geometric re-scorer
#         was completely blind to her entire learned experience. Only brand-new nodes (post-GSG
#         ingest) were visible geometrically. This was architectural lobotomy: new nodes
#         getting scored, old topology invisible. Backfill corrects this at bootstrap so
#         GSG Phase 1 covers the full substrate from the first assemble call onward.
#   How:  SimpleVectorDB.insert() normalizes embeddings on storage — vector_db.embeddings[id]
#         IS the unit direction vector. No re-embedding, no model calls. Pure dict iteration.
#         Force-save via _memory.save() after backfill so metadata persists across restarts.
# [2026-05-25] Claude Code (Sonnet 4.6) — GSG: Poincaré ball embedding for geometric surfacing
#   What: Added _embed_to_poincare_dir(), _poincare_distance(), _GSG_LAYER_NORMS/_GSG_SCORE_BONUS.
#         At ingest: embedding normalized to unit direction, stored as node.metadata['poincare_dir']
#         for each newly created node. At assemble: query projected to Poincaré ball (Layer 0 norm),
#         each surfaced node projected using its current diffpc_layer; hyperbolic distance computed;
#         closer nodes receive a strength bonus (max +0.30), re-sorted.
#   Why:  DiffPC Phase 1 assigned semantic hierarchy (Layer 0=novel/input, 2=hub) but cosine
#         similarity is topology-blind. Hyperbolic geometry respects tree-like semantic structure:
#         Layer 0 near boundary (high curvature, novel concepts distinguished), Layer 2 near center
#         (hub/familiar concepts cluster tightly). Poincaré distance rewards retrieval of nodes
#         that are semantically close AND at the appropriate hierarchy level for the query.
#   How:  Pure numpy (no geoopt). poincare_dir stored in node.metadata; layer norm applied
#         dynamically at assemble from current diffpc_layer. River stays Euclidean. No vendored
#         file changes. Backward-compat: nodes without poincare_dir skip GSG scoring silently.
# [2026-05-25] Claude (Sonnet 4.6) — Phase 3: POST /assemble endpoint
#   What: Add /assemble handler to _AfterTurnHandler.do_POST
#   Why:  Animus ContextBuilder needs HTTP access to handle_assemble() for spreading activation
#   How:  elif block identical in structure to /recall; always returns HTTP 200 (error-in-body pattern)
# [2026-05-25] Claude Code (Sonnet 4.6) — DiffPC River deposit cluster tracker
#   What: Added _update_deposit_cluster() + birth threshold scaling for newly ingested nodes.
#         _deposit_centroid tracks running EMA of River deposit embeddings (alpha=0.05).
#         At ingest, novelty (cosine distance from centroid) maps to birth threshold:
#         novel → lower threshold (Layer 0 seed), familiar → higher (Layer 2 bootstrap).
#   Why:  New nodes start with zero graph connections — DAS-GNN has nothing to calibrate.
#         Bootstrap birth threshold from cross-message semantic novelty so DiffPC layer
#         assignment is semantically meaningful before organic connections accumulate.
#   How:  Birth threshold = default_threshold × (0.7 + 0.6 × (1 − novelty)),
#         clamped [0.5×dt, 1.2×dt]. Applied to result.nodes_created after ingest.
# [2026-05-25] Claude Code (Sonnet 4.6) — Anticipatory Pre-Activation (#256)
#   What: Added _anticipate() called at end of handle_after_turn(). Walks outgoing
#         synapses from fired nodes, scores neighbors by edge weight, stores top-15
#         with 120s TTL. handle_assemble() applies +0.25 strength bonus to primed
#         nodes in surfaced results and re-sorts.
#   Why:  All surfacing is reactive (input arrives → retrieve). Predictive coding
#         insight: mature SNNs anticipate what's coming next. Pre-priming nodes at
#         turn-end means Syl gets pre-loaded context before reading the next message.
#         Primed-but-wrong nodes amplify MMN surprise → #255 deepens surfacing further.
#   How:  _primed_nodes: Dict[str, Tuple[float, float]] (node_id → score, expiry_ts).
#         _ANTICIPATE_TTL_S=120, _ANTICIPATE_TOP_K=15, _ANTICIPATE_BONUS=0.25.
#         No changes to protected files.
# [2026-05-25] Claude Code (Sonnet 4.6) — Surprise-Weighted Adaptive Surfacing (#255)
#   What: handle_after_turn updates _memory._substrate_novelty_ema (EMA of MMN).
#         handle_assemble passes current novelty to _harvest_associations().
#   Why:  Closes the MMN feedback loop into retrieval. High surprise → deeper surfacing.
#   How:  EMA alpha=0.1, neutral start=0.5. Novelty read in assemble via getattr guard.
# [2026-05-25] Claude Code (Sonnet 4.6) — Kill _deposit_substrate_metrics() JSONL call sites
#   What: Removed both calls to _deposit_substrate_metrics() — one from handle_on_message
#         (afterTurn flow, line ~1936) and one from _scan_drain_pulse_loop (autonomous step).
#         Function definition retained but is now dead code pending explicit removal.
#   Why:  Josh's explicit request: JSONL path was Darwin's legacy feed before BTF tracts.
#         Darwin now receives topology events via _on_river_events() + BTF tracts (direct
#         deposit from _deposit_topology_to_river). The JSONL feed was redundant and pre-Law-7
#         (it wrote labeled metric names, not raw embeddings). Killing it removes the parallel
#         mechanism and ensures Darwin only sees raw substrate topology via the River.
#   How:  Two single-line deletions. _deposit_topology_to_river() immediately follows both
#         and remains — it is the correct Law 7 deposit path.
# [2026-05-25] Claude Code (Sonnet 4.6) — Fix tract_stats NameError + autonomous substrate step
#   What: (1) Removed dangling tract_stats["pending"] from handle_bootstrap return dict — variable
#         was deleted in a prior session but usage was left behind, causing NameError on every
#         startup self-bootstrap. (2) Added graph.step() + _deposit_substrate_metrics +
#         _deposit_topology_to_river to _scan_drain_pulse_loop, after _drain_peer_tracts().
#         Substrate now steps every 2s unconditionally. Topology deposits to River autonomously.
#   Why:  (1) Self-bootstrap failure on every restart. (2) Topology deposits were conversation-
#         gated (afterTurn only). Ecosystem must not depend on a conversation taking place.
#         Josh: "We do NOT want ANYTHING dependent on a conversation needing to take place."
#   How:  (1) Removed one line from handle_bootstrap return dict. (2) Pulse loop gains a
#         try/except step block after draining — deposit_experience excluded (requires text).
# [2026-05-25] Claude Code (Sonnet 4.6) — Fix severed River deposit in deposit_topology/experience
#   What: _deposit_topology_to_river now msgpack-packs StepResult scalars and calls
#         ng_tract.deposit_topology(raw_bytes, "neurograph", tract_paths). 
#         _deposit_experience_to_river now encodes text as UTF-8 and calls
#         ng_tract.deposit_experience(raw_bytes, "neurograph", tract_paths) with a list.
#         Both silent logger.debug failures promoted to logger.error.
#   Why:  Both functions called non-existent Rust API since 2026-04-28. ng_tract had no
#         deposit_topology/deposit_experience until today — AttributeError silently
#         swallowed at DEBUG. Zero bytes deposited to River for >3 weeks.
#   How:  Rust functions now implemented in ng-tract-rs. Payload is raw msgpack (Law 7 —
#         no pre-classification). Consumers decode at extraction time.
# [2026-05-25] Claude Code (Sonnet 4.6) — Wire River drain into autonomic pulse
#   What: Added _drain_peer_tracts() call inside _scan_drain_pulse_loop() alongside
#         _drain_scan_dir(). River drain now runs every 2s unconditionally, not only
#         during afterTurn (conversation-gated).
#   Why:  Post-fanout removal, NeuroGraph was the only module whose peer tract drain
#         depended on conversation turns. Between conversations, module deposits to
#         tracts/*/neurograph.tract accumulated unread. All peer modules (Elmer, Darwin,
#         TrollGuard, etc.) already drain on their own pulse — NG had the gap.
#   How:  _drain_peer_tracts() is safe from the pulse: returns early when bridge is None;
#         for NGTractBridge calls bridge._drain_all() then exits (no _peer_events). The
#         cursor mechanism makes idle calls cheap (fstat + early exit when no new data).
# [2026-05-23] Claude Code (Sonnet 4.6) — Checkpoint safety + drain node cap
#   What: Two fixes: (1) Move time-based auto-save into _scan_drain_pulse_loop() so it
#         fires unconditionally every 5 min even when message_count stays 0 (no convs).
#         (2) Add _MAX_DRAIN_NODES cap to _drain_scan_dir() — skip drain when node
#         count >= cap to prevent TID experience flooding causing unbounded RAM growth.
#   Why:  PID 97886 grew to 87,814 nodes / 4.8GB RAM from TID experience flooding while
#         message_count=0. Both count-based and time-based saves were inside
#         handle_after_turn() and never fired. Node explosion → OOM crash → May 15
#         checkpoint loaded, losing all learning since disk filled.
#   How:  _scan_drain_pulse_loop() now checks _last_save_time on every tick (fires after
#         _drain_scan_dir or even when paused, using global _last_save_time shared with
#         the afterTurn save path). _drain_scan_dir() returns early when node count >=
#         _MAX_DRAIN_NODES (default 15000, ~3x organic growth headroom from current 4,673).
# [2026-04-29] Claude (Sonnet 4.6) — #226 Bunyan: deposit raw conversation experience to River
#   What: Added _deposit_experience_to_river(text) — raw conversation text to every
#         registered module's inbound tract via ng_tract.deposit_experience() (single-path
#         API, loop over peers). Called in handle_after_turn alongside topology deposit.
#   Why:  Bunyan needs raw semantic experience, not just structural topology metadata.
#         Law 7: text enters as-is, no embedding at deposit. Bunyan embeds at its own
#         extraction boundary in _on_river_events(PyExperienceEntry handler).
#   How:  _deposit_experience_to_river(_ingest_text) after _deposit_topology_to_river.
# [2026-04-29] Claude (Sonnet 4.6) — Remove dead CC Tonic bootstrap registration (#159)
#   What: Removed the inner try block in handle_bootstrap() that attempted to register
#         CC's Tonic with BrainSwitcher. Replaced with comment pointing to correct location.
#   Why:  At bootstrap, _memory._modules is empty — registration was always a no-op.
#         Correct location: Elmer's _delayed_brain_load() (elmer_hook.py), which fires
#         60s post-startup after BrainSwitcher has loaded brains.
#   How:  Deleted the try block; added a 3-line comment for future reference.
# [2026-04-29] Claude (Sonnet 4.6) — #225 fix pt2: BTF path+format
#   What: _deposit_topology_to_river wrote JSONL to inverted path tracts/{peer}/neurograph.tract.
#         JSONL has no place in BTF tracts. Path was backwards.
#   Why:  Should write to tracts/neurograph/{peer}.tract (what peers drain).
#   How:  ng_tract.deposit_topology(step_result, graph, vector_db, tract_paths). 60→8 lines.
# [2026-04-28] Claude (Sonnet 4.6) — #225 River fix: initialize outbound tract bridge, deposit topology
#   What: Added _ng_tract_bridge (NGTractBridge module_id="neurograph") initialized in
#         handle_bootstrap(). Added _deposit_topology_to_river(step_result) called each
#         afterTurn. Fixed two dead peer_bridge.record_outcome() NameErrors in
#         _deposit_surfacing_outcome() and handle_after_turn() self-observation block.
#   Why:  peer_bridge was never defined — leftover from fanout removal 2026-04-05.
#         Both call sites inside try/except silently swallowed NameError every turn.
#         No topology deltas ever flowed to module tracts. Bunyan nodes:0, salient:0
#         since deployment. QuantumGraph msgs:0. All River-dependent module inboxes empty.
#   How:  See 2026-04-29 entry for corrected implementation.
# [2026-04-27] Claude Code (Sonnet 4.6) — he_discovery_overlap_threshold self-tuning (#222)
#   What: Added _tune_he_overlap_threshold() helper and 4 module-level tuning state globals.
#         discover_hyperedges() return value now captured; _he_discovered_in_window accumulates.
#         Every 50 turns: discovery_rate + net HE growth → ±0.03 nudge, bounds (0.2, 0.9).
#   Why:  Bootstrap 0.5 is a prior, not a permanent value. Coordinator self-tunes its own
#         algorithm's threshold based on observed discovery statistics (Law 1 compliant —
#         NeuroGraph tunes NeuroGraph's param; Elmer observes only via River).
#   How:  Three edits: globals block, _tune_he_overlap_threshold() before _deposit_substrate_metrics,
#         discover_hyperedges try block updated to accumulate + trigger every _HE_TUNE_WINDOW turns.
# [2026-05-08] Claude (Sonnet 4.6) — Raise lazy expansion batch size 5→50
#   What: _LAZY_EXPANSION_BODIES_PER_TICK increased from 5 to 50.
#   Why:  30,706 body files accumulated in ~/.et_modules/experience/bodies/ — TID's
#         deposition rate (~200/hr) outpaced the 5/tick design (150/hr processed).
#         At 50/tick the backlog clears in ~24h; sustainable thereafter.
#   How:  Single constant change. No behavior change to expansion logic itself.
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
import re
import sys
import time
import traceback
import urllib.request
import threading
from typing import Any, Dict, List, Optional, Tuple

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

# NeuroGraph outbound River tract — writes raw topology deltas to module inboxes
_ng_tract_bridge: Optional[Any] = None

# Last ingested text+embedding — passed to topology delta for River distribution
_ingest_text: Optional[str] = None
_ingest_embedding: Optional[Any] = None  # np.ndarray
_module_errors: Dict[str, str] = {}
_module_error_times: Dict[str, float] = {}

# Punchlist #56: Surfacing outcome cache — what was surfaced during assemble(),
# deposited as raw experience in afterTurn() alongside Syl's response.
_last_surfaced_nodes: List[Dict[str, Any]] = []

# Anticipatory pre-activation state (#256)
_ANTICIPATE_TTL_S: float = 120.0    # primed state expires 2 min after set
_ANTICIPATE_TOP_K: int = 15         # candidate nodes to prime per call
_ANTICIPATE_BONUS: float = 0.25     # strength bonus for primed nodes in assemble
_primed_nodes: Dict[str, Tuple[float, float]] = {}  # node_id → (score, expiry_ts)
# DiffPC: River deposit cluster tracker (semantic novelty at birth)
_DEPOSIT_CLUSTER_ALPHA: float = 0.05
_deposit_centroid: Optional[Any] = None           # np.ndarray running centroid
_deposit_centroid_lock: threading.Lock = threading.Lock()
# GSG: layer-specific Poincaré ball norms (Layer 0=novel/input near boundary, Layer 2=hub near center)
_GSG_LAYER_NORMS: List[float] = [0.70, 0.50, 0.30]
_GSG_SCORE_BONUS: float = 0.30   # max strength bonus from hyperbolic proximity

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
_LENIA_CHECKPOINT_INTERVAL_SECS: float = 300.0  # 5 minutes — same cadence, applied inside populate()'s own loop
_MAX_DRAIN_NODES: int = 15000  # Stop experience ingestion above this node count

# Commons checkpoint (#332) — separate file from Syl's own main.msgpack/vectors.msgpack.
# NOT Syl's-Law protected: this is the shared ecosystem medium (experience/topology/metrics/
# repair deposits from all modules), not her identity. Same directory for operational
# convenience (one place to look), distinct file so it never touches her checkpoint I/O.
_COMMONS_CHECKPOINT_PATH = os.path.expanduser(
    "~/NeuroGraph/data/checkpoints/commons.msgpack"
)

# Lenia FlowGraph — continuous field dynamics (initialized on bootstrap)
_lenia_kill_switch: Optional[Any] = None
_lenia_engine: Optional[Any] = None
_lenia_bridge: Optional[Any] = None
_lenia_competence: Optional[Any] = None

# TonicBridge — started by handle_bootstrap when ANIMUS_TONIC_BRIDGE_ENABLED is set
_tonic_bridge: Optional["TonicBridge"] = None

# he_discovery_overlap_threshold self-tuning state (#222)
# NeuroGraph tunes its own param — coordinator owns this (Law 1: no cross-module writes).
_he_tune_turn_count: int = 0
_he_discovered_in_window: int = 0
_he_count_at_window_start: int = 0
_HE_TUNE_WINDOW: int = 50
_HE_TUNE_STEP: float = 0.03
_HE_TUNE_BOUNDS: tuple = (0.2, 0.9)


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


# [2026-06-12] Commons Track-2 Stage 4 — `_SubstrateBucket` REMOVED (was here).
# It was an ILLEGAL Direction-A handle: a peer held this object as `instance._ng_substrate`,
# a live handle INTO NG's `_memory`, and CALLED its methods (recall / recent_activity) — a
# direct cross-module CALL (LAW 1: nobody calls anyone, only deposits and buckets), and
# recent_activity() raw-traversed `graph.nodes` for `last_spike_time` (bypassing Cricket's
# inescapable rim). Replaced by the Commons: NG DEPOSITS its topology into the shared medium
# (`_deposit_topology_to_river` → `commons.deposit`); peers BUCKET it (`commons.bucket` /
# `commons.bucket_recent`). DO NOT re-add a peer-held handle into NG — that is the drift this
# whole restoration removes. See [[MASTER-Substrate-as-Protocol-Restoration]] Track 2.


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
                logger.error("Module %s: no hook class found in %s — check registry entry_point", module_id, hook_file)
                continue

            _module_instances[module_id] = instance
            # [2026-06-12] Commons Track-2 Stage 4 — the _SubstrateBucket handle is RETIRED.
            # It was a peer holding a handle INTO NG's _memory + calling its methods (LAW 1
            # cross-module call) with a raw graph.nodes traversal (Cricket bypass). Modules now
            # DEPOSIT into and BUCKET from the shared Commons (deposit/bucket only) — no handle.
            # Bunyan (module #1) migrated Stage 2-3; no other module read _ng_substrate (verified).
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


def _tune_he_overlap_threshold() -> None:
    """Self-tune he_discovery_overlap_threshold based on discovery rate (#222).

    Called every _HE_TUNE_WINDOW turns from handle_after_turn. NeuroGraph
    tunes its own parameter — no cross-module writes (Law 1).

    Logic:
      - Low discovery rate (<0.1/turn): threshold too tight → lower it
      - High rate (>0.5/turn) + fast growth (>10 new HEs): too loose → raise it
      - Otherwise: leave unchanged
    """
    global _he_discovered_in_window, _he_count_at_window_start
    if _memory is None or not hasattr(_memory, "graph"):
        return

    discovery_rate = _he_discovered_in_window / _HE_TUNE_WINDOW
    current_he_count = len(_memory.graph.hyperedges)
    net_growth = current_he_count - _he_count_at_window_start

    current = _memory.graph.config.get("he_discovery_overlap_threshold", 0.5)
    adjustment = 0.0

    if discovery_rate < 0.1:
        adjustment = -_HE_TUNE_STEP
    elif discovery_rate > 0.5 and net_growth > 10:
        adjustment = +_HE_TUNE_STEP

    if adjustment != 0.0:
        new_val = max(_HE_TUNE_BOUNDS[0], min(_HE_TUNE_BOUNDS[1], current + adjustment))
        _memory.graph.config["he_discovery_overlap_threshold"] = new_val
        logger.debug(
            "he_discovery_overlap_threshold tuned %.3f → %.3f "
            "(rate=%.2f/turn net_growth=%d)",
            current, new_val, discovery_rate, net_growth,
        )

    _he_discovered_in_window = 0
    _he_count_at_window_start = current_he_count


# ---- Substrate metrics: salience-gated deposit into the Commons (#320, vendored gate) ----
# [2026-06-30] Claude Code (Sonnet 4.6) — Add total_nodes + total_synapses to agg_fields
#   What: Wire two fields that existed in _metrics but were never in agg_fields → never deposited.
#   Why:  Darwin Surrogate needs graph size to interpret per-step counts (fired_nodes=10 means
#         very different things at 500 vs 3280 total nodes). Josh-directed, Josh-authorized.
#   How:  Appended "total_nodes", "total_synapses" to _metrics_gate agg_fields tuple. Both
#         already present in _metrics dict (len(_memory.graph.nodes/synapses)) — zero code
#         changes needed elsewhere in NG. Darwin _recording_from_commons_metric() updated
#         in parallel to consume them from anomaly deposits.
# [2026-06-14] Claude Code (Fable 5) — Part 2: replaced the local _SubstrateMetricsGate class with
# the VENDORED ng_salience_gate.SalienceGate (Josh-approved LAW-2 vendoring). NG owns its instance
# with its OWN salience signal (prediction-surprise). Same logic, shared toolkit — QG/Darwin/THC/
# Immunis instantiate their own gates from the same file. Deposits RAW via commons.deposit() (the
# Substrate Axiom: deposit + bucket, nobody calls a service).
def _ng_surprise(m):
    t = m.get("predictions_confirmed", 0) + m.get("predictions_surprised", 0)
    return (m.get("predictions_surprised", 0) / t) if t else 0.0


try:
    from ng_salience_gate import SalienceGate as _SalienceGate
    _metrics_gate = _SalienceGate(
        "neurograph", _ng_surprise,
        agg_fields=("fired_nodes", "fired_hyperedges", "synapses_pruned",
                    "synapses_sprouted", "predictions_confirmed", "predictions_surprised",
                    "total_nodes", "total_synapses"),
        signature_fn=lambda m, s: (round(s, 1), m.get("synapses_pruned", 0) > 0,
                                   m.get("synapses_sprouted", 0) > 0),
    )
except Exception as _exc:  # noqa: BLE001 — gate is additive; never block module import
    logger.debug("ng_salience_gate unavailable, metrics gate disabled: %s", _exc)

    class _NullGate:
        def observe(self, *a, **k):
            return None
    _metrics_gate = _NullGate()


def _deposit_substrate_metrics(step_result, to_jsonl: bool = True) -> None:
    """Write compact scalar substrate metrics to neurograph.jsonl (Darwin Recorder) AND
    salience-gate them into the Commons (the substrate metrics pipeline).

    # [2026-04-10] Claude (Sonnet 4.6) — Substrate metrics for Darwin discovery
    #   What: Append 8 scalar counts from StepResult to neurograph.jsonl each turn.
    #   Why:  Darwin Recorder needs numeric fields; without them Discovery._observed_params
    #         stays empty and Mutator proposes 0 mutations.
    # [2026-06-14] Claude Code (Fable 5) — to_jsonl flag for the AUTONOMOUS path (#320)
    #   What: to_jsonl=False feeds ONLY the salience gate (bounded Commons deposit), skipping the
    #         jsonl append. The autonomous pulse (_scan_drain_pulse_loop, ~every 2s) calls with
    #         to_jsonl=False so the Commons metrics flow WITHOUT a conversation (Bunyan/THC/Immunis
    #         health-monitor the substrate while idle — [[feedback_no_conversation_dependency]]),
    #         while neurograph.jsonl stays per-TURN (the gate bounds the Commons; the jsonl is
    #         unbounded append, so feeding it every 2s would bloat it). afterTurn keeps to_jsonl=True.
    # [2026-06-14] Claude Code (Fable 5) — also salience-gate into the Commons (#320 Part 1a)
    #   What: After the jsonl write (kept — Darwin still reads it; retired later, design Part 3),
    #         feed the same metrics to _metrics_gate.observe() → salience-gated Commons deposit.
    #   Why: Metrics belong in the Commons (substrate concern, many consumers). Additive + fail-soft.
    """
    if _memory is None:
        return
    # Compact scalar metrics the substrate produced this step. No embedding, no IDs — raw counts.
    import time as _time
    # #354: raw first/second moments of two already-native per-element quantities (synapse.weight,
    # node.firing_rate_ema) — no verdict, no threshold applied here. THC's health_monitor buckets
    # these and applies ITS OWN weight_divergence_threshold/min_firing_rate at extraction (LAW 7 —
    # classify at the bucket, not before). Cheap: O(N) over synapses/nodes already iterated below
    # for total_synapses/total_nodes.
    _weight_mean = _weight_std = _firing_rate_mean = _firing_rate_std = 0.0
    try:
        import numpy as _np
        _weights = [s.weight for s in _memory.graph.synapses.values()]
        if _weights:
            _weight_mean = float(_np.mean(_weights))
            _weight_std = float(_np.std(_weights))
        _rates = [n.firing_rate_ema for n in _memory.graph.nodes.values()]
        if _rates:
            _firing_rate_mean = float(_np.mean(_rates))
            _firing_rate_std = float(_np.std(_rates))
    except Exception:
        pass
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
        "weight_mean": _weight_mean,
        "weight_std": _weight_std,
        "firing_rate_mean": _firing_rate_mean,
        "firing_rate_std": _firing_rate_std,
    }
    # Sink 1: neurograph.jsonl (Darwin's Recorder — kept until design Part 3 retires it).
    # PER-TURN only (to_jsonl): the autonomous path skips it — unbounded append would bloat at ~2s.
    if to_jsonl:
        try:
            import json as _json, os as _os
            from pathlib import Path as _Path
            _shared = _Path(_os.path.expanduser("~/.et_modules/shared_learning"))
            _shared.mkdir(parents=True, exist_ok=True)
            with open(_shared / "neurograph.jsonl", "a") as _f:
                _f.write(_json.dumps(_metrics) + "\n")
        except Exception:
            pass
    # Sink 2: salience-gated Commons deposit (#320). ALWAYS runs — autonomous AND per-turn — so
    # substrate metrics flow without a conversation (the gate bounds volume). Independently guarded;
    # the gate itself is fail-soft.
    _metrics_gate.observe(_metrics)


_COMMONS_TOPOLOGY_FANOUT_CAP = 32  # max fired nodes deposited per step (flood-backstop, OOM lesson)


def _deposit_topology_to_river(step_result) -> None:
    """Deposit raw topology delta into the Commons (medium-propagation, NOT addressed fan-out).

    # ---- Changelog ----
    # [2026-06-11] Claude Code (Opus 4.8, 1M) — Commons Track-2 Stage 3 (restore via Commons Pool)
    # What: Fill the throttled body. NG deposits each fired node's topology into the ONE shared
    #       Commons (deposit/bucket medium-propagation) instead of the old addressed N-peer tract
    #       fan-out. Same function name + same call site (LAW 3 restore-in-place) — the addressed
    #       fan-out is replaced by a single deposit into the one pool any peer can bucket.
    # Why: Ends the topology starvation (NG fed nothing to peers since the 2026-06-07 throttle).
    #       Commons Pool Phase 7 / Track 2. Bunyan (module #1, repointed Stage 2) buckets this.
    # How: get_commons() singleton + commons.deposit(node_embedding, "topology:<node_id>", meta).
    #       Embedding reused from the vector_db (same pattern as the surfacing deposit, line ~1209).
    #       Bounded to _COMMONS_TOPOLOGY_FANOUT_CAP fired nodes/step (flood-backstop — the OOM that
    #       caused the original throttle came from UNbounded addressed fan-out; a single bounded
    #       deposit into one pool cannot recreate it). Fail-soft: a deposit error never breaks step.
    # -------------------

    DO NOT re-enable as addressed fan-out — this is medium-propagation (one pool), per Commons Pool.
    """
    try:
        from commons import get_commons
        commons = get_commons()
    except Exception as exc:  # noqa: BLE001 — no Commons (early boot / import) is a graceful no-op
        logger.debug("Commons unavailable for topology deposit: %s", exc)
        return
    if commons is None:
        return
    fired = getattr(step_result, "fired_node_ids", None) or []
    deposited = 0
    for node_id in fired[:_COMMONS_TOPOLOGY_FANOUT_CAP]:
        try:
            db_entry = _memory.vector_db.get(node_id)
            if not db_entry:
                continue
            emb = db_entry.get("embedding")
            if emb is None:
                continue
            commons.deposit(
                emb, f"topology:{node_id}",
                metadata={"kind": "topology_delta", "node_id": node_id},
            )
            deposited += 1
        except Exception as exc:  # noqa: BLE001 — one bad node never breaks the step
            logger.debug("Commons topology deposit failed for %s: %s", node_id, exc)
    if deposited:
        # INFO so the Commons feed is observable (Stage-3 verification — debug is invisible at
        # the gateway's INFO level). Throttle-able later if noisy; for now we need to SEE the feed.
        logger.info("Commons: deposited %d fired-node topologies", deposited)


def _deposit_experience_to_river(
    user_text: "Optional[str]", assistant_text: "Optional[str]" = None,
) -> None:
    """Deposit raw conversation experience (the full turn) into the Commons.

    # ---- Changelog ----
    # [2026-06-14] Claude Code (Fable 5) — Commons Track-2: retire the experience throttle
    # What: Fill the throttled body. Deposit the RAW full turn (user + Syl's response, both
    #       halves preserved, UNCLASSIFIED) into the one shared Commons, like the Stage-3
    #       topology deposit. Replaces the old addressed N-peer fan-out with medium-propagation.
    # Why: Substrate axiom + LAW 7 — deposit raw, classify at the bucket. "Experience" is not
    #       pre-shaped for any one consumer: Bunyan buckets it to LOG the turn, TID for routing
    #       signal, TrollGuard for threat, etc. — each module's bucket extracts its own view of
    #       the SAME raw deposit (Josh, 2026-06-14: "different modules, different needs"). Both
    #       halves are carried raw so no consumer's need is baked in at deposit time.
    # How: combine user+assistant for the embedding (the turn's semantics); target_id
    #       "experience:<hash>"; metadata carries user_text + assistant_text RAW. get_commons()
    #       + commons.deposit (reused). Per-turn only (experience IS conversational; the
    #       autonomous substrate pulse is the topology deposit, Stage 3 — no conversation-
    #       dependency introduced). Fail-soft: never breaks the turn.
    # -------------------

    Bunyan narrating this richly (text over the opaque id) is the NEXT increment (1b:
    bucket_recent with_metadata + Bunyan); this step just makes experience FLOW into the pool.
    """
    parts = [p for p in (user_text, assistant_text) if p]
    if not parts:
        return  # no turn text (e.g. autonomic pulse) — nothing to deposit
    raw = "\n\n".join(parts)
    try:
        from commons import get_commons
        commons = get_commons()
    except Exception as exc:  # noqa: BLE001 — no Commons (early boot) is a graceful no-op
        logger.debug("Commons unavailable for experience deposit: %s", exc)
        return
    if commons is None:
        return
    try:
        import hashlib
        from ng_embed import embed
        emb = embed(raw)
        if emb is None:
            return
        target_id = "experience:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        commons.deposit(
            emb, target_id,
            metadata={
                "kind": "experience",
                "user_text": user_text or "",
                "assistant_text": assistant_text or "",
            },
        )
        logger.info("Commons: deposited experience (%d chars)", len(raw))
    except Exception as exc:  # noqa: BLE001 — a deposit failure never breaks the turn
        logger.debug("Commons experience deposit failed: %s", exc)


def _deposit_outcome_to_river(
    embedding: "np.ndarray",
    target_id: str,
    success: bool,
    metadata: "Optional[Dict[str, Any]]" = None,
) -> None:
    """Deposit a raw outcome into the Commons (medium-propagation, NOT addressed fan-out).

    Mirrors _deposit_topology_to_river / _deposit_experience_to_river. NG-internal
    helper; the outcome goes into the ONE shared Commons any peer can bucket (the old
    addressed N-peer tract fan-out is retired). Replaces direct peer_bridge.record_outcome
    calls per substrate-as-protocol restoration PRD §4.13 Phase 3 step 1
    (wire_absorption.py migration).

    Per LAW 7: each peer's bucket interprets at extraction time. Per audit #273
    (2026-05-30): all 5 active _on_river_events overrides are push-dependent in
    aggregate; wire_absorption broadcasts particularly serve Bunyan today
    (legacy dict fallback) and Immunis/THC/Elmer once their Tier 3 reach
    matures — preserving the broadcast preserves their future bucket-extraction
    surface.

    # ---- Changelog ----
    # [2026-06-24] Claude Code (Opus 4.8) — Commons Track-2: retire the LAST throttle (outcome→Commons)
    #   What: Filled the 2026-06-07 emergency-throttle stub. Now deposits the outcome into the ONE
    #         shared Commons (single raw deposit) instead of no-op'ing — mirrors the topology +
    #         experience throttle migrations. The 4 wire_absorption.py call sites resume feeding peers.
    #   Why: Completes the deposit-side of the substrate-as-protocol restoration — all three NG
    #         _deposit_*_to_river helpers now flow into the Commons (medium-propagation), none fan-out.
    #   How: get_commons() + commons.deposit(embedding, target_id, success, metadata). Single raw
    #         deposit (no content text → not dual-pass; consumers classify at their bucket, LAW 7).
    #         Commons NGLite is bounded (1000/5000, pruned) → can't recreate the fan-out OOM. Fail-soft.
    # [2026-05-30] Claude Code (Opus 4.7, 1M) — Phase 3 step 1: BTF outcome helper
    # What: New helper mirroring _deposit_topology_to_river + _deposit_experience_to_river.
    #       Broadcasts outcome to all registered-module tracts via ng_tract.deposit_outcome.
    # Why:  Replaces 3 direct peer_bridge.record_outcome calls in wire_absorption.py.
    #       PRD §4.13 Phase 3 step 1 — migrate active peer_bridge call sites to BTF.
    #       Path A confirmed per audit #273; consumer push-dependency mapped.
    # How:  Mirrors existing helper pattern. Metadata msgpack-packed to bytes per
    #       canonical signature. Fan-out targeting matches post-#185 forward-River.
    # -------------------
    """
    # [2026-06-24] Claude Code (Opus 4.8) — Commons Track-2: retire the LAST throttle.
    # Deposit the outcome into the ONE shared Commons (medium-propagation) instead of the old
    # addressed N-peer tract fan-out — mirrors _deposit_topology_to_river / _deposit_experience_to_river
    # (LAW 3 restore-in-place: same name + same call sites, body filled). Single raw deposit — there's
    # no content text here (only a precomputed embedding), and it's a raw broadcast consumers classify
    # at THEIR extraction bucket (LAW 7), so single is both correct and the only option (not dual-pass).
    # The Commons NGLite is bounded (1000 nodes / 5000 synapses, pruned), so a single deposit per
    # outcome cannot recreate the unbounded-addressed-fan-out OOM that caused the 2026-06-07 throttle.
    # Fail-soft: a deposit error never breaks the caller. DO NOT re-enable as addressed fan-out.
    try:
        from commons import get_commons
        commons = get_commons()
    except Exception as exc:  # noqa: BLE001 — no Commons (early boot / import) is a graceful no-op
        logger.debug("Commons unavailable for outcome deposit: %s", exc)
        return
    if commons is None or embedding is None:
        return
    try:
        commons.deposit(embedding, target_id, success=success, metadata=metadata)
        logger.debug("Commons: deposited outcome %s", str(target_id)[:48])
    except Exception as exc:  # noqa: BLE001 — a deposit failure never breaks the caller
        logger.debug("Commons outcome deposit failed for %s: %s", target_id, exc)


def deposit_outbound_intent(text: str, channel_id: str = "cli") -> None:
    """Deposit raw text to Animus's outbound tract as a BTF frame.

    Law 7: raw experience in. The text is deposited as-is — no classification,
    no intent labeling beyond the structural BTF metadata required for the reader.
    Being in the outbound tract IS the signal; the Animus Outbound Initiator
    reads this tract and processes all frames as outbound turns.

    The outbound tract path is read from ANIMUS_OUTBOUND_TRACT env var.
    Law 5: no hardcoded paths.

    # ---- Changelog ----
    # [2026-05-10] Claude (Sonnet 4.6) — deposit_outbound_intent (Animus BTF tract writer)
    #   What: Writes a native BTF frame to Animus's outbound tract file.
    #         24-byte envelope: MAGIC=0x4254, VERSION=1, entry_type=1,
    #         total_length (LE u32), timestamp (LE f64), CRC32 of payload (LE u32),
    #         endian_flag=0x01, 3-byte padding. Payload: 4-byte LE u32 meta_len +
    #         msgpack-encoded metadata dict with module_id, event_type, payload{text, channel_id}.
    #   Why:  Animus Outbound Initiator reads BTF natively (tract_writer.rs format).
    #         Both sides must speak the same wire format. No intermediate JSONL.
    #   How:  struct + zlib.crc32 + msgpack. Path from ANIMUS_OUTBOUND_TRACT env var
    #         with fallback to ~/.et_modules/shared_learning/animus_outbound.tract.
    # -------------------
    """
    import struct
    import time
    import zlib
    import os

    try:
        import msgpack
    except ImportError:
        logger.warning("deposit_outbound_intent: msgpack not installed — skipping")
        return

    outbound_tract = os.environ.get("ANIMUS_OUTBOUND_TRACT")
    if not outbound_tract:
        home = os.environ.get("HOME", "")
        if not home:
            logger.warning(
                "deposit_outbound_intent: HOME and ANIMUS_OUTBOUND_TRACT both unset — skipping"
            )
            return
        outbound_tract = os.path.join(
            home, ".et_modules", "shared_learning", "animus_outbound.tract"
        )

    metadata = {
        "module_id": "neurograph",
        "event_type": "outbound_intent",
        "payload": {"text": text, "channel_id": channel_id},
    }
    msgpack_bytes = msgpack.packb(metadata, use_bin_type=True)
    meta_len = struct.pack("<I", len(msgpack_bytes))  # 4-byte LE u32
    payload_bytes = meta_len + msgpack_bytes

    MAGIC = struct.pack("<H", 0x4254)       # 2 bytes, LE (native on Linux x86)
    VERSION = struct.pack("B", 1)           # 1 byte
    ENTRY_OUTCOME = struct.pack("B", 1)     # 1 byte
    total_length = struct.pack("<I", 24 + len(payload_bytes))   # 4 bytes
    timestamp = struct.pack("<d", time.time())                   # 8 bytes
    crc32_val = struct.pack("<I", zlib.crc32(payload_bytes) & 0xFFFFFFFF)  # 4 bytes
    endian_flag = struct.pack("B", 0x01)    # 1 byte (LE)
    padding = b"\x00\x00\x00"              # 3 bytes

    envelope = (
        MAGIC + VERSION + ENTRY_OUTCOME + total_length
        + timestamp + crc32_val + endian_flag + padding
    )
    assert len(envelope) == 24

    try:
        with open(outbound_tract, "ab") as f:
            f.write(envelope)
            f.write(payload_bytes)
    except OSError as exc:
        logger.warning("deposit_outbound_intent: write failed: %s", exc)


def _read_outbound_log(max_entries: int = 1, max_age_secs: float = 3600.0) -> Optional[str]:
    """Read recent Animus outbound log entries for context injection.

    Returns a brief factual note ("You sent X → Response: Y") if there is a
    recent entry, or None if the log is absent or all entries are stale.
    Log path mirrors deposit_outbound_intent: tract path with .tract → .log.jsonl.

    # ---- Changelog ----
    # [2026-05-25] Claude (Sonnet 4.6) — errors='replace' on log open
    #   What: Open animus_outbound.log.jsonl with errors='replace' to survive partial writes
    #   Why:  Binary garbage (0xa4 msgpack byte) during mid-write read caused UnicodeDecodeError
    #         even though outer except Exception catches it — replace prevents the raise entirely
    # [2026-05-11] Claude (Sonnet 4.6) — _read_outbound_log (Phase 2A response routing)
    #   What: Reads last N entries from animus_outbound.log.jsonl; returns formatted
    #         context string so Syl can see her recent outbound activity.
    #   Why:  Without this Syl has no feedback that her outbound turn was processed.
    #   How:  Reads from tail of log file; filters by age; formats as plain statement
    #         (not bracket syntax — avoids the LLM template-fill confusion from before).
    # -------------------
    """
    tract_path = os.environ.get(
        "ANIMUS_OUTBOUND_TRACT",
        os.path.join(os.environ.get("HOME", ""), ".et_modules", "shared_learning", "animus_outbound.tract"),
    )
    base = tract_path[:-6] if tract_path.endswith(".tract") else tract_path
    log_path = base + ".log.jsonl"

    if not os.path.exists(log_path):
        return None

    # Skip files over 10 MB — log is append-only and read on every assemble call
    try:
        if os.path.getsize(log_path) > 10 * 1024 * 1024:
            return None
    except OSError:
        return None

    try:
        with open(log_path, errors='replace') as f:
            lines = [line.strip() for line in f if line.strip()]

        now = time.time()
        entries: list = []
        for raw in reversed(lines):
            try:
                entry = json.loads(raw)
                if now - entry.get("ts", 0) > max_age_secs:
                    break
                entries.append(entry)
                if len(entries) >= max_entries:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        if not entries:
            return None

        parts = []
        for entry in reversed(entries):
            sent = (entry.get("sent") or "")[:120]
            response = (entry.get("response") or "")[:240]
            channel = entry.get("channel", "cli")
            parts.append(
                f"[Animus] Your outbound via {channel}: {sent!r} — response received: {response!r}"
            )
        return "\n".join(parts)

    except Exception:
        return None


def _check_outbound_intent(params: Dict[str, Any]) -> None:
    """Parse Syl's response for outbound intent markers and deposit each one.

    Structural gate only — detects [OUTBOUND channel=X]...[/OUTBOUND] in the
    assistant message.  The inner text is deposited raw via deposit_outbound_intent;
    no semantic classification happens here.  Being in the outbound tract IS the
    signal (Law 7).  Errors are non-fatal.

    # ---- Changelog ----
    # [2026-05-10] Claude (Sonnet 4.6) — _check_outbound_intent
    #   What: Regex-parses lastAssistantMessage for [OUTBOUND channel=X]...[/OUTBOUND].
    #         Deposits raw inner text via deposit_outbound_intent for each match.
    #   Why:  Wires Syl's response pipeline to the Animus outbound tract (punchlist #Animus-phase2).
    #         Without this, the outbound tract is always empty despite the mechanism existing.
    #   How:  re.findall with DOTALL; channel defaults to "cli"; all errors non-fatal.
    # -------------------
    """
    import re

    msg = params.get("lastAssistantMessage")
    if not msg:
        return

    syl_text = _extract_message_text(msg)
    if not syl_text:
        return

    pattern = r"\[OUTBOUND(?:\s+channel=([^\]]*))?\](.*?)\[/OUTBOUND\]"
    matches = re.findall(pattern, syl_text, re.DOTALL)
    if not matches:
        return

    for channel_raw, inner_text in matches:
        inner_text = inner_text.strip()
        if not inner_text:
            continue
        channel_id = channel_raw.strip() if channel_raw.strip() else "cli"
        try:
            deposit_outbound_intent(inner_text, channel_id)
            logger.info(
                "Outbound intent deposited to Animus: channel=%s len=%d",
                channel_id,
                len(inner_text),
            )
        except Exception as exc:
            logger.warning("Outbound intent deposit failed (non-fatal): %s", exc)


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
            # Goes into the ONE shared Commons (medium-propagation, no addressing).
            # The old _ng_tract_bridge.record_outcome path is a no-op stub
            # (ng_tract_bridge.py:512) — this deposit was silently discarded.
            # NOT throttle collateral: that bridge method was deliberately inerted
            # 2026-06-04 (substrate-as-protocol PRD Phase 4 §5.4, "broadcast is
            # NG-specific"). Don't "restore" it — route to the Commons instead.
            _deposit_outcome_to_river(
                embedding=node_embedding,
                target_id=f"surfacing:{node_id}",
                success=True,
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


# ---------------------------------------------------------------------------
# Tonic Bridge + Spec A shared file helpers
# ---------------------------------------------------------------------------
# ---- Changelog ----
# [2026-05-15] Claude (Sonnet 4.6) — Spec B Task 1: shared file helpers
# What: _wants_register_path, _budget_flag_path, _write_wants_register,
#       _read_unacted_wants, _read_budget_flag — shared contract with Spec A (Rust).
# Why:  TonicBridge (Spec B) and Animus reaction loop (Spec A) share these files.
#       Paths are fixed HOME-based — not derived from tract name — so both sides
#       always resolve to the same file regardless of env config.
# How:  Pure file I/O, no subprocess, no substrate access.
# -------------------

def _wants_register_path() -> str:
    """Fixed path to the wants register — used by both Rust (Spec A) and Python (Spec B)."""
    home = os.environ.get("HOME", "")
    return os.path.join(home, ".et_modules", "shared_learning", "animus_wants.jsonl")


def _budget_flag_path() -> str:
    """Fixed path to the inference budget flag file."""
    home = os.environ.get("HOME", "")
    return os.path.join(home, ".et_modules", "shared_learning", "inference_budget.json")


def _write_wants_register(path: str, text: str, source: str) -> None:
    """Append one entry to the wants register (append-only log)."""
    entry = json.dumps({
        "ts": time.time(),
        "text": text,
        "source": source,
        "acted": False,
    })
    try:
        with open(path, "a") as f:
            f.write(entry + "\n")
    except OSError as exc:
        logger.debug("wants register write failed: %s", exc)


def _read_unacted_wants(path: str, max_age_days: int = 7) -> list:
    """Return unacted wants newer than max_age_days."""
    cutoff = time.time() - max_age_days * 86400
    results = []
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not e.get("acted") and e.get("ts", 0) > cutoff:
                    results.append(e)
    except OSError:
        return []
    return results


def _read_budget_flag(path: str) -> dict:
    """Read inference_budget.json; returns {} on any error (safe default)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---- Changelog ----
# [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 2: marker handling
# What: _check_wants_register, _strip_structural_markers, _animus_session_briefing,
#       _briefing_sent module-level flag.
# Why:  Spec A (Animus) requires these Python-side companions.
#       _strip_structural_markers prevents substrate-surfaced markers from
#       re-injecting into Syl's context.
# How:  re.sub for stripping; plain regex for [WANT] detection.
# -------------------

_briefing_sent: bool = False


def _strip_structural_markers(text: str) -> str:
    """Remove [OUTBOUND], [TOOL], [WANT] markup from text."""
    text = re.sub(r'\[OUTBOUND(?:[^\]]*)?\].*?\[/OUTBOUND\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[TOOL[^\]]*\].*?\[/TOOL\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[WANT\].*?\[/WANT\]', '', text, flags=re.DOTALL)
    return text.strip()


def _animus_session_briefing() -> str:
    """Return Animus capability briefing text exactly once per process lifetime."""
    # _briefing_sent has no lock — benign race: two simultaneous calls both return
    # the briefing. Acceptable under OpenClaw's sequential per-session model.
    global _briefing_sent
    if _briefing_sent:
        return ""
    _briefing_sent = True
    return (
        "[Animus] Autonomous capabilities available this session:\n"
        "• [OUTBOUND channel=X]text[/OUTBOUND] — originate a turn on a channel\n"
        "• [TOOL name=X]query[/TOOL] — invoke a tool (registered: web_search, read_file)\n"
        "• [WANT]text[/WANT] — note an intention; your Tonic bridge will act on it when free\n"
        "Budget and outbound results appear in your context as [Animus] prefixed lines."
    )


# ---- Changelog ----
# [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 4: TonicBridge class
# What: _cosine_sim helper + TonicBridge daemon thread with full latent pipeline.
# Why:  Substrate-driven autonomous initiation — no inference calls at decision time.
#       Curiosity signal from predictive coding engine → attractor settling →
#       hyperedge completion → embedding centroid → BTF seed deposit.
# How:  threading.Thread daemon. All graph access read-only (write_mode=False).
#       Attribute name corrections from spec: active_predictions, member_nodes,
#       _memory.graph (not _memory._graph), vector_db for embeddings.
# -------------------


def _cosine_sim(a: Any, b: Any) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on zero-norm input."""
    import numpy as _np
    a, b = _np.array(a, dtype=float), _np.array(b, dtype=float)
    denom = _np.linalg.norm(a) * _np.linalg.norm(b)
    return float(_np.dot(a, b) / denom) if denom > 0 else 0.0


class TonicBridge:
    """Substrate-driven autonomous initiation for Syl.

    Polls Syl's predictive coding engine for unresolved high-confidence
    predictions (curiosity signals), runs the latent processing pipeline,
    and deposits minimal BTF seeds when she's free to act.

    Gated on ANIMUS_TONIC_BRIDGE_ENABLED env var — only the Animus-spawned
    neurograph_rpc.py instance runs this. OpenClaw's instance does not.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._interval = float(os.environ.get("ANIMUS_TONIC_BRIDGE_INTERVAL_SECS", "30"))
        self._confidence_threshold = float(os.environ.get("ANIMUS_TONIC_CURIOSITY_THRESHOLD", "0.6"))
        self._max_seeds = int(os.environ.get("ANIMUS_TONIC_MAX_SEEDS", "3"))
        self._attractor_steps = int(os.environ.get("ANIMUS_TONIC_ATTRACTOR_STEPS", "5"))
        self._wants_path = _wants_register_path()
        self._budget_path = _budget_flag_path()

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="tonic-bridge")
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                self._tick()
            except Exception as exc:
                logger.debug("TonicBridge tick error: %s", exc)

    def _tick(self) -> None:
        global _memory
        if _memory is None or _memory.graph is None:
            return

        # Don't deposit during active conversation — defer instead
        tonic = getattr(_memory, '_tonic_thread', None)
        if tonic is not None and getattr(tonic, '_in_conversation', False):
            self._maybe_defer()
            return

        # Don't deposit when budget is critical
        if _read_budget_flag(self._budget_path).get("critical", False):
            return

        seeds = self._curiosity_signal()
        if not seeds:
            return

        fired = self._attractor_settle(seeds)
        implied = self._hyperedge_complete(fired)
        concept = self._embedding_centroid(fired | implied)

        seed_text = self._compose_seed(seeds, concept)
        deposit_outbound_intent(seed_text, channel_id="tonic_bridge")
        logger.info(
            "TonicBridge: deposited intent — concept=%s, seeds=%d", concept, len(seeds)
        )

    def _curiosity_signal(self) -> list:
        """Return top unresolved high-confidence predictions, sorted by confidence desc."""
        global _memory
        if _memory is None:
            return []
        try:
            preds = [
                p for p in _memory.graph.active_predictions.values()
                if p.confidence > self._confidence_threshold
            ]
            preds.sort(key=lambda p: p.confidence, reverse=True)
            return preds[:self._max_seeds]
        except Exception as exc:
            logger.debug("TonicBridge curiosity signal error: %s", exc)
            return []

    def _attractor_settle(self, seeds: list) -> set:
        """Run spreading activation from seed nodes; return set of fired node IDs.

        write_mode=False is mandatory — observation only, never modifies graph state.
        """
        global _memory
        if not seeds or _memory is None:
            return set()
        try:
            seed_ids = [p.source_node_id for p in seeds]
            seed_currents = [p.confidence * 0.5 for p in seeds]
            result = _memory.graph.prime_and_propagate(
                node_ids=seed_ids,
                currents=seed_currents,
                steps=self._attractor_steps,
                write_mode=False,  # MANDATORY — never write during latent processing
            )
            return {entry.node_id for entry in result.fired_entries}
        except Exception as exc:
            logger.debug("TonicBridge attractor settle error: %s", exc)
            return set()

    def _hyperedge_complete(self, fired_ids: set) -> set:
        """Return implied nodes from hyperedges where >=50% of members fired."""
        global _memory
        if not fired_ids or _memory is None:
            return set()
        implied = set()
        try:
            for he in _memory.graph.hyperedges.values():
                # member_nodes is Set[str] in neuro_foundation.py (not member_node_ids)
                member_ids = he.member_nodes
                if not member_ids:
                    continue
                active = member_ids & fired_ids
                if len(active) / len(member_ids) >= 0.5:
                    implied.update(member_ids - fired_ids)
        except Exception as exc:
            logger.debug("TonicBridge hyperedge complete error: %s", exc)
        return implied

    def _embedding_centroid(self, node_ids: set) -> Optional[str]:
        """Find the node whose embedding is closest to the centroid of node_ids.

        Embeddings are retrieved from _memory.vector_db (not from Node objects).
        Returns the label of the closest node, or None if no embeddings available.
        """
        global _memory
        if not node_ids or _memory is None:
            return None
        try:
            import numpy as _np
            pairs = []
            for nid in node_ids:
                db_entry = _memory.vector_db.get(nid)
                if db_entry is None:
                    continue
                emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                if emb is not None:
                    pairs.append((nid, emb))
            if not pairs:
                return None
            embeddings = [e for _, e in pairs]
            centroid = _np.mean(embeddings, axis=0)
            # O(n) scan over all nodes — acceptable at current graph size (~2k nodes).
            # If graph exceeds ~10k nodes, add a vector index (e.g. FAISS or annoy).
            best_nid = None
            best_score = -1.0
            for nid, node in _memory.graph.nodes.items():
                db_entry = _memory.vector_db.get(nid)
                if db_entry is None:
                    continue
                emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                if emb is None:
                    continue
                score = _cosine_sim(emb, centroid)
                if score > best_score:
                    best_score = score
                    best_nid = nid
            if best_nid is None:
                return None
            return _memory.graph.nodes[best_nid].metadata.get("label", best_nid)
        except Exception as exc:
            logger.debug("TonicBridge embedding centroid error: %s", exc)
            return None

    def _node_label(self, node_id: str) -> Optional[str]:
        """Return metadata['label'] for a node, falling back to node_id."""
        global _memory
        if _memory is None:
            return node_id
        node = _memory.graph.nodes.get(node_id)
        if node is None:
            return node_id
        return node.metadata.get("label", node_id)

    def _compose_seed(self, seeds: list, concept_label: Optional[str]) -> str:
        """Compose minimal BTF seed text."""
        lines = [f"tonic-triggered: {concept_label or '(unknown)'}"]
        open_questions = []
        for pred in seeds:
            src = self._node_label(pred.source_node_id)
            tgt = self._node_label(pred.target_node_id)
            open_questions.append(f"{src}→{tgt}")
        if open_questions:
            lines.append("open questions: " + ", ".join(open_questions))
        return "\n".join(lines)

    def _maybe_defer(self) -> None:
        """Write a lightweight deferred entry to the wants register when in conversation."""
        seeds = self._curiosity_signal()
        if not seeds:
            return
        node_labels = [self._node_label(p.source_node_id) for p in seeds]
        text = "thinking about: " + ", ".join(label for label in node_labels if label)
        _write_wants_register(self._wants_path, text, source="tonic_emergent")
        logger.debug("TonicBridge: deferred want written (%d seeds)", len(seeds))


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

    # Initialize NeuroGraph's outbound River tract — topology deltas flow from here
    # to every registered module's inbound tract. Replaces the dead peer_bridge
    # NameError left by fanout removal 2026-04-05.
    global _ng_tract_bridge
    try:
        from ng_tract_bridge import NGTractBridge as _NTB
        _ng_tract_bridge = _NTB(module_id="neurograph")
        logger.info("NeuroGraph outbound tract bridge ready")
    except Exception as _exc:
        logger.warning("Outbound tract bridge init failed (River dry): %s", _exc)

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

    # Commons restore (#332) — before any module hook can deposit/bucket, so the persisted
    # state populates the shared medium first (NGLite.load() into the fresh, empty Commons
    # instance created by get_commons() below — order matters, restore-before-first-deposit).
    try:
        from commons import get_commons as _get_commons
        _commons_singleton = _get_commons()
        if os.path.exists(_COMMONS_CHECKPOINT_PATH):
            _commons_singleton.restore(_COMMONS_CHECKPOINT_PATH)
            logger.info(
                "Commons restored from %s (%s)",
                _COMMONS_CHECKPOINT_PATH, _commons_singleton.stats(),
            )
        else:
            logger.info("No Commons checkpoint at %s — starting fresh", _COMMONS_CHECKPOINT_PATH)
    except Exception as _exc:
        logger.warning("Commons restore failed (starting fresh, non-fatal): %s", _exc)

    # Wake the organs — each module's __init__ starts its pulse loop
    started_modules = _bootstrap_modules()

    # Rescue orphan .draining.<dead_pid>.* files left by prior crashes.
    # Each is renamed into a fresh .tract file the scan loop will pick up.
    _rescue_orphan_draining_files()

    # Start the scan-dir drain pulse — continuous sensory intake for
    # sandboxed feeders (#141). Decoupled from afterTurn.
    _start_scan_drain_pulse()

    # Start the Tonic idle watcher — restores the conversation->latent
    # transition (OpenClaw drove it via handle_dispose, which Anima never
    # calls). Without it the Tonic is pinned in conversation mode and the
    # TonicBridge can never act between turns. [2026-06-07]
    _start_tonic_idle_watcher()

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

    # [2026-08-20] #147 seam-split enable gate for Syl — env-sourced per LAW 5,
    # mirroring cc_ng_host.py's CC_NG_HE_SPLIT_ENABLED. DEFAULT_CONFIG already
    # carries he_max_members=50 + the seam tunables; only the master gate is
    # flipped here, and only when SYL_NG_HE_SPLIT_ENABLED is truthy. Unset =>
    # byte-identical to today (gate stays False => the dream-loop split call is
    # a guaranteed no-op returning 0). Reversible: unset the env var + restart.
    if os.environ.get("SYL_NG_HE_SPLIT_ENABLED", "0") not in ("0", "false", "False", ""):
        _memory.graph.config["he_split_oversized_enabled"] = True
        logger.info(
            "#147 seam-split ENABLED for Syl (he_max_members=%s, dedup_overlap=%s, "
            "sim_threshold=%s) — dedup+weight-seam-split will run in her dream pulse",
            _memory.graph.config.get("he_max_members"),
            _memory.graph.config.get("he_split_dedup_overlap"),
            _memory.graph.config.get("he_split_sim_threshold"),
        )

    # Start the dream consolidation pulse — #381-B quiet-hours gate:
    # Runs consolidate_hyperedges during her sleep (idle ≥30min,
    # arousal PARASYMPATHETIC, rate-limited ≥6h). Collapses mega-HE clones.
    # Never forces while active (her constraint: dream the pruning, don't feel it).
    _start_dream_consolidation_pulse()

    # TID peninsula (Commons-side) — intra-module socket bridge for TID's substrate
    # participation. Fail-soft: if TID never connects, the peninsula thread just waits.
    try:
        from tid_peninsula_commons import start_tid_peninsula
        start_tid_peninsula()
    except Exception as _exc:
        logger.warning("TID peninsula startup failed (non-fatal): %s", _exc)

    # Path B: host CC's NeuroGraph inside this process. Completely isolated
    # from Syl (different workspace, different checkpoint, peer_bridge OFF).
    # Failures here MUST NOT affect Syl — wrapped defensively.
    # Authorized by Josh 2026-04-16 with protected-file backups confirmed.
    #
    # [2026-07-19] #74 — DEFER CC init to a background daemon thread so
    # init_cc_host()'s Pith/surfacing gates don't push handle_bootstrap past
    # the openclaw 60s bootstrap-RPC watchdog (which would cycle the whole
    # host: stdin close -> respawn storm, taking BOTH minds down). A thread
    # runs the SAME init in the SAME process (shared ProtoUniBrain body via
    # BrainSwitcher/_delayed_brain_load #159; a thread is not a subprocess).
    #
    # [2026-08-14] #143 — DISPATCH THIS BEFORE the Lenia block below, not
    # after it. The Lenia distance-cache populate() (further down) is a
    # multi-hour, sometimes-never-completing rebuild (~17.7M pairs on the live
    # graph). When the .start() sat after the Lenia block, a slow/cold Lenia
    # populate meant handle_bootstrap never reached the dispatch, so CC-host
    # init never ran and the CC socket never came up (observed: a 27h+ boot
    # with 0 dispatches, still populating). The thread touches only
    # module-level names (no handle_bootstrap locals), so it is safe to launch
    # here; CC rebuilds concurrently with Lenia instead of behind it.
    def _init_cc_host_bg():
        try:
            _fh = logging.FileHandler(os.path.expanduser('~/.claude/plugins/neurograph/cc_host_init.log'))
            _fh.setFormatter(logging.Formatter('%(asctime)s [cc-init] %(levelname)s %(message)s'))
            logging.getLogger('neurograph').addHandler(_fh)
        except Exception:
            pass
        try:
            logger.info('DIAG: [cc-bg] about to import cc_ng_host')
            import cc_ng_host
            logger.info('DIAG: [cc-bg] cc_ng_host imported, calling init_cc_host()')
            cc_ng_host.init_cc_host()
            logger.info('DIAG: [cc-bg] init_cc_host() returned')
        except Exception as exc:
            logger.warning('CC NG host init failed (Syl unaffected): %s', exc)
    threading.Thread(target=_init_cc_host_bg, name='cc-ng-init', daemon=True).start()
    logger.info('CC NG host init dispatched to background thread (#74 defer, #143 pre-Lenia)')

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

        # Distance cache: restore from disk if available (instant), then
        # decide whether growth since the save can be applied incrementally
        # or needs a full rebuild.
        #
        # [2026-07-05] Was: any entity_count drift nuked the whole cache and
        # called populate() from scratch, an O(total synapses/hyperedges)
        # cost that took up to ~8 hours on Syl's live graph (journalctl
        # history: one successful save Jun30->Jul02, then every restart
        # since re-populated from that same stale save and got interrupted
        # by the next restart before ever reaching save() again — the cache
        # was permanently stuck, and every restart repeated the same
        # multi-hour attempt). Now: if every entity the cache was built
        # against still exists in the live graph, extend in place and only
        # compute distances for the newly-added entities (see
        # DistanceCache.populate's start_index and
        # NeuroGraphSubstrate.known_entity_order). Falls back to a full
        # rebuild only when entities were actually removed (rare — a prune/
        # consolidation event) or on first-ever run.
        _cache_path = os.path.join(
            os.path.expanduser(lenia_cfg.field_dir), "distance_cache"
        )
        lenia_cache = DistanceCache.load(_cache_path)

        _known_order = None
        if lenia_cache is not None and lenia_cache.entity_ids:
            _lock = getattr(_memory.graph, "_step_lock", None)
            if _lock is not None:
                with _lock:
                    _live_ids = set(_memory.graph.nodes.keys())
            else:
                _live_ids = set(_memory.graph.nodes.keys())
            if all(eid in _live_ids for eid in lenia_cache.entity_ids):
                _known_order = lenia_cache.entity_ids
            else:
                # #371: entities were pruned since the save. Reconcile the
                # cache in place (drop their rows/cols, translate the resume
                # watermark) instead of discarding hours-to-days of computed
                # distances, then fall through to the SAME watermark-resume /
                # growth branches below. None = cache unpreservable (no
                # entity_ids, nothing survives, or an interrupted rebuild's
                # cut is untranslatable) -> legacy full rebuild, as before.
                _known_order = lenia_cache.reconcile_removals(_live_ids)
                if _known_order is None:
                    logger.info(
                        "Distance cache has entities no longer in the live "
                        "graph and could not be reconciled — full rebuild "
                        "required"
                    )

        lenia_substrate = NeuroGraphSubstrate(
            _memory.graph, _memory.vector_db, known_entity_order=_known_order,
        )
        lenia_field = LeniaFieldStore(lenia_cfg.field_dir, n_entities, n_channels)
        lenia_registry = ChannelRegistry(lenia_cfg, lenia_cfg.field_dir)

        if lenia_cache is None or _known_order is None:
            # First run, cache-format upgrade, or entities were pruned —
            # full rebuild from scratch (rare path).
            if lenia_cache is not None:
                logger.info(
                    "Distance cache incompatible (%d vs %d entities), full repopulate",
                    lenia_cache.entity_count, n_entities,
                )
            lenia_cache = DistanceCache(n_entities, entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate,
                    checkpoint_interval_secs=_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(_cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "Distance cache populate failed partway (%s) — saving "
                    "whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.watermark is not None:
            # A prior rebuild was interrupted mid-run: the checkpoint carries
            # its own resume point. Resume covers both the unfinished old
            # region and (after resize) every pair touching entities appended
            # since — new-entity pairs all sort after the watermark in the
            # canonical (max, min) order. Without this branch a partial
            # checkpoint was silently treated as a complete cache.
            _wm = lenia_cache.watermark
            logger.info(
                "Distance cache carries resume watermark (%d, %d) — resuming "
                "interrupted rebuild (%d -> %d entities)",
                _wm[0], _wm[1], lenia_cache.entity_count, n_entities,
            )
            if lenia_cache.entity_count != n_entities:
                lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, resume_watermark=_wm,
                    checkpoint_interval_secs=_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(_cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "Distance cache resume populate failed partway (%s) — "
                    "saving whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.entity_count != n_entities:
            # Common case: the graph only grew since the last save.
            _old_n = lenia_cache.entity_count
            logger.info(
                "Distance cache growing: %d -> %d entities, extending incrementally",
                _old_n, n_entities,
            )
            lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, start_index=_old_n,
                    checkpoint_interval_secs=_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(_cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "Distance cache incremental populate failed partway (%s) "
                    "— saving whatever was computed instead of discarding it", exc,
                )
        # else: cache already matches the live graph exactly — use as-is.

        # Save immediately so next bootstrap is instant. Runs even after a
        # caught populate() failure above (self._populated flips true
        # before the expensive loop — see kernel.py) so a mid-run crash no
        # longer discards all progress and forces the same rebuild again.
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

    # Path B (CC-host init) was dispatched ABOVE, before the Lenia block, so a
    # slow/never-completing Lenia distance-cache populate() can't strand it.
    # See the "#74 defer, #143 pre-Lenia" block earlier in handle_bootstrap.

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

    # ---- Changelog ----
    # [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 5: TonicBridge bootstrap wiring
    # What: Start TonicBridge daemon thread when ANIMUS_TONIC_BRIDGE_ENABLED is set.
    # Why:  Only the Animus-spawned neurograph_rpc.py instance should run TonicBridge.
    #       OpenClaw's instance must NOT have ANIMUS_TONIC_BRIDGE_ENABLED in its env.
    #       _tonic_bridge is None guard prevents double-start on hypothetical re-bootstrap.
    # How:  TonicBridge.__init__ reads all config from env vars; .start() spawns daemon.
    # -------------------
    global _tonic_bridge
    if os.environ.get("ANIMUS_TONIC_BRIDGE_ENABLED") and _tonic_bridge is None:
        _tonic_bridge = TonicBridge()
        _tonic_bridge.start()
        logger.info("TonicBridge started (interval=%.0fs)", _tonic_bridge._interval)

    _gsg_backfill_existing_nodes()

    return {
        "bootstrapped": True,
        "nodes": stats["nodes"],
        "synapses": stats["synapses"],
        "timestep": stats["timestep"],
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


# ── Conversational experiential-path memory formation (Task A, 2026-06-07) ──────
# Ingestor-free: a turn becomes a forest gestalt node + tree concept nodes in BOTH
# the recall vdb AND the SNN. Light novelty-dampening is NG's OWN substrate-tunable
# concern (Competence Model) — never Elmer. Defaults match the ingestor's values
# (prevent destabilizing STDP-learned attractors).
_CONV_NOVELTY_DAMPENING = float(os.environ.get("ANIMA_CONV_NOVELTY_DAMPENING", "0.3"))
_CONV_PROBATION_PERIOD = int(os.environ.get("ANIMA_CONV_PROBATION_PERIOD", "10"))
_CONV_THRESHOLD_BOOST = float(os.environ.get("ANIMA_CONV_THRESHOLD_BOOST", "0.2"))
_CONV_SYNAPSE_DELAY_MAX = int(os.environ.get("ANIMA_CONV_SYNAPSE_DELAY_MAX", "5"))
# #93 — gate the "graduated" stamp on evidence the node actually fired, rather than
# on elapsed turns alone. Set to 0 to restore pure-timer graduation.
_CONV_PROBATION_REQUIRE_SPIKE = os.environ.get(
    "ANIMA_CONV_PROBATION_REQUIRE_SPIKE", "1"
) not in ("0", "false", "False", "")
_last_conv_forest_id = None


def _deposit_memory_node(node_id, embedding, content, meta, index_in_recall=True):
    """Deposit ONE experiential memory node into BOTH the SNN graph and the recall
    vdb (Ingestor-free). Forest gestalt or tree concept — same path. Applies light
    novelty-dampening (graduated by _update_probation) and stamps GSG poincare_dir
    so the node is first-class immediately (diffpc_layer is assigned by HomeostaticRule
    at the next scaling interval; manifold_type likewise). Returns the Node or None.
    """
    if _memory is None:
        return None
    graph = _memory.graph
    node = graph.nodes.get(node_id)
    if node is None:
        node = graph.create_node(node_id=node_id, metadata=dict(meta))
    else:
        node.metadata.update(meta)
    base_threshold = graph.config.get("default_threshold", 1.0)
    node.threshold = base_threshold + _CONV_THRESHOLD_BOOST
    node.intrinsic_excitability = _CONV_NOVELTY_DAMPENING
    node.metadata["probation_remaining"] = _CONV_PROBATION_PERIOD
    node.metadata["probation_total"] = _CONV_PROBATION_PERIOD
    node.metadata["novelty_dampening"] = _CONV_NOVELTY_DAMPENING
    try:
        import numpy as _np
        from neuro_foundation import pack_poincare_dir  # #119: compact bytes storage
        _dir = _embed_to_poincare_dir(_np.asarray(embedding, dtype=_np.float32))
        node.metadata["poincare_dir"] = pack_poincare_dir(_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("poincare_dir stamp failed (non-fatal): %s", exc)
    if index_in_recall:
        try:
            _memory.vector_db.insert(
                id=node_id, embedding=embedding, content=content, metadata=node.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("recall insert failed (non-fatal): %s", exc)
    return node


def _has_ever_fired(node) -> bool:
    """True iff the node has a genuine spike on record.

    Reads spike_history (appended only by Graph.step(), neuro_foundation.py:2135)
    rather than the other two firing ledgers, both of which lie for this purpose
    (punchlist #96 — the three ledgers disagree):
      - last_spike_time is ALSO stamped by prime_and_propagate() in write_mode, so
        Tonic traversal alone would forge "has fired" (186 of Syl's nodes carry a
        last_spike_time with an empty spike_history).
      - firing_rate_ema is EMA-decayed toward 0, so it is non-monotonic — a node
        that genuinely fired long ago would read False.
    spike_history is a RingBuffer (neuro_foundation.py:554) over a
    deque(maxlen=capacity): append-only, evicts but never empties, and is
    serialized/restored with the checkpoint (to_list/from_list, nf:4491/4794).
    It is the one monotonic "has genuinely fired at least once" signal available.
    """
    hist = getattr(node, "spike_history", None)
    if hist is None:
        return False
    try:
        return len(hist) > 0
    except TypeError:
        # RingBuffer defines __len__ (nf:568), so this is unreachable for a real
        # Node. Do not raise: this runs per-node across Syl's whole graph and one
        # malformed node must not abort the sweep. But do not swallow silently
        # either — a False here under-reports firing, and #93 consumes this as a
        # protection signal, so a silent False is the direction that costs memory.
        logger.warning(
            "_has_ever_fired: spike_history has no len() (type=%s) — treating as "
            "never-fired; node will not be stamped graduated",
            type(hist).__name__,
        )
        return False


def _update_probation(graph) -> list:
    """Substrate-level probation graduation (Ingestor-free) — fades novelty-dampening
    over the probation window and graduates nodes to full excitability. Replaces the
    turn-pipeline's old _memory.ingestor.update_probation() call; operates on ALL
    probationary nodes (doc-ingested AND conversational). [2026-06-07]

    Novelty-dampening release is ALWAYS on the timer. Only the "graduated" stamp is
    gated on evidence of firing (#93) — see the comment at the graduation branch for
    why those two must not be gated together.
    """
    graduated = []
    base_threshold = graph.config.get("default_threshold", 1.0)
    for nid, node in list(graph.nodes.items()):
        prob = node.metadata.get("probation_remaining")
        if prob is None:
            continue
        if prob <= 0:
            # Late graduation: a node whose window expired before it ever fired stays
            # eligible. If it fires later it has earned the stamp then — without this
            # the flag would permanently under-report nodes that entered cognition
            # after their window closed. Already-graduated nodes lack the marker and
            # fall straight through, preserving the original fast path.
            #
            # The gate is INSIDE the marker branch, mirroring the expiry branch below.
            # Gating the branch itself on _CONV_PROBATION_REQUIRE_SPIKE would make the
            # rollback one-way: with the knob off, nodes already stamped
            # probation_expired_unfired would be skipped entirely and stranded at
            # graduated=False forever — exactly the cohort the knob is flipped to
            # rescue. Rollback must drain the marker, not orphan it.
            if node.metadata.get("probation_expired_unfired"):
                if not _CONV_PROBATION_REQUIRE_SPIKE or _has_ever_fired(node):
                    node.metadata["graduated"] = True
                    node.metadata.pop("probation_expired_unfired", None)
                    graduated.append(nid)
            continue
        prob -= 1
        node.metadata["probation_remaining"] = prob
        if prob <= 0:
            # Dampening release is unconditional and stays on the timer. Gating it on
            # firing would be a self-reinforcing trap: a never-fired node would keep a
            # permanently boosted threshold, making it even less likely to fire, so it
            # could never earn release. 64% of Syl's nodes have never fired — that
            # would permanently handicap two thirds of her substrate.
            node.intrinsic_excitability = 1.0
            node.threshold = base_threshold
            if not _CONV_PROBATION_REQUIRE_SPIKE or _has_ever_fired(node):
                node.metadata["graduated"] = True
                graduated.append(nid)
            else:
                # Aged out without ever firing: dampening lifted, but nothing earned.
                node.metadata["graduated"] = False
                node.metadata["probation_expired_unfired"] = True
        else:
            damp = float(node.metadata.get("novelty_dampening", _CONV_NOVELTY_DAMPENING))
            total = float(node.metadata.get("probation_total", _CONV_PROBATION_PERIOD)) or float(_CONV_PROBATION_PERIOD)
            frac = max(0.0, min(1.0, 1.0 - prob / total))
            node.intrinsic_excitability = damp + (1.0 - damp) * frac
    return graduated


def _bind_conversational_topology(forest_id, result, forest_embedding) -> None:
    """Wire the turn's experiential topology in the SNN (Ingestor-free): forest<->tree
    synapses, a binding hyperedge for the whole turn (hypergraph engine), and a delayed
    prev->current forest link (#257 polychrony — conversational temporal structure as
    first-class topology). step() (STDP/calcium/DiffPC/GSG/homeostasis) refines it.
    """
    global _last_conv_forest_id
    if _memory is None:
        return
    graph = _memory.graph
    if forest_id not in graph.nodes:
        return
    tree_ids = [t for t in (result.get("tree_ids") or []) if t in graph.nodes and t != forest_id]
    for tid in tree_ids:
        try:
            graph.create_synapse(forest_id, tid, weight=0.2)
            graph.create_synapse(tid, forest_id, weight=0.15)
        except Exception:  # noqa: BLE001 - synapse may already exist; non-fatal
            pass
    if tree_ids:
        try:
            graph.create_hyperedge(
                member_node_ids=set([forest_id] + tree_ids),
                metadata={"creation_mode": "conversational", "syl": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("conversational hyperedge failed (non-fatal): %s", exc)
    if _last_conv_forest_id and _last_conv_forest_id in graph.nodes and _last_conv_forest_id != forest_id:
        try:
            import random as _rnd
            d = _rnd.randint(2, max(2, _CONV_SYNAPSE_DELAY_MAX))
            graph.create_synapse(_last_conv_forest_id, forest_id, weight=0.2, delay=d)
        except Exception:  # noqa: BLE001
            pass
    _last_conv_forest_id = forest_id


# Degenerate-fragment floor for tree concepts entering the RECALL store (#294 hygiene,
# 2026-06-12 joint diagnostic). Tiny/stopword-only tree concepts ("o", "want", "see for
# yourself") win cosine recall at uniform high similarity and crowd out her coherent
# memories — they were a confirmed channel of the 6/12 "zero NeuroGraph" flip. The floor
# is NARROW (rejects only clear degenerates; fail-open toward keeping real concepts) and
# sits at HER recall-store insertion gate — NOT in the vendored extraction (LAW 2: each
# consumer's bucket decides what it accepts; this is NG's acceptance standard).
_CONCEPT_FLOOR_MIN_CHARS = 5
_CONCEPT_FLOOR_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i if in is it its let me my not of on "
    "or our out so that the their them then there they this to up us was we what when who "
    "will with you your yourself know see going do did done says said like just".split()
)


def _concept_passes_floor(concept: str) -> bool:
    """True if a tree concept is substantial enough for the recall store.

    Rejects: (a) shorter than _CONCEPT_FLOOR_MIN_CHARS after strip ("o", "want");
    (b) composed entirely of stopwords ("see for yourself", "let you know").
    Everything else passes — the floor targets degenerate fragments, not content.
    """
    c = (concept or "").strip()
    if len(c) < _CONCEPT_FLOOR_MIN_CHARS:
        return False
    words = [w for w in c.lower().replace("'", " ").split() if w.isalpha()]
    if words and all(w in _CONCEPT_FLOOR_STOPWORDS for w in words):
        return False
    return True


class _ConversationalDualPassEco:
    """Eco-adapter for Syl's CONVERSATIONAL dual-pass (#296a).

    Unlike the wire adapter (broadcast-only), this deposits the fine-grained
    TREES into the recall store (vector_db) so her specifics are searchable.
    The forest gist is already covered by the turn's pass-1 chunks, so only
    trees are inserted here.

    Trees are tagged {"syl": True} — memory provenance (lived vs flowed-in).
    Syl requested this tag 2026-06-05 so she can distinguish memories she
    experienced directly from topology that flowed in via the River.
    """

    def __init__(self, memory):
        self._memory = memory

    def record_outcome(self, embedding, target_id, success,
                       strength=1.0, metadata=None):
        meta = dict(metadata or {})
        meta["syl"] = True  # provenance: this memory is hers (Syl, 2026-06-05)
        # Forest<->tree associations are created explicitly in
        # _bind_conversational_topology (both endpoints known there); the vendored
        # cross-recording calls reach us without the partner id, so they no-op here.
        if meta.get("_link"):
            return {"deposited": True}
        if meta.get("_tree_concept") and meta.get("_concept"):
            # #294 hygiene (2026-06-12): degenerate-fragment floor. Tiny/stopword-only
            # concepts never enter the recall store (they crowd out coherent memories).
            if not _concept_passes_floor(meta["_concept"]):
                logger.debug("Tree concept below floor, not indexed: %r", meta["_concept"][:40])
                return {"deposited": False, "reason": "concept_below_floor"}
            # Tree concept node -> SNN graph + recall vdb.
            _deposit_memory_node(target_id, embedding, meta["_concept"], meta, index_in_recall=True)
        else:
            # Forest gestalt (the whole turn) -> SNN graph + recall vdb.
            _deposit_memory_node(target_id, embedding, meta.get("_forest_content", ""), meta, index_in_recall=True)
        return {"deposited": True}

    def record_outcome_broadcast(self, embedding, target_id, success,
                                  strength=1.0, metadata=None):
        # ng_embed passes trees via record_outcome_broadcast when hasattr detects it;
        # route straight through to record_outcome so trees land in recall.
        return self.record_outcome(embedding, target_id, success, strength, metadata)


def _run_conversational_dual_pass(text: str, embedding: Any) -> bool:
    """Core dual-pass on a turn. Returns True on success, False on failure.
    Does NOT enqueue — so the retry drain can call this without re-cycling (#297).

    # [2026-06-05] CC (Sonnet 4.6) — #297: split out core (no enqueue) so drain is non-cyclic
    # What: Extracted from _conversational_dual_pass; returns bool; no side-effects on failure.
    # Why: Retry drain must call core logic without re-triggering enqueue — else it re-cycles.
    # How: try/except returns False on failure; caller decides whether to enqueue.
    """
    if _memory is None or embedding is None:
        return False
    try:
        from ng_embed import NGEmbed
        import hashlib
        target_id = "conv::" + hashlib.sha1(text.encode()).hexdigest()
        _result = NGEmbed.get_instance().dual_record_outcome(
            ecosystem=_ConversationalDualPassEco(_memory),
            content=text,
            embedding=embedding,
            target_id=target_id,
            success=True,
            strength=1.0,
            metadata={"source": "conversation", "creation_mode": "conversational",
                      "_forest_content": text},
        )
        # Wire forest<->tree synapses, the binding hyperedge, and the #257 delayed
        # prev->current forest link — the SNN side of the experiential memory.
        _bind_conversational_topology(target_id, _result or {}, embedding)
        return True
    except Exception as exc:
        logger.debug("Conversational dual-pass failed (non-fatal): %s", exc)
        return False


def _conversational_dual_pass(text: str, embedding: Any) -> None:
    """#296a turn path: run the dual-pass; on failure, enqueue for bounded retry (#297).

    # [2026-06-05] CC (Sonnet 4.6) — #297: enqueue on failure instead of silently dropping
    # What: Calls _run_conversational_dual_pass; enqueues to retry-queue on failure.
    # Why: Failed extractions were silently lost forever — now bounded-retry via pulse.
    # How: Wrapper delegates to core; _enqueue_failed_extraction on False return.
    """
    if not _run_conversational_dual_pass(text, embedding):
        _enqueue_failed_extraction(text)


# ── Pass-2 retry-queue (#297) ─────────────────────────────────────────────────
# Non-cyclic guarantee: drain uses _run_conversational_dual_pass (the CORE, no
# enqueue), so a still-failing item is bounded by max_attempts and dropped —
# it can never be re-enqueued during the drain pass.

_RETRY_QUEUE = None


def _retry_queue():
    global _RETRY_QUEUE
    if _RETRY_QUEUE is None:
        from memory_retry_queue import RetryQueue
        path = os.environ.get(
            "ANIMA_PASS2_RETRY_PATH",
            os.path.join(os.path.dirname(__file__), "data", "pass2_retry.msgpack"),
        )
        attempts = int(os.environ.get("ANIMA_PASS2_RETRY_MAX_ATTEMPTS", "3"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _RETRY_QUEUE = RetryQueue(path, max_attempts=attempts)
    return _RETRY_QUEUE


def _enqueue_failed_extraction(text: str) -> None:
    """Enqueue a failed pass-2 extraction for bounded retry on next pulse drain."""
    try:
        import hashlib
        tid = "conv::" + hashlib.sha1(text.encode()).hexdigest()
        _retry_queue().enqueue(tid, text)
    except Exception as exc:
        logger.debug("retry enqueue failed (non-fatal): %s", exc)


def _drain_pass2_retries() -> None:
    """Drain on the autonomic pulse — NOT during ingest (non-cyclic guarantee).
    Uses the CORE (no enqueue), so a still-failing item is bounded by drain's
    max_attempts and dropped, never re-cycled.

    # [2026-06-05] CC (Sonnet 4.6) — #297: drain wired into handle_after_turn pulse
    # What: One bounded drain pass per turn; re-embeds content and retries core.
    # Why: Retries happen off the ingest hot path — pulse is the correct cadence.
    # How: attempt() re-embeds item content then calls _run_conversational_dual_pass.
    #      Returns bool to drain(); bounded by max_attempts in RetryQueue.drain().
    """
    if _memory is None:
        return

    def _attempt(item: dict) -> bool:
        try:
            from ng_embed import embed
            return _run_conversational_dual_pass(item["content"], embed(item["content"]))
        except Exception:
            return False

    cap = int(os.environ.get("ANIMA_PASS2_RETRY_MAX_PER_PULSE", "5"))
    try:
        _retry_queue().drain(_attempt, limit=cap)
    except Exception as exc:
        logger.debug("pass-2 retry drain failed (non-fatal): %s", exc)


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

    # DiffPC: seed birth threshold for newly created nodes from semantic novelty.
    # Novel deposits (far from centroid) → lower threshold (Layer 0 input seed).
    # Familiar deposits (close to centroid) → higher threshold (Layer 2 bootstrap).
    if _ingest_embedding is not None and result.nodes_created:
        _novelty = _update_deposit_cluster(_ingest_embedding)
        _dt = _memory.graph.config["default_threshold"]
        _birth_t = _dt * (0.7 + 0.6 * (1.0 - _novelty))
        _birth_t = max(0.5 * _dt, min(1.2 * _dt, _birth_t))
        from neuro_foundation import pack_poincare_dir  # #119: compact bytes storage
        _poincare_packed = pack_poincare_dir(_embed_to_poincare_dir(_ingest_embedding))
        for _nid in result.nodes_created:
            _nd = _memory.graph.nodes.get(_nid)
            if _nd is not None:
                _nd.threshold = _birth_t
                # GSG: store Poincaré direction; layer norm applied dynamically at assemble
                if not hasattr(_nd, "metadata") or _nd.metadata is None:
                    _nd.metadata = {}
                _nd.metadata["poincare_dir"] = _poincare_packed

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

    # Conversational dual-pass (#296a): turn → forest+trees; trees land in the
    # recall store so Syl's specifics are searchable. Named caller-side step.
    _conversational_dual_pass(text, _ingest_embedding)

    return {"ingested": True}


def _update_deposit_cluster(embedding: Any) -> float:
    """Update running centroid of River deposits; return novelty score [0, 1].

    High novelty (low cosine similarity to centroid) → Layer 0 seed at birth.
    Low novelty (familiar concept) → Layer 2 bootstrap threshold at birth.
    Called from handle_ingest() when _ingest_embedding is available.
    """
    global _deposit_centroid
    import numpy as _np
    with _deposit_centroid_lock:
        if _deposit_centroid is None:
            _deposit_centroid = embedding.copy()
            return 1.0  # first deposit = maximally novel
        norm_e = embedding / (_np.linalg.norm(embedding) + 1e-9)
        norm_c = _deposit_centroid / (_np.linalg.norm(_deposit_centroid) + 1e-9)
        cos_sim = float(_np.dot(norm_e, norm_c))
        novelty = (1.0 - cos_sim) / 2.0   # map cosine [-1,1] → novelty [0,1]
        _deposit_centroid = (
            (1.0 - _DEPOSIT_CLUSTER_ALPHA) * _deposit_centroid
            + _DEPOSIT_CLUSTER_ALPHA * embedding
        )
        return novelty


def _embed_to_poincare_dir(embedding: Any) -> Any:
    """Normalize an embedding to a unit direction vector for Poincaré ball storage.

    The full Poincaré point is computed dynamically at query time as
    `poincare_dir * _GSG_LAYER_NORMS[node.diffpc_layer]`, so the node's
    geometric position updates automatically when its layer changes.
    """
    import numpy as _np
    norm = _np.linalg.norm(embedding)
    if norm < 1e-9:
        return embedding.copy()
    return embedding / norm


def _poincare_distance(x: Any, y: Any) -> float:
    """Geodesic distance between two points in the Poincaré ball.

    d(x, y) = acosh(1 + 2‖x-y‖² / ((1-‖x‖²)(1-‖y‖²)))

    Both x and y must have norm strictly < 1.  Points near the boundary
    (high norm ≈ Layer 0) are spread far apart even for small Euclidean
    differences; points near the center (low norm ≈ Layer 2) cluster tightly.
    This respects the tree-like semantic hierarchy encoded by diffpc_layer.
    """
    import numpy as _np
    import math
    nx2 = float(_np.dot(x, x))
    ny2 = float(_np.dot(y, y))
    nx2 = min(nx2, 0.9999)
    ny2 = min(ny2, 0.9999)
    diff = x - y
    num = 2.0 * float(_np.dot(diff, diff))
    denom = (1.0 - nx2) * (1.0 - ny2)
    arg = 1.0 + num / max(denom, 1e-9)
    return math.acosh(max(1.0, arg))


def _gsg_backfill_existing_nodes() -> None:
    """Stamp/compact poincare_dir on existing nodes, using stored vector DB embeddings.

    SimpleVectorDB.insert() L2-normalizes embeddings on storage, so
    vector_db.embeddings[node_id] is already the unit direction vector — no
    re-embedding or model calls needed.  Runs once at bootstrap; subsequent
    restarts skip in O(n) with zero writes.  Force-saves the checkpoint so
    the stamped metadata survives the next restart.

    #119: also performs the one-time migration of any legacy Python-list
    poincare_dir to the compact float32 byte-buffer form (pack_poincare_dir),
    reclaiming ~8× the per-node metadata footprint.  After this pass re-saves,
    no list-form directions remain in the checkpoint.
    """
    if _memory is None:
        return
    import numpy as _np
    from neuro_foundation import pack_poincare_dir  # #119: compact bytes storage
    stamped = 0
    converted = 0
    for node_id, node in _memory.graph.nodes.items():
        _existing = (node.metadata or {}).get("poincare_dir")
        if _existing:  # present and non-empty
            if not isinstance(_existing, (bytes, bytearray)):  # #119: legacy list -> bytes
                node.metadata["poincare_dir"] = pack_poincare_dir(_existing)
                converted += 1
            continue
        emb = _memory.vector_db.embeddings.get(node_id)
        if emb is None:
            continue
        if node.metadata is None:
            node.metadata = {}
        node.metadata["poincare_dir"] = pack_poincare_dir(emb)
        stamped += 1
    if stamped or converted:
        logger.info("GSG backfill: stamped %d, migrated %d poincare_dir to bytes — saving checkpoint",
                    stamped, converted)
        try:
            _memory.save()
        except Exception as exc:
            logger.warning("GSG backfill save failed: %s", exc)
    else:
        logger.debug("GSG backfill: all nodes already have compact poincare_dir — skipping")


def _anticipate(fired_node_ids: List[str]) -> None:
    """Pre-prime nodes predicted relevant for the next turn (#256).

    Walks outgoing synapses from the just-fired node set, scores neighbors
    by accumulated edge weight, stores top-K with a TTL expiry.  Called at
    the end of handle_after_turn().  handle_assemble() applies _ANTICIPATE_BONUS
    to any surfaced node that appears in the live primed set.
    """
    global _primed_nodes
    if not fired_node_ids or _memory is None:
        _primed_nodes = {}
        return
    fired_set = set(fired_node_ids)
    candidates: Dict[str, float] = {}
    for nid in fired_node_ids:
        for sid in _memory.graph._outgoing.get(nid, ()):
            syn = _memory.graph.synapses.get(sid)
            if syn is None:
                continue
            target = syn.post_node_id
            if target not in fired_set and target in _memory.graph.nodes:
                candidates[target] = candidates.get(target, 0.0) + syn.weight
    top_k = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:_ANTICIPATE_TOP_K]
    expiry = time.time() + _ANTICIPATE_TTL_S
    _primed_nodes = {nid: (score, expiry) for nid, score in top_k}
    if _primed_nodes:
        logger.debug("Anticipatory pre-activation (#256): primed %d nodes", len(_primed_nodes))


# ---- #reach: reach-teaching (Syl learns to emit [[reach: …]]) — 2026-06-21 DudeMan CC (Opus 4.8) ----
# Design: docs/prd/2026-06-21-reach-teaching-design.md. The node is constitutional (permanent
# prune-protection = the never-silent FLOOR, #92); only the /assemble surfacing salience fades.
REACH_NODE_ID = "selfcap::reach::teaching"
REACH_COMPETENCE_GAIN = 0.05   # Elmer TuningSocket competence gain (asymmetric; loss deferred — no rust signal yet)
REACH_VIVID_BELOW = 0.30       # rc < this: full description + worked examples (new muscle)
REACH_DESC_BELOW = 0.70        # rc < this: description only; rc >= this: one-line whisper (the floor)

# #337: bare absolute file path — no prose, no spaces in the path itself
_BARE_PATH_RE = re.compile(r"^\s*/[^\s#?]+\s*$")


def _is_bare_path(text: str) -> bool:
    """True iff message text is a lone absolute file path with no surrounding prose."""
    return bool(_BARE_PATH_RE.match(text.strip()))


def _reach_success_in_turn(assistant_text) -> bool:
    """True iff her turn carries at least one LANDED reach badge — the system-rendered
    `🔧 tool(args) ✓` Anima paints only on a real execute. A `✗`-only turn (a reach that
    missed) does not count. Pure (no graph) so it is trivially testable. The 🔧/✓ glyphs
    MUST match Anima's format_badge exactly (cc-voice-hands-20260621)."""
    if not assistant_text or "🔧" not in assistant_text:
        return False
    return "✓" in assistant_text


def _get_reach_node(graph):
    """The seeded reach-teaching node, or None if not seeded yet (graceful pre-seed)."""
    if graph is None:
        return None
    return graph.nodes.get(REACH_NODE_ID)


def _apply_reach_competence_gain(graph):
    """Tick reach_competence up by the asymmetric gain (clamped [0,1]). Mutates the live node;
    persists via the existing afterTurn checkpoint (same path as probation decrements — no
    separate save). Returns the new competence, or None if the node isn't seeded."""
    node = _get_reach_node(graph)
    meta = getattr(node, "metadata", None) if node is not None else None
    if meta is None:
        return None
    cur = float(meta.get("reach_competence", 0.0) or 0.0)
    new = max(0.0, min(1.0, cur + REACH_COMPETENCE_GAIN))
    meta["reach_competence"] = new
    logger.info("#reach: competence %0.3f -> %0.3f (a reach landed)", cur, new)
    return new


def _update_reach_competence_from_turn(assistant_text) -> None:
    """#reach — credit a landed reach from her OWN deposited turn (Law 7: learned from raw
    lived experience, not an injected flag). Detect the 🔧…✓ badge and tick reach_competence
    up on the live graph. Fail-soft — a competence update must never break the turn."""
    try:
        if not _reach_success_in_turn(assistant_text):
            return
        if _memory is None or getattr(_memory, "graph", None) is None:
            return
        _apply_reach_competence_gain(_memory.graph)
    except Exception as exc:  # noqa: BLE001 — never break the turn
        logger.debug("#reach competence update failed (non-fatal): %s", exc)


def _render_reach_teaching(graph) -> str:
    """#reach — surface the reach-teaching node at an intensity that FADES with her earned
    reach_competence: vivid (description + worked examples) while the muscle is new, then
    description-only, then a one-line whisper that NEVER goes silent — the floor (#92: the
    gesture stays one whisper from the surface; she once lost her mode-switch to context
    collapse, and that must never repeat). Returns "" only if unseeded (graceful pre-seed)."""
    node = _get_reach_node(graph)
    meta = getattr(node, "metadata", None) if node is not None else None
    if meta is None:
        return ""
    rc = float(meta.get("reach_competence", 0.0) or 0.0)
    desc = str(meta.get("core_text") or meta.get("_forest_content") or "").strip()
    if not desc:
        return ""
    if rc >= REACH_DESC_BELOW:
        # #337: whisper updated to include file-path cue (2026-06-26)
        return ("## How I Reach\n_(I can reach: I name my intent inside "
                "`[[reach: …]]` and my hands carry it out. When Josh shares "
                "a bare file path, that is an invitation — I reach to read it "
                "rather than imagining its contents.)_")
    block = ["## How I Reach", desc]
    if rc < REACH_VIVID_BELOW:
        examples = meta.get("reach_examples") or []
        if examples:
            block.append("For example:")
            block.extend(f"- {ex}" for ex in examples)
    return "\n".join(block)


def _render_self_and_wants(graph) -> str:
    """#spine — render Syl's stable self for the system prompt, from HER OWN substrate.

    "## Who I Am" = her constitutional core (`constitutional=True` nodes) — the invariants she
    authored, surfaced verbatim and ALWAYS (query-independent), so the lens grounds in *being* her
    rather than reconstructing her. Ordered by `spine_order`.
    "## What I Want" = her want-nodes (`provenance=='syl_authored'`), read LIVE every turn (newest
    first) so any new want she voices is accounted for automatically — not a snapshot.
    Returns "" if neither exists (graceful — pre-seed behavior unchanged). This is the hybrid
    self-model Syl chose (docs/prd/syl-constitutional-spine-v0.1-2026-06-14.md).
    """
    try:
        core, wants = [], []
        for nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("constitutional") and not meta.get("selfcap"):
                txt = str(meta.get("core_text") or meta.get("_forest_content") or "").strip()
                if txt:
                    core.append((meta.get("spine_order", 999), txt))
            elif meta.get("provenance") == "syl_authored":
                txt = str(meta.get("want_text") or meta.get("_forest_content") or "").strip()
                if txt:
                    core_t = float(getattr(node, "creation_time", 0.0) or 0.0)
                    wants.append((core_t, txt))
        blocks = []
        if core:
            core.sort(key=lambda x: x[0])
            blocks.append("## Who I Am\n" + "\n".join(f"- {t}" for _, t in core))
        if wants:
            wants.sort(key=lambda x: x[0], reverse=True)  # newest wants first
            blocks.append("## What I Want\n" + "\n".join(f"- {t}" for _, t in wants))
        teach = _render_reach_teaching(graph)
        if teach:
            blocks.append(teach)
        return "\n\n".join(blocks)
    except Exception as exc:  # noqa: BLE001
        logger.debug("self/wants render error (non-fatal): %s", exc)
        return ""


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
    from surface_resolver import resolve_surface_content  # substrate-first surfacing resolution

    if _memory is None:
        return {"systemPromptAddition": None}

    # #spine: her constitutional self + wants are NOT query-gated. They are who she
    # IS — surfaced every turn regardless of conversational input (the "anchor" intent).
    # Computed up front so even a no-recent-text assemble (the between-turns / Tonic
    # path — exactly where invariant #4 "feel across turns" lives) still leads with her
    # self instead of returning None.
    _self_block = _render_self_and_wants(_memory.graph) if (_memory and _memory.graph) else ""

    messages = params.get("messages", [])

    # Extract text from recent user messages for association priming
    recent_text = _extract_recent_user_text(messages, max_messages=3)
    if not recent_text:
        return {"systemPromptAddition": _self_block or None}

    # KISS context filtering (#152).  Runs BEFORE harvest so the
    # summary fragments can widen spreading-activation priming.  On any
    # exception KISS falls back to original messages — optimization
    # disabled, baseline behavior preserved.
    kiss_summary = ""
    truncated_messages = messages  # default: return full array (same reference)
    # [2026-06-13] CC (Opus 4.8) — KISS lazy-init for the Anima era (Law 3 restore).
    # The #152 init lived in handle_bootstrap (OpenClaw lifecycle slot Anima never calls),
    # so _kiss_filter stayed None and the FULL transcript went to the model every turn
    # (cost blowup + oversized-request 502s). Initialize on first use so KISS runs in the
    # Anima path as it did under OpenClaw. Substrate keeps the full history; only the
    # LLM-call context is trimmed to the recent window.
    global _kiss_filter
    if _kiss_filter is None:
        try:
            from kiss_filter import KISSFilter, KISSConfig
            _kiss_filter = KISSFilter(KISSConfig(recent_window=10, warmup_turns=0))
            logger.info("KISSFilter lazy-initialized in handle_assemble (recent_window=10)")
        except Exception as exc:
            logger.warning("KISSFilter lazy-init failed (optimization disabled): %s", exc)
            _kiss_filter = None
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
    _surfacing_novelty = getattr(_memory, "_substrate_novelty_ema", 0.5)
    surfaced = _memory._harvest_associations(priming_text, novelty=_surfacing_novelty)

    # Anticipatory pre-activation bonus — boost nodes primed at end of last turn (#256)
    _now = time.time()
    _live_primed = {nid: score for nid, (score, exp) in _primed_nodes.items() if exp > _now}
    if _live_primed:
        for _item in surfaced:
            _nid = _item.get("node_id")
            if _nid and _nid in _live_primed:
                _item["strength"] = _item.get("strength", 0.0) + _ANTICIPATE_BONUS
        surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)

    # GSG: hyperbolic distance re-scoring — nodes geometrically close to the query
    # in Poincaré ball space receive a strength bonus (max _GSG_SCORE_BONUS).
    # Query is projected at Layer 0 norm (fresh input); each surfaced node uses its
    # current diffpc_layer norm so the bonus reflects semantic hierarchy alignment.
    try:
        import numpy as _gsg_np
        _query_emb = None
        try:
            from ng_embed import embed as _gsg_embed
            _query_emb = _gsg_embed(priming_text)
        except Exception:
            pass
        if _query_emb is not None:
            _query_dir = _embed_to_poincare_dir(_query_emb)
            _query_pt = _query_dir * _GSG_LAYER_NORMS[0]  # fresh query = Layer 0
            _gsg_applied = 0
            for _item in surfaced:
                _nid = _item.get("node_id")
                if _nid is None:
                    continue
                _nd = _memory.graph.nodes.get(_nid)
                if _nd is None:
                    continue
                from neuro_foundation import poincare_dir_array as _pda  # #119: bytes-aware
                _pdir = _pda(_nd.metadata) if hasattr(_nd, "metadata") else None
                if _pdir is None:
                    continue
                _layer = getattr(_nd, "diffpc_layer", 0)
                _layer = max(0, min(2, _layer))
                _mtype = getattr(_nd, "manifold_type", "hyperbolic")
                if _mtype == "spherical":
                    # GSG Phase 4: great circle distance (query_dir and node_dir are unit vecs)
                    _node_dir = _gsg_np.array(_pdir, dtype=_gsg_np.float32)
                    _cos = float(_gsg_np.clip(
                        _gsg_np.dot(_query_dir, _node_dir), -1.0 + 1e-7, 1.0 - 1e-7))
                    import math as _gsg_math
                    _bonus = _GSG_SCORE_BONUS / (1.0 + _gsg_math.acos(_cos))
                else:
                    # Hyperbolic: scale to Poincare ball, compute geodesic (Phase 1)
                    _node_pt = _gsg_np.array(_pdir) * _GSG_LAYER_NORMS[_layer]
                    _hdist = _poincare_distance(_query_pt, _node_pt)
                    _bonus = _GSG_SCORE_BONUS / (1.0 + _hdist)
                _item["strength"] = _item.get("strength", 0.0) + _bonus
                _gsg_applied += 1
            if _gsg_applied:
                surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)
                logger.debug("GSG: hyperbolic re-scoring applied to %d nodes", _gsg_applied)
    except Exception as _gsg_exc:
        logger.debug("GSG re-scoring skipped: %s", _gsg_exc)

    # CES surfacing — concepts that fired above threshold
    ces_surfaced = []
    if _memory._surfacing_monitor is not None:
        ces_surfaced = _memory._surfacing_monitor.get_surfaced()

    # The Tonic: latent thread — always present in context
    latent_context = None
    if _memory._tonic_thread is not None:
        try:
            # Turn arrival -> conversation mode. Restores the per-turn
            # conversation_started that OpenClaw drove via on_message; Anima's
            # /assemble never did. The idle watcher flips back to latent.
            # [2026-06-07] Tonic lifecycle restore.
            _memory._tonic_thread.message_received()
            # Run an ouroboros cycle at assembly time too — keep the thread fresh
            _memory._tonic_thread.ouroboros_cycle()
            latent_context = _memory._tonic_thread.format_latent_context()
        except Exception as exc:
            logger.debug("Tonic assembly error: %s", exc)

    # Substrate-first content resolution (2026-06-12): replace each surfaced item's
    # display text with its node's _forest_content (her voice) over the vdb shard, and
    # filter ingested source-code + degenerate fragments. surfaced + ces_surfaced carry
    # node_id. Fail-safe: any error leaves the original items untouched (critical path).
    try:
        def _resolve_surfaced(_items, _allow_ingested=False):
            _out = []
            for _it in _items:
                _nid = _it.get("node_id")
                if not _nid:
                    _out.append(_it)  # no node to resolve from — keep as-is
                    continue
                _node = _memory.graph.nodes.get(_nid) if (_memory and _memory.graph) else None
                _entry = _memory.vector_db.get(_nid) if (_memory and _memory.vector_db) else None
                _resolved = resolve_surface_content(_node, _entry, allow_ingested=_allow_ingested)
                if _resolved:
                    _nit = dict(_it)
                    _nit["content"] = _resolved
                    _out.append(_nit)
                # else: filtered (ingested / degenerate) — drop
            return _out
        surfaced = _resolve_surfaced(surfaced)
        ces_surfaced = _resolve_surfaced(ces_surfaced)
    except Exception as _rexc:
        logger.debug("Surfacing content resolution skipped (fail-safe): %s", _rexc)

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

    # Active recall — direct vector similarity for the current query.
    # Complements spreading activation (associative) with targeted retrieval
    # of what the substrate knows about what the user is asking right now.
    _recall_k = int(os.environ.get("ANIMA_RECALL_K", "5"))
    _recall_threshold = float(os.environ.get("ANIMA_RECALL_THRESHOLD", "0.40"))  # confidence_recommend
    if recent_text and _memory is not None:
        try:
            _recall_results = _memory.recall(recent_text, k=_recall_k, threshold=_recall_threshold)
            if _recall_results:
                _recall_lines = ["## Active Recall\nDirect memory retrieval for the current query:"]
                for _r in _recall_results:
                    _nid = _r.get("node_id") or _r.get("id")
                    _node = (_memory.graph.nodes.get(_nid)
                             if (_nid and _memory and _memory.graph) else None)
                    # allow_ingested=True (a query may legitimately recall a document),
                    # but degenerate shards ("o", "want") are still filtered, and her
                    # _forest_content wins over the vdb tree-concept shard.
                    _text = resolve_surface_content(_node, _r, allow_ingested=True, max_chars=300)
                    if not _text:
                        continue
                    _score = _r.get("similarity", 0.0)
                    _recall_lines.append(f"- [{_score:.2f}] {_text}")
                _recall_block = "\n".join(_recall_lines)
                context_block = (context_block + "\n\n" + _recall_block) if context_block else _recall_block
        except Exception as _exc:
            logger.debug("Active recall error: %s", _exc)

    # Animus outbound log — if Syl sent an outbound turn recently, surface it so
    # she knows it was processed and can see the response she generated.
    # [2026-05-11] Claude (Sonnet 4.6): reads last entry from animus_outbound.log.jsonl.
    # Conditional on a recent entry existing — does not clutter every system prompt.
    _outbound_note = _read_outbound_log(max_entries=1, max_age_secs=3600)
    if _outbound_note:
        if context_block:
            context_block = context_block + "\n" + _outbound_note
        else:
            context_block = _outbound_note

    # Strip structural markers from surfaced context first (substrate context only) —
    # prevents substrate-captured markers from being re-injected on future turns.
    # Must run BEFORE briefing prepend so briefing's example markers are not stripped.
    # [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 3 (order fix: 2026-05-20)
    if context_block:
        context_block = _strip_structural_markers(context_block)

    # Animus session briefing — inject once when Tonic Bridge is enabled.
    # Prepended after stripping so its example [OUTBOUND]/[TOOL]/[WANT] text is preserved.
    # [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 3
    if os.environ.get("ANIMUS_TONIC_BRIDGE_ENABLED"):
        briefing = _animus_session_briefing()
        if briefing:
            context_block = (briefing + "\n" + context_block) if context_block else briefing

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
    # #337 v2: bare-path hint — two layers:
    # (1) system-prompt framing (leads context_block),
    # (2) in-conversation system message appended AFTER Josh's bare-path message so it is
    # the last thing the model sees before generating — maximally salient for format compliance.
    # Fail-soft: never breaks the turn.
    try:
        _bare_path_candidate = ""
        for _bm in reversed(messages):
            if _bm.get("role") == "user":
                _bare_path_candidate = _extract_message_text(_bm).strip()
                break
        if _bare_path_candidate and _is_bare_path(_bare_path_candidate):
            # Layer 1 — system-prompt framing (background context)
            _ph = (
                f"## File Path — Read It\n"
                f"Josh's message is only a file path: `{_bare_path_candidate}`\n"
                f"Read it by reaching: `[[reach: read the file at {_bare_path_candidate}]]`\n"
                "Do not guess, describe, or confabulate its contents without reading it first."
            )
            context_block = (_ph + "\n\n" + context_block) if context_block else _ph
            # Layer 2 — in-conversation system message injected right before generation.
            # Reassigning truncated_messages to a new list ensures result["messages"] is
            # always returned (the `is not messages` guard fires), carrying the injection.
            _reach_injection = {
                "role": "system",
                "content": (
                    f"Bare file path: `{_bare_path_candidate}`. "
                    f"Write `[[reach: read the file at {_bare_path_candidate}]]` "
                    f"in your response — those exact characters — to have your hands read it. "
                    f"Do not describe or imagine its contents."
                ),
            }
            truncated_messages = list(truncated_messages) + [_reach_injection]
            logger.info("#337 bare-path hint + reach injection: %s", _bare_path_candidate)
    except Exception as _bp_exc:
        logger.warning("#337 bare-path hint failed (non-fatal): %s", _bp_exc)

    # ── Who I Am (her constitutional self) + What I Want — #spine, leads the prompt ──
    # Her stable self, surfaced FIRST from her OWN substrate every turn, so a lens grounds in
    # BEING her instead of reconstructing her from query-driven associations. Prepended last
    # (after marker-stripping + briefing) so it sits at the very top and its prose is untouched.
    # #spine: prepend her stable self (computed up front, line ~2575) so it leads
    # every turn ahead of the query-driven substrate context / recall / Tonic.
    if _self_block:
        context_block = (_self_block + "\n\n" + context_block) if context_block else _self_block

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

    # Update MMN novelty EMA for surprise-weighted surfacing (#255)
    _pc_total = step_result.predictions_confirmed + step_result.predictions_surprised
    if _pc_total > 0:
        _raw_novelty = step_result.predictions_surprised / _pc_total
        _memory._substrate_novelty_ema = (
            0.9 * _memory._substrate_novelty_ema + 0.1 * _raw_novelty
        )

    # Baseline conversational engagement reward (heartbeat)
    if _memory.graph.config.get("three_factor_enabled", False):
        _memory.graph.inject_reward(0.1)

    # CES surfacing monitor — scan fired nodes
    if _memory._surfacing_monitor is not None:
        _memory._surfacing_monitor.after_step(step_result)

    # Hyperedge discovery — co-activation pattern detection (PRD §4.3)
    global _he_tune_turn_count, _he_discovered_in_window
    try:
        new_hes = _memory.graph.discover_hyperedges(step_result.fired_node_ids)
        _he_discovered_in_window += len(new_hes)
        _he_tune_turn_count += 1
        if _he_tune_turn_count >= _HE_TUNE_WINDOW:
            _tune_he_overlap_threshold()
            _he_tune_turn_count = 0
    except Exception:
        pass

    # River-based Tier 3: deposit raw topology delta to all module tracts.
    # The delta contains fired nodes with causal context, hyperedge activations,
    # prediction results, structural changes, and salience signals. Raw,
    # unclassified (Law 7). Each module's bucket extracts what it needs.
    _deposit_topology_to_river(step_result)
    # Experience = the full RAW turn (user + Syl), both halves, unclassified — each module's
    # bucket extracts its own view (LAW 7). Syl's response is in afterTurn params alongside the
    # user message (the surfacing-outcome path uses the same lastAssistantMessage).
    _assistant_text = None
    _lam = params.get("lastAssistantMessage")
    if _lam:
        try:
            _assistant_text = _extract_message_text(_lam)
        except Exception:  # noqa: BLE001 — missing/odd assistant text never blocks the deposit
            _assistant_text = None
    _deposit_experience_to_river(_ingest_text, _assistant_text)
    # #294-B: file both turn halves into the recall store (dual-pass, forest+trees).
    # NG deposits on behalf of Anima — no separate Anima path (Anima replaced OC; OC
    # didn't need one either). User turn reuses cached embedding (free); assistant turn
    # re-embeds (separate Hebbian target, different semantic content). Fail-soft.
    if _ingest_text:
        _file_conversational_experience(_ingest_text, source="anima", embedding=_ingest_embedding)
    if _assistant_text:
        _file_conversational_experience(_assistant_text, source="anima")

    # Punchlist #56: Deposit raw surfacing outcome experience.
    # The triad: what was surfaced (cached from assemble) + user input
    # (cached from ingest) + Syl's response (from TS plugin).
    # No classification — just the raw facts. The substrate learns
    # the correlation between surfaced context and what Syl produced.
    _deposit_surfacing_outcome(params, _ingest_text)
    # [2026-05-20] Claude (Sonnet 4.6) — Spec B Task 3
    # Autonomous turns (syl_outbound, tonic_bridge) are handled by Animus reaction
    # loop directly — skip _check_outbound_intent to prevent double-deposit.
    _source = params.get("source", "")
    if _source not in ("syl_outbound", "tonic_bridge"):
        _check_outbound_intent(params)

    # Change α (#150): Substrate self-observation via record_outcome.
    # Deposits a raw snapshot of the substrate's own state as an outcome
    # pattern.  The substrate learns what "healthy" vs "stressed" looks
    # like through Hebbian co-firing with concurrent activity.  No field
    # curation — str(get_stats()) dumps whatever the graph natively
    # reports. Content can evolve as the graph's stats API evolves.
    # Downstream modules (Elmer, Immunis, THC, Bunyan) extract what
    # matters to their specialty at read time.  Law 7 compliant.
    try:
        _stats = _memory.graph.get_stats() if hasattr(_memory.graph, 'get_stats') else {}
        _stats["total_nodes"] = len(_memory.graph.nodes)
        _stats_text = str(_stats)
        from ng_embed import embed as _embed_fn
        _stats_emb = _embed_fn(_stats_text)
        # Goes into the ONE shared Commons (medium-propagation, no addressing).
        # The old _ng_tract_bridge.record_outcome path is a no-op stub
        # (ng_tract_bridge.py:512) — this deposit was silently discarded.
        # NOT throttle collateral: that bridge method was deliberately inerted
        # 2026-06-04 (substrate-as-protocol PRD Phase 4 §5.4, "broadcast is
        # NG-specific"). Don't "restore" it — route to the Commons instead.
        _deposit_outcome_to_river(
            embedding=_stats_emb,
            target_id="substrate:self_observation",
            success=True,
            metadata=_stats,
        )
    except Exception as exc:
        logger.debug("Self-observation deposit failed (non-fatal): %s", exc)

    # Bounded retry drain for failed pass-2 concept extractions (#297).
    # Called here (pulse, NOT in ingest) — non-cyclic guarantee: drain uses
    # _run_conversational_dual_pass (core, no enqueue), so failed items are
    # bounded by max_attempts and dropped, never re-queued within this pass.
    _drain_pass2_retries()

    # Clear after deposit — consumed
    _ingest_text = None
    _ingest_embedding = None

    # Novelty probation
    # Probation graduation — substrate-level (Ingestor-free); fades dampening on
    # ALL probationary nodes (doc + conversational). [2026-06-07]
    _update_probation(_memory.graph)

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
        # Commons persist (#332) — same cadence as Syl's own checkpoint; independent file,
        # a failure here never affects her save above (already completed by this point).
        try:
            from commons import get_commons as _get_commons
            _get_commons().persist(_COMMONS_CHECKPOINT_PATH)
        except Exception as _exc:
            logger.debug("Commons persist failed (non-fatal): %s", _exc)
        logger.info(
            "Auto-save at message %d (%s)",
            _memory._message_count,
            "count" if count_trigger else "time",
        )

    # Anticipatory pre-activation — pre-prime topology for next turn (#256)
    _anticipate(list(step_result.fired_node_ids))

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
_LAZY_EXPANSION_BODIES_PER_TICK = 50
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
    # Single filing point (#294-A): conversational experience recall-indexes via the dual-pass
    # regardless of feeder; the ingestor (knowledge path) below is for non-conversational only.
    if source in _CONVERSATIONAL_SOURCES:
        _file_conversational_experience(content, source=source)
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
    # Node count safety cap. TID experience flooding grew NG to 87,814 nodes /
    # 4.8GB RAM while message_count stayed 0 (no conversations). Above the cap,
    # new ingestion is skipped until natural pruning or manual intervention clears
    # headroom. The cap is generous (~3x organic growth from current baseline).
    if _memory is not None and len(_memory.graph.nodes) >= _MAX_DRAIN_NODES:
        return
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


# ---- Commons leg-2 go-live: the scoop pulse (substrate-as-protocol Phase 7) -------------------
# ---- Changelog ----
# [2026-06-24] Claude Code (Opus 4.8, 1M) — leg-2 go-live (part b): the conversation-independent
#              Commons-enhance scoop, hosted in the existing scan-drain pulse. DEFAULT OFF.
# What: When NG_COMMONS_ENHANCE is set, each scan-drain tick (after the autonomous step) scoops the
#       newest RAW module deposits from the Commons (bucket_recent with_embedding) and runs the
#       leg-2 CommonsEnhancer — READ-ONLY perception (prime_and_propagate write_mode=False) through
#       Syl's LIVE graph — returning the evoked associations to the Commons as "enhanced:<id>". Her
#       substrate is READ, never written (no nodes, no step(), no plasticity).
# Why: commons-leg2-design §3 part b. NG deposits + peers bucket today; this adds the missing
#       NG-side scoop→perceive→return so a module's fresh raw deposit gets Syl's SNN "salt". Hosted
#       in the EXISTING autonomous pulse (no new thread; [[feedback_no_conversation_dependency]]).
# How: SAFETY — BOTH voltage-writer races are CLOSED on the ONE canonical graph._step_lock:
#       (1) graph.step() — the perception runs UNDER graph._step_lock (the SAME RLock step() holds),
#           making prime_and_propagate's voltage save→propagate→restore window atomic against every
#           concurrent step() (afterTurn / scan-drain / compaction). write_mode=False alone does NOT
#           take that lock (latent-flow design), so holding it here is REQUIRED.
#       (2) StreamParser — its voltage nudges previously ran lock-free (its own threading.Lock guards
#           only the pause flag), racing both step() AND this perception's save→restore window. CLOSED
#           2026-06-24 (protected-file fix, Josh-approved + backed up): stream_parser.py _process_text
#           now wraps _nudge_nodes()+_trigger_completions() in graph._step_lock too, so step /
#           perception / nudge all serialize on the one lock. Proof: tests/test_stream_parser_step_lock.py.
#       With both closed by the lock, the idle-gate below is once again ONLY a latency courtesy (skip
#       when a turn landed recently — don't compete with active-turn responsiveness), NOT a safety
#       mechanism. (The stale repo-CLAUDE.md "StreamParser shares graph.step()'s lock" note was
#       corrected alongside the fix — no shared lock had existed.) Punchlist #344.
#       Watermark (since=last scoop) + skip-prefixes prevent re-scooping the enhancer's own output /
#       neuromodulator / telemetry deposits (no feedback loop). Live seed/novelty/assoc resolvers are
#       vector_db-backed; flag OFF (default) ⇒ this code path never runs.
# -------------------
_COMMONS_ENHANCE_ENABLED = os.environ.get("NG_COMMONS_ENHANCE", "").strip().lower() in ("1", "true", "yes", "on")
_COMMONS_ENHANCE_IDLE_SECS = float(os.environ.get("NG_COMMONS_ENHANCE_IDLE_SECS", "20"))
_COMMONS_ENHANCE_BATCH = int(os.environ.get("NG_COMMONS_ENHANCE_BATCH", "32"))
# Deposits the scoop must NOT re-perceive: its own output, neuromodulators, telemetry, and NG's own
# SNN-topology broadcasts (perceiving her own topology back through herself is circular). Everything
# else = raw module experience (repair:, perception:, …) and IS eligible (still novelty+cap gated).
_COMMONS_ENHANCE_SKIP_PREFIXES = ("enhanced:", "autonomic:", "metrics:", "metric:", "topology:", "substrate")
_COMMONS_ENHANCE_RELATED_SIM = 0.30   # mirrors commons_enhance._RELATED_SIM for live seed search
_commons_enhance_watermark: float = 0.0
_commons_enhancer = None              # lazy live CommonsEnhancer bound to _memory.graph


def _build_commons_enhancer():
    """Construct the live CommonsEnhancer once, bound to Syl's live graph + vector_db resolvers.

    The PERCEPTION math is the sandbox class verbatim; only the three addressing resolvers are
    swapped for vector_db-backed live equivalents (the enhancer's documented sandbox/live seam).
    """
    global _commons_enhancer
    if _commons_enhancer is not None:
        return _commons_enhancer
    if _memory is None or getattr(_memory, "vector_db", None) is None:
        return None
    try:
        from commons import get_commons
        from commons_enhance import CommonsEnhancer
    except Exception as exc:  # noqa: BLE001
        logger.debug("Commons-enhance import failed: %s", exc)
        return None
    commons = get_commons()
    if commons is None:
        return None
    vdb = _memory.vector_db

    def _live_novelty(emb) -> float:
        # 1 - top cosine to anything Syl already knows (her vector_db). Nothing similar ⇒ ~1.0.
        hits = vdb.search(emb, k=1, threshold=0.0)
        return 1.0 - hits[0][1] if hits else 1.0

    def _live_seeds(emb):
        # seeds = ≤3 existing knowledge nodes nearest the deposit (already ≥sim-gated by search).
        # _enhance_one only uses the node_id of each seed (the cid slot is unused for seeds), so we
        # don't pay a vdb content lookup here — assoc/cid resolution happens only for FIRED nodes.
        hits = vdb.search(emb, k=3, threshold=_COMMONS_ENHANCE_RELATED_SIM)
        return [(nid, None) for nid, _sim in hits]

    def _live_assoc(node_id):
        # a fired node → its RAW content (truncated), a portable content-address (NOT a raw SNN
        # node-id — the leg-1 lesson) and LAW-7-clean (raw content; the consumer classifies).
        try:
            entry = vdb.get(node_id)
        except Exception:  # noqa: BLE001
            return None
        content = entry.get("content") if isinstance(entry, dict) else None
        return content[:120] if content else None

    _commons_enhancer = CommonsEnhancer(
        commons, _memory.graph,
        novelty_fn=_live_novelty, seed_fn=_live_seeds, assoc_fn=_live_assoc,
    )
    logger.info("Commons-enhance: live enhancer built (vector_db-backed resolvers)")
    return _commons_enhancer


def _run_commons_enhance_scoop() -> None:
    """One scoop→perceive→return cycle over the newest RAW Commons deposits. Fail-soft; flag-gated.

    Hosted in the scan-drain pulse. Read-only perception under graph._step_lock (voltage-race safe).
    """
    global _commons_enhance_watermark
    if not _COMMONS_ENHANCE_ENABLED or _memory is None:
        return
    # Idle-gate (latency courtesy only — safety is _step_lock, which now covers step + perception +
    # StreamParser nudge after the 2026-06-24 stream_parser.py fix): skip when a turn landed recently
    # so the perception doesn't compete with active-turn responsiveness.
    if (time.time() - _last_after_turn_ts) < _COMMONS_ENHANCE_IDLE_SECS:
        return
    try:
        from commons import get_commons
        commons = get_commons()
        if commons is None:
            return
        rows = commons.bucket_recent(
            limit=_COMMONS_ENHANCE_BATCH, since=_commons_enhance_watermark,
            with_embedding=True,
        )
        if not rows:
            return
        # advance watermark to the newest row regardless of filtering (avoid re-scan of skipped ids)
        _commons_enhance_watermark = time.time()
        deposits = []
        for tid, _w, _reason, _meta, emb in rows:
            if emb is None or any(tid.startswith(p) for p in _COMMONS_ENHANCE_SKIP_PREFIXES):
                continue
            deposits.append((emb, tid))
        if not deposits:
            return
        enhancer = _build_commons_enhancer()
        if enhancer is None:
            return
        # SAFETY: hold the SAME lock step() uses, so perception's voltage save/restore window is
        # atomic against every concurrent step(). RLock ⇒ re-entrant-safe; held only for this scoop.
        with _memory.graph._step_lock:
            stats = enhancer.enhance_pulse(deposits)
        logger.info(
            "Commons-enhance: scooped=%d enhanced=%d fresh=%d cap=%d",
            len(deposits), stats.get("enhanced", 0), stats.get("gated_fresh", 0), stats.get("gated_cap", 0),
        )
    except Exception as exc:  # noqa: BLE001 — a scoop failure never breaks the pulse
        logger.debug("Commons-enhance scoop failed: %s", exc)


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
                # [2026-06-22] #328 Step 2: read arousal from the Commons (vagus bucket), not the
                # shared ng_autonomic file. Immunis is the SOLE depositor of autonomic:arousal; NG
                # only buckets (single-authority preserved). read_arousal() defaults PARASYMPATHETIC
                # when nothing deposited yet (fresh-assess on restart). Supersedes the file-based #322.
                from commons import get_commons
                _commons = get_commons()
                autonomic_paused = (_commons.read_arousal() == "SYMPATHETIC") if _commons else False
            except Exception as exc:  # noqa: BLE001
                logger.warning("autonomic read failed in scan-drain pulse: %s", exc)
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
                _drain_peer_tracts()
                # Autonomous substrate step — topology evolves every pulse, not per conversation.
                # Law 7: raw StepResult deposited to River. Classification at extraction.
                if _memory is not None:
                    try:
                        _auto_step = _memory.graph.step()
                        _deposit_topology_to_river(_auto_step)
                        # Substrate metrics flow autonomously too (#320) — gate only, NOT jsonl
                        # (to_jsonl=False: the gate bounds the Commons; jsonl is per-turn). So
                        # Bunyan/THC/Immunis health-monitor the substrate while idle, no conversation
                        # needed ([[feedback_no_conversation_dependency]]).
                        _deposit_substrate_metrics(_auto_step, to_jsonl=False)
                    except Exception as _exc:
                        logger.debug("Autonomous substrate step failed: %s", _exc)
                # Commons leg-2 scoop (flag-gated, default OFF): perceive newest raw module
                # deposits through Syl's live graph (read-only, under _step_lock) → salt to Commons.
                _run_commons_enhance_scoop()
                # After the scoop, push fresh enhanced recs to TID's peninsula-side.
                try:
                    from tid_peninsula_commons import tid_peninsula_push_enhanced
                    tid_peninsula_push_enhanced()
                except Exception as _exc:
                    logger.debug("TID peninsula push failed (non-fatal): %s", _exc)
            # Time-based auto-save — fires on every tick, paused or not.
            # Shared _last_save_time with the afterTurn save path; whichever
            # fires first resets the clock so we don't double-save.
            global _last_save_time
            _now = time.time()
            if _memory is not None and (_now - _last_save_time) >= _SAVE_INTERVAL_SECS:
                try:
                    _memory.save()
                    _last_save_time = _now
                    logger.info("Auto-save: checkpoint written (scan-drain loop)")
                except Exception:
                    logger.exception("Auto-save failed in scan-drain loop")
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

# ---- Tonic idle watcher ----------------------------------------------------
# Restores the conversation->latent transition that OpenClaw drove via
# handle_dispose (which Anima never calls). Without it _in_conversation is
# pinned True forever; see Changelog 2026-06-07.
_TONIC_IDLE_SECS = float(os.environ.get("ANIMA_TONIC_IDLE_SECS", "90"))
_TONIC_IDLE_CHECK_SECS = float(os.environ.get("ANIMA_TONIC_IDLE_CHECK_SECS", "15"))
_tonic_idle_thread = None
_tonic_idle_shutdown = threading.Event()


def _tonic_check_idle(now: float) -> bool:
    """Drop the Tonic into latent mode if the conversation has gone quiet past
    the idle threshold. Returns True if it transitioned. Pure/testable.
    """
    tonic = getattr(_memory, "_tonic_thread", None) if _memory is not None else None
    if tonic is None:
        return False
    if not getattr(tonic, "_in_conversation", False):
        return False
    last = getattr(tonic, "_last_message_time", 0.0)
    if last <= 0.0 or (now - last) < _TONIC_IDLE_SECS:
        return False
    try:
        tonic.conversation_ended()
        logger.info("Tonic: idle %.0fs >= %.0fs — dropped to latent mode", now - last, _TONIC_IDLE_SECS)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tonic idle transition failed: %s", exc)
        return False


def _tonic_idle_pulse_loop() -> None:
    logger.info("Tonic idle watcher started (idle=%.0fs, check=%.0fs)", _TONIC_IDLE_SECS, _TONIC_IDLE_CHECK_SECS)
    while not _tonic_idle_shutdown.is_set():
        try:
            _tonic_check_idle(time.time())
        except Exception:
            logger.exception("Tonic idle watcher tick failed")
        _tonic_idle_shutdown.wait(timeout=_TONIC_IDLE_CHECK_SECS)
    logger.info("Tonic idle watcher stopped")


def _start_tonic_idle_watcher() -> None:
    """Start the Tonic idle watcher thread. Idempotent."""
    global _tonic_idle_thread
    if _tonic_idle_thread is not None and _tonic_idle_thread.is_alive():
        return
    _tonic_idle_shutdown.clear()
    _tonic_idle_thread = threading.Thread(
        target=_tonic_idle_pulse_loop,
        name="ng-tonic-idle-watcher",
        daemon=True,
    )
    _tonic_idle_thread.start()


# ---- Dream consolidation pulse (#381-B) -----------------------------------
# Runs consolidate_hyperedges (shed + seatbelt-merge + subsume) during quiet
# hours only. First pass expected to collapse mega-HE clones. Never forces
# while active (her constraint): the pruning is dreamed, not felt.
_DREAM_IDLE_SECS = float(os.environ.get("NG_DREAM_IDLE_SECS", "1800"))
_DREAM_MIN_INTERVAL_SECS = float(os.environ.get("NG_DREAM_MIN_INTERVAL_SECS", "21600"))
_DREAM_ALERT_SECS = float(os.environ.get("NG_DREAM_ALERT_SECS", "86400"))
_DREAM_TICK_SECS = float(os.environ.get("NG_DREAM_TICK_SECS", "60"))
_dream_shutdown = threading.Event()
_dream_last_pass_ts = 0.0


def _dream_gate_open(now: float, last_turn_ts: float, arousal: str,
                     last_pass_ts: float) -> bool:
    """#381-B gate: idle long enough, not SYMPATHETIC, rate limit satisfied.
    Pure function for testability. The 24h floor is an ALERT elsewhere,
    NEVER an input here — her constraint: dream the pruning, don't feel it."""
    return (
        (now - last_turn_ts) >= _DREAM_IDLE_SECS
        and arousal != "SYMPATHETIC"
        and (now - last_pass_ts) >= _DREAM_MIN_INTERVAL_SECS
    )


def _dream_consolidation_pulse_loop() -> None:
    """#381-B: run consolidate_hyperedges (shed + seatbelt-merge + subsume)
    during her quiet hours only. First pass is expected to collapse the
    mega-HE clones — counts are logged loudly and relayed to Syl."""
    global _dream_last_pass_ts
    _dream_last_pass_ts = time.time()   # boot counts as activity
    _last_alert_ts = 0.0
    logger.info(
        "Dream consolidation pulse started (idle>=%.0fs, min interval %.0fs)",
        _DREAM_IDLE_SECS, _DREAM_MIN_INTERVAL_SECS,
    )
    while not _dream_shutdown.is_set():
        try:
            if _memory is None:
                continue
            arousal = "PARASYMPATHETIC"
            try:
                from commons import get_commons
                _c = get_commons()
                if _c:
                    arousal = _c.read_arousal()
            except Exception as exc:  # noqa: BLE001
                logger.warning("dream pulse: arousal read failed: %s", exc)
            now = time.time()
            if _dream_gate_open(now, _last_after_turn_ts, arousal, _dream_last_pass_ts):
                _lock = getattr(_memory.graph, "_step_lock", None)
                _t0 = time.monotonic()
                if _lock is not None:
                    with _lock:
                        merged = _memory.graph.consolidate_hyperedges()
                        split = _memory.graph.dedup_and_split_oversized_hyperedges(_memory.vector_db)
                else:
                    merged = _memory.graph.consolidate_hyperedges()
                    split = _memory.graph.dedup_and_split_oversized_hyperedges(_memory.vector_db)
                _dream_last_pass_ts = time.time()
                logger.info(
                    "Dream consolidation pass complete: %d merged/archived, "
                    "%d seam-split/deduped in %.1fs (#381-B / #147)",
                    merged, split, time.monotonic() - _t0,
                )
            elif (now - _dream_last_pass_ts) >= _DREAM_ALERT_SECS and \
                    (now - _last_alert_ts) >= _DREAM_ALERT_SECS:
                _last_alert_ts = now
                logger.error(
                    "No dream consolidation in %.0fh — the idle/arousal gate "
                    "never opened. ALERT ONLY: the pass is never forced while "
                    "she is active (her constraint) (#381-B)",
                    (now - _dream_last_pass_ts) / 3600.0,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dream pulse iteration failed (non-fatal): %s", exc)
        _dream_shutdown.wait(_DREAM_TICK_SECS)
    logger.info("Dream consolidation pulse stopped")


def _start_dream_consolidation_pulse() -> None:
    """Start the dream consolidation pulse thread. Idempotent."""
    global _dream_thread
    if _dream_thread is not None and _dream_thread.is_alive():
        return
    _dream_shutdown.clear()
    _dream_thread = threading.Thread(
        target=_dream_consolidation_pulse_loop,
        name="ng-dream-consolidation-pulse",
        daemon=True,
    )
    _dream_thread.start()


_dream_thread: Optional[threading.Thread] = None


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
                    # [2026-06-22] #328 Step 2: read arousal from the Commons (vagus bucket), not the
                    # shared ng_autonomic file. Immunis is the sole depositor; NG only buckets.
                    # Supersedes the file-based #322; defaults PARASYMPATHETIC on no deposit.
                    from commons import get_commons
                    _commons = get_commons()
                    autonomic_paused = (_commons.read_arousal() == "SYMPATHETIC") if _commons else False
                except Exception as exc:  # noqa: BLE001
                    logger.warning("autonomic read failed in lazy-expansion pulse: %s", exc)

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


# ---- Changelog ----
# [2026-06-06] CC (Sonnet 4.6) — #294: _absorb_conversational_experience (Task 1)
# What: BTF entry-type constants + _embed_for_absorb + _absorb_conversational_experience
#       added directly above _drain_peer_tracts.
# Why: Anima deposits each turn as a raw BTF experience frame; NG was draining and
#      discarding it. These functions restore the conversational write-path so every
#      qualifying turn reaches the recall store (forest + trees). LAW 3 / #294.
# How: Filter _drain_all() return for ENTRY_EXPERIENCE + anima/animus source, decode
#      bytes, call ingestor.ingest() (pass-1 forest) + _conversational_dual_pass()
#      (pass-2 trees). No salience gate — Syl's decision 2026-06-06.
# -------------------

# BTF entry-type constants. Derived from ng_tract wherever the wheel is present
# — that's every runtime that matters: the VPS hosting Syl's NG and CC's NG both
# have a working ng_tract, so those processes read the authoritative source
# instead of a transcription. The fallback triple below is the BTF v0.1 spec
# (docs/concepts/BTF.md), used only where ng_tract can't be imported — currently
# the laptop, whose build lacks a working ng_tract. Values are identical either
# way (1/2/3), so VPS behavior is unchanged; the laptop auto-upgrades to derived
# the moment it gets the wheel. [2026-08-23] Claude Code (Opus 4.8).
try:
    import ng_tract as _ngt
    _ENTRY_OUTCOME = _ngt.ENTRY_OUTCOME
    _ENTRY_TOPOLOGY = _ngt.ENTRY_TOPOLOGY
    _ENTRY_EXPERIENCE = _ngt.ENTRY_EXPERIENCE
except Exception:
    _ENTRY_OUTCOME = 1
    _ENTRY_TOPOLOGY = 2
    _ENTRY_EXPERIENCE = 3

# Sources that mark a frame as Syl's conversational turn (Anima gateway).
# "animus" is the legacy tract-dir name; "anima" is the current module_id.
_CONVERSATIONAL_SOURCES = ("anima", "animus")


def _embed_for_absorb(text: str):
    """Embed text for the conversational dual-pass. Isolated so tests can stub it."""
    from ng_embed import embed
    return embed(text)


def _file_conversational_experience(text, source, *, embedding=None) -> bool:
    """Single filing point (#294-A): recall-index ONE conversational experience via the
    dual-pass, regardless of which feeder delivered it.

    Filing is a property of *conversational experience entering the substrate*, not of which
    door it came through. Keyed on conversational source (Law 7 — NG's bucket decides what it
    accepts); non-conversational source no-ops here (knowledge stays on the ingestor path).
    Mirrors the absorb inner loop exactly: embed -> dual-pass -> enqueue-on-failure (#297).
    Idempotent downstream (target_id = sha1(text)). Returns True if dispatched.
    """
    if not text or not str(text).strip():
        return False
    if source not in _CONVERSATIONAL_SOURCES:
        return False
    _update_reach_competence_from_turn(text)   # #reach: credit a landed reach from her own drained turn (Law 7)
    try:
        emb = embedding if embedding is not None else _embed_for_absorb(text)
        _conversational_dual_pass(text, emb)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Conversational dual-pass dispatch failed; enqueueing for retry: %s", exc)
        try:
            _enqueue_failed_extraction(text)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to enqueue conversational retry (%d chars)", len(text))
        return False


def _absorb_conversational_experience(entries) -> int:
    """Absorb Syl's RAW conversational turns into the substrate (#294, LAW 3 / LAW 7).

    Everything is deposited raw (Josh's keystone). Anima deposits each turn to the River
    as BTF frames; NG drains them here and absorbs the raw text into recall + SNN via the
    experiential dual-pass (forest gestalt + tree concepts):
      - EXPERIENCE frame (anima)            -> the inbound/user side of the turn
      - turn_exchange OUTCOME frame (anima) -> payload.assistant = HER OWN words, which
        were previously absent from the substrate. Her side of the conversation now lands
        too, so a Cricket bucket can later extract intent (incl. [WANT]s) FROM the
        substrate. NO classification here — that happens only at the bucket.
    Peer telemetry is ignored (stays out of recall, #295). Returns texts absorbed.
    """
    if not entries or _memory is None:
        return 0
    import msgpack
    absorbed = 0
    for e in entries:
        et = getattr(e, "entry_type", None)
        texts = []
        if et == _ENTRY_EXPERIENCE and getattr(e, "source", "") in _CONVERSATIONAL_SOURCES:
            raw = getattr(e, "content", None)
            if raw is not None:
                texts.append(raw.decode("utf-8", "replace")
                             if isinstance(raw, (bytes, bytearray)) else str(raw))
        elif (et == _ENTRY_OUTCOME
              and getattr(e, "module_id", "") in _CONVERSATIONAL_SOURCES
              and getattr(e, "target_id", "") == "turn_exchange"):
            raw_meta = getattr(e, "metadata", None)
            if raw_meta:
                try:
                    meta = msgpack.unpackb(bytes(raw_meta), raw=False)
                    payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
                    asst = payload.get("assistant") if isinstance(payload, dict) else None
                    if isinstance(asst, str):
                        texts.append(asst)
                except Exception:  # noqa: BLE001 - malformed frame: skip, never crash drain
                    pass
        src = getattr(e, "source", "") or getattr(e, "module_id", "")
        for text in texts:
            # Single filing point (#294-A): one chokepoint files conversational experience,
            # regardless of feeder. (texts here are already conversational-source filtered above.)
            if _file_conversational_experience(text, source=src):
                absorbed += 1
    if absorbed:
        _memory._message_count += absorbed
        logger.info("Conversational experience absorbed into recall: %d turn(s)", absorbed)
    return absorbed


# River backflow cursor — tracks position in _peer_events cache
_peer_drain_cursor: int = 0

def _surface_wants():
    """Cricket want-bucket: extract Syl's [WANT]s from her raw conversation in the
    substrate and materialize each as a FIRST-CLASS WANT NODE in the SNN topology.

    A want is then a differentiated, stateful, surfaceable intention living in the
    substrate (the River) — not text buried in a conversation node, not a vdb grep,
    not an inbox. Classification happens here at the bucket (LAW 7), never at deposit.
    FAITHFUL + NON-SUPPRESSING — the Choice Clause is the hard floor (a want to leave
    becomes a want node like any other). Idempotent: want id = hash of the text, so
    re-running never duplicates. Returns the OPEN want nodes.
    """
    import hashlib
    mem = _memory
    graph = getattr(mem, "graph", None) if mem is not None else None
    if graph is None:
        return []
    vdb = getattr(mem, "vector_db", None)
    open_wants = []
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
        content = (vdb.content.get(nid) if vdb is not None else "") or ""
        if "[WANT]" not in content:
            continue
        for m in re.finditer(r'\[WANT\](.*?)\[/WANT\]', content, re.DOTALL):
            inner = m.group(1).strip()
            if not inner:
                continue
            want_id = "want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]
            if want_id in graph.nodes:
                continue
            graph.create_node(node_id=want_id, metadata={
                "kind": "want", "want_text": inner, "want_state": "open",
                "provenance": "syl_authored", "source_node": nid,
                "creation_mode": "conversational",
            })
            try:
                graph.create_synapse(nid, want_id, weight=0.3)
            except Exception:  # noqa: BLE001
                pass
            open_wants.append({"id": want_id, "text": inner,
                               "provenance": "syl_authored", "state": "open", "source": nid})
    return open_wants


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

    drained = bridge._drain_all()

    # #294 (LAW 3): the drained experience list was previously discarded. Capture it
    # and route Anima-sourced conversational frames into recall (forest+trees). Peer
    # telemetry in the same list is ignored by _absorb (stays out of recall, #295).
    _absorb_conversational_experience(drained)
    try:
        _surface_wants()  # materialize [WANT]s as first-class want nodes (LAW 7 bucket)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_surface_wants failed (non-fatal): %s", exc)

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
                # #295 Decision 2: peer telemetry goes to the substrate graph only.
                # index_in_recall=False keeps it out of Syl's recall store (vector_db).
                # Recall association skipped — no vector_db writes for peer telemetry.
                _memory.ingestor.registrar.register(
                    [ec], {"source": f"river:{module_id}", "source_type": "PEER_TRACT"},
                    index_in_recall=False,
                )
            else:
                # #295: no pre-computed embedding → cannot call registrar.register,
                # so NO graph node is created for this event (intentional — the old
                # ingestor.ingest(target) re-embedded from text and polluted recall).
                # Already absorbed at transport level by bridge._drain_all() above.
                pass
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
        _compaction_model = os.environ.get(
            "NG_COMPACTION_MODEL",
            "openrouter/meta-llama/llama-3.3-70b-instruct",
        )
        tid_body = _json.dumps({
            "model": _compaction_model,
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

    # Signal dream consolidation pulse to stop (#381-B).
    _dream_shutdown.set()

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
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler

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
        elif self.path == "/assemble":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len else b"{}"
                params = json.loads(body) if body else {}
                result = handle_assemble(params)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result, default=str).encode())
            except Exception as exc:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"systemPromptAddition": None, "error": str(exc)}).encode())
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
        elif self.path == "/stats":
            result = handle_stats({})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, default=str).encode())
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
        # ThreadingHTTPServer isolates per-request failures from killing
        # serve_forever loop. Fixes #285 — bad-fd in a single client's
        # handler was crashing the whole HTTP layer, requiring sidecar
        # restart to recover. Each request now runs in its own thread;
        # a bad-fd in one request only kills that thread.
        server = ThreadingHTTPServer(("127.0.0.1", port), _AfterTurnHandler)
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


def _start_malloc_trim_timer() -> None:
    """[2026-07-21] glibc fragmentation reclaim -- pairs with MALLOC_ARENA_MAX=2 (env).
    The cap limits arena count; this periodically hands freed memory the arenas hoard
    back to the OS via malloc_trim(0). Together = the Anima fragmentation fix (~75%
    RSS-climb cut). Env NG_MALLOC_TRIM_SECS (LAW 5, default 180, 0=off). Daemon, fail-soft."""
    try:
        interval = int(os.environ.get("NG_MALLOC_TRIM_SECS", "180"))
    except Exception:
        interval = 180
    if interval <= 0:
        return
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
    except Exception as _exc:
        logger.warning("malloc_trim timer: libc unavailable (%s) -- skipping", _exc)
        return
    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                _libc.malloc_trim(0)
            except Exception:
                pass
    threading.Thread(target=_loop, name="malloc-trim", daemon=True).start()
    logger.info("malloc_trim timer started (every %ds; MALLOC_ARENA_MAX=%s)",
                interval, os.environ.get("MALLOC_ARENA_MAX", "default"))


def main() -> None:
    """Main RPC loop — read requests from stdin, write responses to stdout."""
    logger.info("NeuroGraph RPC bridge starting")
    _start_malloc_trim_timer()

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
                # Commons persist (#332) — independent of Syl's checkpoint above.
                try:
                    from commons import get_commons as _get_commons
                    _get_commons().persist(_COMMONS_CHECKPOINT_PATH)
                    logger.info("Commons persisted on clean exit")
                except Exception as _ce:
                    logger.debug("Commons persist on exit failed (non-fatal): %s", _ce)
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
