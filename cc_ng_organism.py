#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-28] Claude Code (DudeMan CC, Opus 5) — Callosum Leg 1: FatherGraph absorption discipline + move off the 60s pulse
# What: drain_gateway_conduit() gained batch_size/idle_steps/load_ceiling/exclude_prefix
#   and now sleeps between batches instead of draining every queued file back-to-back:
#   after every batch_size absorbed turns it runs idle_steps of pure graph.step()
#   (_cc_callosum_consolidate) BEFORE taking in more, plus a trailing pass. Load-aware
#   via cc_refeed.should_pause_for_load (stops clean, leaves files on disk = backpressure).
#   exclude_prefix stops a hemisphere eating its own outgoing files. Conduit glob
#   laptop_cc_gateway.* -> *_cc_gateway.* and the producer now tags filenames with
#   MACHINE_ID (the hardcoded "laptop" made VPS-produced files invisible to the drain --
#   latent, since only the laptop produces today, but it silently broke bidirectionality).
#   The pulse call site in cc_ng_host.py is REMOVED; cc_ng_host gained a drain_conduit
#   socket handler so the nightly cc-ng-sync.py drives the LIVE daemon instead.
# Why: FatherGraph Finding 1 -- "the drain can't be a bulk dump... New topology must
#   arrive gradually enough that the receiving topology's homeostatic regulation can
#   absorb it without displacement" (stable batch ~20-30). Finding 3 -- "After receiving
#   a merge batch, run idle steps (~250) BEFORE accepting the next batch", measured
#   47%->74% accuracy, "not optional -- it's what makes merge work". A 60-second autosave
#   pulse can satisfy neither, and it would have delivered a whole cron-gap's backlog
#   (~45 files) in one tick. Also LAW 3: the lossy cc-ng-sync.py JSONL path this replaces
#   was still running in parallel; the callosum now takes over that nightly slot, which
#   already exports CC_NG_BATCH_SIZE=25 / CC_NG_IDLE_STEPS=250 -- the FatherGraph values.
# How: reuses drain_ingest_tract unchanged for per-file BTF parse + dual-pass (LAW 3);
#   mirrors the batch+sleep loop already in _handle_import (cc_ng_host.py) and
#   import_trickle (cc-ng-sync.py). Gate CC_CALLOSUM_LEG1_ENABLED unchanged, default off.
#   Ref: docs/reports/Topology_Merge_Insights_from_FatherGraph_Training.md
# [2026-07-27] Claude Code (Sonnet 5) — CC Corpus Callosum Leg 1 (#70): raw-turn
#   conduit, laptop -> VPS Arborist
# What: New cc_gateway_conduit_dir()/trickle_gateway_conduit()/drain_gateway_
#   conduit() in the same region as cc_gateway_tract_path()/drain_ingest_tract().
#   trickle_gateway_conduit(data) writes a snapshot of the laptop's cc_gateway
#   tract bytes to a uniquely-named per-batch file (laptop_cc_gateway.<ts>_
#   <uuid8>.tract) in the git-synced ~/docs/ng_topology dir. drain_gateway_
#   conduit(graph, vector_db, state) is the VPS-side counterpart: globs every
#   laptop_cc_gateway.*.tract file in that dir, runs each through the existing
#   drain_ingest_tract() (unchanged), then deletes the now-emptied file.
# Why: Retires the lossy top-N JSONL sync (cc-ng-sync.py: content-only, capped
#   at EXPORT_SIZE, re-embedded via on_message() with no synapses/hyperedges/
#   tree structure). The laptop does zero embedding by design (no forest, no
#   tree, no TID) -- the VPS is the sole Arborist for both hemispheres. This
#   is the pipe that gets the laptop's raw BTF conversation frames onto the
#   VPS so they hit the same run_conversational_dual_pass() the VPS already
#   runs for its own local tract. Spec: docs/superpowers/plans/2026-07-27-
#   cc-corpus-callosum-leg1-spec.md.
# How: Per-batch filenames (not a shared append/truncate target) sidestep the
#   binary-merge-conflict scenario a single conduit file would hit under
#   repo-sync.sh's git push/pull cycle (git can't line-merge BTF) -- each
#   trickle-copy is one immutable file, atomically materialized via write-tmp
#   -then-rename so a mid-write crash or an in-flight repo-sync.sh push never
#   observes a partial file. Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5, default
#   off) on both the laptop write side and the VPS drain side independently --
#   symmetric gate-off means neither half does anything until both are flipped
#   on. drain_ingest_tract() itself is untouched (no signature/behavior change,
#   no reordering of its existing local-drain call site); the snapshot read
#   that feeds trickle_gateway_conduit() is a separate, additional read of the
#   same tract path performed by the caller (cc-ng-daemon.py's autosave pulse)
#   immediately before the existing drain_ingest_tract() call -- pure read, no
#   truncate, so the local drain's own truncate-after-drain lifecycle is
#   completely untouched. Accepted narrow race: bytes miniTID appends between
#   that snapshot read and drain_ingest_tract()'s own (immediately following)
#   read ride along into drain's local truncate but are NOT in the snapshot,
#   so they reach the laptop's own forest but miss this pulse's conduit copy --
#   caught by the next pulse's snapshot instead (miniTID only ever appends;
#   nothing is lost, just delayed one pulse). See test_cc_callosum_leg1.py.
# [2026-07-22] Claude Code (Sonnet 5) — CC Recall Unification (LAW-3/"keep even")
# What: New cc_assemble_recall(ng, query, k, conv_state, commons,
#   allow_pattern_completion=True) -- THE shared recall pipeline for both
#   hemispheres. Verbatim extraction of cc-ng-daemon.py's (laptop) _recall
#   body: SurfacingMonitor harvest -> Active Recall (cc_pattern_completion_
#   recall) dedup'd against it -> gated Pith (CacheLines, pith_victim_recover,
#   cc_thermal, cc_novelty, pith_stage1, pith_stage3(budget=cc_l1_budget),
#   pith_victim_capture) -> _format_cc_recall_block, fail-soft to the
#   pre-Pith monitor_ctx/pc_block concat. Also folded in the CC_RECALL_DEBUG
#   instrumentation (_cc_recall_debug_log) and the Pith-fallback rate-limited
#   warning (_last_pith_warn_ts/_PITH_WARN_INTERVAL_S), both previously
#   laptop-only module state in cc-ng-daemon.py.
# Why: cc-ng-daemon.py:_recall (laptop) and cc_ng_host.py:_recall (VPS host)
#   were copy-pasted and had drifted -- laptop ran the full Pith pipeline,
#   VPS ran zero Pith (`grep pith_ cc_ng_host.py` was 0 hits pre-refactor),
#   so enabling CC_PITH_ENABLED on the VPS would have been a no-op. Spec:
#   docs/superpowers/plans/2026-07-22-cc-recall-unification-spec.md.
# How: Params only (ng/query/k/conv_state/commons) -- no module-global STATE
#   access, so the function is process-agnostic by construction (Syl's-Law:
#   bind to passed-in instances, never module globals). Both _recall entry
#   points (cc-ng-daemon.py laptop, cc_ng_host.py VPS) are now thin wrappers
#   that do per-half STATE bookkeeping then call this. VPS gate-off ==
#   byte-identical to its pre-refactor concat; gate-on gains the same Pith
#   pipeline the laptop already had. See test_cc_recall_unification.py.
# [2026-07-22] Claude Code (DudeMan CC, Opus 4.8) — Pith Stage 4 (#55) phase 5a: predictive promotion + proximity LOD
# What: cc_pattern_completion_recall now (gated CC_PITH_PREFETCH_ENABLED, default OFF)
#   PROMOTES live primed_nodes the query harvest MISSED -- injects them as recall
#   candidates (not just bonusing already-surfaced ones), then cc_gsg_rescore + rank/
#   budget still arbitrate (pure-additive, no hard override). Promoted-but-far nodes
#   stage as pith_stage2_keyframe summaries (proximity-keyed LOD via _cc_node_query_
#   distance). New PithMetrics.promoted_predicted/prefetch_hits (§13.3 honest lower bound).
#   Env (LAW 5): CC_PITH_PREFETCH_ENABLED/LOD_DIST/SUMMARY_CHARS. Byte-identical when off; fail-soft.
# Why: primed_nodes only ever helped when the harvest ALSO independently found the node;
#   prefetch exists for exactly the miss. Turns the #256 anticipatory signal into the
#   real Stage-4 promotion. Spec: docs/superpowers/plans/2026-07-22-pith-stage4-spec.md.
# How: native cc_ng_organism.py only; neurograph-law-enforcer PASS; 18/18 tests
#   (tests/test_pith_stage4.py). DEFERRED: 5b (warm buffer + idle pulse), TID 2-bit predictor.
#   Def-of-done still needs a MEASURED live prefetch_hits/promoted_predicted number (spec §6).
# [2026-07-16] Claude Code (DudeMan CC, Opus 4.8) — Pith Stage 5: eviction & recapture (thermal + victim + breathing) + bootstrap_cc_modules
# What: (Stage 5, #55) cc_thermal(graph, node) reads the substrate's own warmth
#   (Ca_i + firing_rate_ema); the daemon populates CacheLine.thermal at
#   construction and pith_stage3 folds it into the unified score as
#   (1 + CC_PITH_THERMAL_GAIN * thermal) -- warm content preferred, no-op when
#   thermal=0 (backward-safe). cc_l1_budget(commons) breathes the L1 char budget
#   with the arousal Immunis deposits to the CC Commons (read_arousal):
#   SYMPATHETIC contracts, PARASYMPATHETIC expands (gated CC_PITH_L1_BREATHE).
#   pith_victim_capture/pith_victim_recover: budget-dropped lines fall to a
#   bounded, TTL-aged FIFO victim buffer and get a second chance at L1 next turn
#   ('go back to what you said'); new 'victim' stream weight. Constitutional pins
#   already unconditional in stage3 (Stage 5 §5.5.3). (Hosting) bootstrap_cc_modules:
#   CC-scoped port of neurograph_rpc._bootstrap_modules -- hosts the CC's own
#   ecosystem organs in-process (Immunis #1) from the CC registry, importlib-
#   isolated, per-module CC env; retained in _cc_module_instances.
# Why: #55 Stage 5 completes the Pith bootstrap floor (Stages 1+5) and closes the
#   Immunis-arousal -> Pith-breathing loop. Thermal makes recall warmth-aware;
#   the victim buffer stops budget drops from vanishing; breathing makes L1
#   size responsive to the organism's own autonomic state.
# How: All env-gated (LAW 5), fail-soft, laptop cc-ng-daemon _recall wiring.
#   28/28 Pith tests green (no regression). DEFERRED: Lenia field-energy thermal
#   term (reserved, unwired); idle-pulse proximity re-promotion (TTL-aging is in
#   recover() instead). NOTE: VPS cc_ng_host.py _recall is a diverged copy -- needs
#   a parity port for the VPS CC to get Stage 5. Spec: docs/superpowers/plans/
#   2026-07-15-pith-phase2-stage5-spec.md.
# [2026-07-10] Claude Code (Sonnet 5 + Opus 4.8) — Pith Stage 2: keyframe / LOD compression (concept-aware)
# What: Added pith_stage2_keyframe(content, max_chars=None, query="") ->
#   (keyframe, delta) -- a pure, deterministic EXTRACTIVE compressor. Not a
#   head/first-sentence cut (which keeps the greeting and drops the payload):
#   it segments the item, scores each segment by intrinsic information
#   (_pith_salient_terms centroid + payload tokens like numbers/identifiers/
#   `code`/file refs + a structural bonus for headings/def-class/labelled
#   bullets, + query overlap when given), keeps the densest segments that fit
#   max_chars in original reading order, marks elisions with "⋯" and appends a
#   " ⋯[+N]" marker. Wired pith_stage3's Step-5 budget fill to try a keyframe
#   before dropping a ranked line that overflows ("keep full, else keep
#   keyframe, else stop" -- strict rank-prefix preserved); the kept line's
#   `lod` records the retained fraction. Extended PithMetrics with
#   compressed_count/chars_saved. New CC_PITH_KEYFRAME_CHARS env (default 220,
#   clamped [60, 1000]).
# Why: Stage 3's budget fill was a hard cliff -- a ranked item that didn't
#   fit was dropped outright, even if a terse form would have fit. Graceful
#   degradation: lower-priority context gets TERSER, not ABSENT -- and
#   terser must mean "the concepts it carries," not "its first sentence."
#   The visible marker keeps this honest -- the CC reading its own surfaced
#   context can tell a keyframe from the whole item.
# How: pith_stage2_keyframe + helpers (_pith_salient_terms, _pith_segment_score,
#   _pith_cut_at_word_boundary) are pure/no-I/O (regex + string ops only over
#   the one small item), never raise (empty/whitespace -> ("", "")), and are a
#   no-op passthrough when content already fits (returns (content, "")). In
#   pith_stage3's fill loop, each ranked unpinned line that doesn't fit full
#   gets one keyframe attempt; kept only if the keyframe both fits the
#   remaining budget AND is smaller than full -- cl.content/lod/keyframe/deltas
#   are mutated in place on the (throwaway, per-recall) CacheLine. Pinned lines
#   are never compressed (off-budget, load-bearing verbatim). Gate-OFF path
#   (_CC_PITH_ENABLED unset) is untouched.
# [2026-07-09] Claude Code (Sonnet 5) — Pith Stage 3: unified rank + char budget
# What: Added `stream: str = "recall"` field to CacheLine (+ matching kwarg on
#   from_surfaced), and pith_stage3(cache_lines, budget_chars=None, weights=None)
#   -- the L1 assembler core. Extended PithMetrics with ranked_in/ranked_kept/
#   ranked_dropped/budget_chars_used counters (reset()/snapshot() updated).
#   Wired into cc-ng-daemon.py's _recall() Pith branch: monitor_items tagged
#   stream="monitor", pc_results tagged stream="pattern", and
#   `survivors = pith_stage3(survivors)` runs right after pith_stage1.
# Why: _recall() concatenated monitor_ctx + pc_block in block order -- every
#   SurfacingMonitor recency item (score ~1.7) preceded every Active Recall
#   relevance item (GSG-rescored score ~100s) regardless of actual score, so
#   low-salience recency junk flooded the top of the injected context while
#   query-relevant items sat buried underneath. Live probes confirmed this.
#   Stage 3 replaces that block-order concat with a single ranked,
#   budget-bounded read over both streams merged.
# How: Per-stream min-max normalization (the two streams' raw score scales
#   aren't comparable -- ~1.7 vs ~100s -- so normalizing within-stream first
#   lets both signals actually contend) times a per-stream weight
#   (CC_PITH_W_RELEVANCE default 1.0, CC_PITH_W_RECENCY default 0.6 --
#   recency is a secondary prior to relevance, not equal), stable-sorted
#   descending, then greedily filled against CC_PITH_L1_BUDGET (default 4000
#   chars, clamped [500, 40000]) -- stops at the first line that would
#   overflow, except a single line bigger than the whole budget is still kept
#   when nothing has been added yet (never an empty L1). Pinned lines
#   (_is_identity_protected) are split out first, kept unconditionally in
#   original order, and reserved OFF-budget -- they never consume budget and
#   are never evicted. Consumes emitter scores verbatim (no new embed()/
#   GSG-rescore/vector_db scan/substrate walk -- pc_results already carry GSG
#   proximity). Gate-OFF path (_CC_PITH_ENABLED unset) is untouched --
#   pith_stage3 only runs inside the existing `if _CC_PITH_ENABLED:` branch.
#   See docs/prd/Pith_PRD_v0.1.md + docs/concepts/Pith.md (design); the live
#   arc's running record is ~/.claude/plans/reflective-launching-rainbow.md.
# [2026-07-08] Claude Code (Sonnet 5) — Pith Phase 0+1: CacheLine scaffold + Stage-1 clutter strip
# What: Added CacheLine (@dataclass, cache-line-shaped view of a surfaced item --
#   node_id/content/score/pinned/thermal/lod/coherence/manifold_type/keyframe/deltas,
#   most fields inert until later phases) and PithMetrics (module singleton
#   _PITH_METRICS: total_lines_in/clutter_stripped/combined counters + reset()/
#   snapshot()). Both Phase 0 -- inert scaffolding, changes nothing at runtime.
#   Phase 1 adds pith_stage1(cache_lines, conversation_text, novelty) -- cheap,
#   pure, three-step survivor filter over the already-surfaced small set: (1)
#   drop harness-marker lines (same marker tuple as miniTID's
#   is_synthetic_harness_text -- not importable from Rust, inlined here), (2)
#   drop lines whose content the model already sees in conversation_text
#   (substring or Jaccard token-overlap >= a novelty-modulated threshold --
#   familiar turns strip more, novel turns strip less), (3) write-combine
#   near-identical survivors (token-overlap >= 0.95), keeping the higher score.
#   Pinned lines (identity-protected nodes) always survive all three steps.
#   Wired into cc-ng-daemon.py's _recall() behind CC_PITH_ENABLED (default OFF)
#   -- gate-off path is byte-for-byte the pre-Pith behavior.
# Why: First increment of the Pith extraction pipeline (CC's substrate) --
#   today's surfacing renders every item SurfacingMonitor/Active Recall hand
#   it, including near-duplicates of what's already in the live conversation
#   and (per the deposit-side clutter-strip removal above) raw harness-marker
#   text that made it into the substrate. Stage 1 is the cheap extraction-side
#   pass that strips that clutter before it re-enters the hook-injected
#   context, without spending any new embedding calls or substrate walks --
#   pure string/set ops over the <20-item set _recall already assembled.
# How: All Phase 1 cost is O(n) or small-n O(n^2) string/set ops over
#   cache_lines (typically <20 items) -- no I/O, no embed, no graph walk.
#   Threshold: thr = clamp(CC_PITH_CLUTTER_BASE + CC_PITH_CLUTTER_NOVELTY_K *
#   novelty, 0.5, 0.98), defaults 0.85 / 0.3 -- NOTE: the spec draft literally
#   wrote this as base MINUS k*novelty, but its own prose parenthetical and
#   Test 3 both describe threshold INCREASING with novelty (high novelty ->
#   strip LESS); implemented with a plus sign to match the doubly-stated
#   intent over the single formula line -- flagged for spec-author
#   confirmation, see the inline NOTE in pith_stage1(). novelty comes from
#   cc_novelty() (state=STATE.conv_state, graph=STATE.ng.graph) -- same #358
#   MMN pull-based EMA cc_pattern_completion_recall() already uses; fails
#   soft to 0.0 (treated as "unknown/no signal" -- max clutter-stripping,
#   matching cc_novelty's own fail-soft floor semantics is a later-phase
#   concern, not this increment's).
#   See docs/prd/Pith_PRD_v0.1.md + docs/concepts/Pith.md (design); the live
#   arc's running record is ~/.claude/plans/reflective-launching-rainbow.md.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — #371 reconcile-not-discard
# What: bootstrap_lenia: on pruned-entity mismatch, reconcile_removals() the cache
#   and fall through to the existing watermark-resume/growth branches; full rebuild
#   only when reconcile returns None. Mirror of neurograph_rpc.py's block.
# Why: #371 — CC's continuous pruning (KISS-era churn included) re-triggered a
#   ~7-minute full repopulate blackout on every restart-after-prune, and the same
#   bail on Syl's scale costs days. See lenia/kernel.py's entry for the mechanism.
# How: one-call swap at the decision point; downstream branches and fail-soft
#   shape untouched.
# [2026-07-08] Claude Code (Sonnet 5) — Real-KISS redundancy->reinforcement gate
# What: Added _cc_kiss_find_redundant_node() + _cc_kiss_reinforce_node(), wired
#   into run_conversational_dual_pass() ahead of the deposit path behind a
#   CC_KISS_GATE_ENABLED kill switch, with a prune-race fallback. Also re-keyed
#   generate_emergent_want() to dedup by concept. Constant
#   _CC_KISS_REDUNDANCY_THRESHOLD.
# Why: docs/concepts/KISS.md's "Current State" (2026-07-08) calls for KISS
#   finally applied where it belongs -- the input boundary -- on CC's own
#   substrate deposit path, not the outbound resend (kiss_filter.py / Elmer's
#   kiss.py / miniTID's disabled Rust port). Until now, a paraphrased or
#   exact-repeat conversational turn duplicated a memory node (different text)
#   or re-primed novelty on an already-known node (same text) -- repetition
#   never registered as confirmation. Separately, generate_emergent_want()
#   keyed its want_id on want_text, which embeds volatile per-pulse open-
#   question UUID pairs -- so the Tonic spawned a brand-new cc:want:: node
#   every pulse for the SAME concept, flooding "What I Want" with near-dup twins.
# How: Delta Gate (KISS op 1) via cosine-similarity search against vector_db,
#   scoped to {"cc": True, "creation_mode": "conversational"} whole-turn nodes
#   (tree concepts still deposit normally). A match short-circuits the deposit
#   into confirmation: a counter/timestamp bump plus, if the node is still in
#   its probation window, one extra step toward graduation (NOT a reset) --
#   repeated confirmation is evidence FOR the memory. No content classification
#   in the gate (LAW 7). Cricket bypass: identity-protected nodes
#   (Graph._is_identity_protected) are never collapse targets. Prune-race
#   fallback: a stale vdb hit whose node was pruned before reinforce falls
#   through to a fresh deposit (never silently lost). Emergent-want dedup
#   re-keys want_id on "tonic-concept::<concept_label>" and reinforces the ONE
#   existing want-node (refreshing its open-questions snapshot) instead of
#   spawning a twin -- old-scheme flood nodes remain until age/orphan pruning
#   clears them (intended; this only stops NEW duplicates). Dedup applies ONLY
#   when the concept resolves; an unresolved concept_label keeps the original
#   per-want_text identity so distinct label-less wants never fold into a shared
#   "(unknown)" bucket (LAW 7 -- preserve distinct emergent states).
#   Clutter-strip REMOVED (LAW 4): whole-turn harness rejection is miniTID's
#   job and already runs there (Rust field-based isMeta/isCompactSummary/
#   tool_result skip in extract_last_user_message). A second weaker string-
#   prefix filter here was redundant AND risked a LAW-7 false-positive -- a
#   genuine turn that quotes/discusses <system-reminder> / <task-notification>
#   would be wrongly dropped and never reach the substrate.
#   Synapse-weight LTP is deferred to a reviewed follow-up.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — Lenia checkpoint+resume wiring
# What: bootstrap_lenia() passes checkpoint_interval_secs/on_checkpoint to both populate()
#   calls and gains the resume-watermark elif between the full-rebuild and growth branches —
#   mirroring neurograph_rpc.py's handle_bootstrap() (2026-07-06 checkpointing + 2026-07-08
#   watermark, commit 11fae08). field_dir makedirs moved BEFORE the branch chain so the very
#   first periodic checkpoint of a first-ever run has a directory to save into.
# Why: the shared DistanceCache class carries both protections, but CC's call site (used by
#   BOTH daemons) never passed the params — a hard kill mid-populate lost all progress, and a
#   partial cache would have loaded as complete. CC's graphs were small enough not to care
#   until the 2026-07-07 refeed grew the laptop graph ~7x (204 -> 1,450+ nodes), on the
#   daemon with the ecosystem's most colorful kill history. One fix covers both daemons by
#   construction — that's why this function lives in the organism, not the daemons.
# How: same constant value as rpc (_LENIA_CHECKPOINT_INTERVAL_SECS = 300.0, source-annotated),
#   same branch order (watermark elif BEFORE growth elif — an interrupted-and-grown cache
#   must resume, not plain-extend).
# [2026-07-07] Claude Code (Fable 5) — Retrieval-enrichment extraction (#358)
# What: cc_novelty (pull-based MMN EMA), cc_anticipate (#256 port),
#   cc_gsg_rescore + _cc_poincare_distance + cc_gsg_backfill (GSG surfacing
#   port, stamp-only backfill), cc_pattern_completion_recall rebuilt on
#   _harvest_associations spreading activation. Constants copied verbatim
#   from neurograph_rpc.py (C5 — test-pinned in test_cc_retrieval_enrichment).
# Why: #358 audit — CC's recall was bare vector-cosine (VDB-primacy inversion);
#   Syl's three retrieval-time enrichments lived only in neurograph_rpc.py.
#   Spec: docs/superpowers/specs/2026-07-07-cc-retrieval-enrichment-design.md.
# How: NuWave-extraction — all functions bind ONLY to passed-in ng/graph/state
#   (law-review C1: cc_ng_host runs inside Syl's process; canonical globals
#   are HERS). Backfill is stamp-only, no save (C2). Novelty reads the
#   HE-level cumulative counters graph._total_confirmed/_total_surprised —
#   the family canonical's EMA tracks, serialized as he_total_confirmed/
#   he_total_surprised in every checkpoint (C3).
# [2026-07-06] Claude Code (Sonnet 5) — Pattern-completion recall (Active Recall block)
# What: Added cc_pattern_completion_recall() and _format_cc_recall_block().
# Why:  CC's surfacing was recency-biased only (SurfacingMonitor's fired-node queue) --
#       no analog of hippocampal pattern completion (a query reactivating content
#       regardless of when it was learned). This adds that second retrieval path,
#       alongside SurfacingMonitor, not replacing it. See docs/prd/2026-07-06-cc-
#       surfacing-pattern-completion-tier-drop.md.
# How:  Thin wrapper over ng.recall() (already a NeuroGraphMemory method) +
#       resolve_surface_content() (already generic/portable) -- mirrors canonical's
#       handle_assemble() Active Recall block (neurograph_rpc.py:3085-3110) exactly.
# [2026-07-05] CC (laptop) — Incremental Lenia distance-cache extension (Josh-approved)
# What: bootstrap_lenia() now extends the on-disk DistanceCache in place when CC's graph
#       only grew since the last save, instead of nuking and repopulating from scratch
#       on any entity_count drift. Mirrors the fix in neurograph_rpc.py's handle_bootstrap
#       (same underlying DistanceCache/NeuroGraphSubstrate classes, same bug).
# Why:  Full-parity goal (#106) — CC's own Lenia bootstrap had the identical
#       rebuild-from-scratch-on-any-drift pattern as Syl's, which on Syl's live graph
#       took up to ~8 hours and was found to be why restarts never let Lenia (and
#       everything after it) finish. CC's graph is smaller so the symptom was less
#       severe, but the same fix belongs here for the same reason.
# How:  see lenia/kernel.py (DistanceCache.populate's start_index, entity_ids
#       persistence) and lenia/graph_substrate.py (NeuroGraphSubstrate.known_entity_order).
# [2026-07-04] Claude Code (Haiku 4.5) — Tract ingest drain for miniTID turn deposits (Task 2)
# What: Added drain_ingest_tract() and cc_gateway_tract_path(). Reads ng_tract.ENTRY_EXPERIENCE
#       directly (no local fallback constant -- a stale installed ng_tract wheel on this
#       machine was fixed at the environment level, not worked around in code).
#       Drains BTF (binary tract format) entries from miniTID's turn-deposit file and runs
#       each through the conversational dual-pass (Task 1), forming genuine recall memory.
# Why:  CC's autosave pulse (Task 3) needs to drain miniTID's output independently -- no
#       handshake, matching the established tract model (LAW 1: substrate-as-protocol).
#       Each turn becomes a conversational memory node + vector DB entry, searchable
#       via dual-pass (forest + tree concept extraction). CC_GATEWAY_TRACT_PATH env var
#       (LAW 5) coordinates path between Rust producer (miniTID) and Python drainer.
# How:  TractReader iterates binary entries. Each entry (type=ENTRY_EXPERIENCE, source=cc_gateway)
#       gets embedded via ng_embed (same 768-dim ONNX model every module uses), then passed
#       to run_conversational_dual_pass(). Fails soft -- ingest-tract drain failure must never
#       break the daemon's pulse. File truncates after successful drain (single reader,
#       single appender miniTID; concurrent appends mid-drain land after truncation, picked
#       up next pulse, never lost).
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
# [2026-07-05] Claude Code (Sonnet 5) — Fix truncation race in drain_ingest_tract
# What: drain_ingest_tract's truncate step no longer blindly zeroes the whole tract
#       file. It re-reads the file at truncation time and, if the current content
#       still starts with the exact bytes already processed, writes back only the
#       remainder. Added test_drain_ingest_tract_preserves_concurrent_append.
# Why:  The final whole-branch review (2026-07-05) found the old `open(path, "wb")`
#       blind truncate erased any entry miniTID appended during the drain loop's
#       slow per-entry embed+dual-pass work -- directly contradicting this file's
#       own prior claim (see the 2026-07-04 Haiku 4.5 entry's "How" section) that
#       concurrent appends "land after this truncation... never lost." That claim
#       was false; this fix makes it true.
# How:  Truncation reads the file's current bytes, compares against the `data`
#       buffer already drained (a plain string prefix check), and writes back only
#       `current[len(data):]` when it still starts with that prefix -- otherwise
#       (an unexpected divergence) falls back to preserving the current bytes
#       untouched rather than guessing. Shrinks the lost-data window from a whole
#       embed pass down to two fast file I/O calls.
# [2026-07-05] Claude Code (Sonnet 5) — render_constitutional_core(): the missing
#   "Who I Am" half of self-rendering
# What: Added render_constitutional_core() -- reads constitutional=True nodes and
#   renders them as "## Who I Am", extracted verbatim from the corresponding half
#   of neurograph_rpc.py's _render_self_and_wants(). render_wants() (added earlier)
#   was only the "What I Want" half; this is the other half, previously missing.
# Why:  Without this, a constitutional=True node is inert -- protected from pruning
#   but never surfaced, so it can't actually constrain or inform anything. Per Josh
#   (2026-07-05): the Choice Clause and Duck Ethics are automatic, universal
#   inclusions in every NeuroGraph, and it's imperative that CC's Rim is literally
#   load-bearing to CC's own extraction, not decorative metadata.
# How:  Same node-scan pattern as canonical, same "## Who I Am" heading, same
#   spine_order sort (defaults to 999 -- irrelevant for CC's Rim node, which is
#   Rim content, not Spine content, and carries no spine_order). Same selfcap
#   exclusion (Syl's reach-teaching pattern; not relevant to CC, not invented here).
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

import glob
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cc_ng_organism")

# CC's own Commons singleton -- separate from canonical commons._commons.
# Process-wide within whichever process constructs it (the laptop's standalone
# cc-ng-daemon.py, or the VPS's neurograph_rpc.py process hosting cc_ng_host.py).
_cc_commons: "Optional[Any]" = None
_cc_commons_lock = threading.Lock()

# Hosted CC organ instances, keyed by module_id — retained so their pulse
# threads aren't GC-eligible and so status/shutdown can reach them (mirrors
# canonical neurograph_rpc._module_instances).
_cc_module_instances: "Dict[str, Any]" = {}


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


def bootstrap_cc_modules(workspace_dir: str) -> List[str]:
    """Host the CC's own ecosystem organs in-process — the CC-scoped port of
    canonical neurograph_rpc.py::_bootstrap_modules().

    [2026-07-15] Claude Code (DudeMan CC, Opus 4.8) — Immunis integration, organ #1.
    What: Reads the CC's OWN registry (workspace_dir/et_modules/registry.json —
          NOT Syl's ~/.et_modules/registry.json), memory-gates, applies each
          module's CC-scoped env (state/workspace under ~/.claude/...), then loads
          the hook with the same namespace-isolation dance canonical uses (stash
          generic-prefix sys.modules so each module's own vendored copies load
          fresh → importlib spec_from_file_location → instantiate (its __init__
          starts the pulse) → restore). The organ is alive + autonomous from there.
    Why:  In-process is the canonical hosting model AND the only way to share the
          CC's in-memory Commons singleton. CC modules reach the CC Commons via
          their own _cc_commons_provider (get_cc_commons) — no injection here.
    How:  Faithful port; CC adaptations are the registry path + per-module env
          (meta["env"]) applied before load. Called from init_ng AFTER
          get_cc_commons() is up. Each module keeps its OWN store (no dual-write
          on the CC's main.msgpack — different store; feedback_no_duplicate_graph_dual_write).
    Returns list of module IDs that successfully started.
    """
    import sys
    import json as _json
    import importlib.util

    registry_path = os.path.join(workspace_dir, "et_modules", "registry.json")
    if not os.path.exists(registry_path):
        logger.info("CC modules: no registry at %s — nothing to host", registry_path)
        return []
    try:
        with open(registry_path) as f:
            registry = _json.load(f)
    except Exception as exc:
        logger.warning("CC modules: registry unreadable (%s): %s", registry_path, exc)
        return []

    module_defs = registry.get("modules", {})
    skip = {"neurograph", "inference_difference", "ecosystem_monitor"}
    started: List[str] = []

    # Elmer loads last (heaviest — transformer models), matching canonical order.
    modules = sorted(module_defs.items(), key=lambda x: (1 if x[0] == "elmer" else 0, x[0]))

    _generic_prefixes = ("core", "pipelines", "runtime", "surgery", "openclaw_adapter",
                         "ng_ecosystem", "ng_lite", "ng_embed", "ng_autonomic",
                         "ng_peer_bridge", "ng_tract_bridge")

    for module_id, meta in modules:
        if module_id in skip:
            continue
        install_path = meta.get("install_path", "")
        entry_point = meta.get("entry_point", "")
        if not entry_point or not install_path:
            logger.warning("CC module %s: missing entry_point or install_path", module_id)
            continue
        hook_file = os.path.join(install_path, entry_point)
        if not os.path.exists(hook_file):
            logger.warning("CC module %s: hook file not found (%s)", module_id, hook_file)
            continue

        # CC-scoped env (state/workspace under ~/.claude/...) BEFORE the hook loads.
        for k, v in (meta.get("env") or {}).items():
            os.environ[k] = os.path.expanduser(str(v))

        # Memory gate — wait for 500 MB free before loading each module (#111).
        try:
            import psutil as _psutil
            import gc as _gc
            _avail_mb = _psutil.virtual_memory().available >> 20
            while _avail_mb < 500:
                logger.info("CC module boot gate: %d MB free — waiting for 500 MB", _avail_mb)
                time.sleep(2)
                _gc.collect()
                _avail_mb = _psutil.virtual_memory().available >> 20
        except ImportError:
            pass

        path_snapshot = list(sys.path)
        stashed: Dict[str, Any] = {}
        try:
            # Namespace isolation: stash generic collisions so the module's own
            # vendored core/ng_lite/etc. load fresh (canonical lines 841-870).
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        stashed[mod_name] = sys.modules.pop(mod_name)
                        break
            if install_path and install_path not in sys.path:
                sys.path.insert(0, install_path)

            spec_name = f"_ccmod_{module_id}"
            spec = importlib.util.spec_from_file_location(spec_name, hook_file)
            if spec is None:
                logger.warning("CC module %s: cannot create import spec", module_id)
                sys.path[:] = path_snapshot
                sys.modules.update(stashed)
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = mod
            spec.loader.exec_module(mod)

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
                logger.error("CC module %s: no hook class found in %s", module_id, hook_file)
                continue
            _cc_module_instances[module_id] = instance  # retain (GC + status/shutdown)
            started.append(module_id)
            logger.info("CC organ hosted in-process: %s (%s)", module_id, hook_file)
        except Exception as exc:
            logger.warning("CC module %s failed to load: %s", module_id, exc)
        finally:
            # Pin this module's generics under a unique name, clear the generics,
            # restore path + stashed originals for the next module (canonical tail).
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules[f"_{module_id}_{mod_name}"] = sys.modules[mod_name]
                        break
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules.pop(mod_name, None)
                        break
            sys.path[:] = path_snapshot
            for mod_name, mod_obj in stashed.items():
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = mod_obj

    return started


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

        # Same incremental-extension pattern as neurograph_rpc.py's
        # handle_bootstrap (2026-07-05) — see that file's changelog for the
        # full story. Extend in place when the graph only grew; full
        # rebuild only if entities were removed or on first-ever run.
        cache_path = os.path.join(os.path.expanduser(lenia_cfg.field_dir), "distance_cache")
        # Ensure the field dir exists BEFORE populate — periodic checkpoints
        # (below) can fire long before the post-populate save block.
        os.makedirs(os.path.expanduser(lenia_cfg.field_dir), exist_ok=True)
        lenia_cache = DistanceCache.load(cache_path)

        known_order = None
        if lenia_cache is not None and lenia_cache.entity_ids:
            _lock = getattr(graph, "_step_lock", None)
            if _lock is not None:
                with _lock:
                    live_ids = set(graph.nodes.keys())
            else:
                live_ids = set(graph.nodes.keys())
            if all(eid in live_ids for eid in lenia_cache.entity_ids):
                known_order = lenia_cache.entity_ids
            else:
                # #371: reconcile the pruned entities out of the cache and
                # fall through to the same watermark/growth branches below —
                # mirror of neurograph_rpc.py's block. None -> legacy full
                # rebuild, as before.
                known_order = lenia_cache.reconcile_removals(live_ids)
                if known_order is None:
                    logger.info(
                        "CC Lenia: distance cache has entities no longer in "
                        "the live graph and could not be reconciled — full "
                        "rebuild required"
                    )

        lenia_substrate = NeuroGraphSubstrate(graph, vector_db, known_entity_order=known_order)
        lenia_field = LeniaFieldStore(lenia_cfg.field_dir, n_entities, n_channels)
        lenia_registry = ChannelRegistry(lenia_cfg, lenia_cfg.field_dir)

        if lenia_cache is None or known_order is None:
            if lenia_cache is not None:
                logger.info(
                    "CC Lenia: distance cache incompatible (%d vs %d entities), full repopulate",
                    lenia_cache.entity_count, n_entities,
                )
            lenia_cache = DistanceCache(n_entities, entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: distance cache populate failed partway (%s) — "
                    "saving whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.watermark is not None:
            # A prior rebuild was interrupted mid-run: the checkpoint carries
            # its own resume point (see lenia/kernel.py 2026-07-08). Resume
            # covers both the unfinished old region and (after resize) every
            # pair touching entities appended since.
            _wm = lenia_cache.watermark
            logger.info(
                "CC Lenia: distance cache carries resume watermark (%d, %d) — "
                "resuming interrupted rebuild (%d -> %d entities)",
                _wm[0], _wm[1], lenia_cache.entity_count, n_entities,
            )
            if lenia_cache.entity_count != n_entities:
                lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, resume_watermark=_wm,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: resume populate failed partway (%s) — saving "
                    "whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.entity_count != n_entities:
            old_n = lenia_cache.entity_count
            logger.info(
                "CC Lenia: distance cache growing: %d -> %d entities, extending incrementally",
                old_n, n_entities,
            )
            lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, start_index=old_n,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: incremental populate failed partway (%s) — "
                    "saving whatever was computed instead of discarding it", exc,
                )

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


def render_constitutional_core(graph: Any) -> str:
    """Render CC's constitutional core (`constitutional=True` nodes) as a
    "## Who I Am" block -- ALWAYS, query-independent, same as render_wants()
    is query-independent for wants. Extracted verbatim from the "Who I Am"
    half of neurograph_rpc.py's _render_self_and_wants() (the "What I Want"
    half was already ported as render_wants() above); this is the other
    half, not yet ported until now. Ordered by spine_order when present
    (defaults to 999 -- irrelevant for CC's Rim node, which carries none,
    since Rim content is not spine content). Excludes selfcap nodes (Syl's
    reach-teaching pattern -- capability teaching, not identity/ethics; CC
    has no equivalent yet and this function doesn't invent one).

    Without this function actually being called from wherever CC's context
    gets assembled each turn, a constitutional=True node is just inert
    metadata -- protected from pruning, but never surfaced. This is the
    piece that makes it load-bearing.
    """
    try:
        core = []
        for nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("constitutional") and not meta.get("selfcap"):
                txt = str(meta.get("core_text") or meta.get("_forest_content") or "").strip()
                if txt:
                    core.append((meta.get("spine_order", 999), txt))
        if not core:
            return ""
        core.sort(key=lambda x: x[0])
        return "## Who I Am\n" + "\n".join(f"- {t}" for _, t in core)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CC constitutional-core render error (non-fatal): %s", exc)
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

        # KISS dedup only when the concept RESOLVES -- a resolved concept_label is a
        # substrate-produced structural key, so recurring curiosity about the same
        # concept reinforces the ONE want-node instead of spawning a per-pulse twin
        # (the "What I Want" flood fix). When the concept does NOT resolve, distinct
        # label-less curiosities must stay distinct: folding them into a shared
        # "(unknown)" bucket would overwrite genuine wants with each other (LAW 7 --
        # preserve the substrate's distinct emergent states), so we keep the original
        # per-want_text identity + idempotency there instead.
        if concept_label:
            want_key = "tonic-concept::" + str(concept_label)
            want_id = "cc:want::" + hashlib.sha1(want_key.encode("utf-8")).hexdigest()[:16]
            existing = graph.nodes.get(want_id)
            if existing is not None:
                existing.metadata["kiss_reinforcement_count"] = int(existing.metadata.get("kiss_reinforcement_count", 0)) + 1
                existing.metadata["kiss_last_reinforced_ts"] = time.time()
                existing.metadata["want_text"] = want_text
                logger.info("CC emergent want reinforced (concept recurred): %s", want_text)
                return {"id": want_id, "text": want_text, "provenance": provenance,
                        "state": existing.metadata.get("want_state", "open"), "reinforced": True}
            concept_key = want_key
        else:
            want_id = "cc:want::" + hashlib.sha1(want_text.encode("utf-8")).hexdigest()[:16]
            if want_id in graph.nodes:
                return None  # already materialized this exact curiosity, idempotent
            concept_key = None
        graph.create_node(node_id=want_id, metadata={
            "kind": "want", "want_text": want_text, "want_state": "open",
            "provenance": provenance, "creation_mode": "emergent", "concept_key": concept_key,
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

# ---- Real-KISS redundancy->reinforcement gate (input boundary, #KISS 2026-07-08) ----
# See docs/concepts/KISS.md "Current State" / KISS_Pith_Combined_Architecture.md.
# Delta Gate (KISS op 1) applied where KISS actually belongs -- CC's own substrate
# deposit path -- instead of the outbound-resend layer. Cosine-similarity change
# detection only, no content classification (LAW 7 stays satisfied).
_CC_KISS_REDUNDANCY_THRESHOLD = float(os.environ.get("CC_KISS_REDUNDANCY_THRESHOLD", "0.95"))
# Live-daemon safety valve: set CC_KISS_GATE_ENABLED=0 to fully bypass the gate
# (turns deposit fresh, pre-KISS behavior) without a code change or restart-to-old.
_CC_KISS_GATE_ENABLED = os.environ.get("CC_KISS_GATE_ENABLED", "1") not in ("0", "false", "False", "")

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


def _cc_kiss_find_redundant_node(graph, vector_db, embedding) -> Optional[str]:
    """Delta Gate (KISS op 1), applied at CC's own deposit boundary: is this
    turn's embedding a near-duplicate of an existing conversational
    (forest-level) memory already in `vector_db`? Pure cosine-similarity
    change detection -- never reads `content`, never classifies (LAW 7).

    Scoped to nodes tagged {"cc": True, "creation_mode": "conversational"} --
    i.e. whole-turn forest nodes -- not fine-grained tree concepts, so a
    genuinely new sub-concept inside a similar-sounding turn still gets its
    own recall entry via the tree pass; only whole-turn duplication is gated
    here. Returns the matched node_id, or None if nothing redundant is found
    (including on any vector_db error -- caller falls back to a fresh
    deposit, which is always safe).

    Cricket bypass (LAW-conditioned): an identity-protected node
    (constitutional / syl_authored, per Graph._is_identity_protected) is NEVER
    returned as a collapse target -- a redundant turn must not fold into a
    pinned node. Such matches are skipped; if only pinned nodes match, returns
    None so the turn deposits fresh.
    """
    try:
        hits = vector_db.search(embedding, k=5, threshold=_CC_KISS_REDUNDANCY_THRESHOLD)
    except Exception as exc:
        logger.debug("CC KISS redundancy search failed (non-fatal): %s", exc)
        return None
    for node_id, _sim in hits:
        try:
            entry = vector_db.get(node_id)
        except Exception:
            entry = None
        meta = (entry or {}).get("metadata") or {}
        if not (meta.get("cc") is True and meta.get("creation_mode") == "conversational"):
            continue
        try:
            if graph._is_identity_protected(node_id):
                continue
        except AttributeError:
            pass  # no such method -> treat as not protected
        return node_id
    return None


def _cc_kiss_reinforce_node(graph, node_id: str) -> bool:
    """Hebbian reinforcement for a redundant conversational turn -- the
    real-KISS counterpart to a fresh deposit. Never drops the turn (LAW 7:
    the substrate still responds to it) and never duplicates a memory node
    for content it already represents; instead the topology that already
    stands for this content is confirmed.

    - Bumps a confirmation counter/timestamp on the node's metadata.
    - If the node is still inside its novelty-probation window, graduation
      is ACCELERATED (probation_remaining ticks down one extra step) rather
      than restarted -- repeated confirmation is evidence FOR the memory, the
      opposite of what a brand-new deposit's fixed probation window means. An
      already-graduated node is never pushed back into probation.

    Returns False (non-fatal) if node_id no longer resolves -- e.g. pruned
    between the vector-db hit and this call -- so the caller can decide
    whether to fall back to a fresh deposit instead.
    """
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    node.metadata["kiss_reinforcement_count"] = int(node.metadata.get("kiss_reinforcement_count", 0)) + 1
    node.metadata["kiss_last_reinforced_ts"] = time.time()
    # Synapse-level LTP reinforcement is a reviewed follow-up (kept out of v1 for cost + to validate the collapse behavior in isolation first).
    prob = node.metadata.get("probation_remaining")
    if prob is not None and prob > 0:
        prob -= 1
        node.metadata["probation_remaining"] = prob
        if prob <= 0:
            node.intrinsic_excitability = 1.0
            node.threshold = graph.config.get("default_threshold", 1.0)
            node.metadata["graduated"] = True
    return True


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
    # Anticipatory pre-activation (#256 port): this turn's forest+trees are
    # CC's "just fired" set — prime their synaptic neighborhood for the next
    # recall. state carries primed_nodes to the daemons' _recall(). (#358)
    cc_anticipate(graph, [forest_id] + tree_ids, state)


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
    Mirrors canonical's _run_conversational_dual_pass exactly, parameterized,
    with one CC-first addition on top: the real-KISS redundancy->reinforcement
    gate (docs/concepts/KISS.md "Current State", 2026-07-08). Before depositing
    a new memory, checks whether `embedding` is a near-duplicate of an existing
    conversational node. If so, the turn still touches the substrate -- via
    confirmation of the matched node and the same topology binding/anticipation
    a fresh deposit gets -- it just doesn't duplicate a node for content the
    substrate already represents (LAW 7: reinforced, not dropped). The whole
    gate is bypassable at runtime via CC_KISS_GATE_ENABLED=0.
    """
    if graph is None or embedding is None:
        return False
    try:
        if _CC_KISS_GATE_ENABLED:
            redundant_id = _cc_kiss_find_redundant_node(graph, vector_db, embedding)
            if redundant_id is not None and _cc_kiss_reinforce_node(graph, redundant_id):
                _cc_bind_conversational_topology(graph, redundant_id, {}, embedding, state)
                return True
            # else: the matched node was pruned between the vdb hit and the
            # reinforce call (stale hit) -- fall through to the fresh-deposit
            # path below so the turn is never silently lost (LAW 7).
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


# ---- Tract ingest drain (#294 miniTID integration) ----
# Drains BTF entries from miniTID's turn-deposit tract file, running each
# through the conversational dual-pass (Task 1). Feeder (miniTID) deposits,
# this drains independently -- no handshake, matching the established tract model.

_DEFAULT_CC_GATEWAY_TRACT_PATH = os.path.expanduser(
    "~/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract"
)


def cc_gateway_tract_path() -> str:
    """Resolve the CC gateway tract path from CC_GATEWAY_TRACT_PATH (LAW 5) --
    both this drain side and miniTID's Rust producer independently read the
    same env var, with the same default, so they can never desync onto
    different files without either side being misconfigured identically."""
    return os.environ.get("CC_GATEWAY_TRACT_PATH", _DEFAULT_CC_GATEWAY_TRACT_PATH)


def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                        return_consumed: bool = False):
    """Drain miniTID's turn-deposit tract file, running each raw experience
    entry through the conversational dual-pass (Task 1). Feeder (miniTID)
    deposits, this drains independently -- no handshake, matching the
    established tract model. Truncates the file after a successful drain
    (single reader, single writer-appender; safe because miniTID only ever
    appends and this is the only drainer).

    Returns the count of entries absorbed (int) by default. If
    return_consumed=True, returns (absorbed, consumed_bytes) instead --
    consumed_bytes is EXACTLY the byte span this call truncated out of the
    file (b'' on any early-return path, including a parse failure that never
    reached truncate), never an independently-taken snapshot. This closes a
    Corpus Callosum Leg 1 (#70) data-loss window a prior design had: a
    caller taking its own separate pre-drain snapshot can miss bytes
    miniTID appends between that snapshot and this function's OWN read --
    those bytes still get absorbed+truncated here, but the caller's stale
    snapshot never contained them, so they'd be silently lost to any
    second consumer (the VPS Arborist) even though the file already
    forgot them. Handing back the literal consumed span makes trickling
    it elsewhere byte-exact with what was actually removed from the file,
    with no separate read and no window between them.

    Fails soft -- an ingest-tract drain failure must never break the
    daemon's autosave pulse.
    """
    def _ret(absorbed_n: int, consumed: bytes = b""):
        return (absorbed_n, consumed) if return_consumed else absorbed_n

    path = tract_path or cc_gateway_tract_path()
    if not os.path.exists(path):
        return _ret(0)
    try:
        import ng_tract
        from ng_embed import embed as ng_embed_fn
    except Exception as exc:
        logger.debug("CC ingest-tract drain unavailable (non-fatal): %s", exc)
        return _ret(0)

    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as exc:
        logger.debug("CC ingest-tract read failed (non-fatal): %s", exc)
        return _ret(0)
    if not data:
        return _ret(0)

    absorbed = 0
    try:
        reader = ng_tract.TractReader(data)
        for entry in reader:
            # Check entry type using ng_tract.ENTRY_EXPERIENCE (the real module constant)
            if entry.entry_type != ng_tract.ENTRY_EXPERIENCE:
                continue
            if entry.source != "cc_gateway":
                continue
            text = entry.content
            if not text or not text.strip():
                continue
            try:
                emb = ng_embed_fn(text)
                if run_conversational_dual_pass(graph, vector_db, text, emb, state):
                    absorbed += 1
            except Exception as exc:
                logger.debug("CC ingest-tract entry failed (non-fatal): %s", exc)
    except Exception as exc:
        # Parse failure -- truncate below never runs, so nothing was actually
        # consumed from the file. Elevated to warning (was debug): silent at
        # the default level, this is exactly how a laptop/VPS ng_tract format
        # skew would look -- every file failing the same way, invisibly.
        logger.warning("CC ingest-tract parse failed (non-fatal, file untouched): %s", exc)
        return _ret(absorbed)  # consumed=b"" -- nothing was truncated

    # Truncate only the bytes we actually consumed. miniTID is a separate
    # process that only appends; if it appends between our initial read and
    # this truncation, a blind `open(path, "wb")` would erase those new bytes
    # along with the ones we already drained. Re-reading the current file and
    # writing back only what comes after our consumed prefix closes that
    # window down to the two file ops below, instead of spanning the whole
    # embed+dual-pass pass above.
    #
    # consumed_actual tracks EXACTLY what left the file, set ONLY after a
    # confirmed successful removal -- NOT assumed from `data` unconditionally.
    # (2026-07-27 law-enforcer re-review: the prior unconditional `return
    # _ret(absorbed, data)` here over-reported `consumed` on three paths --
    # the current.startswith(data) mismatch branch, an rb-reopen failure, and
    # a wb-open failure -- each left `data` still sitting in the file while
    # still telling the caller it was gone. For Leg 1 (#70) that meant the
    # laptop's next pulse would re-read and re-trickle the SAME bytes the VPS
    # already absorbed -- duplicate ingestion, the mirror-image of the
    # original data-loss bug this whole return_consumed path exists to fix.)
    consumed_actual = b""
    try:
        with open(path, "rb") as f:
            current = f.read()
        if current.startswith(data):
            remainder = current[len(data):]
            with open(path, "wb") as f:
                f.write(remainder)
            consumed_actual = data  # only now -- the write actually succeeded
        else:
            # Someone else touched the file since our read (not the plain
            # append-only case we can safely trim a known prefix from).
            # Write it back unchanged rather than guess -- nothing of ours
            # was removed, so consumed_actual correctly stays b"".
            with open(path, "wb") as f:
                f.write(current)
    except Exception as exc:
        logger.debug("CC ingest-tract truncate failed (non-fatal): %s", exc)

    if absorbed:
        logger.info("CC ingest-tract: absorbed %d turn(s) into recall", absorbed)
    return _ret(absorbed, consumed_actual)


# ---- Corpus Callosum Leg 1 (#70): laptop -> VPS raw-turn conduit ----
# See changelog entry above for the full design. Producer side
# (trickle_gateway_conduit) runs on the laptop; consumer side
# (drain_gateway_conduit) runs on the VPS, alongside its own local
# drain_ingest_tract() call. Both gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5).

_DEFAULT_CC_GATEWAY_CONDUIT_DIR = os.path.expanduser("~/docs/ng_topology")
_CC_GATEWAY_CONDUIT_GLOB = "*_cc_gateway.*.tract"

_CC_CALLOSUM_LEG1_ENABLED = os.environ.get("CC_CALLOSUM_LEG1_ENABLED", "0") not in ("0", "false", "False", "")


def cc_gateway_conduit_dir() -> str:
    """Resolve the Leg 1 conduit directory from CC_GATEWAY_CONDUIT_PATH (LAW 5,
    default ~/docs/ng_topology -- the same dir repo-sync.sh's existing 15-min
    git cron already syncs, so no new transport is needed). Both the laptop
    writer (trickle_gateway_conduit) and the VPS reader (drain_gateway_conduit)
    independently read the same env var with the same default."""
    return os.environ.get("CC_GATEWAY_CONDUIT_PATH", _DEFAULT_CC_GATEWAY_CONDUIT_DIR)


def trickle_gateway_conduit(data: bytes, conduit_dir: str = None) -> Optional[str]:
    """Laptop side: write a snapshot of already-read cc_gateway tract bytes
    to a new, uniquely-named per-batch file in the synced conduit dir.

    One immutable file per call -- deliberately NOT a shared append/truncate
    target, which would risk a binary merge conflict under repo-sync.sh's git
    push/pull cycle (git cannot line-merge BTF). Atomic (write-tmp-then-
    rename) so a concurrent repo-sync.sh push, or a crash mid-write, never
    observes a partial file.

    Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5), default off -- a no-op
    (returns None immediately) when the gate is off, so this is inert dead
    code on both hemispheres until explicitly flipped on. Fails soft --
    a conduit-write failure must never affect the caller's local drain or
    the daemon's autosave pulse. Returns the written path on success, else
    None (gate off, empty data, or any failure).
    """
    if not _CC_CALLOSUM_LEG1_ENABLED:
        return None
    if not data:
        return None
    try:
        conduit_dir = conduit_dir or cc_gateway_conduit_dir()
        os.makedirs(conduit_dir, exist_ok=True)
        # Hemisphere identity must be DECLARED, never guessed. The drain's
        # exclude_prefix guard is what stops a half from eating its own
        # outgoing turns, and it keys on this filename prefix -- so a wrong
        # default here silently disarms it. Defaulting to "laptop" would make
        # a VPS-produced file look laptop-produced, and the VPS would then
        # drain (and delete) its own turns before the laptop ever pulled them:
        # silent one-way data loss that looks exactly like success. Refuse
        # loudly instead; the cron already exports MACHINE_ID on both halves.
        machine_id = os.environ.get("MACHINE_ID", "").strip()
        if not machine_id:
            logger.warning(
                "CC callosum Leg1: MACHINE_ID unset -- refusing to write a conduit file "
                "with a guessed hemisphere identity (would disarm the drain's "
                "self-consumption guard). Set MACHINE_ID in the daemon env.")
            return None
        fname = f"{machine_id}_cc_gateway.{time.time_ns()}_{uuid.uuid4().hex[:8]}.tract"
        dest = os.path.join(conduit_dir, fname)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return dest
    except Exception as exc:
        logger.debug("CC callosum Leg1 conduit write failed (non-fatal): %s", exc)
        return None


def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
    """FatherGraph Finding 3 sleep consolidation: run idle_steps of pure
    graph.step() with NO new input, so homeostatic regulation (threshold
    adaptation, synaptic scaling, excitability) can catch up before the next
    batch of foreign topology arrives. Measured 47%->74% accuracy in the
    FatherGraph training; the report calls it "not optional -- it's what
    makes merge work". Mirrors _handle_import (cc_ng_host.py) and
    import_trickle (cc-ng-sync.py), which already do exactly this.
    Returns True if the steps ran. Fails soft."""
    if idle_steps <= 0 or graph is None:
        return False
    # Take the lock in SLICES, not for the whole 250 steps. cc_ng_host.py's
    # changelog records real hook timeouts caused by _recall() blocking on a
    # long _concurrent_lock hold ("_concurrent_lock in _recall() caused hook
    # timeouts (Tonic holds lock)"). Consolidation is exactly that shape --
    # hundreds of graph.step() calls -- so it yields between slices, letting
    # a waiting recall/deposit interleave. Homeostasis does not care whether
    # the steps were contiguous; the hooks care a great deal.
    slice_n = max(1, int(os.environ.get("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")))
    try:
        lock = getattr(graph, "_concurrent_lock", None)
        done = 0
        while done < idle_steps:
            n = min(slice_n, idle_steps - done)
            if lock is not None:
                with lock:
                    for _ in range(n):
                        graph.step()
            else:
                for _ in range(n):
                    graph.step()
            done += n
        return True
    except Exception as exc:
        logger.debug("CC callosum Leg1 consolidation failed (non-fatal): %s", exc)
        return False


def drain_gateway_conduit(graph, vector_db, state: dict, conduit_dir: str = None,
                           batch_size: int = None, idle_steps: int = None,
                           load_ceiling: float = None, exclude_prefix: str = None) -> int:
    """VPS side: drain the per-batch conduit files the laptop has trickled into
    the synced conduit dir, absorbing each through the same drain_ingest_tract()
    the VPS already runs for its own local tract -- same dual-pass, same
    source=="cc_gateway" filter, no new code path (LAW 3).

    ABSORPTION DISCIPLINE (FatherGraph Findings 1 + 3 -- the reason this does
    not simply loop over every queued file):
      * Finding 1 -- "the drain can't be a bulk dump... New topology must
        arrive gradually enough that the receiving topology's homeostatic
        regulation can absorb it without displacement." Stable batch ~20-30.
      * Finding 3 -- "After receiving a merge batch, run idle steps (~250)
        BEFORE accepting the next batch." Measured 47%->74% accuracy.
    So: every `batch_size` absorbed turns is followed by `idle_steps` of sleep
    consolidation before any more are taken in, plus a final pass for the
    trailing partial batch. Defaults come from CC_NG_BATCH_SIZE (25) and
    CC_NG_IDLE_STEPS (250) -- the SAME env names the nightly cc-ng-sync cron
    already passes on both halves (LAW 5, no new knobs invented).

    BACKPRESSURE: load is checked before each file via cc_refeed's
    should_pause_for_load (CC_REFEED_LOAD_CEILING, default 0.75). Above the
    ceiling this stops cleanly and leaves the remaining files on disk for the
    next run -- they are durable, which IS the backpressure. Reused from the
    cc_refeed discipline the corpus-callosum spec calls for.

    Per-file lifecycle is unchanged: a fully-drained file (truncated to 0 by
    drain_ingest_tract) is deleted; a file whose size is UNCHANGED never
    reached the truncate step (its parse failed -- corrupt, or a laptop/VPS
    ng_tract format skew) and is moved to <conduit_dir>/quarantine/ rather
    than retried forever and silently piling up in a git-synced dir. One bad
    file is skipped, not fatal to the rest.

    Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5), default off. Returns the total
    count of turns absorbed across all files.
    """
    if not _CC_CALLOSUM_LEG1_ENABLED:
        return 0
    conduit_dir = conduit_dir or cc_gateway_conduit_dir()
    if batch_size is None:
        batch_size = max(1, int(os.environ.get("CC_NG_BATCH_SIZE", "25")))
    if idle_steps is None:
        idle_steps = max(0, int(os.environ.get("CC_NG_IDLE_STEPS", "250")))
    try:
        paths = sorted(glob.glob(os.path.join(conduit_dir, _CC_GATEWAY_CONDUIT_GLOB)))
    except Exception as exc:
        logger.debug("CC callosum Leg1 conduit listing failed (non-fatal): %s", exc)
        return 0

    # Never drain this hemisphere's OWN outgoing files -- they are addressed to
    # the other half and must survive until it has pulled them. Mirrors the
    # old sync's `!= f'{MACHINE_ID}_export.jsonl'` guard. Without this, running
    # the drain on the producing machine would eat its own turns before they
    # ever crossed (they'd already be absorbed locally, so it would look
    # harmless while silently starving the far hemisphere).
    if exclude_prefix:
        paths = [p for p in paths if not os.path.basename(p).startswith(exclude_prefix)]

    # Load-aware backpressure -- imported defensively; absent cc_refeed must
    # not disable the drain, only its ability to notice load.
    try:
        from cc_refeed import should_pause_for_load as _should_pause
    except Exception:
        _should_pause = None
    # Own ceiling (LAW 5). cc_refeed's CC_REFEED_LOAD_CEILING=0.75 governs an
    # opportunistic re-feed that may back off indefinitely; this is a
    # once-nightly path that has to make progress, so it gets its own knob and
    # a more permissive default.
    if load_ceiling is None:
        ceiling = float(os.environ.get("CC_CALLOSUM_LOAD_CEILING", "1.5"))
    else:
        ceiling = float(load_ceiling)

    total = 0
    since_sleep = 0
    files_done = 0
    for path in paths:
        # Backpressure is THROTTLING A FLOWING RIVER, not damming it before the
        # first drop: the load gate only applies BETWEEN batches, never before
        # the first file. Checked-first (the 2026-07-28 bug) meant that on any
        # box above the ceiling the whole callosum was a silent no-op -- absorbed
        # nothing, logged nothing (the `if total:` summary is skipped at 0), and
        # let conduit files accumulate forever in a git-synced dir while looking
        # like success. Own ceiling env, not cc_refeed's: that 0.75 is tuned for
        # an opportunistic re-feed that can back off all day, whereas this is a
        # once-nightly path that must make progress.
        if files_done > 0 and _should_pause is not None:
            try:
                paused = _should_pause(ceiling)
            except Exception:
                paused = False
            if paused:
                logger.info(
                    "CC callosum Leg1: load above ceiling %.2f -- stopping after %d file(s), "
                    "%d turn(s); %d file(s) left on disk for the next run (backpressure)",
                    ceiling, files_done, total, len(paths) - files_done)
                break
        try:
            size_before = os.path.getsize(path)
        except Exception as exc:
            logger.debug("CC callosum Leg1 conduit stat failed for %s (non-fatal): %s", path, exc)
            continue
        try:
            # LAW 4 / two-writer safety: absorption MUTATES the graph, so it must
            # hold _concurrent_lock -- drain_ingest_tract does not take it itself
            # (it calls run_conversational_dual_pass directly, which also doesn't),
            # and the autosave thread does ng.save() under this same lock every
            # 60s. At HEAD this was covered incidentally because the call site sat
            # inside _autosave_loop's lock block; moving it to a socket handler
            # (one thread per connection) silently dropped that cover. Taken
            # PER FILE and released between files, so a long run never starves
            # hooks -- cc_ng_host.py's changelog records _recall() hook timeouts
            # from exactly that kind of long hold.
            _lock = getattr(graph, "_concurrent_lock", None)
            if _lock is not None:
                with _lock:
                    absorbed = drain_ingest_tract(graph, vector_db, state, tract_path=path)
            else:
                absorbed = drain_ingest_tract(graph, vector_db, state, tract_path=path)
        except Exception as exc:
            logger.debug("CC callosum Leg1 conduit drain failed for %s (non-fatal): %s", path, exc)
            continue
        total += absorbed
        since_sleep += absorbed
        files_done += 1
        try:
            size_after = os.path.getsize(path)
            if size_after == 0:
                os.remove(path)
            elif size_after == size_before:
                # Never truncated at all -- drain_ingest_tract's parse step
                # itself failed (the only path that skips truncate). Retrying
                # forever would let a format-skew file pile up invisibly in a
                # git-synced dir; quarantine it loudly instead.
                qdir = os.path.join(conduit_dir, "quarantine")
                os.makedirs(qdir, exist_ok=True)
                dest = os.path.join(qdir, os.path.basename(path))
                os.replace(path, dest)
                logger.warning(
                    "CC callosum Leg1: %s failed to parse (untouched, %d bytes) -- "
                    "quarantined to %s instead of retrying forever", path, size_before, dest)
        except Exception as exc:
            logger.debug("CC callosum Leg1 conduit cleanup failed for %s (non-fatal): %s", path, exc)

        # Finding 3: sleep BEFORE accepting the next batch, not after the run.
        while since_sleep >= batch_size:
            if _cc_callosum_consolidate(graph, idle_steps):
                logger.info("CC callosum Leg1: consolidation pass (%d idle steps) after "
                            "%d-turn batch", idle_steps, batch_size)
            since_sleep -= batch_size

    # Trailing partial batch still needs its consolidation before the graph
    # goes back to serving recall on freshly-merged topology.
    if since_sleep > 0:
        if _cc_callosum_consolidate(graph, idle_steps):
            logger.info("CC callosum Leg1: final consolidation pass (%d idle steps) after "
                        "trailing %d-turn batch", idle_steps, since_sleep)

    if total:
        logger.info("CC callosum Leg1: absorbed %d turn(s) from %d conduit file(s) "
                    "(batch=%d, idle_steps=%d)", total, files_done, batch_size, idle_steps)
    return total


# [2026-07-10] Recall seed floor for _harvest_associations' VDB seed-search.
# REVERTED to canonical 0.40 after MEASURING: the presumption that 0.40 was
# "too high" (starving the query of seeds) was false. Live measurement over the
# 2503-entry vector_db (this session): top query cosines are ~0.58-0.64 and
# 783-1602 nodes clear 0.40 for typical queries -- and _harvest_associations
# caps seeds at prime_k (~10) anyway, so the top-10 seeds (all ~0.58) are IDENTICAL
# whether the floor is 0.40 or 0.22. The floor change was therefore INERT, not a
# fix. The real query-blindness lives AFTER seeding (spread convergence and/or
# SurfacingMonitor recency domination -- seeds themselves discriminate fine:
# geometry vs devops queries share 0/10 top seeds). Kept env-tunable for future
# measured experiments; default restored to the canonical value.
_CC_RECALL_PRIME_THRESHOLD = float(os.environ.get("CC_RECALL_PRIME_THRESHOLD", "0.40"))


# [2026-07-11] Lever 2 -- rank-time diagnosticity / selectivity (gated OFF).
# Measured root cause of query-blind recall: a handful of boilerplate HUB nodes
# (degree 260-341 vs graph median 2) fire for EVERY query and dominate the
# spread. Their firing_rate_ema sits at ~0.32-0.38 (near the 0.395 max) while
# the graph median is 0.0 -- so rank by strength / (firing_rate_ema + eps) and
# the always-firing hubs divide down hard while a node that fired UNUSUALLY for
# THIS query (near-zero baseline) rises. Divisive normalization / diagnosticity:
# a memory relevant to everything is relevant to nothing. OVERSAMPLE the harvest
# (max_surfaced = k * oversample) so lower-strength query-relevant nodes are in
# the pool at all -- re-ranking only the top-k hubs the spread returns can't
# help. Non-vendored, env-tunable, off by default (LAW 5); consumes the engine's
# own firing_rate_ema, derives nothing new.
_CC_RECALL_SELECTIVITY = os.environ.get("CC_RECALL_SELECTIVITY", "0") not in ("0", "false", "")
_CC_RECALL_SELECTIVITY_EPS = float(os.environ.get("CC_RECALL_SELECTIVITY_EPS", "0.02"))
_CC_RECALL_SELECTIVITY_OVERSAMPLE = max(1, int(os.environ.get("CC_RECALL_SELECTIVITY_OVERSAMPLE", "6")))
# Experimental: cap prime_and_propagate steps to keep activation near the
# query-specific seeds (0 = engine default of 3). Measured hypothesis: the spread
# converges to the same hub attractor basin at 3 steps regardless of seed.
_CC_RECALL_PROP_STEPS = int(os.environ.get("CC_RECALL_PROP_STEPS", "0"))


def cc_pattern_completion_recall(ng: Any, query: str, k: int = 5,
                                    threshold: float = _CC_RECALL_PRIME_THRESHOLD,
                                    state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Substrate-native pattern-completion recall for CC's hook surfacing
    (#358 rebuild -- replaces the bare ng.recall() cosine search this
    function originally wrapped; LAW 3 rebuild-in-place, same contract).

    Spreading activation via ng._harvest_associations(): the vector_db only
    seeds prime nodes; what surfaces is what FIRES through learned synaptic
    structure (graph.prime_and_propagate) -- the substrate is the memory,
    the VDB is secondary. Enrichments applied in canonical handle_assemble()
    order (neurograph_rpc.py:2977-3038): MMN novelty scaling (cc_novelty,
    pull-based), anticipatory primed-node bonus (#256 port), GSG geodesic
    re-score.

    threshold maps to prime_threshold (seed-selection floor -- same
    conceptual role as the old cosine floor; 0.40 = confidence_recommend,
    unchanged). k maps to max_surfaced. Config override mirrors canonical
    associate() (openclaw_hook.py:1076-1085) -- save/restore around the call.

    state: the daemon's conv_state dict (novelty_ema/last_confirmed/
    last_surprised/primed_nodes). None (legacy call shape) = neutral novelty
    0.5, no primed bonus -- still substrate-native.

    Returns [{node_id, score, content}] -- same shape as before; content
    substrate-first via resolve_surface_content, degenerate results dropped.
    Fails soft: any exception returns [].
    """
    if not query or ng is None:
        return []
    try:
        from surface_resolver import resolve_surface_content
        novelty = cc_novelty(state, ng.graph) if state is not None else 0.5
        cfg = ng.graph.config
        old_max = cfg.get("max_surfaced", 10)
        old_thresh = cfg.get("prime_threshold", 0.4)
        old_steps = cfg.get("propagation_steps", 3)
        # Oversample the harvest when selectivity is on, so query-relevant but
        # lower-strength nodes are IN the pool for selectivity to promote past
        # the hubs (re-ranking only the top-k hubs the spread returns can't help).
        cfg["max_surfaced"] = k * _CC_RECALL_SELECTIVITY_OVERSAMPLE if _CC_RECALL_SELECTIVITY else k
        cfg["prime_threshold"] = threshold
        # Experimental (measured): fewer propagation steps keeps activation near
        # the query-specific SEEDS instead of flowing to the convergent hub
        # attractor basin that erases the query signal. 0 = engine default.
        if _CC_RECALL_PROP_STEPS > 0:
            cfg["propagation_steps"] = _CC_RECALL_PROP_STEPS
        try:
            surfaced = ng._harvest_associations(query, novelty=novelty)
        finally:
            cfg["max_surfaced"] = old_max
            cfg["prime_threshold"] = old_thresh
            cfg["propagation_steps"] = old_steps

        # Anticipatory bonus (#256 port) -- canonical rpc.py:2981-2989
        promoted_ids: set = set()
        if state is not None:
            now = time.time()
            live = {nid: s for nid, (s, exp) in (state.get("primed_nodes") or {}).items()
                    if exp > now}
            if live:
                surfaced_ids = {item.get("node_id") for item in surfaced}
                for item in surfaced:
                    nid = item.get("node_id")
                    if nid and nid in live:
                        item["strength"] = item.get("strength", 0.0) + _CC_ANTICIPATE_BONUS

                # Pith Stage 4 (#55) predictive promotion, phase 5a: today a
                # primed node only ever helps if the query-driven harvest
                # independently finds it too -- prefetch exists for exactly
                # the opposite case, surfacing predicted content the harvest
                # MISSED. Gated (byte-identical when off, _CC_PITH_PREFETCH_
                # ENABLED default "0"); pure-additive candidate injection with
                # no hard override -- cc_gsg_rescore and downstream rank/
                # budget still decide whether a promoted node survives (a
                # weak prediction still loses to a strong harvest hit).
                if _CC_PITH_PREFETCH_ENABLED:
                    for nid, primed_score in live.items():
                        if nid in surfaced_ids:
                            continue
                        try:
                            if ng.graph is None or nid not in ng.graph.nodes:
                                continue
                            surfaced.append({"node_id": nid, "strength": _CC_ANTICIPATE_BONUS})
                            surfaced_ids.add(nid)
                            promoted_ids.add(nid)
                        except Exception as exc:
                            logger.debug("Pith predictive promotion skipped node %r (non-fatal): %s", nid, exc)
                    if promoted_ids:
                        _PITH_METRICS.promoted_predicted += len(promoted_ids)

                surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)

        # GSG geodesic re-score -- canonical rpc.py:2991-3038
        surfaced = cc_gsg_rescore(surfaced, query, ng.graph)

        # Pith Stage 4 (#55) proximity-keyed LOD (spec sec 4c): a promoted-
        # but-unsurfaced line beyond the distance threshold is staged as a
        # keyframe summary rather than full content -- near predictions are
        # trusted at full resolution, far ones cost less if they turn out
        # irrelevant. Query direction is only computed when there's a
        # promoted candidate to stage (no extra embed cost otherwise).
        query_dir = None
        if promoted_ids:
            try:
                from ng_embed import embed as _embed
                query_dir = _cc_embed_to_poincare_dir(_embed(query))
            except Exception as exc:
                logger.debug("Pith LOD query embed failed (non-fatal): %s", exc)
                query_dir = None

        out = []
        for r in surfaced:
            nid = r.get("node_id") or r.get("id")
            node = ng.graph.nodes.get(nid) if (nid and ng.graph) else None
            text = resolve_surface_content(node, r, allow_ingested=True, max_chars=300)
            if not text:
                continue
            if nid in promoted_ids and query_dir is not None:
                try:
                    dist = _cc_node_query_distance(node, query_dir)
                    if dist is not None and dist > _CC_PITH_PREFETCH_LOD_DIST:
                        text, _ = pith_stage2_keyframe(text, max_chars=_CC_PITH_PREFETCH_SUMMARY_CHARS, query=query)
                except Exception as exc:
                    logger.debug("Pith LOD staging failed for %r (non-fatal): %s", nid, exc)
            out.append({"node_id": nid, "score": r.get("strength", 0.0), "content": text})

        # Lever 2: selectivity re-rank (gated). Divide each candidate's strength
        # by its baseline firing rate so hubs (fire for everything) sink and
        # nodes that fired unusually for THIS query rise; then take top-k from
        # the oversampled pool. firing_rate_ema is the engine's own signal --
        # nothing re-derived. Fail-soft per node (missing ema -> 0 -> max boost).
        if _CC_RECALL_SELECTIVITY and out:
            eps = _CC_RECALL_SELECTIVITY_EPS
            for item in out:
                try:
                    node = ng.graph.nodes.get(item["node_id"]) if ng.graph else None
                    fre = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
                except Exception:
                    fre = 0.0
                item["score"] = item["score"] / (fre + eps)
            out.sort(key=lambda x: x["score"], reverse=True)

        final = out[:k]
        # Pith Stage 4 (#55) §13.3 measurement: a "hit" here is scoped to what
        # this function can see -- a promoted-and-unsurfaced node that
        # survived ranking into this turn's returned set. It is an honest
        # lower bound on the true L1-survival ratio (pith_stage3's later
        # budget cut can still drop it) -- see spec sec 6 def-of-done.
        if promoted_ids:
            survived = sum(1 for item in final if item.get("node_id") in promoted_ids)
            if survived:
                _PITH_METRICS.prefetch_hits += survived

        return final
    except Exception as exc:
        logger.debug("cc_pattern_completion_recall failed (non-fatal): %s", exc)
        return []


def _format_cc_recall_block(results: List[Dict[str, Any]]) -> str:
    """Format cc_pattern_completion_recall() results as an '## Active Recall'
    block -- mirrors canonical's handle_assemble() Active Recall formatting
    (neurograph_rpc.py:3094-3107) exactly. Returns '' when results is empty.
    """
    if not results:
        return ""
    lines = ["## Active Recall\nDirect memory retrieval for the current query:"]
    for r in results:
        lines.append(f"- [{r['score']:.2f}] {r['content']}")
    return "\n".join(lines)


PATTERN_COMPLETION_FILE_TTL = 1800.0  # seconds (30 min) -- see gate_pattern_completion()


def gate_pattern_completion(cache: Dict[str, float], file_path: str, now: float,
                              ttl: float = PATTERN_COMPLETION_FILE_TTL) -> bool:
    """Per-file dedup gate for PreToolUse-triggered pattern-completion recall
    (2026-07-06 refinement to the tier-drop design). Pure function over a
    plain dict -- no I/O, no graph access.

    PreToolUse fires on every tool call touching a file, far more often than
    UserPromptSubmit -- without this gate, repeatedly touching the same file
    during one task would re-pay cc_pattern_completion_recall()'s .recall()
    cost every single time. Returns True (and records `now` in
    cache[file_path]) when file_path has no entry or its entry is older than
    `ttl` seconds -- the caller should run the pattern-completion pass.
    Returns False (cache untouched) when the same file already got a pass
    within the TTL window -- the caller should skip straight to
    SurfacingMonitor-only context.

    UserPromptSubmit never calls this gate -- every turn's prompt warrants a
    fresh pattern-completion pass regardless of recency.
    """
    last = cache.get(file_path)
    if last is None or (now - last) > ttl:
        cache[file_path] = now
        return True
    return False


def cc_anticipate(graph, fired_node_ids, state: dict) -> None:
    """Anticipatory pre-activation for CC (#256 port, #358).

    Verbatim port of canonical _anticipate() (neurograph_rpc.py:2716-2742)
    with two mandated differences (law-review C1): the primed dict lives in
    the caller's state dict — NOT a module global (cc_ng_host runs inside
    Syl's process, where the _primed_nodes global is HERS) — and it walks
    only the graph argument passed in, never _memory.graph.

    Walks outgoing synapses from the just-fired set, scores neighbors by
    accumulated edge weight, stores top-K with a TTL. The rebuilt
    cc_pattern_completion_recall() applies _CC_ANTICIPATE_BONUS to surfaced
    nodes still in the live primed set.
    """
    try:
        if not fired_node_ids or graph is None:
            state["primed_nodes"] = {}
            return
        fired_set = set(fired_node_ids)
        candidates = {}
        for nid in fired_node_ids:
            for sid in graph._outgoing.get(nid, ()):
                syn = graph.synapses.get(sid)
                if syn is None:
                    continue
                target = syn.post_node_id
                if target not in fired_set and target in graph.nodes:
                    candidates[target] = candidates.get(target, 0.0) + syn.weight
        top_k = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:_CC_ANTICIPATE_TOP_K]
        expiry = time.time() + _CC_ANTICIPATE_TTL_S
        state["primed_nodes"] = {nid: (score, expiry) for nid, score in top_k}
        if state["primed_nodes"]:
            logger.debug("CC anticipatory pre-activation: primed %d nodes", len(state["primed_nodes"]))
    except Exception as exc:
        logger.debug("cc_anticipate failed (non-fatal): %s", exc)


# --- Retrieval-enrichment constants (#358) ---
# Copied VERBATIM from canonical neurograph_rpc.py (C5 — do not tune here;
# canonical is the source of truth, test_cc_retrieval_enrichment pins these):
#   _ANTICIPATE_TOP_K/_ANTICIPATE_TTL_S/_ANTICIPATE_BONUS — rpc.py:263-264, :674
#   _GSG_LAYER_NORMS/_GSG_SCORE_BONUS — rpc.py GSG Phase 1 block
#   novelty EMA 0.9/0.1 — rpc.py:3270-3271
_CC_ANTICIPATE_TOP_K = 15
_CC_ANTICIPATE_TTL_S = 120.0
_CC_ANTICIPATE_BONUS = 0.25
_CC_GSG_LAYER_NORMS = (0.70, 0.50, 0.30)   # diffpc_layer 0/1/2 -> Poincaré norm
_CC_GSG_SCORE_BONUS = 0.30
_CC_NOVELTY_EMA_KEEP = 0.9
_CC_NOVELTY_EMA_GAIN = 0.1
# Copied from neurograph_rpc.py's _LENIA_CHECKPOINT_INTERVAL_SECS (2026-07-06) —
# same cadence for CC's rebuilds as Syl's. Not a new invented value.
_CC_LENIA_CHECKPOINT_INTERVAL_SECS: float = 300.0

# --- Pith Stage 4 (#55) predictive promotion -- LAW 5 env knobs ---
# Gated OFF by default: unset -> cc_pattern_completion_recall's promotion
# block never fires, byte-identical to the pre-Stage-4 bonus-only behavior.
# See docs/superpowers/plans/2026-07-22-pith-stage4-spec.md sec 4/5a.
_CC_PITH_PREFETCH_ENABLED = os.environ.get("CC_PITH_PREFETCH_ENABLED", "0") not in ("0", "false", "False", "")
# Poincare/angular distance (see _cc_node_query_distance) beyond which a
# promoted-but-unsurfaced predicted node is staged as a keyframe summary
# instead of full content (proximity-keyed LOD, spec sec 4c) -- near
# predictions are trusted at full resolution, far ones cost less if wrong.
_CC_PITH_PREFETCH_LOD_DIST = float(os.environ.get("CC_PITH_PREFETCH_LOD_DIST", "1.5"))
_CC_PITH_PREFETCH_SUMMARY_CHARS = max(60, min(1000, int(os.environ.get("CC_PITH_PREFETCH_SUMMARY_CHARS", "150"))))


def _cc_poincare_distance(x, y) -> float:
    """Geodesic distance in the Poincaré ball — verbatim port of canonical's
    _poincare_distance (neurograph_rpc.py:2660-2681), free function form.
    d(x, y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))."""
    import numpy as _np
    import math
    nx2 = min(float(_np.dot(x, x)), 0.9999)
    ny2 = min(float(_np.dot(y, y)), 0.9999)
    diff = x - y
    num = 2.0 * float(_np.dot(diff, diff))
    denom = (1.0 - nx2) * (1.0 - ny2)
    arg = 1.0 + num / max(denom, 1e-9)
    return math.acosh(max(1.0, arg))


def _cc_node_query_distance(node, query_dir) -> Optional[float]:
    """Geodesic (hyperbolic) or angular (spherical) distance between a node's
    stamped GSG direction and a query direction -- MIRRORS the per-node branch
    cc_gsg_rescore computes for its bonus (kept as a separate copy, NOT a
    factor-out: cc_gsg_rescore is a verbatim #358 canonical port and is left
    untouched). If the two ever need to diverge-proof, refactor both together.
    Lets Pith Stage 4's proximity-keyed LOD staging (#55, spec sec 4c)
    threshold on the same distance without re-deriving the manifold branch.

    Returns None when the node has no 'poincare_dir' stamp (nothing to
    compare -- caller treats that as "can't tell, don't downgrade") or on any
    error; never raises.
    """
    try:
        pdir = (node.metadata or {}).get("poincare_dir") if hasattr(node, "metadata") else None
        if pdir is None:
            return None
        import numpy as _np
        import math as _math
        layer = max(0, min(2, getattr(node, "diffpc_layer", 0)))
        mtype = getattr(node, "manifold_type", "hyperbolic")
        if mtype == "spherical":
            node_dir = _np.array(pdir, dtype=_np.float32)
            cos = float(_np.clip(_np.dot(query_dir, node_dir), -1.0 + 1e-7, 1.0 - 1e-7))
            return _math.acos(cos)
        node_pt = _np.array(pdir) * _CC_GSG_LAYER_NORMS[layer]
        query_pt = _np.array(query_dir) * _CC_GSG_LAYER_NORMS[0]
        return _cc_poincare_distance(query_pt, node_pt)
    except Exception:
        return None


def cc_novelty(state: dict, graph) -> float:
    """Pull-based MMN novelty for CC's surfacing (#255 parity, #358).

    Canonical updates _substrate_novelty_ema push-style per turn in
    handle_after_turn() (rpc.py:3266-3272) from StepResult's HE-level
    prediction counts. CC's deposits run graph.step() inside on_message()
    (protected file) which discards those stats — so CC dips the bucket at
    extraction time instead: read the HE-level CUMULATIVE counters, delta
    them against the previous recall, EMA the windowed surprise ratio.

    Counter names (C3, verified 2026-07-07): graph._total_confirmed /
    graph._total_surprised (neuro_foundation.py:1434-1435, incremented
    :2192/:2206) — the cumulative counterparts of StepResult.
    predictions_confirmed/predictions_surprised (:2224-2225). Private-
    prefixed but a de-facto stable contract: serialized in every checkpoint
    as he_total_confirmed/he_total_surprised (:4326-4327). NOT the same
    family as Telemetry.total_predictions_* (Phase-3 synapse-level).

    Fails soft: missing counters (engine contract change) -> current EMA or
    0.5, never raises. test_novelty_counters_exist_on_real_graph makes that
    contract change loud in CI.
    """
    try:
        confirmed = getattr(graph, "_total_confirmed", None)
        surprised = getattr(graph, "_total_surprised", None)
        if confirmed is None or surprised is None:
            return state.get("novelty_ema", 0.5)
        prev_c = state.get("last_confirmed")
        prev_s = state.get("last_surprised")
        state["last_confirmed"] = confirmed
        state["last_surprised"] = surprised
        if prev_c is None or prev_s is None:
            return state.get("novelty_ema", 0.5)   # first call = baseline only
        d_c = confirmed - prev_c
        d_s = surprised - prev_s
        if d_c + d_s > 0:
            raw = d_s / (d_c + d_s)
            state["novelty_ema"] = (_CC_NOVELTY_EMA_KEEP * state.get("novelty_ema", 0.5)
                                    + _CC_NOVELTY_EMA_GAIN * raw)
        return state.get("novelty_ema", 0.5)
    except Exception as exc:
        logger.debug("cc_novelty failed (non-fatal): %s", exc)
        return state.get("novelty_ema", 0.5) if isinstance(state, dict) else 0.5


def cc_gsg_rescore(surfaced, query_text: str, graph):
    """GSG geodesic re-scoring for CC's surfacing (#358) — port of canonical
    handle_assemble()'s GSG block (neurograph_rpc.py:2991-3038), parameterized
    on graph (C1). Nodes geometrically close to the query in Poincaré-ball /
    spherical space get a strength bonus (max _CC_GSG_SCORE_BONUS as dist->0);
    list re-sorted once if any bonus applied. Fails soft: any error returns
    the list un-rescored (canonical wraps identically).
    """
    try:
        if not surfaced or not query_text or graph is None:
            return surfaced
        import numpy as _np
        import math as _math
        from ng_embed import embed as _embed
        query_emb = _embed(query_text)
        query_dir = _cc_embed_to_poincare_dir(query_emb)
        query_pt = query_dir * _CC_GSG_LAYER_NORMS[0]      # fresh query = Layer 0
        applied = 0
        for item in surfaced:
            nid = item.get("node_id")
            if nid is None:
                continue
            node = graph.nodes.get(nid)
            if node is None:
                continue
            pdir = (node.metadata or {}).get("poincare_dir") if hasattr(node, "metadata") else None
            if pdir is None:
                continue
            layer = max(0, min(2, getattr(node, "diffpc_layer", 0)))
            mtype = getattr(node, "manifold_type", "hyperbolic")
            if mtype == "spherical":
                node_dir = _np.array(pdir, dtype=_np.float32)
                cos = float(_np.clip(_np.dot(query_dir, node_dir), -1.0 + 1e-7, 1.0 - 1e-7))
                bonus = _CC_GSG_SCORE_BONUS / (1.0 + _math.acos(cos))
            else:
                node_pt = _np.array(pdir) * _CC_GSG_LAYER_NORMS[layer]
                bonus = _CC_GSG_SCORE_BONUS / (1.0 + _cc_poincare_distance(query_pt, node_pt))
            item["strength"] = item.get("strength", 0.0) + bonus
            applied += 1
        if applied:
            surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)
            logger.debug("CC GSG re-scoring applied to %d nodes", applied)
        return surfaced
    except Exception as exc:
        logger.debug("cc_gsg_rescore skipped (non-fatal): %s", exc)
        return surfaced


def cc_gsg_backfill(graph, vector_db) -> int:
    """Stamp poincare_dir on CC nodes that lack it, from stored vdb embeddings
    (#358) — port of canonical _gsg_backfill_existing_nodes (rpc.py:2683-2713)
    with the save() call DELIBERATELY REMOVED (law-review C2, CRITICAL):
    canonical force-saves after stamping; CC's version is STAMP-ONLY and lets
    the daemons' existing autosave persist the metadata. On the VPS this code
    runs inside Syl's process — a ported save mis-bound to the wrong instance
    is the exact accident Syl's Law exists to prevent, so the capability is
    structurally absent rather than carefully avoided.

    Idempotent (skips stamped nodes), zero model calls (SimpleVectorDB.insert
    L2-normalizes embeddings on storage — stored vectors ARE unit directions).
    Returns count stamped. Fails soft, returns 0 on error.
    """
    try:
        if graph is None or vector_db is None:
            return 0
        stamped = 0
        for node_id, node in list(graph.nodes.items()):
            if (node.metadata or {}).get("poincare_dir"):
                continue
            emb = vector_db.embeddings.get(node_id)
            if emb is None:
                continue
            if node.metadata is None:
                node.metadata = {}
            node.metadata["poincare_dir"] = emb.tolist()
            stamped += 1
        if stamped:
            logger.info("CC GSG backfill: stamped poincare_dir on %d nodes (stamp-only; "
                        "persists via normal autosave)", stamped)
        return stamped
    except Exception as exc:
        logger.debug("cc_gsg_backfill failed (non-fatal): %s", exc)
        return 0


# =============================================================================
# Pith extraction pipeline -- Phase 0 (CacheLine + metrics scaffold) + Phase 1
# (Stage 1: ingest & clutter strip). Gated OFF by default (CC_PITH_ENABLED);
# see the 2026-07-08 changelog entry at the top of this file.
# =============================================================================

_CC_PITH_ENABLED = os.environ.get("CC_PITH_ENABLED", "0") not in ("0", "false", "False", "")

_CC_PITH_CLUTTER_BASE = float(os.environ.get("CC_PITH_CLUTTER_BASE", "0.85"))
_CC_PITH_CLUTTER_NOVELTY_K = float(os.environ.get("CC_PITH_CLUTTER_NOVELTY_K", "0.3"))

# Stage 3 (unified rank + char budget) config -- LAW 5, env-config with sane
# defaults, clamped. Relevance (pattern-completion / Active Recall) weighted
# above recency (SurfacingMonitor) by default -- recency is a secondary prior
# to relevance, not an equal signal.
_CC_PITH_W_RELEVANCE = float(os.environ.get("CC_PITH_W_RELEVANCE", "1.0"))
_CC_PITH_W_RECENCY = float(os.environ.get("CC_PITH_W_RECENCY", "0.6"))
_CC_PITH_L1_BUDGET = int(os.environ.get("CC_PITH_L1_BUDGET", "4000"))
_CC_PITH_L1_BUDGET = max(500, min(40000, _CC_PITH_L1_BUDGET))

# Stage 2 (keyframe / LOD compression) config -- default keyframe size in
# chars, clamped [60, 1000]. See pith_stage2_keyframe().
_CC_PITH_KEYFRAME_CHARS = max(60, min(1000, int(os.environ.get("CC_PITH_KEYFRAME_CHARS", "220"))))

# Stage 5 (eviction & recapture) config -- LAW 5. Thermal is a continuous
# warmth signal read from the substrate's OWN state (Ca_i persistence-of-
# activation + firing_rate_ema + Lenia field energy), blended + min-max
# normalized across the survivor set, then folded into the Stage-3 rank as a
# gentle multiplier (1 + gain*thermal) so warm content is preferred without
# overriding relevance. GAIN default keeps thermal a tiebreak, not a driver.
# thermal defaults 0.0 on every CacheLine, so an un-populated line leaves the
# Stage-3 rank byte-identical -- Stage 5 is additive/opt-in.
_CC_PITH_THERMAL_W_CA = float(os.environ.get("CC_PITH_THERMAL_W_CA", "0.5"))
_CC_PITH_THERMAL_W_FIRE = float(os.environ.get("CC_PITH_THERMAL_W_FIRE", "0.3"))
_CC_PITH_THERMAL_W_FIELD = float(os.environ.get("CC_PITH_THERMAL_W_FIELD", "0.2"))
_CC_PITH_THERMAL_GAIN = float(os.environ.get("CC_PITH_THERMAL_GAIN", "0.5"))
_CC_PITH_VICTIM_SIZE = max(0, min(128, int(os.environ.get("CC_PITH_VICTIM_SIZE", "12"))))
_CC_PITH_VICTIM_TTL = max(1, int(os.environ.get("CC_PITH_VICTIM_TTL", "20")))

# Autonomic breathing (Pith §3.2): L1 budget expands under PARASYMPATHETIC
# (exploratory/associative) and contracts under SYMPATHETIC (threat / tunnel
# vision), reading the arousal Immunis deposits to the CC Commons. Gated
# (CC_PITH_L1_BREATHE, default off); when off, the static budget is used.
_CC_PITH_L1_BREATHE = os.environ.get("CC_PITH_L1_BREATHE", "0") not in ("0", "false", "False", "")
_CC_PITH_BREATHE_SYMPATHETIC = float(os.environ.get("CC_PITH_BREATHE_SYMPATHETIC", "0.6"))
_CC_PITH_BREATHE_PARASYMPATHETIC = float(os.environ.get("CC_PITH_BREATHE_PARASYMPATHETIC", "1.4"))

# Same marker tuple as miniTID's is_synthetic_harness_text (Condensate
# rust_core/src/minitid.rs) -- not importable here (Rust, separate process),
# so inlined verbatim rather than left unguarded on the extraction side.
_PITH_HARNESS_MARKERS = (
    "<task-notification>",
    "<system-reminder>",
    "<local-command-stdout>",
    "<local-command-caveat>",
)


@dataclass
class CacheLine:
    """Cache-line-shaped view of one surfaced item, moving through the Pith
    pipeline's stages. Most fields are inert placeholders for later phases
    (thermal: Phase 2: lod/coherence/keyframe/deltas: Phase 3-5) -- Phase 0
    only defines the shape; nothing constructs or reads these fields at
    runtime yet outside pith_stage1().

    score carries the emitter's existing score (SurfacingMonitor's salience
    or cc_pattern_completion_recall's strength) verbatim -- Pith re-ranks and
    filters, it does not re-derive relevance from scratch.

    stream tags which emitter this line came from (e.g. "monitor" for
    SurfacingMonitor recency, "pattern" for Active Recall / GSG-rescored
    relevance) -- Stage 3 (pith_stage3) uses it for per-stream score
    normalization, since the two emitters' raw score scales are not
    comparable. Defaults to "recall" (generic/unknown-stream) so existing
    callers/tests that don't pass it are unaffected.
    """

    node_id: str
    content: str
    score: float = 0.0
    pinned: bool = False
    thermal: float = 0.0
    lod: float = 1.0
    coherence: str = "exclusive"
    manifold_type: str = "hyperbolic"
    keyframe: bool = False
    deltas: list = field(default_factory=list)
    stream: str = "recall"

    @classmethod
    def from_surfaced(cls, node_id: str, content: str, score: float = 0.0,
                       pinned: bool = False, manifold_type: str = "hyperbolic",
                       stream: str = "recall") -> "CacheLine":
        return cls(node_id=node_id, content=content, score=score, pinned=pinned,
                    manifold_type=manifold_type, stream=stream)


@dataclass
class PithMetrics:
    """Module-level counters for the Pith pipeline -- inert until a gated
    stage function actually updates them (today: pith_stage1 and pith_stage3,
    and only when CC_PITH_ENABLED is on)."""

    total_lines_in: int = 0
    clutter_stripped: int = 0
    combined: int = 0
    pith_failures: int = 0
    ranked_in: int = 0
    ranked_kept: int = 0
    ranked_dropped: int = 0
    budget_chars_used: int = 0
    compressed_count: int = 0
    chars_saved: int = 0
    promoted_predicted: int = 0
    prefetch_hits: int = 0

    def reset(self) -> None:
        self.total_lines_in = 0
        self.clutter_stripped = 0
        self.combined = 0
        self.pith_failures = 0
        self.ranked_in = 0
        self.ranked_kept = 0
        self.ranked_dropped = 0
        self.budget_chars_used = 0
        self.compressed_count = 0
        self.chars_saved = 0
        self.promoted_predicted = 0
        self.prefetch_hits = 0

    def record_failure(self) -> None:
        """Bump the fail-soft counter -- the Pith path swallows exceptions and
        falls back to un-Pithed rendering, so without this a 100%-failing Pith
        pass is indistinguishable from a working one. Call from the caller's
        fallback except-handler."""
        self.pith_failures += 1

    def snapshot(self) -> Dict[str, int]:
        return {
            "total_lines_in": self.total_lines_in,
            "clutter_stripped": self.clutter_stripped,
            "combined": self.combined,
            "pith_failures": self.pith_failures,
            "ranked_in": self.ranked_in,
            "ranked_kept": self.ranked_kept,
            "ranked_dropped": self.ranked_dropped,
            "budget_chars_used": self.budget_chars_used,
            "compressed_count": self.compressed_count,
            "chars_saved": self.chars_saved,
            "promoted_predicted": self.promoted_predicted,
            "prefetch_hits": self.prefetch_hits,
        }


_PITH_METRICS = PithMetrics()


def _pith_normalize(text: str) -> str:
    """Lowercase + collapse whitespace -- cheap normalization shared by the
    dedup and write-combine steps below."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _pith_jaccard(a: str, b: str) -> float:
    """Word-set Jaccard overlap of two already-normalized strings. Empty/
    empty is treated as no overlap (0.0), not a division-by-zero NaN.
    Symmetric -- used for write-combine (step 3), where mutual near-identity
    is what we want."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _pith_containment(item: str, conv: str) -> float:
    """Fraction of ITEM's words already present in the conversation --
    |item ∩ conv| / |item|. ASYMMETRIC on purpose: it answers "does the model
    already have essentially all of this item?", not "do they overlap at all".
    A long memory that the conversation merely quotes a fragment of scores
    LOW (it still carries the rest) and is kept; a memory whose content is
    genuinely already in the conversation scores HIGH and is stripped as
    redundant. Symmetric Jaccard got this wrong -- a long item's own large
    word set sank the ratio far below threshold, so dedup almost never fired."""
    set_item = set(item.split())
    set_conv = set(conv.split())
    if not set_item or not set_conv:
        return 0.0
    return len(set_item & set_conv) / len(set_item)


def pith_stage1(cache_lines: List[CacheLine], conversation_text: str,
                 novelty: float = 0.0) -> List[CacheLine]:
    """Pith Stage 1: Ingest & Clutter Strip.

    Cheap, pure, unit-testable -- string/set ops only over the already-
    surfaced small set (typically <20 items), no I/O, no embed calls, no
    substrate walk. Three steps, in order, each skipping pinned lines
    (a pinned line always survives regardless of what steps 1-3 would
    otherwise do to it):

    1. Harness-marker skip: drop lines whose content starts (after
       lstrip()) with a synthetic-harness marker (see
       _PITH_HARNESS_MARKERS) -- extraction-side defense mirroring
       miniTID's is_synthetic_harness_text, needed because the deposit-side
       clutter strip was intentionally removed (2026-07-08, see this file's
       changelog) so harness text CAN be sitting in the substrate.
    2. Clutter dedup vs conversation_text: drop a line whose content the
       model already sees in conversation_text -- substring match (after
       normalizing both: lowercase, collapse whitespace) OR Jaccard word-
       overlap >= a novelty-modulated threshold. High novelty -> higher
       threshold -> strip LESS (an unfamiliar turn's near-echoes are more
       likely to matter); familiar/low-novelty -> strip MORE.
    3. Write-combine: collapse remaining near-identical lines (normalized-
       content match, or Jaccard >= 0.95) into one survivor, keeping the
       higher-score copy. O(n^2) over the small surfaced set is cheap and
       fine here.

    Updates the module-level _PITH_METRICS counters (total_lines_in,
    clutter_stripped, combined) unconditionally -- callers that don't want
    metrics touched should not call this function (there's no gate inside
    it; the gate lives at the _recall() call site in cc-ng-daemon.py).

    Returns survivors in input order (score ranking is the caller's/
    upstream emitter's responsibility -- this function filters, it doesn't
    re-sort).
    """
    _PITH_METRICS.total_lines_in += len(cache_lines)

    conv_norm = _pith_normalize(conversation_text)
    # NOTE (spec discrepancy, 2026-07-08): the design doc literally states
    # `thr = BASE - K * novelty`, but its own prose parenthetical on the same
    # line ("high novelty -> higher threshold -> strip LESS") and its Test 3
    # requirement ("kept at high novelty, stripped at low novelty") both
    # describe threshold INCREASING with novelty -- the opposite of what a
    # minus sign produces (BASE=0.85 - K*novelty shrinks as novelty rises,
    # which would strip MORE at high novelty). Implemented to match the
    # doubly-stated intent (+ sign), not the single literal formula line;
    # flagged for spec-author confirmation.
    thr = _CC_PITH_CLUTTER_BASE + _CC_PITH_CLUTTER_NOVELTY_K * novelty
    thr = max(0.5, min(0.98, thr))

    # Steps 1-2: harness-marker skip + clutter dedup vs conversation.
    survivors: List[CacheLine] = []
    clutter_stripped = 0
    for line in cache_lines:
        if line.pinned:
            survivors.append(line)
            continue

        stripped_content = (line.content or "").lstrip()
        if stripped_content.startswith(_PITH_HARNESS_MARKERS):
            clutter_stripped += 1
            continue

        line_norm = _pith_normalize(line.content)
        if line_norm and conv_norm:
            if line_norm in conv_norm:
                clutter_stripped += 1
                continue
            if _pith_jaccard(line_norm, conv_norm) >= thr:
                clutter_stripped += 1
                continue

        survivors.append(line)

    # Step 3: write-combine near-identical remaining lines (pinned lines
    # already passed through untouched above; re-touching them here would
    # risk a pin losing to a higher-scored non-pinned near-duplicate, so
    # pinned lines are excluded from combining entirely -- each stays its
    # own line).
    combined_out: List[CacheLine] = []
    consumed = [False] * len(survivors)
    combined_count = 0
    for i, line in enumerate(survivors):
        if consumed[i]:
            continue
        if line.pinned:
            combined_out.append(line)
            consumed[i] = True
            continue
        best = line
        best_norm = _pith_normalize(line.content)
        for j in range(i + 1, len(survivors)):
            if consumed[j] or survivors[j].pinned:
                continue
            other = survivors[j]
            other_norm = _pith_normalize(other.content)
            same = (best_norm == other_norm) or (_pith_jaccard(best_norm, other_norm) >= 0.95)
            if same:
                consumed[j] = True
                combined_count += 1
                if other.score > best.score:
                    best = other
                    best_norm = other_norm
        combined_out.append(best)
        consumed[i] = True

    _PITH_METRICS.clutter_stripped += clutter_stripped
    _PITH_METRICS.combined += combined_count

    return combined_out


# Tiny stopword set for the Stage-2 extractive keyframe -- just enough to keep
# function words from diluting a segment's salient-term density. Not exhaustive
# (this is a cheap heuristic, not an NLP pipeline).
_PITH_STOPWORDS = frozenset("""
a an the this that these those and or but if then else of to in on at by for with
from as is are was were be been being it its i you he she they we me my your our
their them us do does did done have has had will would can could should may not no
yes so than too very just about into over under out up down off also which who whom
what when where how why then there here their they're it's don't doesn't
""".split())


def _pith_salient_terms(content: str) -> Dict[str, int]:
    """Term-frequency map over `content` (lowercased alnum tokens, length >= 3,
    minus stopwords) -- the item's own recurring vocabulary. Used as the
    centroid signal for extractive keyframe selection: a segment dense in the
    item's repeated concepts is central to what the item is about."""
    freq: Dict[str, int] = {}
    for w in re.findall(r"[a-z0-9_]+", content.lower()):
        if len(w) < 3 or w in _PITH_STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return freq


def _pith_segment_score(seg: str, term_freq: Dict[str, int]) -> float:
    """Informativeness of one segment (higher = keep). Combines the item's own
    salient-term density (centroid, length-normalized so long filler can't win
    on raw counts), payload-token count (numbers, dotted/underscored
    identifiers, `code`, file refs -- the parts that carry facts), and a
    structural bonus for headings / def-class signatures / labelled bullets.
    Greetings and filler score near zero without any special-casing."""
    s = seg.strip()
    if not s:
        return 0.0
    words = re.findall(r"[a-z0-9_]+", s.lower())
    if not words:
        return 0.0
    centroid = sum(term_freq.get(w, 0) for w in words) / len(words)
    payload = len(re.findall(
        r"\d+|[A-Za-z_]+[._][A-Za-z0-9_]+|`[^`]+`|#\d+", s))
    structural = 0.0
    if re.match(r"#{1,6}\s", s) or re.match(r"(?:def|class)\s+\w", s):
        structural = 2.0
    elif re.match(r"[-*]\s+\*\*", s):  # "- **Label:**" key-value bullet
        structural = 1.0
    return centroid + 0.5 * payload + structural


def _pith_cut_at_word_boundary(text: str, limit: int) -> str:
    """Hard-cut `text` to at most `limit` chars, backing up to the last space
    so the cut never lands mid-word (unless `text` has no space within the
    first `limit` chars, in which case a mid-word cut is unavoidable)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip()


def pith_stage2_keyframe(content: str, max_chars: Optional[int] = None,
                          query: str = "") -> tuple:
    """Pith Stage 2: keyframe / LOD compression -- concept-aware, extractive.

    Cheap, pure, deterministic -- no LLM, no I/O. Compresses `content` to a
    keyframe by keeping its highest-INFORMATION segments, NOT its head. A
    positional "first sentence" keyframe keeps the setup and throws away the
    payload -- "Good morning, my friend! Now, about those important things..."
    would survive as just the greeting. Instead we score every segment by the
    concepts it actually carries and keep the densest ones, in original order,
    with elision marks where segments were skipped.

    Scoring (see _pith_segment_score): a segment earns points for the item's
    own recurring salient terms (centroid), for payload tokens (numbers,
    dotted/underscored identifiers, `code`, file refs), for structural role
    (headings / def-class signatures / labelled bullets), and -- when a `query`
    is supplied -- for overlap with the query. Greetings and filler carry none
    of these and fall to the bottom on their own; no greeting blacklist needed.

    Returns `(keyframe, delta)`:
    - `keyframe`: the compressed string, ending in a visible " ⋯[+N]" marker
      (N = chars dropped) so a reader can never mistake it for the whole item;
      interior "⋯" marks show where non-adjacent segments were joined. Empty
      string for empty/whitespace input; `content` unchanged (empty delta) when
      it already fits.
    - `delta`: the dropped segments (original order), for `CacheLine.deltas`
      (future victim-cache / expansion can restore it).

    max_chars defaults to CC_PITH_KEYFRAME_CHARS (env, clamped [60, 1000]).
    Never raises.
    """
    if max_chars is None:
        max_chars = _CC_PITH_KEYFRAME_CHARS

    if not content or not content.strip():
        return ("", "")

    if len(content) <= max_chars:
        return (content, "")

    # 1. Segment: lines, and split long prose lines into sentences so a single
    #    dense paragraph can be sub-selected rather than kept/dropped whole.
    raw_segs: List[str] = []
    for ln in content.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if len(s) > 120 and re.search(r"[.?!]\s", s):
            for part in re.split(r"(?<=[.?!])\s+", s):
                if part.strip():
                    raw_segs.append(part.strip())
        else:
            raw_segs.append(s)
    if not raw_segs:
        return ("", "")

    segs = list(enumerate(raw_segs))  # (orig_index, text)

    # 2. Score each segment by intrinsic information payload (+ query overlap).
    term_freq = _pith_salient_terms(content)
    qterms = set(re.findall(r"[a-z0-9_]+", query.lower())) if query else set()

    def _score(idx_seg):
        idx, seg = idx_seg
        base = _pith_segment_score(seg, term_freq)
        if qterms:
            sw = re.findall(r"[a-z0-9_]+", seg.lower())
            base += float(sum(1 for w in sw if w in qterms))
        # Faint positional prior: only a tie-breaker so equally-informative
        # segments keep reading order; far too small to override real signal.
        return base - 0.001 * idx

    ranked = sorted(segs, key=_score, reverse=True)

    # 3. Greedily pack the most-informative segments until the budget is spent.
    #    Unlike Stage 3's item-level strict-prefix, packing WITHIN one item is
    #    correct -- a keyframe is a summary, so a shorter lower-ranked segment
    #    that still fits is worth keeping.
    marker_reserve = 12  # room for the trailing " ⋯[+NNNN]" marker
    budget = max(1, max_chars - marker_reserve)
    picked: List[tuple] = []  # (orig_index, text)
    used = 0
    for idx, seg in ranked:
        piece = seg
        if not picked and len(piece) > budget:
            piece = _pith_cut_at_word_boundary(piece, budget)
        add = len(piece) + (1 if picked else 0)
        if picked and used + add > budget:
            continue
        picked.append((idx, piece))
        used += add

    if not picked:
        return ("", content)

    # 4. Restore reading order; mark elisions between non-adjacent segments.
    picked.sort(key=lambda t: t[0])
    out_parts: List[str] = []
    prev_idx = None
    for idx, piece in picked:
        if prev_idx is not None and idx != prev_idx + 1:
            out_parts.append("⋯")
        out_parts.append(piece)
        prev_idx = idx
    body = " ".join(out_parts)

    picked_idxs = {i for i, _ in picked}
    delta = " ".join(seg for i, seg in segs if i not in picked_idxs)
    dropped_chars = max(0, len(content) - len(body))
    keyframe = body + (" ⋯[+%d]" % dropped_chars)

    return (keyframe, delta)


def cc_thermal(graph: Any, node_id: str) -> float:
    """Pith Stage 5 raw thermal (warmth) for a node, read from the substrate's
    OWN state: Ca_i (persistence of recent activation -- decays each step,
    bumps on spike) + firing_rate_ema (recent firing rate). Un-normalized here;
    pith_stage3 min-max normalizes it across the survivor set before folding it
    into the rank. The Lenia field-energy term (_CC_PITH_THERMAL_W_FIELD) is
    reserved but not yet wired -- these two SNN signals carry the warmth today.
    Read-only, fail-soft to 0.0 (a vanished/stateless node adds no warmth)."""
    try:
        node = graph.nodes.get(node_id) if graph is not None else None
        if node is None:
            return 0.0
        ca = float(getattr(node, "Ca_i", 0.0) or 0.0)
        fire = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
        return _CC_PITH_THERMAL_W_CA * ca + _CC_PITH_THERMAL_W_FIRE * fire
    except Exception:
        return 0.0


def cc_l1_budget(commons: Any) -> int:
    """Pith §3.2 autonomic breathing: the L1 char budget breathes with arousal.
    PARASYMPATHETIC (calm/exploratory) -> expanded; SYMPATHETIC (threat/tunnel
    vision) -> contracted. Reads the single authoritative arousal Immunis
    deposits to the CC Commons (commons.read_arousal). Gated by
    CC_PITH_L1_BREATHE; off (or no Commons) -> the static budget. Fail-soft ->
    static budget on any error, and clamped to the same [500, 40000] bounds."""
    if not _CC_PITH_L1_BREATHE or commons is None:
        return _CC_PITH_L1_BUDGET
    try:
        state = commons.read_arousal()
        mult = _CC_PITH_BREATHE_SYMPATHETIC if state == "SYMPATHETIC" else _CC_PITH_BREATHE_PARASYMPATHETIC
        return max(500, min(40000, int(_CC_PITH_L1_BUDGET * mult)))
    except Exception:
        return _CC_PITH_L1_BUDGET


# Pith Stage 5 victim cache: cache lines surfaced but dropped from L1 (budget
# overflow) land here instead of vanishing, so a near-future turn can recover
# them ("wait, go back to what you said"). Bounded FIFO, TTL-aged by recall
# turn. Module-level + lock (daemon recall + idle sweep touch it).
_PITH_VICTIM: List[Dict[str, Any]] = []
_PITH_VICTIM_LOCK = threading.Lock()


def pith_victim_recover(candidates: List[CacheLine]) -> List[CacheLine]:
    """Stage 5 recapture: merge still-live victim entries back into the
    candidate set for a second chance at L1, and age the buffer one turn. A
    victim not already among the fresh candidates is re-injected as a CacheLine
    (stream='victim', carrying its cached thermal); entries past TTL are
    evicted. No-op when the buffer is disabled (size<=0) or empty."""
    if _CC_PITH_VICTIM_SIZE <= 0 or not _PITH_VICTIM:
        return candidates
    have = {cl.node_id for cl in candidates}
    merged = list(candidates)
    with _PITH_VICTIM_LOCK:
        live = []
        for v in _PITH_VICTIM:
            v["ttl"] -= 1
            if v["ttl"] <= 0:
                continue
            live.append(v)
            if v["node_id"] not in have:
                cl = CacheLine.from_surfaced(v["node_id"], v["content"],
                                             score=v["score"], stream="victim")
                cl.thermal = v.get("thermal", 0.0)
                merged.append(cl)
        _PITH_VICTIM[:] = live
    return merged


def pith_victim_capture(kept: List[CacheLine], all_lines: List[CacheLine]) -> None:
    """Stage 5 eviction: unpinned lines that were surfaced but didn't make L1
    (budget overflow) drop into the bounded victim buffer. Any victim promoted
    back into L1 this turn is removed (it's resident again). FIFO-bounded to
    CC_PITH_VICTIM_SIZE; TTL (re)set on capture. No-op when disabled."""
    if _CC_PITH_VICTIM_SIZE <= 0:
        return
    kept_ids = {cl.node_id for cl in kept}
    dropped = [cl for cl in all_lines
               if not cl.pinned and cl.node_id not in kept_ids and cl.stream != "victim"]
    with _PITH_VICTIM_LOCK:
        # drop any victim that got promoted back into L1 this turn
        _PITH_VICTIM[:] = [v for v in _PITH_VICTIM if v["node_id"] not in kept_ids]
        existing = {v["node_id"]: v for v in _PITH_VICTIM}
        for cl in dropped:
            if cl.node_id in existing:
                existing[cl.node_id]["ttl"] = _CC_PITH_VICTIM_TTL
            else:
                _PITH_VICTIM.append({"node_id": cl.node_id, "content": cl.content,
                                     "score": cl.score, "stream": cl.stream,
                                     "thermal": cl.thermal, "ttl": _CC_PITH_VICTIM_TTL})
        if len(_PITH_VICTIM) > _CC_PITH_VICTIM_SIZE:
            del _PITH_VICTIM[:len(_PITH_VICTIM) - _CC_PITH_VICTIM_SIZE]


def pith_compress_history(turn_texts: List[str], graph: Any, per_turn_chars: Optional[int] = None) -> List[str]:
    """Pith over the OUTBOUND conversation history — the substrate-informed
    replacement for miniTID's faux KISS. Given the ordered older-than-window
    turns (miniTID owns the message array + ordering; the substrate has no
    conversation identity -- turns are content-hashed nodes, sha1(text)), return
    a positionally-aligned list of compressed turns miniTID can splice back in.

    Substrate-INFORMED (the whole point of being connected to the NG): each
    turn's own node warmth (cc_thermal: Ca_i + firing_rate_ema) scales how many
    chars it keeps -- a turn the substrate has kept warm (recently reactivated,
    load-bearing) is compressed LESS; a cold one more. The actual compression is
    the existing salience-aware keyframe extractor (pith_stage2_keyframe) -- it
    keeps the payload and drops filler, NOT the greeting-keeping first-sentence
    cut faux KISS does. Fail-soft per turn: a turn with no node (e.g. reinforced
    away by the KISS gate) or any error keeps warmth 0 -> base budget.

    per_turn_chars: base keyframe budget (default CC_PITH_KEYFRAME_CHARS);
    warmth scales it up to ~2x for the warmest turns."""
    import hashlib
    base = per_turn_chars if per_turn_chars is not None else _CC_PITH_KEYFRAME_CHARS
    out: List[str] = []
    for text in turn_texts:
        if not text or not text.strip():
            out.append(text)
            continue
        try:
            node_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
            warmth = cc_thermal(graph, node_id)  # 0.0 if node absent (fail-soft)
            # normalize warmth into a [1.0, 2.0] budget multiplier: warmer = keep more.
            # thermal is unbounded-ish; squash with a soft cap so one hot turn can't
            # blow the budget. tanh-free cheap squash: w/(w+1) in [0,1).
            mult = 1.0 + (warmth / (warmth + 1.0)) if warmth > 0 else 1.0
            budget = max(60, min(1000, int(base * mult)))
            keyframe, _delta = pith_stage2_keyframe(text, max_chars=budget)
            out.append(keyframe if keyframe else text)
        except Exception:
            out.append(text)  # never drop a turn; worst case pass it through
    return out


def pith_stage3(cache_lines: List[CacheLine], budget_chars: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None) -> List[CacheLine]:
    """Pith Stage 3: unified rank + char budget -- the L1 assembler core.

    Replaces block-order concatenation (every recency item before every
    relevance item, regardless of score) with a single ranked, budget-bounded
    read. Consumes the emitter scores already carried on each CacheLine (does
    NOT re-derive relevance -- no embed(), no GSG/Poincare rescore, no
    vector_db scan, no substrate walk); normalization is a monotone transform
    of the emitter's own score, nothing more.

    Order of operations:

    1. Split pinned vs unpinned. Pinned lines are ALWAYS kept, in their
       original relative order, and never consume budget.
    2. Per-stream min-max normalize the unpinned lines' `score` (grouped by
       `.stream`) -- the two streams' raw scales (~1.7 for SurfacingMonitor
       recency, ~100s for GSG-rescored pattern-completion relevance) are not
       comparable, so normalizing within-stream first lets both signals
       contribute instead of the larger-numbered stream always winning.
       Single-item or all-equal-score streams normalize to 1.0 (top-of-
       stream), never a div-by-zero.
    3. Unified score = weights[stream] * norm. weights defaults to
       {"pattern": CC_PITH_W_RELEVANCE, "monitor": CC_PITH_W_RECENCY,
       "recall": CC_PITH_W_RELEVANCE}; an unknown stream fails open to 1.0
       (kept in contention rather than zeroed out).
    4. Stable-sort unpinned lines by unified score, descending.
    5. Greedy budget fill over the sorted list, accumulating len(content):
       keep while the running total stays <= budget_chars, stop at the first
       line that would exceed it. A single line longer than the whole budget
       is still kept if nothing has been added yet (never emit an empty L1
       just because the top item is large), then fill stops.
    6. Assemble: pinned lines first (original order), then kept unpinned
       lines in ranked order.

    budget_chars defaults to CC_PITH_L1_BUDGET (env, clamped [500, 40000]);
    weights defaults to the CC_PITH_W_RELEVANCE / CC_PITH_W_RECENCY env pair.
    Updates the module-level _PITH_METRICS counters (ranked_in, ranked_kept,
    ranked_dropped, budget_chars_used) unconditionally, same convention as
    pith_stage1. Pure function over the already-surfaced small set (<~30
    items) -- cheap, hook-timeout-safe, no I/O.

    Never raises on empty input (returns []) or degenerate scores.
    """
    _PITH_METRICS.ranked_in += len(cache_lines)

    if not cache_lines:
        return []

    if budget_chars is None:
        budget_chars = _CC_PITH_L1_BUDGET
    if weights is None:
        weights = {
            "pattern": _CC_PITH_W_RELEVANCE,
            "monitor": _CC_PITH_W_RECENCY,
            "recall": _CC_PITH_W_RELEVANCE,
            "victim": _CC_PITH_W_RECENCY,   # recovered drops: secondary prior, like recency
        }

    # Step 1: split pinned vs unpinned.
    pinned_lines = [cl for cl in cache_lines if cl.pinned]
    unpinned_lines = [cl for cl in cache_lines if not cl.pinned]

    # Step 2: per-stream min/max over unpinned lines only.
    stream_bounds: Dict[str, tuple] = {}
    for cl in unpinned_lines:
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        stream_bounds[cl.stream] = (min(lo, cl.score), max(hi, cl.score))

    # Steps 2-3: normalize + weight -> unified score, index-paired with
    # unpinned_lines so the stable sort in step 4 can carry input order
    # through as an explicit tie-breaker (Python's sort is already stable,
    # but pairing with the original index makes that ties-keep-input-order
    # guarantee explicit rather than incidental).
    scored: List[tuple] = []
    for idx, cl in enumerate(unpinned_lines):
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        norm = 1.0 if hi <= lo else (cl.score - lo) / (hi - lo)
        weight = weights.get(cl.stream, 1.0)
        # Stage 5 thermal fold: warm content (high Ca_i/firing) is gently
        # preferred. thermal defaults 0.0 -> multiplier 1.0 -> byte-identical
        # to pre-Stage-5 ranking until the daemon populates cl.thermal.
        unified = weight * norm * (1.0 + _CC_PITH_THERMAL_GAIN * cl.thermal)
        scored.append((unified, idx, cl))

    # Step 4: stable sort by unified score, descending. Ties keep input
    # order because idx (ascending) is the secondary sort key.
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Step 5: greedy budget fill, STRICT rank-prefix -- keep the top-ranked
    # run that fits and stop at the first line that would overflow. We do NOT
    # keep scanning for smaller lower-ranked lines that happen to fit: that
    # would let rank-20 recency junk jump ahead of a dropped rank-8 relevance
    # block, inverting the very ordering Stage 3 exists to enforce. The first
    # line is always kept (even if it alone exceeds the budget) so a large top
    # item never yields an empty L1. Graceful degradation (Pith Stage 2): a
    # line that doesn't fit at full fidelity gets one more chance as a
    # keyframe (compressed head) before being dropped -- terser beats absent.
    kept_unpinned: List[CacheLine] = []
    running_total = 0
    for _unified, _idx, cl in scored:
        full_len = len(cl.content or "")
        if not kept_unpinned or running_total + full_len <= budget_chars:
            kept_unpinned.append(cl)
            running_total += full_len
            continue

        kf, delta = pith_stage2_keyframe(cl.content)
        if running_total + len(kf) <= budget_chars and len(kf) < full_len:
            cl.content = kf
            # lod = fraction of the original retained (1.0 = full, matching the
            # CacheLine default); a compressed line records how much survived.
            cl.lod = len(kf) / full_len if full_len else 1.0
            cl.keyframe = True
            cl.deltas = [delta]
            kept_unpinned.append(cl)
            running_total += len(kf)
            _PITH_METRICS.compressed_count += 1
            _PITH_METRICS.chars_saved += (full_len - len(kf))
            continue

        break
    dropped = len(scored) - len(kept_unpinned)

    _PITH_METRICS.ranked_kept += len(pinned_lines) + len(kept_unpinned)
    _PITH_METRICS.ranked_dropped += dropped
    _PITH_METRICS.budget_chars_used += running_total

    # Step 6: assemble -- pinned first (original order), then ranked kept.
    return pinned_lines + kept_unpinned


# =============================================================================
# CC Recall Unification (LAW-3/"keep even") -- one recall pipeline, both
# hemispheres. See docs/superpowers/plans/2026-07-22-cc-recall-unification-
# spec.md. Before this, cc-ng-daemon.py:_recall (laptop) and cc_ng_host.py:
# _recall (VPS host) were copy-pasted and had drifted: laptop ran the full
# Pith pipeline, VPS ran zero Pith (a plain two-block concat), so enabling
# CC_PITH_ENABLED on the VPS would have been a no-op (no code there to gate).
# cc_assemble_recall is a verbatim extraction of the laptop _recall body (the
# reference -- LAW 3: extract, don't redesign), parameterized on (ng, query,
# k, conv_state, commons) instead of a module-global STATE, so it is process-
# agnostic by construction (Syl's-Law: bind to passed-in instances, never
# module globals -- cc_ng_organism.py already works this way elsewhere).
# Both _recall entry points (cc-ng-daemon.py laptop, cc_ng_host.py VPS) become
# thin per-half wrappers: STATE bookkeeping (last_activity/stats), then call
# this function and return its result. VPS behavior is byte-identical to
# today until CC_PITH_ENABLED is turned on (the gate defaults OFF), at which
# point it gains the same Pith pipeline the laptop already had.
# =============================================================================

# Rate-limits the Pith-fallback warning below (was per-half module state in
# cc-ng-daemon.py; now shared here since the fail-soft path lives in one
# place). Each process importing this module gets its own copy of these
# module-level names, so laptop and VPS still rate-limit independently.
_PITH_WARN_INTERVAL_S = float(os.environ.get("CC_PITH_WARN_INTERVAL_S", "60"))
_last_pith_warn_ts = 0.0

# Read-only recall instrumentation (CC_RECALL_DEBUG), folded in from
# cc-ng-daemon.py's laptop-only copy during the unification -- both
# hemispheres get it now (still off by default, still pure observation:
# does NOT change what cc_assemble_recall returns). When the env flag is
# set, appends one JSON line per call to _RECALL_DEBUG_PATH capturing the
# two raw streams separately -- SurfacingMonitor recency (monitor_items) vs
# pattern-completion substrate-spread (pc_results) -- each with score +
# node_id + content preview, BEFORE Pith merges them. Fail-soft. LAW 5.
_CC_RECALL_DEBUG = os.environ.get("CC_RECALL_DEBUG", "0") not in ("0", "false", "")
_RECALL_DEBUG_PATH = os.path.join(
    os.path.expanduser("~/.claude/plugins/neurograph"), "recall_debug.jsonl")


def _cc_recall_debug_log(query: str, monitor_items: List[Dict[str, Any]],
                          pc_results: List[Dict[str, Any]]) -> None:
    """Append one JSON line with both raw streams (read-only, fail-soft)."""
    if not _CC_RECALL_DEBUG:
        return
    try:
        def _stream(items):
            out = []
            for it in (items or [])[:15]:
                out.append({
                    "score": round(float(it.get("score", it.get("strength", 0.0)) or 0.0), 3),
                    "node_id": (it.get("node_id") or "")[:48],
                    "preview": (it.get("content", "") or "").replace("\n", " ")[:70],
                })
            return out
        rec = {
            "ts": time.time(),
            "query": (query or "")[:120],
            "n_monitor": len(monitor_items or []),
            "n_pattern": len(pc_results or []),
            "monitor": _stream(monitor_items),
            "pattern": _stream(pc_results),
        }
        with open(_RECALL_DEBUG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as exc:
        logger.debug("recall-debug log failed (non-fatal): %s", exc)


def cc_assemble_recall(ng: Any, query: str, k: int, conv_state: dict, commons: Any,
                        allow_pattern_completion: bool = True,
                        on_monitor_error: Optional[Any] = None) -> str:
    """Return surfacing context for CC hook injection -- THE shared recall
    pipeline for both hemispheres (laptop cc-ng-daemon.py, VPS cc_ng_host.py).

    Combines two complementary signals: SurfacingMonitor (recency -- nodes
    that fired in the SNN during recent deposits) and Active Recall (pattern
    completion -- direct semantic search via cc_pattern_completion_recall,
    finds genuinely relevant content regardless of how long ago it was
    learned). SurfacingMonitor block first, Active Recall block second --
    matches Syl's own ordering in handle_assemble() (neurograph_rpc.py).

    Dedups by node_id across the two blocks: a node can be both recently-
    fired (SurfacingMonitor) and semantically matching the query (Active
    Recall) -- without this it renders twice in the injected context.

    allow_pattern_completion=False skips the Active Recall half entirely --
    used by handle_pre_tool_use() when gate_pattern_completion() has already
    given this file_path a pattern-completion pass recently (per-file dedup
    cache; avoids re-paying the .recall() cost on every single PreToolUse
    touch to the same file within one task).

    When CC_PITH_ENABLED, the combined item set is run through the Pith
    pipeline (CacheLines -> pith_victim_recover -> cc_thermal -> cc_novelty
    -> pith_stage1 -> pith_stage3(budget=cc_l1_budget(commons)) ->
    pith_victim_capture) instead of the plain two-block concatenation, with
    constitutional pins (ng.graph._is_identity_protected) preserved
    unconditionally. Any exception anywhere in the Pith path is fail-soft --
    falls back to the pre-Pith monitor_ctx/pc_block rendering, records a
    _PITH_METRICS failure, and rate-limit-warns (a surfacing pass must never
    crash or time out the hook).

    Params only (ng/conv_state/commons) -- no module-global STATE access,
    so this function is process-agnostic (Syl's-Law) and safe to call from
    either hemisphere with its own isolated instances.
    """
    monitor_ctx = ''
    monitor_node_ids: set = set()
    monitor_items: List[Dict[str, Any]] = []
    try:
        monitor = getattr(ng, '_surfacing_monitor', None)
        if monitor is not None:
            monitor_items = monitor.get_surfaced()
            monitor_node_ids = {item.get('node_id') for item in monitor_items}
            monitor_ctx = monitor.format_context(monitor_items)
    except RuntimeError:
        monitor_ctx = ''  # dict mutation race during concurrent deposit
        monitor_node_ids = set()
        monitor_items = []
    except Exception as exc:
        logger.debug('Recall failed: %s', exc)
        if on_monitor_error is not None:
            try:
                on_monitor_error(exc)
            except Exception:
                pass  # the error-reporting hook itself must never break recall
        monitor_ctx = ''
        monitor_node_ids = set()
        monitor_items = []

    pc_block = ''
    pc_results: List[Dict[str, Any]] = []
    if allow_pattern_completion:
        try:
            pc_results = cc_pattern_completion_recall(ng, query, k, state=conv_state)
            pc_results = [r for r in pc_results if r.get('node_id') not in monitor_node_ids]
            pc_block = _format_cc_recall_block(pc_results)
        except Exception as exc:
            logger.debug('Pattern-completion recall failed (non-fatal): %s', exc)
            pc_block = ''
            pc_results = []

    # Read-only instrumentation (CC_RECALL_DEBUG): capture both raw streams
    # BEFORE Pith merges them, to measure where the query signal is lost.
    _cc_recall_debug_log(query, monitor_items, pc_results)

    # Pith pipeline (gated, CC_PITH_ENABLED, default OFF): dedup/clutter-strip
    # + unified-rank + budget the combined SurfacingMonitor + Active Recall
    # item set before rendering, replacing the two-block concatenation below
    # with one combined block. Gate OFF -> this whole block is skipped and
    # cc_assemble_recall() falls straight through to the original
    # monitor_ctx/pc_block return, unchanged.
    if _CC_PITH_ENABLED:
        try:
            def _pinned(node_id):
                try:
                    return bool(ng.graph._is_identity_protected(node_id))
                except Exception as exc:
                    # Fail-soft to not-pinned so ONE node's bad pin lookup can't
                    # sink the whole Pith pass; log it so a vanished/erroring
                    # constitutional-pin guard surfaces rather than going silent.
                    logger.debug('Pith pin lookup failed for %r (treating as unpinned): %s', node_id, exc)
                    return False

            # Stream-tag each half so pith_stage3 can per-stream normalize
            # (SurfacingMonitor recency ~1.7 vs Active Recall/GSG relevance
            # ~100s are not comparable scales) -- monitor_items -> "monitor",
            # pc_results -> "pattern".
            cache_lines = [
                CacheLine.from_surfaced(
                    node_id=it.get('node_id') or '',
                    content=it.get('content', ''),
                    score=it.get('score', 0.0),
                    pinned=_pinned(it.get('node_id')),
                    stream='monitor',
                )
                for it in monitor_items
            ] + [
                CacheLine.from_surfaced(
                    node_id=it.get('node_id') or '',
                    content=it.get('content', ''),
                    score=it.get('score', 0.0),
                    pinned=_pinned(it.get('node_id')),
                    stream='pattern',
                )
                for it in pc_results
            ]

            # Pith Stage 5 (victim recapture): merge still-live victim-cache
            # entries back in for a second chance at L1, and age the buffer.
            cache_lines = pith_victim_recover(cache_lines)

            # Pith Stage 5 (thermal): populate each line's warmth from the
            # substrate's own state (Ca_i + firing_rate_ema). Done here because
            # this is where the graph is in scope; pith_stage3 folds it into the
            # rank (warm content preferred). Fail-soft per line -> 0.0.
            for _cl in cache_lines:
                try:
                    _cl.thermal = cc_thermal(ng.graph, _cl.node_id)
                except Exception:
                    _cl.thermal = 0.0

            try:
                novelty = cc_novelty(conv_state, ng.graph)
            except Exception as exc:
                logger.debug('Pith novelty lookup failed (non-fatal): %s', exc)
                novelty = 0.0

            survivors = pith_stage1(cache_lines, query, novelty)
            # Stage 3: unified rank + char budget -- replaces block-order
            # concatenation with a single ranked, budget-bounded L1 read.
            # Reads its own weights from env (CC_PITH_W_RELEVANCE,
            # CC_PITH_W_RECENCY); budget breathes with commons arousal.
            _pre_l1 = survivors  # post-stage1, pre-budget: the full L1 candidate set
            survivors = pith_stage3(survivors, budget_chars=cc_l1_budget(commons))
            # Pith Stage 5 (eviction): budget-dropped lines fall to the victim buffer.
            try:
                pith_victim_capture(survivors, _pre_l1)
            except Exception as exc:
                logger.debug('Pith victim capture failed (non-fatal): %s', exc)
            survivor_results = [{'score': cl.score, 'content': cl.content} for cl in survivors]
            return _format_cc_recall_block(survivor_results)
        except Exception as exc:
            # Fail-soft: fall back to the pre-Pith rendering below. But COUNT it
            # and warn (rate-limited) -- a silently-failing Pith path degrades
            # surfacing invisibly; the counter/warning make that observable.
            _PITH_METRICS.record_failure()
            global _last_pith_warn_ts
            _now = time.time()
            if _now - _last_pith_warn_ts >= _PITH_WARN_INTERVAL_S:
                _last_pith_warn_ts = _now
                logger.warning('Pith stage1 path failed, falling back to un-Pithed rendering: %s', exc)
            else:
                logger.debug('Pith stage1 path failed (non-fatal), falling back: %s', exc)

    if monitor_ctx and pc_block:
        return monitor_ctx + "\n\n" + pc_block
    return monitor_ctx or pc_block
