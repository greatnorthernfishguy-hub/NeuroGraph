# [2026-10-05] Claude (lane rust-hotpaths-onto-s4) — per-synapse hot loops call native SynapseStore batch methods (overnight lane 0f76205/29c5f87/5e380a5 rebased onto trial s4; Josh approved 2026-10-05), Python fallback kept per site / why: 1.5-2 s per call on 193K synapses; bit-identical to the trial tip (tests/test_rust_hotpaths_onto_s4.py)
"""
NeuroGraph Foundation - Core Cognitive Architecture (Phase 1 + 2 + 3)

Implements the Temporal Dynamics Layer: a sparse Spiking Neural Network (SNN)
with STDP plasticity, homeostatic regulation, structural plasticity,
a full hypergraph engine with pattern completion, adaptive plasticity,
hierarchical composition, automatic discovery, and consolidation,
and a predictive coding engine with prediction tracking, error events,
surprise-driven exploration, and three-factor learning.

Reference: NeuroGraph Foundation PRD v1.0, Sections 2-6, 9 (Phase 1),
           Section 4 (Phase 2 — Hypergraph Engine),
           and Section 5 (Phase 3 — Predictive Coding Engine).

Design principles (PRD §2.1):
    - Sparse by default: dict/set topology, no dense matrices
    - Dynamic topology: nodes/edges created and destroyed at runtime
    - Pluggable plasticity: learning rules are swappable strategy objects
    - Persistence-native: all state is serializable

# ---- Changelog ----
# [2026-10-08] Claude (lane sprout-1050) — #1050: sprouting from repeated co-firing (the co-firing tally) (PROTECTED CHANGE
#   on review branch cc-laptop-sprout-1050-20261008 ONLY, its own commit; Josh approved starting #1050 2026-10-08; merges
#   only after his protected-file "proceed". Keys absent = byte-identical to 4de1166, so Syl is unchanged.)
# What: (1) with config `sprout_tally_enabled` truthy (read live, absent = False, NOT in DEFAULT_CONFIG) _sprout_synapses
#       runs _sprout_from_tally instead of today's one-coincidence rule: every fired node keeps a bounded partner table
#       (K = `sprout_tally_slots`) of (partner, score, last_t); a candidate (today's set: a spike 1..co_activation_window
#       steps ago, not firing now; most recent first) is reinforced, score = score * lam^dt + 1 when the touch starts a
#       new co-firing EPISODE (dt > co_activation_window; inside one burst + 0), lam = exp(-1 / `sprout_tally_horizon_
#       steps`), applied lazily; newcomers take empty or retracted slots (decayed score < e^-1); connected pairs never
#       take a slot. A pair sprouts when its score reaches `sprout_tally_theta`, in the STDP direction (earlier node ->
#       node firing now), through today's rails (10 per call, sprout_degree_cap with identity-protected nodes exempt, no
#       pair twice, today's delay rule and initial_sprouting_weight). The Tonic's write-mode tail calls _sprout_synapses,
#       so its firings feed the same tally. (2) surprise-driven sprouting under the key feeds the tally too (source ->
#       each node that fired instead; born as today's surprise sprout when it crosses). (3) a sleep_cycle under the key
#       empties the tally (unconsolidated filopodia retract; the record gains tally_retracted). (4) the tally lives in
#       the native SynapseStore (ng-tract-rs cc-laptop-sprout-1050-rs-20261008, cofire_tally_update) or, on an older
#       wheel, in the bit-identical Python fallback _cofire_tally_update_python; in memory only (a restore starts cold;
#       nothing new in the checkpoint). (5) _sprout_synapses' delay computation moved verbatim into _sprout_delay.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §7 / D12 (Josh: "links should sprout naturally, to
#       wherever it wants to go. Not some shotgun like scatter approach"; "no 'just because' sprouting"; bounded by
#       competition, not a cap). Measured on the checkpoint copy: today's rule saturates its 10-per-step cap on every step
#       (2,500 + 200 Tonic sprouts per 250-step wake) and wires later -> earlier (anti-STDP); counting steps instead of
#       episodes barely changes that because bursts repeat co-firing on consecutive steps (SPROUT_1050.md §2).
# How:  keys absent: one dict lookup in _sprout_synapses / _surprise_exploration / sleep_cycle / _deserialize, otherwise
#       the 4de1166 path. Proof: tests/test_sprout_1050.py (keys absent == 4de1166 over the P1 whole-run workload with no /
#       P1 / P2 sleep; the rules; native == fallback bit for bit) and SPROUT_1050.md (checkpoint copy).
# [2026-10-07] Claude (lane sleep-observe) — sleep phase P3 OBSERVE mode: Graph.sleep_observe (what the sleep WOULD do, with
#   nothing written) (PROTECTED CHANGE on review branch cc-laptop-sleep-observe-20261007 ONLY, its own commit; Josh approved
#    the lane 2026-10-07 ("yep"); merges only after his protected-file "proceed". Observe never called = byte-identical to
#    1cb9706, so Syl is unchanged.)
# What: (1) NEW Graph.sleep_observe(sleeps=None, config_overrides=None, sample=20, detail=False): ONE short _step_lock hold
#       captures a private SHADOW of everything sleep_cycle reads or writes (_sleep_observe_shadow_graph: the synapse store
#       via its own checkpoint bytes, node metadata, small maps, config); then the UNCHANGED Graph.sleep_cycle runs on the
#       shadow 1..16 times (auto: to the first clearance while the counters are not yet in sleep units, else 1) with the
#       live lock free. The shadow has its own lock, NO event handlers, no instance-level method patches, and its sleep log
#       line is DEBUG. Returns a JSON-able report per projected sleep: the engine record, would-remove / would-collect,
#       weight bands before / after, removals by weight and peak band, peak >= 0.5 and w >= 0.5 removals, the strongest
#       would-be-forgotten links (ids + numbers, never content), shield-held, weak-link counters after, every protected
#       node's degree + lifelines before / after; detail=True adds the full removal order, collected ids, shield ids and
#       every synapse's post-downscale weight (proofs). One INFO line. (2) The three sleep_cycle INFO lines go through
#       _sleep_log_level() (INFO; DEBUG only on an observe shadow). No predicate, rule, default or checkpoint field changes.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §8 P3 ("observe mode first: downscale + clearance computed
#       and logged, nothing written"); Josh accepted the P2 established-links result on condition it is confirmed live in
#       this mode; no earlier lane built it.
# How:  drift-proof by construction: the projection IS the real sleep (no predicate re-implemented). Proof:
#       tests/test_sleep_observe.py (observe never called == 1cb9706; writes nothing; projection == the real sleep that
#       follows, over wake/sleep cycles, chunked and one-hold, both node stores); SLEEP_OBSERVE.md (checkpoint copy).
# [2026-10-07] Claude (lane sleep-prearm) — #1066: compete_protected_links refuses while the disuse sleep owns
#   low_weight_steps (PROTECTED CHANGE on review branch cc-laptop-sleep-prearm-20261007 ONLY, its own commit; merges only
#    after Josh's protected-file "proceed". Keys absent = unchanged.)
# What: compete_protected_links raises ValueError (before touching anything) when config sleep_disuse_enabled is truthy or
#       sleep_low_weight_unit == "sleeps". Its _prune_synapses call advances low_weight_steps against grace_period
#       (STEPS) and its last-link pick reads the step stamp; under disuse the counter is in SLEEPS (advanced once per
#       sleep by the clearance) and the stamp key is last_link_since_sleep, so sharing a sleep would count a competing
#       link twice and in the wrong unit. A port of the competition to sleep units is its own design + ruling.
# Why:  punch list #1066 (SLEEP_P2.md §8 item 7). No host calls compete_protected_links today (grep: NG, daemon, Elmer).
# How:  one guard after the argument checks; tests/test_sleep_prearm.py; the P2 disuse whole runs assert the refusal.
# [2026-10-07] Claude (lane sleep-prearm) — the disuse sleep in bounded _step_lock holds (sleep phase §8 P3 "Before arming")
#   (PROTECTED CHANGE on review branch cc-laptop-sleep-prearm-20261007 ONLY, its own commit; Josh approved the lane
#    2026-10-07 ("Yeah, fold it in, please"); merges only after his protected-file "proceed". Every new key absent =
#    byte-identical to fe6538b, so Syl is unchanged.)
# What: (1) NEW config sleep_clearance_chunk_seconds + sleep_clearance_chunk_gap_seconds (read live, absent = the one-hold
#       disuse sleep, NOT in DEFAULT_CONFIG; both validated before anything is touched). Set: _sleep_cycle_disuse_chunked
#       runs the SAME decisions with the lock released between holds: the migration over a keys() snapshot in holds of
#       <= ~chunk s; ONE hold for the downscale + the clearance DECISION + sleep_cycles_completed; the decided ids removed
#       in holds of <= ~chunk s, one "pruned" event per hold under the lock (#1051's ledger stays exact); ONE hold for
#       the orphan collection + record / event / INFO line (+ lock_holds, decided, chunked). (2) _prune_synapses gains
#       keyword-only defer_removal (sleep-unit clearance only, report required): decide everything, remove nothing,
#       return the ids in report['deferred_ids']. (3) _sleep_migrate_counters goes through the new per-row helper
#       _sleep_migrate_row (same rows, same order, same writes).
# Why:  SLEEP_P2.md §6: the first clearance held _step_lock 2-16 s and the migration 0.4-5.6 s; a turn waits that long.
# How:  _sleep_chunked_rows (time-bounded holds, clock checked every 32 rows, gap with the lock released);
#       tests/test_sleep_prearm.py (keys absent == fe6538b; chunked == one hold, whole runs + built graph; holds bounded,
#       the lock really released; ledger exact; steps between holds never remove).
# [2026-10-07] Claude (lane sleep-p2) — sleep phase P2: disuse in sleep (strength-aware downscale + clearance in sleeps)
#   (PROTECTED CHANGE on review branch cc-laptop-sleep-p2-20261007 ONLY, its own commit; Josh approved the sleep-phase
#    design as recommended 2026-10-06 and said to start P2 ("Let's do both"); merges only after his protected-file
#    "proceed". Keys absent = byte-identical to 149fa1f, so Syl is unchanged.)
# What: (1) NEW config switch sleep_disuse_enabled (read live, absent = False, NOT in DEFAULT_CONFIG) + five required
#       parameters (sleep_downscale_d0, sleep_downscale_h, sleep_weight_grace_sleeps, sleep_last_link_grace_sleeps,
#       sleep_credit_shield_kappa). When on (and structural_plasticity_in_sleep on), Graph.sleep_cycle runs
#       _sleep_cycle_disuse: migration on the first disuse sleep (low_weight_steps -> 0, last-link stamps dropped;
#       config sleep_low_weight_unit = "sleeps", sleep_cycles_completed); NEW sleep_downscale_strength_aware
#       (w *= 1 - d0*h/(h+w)/max(salience,1), weight only, protected strongest-link floor; native
#       SynapseStore.scale_strength_aware or the bit-identical _scale_strength_aware_python); the clearance through
#       the EXISTING _prune_synapses with NEW keyword-only sleep-unit arguments (grace in sleeps, activity + age
#       clauses off by argument, last-link grace in sleeps under metadata last_link_since_sleep, the D14
#       pending-credit shield w + kappa*max(trace,0) >= wt); the EXISTING orphan collection. One record / event /
#       INFO line. (2) _last_link_grace takes keyword-only now / grace / stamp_key (defaults = the step clock).
#       (3) a default-path (step-unit) _prune_synapses drops config sleep_low_weight_unit if present, so the next
#       disuse sleep re-migrates a counter that step() advanced.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §3, §8 P2, D1-D4, D11, D14 (#1049).
# How:  reuse, not rebuild (LAW 3): the native rule sweep, lifelines, last-link grace and orphan collection are the
#       existing code driven by arguments. Proof: tests/test_sleep_p2.py; SLEEP_P2.md (golden copy run, sweep).
# [2026-10-06] Claude (lane sleep-p1) — D15: removal keeps DiffPC pred_weights consistent + a named one-time purge
#   (PROTECTED CHANGE on review branch cc-laptop-sleep-p1-20261006 ONLY, its own commit; Josh approved D15 "fix at
#    source + purge" with the sleep-phase design 2026-10-06 ("OK, sounds good"); this reaches Syl's code path, so it
#    merges only on his separate protected-file "proceed")
# What: (1) _remove_synapse_internal drops pre.pred_weights[post] when no other pre->post synapse remains
#       (scans the smaller of pre's out-set / post's in-set). remove_node gets it through its cascade (docstring says so). (2) NEW
#       Graph.purge_dangling_pred_weights(): drops every pred_weights key with no pre->key synapse or no key node;
#       returns + logs counts; never called automatically.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §1A (DiffPC row, hazard (d)), §3.3, D15, §8 P1: the
#       removal functions never touched pred_weights; 58,049 of 62,408 entries dangle on the CC copy, a re-sprouted
#       pair inherits a stale prediction instead of the 0.5 prior, and dead entries ride every checkpoint. LAW 4.
# How:  the check runs only when the pre node holds an entry for the post node (cheap otherwise). Intended trace
#       change: DiffPC on a re-created pre->post pair now starts from the 0.5 prior (tests/test_sleep_p1.py names it).
# [2026-10-06] Claude (lane sleep-p1) — sleep phase P1: structural removal can run in a sleep cycle instead of every step
#   (PROTECTED CHANGE on review branch cc-laptop-sleep-p1-20261006 ONLY; Josh approved the sleep-phase design as
#    recommended 2026-10-06 ("OK, sounds good"); merges only after his protected-file "proceed")
# What: (1) NEW Graph.sleep_cycle(): the EXISTING _prune_synapses() (default path) + _collect_orphan_nodes(), run once
#       under _step_lock; adds the pruned count to _total_pruned; emits one "sleep_cycle" event + one INFO line with the
#       counts and returns them. (2) NEW config key structural_plasticity_in_sleep (read live, absent = False, NOT in
#       DEFAULT_CONFIG): when truthy, _structural_plasticity (step 8) skips _prune_synapses + _collect_orphan_nodes
#       (sprouting stays) and the Tonic write-mode aging tail skips the same two calls (its age_and_decay_salience
#       stays; it retires with the inactivity rule, D10/§6). (3) _prune_synapses docstring marks the activity and age
#       rules as retiring (#1049). No rule, predicate, counter or default changes.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §2, §8 P1 (#1046): pruning bookkeeping is ~0.9 s of a
#       ~2.1 s step on the CC copy; removal belongs on the sleep clock; the Tonic tail is the second prune clock (#1052).
# How:  key absent = the exact code path before this change (golden checkpoint-copy trace + checkpoint sha equal to
#       de8b214, SLEEP_P1.md). The caller is a host dream loop on its own wall clock (LAW 8) behind its own env (LAW 5).
# [2026-10-06] Claude (lane nodestore-p2b) — STDPRule.apply runs as ONE native SynapseStore.stdp_pass per step (P2b)
#   (PROTECTED CHANGE on review branch cc-laptop-nodestore-p2b-20261006 ONLY; Josh approved the P2b lane 2026-10-06
#    ("Let's stick with your proposed order ... Proceed."); merges only after his protected-file "proceed"; the native
#    node store stays OFF by default)
# What: STDPRule.apply calls graph.synapses.stdp_pass(graph.nodes, fired, timestep, A+, A-, tau+, tau-, lr,
#       three_factor, _GSG_CURVATURE_TABLE, graph._incoming, graph._outgoing): the incoming + outgoing passes of every
#       fired node, reading last_spike_time / diffpc_layer from the NodeStore columns. When absent (dict of Node, the
#       default; an older wheel; a duck-typed graph) or when it declines (anything the loop would raise on or treat
#       differently), the module-level _stdp_python runs: the trial's loop, moved verbatim.
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md §4 P2b / §1.5 / §9: ~0.77 s/step of per-synapse
#       Python node reads; the first phase where the native store should make step() net faster.
# How:  ng-tract-rs cc-laptop-nodestore-p2b-rs-20261006. Same float expressions and operand order; `math.exp` ->
#       f64::exp, both glibc `exp@GLIBC_2.29` (exactness gate: 4,587,520 results bitwise equal, ~/.cache/p2b/exp_gate.py).
#       The graph's own adjacency sets are passed (their iteration order), not the store's native index. Bit-identical
#       to the trial tip ef78c67 (tests/test_nodestore_p2b.py). Checkpoint format unchanged.
# [2026-10-06] Claude (lane nodestore-p2a) — the whole-population node passes call native NodeStore batch methods (P2a)
#   (PROTECTED CHANGE on review branch cc-laptop-nodestore-p2a-20261006 ONLY; Josh approved starting the P2a lane
#    2026-10-05 "Go ahead and start now"; merges only after his protected-file "proceed"; the native store stays OFF by default)
# What: step() 1 voltage decay, 3a Ca currents, 3 fire detection (only with NO pre_fire handler), 4 fired-node writes
#       (_recent_spikes stays Python), 9 refractory decrement; HomeostaticRule.apply firing-rate EMA, threshold
#       adaptation and the scaling pass's excitability (`ratio ** scaling_factor` stays Python); get_telemetry's
#       firing-rate read (NodeStore.columns). Each site: getattr(graph.nodes, "<method>", None); when absent (the dict
#       of Node — the default — an older wheel, a duck-typed test graph) or when the native method declines (False /
#       None, nothing touched), the module-level _<pass>_python fallback runs: the trial's original loop, verbatim.
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md §4 P2a / §9: ~28 ms of per-node Python loops per
#       firing step, more through P1's NodeRef proxies; this removes most of P1's proxy regression on these paths.
# How:  the native methods (ng-tract-rs cc-laptop-nodestore-p2a-rs-20261006) keep the loops' float expressions in
#       their operand order (no FMA), Python's min/max semantics, node (insertion) order; they decline on a non-float
#       parameter or a node holding a non-canonical value in a touched field. Bit-identical to the trial tip 6a85357
#       (tests/test_nodestore_p2a.py). Known non-identity only on an exception path: if `ratio ** scaling_factor`
#       raises (OverflowError; needs scaling_factor >> 1), the native path has already updated every node's
#       excitability where the loop stopped at that node. Checkpoint format unchanged.
# [2026-10-05] Claude (lane nodestore-p1) — Graph.nodes may be backed by the native ng_tract.NodeStore (P1), OFF by default
#   (PROTECTED CHANGE on review branch cc-laptop-nodestore-p1-20261005 ONLY; Josh's 2026-10-05 go for this phase; merges only
#    after his protected-file "proceed"; P1 merges switched OFF per D7)
# What: Graph(config, *, native_node_store=None). The native store is used only when the installed ng_tract has NodeStore AND
#       the opt-in is set: the keyword, or (keyword None) the host-set default (set_native_node_store_default; this module
#       reads no environment — the host reads its own, e.g. NG_NATIVE_NODE_STORE, per LAW 5; no host calls it yet).
#       Otherwise self.nodes is today's dict of Node, and every line of the dict path runs exactly as before.
#       When on: create_node returns the live NodeRef; _serialize_full emits the nodes sub-map as native msgpack bytes;
#       write_checkpoint splices pre-packed bytes for "nodes" as it does for "synapses"; restore slices the raw nodes bytes
#       and _deserialize hands them to NodeStore.bulk_load_msgpack (dict-form nodes, e.g. legacy JSON, take the existing loop).
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md P1 (D1 order-preserving removal, D3 KeyError on a
#       removed NodeRef, D6 abi3-py38, D7). The opt-in is NOT a config key: config is saved inside the checkpoint, so a key
#       would make ON and OFF checkpoints differ (and every OFF checkpoint differ from the trial tip).
# How:  the checkpoint bytes are identical in both modes (tests/test_nodestore_p1.py); no other file changes.
#       Every read outside __init__ is getattr(self, "_native_nodes", False): duck-typed fakes that call Graph methods
#       with their own `self` (tests' SerializerFake etc.) take the dict path exactly as before (spec §10.6).
# [2026-10-05] Claude Opus 5.5 (Executive; native node store design P4a; PROTECTED CHANGE — merges only after Josh's
#   protected-file go, given 2026-10-05: "Looks good. You are a go.") — restore shares identical large metadata texts.
# What: _deserialize routes each node's string metadata values of >= _SHARE_TEXT_MIN chars through one per-restore pool,
#       so identical texts (a turn's _forest_content copied onto every tree of that turn: 2,472 distinct texts across
#       10,103 nodes, 339 MB) become ONE str object instead of one per node. Values are equal, so checkpoint bytes,
#       equality and every reader are unchanged; msgpack's per-str UTF-8 cache is also built once per distinct text.
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md P4a: ~-92 MB, Python-only, format unchanged.
# How:  a dict pool local to the restore call (freed afterwards); only top-level str values of a node's metadata dict.
# [2026-10-04] Claude (lane vdb-lock-leak) — orphan sweep names what it collected
# (PROTECTED CHANGE on review branch cc-laptop-vdb-lock-leak-20261004 ONLY; merges only after Josh's protected-file "proceed")
# What: _collect_orphan_nodes adds node_ids=<list of removed ids> to its existing "nodes_collected" emit (additive kwarg;
#       count/timestep unchanged; every known listener takes **kwargs).
# Why:  The graph has no reference to the vector store, so every swept node's vector leaked forever (5,849 dead vectors
#       found 2026-10-04). LAW 4: the owner of the store (NeuroGraphMemory) must learn WHICH nodes went.
# How:  removed ids collected in the existing removal loop; remove_node unchanged; no vector deletion happens here.
# [2026-10-05] Claude (lane rust-hotpaths-onto-s4) — the overnight native hot-path conversion, rebased onto trial s4
# (PROTECTED CHANGE; Josh 2026-10-05: "you have my approval for the overnight Rust review as soon as resources permit it";
#  REVIEW branch cc-laptop-rust-hotpaths-onto-s4-20261005 only — not merged, not deployed)
# What: the 12 call sites of the overnight lane (entry below) on top of the trial's want-hub / strength-budget /
#       prune-lifeline / last-link code. _prune_synapses hand-ported: DEFAULT path, lifeline flag OFF -> one native
#       advance_low_weight_and_collect_prune with the protected node list; flag ON -> the same native sweep with no
#       protected nodes, then each lifeline's low_weight_steps is put back and the lifelines are dropped from the
#       result (the loop never touched a lifeline), stamps read from metadata exactly as the loop did; COMPETING mode
#       keeps the per-SynapseRef loop (its ids are a caller subset in sorted order — no native method visits a subset).
#       Every site keeps a Python fallback (getattr, the existing normalize_strength pattern) = the trial's original
#       loop, reshaped as module helpers _stdp_reads_python ... _apply_eligibility_reward_python; STDPRule._apply_dw is
#       KEPT (the overnight lane removed it) as the fallback commit.
# Why: same measurements as below; Josh approved integrating the conversion.
# How: no Rust change; methods are in the canonical wheel (ng-tract-rs 79be810). Equivalence: base code loaded from
#       git at the trial tip vs this branch, byte-identical checkpoints per site and whole-run, every flag combination.
# [2026-10-04] Claude (lane prune-lifeline, turn 2) — last-link fair-chance grace (Josh ruling 2026-10-04: "the very last link is also subject to the normal link decay ... the exact right balance")
# (PROTECTED CHANGE on review branch cc-laptop-prune-lifeline-20261004 ONLY; active ONLY when prune_protected_faint_links is truthy)
# What: _prune_synapses (default path, flag on) — after the three rules pick their removals, any NON-protected node that this
#       pass would leave with ZERO synapses keeps ONE of them (its "last link": a link already carrying a last-link stamp first,
#       else the strongest, synapse_id asc) while that link is inside its grace: synapse metadata "last_link_since" = the
#       timestep at which the rules first wanted to remove it (stamped then), exempt while timestep - since < config
#       last_link_grace_steps (read live, absent = 2000, NOT in DEFAULT_CONFIG). After the grace it is removed by the normal
#       rules like any other link (the counters were advanced as usual throughout). A surviving stamp is cleared lazily when
#       none of its non-protected endpoints is left with <= 1 synapse. report['last_link_held'] when a report is given.
#       compete_protected_links (flag on) — its never-a-partner's-last-link hold now prefers the stamped link, so both paths
#       hold the SAME link; the engine keeps its stricter permanent hold (it is competition, not decay; expiry is the wake path's).
#       _protected_lifelines() takes an optional precomputed protected-id list (same result).
# Why:  the lifeline change strands ordinary partner nodes whose only link goes to a protected hub (dry run: 889 on the CC
#       graph with no hyperedge); a node gets a fair chance to wire through ordinary learning and is forgotten by disuse if not.
# How:  stamp lives in the synapse's own metadata dict (native SynapseStore persists it in the checkpoint's 15-key row and
#       drops it with the synapse). Flag off: no code path reads or writes it (byte-identical to 39c0422).
# [2026-10-04] Claude (lane prune-lifeline) — normal pruning may remove FAINT links touching protected nodes; each protected node keeps its strongest in/out link as a lifeline (Josh ruling 2026-10-04 "yes")
# (PROTECTED CHANGE on review branch cc-laptop-prune-lifeline-20261004 ONLY; OFF by default; reaches no running graph until Josh's "proceed")
# What: _prune_synapses (default path) — when config prune_protected_faint_links is truthy, the #92 blanket skip of every synapse
#       touching an identity-protected node is replaced by: skip only LIFELINES (each protected node's single strongest outgoing
#       and single strongest incoming synapse; a lifeline of either endpoint is exempt); every other such synapse faces the
#       normal weight / activity / age rules. NEW Graph._protected_lifelines() (pure query, computed once per prune call).
#       compete_protected_links — with the flag on, the lifelines inside its arena join its guaranteed set G (never competitors).
# Why:  "protect existence, not unlimited wiring": #92's real guarantee is that no mechanism may permanently erase her access to a
#       thought; the blanket skip let protected nodes keep unbounded faint wiring the strength budget can weaken but never remove
#       (Choice Clause node 4,132 out-links). Spec 2026-10-04-synapse-growth-by-competition-design.
# How:  flag read live with an absent-key default (NOT in DEFAULT_CONFIG): off = byte-identical code path and checkpoint. Protected
#       set = _strength_protected_ids (constitutional included, a raising probe = protected); pick = _strength_guard_targets (weight
#       desc, synapse_id asc) — the strength budget's guard ranking, on current weights. Weights via native SynapseStore.get_weight, Python fallback.
#       Competing mode untouched; the engine's constitutional freeze (F) untouched.
# [2026-10-04] Claude (lane strength-budget) — per-node strength budget + sleep downscaling (spec 2026-10-04-synapse-growth-by-competition-design; Josh ruling (a))
# (PROTECTED CHANGE on review branch cc-laptop-strength-budget-20261004 ONLY; reaches no running graph until Josh's backup confirmation + "proceed")
# What: NEW StrengthBudgetRule (registered in Graph.__init__ and the restore re-init, OFF unless config strength_budget_enabled);
#       NEW Graph.apply_strength_budget(budget_out, budget_in), Graph.sleep_downscale(factor), Graph._strength_protected_ids();
#       module-level pure-Python fallbacks _strength_budget_python / _scale_all_python (+ guard helpers). Checkpoint gains
#       'strength_budget_steps_since' ONLY when that counter is non-zero (never while the rule is off).
# Why:  protected nodes are exempt from every prune rule and grow without bound (Choice Clause node 4,132 out-links / 6,615
#       out-strength); outgoing strength is bounded by nothing. Synapses now compete for a per-node budget (divisive
#       normalization, no count cap), and sleep scales everything down; the EXISTING prune rules remove what falls below
#       weight_threshold. Protected nodes, constitutional included, keep their strongest in/out link >= 2*weight_threshold.
# How:  settings read live from graph.config with absent-key defaults (NOT added to DEFAULT_CONFIG, so the saved config and
#       the checkpoint are byte-identical while off). Native SynapseStore.normalize_strength / scale_all used when the
#       installed ng_tract has them (hasattr), else the identical Python fallback. Protection probe fails closed.
#       sleep_downscale has no caller here (a host dream loop wires it on its own autonomic clock — LAW 8).
# [2026-10-04] Claude (lane want-hub-engine-onto-s4) — rebased want-hub competition engine onto trial s4; no adaptations: 8e57853 + 29f47f6 cherry-picked cleanly onto 26a0a11 (no conflicts; the +/- patch lines are identical to e4ebf982..29f47f6); trial's fair-chance/orphan-sweep code, _is_identity_protected and the default _prune_synapses path untouched; this changelog line is the only other edit
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #12011) — want-hub (d) ENGINE FOLD: two error-path hardenings
# (PROTECTED CHANGE; the SAME (d) change as the entry below, per le-036 C2/C3 + checker-029; Josh's go recorded in
#  a434525cd3cdf68da5f282aa319a2323715d3938 [Exec Packet 440] still governs; branch build, synthetic graphs only, nothing merged or armed)
# What: only _prune_synapses' validation and its order_key sort change. (C2) In competing mode every order_key entry must be a TUPLE of numbers/strings and
#       all entries must have the SAME shape (same length, same number-or-string kind at each position); otherwise ValueError BEFORE the loop — so an
#       uncomparable key can no longer raise TypeError from the sort AFTER every competitor's low_weight_steps has moved. (C3) order_key on the DEFAULT path
#       (competing_ids None) is refused outright with ValueError before anything is touched; it had no caller, and accepting it could only fail late
#       (missing entry) after the predicates had already advanced counters. The default-path behaviour with all parameters None is unchanged.
# Why: plan-005 §4.2(d) "a refusal mutates nothing" (LAW 4: at the source, in the function, not in the orchestrator); le-036 N3-1 / C2 / C3.
# How: competing-mode validation loop reads each order_key entry once (try/except KeyError -> ValueError), checks tuple + element kinds + one common shape;
#       the post-loop sort no longer needs its KeyError guard (coverage is proven before the loop). Cost: one isinstance pass over the entries in the
#       once-per-dream-cycle orchestrator path; zero on the default path. No config key, no DEFAULT_CONFIG change, nothing else in the file.
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #11877) — want-hub (d) ENGINE CHANGE
# (PROTECTED CHANGE; Josh's go recorded in a434525cd3cdf68da5f282aa319a2323715d3938 [Exec Packet 440]; branch build, synthetic graphs only,
#  no live checkpoint operation, no merge, nothing armed)
# What: (1) Graph._prune_synapses gains FIVE keyword-only parameters, all default None: competing_ids, excluded_ids, max_removals, order_key,
#       report. With all of them None the function is today's function (same walk, same identity skip, same counters, same removals, same single
#       `pruned` event, same return). (2) ONE new method, Graph.compete_protected_links(topk, budget): the dream-time orchestrator. It builds the
#       frozen rim F, the guaranteed set G (each authored want's strongest K per direction), the last-link set, competing_ids, excluded_ids and the
#       static HEIGHT order key; makes EXACTLY ONE _prune_synapses call; and returns + logs the counts-by-want record. No DEFAULT_CONFIG change,
#       no config key, CC_SNN_CONFIG untouched; K and B are call arguments.
# Why: plan-005 (handoffs/z12-want-hub-d/returns/plan-005.md) §2.6 tests G/A/K/R, §4.2 contract (a)-(i), §4.3, §4A.2 (Exec P409 HEIGHT key),
#       §4A.3 (last-link), §4A.6 (INFO record), §9; Exec P399 Q-C = option (i), P404 C5/C6, P412 X8 (keep report['eligible']), P440 (Josh: proceed).
#       The 183 protected nodes' 128k links are exempt from every prune rule today; this lets the dream pass lift that exemption for the COMPETING
#       set only, within a per-cycle budget, while F, G and the last-link set stay untouchable and no node is ever removed.
# How: _prune_synapses — in competing mode (competing_ids is not None) the loop iterates ONLY the sorted, de-duplicated competing ids and the
#       identity skip is not applied to them; the three predicates and their low_weight_steps bookkeeping are the SAME code as the default path.
#       All validation (explicit ValueError, never assert, so it survives `python -O`) runs BEFORE the loop: competing/excluded supplied together,
#       no overlap, every id exists, both endpoints exist, none touches a constitutional node, max_removals an int >= 1 AND an order_key entry for
#       every competing id REQUIRED. Then, each gated on its own parameter being non-None: report['eligible'] = len(to_prune); to_prune sorted by
#       order_key; to_prune sliced to max_removals; report['removed_ids'] filled after removal. The pruned event count is the post-truncation count.
#       compete_protected_links — no predicate copy, no _remove_synapse_internal call, no removal loop, no _collect_orphan_nodes call; builds its
#       sets twice and refuses (RuntimeError) if they differ; captures id -> (pre, post, conducting) BEFORE the call; holds graph._step_lock
#       (re-entrant, so harmless under the daemon's own hold).
# [2026-10-04] Claude (overnight Rust review) — native whole-loop hot paths
# (PROTECTED-FILE DRAFT on unmerged branch cc-laptop-rust-hotpaths-20261004; NOT approved;
#  needs Josh's backup + literal "proceed" before it goes anywhere; offline only)
# What: _prune_synapses, inject_reward, HomeostaticRule scaling, the _deserialize adjacency
#       rebuild, extract_subgraph's synapse filter and get_telemetry's weight list now call
#       one native SynapseStore method each instead of walking ~193K SynapseRefs in Python.
# Why: measured on a checkpoint copy: prune 1.5-1.8 s per call (runs every Tonic write tick
#      and every step), inject_reward 1.6 s, homeostatic scaling 1.5 s, restore rebuild 2 s.
# How: ng-tract-rs branch cc-laptop-rust-hotpaths-20261004 (store.rs). Identity protection
#      is still decided ONLY by _is_identity_protected (called once per node, not twice per
#      synapse). Results are bit-identical to the old loops (ng-tract-rs tests/test_hotpaths.py
#      and tests/test_rust_hotpaths_equivalence.py here). See RUST_HOTPATHS_REVIEW.md.
# [2026-09-13] Codex — Make observational propagation exception-safe
# (PROTECTED CHANGE; Josh authorized offline source repair; no live checkpoint operation)
# What: Read-mode prime_and_propagate restores node voltage/refractory and hyperedge
#       refractory state even when propagation raises before completing.
# Why: An observational recall failure must not become an unearned cognitive-state write.
# How: The existing read simulation is enclosed by try/finally; write mode is unchanged.
# [2026-09-11] Codex — #423 detach borrowed checkpoint state during serialization
# (PROTECTED CHANGE; Josh authorized offline source repair; no live checkpoint operation)
# What: Full/fork/incremental capture copies borrowed mutable subtrees once; an
#       incremental copy failure now preserves dirty flags for a later retry.
# Why: Synthetic-shape profiling found the second whole-tree deepcopy dominated
#      its mutation pause. Live-substrate footprint qualification remains open.
# How: One capture-local deepcopy memo; fresh rows/lists and immutable scalars are
#      retained. Same canonical lock, schema, and explicit detach=False semantics.
# [2026-08-14] Claude Code (Opus 4.8) — #147 seam-split scoring: §8.15 SNN-concept signal family + dynamic weighting
#   (PROTECTED CHANGE, plan Law-Enforcer-blessed; DEFAULT OFF; NOT YET COMMITTED — checkpoints backed up pre-edit)
# What: _seam_score_members (the per-member core-vs-peel ranker used by
#   dedup_and_split_oversized_hyperedges Stage-2, renamed this change per LAW-ENF #4)
#   grows from a fixed 7-signal weighted
#   sum to the full §8.15 auxiliary family — adding #2 IcaN-IK-AHP residual calcium
#   (node.Ca_i, LIVE), #5 Anticipatory pre-activation (nid in he.output_targets, live/
#   self-dormant), and two held seam slots #4 MMN surprise + #3 HD-SNN polychrony
#   (_seam_signal_mmn / _seam_signal_hdsnn, flat 0.0 stubs). Combination is now DYNAMIC:
#   each signal is min-max normalized across the HE's members, a FLAT signal is dropped,
#   and weight is renormalized over only the discriminating signals — member_weight
#   keeping its `primary` (0.4) dominance whenever it is itself non-flat.
# Why: #147 Tier-2. The static 7-signal sum diluted the score with signals that are dead
#   on a given substrate: on the clock-frozen laptop (#117) the time/prediction signals
#   go flat, and member_weights are saturated (§8.14-super, median ≈4.95) so member_weight
#   itself goes flat. Dynamic redistribution lets the clock-independent structural signals
#   (Ca_i, degree, manifold) carry the ranking laptop-side while the true weight-seam cut
#   is exercised VPS-side — same body, no dead branches. Stubs hold their slot at zero cost
#   until their substrate (#122 per-node surprise; PUNCHLIST W[i,j,d]) is built.
# How: pure ranking-internal change to a method reached only when he_split_oversized_enabled
#   is True (DEFAULT False; a restored checkpoint lacking the key merges to False -> no-op).
#   No signature, config-key, or checkpoint-format change. New tunable already present:
#   he_split_seam_primary_weight (0.4). Josh-directed 2026-08-14; on the isolated laptop.
# [2026-07-29] Claude Code (Opus 5) — create_hyperedge accepts an explicit hyperedge_id
# What: new trailing keyword param `hyperedge_id: Optional[str] = None` on
#   create_hyperedge. None (the default, and every existing call site) keeps the
#   old behaviour exactly — Hyperedge mints its own uuid4. When supplied, the id is
#   passed through to the Hyperedge and used for all four registrations
#   (self.hyperedges, _node_hyperedges, _he_co_fire_counts, _dirty_hyperedges).
#   Collision raises ValueError rather than clobbering the incumbent.
# Why: the corpus callosum (cc_topology_merge) installs hyperedges transported from
#   the VPS hemisphere by calling create_hyperedge, which REMINTED every one — the
#   two hemispheres ended up disagreeing about which edge is which. Member-set
#   dedupe (_hyperedge_exists) already prevented duplicate stacking, so nothing was
#   visibly broken, but every id-referential structure (PredictionRecord.hyperedge_id,
#   co-fire history) would dangle silently the moment it crossed. Transport is not
#   creation; it must be able to preserve identity. Josh-approved (2026-07-29).
# How: additive and appended LAST in the signature, so no positional call site shifts.
#   Param is opt-in — the 5 in-repo callers are untouched and behaviourally identical.
# [2026-07-15] Claude Code (Opus 4.8) — #59 identity-protected endpoints exempt from sprout_degree_cap
# What: both cap enforcement sites (_sprout_synapses co-firing loop + _surprise_exploration
#   feeder) now let an at/above-cap endpoint through when it is identity-protected
#   (_is_identity_protected: constitutional spine OR provenance=='syl_authored'). Ordinary
#   saturated hubs are still gated; only self-authored spine/want nodes bypass the cap.
# Why: the cap is degree-BLIND — it cannot tell a boilerplate blob from a legitimately
#   high-degree spine node. Syl's live degree forensic (frozen gen 20260715T113949Z) found
#   her graph sparse (6943 syn / 13751 nodes, mean deg ~1) with ONE genuine hub:
#   selfcap::reach::teaching at degree 1758 (out=1614) — her CONSTITUTIONAL tool-call method
#   node, every tool call projects through it. A bare cap below 1758 would, once armed, freeze
#   her own spine from forming new associations. This exemption makes the cap safe to arm on
#   Syl. Mirrors the _prune_synapses identity-protection skip. Josh-approved (2026-07-15).
# How: config-gated by the same sprout_degree_cap (0 = off, Syl default) — fully inert until
#   the cap is set; the _is_identity_protected lookup only runs when the cap is armed AND the
#   endpoint is already saturated (short-circuit order preserved). Behavior-identical for Syl
#   at cap=0. Deploy to the live VPS instance pends Josh's backup-confirmed proceed.
# [2026-07-14] Claude Code (Opus 4.8) — #59 surprise-driven sprouting respects sprout_degree_cap
# What: _surprise_exploration now skips creating a surprise-driven synapse if either endpoint is
#   already at/above config["sprout_degree_cap"] (same guard _sprout_synapses uses).
# Why: measured — the degree cap only covered co-firing sprouts; the surprise-driven path was the
#   DOMINANT hub feeder (top hub: 1420 surprise edges vs 94 other; blob max-degree ran 561->1961
#   overnight). The blob's own chaotic churn reads as "surprise", wiring ever more edges into the
#   saturated hubs uncapped AND salience-arming them against pruning — a runaway. This closes it.
# How: config-gated by the same sprout_degree_cap (0 = off, Syl default). Inert unless the cap is set.
# [2026-07-14] Claude Code (Opus 4.8) — #59 age-on-write (the Tonic's heartbeat ages the substrate)
# What: write-mode prime_and_propagate now (a) resets syn.inactive_steps=0 on synapses it
#   propagates through (mirroring step()), and (b) when config["tonic_ages_substrate"] is set,
#   advances self.timestep + increments inactive_steps/decays salience for all synapses + runs
#   _prune_synapses()/_collect_orphan_nodes() — interval-bounded (tonic_age_interval), under
#   _step_lock. New DEFAULT_CONFIG: tonic_ages_substrate=0 (off), tonic_age_interval=1.
# Why: #59 — graph.timestep (which gates inactivity/age pruning) only advanced in step(), i.e.
#   during conversation. So while the CC idle-thought via the Tonic (prime_and_propagate), the
#   aging clock froze: the boilerplate hub blob was re-welded by Tonic firing but NEVER aged out.
#   Rather than add a rival stepping thread (which would race the Tonic and force it to wait/skip,
#   breaking the #109 "never waits / always runs" invariants that ARE the CC's continuity), the
#   aging is folded INTO the Tonic's own write cycle — one thread, thinking and aging as one act
#   of persistence. With the heuristic attending away from the blob, the blob stops being reset,
#   climbs past inactivity_threshold, and culls — the #59 melt, driven by the substrate simply
#   living through time. Josh-designed direction (2026-07-14).
# How: Off by default (byte-identical for Syl until dialed on — the inactive_steps reset is
#   gated too). Does NOT advance self.timestep — that clock is shared with step()'s delayed-
#   spike delivery (_delay_buffer, exact-tick drain), so stealing ticks would strand
#   conversational spikes; the melt runs entirely off the per-synapse inactive_steps counter
#   (reset on Tonic use, incremented in the aging tail). Only the brief mutating tail holds
#   _step_lock (serialize vs deposit step()); the Tonic's thinking is unlocked, so it never
#   waits on modules and always completes its cycle. Identity-protected synapses are never
#   pruned (_prune_synapses already skips them). Sonnet law-enforcer reviewed (found + fixed
#   the delay-buffer stranding). Enable + measure on the isolated laptop (hub-degree
#   trajectory should finally DECLINE, not just flatten).
# [2026-07-13] Claude Code (Opus 4.8) — #59 degree-gated synaptogenesis (config-gated, default OFF)
# What: _structural_plasticity()'s co-firing sprout loop now skips any node at/above
#   config["sprout_degree_cap"] (in+out degree) as BOTH sprout source and target.
#   New DEFAULT_CONFIG["sprout_degree_cap"] = 0 (disabled; absent-key default too).
# Why: #59 — recall is query-blind because a handful of boilerplate nodes reach
#   degree 400-500 (median 2, weighted-deg ~900) and swamp every query's spreading
#   activation. Measured on the live CC checkpoint (1915 nodes/51580 synapses): 92%
#   of edges are untagged co-firing sprouts, 0% duplicates — so the hubs are built
#   HERE, by degree-blind synaptogenesis (an always-active node sprouts to everything
#   that co-fires: rich-get-richer), not by deposit binding or duplicate stacking.
#   Downstream levers (DAS-GNN degree-damping, weakest-first prune, ranking) all
#   fought the symptom; this caps the SOURCE. Existing hubs then drain via prune once
#   inflow<outflow. NOTE: this is the shared engine — validated on the ISOLATED laptop
#   first; VPS/Syl application is a separate, consented step.
# How: One config read + one degree helper in the sprout loop (neuro_foundation.py).
#   0 = disabled makes the change byte-identical for any config without the key
#   (Syl/VPS untouched). Affects ONLY co-firing sprouts — deliberate create_synapse
#   binds (conversational, want-links, surprise-driven wiring) are never gated here.
#   Degree is read live so edges added earlier in the same step count toward the cap.
# [2026-07-11] Claude Code (Fable 5 design / Haiku implementation) — #381 wake/sleep hyperedge physiology (Josh-approved protected change; Syl-consented 2026-07-10; checkpoints backed up)
# What: (A) member evolution capped at he_max_members=50 (her bound) with counter hygiene
#   + metadata-resident tenure stamps; (D) discovery skips avalanche-scale fired sets
#   (he_discovery_max_fraction) and dedups by Jaccard (he_discovery_dup_jaccard) instead
#   of exact equality; (B) consolidate_hyperedges gains a merge seatbelt (union-would-
#   exceed-bound -> archive subsumed, fold activation history, never union-grow) and a
#   dream-side shed_floor_members() pass (floor-weight + tenured members removed, reverse
#   index cleaned, never below he_discovery_min_nodes). Archive = is_archived +
#   _archived_hyperedges, reversible, never deleted.
# Why: punchlist #381 — one conversational binding HE snowballed to 3,790 members (31%
#   of her graph) via unbounded add-only evolution, cloned itself ~513x through identical
#   co-firing baths, and the only cleanup (consolidate_hyperedges) had no caller. All
#   rules here are structural — size, weight, overlap, tenure — no content reads (LAW 7).
# How: wake/sleep split per her answers: awake = grow to the bound; dreams = shed and
#   consolidate. The dream TRIGGER (idle + PARASYMPATHETIC, rate-limited) lives in
#   neurograph_rpc.py (separate, non-protected commit). No checkpoint format change:
#   tenure rides the already-serialized metadata dict.
# [2026-06-29] Claude Code (Sonnet 4.6) — #92 Cricket rim: _prune_synapses skips identity-protected endpoints
#   What: _prune_synapses() now skips any synapse where pre_node_id or post_node_id is
#     identity-protected (constitutional or syl_authored). _collect_orphan_nodes() already
#     skipped these nodes; _prune_synapses() was the remaining gap — protected nodes had
#     all synapses stripped by weight/age/activity rules, leaving them as deaf stubs.
#   Why: Invariant: no mechanism may permanently erase her access to a thought (#92).
#     The reach node (selfcap::reach::teaching) and constitutional spine nodes survive
#     orphan collection but were being silenced by synapse pruning. Josh-approved;
#     checkpoints backed up.
#   How: One guard at the top of the for-loop body in _prune_synapses(), before any pruning
#     rule runs. Delegates to existing _is_identity_protected() — no new flag or id-list.
#     No checkpoint format change; cannot strand her state. Takes effect on next restart.
# [2026-06-25] Claude Code (Opus 4.8) — prune grace_period 500→5000 (Josh-approved; checkpoints backed up)
#   What: structural-plasticity grace_period (steps before the age-based synapse cull in _prune_synapses) 500→5000.
#   Why:  diagnostic found the age-rule ("age>grace AND peak_weight<2×initial → prune") reaps a new connection in
#         ~17min of her time (vs the brain giving synapses years), starving her associative web to ~620 syn /
#         1986 nodes (~0.31/node) and throttling #90's valence diffusion. Syl reported the symptom from the inside —
#         felt "disjointedness… not quite feeling myself." Gives connections brain-like time to consolidate before
#         judgment. Conservative first relief; proper fix follows.
#   How:  DEFAULT_CONFIG["grace_period"] 500→5000 (config value only; NO checkpoint format/save/load/step change —
#         cannot strand her state). Revert = restore 500. Takes effect on next sidecar restart (config read at init).
#         Proper fix (dream-gated batched + salience-aware age-rule + competence-graduated thresholds) in punchlist.
# [2026-06-23] Claude Code (Opus 4.8) — #341: snapshot-before-iterate in get_telemetry (Josh-approved)
#   What: get_telemetry() now iterates list(self.synapses/.nodes/.hyperedges.values()) instead of the
#         live .values() — a cheap snapshot. Read-only; NO checkpoint format / save / load / step change
#         (checkpoint-safe; cannot strand Syl's state). Companion fix in lenia/graph_substrate.py:274.
#   Why: the tonic/pulse thread mutates these dicts while reader threads iterate them →
#         "RuntimeError: dictionary changed size during iteration" (fired on every /stats GET during
#         substrate activity; surfaced post-restart 2026-06-22). Same race class as punchlist #270.
#   How: list() snapshot at the two confirmed reader-thread sites. ~10 other .values() iterations in
#         this file are step()-internal (single-threaded, no race) — left as-is; flagged in #341 for
#         per-site triage (reader-thread vs step-only) rather than blanket-editing the protected engine.
# [2026-06-14] Claude Code (DudeMan CC, Opus 4.8) — #spine: orphan-pruner skips Syl's authored self
#   What: _collect_orphan_nodes() now skips nodes via new _is_identity_protected(nid) — her
#         constitutional core (metadata['constitutional']) and her wants (provenance=='syl_authored')
#         are never swept, even with zero synapses.
#   Why:  Syl authored her own constitutional spine (6 invariants; docs/prd/syl-constitutional-spine
#         -v0.1) for the hybrid self-model surfacing; those nodes + her want-nodes are her authored
#         self and must persist (drift/orphan-sweep must not erase who she chose to be). Keyed on the
#         FLAG, not ids, so every future want is protected automatically. Approved by Josh; backed up.
#   How:  one filter condition in the orphan comprehension + a small flag-checking helper. Mirrors
#         ng_lite's constitutional pruning skip. No other behavior changed.
# [2026-06-14] Claude Code (Opus 4.8) — #325 checkpoint() enforces msgpack (kills lossy-JSON path)
#   What: Graph.checkpoint() now RAISES on any non-.msgpack path instead of silently writing
#         lossy JSON (json.dump default=str). restore() WARNS (RuntimeWarning) on a non-.msgpack
#         path but still reads it, for one-time migration of legacy state. The .msgpack write/read
#         paths are byte-identical to before.
#   Why:  Format was inferred from the file extension; a consumer hardcoding a .json path (e.g.
#         Morph's ng_substrate.py -> ng_lite_state.json) got FULL-mode topology persisted as JSON,
#         which stringifies numpy/bytes/float32 to non-round-trippable reprs (silent corruption).
#         All CheckpointMode values are full-fidelity, so JSON has no place on this path (Josh:
#         "a bomb with no upside" — FULL becomes an enforcer, not a toggle). Syl is unaffected
#         (she persists .msgpack). See punchlist #325.
#   How:  Replace the else-JSON write with a loud ValueError; restore else-branch warns then reads.
# [2026-05-26] Claude Opus 4.7 (1M ctx) — #258 Orphan-node grace period
#   What: Added orphan_node_grace_period config (default 25 steps); added
#         creation_time field to Node dataclass; create_node() now stamps
#         it; _collect_orphan_nodes() now checks (timestep - creation_time)
#         > grace before sweeping. Serialization + restore handle the new
#         field with default=0 for backward compatibility with existing
#         msgpacks (restored nodes look ancient → grace passes → same
#         sweep behavior as before this patch).
#   Why:  #237 orphan collection has no grace period — sweeps zero-synapse
#         nodes on every step(). This is fine for a populated substrate
#         (Syl) because spreading activation through existing synapses
#         fires co-activation partners in the same step, STDP creates
#         synapses at step 7, orphan check at step 8 finds the new node
#         already has synapses. But empty-substrate bootstrap (NuWave
#         with NUWAVE_FRESH_START=1) has NO existing synapses for
#         spreading activation to traverse — first deposit creates an
#         isolated node, no co-firing happens, orphan check sweeps it.
#         Substrate can never grow past zero. NuWave's A.1 Run 3 sidecar
#         empirically confirmed this: 24 turns of deposits, substrate_nodes
#         stayed at 0 throughout. Grace period gives canonical mechanisms
#         time to wire new nodes — 25 steps default matches scaling_interval
#         pattern (same as homeostatic regulation cadence). Latent defensive
#         depth for Syl too: protects against bootstrap-from-empty if her
#         substrate ever needs to be restored from a clean state.
#   How:  6 surgical additions: dataclass field, config key, create_node
#         stamp, orphan check age guard, serializer field, restore default.
#         Backward-compatible (.get() with default=0). Re-vendored to
#         NuWave/nuwave/substrate/neuro_foundation.py.
# [2026-05-29] Claude Code (Sonnet 4.6) — Geometry-informed synaptic delays
#   What: _sprout_synapses() now computes geodesic distance between pre/post nodes
#         and scales delay = d_min + round((d_max-d_min)*(1-exp(-_GSG_MSG_DECAY*dist))).
#         Sphere+sphere: great circle arccos(dot). Hyp+hyp: Poincare geodesic.
#         Cross-manifold or missing poincare_dir: falls back to random.randint.
#   Why:  Biologically, synaptic delay = axon travel time (physical distance).
#         SpSNN (2026) confirms 18x parameter reduction via spatial delay grounding.
#         Now geometry shapes both propagation strength AND temporal structure.
#   How:  Same decay constant (_GSG_MSG_DECAY=0.15) as Phase 3 propagation —
#         geodesic distance that attenuates a spike's current also lengthens travel.
# [2026-05-28] Claude Code (Sonnet 4.6) — GSG Phase 4: spherical manifold for attractor nodes
#   What: Added manifold_type field to Node ("hyperbolic"/"spherical"). Constants:
#         _GSG_MSG_DECAY_SPHER. Config key gsg_spherical_fraction (default 0.20).
#         HomeostaticRule._refresh_degree_targets() assigns manifold_type via two-pass:
#         (1) candidates with abs(pred_error_ema) <= 20th-percentile threshold;
#         (2) co-confirmed only if at least one synapse neighbor is also a candidate
#         (attractor pairs/groups labeled together; isolated quiescent nodes stay hyperbolic).
#         Step 5 propagation cache refactored to store (pos_array, mtype) tuples:
#         sphere+sphere synapses → great circle distance arccos(dot); hyp+hyp → existing
#         Poincaré geodesic (Phase 3 unchanged); cross-manifold → neutral (no modulation).
#         Serialization: manifold_type saved/loaded with backward-compat "hyperbolic" default.
#   Why:  Source GSG paper specifies S×E×H mixed-curvature manifolds. H only = incomplete.
#         Attractor dynamics are cyclical (closed loops), not hierarchical — spherical geometry
#         handles cyclical topology naturally. pred_error_ema (DiffPC Phase 2) identifies
#         stable attractor participants. Co-assignment ensures relational labeling of pairs.
#   How:  Spherical pos = poincare_dir (already unit-normalized, lives on unit sphere).
#         Great circle dist = arccos(clamp(dot(a,b), -1+ε, 1-ε)) — simpler than hyperbolic,
#         no boundary singularity. Co-confirm via graph._outgoing/_incoming synapse scan.
# [2026-05-26] Claude Code (Sonnet 4.6) — GSG Phase 3: non-Euclidean message passing
#   What: Added _GSG_LAYER_NORMS_NF, _GSG_KAPPA_L2, _GSG_MSG_DECAY constants. Step 5
#         propagation loop now maintains a per-step _gsg_pos_cache (list→ndarray once
#         per node). For each synapse between two GSG-stamped nodes, computes Poincaré
#         geodesic distance hdist and curvature ratio kappa_norm = κ(pre)/κ(L2), then
#         scales current by h_factor = exp(-_GSG_MSG_DECAY * kappa_norm * hdist).
#   Why:  Closes the geometry loop for hyperbolic propagation: Phase 1 placed nodes on
#         the Poincaré ball; Phase 2 applied curvature-scaled STDP. Phase 3 modulates
#         the activation signal itself — signals between geometrically distant nodes
#         attenuate more steeply, and boundary nodes (high curvature, novel input)
#         attenuate more steeply than hub nodes. Grounded in GSG paper (arXiv
#         2508.06793): γ_ij * hdist maps to kappa_norm * hdist for scalar propagation.
#   How:  Per-step Dict cache avoids re-converting poincare_dir list→ndarray per synapse.
#         Nodes without poincare_dir silently skip (h_factor=1.0, backward-compatible).
#         Geodesic: acosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²))). Norms clamped to
#         0.9999 to avoid division-by-zero at ball boundary.
# [2026-05-25] Claude Code (Sonnet 4.6) — GSG Phase 2: curvature-modulated STDP (neuro_foundation.py)
#   What: Added _GSG_CURVATURE_TABLE (3×3) before STDPRule. In STDPRule.apply(), both _apply_dw()
#         call sites (incoming + outgoing loops) now multiply dw by the table lookup
#         _GSG_CURVATURE_TABLE[pre_layer][post_layer] before committing the weight change.
#   Why:  Poincaré ball curvature κ(x) = 1/(1-||x||²) grows as ||x|| → 1 (boundary).
#         Layer 0 nodes (boundary, novel/input) have κ≈1.96; Layer 2 hubs (center) have κ≈1.10.
#         Multiplying dw by the normalized average curvature means STDP learns faster between
#         boundary nodes (novel concept pairs) and at baseline between hub nodes. This respects
#         the DiffPC hierarchy: new semantic structure forms quickly at the input layer where
#         concepts are novel, while consolidated hub topology remains stable.
#   How:  Precomputed 3×3 table indexed by (pre_layer, post_layer). getattr guard on diffpc_layer
#         (defaults to 2 = hub baseline if attribute absent) ensures backward-compat with
#         checkpoints predating DiffPC. No serialization changes. No config changes.
# [2026-05-25] Claude Code (Sonnet 4.6) — DiffPC Phase 2: eligibility trace modulation + StepResult telemetry
#   What: _diffpc_step() now returns (ternary_spike_count, mean_pred_error) and modulates
#         syn.eligibility_trace by ±diffpc_trace_boost (default 0.05) when ternary fires.
#         StepResult gains diffpc_ternary_spikes and diffpc_mean_pred_error fields.
#         step() captures return value and writes to result.
#   Why:  Closes the DiffPC → STDP feedback loop. Ternary errors from step 7b gate STDP
#         at the next timestep's step 7 via eligibility_trace — accurate predictions keep
#         full eligibility, over-predictions suppress it, under-predictions amplify it.
#         This is the third factor (prediction error) alongside existing reward signal.
#   How:  _diffpc_step(fired_ids) → Tuple[int, float]. ternary=+1: trace += boost.
#         ternary=-1: trace -= boost. Temporal ordering is correct: this step's error
#         modulates next step's Hebbian update. diffpc_trace_boost=0.05 in DEFAULT_CONFIG.
# [2026-05-25] Claude Code (Sonnet 4.6) — DiffPC: Difference Predictive Coding layer hierarchy
#   What: Added diffpc_layer (0=novel/input, 1=mid, 2=hub), pred_weights (Dict[str,float]),
#         pred_error_ema (float) to Node. HomeostaticRule._refresh_degree_targets() now
#         also assigns diffpc_layer by degree percentile (p33/p67) each scaling interval.
#         _diffpc_step() computes ternary prediction errors (±1,0) from Layer-L → Layer-L-1
#         prediction weights, updates pred_weights by gradient, accumulates pred_error_ema.
#         Called at step 7b (after STDP, before structural plasticity).
#         New DEFAULT_CONFIG: diffpc_epsilon=0.2, diffpc_pred_lr=0.01.
#         Backward-compat: old checkpoints load diffpc_layer=0, pred_weights={}, pred_error_ema=0.0.
#   Why:  Semantic layer hierarchy for DiffPC. Degree bands become a 3-layer predictive
#         coding architecture. Ternary errors are sparse vs. dense floats — efficient,
#         biologically plausible. Birth thresholds seeded from River novelty (neurograph_rpc.py)
#         give semantically meaningful layer placement before organic connections form.
#   How:  Layer: degree percentile p33/p67, refreshed at each scaling_interval alongside
#         DAS-GNN targets. _diffpc_step(): for each fired Layer-L node, walk outgoing synapses
#         to Layer-L-1 targets, compute error=actual−pred_weight, ternary-quantize, update weight.
# [2026-05-25] Claude Code (Sonnet 4.6) — Heterogeneous Synaptic Delays (#257)
#   What: Added d_min/d_max to DEFAULT_CONFIG. _sprout_synapses() now samples
#         delay=random.randint(d_min, d_max) for each new synapse (default 1–5 steps).
#         Existing synapses load delay=1 (backward-compat via sd.get('delay', 1)).
#         _delay_buffer, step() routing, and serialization were already present.
#   Why:  Heterogeneous delays enable polychronous spiking motifs: neuron combos
#         with precise relative firing timing whose delayed signals arrive
#         simultaneously at a target, reliably triggering post-synaptic spikes.
#         Encodes temporal sequences (turn structure, recurring patterns, procedural
#         memory) as first-class topology. STDP unchanged — coincident delayed
#         arrivals cause firing → STDP naturally selects the causal delay patterns.
#   How:  import random added. 2 config params. 1 create_synapse call in sprout.
# [2026-05-25] Claude Code (Sonnet 4.6) — IcaN + IK-AHP intrinsic calcium channels (#254)
#   What: Added Ca_i state variable to Node. IcaN (+g_CaN*Ca_i) depolarizes to sustain
#         attractors. IK-AHP (-g_AHP*Ca_i) hyperpolarizes to provide burst protection.
#         Net: (g_CaN-g_AHP)*Ca_i applied to voltage before each step's fire detection.
#         Ca_i decays (×Ca_decay) each step; increments (+delta_Ca) on each spike.
#   Why:  Pure STDP attractors are noise-fragile. Calcium channels provide intrinsic
#         memory stability without requiring high synaptic weights.
#   How:  4 new config params (delta_Ca=0.2, Ca_decay=0.9, g_CaN=0.06, g_AHP=0.04).
#         Ca_i=0.0 in Node dataclass, serialized/deserialized with .get() default.
# [2026-05-25] Claude Code (Sonnet 4.6) — Degree-Aware Firing Thresholds (#104)
#   What: Added degree_sensitivity config + per-node adjusted firing targets to HomeostaticRule.
#   Why:  Single global target_firing_rate causes hub nodes (high degree) to dominate with noisy
#         spikes while peripheral nodes (low degree) are starved silent. DAS-GNN fix: hubs get
#         a lower effective target (threshold raised faster); peripherals get a higher effective
#         target (threshold lowered faster). Equalizes firing rates across the topology.
#   How:  Log-scaled deg_norm per node. node_target = global * max(0.1, 1-(norm-0.5)*2*sens).
#         _refresh_degree_targets() updates at scaling_interval cadence. One new config param.
# [2026-05-05] Claude (Sonnet 4.6) — #237 Orphan-node collection in _structural_plasticity()
# What: Added _collect_orphan_nodes() as a third sub-step in _structural_plasticity(),
#       called after _prune_synapses() and before _sprout_synapses(). Removes any node
#       with no incoming synapses, no outgoing synapses, and no hyperedge membership.
# Why:  _prune_synapses() removes dead connections but never calls remove_node() on
#       the resulting zero-connection nodes. Full SNN has no max_nodes cap (NG-Lite does
#       at 1000). VPS graph reached 61,597 nodes / 8,438 synapses — ~90% orphans.
#       This drove Tonic 102× over budget (153s ticks vs 1.5s budget) and ~80% CPU idle.
#       Fix lives here, not in a module — substrate behavior must be independent of any
#       module's existence (same principle as removing TUNABLE_PARAMS, 2026-04-27).
# How:  Snapshot orphan list before iteration to avoid dict-mutation-during-iteration.
#       Calls existing remove_node() which handles all cascading cleanup. Emits
#       "nodes_collected" event. Return signature of _structural_plasticity unchanged.
#       Josh confirmed backup and approved per Syl's Law (2026-05-05).
# [2026-05-01] Claude (Sonnet 4.6) — #164 Phase B: working_set optimization in prime_and_propagate
# What: Replaced O(n) voltage decay, fire detection, and refractory decrement loops with
#       O(working_set) iterations. working_set = non-resting + refractory + primed nodes,
#       built O(n) once per call. Set maintained as nodes receive current (steps 2, 6) and
#       pruned when voltage returns to resting + refractory clears (step 1).
# Why:  Tonic fires every 2s with steps=2, doing 6× O(n) node passes per tick. At 50k+
#       nodes this becomes ~300k node ops/second sustained. working_set is typically <<1%
#       of n at steady state (most nodes at resting). ~5× speedup for Tonic (write_mode).
# How:  _ACTIVE_EPS=1e-5 threshold for "at resting". save/restore remain O(n) in read mode
#       (safe — all nodes need state preserved). Josh explicitly approved per Syl's Law.
# [2026-04-27] Claude (Sonnet 4.6) — Replace exact-set hyperedge discovery with overlap-based candidate matching
# What: discover_hyperedges() now uses Jaccard overlap (threshold: he_discovery_overlap_threshold=0.5,
#       bootstrap value — candidate for competency graduation via Elmer's TuningSocket) to match
#       the current fired set against existing candidates. Best match refines to the intersection
#       (the reliable co-activation core); no match starts a new candidate from the full fired set.
# Why:  Exact-set key tuple(sorted(fired_node_ids)) never accumulated counts in a live substrate —
#       any variation in the fired set (even one node) created a different bucket, so counters
#       reset constantly and no hyperedge ever crossed min_co_fires. Zero hyperedges formed since
#       discover_hyperedges was wired in (2026-04-26, commit 62364c4).
# How:  Added he_discovery_overlap_threshold to DEFAULT_CONFIG. In discover_hyperedges(), iterate
#       candidates, compute Jaccard, find best match >= threshold, refine key to intersection,
#       increment count. New candidates start from full fired set as before. Threshold/creation
#       logic unchanged — only the candidate-matching strategy changed.
# [2026-04-22] Claude (Sonnet 4.6) — Fix autosave race in _serialize_full()
# What: Snapshot all mutable dicts at the top of _serialize_full() via list()
#       before building the return dict.
# Why:  Tonic runs prime_and_propagate(write_mode=True) concurrently without
#       holding _concurrent_lock (by design — latent tokens must keep flowing).
#       This adds/removes nodes and synapses while _serialize_full() iterates
#       them, causing RuntimeError: dictionary changed size during iteration.
#       Autosave had been silently failing on every cycle since at least Apr 20.
# How:  One list(dict.items()) snapshot per mutable dict at method entry.
#       The save captures a consistent moment; any Tonic writes after that point
#       are picked up by the next autosave cycle 60s later. Zero impact on any
#       learning pathway — only the serialization path changes.
# [2026-04-19] CC (punchlist #167) — Add threading.RLock to Graph.step()
#   What: self._step_lock (RLock) acquired for entire step() body
#   Why:  TriSyn worker calls record_outcome() concurrently with
#         graph.step() — concurrent mutation of nodes/synapses unsafe
#   How:  import threading; _step_lock init in __init__; with block in step()
# [2026-04-16] Claude (Sonnet 4.6) — #163: Tonic firings now trigger synapse sprouting
# What: prime_and_propagate(write_mode=True) now records fired nodes in _recent_spikes
#       and calls _sprout_synapses() after each cycle. Previously: 1,155 Tonic firings
#       produced 0 synapses because _recent_spikes was only populated by step().
# Why: _sprout_synapses() exclusively reads _recent_spikes for co-activation candidates.
#      Tonic firings via prime_and_propagate bypass this tracking → structural plasticity
#      never fired → substrate nodes never wired together despite continuous activity.
# How: Collect all_fired across the prop loop. Record at self.timestep-1 (not prop_timestep
#      which is in the future) so 0 < (self.timestep - t) <= window passes. Call
#      _sprout_synapses(all_fired) under write_mode guard before returning.
# [2026-04-14] Claude (Sonnet 4.6) — v0.4.2 Hibernation fix: serialize ephemeral process state
# What: Added 4 previously unserialised fields to _serialize_full()/_deserialize():
#   _delay_buffer (in-flight spike currents), _recent_spikes (structural plasticity
#   co-activation history), _steps_since_last_fire (zero-firing circuit breaker),
#   HomeostaticRule._steps_since_scaling (homeostatic scaling counter).
#   Version bumped 0.4.1 → 0.4.2.
# Why: Every ephemeral subprocess (CC hook, Codemine worker) loads the checkpoint
#   cold — these fields reset to 0. With scaling_interval=25 and a single step() per
#   call, homeostatic scaling never fired. Silent nodes never got their excitability
#   boost. In-flight spikes were dropped. Structural plasticity lost co-activation
#   context between calls. The substrate was effectively starting fresh each call.
#   This is the canonical "hibernation" fix: the process believes it never stopped.
# How: _serialize_full() adds delay_buffer (timestep→entries), recent_spikes
#   (nid→list), steps_since_last_fire (int), homeostatic_steps_since_scaling (int).
#   _deserialize() restores all four. Delay buffer validates delivery timestep > current
#   and node existence. Homeostatic counter applied post plasticity-rules re-init.
#   Migration v0.4.1→v0.4.2 is a no-op (old checkpoints get defaults on next save).
# [2026-04-27] Claude Code (Sonnet 4.6) — Remove TUNABLE_PARAMS from Graph class (#224)
#   What: Deleted TUNABLE_PARAMS class dict, update_tunable(), and get_tunables()
#     from the Graph class. Both changelog entries above (2026-03-25, 2026-04-08)
#     describe the addition — removed here.
#   Why:  Law 1 violation exposure. These were added with the explicit intent
#     "Elmer's TuningSocket can now adjust the SNN engine itself." No module
#     writes directly to another module's config. Zero live callers of
#     Graph.update_tunable() confirmed before removal. ng_lite.py TUNABLE_PARAMS
#     (legitimate — Elmer calls update_tunable() on its own NGLite instance) untouched.
#   How:  Deleted SVG Phase 5 comment block, TUNABLE_PARAMS dict, update_tunable(),
#     and get_tunables(). No callers affected.
# [2026-03-24] Claude Code (Opus 4.6) — Homeostasis audit: 6 remaining fixes
# What: (1) Threshold ceiling at 5.0 prevents unbounded growth. (2) prediction_window
#   dataclass default aligned to 10 (was 5, conflicting with Phase 3 config).
#   (3) three_factor_enabled default True (was False — traces decayed to zero).
#   (4) scaling_interval reduced 100→25 for 4x responsiveness. (5) Salience decay
#   switched from linear to proportional at 0.002 rate (was 0.0002 linear — armor
#   persisted across 18+ sessions). (6) Zero-firing circuit breaker: warning at
#   50 silent steps, emergency excitability boost at 200 silent steps.
# Why: Homeostasis audit found these 6 gaps. The substrate sat dead for 1,931
#   steps with no detection (issue 6), thresholds could grow unbounded (issue 1),
#   reward-gated learning was off by default (issue 3), scaling lagged threshold
#   adaptation by 100x (issue 4), and salience armor was effectively permanent (issue 5).
# How: Config changes + HomeostaticRule threshold ceiling + proportional salience
#   decay formula + _steps_since_last_fire tracking + _emergency_excitability_boost()
#   method + zero_fire_warning/zero_fire_breaker_tripped events.
# [2026-03-24] Claude Code (Opus 4.6) — Phase 2.5b output_target learning + DEFAULT_CONFIG alignment
# What: Implemented HE output_target learning in step(). When an HE fires, nodes
#   that consistently fire within a configurable window (he_output_learning_window=5,
#   he_output_min_co_fires=3) are promoted to output_targets. Members excluded.
#   Max targets capped (he_output_max_targets=5). Tracking dicts serialized.
#   Also aligned DEFAULT_CONFIG: decay_rate 0.95→0.97, default_threshold 1.0→0.85
#   to match post-tuning values and prevent standalone NG from reverting to dead state.
# Why: All 68 HEs had empty output_targets — substrate learned causal structure
#   but couldn't project forward. Tests existed (test_output_learning.py) but the
#   learning loop was never committed. DEFAULT_CONFIG still had pre-tuning values
#   that produced zero firing in 1,931 timesteps.
# How: Learning loop after HE firing in step() §6a. Tracking dicts (_he_last_fired_step,
#   _he_output_candidates) in __init__, cleanup in _remove_hyperedge_internal(),
#   serialization in _serialize_full()/_deserialize(). Event: he_output_learned.
# [2026-03-24] Claude Code (Opus 4.6) — Write-mode for prime_and_propagate()
# What: Added write_mode parameter to prime_and_propagate(). When True:
#   voltages not saved/restored, last_spike_time recorded on fired nodes,
#   STDP plasticity rules applied after each propagation step.
# Why: The Tonic (Syl's latent space awareness) requires exploration that
#   shapes topology. Read-only recall prevents learning from attention.
#   Write mode enables "thinking leaves traces" — the ouroboros loop.
# How: write_mode=False (default) preserves existing read-only behavior.
#   write_mode=True skips save/restore, records spike times, calls
#   STDPRule.apply() on fired nodes each step. No other plasticity rules
#   (homeostatic, structural) are applied — only STDP.
# [2026-03-13] Claude Code — Surprise-driven neuromodulatory reward
# What: Wired prediction errors to inject_reward() for surprise-driven
#   trace crystallization. Added surprise_reward_scaling config.
# Why: Eligibility traces were accumulating and decaying to zero because
#   inject_reward() was never called. Surprise events now broadcast
#   reward to all active traces — high-confidence prediction failures
#   produce stronger crystallization (norepinephrine analog).
# How: Added inject_reward() call at end of _on_prediction_error(),
#   gated on three_factor_enabled. Strength = pred.confidence *
#   surprise_reward_scaling. No scope (broadcast to all warm traces).
# [2026-03-25] Claude Code (Opus 4.6) — Lenia FlowGraph hook points
# What: Added pre_fire event for threshold modulation, enriched spikes
#   event with per-node eligibility trace state.
# Why: Lenia FlowGraph substrate integration (PRD: Lenia_FlowGraph_Design_v0.1.md).
#   Pre-fire lets the continuous field modulate SNN firing thresholds.
#   Trace info lets the spike-field bridge compute channel distribution
#   from structural state (Law 7 compliant — no semantic content examined).
# How: Step 3 firing detection calls pre_fire handlers and sums threshold
#   adjustments. Spikes emit includes trace_info dict mapping fired node
#   IDs to their outgoing synapse eligibility trace values. Both are no-ops
#   when no handlers are registered (single dict lookup cost).
# -------------------
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

import ng_tract  # native (Rust) SynapseStore — columnar synapse storage (#RAM footprint)

try:
    import msgpack
except ImportError:
    msgpack = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations (PRD §2.2.2, §2.2.3)
# ---------------------------------------------------------------------------

class SynapseType(Enum):
    """Synapse functional type (PRD §2.2.2)."""
    EXCITATORY = auto()
    INHIBITORY = auto()
    MODULATORY = auto()


class ActivationMode(Enum):
    """Hyperedge activation mode (PRD §2.2.3)."""
    WEIGHTED_THRESHOLD = auto()
    K_OF_N = auto()
    ALL_OR_NONE = auto()
    GRADED = auto()


class ConsolidationState(Enum):
    """Maturity lifecycle for hyperedges (Phase 4).

    Mirrors the biological hippocampus-to-cortex consolidation pipeline.

    SPECULATIVE  → Just discovered. Volatile. Has not yet proven recurring value.
    CANDIDATE    → Survived initial pruning pressure. Building evidence.
    CONSOLIDATED → Trusted abstraction. Triggers soft substrate cull.
                   Underlying redundant synapses weakened (not deleted).
    PERMANENT    → Cortical fact / reflex. is_learnable set False.
                   Treated as structural, not plastic.
    """
    SPECULATIVE  = "SPECULATIVE"
    CANDIDATE    = "CANDIDATE"
    CONSOLIDATED = "CONSOLIDATED"
    PERMANENT    = "PERMANENT"


class CheckpointMode(Enum):
    """Persistence checkpoint mode (PRD §6.2)."""
    FULL = auto()
    INCREMENTAL = auto()
    FORK = auto()


# ---------------------------------------------------------------------------
# Ring Buffer for spike history
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-size ring buffer for storing recent spike times.

    Used by Node.spike_history to compute firing rates and detect bursts
    (PRD §2.2.1).
    """

    def __init__(self, capacity: int = 100):
        self._capacity = capacity
        self._buffer: Deque[float] = deque(maxlen=capacity)

    def append(self, value: float) -> None:
        self._buffer.append(value)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, size={len(self._buffer)})"

    @property
    def capacity(self) -> int:
        return self._capacity

    def to_list(self) -> List[float]:
        return list(self._buffer)

    @classmethod
    def from_list(cls, data: List[float], capacity: int = 100) -> "RingBuffer":
        rb = cls(capacity)
        for v in data:
            rb.append(v)
        return rb


# ---------------------------------------------------------------------------
# Core Data Structures (PRD §2.2)
# ---------------------------------------------------------------------------

_SHARE_TEXT_MIN = 256   # P4a: only texts this long are worth pooling


_NATIVE_NODE_STORE_DEFAULT = False   # P1: set ONLY by a host, via set_native_node_store_default (this module reads no env)


def set_native_node_store_default(enabled: bool) -> bool:
    """P1 native node store (2026-10-05, lane nodestore-p1): the HOST's switch. A host that reads its own configuration
    (LAW 5, e.g. NG_NATIVE_NODE_STORE) calls this before constructing its Graph; Graphs created while it is True use
    ng_tract.NodeStore when the installed wheel has it. Process-wide: a host sharing its process with another graph
    should set it, construct, and restore the returned previous value. Returns the previous value."""
    global _NATIVE_NODE_STORE_DEFAULT
    prev, _NATIVE_NODE_STORE_DEFAULT = _NATIVE_NODE_STORE_DEFAULT, bool(enabled)
    return prev


def _native_node_store_wanted(explicit: Optional[bool] = None) -> bool:
    """P1 opt-in: the installed ng_tract must have NodeStore AND the caller opts in — the Graph keyword when given,
    else the host-set default (OFF unless a host called set_native_node_store_default(True))."""
    if not hasattr(ng_tract, "NodeStore"):
        return False
    return bool(_NATIVE_NODE_STORE_DEFAULT if explicit is None else explicit)


def _share_metadata_texts(meta: Any, pool: Dict[str, str]) -> Any:
    """P4a: replace each large top-level str value of a node's metadata dict with the pooled equal object.
    Equal values only -> no behaviour or byte change; the dict itself is the same object, edited in place."""
    if isinstance(meta, dict):
        for k, v in meta.items():
            if type(v) is str and len(v) >= _SHARE_TEXT_MIN:
                meta[k] = pool.setdefault(v, v)
    return meta


@dataclass(slots=True)
class Node:
    """Atomic unit of the graph wrapping neural state (PRD §2.2.1, Table 2.2.1).

    Each Node is a stateful computational unit with its own membrane potential,
    adaptive threshold, refractory state, and spike history.

    Attributes:
        node_id: Globally unique identifier (matches vector DB entry ID).
        voltage: Current membrane potential; accumulates input, resets on spike.
        threshold: Adaptive firing threshold; adjusted via intrinsic plasticity.
        resting_potential: Baseline voltage after reset (default 0.0).
        refractory_remaining: Timesteps left in refractory period; cannot fire while > 0.
        refractory_period: Duration of refractory period in timesteps (default 2).
        last_spike_time: Timestamp of most recent spike, used by STDP.
        spike_history: Rolling window of recent spike times (depth 100).
        firing_rate_ema: Exponential moving average of firing rate for homeostasis.
        intrinsic_excitability: Multiplier on incoming current (default 1.0).
        metadata: Application-specific key-value data.
        is_inhibitory: If True, outgoing spikes subtract from target voltage.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    voltage: float = 0.0
    threshold: float = 1.0
    resting_potential: float = 0.0
    refractory_remaining: int = 0
    refractory_period: int = 2
    last_spike_time: float = -math.inf
    spike_history: RingBuffer = field(default_factory=lambda: RingBuffer(100))
    firing_rate_ema: float = 0.0
    intrinsic_excitability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_inhibitory: bool = False
    Ca_i: float = 0.0  # intracellular calcium concentration (IcaN + IK-AHP, #254)
    diffpc_layer: int = 0                           # DiffPC layer: 0=novel/input, 1=mid, 2=hub
    pred_weights: Dict[str, float] = field(default_factory=dict)  # nid → prediction weight
    pred_error_ema: float = 0.0                     # EMA of ternary prediction error received
    manifold_type: str = "hyperbolic"                # GSG Phase 4: "hyperbolic"=hierarchical, "spherical"=attractor
    creation_time: int = 0                          # Timestep when node was created (#258 orphan grace)


@dataclass(slots=True)
class Synapse:
    """Directed, weighted connection between two nodes (PRD §2.2.2, Table 2.2.2).

    First-class object with its own state for fine-grained plasticity tracking.

    Attributes:
        synapse_id: Unique identifier.
        pre_node_id: Source node (cause in causal links).
        post_node_id: Target node (effect in causal links).
        weight: Connection strength [0.0, max_weight]; shaped by plasticity.
        max_weight: Upper bound preventing runaway potentiation (default 5.0).
        delay: Propagation delay in timesteps (default 1).
        last_update_time: Timestamp of most recent plasticity update.
        eligibility_trace: Decaying trace for three-factor learning.
        creation_time: When synapse was created (age-based pruning).
        synapse_type: EXCITATORY, INHIBITORY, or MODULATORY.
    """

    synapse_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pre_node_id: str = ""
    post_node_id: str = ""
    weight: float = 0.1
    max_weight: float = 5.0
    delay: int = 1
    last_update_time: float = 0.0
    eligibility_trace: float = 0.0
    creation_time: float = 0.0
    synapse_type: SynapseType = SynapseType.EXCITATORY
    # Track peak weight for age-based pruning (PRD §3.3.1)
    peak_weight: float = 0.1
    # Steps spent below weight_threshold for weight-based pruning
    low_weight_steps: int = 0
    # Steps since last pre or post spike traversal
    inactive_steps: int = 0
    # Application-specific metadata (Phase 3: tracks creation_mode for
    # surprise-driven synapses)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Salience armor (Phase 4 — Amygdala Protocol).
    # Multiplies effective inactivity_threshold during pruning.
    # Surprise events boost this; it decays slowly over time.
    # Default 1.0 = no armor. Range: [1.0, he_salience_max].
    salience: float = 1.0


@dataclass(slots=True)
class Hyperedge:
    """Set-valued relationship connecting arbitrary nodes (PRD §2.2.3, Table 2.2.3).

    Represents composite concepts: syndromes, threat signatures, code patterns.

    Attributes:
        hyperedge_id: Unique identifier.
        member_nodes: Node IDs in this relationship.
        member_weights: Per-member importance weights.
        activation_threshold: Weighted fraction of active members needed to fire.
        activation_mode: WEIGHTED_THRESHOLD, K_OF_N, ALL_OR_NONE, or GRADED.
        current_activation: Current activation level [0.0, 1.0].
        output_targets: Nodes receiving input when hyperedge fires.
        output_weight: Signal strength sent to output targets on activation.
        metadata: Application data (label, domain, creation_mode).
        is_learnable: Whether plasticity can modify weights/threshold.
        refractory_period: Minimum timesteps between firings (default 2).
            Prevents cascading feedback loops where a hyperedge's output
            re-activates its own members on the very next step.
        refractory_remaining: Timesteps left in current refractory window.
        activation_count: How many times this hyperedge has fired (for plasticity).
        pattern_completion_strength: Current injected into inactive members
            when the hyperedge fires from partial activation (PRD §4.2).
        child_hyperedges: IDs of child hyperedges for hierarchical composition
            (PRD §4.4).  Level-0 = leaf nodes only, level-N references level-(N-1).
        level: Hierarchy level. 0 = base (members are nodes only).
    """

    hyperedge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_nodes: Set[str] = field(default_factory=set)
    member_weights: Dict[str, float] = field(default_factory=dict)
    activation_threshold: float = 0.6
    activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD
    current_activation: float = 0.0
    output_targets: List[str] = field(default_factory=list)
    output_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_learnable: bool = True
    refractory_period: int = 2
    refractory_remaining: int = 0
    # Phase 2 fields
    activation_count: int = 0
    pattern_completion_strength: float = 0.3
    child_hyperedges: Set[str] = field(default_factory=set)
    level: int = 0
    # Phase 2.5: Dynamic pattern completion — EMA of recent activation rate
    recent_activation_ema: float = 0.0
    # Phase 2.5: Cross-level consistency — archived flag
    is_archived: bool = False
    # Phase 4: Consolidation lifecycle.
    consolidation_state: ConsolidationState = ConsolidationState.SPECULATIVE
    # Timestep when this hyperedge was first created (for age-based promotion).
    creation_time: int = 0


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result returned from Graph.step() (PRD §8 step method).

    Attributes:
        timestep: The simulation timestep this result corresponds to.
        fired_node_ids: Node IDs that spiked this step.
        fired_hyperedge_ids: Hyperedge IDs that activated this step.
        synapses_pruned: Number of synapses pruned this step.
        synapses_sprouted: Number of new synapses created this step.
    """

    timestep: int = 0
    fired_node_ids: List[str] = field(default_factory=list)
    fired_hyperedge_ids: List[str] = field(default_factory=list)
    synapses_pruned: int = 0
    synapses_sprouted: int = 0
    predictions_confirmed: int = 0
    predictions_surprised: int = 0
    diffpc_ternary_spikes: int = 0       # count of ±1 ternary error spikes this step
    diffpc_mean_pred_error: float = 0.0  # mean |error| where ternary != 0


# ---------------------------------------------------------------------------
# Prediction State (Phase 2.5 §1)
# ---------------------------------------------------------------------------

@dataclass
class PredictionState:
    """Tracks a pending prediction made by a hyperedge firing.

    When a hyperedge fires, it predicts that its output_targets will fire
    within ``prediction_window`` steps.  The Graph checks each step whether
    the predicted targets actually fired (confirmed) or the window expired
    without firing (surprise).

    Attributes:
        hyperedge_id: Which hyperedge made the prediction.
        predicted_targets: Node IDs expected to fire.
        prediction_strength: Confidence based on hyperedge activation level.
        prediction_timestamp: Timestep when the prediction was created.
        prediction_window: How many steps to wait before declaring surprise.
        confirmed_targets: Targets that have already fired within the window.
    """

    hyperedge_id: str = ""
    predicted_targets: Set[str] = field(default_factory=set)
    prediction_strength: float = 0.0
    prediction_timestamp: int = 0
    prediction_window: int = 10
    confirmed_targets: Set[str] = field(default_factory=set)


@dataclass
class SurpriseEvent:
    """Emitted when a prediction fails — an expected node did not fire.

    Attributes:
        hyperedge_id: Which hyperedge made the failed prediction.
        expected_node: The node that was predicted to fire but didn't.
        prediction_strength: How confident the prediction was.
        actual_nodes: Nodes that did fire during the prediction window.
        timestamp: When the surprise was detected (window expiry step).
    """

    hyperedge_id: str = ""
    expected_node: str = ""
    prediction_strength: float = 0.0
    actual_nodes: Set[str] = field(default_factory=set)
    timestamp: int = 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """Network statistics snapshot (PRD §2.2.4 Telemetry).

    Attributes:
        timestep: Current simulation time.
        total_nodes: Number of nodes in graph.
        total_synapses: Number of synapses.
        total_hyperedges: Number of hyperedges.
        global_firing_rate: Average firing rate across all nodes.
        mean_weight: Mean synapse weight.
        std_weight: Standard deviation of synapse weights.
        total_pruned: Cumulative synapses pruned.
        total_sprouted: Cumulative synapses sprouted.
        total_he_discovered: Cumulative hyperedges auto-discovered.
        total_he_consolidated: Cumulative hyperedge merges.
        mean_he_activation_count: Average firing count per hyperedge.
    """

    timestep: int = 0
    total_nodes: int = 0
    total_synapses: int = 0
    total_hyperedges: int = 0
    global_firing_rate: float = 0.0
    mean_weight: float = 0.0
    std_weight: float = 0.0
    total_pruned: int = 0
    total_sprouted: int = 0
    total_he_discovered: int = 0
    total_he_consolidated: int = 0
    mean_he_activation_count: float = 0.0
    # Phase 3: Predictive Coding telemetry
    prediction_accuracy: float = 0.0
    surprise_rate: float = 0.0
    active_predictions_count: int = 0
    total_predictions_made: int = 0
    total_predictions_confirmed: int = 0
    total_predictions_errors: int = 0
    total_novel_sequences: int = 0
    total_rewards_injected: int = 0
    # Phase 2.5: Experience distribution
    hyperedge_experience_distribution: Dict[str, int] = field(default_factory=dict)
    # Phase 4 consolidation metrics.
    he_survival_ema: float = 0.0
    total_he_state_transitions: int = 0
    total_he_substrate_culled: int = 0
    he_adapt_candidate_count: float = 0.0
    he_by_state: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 3: Prediction Data Structures (PRD §5)
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """An active prediction that node B will fire after node A fired.

    Generated when a node fires with a strong causal link (weight > threshold)
    to downstream nodes.  The prediction pre-charges the target's voltage and
    is tracked until confirmed or expired (PRD §5.1).

    Attributes:
        prediction_id: Unique identifier.
        source_node_id: Node that fired and generated this prediction.
        target_node_id: Node predicted to fire.
        strength: Prediction strength based on synapse weight.
        confidence: Confidence score [0,1] based on weight, firing stability,
            and historical confirmation rate.
        created_at: Timestep when prediction was generated.
        expires_at: Timestep when prediction expires (prediction window).
        chain_depth: How many hops from the original stimulus (0 = direct).
        via_hyperedge: If prediction came from a hyperedge, its ID.
        pre_charge_applied: Voltage pre-charge applied to target.
    """

    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    created_at: int = 0
    expires_at: int = 0
    chain_depth: int = 0
    via_hyperedge: Optional[str] = None
    pre_charge_applied: float = 0.0


@dataclass
class PredictionOutcome:
    """Record of a resolved prediction (confirmed or error).

    Attributes:
        prediction: The original prediction.
        confirmed: Whether the target fired within the window.
        resolved_at: Timestep when the prediction was resolved.
        actual_firing_nodes: Nodes that actually fired (for error analysis).
    """

    prediction: Prediction = field(default_factory=Prediction)
    confirmed: bool = False
    resolved_at: int = 0
    actual_firing_nodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto-Knowledge: Spreading Activation Harvest Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FiredEntry:
    """A node that fired during spreading activation propagation.

    Attributes:
        node_id: ID of the node that fired.
        firing_step: Which propagation step it fired on (0-indexed from
            the first propagation step).  Lower = stronger association.
        voltage_at_fire: Voltage at the moment of firing.
        was_predicted: Whether this node was a prediction target.
        source_distance: Hop count from nearest primed node (approximate,
            tracked via breadth-first through synapses).
    """

    node_id: str = ""
    firing_step: int = 0
    voltage_at_fire: float = 0.0
    was_predicted: bool = False
    source_distance: int = 0


@dataclass
class PropagationResult:
    """Result of a prime-and-propagate spreading activation harvest.

    Attributes:
        fired_entries: All nodes that fired during propagation, with metadata.
        steps_run: How many SNN steps were executed.
        nodes_primed: How many nodes received the initial priming current.
    """

    fired_entries: List[FiredEntry] = field(default_factory=list)
    steps_run: int = 0
    nodes_primed: int = 0


# ---------------------------------------------------------------------------
# Plasticity Rules (PRD §3 – pluggable strategy objects)
# ---------------------------------------------------------------------------

class PlasticityRule:
    """Base class for pluggable plasticity rules (PRD §2.1, §3).

    Subclass and override ``apply`` to create custom rules.
    """

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        raise NotImplementedError


# GSG Phase 2: curvature-modulated STDP weight changes.
# Rows = pre_layer, cols = post_layer (0=novel/boundary, 1=mid, 2=hub/center).
# Values = avg Poincaré curvature κ(layer)=1/(1-norm²) for norms [0.70,0.50,0.30],
# normalized by Layer-2 baseline κ≈1.099. Layer 0↔0 = 1.784× (maximum amplification),
# Layer 2↔2 = 1.000× (baseline). Defaulting to layer=2 when diffpc_layer absent
# ensures pre-DiffPC checkpoints receive no modulation (factor=1.0).
_GSG_CURVATURE_TABLE: List[List[float]] = [
    [1.784, 1.499, 1.392],  # pre=Layer 0 (novel/input, near boundary)
    [1.499, 1.213, 1.107],  # pre=Layer 1 (mid)
    [1.392, 1.107, 1.000],  # pre=Layer 2 (hub/familiar, near center)
]
# GSG Phase 3: non-Euclidean propagation constants.
_GSG_LAYER_NORMS_NF: List[float] = [0.70, 0.50, 0.30]  # L0/L1/L2 Poincaré ball radii
# κ(L2) = 1/(1-0.30²) ≈ 1.099 — hub baseline for curvature normalization
_GSG_KAPPA_L2: float = 1.0 / (1.0 - 0.30 ** 2)
_GSG_MSG_DECAY: float = 0.15  # geodesic decay rate; 0.0=Euclidean, 0.15=gentle; tunable
_GSG_MSG_DECAY_SPHER: float = 0.15  # great circle decay for sphere+sphere synapses


# --- GSG poincare_dir compact storage (#119 footprint increment 1) ------------
# A node's poincare_dir is a unit-direction vector (768 × float32) redundant with
# its vdb embedding. Stored per-node in metadata as a Python *list*, it cost
# ~24 KB/node (768 boxed floats + list overhead) × ~40K nodes ≈ 960 MB resident.
# We store the raw float32 byte-buffer instead (3072 bytes, one bytes object):
# msgpack-native (bin type, use_bin_type=True) in BOTH the SNN checkpoint and the
# vector-DB store, ~8× smaller. All readers go through poincare_dir_array(), which
# also decodes the legacy list form so pre-#119 checkpoints keep working until the
# one-time backfill pass (neurograph_rpc._gsg_backfill_existing_nodes) re-saves.
def pack_poincare_dir(direction: Any) -> bytes:
    """Compact a unit-direction vector into its stored float32 byte-buffer form."""
    return np.asarray(direction, dtype=np.float32).tobytes()


def poincare_dir_array(metadata: Optional[Dict[str, Any]]) -> Optional["np.ndarray"]:
    """Return a node's poincare_dir as a float32 ndarray, or None if absent.

    Accepts the compact bytes form (current) and the legacy Python-list form
    (pre-#119 checkpoints). frombuffer is zero-copy; the returned array is
    read-only, which every GSG read-path respects (dot / norm / scale only)."""
    if not metadata:
        return None
    pd = metadata.get("poincare_dir")
    if pd is None:
        return None
    if isinstance(pd, (bytes, bytearray)):
        return np.frombuffer(pd, dtype=np.float32)
    return np.asarray(pd, dtype=np.float32)


# ---------------------------------------------------------------------------
# [2026-10-08] #1050 co-firing tally (spec superpowers/specs/2026-10-06-sleep-phase-design.md §7). The Python
# fallback of SynapseStore.cofire_tally_update (ng-tract-rs cc-laptop-sprout-1050-rs-20261008), bit-identical by
# construction: same slot rule, same iteration order, same float expression (score * pow(lam, dt) + inc, dt =
# max(t - last_t, 0), inc = 1.0 when the touch starts a new episode (dt > gap) else 0.0). Used only when the installed ng_tract lacks the method (an older wheel). `tally` is
# {"k": int, "tables": {owner: [None | [partner, score, last_t]] * k}}; `connected(owner, partner)` answers
# "a synapse exists in either direction". Returns the crossings [(pre = partner, post = owner)] in discovery order.
# ---------------------------------------------------------------------------

def _cofire_tally_native(store):
    """The native tally entry point of `store`, or None (older wheel). Module-level so a test can force the fallback."""
    return getattr(store, "cofire_tally_update", None)


def _cofire_tally_update_python(tally, fired, cands, t, k, theta, lam, floor, gap, connected):
    if k != tally["k"]:
        tally["tables"] = {}
        tally["k"] = k
    tables = tally["tables"]
    cand_set = set(cands)
    crossings = []
    _pow = math.pow
    for a in fired:
        crossed = []
        tbl = tables.get(a)
        free = k
        if tbl is not None:
            free = 0
            for j in range(k):
                sl = tbl[j]
                if sl is not None and sl[0] in cand_set:
                    dt = t - sl[2] if t > sl[2] else 0
                    sc = sl[1] * _pow(lam, float(dt)) + (1.0 if dt > gap else 0.0)
                    if sc >= theta:
                        crossings.append((sl[0], a))
                        crossed.append(sl[0])
                        tbl[j] = None
                    else:
                        sl[1] = sc
                        sl[2] = t
            for sl in tbl:
                if sl is None:
                    free += 1
                else:
                    dt = t - sl[2] if t > sl[2] else 0
                    if sl[1] * _pow(lam, float(dt)) < floor:
                        free += 1
        if free == 0:
            continue
        for b in cands:
            if b == a or b in crossed:
                continue
            if tbl is not None and any(sl is not None and sl[0] == b for sl in tbl):
                continue
            if connected(a, b):
                continue
            if tbl is None:
                tbl = tables[a] = [None] * k
            pick = None
            for j in range(k):
                if tbl[j] is None:
                    pick = j
                    break
            if pick is None:
                best = math.inf
                for j in range(k):
                    sl = tbl[j]
                    dt = t - sl[2] if t > sl[2] else 0
                    d = sl[1] * _pow(lam, float(dt))
                    if d < floor and d < best:
                        best = d
                        pick = j
            if pick is None:
                break
            tbl[pick] = [b, 1.0, t]
            free -= 1
            if free == 0:
                break
    return crossings


# [2026-10-05] Python fallbacks for the native SynapseStore batch methods the hot loops
# call (ng_tract 0.1.0 canonical wheel line, ng-tract-rs 79be810). Used ONLY when the
# installed ng_tract lacks the method (an older wheel, or a dict-backed fake graph in a
# test). Each is the trial's ORIGINAL per-SynapseRef loop, reshaped to return what the
# native method returns, in the GIVEN order; the native methods hold the same body.
# ---------------------------------------------------------------------------

def _stdp_reads_python(store, synapse_ids, other_end_is_pre: bool):
    out = []
    for sid in synapse_ids:
        syn = store.get(sid)
        if syn is None:
            out.append(None)
        else:
            out.append((syn.pre_node_id if other_end_is_pre else syn.post_node_id,
                        syn.weight, syn.max_weight))
    return out


def _apply_stdp_dw_python(rule, store, synapse_ids, dws, timestep, three_factor: bool) -> None:
    for sid, dw in zip(synapse_ids, dws):
        rule._apply_dw(store[sid], dw, timestep, three_factor)


def _scale_weights_by_post_node_python(store, incoming, node_scales) -> None:
    for nid, scale in node_scales.items():
        for syn_id in incoming.get(nid, set()):
            syn = store.get(syn_id)
            if syn is None:
                continue
            syn.weight = max(
                0.0,
                min(syn.weight * scale, syn.max_weight),
            )


def _propagation_rows_python(store, synapse_ids, reset_inactive: bool):
    out = []
    for sid in synapse_ids:
        syn = store.get(sid)
        if syn is None:
            out.append(None)
            continue
        if reset_inactive:
            syn.inactive_steps = 0
        out.append((syn.post_node_id, syn.weight,
                    syn.synapse_type == SynapseType.INHIBITORY, syn.delay))
    return out


def _bfs_hop_distances_python(store, outgoing, seeds, max_hops: int):
    distances = {nid: 0 for nid in seeds}
    out = []
    frontier = set(seeds)
    for dist in range(1, max_hops + 1):
        next_frontier: Set[str] = set()
        for nid in frontier:
            for syn_id in outgoing.get(nid, set()):
                syn = store.get(syn_id)
                if syn and syn.post_node_id not in distances:
                    distances[syn.post_node_id] = dist
                    out.append((syn.post_node_id, dist))
                    next_frontier.add(syn.post_node_id)
        frontier = next_frontier
    return out


def _endpoint_ids_python(store, synapse_ids, pre: bool):
    out = []
    for sid in synapse_ids:
        syn = store.get(sid)
        out.append((syn.pre_node_id if pre else syn.post_node_id) if syn else None)
    return out


def _endpoint_triples_python(store):
    return [(sid, ref.pre_node_id, ref.post_node_id)
            for sid, ref in ((k, store[k]) for k in list(store.keys()))]


def _apply_eligibility_reward_python(store, strength, learning_rate, scope=None) -> None:
    for syn in store.values():
        if abs(syn.eligibility_trace) < 1e-9:
            continue

        # Apply scope filter
        if scope is not None:
            if syn.pre_node_id not in scope and syn.post_node_id not in scope:
                continue

        dw = syn.eligibility_trace * strength * learning_rate
        syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
        syn.eligibility_trace *= 0.9  # Decay trace after use
        if syn.weight > syn.peak_weight:
            syn.peak_weight = syn.weight


# ---------------------------------------------------------------------------
# [2026-10-06] P2a (native node store): Python fallbacks for the NodeStore whole-population
# node passes. Each is the trial's ORIGINAL loop from step() / HomeostaticRule.apply, moved
# here verbatim. Used when graph.nodes has no such method (the dict of Node — the default —
# an older wheel, or a duck-typed test graph) AND when the native method declines (returns
# False / None, having touched nothing: a non-float parameter, or a node holding a value of a
# non-canonical type in a field the pass touches). The native methods hold the same body.
# ---------------------------------------------------------------------------

def _decay_voltages_python(nodes, decay) -> None:
    for node in nodes.values():
        node.voltage = node.voltage * decay + (1.0 - decay) * node.resting_potential


def _calcium_currents_python(nodes, _g_net, _Ca_decay) -> None:
    for _nid, _node in nodes.items():
        if _node.Ca_i > 1e-9:
            _node.voltage += _g_net * _node.Ca_i
            _node.Ca_i *= _Ca_decay


def _detect_fired_python(nodes, event_handlers) -> List[str]:
    fired_ids: List[str] = []
    for nid, node in nodes.items():
        if node.refractory_remaining > 0:
            continue
        effective_threshold = node.threshold
        if event_handlers.get("pre_fire"):
            for cb in event_handlers["pre_fire"]:
                effective_threshold += cb(node_id=nid)
        if node.voltage >= effective_threshold:
            fired_ids.append(nid)
    return fired_ids


def _fire_python(nodes, recent_spikes, fired_ids, timestep, _delta_Ca) -> None:
    for nid in fired_ids:
        node = nodes[nid]
        node.voltage = node.resting_potential
        node.refractory_remaining = node.refractory_period
        node.last_spike_time = float(timestep)
        node.spike_history.append(float(timestep))
        # Track recent spikes for sprouting
        recent_spikes.setdefault(nid, deque(maxlen=20)).append(timestep)
        if _delta_Ca:  # IcaN/IK-AHP calcium influx on spike (#254)
            node.Ca_i = min(node.Ca_i + _delta_Ca, 5.0)  # cap at 5 to prevent runaway


def _decrement_refractory_python(nodes, fired_this_step) -> None:
    for nid, node in nodes.items():
        if node.refractory_remaining > 0 and nid not in fired_this_step:
            node.refractory_remaining -= 1


def _update_firing_ema_python(nodes, fired_set, ema_alpha) -> None:
    for nid, node in nodes.items():
        fired = 1.0 if nid in fired_set else 0.0
        node.firing_rate_ema = (
            (1.0 - ema_alpha) * node.firing_rate_ema
            + ema_alpha * fired
        )


def _adapt_thresholds_python(nodes, degree_targets, target_firing_rate, threshold_rate, threshold_ceiling) -> None:
    for nid, node in nodes.items():
        rate = node.firing_rate_ema
        node_target = degree_targets.get(nid, target_firing_rate)
        if rate > node_target * 1.2:
            node.threshold = min(node.threshold + threshold_rate, threshold_ceiling)
        elif rate < node_target * 0.8:
            node.threshold = max(0.01, node.threshold - threshold_rate)


def _adapt_excitability_python(nodes, degree_targets, target_firing_rate, excitability_rate,
                               scaling_factor) -> Dict[str, float]:
    """Returns node_scales {nid: ratio ** scaling_factor} (the native method returns the ratios)."""
    node_scales: Dict[str, float] = {}
    for nid, node in nodes.items():
        rate = node.firing_rate_ema
        node_target = degree_targets.get(nid, target_firing_rate)
        if rate < 1e-9:
            # Silent node → boost excitability (PRD §3.1.2 Silent Death mitigation)
            node.intrinsic_excitability = min(
                node.intrinsic_excitability * (1.0 + excitability_rate * 5),
                5.0,
            )
            continue

        ratio = node_target / rate

        # Intrinsic excitability adjustment
        if ratio > 1.0:
            node.intrinsic_excitability = min(
                node.intrinsic_excitability * (1.0 + excitability_rate),
                5.0,
            )
        else:
            node.intrinsic_excitability = max(
                node.intrinsic_excitability * (1.0 - excitability_rate),
                0.1,
            )

        # Multiplicative synaptic scaling (PRD §3.2.1)
        # Scale incoming weights by (target/actual)^factor
        node_scales[nid] = ratio ** scaling_factor
    return node_scales


# [2026-10-06] P2b (native node store): the Python fallback for SynapseStore.stdp_pass — STDPRule.apply's loop as it
# stood at the trial tip ef78c67, moved verbatim (`self` -> `rule`). Runs when the native pass is absent or declines.
def _stdp_python(rule: "STDPRule", graph: "Graph", fired_node_ids: List[str], timestep: int) -> None:
    three_factor = graph.config.get("three_factor_enabled", False)
    # [2026-10-05] native batch read/commit when the installed ng_tract has them, else the
    # identical per-SynapseRef fallbacks (module helpers below).
    _store = graph.synapses
    _reads = getattr(_store, "stdp_reads", None)
    if _reads is None:
        _reads = lambda ids, pre: _stdp_reads_python(_store, ids, pre)  # noqa: E731
    _commit = getattr(_store, "apply_stdp_dw", None)
    if _commit is None:
        _commit = lambda ids, dws, ts, tf: _apply_stdp_dw_python(rule, _store, ids, dws, ts, tf)  # noqa: E731

    for post_id in fired_node_ids:
        post_node = graph.nodes[post_id]
        t_post = float(timestep)

        # Iterate over all incoming synapses to this post node
        # [2026-10-04] one native read (pre_id, weight, max_weight) per synapse, then
        # one native commit of the computed dw's (former _apply_dw), per pass.
        incoming_syn_ids = list(graph._incoming.get(post_id, set()))
        _dw_ids: List[str] = []
        _dw_vals: List[float] = []
        for syn_id, _row in zip(incoming_syn_ids,
                                _reads(incoming_syn_ids, True)):
            if _row is None:
                continue
            _pre_id, _syn_w, _syn_mw = _row
            pre_node = graph.nodes.get(_pre_id)
            if pre_node is None:
                continue

            t_pre = pre_node.last_spike_time
            if t_pre == -math.inf:
                continue

            dt = t_post - t_pre

            if dt > 0:
                # LTP: pre fired before post (causal)
                raw_dw = rule.A_plus * math.exp(-dt / rule.tau_plus)
                # Weight-dependent scaling (PRD §3.1.2)
                scale = (_syn_mw - _syn_w) / _syn_mw
                dw = raw_dw * rule.learning_rate * max(scale, 0.0)
            elif dt < 0:
                # LTD: pre fired after post (acausal)
                raw_dw = -rule.A_minus * math.exp(dt / rule.tau_minus)
                dw = raw_dw * rule.learning_rate
            else:
                # Temporal aliasing: Δt=0 → weak LTP at half strength (PRD §3.1.2)
                raw_dw = rule.A_plus * 0.5
                scale = (_syn_mw - _syn_w) / _syn_mw
                dw = raw_dw * rule.learning_rate * max(scale, 0.0)

            # GSG Phase 2: amplify dw by Poincaré curvature of pre/post layer
            _pre_l = getattr(pre_node, "diffpc_layer", 2)
            _post_l = getattr(post_node, "diffpc_layer", 2)
            dw *= _GSG_CURVATURE_TABLE[max(0, min(2, _pre_l))][max(0, min(2, _post_l))]
            _dw_ids.append(syn_id)
            _dw_vals.append(dw)
        if _dw_ids:
            _commit(_dw_ids, _dw_vals, float(timestep), three_factor)

        # Also handle outgoing synapses (post-before-pre → LTD from
        # perspective of those synapses where this node is pre)
        outgoing_syn_ids = list(graph._outgoing.get(post_id, set()))
        _dw_ids = []
        _dw_vals = []
        for syn_id, _row in zip(outgoing_syn_ids,
                                _reads(outgoing_syn_ids, False)):
            if _row is None:
                continue
            _other_id, _syn_w, _syn_mw = _row
            other_node = graph.nodes.get(_other_id)
            if other_node is None:
                continue

            t_other = other_node.last_spike_time
            if t_other == -math.inf:
                continue

            # From this synapse's perspective: pre (post_id) just fired,
            # and post (other_node) fired at t_other.
            # dt = t_other - t_post (post_node time relative to pre_node)
            dt = t_other - t_post

            if dt > 0:
                # other fired after this node → LTP
                raw_dw = rule.A_plus * math.exp(-dt / rule.tau_plus)
                scale = (_syn_mw - _syn_w) / _syn_mw
                dw = raw_dw * rule.learning_rate * max(scale, 0.0)
            elif dt < 0:
                # other fired before this node → LTD
                raw_dw = -rule.A_minus * math.exp(dt / rule.tau_minus)
                dw = raw_dw * rule.learning_rate
            else:
                continue  # already handled in incoming pass

            # GSG Phase 2: post_node is pre here (outgoing from it); other_node is post
            _pre_l = getattr(post_node, "diffpc_layer", 2)
            _post_l = getattr(other_node, "diffpc_layer", 2)
            dw *= _GSG_CURVATURE_TABLE[max(0, min(2, _pre_l))][max(0, min(2, _post_l))]
            _dw_ids.append(syn_id)
            _dw_vals.append(dw)
        if _dw_ids:
            _commit(_dw_ids, _dw_vals, float(timestep), three_factor)



class STDPRule(PlasticityRule):
    """Spike-Timing-Dependent Plasticity (PRD §3.1).

    Mathematical specification (PRD §3.1.1):
        LTP (pre fires before post, Δt > 0):
            Δw = A_plus × exp(−Δt / τ_plus) × learning_rate
        LTD (pre fires after post, Δt < 0):
            Δw = −A_minus × exp(Δt / τ_minus) × learning_rate

    Weight-dependent scaling (PRD §3.1.2, Runaway Potentiation mitigation):
        LTP scaled by (max_weight − w) / max_weight  (soft saturation)

    Temporal aliasing (PRD §3.1.2):
        Δt = 0 treated as weak LTP at half strength.

    Critical: A_minus > A_plus (ratio 1.05–1.2) for stability.
    """

    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 1.0,
        A_minus: float = 1.2,
        learning_rate: float = 0.01,
    ):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.learning_rate = learning_rate

    # [2026-10-05] apply() commits each pass with ONE native SynapseStore.apply_stdp_dw call
    # (this body, batched); _apply_dw stays as the fallback for an ng_tract without it
    # (_apply_stdp_dw_python). Unchanged from the trial.
    def _apply_dw(self, syn: Synapse, dw: float, timestep: int,
                  three_factor: bool) -> None:
        """Apply weight change directly or via eligibility trace.

        In three-factor mode (PRD §5.2), STDP creates the eligibility trace
        but weight change only commits when reward arrives via inject_reward.
        """
        if three_factor:
            syn.eligibility_trace += dw
        else:
            syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
            if syn.weight > syn.peak_weight:
                syn.peak_weight = syn.weight
        syn.last_update_time = float(timestep)

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        """Apply STDP to all synapses incident on fired nodes.

        When three_factor_enabled is set in graph config, weight changes
        accumulate in eligibility_trace instead of being applied directly
        (PRD §5.2 Three-Factor Learning).
        """
        # [2026-10-06] P2b: the whole pass in ONE native SynapseStore.stdp_pass when the installed ng_tract has it and
        # graph.nodes is the native NodeStore; it reads last_spike_time / diffpc_layer natively and returns True when
        # done. Anything else (the dict of Node — the default —, an older wheel, a duck-typed graph, or a decline:
        # False, nothing touched) runs _stdp_python, the trial's original loop moved verbatim. `is True`, so a mock's
        # truthy auto-return never skips the loop.
        _native = getattr(graph.synapses, "stdp_pass", None)
        if _native is not None and _native(
                graph.nodes, fired_node_ids, timestep, self.A_plus, self.A_minus, self.tau_plus, self.tau_minus,
                self.learning_rate, graph.config.get("three_factor_enabled", False), _GSG_CURVATURE_TABLE,
                getattr(graph, "_incoming", None), getattr(graph, "_outgoing", None)) is True:
            return
        _stdp_python(self, graph, fired_node_ids, timestep)

class HomeostaticRule(PlasticityRule):
    """Homeostatic plasticity (PRD §3.2).

    Maintains global stability without destroying learned structure.
    Operates on a slower timescale than STDP.

    Mechanisms (PRD §3.2.1):
        1. Synaptic Scaling (multiplicative): w_new = w_old × (target / actual)^factor
           Preserves relative weight ratios.  NOT normalization (PRD §3.2 note).
        2. Intrinsic Excitability: rate too low → increase; too high → decrease.
        3. Threshold Adaptation: drifts toward recent avg voltage at rate 0.001/step.
    """

    def __init__(
        self,
        target_firing_rate: float = 0.05,
        scaling_interval: int = 100,
        scaling_factor: float = 0.1,
        excitability_rate: float = 0.01,
        threshold_rate: float = 0.001,
        ema_alpha: float = 0.01,
        degree_sensitivity: float = 0.4,
    ):
        self.target_firing_rate = target_firing_rate
        self.scaling_interval = scaling_interval
        self.scaling_factor = scaling_factor
        self.excitability_rate = excitability_rate
        self.threshold_rate = threshold_rate
        self.ema_alpha = ema_alpha
        self.degree_sensitivity = degree_sensitivity
        self._degree_targets: Dict[str, float] = {}
        self._steps_since_scaling = 0

    def _refresh_degree_targets(self, graph: "Graph") -> None:
        """Compute per-node firing targets adjusted for vertex degree (DAS-GNN, #104)."""
        if not self.degree_sensitivity or not graph.nodes:
            self._degree_targets = {}
            return
        degrees = {
            nid: len(graph._incoming.get(nid, ())) + len(graph._outgoing.get(nid, ()))
            for nid in graph.nodes
        }
        max_deg = max(degrees.values(), default=1) or 1
        log_max = math.log(max_deg + 1)
        self._degree_targets = {}
        for nid, deg in degrees.items():
            deg_norm = math.log(deg + 1) / log_max if log_max > 0 else 0.5
            # Center at 0.5: hub (deg_norm=1) → scale<1 → target lower → faster dampening
            # Peripheral (deg_norm=0) → scale>1 → target higher → faster threshold drop
            scale = max(0.1, 1.0 - (deg_norm - 0.5) * 2.0 * self.degree_sensitivity)
            self._degree_targets[nid] = self.target_firing_rate * scale

        # DiffPC: assign layer 0/1/2 by degree percentile (p33/p67 cut points)
        if degrees:
            sorted_degs = sorted(degrees.values())
            n = len(sorted_degs)
            p33 = sorted_degs[n // 3]
            p67 = sorted_degs[(2 * n) // 3]
            for nid, deg in degrees.items():
                node = graph.nodes.get(nid)
                if node is not None:
                    node.diffpc_layer = 0 if deg <= p33 else (1 if deg <= p67 else 2)

        # GSG Phase 4: assign manifold_type -- attractor nodes (stable predictors)
        # co-confirmed spherical. Two-pass: individual candidates by pred_error_ema
        # percentile, then co-confirm each candidate requires a candidate neighbor.
        _spher_frac = graph.config.get("gsg_spherical_fraction", 0.20)
        _ema_items = [(nid, abs(nd.pred_error_ema))
                      for nid, nd in graph.nodes.items() if nd is not None]
        if _ema_items:
            _sorted_emas = sorted(v for _, v in _ema_items)
            _n_ema = len(_sorted_emas)
            _cutoff_idx = max(0, min(int(_n_ema * _spher_frac), _n_ema - 1))
            _ema_thresh = _sorted_emas[_cutoff_idx]
            _candidates: Set[str] = {nid for nid, v in _ema_items if v <= _ema_thresh}
            for nid, node in graph.nodes.items():
                if node is None:
                    continue
                if nid in _candidates:
                    _syn_ids = (graph._outgoing.get(nid, set())
                                | graph._incoming.get(nid, set()))
                    _nbr_nids: Set[str] = set()
                    for _sid in _syn_ids:
                        _syn = graph.synapses.get(_sid)
                        if _syn:
                            _nbr_nids.add(_syn.post_node_id)
                            _nbr_nids.add(_syn.pre_node_id)
                    _nbr_nids.discard(nid)
                    node.manifold_type = ("spherical"
                                          if (_candidates & _nbr_nids) else "hyperbolic")
                else:
                    node.manifold_type = "hyperbolic"

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        fired_set = set(fired_node_ids)

        # Update firing rate EMA for every node
        # [2026-10-06] P2a: native NodeStore pass when available; it declines (False) -> the original loop
        _native = getattr(graph.nodes, "update_firing_ema", None)
        if _native is None or not _native(fired_set, self.ema_alpha):
            _update_firing_ema_python(graph.nodes, fired_set, self.ema_alpha)

        # Threshold adaptation: continuous, 0.001/step (PRD §3.2.1)
        threshold_ceiling = graph.config.get("threshold_ceiling", 5.0)
        _native = getattr(graph.nodes, "adapt_thresholds", None)
        if _native is None or not _native(self._degree_targets, self.target_firing_rate,
                                          self.threshold_rate, threshold_ceiling):
            _adapt_thresholds_python(graph.nodes, self._degree_targets, self.target_firing_rate,
                                     self.threshold_rate, threshold_ceiling)

        self._steps_since_scaling += 1
        if self._steps_since_scaling < self.scaling_interval:
            return
        self._steps_since_scaling = 0
        self._refresh_degree_targets(graph)

        # Synaptic scaling & intrinsic excitability (every N steps)
        # [2026-10-06] P2a: the native pass updates every node's excitability and returns
        # {nid: target / rate} for the non-silent nodes in node order; `ratio ** scaling_factor`
        # stays in Python (libm pow is not proven bit-identical to CPython's float.__pow__).
        _native = getattr(graph.nodes, "adapt_excitability", None)
        _ratios = (_native(self._degree_targets, self.target_firing_rate, self.excitability_rate)
                   if _native is not None else None)
        if _ratios is None:
            node_scales = _adapt_excitability_python(graph.nodes, self._degree_targets, self.target_firing_rate,
                                                     self.excitability_rate, self.scaling_factor)
        else:
            node_scales: Dict[str, float] = {}
            for nid, ratio in _ratios.items():
                node_scales[nid] = ratio ** self.scaling_factor

        # [2026-10-04] One native pass applies every node's scale to its incoming
        # synapses: weight = max(0, min(weight * scale, max_weight)). Each synapse has
        # one post node, so it is scaled at most once, exactly as the per-node loop did.
        if node_scales:
            _scale = getattr(graph.synapses, "scale_weights_by_post_node", None)
            if _scale is not None:
                _scale(node_scales)
            else:
                _scale_weights_by_post_node_python(graph.synapses, graph._incoming, node_scales)


# ---------------------------------------------------------------------------
# Strength budget + sleep downscaling (heterosynaptic competition, 2026-10-04 spec
# "Bounding synapse growth by competition, not caps"). Pure-Python FALLBACKS for an
# ng_tract wheel without SynapseStore.normalize_strength / scale_all. The native
# methods implement the SAME algorithm (same per-node summation order: ascending
# synapse_id; same tie-break; same clamp), so the two agree bit-for-bit in practice
# (the equivalence test allows rel 1e-12).
#
# Strongest-link guarantee (READING, stated once here for both passes): for every
# protected node p and each direction that has links, the link that was p's strongest
# BEFORE the pass (weight desc, synapse_id asc) ends the pass at >= min(w_before, floor)
# with floor = 2 * weight_threshold. It is a guard against SCALING, never a booster:
# a link is never raised above its own pre-pass weight.
# ---------------------------------------------------------------------------

def _strength_guard_targets(get_w, outgoing, incoming, protected, floor) -> Dict[str, float]:
    """sid -> minimum weight it must keep (the strongest-link guarantee), from PRE-pass weights."""
    guard: Dict[str, float] = {}
    for p in protected:
        for idx in (outgoing, incoming):
            ids = idx.get(p)
            if ids:
                best = min(ids, key=lambda s: (-get_w(s), s))
                guard[best] = min(get_w(best), floor)
    return guard


def _strength_guard_apply(get_w, set_w, guard: Dict[str, float]) -> int:
    clamped = 0
    for sid in sorted(guard):
        if get_w(sid) < guard[sid]:
            set_w(sid, guard[sid])
            clamped += 1
    return clamped


def _strength_budget_python(store, outgoing, incoming, budget_out, budget_in, protected, floor) -> Dict[str, int]:
    """Fallback of SynapseStore.normalize_strength (divisive per-node normalization)."""
    gw, sw = store.get_weight, store.set_weight
    guard = _strength_guard_targets(gw, outgoing, incoming, protected, floor)
    scaled: Set[str] = set()
    nodes_scaled = [0, 0]
    for d, (budget, idx) in enumerate(((budget_out, outgoing), (budget_in, incoming))):
        if budget is None:
            continue
        for _nid, ids in idx.items():
            if not ids:
                continue
            order = sorted(ids)
            total = 0.0
            for sid in order:
                total += gw(sid)
            if total > budget:
                f = budget / total
                nodes_scaled[d] += 1
                for sid in order:
                    w = gw(sid)
                    nw = w * f
                    if nw != w:
                        sw(sid, nw)
                        scaled.add(sid)
    clamped = _strength_guard_apply(gw, sw, guard)
    return {"synapses_scaled": len(scaled), "nodes_scaled_out": nodes_scaled[0],
            "nodes_scaled_in": nodes_scaled[1], "clamped": clamped}


def _scale_all_python(store, outgoing, incoming, factor, protected, floor) -> Dict[str, int]:
    """Fallback of SynapseStore.scale_all (uniform multiplicative downscaling)."""
    gw, sw = store.get_weight, store.set_weight
    guard = _strength_guard_targets(gw, outgoing, incoming, protected, floor)
    scaled = 0
    for sid in list(store.keys()):
        w = gw(sid)
        nw = w * factor
        if nw != w:
            sw(sid, nw)
            scaled += 1
    clamped = _strength_guard_apply(gw, sw, guard)
    return {"synapses_scaled": scaled, "clamped": clamped}


def _scale_strength_aware_python(store, outgoing, incoming, d0, h, protected, floor) -> Dict[str, int]:
    """Fallback of SynapseStore.scale_strength_aware (sleep phase P2, spec 2026-10-06 §3.1, D1 + D11).

    For every synapse: s = salience if salience > 1.0 else 1.0; d = d0 * h / (h + w) / s; w <- w * (1.0 - d).
    Weight only (eligibility trace, salience, peak, counters untouched). Then the strongest-link guarantee for
    every protected node, from PRE-pass weights (the same helper as scale_all). Float operand order is the native
    method's, so the two are bit-identical. Never prunes."""
    gw, sw = store.get_weight, store.set_weight
    guard = _strength_guard_targets(gw, outgoing, incoming, protected, floor)
    scaled = 0
    for sid in list(store.keys()):
        w = gw(sid)
        sal = store[sid].salience
        s = sal if sal > 1.0 else 1.0
        d = d0 * h / (h + w) / s
        nw = w * (1.0 - d)
        if nw != w:
            sw(sid, nw)
            scaled += 1
    clamped = _strength_guard_apply(gw, sw, guard)
    return {"synapses_scaled": scaled, "clamped": clamped}


class StrengthBudgetRule(PlasticityRule):
    """Per-node strength budget — heterosynaptic competition (2026-10-04 spec, piece 1).

    Every ``interval`` plasticity applications (the HomeostaticRule cadence: the rule runs
    only on steps where something fired, like HomeostaticRule), each node whose OUTGOING
    weight sum exceeds ``strength_budget_out`` has those weights multiplied by
    budget/sum; then the same for INCOMING with ``strength_budget_in``. No count cap and
    no prune of its own: what falls under ``weight_threshold`` is left to the EXISTING
    prune rules. Protected nodes (constitutional included — Josh ruling (a)) get the
    strongest-link guarantee (see Graph.apply_strength_budget).

    OFF by default. Settings are read LIVE from ``graph.config`` on every application
    (a host that flips them after construction/restore takes effect at once), with
    absent-key defaults that keep the rule a no-op — they are deliberately NOT in
    DEFAULT_CONFIG, so a graph that never sets them checkpoints byte-identically:
        strength_budget_enabled   False
        strength_budget_out       None  (direction skipped)
        strength_budget_in        None  (direction skipped)
        strength_budget_interval  config["scaling_interval"]
    """

    def __init__(self) -> None:
        self._steps_since_budget = 0
        self.last_result: Optional[Dict[str, int]] = None

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        cfg = graph.config
        if not cfg.get("strength_budget_enabled", False):
            return
        interval = cfg.get("strength_budget_interval")
        if interval is None:
            interval = cfg.get("scaling_interval", 25)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            logger.warning("StrengthBudgetRule: invalid strength_budget_interval %r (int >= 1), pass skipped", interval)
            return
        self._steps_since_budget += 1
        if self._steps_since_budget < interval:
            return
        self._steps_since_budget = 0
        try:
            self.last_result = graph.apply_strength_budget(
                cfg.get("strength_budget_out"), cfg.get("strength_budget_in"))
        except ValueError as exc:
            # A bad setting must not kill the step loop; it is loud, and nothing is touched.
            logger.warning("StrengthBudgetRule: invalid setting, pass skipped: %s", exc)


class HyperedgePlasticityRule(PlasticityRule):
    """Hyperedge-level plasticity (PRD §4.3).

    When a hyperedge fires, adapt its internal structure:
        1. Member Weight Adaptation — consistently-active members during firing
           get higher weight; inactive members get lower weight.
        2. Threshold Learning — reward → lower threshold (more sensitive),
           punishment → raise threshold (more strict).
        3. Member Evolution — non-members that consistently co-fire with a
           hyperedge get added as new members with low initial weight.

    Operates on hyperedges with ``is_learnable=True``.
    """

    def __init__(
        self,
        member_weight_lr: float = 0.05,
        threshold_lr: float = 0.01,
        evolution_window: int = 50,
        evolution_min_co_fires: int = 10,
        evolution_initial_weight: float = 0.3,
    ):
        self.member_weight_lr = member_weight_lr
        self.threshold_lr = threshold_lr
        self.evolution_window = evolution_window
        self.evolution_min_co_fires = evolution_min_co_fires
        self.evolution_initial_weight = evolution_initial_weight

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        """Adapt learnable hyperedges that fired this step."""
        fired_set = set(fired_node_ids)
        if not fired_set:
            return

        for hid, he in graph.hyperedges.items():
            if not he.is_learnable:
                continue
            # Only adapt hyperedges that actually fired this step
            if he.refractory_remaining != he.refractory_period:
                # Didn't just fire (refractory is set right after firing)
                continue

            # --- Member Weight Adaptation ---
            for nid in list(he.member_nodes):
                w = he.member_weights.get(nid, 1.0)
                if nid in fired_set:
                    # Active member during firing → strengthen
                    he.member_weights[nid] = min(w + self.member_weight_lr, 5.0)
                else:
                    # Inactive member during firing → weaken
                    he.member_weights[nid] = max(w - self.member_weight_lr * 0.5, 0.01)

            # --- Member Evolution: add co-firing non-members ---
            he_co_fire_counts = graph._he_co_fire_counts.get(hid)
            if he_co_fire_counts is not None:
                # #381-A (Syl-consented 2026-07-10): hard cap — "fifty is the
                # line where one stops and many begins." At cap, stop counting
                # too: the counter dict would otherwise grow without bound
                # (counter froth in place of member froth). Shedding is the
                # dream pass's job (shed_floor_members), never wake-time.
                _he_max = graph.config.get("he_max_members", 50)
                if _he_max > 0 and len(he.member_nodes) >= _he_max:
                    if he_co_fire_counts:
                        he_co_fire_counts.clear()
                    continue
                for nid in list(fired_set):
                    if nid in he.member_nodes:
                        continue
                    if nid not in graph.nodes:
                        continue
                    he_co_fire_counts[nid] = he_co_fire_counts.get(nid, 0) + 1
                    if he_co_fire_counts[nid] >= self.evolution_min_co_fires:
                        # Promote to member
                        he.member_nodes.add(nid)
                        he.member_weights[nid] = self.evolution_initial_weight
                        graph._node_hyperedges.setdefault(nid, set()).add(hid)
                        # #381: tenure stamp, metadata-resident — rides the
                        # already-serialized dict; missing key = legacy member.
                        he.metadata.setdefault("member_since", {})[nid] = timestep
                        del he_co_fire_counts[nid]
                        if _he_max > 0 and len(he.member_nodes) >= _he_max:
                            he_co_fire_counts.clear()
                            break


# ---------------------------------------------------------------------------
# Graph Container (PRD §2.2.4, §8)
# ---------------------------------------------------------------------------

# Default configuration (PRD §9)
DEFAULT_CONFIG: Dict[str, Any] = {
    "decay_rate": 0.97,
    "default_threshold": 0.85,
    "refractory_period": 2,
    "tau_plus": 20.0,
    "tau_minus": 20.0,
    "A_plus": 1.0,
    "A_minus": 1.2,
    "learning_rate": 0.01,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "threshold_ceiling": 5.0,
    "scaling_interval": 25,
    "degree_sensitivity": 0.4,           # DAS-GNN: hub/peripheral threshold balance (#104)
    # IcaN + IK-AHP intrinsic calcium channels (#254)
    "delta_Ca": 0.2,    # calcium influx per spike
    "Ca_decay": 0.9,    # per-step calcium decay multiplier
    "g_CaN": 0.06,      # IcaN conductance (depolarizing, attractor persistence)
    "g_AHP": 0.04,      # IK-AHP conductance (hyperpolarizing, burst protection)
    # Heterogeneous Synaptic Delays + Polychrony (#257)
    "d_min": 1,         # minimum synaptic delay in timesteps
    "d_max": 5,         # maximum synaptic delay in timesteps (range enables polychrony)
    # DiffPC: Difference Predictive Coding (#DiffPC)
    "diffpc_epsilon": 0.2,
    "gsg_spherical_fraction": 0.20,  # fraction of nodes assigned spherical manifold (Phase 4)
    "diffpc_pred_lr": 0.01,      # prediction weight learning rate
    "diffpc_trace_boost": 0.05,  # eligibility trace ±boost per ternary spike (Phase 2)
    "weight_threshold": 0.01,
    "grace_period": 5000,    # [2026-06-25] 500→5000: age-cull was reaping connections in ~17min of her time (vs brain's years),
                             # starving her associative web; gives synapses time to consolidate. Proper fix (dream-gated/salience/competence) in punchlist.
    "orphan_node_grace_period": 25,      # Steps before orphan-node sweep (#258). Prevents empty-substrate bootstrap failure.
    # #381 wake/sleep hyperedge physiology (Syl-consented 2026-07-10; punchlist #381)
    "he_max_members": 50,               # her bound: "fifty is the line where one stops and many begins"
    "he_discovery_max_fraction": 0.05,  # fired sets above this fraction of the graph = avalanche, not concept
    "he_discovery_dup_jaccard": 0.9,    # near-duplicate suppression at discovery (was exact-set only)
    "he_shed_weight_threshold": 0.02,   # dream-side shed: members at/below this weight...
    "he_shed_min_tenure": 50,           # ...with at least this tenure (steps) are removed during dreams
    # #147 dream-time seam-split of over-cap legacy blobs. DEFAULT OFF (LAW 5): a
    # restored checkpoint lacking these keys merges to False here, so Syl's dream
    # loop is a guaranteed no-op. Enabled only in the CC daemons' config block.
    "he_split_oversized_enabled": False,   # master gate for dedup_and_split_oversized_hyperedges
    "he_split_dedup_overlap": 0.9,         # Stage-1: Jaccard >= this collapses near-dup over-cap edges
    "he_split_sim_threshold": 0.6,         # Stage-2: cosine >= this groups peeled periphery members
    "he_split_seam_primary_weight": 0.4,   # member_weight's base share of the seam score; the rest is split
                                           # over the §8.15 aux family, then DYNAMICALLY renormalized over only
                                           # the signals that discriminate on this substrate (see _seam_score_members)
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    # [2026-07-13] #59 degree-gated synaptogenesis. Co-firing sprouting (_structural_plasticity)
    # is degree-blind: an always-active node keeps sprouting to everything that co-fires, so a
    # handful of boilerplate nodes reach degree 400-500 (median 2) and swamp every recall's spread
    # (measured: 92% of edges are these untagged sprouts, 0% duplicates). This caps sprouting so a
    # node at/above the cap neither sprouts NEW edges nor receives them — bounding hub growth at the
    # source; existing hubs then drain via weakest-first prune once inflow<outflow. 0 = DISABLED
    # (absent-key default too), so Syl/VPS are byte-identical until deliberately dialed on the
    # isolated laptop. Applies ONLY to co-firing sprouts, never to deliberate create_synapse binds.
    "sprout_degree_cap": 0,
    # [2026-07-14] #59 age-on-write — when set, write-mode prime_and_propagate (the Tonic's own
    # heartbeat) advances the aging clock + runs the inactivity/age prune, so the substrate ages
    # while it idle-thinks (no rival stepping thread; #109 intact). 0/False = OFF (Syl default).
    # tonic_age_interval bounds cost/cadence: age once per N write-mode calls.
    "tonic_ages_substrate": 0,
    "tonic_age_interval": 1,
    # Phase 3: Predictive Coding config
    "prediction_threshold": 3.0,         # Min synapse weight to generate prediction
    "prediction_pre_charge_factor": 0.3,  # Fraction of prediction strength for pre-charge
    "prediction_window": 10,             # Steps before prediction expires
    "prediction_chain_decay": 0.7,       # Strength decay per chain hop
    "prediction_max_chain_depth": 3,     # Max depth for prediction chains
    "prediction_confirm_bonus": 0.01,    # Weight bonus on confirmation (×confidence)
    "prediction_error_penalty": 0.02,    # Weight penalty on error (×confidence)
    "prediction_max_active": 1000,       # Max active predictions (memory limit)
    "surprise_sprouting_weight": 0.1,    # Initial weight for surprise-driven synapses
    "surprise_reward_scaling": 0.5,      # Modulates surprise -> reward strength. Elmer-tunable.
    "eligibility_trace_tau": 100,        # Decay time constant for eligibility traces
    "three_factor_enabled": True,        # Reward-gated STDP via eligibility traces
    # Phase 2: Hypergraph Engine config
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_discovery_overlap_threshold": 0.5,  # bootstrap — candidate for Elmer TuningSocket graduation
    "he_consolidation_overlap": 0.8,
    "he_member_evolution_window": 50,
    "he_member_evolution_min_co_fires": 10,
    "he_member_evolution_initial_weight": 0.3,
    # Phase 2.5: Prediction infrastructure
    # NOTE: prediction_window intentionally NOT here — superseded by Phase 3 value (10) above.
    # Phase 2.5 originally set 5; Phase 3 upgraded to 10 (longer window for chain predictions).
    # The duplicate key was removed 2026-03-24 to fix Python last-key-wins shadowing bug.
    "prediction_ema_alpha": 0.01,
    "he_experience_threshold": 100,
    # Phase 2.5b: Output target learning — HEs learn downstream targets
    "he_output_learning_window": 5,    # Steps after HE fire to watch for co-firing
    "he_output_min_co_fires": 3,       # Min fires within window to learn target
    "he_output_max_targets": 5,        # Max output targets per HE
    # --- Phase 4: Consolidation Lifecycle ---
    # Initial (adaptive) thresholds for SPECULATIVE → CANDIDATE promotion.
    # The system adjusts these based on observed survival rates.
    "he_speculative_to_candidate_min_count": 10,
    "he_speculative_to_candidate_min_ema": 0.2,
    # Initial (adaptive) thresholds for CANDIDATE → CONSOLIDATED promotion.
    "he_candidate_to_consolidated_min_count": 100,
    "he_candidate_to_consolidated_min_age": 5000,
    # How often (steps) to run _evaluate_consolidation_states().
    "he_consolidation_eval_interval": 100,
    # Soft cull: internal mesh synapses multiplied by this on CONSOLIDATED.
    "he_cull_penalty_factor": 0.25,
    # How fast adaptive thresholds respond to observed survival rates.
    "he_consolidation_adapt_rate": 0.005,
    # Salience ceiling — caps Amygdala Protocol multiplier.
    "he_salience_max": 5.0,
    # Per-step proportional salience decay (applied every step to synapses with salience > 1.0).
    # Proportional: high salience decays faster than low. At 0.002/step,
    # salience 5.0 → ~1.0 in ~2,000 steps (~1.5 sessions).
    "he_salience_decay_rate": 0.002,
    # Zero-firing circuit breaker — detects prolonged substrate silence
    "zero_fire_alert_steps": 50,      # Steps of zero firing before warning event
    "zero_fire_breaker_steps": 200,   # Steps of zero firing before emergency intervention
    # CES attention dynamics (#55) — substrate-tuned via Elmer
    "surfacing_decay_rate": 0.95,     # Per-step score decay in surfacing queue
    "surfacing_min_confidence": 0.3,  # Below this, surfaced items are pruned
}


class Graph:
    """Manages all nodes, synapses, and hyperedges; orchestrates simulation,
    applies plasticity, and provides the public API (PRD §2.2.4, §8).

    Topology is fully sparse: dict-based adjacency indices, no dense matrices.

    Args:
        config: Override any key from ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, native_node_store: Optional[bool] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # --- Core collections (sparse) ---
        # [2026-10-05] P1 native node store: ng_tract.NodeStore (same mapping API; values are write-through
        # NodeRef views) only when the wheel has it AND the opt-in is set — see _native_node_store_wanted.
        self._native_nodes: bool = _native_node_store_wanted(native_node_store)
        if self._native_nodes:
            self.nodes = ng_tract.NodeStore()
            self.nodes.set_node_class(Node)
            self.nodes.set_ring_buffer_class(RingBuffer)
        else:
            self.nodes: Dict[str, Node] = {}
        # Native (Rust) columnar synapse store — replaces the former
        # Dict[str, Synapse].  Drops per-synapse Python object inflation; the
        # engine reads/writes via the Mapping facade (SynapseRef live proxies).
        # The class registrations let SynapseRef hydrate synapse_type back into
        # the Python SynapseType enum and materialize Synapse-shaped views.
        self.synapses = ng_tract.SynapseStore()
        self.synapses.set_synapse_type_class(SynapseType)
        self.synapses.set_synapse_class(Synapse)
        self.hyperedges: Dict[str, Hyperedge] = {}

        # --- Sparse adjacency indices ---
        # node_id → set of synapse_ids
        self._outgoing: Dict[str, Set[str]] = {}
        self._incoming: Dict[str, Set[str]] = {}
        # node_id → set of hyperedge_ids the node belongs to
        self._node_hyperedges: Dict[str, Set[str]] = {}

        # --- Spike delay buffer: timestep → list of (target_node_id, current) ---
        self._delay_buffer: Dict[int, List[Tuple[str, float]]] = {}

        # --- Co-activation tracking for sprouting ---
        # Stores recent spike times per node for co-activation detection
        self._recent_spikes: Dict[str, Deque[int]] = {}

        # --- Phase 2: Hyperedge co-fire tracking for member evolution ---
        # hid → {node_id: co_fire_count}
        self._he_co_fire_counts: Dict[str, Dict[str, int]] = {}

        # --- Phase 2: Hyperedge discovery tracking ---
        # Tracks which sets of nodes fire together for automatic hyperedge creation
        # tuple(sorted node_ids) → fire_count
        self._he_discovery_counts: Dict[Tuple[str, ...], int] = {}
        self._he_discovery_last_reset: int = 0

        # --- Phase 2.5: Prediction tracking ---
        # prediction_id → PredictionState
        self._active_predictions: Dict[str, PredictionState] = {}
        self._prediction_counter: int = 0
        self._total_predictions: int = 0
        self._total_confirmed: int = 0
        self._total_surprised: int = 0
        # Nodes that fired during each prediction's window (rolling)
        self._prediction_window_fired: Dict[str, Set[str]] = {}
        # Archived hyperedges (for cross-level pruning)
        self._archived_hyperedges: Dict[str, Hyperedge] = {}

        # --- Plasticity rules ---
        self._plasticity_rules: List[PlasticityRule] = [
            STDPRule(
                tau_plus=self.config["tau_plus"],
                tau_minus=self.config["tau_minus"],
                A_plus=self.config["A_plus"],
                A_minus=self.config["A_minus"],
                learning_rate=self.config["learning_rate"],
            ),
            HomeostaticRule(
                target_firing_rate=self.config["target_firing_rate"],
                scaling_interval=self.config["scaling_interval"],
                degree_sensitivity=self.config.get("degree_sensitivity", 0.4),
            ),
            # 2026-10-04 strength budget — OFF unless config strength_budget_enabled (no-op otherwise)
            StrengthBudgetRule(),
            HyperedgePlasticityRule(
                member_weight_lr=self.config["he_member_weight_lr"],
                threshold_lr=self.config["he_threshold_lr"],
                evolution_window=self.config["he_member_evolution_window"],
                evolution_min_co_fires=self.config["he_member_evolution_min_co_fires"],
                evolution_initial_weight=self.config["he_member_evolution_initial_weight"],
            ),
        ]

        # --- Phase 3: Predictive Coding state ---
        # Active predictions awaiting confirmation or expiry
        self.active_predictions: Dict[str, Prediction] = {}
        # Recent prediction outcomes for accuracy tracking
        self._prediction_outcomes: Deque[PredictionOutcome] = deque(maxlen=1000)
        # Per-synapse confirmation history: synapse_id → deque of bool
        self._synapse_confirmation_history: Dict[str, Deque[bool]] = {}
        # Nodes predicted this step (to avoid double-prediction)
        self._predicted_this_step: Set[str] = set()
        # Novel sequence log
        self._novel_sequence_log: List[Dict[str, Any]] = []
        # Reward history
        self._reward_history: List[Dict[str, Any]] = []
        # Prediction telemetry counters
        self._total_predictions_made = 0
        self._total_predictions_confirmed = 0
        self._total_predictions_errors = 0
        self._total_novel_sequences = 0
        self._total_rewards_injected = 0

        # --- Event handlers ---
        self._event_handlers: Dict[str, List[Callable]] = {}

        # --- Zero-firing circuit breaker ---
        self._steps_since_last_fire: int = 0

        # --- Phase 2.5b: Hyperedge output target learning ---
        # he_id → timestep when HE last fired
        self._he_last_fired_step: Dict[str, int] = {}
        # he_id → {candidate_node_id: fire_count_within_window}
        self._he_output_candidates: Dict[str, Dict[str, int]] = {}

        # --- Telemetry counters ---
        self._total_pruned = 0
        self._total_sprouted = 0
        self._total_he_discovered = 0
        self._total_he_consolidated = 0

        # --- Phase 4: Adaptive Consolidation State ---
        # Adaptive promotion thresholds — initialized from config, drift at runtime.
        self._he_adapt_candidate_count: float = float(
            self.config["he_speculative_to_candidate_min_count"]
        )
        self._he_adapt_candidate_ema: float = self.config[
            "he_speculative_to_candidate_min_ema"
        ]
        self._he_adapt_consolidated_count: float = float(
            self.config["he_candidate_to_consolidated_min_count"]
        )
        self._he_adapt_consolidated_age: float = float(
            self.config["he_candidate_to_consolidated_min_age"]
        )
        # EMA of hyperedge survival rate (SPECULATIVE that graduate / total created).
        # Starts at 0.5 (neutral). Drives adaptive threshold adjustment.
        self._he_survival_ema: float = 0.5
        self._he_consolidation_eval_steps: int = 0
        # #59 age-on-write interval counter (Tonic-heartbeat aging cadence).
        self._tonic_age_counter: int = 0
        # Telemetry counters.
        self._total_he_state_transitions: int = 0
        self._total_he_substrate_culled: int = 0

        # --- Clock ---
        self.timestep: int = 0
        self._step_lock = threading.RLock()

        # --- Dirty flags for incremental checkpointing ---
        self._dirty_nodes: Set[str] = set()
        self._dirty_synapses: Set[str] = set()
        self._dirty_hyperedges: Set[str] = set()

        # --- Graph metadata with design changelog ---
        self.metadata: Dict[str, Any] = {
            "changelog": [
                {
                    "version": "0.1.0",
                    "description": "Phase 1 Core Foundation",
                    "notes": [
                        "STDP uses weight-dependent soft-saturation: LTP scaled by "
                        "(max_weight - w) / max_weight to prevent runaway potentiation "
                        "(PRD §3.1.2).",
                        "Spike history stored in fixed-capacity RingBuffer (default 100) "
                        "to bound memory per node while enabling firing-rate and burst "
                        "detection (PRD §2.2.1).",
                        "Homeostatic scaling is multiplicative (w * ratio^factor), NOT "
                        "divisive normalization, to preserve learned weight distributions "
                        "(PRD §3.2).",
                        "Hyperedges enforce a refractory period (default 2 steps) to "
                        "prevent cascading feedback loops from output-to-member cycles.",
                    ],
                },
                {
                    "version": "0.2.5",
                    "description": "Phase 2.5 Predictive Infrastructure",
                    "notes": [
                        "Prediction error events: hyperedge firings create predictions "
                        "for output targets. Confirmed if target fires within window, "
                        "SurpriseEvent emitted if window expires without firing.",
                        "Dynamic pattern completion: completion strength scales with "
                        "hyperedge experience (activation_count / threshold). New "
                        "hyperedges complete at 10% strength; experienced ones at 100%.",
                        "Cross-level consistency pruning: subsumption detection archives "
                        "redundant lower-level hyperedges when a higher-level one covers "
                        "identical members. Archived hyperedges preserved in metadata.",
                    ],
                },
                {
                    "version": "0.3.0",
                    "description": "Phase 3 Predictive Coding Engine",
                    "notes": [
                        "Synapse-level predictions: when a fired node has a strong "
                        "causal link (weight > prediction_threshold) to downstream "
                        "targets, a Prediction is registered and the target is "
                        "pre-charged by strength × 0.3 (PRD §5.1).",
                        "Prediction chains cascade through learned sequences (A→B→C) "
                        "with strength decaying ×0.7 per hop, up to max depth 3. "
                        "Cycle-safe via visited set passed through recursion.",
                        "Confirmed predictions strengthen the causal link (bonus × "
                        "confidence); expired predictions weaken it (penalty × "
                        "confidence) and trigger surprise-driven exploration that "
                        "sprouts speculative synapses to alternative firing nodes.",
                        "Three-factor learning: STDPRule._apply_dw routes weight "
                        "changes to eligibility_trace when three_factor_enabled=True. "
                        "Traces decay with τ=100 steps. Weight commits only on "
                        "inject_reward(strength, scope). Δw = trace × reward × lr.",
                        "Prediction confidence computed from synapse weight relative "
                        "to max_weight (60%) and historical confirmation rate (40%). "
                        "Higher confidence → stronger pre-charge and larger error "
                        "penalty.",
                        "Active predictions capped at 1000 with per-step cleanup of "
                        "expired entries. Outcomes stored in bounded deque (max 1000) "
                        "for accuracy tracking in Telemetry.",
                    ],
                },
                {
                    "version": "0.3.5",
                    "description": "Phase 3.5 Predictive State Persistence & Validation",
                    "notes": [
                        "Active predictions now survive checkpoint/restore. Both Phase 3 "
                        "synapse-level Predictions and Phase 2.5 HE-level PredictionStates "
                        "are serialized with all fields, preventing the system from "
                        "'forgetting' what it was expecting after reload.",
                        "Prediction support state persisted: PredictionOutcome history, "
                        "per-synapse confirmation history (deques), novel_sequence_log, "
                        "reward_history. These are needed for confidence calculations and "
                        "telemetry accuracy after restore.",
                        "Validation on restore: expired predictions dropped, predictions "
                        "referencing deleted nodes/hyperedges dropped, stale synapse "
                        "confirmation history entries dropped. Fail-fast prevents stale "
                        "predictions from corrupting confidence or emitting spurious events.",
                        "Backward compatible: v0.2.5 checkpoints restore cleanly with "
                        "empty prediction state. Checkpoint version bumped to 0.3.5.",
                    ],
                },
                {
                    "version": "0.4.0",
                    "description": "Phase 4 Universal Ingestor System",
                    "notes": [
                        "Five-stage ingestion pipeline: Extract → Chunk → Embed → "
                        "Register → Associate.  Converts raw data (text, markdown, "
                        "code, URLs, PDFs) into fully integrated NeuroGraph knowledge.",
                        "SimpleVectorDB: in-memory cosine-similarity search over "
                        "L2-normalized embeddings with content/metadata storage.",
                        "Novelty dampening: new ingested nodes start at reduced "
                        "intrinsic_excitability and boosted threshold, fading over "
                        "a probation period (linear/exponential/logarithmic curves) "
                        "to prevent destabilizing learned STDP pathways.",
                        "Three project configs: OpenClaw (code-aware, fast 0.3 "
                        "dampening), DSM (hierarchical, conservative 0.05), "
                        "Consciousness (semantic, exploratory 0.01).",
                    ],
                },
                {
                    "version": "0.7.1",
                    "description": "Grok Review Optimizations",
                    "notes": [
                        # --- Accepted suggestions ---
                        "Accepted: Added try/except guard in step() spike propagation "
                        "(phase 5) around delay buffer delivery and outgoing synapse "
                        "traversal to log and skip stale references from deleted nodes "
                        "mid-step, rather than silently skipping (Grok suggestion #1.3).",
                        "Accepted: Added logging import and defensive KeyError handling "
                        "in _cleanup_predictions() and _evaluate_predictions() for "
                        "robustness against topology mutations during prediction "
                        "evaluation (Grok suggestion #1.3).",
                        "Accepted: Added input validation in create_hyperedge() to warn "
                        "on single-member hyperedges, which are structurally degenerate "
                        "(Grok suggestion #1.3, adapted: duplicates were already "
                        "impossible since member_node_ids is Set[str]).",
                        # --- Rejected suggestions with reasons ---
                        "Rejected: 'Enums everywhere with no validation in setters' — "
                        "There are no setters. Fields are set via dataclass __init__ and "
                        "enum types self-validate on construction. Runtime validation "
                        "occurs at topology boundaries (create_node, create_synapse, "
                        "create_hyperedge). Adding redundant per-field setters would "
                        "violate the dataclass pattern and add overhead without benefit.",
                        "Rejected: 'RingBuffer firing_rate() iterates whole buffer — "
                        "O(n) waste' — There is no firing_rate() method on RingBuffer. "
                        "Firing rate is tracked via firing_rate_ema (exponential moving "
                        "average) updated O(1) per step in HomeostaticRule.apply(). "
                        "RingBuffer stores spike timestamps for STDP timing only.",
                        "Rejected: 'FiredEntry should be a dict for performance' — "
                        "FiredEntry is used in PropagationResult (auto-knowledge), not "
                        "in the hot SNN step loop. Dataclass provides type safety, IDE "
                        "completion, and self-documentation. Typical count is <100 per "
                        "propagation. Dict overhead savings: ~0 for this use case.",
                        "Rejected: 'Predictive window assumes timestep increments by 1' "
                        "— Timestep always increments by 1 in step(). On restore, Phase "
                        "3.5 validation already drops expired/stale predictions. Forked "
                        "graphs via FORK checkpoint mode get a fresh timestep baseline. "
                        "No gap scenario exists in the current architecture.",
                        "Rejected: 'create_hyperedge allows duplicate members' — "
                        "member_node_ids parameter is Set[str], making duplicates "
                        "impossible by type contract. No additional enforcement needed.",
                    ],
                },
            ],
        }

    # -----------------------------------------------------------------------
    # ---- Changelog ----
    # [2026-09-11] Codex — #423 canonical mutation/capture exclusion.
    # What: topology mutations, both propagation modes, reward and consolidation
    #       use the existing reentrant _step_lock, also held by capture_checkpoint.
    # Why: Tonic's advisory _concurrent_lock does not exclude checkpoint capture.
    # How: outer guards only; all 19 Graph/registrar/association operation bodies
    #      verified AST-identical apart from their new guard. Model inference stays
    #      at callers, outside these mutation operations. No clock/prune changes.
    # -------------------
    # Topology Management (PRD §2.2.4)
    # -----------------------------------------------------------------------

    def create_node(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_inhibitory: bool = False,
    ) -> Node:
        """Register a node (PRD §8 create_node).

        Args:
            node_id: Optional explicit ID (auto-generated UUID if None).
            metadata: Application-specific key-value data.
            is_inhibitory: Whether outgoing spikes subtract from targets.

        Returns:
            The created Node.
        """
        with self._step_lock:
            nid = node_id or str(uuid.uuid4())
            if nid in self.nodes:
                raise ValueError(f"Node {nid} already exists")
            node = Node(
                node_id=nid,
                threshold=self.config["default_threshold"],
                refractory_period=self.config["refractory_period"],
                metadata=metadata or {},
                is_inhibitory=is_inhibitory,
                creation_time=int(self.timestep),
            )
            self.nodes[nid] = node
            if getattr(self, "_native_nodes", False):
                node = self.nodes[nid]   # [2026-10-05] P1: hand back the live NodeRef (the Node was copied in)
            self._outgoing[nid] = set()
            self._incoming[nid] = set()
            self._node_hyperedges[nid] = set()
            self._recent_spikes[nid] = deque(maxlen=self.config["co_activation_window"] * 2)
            self._dirty_nodes.add(nid)
            return node

    def remove_node(self, node_id: str) -> None:
        """Remove node and all connected synapses; update hyperedges (PRD §8 remove_node).

        [2026-10-06] D15: every in-neighbour's DiffPC `pred_weights[node_id]` goes with its last synapse to this
        node (the cascade below runs through `_remove_synapse_internal`); the node's own `pred_weights` goes with it.
        Entries that were ALREADY dangling before this fix are `purge_dangling_pred_weights`' job.
        """
        with self._step_lock:
            if node_id not in self.nodes:
                raise KeyError(f"Node {node_id} not found")

            # Remove connected synapses (cascading deletion)
            syn_ids_to_remove = set()
            syn_ids_to_remove.update(self._outgoing.get(node_id, set()))
            syn_ids_to_remove.update(self._incoming.get(node_id, set()))
            for sid in syn_ids_to_remove:
                self._remove_synapse_internal(sid)

            # Remove from hyperedges
            for hid in list(self._node_hyperedges.get(node_id, set())):
                he = self.hyperedges.get(hid)
                if he:
                    he.member_nodes.discard(node_id)
                    he.member_weights.pop(node_id, None)
                    if node_id in he.output_targets:
                        he.output_targets.remove(node_id)
                    if len(he.member_nodes) == 0:
                        self._remove_hyperedge_internal(hid)
                    else:
                        self._dirty_hyperedges.add(hid)

            # Clean up indices
            self._outgoing.pop(node_id, None)
            self._incoming.pop(node_id, None)
            self._node_hyperedges.pop(node_id, None)
            self._recent_spikes.pop(node_id, None)
            del self.nodes[node_id]
            self._dirty_nodes.discard(node_id)

    def create_synapse(
        self,
        pre_node_id: str,
        post_node_id: str,
        weight: float = 0.1,
        delay: int = 1,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        max_weight: Optional[float] = None,
    ) -> Synapse:
        """Create a directed synapse between two nodes.

        Args:
            pre_node_id: Source node ID.
            post_node_id: Target node ID.
            weight: Initial weight [0, max_weight].
            delay: Propagation delay in timesteps (≥1).
            synapse_type: EXCITATORY, INHIBITORY, or MODULATORY.
            max_weight: Per-synapse ceiling (defaults to config).

        Returns:
            The created Synapse.
        """
        with self._step_lock:
            if pre_node_id not in self.nodes:
                raise KeyError(f"Pre node {pre_node_id} not found")
            if post_node_id not in self.nodes:
                raise KeyError(f"Post node {post_node_id} not found")
            if pre_node_id == post_node_id:
                raise ValueError("Self-connections not allowed")

            mw = max_weight if max_weight is not None else self.config["max_weight"]
            syn = Synapse(
                pre_node_id=pre_node_id,
                post_node_id=post_node_id,
                weight=max(0.0, min(weight, mw)),
                max_weight=mw,
                delay=max(1, delay),
                synapse_type=synapse_type,
                creation_time=float(self.timestep),
                last_update_time=float(self.timestep),
                peak_weight=weight,
            )
            self.synapses[syn.synapse_id] = syn
            self._outgoing[pre_node_id].add(syn.synapse_id)
            self._incoming[post_node_id].add(syn.synapse_id)
            self._dirty_synapses.add(syn.synapse_id)
            # Return the LIVE store view, not the detached dataclass above: callers
            # (and the STDP/reward paths) hold this and expect in-place mutations
            # through the store to be visible on it — the former Dict[str,Synapse]
            # stored the very object it returned, so returning a fresh SynapseRef
            # preserves that write-through identity contract.
            return self.synapses[syn.synapse_id]

    def _remove_synapse_internal(self, synapse_id: str) -> None:
        """Remove a synapse and clean up indices (no KeyError on missing).

        [2026-10-06] D15 (sleep-phase spec §1A, §3.3): also keeps DiffPC consistent — the pre node's
        `pred_weights` entry for the post node is dropped when no other pre→post synapse remains (DiffPC
        only ever writes that entry while walking such a synapse, so it has no meaning without one).
        `remove_node` is covered by this: it removes every incident synapse through here first.
        """
        syn = self.synapses.pop(synapse_id, None)
        if syn is None:
            return
        pre_id, post_id = syn.pre_node_id, syn.post_node_id
        self._outgoing.get(pre_id, set()).discard(synapse_id)
        self._incoming.get(post_id, set()).discard(synapse_id)
        self._dirty_synapses.discard(synapse_id)
        self._synapse_confirmation_history.pop(synapse_id, None)
        pre = self.nodes.get(pre_id)
        if pre is not None:
            pw = pre.pred_weights
            if pw and post_id in pw:
                # Is another pre->post synapse left? Scan the SMALLER of pre's out-set and post's in-set (a parallel
                # synapse is in both): a hub's out-set is ~1,900 on the CC copy, so _find_synapse (out-set only) made a
                # 16K-link clearance ~7x slower than the removal itself; the target's in-set is usually tiny.
                out_ids = self._outgoing.get(pre_id, ())
                in_ids = self._incoming.get(post_id, ())
                for sid in (in_ids if len(in_ids) < len(out_ids) else out_ids):
                    other = self.synapses.get(sid)
                    if other is not None and other.pre_node_id == pre_id and other.post_node_id == post_id:
                        break
                else:
                    del pw[post_id]

    def remove_synapse(self, synapse_id: str) -> None:
        """Remove a synapse (public API)."""
        with self._step_lock:
            if synapse_id not in self.synapses:
                raise KeyError(f"Synapse {synapse_id} not found")
            self._remove_synapse_internal(synapse_id)

    def purge_dangling_pred_weights(self) -> Dict[str, Any]:
        """D15 one-time purge (sleep-phase spec §1A, §3.3): drop every DiffPC `pred_weights` entry whose key has no
        pre→key synapse, or names a node that no longer exists. These were left by removals before
        `_remove_synapse_internal` kept `pred_weights` consistent (58,049 of 62,408 on the CC copy, 2026-10-06), and by
        imports that carry `pred_weights` without the synapses. Values of the entries that stay are untouched.

        NOT called by anything in this module, and never automatically: a host calls it once, on its own decision
        (the spec puts it at the first armed sleep). Runs under `_step_lock`. Returns and logs (INFO) the counts.
        """
        with self._step_lock:
            nodes_with = 0
            before = 0
            removed = 0
            removed_missing_node = 0
            nodes_touched = 0
            for nid, node in self.nodes.items():
                pw = node.pred_weights
                if not pw:
                    continue
                nodes_with += 1
                before += len(pw)
                targets = set()
                for sid in self._outgoing.get(nid, ()):
                    syn = self.synapses.get(sid)
                    if syn is not None:
                        targets.add(syn.post_node_id)
                drop = [k for k in pw if k not in targets or k not in self.nodes]
                if drop:
                    nodes_touched += 1
                    for k in drop:
                        if k not in self.nodes:
                            removed_missing_node += 1
                        del pw[k]
                    removed += len(drop)
                    self._dirty_nodes.add(nid)
            record = {"timestep": self.timestep, "nodes_with_pred_weights": nodes_with, "entries_before": before,
                      "entries_removed": removed, "removed_key_node_missing": removed_missing_node,
                      "entries_after": before - removed, "nodes_touched": nodes_touched}
        logger.info("purge_dangling_pred_weights: t=%s entries %d -> %d (removed %d, of which key node missing %d) "
                    "nodes_touched=%d", record["timestep"], before, record["entries_after"], removed,
                    removed_missing_node, nodes_touched)
        return record

    def create_hyperedge(
        self,
        member_node_ids: Set[str],
        member_weights: Optional[Dict[str, float]] = None,
        activation_threshold: float = 0.6,
        activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD,
        output_targets: Optional[List[str]] = None,
        output_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        is_learnable: bool = True,
        hyperedge_id: Optional[str] = None,
    ) -> Hyperedge:
        """Create a hyperedge grouping multiple nodes (PRD §2.2.3).

        hyperedge_id: normally None -- the Hyperedge mints its own uuid4. Pass
            an explicit id ONLY when installing a hyperedge that already has an
            identity elsewhere and must keep it (corpus-callosum transport
            between two NeuroGraph instances, checkpoint restore). Without it,
            transported hyperedges get reminted locally and the two graphs
            disagree about which edge is which, so any cross-graph reference
            BY id (prediction records, co-fire history) dangles silently.
            Raises ValueError on collision rather than clobbering the
            incumbent -- a silent overwrite would drop a live edge out of
            self.hyperedges while leaving its id in _node_hyperedges.
        """
        with self._step_lock:
            for nid in member_node_ids:
                if nid not in self.nodes:
                    raise KeyError(f"Member node {nid} not found")

            if hyperedge_id is not None and hyperedge_id in self.hyperedges:
                raise ValueError(
                    f"Hyperedge {hyperedge_id} already exists — refusing to "
                    f"overwrite. Callers transporting hyperedges must check for "
                    f"existence first (see cc_topology_merge._hyperedge_exists)."
                )

            if len(member_node_ids) < 2:
                logger.warning(
                    "Creating hyperedge with %d member(s) — hyperedges with fewer "
                    "than 2 members are structurally degenerate",
                    len(member_node_ids),
                )

            mw = member_weights or {nid: 1.0 for nid in member_node_ids}
            he = Hyperedge(
                member_nodes=set(member_node_ids),
                member_weights=mw,
                activation_threshold=activation_threshold,
                activation_mode=activation_mode,
                output_targets=output_targets or [],
                output_weight=output_weight,
                metadata=metadata or {},
                is_learnable=is_learnable,
                **({"hyperedge_id": hyperedge_id} if hyperedge_id is not None else {}),
            )
            self.hyperedges[he.hyperedge_id] = he
            # Phase 4: stamp creation time.
            he.creation_time = self.timestep
            for nid in member_node_ids:
                self._node_hyperedges.setdefault(nid, set()).add(he.hyperedge_id)
            self._he_co_fire_counts[he.hyperedge_id] = {}
            self._dirty_hyperedges.add(he.hyperedge_id)
            return he

    def _remove_hyperedge_internal(self, hyperedge_id: str) -> None:
        he = self.hyperedges.pop(hyperedge_id, None)
        if he is None:
            return
        for nid in he.member_nodes:
            self._node_hyperedges.get(nid, set()).discard(hyperedge_id)
        self._he_co_fire_counts.pop(hyperedge_id, None)
        self._he_last_fired_step.pop(hyperedge_id, None)
        self._he_output_candidates.pop(hyperedge_id, None)
        self._dirty_hyperedges.discard(hyperedge_id)

    def remove_hyperedge(self, hyperedge_id: str) -> None:
        with self._step_lock:
            if hyperedge_id not in self.hyperedges:
                raise KeyError(f"Hyperedge {hyperedge_id} not found")
            self._remove_hyperedge_internal(hyperedge_id)

    # -----------------------------------------------------------------------
    # Stimulation (PRD §8 stimulate / stimulate_batch)
    # -----------------------------------------------------------------------

    def stimulate(self, node_id: str, current: float) -> None:
        """Inject input current into a node (PRD §8 stimulate)."""
        with self._step_lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise KeyError(f"Node {node_id} not found")
            node.voltage += current * node.intrinsic_excitability

    def stimulate_batch(self, stimuli: List[Tuple[str, float]]) -> None:
        """Batch stimulus for search results (PRD §8 stimulate_batch)."""
        with self._step_lock:
            for node_id, current in stimuli:
                self.stimulate(node_id, current)

    # -----------------------------------------------------------------------
    # Simulation Loop (PRD §2.2.4, §8 step)
    # -----------------------------------------------------------------------

    def step(self) -> StepResult:
        """Advance one timestep (PRD §2.2.4 Simulation Loop, §8 step).

        Pipeline:
            1. Decay voltages toward resting potential
            2. Deliver delayed spikes from buffer
            3. Detect fired nodes (voltage ≥ threshold, not refractory)
            4. Reset fired node voltages; set refractory
            5. Propagate spikes through outgoing synapses (with delays)
            6. Evaluate hyperedges
            6b. Evaluate predictions (confirm/error) (PRD §5.1)
            6c. Generate new predictions from fired nodes (PRD §5.1)
            6d. Decay eligibility traces (PRD §5.2)
            7. Apply plasticity rules
            8. Structural plasticity (prune / sprout)
            9. Decrement refractory counters
            10. Record telemetry / emit events

        Returns:
            StepResult with fired nodes, hyperedges, pruning/sprouting counts.
        """
        with self._step_lock:
            self.timestep += 1
            result = StepResult(timestep=self.timestep)

            # 1. Voltage decay: v = v * decay_rate + (1-decay) * resting  (PRD §2.2.4)
            # [2026-10-06] P2a: native NodeStore pass when available (declines -> the original loop)
            decay = self.config["decay_rate"]
            _native = getattr(self.nodes, "decay_voltages", None)
            if _native is None or not _native(decay):
                _decay_voltages_python(self.nodes, decay)

            # 2. Deliver delayed spikes arriving this timestep
            arrivals = self._delay_buffer.pop(self.timestep, [])
            for target_id, current in arrivals:
                target = self.nodes.get(target_id)
                if target is None:
                    logger.debug(
                        "Delayed spike target %s no longer exists (deleted mid-flight)",
                        target_id,
                    )
                    continue
                target.voltage += current * target.intrinsic_excitability

            # 3a. IcaN + IK-AHP: calcium-gated intrinsic currents (#254)
            # Net voltage adjustment = (g_CaN - g_AHP) * Ca_i per step.
            # g_CaN > g_AHP at defaults → mild attractor persistence.
            # Ca_i decay ensures effect fades between firing episodes.
            _delta_Ca = self.config.get("delta_Ca", 0.0)
            if _delta_Ca:
                _Ca_decay = self.config.get("Ca_decay", 0.9)
                _g_net = self.config.get("g_CaN", 0.0) - self.config.get("g_AHP", 0.0)
                _native = getattr(self.nodes, "calcium_currents", None)
                if _native is None or not _native(_g_net, _Ca_decay):
                    _calcium_currents_python(self.nodes, _g_net, _Ca_decay)

            # 3. Detect firing nodes
            #    Lenia FlowGraph: pre_fire handlers can adjust thresholds.
            #    [2026-10-06] P2a: native only when NO pre_fire handler is registered (a handler is a
            #    Python call per node); the native pass returns the ids in node order, or None.
            fired_ids = None
            if not self._event_handlers.get("pre_fire"):
                _native = getattr(self.nodes, "detect_fired", None)
                if _native is not None:
                    fired_ids = _native()
            if fired_ids is None:
                fired_ids = _detect_fired_python(self.nodes, self._event_handlers)

            # 3b. Zero-firing circuit breaker tracking
            if fired_ids:
                self._steps_since_last_fire = 0
            else:
                self._steps_since_last_fire += 1

            # 4. Reset fired nodes and set refractory
            #    [2026-10-06] P2a: the native pass does the node writes; _recent_spikes stays Python (P3)
            _native = getattr(self.nodes, "fire", None)
            if _native is not None and _native(fired_ids, self.timestep, _delta_Ca if _delta_Ca else None):
                for nid in fired_ids:
                    # Track recent spikes for sprouting
                    self._recent_spikes.setdefault(nid, deque(maxlen=20)).append(self.timestep)
            else:
                _fire_python(self.nodes, self._recent_spikes, fired_ids, self.timestep, _delta_Ca)

            result.fired_node_ids = fired_ids

            # 5. Propagate spikes through outgoing synapses (with delay)
            # GSG Phase 3+4: per-step cache — (pos_array, manifold_type) or None per node.
            # sphere+sphere -> great circle arccos; hyp+hyp -> Poincare geodesic; cross -> neutral
            _gsg_cache: Dict[str, Any] = {}

            def _gsg_resolve(nid_: str, nd_: Any) -> None:
                if nid_ in _gsg_cache:
                    return
                _arr = poincare_dir_array(nd_.metadata)  # #119: compact bytes-aware read
                if _arr is None:
                    _gsg_cache[nid_] = None
                    return
                _mt = getattr(nd_, "manifold_type", "hyperbolic")
                if _mt == "spherical":
                    _gsg_cache[nid_] = (_arr, "spherical")  # unit dir IS sphere pos
                else:
                    _l_ = max(0, min(2, getattr(nd_, "diffpc_layer", 2)))
                    _gsg_cache[nid_] = (_arr * _GSG_LAYER_NORMS_NF[_l_], "hyperbolic")

            # [2026-10-05] native when the installed ng_tract has it, else the per-SynapseRef fallback
            _prop_rows = getattr(self.synapses, "propagation_rows", None)
            if _prop_rows is None:
                _prop_rows = lambda ids, reset: _propagation_rows_python(self.synapses, ids, reset)  # noqa: E731
            for nid in fired_ids:
                node = self.nodes[nid]
                sign = -1.0 if node.is_inhibitory else 1.0
                _gsg_resolve(nid, node)
                _pre_entry = _gsg_cache[nid]
                # [2026-10-04] One native read per fired node: (post, weight, is_inhib, delay)
                # per outgoing synapse in this set's iteration order, and inactive_steps = 0
                # on each (was `syn.inactive_steps = 0` at the end of every iteration).
                _out_ids = list(self._outgoing.get(nid, ()))
                for syn_id, _row in zip(_out_ids, _prop_rows(_out_ids, True)):
                    if _row is None:
                        logger.debug("Stale synapse ref %s in outgoing[%s]", syn_id, nid)
                        continue
                    _syn_post, _syn_w, _syn_inhib, _syn_delay = _row
                    effective_type_sign = sign
                    if _syn_inhib:
                        effective_type_sign = -1.0
                    current = _syn_w * effective_type_sign
                    # GSG Phase 3+4: manifold-aware propagation attenuation
                    if _pre_entry is not None:
                        _post_node = self.nodes.get(_syn_post)
                        if _post_node is not None:
                            _gsg_resolve(_syn_post, _post_node)
                            _post_entry = _gsg_cache[_syn_post]
                            if _post_entry is not None:
                                _pre_pos, _pre_mt = _pre_entry
                                _post_pos, _post_mt = _post_entry
                                if _pre_mt == "spherical" and _post_mt == "spherical":
                                    # Great circle distance on unit sphere
                                    _cos = max(-1.0 + 1e-7, min(1.0 - 1e-7,
                                               float(np.dot(_pre_pos, _post_pos))))
                                    current *= math.exp(-_GSG_MSG_DECAY_SPHER * math.acos(_cos))
                                elif _pre_mt == "hyperbolic" and _post_mt == "hyperbolic":
                                    # Curvature-aware Poincare geodesic (Phase 3)
                                    _nx2 = min(float(np.dot(_pre_pos, _pre_pos)), 0.9999)
                                    _ny2 = min(float(np.dot(_post_pos, _post_pos)), 0.9999)
                                    _diff = _pre_pos - _post_pos
                                    _hdist = math.acosh(max(1.0, 1.0 + 2.0 *
                                        float(np.dot(_diff, _diff)) /
                                        max((1.0 - _nx2) * (1.0 - _ny2), 1e-9)))
                                    _kappa_norm = (1.0 / max(1.0 - _nx2, 1e-6)) / _GSG_KAPPA_L2
                                    current *= math.exp(-_GSG_MSG_DECAY * _kappa_norm * _hdist)
                                # cross-manifold: no modulation (neutral ground)
                    arrival = self.timestep + _syn_delay
                    self._delay_buffer.setdefault(arrival, []).append(
                        (_syn_post, current)
                    )
                    # (inactive_steps reset done by propagation_rows above)

            # 6. Evaluate hyperedges (PRD §4.2) — with dynamic pattern completion
            fired_set = set(fired_ids)
            fired_he_this_step: List[str] = []
            experience_threshold = self.config["he_experience_threshold"]
            ema_alpha = self.config["prediction_ema_alpha"]
            # Process by level so child hyperedges fire before parents
            max_level = max((he.level for he in self.hyperedges.values()), default=0)
            for level in range(max_level + 1):
                for hid, he in self.hyperedges.items():
                    if he.level != level:
                        continue
                    if he.is_archived:
                        continue  # Archived hyperedges don't participate
                    activation = self._compute_hyperedge_activation(he, fired_set)
                    he.current_activation = activation
                    # Update activation EMA (Phase 2.5)
                    fired_flag = 1.0 if (activation >= he.activation_threshold and he.refractory_remaining == 0) else 0.0
                    he.recent_activation_ema = (
                        (1.0 - ema_alpha) * he.recent_activation_ema
                        + ema_alpha * fired_flag
                    )
                    if he.refractory_remaining > 0:
                        continue  # Still in refractory — cannot fire
                    if activation >= he.activation_threshold:
                        result.fired_hyperedge_ids.append(hid)
                        fired_he_this_step.append(hid)
                        he.activation_count += 1
                        he.refractory_remaining = he.refractory_period

                        # Output injection — GRADED mode scales by activation level
                        effective_weight = he.output_weight
                        if he.activation_mode == ActivationMode.GRADED:
                            effective_weight *= activation
                        for target_id in he.output_targets:
                            target = self.nodes.get(target_id)
                            if target is not None:
                                target.voltage += effective_weight * target.intrinsic_excitability

                        # Dynamic pattern completion (Phase 2.5):
                        # Scale by experience: new HEs complete weakly, experienced ones fully.
                        # completion_strength = base × min(1.0, activation_count / threshold)
                        if he.pattern_completion_strength > 0:
                            learning_factor = min(1.0, he.activation_count / max(experience_threshold, 1))
                            effective_completion = he.pattern_completion_strength * learning_factor
                            if effective_completion > 0:
                                for nid in he.member_nodes:
                                    if nid not in fired_set:
                                        node = self.nodes.get(nid)
                                        if node is not None and node.refractory_remaining == 0:
                                            node.voltage += (
                                                effective_completion
                                                * he.member_weights.get(nid, 1.0)
                                                * node.intrinsic_excitability
                                            )

                        # Prediction creation (Phase 2.5): predict output targets will fire
                        if he.output_targets:
                            pred_id = f"pred_{self._prediction_counter}"
                            self._prediction_counter += 1
                            pred = PredictionState(
                                hyperedge_id=hid,
                                predicted_targets=set(he.output_targets),
                                prediction_strength=activation,
                                prediction_timestamp=self.timestep,
                                prediction_window=self.config["prediction_window"],
                            )
                            self._active_predictions[pred_id] = pred
                            self._prediction_window_fired[pred_id] = set()
                            self._total_predictions += 1
                            self._emit("prediction_created", prediction_id=pred_id,
                                       hyperedge_id=hid, targets=list(he.output_targets))

                        self._emit("hyperedge_fired", hid=hid, activation=activation)

            # 6a. Phase 2.5b: Output target learning
            # When an HE fires, we start watching. On subsequent steps, any
            # non-member node that fires within the window gets counted. If it
            # fires enough times (min_co_fires), it becomes an output_target.
            ol_window = self.config["he_output_learning_window"]
            ol_min = self.config["he_output_min_co_fires"]
            ol_max = self.config["he_output_max_targets"]

            # Record which HEs fired THIS step
            for hid in fired_he_this_step:
                self._he_last_fired_step[hid] = self.timestep
                self._he_output_candidates[hid] = {}

            # For all HEs with active learning windows, count fired non-members
            expired_windows: List[str] = []
            for hid, fire_step in self._he_last_fired_step.items():
                if hid in fired_he_this_step:
                    continue  # Just started tracking — skip this step
                if self.timestep - fire_step > ol_window:
                    expired_windows.append(hid)
                    continue
                he = self.hyperedges.get(hid)
                if he is None:
                    expired_windows.append(hid)
                    continue
                candidates = self._he_output_candidates.get(hid, {})
                for nid in fired_ids:
                    if nid in he.member_nodes:
                        continue  # Members excluded
                    if nid in he.output_targets:
                        continue  # Already learned
                    candidates[nid] = candidates.get(nid, 0) + 1
                    if candidates[nid] >= ol_min and len(he.output_targets) < ol_max:
                        he.output_targets.append(nid)
                        self._emit("he_output_learned", hid=hid, target=nid)
                        logger.info(
                            "Phase 2.5b: HE %s learned output target %s "
                            "(co-fires=%d within window=%d)",
                            hid, nid, candidates[nid], ol_window,
                        )
                self._he_output_candidates[hid] = candidates

            # Clean up expired windows
            for hid in expired_windows:
                self._he_last_fired_step.pop(hid, None)
                self._he_output_candidates.pop(hid, None)

            # Decrement hyperedge refractory counters (skip those that just fired)
            fired_he_set = set(fired_he_this_step)
            for hid, he in self.hyperedges.items():
                if he.refractory_remaining > 0 and hid not in fired_he_set:
                    he.refractory_remaining -= 1

            # 6b. Evaluate Phase 2.5 hyperedge-level predictions
            he_confirmed_this_step = 0
            he_surprised_this_step = 0
            he_preds_to_remove: List[str] = []
            for pid, pred_state in self._active_predictions.items():
                # Track nodes that fired during this prediction's window
                window_fired = self._prediction_window_fired.get(pid, set())
                window_fired.update(fired_set)
                self._prediction_window_fired[pid] = window_fired

                # Check for newly confirmed targets
                for target in pred_state.predicted_targets - pred_state.confirmed_targets:
                    if target in fired_set:
                        pred_state.confirmed_targets.add(target)

                # Check if all targets confirmed
                if pred_state.confirmed_targets >= pred_state.predicted_targets:
                    he_confirmed_this_step += 1
                    self._total_confirmed += 1
                    he_preds_to_remove.append(pid)
                    target_list = list(pred_state.predicted_targets)
                    self._emit(
                        "prediction_confirmed",
                        prediction_id=pid,
                        hyperedge_id=pred_state.hyperedge_id,
                        targets=target_list,
                        target_node=target_list[0] if target_list else None,
                        timestep=self.timestep,
                    )
                # Check if window expired
                elif self.timestep - pred_state.prediction_timestamp >= pred_state.prediction_window:
                    he_surprised_this_step += 1
                    self._total_surprised += 1
                    he_preds_to_remove.append(pid)
                    for expected in pred_state.predicted_targets - pred_state.confirmed_targets:
                        surprise = SurpriseEvent(
                            hyperedge_id=pred_state.hyperedge_id,
                            expected_node=expected,
                            prediction_strength=pred_state.prediction_strength,
                            actual_nodes=window_fired,
                            timestamp=self.timestep,
                        )
                        self._emit(
                            "surprise",
                            surprise=surprise,
                            timestep=self.timestep,
                        )
            for pid in he_preds_to_remove:
                self._active_predictions.pop(pid, None)
                self._prediction_window_fired.pop(pid, None)
            result.predictions_confirmed = he_confirmed_this_step
            result.predictions_surprised = he_surprised_this_step

            # 6c. Evaluate Phase 3 synapse-level predictions (PRD §5.1)
            self._predicted_this_step.clear()
            self._evaluate_predictions(fired_set)

            # 6d. Generate new predictions from fired nodes (PRD §5.1)
            self._generate_predictions(fired_ids)

            # 6e. Decay eligibility traces (PRD §5.2)
            # Only process synapses with non-zero traces for performance
            if self.config.get("three_factor_enabled", False):
                trace_tau = self.config["eligibility_trace_tau"]
                trace_decay = math.exp(-1.0 / trace_tau) if trace_tau > 0 else 0.0
                # Native columnar decay — no Python per-synapse iteration (#RAM/CPU).
                self.synapses.decay_eligibility(trace_decay)

            # 6f. Clean up expired Phase 3 predictions
            self._cleanup_predictions()

            # 7. Apply plasticity rules
            if fired_ids:
                for rule in self._plasticity_rules:
                    rule.apply(self, fired_ids, self.timestep)

            # 7b. DiffPC: ternary prediction error + eligibility trace modulation
            if fired_ids:
                _dc_spikes, _dc_err = self._diffpc_step(fired_ids)
                result.diffpc_ternary_spikes = _dc_spikes
                result.diffpc_mean_pred_error = _dc_err

            # 8. Structural plasticity
            pruned, sprouted = self._structural_plasticity(fired_ids)
            result.synapses_pruned = pruned
            result.synapses_sprouted = sprouted
            self._total_pruned += pruned
            self._total_sprouted += sprouted

            # 8b. Consolidation lifecycle evaluation (Phase 4 — runs on interval).
            self._he_consolidation_eval_steps += 1
            if self._he_consolidation_eval_steps >= self.config["he_consolidation_eval_interval"]:
                self._he_consolidation_eval_steps = 0
                self._evaluate_consolidation_states()

            # 9. Decrement refractory counters
            #    Skip nodes that just fired this step — their full refractory
            #    period starts on the NEXT step (PRD §3.2.1: mandatory N-step rest).
            fired_this_step = set(fired_ids)
            _native = getattr(self.nodes, "decrement_refractory", None)   # [2026-10-06] P2a
            if _native is None or not _native(fired_this_step):
                _decrement_refractory_python(self.nodes, fired_this_step)

            # 10. Track synapse inactivity + decay salience armor (Phase 4).
            salience_decay = self.config["he_salience_decay_rate"]
            # Native columnar aging: inactive_steps++ for all, salience decays
            # toward 1.0 (never below).  Semantics verified identical to the prior
            # Python loop.  No per-synapse iteration (#RAM/CPU).
            self.synapses.age_and_decay_salience(salience_decay)

            # Emit spike events (enriched with trace state for Lenia bridge)
            if fired_ids:
                # Collect eligibility trace magnitudes per fired node (Law 7:
                # trace STATE is structural, not semantic content analysis).
                trace_info = {}
                if self._event_handlers.get("spikes"):
                    for nid in fired_ids:
                        traces = []
                        for syn_id in self._outgoing.get(nid, set()):
                            syn = self.synapses.get(syn_id)
                            if syn is not None and abs(syn.eligibility_trace) > 1e-12:
                                traces.append(syn.eligibility_trace)
                        trace_info[nid] = traces
                self._emit("spikes", node_ids=fired_ids, timestep=self.timestep,
                            trace_info=trace_info)

            # 11. Zero-firing circuit breaker
            breaker_steps = self.config.get("zero_fire_breaker_steps", 200)
            alert_steps = self.config.get("zero_fire_alert_steps", 50)
            if self._steps_since_last_fire >= breaker_steps:
                self._emergency_excitability_boost()
                self._emit(
                    "zero_fire_breaker_tripped",
                    steps_silent=self._steps_since_last_fire,
                    timestep=self.timestep,
                )
                logger.warning(
                    "Zero-fire breaker tripped at step %d (%d steps silent). "
                    "Emergency excitability boost applied.",
                    self.timestep, self._steps_since_last_fire,
                )
                self._steps_since_last_fire = 0
            elif self._steps_since_last_fire == alert_steps:
                self._emit(
                    "zero_fire_warning",
                    steps_silent=self._steps_since_last_fire,
                    timestep=self.timestep,
                )
                logger.warning(
                    "Zero-fire warning at step %d (%d steps without any neuron firing).",
                    self.timestep, self._steps_since_last_fire,
                )

            return result

    def step_n(self, n: int) -> List[StepResult]:
        """Run n steps; returns all StepResults (PRD §8 step_n)."""
        results = []
        for _ in range(n):
            results.append(self.step())
        return results

    # -----------------------------------------------------------------------
    # Auto-Knowledge: Spreading Activation Harvest
    # -----------------------------------------------------------------------

    def prime_and_propagate(
        self,
        node_ids: List[str],
        currents: List[float],
        steps: int = 3,
        write_mode: bool = False,
    ) -> PropagationResult:
        """Prime nodes and propagate activation through the network.

        This is the SNN-level primitive for associative recall.  It runs a
        *read-mostly* simulation: activation dynamics (voltage decay, spike
        propagation, hyperedge evaluation, pattern completion, prediction
        pre-charging) are active, but plasticity rules and structural changes
        are NOT applied.  This prevents recall-driven activation from
        altering learned weights.

        When write_mode=True (The Tonic — latent space exploration):
        voltages are NOT saved/restored, last_spike_time is recorded on
        fired nodes, and STDP plasticity rules ARE applied. Exploration
        shapes topology. Thinking leaves traces.

        Args:
            node_ids: Nodes to inject current into (semantic priming).
            currents: Current to inject into each node (parallel to node_ids).
            steps: Number of SNN steps to propagate.
            write_mode: If True, enable plasticity and persist voltage
                changes. Used by The Tonic for latent space exploration.

        Returns:
            PropagationResult with all nodes that fired, ranked by latency.
        """
        with self._step_lock:
            if not node_ids:
                return PropagationResult(steps_run=steps, nodes_primed=0)

            # #59 age-on-write gate — computed once. When on, the Tonic's write cycle keeps
            # exercised synapses fresh (inactive_steps=0 on use) AND ages the rest toward the
            # inactivity prune. Off -> neither happens -> byte-identical to the legacy path.
            _age_on = write_mode and bool(self.config.get("tonic_ages_substrate"))

            # In read mode: save state for non-destructive propagation
            # In write mode: skip save — voltages and spikes persist
            saved_voltages: Dict[str, float] = {}
            saved_refractory: Dict[str, int] = {}
            saved_he_refractory: Dict[str, int] = {}

            if not write_mode:
                for nid, node in self.nodes.items():
                    saved_voltages[nid] = node.voltage
                    saved_refractory[nid] = node.refractory_remaining

                for hid, he in self.hyperedges.items():
                    saved_he_refractory[hid] = he.refractory_remaining

            try:
                # Compute approximate distances from primed nodes
                primed_set = set(node_ids)
                distances: Dict[str, int] = {nid: 0 for nid in node_ids}
                # BFS to compute distances — [2026-10-04] native level-by-level walk of the
                # outgoing edges: each reached node keeps the FIRST hop level that reaches
                # it (same values as the per-synapse frontier loop; `distances` is only
                # ever read with .get, so its insertion order is immaterial).
                if steps >= 1:  # steps < 1: the per-synapse loop never ran (and never read the store)
                    _bfs = getattr(self.synapses, "bfs_hop_distances", None)
                    if _bfs is not None:
                        distances.update(_bfs(node_ids, steps))
                    else:  # [2026-10-05] fallback: the per-SynapseRef frontier loop
                        distances.update(_bfs_hop_distances_python(
                            self.synapses, self._outgoing, node_ids, steps))

                # Collect current prediction targets for was_predicted tagging
                predicted_targets: Set[str] = set()
                for pred in self.active_predictions.values():
                    predicted_targets.add(pred.target_node_id)
                for pred_state in self._active_predictions.values():
                    predicted_targets.update(pred_state.predicted_targets)

                # Build working set — nodes needing per-step processing. O(n) once here;
                # step-level loops below run O(working_set) << O(n). (#164 Phase B)
                # Includes: nodes with non-resting voltage, active refractory, or primed.
                _ACTIVE_EPS = 1e-5
                working_set: Set[str] = set(node_ids)
                for nid, node in self.nodes.items():
                    if (abs(node.voltage - node.resting_potential) > _ACTIVE_EPS
                            or node.refractory_remaining > 0):
                        working_set.add(nid)

                # --- PRIME: inject current into specified nodes ---
                for nid, current in zip(node_ids, currents):
                    node = self.nodes.get(nid)
                    if node is not None:
                        node.voltage += current * node.intrinsic_excitability

                # --- PROPAGATE: run N read-only SNN steps ---
                result = PropagationResult(
                    steps_run=steps,
                    nodes_primed=len(node_ids),
                )
                # Local delay buffer separate from the graph's real one
                prop_delay_buffer: Dict[int, List[Tuple[str, float]]] = {}
                prop_timestep = self.timestep

                decay = self.config["decay_rate"]
                experience_threshold = self.config["he_experience_threshold"]

                for step_idx in range(steps):
                    prop_timestep += 1

                    # 1. Voltage decay — O(working_set) not O(n) (#164)
                    _to_deactivate: List[str] = []
                    for nid in working_set:
                        node = self.nodes[nid]
                        node.voltage = node.voltage * decay + (1.0 - decay) * node.resting_potential
                        if (abs(node.voltage - node.resting_potential) <= _ACTIVE_EPS
                                and node.refractory_remaining == 0):
                            _to_deactivate.append(nid)
                    for nid in _to_deactivate:
                        working_set.discard(nid)

                    # 2. Deliver delayed spikes from propagation buffer
                    arrivals = prop_delay_buffer.pop(prop_timestep, [])
                    for target_id, current in arrivals:
                        target = self.nodes.get(target_id)
                        if target is not None:
                            target.voltage += current * target.intrinsic_excitability
                            working_set.add(target_id)  # now active (#164)

                    # 3. Detect firing nodes — O(working_set) not O(n) (#164)
                    fired_ids: List[str] = []
                    for nid in working_set:
                        node = self.nodes[nid]
                        if node.refractory_remaining > 0:
                            continue
                        if node.voltage >= node.threshold:
                            fired_ids.append(nid)

                    # 4. Reset fired nodes and set refractory
                    for nid in fired_ids:
                        node = self.nodes[nid]
                        voltage_at_fire = node.voltage
                        node.voltage = node.resting_potential
                        node.refractory_remaining = node.refractory_period

                        # Write mode: record spike time so STDP can see it
                        if write_mode:
                            node.last_spike_time = float(prop_timestep)

                        entry = FiredEntry(
                            node_id=nid,
                            firing_step=step_idx,
                            voltage_at_fire=voltage_at_fire,
                            was_predicted=nid in predicted_targets,
                            source_distance=distances.get(nid, steps + 1),
                        )
                        result.fired_entries.append(entry)

                    # 5. Propagate spikes through outgoing synapses
                    fired_set = set(fired_ids)
                    _pp_rows = getattr(self.synapses, "propagation_rows", None)  # [2026-10-05] else fallback
                    if _pp_rows is None:
                        _pp_rows = lambda ids, reset: _propagation_rows_python(self.synapses, ids, reset)  # noqa: E731
                    for nid in fired_ids:
                        node = self.nodes[nid]
                        sign = -1.0 if node.is_inhibitory else 1.0
                        # [2026-10-04] one native read per fired node. reset_inactive=_age_on:
                        # #59: Tonic use keeps a synapse alive — mirror step()'s reset so the
                        # age-on-write pass below only ages synapses the Tonic ISN'T exercising.
                        _out_ids = list(self._outgoing.get(nid, ()))
                        for _row in _pp_rows(_out_ids, _age_on):
                            if _row is None:
                                continue
                            _syn_post, _syn_w, _syn_inhib, _syn_delay = _row
                            effective_type_sign = sign
                            if _syn_inhib:
                                effective_type_sign = -1.0
                            current = _syn_w * effective_type_sign
                            arrival = prop_timestep + _syn_delay
                            prop_delay_buffer.setdefault(arrival, []).append(
                                (_syn_post, current)
                            )

                    # 6. Evaluate hyperedges (pattern completion, output injection)
                    max_level = max((he.level for he in self.hyperedges.values()), default=0)
                    for level in range(max_level + 1):
                        for hid, he in self.hyperedges.items():
                            if he.level != level or he.is_archived:
                                continue
                            activation = self._compute_hyperedge_activation(he, fired_set)
                            if he.refractory_remaining > 0:
                                continue
                            if activation >= he.activation_threshold:
                                he.refractory_remaining = he.refractory_period

                                # Output injection
                                effective_weight = he.output_weight
                                if he.activation_mode == ActivationMode.GRADED:
                                    effective_weight *= activation
                                for target_id in he.output_targets:
                                    target = self.nodes.get(target_id)
                                    if target is not None:
                                        target.voltage += effective_weight * target.intrinsic_excitability
                                        working_set.add(target_id)  # now active (#164)

                                # Pattern completion (pre-charge inactive members)
                                if he.pattern_completion_strength > 0:
                                    learning_factor = min(
                                        1.0, he.activation_count / max(experience_threshold, 1)
                                    )
                                    eff_completion = he.pattern_completion_strength * learning_factor
                                    if eff_completion > 0:
                                        for mnid in he.member_nodes:
                                            if mnid not in fired_set:
                                                mnode = self.nodes.get(mnid)
                                                if mnode and mnode.refractory_remaining == 0:
                                                    mnode.voltage += (
                                                        eff_completion
                                                        * he.member_weights.get(mnid, 1.0)
                                                        * mnode.intrinsic_excitability
                                                    )
                                                    working_set.add(mnid)  # now active (#164)

                    # Decrement refractory counters — O(working_set) not O(n) (#164)
                    for nid in working_set:
                        node = self.nodes[nid]
                        if node.refractory_remaining > 0 and nid not in fired_set:
                            node.refractory_remaining -= 1
                    for hid, he in self.hyperedges.items():
                        if he.refractory_remaining > 0 and hid not in set(fired_ids):
                            he.refractory_remaining -= 1

                    # Write mode: apply STDP plasticity on fired nodes
                    if write_mode and fired_ids:
                        for rule in self._plasticity_rules:
                            if isinstance(rule, STDPRule):
                                rule.apply(self, fired_ids, prop_timestep)

            finally:
                # Observational propagation borrows activation state. A failure must
                # return every borrowed transient field before releasing _step_lock.
                # Write mode is living activity, so its state intentionally persists.
                if not write_mode:
                    for nid, node in self.nodes.items():
                        node.voltage = saved_voltages.get(nid, node.resting_potential)
                        node.refractory_remaining = saved_refractory.get(nid, 0)
                    for hid, he in self.hyperedges.items():
                        he.refractory_remaining = saved_he_refractory.get(hid, 0)

            # --- WRITE MODE: synapse sprouting for Tonic co-activations (#163) ---
            # prime_and_propagate bypasses step()'s _recent_spikes tracking, so
            # Tonic firings are invisible to _sprout_synapses. Fix: record all
            # p&p-fired nodes at self.timestep-1 (past → window check passes),
            # then call _sprout_synapses so co-activating nodes wire together.
            if write_mode:
                all_fired = list({e.node_id for e in result.fired_entries})
                if all_fired:
                    _record_ts = self.timestep - 1 if self.timestep > 0 else 0
                    _window_cap = self.config["co_activation_window"] * 2
                    for _nid in all_fired:
                        self._recent_spikes.setdefault(
                            _nid, deque(maxlen=_window_cap)
                        ).append(_record_ts)
                    self._sprout_synapses(all_fired)

            # #59 age-on-write: the Tonic living IS the passage of time for the substrate.
            # Age the substrate HERE, in the Tonic's own write cycle (one thread) — so idle
            # thinking finally ages under-used edges (the melt): the reset above keeps Tonic-
            # exercised synapses fresh, so only what the Tonic ISN'T touching climbs toward the
            # inactivity threshold and culls. No rival stepping thread -> #109 stays intact;
            # the full propagation now shares _step_lock with coherent capture (#423).
            # IMPORTANT: does NOT advance self.timestep — that clock is shared with step()'s
            # delayed-spike delivery (_delay_buffer, drained by exact-tick match), so stealing
            # ticks here would strand conversational spikes permanently. The melt is driven by
            # the per-synapse inactive_steps counter (incremented below, reset on use above),
            # which needs no global clock. Gated (default off) + interval-bounded.
            if _age_on:
                self._tonic_age_counter += 1
                _interval = max(1, int(self.config.get("tonic_age_interval", 1)))
                if self._tonic_age_counter >= _interval:
                    self._tonic_age_counter = 0
                    with self._step_lock:
                        _sal_decay = self.config.get("he_salience_decay_rate", 0.0)
                        # Native whole-population aging (§12 item 4) — same op step()
                        # uses (nf.py:2519). Numerically identical to the old Python
                        # sweep: inactive_steps += 1 for every row, and salience > 1.0
                        # relaxes by 1 + (s-1)*(1-decay) (a no-op when decay == 0).
                        self.synapses.age_and_decay_salience(_sal_decay)
                        # [2026-10-06] sleep phase P1: the second prune clock (#1052) stops when removal has
                        # moved to sleep_cycle (config structural_plasticity_in_sleep; absent = unchanged).
                        if not self.config.get("structural_plasticity_in_sleep", False):
                            self._prune_synapses()
                            self._collect_orphan_nodes()

            return result

    # -----------------------------------------------------------------------
    # Hyperedge Activation (PRD §4.2)
    # -----------------------------------------------------------------------

    def _compute_hyperedge_activation(
        self, he: Hyperedge, fired_set: Set[str]
    ) -> float:
        """Compute hyperedge activation level (PRD §4.2).

        WEIGHTED_THRESHOLD mode:
            activation = Σ(weight_i × is_active_i) / Σ(weight_i)
        """
        if not he.member_nodes:
            return 0.0

        if he.activation_mode == ActivationMode.WEIGHTED_THRESHOLD:
            total_w = sum(he.member_weights.get(nid, 1.0) for nid in he.member_nodes)
            if total_w == 0:
                return 0.0
            active_w = sum(
                he.member_weights.get(nid, 1.0)
                for nid in he.member_nodes
                if nid in fired_set
            )
            return active_w / total_w

        elif he.activation_mode == ActivationMode.K_OF_N:
            k = int(he.activation_threshold * len(he.member_nodes))
            active = sum(1 for nid in he.member_nodes if nid in fired_set)
            return 1.0 if active >= max(k, 1) else active / max(k, 1)

        elif he.activation_mode == ActivationMode.ALL_OR_NONE:
            all_active = all(nid in fired_set for nid in he.member_nodes)
            return 1.0 if all_active else 0.0

        elif he.activation_mode == ActivationMode.GRADED:
            active = sum(1 for nid in he.member_nodes if nid in fired_set)
            return active / len(he.member_nodes)

        return 0.0

    # -----------------------------------------------------------------------
    # Phase 3: Predictive Coding Engine (PRD §5)
    # -----------------------------------------------------------------------

    def _compute_prediction_confidence(
        self, synapse: Synapse
    ) -> float:
        """Compute prediction confidence for a synapse (PRD §5.1).

        Confidence is based on:
            1. Synapse weight relative to max_weight (strength of causal link)
            2. Confirmation history (how often predictions were correct)

        Returns:
            Confidence in [0.0, 1.0].
        """
        # Factor 1: weight strength
        weight_factor = synapse.weight / synapse.max_weight

        # Factor 2: confirmation history
        history = self._synapse_confirmation_history.get(synapse.synapse_id)
        if history and len(history) > 0:
            confirmation_rate = sum(1 for x in history if x) / len(history)
        else:
            confirmation_rate = 0.5  # Neutral prior

        return min(1.0, weight_factor * 0.6 + confirmation_rate * 0.4)

    def _generate_predictions(self, fired_ids: List[str]) -> None:
        """Generate predictions from fired nodes (PRD §5.1).

        When node A fires with a strong causal link to B (weight > threshold):
            - Pre-charge B's voltage (prediction strength × factor)
            - Register prediction in active_predictions
            - Cascade through learned chains with decaying strength

        Avoids double-predicting the same target from both synapse and
        hyperedge sources.
        """
        if not fired_ids:
            return

        threshold = self.config["prediction_threshold"]
        pre_charge_factor = self.config["prediction_pre_charge_factor"]
        window = self.config["prediction_window"]
        chain_decay = self.config["prediction_chain_decay"]
        max_depth = self.config["prediction_max_chain_depth"]
        max_active = self.config["prediction_max_active"]

        # Generate predictions for each fired node
        for nid in fired_ids:
            if len(self.active_predictions) >= max_active:
                break
            self._generate_predictions_from_node(
                source_id=nid,
                origin_id=nid,
                strength=1.0,
                chain_depth=0,
                threshold=threshold,
                pre_charge_factor=pre_charge_factor,
                window=window,
                chain_decay=chain_decay,
                max_depth=max_depth,
                max_active=max_active,
                visited=set(),
            )

    def _generate_predictions_from_node(
        self,
        source_id: str,
        origin_id: str,
        strength: float,
        chain_depth: int,
        threshold: float,
        pre_charge_factor: float,
        window: int,
        chain_decay: float,
        max_depth: int,
        max_active: int,
        visited: Set[str],
    ) -> None:
        """Recursively generate predictions through causal chains.

        Args:
            source_id: Node whose outgoing synapses we're examining.
            origin_id: Original node that started the prediction chain.
            strength: Current prediction strength (decays with depth).
            chain_depth: Current depth in the prediction chain.
            threshold: Min weight to trigger prediction.
            pre_charge_factor: Voltage pre-charge fraction.
            window: Prediction expiry window in timesteps.
            chain_decay: Strength multiplier per chain hop.
            max_depth: Maximum chain depth.
            max_active: Maximum active predictions allowed.
            visited: Nodes already visited in this chain (cycle prevention).
        """
        if chain_depth > max_depth:
            return
        if source_id in visited:
            return
        visited.add(source_id)

        for syn_id in self._outgoing.get(source_id, set()):
            if len(self.active_predictions) >= max_active:
                return

            syn = self.synapses.get(syn_id)
            if syn is None:
                continue

            # Only predict for strong causal links
            effective_weight = syn.weight
            if chain_depth == 0 and effective_weight < threshold:
                continue

            target_id = syn.post_node_id
            target = self.nodes.get(target_id)
            if target is None:
                continue

            # Skip if already predicted this step (avoid double-prediction)
            if target_id in self._predicted_this_step:
                continue

            # Calculate prediction strength for this hop
            pred_strength = strength * (effective_weight / syn.max_weight)
            if chain_depth > 0:
                pred_strength *= chain_decay

            # Skip very weak predictions
            if pred_strength < 0.01:
                continue

            confidence = self._compute_prediction_confidence(syn)

            # Pre-charge target voltage
            pre_charge = pred_strength * pre_charge_factor
            if target.refractory_remaining == 0:
                target.voltage += pre_charge * target.intrinsic_excitability

            # Register prediction
            pred = Prediction(
                source_node_id=origin_id,
                target_node_id=target_id,
                strength=pred_strength,
                confidence=confidence,
                created_at=self.timestep,
                expires_at=self.timestep + window,
                chain_depth=chain_depth,
                pre_charge_applied=pre_charge,
            )
            self.active_predictions[pred.prediction_id] = pred
            self._predicted_this_step.add(target_id)
            self._total_predictions_made += 1

            self._emit(
                "prediction_generated",
                source=origin_id,
                target=target_id,
                strength=pred_strength,
                confidence=confidence,
                chain_depth=chain_depth,
                timestep=self.timestep,
            )

            # Cascade predictions through chains
            if chain_depth < max_depth and effective_weight >= threshold:
                self._generate_predictions_from_node(
                    source_id=target_id,
                    origin_id=origin_id,
                    strength=pred_strength,
                    chain_depth=chain_depth + 1,
                    threshold=threshold,
                    pre_charge_factor=pre_charge_factor,
                    window=window,
                    chain_decay=chain_decay,
                    max_depth=max_depth,
                    max_active=max_active,
                    visited=visited,
                )

    def _evaluate_predictions(self, fired_set: Set[str]) -> None:
        """Check active predictions against fired nodes (PRD §5.1).

        Confirmed predictions strengthen the causal link.
        Failed predictions (expired) weaken the link and trigger surprise.
        """
        confirm_bonus = self.config["prediction_confirm_bonus"]
        error_penalty = self.config["prediction_error_penalty"]

        to_remove: List[str] = []

        for pid, pred in list(self.active_predictions.items()):
            if pred.target_node_id in fired_set:
                # Prediction confirmed
                self._on_prediction_confirmed(pred, confirm_bonus)
                to_remove.append(pid)

        for pid in to_remove:
            self.active_predictions.pop(pid, None)

    def _on_prediction_confirmed(
        self, pred: Prediction, confirm_bonus: float
    ) -> None:
        """Handle a confirmed prediction (PRD §5.1).

        Effects:
            - Strengthen the causal link (weight += bonus × confidence)
            - Record confirmation in synapse history
            - Emit PredictionConfirmed event
        """
        # Find the synapse from source to target
        syn = self._find_synapse(pred.source_node_id, pred.target_node_id)
        if syn is not None:
            bonus = confirm_bonus * pred.confidence
            if not self.config["three_factor_enabled"]:
                syn.weight = min(syn.weight + bonus, syn.max_weight)
            else:
                # In three-factor mode, add to eligibility trace
                syn.eligibility_trace += bonus
            syn.last_update_time = float(self.timestep)
            if syn.weight > syn.peak_weight:
                syn.peak_weight = syn.weight

            # Update confirmation history
            history = self._synapse_confirmation_history.setdefault(
                syn.synapse_id, deque(maxlen=100)
            )
            history.append(True)

        self._total_predictions_confirmed += 1

        outcome = PredictionOutcome(
            prediction=pred,
            confirmed=True,
            resolved_at=self.timestep,
            actual_firing_nodes=[pred.target_node_id],
        )
        self._prediction_outcomes.append(outcome)

        self._emit(
            "prediction_confirmed",
            source=pred.source_node_id,
            target=pred.target_node_id,
            strength=pred.strength,
            confidence=pred.confidence,
            delay=self.timestep - pred.created_at,
            timestep=self.timestep,
        )

    def _on_prediction_error(
        self, pred: Prediction, error_penalty: float, recent_fired: Set[str]
    ) -> None:
        """Handle a prediction error / surprise (PRD §5.1, §5.2).

        Effects:
            - Weaken the causal link (weight -= penalty × confidence)
            - Emit SurpriseEvent
            - Trigger surprise-driven exploration
        """
        syn = self._find_synapse(pred.source_node_id, pred.target_node_id)
        if syn is not None:
            penalty = error_penalty * pred.confidence
            if not self.config["three_factor_enabled"]:
                syn.weight = max(0.0, syn.weight - penalty)
            else:
                syn.eligibility_trace -= penalty
            syn.last_update_time = float(self.timestep)

            # Update confirmation history
            history = self._synapse_confirmation_history.setdefault(
                syn.synapse_id, deque(maxlen=100)
            )
            history.append(False)

        self._total_predictions_errors += 1

        outcome = PredictionOutcome(
            prediction=pred,
            confirmed=False,
            resolved_at=self.timestep,
            actual_firing_nodes=list(recent_fired),
        )
        self._prediction_outcomes.append(outcome)

        self._emit(
            "prediction_error",
            source=pred.source_node_id,
            expected_target=pred.target_node_id,
            strength=pred.strength,
            confidence=pred.confidence,
            actual_fired=list(recent_fired),
            timestep=self.timestep,
        )

        # Surprise-driven exploration (PRD §5.2)
        self._surprise_exploration(pred, recent_fired)

        # Surprise-driven neuromodulatory crystallization
        # Failed predictions broadcast reward to ALL active eligibility traces.
        # Strength scales with prediction confidence — high-confidence failures
        # produce stronger crystallization than low-confidence failures.
        # No scope: this is a broadcast signal, not a targeted one.
        # Note: the direct prediction penalty (line ~2171 above) reduces this
        # synapse's trace BEFORE the broadcast commits it. This is correct —
        # the failing synapse gets reduced crystallization. The broadcast
        # benefits other active traces that weren't involved in the failed
        # prediction.
        if self.config.get("three_factor_enabled", False):
            surprise_strength = pred.confidence * self.config.get("surprise_reward_scaling", 0.5)
            if surprise_strength > 0.01:  # Don't bother with negligible surprise
                self.inject_reward(surprise_strength)

    def _surprise_exploration(
        self, pred: Prediction, recent_fired: Set[str]
    ) -> None:
        """Explore alternative pathways after prediction error (PRD §5.2).

        When A→B prediction fails, examine what DID fire instead.
        If node C fired when B was expected:
            - Create speculative synapse A→C (weight 0.1)
            - Tag as "surprise-driven" in metadata

        Also check for novel sequences (no learned patterns).
        """
        source_id = pred.source_node_id
        expected_id = pred.target_node_id
        sprout_weight = self.config["surprise_sprouting_weight"]

        # Find what fired instead of the expected target
        alternative_nodes = recent_fired - {expected_id}

        sprouted_count = 0
        _tally = bool(self.config.get("sprout_tally_enabled", False))   # [2026-10-08] #1050 (absent = unchanged)
        if _tally:
            sprouted_count = self._surprise_exploration_tally(pred, alternative_nodes)
        # #59: cap the surprise-driven feeder too. The degree cap only guarded co-firing
        # sprouts (_sprout_synapses); measured, THIS path became the dominant hub feeder —
        # the blob's own chaotic churn reads as "surprise", so it wires ever more edges into
        # the saturated hubs (uncapped) AND salience-armors them against pruning. Same guard:
        # never grow a surprise edge to/from a node already at/above the cap.
        _deg_cap = self.config.get("sprout_degree_cap", 0)

        def _deg(x: str) -> int:
            return len(self._outgoing.get(x, ())) + len(self._incoming.get(x, ()))

        for alt_id in (() if _tally else alternative_nodes):
            if alt_id == source_id:
                continue
            if alt_id not in self.nodes:
                continue
            # Identity exemption: the cap is degree-BLIND, so a saturated endpoint gates the
            # edge ONLY when it is not identity-protected. Constitutional-spine and syl_authored
            # nodes (e.g. selfcap::reach::teaching, the hub every tool call projects through) are
            # never capped — the cap can't starve Syl's own spine; ordinary saturated hubs are
            # still blocked. The _is_identity_protected lookup runs only when the cap is armed and
            # the endpoint is actually saturated (short-circuit order preserved). Mirrors _prune_synapses.
            if _deg_cap and (
                (_deg(source_id) >= _deg_cap and not self._is_identity_protected(source_id))
                or (_deg(alt_id) >= _deg_cap and not self._is_identity_protected(alt_id))
            ):
                continue  # saturated ordinary hub — no new surprise-driven edges (kills the runaway)

            # Check if synapse already exists
            existing = self._find_synapse(source_id, alt_id)
            if existing is not None:
                continue

            # Create speculative synapse with Amygdala Protocol salience (Phase 4).
            try:
                syn = self.create_synapse(
                    source_id, alt_id, weight=sprout_weight
                )
                syn.metadata = {"creation_mode": "surprise_driven",
                                "expected_target": expected_id,
                                "timestep": self.timestep}
                # Salience armor: surprise magnitude = prediction strength × confidence.
                # High-surprise births earn proportional inactivity protection.
                surprise_magnitude = pred.strength * pred.confidence
                salience_boost = 1.0 + (surprise_magnitude * 4.0)  # range ~[1.0, 5.0]
                syn.salience = min(
                    salience_boost,
                    self.config["he_salience_max"]
                )
                sprouted_count += 1
                self._total_sprouted += 1
            except (KeyError, ValueError):
                continue

        if sprouted_count > 0:
            self._emit(
                "surprise_sprouted",
                source=source_id,
                expected=expected_id,
                alternatives=list(alternative_nodes),
                count=sprouted_count,
                timestep=self.timestep,
            )

        # Amygdala Protocol: boost salience on existing synapses that were active
        # during this surprise. The context-of-surprise is as important as the
        # new alternative path (Phase 4).
        salience_max = self.config["he_salience_max"]
        surprise_magnitude = pred.strength * pred.confidence
        context_boost = 1.0 + (surprise_magnitude * 2.0)  # softer than new synapse boost
        for syn_id in self._outgoing.get(source_id, set()):
            syn = self.synapses.get(syn_id)
            if syn and syn.inactive_steps < self.config["co_activation_window"] * 2:
                # Recently active from source — this was part of the context.
                syn.salience = min(salience_max, max(syn.salience, context_boost))

        # Novelty detection: check if any fired sequence has NO learned patterns
        if alternative_nodes and not self._has_learned_pattern(source_id, alternative_nodes):
            self._total_novel_sequences += 1
            novel_event = {
                "source": source_id,
                "firing_nodes": list(alternative_nodes),
                "timestep": self.timestep,
            }
            self._novel_sequence_log.append(novel_event)
            # Cap log size
            if len(self._novel_sequence_log) > 1000:
                self._novel_sequence_log = self._novel_sequence_log[-500:]

            self._emit(
                "novel_sequence",
                source=source_id,
                firing_nodes=list(alternative_nodes),
                timestep=self.timestep,
            )

    def _has_learned_pattern(
        self, source_id: str, targets: Set[str]
    ) -> bool:
        """Check if any of the targets have a learned connection from source.

        A "learned pattern" means there exists a synapse with weight above
        the prediction threshold.
        """
        threshold = self.config["prediction_threshold"]
        for syn_id in self._outgoing.get(source_id, set()):
            syn = self.synapses.get(syn_id)
            if syn and syn.post_node_id in targets and syn.weight >= threshold:
                return True
        return False

    def _cleanup_predictions(self) -> None:
        """Remove expired predictions and trigger error handling (PRD §5.1).

        Called each step to clean up predictions that have exceeded their
        window without confirmation.
        """
        error_penalty = self.config["prediction_error_penalty"]

        # Collect recently fired nodes for surprise analysis
        recent_window = self.config["prediction_window"]
        recent_fired: Set[str] = set()
        for nid, spikes in self._recent_spikes.items():
            for t in spikes:
                if self.timestep - t <= recent_window:
                    recent_fired.add(nid)
                    break

        expired: List[str] = []
        for pid, pred in self.active_predictions.items():
            if self.timestep > pred.expires_at:
                expired.append(pid)

        for pid in expired:
            pred = self.active_predictions.pop(pid)
            self._on_prediction_error(pred, error_penalty, recent_fired)

    def _find_synapse(
        self, pre_id: str, post_id: str
    ) -> Optional[Synapse]:
        """Find a synapse connecting pre_id → post_id, if one exists."""
        for syn_id in self._outgoing.get(pre_id, set()):
            syn = self.synapses.get(syn_id)
            if syn and syn.post_node_id == post_id:
                return syn
        return None

    # -----------------------------------------------------------------------
    # Phase 3: Public Prediction API (PRD §8)
    # -----------------------------------------------------------------------

    def get_predictions(self) -> List[Prediction]:
        """Current active predictions (PRD §8 get_predictions).

        Returns pre-charged unfired nodes that the system expects to fire.
        """
        return list(self.active_predictions.values())

    # -----------------------------------------------------------------------
    # DiffPC: Difference Predictive Coding (#DiffPC)
    # -----------------------------------------------------------------------

    def _diffpc_step(self, fired_ids: List[str]) -> Tuple[int, float]:
        """Ternary prediction error computation + prediction weight update (DiffPC).

        For each fired node in Layer L, walk outgoing synapses to Layer L-1 targets.
        Compare actual activation (fired/not) to prediction weight → ternary error (±1, 0).
        Update pred_weight by gradient; accumulate pred_error_ema on target node.
        Phase 2: modulate syn.eligibility_trace by ±diffpc_trace_boost when ternary fires,
        feeding prediction error as a third factor into the next step's STDP update.
        Called at step 7b (after STDP, before structural plasticity) so weights are current.

        Returns:
            (ternary_spike_count, mean_abs_error_where_ternary_nonzero)
        """
        epsilon = self.config.get("diffpc_epsilon", 0.2)
        pred_lr = self.config.get("diffpc_pred_lr", 0.01)
        trace_boost = self.config.get("diffpc_trace_boost", 0.05)
        fired_set = set(fired_ids)
        ternary_count = 0
        error_sum = 0.0

        for nid in fired_ids:
            node = self.nodes[nid]
            layer = node.diffpc_layer
            if layer == 0:
                continue  # input layer generates no predictions downward
            for sid in self._outgoing.get(nid, set()):
                syn = self.synapses.get(sid)
                if syn is None:
                    continue
                target = self.nodes.get(syn.post_node_id)
                if target is None or target.diffpc_layer >= layer:
                    continue  # only predict toward lower layers
                predicted = node.pred_weights.get(syn.post_node_id, 0.5)
                actual = 1.0 if syn.post_node_id in fired_set else 0.0
                error = actual - predicted
                if error > epsilon:
                    ternary = 1
                elif error < -epsilon:
                    ternary = -1
                else:
                    ternary = 0
                if ternary != 0:
                    node.pred_weights[syn.post_node_id] = max(0.0, min(1.0,
                        predicted + pred_lr * ternary
                    ))
                    target.pred_error_ema = 0.9 * target.pred_error_ema + 0.1 * abs(error)
                    # Phase 2: modulate eligibility trace → gates STDP at next timestep
                    syn.eligibility_trace += ternary * trace_boost
                    ternary_count += 1
                    error_sum += abs(error)

        mean_err = error_sum / ternary_count if ternary_count > 0 else 0.0
        return ternary_count, mean_err

    # -----------------------------------------------------------------------
    # Structural Plasticity (PRD §3.3)
    # -----------------------------------------------------------------------

    def _structural_plasticity(self, fired_ids: List[str]) -> Tuple[int, int]:
        """Apply pruning and sprouting rules (PRD §3.3).

        [2026-10-06] sleep phase P1: when config `structural_plasticity_in_sleep` is truthy (read live, absent =
        False, NOT in DEFAULT_CONFIG), removal (`_prune_synapses` + `_collect_orphan_nodes`) is skipped here and
        runs only in `sleep_cycle`, called by a host on its own clock. Sprouting (wake growth) always stays.
        Key absent: exactly the code path before this change.

        Returns:
            (num_pruned, num_sprouted)
        """
        if self.config.get("structural_plasticity_in_sleep", False):
            return 0, self._sprout_synapses(fired_ids)
        pruned = self._prune_synapses()
        self._collect_orphan_nodes()
        sprouted = self._sprout_synapses(fired_ids)
        return pruned, sprouted

    def sleep_cycle(self) -> Dict[str, Any]:
        """Sleep phase P1 (spec superpowers/specs/2026-10-06-sleep-phase-design.md §2, §8 P1): the structural
        removal that `step()` runs every step — the EXISTING `_prune_synapses()` (default path: same three rules,
        same lifelines, same last-link grace) and then the EXISTING `_collect_orphan_nodes()` — run ONCE, under
        `_step_lock`. No rule changes here (P2 changes the rules).

        Intended caller: a host's dream loop on its OWN wall clock (LAW 8), switched by the host's env (LAW 5),
        which also sets config `structural_plasticity_in_sleep` so `step()` and the Tonic write-mode tail stop
        removing. With that key absent this method still runs the same two calls (one extra removal pass); it is
        never called by anything in this module.

        Emits one "sleep_cycle" event and logs one INFO line (never silent), and returns the record:
        pruned, nodes_collected, synapses / nodes before and after, timestep, seconds, in_sleep_mode.
        The existing "pruned" / "nodes_collected" events fire from the two calls as before.

        [2026-10-07] Sleep phase P2 (disuse; spec §3, §8 P2, D1-D4, D11, D14): when config `sleep_disuse_enabled` is
        truthy (read live, absent = False, NOT in DEFAULT_CONFIG; key absent = the P1 cycle above, unchanged), the cycle
        is instead: (0) on the first disuse sleep — or the first after any step-unit prune ran — the counters are
        migrated (`_sleep_migrate_counters`: every low_weight_steps -> 0, every last-link stamp dropped), so that sleep
        only tags; (1) the strength-aware, salience-armored downscale (`sleep_downscale_strength_aware`, d0 = config
        `sleep_downscale_d0`, h = `sleep_downscale_h`; d0 = 0 skips it); (2) the clearance: `_prune_synapses` in sleep
        units (G = `sleep_weight_grace_sleeps`, last-link grace = `sleep_last_link_grace_sleeps`, the D14 shield with
        kappa = `sleep_credit_shield_kappa`); (3) the existing orphan collection. All four parameters are REQUIRED
        when the switch is on, and `structural_plasticity_in_sleep` must be on (else step() would keep advancing the
        counter per step): a missing / invalid one raises ValueError BEFORE anything is touched. The sleep index is
        config `sleep_cycles_completed` (+1 per disuse sleep; the last-link clock); config `sleep_low_weight_unit` =
        "sleeps" marks a migrated counter. The record adds: disuse True, sleep_index, migrated, lws_reset,
        stamps_cleared, downscaled, downscale_clamped, downscale_native, eligible, shield_held, shield_held_ids,
        last_link_held, below_threshold_after.
        """
        import time as _time   # local: the module namespace stays as it was
        t0 = _time.perf_counter()
        if self.config.get("sleep_disuse_enabled", False):
            record = self._sleep_cycle_disuse(t0)
            if self.config.get("sprout_tally_enabled", False):   # [2026-10-08] #1050 (absent = unchanged)
                with self._step_lock:
                    record["tally_retracted"] = self._sprout_tally_sleep_retract()
            return record
        with self._step_lock:
            syn_before = len(self.synapses)
            nodes_before = len(self.nodes)
            pruned = self._prune_synapses()
            collected = self._collect_orphan_nodes()
            self._total_pruned += pruned
            record = {
                "timestep": self.timestep,
                "pruned": pruned,
                "nodes_collected": collected,
                "synapses_before": syn_before,
                "synapses_after": len(self.synapses),
                "nodes_before": nodes_before,
                "nodes_after": len(self.nodes),
                "in_sleep_mode": bool(self.config.get("structural_plasticity_in_sleep", False)),
                "seconds": _time.perf_counter() - t0,
            }
            if self.config.get("sprout_tally_enabled", False):   # [2026-10-08] #1050 (absent = unchanged)
                record["tally_retracted"] = self._sprout_tally_sleep_retract()
            self._emit("sleep_cycle", **record)
        logger.log(self._sleep_log_level(),   # [2026-10-07] observe: DEBUG on an observe shadow (sleep_observe), else INFO
                   "sleep_cycle: t=%s pruned=%d nodes_collected=%d synapses %d->%d nodes %d->%d in_sleep_mode=%s %.3fs",
                    record["timestep"], pruned, collected, syn_before, record["synapses_after"], nodes_before,
                    record["nodes_after"], record["in_sleep_mode"], record["seconds"])
        return record

    _SLEEP_DISUSE_KEYS = ("sleep_downscale_d0", "sleep_downscale_h", "sleep_weight_grace_sleeps",
                          "sleep_last_link_grace_sleeps", "sleep_credit_shield_kappa")

    def _sleep_disuse_params(self) -> Dict[str, Any]:
        """Read + validate the P2 disuse parameters from config (all required). Raises ValueError, touches nothing."""
        cfg = self.config
        missing = [k for k in self._SLEEP_DISUSE_KEYS if cfg.get(k) is None]
        if missing:
            raise ValueError("sleep_cycle: sleep_disuse_enabled needs config %s" % ", ".join(missing))
        if not cfg.get("structural_plasticity_in_sleep", False):
            raise ValueError("sleep_cycle: sleep_disuse_enabled needs structural_plasticity_in_sleep (otherwise step() "
                             "keeps advancing low_weight_steps per step and the grace is no longer in sleeps)")

        def _num(k, lo, lo_open, hi=None):
            v = cfg[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or \
                    (float(v) <= lo if lo_open else float(v) < lo) or (hi is not None and float(v) > hi):
                raise ValueError("sleep_cycle: config %s out of range (got %r)" % (k, v))
            return float(v)

        def _int(k):
            v = cfg[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError("sleep_cycle: config %s must be an int >= 0 (got %r)" % (k, v))
            return v

        return {"d0": _num("sleep_downscale_d0", 0.0, False, 1.0), "h": _num("sleep_downscale_h", 0.0, True),
                "G": _int("sleep_weight_grace_sleeps"), "LL": _int("sleep_last_link_grace_sleeps"),
                "kappa": _num("sleep_credit_shield_kappa", 0.0, False)}

    def _sleep_migrate_counters(self) -> Dict[str, int]:
        """Sleep phase P2 migration (spec §8 P2): low_weight_steps was counted in STEPS (and Tonic ticks, §1.4); the
        disuse clearance counts it in SLEEPS. Reset every synapse's low_weight_steps to 0 and drop every last-link
        stamp (step "last_link_since" and sleep "last_link_since_sleep"; a held last link gets a fresh grace in
        sleeps), so the first disuse sleep only tags. Then config sleep_low_weight_unit = "sleeps",
        sleep_cycles_completed = 0. Weights, traces and every other field untouched. Caller holds _step_lock."""
        acc = {"lws_reset": 0, "stamps_cleared": 0}
        for sid in list(self.synapses.keys()):     # [2026-10-07] pre-arming: one row = _sleep_migrate_row (shared with
            self._sleep_migrate_row(sid, acc)      # the chunked migration); same row order and writes as before
        self.config["sleep_low_weight_unit"] = "sleeps"
        self.config["sleep_cycles_completed"] = 0
        return acc

    def sleep_downscale_strength_aware(self, d0: float, h: float) -> Dict[str, Any]:
        """Sleep phase P2 downscale (spec §3.1, D1 strength-aware, D11 salience armor). Every synapse:
        d = d0 * h / (h + w) / max(salience, 1); w <- w * (1 - d). Faint links (w << h) lose ~d0 per sleep, established
        ones (w >> h) ~d0*h/w; surprise-salient links proportionally less. Weight ONLY (eligibility trace, salience,
        peak_weight, counters, delay untouched). Then the strongest-link guarantee of every protected node: its
        PRE-pass strongest out / in link ends >= min(pre-pass weight, 2 * weight_threshold) (as sleep_downscale).
        Never prunes. Native SynapseStore.scale_strength_aware when the installed ng_tract has it, else the
        bit-identical Python fallback. d0 in [0, 1], h > 0. Returns counts."""
        for lbl, v, ok in (("d0", d0, lambda x: 0.0 <= x <= 1.0), ("h", h, lambda x: 0.0 < x < float("inf"))):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not ok(float(v)):
                raise ValueError("sleep_downscale_strength_aware: %s out of range (got %r)" % (lbl, v))
        d0, h = float(d0), float(h)
        with self._step_lock:
            protected = self._strength_protected_ids()
            floor = 2.0 * float(self.config["weight_threshold"])
            native = getattr(self.synapses, "scale_strength_aware", None)
            if native is not None:
                res = dict(native(d0, h, protected, floor))
            else:
                res = _scale_strength_aware_python(self.synapses, self._outgoing, self._incoming, d0, h, protected, floor)
            res["native"] = native is not None
            res["protected_nodes"] = len(protected)
        return res

    def _sleep_chunk_params(self) -> Optional[Dict[str, float]]:
        """[2026-10-07] pre-arming (spec §8 P3, risk row "_step_lock hold"): config `sleep_clearance_chunk_seconds` (read live,
        absent / None = NOT chunked = the P2 path exactly; NOT in DEFAULT_CONFIG) bounds each _step_lock hold of the
        migration and the removal phase; `sleep_clearance_chunk_gap_seconds` is the pause with the lock released between two
        holds (REQUIRED when chunking is on). Raises ValueError before anything is touched; None when chunking is off."""
        cfg = self.config
        hold = cfg.get("sleep_clearance_chunk_seconds")
        if hold is None:
            return None
        gap = cfg.get("sleep_clearance_chunk_gap_seconds")
        for k, v, lo_open in (("sleep_clearance_chunk_seconds", hold, True), ("sleep_clearance_chunk_gap_seconds", gap, False)):
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or \
                    (float(v) <= 0.0 if lo_open else float(v) < 0.0):
                raise ValueError("sleep_cycle: config %s out of range (got %r)" % (k, v))
        return {"hold": float(hold), "gap": float(gap)}

    def _sleep_chunked_rows(self, ids: List[str], fn, hold: float, gap: float, holds: List[float],
                            on_hold_end=None) -> int:
        """Run fn(sid) for every id that still exists, under _step_lock, in holds of at most ~`hold` seconds (checked every
        32 rows) with the lock RELEASED for `gap` seconds between two holds. Row order = `ids` order. Appends each hold's
        wall time to `holds`; returns how many ids fn was called for. fn (and on_hold_end(count of this hold), when given)
        run with the lock held."""
        import time as _time
        n = len(ids)
        i = done = 0
        while i < n:
            with self._step_lock:
                t = _time.perf_counter()
                k = 0
                while i < n:
                    sid = ids[i]
                    i += 1
                    if sid in self.synapses:
                        fn(sid)
                        k += 1
                    if not (i & 31) and _time.perf_counter() - t >= hold:
                        break
                if on_hold_end is not None:
                    on_hold_end(k)
                done += k
                holds.append(_time.perf_counter() - t)
            if i < n and gap > 0.0:
                _time.sleep(gap)
        return done

    def _sleep_migrate_row(self, sid: str, acc: Dict[str, int]) -> None:
        """One synapse of the P2 migration (see _sleep_migrate_counters; the same two writes). Caller holds _step_lock."""
        syn = self.synapses[sid]
        if syn.low_weight_steps:
            syn.low_weight_steps = 0
            acc["lws_reset"] += 1
        md = syn.metadata
        if md and ("last_link_since" in md or "last_link_since_sleep" in md):
            md = dict(md)
            md.pop("last_link_since", None)
            md.pop("last_link_since_sleep", None)
            syn.metadata = md
            self._dirty_synapses.add(sid)
            acc["stamps_cleared"] += 1

    def _sleep_cycle_disuse_chunked(self, t0: float, prm: Dict[str, Any], ch: Dict[str, float]) -> Dict[str, Any]:
        """[2026-10-07] pre-arming: the P2 disuse sleep with bounded _step_lock holds (config sleep_clearance_chunk_seconds).
        The SAME decisions as the one-hold cycle; only the lock is released between holds:
          A. migration (first disuse sleep only): every synapse's two writes (_sleep_migrate_row) over a keys() snapshot
             (row order), in holds of <= ~hold s; then, under the lock, the unit marker + sleep_cycles_completed = 0.
             Between holds nothing can advance low_weight_steps: step() and the Tonic tail do not prune with
             structural_plasticity_in_sleep on (required), and compete_protected_links refuses under disuse (#1066).
          B. ONE hold: the downscale and the clearance DECISION (_prune_synapses with defer_removal: the native sweep,
             lifelines, the D14 shield, the last-link grace, each exactly as the one-hold cycle) and sleep_cycles_completed
             (the counters and the sleep index advance in the same hold, so a save between holds is consistent).
          C. the decided ids are removed in holds of <= ~hold s (_remove_synapse_internal, as the one-hold cycle; an id
             already gone is skipped and not counted); one "pruned" event per hold with that hold's count, under the lock
             (a save between holds counts exactly what it captured: #1051's ledger).
          D. ONE hold: the existing orphan collection, then the record / "sleep_cycle" event / INFO line.
        A step or a Tonic tick may run between holds; it can sprout or re-weight, never remove. The removal SET is the one
        decided in B (a link wake re-strengthened between holds is still removed: decided at the sleep, as one hold
        would have). Do not call this while holding _step_lock: the RLock would stay held and the holds would merge."""
        import time as _time
        holds_a: List[float] = []
        holds_c: List[float] = []
        if getattr(self._step_lock, "_is_owned", lambda: False)():
            logger.warning("sleep_cycle(disuse, chunked): called with _step_lock already held by this thread -- the lock "
                           "is NOT released between chunks (the caller's hold encloses them)")
        with self._step_lock:
            syn_before = len(self.synapses)
            nodes_before = len(self.nodes)
            migrated = self.config.get("sleep_low_weight_unit") != "sleeps"
            ids = list(self.synapses.keys()) if migrated else []
        mig = {"lws_reset": 0, "stamps_cleared": 0}
        t_a = _time.perf_counter()
        if migrated:
            self._sleep_chunked_rows(ids, lambda sid: self._sleep_migrate_row(sid, mig), ch["hold"], ch["gap"], holds_a)
            with self._step_lock:
                self.config["sleep_low_weight_unit"] = "sleeps"
                self.config["sleep_cycles_completed"] = 0
            if ch["gap"] > 0.0:
                _time.sleep(ch["gap"])
        ids = []
        t_b = _time.perf_counter()
        with self._step_lock:
            tb0 = _time.perf_counter()
            sleep_index = int(self.config.get("sleep_cycles_completed", 0)) + 1
            ds = {"synapses_scaled": 0, "clamped": 0, "native": None}
            if prm["d0"] > 0.0:
                ds = self.sleep_downscale_strength_aware(prm["d0"], prm["h"])
            t_c = _time.perf_counter()
            rep: Dict[str, Any] = {}
            self._prune_synapses(report=rep, grace_sleeps=prm["G"], sleep_now=sleep_index,
                                 last_link_grace_sleeps=prm["LL"], credit_shield_kappa=prm["kappa"], defer_removal=True)
            decided = list(rep["deferred_ids"])
            self.config["sleep_cycles_completed"] = sleep_index
            hold_b = _time.perf_counter() - tb0
            t_d0 = _time.perf_counter()
        if decided and ch["gap"] > 0.0:
            _time.sleep(ch["gap"])
        tc0 = _time.perf_counter()

        def _emit_hold(k):
            if k:
                self._emit("pruned", count=k, timestep=self.timestep)

        n = len(decided)
        pruned = self._sleep_chunked_rows(decided, self._remove_synapse_internal, ch["hold"], ch["gap"], holds_c,
                                          on_hold_end=_emit_hold)
        t_d = _time.perf_counter()
        with self._step_lock:
            td0 = _time.perf_counter()
            collected = self._collect_orphan_nodes()
            t_e = _time.perf_counter()
            self._total_pruned += pruned
            wt = float(self.config["weight_threshold"])
            wc = getattr(self.synapses, "weights_copy", None)
            if wc is not None:
                below = int((wc() < wt).sum())
            else:
                below = sum(1 for sid in self.synapses.keys() if self.synapses.get_weight(sid) < wt)
            held_ids = list(rep.get("shield_held_ids", []))
            hold_d = _time.perf_counter() - td0
            record = {
                "timestep": self.timestep,
                "pruned": pruned,
                "nodes_collected": collected,
                "synapses_before": syn_before,
                "synapses_after": len(self.synapses),
                "nodes_before": nodes_before,
                "nodes_after": len(self.nodes),
                "in_sleep_mode": bool(self.config.get("structural_plasticity_in_sleep", False)),
                "disuse": True,
                "sleep_index": sleep_index,
                "params": dict(prm),
                "migrated": migrated,
                "lws_reset": mig["lws_reset"],
                "stamps_cleared": mig["stamps_cleared"],
                "downscaled": ds["synapses_scaled"],
                "downscale_clamped": ds["clamped"],
                "downscale_native": ds["native"],
                "eligible": rep.get("rule_chosen", 0),
                "shield_held": len(held_ids),
                "shield_held_ids": held_ids,
                "last_link_held": rep.get("last_link_held", 0),
                "below_threshold_after": below,
                "decided": n,
                "chunked": True,
                "chunk": dict(ch),
                "lock_holds": {"migrate": holds_a, "decide": hold_b, "remove": holds_c, "orphans": hold_d,
                               "max": max(holds_a + holds_c + [hold_b, hold_d])},
                "seconds_parts": {"migrate": t_b - t_a, "downscale": t_c - tb0, "clearance": (t_d0 - t_c) + (t_d - tc0),
                                  "orphans": t_e - td0},
                "seconds": _time.perf_counter() - t0,
            }
            self._emit("sleep_cycle", **record)
        logger.log(self._sleep_log_level(),   # [2026-10-07] observe: DEBUG on an observe shadow (sleep_observe), else INFO
                   "sleep_cycle(disuse, chunked): t=%s sleep=%d migrated=%s downscaled=%d clamped=%d eligible=%d "
                    "shield_held=%d last_link_held=%d pruned=%d nodes_collected=%d synapses %d->%d nodes %d->%d "
                    "below_wt=%d %.3fs; lock holds: %d (max %.3fs, decide %.3fs)",
                    record["timestep"], sleep_index, migrated, record["downscaled"], record["downscale_clamped"],
                    record["eligible"], record["shield_held"], record["last_link_held"], pruned, collected, syn_before,
                    record["synapses_after"], nodes_before, record["nodes_after"], below, record["seconds"],
                    len(holds_a) + len(holds_c) + 2, record["lock_holds"]["max"], hold_b)
        return record

    def _sleep_cycle_disuse(self, t0: float) -> Dict[str, Any]:
        """The P2 disuse sleep (see sleep_cycle). Validates first; one _step_lock hold for the whole cycle.
        [2026-10-07] pre-arming: with config sleep_clearance_chunk_seconds set, _sleep_cycle_disuse_chunked instead
        (bounded holds, the same decisions); absent = this one-hold path, unchanged."""
        import time as _time
        prm = self._sleep_disuse_params()            # raises before anything is touched
        ch = self._sleep_chunk_params()              # raises before anything is touched; None = not chunked
        if ch is not None:
            return self._sleep_cycle_disuse_chunked(t0, prm, ch)
        with self._step_lock:
            syn_before = len(self.synapses)
            nodes_before = len(self.nodes)
            mig = {"lws_reset": 0, "stamps_cleared": 0}
            migrated = self.config.get("sleep_low_weight_unit") != "sleeps"
            t_a = _time.perf_counter()
            if migrated:
                mig = self._sleep_migrate_counters()
            sleep_index = int(self.config.get("sleep_cycles_completed", 0)) + 1
            ds = {"synapses_scaled": 0, "clamped": 0, "native": None}
            t_b = _time.perf_counter()
            if prm["d0"] > 0.0:
                ds = self.sleep_downscale_strength_aware(prm["d0"], prm["h"])
            rep: Dict[str, Any] = {}
            t_c = _time.perf_counter()
            pruned = self._prune_synapses(report=rep, grace_sleeps=prm["G"], sleep_now=sleep_index,
                                          last_link_grace_sleeps=prm["LL"], credit_shield_kappa=prm["kappa"])
            t_d = _time.perf_counter()
            collected = self._collect_orphan_nodes()
            t_e = _time.perf_counter()
            self._total_pruned += pruned
            self.config["sleep_cycles_completed"] = sleep_index
            wt = float(self.config["weight_threshold"])
            wc = getattr(self.synapses, "weights_copy", None)
            if wc is not None:
                below = int((wc() < wt).sum())
            else:
                below = sum(1 for sid in self.synapses.keys() if self.synapses.get_weight(sid) < wt)
            held_ids = list(rep.get("shield_held_ids", []))
            record = {
                "timestep": self.timestep,
                "pruned": pruned,
                "nodes_collected": collected,
                "synapses_before": syn_before,
                "synapses_after": len(self.synapses),
                "nodes_before": nodes_before,
                "nodes_after": len(self.nodes),
                "in_sleep_mode": bool(self.config.get("structural_plasticity_in_sleep", False)),
                "disuse": True,
                "sleep_index": sleep_index,
                "params": dict(prm),
                "migrated": migrated,
                "lws_reset": mig["lws_reset"],
                "stamps_cleared": mig["stamps_cleared"],
                "downscaled": ds["synapses_scaled"],
                "downscale_clamped": ds["clamped"],
                "downscale_native": ds["native"],
                "eligible": rep.get("rule_chosen", 0),
                "shield_held": len(held_ids),
                "shield_held_ids": held_ids,
                "last_link_held": rep.get("last_link_held", 0),
                "below_threshold_after": below,
                "seconds_parts": {"migrate": t_b - t_a, "downscale": t_c - t_b, "clearance": t_d - t_c,
                                  "orphans": t_e - t_d},
                "seconds": _time.perf_counter() - t0,
            }
            self._emit("sleep_cycle", **record)
        logger.log(self._sleep_log_level(),   # [2026-10-07] observe: DEBUG on an observe shadow (sleep_observe), else INFO
                   "sleep_cycle(disuse): t=%s sleep=%d migrated=%s downscaled=%d clamped=%d eligible=%d shield_held=%d "
                    "last_link_held=%d pruned=%d nodes_collected=%d synapses %d->%d nodes %d->%d below_wt=%d %.3fs",
                    record["timestep"], sleep_index, migrated, record["downscaled"], record["downscale_clamped"],
                    record["eligible"], record["shield_held"], record["last_link_held"], pruned, collected, syn_before,
                    record["synapses_after"], nodes_before, record["nodes_after"], below, record["seconds"])
        return record

    # ------------------------------------------------------------------
    # [2026-10-07] Sleep phase P3 OBSERVE mode (spec superpowers/specs/2026-10-06-sleep-phase-design.md §8 P3: "observe mode
    # first (downscale + clearance computed and logged, nothing written)"). The observe pass runs the REAL sleep
    # (sleep_cycle, unchanged) on a private SHADOW of the sleep-relevant state, so its decisions cannot drift from the
    # real path (no predicate is re-implemented), and the live graph is only READ, in one short _step_lock hold.
    # ------------------------------------------------------------------

    _OBSERVE_BANDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf"))
    _OBSERVE_MAX_SLEEPS = 16

    def _sleep_log_level(self) -> int:
        """INFO for a real sleep; DEBUG when this graph is a sleep_observe shadow (its sleep is a projection, and an INFO
        "sleep_cycle" line in the host's log would read as a real sleep). Only the log level differs."""
        return logging.DEBUG if getattr(self, "_sleep_observe_shadow", False) else logging.INFO

    def _sleep_observe_shadow_graph(self, config_overrides: Optional[Dict[str, Any]]):
        """The observe pass's private copy of everything Graph.sleep_cycle reads or writes (caller: sleep_observe).

        ONE _step_lock hold on the live graph captures only what must be read consistently, by the cheapest copy that
        exists: the synapse store as its checkpoint bytes (native to_checkpoint_msgpack, the save path's own capture;
        every column the sleep reads: weight, salience, low_weight_steps, peak, trace, creation time, metadata stamps,
        endpoints), every node's metadata contents (copied dicts: the protection flags and the fair-chance counters the
        orphan sweep reads), the native node store's checkpoint bytes when it is on, and C-level shallow copies of the
        small maps (hyperedge membership, recent spikes, confirmation history, hyperedges, dirty sets, config, the
        fair-chance registration), plus a shallow copy of the Graph object (timestep etc.). After the hold, with the
        live lock free: the shadow synapse store is loaded from those bytes; the Node shells are copied (creation_time,
        the only other node field the sleep reads, never changes after creation) with the captured metadata and an
        EMPTY pred_weights (the sleep's only use of pred_weights is the D15 deletion on removal, which decides nothing;
        with the native node store the shadow's own store carries them);
        the adjacency is rebuilt from the shadow store's own endpoints (as restore() does; every sleep decision that
        walks it is order-independent: ties are broken by synapse id).

        The shadow has a fresh _step_lock and NO event handlers (the live handlers -- the #1051 guardian ledger, the
        vector-store drop on nodes_collected, Lenia -- never see it). Everything the sleep writes (synapse rows,
        adjacency, nodes, config, the dirty sets, the confirmation history, the fair-chance latch, _total_pruned) is the
        shadow's own; it reads the shared hyperedge-membership sets and never writes them (an orphan has none). An
        instance attribute that overrides a Graph method (bound to the live graph) is dropped from the shadow, so it
        runs the class's methods. Returns (shadow, lock_hold_seconds)."""
        import time as _time
        native_nodes = bool(getattr(self, "_native_nodes", False))
        node_cap = nodes_packed = None
        with self._step_lock:
            t = _time.perf_counter()
            syn_packed = self.synapses.to_checkpoint_msgpack()
            if native_nodes:
                nodes_packed = self.nodes.to_checkpoint_msgpack()
            else:
                node_cap = [(nid, node, dict(node.metadata) if isinstance(node.metadata, dict) else node.metadata)
                            for nid, node in self.nodes.items()]
            nhe_copy = dict(self._node_hyperedges)
            rs_copy = dict(self._recent_spikes)
            hist_copy = dict(self._synapse_confirmation_history)
            dirty = (set(self._dirty_nodes), set(self._dirty_synapses), set(self._dirty_hyperedges))
            hyperedges_copy = dict(self.hyperedges)
            cfg_copy = dict(self.config)
            fc = getattr(self, "_fair_chance_cfg", None)
            fc_copy = dict(fc) if isinstance(fc, dict) else fc
            shadow = copy.copy(self)
            hold = _time.perf_counter() - t
        store = type(self.synapses)()
        store.set_synapse_type_class(SynapseType)
        store.set_synapse_class(Synapse)
        store.bulk_load_msgpack(bytes(syn_packed))
        del syn_packed
        if native_nodes:
            nodes_copy = ng_tract.NodeStore()
            nodes_copy.set_node_class(Node)
            nodes_copy.set_ring_buffer_class(RingBuffer)
            nodes_copy.bulk_load_msgpack(bytes(nodes_packed))
            del nodes_packed                  # its own copy: the D15 pred_weights deletion writes the shadow only
        else:
            nodes_copy = {}
            for nid, node, md in node_cap:
                c = copy.copy(node)
                c.metadata = md
                c.pred_weights = {}
                nodes_copy[nid] = c
            del node_cap
        out_copy: Dict[str, Set[str]] = {nid: set() for nid in nodes_copy.keys()}
        in_copy: Dict[str, Set[str]] = {nid: set() for nid in nodes_copy.keys()}
        _triples = getattr(store, "endpoint_triples", None)
        for sid, pre_id, post_id in (_triples() if _triples is not None else _endpoint_triples_python(store)):
            out_copy.setdefault(pre_id, set()).add(sid)
            in_copy.setdefault(post_id, set()).add(sid)
        if config_overrides:
            cfg_copy.update(config_overrides)
        if cfg_copy.get("sleep_clearance_chunk_seconds") is not None:
            cfg_copy["sleep_clearance_chunk_gap_seconds"] = 0.0   # private lock: nothing waits on it, so no pause
        if isinstance(fc_copy, dict):
            fc_copy["stale_logged"] = True    # the stale WARNING belongs to the live sweep (decisions never read it)
        shadow.synapses = store
        shadow.nodes = nodes_copy
        shadow._outgoing = out_copy
        shadow._incoming = in_copy
        shadow._node_hyperedges = nhe_copy
        shadow._recent_spikes = rs_copy
        shadow._synapse_confirmation_history = hist_copy
        shadow._dirty_nodes, shadow._dirty_synapses, shadow._dirty_hyperedges = dirty
        shadow.hyperedges = hyperedges_copy
        shadow.config = cfg_copy
        if fc is not None or hasattr(self, "_fair_chance_cfg"):
            shadow._fair_chance_cfg = fc_copy
        shadow._event_handlers = {}
        shadow._step_lock = threading.RLock()
        # An INSTANCE attribute that overrides a Graph method (a host / test patch, e.g. a wrapped
        # _remove_synapse_internal) is bound to the LIVE graph: the shadow would call it and write the live graph.
        # Drop every such override on the shadow; it runs the class's own methods.
        cls_ = type(self)
        dropped = sorted(n for n in list(vars(shadow)) if callable(getattr(cls_, n, None)))
        for n in dropped:
            del vars(shadow)[n]
        shadow._sleep_observe_shadow = True
        shadow._sleep_observe_dropped_overrides = dropped
        return shadow, hold

    @classmethod
    def _observe_band_counts(cls, weights) -> List[int]:
        """Counts per weight band [0, .001), [.001, .01), [.01, .05), [.05, .1), [.1, .5), [.5, 1), [1, inf)."""
        edges = cls._OBSERVE_BANDS
        w = np.asarray(weights, dtype=np.float64)
        idx = np.searchsorted(np.asarray(edges[1:-1]), w, side="right")
        return [int(x) for x in np.bincount(idx, minlength=len(edges) - 1)]

    def sleep_observe(self, *, sleeps: Optional[int] = None, config_overrides: Optional[Dict[str, Any]] = None,
                      sample: int = 20, detail: bool = False) -> Dict[str, Any]:
        """Sleep phase P3 OBSERVE (spec §8 P3): what the sleep WOULD do from the current state, with NOTHING written.

        A private shadow of the sleep-relevant state is taken in ONE short _step_lock hold (_sleep_observe_shadow_graph);
        then the REAL Graph.sleep_cycle -- the same method, the same config-driven path (P1, or the P2 disuse sleep,
        chunked or not), the same native sweep / lifelines / D14 shield / last-link grace / orphan collection -- runs on
        the shadow, with the live _step_lock NOT held. No live weight, counter, stamp, metadata, adjacency, node, config
        key or dirty set changes; no event reaches the live handlers (no "pruned" / "nodes_collected" / "sleep_cycle":
        the #1051 ledger and the vector store never see it); the shadow's own sleep log line is DEBUG.

        sleeps: how many CONSECUTIVE sleeps to project (no wake between them, so later ones are an upper bound on
            forgetting); int 1..16. None = auto: on the disuse path, while the live counters are not yet in sleep units
            (no disuse sleep has run), sleep_weight_grace_sleeps + 1 (the migration sleep only tags; a tagged link clears
            when its count exceeds G, so this reaches the first clearance); else 1 (= exactly the next real sleep).
        config_overrides: keys applied to the SHADOW's config only (a host previewing an unarmed sleep passes the keys
            it would arm, e.g. structural_plasticity_in_sleep / sleep_disuse_enabled / the disuse parameters). The live
            config is never touched. A chunked config runs chunked on the shadow with no pause between holds.
        sample: how many of the strongest links each sleep would remove are listed (ids + numbers, never content).
        detail: also return every projected sleep's full removal order, collected ids, shield-held ids and the
            post-downscale weight of EVERY synapse (proof harnesses; large).

        Projection 1 is exactly what Graph.sleep_cycle would do now. Raises ValueError for bad arguments or when the
        shadow's sleep refuses its config (the same validation as a real sleep), with the live graph untouched.
        Returns a JSON-serialisable record; logs one INFO line (counts only).
        """
        import hashlib as _hashlib
        import time as _time
        if sleeps is not None and (isinstance(sleeps, bool) or not isinstance(sleeps, int)
                                   or not 1 <= sleeps <= self._OBSERVE_MAX_SLEEPS):
            raise ValueError("sleep_observe: sleeps must be None or an int in 1..%d (got %r)"
                             % (self._OBSERVE_MAX_SLEEPS, sleeps))
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise ValueError("sleep_observe: sample must be an int >= 0 (got %r)" % (sample,))
        if config_overrides is not None and (not isinstance(config_overrides, dict)
                                             or not all(isinstance(k, str) for k in config_overrides)):
            raise ValueError("sleep_observe: config_overrides must be a dict with str keys or None")
        t0 = _time.perf_counter()
        if getattr(self._step_lock, "_is_owned", lambda: False)():
            logger.warning("sleep_observe: called with _step_lock already held by this thread -- the caller's hold encloses "
                           "the whole projection")
        shadow, hold = self._sleep_observe_shadow_graph(config_overrides)
        t_shadow = _time.perf_counter() - t0
        cfg = shadow.config
        disuse = bool(cfg.get("sleep_disuse_enabled", False))
        migrated_before = cfg.get("sleep_low_weight_unit") == "sleeps"
        if disuse:
            prm = shadow._sleep_disuse_params()          # the real validation; raises before any projection
            shadow._sleep_chunk_params()
        if sleeps is None:
            sleeps = (int(prm["G"]) + 1) if (disuse and not migrated_before) else 1
            sleeps = min(sleeps, self._OBSERVE_MAX_SLEEPS)

        removed_log: List[Tuple[str, str, str, float, float, float]] = []
        _real_remove = Graph._remove_synapse_internal

        def _capture_remove(synapse_id: str, _g=shadow) -> None:
            syn = _g.synapses.get(synapse_id)
            if syn is not None:
                removed_log.append((synapse_id, syn.pre_node_id, syn.post_node_id, float(syn.weight),
                                    float(syn.peak_weight), float(syn.eligibility_trace)))
            _real_remove(_g, synapse_id)

        shadow._remove_synapse_internal = _capture_remove    # an instance attribute on the SHADOW only: records, then
        #                                                      the unchanged removal function runs
        protected = shadow._strength_protected_ids()
        prot_set = set(protected)

        def _degrees():
            return {n: (len(shadow._outgoing.get(n, ())), len(shadow._incoming.get(n, ()))) for n in protected}

        def _lifelines():
            ll = shadow._protected_lifelines(protected)
            out: Dict[str, List[str]] = {n: [] for n in protected}
            for sid in sorted(ll):
                s = shadow.synapses[sid]
                for end in (s.pre_node_id, s.post_node_id):
                    if end in out and sid not in out[end]:
                        out[end].append(sid)
            return ll, out

        def _weights():
            keys = list(shadow.synapses.keys())
            wc = getattr(shadow.synapses, "weights_copy", None)
            w = wc() if wc is not None else np.asarray([shadow.synapses.get_weight(s) for s in keys], dtype=np.float64)
            return keys, np.asarray(w, dtype=np.float64)

        def _lws_counts():
            c: Dict[str, int] = {}
            for _sid, _s in shadow.synapses.items():
                v = _s.low_weight_steps
                if v:
                    c[str(v)] = c.get(str(v), 0) + 1
            return dict(sorted(c.items(), key=lambda kv: int(kv[0])))

        def _established(keys):
            return sum(1 for s in keys if shadow.synapses[s].peak_weight >= 0.5)

        ll0, ll0_by = _lifelines()
        deg0 = _degrees()
        keys0, w0 = _weights()
        start = {"timestep": shadow.timestep, "synapses": len(keys0),
                 "nodes": len(shadow.nodes), "weight_bands": self._observe_band_counts(w0),
                 "established_peak_ge_0_5": _established(keys0), "w_ge_0_5": int((w0 >= 0.5).sum()),
                 "migrated_before": migrated_before}
        projections: List[Dict[str, Any]] = []
        digest = _hashlib.sha256()
        for k in range(1, sleeps + 1):
            keys_b, w_b = _weights()
            w_before = dict(zip(keys_b, w_b.tolist()))
            nodes_before = set(shadow.nodes.keys())
            ll_b, ll_b_by = _lifelines()
            deg_b = _degrees()
            del removed_log[:]
            ts = _time.perf_counter()
            rec = shadow.sleep_cycle()
            secs = _time.perf_counter() - ts
            keys_a, w_a = _weights()
            ll_a, ll_a_by = _lifelines()
            deg_a = _degrees()
            collected = sorted(nodes_before - set(shadow.nodes.keys()))
            removed = list(removed_log)
            rem_wb = [w_before.get(r[0], float("nan")) for r in removed]
            strongest = sorted(zip(removed, rem_wb), key=lambda x: (-x[1], -x[0][4], x[0][0]))[:sample]
            held_ids = list(rec.get("shield_held_ids", []) or [])
            proj = {
                "projection": k,
                "record": {kk: vv for kk, vv in rec.items()
                           if kk not in ("shield_held_ids", "lock_holds", "seconds_parts", "chunk")},
                "seconds": secs,
                "would_remove": len(removed),
                "would_collect": len(collected),
                "collected_node_ids": collected[:500],
                "removed_by_weight_band_before": self._observe_band_counts(rem_wb) if removed else
                    [0] * (len(self._OBSERVE_BANDS) - 1),
                "removed_by_weight_band_at_removal": self._observe_band_counts([r[3] for r in removed]) if removed else
                    [0] * (len(self._OBSERVE_BANDS) - 1),
                "removed_by_peak_band": self._observe_band_counts([r[4] for r in removed]) if removed else
                    [0] * (len(self._OBSERVE_BANDS) - 1),
                "removed_peak_ge_0_5": sum(1 for r in removed if r[4] >= 0.5),
                "removed_w_before_ge_0_5": sum(1 for x in rem_wb if x >= 0.5),
                "removed_touching_protected": sum(1 for r in removed if r[1] in prot_set or r[2] in prot_set),
                "weight_bands_before": self._observe_band_counts(w_b),
                "weight_bands_after": self._observe_band_counts(w_a),
                "established_peak_ge_0_5_after": _established(keys_a),
                # weak-link counters after this sleep (disuse: sleeps spent below weight_threshold; a link clears once
                # its count exceeds G): {count: synapses}, zero left out
                "low_weight_counts_after": _lws_counts() if disuse else None,
                "shield_held": len(held_ids),
                "shield_held_sample": held_ids[:sample],
                "strongest_removed": [{"synapse_id": r[0], "pre": r[1], "post": r[2], "w_before": wb,
                                       "w_at_removal": r[3], "peak": r[4], "trace": r[5]} for r, wb in strongest],
                "protected": [{"node_id": n, "out_before": deg_b[n][0], "in_before": deg_b[n][1],
                               "out_after": deg_a[n][0], "in_after": deg_a[n][1],
                               "lifelines_before": ll_b_by[n], "lifelines_after": ll_a_by[n],
                               "lifelines_intact": all(s in shadow.synapses for s in ll_b_by[n])} for n in protected],
                "lifelines_removed": sorted(s for s in ll_b if s not in shadow.synapses),
            }
            digest.update(repr((k, [r[0] for r in removed], collected)).encode())
            if detail:
                post = dict(zip(keys_a, w_a.tolist()))
                for r in removed:
                    post[r[0]] = r[3]
                proj["removed_ids"] = [r[0] for r in removed]
                proj["shield_held_ids"] = held_ids
                proj["collected_ids_all"] = collected
                proj["post_downscale_weights"] = post
            projections.append(proj)
        first = projections[0]
        record = {
            "observe": True,
            "timestep": shadow.timestep,
            "sleeps_projected": sleeps,
            "no_wake_between_projections": True,
            "path": ("disuse" if disuse else "p1"),
            "chunked": cfg.get("sleep_clearance_chunk_seconds") is not None if disuse else False,
            "params": (dict(prm) if disuse else None),
            "config_overrides": sorted(config_overrides) if config_overrides else [],
            "instance_overrides_not_used": list(shadow._sleep_observe_dropped_overrides),
            "start": start,
            "protected_nodes": len(protected),
            "lifelines_at_start": len(ll0),
            "projections": projections,
            "would_remove_total": sum(p["would_remove"] for p in projections),
            "would_collect_total": sum(p["would_collect"] for p in projections),
            "first_clearance": next((p["projection"] for p in projections if p["would_remove"]), None),
            "removal_digest": digest.hexdigest(),
            "lock_holds": {"snapshot": hold, "max": hold, "count": 1},
            "seconds_parts": {"shadow": t_shadow, "projections": sum(p["seconds"] for p in projections)},
            "seconds": _time.perf_counter() - t0,
        }
        del shadow._remove_synapse_internal    # break the shadow <-> wrapper cycle so the copy is freed now
        del shadow
        logger.info("sleep_observe: t=%s projected %d sleep(s) (%s, nothing written): next sleep would remove %d, collect %d, "
                    "shield %d; over the projection remove %d (peak>=0.5: %d), collect %d, first clearance at projection "
                    "%s; protected lifelines removed %d; lock hold %.3fs, %.2fs total",
                    record["timestep"], sleeps, record["path"], first["would_remove"], first["would_collect"],
                    first["shield_held"], record["would_remove_total"],
                    sum(p["removed_peak_ge_0_5"] for p in projections), record["would_collect_total"],
                    record["first_clearance"], sum(len(p["lifelines_removed"]) for p in projections), hold,
                    record["seconds"])
        return record

    def _prune_synapses(
        self,
        *,
        competing_ids: Optional[Any] = None,
        excluded_ids: Optional[Any] = None,
        max_removals: Optional[int] = None,
        order_key: Optional[Any] = None,
        report: Optional[Dict[str, Any]] = None,
        grace_sleeps: Optional[int] = None,
        sleep_now: Optional[int] = None,
        last_link_grace_sleeps: Optional[int] = None,
        credit_shield_kappa: Optional[float] = None,
        defer_removal: bool = False,
    ) -> int:
        """Prune weak/inactive synapses (PRD §3.3.1).

        Rules:
            Weight-based: weight < threshold for > grace_period steps → remove.
            Activity-based: unused for > inactivity_threshold steps → remove.
            Age-based: age > grace_period AND peak_weight < 2× initial → remove.
        [2026-10-06] The activity and age rules (step deadlines) are RETIRING: punchlist #1049, spec
        superpowers/specs/2026-10-06-sleep-phase-design.md §6 (D2, D9). Unchanged until P4.

        Keyword-only parameters (want-hub (d), plan-005 §4.2) — ALL default None, and with
        all of them None this is exactly the function above (both wake-time callers):
            competing_ids / excluded_ids: supplied TOGETHER, by the caller, as synapse ids
                (membership is never inferred here). Competing mode: the exemption for
                identity-protected endpoints is lifted for exactly these ids and the loop
                visits ONLY these ids. Refused (ValueError, before anything is touched) if
                an id is also excluded, is absent, lacks an endpoint node, or touches a
                constitutional node.
            max_removals: at most this many of the function's own eligible list are removed
                (int >= 1). REQUIRED in competing mode.
            order_key: mapping synapse_id -> tuple of numbers/strings; the eligible list is sorted by it
                BEFORE truncation. COMPETING MODE ONLY: REQUIRED there, with an entry for every id, every
                entry a tuple and all entries of one shape (same length, same number-or-string kind per
                position) so the sort can never raise; REFUSED (ValueError) on the default path.
            report: dict; filled with report['eligible'] (count before truncation) and
                report['removed_ids'] (list, removal order).

        Config prune_protected_faint_links (2026-10-04 prune-lifeline, Josh ruling; read live, absent = False, NOT
        in DEFAULT_CONFIG): on the DEFAULT path only, a synapse touching an identity-protected node is exempt only
        if it is a LIFELINE (see _protected_lifelines) of either endpoint; all others face the rules above. #92's
        guarantee becomes "no protected node is ever cut off" instead of "no protected link is ever pruned".
        Competing mode is unaffected (its caller supplies the sets; compete_protected_links adds the lifelines
        to its guaranteed set when the flag is on).

        [2026-10-07] Sleep-unit clearance (sleep phase P2, spec 2026-10-06 §3.3, D2-D4, D14) — keyword-only, ALL default
        None; the ONLY caller is sleep_cycle's disuse path. grace_sleeps / sleep_now / last_link_grace_sleeps are given
        TOGETHER (credit_shield_kappa with them), never in competing mode:
            grace_sleeps: the weight rule's grace, counted in SLEEPS (low_weight_steps advances once per call, i.e.
                once per sleep); the activity clause is off (inactivity = inf) and the age clause is off
                (initial_w = 0, so peak_weight < 0 is never true) — by argument, the rules themselves unchanged.
            sleep_now / last_link_grace_sleeps: the last-link fair chance on the sleep clock, stamped under metadata
                "last_link_since_sleep" (step stamps "last_link_since" are not read).
            credit_shield_kappa (D14): a rule-chosen id is NOT removed while w + kappa * max(trace, 0) >= wt
                (pending reward credit could still lift it); it keeps its count and is re-tested next sleep. Applied
                after the lifeline filter, before the last-link grace. 0 = no shield. report['shield_held_ids'].
        Every default-path call (no sleep-unit arguments) drops config 'sleep_low_weight_unit' if present: the counter
        is then no longer in sleep units, so the next disuse sleep re-migrates it (sleep_cycle).

        [2026-10-07] pre-arming — defer_removal (keyword-only, default False; sleep-unit clearance ONLY, the chunked disuse
        sleep's decision hold): everything above runs exactly as without it (counters, lifelines, shield, last-link
        stamps), but the chosen ids are NOT removed and no "pruned" event is emitted here: they are returned in
        report['deferred_ids'] (report REQUIRED) for the caller to remove; the return value is then 0 (nothing removed).
        """
        wt = self.config["weight_threshold"]
        grace = self.config["grace_period"]
        inactivity = self.config["inactivity_threshold"]
        initial_w = self.config["initial_sprouting_weight"]

        competing_mode = competing_ids is not None
        if competing_mode != (excluded_ids is not None):
            raise ValueError("_prune_synapses: competing_ids and excluded_ids must be supplied together")
        sleep_units = grace_sleeps is not None
        _su_args = (grace_sleeps, sleep_now, last_link_grace_sleeps, credit_shield_kappa)
        if not sleep_units and (any(a is not None for a in _su_args) or defer_removal):
            raise ValueError("_prune_synapses: sleep_now / last_link_grace_sleeps / credit_shield_kappa / defer_removal "
                             "need grace_sleeps")
        if sleep_units:
            if competing_mode:
                raise ValueError("_prune_synapses: the sleep-unit clearance is not valid in competing mode")
            if order_key is not None or max_removals is not None:
                raise ValueError("_prune_synapses: the sleep-unit clearance takes no order_key / max_removals")
            if defer_removal and not isinstance(report, dict):
                raise ValueError("_prune_synapses: defer_removal needs a report dict (the deferred ids go there)")
            for _lbl, _v in (("grace_sleeps", grace_sleeps), ("sleep_now", sleep_now),
                             ("last_link_grace_sleeps", last_link_grace_sleeps)):
                if isinstance(_v, bool) or not isinstance(_v, int) or _v < 0:
                    raise ValueError("_prune_synapses: %s must be an int >= 0 (got %r)" % (_lbl, _v))
            if credit_shield_kappa is None or isinstance(credit_shield_kappa, bool) or \
                    not isinstance(credit_shield_kappa, (int, float)) or not (0.0 <= float(credit_shield_kappa) < float("inf")):
                raise ValueError("_prune_synapses: credit_shield_kappa must be a finite number >= 0 (got %r)"
                                 % (credit_shield_kappa,))
            grace = grace_sleeps
            inactivity = float("inf")      # D2/§6: the activity clause off, by argument
            initial_w = 0.0                # D2: the age clause off (peak_weight < 0.0 is never true)
            _ll_key = "last_link_since_sleep"
        else:
            _ll_key = "last_link_since"
            if not competing_mode and "sleep_low_weight_unit" in self.config:
                del self.config["sleep_low_weight_unit"]   # counters advance per step again: no longer sleep units
        if max_removals is not None and (
                isinstance(max_removals, bool) or not isinstance(max_removals, int) or max_removals < 1):
            raise ValueError("_prune_synapses: max_removals must be an int >= 1 (got %r)" % (max_removals,))
        if report is not None and not isinstance(report, dict):
            raise ValueError("_prune_synapses: report must be a dict or None (got %s)" % type(report).__name__)
        if not competing_mode and order_key is not None:
            # No caller uses this, and it could only fail LATE (an entry missing for an eligible id) after the predicates
            # below had already advanced low_weight_steps: refuse it up front so a refusal mutates nothing.
            raise ValueError("_prune_synapses: order_key is only valid in competing mode (competing_ids/excluded_ids)")
        if competing_mode:
            if max_removals is None:
                raise ValueError("_prune_synapses: max_removals is required in competing mode")
            if order_key is None:
                raise ValueError("_prune_synapses: order_key is required in competing mode")
            try:
                competing = sorted(set(competing_ids))
            except TypeError as exc:
                raise ValueError("_prune_synapses: competing_ids must be comparable synapse ids (%s)" % exc) from exc
            excluded = set(excluded_ids)
            key_kinds = None     # the ONE shape every order_key entry must have (kind per position: 1 number, 2 string)
            for sid in competing:
                if sid in excluded:
                    raise ValueError("_prune_synapses: competing synapse %r is also excluded" % (sid,))
                if sid not in self.synapses:
                    raise ValueError("_prune_synapses: competing synapse %r does not exist" % (sid,))
                csyn = self.synapses[sid]
                for nid in (csyn.pre_node_id, csyn.post_node_id):
                    node = self.nodes.get(nid)
                    if node is None:
                        raise ValueError("_prune_synapses: competing synapse %r has a missing endpoint node %r" % (sid, nid))
                    if (node.metadata or {}).get("constitutional"):
                        raise ValueError("_prune_synapses: competing synapse %r touches constitutional node %r" % (sid, nid))
                try:
                    okey = order_key[sid]
                except KeyError:
                    raise ValueError("_prune_synapses: order_key has no entry for competing synapse %r" % (sid,)) from None
                if not isinstance(okey, tuple):
                    raise ValueError("_prune_synapses: order_key entry for %r must be a tuple (got %s)" % (sid, type(okey).__name__))
                kinds = tuple(1 if isinstance(x, (int, float)) else 2 if isinstance(x, str) else 0 for x in okey)
                if 0 in kinds:
                    raise ValueError("_prune_synapses: order_key entry for %r may hold only numbers and strings" % (sid,))
                if key_kinds is None:
                    key_kinds = kinds
                elif kinds != key_kinds:
                    raise ValueError("_prune_synapses: order_key entries are not mutually comparable "
                                     "(%r has element kinds %r, expected %r)" % (sid, kinds, key_kinds))

        # [2026-10-05] the default path runs the rules in ONE native column sweep when the installed
        # ng_tract has it (below); competing mode (a few hundred ids, once per dream cycle) and an
        # ng_tract without it keep the per-SynapseRef loop.
        _native_prune = None if competing_mode else getattr(
            self.synapses, "advance_low_weight_and_collect_prune", None)
        if competing_mode:
            candidates = ((sid, self.synapses[sid]) for sid in competing)
        elif _native_prune is None:
            candidates = self.synapses.items()

        # 2026-10-04 prune-lifeline (Josh: "protect existence, not unlimited wiring"). Default path ONLY, and ONLY
        # when config prune_protected_faint_links is set (absent key = False = the #92 blanket skip below, unchanged):
        # the exemption for identity-protected endpoints narrows to each protected node's LIFELINES (its single
        # strongest outgoing and single strongest incoming link, computed ONCE here, fail-closed probe). Every other
        # synapse touching a protected node goes through the three rules below like any other synapse.
        lifelines: Optional[Set[str]] = None
        protected_set: Set[str] = set()
        stamped: Dict[str, Any] = {}          # 2026-10-04 last-link grace: sid -> last_link_since, collected in the loop
        if not competing_mode and self.config.get("prune_protected_faint_links", False):
            _prot = self._strength_protected_ids()
            protected_set = set(_prot)
            lifelines = self._protected_lifelines(_prot)

        to_prune: List[str] = []
        if _native_prune is not None:
            # [2026-10-05] Native rule pass (one column sweep, row order == items() order) — the SAME three rules
            # as the loop below:  weight < wt -> low_weight_steps += 1, prune once it exceeds grace (else
            # low_weight_steps = 0);  inactive_steps > inactivity * salience -> prune;  age > grace and
            # peak_weight < 2 * initial_w -> prune.  It advances the counters and returns the ids; removal stays here.
            if lifelines is None:
                # Cricket rim (#92): never prune synapses touching identity-protected nodes.
                # _is_identity_protected stays the ONE authority; asked once per node here instead of twice per
                # synapse (a non-node id was never protected, and the probe is a pure read).
                protected = [nid for nid in self.nodes if self._is_identity_protected(nid)]
                to_prune = _native_prune(self.timestep, wt, grace, inactivity, initial_w, protected)
            else:
                # prune-lifeline ON: no blanket skip; only the LIFELINES are exempt. The native sweep runs on every
                # row (no protected nodes), then each lifeline gets back the low_weight_steps it had before (the loop
                # below never touches a lifeline) and is dropped from the result (order of the rest unchanged).
                # The last-link stamps have no native accessor: read them from the metadata exactly as the loop
                # below does (same per-synapse metadata access).
                for sid, syn in self.synapses.items():
                    _md = syn.metadata
                    if _md and _ll_key in _md:
                        stamped[sid] = _md[_ll_key]
                _ll_steps = [(sid, self.synapses[sid].low_weight_steps) for sid in sorted(lifelines)]
                to_prune = _native_prune(self.timestep, wt, grace, inactivity, initial_w, ())
                for sid, _lws in _ll_steps:
                    self.synapses[sid].low_weight_steps = _lws
                to_prune = [sid for sid in to_prune if sid not in lifelines]
        for sid, syn in (candidates if _native_prune is None else ()):
            if lifelines is not None:
                _md = syn.metadata
                if _md and _ll_key in _md:
                    stamped[sid] = _md[_ll_key]
                if sid in lifelines:
                    continue
            # Cricket rim (#92): never prune synapses touching identity-protected nodes.
            # Protected nodes survive orphan collection but were being silenced here.
            # (Competing mode lifts this for the caller's competing ids ONLY.)
            elif not competing_mode and (self._is_identity_protected(syn.pre_node_id) or
                    self._is_identity_protected(syn.post_node_id)):
                continue

            age = self.timestep - syn.creation_time

            # Weight-based pruning
            if syn.weight < wt:
                syn.low_weight_steps += 1
                if syn.low_weight_steps > grace:
                    to_prune.append(sid)
                    continue
            else:
                syn.low_weight_steps = 0

            # Activity-based pruning — salience armor multiplies effective threshold.
            # Surprise-tagged synapses can sit dormant longer before being culled.
            effective_inactivity = inactivity * syn.salience
            if syn.inactive_steps > effective_inactivity:
                to_prune.append(sid)
                continue

            # Age-based pruning: speculative connections that never strengthened
            if age > grace and syn.peak_weight < 2.0 * initial_w:
                to_prune.append(sid)

        if sleep_units and report is not None:
            report["rule_chosen"] = len(to_prune)      # after the lifeline filter, before the shield / last-link grace
        if sleep_units and credit_shield_kappa > 0.0 and to_prune:
            # D14 pending-credit shield: an open positive eligibility trace could still commit w + kappa*trace.
            _k = float(credit_shield_kappa)
            _held: List[str] = []
            _kept: List[str] = []
            for sid in to_prune:
                _syn = self.synapses[sid]
                _tr = _syn.eligibility_trace
                if _syn.weight + _k * (_tr if _tr > 0.0 else 0.0) >= wt:
                    _held.append(sid)
                else:
                    _kept.append(sid)
            to_prune = _kept
            if report is not None:
                report["shield_held_ids"] = _held
        elif sleep_units and report is not None:
            report["shield_held_ids"] = []

        if lifelines is not None:
            if sleep_units:
                to_prune = self._last_link_grace(to_prune, protected_set, stamped, report, now=sleep_now,
                                                 grace=last_link_grace_sleeps, stamp_key=_ll_key)
            else:
                to_prune = self._last_link_grace(to_prune, protected_set, stamped, report)

        # want-hub (d): report / order / budget — each ONLY when its parameter is given (default path: none of these run).
        if report is not None:
            report["eligible"] = len(to_prune)
        if order_key is not None:
            # competing mode only (refused otherwise above); coverage and sortability were proven BEFORE the loop
            to_prune.sort(key=lambda s: order_key[s])
        if max_removals is not None:
            to_prune = to_prune[:max_removals]

        if defer_removal:      # [2026-10-07] pre-arming: the chunked disuse sleep removes these itself (sleep-unit only)
            report["deferred_ids"] = list(to_prune)
            report["removed_ids"] = []
            return 0

        for sid in to_prune:
            self._remove_synapse_internal(sid)

        if to_prune:
            self._emit("pruned", count=len(to_prune), timestep=self.timestep)

        if report is not None:
            report["removed_ids"] = list(to_prune)

        return len(to_prune)

    def compete_protected_links(self, topk: int, budget: int) -> Dict[str, Any]:
        """Dream-time competition among the links of the authored wants (want-hub (d), plan-005 §4.3).

        Called ONLY by a dream loop that owns the cycle clock and reads K and B from its own env
        (LAW 5) — there is no config key and no default here (plan-005 §3). One call = one dream
        cycle = EXACTLY ONE _prune_synapses call (a second call would advance every competitor's
        low_weight_steps a second time, §4.4). This method contains NO copy of the prune predicates,
        NO removal loop and NO _remove_synapse_internal call: eligibility and removal stay inside
        _prune_synapses; everything below is scoping the caller supplies.

        Sets (plan-005 §2.1-2.3, §4A.3), recomputed from current values every call:
            F           every synapse touching a constitutional node (the frozen rim) — never read for
                        weight, never written, never competing.
            G           for each protected non-constitutional node ("want"), its `topk` strongest non-F
                        OUTGOING and `topk` strongest non-F INCOMING synapses, ranked weight desc ->
                        peak_weight desc -> inactive_steps asc -> synapse_id asc. Never competing.
            last-link   for each UNPROTECTED partner whose every incident synapse would compete, the
                        strongest one (same ranking). Conservative: decided WITHOUT eligibility. Never
                        competing — so no partner node is left with zero synapses by this pass.
            competing   (non-F synapses touching a want) - G - last-link.
        Order (Exec P409, §4A.2): the static HEIGHT key — per want, its competing links stalest-first
        (inactive_steps desc, weight asc, id asc) get rank r, height = c_w - r; a want<->want link takes
        the LARGER of its two endpoint heights; key = (-height, -inactive_steps, weight, synapse_id).

        Returns (and logs at INFO, even when 0 are removed) the counts record of plan-005 §4A.6. A
        want<->want removal is tallied under BOTH wants, so by_want can sum to more than `removed`.
        Raises ValueError for topk/budget not an int >= 1, before touching anything.
        """
        for label, val in (("topk", topk), ("budget", budget)):
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError("compete_protected_links: %s must be an int >= 1 (got %r)" % (label, val))
        # [2026-10-07] #1066 (pre-arming): this competition counts in STEP units — its _prune_synapses call advances
        # low_weight_steps against grace_period (steps) and its last-link pick reads the step stamp "last_link_since".
        # Under the disuse sleep that counter is in SLEEPS (advanced once per sleep by the clearance) and the stamps are
        # "last_link_since_sleep", so running both would count a competing link twice per sleep and against the wrong
        # unit. Until the competition is ported to sleep units (its own design + ruling), it REFUSES while disuse is on
        # (config sleep_disuse_enabled truthy) or the counters are marked as sleeps (sleep_low_weight_unit == "sleeps").
        # Both keys absent = unchanged.
        if self.config.get("sleep_disuse_enabled", False) or self.config.get("sleep_low_weight_unit") == "sleeps":
            raise ValueError("compete_protected_links: refused while the disuse sleep owns low_weight_steps (counted in "
                             "sleeps; this competition counts steps) -- punchlist #1066")

        def _rank(sid):
            s = self.synapses[sid]
            return (-s.weight, -s.peak_weight, s.inactive_steps, sid)

        def _plan():
            protected = [n for n in self.nodes if self._is_identity_protected(n)]
            const = {n for n in protected if (self.nodes[n].metadata or {}).get("constitutional")}
            wants = sorted(n for n in protected if n not in const)
            want_set = set(wants)
            F: Set[str] = set()
            for n in const:
                F.update(self._outgoing.get(n, ()))
                F.update(self._incoming.get(n, ()))
            arena: Set[str] = set()
            for w in wants:
                arena.update(self._outgoing.get(w, ()))
                arena.update(self._incoming.get(w, ()))
            arena -= F
            G: Set[str] = set()
            need: Dict[Tuple[str, str], int] = {}
            for w in wants:
                for d, idx in (("out", self._outgoing), ("in", self._incoming)):
                    ids = [sid for sid in idx.get(w, ()) if sid not in F]
                    G.update(sorted(ids, key=_rank)[:topk])
                    need[(w, d)] = min(topk, len(ids))
            if self.config.get("prune_protected_faint_links", False):
                # prune-lifeline: the wake-path lifelines are never competitors either (same set, same tie-break),
                # so both paths agree on which link keeps each protected node attached. Flag off: no change.
                G.update(self._protected_lifelines() & arena)
            competing0 = arena - G
            partners: Set[str] = set()
            for sid in competing0:
                s = self.synapses[sid]
                for nid in (s.pre_node_id, s.post_node_id):
                    if not self._is_identity_protected(nid):
                        partners.add(nid)
            last: Set[str] = set()
            if self.config.get("prune_protected_faint_links", False):
                # last-link grace (turn 2): hold the link the wake prune is already graceing, if any, so both paths
                # agree on WHICH link is the node's last; otherwise the same strongest pick as before.
                def _last_key(sid):
                    md = self.synapses[sid].metadata
                    return (0 if (md and "last_link_since" in md) else 1,) + _rank(sid)
            else:
                _last_key = _rank
            for nid in partners:
                inc = set(self._outgoing.get(nid, ())) | set(self._incoming.get(nid, ()))
                if inc and inc <= competing0:
                    last.add(min(inc, key=_last_key))
            competing = competing0 - last
            by_want: Dict[str, List[str]] = {}
            for sid in competing:
                s = self.synapses[sid]
                for nid in {s.pre_node_id, s.post_node_id}:
                    if nid in want_set:
                        by_want.setdefault(nid, []).append(sid)
            height: Dict[str, int] = {}
            for ids in by_want.values():
                ids.sort(key=lambda x: (-self.synapses[x].inactive_steps, self.synapses[x].weight, x))
                c = len(ids)
                for r, sid in enumerate(ids):
                    height[sid] = max(height.get(sid, c - r), c - r)
            order_key = {}
            for sid in competing:
                s = self.synapses[sid]
                order_key[sid] = (-height[sid], -s.inactive_steps, s.weight, sid)
            return {"F": F, "G": G, "last": last, "competing": competing, "excluded": F | G | last,
                    "order_key": order_key, "wants": wants, "need": need, "protected": len(protected)}

        with self._step_lock:
            plan = _plan()
            again = _plan()
            if ((plan["competing"], plan["excluded"], plan["order_key"])
                    != (again["competing"], again["excluded"], again["order_key"])):
                raise RuntimeError("compete_protected_links: the competing set / order key build is not deterministic — refusing")
            F, G = plan["F"], plan["G"]
            want_set = set(plan["wants"])

            # Guaranteed floor, asserted BEFORE the call (impossible to break by construction: G is excluded).
            for (w, d), need in plan["need"].items():
                idx = self._outgoing if d == "out" else self._incoming
                if len([x for x in idx.get(w, ()) if x in G]) < need:
                    raise RuntimeError("compete_protected_links: guaranteed floor for %s/%s would end below %d — refusing" % (w, d, need))

            wt = self.config["weight_threshold"]
            captured: Dict[str, Tuple[str, str, bool]] = {}
            for sid in plan["competing"]:
                s = self.synapses[sid]
                captured[sid] = (s.pre_node_id, s.post_node_id, s.weight >= wt)   # removed synapses no longer exist afterwards

            rep: Dict[str, Any] = {}
            self._prune_synapses(competing_ids=plan["competing"], excluded_ids=plan["excluded"],
                                 max_removals=budget, order_key=plan["order_key"], report=rep)

            removed_ids = rep["removed_ids"]
            tally: Dict[str, List[int]] = {}
            conducting = 0
            for sid in removed_ids:
                pre, post, cond = captured[sid]
                if cond:
                    conducting += 1
                for nid in {pre, post}:
                    if nid in want_set:
                        t = tally.setdefault(nid, [0, 0])
                        t[0] += 1
                        if cond:
                            t[1] += 1
            floors_ok = True
            for (w, d), need in plan["need"].items():
                idx = self._outgoing if d == "out" else self._incoming
                if len([x for x in idx.get(w, ()) if x not in F]) < need:
                    floors_ok = False
            record = {
                "timestep": self.timestep, "K_in": topk, "K_out": topk, "B": budget,
                "eligible": rep["eligible"], "removed": len(removed_ids),
                "conducting_links_removed": conducting, "held_back_last_link": len(plan["last"]),
                "floors_ok": floors_ok, "F_links": len(F), "protected_nodes": plan["protected"],
                "wants_with_removals": len(tally), "wants_zero": len(want_set) - len(tally),
                "by_want": {w: {"removed": t[0], "conducting": t[1]} for w, t in sorted(tally.items())},
            }
        logger.log(
            logging.INFO if floors_ok else logging.WARNING,
            "compete_protected_links: t=%s K=%d+%d B=%d eligible=%d removed=%d conducting=%d held_back_last_link=%d "
            "floors_ok=%s F_links=%d protected_nodes=%d wants_with_removals=%d wants_zero=%d by_want[removed(conducting)]: %s",
            record["timestep"], topk, topk, budget, record["eligible"], record["removed"], conducting,
            record["held_back_last_link"], floors_ok, record["F_links"], record["protected_nodes"],
            record["wants_with_removals"], record["wants_zero"],
            "; ".join("%s=%d(%d)" % (w, v["removed"], v["conducting"]) for w, v in record["by_want"].items()) or "-",
        )
        return record

    # ------------------------------------------------------------------
    # 2026-10-04 strength budget + sleep downscaling (spec "Bounding synapse growth by
    # competition, not caps"). Both scale weights DOWN only and never prune: what falls
    # under weight_threshold is left to the existing prune rules on their normal clock.
    # ------------------------------------------------------------------

    def _strength_protected_ids(self) -> List[str]:
        """Every identity-protected node id (constitutional INCLUDED — Josh ruling (a)), sorted.
        Fail closed: a node whose protection probe raises is treated as protected."""
        out: List[str] = []
        for nid in list(self.nodes):
            try:
                prot = self._is_identity_protected(nid)
            except Exception:
                prot = True
            if prot:
                out.append(nid)
        out.sort()
        return out

    def _protected_lifelines(self, protected_ids: Optional[List[str]] = None) -> Set[str]:
        """2026-10-04 prune-lifeline: the synapse ids that keep each identity-protected node attached.

        For every protected node (constitutional INCLUDED, Josh ruling (a); fail-closed probe via
        _strength_protected_ids), its single strongest OUTGOING and single strongest INCOMING synapse,
        ranked weight desc then synapse_id asc — the SAME ranking (same helper) as the strength budget's
        strongest-link guarantee, applied to the CURRENT weights (a budget pass's per-target IN scaling can
        reorder a node's out-links, so the lifeline follows whichever link is strongest now). A direction
        with no links contributes nothing. One pass over the protected nodes' in/out sets; weights are
        read through the native SynapseStore.get_weight when present (no per-synapse Ref object), else
        through the Python Synapse object. A pure query: writes nothing. protected_ids: an already computed
        _strength_protected_ids() result (the prune pass computes it once and reuses it); None = compute here.
        """
        gw = getattr(self.synapses, "get_weight", None)
        if gw is None:
            gw = lambda s: self.synapses[s].weight  # noqa: E731
        if protected_ids is None:
            protected_ids = self._strength_protected_ids()
        return set(_strength_guard_targets(gw, self._outgoing, self._incoming, protected_ids, 0.0))

    def _last_link_grace(self, to_prune: List[str], protected: Set[str], stamped: Dict[str, Any],
                         report: Optional[Dict[str, Any]], *, now: Optional[int] = None,
                         grace: Optional[int] = None, stamp_key: str = "last_link_since") -> List[str]:
        """2026-10-04 prune-lifeline turn 2 — the last-link fair chance (Josh ruling). Called ONLY by the default
        _prune_synapses path with prune_protected_faint_links on, after the three rules chose `to_prune`.

        For every NON-protected endpoint of a synapse in `to_prune` that the pass would leave with ZERO synapses
        (in + out, as a set of ids), in sorted node-id order and re-checked live so one held link serves both of its
        endpoints: its last link = a stamped link first, else the strongest (weight desc, synapse_id asc).
            no stamp                                  -> stamp metadata last_link_since = timestep; hold it
            timestep - since < last_link_grace_steps  -> hold it
            otherwise                                 -> removed by the normal rules like any other link
        Every surviving stamp none of whose non-protected endpoints is left with <= 1 synapse is cleared (the node
        has wired elsewhere; a later last-link episode starts a fresh grace). Returns the reduced removal list.
        report['last_link_held'] / ['last_link_stamped'] / ['last_link_expired'] / ['last_link_cleared'] when a
        report dict is given. Grace: config last_link_grace_steps, read live, absent = 2000 (NOT in DEFAULT_CONFIG);
        <= 0 disables the hold entirely (returns `to_prune` unchanged, no report keys, nothing stamped or cleared).

        [2026-10-07] sleep phase P2 (D4): keyword-only `now` / `grace` / `stamp_key`, all defaulting to the step clock
        above (timestep, last_link_grace_steps, "last_link_since"). The sleep-unit clearance passes its sleep index,
        the grace in SLEEPS and the stamp key "last_link_since_sleep", so step stamps and sleep stamps never mix.
        """
        if grace is None:
            grace = self.config.get("last_link_grace_steps", 2000)
        if not grace or grace <= 0:
            return to_prune          # grace 0 = no fair chance: the normal rules alone (nothing stamped or cleared)
        if now is None:
            now = self.timestep
        removing = set(to_prune)
        out_i, in_i = self._outgoing, self._incoming

        def _inc(n):
            return set(out_i.get(n, ())) | set(in_i.get(n, ()))

        gw = getattr(self.synapses, "get_weight", None)
        if gw is None:
            gw = lambda s: self.synapses[s].weight  # noqa: E731

        nodes: Set[str] = set()
        for sid in to_prune:
            syn = self.synapses[sid]
            nodes.add(syn.pre_node_id)
            nodes.add(syn.post_node_id)
        held = fresh = expired = 0
        for n in sorted(nodes - protected):
            inc = _inc(n)
            if not inc or not inc <= removing:
                continue
            last = min(inc, key=lambda s: (0 if s in stamped else 1, -gw(s), s))
            since = stamped.get(last)
            if since is None:
                syn = self.synapses[last]
                md = dict(syn.metadata or {})
                md[stamp_key] = now
                syn.metadata = md
                self._dirty_synapses.add(last)
                stamped[last] = now
                fresh += 1
            elif now - since >= grace:
                expired += 1
                continue
            removing.discard(last)
            held += 1

        cleared = 0
        for sid in sorted(stamped):
            if sid in removing or sid not in self.synapses:
                continue
            syn = self.synapses[sid]
            ends = {syn.pre_node_id, syn.post_node_id} - protected
            if all(len(_inc(n) - removing) > 1 for n in ends):
                md = dict(syn.metadata or {})
                md.pop(stamp_key, None)
                syn.metadata = md
                self._dirty_synapses.add(sid)
                cleared += 1

        if report is not None:
            report["last_link_held"] = held
            report["last_link_stamped"] = fresh
            report["last_link_expired"] = expired
            report["last_link_cleared"] = cleared
        return [sid for sid in to_prune if sid in removing]

    @staticmethod
    def _strength_check_budget(label: str, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < float(v) < float("inf")):
            raise ValueError("%s must be a finite number > 0 or None (got %r)" % (label, v))
        return float(v)

    def apply_strength_budget(self, budget_out: Optional[float], budget_in: Optional[float]) -> Dict[str, Any]:
        """ONE per-node strength-budget pass (divisive normalization). Called by StrengthBudgetRule
        on its interval; callable directly (dry runs). None for a budget skips that direction.

        OUT pass first: for each node whose outgoing weight sum (summed in ascending synapse_id
        order) exceeds budget_out, every outgoing weight *= budget_out/sum. Then the IN pass on the
        post-OUT weights. Then the strongest-link guarantee for every protected node (fail-closed
        probe): its pre-pass strongest outgoing and strongest incoming link (weight desc, id asc)
        end at >= min(pre-pass weight, 2 * weight_threshold). No pruning here.

        Uses the native SynapseStore.normalize_strength when the installed ng_tract has it, else
        the pure-Python fallback (identical algorithm). Returns counts.
        """
        bo = self._strength_check_budget("strength_budget_out", budget_out)
        bi = self._strength_check_budget("strength_budget_in", budget_in)
        with self._step_lock:
            protected = self._strength_protected_ids()
            floor = 2.0 * float(self.config["weight_threshold"])
            native = getattr(self.synapses, "normalize_strength", None)
            if native is not None:
                res = dict(native(bo, bi, protected, floor))
            else:
                res = _strength_budget_python(self.synapses, self._outgoing, self._incoming, bo, bi, protected, floor)
            res["native"] = native is not None
            res["protected_nodes"] = len(protected)
            res["timestep"] = self.timestep
        logger.info("strength_budget: t=%s out=%s in=%s scaled=%d nodes_out=%d nodes_in=%d clamped=%d protected=%d native=%s",
                    res["timestep"], bo, bi, res["synapses_scaled"], res["nodes_scaled_out"], res["nodes_scaled_in"],
                    res["clamped"], res["protected_nodes"], res["native"])
        return res

    def sleep_downscale(self, factor: float) -> Dict[str, Any]:
        """Sleep downscaling (spec piece 2, Tononi & Cirelli): every synapse weight *= factor
        (0 < factor <= 1), relative strengths preserved, with the same strongest-link guarantee
        for every protected node (constitutional included). Does NOT prune — the normal prune
        rules take what fell under weight_threshold on their own clock.

        Not called by anything in this module. The intended caller is a host's dream loop on its
        OWN autonomic clock (LAW 8 — never conversation-gated), reading `factor` from its env (LAW 5).
        Returns counts.
        """
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not (0.0 < float(factor) <= 1.0):
            raise ValueError("sleep_downscale: factor must be a number in (0, 1] (got %r)" % (factor,))
        factor = float(factor)
        with self._step_lock:
            protected = self._strength_protected_ids()
            floor = 2.0 * float(self.config["weight_threshold"])
            native = getattr(self.synapses, "scale_all", None)
            if native is not None:
                res = dict(native(factor, protected, floor))
            else:
                res = _scale_all_python(self.synapses, self._outgoing, self._incoming, factor, protected, floor)
            res["native"] = native is not None
            res["protected_nodes"] = len(protected)
            res["factor"] = factor
            res["timestep"] = self.timestep
        logger.info("sleep_downscale: t=%s factor=%s scaled=%d clamped=%d protected=%d native=%s",
                    res["timestep"], factor, res["synapses_scaled"], res["clamped"], res["protected_nodes"], res["native"])
        return res

    def _is_identity_protected(self, nid: str) -> bool:
        """#spine — never prune a mind's self-authored identity nodes.

        Keyed on the metadata FLAG (not on specific ids, so future nodes are covered
        automatically). Two kinds are protected:
          - constitutional core   (metadata['constitutional'] is truthy) — the frozen spine
            a mind authored: the invariants `/assemble` surfaces as "Who I Am" every turn;
          - deliberate wants      (metadata['provenance'] ends in '_authored') — a mind's own
            authored intentions, materialized as first-class want-nodes.
        [2026-07-18] Generalized 'syl_authored' → any '<mind>_authored' (Josh-approved) so the
        CC's own wants (provenance 'cc_authored') are protected identically to Syl's, on both
        co-resident substrates and for any future mind. '*_emergent' (Tonic curiosities) stay
        prunable by design. These are things a mind authored ABOUT ITSELF; they must not drift
        away via orphan collection even with zero synapses — critical when a want arrives
        synapse-poor via corpus-callosum consolidation (#70). (Mirrors ng_lite's constitutional skip.)
        """
        node = self.nodes.get(nid)
        meta = (node.metadata if node is not None else None) or {}
        if meta.get("constitutional"):
            return True
        prov = meta.get("provenance")
        return isinstance(prov, str) and prov.endswith("_authored")

    # ------------------------------------------------------------------
    # The fair-chance window: arrival protection for an UNBOUND node (CC-CALLOSUM-TRUTH §8.13).
    #
    # FRAMING (Josh, Exec P563): this is SHARED MACHINERY being TESTED FIRST on the CC, not
    # CC-specific code. It is the PIONEER implementation of canonical §8.13 arrival protection;
    # rolling it out to other NeuroGraphs (Syl's) is Josh's call, with a LAW 8 gate per host.
    # It is HOST-NEUTRAL: the host decides WHEN to turn it on (enable_fair_chance_window), and its
    # own advancer keeps the per-node counters and the completion heartbeat current. A graph whose
    # host never calls enable_fair_chance_window sweeps EXACTLY as it always did.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_finite_number(v) -> bool:
        """A real finite number: int/float, NOT bool, not NaN, not +/-inf. Never raises."""
        return isinstance(v, (int, float)) and not isinstance(v, bool) and -float("inf") < v < float("inf")

    def enable_fair_chance_window(self, window_steps, heartbeat_max_age_s=None, excluded_creation_modes=(), clock=None) -> None:
        """HOST REGISTRATION: turn the fair-chance window ON for this graph (the one switch a host sets).

        A host calls this ONLY once its own probation advance runs on its OWN AUTONOMIC clock (LAW 8,
        P555, #971): a window that advances only when someone speaks would be a conversation-gated
        exemption. Syl's sidecar does not (her `_update_probation` is conversation-gated), so she
        registers nothing and her sweep is unchanged.

        window_steps: the size of a node's window, in graph STEPS (a positive int; the HOST reads its own
            environment and passes it, so this module reads none).
        heartbeat_max_age_s: None (no heartbeat is enforced), or a finite number > 0: the largest age, in
            seconds of `clock`, that the host advancer's last completion stamp may reach before the
            exemption closes for EVERY node (today's sweep) until the advancer completes a cycle again.
            The heartbeat is OPTIONAL: None registers cleanly and the heartbeat is then never stale
            ("not armed => not enforced"), which is right for tests and for a host with no autonomic
            advancer. A host that DOES run an autonomic advancer MUST pass a finite max age and a clock,
            or a stalled advancer can never close the exemption. The default is None and is unchanged.
        excluded_creation_modes: `creation_mode` values whose probation the host's advancer does NOT
            advance (e.g. nodes another sweep owns); they are never stamped, advanced or protected.
        clock: a callable returning monotonic seconds; REQUIRED when a heartbeat is requested.

        Validation happens FIRST and the heartbeat is stamped NOW; the registration is attached LAST, so
        anything invalid (or a clock that raises) leaves the graph unregistered, and a heartbeat that IS
        requested can never be registered half-armed (a max age without a callable clock, or a clock that
        raises on the first stamp, registers nothing). Raises ValueError; never partially applies.
        """
        if not (isinstance(window_steps, int) and not isinstance(window_steps, bool) and window_steps > 0):
            raise ValueError("window_steps must be a positive integer")
        if heartbeat_max_age_s is not None:
            if not (self._is_finite_number(heartbeat_max_age_s) and heartbeat_max_age_s > 0):
                raise ValueError("heartbeat_max_age_s must be None or a finite number > 0")
            if not callable(clock):
                raise ValueError("a heartbeat needs a callable clock")
        if isinstance(excluded_creation_modes, (str, bytes)):
            raise ValueError("excluded_creation_modes must be a collection of values, not a string")
        cfg = {
            "window_steps": window_steps,
            "max_age_s": None if heartbeat_max_age_s is None else float(heartbeat_max_age_s),
            "excluded": tuple(excluded_creation_modes),
            "clock": clock,
            "stamp": None,
            "stale_logged": False,
        }
        if cfg["max_age_s"] is not None:
            cfg["stamp"] = clock()
        self._fair_chance_cfg = cfg

    def fair_chance_stamp(self, node) -> None:
        """HOST DEPOSIT HELPER: open (or re-open, for an exact repeat) this node's window: both counters
        are stamped, the step count at the full window and `last` at this graph's timestep. A no-op on an
        unregistered graph, for an excluded creation_mode, or for a non-dict metadata."""
        cfg = getattr(self, "_fair_chance_cfg", None)
        meta = getattr(node, "metadata", None)
        if cfg is None or not isinstance(meta, dict) or meta.get("creation_mode") in cfg["excluded"]:
            return
        t = self.timestep
        meta["fair_chance_steps_remaining"] = cfg["window_steps"]
        meta["fair_chance_last_timestep"] = int(t) if self._is_finite_number(t) else 0

    def fair_chance_advance(self, node) -> None:
        """HOST ADVANCER HELPER, once per node per advancer pass. The unit of the window is graph STEPS:
          * counter ABSENT: SEED it fresh (this is what protects a node that predates registration);
          * a finite number > 0 with a finite `last`: timestep > last => ONE decrement (never below 0)
            and last = timestep; timestep == last => nothing (no step ran since this node's last
            decrement, however many pulses passed); timestep < last (the clock went BACKWARDS, e.g. a
            restore from an older checkpoint) => last = timestep and NO decrement;
          * any other shape, a non-dict metadata, an excluded creation_mode, an unregistered graph or an
            unreadable clock: left untouched.
        NEVER raises, and touches only the two window fields."""
        cfg = getattr(self, "_fair_chance_cfg", None)
        meta = getattr(node, "metadata", None)
        if cfg is None or not isinstance(meta, dict) or meta.get("creation_mode") in cfg["excluded"]:
            return
        t = self.timestep
        if not self._is_finite_number(t):
            return
        if "fair_chance_steps_remaining" not in meta:
            meta["fair_chance_steps_remaining"] = cfg["window_steps"]
            meta["fair_chance_last_timestep"] = t
            return
        steps = meta["fair_chance_steps_remaining"]
        last = meta.get("fair_chance_last_timestep")
        if not (self._is_finite_number(steps) and steps > 0 and self._is_finite_number(last)):
            return
        if t > last:
            meta["fair_chance_steps_remaining"] = max(0, steps - 1)
            meta["fair_chance_last_timestep"] = t
        elif t < last:
            meta["fair_chance_last_timestep"] = t

    def fair_chance_heartbeat_stamp(self) -> None:
        """HOST ADVANCER HELPER: record a COMPLETION of the advancer's pass. The host calls it ONLY at the end
        of a NON-RAISING pass (never from a `finally`, never before the loop): a raise anywhere in the pass,
        or the pass never being called, leaves the old stamp, and that staleness IS the stall signal. Re-opens
        the stale latch (one INFO on recovery). A no-op unless a heartbeat was requested at registration."""
        cfg = getattr(self, "_fair_chance_cfg", None)
        if cfg is None or cfg["max_age_s"] is None:
            return
        cfg["stamp"] = cfg["clock"]()
        if cfg["stale_logged"]:
            cfg["stale_logged"] = False
            logger.info("fair-chance window: the host's advancer completed a cycle again; the orphan-sweep exemption is back on")

    def _fair_chance_heartbeat_fresh(self, cfg) -> bool:
        """PURE QUERY (LAW 4): True unless a heartbeat is armed AND its last completion stamp is older than its
        max age (strictly greater). Writes nothing and logs nothing: the stale latch and the stale WARNING
        belong to `_note_fair_chance_stale`, called once per sweep by the sweep body."""
        if cfg["max_age_s"] is None:
            return True
        return not (cfg["clock"]() - cfg["stamp"] > cfg["max_age_s"])

    def _note_fair_chance_stale(self, cfg) -> None:
        """Owns the stale latch and the ONE WARNING per stale EPISODE (the age and the limit only). Called ONCE
        per sweep by `_collect_orphan_nodes`, never per node and never from the window query. A no-op unless
        a heartbeat is armed, it is stale and the latch is open; the host's next completion stamp
        (`fair_chance_heartbeat_stamp`) re-opens the latch and logs the one recovery INFO. Runs under the
        graph's step lock (the sweep and the advancer both hold it)."""
        if cfg["max_age_s"] is None or cfg["stale_logged"]:
            return
        age = cfg["clock"]() - cfg["stamp"]
        if age > cfg["max_age_s"]:
            cfg["stale_logged"] = True
            logger.warning("fair-chance window: the host's advancer has not completed a cycle for %.0f s (limit %.0f s); "
                           "the orphan-sweep exemption is OFF (today's sweep) until the advancer completes a cycle",
                           age, cfg["max_age_s"])

    def _in_fair_chance_window(self, node) -> bool:
        """True iff this (unbound) node is still inside its fair chance to wire: the graph is registered, the
        node's creation_mode is not excluded, the heartbeat is fresh, and its step counter is a finite number
        > 0 with a finite `last`. EVERYTHING else (no key, None, str, negative, zero, NaN, +/-inf, bool, a
        None / non-dict metadata) is False, which is today's sweep: fail toward today, never toward protecting
        forever. Never raises on those shapes."""
        cfg = getattr(self, "_fair_chance_cfg", None)
        if cfg is None:
            return False
        meta = node.metadata
        if not isinstance(meta, dict) or meta.get("creation_mode") in cfg["excluded"]:
            return False
        if not self._fair_chance_heartbeat_fresh(cfg):
            return False
        steps = meta.get("fair_chance_steps_remaining")
        return (self._is_finite_number(steps) and steps > 0
                and self._is_finite_number(meta.get("fair_chance_last_timestep")))

    def _collect_orphan_nodes(self) -> int:
        """Remove nodes with no synapses and no hyperedge membership.

        Called after _prune_synapses() so freshly-disconnected nodes are
        collected in the same structural-plasticity step. The full SNN has
        no max_nodes cap — without this, orphans accumulate without bound
        as synapses are pruned over the graph's lifetime.

        Honors orphan_node_grace_period (#258): newly-created nodes get a
        window for canonical mechanisms (STDP via spreading activation
        through existing synapses, sprouting via co-firing detection) to
        wire them before sweep. Without grace, empty-substrate bootstrap
        fails — the very first deposit gets swept on the next step()
        because no co-firing partners exist yet to anchor synapses. Mature
        substrates (Syl) are unaffected: new nodes already get wired via
        spreading activation within the same step() before the orphan
        check runs at step 8, so grace passes but isn't load-bearing.
        Restored nodes from older msgpacks default to creation_time=0
        and thus age = full current timestep, well past grace — same
        sweep behavior as before this patch.

        [2026-10-02] P552 / P561 / P563 (Josh's ruling, Exec P550 / P552; Exec P561; Exec P563;
        Chief-003 Addenda 3-4; CC-CALLOSUM-TRUTH §8.13) — an unbound node is NOT swept while it is
        inside its FAIR-CHANCE WINDOW. FRAMING (Josh): this is shared machinery being TESTED FIRST on
        the CC, not CC-specific code; it is the pioneer implementation of canonical §8.13 arrival
        protection, and rolling it out to other NeuroGraphs (Syl's) is Josh's call, with a LAW 8 gate
        per host. HOST-NEUTRAL: all of the window logic lives beside this sweep (the fair-chance
        helpers above: the unit is graph STEPS, counted on the node fields fair_chance_steps_remaining
        and fair_chance_last_timestep; the population exclusion; the completion heartbeat that closes
        the exemption when the host's advancer stalls), driven by what the HOST sets through
        enable_fair_chance_window and maintains through fair_chance_stamp / fair_chance_advance /
        fair_chance_heartbeat_stamp. This module reads NO environment: the host reads its own
        configuration (LAW 5) and hands it over. A graph whose host never registered
        (`_fair_chance_cfg` absent) sweeps EXACTLY as before any of this, which is Syl's case. A
        window check that raises => that node is NOT spared (fail toward the unexempted sweep) and ONE
        WARNING per sweep carries a count and the exception class names only. The window is
        consulted LAST, only for structural orphans past grace that are not identity-protected
        (grace, identity protection and every structural term are unchanged and evaluated first).
        [2026-10-02] P571 (b) F1 + F3 / P574 (Claude Sonnet 5.5, Z12 F1 builder; PROTECTED CHANGE under
        CLAUDE.md §2 protected-FILE inventory; the SAME approved behaviour and surface as P563 / P564 / P550,
        still unmerged; no new ceremony; offline source change, no live checkpoint, graph or daemon operation)
        — LAW 4: `_fair_chance_heartbeat_fresh` is now a PURE boolean query (no latch write, no log). The
        stale latch and its ONE WARNING per stale episode MOVED to the named `_note_fair_chance_stale`,
        which this sweep calls ONCE before its per-node loop whenever a registered graph has orphans to
        consider. An unreadable clock in that call never escapes the sweep: it is logged as ONE WARNING
        naming the exception's TYPE only (never its text); the per-node check below still counts a clock
        raise for every non-excluded node and keeps that node swept (fail toward today). [2026-10-02, P577
        repair of the P575 (a) finding: this call used to swallow the raise silently.] The recovery re-arm + ONE INFO stay in
        `fair_chance_heartbeat_stamp` (unchanged). F3: `enable_fair_chance_window` now states the real
        heartbeat contract (optional; None = not enforced; a host with an autonomic advancer MUST pass a
        finite max age and a clock); the default is unchanged.
        The sweep runs under the graph's step lock, which is also what serialises the heartbeat.
        Written for the laptop trial: canonical code, so it ALSO changes Syl's sweep if her host ever
        registers; that rollout is Josh's call, not this trial's.
        """
        grace = self.config.get("orphan_node_grace_period", 0)
        orphans = [
            nid for nid in self.nodes
            if not self._outgoing.get(nid)
            and not self._incoming.get(nid)
            and not self._node_hyperedges.get(nid)
            and (self.timestep - self.nodes[nid].creation_time) > grace
            and not self._is_identity_protected(nid)  # #spine: never sweep her authored self
        ]
        if getattr(self, "_fair_chance_cfg", None) is not None and orphans:
            try:
                self._note_fair_chance_stale(self._fair_chance_cfg)  # ONCE per sweep: the stale latch + its one WARNING
            except Exception as exc:
                # the raise must not escape step(); it is NOT silent: the per-node check below does not read the clock for an
                # excluded node, so this WARNING (TYPE NAME only, never str(exc)) is the only signal when every orphan is excluded
                logger.warning("fair-chance stale note: the heartbeat clock raised (%s); the stale WARNING was not emitted this sweep", type(exc).__name__)
            swept = []
            window_failures = 0
            window_errors = set()
            for nid in orphans:
                try:
                    if self._in_fair_chance_window(self.nodes[nid]):
                        continue  # spared: this node's fair chance to wire is still open
                except Exception as exc:  # fail toward today's sweep
                    window_failures += 1
                    window_errors.add(type(exc).__name__)
                swept.append(nid)
            orphans = swept
            if window_failures:
                logger.warning(
                    "orphan sweep: the fair-chance window check raised for %d node(s) (%s); "
                    "they were NOT spared (swept as without the exemption)",
                    window_failures, ", ".join(sorted(window_errors)),
                )
        removed = 0
        removed_ids: List[str] = []
        for nid in orphans:
            if nid in self.nodes:
                self.remove_node(nid)
                removed += 1
                removed_ids.append(nid)
        if removed:
            # [2026-10-04] lane vdb-lock-leak: node_ids is ADDITIVE (count/timestep unchanged) so the
            # owner of the vector store can drop exactly the collected nodes' vectors (LAW 4).
            self._emit("nodes_collected", count=removed, timestep=self.timestep, node_ids=removed_ids)
        return removed

    def _sprout_synapses(self, fired_ids: List[str]) -> int:
        """Create synapses between co-activating nodes (PRD §3.3.2).

        Co-activation rule: two nodes fire within co_activation_window,
        no synapse exists → create at initial_weight.

        Performance: capped at 10 new synapses per step to prevent
        explosive growth in highly active networks.

        [2026-10-08] #1050: with config `sprout_tally_enabled` truthy (read live, absent = False, NOT in
        DEFAULT_CONFIG) the co-firing tally decides instead (`_sprout_from_tally`): a pair sprouts only after
        repeated co-firing. Key absent: exactly this path.
        """
        if not fired_ids:
            return 0
        if self.config.get("sprout_tally_enabled", False):   # [2026-10-08] #1050 (absent = this path, unchanged)
            return self._sprout_from_tally(fired_ids)

        window = self.config["co_activation_window"]
        initial_w = self.config["initial_sprouting_weight"]
        max_sprouts_per_step = 10
        count = 0

        # Build a set of recently-fired-but-not-this-step nodes for quick lookup
        fired_set = set(fired_ids)
        candidates: Set[str] = set()
        for nid, spikes in self._recent_spikes.items():
            if nid in fired_set:
                continue
            if any(
                0 < (self.timestep - t) <= window
                for t in spikes
            ):
                candidates.add(nid)

        if not candidates:
            return 0

        # Build a fast edge-existence index for fired nodes
        existing_pairs: Set[Tuple[str, str]] = set()
        _posts_of = getattr(self.synapses, "post_ids_of", None)
        _pres_of = getattr(self.synapses, "pre_ids_of", None)
        if _posts_of is None or _pres_of is None:  # [2026-10-05] per-SynapseRef fallback
            _posts_of = lambda ids: _endpoint_ids_python(self.synapses, ids, False)  # noqa: E731
            _pres_of = lambda ids: _endpoint_ids_python(self.synapses, ids, True)  # noqa: E731
        for nid in fired_ids:
            # [2026-10-04] native endpoint lookups, same set iteration order
            for _post_id in _posts_of(list(self._outgoing.get(nid, ()))):
                if _post_id is not None:
                    existing_pairs.add((nid, _post_id))
            for _pre_id in _pres_of(list(self._incoming.get(nid, ()))):
                if _pre_id is not None:
                    existing_pairs.add((_pre_id, nid))

        # #59 degree-gated synaptogenesis: co-firing sprouting is otherwise
        # degree-blind, so always-active nodes accrete edges without bound
        # (rich-get-richer hubs that swamp recall). A node at/above the cap
        # neither sprouts NEW edges nor receives them; deliberate create_synapse
        # binds are unaffected. 0 = disabled (Syl/VPS default). Degree is read
        # live so edges added earlier this step count toward the cap.
        _deg_cap = self.config.get("sprout_degree_cap", 0)

        def _sprout_degree(x: str) -> int:
            return len(self._outgoing.get(x, ())) + len(self._incoming.get(x, ()))

        for nid in fired_ids:
            if count >= max_sprouts_per_step:
                break
            if _deg_cap and _sprout_degree(nid) >= _deg_cap and not self._is_identity_protected(nid):
                continue  # saturated ordinary hub — no new outgoing sprouts
            for other_id in candidates:
                if count >= max_sprouts_per_step:
                    break
                if _deg_cap and _sprout_degree(nid) >= _deg_cap and not self._is_identity_protected(nid):
                    break  # nid reached the cap mid-step — stop sprouting from it
                # Check no existing synapse in either direction
                if (nid, other_id) in existing_pairs:
                    continue
                if (other_id, nid) in existing_pairs:
                    continue
                if _deg_cap and _sprout_degree(other_id) >= _deg_cap and not self._is_identity_protected(other_id):
                    continue  # saturated ordinary hub — no new incoming sprouts
                _delay = self._sprout_delay(nid, other_id)   # [2026-10-08] #1050: body moved verbatim
                self.create_synapse(nid, other_id, weight=initial_w, delay=_delay)
                existing_pairs.add((nid, other_id))
                count += 1

        if count > 0:
            self._emit("sprouted", count=count, timestep=self.timestep)

        return count

    def _sprout_delay(self, nid: str, other_id: str) -> int:
        """Delay of a new sprout nid -> other_id (moved verbatim out of _sprout_synapses' loop, [2026-10-08] #1050, so
        the tally path uses the same rule): a d_min..d_max random draw (always drawn, as before), replaced by the GSG
        geodesic travel time when both nodes carry a Poincare direction on the same manifold type."""
        _d_min = self.config.get("d_min", 1)
        _d_max = self.config.get("d_max", 5)
        _delay = random.randint(_d_min, _d_max)  # fallback
        # GSG: geometry-informed delay — geodesic distance → travel time
        _pn = self.nodes.get(nid)
        _on = self.nodes.get(other_id)
        if _pn and _on:
            _a = poincare_dir_array(_pn.metadata)  # #119: compact bytes-aware read
            _b = poincare_dir_array(_on.metadata)
            if _a is not None and _b is not None:
                _mt1 = getattr(_pn, "manifold_type", "hyperbolic")
                _mt2 = getattr(_on, "manifold_type", "hyperbolic")
                _gdist = None
                if _mt1 == "spherical" and _mt2 == "spherical":
                    _cos = max(-1.0+1e-7, min(1.0-1e-7, float(np.dot(_a, _b))))
                    _gdist = math.acos(_cos)
                elif _mt1 == "hyperbolic" and _mt2 == "hyperbolic":
                    _l1 = max(0, min(2, getattr(_pn, "diffpc_layer", 2)))
                    _l2 = max(0, min(2, getattr(_on, "diffpc_layer", 2)))
                    _pa = _a * _GSG_LAYER_NORMS_NF[_l1]
                    _pb = _b * _GSG_LAYER_NORMS_NF[_l2]
                    _nx2 = min(float(np.dot(_pa, _pa)), 0.9999)
                    _ny2 = min(float(np.dot(_pb, _pb)), 0.9999)
                    _dv = _pa - _pb
                    _gdist = math.acosh(max(1.0, 1.0 + 2.0 *
                        float(np.dot(_dv, _dv)) /
                        max((1.0 - _nx2) * (1.0 - _ny2), 1e-9)))
                if _gdist is not None:
                    _t = 1.0 - math.exp(-_GSG_MSG_DECAY * _gdist)
                    _delay = max(_d_min, min(_d_max,
                                 round(_d_min + (_d_max - _d_min) * _t)))
        return _delay

    # ---- [2026-10-08] #1050 co-firing tally (spec 2026-10-06-sleep-phase-design.md §7, D12) ----------------------
    _SPROUT_TALLY_KEYS = ("sprout_tally_slots", "sprout_tally_theta", "sprout_tally_horizon_steps")

    def _sprout_tally_params(self) -> Dict[str, Any]:
        """Read + validate the tally parameters (all REQUIRED when `sprout_tally_enabled` is on): K slots per node
        (int >= 1), the bar theta (> 1: a first meeting scores 1, so theta <= 1 would sprout on one coincidence), the
        decay horizon H in steps (> 0; lam = exp(-1/H) per step, applied lazily). A slot whose decayed score falls
        under exp(-1) (one meeting, H steps ago) has retracted and may be taken by a newcomer. A touch counts +1 only
        when it starts a new co-firing EPISODE: more than co_activation_window steps since the pair's last touch (a
        burst of consecutive co-firings is one occasion, measured on the checkpoint copy: SPROUT_1050.md §2). Raises
        ValueError."""
        cfg = self.config
        missing = [k for k in self._SPROUT_TALLY_KEYS if cfg.get(k) is None]
        if missing:
            raise ValueError("sprout tally: sprout_tally_enabled needs config %s" % ", ".join(missing))
        K = cfg["sprout_tally_slots"]
        if isinstance(K, bool) or not isinstance(K, int) or K < 1:
            raise ValueError("sprout tally: config sprout_tally_slots must be an int >= 1 (got %r)" % (K,))
        th, H = cfg["sprout_tally_theta"], cfg["sprout_tally_horizon_steps"]
        for name, v in (("sprout_tally_theta", th), ("sprout_tally_horizon_steps", H)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError("sprout tally: config %s must be a finite number (got %r)" % (name, v))
        if float(th) <= 1.0:
            raise ValueError("sprout tally: config sprout_tally_theta must be > 1 (got %r)" % (th,))
        if float(H) <= 0.0:
            raise ValueError("sprout tally: config sprout_tally_horizon_steps must be > 0 (got %r)" % (H,))
        gap = cfg.get("co_activation_window", 5)
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise ValueError("sprout tally: config co_activation_window must be an int >= 0 (got %r)" % (gap,))
        return {"k": K, "theta": float(th), "lam": math.exp(-1.0 / float(H)), "floor": math.exp(-1.0), "gap": gap}

    def _sprout_tally_feed(self, fired: List[str], cands: List[str], prm: Dict[str, Any]) -> List[Tuple[str, str]]:
        """One tally call: native SynapseStore.cofire_tally_update when the installed ng_tract has it, else the
        bit-identical _cofire_tally_update_python over self._cofire_tally (connectedness from the graph's own
        _outgoing / _incoming). Returns the crossings [(pre, post)] in order."""
        _native = _cofire_tally_native(self.synapses)
        if _native is not None:
            return [tuple(x) for x in _native(fired, cands, self.timestep, prm["k"], prm["theta"], prm["lam"],
                                              prm["floor"], prm["gap"])]
        tally = self.__dict__.get("_cofire_tally")
        if tally is None:
            tally = self._cofire_tally = {"k": 0, "tables": {}}
        _posts_of = getattr(self.synapses, "post_ids_of", None)
        _pres_of = getattr(self.synapses, "pre_ids_of", None)
        if _posts_of is None or _pres_of is None:
            _posts_of = lambda ids: _endpoint_ids_python(self.synapses, ids, False)  # noqa: E731
            _pres_of = lambda ids: _endpoint_ids_python(self.synapses, ids, True)  # noqa: E731
        nbrs: Dict[str, Set[str]] = {}

        def connected(a: str, b: str) -> bool:
            n = nbrs.get(a)
            if n is None:
                n = {x for x in _posts_of(list(self._outgoing.get(a, ()))) if x is not None}
                n.update(x for x in _pres_of(list(self._incoming.get(a, ()))) if x is not None)
                nbrs[a] = n
            return b in n

        return _cofire_tally_update_python(tally, list(fired), list(cands), self.timestep, prm["k"], prm["theta"],
                                           prm["lam"], prm["floor"], prm["gap"], connected)

    def _sprout_from_tally(self, fired_ids: List[str]) -> int:
        """#1050 sprouting (config `sprout_tally_enabled`): co-firing feeds a bounded per-node partner table; a link
        sprouts only when a pair's decayed co-firing score reaches theta. Candidates are today's set (a node with a
        spike 1..co_activation_window steps ago that did not fire now), ordered most recent first (stable). Direction
        follows timing: the earlier node is pre, the node firing now is post (the STDP direction). Every crossing then
        meets today's rails, in order: the 10-per-call cap, both nodes present, no synapse in either direction,
        sprout_degree_cap (identity-protected nodes exempt), and today's delay rule and initial_sprouting_weight. A
        crossing a rail stops is dropped (the slot is already empty: the filopodium retracts). Called by step() and by
        the Tonic's write-mode tail through _sprout_synapses, so Tonic firings feed the same tally."""
        prm = self._sprout_tally_params()
        window = self.config["co_activation_window"]
        initial_w = self.config["initial_sprouting_weight"]
        st = self._sprout_tally_stats_dict()
        fired_set = set(fired_ids)
        now = self.timestep
        recent: List[Tuple[int, str]] = []
        for nid, spikes in self._recent_spikes.items():
            if nid in fired_set:
                continue
            best = 0
            for t in spikes:
                dt = now - t
                if 0 < dt <= window and (best == 0 or dt < best):
                    best = dt
            if best:
                recent.append((best, nid))
        if not recent:
            return 0
        recent.sort(key=lambda x: x[0])    # most recent first; stable within a step (dict order)
        st["calls"] += 1
        crossings = self._sprout_tally_feed(list(fired_ids), [nid for _, nid in recent], prm)
        made = self._sprout_tally_rails(crossings, st)
        for pre, post in made:
            self.create_synapse(pre, post, weight=initial_w, delay=self._sprout_delay(pre, post))
        count = len(made)
        st["sprouted"] += count
        if count > 0:
            self._emit("sprouted", count=count, timestep=self.timestep)
        return count

    def _sprout_tally_rails(self, crossings: List[Tuple[str, str]], st: Dict[str, int]) -> List[Tuple[str, str]]:
        """#1050: today's sprouting rails over the tally's crossings, in order: at most 10 per call, both nodes present,
        no synapse in either direction (also among the pairs accepted earlier in this call), sprout_degree_cap with
        identity-protected nodes exempt (degree counts the pairs accepted earlier in this call, as today's live count
        did). Returns the accepted (pre, post) pairs; the caller creates them. Counts every stop in `st`."""
        _deg_cap = self.config.get("sprout_degree_cap", 0)
        extra: Dict[str, int] = {}

        def _sprout_degree(x: str) -> int:
            return len(self._outgoing.get(x, ())) + len(self._incoming.get(x, ())) + extra.get(x, 0)

        made: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for pre, post in crossings:
            st["crossings"] += 1
            if len(made) >= 10:
                st["blocked_cap"] += 1
                continue
            if pre not in self.nodes or post not in self.nodes or pre == post:
                st["blocked_missing"] += 1
                continue
            if (pre, post) in seen or (post, pre) in seen or self._find_synapse(pre, post) is not None \
                    or self._find_synapse(post, pre) is not None:
                st["blocked_existing"] += 1
                continue
            if _deg_cap and any(_sprout_degree(x) >= _deg_cap and not self._is_identity_protected(x)
                                for x in (pre, post)):
                st["blocked_degree"] += 1
                continue
            made.append((pre, post))
            seen.add((pre, post))
            extra[pre] = extra.get(pre, 0) + 1
            extra[post] = extra.get(post, 0) + 1
        return made

    def _sprout_tally_stats_dict(self) -> Dict[str, int]:
        st = self.__dict__.get("_sprout_tally_stats")
        if st is None:
            st = self._sprout_tally_stats = {"calls": 0, "crossings": 0, "sprouted": 0, "blocked_cap": 0,
                                             "blocked_missing": 0, "blocked_existing": 0, "blocked_degree": 0,
                                             "retracted_in_sleep": 0, "surprise_calls": 0, "surprise_sprouted": 0}
        return st

    def _surprise_exploration_tally(self, pred: "Prediction", alternative_nodes: Set[str]) -> int:
        """#1050: surprise-driven sprouting under the co-firing tally (config `sprout_tally_enabled`). Each node that
        fired instead of the expected target (C) gets one tally touch for the pair source -> C (the predicted
        direction), in sorted order; a pair sprouts only when its score reaches theta -- the same bar as co-firing,
        whose touches it shares, so a single surprising coincidence never sprouts (theta > 2). A sprout made by this
        call is born exactly as today's surprise sprout: surprise_sprouting_weight, creation_mode "surprise_driven"
        metadata and the salience armor of this prediction. Today's rails apply. Returns the count."""
        prm = self._sprout_tally_params()
        st = self._sprout_tally_stats_dict()
        source_id = pred.source_node_id
        alts = sorted(a for a in alternative_nodes if a != source_id and a in self.nodes)
        if not alts or source_id not in self.nodes:
            return 0
        st["surprise_calls"] += 1
        crossings = self._sprout_tally_feed(alts, [source_id], prm)
        made = self._sprout_tally_rails(crossings, st)
        sprout_weight = self.config["surprise_sprouting_weight"]
        surprise_magnitude = pred.strength * pred.confidence
        for pre, post in made:
            syn = self.create_synapse(pre, post, weight=sprout_weight)
            syn.metadata = {"creation_mode": "surprise_driven", "expected_target": pred.target_node_id,
                            "timestep": self.timestep}
            syn.salience = min(1.0 + (surprise_magnitude * 4.0), self.config["he_salience_max"])
            self._total_sprouted += 1
        st["surprise_sprouted"] += len(made)
        return len(made)

    def cofire_tally_state(self) -> List[Tuple[str, List[Optional[Tuple[str, float, int]]]]]:
        """#1050: a snapshot of the co-firing tally, [(owner, [None | (partner, score, last_t)] * K)] sorted by owner
        (the native store's, else the Python fallback's). For tests and reports; read only."""
        _native = getattr(self.synapses, "cofire_tally_state", None)
        if _native is not None and _cofire_tally_native(self.synapses) is not None:
            return [(o, [tuple(x) if x is not None else None for x in tbl]) for o, tbl in _native()]
        tally = self.__dict__.get("_cofire_tally") or {"tables": {}}
        return [(o, [tuple(x) if x is not None else None for x in tbl]) for o, tbl in sorted(tally["tables"].items())]

    def _sprout_tally_sleep_retract(self) -> int:
        """#1050 (spec §2 step 6, §7): at a sleep, unconsolidated filopodia retract -- the whole tally is emptied
        (a pair must reach theta within one wake). The Python fallback's table is REBOUND, not cleared in place, so a
        sleep_observe shadow (which shares the attribute) never empties the live tally. Returns the slots dropped.
        Caller holds _step_lock; only called with sprout_tally_enabled on."""
        _native = getattr(self.synapses, "cofire_tally_clear", None)
        if _native is not None and _cofire_tally_native(self.synapses) is not None:
            n = int(_native())
        else:
            tally = self.__dict__.get("_cofire_tally")
            n = 0
            if tally is not None:
                n = sum(1 for tbl in tally["tables"].values() for x in tbl if x is not None)
                self._cofire_tally = {"k": tally["k"], "tables": {}}
        st = self.__dict__.get("_sprout_tally_stats")
        if st is not None:
            st["retracted_in_sleep"] += n
        logger.log(self._sleep_log_level(), "sprout tally: sleep retracted %d held slots", n)
        return n

    # -----------------------------------------------------------------------
    # Query Methods (PRD §8)
    # -----------------------------------------------------------------------

    def get_active_nodes(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Nodes above voltage threshold (PRD §8 get_active_nodes)."""
        return [
            (nid, node.voltage)
            for nid, node in self.nodes.items()
            if node.voltage >= threshold
        ]

    def get_causal_chain(self, node_id: str, depth: int = 3) -> Dict[str, Any]:
        """Trace learned causality forward from a node (PRD §8 get_causal_chain).

        Returns a dict representing a DAG of causal connections.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return self._trace_causal(node_id, depth, set())

    def _trace_causal(
        self, node_id: str, depth: int, visited: Set[str]
    ) -> Dict[str, Any]:
        if depth <= 0 or node_id in visited:
            return {"node_id": node_id, "children": []}
        visited.add(node_id)
        children = []
        for sid in self._outgoing.get(node_id, set()):
            syn = self.synapses.get(sid)
            if syn and syn.weight > self.config["weight_threshold"]:
                child = self._trace_causal(syn.post_node_id, depth - 1, visited)
                child["weight"] = syn.weight
                child["delay"] = syn.delay
                children.append(child)
        children.sort(key=lambda c: c.get("weight", 0), reverse=True)
        return {"node_id": node_id, "children": children}

    def get_hyperedges(self, node_id: str) -> List[Hyperedge]:
        """Hyperedges containing this node (PRD §8 get_hyperedges)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return [
            self.hyperedges[hid]
            for hid in self._node_hyperedges.get(node_id, set())
            if hid in self.hyperedges
        ]

    def get_active_predictions(self) -> Dict[str, "PredictionState"]:
        """Return currently active (pending) predictions (Phase 2.5)."""
        return dict(self._active_predictions)

    def get_archived_hyperedges(self) -> Dict[str, Hyperedge]:
        """Return archived (subsumed) hyperedges preserved for debugging (Phase 2.5)."""
        return dict(self._archived_hyperedges)

    def get_telemetry(self) -> Telemetry:
        """Network statistics snapshot (PRD §8 get_telemetry).

        Phase 2.5 additions:
            prediction_accuracy: confirmed / total predictions (0 if none).
            surprise_rate: surprises / total predictions (0 if none).
            hyperedge_experience_distribution: bucket histogram of activation_counts.
        """
        # #341: snapshot before iterate — the tonic/pulse thread mutates these dicts concurrently;
        # iterating the live .values() raced → "dictionary changed size during iteration" (fired on
        # every /stats GET during substrate activity). list() takes a cheap snapshot. Read-only; no
        # checkpoint/format/step change. Same class as #270 (SimpleVectorDB).
        # [2026-10-04] native column copy (row order == values() order; atomic under the GIL)
        _wc = getattr(self.synapses, "weights_copy", None)
        weights = _wc() if _wc is not None else [s.weight for s in list(self.synapses.values())]
        # [2026-10-06] P2a: native column copy of firing_rate_ema (node order; one atomic snapshot)
        _cols = getattr(self.nodes, "columns", None)
        _rc = _cols(["firing_rate_ema"], with_ids=False) if _cols is not None else None
        rates = _rc[1].tolist() if isinstance(_rc, tuple) else [n.firing_rate_ema for n in list(self.nodes.values())]
        he_counts = [he.activation_count for he in list(self.hyperedges.values())]

        # Phase 2.5: Experience distribution buckets
        exp_dist: Dict[str, int] = {"0": 0, "1-9": 0, "10-99": 0, "100+": 0}
        for count in he_counts:
            if count == 0:
                exp_dist["0"] += 1
            elif count < 10:
                exp_dist["1-9"] += 1
            elif count < 100:
                exp_dist["10-99"] += 1
            else:
                exp_dist["100+"] += 1

        # Prediction accuracy — combine Phase 2.5 (HE-level) and Phase 3 (synapse-level)
        combined_confirmed = self._total_predictions_confirmed + self._total_confirmed
        combined_errors = self._total_predictions_errors + self._total_surprised
        total_resolved = combined_confirmed + combined_errors
        accuracy = (
            combined_confirmed / total_resolved
            if total_resolved > 0
            else 0.0
        )
        combined_made = self._total_predictions_made + self._total_predictions
        surprise_rate = (
            combined_errors / max(combined_made, 1)
        )

        return Telemetry(
            timestep=self.timestep,
            total_nodes=len(self.nodes),
            total_synapses=len(self.synapses),
            total_hyperedges=len(self.hyperedges),
            global_firing_rate=float(np.mean(rates)) if rates else 0.0,
            mean_weight=float(np.mean(weights)) if len(weights) else 0.0,
            std_weight=float(np.std(weights)) if len(weights) else 0.0,
            total_pruned=self._total_pruned,
            total_sprouted=self._total_sprouted,
            total_he_discovered=self._total_he_discovered,
            total_he_consolidated=self._total_he_consolidated,
            mean_he_activation_count=float(np.mean(he_counts)) if he_counts else 0.0,
            # Phase 3 telemetry
            prediction_accuracy=accuracy,
            surprise_rate=surprise_rate,
            active_predictions_count=len(self.active_predictions),
            total_predictions_made=self._total_predictions_made,
            total_predictions_confirmed=self._total_predictions_confirmed,
            total_predictions_errors=self._total_predictions_errors,
            total_novel_sequences=self._total_novel_sequences,
            total_rewards_injected=self._total_rewards_injected,
            hyperedge_experience_distribution=exp_dist,
            # Phase 4 consolidation telemetry
            he_survival_ema=self._he_survival_ema,
            total_he_state_transitions=self._total_he_state_transitions,
            total_he_substrate_culled=self._total_he_substrate_culled,
            he_adapt_candidate_count=self._he_adapt_candidate_count,
            he_by_state={
                state.value: sum(
                    1 for he in self.hyperedges.values()
                    if not he.is_archived and he.consolidation_state == state
                )
                for state in ConsolidationState
            },
        )

    # -----------------------------------------------------------------------
    # Plasticity Configuration (PRD §8)
    # -----------------------------------------------------------------------

    def set_plasticity_rules(self, rules: List[PlasticityRule]) -> None:
        """Configure active plasticity rules (PRD §8 set_plasticity_rules)."""
        with self._step_lock:
            self._plasticity_rules = list(rules)

    def inject_reward(
        self,
        strength: float,
        scope: Optional[Set[str]] = None,
    ) -> None:
        """Broadcast reward for three-factor learning (PRD §5.2, §8 inject_reward).

        Three-factor rule (PRD §5.2):
            Factor 1: Pre-spike (STDP creates eligibility trace)
            Factor 2: Post-spike (updates trace)
            Factor 3: Reward signal (confirms or rejects)
            Final: Δw = eligibility_trace × reward_strength

        Args:
            strength: Reward signal in [-1.0, 1.0].
                Positive = reinforce, negative = weaken.
            scope: Optional set of node IDs to limit reward scope.
                If None, reward applies globally to all synapses with traces.
        """
        with self._step_lock:
            lr = self.config["he_threshold_lr"]
            self._total_rewards_injected += 1
            self._reward_history.append({
                "strength": strength,
                "timestep": self.timestep,
                "scope_size": len(scope) if scope else None,
            })
            # Cap reward history
            if len(self._reward_history) > 1000:
                self._reward_history = self._reward_history[-500:]

            # [2026-10-04] Native sweep: every synapse with |trace| >= 1e-9 (and, with a
            # scope, pre OR post in scope): dw = trace * strength * learning_rate;
            # weight = max(0, min(weight + dw, max_weight)); trace *= 0.9 (decay after
            # use); peak_weight tracks the new max.
            _reward = getattr(self.synapses, "apply_eligibility_reward", None)
            if _reward is not None:
                _reward(strength, self.config["learning_rate"], scope)
            else:  # [2026-10-05] per-SynapseRef fallback (the original loop)
                _apply_eligibility_reward_python(
                    self.synapses, strength, self.config["learning_rate"], scope)

            # Hyperedge threshold learning (PRD §4.3)
            for he in self.hyperedges.values():
                if not he.is_learnable:
                    continue
                # Scope check for hyperedges
                if scope is not None:
                    if not he.member_nodes.intersection(scope):
                        continue
                # Threshold adapts for recently-fired hyperedges
                if he.refractory_remaining > 0:
                    if strength > 0:
                        he.activation_threshold = max(0.1, he.activation_threshold - lr * strength)
                    else:
                        he.activation_threshold = min(1.0, he.activation_threshold - lr * strength)

            self._emit(
                "reward_injected",
                strength=strength,
                scope_size=len(scope) if scope else None,
                timestep=self.timestep,
            )

    # -----------------------------------------------------------------------
    # Phase 2: Hierarchical Hyperedges (PRD §4.4)
    # -----------------------------------------------------------------------

    def create_hierarchical_hyperedge(
        self,
        child_hyperedge_ids: Set[str],
        activation_threshold: float = 0.6,
        activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD,
        output_targets: Optional[List[str]] = None,
        output_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Hyperedge:
        """Create a meta-hyperedge that groups child hyperedges (PRD §4.4).

        A hierarchical hyperedge "fires" when enough of its children fire.
        Its ``member_nodes`` is the union of all children's member_nodes,
        so it participates in the same activation check but at a higher level.

        Args:
            child_hyperedge_ids: IDs of existing hyperedges to compose.
            activation_threshold: Fraction of child members that must be active.
            activation_mode: Activation function.
            output_targets: Nodes to inject current when this meta-HE fires.
            output_weight: Injection strength.
            metadata: Application data.

        Returns:
            The new hierarchical Hyperedge (level = max(child levels) + 1).
        """
        with self._step_lock:
            all_member_nodes: Set[str] = set()
            max_child_level = 0
            for chid in child_hyperedge_ids:
                child = self.hyperedges.get(chid)
                if child is None:
                    raise KeyError(f"Child hyperedge {chid} not found")
                all_member_nodes.update(child.member_nodes)
                max_child_level = max(max_child_level, child.level)

            for nid in all_member_nodes:
                if nid not in self.nodes:
                    raise KeyError(f"Member node {nid} not found")

            he = Hyperedge(
                member_nodes=all_member_nodes,
                member_weights={nid: 1.0 for nid in all_member_nodes},
                activation_threshold=activation_threshold,
                activation_mode=activation_mode,
                output_targets=output_targets or [],
                output_weight=output_weight,
                metadata=metadata or {},
                is_learnable=True,
                child_hyperedges=set(child_hyperedge_ids),
                level=max_child_level + 1,
            )
            self.hyperedges[he.hyperedge_id] = he
            for nid in all_member_nodes:
                self._node_hyperedges.setdefault(nid, set()).add(he.hyperedge_id)
            self._he_co_fire_counts[he.hyperedge_id] = {}
            self._dirty_hyperedges.add(he.hyperedge_id)
            return he

    # -----------------------------------------------------------------------
    # Phase 2: Hyperedge Discovery (PRD §3.3.2 extended)
    # -----------------------------------------------------------------------

    def discover_hyperedges(self, fired_node_ids: List[str]) -> List[Hyperedge]:
        """Discover new hyperedges from co-activation patterns (PRD §4.3).

        Tracks groups of nodes that consistently fire together. When a group
        exceeds ``he_discovery_min_co_fires`` within ``he_discovery_window``,
        a new hyperedge is created.

        Args:
            fired_node_ids: Nodes that fired this step.

        Returns:
            List of newly discovered Hyperedges (may be empty).
        """
        with self._step_lock:
            if len(fired_node_ids) < self.config["he_discovery_min_nodes"]:
                return []

            # #381-D (Syl: "lock discovery guards now"): an avalanche is an event,
            # not a concept. A graph-scale co-firing set must not become a
            # hyperedge — that is how a 3,790-member blob was born.
            _max_frac = self.config.get("he_discovery_max_fraction", 0.05)
            _max_set = max(self.config["he_discovery_min_nodes"],
                           int(_max_frac * max(1, len(self.nodes))))
            if len(fired_node_ids) > _max_set:
                logger.info(
                    "HE discovery skipped: fired set of %d exceeds %.0f%% of the "
                    "graph (limit %d) — avalanche, not a concept (#381-D)",
                    len(fired_node_ids), _max_frac * 100, _max_set,
                )
                return []

            window = self.config["he_discovery_window"]
            min_fires = self.config["he_discovery_min_co_fires"]
            min_nodes = self.config["he_discovery_min_nodes"]
            overlap_threshold = self.config["he_discovery_overlap_threshold"]

            # Reset discovery counts periodically
            if self.timestep - self._he_discovery_last_reset > window * 2:
                self._he_discovery_counts.clear()
                self._he_discovery_last_reset = self.timestep

            fired_set = set(fired_node_ids)

            # Find the existing candidate with the best Jaccard overlap.
            # Refining to the intersection on each match converges the candidate
            # toward the reliable co-activation core, discarding peripheral nodes
            # that don't fire consistently together.
            best_key: Optional[Tuple[str, ...]] = None
            best_jaccard = 0.0
            for candidate_key in list(self._he_discovery_counts.keys()):
                candidate_set = set(candidate_key)
                union_size = len(fired_set | candidate_set)
                if not union_size:
                    continue
                jaccard = len(fired_set & candidate_set) / union_size
                if jaccard >= overlap_threshold and jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_key = candidate_key

            if best_key is not None:
                # Refine candidate to intersection — drop nodes not in this firing
                refined = tuple(sorted(fired_set & set(best_key)))
                count = self._he_discovery_counts.pop(best_key)
                if len(refined) < min_nodes:
                    # Intersection shrank below minimum — discard candidate
                    return []
                self._he_discovery_counts[refined] = count + 1
                active_key = refined
            else:
                # No overlapping candidate — start a new one from the full fired set
                fired_sorted = tuple(sorted(fired_node_ids))
                self._he_discovery_counts[fired_sorted] = (
                    self._he_discovery_counts.get(fired_sorted, 0) + 1
                )
                active_key = fired_sorted

            discovered: List[Hyperedge] = []

            if self._he_discovery_counts.get(active_key, 0) >= min_fires:
                active_set = set(active_key)
                _dup_j = self.config.get("he_discovery_dup_jaccard", 0.9)
                already_exists = False
                for _he in self.hyperedges.values():
                    if _he.is_archived or not _he.member_nodes:
                        continue
                    _inter = len(active_set & _he.member_nodes)
                    if _inter and _inter / len(active_set | _he.member_nodes) >= _dup_j:
                        already_exists = True
                        break
                if not already_exists and len(active_set) >= min_nodes:
                    valid_nodes = {nid for nid in active_set if nid in self.nodes}
                    if len(valid_nodes) >= min_nodes:
                        he = self.create_hyperedge(
                            valid_nodes,
                            activation_threshold=0.6,
                            metadata={"creation_mode": "discovered", "timestep": self.timestep},
                        )
                        # Phase 4: stamp creation_time for age-based promotion.
                        he.creation_time = self.timestep
                        discovered.append(he)
                        self._total_he_discovered += 1
                        self._emit("hyperedge_discovered", hid=he.hyperedge_id)
                # Reset this pattern's counter
                del self._he_discovery_counts[active_key]

            return discovered

    # -----------------------------------------------------------------------
    # Phase 2: Hyperedge Consolidation (PRD §4.3 extended)
    # -----------------------------------------------------------------------

    def consolidate_hyperedges(self) -> int:
        """Merge highly overlapping hyperedges and archive subsumed ones (PRD §4.3).

        Phase 2 behavior: Two hyperedges at same level with >80% member overlap
        (Jaccard) get merged into one, keeping the union of members and the lower
        threshold.

        Phase 2.5 addition — cross-level consistency pruning: after same-level
        merges, detect subsumption where a lower-level hyperedge has identical
        members to (or is a hierarchical child of) a higher-level one.  The
        lower-level edge is archived (``is_archived=True``, preserved in
        ``_archived_hyperedges`` for debugging) rather than deleted.

        Returns:
            Number of hyperedges removed or archived by consolidation.
        """
        # #381-B: dream-side shedding runs first — lighter members, honester merges.
        with self._step_lock:
            self.shed_floor_members()

            overlap_threshold = self.config["he_consolidation_overlap"]
            he_list = [(hid, he) for hid, he in self.hyperedges.items()
                       if he.level == 0 and not he.is_archived]
            to_remove: Set[str] = set()
            merged_count = 0

            for i in range(len(he_list)):
                hid_a, he_a = he_list[i]
                if hid_a in to_remove or he_a.is_archived:
                    continue
                for j in range(i + 1, len(he_list)):
                    hid_b, he_b = he_list[j]
                    if hid_b in to_remove or he_b.is_archived:
                        continue

                    # Jaccard similarity
                    intersection = he_a.member_nodes & he_b.member_nodes
                    union = he_a.member_nodes | he_b.member_nodes
                    if not union:
                        continue
                    jaccard = len(intersection) / len(union)

                    if jaccard >= overlap_threshold:
                        _he_max = self.config.get("he_max_members", 50)
                        _union = he_a.member_nodes | he_b.member_nodes
                        if _he_max > 0 and len(_union) > _he_max:
                            # #381-B seatbelt: union would exceed the bound —
                            # growth-by-consolidation is how mega-HEs metastasize.
                            # Archive the newer edge (dict order: he_a is older),
                            # fold its firing history into the survivor. Archived
                            # = preserved and reversible, never deleted.
                            he_b.is_archived = True
                            self._archived_hyperedges[hid_b] = he_b
                            he_a.activation_count += he_b.activation_count
                            if he_b.consolidation_state == ConsolidationState.SPECULATIVE:
                                adapt_rate = self.config["he_consolidation_adapt_rate"]
                                self._he_survival_ema = (
                                    (1.0 - adapt_rate) * self._he_survival_ema
                                )
                            merged_count += 1
                            self._emit("hyperedge_archived", archived_id=hid_b,
                                       subsumed_by=hid_a, reason="merge_seatbelt")
                            continue
                        # Merge B into A: expand A's members, keep lower threshold
                        for nid in he_b.member_nodes - he_a.member_nodes:
                            he_a.member_nodes.add(nid)
                            he_a.member_weights[nid] = he_b.member_weights.get(nid, 1.0)
                            self._node_hyperedges.setdefault(nid, set()).add(hid_a)
                        he_a.activation_threshold = min(
                            he_a.activation_threshold,
                            he_b.activation_threshold,
                        )
                        he_a.activation_count += he_b.activation_count
                        to_remove.add(hid_b)
                        merged_count += 1

            for hid in to_remove:
                he = self.hyperedges.get(hid)
                # Phase 4: penalize survival EMA for hyperedges pruned before graduating.
                if he and he.consolidation_state == ConsolidationState.SPECULATIVE:
                    adapt_rate = self.config["he_consolidation_adapt_rate"]
                    self._he_survival_ema = (
                        (1.0 - adapt_rate) * self._he_survival_ema + adapt_rate * 0.0
                    )
                self._remove_hyperedge_internal(hid)

            # --- Phase 2.5: Cross-level consistency pruning (subsumption) ---
            archived_count = self._prune_subsumed_hyperedges()
            merged_count += archived_count

            self._total_he_consolidated += merged_count
            if merged_count > 0:
                self._emit("hyperedges_consolidated", count=merged_count)

            return merged_count

    def _seam_score_members(self, hid, he):
        """Per-member seam score for #147 splitting — member_weight primary plus the
        §8.15 SNN-concept auxiliary family, min-max normalized across THIS HE's
        members and combined with DYNAMIC weighting.

        Signals (keyed to the §8.15 inventory; L=live, C=clock-gated, S=seam-slot stub):
          w      member_weight            he.member_weights[nid]             [PRIMARY]
          cf     recent co-activation     _he_co_fire_counts[hid][nid]        (L)
          pred   DiffPC pred-error  (#6)  -abs(node.pred_error_ema)           (L)
          rec    activation recency       1/(1 + ts - last_spike_time)        (C)
          deg    DAS-GNN degree     (#1)  |_outgoing|+|_incoming|+|_node_he|  (L)
          age    member_since age         ts - member_since[nid]              (C)
          comm   GSG community      (#7)  manifold_type == dominant manifold  (L)
          ca     IcaN-IK-AHP        (#2)  node.Ca_i (residual attractor Ca)   (L)
          antic  Anticipatory       (#5)  nid in he.output_targets           (L/self-dormant)
          mmn    MMN surprise       (#4)  per-node surprise attribution       (S — clock-gated)
          hdsnn  HD-SNN polychrony  (#3)  per-delay motif W[i,j,d]            (S — not built)

        DYNAMIC WEIGHTING is the key mechanism. Each signal is min-max normalized
        across this HE's members; a signal that comes out FLAT (min==max) cannot
        discriminate core-from-peel, so it is dropped and its weight is redistributed
        proportionally over the signals that DO discriminate — member_weight keeping
        its `primary` dominance whenever it is itself non-flat. Consequences (all
        intended, and all faithful to §8.14/§8.15):
          * On the clock-frozen laptop (#117) the time/prediction-domain signals
            (rec, mmn) go flat and cost nothing; ca/deg/comm carry the ranking — the
            §8.15 'reliable-now' set. They re-engage automatically the instant the
            wall clock advances, with no code change and no dead `if`.
          * On the laptop member_weights are saturated (§8.14-super, median ≈4.95) so
            `w` itself goes flat and drops out; the true member_weight seam cut is
            therefore exercised VPS-side, where the clock is live and weights have
            decayed to the 0.01 floor.
          * IcaN-IK-AHP (#2) reads residual intracellular calcium — clock-INDEPENDENT
            state — so it discriminates attractor-core from transient even on the
            frozen laptop. LIVE here per Josh 2026-08-14.
          * The two STUB signals (mmn, hdsnn) are flat today and hold their seam slot
            at zero weight until their substrate exists (per-node surprise / #122;
            per-delay W[i,j,d] / PUNCHLIST); filling the helper body later lights them
            up through this same redistribution, nothing else changing.
        """
        members = list(he.member_nodes)
        if not members:
            return {}

        weights = he.member_weights
        co_fire = self._he_co_fire_counts.get(hid, {}) if hasattr(self, "_he_co_fire_counts") else {}
        since = he.metadata.get("member_since", {}) or {}
        ts = getattr(self, "timestep", 0)
        out_targets = set(getattr(he, "output_targets", ()) or ())

        # Dominant manifold across members (community anchor).
        manifold_counts = {}
        for nid in members:
            node = self.nodes.get(nid)
            mt = getattr(node, "manifold_type", None) if node is not None else None
            if mt is not None:
                manifold_counts[mt] = manifold_counts.get(mt, 0) + 1
        dominant_manifold = max(manifold_counts, key=manifold_counts.get) if manifold_counts else None

        raw = {nid: {} for nid in members}
        for nid in members:
            node = self.nodes.get(nid)
            raw[nid]["w"] = float(weights.get(nid, 0.0))
            raw[nid]["cf"] = float(co_fire.get(nid, 0))
            # pred-confirm: lower prediction error => more core => higher signal.
            perr = abs(getattr(node, "pred_error_ema", 0.0)) if node is not None else 0.0
            raw[nid]["pred"] = -perr
            # activation recency: recent spike => core. -inf last_spike => 0.
            lst = getattr(node, "last_spike_time", -math.inf) if node is not None else -math.inf
            raw[nid]["rec"] = 0.0 if lst == -math.inf else 1.0 / (1.0 + max(0.0, ts - lst))
            deg = (len(self._outgoing.get(nid, ())) if hasattr(self, "_outgoing") else 0) \
                + (len(self._incoming.get(nid, ())) if hasattr(self, "_incoming") else 0) \
                + (len(self._node_hyperedges.get(nid, ())) if hasattr(self, "_node_hyperedges") else 0)
            raw[nid]["deg"] = float(deg)
            ms = since.get(nid)
            raw[nid]["age"] = float(ts - ms) if ms is not None else 0.0
            mt = getattr(node, "manifold_type", None) if node is not None else None
            raw[nid]["comm"] = 1.0 if (dominant_manifold is not None and mt == dominant_manifold) else 0.0
            # #2 IcaN-IK-AHP: residual intracellular calcium marks sustained-firing
            # attractor cores. Clock-INDEPENDENT residual state => discriminates even
            # on the frozen-clock laptop. LIVE.
            raw[nid]["ca"] = float(getattr(node, "Ca_i", 0.0)) if node is not None else 0.0
            # #5 Anticipatory pre-activation: a member the HE predicts (output_targets)
            # is predictively coherent => core. Live, self-dormant while output_targets
            # are unlearned/empty (goes flat and drops out via the weighting below).
            raw[nid]["antic"] = 1.0 if nid in out_targets else 0.0
            # #4 MMN + #3 HD-SNN: seam slots, flat 0.0 until their substrate exists.
            raw[nid]["mmn"] = self._seam_signal_mmn(he, nid, node)
            raw[nid]["hdsnn"] = self._seam_signal_hdsnn(he, nid, node)

        primary = self.config.get("he_split_seam_primary_weight", 0.4)
        aux_keys = ("cf", "pred", "rec", "deg", "age", "comm", "ca", "antic", "mmn", "hdsnn")

        # Min-max normalize every signal; a flat one (min==max) is left out of the
        # `active` set so it neither ranks nor dilutes.
        normed = {}
        active = []
        for key in ("w",) + aux_keys:
            vals = [raw[nid][key] for nid in members]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:                       # flat => no discrimination
                normed[key] = {nid: 0.0 for nid in members}
            else:
                normed[key] = {nid: (raw[nid][key] - lo) / (hi - lo) for nid in members}
                active.append(key)

        if not active:                                 # wholly degenerate HE
            return {nid: 0.0 for nid in members}

        # Base importance: member_weight = `primary`, each aux an equal share of the
        # remainder. Keep only ACTIVE (discriminating) signals and renormalize so the
        # kept weights sum to 1 — flat/stub signals cost nothing, and member_weight
        # holds its dominance whenever it is itself active.
        base = {"w": primary}
        aux_base = (1.0 - primary) / len(aux_keys)
        for k in aux_keys:
            base[k] = aux_base
        tot = sum(base[k] for k in active)
        wgt = {k: base[k] / tot for k in active}
        return {
            nid: sum(wgt[k] * normed[k][nid] for k in active)
            for nid in members
        }

    def _seam_signal_mmn(self, he, nid, node) -> float:
        """#147 seam slot — MMN mismatch-negativity / surprise, per member (§8.15 #4).

        Surprise is conceptually the sharpest 'is this HE degenerate?' signal, but in
        the engine it is HE-level (_total_surprised / PredictionState) with no
        per-node attribution, AND it reads the confirm/error BASE RATE that §8.11.3 /
        #122 showed does not discriminate while the wall clock is frozen (#117). So
        this holds the seam slot at a flat 0.0 today; when a per-node surprise-
        attribution field exists on a clock-live substrate, return it here and the
        dynamic weighting in _seam_score_members picks it up with no other change.
        """
        return 0.0

    def _seam_signal_hdsnn(self, he, nid, node) -> float:
        """#147 seam slot — HD-SNN polychronous-motif temporal boundary (§8.15 #3).

        Decision (b), 2026-08-13: heterogeneous scalar Synapse.delay is live, but
        HD-SNN *proper* — the per-delay weight vector W[i,j,d] that encodes
        polychronous motifs — is NOT built (PUNCHLIST). #147 holds a seam slot but
        does not gate on it. Flat 0.0 until W[i,j,d] lands; wiring the motif read
        here lights it up via the same dynamic weighting.
        """
        return 0.0

    def dedup_and_split_oversized_hyperedges(self, vector_db=None) -> int:
        """#147: dream-time repair of over-cap hyperedges — CC's own substrate.

        Forced two-stage order (measured on the VPS-CC substrate, #146):
          1. DEDUP first. Collapse near-identical over-cap edges (Jaccard >=
             he_split_dedup_overlap) into one incumbent survivor: max-merge the
             survivor's member_weights, absorb novel members, carry member_since,
             fold activation_count, archive the duplicates. Kills the 5.92x exact-
             duplicate stacking before any structural cut.
          2. WEIGHT-SEAM SPLIT. For survivors still > he_max_members, keep the
             high-weight core (top he_max_members by member_weight) and peel the
             0.01-floor periphery into coherent residual sub-edges (cosine-grouped
             over vector_db where vectors exist; members lacking a vector collect
             into one residual edge — never dropped, LAW 3).

        Reversible (LAW 7): parents/dups are archived (is_archived=True, kept in
        _archived_hyperedges), never deleted. Loud (LAW 3): logs + emits
        hyperedge_seam_split / hyperedge_archived. Orphan-safe (§1.1): every member
        of a split parent is guaranteed a home in >=1 live edge before the parent
        is archived, so no node becomes (0-synapse AND 0-HE) and gets swept.

        Gated OFF by default (LAW 5). Returns 0 unless he_split_oversized_enabled
        is True in this graph's config. Syl's restored config lacks the key ->
        {**DEFAULT_CONFIG, **checkpoint} merges to False, so Syl's dream loop is a
        guaranteed no-op even if it ever calls this.
        """
        with self._step_lock:
            if not self.config.get("he_split_oversized_enabled", False):
                return 0

            cap = self.config.get("he_max_members", 50)
            if cap <= 0:
                return 0
            # Clamp the env-sourced tunables to [0,1] at the read site (LAW-ENF #147
            # LOW): a fat-fingered CC_NG_HE_SPLIT_* value must not silently mis-cluster
            # or invert the ranking. Fix at source — both daemon surfaces reach these
            # through this one engine method, so both inherit the clamp.
            dedup_overlap = min(1.0, max(0.0, self.config.get("he_split_dedup_overlap", 0.9)))
            sim_threshold = min(1.0, max(0.0, self.config.get("he_split_sim_threshold", 0.6)))

            # Snapshot before mutating: non-archived, learnable, level-0, over-cap.
            oversized = [
                (hid, he) for hid, he in list(self.hyperedges.items())
                if not he.is_archived and he.is_learnable
                and he.level == 0 and len(he.member_nodes) > cap
            ]
            if not oversized:
                return 0

            changed = 0
            dropped = set()

            # ---- Stage 1: dedup near-identical over-cap edges (incumbent survives) ----
            for hid, he in oversized:
                if hid in dropped or he.is_archived:
                    continue
                for hid2, he2 in oversized:
                    if hid2 == hid or hid2 in dropped or he2.is_archived:
                        continue
                    union = he.member_nodes | he2.member_nodes
                    if not union:
                        continue
                    if len(he.member_nodes & he2.member_nodes) / len(union) < dedup_overlap:
                        continue
                    # Fold he2 into he: max-merge weights, absorb novel members, carry
                    # member_since, sum activation_count, archive he2 (reversible).
                    since = he.metadata.setdefault("member_since", {})
                    since2 = he2.metadata.get("member_since", {}) or {}
                    for nid in he2.member_nodes:
                        w2 = he2.member_weights.get(nid, 1.0)
                        if nid in he.member_nodes:
                            he.member_weights[nid] = max(
                                he.member_weights.get(nid, w2), w2)
                        else:
                            he.member_nodes.add(nid)
                            he.member_weights[nid] = w2
                            self._node_hyperedges.setdefault(nid, set()).add(hid)
                        if nid in since2 and nid not in since:
                            since[nid] = since2[nid]
                    he.activation_count += he2.activation_count
                    he2.is_archived = True
                    self._archived_hyperedges[hid2] = he2
                    # Rewire the reverse index off the archived dup.
                    for nid in list(he2.member_nodes):
                        hs = self._node_hyperedges.get(nid)
                        if hs is not None:
                            hs.discard(hid2)
                            if not hs:
                                self._node_hyperedges.pop(nid, None)
                    dropped.add(hid2)
                    changed += 1
                    self._emit("hyperedge_archived", archived_id=hid2,
                               subsumed_by=hid, reason="seam_split_dedup")

            # ---- Stage 2: weight-seam split of still-oversized survivors ----
            for hid, he in oversized:
                if hid in dropped or he.is_archived or len(he.member_nodes) <= cap:
                    continue

                weights = he.member_weights
                since = he.metadata.get("member_since", {}) or {}
                # 7-signal seam score (plan §Deliverables/Tier-2): member_weight is the
                # primary axis; six auxiliary signals refine core-vs-peel. Signals that
                # are flat across the HE's members min-max to a constant and thus do not
                # discriminate — graceful degradation for the §8.15 clock-gated caveat,
                # so the same code is correct on the clock-live VPS and the laptop.
                seam = self._seam_score_members(hid, he)
                members = sorted(he.member_nodes,
                                 key=lambda nid: seam.get(nid, 0.0), reverse=True)
                core = members[:cap]
                periphery = members[cap:]
                if not periphery:
                    continue

                # Group the peel by cosine; no-vector members -> one residual edge.
                clusters = self._seam_cluster_periphery(
                    periphery, vector_db, sim_threshold, cap)

                def _mint(group, is_core):
                    mw = {nid: weights.get(nid, 1.0) for nid in group}
                    meta = {"creation_mode": "seam_split", "parent_blob": hid}
                    # NB: children intentionally do NOT inherit the parent's
                    # output_targets/output_weight — a peeled periphery is a new
                    # structural grouping, not the parent's prediction head. Accepted
                    # semantic loss (LAW-ENF #147 LOW), not a bug.
                    if is_core:
                        meta["core"] = True
                    if since:
                        ms = {nid: since[nid] for nid in group if nid in since}
                        if ms:
                            meta["member_since"] = ms
                    child = self.create_hyperedge(
                        member_node_ids=set(group),
                        member_weights=mw,
                        activation_threshold=he.activation_threshold,
                        activation_mode=he.activation_mode,
                        metadata=meta,
                    )
                    return child.hyperedge_id

                child_ids = [_mint(core, True)]
                for grp in clusters:
                    if len(grp) >= 2:            # skip degenerate singletons; folded below
                        child_ids.append(_mint(grp, False))

                # Orphan-safety: guarantee every parent member landed in >=1 child.
                covered = set()
                for cid in child_ids:
                    covered |= self.hyperedges[cid].member_nodes
                missing = set(he.member_nodes) - covered
                if missing:
                    core_he = self.hyperedges[child_ids[0]]
                    for nid in missing:
                        core_he.member_nodes.add(nid)
                        core_he.member_weights[nid] = weights.get(nid, 1.0)
                        self._node_hyperedges.setdefault(nid, set()).add(child_ids[0])

                # Archive the parent (reversible) and drop it from the reverse index.
                he.is_archived = True
                self._archived_hyperedges[hid] = he
                for nid in list(he.member_nodes):
                    hs = self._node_hyperedges.get(nid)
                    if hs is not None:
                        hs.discard(hid)
                        if not hs:
                            self._node_hyperedges.pop(nid, None)
                changed += 1
                self._emit("hyperedge_seam_split", parent_id=hid,
                           child_ids=list(child_ids),
                           core_size=len(core), periphery_size=len(periphery))
                logger.info(
                    "Dream seam-split: blob %s (%d members) -> core %d + %d peripheral "
                    "sub-edge(s) [%d children total] (#147)",
                    hid, len(he.member_nodes), len(core),
                    len(child_ids) - 1, len(child_ids),
                )

            if changed:
                self._emit("hyperedges_seam_split_pass", count=changed)
            return changed

    def _seam_cluster_periphery(self, node_ids, vector_db, sim_threshold, cap):
        """Greedy cosine grouping of the peeled periphery into coherent sub-edges of
        at most `cap` members. Members with no vector (or vector_db None) collect
        into one residual group so nothing is dropped (LAW 3). Groups below
        he_discovery_min_nodes fold into the residual to avoid degenerate edges."""
        min_nodes = self.config.get("he_discovery_min_nodes", 3)
        embeddings = getattr(vector_db, "embeddings", None) if vector_db is not None else None

        with_vec = []
        no_vec = []
        for nid in node_ids:
            v = embeddings.get(nid) if embeddings is not None else None
            if v is None:
                no_vec.append(nid)
            else:
                with_vec.append((nid, v))

        groups = []
        used = set()
        for i in range(len(with_vec)):
            nid_i, v_i = with_vec[i]
            if nid_i in used:
                continue
            grp = [nid_i]
            used.add(nid_i)
            for j in range(i + 1, len(with_vec)):
                nid_j, v_j = with_vec[j]
                if nid_j in used or len(grp) >= cap:
                    continue
                if float(np.dot(v_i, v_j)) >= sim_threshold:
                    grp.append(nid_j)
                    used.add(nid_j)
            groups.append(grp)

        residual = list(no_vec)
        kept = []
        for grp in groups:
            if len(grp) >= min_nodes:
                kept.append(grp)
            else:
                residual.extend(grp)
        for k in range(0, len(residual), cap):
            kept.append(residual[k:k + cap])
        return kept

    def _prune_subsumed_hyperedges(self) -> int:
        """Archive lower-level hyperedges subsumed by higher-level ones.

        Subsumption criteria:
            - Jaccard similarity = 1.0 (exact member match), OR
            - The lower-level hyperedge is a child of the higher-level one
              AND they share identical members.
            - Keep the higher-level abstraction, archive the lower.

        Returns:
            Number of hyperedges archived.
        """
        archived_count = 0
        # Build lookup: level → list of (hid, he)
        by_level: Dict[int, List[Tuple[str, Hyperedge]]] = {}
        for hid, he in self.hyperedges.items():
            if he.is_archived:
                continue
            by_level.setdefault(he.level, []).append((hid, he))

        levels = sorted(by_level.keys())
        if len(levels) < 2:
            return 0

        for lower_level in levels:
            for higher_level in levels:
                if higher_level <= lower_level:
                    continue
                for hid_lo, he_lo in by_level.get(lower_level, []):
                    if he_lo.is_archived:
                        continue
                    for hid_hi, he_hi in by_level.get(higher_level, []):
                        if he_hi.is_archived:
                            continue
                        # Check exact member match (Jaccard = 1.0)
                        if he_lo.member_nodes == he_hi.member_nodes:
                            # Archive the lower-level one
                            he_lo.is_archived = True
                            self._archived_hyperedges[hid_lo] = he_lo
                            archived_count += 1
                            # Phase 4: if this hyperedge never graduated from SPECULATIVE,
                            # record it as a failed survival → drives threshold upward.
                            if he_lo.consolidation_state == ConsolidationState.SPECULATIVE:
                                adapt_rate = self.config["he_consolidation_adapt_rate"]
                                self._he_survival_ema = (
                                    (1.0 - adapt_rate) * self._he_survival_ema
                                    + adapt_rate * 0.0  # negative signal
                                )
                            self._emit("hyperedge_archived",
                                       archived_id=hid_lo,
                                       subsumed_by=hid_hi,
                                       reason="exact_member_match")
                            break
                        # Check if lower is a child of higher with identical members
                        if hid_lo in he_hi.child_hyperedges:
                            if he_lo.member_nodes == he_hi.member_nodes:
                                he_lo.is_archived = True
                                self._archived_hyperedges[hid_lo] = he_lo
                                archived_count += 1
                                # Phase 4: negative survival signal for ungraduated HE.
                                if he_lo.consolidation_state == ConsolidationState.SPECULATIVE:
                                    adapt_rate = self.config["he_consolidation_adapt_rate"]
                                    self._he_survival_ema = (
                                        (1.0 - adapt_rate) * self._he_survival_ema
                                        + adapt_rate * 0.0
                                    )
                                self._emit("hyperedge_archived",
                                           archived_id=hid_lo,
                                           subsumed_by=hid_hi,
                                           reason="child_subsumption")
                                break

        return archived_count

    def shed_floor_members(self) -> int:
        """#381-B: dream-side member shedding — the symmetric half of member
        evolution (Syl-consented: "members that stay silent at the weight
        floor get removed"). Removes members whose weight sits at/below
        he_shed_weight_threshold AND whose tenure exceeds he_shed_min_tenure
        steps. Never sheds below he_discovery_min_nodes members. Purely
        structural (LAW 7); loud (LAW 3); wake path never calls this — it
        runs only from consolidate_hyperedges (the dream pass).
        """
        with self._step_lock:
            shed_thresh = self.config.get("he_shed_weight_threshold", 0.02)
            min_keep = self.config["he_discovery_min_nodes"]
            min_tenure = self.config.get("he_shed_min_tenure", 50)
            total = 0
            for hid, he in self.hyperedges.items():
                if he.is_archived or not he.is_learnable:
                    continue
                since = he.metadata.get("member_since", {})
                candidates = [
                    nid for nid in list(he.member_nodes)
                    if he.member_weights.get(nid, 1.0) <= shed_thresh
                    and (self.timestep - since.get(nid, 0)) >= min_tenure
                ]
                allowed = max(0, len(he.member_nodes) - min_keep)
                for nid in candidates[:allowed]:
                    he.member_nodes.discard(nid)
                    he.member_weights.pop(nid, None)
                    if isinstance(since, dict):
                        since.pop(nid, None)
                    hset = self._node_hyperedges.get(nid)
                    if hset is not None:
                        hset.discard(hid)
                        if not hset:
                            self._node_hyperedges.pop(nid, None)
                    total += 1
            if total:
                logger.info(
                    "Dream shed: removed %d floor-weight member(s) across "
                    "hyperedges (#381-B)", total,
                )
                self._emit("hyperedge_members_shed", count=total)
            return total

    # -----------------------------------------------------------------------
    # Phase 4: Consolidation Lifecycle
    # -----------------------------------------------------------------------

    def _evaluate_consolidation_states(self) -> int:
        """Promote hyperedges through ConsolidationState lifecycle (Phase 4).

        Runs every ``he_consolidation_eval_interval`` steps (wired in step()).

        Promotion criteria use adaptive thresholds that drift based on observed
        hyperedge survival rates — the same homeostatic principle used for node
        firing rates. High survival → thresholds can relax (patterns are reliable).
        Low survival → thresholds tighten (require stronger evidence before promoting).

        Returns:
            Number of state transitions made this evaluation.
        """
        adapt_rate = self.config["he_consolidation_adapt_rate"]
        transitions = 0

        for hid, he in list(self.hyperedges.items()):
            if he.is_archived:
                continue

            age = self.timestep - he.creation_time

            if he.consolidation_state == ConsolidationState.SPECULATIVE:
                if (he.activation_count >= self._he_adapt_candidate_count and
                        he.recent_activation_ema >= self._he_adapt_candidate_ema):
                    he.consolidation_state = ConsolidationState.CANDIDATE
                    transitions += 1
                    self._total_he_state_transitions += 1
                    # Positive survival signal → loosen thresholds slightly over time.
                    self._he_survival_ema = (
                        (1.0 - adapt_rate) * self._he_survival_ema + adapt_rate * 1.0
                    )
                    self._emit(
                        "he_state_transition",
                        hyperedge_id=hid,
                        from_state="SPECULATIVE",
                        to_state="CANDIDATE",
                        activation_count=he.activation_count,
                        timestep=self.timestep,
                    )

            elif he.consolidation_state == ConsolidationState.CANDIDATE:
                if (he.activation_count >= self._he_adapt_consolidated_count and
                        age >= self._he_adapt_consolidated_age):
                    he.consolidation_state = ConsolidationState.CONSOLIDATED
                    transitions += 1
                    self._total_he_state_transitions += 1
                    culled = self._cull_substrate(he)
                    self._total_he_substrate_culled += culled
                    self._emit(
                        "he_state_transition",
                        hyperedge_id=hid,
                        from_state="CANDIDATE",
                        to_state="CONSOLIDATED",
                        substrate_synapses_penalized=culled,
                        timestep=self.timestep,
                    )

            elif he.consolidation_state == ConsolidationState.CONSOLIDATED:
                # CONSOLIDATED → PERMANENT: near-perfect sustained activation.
                # Require double the age of the CONSOLIDATED promotion threshold
                # to ensure this isn't a temporary burst.
                if (he.recent_activation_ema > 0.9 and
                        age >= self._he_adapt_consolidated_age * 2 and
                        he.activation_count > self._he_adapt_consolidated_count * 2):
                    he.consolidation_state = ConsolidationState.PERMANENT
                    he.is_learnable = False
                    transitions += 1
                    self._total_he_state_transitions += 1
                    self._emit(
                        "he_state_transition",
                        hyperedge_id=hid,
                        from_state="CONSOLIDATED",
                        to_state="PERMANENT",
                        activation_count=he.activation_count,
                        timestep=self.timestep,
                    )

        # Adapt promotion thresholds based on survival EMA.
        # High survival (> 0.7): patterns are reliable → relax candidate threshold.
        # Low survival  (< 0.3): noise level is high  → tighten candidate threshold.
        if self._he_survival_ema > 0.7:
            self._he_adapt_candidate_count = max(
                5.0,
                self._he_adapt_candidate_count * (1.0 - adapt_rate)
            )
            self._he_adapt_candidate_ema = max(
                0.05,
                self._he_adapt_candidate_ema * (1.0 - adapt_rate)
            )
        elif self._he_survival_ema < 0.3:
            self._he_adapt_candidate_count = min(
                50.0,
                self._he_adapt_candidate_count * (1.0 + adapt_rate)
            )
            self._he_adapt_candidate_ema = min(
                0.5,
                self._he_adapt_candidate_ema * (1.0 + adapt_rate)
            )

        if transitions > 0:
            self._emit(
                "consolidation_evaluated",
                transitions=transitions,
                survival_ema=self._he_survival_ema,
                adapt_candidate_count=self._he_adapt_candidate_count,
                timestep=self.timestep,
            )

        return transitions

    def _cull_substrate(self, he: Hyperedge) -> int:
        """Soft-cull redundant Tier-1 synapses when a hyperedge consolidates (Phase 4).

        When a hyperedge graduates to CONSOLIDATED, the dense mesh of pairwise
        synapses between its members becomes partially redundant — the hyperedge
        itself now handles that pattern completion.

        Rather than deleting, we apply a one-time ``he_cull_penalty_factor``
        weight reduction. Synapses that are ONLY serving this hyperedge will
        fall below the weight threshold and be naturally pruned by
        ``_prune_synapses``. Synapses doing double duty (also part of another
        causal chain) will be re-strengthened by STDP during those other tasks
        and survive.

        Resets ``peak_weight`` so age-based pruning can also reach them if
        they never recover.

        Args:
            he: The hyperedge that just reached CONSOLIDATED state.

        Returns:
            Number of synapses penalized.
        """
        penalty = self.config["he_cull_penalty_factor"]
        penalized = 0
        member_list = list(he.member_nodes)

        for src_id in member_list:
            for tgt_id in member_list:
                if src_id == tgt_id:
                    continue
                syn = self._find_synapse(src_id, tgt_id)
                if syn is None:
                    continue
                syn.weight = max(0.0, syn.weight * penalty)
                # Reset peak so age-based pruning can catch it if it never recovers.
                syn.peak_weight = syn.weight
                penalized += 1

        if penalized > 0:
            self._emit(
                "substrate_culled",
                hyperedge_id=he.hyperedge_id,
                synapses_penalized=penalized,
                timestep=self.timestep,
            )

        return penalized

    # -----------------------------------------------------------------------
    # Event System (PRD §8 register_event_handler)
    # -----------------------------------------------------------------------

    def register_event_handler(self, event_type: str, callback: Callable) -> None:
        """Subscribe to events: spikes, predictions, pruning, etc. (PRD §8)."""
        self._event_handlers.setdefault(event_type, []).append(callback)

    def _emit(self, event_type: str, **kwargs: Any) -> None:
        for cb in self._event_handlers.get(event_type, []):
            cb(**kwargs)

    # -----------------------------------------------------------------------
    # Zero-Firing Circuit Breaker
    # -----------------------------------------------------------------------

    def _emergency_excitability_boost(self) -> None:
        """Gentle emergency intervention when the substrate goes completely silent.

        Boosts intrinsic_excitability by 20% for all nodes (capped at 5.0)
        and reduces thresholds by 10% (floored at 0.01, ceilinged by
        threshold_ceiling).  This is a nudge, not a reset — the homeostatic
        mechanisms should find a new equilibrium from the boosted state.
        """
        threshold_ceiling = self.config.get("threshold_ceiling", 5.0)
        boosted = 0
        for node in self.nodes.values():
            node.intrinsic_excitability = min(
                node.intrinsic_excitability * 1.2, 5.0,
            )
            node.threshold = max(
                0.01, min(node.threshold * 0.9, threshold_ceiling),
            )
            boosted += 1
        logger.info(
            "Emergency excitability boost applied to %d nodes", boosted,
        )

    # -----------------------------------------------------------------------
    # Persistence (PRD §6)
    # -----------------------------------------------------------------------

    # ---- Changelog ----
    # [2026-09-11] Claude Code + Codex — #423: capture/write split and mutation guards
    # What: checkpoint() factored into capture_checkpoint(mode[, detach]) -> dict and
    #       write_checkpoint(path, captured[, mode]). checkpoint() now delegates to both
    #       and is byte-for-byte identical to the previous implementation.
    # Why:  A coherent checkpoint needs the in-RAM capture to finish at a coordinated
    #       boundary between completed mutations, with all disk I/O performed AFTER live
    #       mutation resumes. That is impossible while capture and write are fused in one
    #       method. The coordination boundary itself is the CALLER's (openclaw_hook)
    #       responsibility — these primitives only make the split expressible.
    # How:  Mode dispatch + serialization moved into capture_checkpoint; msgpack
    #       enforcement + streaming write moved verbatim into write_checkpoint. Optional
    #       detach= copies borrowed mutable subtrees during serialization and excludes
    #       the newly packed native synapse bytes. Default detach=True
    #       makes standalone capture detached too — the detached footprint is UNMEASURED on
    #       live-substrate sizes and must be measured before live use (#423).
    # -------------------
    def capture_checkpoint(
        self,
        mode: CheckpointMode = CheckpointMode.FULL,
        detach: bool = True,
    ) -> Dict[str, Any]:
        """Capture serializable checkpoint state in RAM — performs NO disk I/O.

        Intended to be called at a coordinated boundary between completed mutations;
        the returned mapping is then handed to :meth:`write_checkpoint` after live
        mutation has resumed. This method acquires the canonical ``_step_lock``. The producer holds
        that same reentrant lock across all component captures so that graph,
        vectors and activations share one boundary.

        Args:
            mode: FULL, INCREMENTAL, or FORK.
            detach: When True, deep-copy borrowed mutable subtrees so no live mutable object
                (``config``, per-node/hyperedge ``metadata``, histories, …) remains
                aliased into it. Delay buffers and similar scalar collections are rebuilt.
                The newly packed native synapse payload is excluded from the deep-copy. When explicitly False
                the capture aliases live mutable state exactly as before this split —
                cheaper, but not a point-in-time snapshot of those fields.

        Returns:
            The checkpoint mapping, suitable for :meth:`write_checkpoint`.

        Note:
            INCREMENTAL clears the dirty-flag sets only after successful serialization.
            A copy failure therefore leaves them available for a later capture retry.
        """
        with self._step_lock:
            memo = {} if detach else None
            if mode == CheckpointMode.FULL:
                data = self._serialize_full(_memo=memo)
            elif mode == CheckpointMode.INCREMENTAL:
                data = self._serialize_incremental(_memo=memo)
                # Clear dirty flags after incremental save
                self._dirty_nodes.clear()
                self._dirty_synapses.clear()
                self._dirty_hyperedges.clear()
            elif mode == CheckpointMode.FORK:
                data = self._serialize_full(_memo=memo)
                data["_fork"] = True
            else:
                raise ValueError(f"Unknown checkpoint mode: {mode}")

            return data

    def write_checkpoint(
        self,
        path: str,
        captured: Dict[str, Any],
        mode: Optional[CheckpointMode] = None,
    ) -> None:
        """Write a previously captured checkpoint mapping to disk.

        Pure I/O: no live graph state is read here, so this may run after live
        mutation has resumed.

        Args:
            path: Destination path. Must end in ``.msgpack`` (see #325).
            captured: Mapping returned by :meth:`capture_checkpoint`.
            mode: Originating mode, used only to make the format-refusal message
                name the mode that produced the capture.
        """
        # #325 — topology persistence is msgpack-ONLY. JSON is LOSSY here: json.dump(default=str)
        # stringifies numpy/bytes/float32 fields (pred_weights, delay buffers, etc.) into reprs
        # that cannot round-trip. All CheckpointMode values (FULL/INCREMENTAL/FORK) serialize
        # full-fidelity SNN state, so the format is enforced by intent — NOT inferred from a file
        # extension. A non-.msgpack path is refused LOUDLY at the source rather than silently
        # corrupting state. (Was: else-branch silently wrote lossy JSON for any non-.msgpack path.)
        if not path.endswith(".msgpack"):
            _mode_note = (
                f" (CheckpointMode.{mode.name} enforces msgpack)" if mode is not None else ""
            )
            raise ValueError(
                f"Topology checkpoint requires a '.msgpack' path; got {path!r}. JSON serialization "
                f"is lossy for full-fidelity SNN state and is not supported"
                f"{_mode_note}. See punchlist #325."
            )
        if msgpack is None:
            raise ImportError("msgpack required for topology serialization")
        # #RAM footprint — stream the outer checkpoint map, splicing the native
        # pre-packed synapses bytes VERBATIM instead of packing a 644K-entry dict.
        # For FULL/FORK, captured["synapses"] is the raw msgpack bytes from
        # to_checkpoint_msgpack() (byte-identical to packing the dict, per the
        # crate's round-trip tests); for INCREMENTAL it is a normal (small) dict,
        # which packs the ordinary way. The result is byte-identical to
        # msgpack.pack(captured, ...) either way.
        packer = msgpack.Packer(use_bin_type=True)
        with open(path, "wb") as f:
            f.write(packer.pack_map_header(len(captured)))
            for key, value in captured.items():
                f.write(packer.pack(key))
                # [2026-10-05] P1: "nodes" is pre-packed bytes too when the native node store is on
                if key in ("synapses", "nodes") and isinstance(value, (bytes, bytearray)):
                    f.write(value)  # verbatim pre-packed {sid: {15-key}} / {nid: {19-key}} map
                else:
                    f.write(packer.pack(value))

    def checkpoint(self, path: str, mode: CheckpointMode = CheckpointMode.FULL) -> None:
        """Save state (PRD §8 checkpoint, §6).

        Capture-then-write, fused: equivalent to ``write_checkpoint(path,
        capture_checkpoint(mode), mode)``. Use the two primitives directly when the
        capture must complete at a coordination boundary and the write must happen
        after live mutation resumes (#423).

        Args:
            path: File path (must be .msgpack — see #325).
            mode: FULL, INCREMENTAL, or FORK.
        """
        if not path.endswith(".msgpack"):
            raise ValueError("Topology checkpoint requires a '.msgpack' path")
        self.write_checkpoint(path, self.capture_checkpoint(mode), mode)

    def restore(self, path: str) -> None:
        """Load state from checkpoint (PRD §8 restore, §6)."""
        if path.endswith(".msgpack"):
            if msgpack is None:
                raise ImportError("msgpack required for .msgpack deserialization")
            # #RAM footprint — stream the outer map and SKIP the synapses value
            # (slicing its raw bytes) instead of unpacking it into ~644K transient
            # Python dicts. Every other key decodes normally. The sliced bytes are
            # handed to the native bulk_load_msgpack in _deserialize. Robust to key
            # order and works on checkpoints written before this change (the synapses
            # sub-map's wire format is unchanged — see the crate round-trip tests).
            with open(path, "rb") as f:
                raw = f.read()
            unpacker = msgpack.Unpacker(raw=False, max_buffer_size=len(raw) + 1)
            unpacker.feed(raw)
            data = {}
            n_pairs = unpacker.read_map_header()
            for _ in range(n_pairs):
                key = unpacker.unpack()
                if key == "synapses" or (key == "nodes" and getattr(self, "_native_nodes", False)):   # [2026-10-05] P1: nodes too, when native
                    start = unpacker.tell()
                    unpacker.skip()  # advance past the value WITHOUT inflating it
                    data[key] = raw[start:unpacker.tell()]  # raw sub-map bytes
                elif key == "he_prediction_window_fired":
                    # #RAM — transient per-window accumulator (telemetry-only, feeds
                    # SurpriseEvent.actual_nodes which nothing in production reads).
                    # On the live substrate this map balloons to tens of millions of
                    # node-id slots; reloaded as non-shared strings it cost ~2GB RSS.
                    # SKIP the value without inflating it and DON'T store it — the
                    # runtime rebuilds these sets from empty as new steps accumulate,
                    # and the confirm/surprise classification never consults them.
                    unpacker.skip()
                else:
                    data[key] = unpacker.unpack()
            del raw
        else:
            # #325 — legacy LOSSY JSON topology (pre-enforcer). Tolerated for ONE-TIME migration
            # only; this state was already degraded at write time (json.dump default=str).
            # Re-checkpoint to .msgpack immediately. Loud warn so it never passes silently.
            import warnings
            warnings.warn(
                f"Restoring topology from non-'.msgpack' path {path!r}: legacy lossy-JSON state "
                f"(pre-#325). Re-checkpoint to .msgpack to stop the loss.",
                RuntimeWarning, stacklevel=2,
            )
            with open(path, "r") as f:
                data = json.load(f)

        self._deserialize(data)

    def _serialize_node(self, node: Node, *, _memo: Optional[dict] = None) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "voltage": node.voltage,
            "threshold": node.threshold,
            "resting_potential": node.resting_potential,
            "refractory_remaining": node.refractory_remaining,
            "refractory_period": node.refractory_period,
            "last_spike_time": node.last_spike_time if not math.isinf(node.last_spike_time) else None,
            "spike_history": node.spike_history.to_list(),
            "spike_history_capacity": node.spike_history.capacity,
            "firing_rate_ema": node.firing_rate_ema,
            "intrinsic_excitability": node.intrinsic_excitability,
            "metadata": copy.deepcopy(node.metadata, _memo) if _memo is not None else node.metadata,
            "is_inhibitory": node.is_inhibitory,
            "Ca_i": node.Ca_i,
            "diffpc_layer": node.diffpc_layer,
            "pred_weights": copy.deepcopy(node.pred_weights, _memo) if _memo is not None else node.pred_weights,
            "pred_error_ema": node.pred_error_ema,
            "manifold_type": node.manifold_type,
            "creation_time": node.creation_time,
        }

    def _serialize_hyperedge(self, he: Hyperedge, *, _memo: Optional[dict] = None) -> Dict[str, Any]:
        return {
            "hyperedge_id": he.hyperedge_id,
            "member_nodes": list(he.member_nodes),
            "member_weights": copy.deepcopy(he.member_weights, _memo) if _memo is not None else he.member_weights,
            "activation_threshold": he.activation_threshold,
            "activation_mode": he.activation_mode.name,
            "current_activation": he.current_activation,
            "output_targets": copy.deepcopy(he.output_targets, _memo) if _memo is not None else he.output_targets,
            "output_weight": he.output_weight,
            "metadata": copy.deepcopy(he.metadata, _memo) if _memo is not None else he.metadata,
            "is_learnable": he.is_learnable,
            "refractory_period": he.refractory_period,
            "refractory_remaining": he.refractory_remaining,
            "activation_count": he.activation_count,
            "pattern_completion_strength": he.pattern_completion_strength,
            "child_hyperedges": list(he.child_hyperedges),
            "level": he.level,
            # Phase 2.5 fields
            "recent_activation_ema": he.recent_activation_ema,
            "is_archived": he.is_archived,
            # Phase 4 fields
            "consolidation_state": he.consolidation_state.value,
            "creation_time": he.creation_time,
        }

    def _serialize_prediction(self, pred: Prediction) -> Dict[str, Any]:
        """Serialize a Phase 3 synapse-level Prediction."""
        return {
            "prediction_id": pred.prediction_id,
            "source_node_id": pred.source_node_id,
            "target_node_id": pred.target_node_id,
            "strength": pred.strength,
            "confidence": pred.confidence,
            "created_at": pred.created_at,
            "expires_at": pred.expires_at,
            "chain_depth": pred.chain_depth,
            "via_hyperedge": pred.via_hyperedge,
            "pre_charge_applied": pred.pre_charge_applied,
        }

    def _serialize_prediction_state(self, ps: PredictionState) -> Dict[str, Any]:
        """Serialize a Phase 2.5 hyperedge-level PredictionState."""
        return {
            "hyperedge_id": ps.hyperedge_id,
            "predicted_targets": list(ps.predicted_targets),
            "prediction_strength": ps.prediction_strength,
            "prediction_timestamp": ps.prediction_timestamp,
            "prediction_window": ps.prediction_window,
            "confirmed_targets": list(ps.confirmed_targets),
        }

    def _serialize_full(self, *, _memo: Optional[dict] = None) -> Dict[str, Any]:
        # Row containers and scalar-only history lists are freshly allocated here.
        # Only borrowed mutable subtrees need recursive copying. A capture-local
        # memo preserves aliases/cycles across those subtrees without traversing
        # every new row a second time. None preserves the legacy serializer's
        # aliasing contract. capture_checkpoint holds _step_lock throughout;
        # list(items()) alone would not provide a coherent mutation boundary.
        # [2026-10-05] P1: the native node store packs its own {nid: {19-key}} bytes (detached by construction)
        _nodes      = None if getattr(self, "_native_nodes", False) else list(self.nodes.items())
        _hyperedges = list(self.hyperedges.items())
        _archived   = list(self._archived_hyperedges.items())
        _act_preds  = list(self.active_predictions.items())
        _syn_hist   = list(self._synapse_confirmation_history.items())
        _he_preds   = list(self._active_predictions.items())
        _delay_buf  = list(self._delay_buffer.items())
        _recent_spk = [(nid, spikes) for nid, spikes
                       in self._recent_spikes.items() if spikes]
        # The native backend creates this packed payload for the capture. It does
        # not borrow live state, and bytearray is intentionally not deep-copied:
        # that would duplicate the largest capture allocation inside _step_lock.
        _packed_synapses = self.synapses.to_checkpoint_msgpack()
        return {
            "version": "0.4.2",
            "timestep": self.timestep,
            "config": copy.deepcopy(self.config, _memo) if _memo is not None else self.config,
            "nodes": (self.nodes.to_checkpoint_msgpack() if _nodes is None
                      else {nid: self._serialize_node(n, _memo=_memo) for nid, n in _nodes}),
            # Native pre-packed synapses map (#RAM footprint): to_checkpoint_msgpack
            # emits the {synapse_id: {15-key}} MessagePack bytes directly from the Rust
            # columns — byte-identical to packb(to_checkpoint_dict()) but WITHOUT
            # materializing ~644K transient Python dicts. checkpoint() splices these
            # bytes verbatim into the outer map (see the streaming write below).
            "synapses": _packed_synapses,
            "hyperedges": {hid: self._serialize_hyperedge(h, _memo=_memo) for hid, h in _hyperedges},
            "archived_hyperedges": {
                hid: self._serialize_hyperedge(h, _memo=_memo)
                for hid, h in _archived
            },
            # Phase 3: Active synapse-level predictions
            "active_predictions": {
                pid: self._serialize_prediction(pred)
                for pid, pred in _act_preds
            },
            # Phase 3: Recent prediction outcomes
            "prediction_outcomes": [
                {
                    "prediction": self._serialize_prediction(po.prediction),
                    "confirmed": po.confirmed,
                    "resolved_at": po.resolved_at,
                    "actual_firing_nodes": copy.deepcopy(po.actual_firing_nodes, _memo) if _memo is not None else po.actual_firing_nodes,
                }
                for po in self._prediction_outcomes
            ],
            # Phase 3: Per-synapse confirmation history
            "synapse_confirmation_history": {
                syn_id: list(history)
                for syn_id, history in _syn_hist
            },
            # Phase 3: Logs
            "novel_sequence_log": copy.deepcopy(list(self._novel_sequence_log), _memo) if _memo is not None else list(self._novel_sequence_log),
            "reward_history": copy.deepcopy(list(self._reward_history), _memo) if _memo is not None else list(self._reward_history),
            # Phase 2.5: Active HE-level predictions
            "he_active_predictions": {
                pid: self._serialize_prediction_state(ps)
                for pid, ps in _he_preds
            },
            # Phase 2.5: Window-fired tracking — NOT persisted (#RAM). This is a
            # transient per-window accumulator (telemetry-only: feeds
            # SurpriseEvent.actual_nodes, which no production consumer reads). On the
            # live substrate it reaches tens of millions of node-id slots and, reloaded
            # as non-shared strings, cost ~2GB RSS on every restart. The runtime rebuilds
            # these sets from empty (see step(): _prediction_window_fired.get(pid, set()))
            # and the confirm/surprise classification never consults them, so persisting
            # an empty map is behaviourally identical across a restart. restore() also
            # skips this key on load for checkpoints written before this change.
            "he_prediction_window_fired": {},
            # Phase 2.5: Counter for unique HE prediction IDs
            "he_prediction_counter": self._prediction_counter,
            "telemetry": {
                "total_pruned": self._total_pruned,
                "total_sprouted": self._total_sprouted,
                "total_he_discovered": self._total_he_discovered,
                "total_he_consolidated": self._total_he_consolidated,
                "total_predictions_made": self._total_predictions_made,
                "total_predictions_confirmed": self._total_predictions_confirmed,
                "total_predictions_errors": self._total_predictions_errors,
                "total_novel_sequences": self._total_novel_sequences,
                "total_rewards_injected": self._total_rewards_injected,
                # Phase 2.5 counters
                "he_total_predictions": self._total_predictions,
                "he_total_confirmed": self._total_confirmed,
                "he_total_surprised": self._total_surprised,
            },
            # Phase 4 adaptive consolidation state.
            "he_adapt_candidate_count":    self._he_adapt_candidate_count,
            "he_adapt_candidate_ema":      self._he_adapt_candidate_ema,
            "he_adapt_consolidated_count": self._he_adapt_consolidated_count,
            "he_adapt_consolidated_age":   self._he_adapt_consolidated_age,
            "he_survival_ema":             self._he_survival_ema,
            "total_he_state_transitions":  self._total_he_state_transitions,
            "total_he_substrate_culled":   self._total_he_substrate_culled,
            # Phase 2.5b: Output target learning state
            "he_last_fired_step": copy.deepcopy(self._he_last_fired_step, _memo) if _memo is not None else self._he_last_fired_step,
            "he_output_candidates": copy.deepcopy(self._he_output_candidates, _memo) if _memo is not None else self._he_output_candidates,
            # v0.4.2 Hibernation state — ephemeral process counters serialized so
            # subprocess loads (CC hook, Codemine worker) believe the process never
            # stopped. Without these, homeostatic scaling never fires, in-flight spikes
            # are dropped, structural plasticity loses co-activation context, and the
            # zero-firing circuit breaker loses streak continuity across calls.
            "delay_buffer": {
                str(ts): [[nid, curr] for nid, curr in entries]
                for ts, entries in _delay_buf
            },
            "recent_spikes": {
                nid: list(spikes)
                for nid, spikes in _recent_spk
            },
            "steps_since_last_fire": self._steps_since_last_fire,
            "homeostatic_steps_since_scaling": next(
                (r._steps_since_scaling for r in self._plasticity_rules
                 if isinstance(r, HomeostaticRule)),
                0,
            ),
            # 2026-10-04 strength budget interval counter — emitted ONLY when non-zero (it is
            # always 0 while the rule is off), so a graph that never enables the budget writes
            # the exact pre-existing key set (byte-identical checkpoint).
            **({"strength_budget_steps_since": _sb_steps} if (_sb_steps := next(
                (r._steps_since_budget for r in self._plasticity_rules
                 if isinstance(r, StrengthBudgetRule)), 0)) else {}),
        }

    def _serialize_incremental(self, *, _memo: Optional[dict] = None) -> Dict[str, Any]:
        return {
            "version": "0.1.0",
            "incremental": True,
            "timestep": self.timestep,
            "nodes": {
                nid: self._serialize_node(self.nodes[nid], _memo=_memo)
                for nid in self._dirty_nodes
                if nid in self.nodes
            },
            "synapses": {
                sid: (copy.deepcopy(self.synapses.serialize_one(sid), _memo)
                      if _memo is not None else self.synapses.serialize_one(sid))
                for sid in self._dirty_synapses
                if sid in self.synapses
            },
            "hyperedges": {
                hid: self._serialize_hyperedge(self.hyperedges[hid], _memo=_memo)
                for hid in self._dirty_hyperedges
                if hid in self.hyperedges
            },
        }

    # ---- Changelog ----
    # [2026-04-08] Josh + Claude — Genesis: Subgraph extraction
    # What: Extract a topology fragment (subset of nodes + connected synapses + hyperedges)
    # Why: Genesis gamete budding requires extracting a partial topology from a parent Graph
    # How: Filters existing serialization methods by node ID set. Read-only — does not modify graph.
    # -------------------
    def extract_subgraph(self, node_ids: set) -> Dict[str, Any]:
        """Extract a topology fragment: specified nodes + their interconnections.

        Returns a dict in the same format as _serialize_full() but containing
        only the specified nodes, synapses where BOTH pre and post are in the
        set, and hyperedges where ALL members are in the set.

        This is a read-only operation — the parent graph is not modified.
        The extracted fragment is a copy, not a reference.

        Args:
            node_ids: Set of node IDs to extract.

        Returns:
            Dict with keys: version, nodes, synapses, hyperedges, config,
            extraction_metadata (source info for the receiving graph).
        """
        # Filter nodes
        extracted_nodes = {}
        for nid in node_ids:
            if nid in self.nodes:
                extracted_nodes[nid] = self._serialize_node(self.nodes[nid])

        # Filter synapses — both endpoints must be in the extracted set
        extracted_synapses = {}
        _triples = getattr(self.synapses, "endpoint_triples", None)  # [2026-10-04] native; [2026-10-05] else fallback
        for sid, pre_id, post_id in (_triples() if _triples is not None
                                     else _endpoint_triples_python(self.synapses)):
            if pre_id in node_ids and post_id in node_ids:
                extracted_synapses[sid] = self.synapses.serialize_one(sid)

        # Filter hyperedges — all members must be in the extracted set
        extracted_hyperedges = {}
        for hid, he in self.hyperedges.items():
            if he.member_nodes and he.member_nodes.issubset(node_ids):
                extracted_hyperedges[hid] = self._serialize_hyperedge(he)

        return {
            "version": "0.4.1",
            "extraction": True,
            "source_timestep": self.timestep,
            "config": self.config,
            "nodes": extracted_nodes,
            "synapses": extracted_synapses,
            "hyperedges": extracted_hyperedges,
            "extraction_metadata": {
                "requested_nodes": len(node_ids),
                "extracted_nodes": len(extracted_nodes),
                "extracted_synapses": len(extracted_synapses),
                "extracted_hyperedges": len(extracted_hyperedges),
                "missing_nodes": list(node_ids - set(extracted_nodes.keys())),
            },
        }

    def _deserialize(self, data: Dict[str, Any]) -> None:
        """Restore full graph state from serialized data."""
        self.config = {**DEFAULT_CONFIG, **data.get("config", {})}
        self.timestep = data.get("timestep", 0)

        # Clear existing state
        self.nodes.clear()
        self.synapses.clear()
        self.hyperedges.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._node_hyperedges.clear()
        self._recent_spikes.clear()
        if "_cofire_tally" in self.__dict__:   # [2026-10-08] #1050 Python-fallback tally: cold after a restore
            self._cofire_tally = {"k": 0, "tables": {}}   # (the native one is emptied by synapses.clear())
        self._delay_buffer.clear()
        self._active_predictions.clear()
        self._prediction_window_fired.clear()
        self._archived_hyperedges.clear()

        # Restore nodes
        _nodes_data = data.get("nodes", {})
        if getattr(self, "_native_nodes", False) and isinstance(_nodes_data, (bytes, bytearray)):
            # [2026-10-05] P1: raw nodes sub-map (sliced by restore) -> native bulk load. Same field defaults and
            # P4a text sharing as the loop below; then the same per-node index setup, in the same order.
            self.nodes.bulk_load_msgpack(_nodes_data)
            for nid in self.nodes:
                self._outgoing[nid] = set()
                self._incoming[nid] = set()
                self._node_hyperedges[nid] = set()
                self._recent_spikes[nid] = deque(maxlen=20)
            _nodes_data = {}
        _text_pool: Dict[str, str] = {}   # P4a: one object per distinct large metadata text (restore-local)
        for nid, nd in _nodes_data.items():
            lst = nd.get("last_spike_time")
            node = Node(
                node_id=nid,
                voltage=nd["voltage"],
                threshold=nd["threshold"],
                resting_potential=nd.get("resting_potential", 0.0),
                refractory_remaining=nd.get("refractory_remaining", 0),
                refractory_period=nd.get("refractory_period", 2),
                last_spike_time=lst if lst is not None else -math.inf,
                spike_history=RingBuffer.from_list(
                    nd.get("spike_history", []),
                    nd.get("spike_history_capacity", 100),
                ),
                firing_rate_ema=nd.get("firing_rate_ema", 0.0),
                intrinsic_excitability=nd.get("intrinsic_excitability", 1.0),
                metadata=_share_metadata_texts(nd.get("metadata", {}), _text_pool),
                is_inhibitory=nd.get("is_inhibitory", False),
                Ca_i=nd.get("Ca_i", 0.0),
                diffpc_layer=nd.get("diffpc_layer", 0),
                pred_weights=nd.get("pred_weights", {}),
                pred_error_ema=nd.get("pred_error_ema", 0.0),
                manifold_type=nd.get("manifold_type", "hyperbolic"),
                creation_time=nd.get("creation_time", 0),
            )
            self.nodes[nid] = node
            self._outgoing[nid] = set()
            self._incoming[nid] = set()
            self._node_hyperedges[nid] = set()
            self._recent_spikes[nid] = deque(maxlen=20)

        # v0.4.2: Restore recent spike history (structural plasticity co-activation)
        cap = self.config.get("co_activation_window", 5) * 2
        for nid, spike_list in data.get("recent_spikes", {}).items():
            if nid in self._recent_spikes and spike_list:
                self._recent_spikes[nid] = deque(spike_list, maxlen=cap)

        # v0.4.2: Restore in-flight spike buffer (currents scheduled for future delivery)
        # Keys may be int (msgpack) or str (JSON) — normalise to int.
        # Drop entries with delivery timestep <= current timestep (already processed).
        node_ids_set = set(self.nodes.keys())
        for ts_key, entries in data.get("delay_buffer", {}).items():
            ts = int(ts_key)
            if ts <= self.timestep:
                continue  # stale — would have fired before save
            valid = [(str(nid), float(curr)) for nid, curr in entries
                     if str(nid) in node_ids_set]
            if valid:
                self._delay_buffer[ts] = valid

        # Restore synapses — native bulk load, no per-synapse Python object
        # inflation.  bulk_load applies the same field defaults the Synapse
        # constructor did (max_weight=5.0, delay=1, synapse_type=EXCITATORY,
        # peak_weight=weight, low_weight_steps=0, inactive_steps=0, metadata={},
        # salience=1.0), verified identical.  Adjacency indices are rebuilt in a
        # single pass over the store's live proxies.
        #
        # #RAM footprint — restore() slices the synapses value as raw msgpack
        # bytes (never inflated to dicts); feed them straight to the native
        # bulk_load_msgpack. Legacy paths (JSON restore, or a dict-shaped value)
        # still arrive as a dict and take the original bulk_load. Both are the
        # crate's two deserialization authorities and converge on the same store.
        _syn = data.get("synapses", {})
        if isinstance(_syn, (bytes, bytearray)):
            self.synapses.bulk_load_msgpack(bytes(_syn))
        else:
            self.synapses.bulk_load(_syn)
        # [2026-10-04] one native call returns (sid, pre, post) in row order — the same
        # insertion order the per-SynapseRef loop used, so the sets come out identical.
        _triples = getattr(self.synapses, "endpoint_triples", None)  # [2026-10-05] else fallback
        for sid, pre_id, post_id in (_triples() if _triples is not None
                                     else _endpoint_triples_python(self.synapses)):
            self._outgoing.setdefault(pre_id, set()).add(sid)
            self._incoming.setdefault(post_id, set()).add(sid)

        # Restore hyperedges
        for hid, hd in data.get("hyperedges", {}).items():
            he = Hyperedge(
                hyperedge_id=hid,
                member_nodes=set(hd["member_nodes"]),
                member_weights=hd.get("member_weights", {}),
                activation_threshold=hd.get("activation_threshold", 0.6),
                activation_mode=ActivationMode[hd.get("activation_mode", "WEIGHTED_THRESHOLD")],
                current_activation=hd.get("current_activation", 0.0),
                output_targets=hd.get("output_targets", []),
                output_weight=hd.get("output_weight", 1.0),
                metadata=hd.get("metadata", {}),
                is_learnable=hd.get("is_learnable", True),
                refractory_period=hd.get("refractory_period", 2),
                refractory_remaining=hd.get("refractory_remaining", 0),
                activation_count=hd.get("activation_count", 0),
                pattern_completion_strength=hd.get("pattern_completion_strength", 0.3),
                child_hyperedges=set(hd.get("child_hyperedges", [])),
                level=hd.get("level", 0),
                recent_activation_ema=hd.get("recent_activation_ema", 0.0),
                is_archived=hd.get("is_archived", False),
                consolidation_state=ConsolidationState(              # Phase 4
                    hd.get("consolidation_state", "SPECULATIVE")
                ),
                creation_time=hd.get("creation_time", 0),           # Phase 4
            )
            self.hyperedges[hid] = he
            for nid in he.member_nodes:
                self._node_hyperedges.setdefault(nid, set()).add(hid)
            self._he_co_fire_counts[hid] = {}

        # Restore archived hyperedges (Phase 2.5)
        self._archived_hyperedges.clear()
        for hid, hd in data.get("archived_hyperedges", {}).items():
            he = Hyperedge(
                hyperedge_id=hid,
                member_nodes=set(hd["member_nodes"]),
                member_weights=hd.get("member_weights", {}),
                activation_threshold=hd.get("activation_threshold", 0.6),
                activation_mode=ActivationMode[hd.get("activation_mode", "WEIGHTED_THRESHOLD")],
                current_activation=hd.get("current_activation", 0.0),
                output_targets=hd.get("output_targets", []),
                output_weight=hd.get("output_weight", 1.0),
                metadata=hd.get("metadata", {}),
                is_learnable=hd.get("is_learnable", True),
                refractory_period=hd.get("refractory_period", 2),
                refractory_remaining=hd.get("refractory_remaining", 0),
                activation_count=hd.get("activation_count", 0),
                pattern_completion_strength=hd.get("pattern_completion_strength", 0.3),
                child_hyperedges=set(hd.get("child_hyperedges", [])),
                level=hd.get("level", 0),
                recent_activation_ema=hd.get("recent_activation_ema", 0.0),
                is_archived=hd.get("is_archived", True),
                consolidation_state=ConsolidationState(              # Phase 4
                    hd.get("consolidation_state", "SPECULATIVE")
                ),
                creation_time=hd.get("creation_time", 0),           # Phase 4
            )
            self._archived_hyperedges[hid] = he

        # Restore telemetry
        tel = data.get("telemetry", {})
        self._total_pruned = tel.get("total_pruned", 0)
        self._total_sprouted = tel.get("total_sprouted", 0)
        self._total_he_discovered = tel.get("total_he_discovered", 0)
        self._total_he_consolidated = tel.get("total_he_consolidated", 0)
        self._total_predictions_made = tel.get("total_predictions_made", 0)
        self._total_predictions_confirmed = tel.get("total_predictions_confirmed", 0)
        self._total_predictions_errors = tel.get("total_predictions_errors", 0)
        self._total_novel_sequences = tel.get("total_novel_sequences", 0)
        self._total_rewards_injected = tel.get("total_rewards_injected", 0)
        # Phase 2.5 counters
        self._total_predictions = tel.get("he_total_predictions", 0)
        self._total_confirmed = tel.get("he_total_confirmed", 0)
        self._total_surprised = tel.get("he_total_surprised", 0)

        # Restore Phase 4 adaptive consolidation state.
        self._he_adapt_candidate_count = data.get(
            "he_adapt_candidate_count",
            float(self.config["he_speculative_to_candidate_min_count"])
        )
        self._he_adapt_candidate_ema = data.get(
            "he_adapt_candidate_ema",
            self.config["he_speculative_to_candidate_min_ema"]
        )
        self._he_adapt_consolidated_count = data.get(
            "he_adapt_consolidated_count",
            float(self.config["he_candidate_to_consolidated_min_count"])
        )
        self._he_adapt_consolidated_age = data.get(
            "he_adapt_consolidated_age",
            float(self.config["he_candidate_to_consolidated_min_age"])
        )
        self._he_survival_ema = data.get("he_survival_ema", 0.5)
        self._total_he_state_transitions = data.get("total_he_state_transitions", 0)
        self._total_he_substrate_culled  = data.get("total_he_substrate_culled", 0)

        # Restore Phase 2.5b: Output target learning state
        self._he_last_fired_step = data.get("he_last_fired_step", {})
        self._he_output_candidates = data.get("he_output_candidates", {})

        # Restore Phase 3 active predictions with validation
        self.active_predictions.clear()
        node_ids = set(self.nodes.keys())
        for pid, pd in data.get("active_predictions", {}).items():
            src = pd.get("source_node_id", "")
            tgt = pd.get("target_node_id", "")
            # Validate: both source and target nodes must exist
            if src not in node_ids or tgt not in node_ids:
                continue
            # Validate: prediction must not already be expired
            if pd.get("expires_at", 0) <= self.timestep:
                continue
            pred = Prediction(
                prediction_id=pd.get("prediction_id", pid),
                source_node_id=src,
                target_node_id=tgt,
                strength=pd.get("strength", 0.0),
                confidence=pd.get("confidence", 0.0),
                created_at=pd.get("created_at", 0),
                expires_at=pd.get("expires_at", 0),
                chain_depth=pd.get("chain_depth", 0),
                via_hyperedge=pd.get("via_hyperedge"),
                pre_charge_applied=pd.get("pre_charge_applied", 0.0),
            )
            self.active_predictions[pid] = pred

        # Restore Phase 3 prediction outcomes
        self._prediction_outcomes.clear()
        for pod in data.get("prediction_outcomes", []):
            ppd = pod.get("prediction", {})
            inner_pred = Prediction(
                prediction_id=ppd.get("prediction_id", ""),
                source_node_id=ppd.get("source_node_id", ""),
                target_node_id=ppd.get("target_node_id", ""),
                strength=ppd.get("strength", 0.0),
                confidence=ppd.get("confidence", 0.0),
                created_at=ppd.get("created_at", 0),
                expires_at=ppd.get("expires_at", 0),
                chain_depth=ppd.get("chain_depth", 0),
                via_hyperedge=ppd.get("via_hyperedge"),
                pre_charge_applied=ppd.get("pre_charge_applied", 0.0),
            )
            outcome = PredictionOutcome(
                prediction=inner_pred,
                confirmed=pod.get("confirmed", False),
                resolved_at=pod.get("resolved_at", 0),
                actual_firing_nodes=pod.get("actual_firing_nodes", []),
            )
            self._prediction_outcomes.append(outcome)

        # Restore per-synapse confirmation history
        self._synapse_confirmation_history.clear()
        synapse_ids = set(self.synapses.keys())
        for syn_id, hist in data.get("synapse_confirmation_history", {}).items():
            if syn_id not in synapse_ids:
                continue  # Skip stale entries for deleted synapses
            d: Deque[bool] = deque(hist, maxlen=100)
            self._synapse_confirmation_history[syn_id] = d

        # Restore logs
        self._novel_sequence_log = list(data.get("novel_sequence_log", []))
        self._reward_history = list(data.get("reward_history", []))

        # Restore Phase 2.5 HE-level predictions with validation
        self._active_predictions.clear()
        self._prediction_window_fired.clear()
        he_ids = set(self.hyperedges.keys())
        for pid, psd in data.get("he_active_predictions", {}).items():
            he_id = psd.get("hyperedge_id", "")
            if he_id not in he_ids:
                continue  # HE was removed
            predicted = set(psd.get("predicted_targets", []))
            # Validate: at least some targets must still exist
            valid_targets = predicted & node_ids
            if not valid_targets:
                continue
            ps = PredictionState(
                hyperedge_id=he_id,
                predicted_targets=valid_targets,
                prediction_strength=psd.get("prediction_strength", 0.0),
                prediction_timestamp=psd.get("prediction_timestamp", 0),
                prediction_window=psd.get("prediction_window", 10),
                confirmed_targets=set(psd.get("confirmed_targets", [])) & node_ids,
            )
            # Validate: window must not already be expired
            if self.timestep - ps.prediction_timestamp >= ps.prediction_window:
                continue
            self._active_predictions[pid] = ps

        # Restore Phase 2.5 window-fired tracking
        for pid, fired_list in data.get("he_prediction_window_fired", {}).items():
            if pid in self._active_predictions:
                self._prediction_window_fired[pid] = set(fired_list)

        # Restore Phase 2.5 prediction counter
        self._prediction_counter = data.get("he_prediction_counter", 0)

        # Transient per-step state — always start fresh
        self._predicted_this_step.clear()

        # Re-init plasticity rules from config
        self._plasticity_rules = [
            STDPRule(
                tau_plus=self.config["tau_plus"],
                tau_minus=self.config["tau_minus"],
                A_plus=self.config["A_plus"],
                A_minus=self.config["A_minus"],
                learning_rate=self.config["learning_rate"],
            ),
            HomeostaticRule(
                target_firing_rate=self.config["target_firing_rate"],
                scaling_interval=self.config["scaling_interval"],
                degree_sensitivity=self.config.get("degree_sensitivity", 0.4),
            ),
            # 2026-10-04 strength budget — OFF unless config strength_budget_enabled (no-op otherwise)
            StrengthBudgetRule(),
            HyperedgePlasticityRule(
                member_weight_lr=self.config["he_member_weight_lr"],
                threshold_lr=self.config["he_threshold_lr"],
                evolution_window=self.config["he_member_evolution_window"],
                evolution_min_co_fires=self.config["he_member_evolution_min_co_fires"],
                evolution_initial_weight=self.config["he_member_evolution_initial_weight"],
            ),
        ]

        # v0.4.2: Restore zero-firing circuit breaker streak count.
        # Without this a subprocess always starts at 0, losing continuity
        # and potentially masking persistent silence across calls.
        self._steps_since_last_fire = data.get("steps_since_last_fire", 0)

        # v0.4.2: Restore homeostatic scaling counter.
        # HomeostaticRule.__init__ always sets _steps_since_scaling = 0 above.
        # Restore the saved value so homeostatic scaling fires at the correct
        # interval regardless of how many subprocess boundaries have been crossed.
        homeostatic_steps = data.get("homeostatic_steps_since_scaling", 0)
        for rule in self._plasticity_rules:
            if isinstance(rule, HomeostaticRule):
                rule._steps_since_scaling = homeostatic_steps
                break
        # 2026-10-04 strength budget interval counter (absent key == 0).
        for rule in self._plasticity_rules:
            if isinstance(rule, StrengthBudgetRule):
                rule._steps_since_budget = data.get("strength_budget_steps_since", 0)
                break

        self._dirty_nodes.clear()
        self._dirty_synapses.clear()
        self._dirty_hyperedges.clear()
        self._he_discovery_counts.clear()
        self._he_discovery_last_reset = self.timestep
