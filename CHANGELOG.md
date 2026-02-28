# NeuroGraph Changelog

Detailed changelog for each implementation phase. For project instructions, see [CLAUDE.md](CLAUDE.md).

## Changelog

### Phase 1: Core Foundation (v0.1.0)
**Commit:** Initial implementation of `neuro_foundation.py`

**What was built:**
- `Node`, `Synapse`, `Hyperedge`, `Graph` dataclasses (PRD §2.2)
- `RingBuffer` for bounded spike history (capacity 100)
- Sparse SNN simulation loop: voltage decay → delayed spike delivery → fire detection → spike propagation → hyperedge evaluation → plasticity → structural plasticity → refractory decrement (PRD §2.2.4)
- `STDPRule`: LTP/LTD with weight-dependent soft-saturation, temporal aliasing at Δt=0 (PRD §3.1)
- `HomeostaticRule`: multiplicative synaptic scaling, intrinsic excitability, threshold adaptation (PRD §3.2)
- Structural plasticity: weight/activity/age-based pruning, co-activation sprouting with per-step cap of 10 (PRD §3.3)
- JSON and MessagePack serialization with FULL/INCREMENTAL/FORK checkpoint modes (PRD §6)
- Event system: `register_event_handler` / `_emit` for spikes, pruning, sprouting
- Three-factor learning via `inject_reward` with eligibility traces
- Causal chain tracing via `get_causal_chain`

**Key design decisions:**
- STDP uses weight-dependent soft-saturation: LTP scaled by `(max_weight - w) / max_weight` to prevent runaway potentiation (PRD §3.1.2)
- Spike history stored in fixed-capacity `RingBuffer` (default 100) to bound memory per node
- Homeostatic scaling is multiplicative (`w * ratio^factor`), NOT divisive normalization, to preserve learned weight distributions (PRD §3.2)
- Refractory counter skips decrement on the step a node fires (full N-step rest starts next step)
- Sprouting capped at 10 new synapses per step with pre-built edge-existence index for performance

**Tests (57 total):**
- `test_snn.py` (12): voltage decay, spike detection, refractory, propagation, 1K-node/10K-step stability
- `test_stdp.py` (9): LTP, LTD, weight-dependent STDP, temporal aliasing, asymmetry
- `test_hypergraph.py` (18): creation, 4 activation modes, output injection, node removal cascade
- `test_integration.py` (18): homeostatic regulation, pruning, serialization, causal chains, telemetry, events, reward, edge cases, memory stability

**Files created:**
- `neuro_foundation.py`, `requirements.txt`, `.gitignore`
- `tests/__init__.py`, `tests/test_snn.py`, `tests/test_stdp.py`, `tests/test_hypergraph.py`, `tests/test_integration.py`
- `examples/simple_usage.py`, `examples/project_configs.py`

---

### Phase 1.5: Stability Enhancements (v0.1.5)
**Commit:** Phase 1.5 stability enhancements

**What was built:**
1. **Changelog metadata in `Graph.__init__`** — `self.metadata["changelog"]` documents design decisions (STDP soft-saturation, RingBuffer usage, multiplicative scaling, hyperedge refractory) for auditability.
2. **Refractory period for `Hyperedge` class** — `refractory_period` (default 2) and `refractory_remaining` fields prevent cascading feedback loops where a hyperedge's output re-activates its own members on the very next step. Uses same skip-on-fire-step pattern as nodes.

**Key design decisions:**
- Hyperedge refractory uses identical pattern to node refractory: set counter on fire, skip decrement on firing step, decrement on subsequent steps
- Default refractory_period=2 for both nodes and hyperedges balances responsiveness with stability

**Tests added (2):**
- `test_hypergraph.py::TestHyperedgeRefractory::test_cannot_refire_during_refractory`
- `test_hypergraph.py::TestHyperedgeRefractory::test_custom_refractory_period`

---

### Phase 2: Hypergraph Engine (v0.2.0)
**Commit:** Implement Phase 2: Hypergraph Engine (PRD Section 4)

**What was built:**
1. **Enhanced activation dynamics** — All 4 modes (WEIGHTED_THRESHOLD, K_OF_N, ALL_OR_NONE, GRADED) with GRADED output scaling: `effective_weight = output_weight × activation_level` (PRD §4.2)
2. **Pattern completion** — When hyperedge fires from partial activation, inactive members get pre-charged: `voltage += completion_strength × member_weight × excitability` (PRD §4.2)
3. **`HyperedgePlasticityRule`** (PRD §4.3):
   - Member weight adaptation: active members during firing get `+lr`, inactive get `-lr×0.5`
   - Threshold learning via `inject_reward`: positive reward lowers threshold (more sensitive), negative raises (more strict), bounds [0.1, 1.0]
   - Member evolution: non-members that consistently co-fire (tracked via `_he_co_fire_counts`) get promoted with initial weight 0.3
4. **Hierarchical hyperedges** — `create_hierarchical_hyperedge()` composes child HEs into meta-HEs. Members = union of children's members. Level = max(child levels) + 1. Step processes level 0 first, then 1, etc. (PRD §4.4)
5. **Hyperedge discovery** — `discover_hyperedges()` tracks co-activation patterns. When a node group fires together `≥ min_co_fires` times within a window, a new hyperedge is auto-created with `creation_mode: "discovered"` metadata.
6. **Consolidation** — `consolidate_hyperedges()` merges same-level hyperedges with Jaccard similarity ≥ 0.8. Surviving HE gets union of members, lower threshold, summed activation counts.

**New Hyperedge fields:**
- `activation_count`, `pattern_completion_strength`, `child_hyperedges`, `level`

**New config keys:**
- `he_pattern_completion_strength`, `he_member_weight_lr`, `he_threshold_lr`
- `he_discovery_window`, `he_discovery_min_co_fires`, `he_discovery_min_nodes`
- `he_consolidation_overlap`, `he_member_evolution_window`, `he_member_evolution_min_co_fires`, `he_member_evolution_initial_weight`

**Tests added (26):**
- `test_hypergraph_phase2.py`: pattern completion (4), GRADED scaling (1), member weight adaptation (2), threshold learning (3), member evolution (2), hierarchical HEs (5), discovery (3), consolidation (4), activation count (1), serialization roundtrip (1)

**Files created:**
- `tests/test_hypergraph_phase2.py`, `examples/hypergraph_demo.py`

---

### Phase 2.5: Predictive Infrastructure Enhancements (v0.2.5)
**Commit:** Implement Phase 2.5: Predictive Infrastructure Enhancements

**What was built:**
1. **Prediction Error Events (HIGH PRIORITY):**
   - `PredictionState` dataclass: tracks hyperedge_id, predicted_targets, prediction_strength, timestamp, window, confirmed_targets
   - `SurpriseEvent` dataclass: hyperedge_id, expected_node, prediction_strength, actual_nodes, timestamp
   - When hyperedge with `output_targets` fires → `PredictionState` created in `_active_predictions`
   - Each step checks active predictions: target fired within window → `prediction_confirmed` event; window expired → `SurpriseEvent` emitted via `surprise` event
   - Counters: `_total_predictions`, `_total_confirmed`, `_total_surprised`
   - `StepResult` extended with `predictions_confirmed`, `predictions_surprised`

2. **Dynamic Pattern Completion (MEDIUM PRIORITY):**
   - Completion strength scales by experience: `effective = base_strength × min(1.0, activation_count / he_experience_threshold)`
   - New hyperedges (count=0) → zero completion; experienced (count ≥ 100) → full strength
   - Prevents aggressive pattern completion from unproven concepts
   - `recent_activation_ema` field on Hyperedge tracks firing rate EMA

3. **Cross-Level Consistency Pruning (LOW PRIORITY):**
   - `_prune_subsumed_hyperedges()` called after same-level merges in `consolidate_hyperedges()`
   - Subsumption: lower-level HE with identical members to higher-level HE → archived
   - Child subsumption: child of hierarchical HE with identical members → archived
   - `is_archived` flag on Hyperedge; archived HEs skipped in `step()` evaluation
   - `_archived_hyperedges` dict preserves archived HEs for debugging
   - `get_archived_hyperedges()` public accessor

4. **Telemetry updates:**
   - `prediction_accuracy`: confirmed / total outcomes (0 if no predictions resolved)
   - `surprise_rate`: surprised / total outcomes
   - `hyperedge_experience_distribution`: bucketed histogram {"0", "1-9", "10-99", "100+"}

**New Hyperedge fields:**
- `recent_activation_ema`, `is_archived`

**New config keys:**
- `prediction_window` (default 5), `prediction_ema_alpha` (default 0.01), `he_experience_threshold` (default 100)

**Serialization updates:**
- `_serialize_hyperedge` includes `recent_activation_ema`, `is_archived`
- `_serialize_full` includes `archived_hyperedges` dict, prediction counters in telemetry
- `_deserialize` restores archived HEs and prediction counters

**Tests added (26):**
- `test_phase25.py`: prediction creation (3), confirmation (2), surprise detection (3), dynamic completion (3), subsumption pruning (6), telemetry (4), serialization roundtrip (3), activation EMA (2)

**Files created:**
- `tests/test_phase25.py`

---

### Phase 3: Predictive Coding Engine (v0.3.0)
**Commit:** Implement Phase 3: Predictive Coding Engine (PRD Section 5)

**What was built:**
1. **Synapse-level prediction generation (PRD §5.1):**
   - `Prediction` dataclass: source_node_id, target_node_id, strength, confidence, created_at, expires_at, chain_depth, pre_charge_applied
   - `PredictionOutcome` dataclass: records confirmed/error results with actual firing nodes
   - When node fires with strong causal link (weight > `prediction_threshold`), prediction registered in `Graph.active_predictions` and target pre-charged by `strength × 0.3`
   - `get_predictions()` public API returns active predictions

2. **Prediction chains:**
   - `_generate_predictions_from_node()` recursively cascades predictions through learned sequences (A→B→C)
   - Strength decays ×0.7 per hop (`prediction_chain_decay`), max depth 3 (`prediction_max_chain_depth`)
   - Cycle-safe via `visited` set passed through recursion
   - `_predicted_this_step` set prevents double-predicting the same target from both synapse and hyperedge sources

3. **Prediction confirmation and error (PRD §5.1):**
   - `_evaluate_predictions()`: each step checks if predicted targets fired → `_on_prediction_confirmed()`
   - `_cleanup_predictions()`: expired predictions → `_on_prediction_error()`
   - Confirmation: weight += `prediction_confirm_bonus` × confidence (default 0.01)
   - Error: weight -= `prediction_error_penalty` × confidence (default 0.02)
   - Events emitted: `prediction_generated`, `prediction_confirmed`, `prediction_error`

4. **Surprise-driven exploration (PRD §5.2):**
   - `_surprise_exploration()`: when A→B prediction fails, examines what fired instead
   - If node C fired when B was expected: creates speculative synapse A→C (weight 0.1) tagged `{"creation_mode": "surprise_driven"}` in `Synapse.metadata`
   - `_has_learned_pattern()`: novelty detection — if alternative firing nodes have no strong connections from source, emits `novel_sequence` event
   - Novel sequences logged in `_novel_sequence_log` (bounded to 1000)

5. **Three-factor learning (PRD §5.2):**
   - `STDPRule._apply_dw()`: when `three_factor_enabled=True`, weight changes accumulate in `Synapse.eligibility_trace` instead of being applied directly
   - Eligibility traces decay each step: `trace *= exp(-1/τ)` with τ=100 (`eligibility_trace_tau`), only when three-factor is enabled
   - `inject_reward(strength, scope)`: commits traces as `Δw = trace × reward × learning_rate`. Optional `scope` (set of node IDs) limits which synapses are affected
   - Reward history tracked in `_reward_history` (bounded to 1000)

6. **Telemetry extensions:**
   - `Telemetry` dataclass: `prediction_accuracy`, `surprise_rate`, `active_predictions_count`, `total_predictions_made`, `total_predictions_confirmed`, `total_predictions_errors`, `total_novel_sequences`, `total_rewards_injected`
   - `get_telemetry()` computes accuracy as confirmed / (confirmed + errors)

7. **Synapse.metadata field:**
   - New `Dict[str, Any]` field on `Synapse` for application-specific data
   - Used by surprise exploration to tag creation mode
   - Serialized/deserialized with full state

**Key design decisions:**
- Prediction confidence = 60% weight factor + 40% confirmation history rate — balances link strength with empirical accuracy
- Pre-charge factor of 0.3 chosen to nudge targets toward threshold without guaranteeing firing — predictions are suggestions, not commands
- Chain predictions use a `visited` set rather than depth-only limiting — handles cyclic topologies safely without infinite recursion
- Three-factor mode modifies `STDPRule._apply_dw()` with a single branch rather than a separate rule class — preserves all existing STDP logic (weight-dependent scaling, temporal aliasing) without duplication
- Eligibility trace decay skipped entirely when `three_factor_enabled=False` — zero overhead for non-three-factor usage
- Prediction state bounded at 1000 active + 1000 outcomes — prevents memory leaks in long-running simulations without losing recent history

**New config keys:**
- `prediction_threshold` (3.0), `prediction_pre_charge_factor` (0.3), `prediction_window` (10), `prediction_chain_decay` (0.7), `prediction_max_chain_depth` (3)
- `prediction_confirm_bonus` (0.01), `prediction_error_penalty` (0.02), `prediction_max_active` (1000)
- `surprise_sprouting_weight` (0.1), `eligibility_trace_tau` (100), `three_factor_enabled` (False)

**Serialization updates:**
- `_serialize_synapse` includes `metadata` field
- `_serialize_full` telemetry includes prediction counters (`total_predictions_made`, `total_predictions_confirmed`, `total_predictions_errors`, `total_novel_sequences`, `total_rewards_injected`)
- `_deserialize` restores prediction counters and synapse metadata; clears transient prediction state

**Tests added (38):**
- `test_prediction.py`: prediction generation (5), chain predictions (5), confirmation (3), errors (3), surprise exploration (2), novelty detection (2), three-factor learning (4), reward scoping (2), state management (3), telemetry (2), sequence learning (2), get_predictions API (3), serialization (2)

**Files created:**
- `tests/test_prediction.py`, `examples/predictive_demo.py`

---

### Phase 3.5: Predictive State Persistence & Validation (v0.3.5)
**Commit:** Implement Phase 3.5: Predictive State Persistence & Validation

**What was built:**
1. **Active prediction serialization (CRITICAL FIX):**
   - `_serialize_prediction()`: serializes Phase 3 `Prediction` objects (all 10 fields)
   - `_serialize_prediction_state()`: serializes Phase 2.5 `PredictionState` objects (all 6 fields)
   - `_serialize_full()` now includes: `active_predictions`, `he_active_predictions`, `he_prediction_window_fired`, `he_prediction_counter`
   - Previously, active predictions were lost on checkpoint/restore — the system "forgot" what it was expecting

2. **Prediction support state serialization:**
   - `prediction_outcomes`: resolved prediction history (Prediction + confirmed/error + actual nodes)
   - `synapse_confirmation_history`: per-synapse boolean deques tracking confirmation rates (used for confidence calculations)
   - `novel_sequence_log`: bounded list of novel sequence events
   - `reward_history`: bounded list of reward injection events

3. **Validation on restore:**
   - Phase 3 predictions: dropped if `expires_at <= timestep` (expired) or source/target node missing
   - Phase 2.5 HE predictions: dropped if hyperedge removed, all target nodes removed, or window expired
   - Synapse confirmation history: entries for deleted synapses are dropped
   - Phase 2.5 window-fired tracking: only restored for predictions that survived validation
   - Ensures no dangling references after topology changes between checkpoint and restore

4. **Backward compatibility:**
   - Restoring a v0.2.5 checkpoint (no prediction fields) works cleanly — all prediction state defaults to empty
   - Version bumped to `0.3.5`

**Key design decisions:**
- Validation filters predictions at restore time rather than lazily during step — fail-fast prevents stale predictions from corrupting confidence calculations or emitting spurious events
- `_prediction_counter` is serialized to prevent HE prediction ID collisions after restore (Phase 3 uses UUIDs so no collision risk there)
- `PredictionOutcome.prediction` is serialized inline (not by reference) because the parent prediction may have been cleaned up from active_predictions by the time the outcome was recorded
- Confirmation history uses `deque(hist, maxlen=100)` on restore to maintain the bounded size invariant

**Serialization version:** `0.3.5` (up from `0.2.5`)

**Tests added (20):**
- `test_phase35.py`:
  - Phase 3 persistence: active predictions restored (1), field preservation (1), confirmable after restore (1), error after restore (1)
  - Phase 2.5 persistence: HE predictions restored (1), confirmable after restore (1), counter preserved (1), window-fired preserved (1)
  - Support state: outcomes (1), confirmation history (1), novel sequence log (1), reward history (1)
  - Validation: expired P3 dropped (1), deleted-node P3 dropped (1), expired HE dropped (1), deleted-HE dropped (1), stale confirmation history dropped (1)
  - Integration: version bump (1), full mid-flight roundtrip (1), backward-compatible restore (1)

**Files created:**
- `tests/test_phase35.py`

---

### Phase 4: Universal Ingestor System (v0.4.0)
**Commit:** Implement Phase 4: Universal Ingestor System (PRD Addendum §1-4)

**What was built:**
1. **SimpleVectorDB (PRD §7):**
   - In-memory vector database using numpy
   - L2-normalized storage for cosine similarity via dot product
   - `insert()`, `search()`, `get()`, `delete()`, `count()`, `all_ids()` API
   - Top-k search with configurable similarity threshold

2. **Five-Stage Ingestion Pipeline (PRD Addendum §2):**
   - **Stage 1 — Extract:** `ExtractorRouter` with auto-detection + 5 extractors:
     - `TextExtractor`: plain text passthrough
     - `MarkdownExtractor`: preserves heading hierarchy as structure entries
     - `CodeExtractor`: regex-based AST detection for Python/JS (functions, classes, async)
     - `URLExtractor`: HTML→text via BeautifulSoup (regex fallback)
     - `PDFExtractor`: page-aware text extraction via PyPDF2
   - **Stage 2 — Chunk:** `AdaptiveChunker` with 4 strategies:
     - SEMANTIC: paragraph/sentence boundary splitting (200-500 tokens)
     - CODE_AWARE: split by function/class definition boundaries
     - HIERARCHICAL: follows heading structure with parent-child relationships
     - FIXED_SIZE: sliding window (512 tokens, 64 overlap) fallback
   - **Stage 3 — Embed:** `EmbeddingEngine` with sentence-transformers/all-mpnet-base-v2:
     - Batch processing, LRU cache (10K entries), auto-normalization
     - Deterministic SHA-256 hash fallback for testing without model
   - **Stage 4 — Register:** `NodeRegistrar` with novelty dampening (PRD Addendum §4.2-4.3):
     - New nodes start at reduced `intrinsic_excitability` (dampening factor)
     - Higher initial `threshold` (configurable boost)
     - Probation period with 3 fade curves: LINEAR, EXPONENTIAL, LOGARITHMIC
     - `update_probation()` called per step graduates nodes to full weight
   - **Stage 5 — Associate:** `HypergraphAssociator` with 3 strategies:
     - Similarity-based: bidirectional synapses for chunks above similarity threshold
     - Structural: sequential links, parent→child links, code definition→usage links
     - Hypergraph clustering: greedy groups of similar nodes → hyperedges

3. **UniversalIngestor coordinator (PRD Addendum §4.1):**
   - `ingest(source, source_type)`: main entry point, executes all 5 stages
   - `ingest_batch()`: sequential multi-source ingestion
   - `update_probation()`: advance novelty dampening for all ingested nodes
   - `query_similar()`: vector similarity search over ingested content
   - `ingestion_log`: history of all ingestion results

4. **Project configurations (PRD Addendum §3):**
   - `OPENCLAW_INGESTOR_CONFIG`: code-aware chunking, novelty_dampening=0.3, similarity=0.7
   - `DSM_INGESTOR_CONFIG`: hierarchical chunking, novelty_dampening=0.05, similarity=0.8
   - `CONSCIOUSNESS_INGESTOR_CONFIG`: semantic chunking, novelty_dampening=0.01, similarity=0.65
   - `get_ingestor_config(project)` accessor

**Key design decisions:**
- Embedding engine falls back to deterministic SHA-256 hash-based vectors when sentence-transformers is unavailable — enables full test coverage without heavy model dependencies
- Novelty dampening modifies `intrinsic_excitability` (which scales incoming current in the SNN step loop) rather than adding a separate dampening pathway — leverages existing SNN infrastructure
- Similarity synapses are bidirectional (A→B and B→A) because semantic similarity is symmetric — causal directionality emerges later through STDP
- Code definition→usage links use text search rather than full AST analysis — pragmatic choice balancing accuracy with language-independence
- LRU cache on embeddings (10K entries) bounds memory while avoiding recomputation for repeated content during batch or multi-source ingestion
- Probation graduation restores both excitability and threshold in a single step — no partial graduation states

**New data structures:**
- `ExtractedContent`, `Chunk`, `EmbeddedChunk`, `IngestionResult`
- `SourceType`, `ChunkStrategy`, `DampeningCurve` enumerations
- `SimpleVectorDB`, `IngestorConfig`

**New dependencies:**
- `sentence-transformers>=2.2.0` (optional — hash fallback available)
- `beautifulsoup4>=4.12.0` (optional — regex fallback available)
- `PyPDF2>=3.0.0` (required only for PDF extraction)

**Tests added (88):**
- `test_ingestor.py`:
  - SimpleVectorDB: insert/get (1), normalization (1), search cosine (1), threshold (1), top-k (1), empty (1), delete (1), count (1), all_ids (1)
  - Extractors: text (2), markdown (3), code Python (2), code JS (1), URL/HTML (2), router (6)
  - Chunking: semantic (2), code-aware (1), hierarchical (2), fixed-size (1), empty (1), token estimation (1), fallback (1)
  - Embedding: vectors (1), deterministic (1), caching (1), single text (1), different texts (1), model name (1)
  - Novelty dampening: reduced excitability (1), boosted threshold (1), metadata (1), decrement (1), graduation (1), fade (1), vector DB (1), exponential curve (1)
  - Association: similarity (2), sequential (1), parent-child (1), code links (1), clustering (1)
  - End-to-end: text (1), markdown (1), code (1), empty (1), graph nodes (1), vector DB (1), log (1), query (1), probation (1), batch (1), sequential in pipeline (1)
  - Project configs: OpenClaw (4), DSM (4), Consciousness (3), get_config (2)
  - IngestorConfig: access (1), missing (1), get (1)
  - Edge cases: single word (1), whitespace (1), long input (1), unicode (1), accumulate (1), zero vector (1)

**Files created:**
- `universal_ingestor.py`
- `tests/test_ingestor.py`
- `examples/ingest_code.py`, `examples/ingest_document.py`, `examples/ingest_multi_source.py`

---

### Phase 5: Deployment & Modular Setup (v0.5.0)
**Commit:** Implement Phase 5: Deployment, OpenClaw integration, and migration framework

**What was built:**
1. **OpenClaw Integration Hook (`openclaw_hook.py`):**
   - `NeuroGraphMemory` singleton class wrapping Graph + UniversalIngestor + SimpleVectorDB
   - `get_instance()` / `reset_instance()` class methods for singleton lifecycle
   - `on_message(text)`: ingest content, run STDP step, update probation, auto-save every N messages
   - `recall(query, k, threshold)`: semantic similarity search over ingested knowledge
   - `save()`: force checkpoint to disk
   - `stats()`: telemetry dict with nodes, synapses, predictions, accuracy, etc.
   - `ingest_file(path)`: auto-detect source type from extension, ingest file content
   - `ingest_directory(directory, extensions, recursive)`: batch ingest matching files
   - `step(n)`: run N SNN learning steps without ingestion
   - OpenClaw-tuned SNN config: `learning_rate=0.02`, `tau=15.0`, fast novelty dampening

2. **Checkpoint Migration Framework (`neurograph_migrate.py`):**
   - Schema version registry: 0.1.0 → 0.2.0 → 0.2.5 → 0.3.0 → 0.3.5 → 0.4.0
   - `get_checkpoint_version(path)`: detect schema version of any checkpoint file
   - `get_checkpoint_info(path)`: detailed checkpoint metadata (version, counts, size)
   - `plan_migration(from_version, target_version)`: compute migration path
   - `migrate_data(data, target)`: in-memory migration with deep copy (original unchanged)
   - `upgrade_checkpoint(path, target, backup, dry_run)`: file-based upgrade with backup
   - `rollback_checkpoint(path, backup_path)`: restore from backup
   - `list_backups(path)`: enumerate all backup files for a checkpoint
   - 5 migration functions covering all schema transitions:
     - 0.1.0→0.2.0: adds hyperedge engine fields (activation_count, pattern_completion, etc.)
     - 0.2.0→0.2.5: adds predictive infrastructure (recent_activation_ema, archived HEs, etc.)
     - 0.2.5→0.3.0: adds predictive coding (eligibility_trace, synapse metadata, etc.)
     - 0.3.0→0.3.5: adds prediction persistence (active_predictions, outcomes, etc.)
     - 0.3.5→0.4.0: version bump only (no schema changes for Universal Ingestor)

3. **CLI Tools:**
   - `feed-syl`: ingestion/status CLI (`--status`, `--text`, `--file`, `--dir`, `--workspace`, `--query`, `--save`, `--step`, `--upgrade`)
   - `neurograph`: management CLI with subcommands (`setup`, `status`, `upgrade`, `rollback`, `info`, `backups`, `verify`)
   - `neurograph setup`: interactive wizard — checks deps, tests graph/ingestor, configures workspace
   - `neurograph verify`: installation health check with dependency status

4. **Deployment Script (`deploy.sh`):**
   - One-command installation: `./deploy.sh`
   - Installs dependencies (sentence-transformers from source, CPU PyTorch, etc.)
   - Deploys files to `~/.openclaw/skills/neurograph/`
   - Installs `feed-syl` to `~/.local/bin/`
   - Configures OpenClaw JSON (`openclaw.json` neurograph skill entry)
   - Verification step: checks file presence, Python imports, embedding backend
   - Flags: `--deps-only`, `--files-only`, `--uninstall`, `--verify`

5. **Skill Definition (`SKILL.md`):**
   - `autoload: true` for automatic OpenClaw integration
   - Documents environment variables, capabilities, CLI usage

**Key design decisions:**
- Singleton pattern ensures one NeuroGraphMemory per process — prevents checkpoint contention
- Migration uses deep copy to preserve original data, enabling safe rollback without file backup
- Backup files use timestamp suffixes (`.backup-{unix_time}`) for deterministic ordering
- `deploy.sh` uses `--break-system-packages` with fallback to standard pip — handles both bare-metal and virtualenv environments
- `feed-syl` adds both script directory and skill directory to `sys.path` — works whether run from repo or installed location
- Auto-save interval defaults to 10 messages — balances persistence safety with I/O overhead

**Tests added (50):**
- `test_migration.py` (32): version detection (4), migration planning (4), individual steps (5), full migration (4), file upgrade (5), backup/rollback (5), checkpoint info (2), Graph integration (3)
- `test_openclaw_hook.py` (18): singleton (2), message ingestion (5), recall (2), auto-save (2), stats (2), persistence (1), file ingestion (3), step (1)

**Files created:**
- `openclaw_hook.py`, `neurograph_migrate.py`, `neurograph`, `feed-syl`, `deploy.sh`, `SKILL.md`
- `tests/test_migration.py`, `tests/test_openclaw_hook.py`

---

### Phase 5.5: Management GUI (v0.5.5)
**Commit:** Implement Phase 5.5: NeuroGraph Management GUI

**What was built:**
1. **tkinter Management GUI (`neurograph_gui.py`):**
   - `GUIConfig`: manages `~/.neurograph/config.json` with defaults, directory creation
   - `FileWatcher`: watchdog-based inbox monitor with stability checking (waits for file write completion), polling fallback when watchdog unavailable, ignore patterns for hidden/temp/unsupported files
   - `GitUpdater`: git clone/fetch/pull with neurograph-patch integration for deployment, all operations on daemon threads, auto-rollback on validation failure
   - `IngestionManager`: wraps NeuroGraphMemory.ingest_file() with post-ingestion file movement to `ingested/YYYY-MM-DD/`, name collision handling, lazy NeuroGraphMemory initialization
   - `GUIMessageQueue`: thread-safe bridge (queue.Queue + root.after polling) for background thread → tkinter main loop communication
   - `NeuroGraphGUI`: main window with ttk.Notebook, four tabs:
     - **Status tab**: live telemetry grid (nodes, synapses, predictions, accuracy, etc.), auto-refresh every 5s, Save Checkpoint button
     - **Ingestion tab**: inbox path display, watcher ON/OFF toggle, inbox file Listbox, Ingest All/Selected/Add Files buttons, ingestion history Listbox
     - **Updates tab**: version and install info, Check for Updates button, Update Now button, scrolling update log
     - **Logs tab**: source selector (events.jsonl / gui.log), formatted event display, Refresh/Clear buttons
   - Settings dialog for editing all config values
   - Menu bar with File (Settings, Quit) and Help (About)

2. **Linux Desktop Entry (`neurograph.desktop`):**
   - Standard freedesktop.org `.desktop` file
   - Installed to `~/.local/share/applications/` for application launcher visibility
   - Exec path resolved at deploy time to point to installed `neurograph_gui.py`

3. **Directory structure (`~/.neurograph/`):**
   - `inbox/` — drop files here for auto-ingestion by watchdog
   - `ingested/YYYY-MM-DD/` — successfully ingested files moved here
   - `repo/` — shallow git clone for update mechanism
   - `logs/gui.log` — GUI activity log
   - `config.json` — persistent GUI settings
   - Intentionally separate from `~/.openclaw/neurograph/` to prevent GUI data from mixing with learned knowledge

**Key design decisions:**
- Non-GUI classes (`GUIConfig`, `FileWatcher`, `GitUpdater`, `IngestionManager`) defined before tkinter imports — testable in headless environments without a display server
- watchdog import guarded with try/except; polling fallback (2s interval) when unavailable — GUI works without the dependency, just less responsive
- Lazy `NeuroGraphMemory` initialization — avoids loading embeddings/checkpoint at startup if user only wants to check for updates
- GitUpdater imports neurograph-patch from the pulled repo (not installed copy) using `importlib.machinery.SourceFileLoader` — ensures latest MANIFEST and migration logic are always used for deployment
- All background operations (git, ingestion, file watching) run on daemon threads — GUI never blocks
- File stability checking: records file size, re-checks after 0.5s; only fires ingestion callback when size unchanged — handles partial writes, downloads-in-progress
- Post-ingestion files moved to dated subdirectories with automatic name collision resolution (append `_1`, `_2`, etc.)
- `~/.neurograph/` never deleted on uninstall — user's inbox and ingested files are preserved

**Deployment integration:**
- `deploy.sh` updated: installs `neurograph_gui.py` to skill dir, generates `.desktop` file with resolved Exec path, creates `~/.neurograph/` directories, installs watchdog dependency
- `neurograph-patch` MANIFEST updated: includes `neurograph_gui.py` and `neurograph.desktop` — future patches automatically deploy GUI updates
- `requirements.txt` updated: added `watchdog>=3.0.0`

**Tests added (25):**
- `test_gui.py`:
  - GUIConfig: defaults (1), custom default (1), set/get (1), save creates file (1), load existing (1), corrupt JSON fallback (1), ensure directories (1), save then reload (1)
  - FileWatcher: hidden files (1), temp files (1), unsupported extensions (1), supported extensions (1), no extension (1), start/stop lifecycle (1), stable file detection (1), hidden file ignored (1)
  - GitUpdater: clone when missing (1), skip if exists (1), get local commit (1), check no updates (1), check has updates (1), check error (1)
  - IngestionManager: move creates date dir (1), name collision (1), multiple collisions (1), success worker (1), error worker (1), batch ingest (1)
  - Integration: file drop pipeline (1)

**Files created:**
- `neurograph_gui.py`, `neurograph.desktop`
- `tests/test_gui.py`

**Files modified:**
- `neurograph-patch` (MANIFEST entries for GUI files)
- `deploy.sh` (GUI deployment, watchdog dep, .desktop installation)
- `requirements.txt` (watchdog>=3.0.0)
- `CLAUDE.md` (Phase 5.5 documentation)

---

### Phase 6: NG-Lite Canonical Source & Bridge (v0.6.0)
**Commit:** Add canonical NG-Lite learning substrate and NGSaaSBridge

**What was built:**
1. **NG-Lite (`ng_lite.py`) — Canonical lightweight learning substrate:**
   - Single-file vendorable design: numpy + stdlib only
   - `NGLiteNode`: pattern nodes with embedding hash, activation tracking, LRU metadata
   - `NGLiteSynapse`: weighted connections with Hebbian learning, success/failure counts
   - `NGLite` core class with full API:
     - `find_or_create_node(embedding)`: hash-first lookup, cosine similarity fallback, LRU pruning
     - `record_outcome(embedding, target_id, success)`: Hebbian weight update with soft saturation
     - `get_recommendations(embedding, top_k)`: ranked target retrieval by weight
     - `detect_novelty(embedding)`: cosine distance from known patterns
     - `save(filepath)` / `load(filepath)`: JSON persistence
   - `NGBridge` abstract interface for tier upgrades (Tier 1→2→3)
   - Bridge-first API: all methods check bridge before falling back to local
   - Bounded memory: configurable max_nodes (1000), max_synapses (5000) with LRU/weight pruning
   - Embedding hashing: SHA-256 of first 128 dims for O(1) exact-match lookup

2. **NGSaaSBridge (`ng_bridge.py`) — Tier 3 bridge implementation:**
   - Connects NG-Lite to full NeuroGraph Foundation via NeuroGraphMemory
   - `record_outcome()`: forwards outcomes as structured messages for SNN ingestion
   - `get_recommendations()`: semantic recall from full graph, extracts target IDs from content
   - `detect_novelty()`: cross-module novelty via full graph's vector DB
   - `sync_state()`: periodic state sync — ingests high-weight (>0.7) synapses into full SNN
   - Weight normalization: `[0,1] * max_weight ↔ [0,max_weight] / max_weight`
   - Disconnect/reconnect with automatic fallback to local learning

**Key design decisions:**
- **JSON for ng_lite, msgpack for neuro_foundation** — deliberate. NG-Lite state is small (≤1000 nodes), human-readable JSON aids debugging, and json is stdlib (no dependency). Full NeuroGraph has 20+ fields per node with spike histories and numpy arrays where msgpack's binary handling is 3-5x faster
- **Weight range [0,1] for ng_lite, [0,5] for full** — ng_lite's normalized range is simpler for Hebbian learning with soft saturation. Bridge handles translation. The ranges are equivalent under linear mapping
- **Incremental node IDs ("n_1") for ng_lite, UUIDs for full** — compactness vs global uniqueness. Bridge maintains mapping tables
- **Synapse keys as tuples, serialized as "src|tgt"** — JSON doesn't support tuple keys, pipe delimiter chosen for unambiguous splitting since node IDs never contain pipes
- **Bridge-first API pattern** — every method (recommendations, novelty, outcome) tries the bridge first, falls back to local on failure or disconnection. This makes tier transitions transparent to the consuming module

**Compatibility with The-Inference-Difference:**
- This is the canonical source for ng_lite.py (as declared in its docstring)
- Byte-for-byte compatible with the vendored copy in The-Inference-Difference
- router.py consumption pattern: `_classification_to_embedding()` → `record_outcome()` / `get_recommendations()` — works unchanged
- NGSaaSBridge is the upgrade path from Tier 1 (standalone) to Tier 3 (full SNN)

**Tests added (52):**
- `test_ng_lite.py`:
  - Node management: create (1), find existing (1), similar reuse (1), novel creates new (1), capacity pruning (1), incremental IDs (1)
  - Learning: success strengthens (1), failure weakens (1), approaches 1.0 (1), approaches 0.0 (1), bounded [0,1] (1), independent targets (1), success/failure counts (1), synapse pruning (1)
  - Recommendations: empty (1), sorted by weight (1), top_k limit (1)
  - Novelty: empty graph (1), known pattern (1), different pattern (1), similar pattern (1)
  - Persistence: save/load roundtrip (1), weight preservation (1), valid JSON (1), synapse key format (1), forward compatible (1), counters preserved (1)
  - Bridge interface: record outcome (1), recommendations (1), novelty (1), sync (1), disconnected fallback (1), connect/disconnect (1), failure fallback (1)
  - Stats: initial (1), after learning (1), memory estimate (1)
  - NGSaaSBridge: connected (1), disconnect/reconnect (1), record outcome (1), disconnected (1), recommendations (1), no results (1), novelty (1), empty graph novelty (1), sync state (1), sync disconnected (1), weight normalization (1), extract target (1)
  - Integration: full pipeline (1), bridge disconnect fallback (1)
  - Edge cases: zero vector (1), single dim (1), concurrent targets (1), empty module ID (1), version string (1)

**Files created:**
- `ng_lite.py`, `ng_bridge.py`
- `tests/test_ng_lite.py`

**Files modified:**
- `neurograph-patch` (MANIFEST entries for ng_lite.py and ng_bridge.py)
- `CLAUDE.md` (Phase 6 documentation)

---

### Phase 7: ET Module Manager Integration (v0.7.0)
**Commit:** ET Module Manager integration — NGPeerBridge, unified discovery, cross-module learning

**What was built:**
1. **NGPeerBridge (`ng_peer_bridge.py`) — Tier 2 cross-module learning:**
   - Implements `NGBridge` interface for peer-to-peer learning between co-located NG-Lite instances
   - File-based event exchange: each module writes JSONL events to `~/.et_modules/shared_learning/<module_id>.jsonl`
   - Asynchronous sync: reads peers' event files every N outcomes (default 100, NeuroGraph uses 50)
   - Incremental reads via tracked file positions — no re-reading on each sync
   - Cross-module recommendations via embedding cosine similarity (threshold 0.3)
   - Cross-module novelty detection: checks if a pattern is novel across ALL peer modules
   - Peer registry at `~/.et_modules/shared_learning/_peer_registry.json`
   - Bounded peer event cache (500 entries) with LRU eviction
   - `get_stats()` telemetry: sync count, cached events, connection status

2. **ET Module Manager (`et_modules/manager.py`):**
   - `ModuleManifest` dataclass: module identity, version, git remote, dependencies, install path
   - `ModuleStatus` dataclass: health, tier, update availability, peer bridge connectivity
   - `ETModuleManager` class with full lifecycle API:
     - `discover()`: scans known locations + registry for et_module.json manifests
     - `status()`: health check with tier assignment (1=isolated, 2=peer, 3=full SNN)
     - `register()`: add/update a module in the registry
     - `update_all()` / `update_module()`: git-pull updates with optional service restart
     - `get_peer_modules()`: find non-NeuroGraph modules for Tier 3 upgrade offers
     - `get_neurograph_path()`: peer modules use this to find the full SNN backend
     - `get_shared_learning_dir()`: returns path for NGPeerBridge coordination
   - Registry persistence at `~/.et_modules/registry.json`
   - Known install locations: `~/.openclaw/skills/neurograph`, `/opt/inference-difference`, `/opt/trollguard`, `~/.et_modules/modules`

3. **Module manifest (`et_module.json`):**
   - Declares NeuroGraph's identity: module_id="neurograph", version="0.6.0"
   - Git remote, branch, entry point, ng_lite_version for ecosystem coordination

4. **NeuroGraphMemory integration (`openclaw_hook.py`):**
   - NGPeerBridge initialized in `__init__` with graceful degradation (try/except)
   - `_write_peer_learning_event()`: after each ingestion, embeds the text and writes a learning event to the shared directory so peer modules can absorb patterns
   - `get_peer_modules()`: discovers co-located E-T Systems modules via ETModuleManager
   - `stats()` now includes `peer_bridge` status (connected, sync count, cached events)
   - Version bumped to 0.6.0 in stats output
   - Peer bridge disabled via `config={"peer_bridge": {"enabled": False}}`

5. **Deployment updates (`deploy.sh`):**
   - Deploys ng_lite.py, ng_bridge.py, ng_peer_bridge.py to skill directory
   - Deploys et_modules/ package (manager.py + __init__.py) and et_module.json
   - Creates `~/.et_modules/shared_learning/` directory
   - Registers NeuroGraph in `~/.et_modules/registry.json` with full manifest data
   - Uninstall preserves `~/.et_modules/` (shared learning data is cross-module state)

**Key design decisions:**
- **NeuroGraph is both Tier 2 and Tier 3** — it writes to the shared learning directory (Tier 2 peer participation) AND provides the full SNN backend (Tier 3 via NGSaaSBridge). This means peer modules get NeuroGraph's learning events even without a direct SaaS connection
- **sync_interval=50 for NeuroGraph** (vs default 100) — NeuroGraph processes more events per session than typical modules, so more frequent syncs keep peers up to date
- **Peer bridge initialized with try/except** — import failures (missing ng_peer_bridge.py in standalone installs) gracefully degrade to standalone mode without breaking core functionality
- **File-based exchange, not IPC** — shared JSONL files are dead simple, survive process restarts, work across languages, and let modules run on different schedules. No coordination daemon needed
- **Incremental file reads** — `_peer_read_positions` tracks byte offsets per peer file, so each sync only reads new events. Critical for performance as event logs grow
- **Bounded peer event cache** (500) — prevents memory growth from long-running peers with high event volumes

**Tests added (45):**
- `test_et_modules.py`:
  - ModuleManifest: defaults (1), from_file (1), missing file (1), invalid JSON (1), extra fields ignored (1), to_file (1), roundtrip (1)
  - ETModuleManager: init directories (1), empty registry (1), register (1), register+discover (1), status (1), tier assignment (1), neurograph tier 3 (1), peer modules (1), neurograph path (1), shared learning dir (1), update not registered (1), update no git (1)
  - NGPeerBridge: init shared dir (1), init registers (1), connected (1), disconnect/reconnect (1), record outcome writes (1), record disconnected (1), sync from peers (1), auto sync on interval (1), recommendations none (1), recommendations from peers (1), filters own module (1), novelty no peers (1), novelty known (1), novelty new (1), sync state (1), sync disconnected (1), stats (1), bounded events (1), incremental read (1)
  - NeuroGraphMemory integration: bridge initialized (1), on_message writes peer event (1), stats includes bridge (1), bridge disabled (1), works without bridge (1), get_peer_modules (1)
  - Manifest validation: neurograph et_module.json valid (1)

**Files created:**
- `ng_peer_bridge.py`, `et_module.json`
- `et_modules/__init__.py`, `et_modules/manager.py`
- `tests/test_et_modules.py`

**Files modified:**
- `openclaw_hook.py` (peer bridge integration, shared learning events, version bump)
- `deploy.sh` (NG-Lite ecosystem files, ET Module Manager registration, shared learning dir)
- `CLAUDE.md` (Phase 7 documentation)

---

### Phase 9: Cognitive Enhancement Suite (v0.9.0)
**Commit:** Implement Phase 9: Cognitive Enhancement Suite (CES)

**What was built:**
1. **CES Config (`ces_config.py`):**
   - `CESConfig` dataclass with four section dataclasses: `StreamingConfig`, `SurfacingConfig`, `PersistenceConfig`, `MonitoringConfig`
   - `load_ces_config(overrides, config_path)` factory with JSON file + dict override layering
   - All CES defaults centralised: Ollama model, chunk sizes, thresholds, HTTP port, etc.

2. **Stream Parser (`stream_parser.py`):**
   - `StreamParser` class with background daemon thread consuming from `queue.Queue`
   - Pipeline: `feed(text)` → `_chunk_text()` (overlapping word-level chunks) → `_embed_chunk()` (Ollama HTTP API) → `_find_similar()` (vector DB search) → `_nudge_nodes()` (voltage injection) → `_trigger_completions()` (hyperedge pattern completion)
   - Ollama availability detection with cached periodic re-check (60s interval)
   - Fallback chain: Ollama → sentence-transformers → hash embedder
   - Lifecycle: `pause()`, `resume()`, `stop()`, `is_running`, `get_stats()`

3. **Activation Persistence (`activation_persistence.py`):**
   - `ActivationPersistence` class writing JSON sidecar alongside msgpack checkpoint
   - `capture(graph)` → dict of node voltages, excitability, timestamps
   - `save(graph, checkpoint_path)` / `restore(graph, checkpoint_path)` with temporal decay
   - Exponential decay: `voltage *= (1 - decay_per_hour) ^ elapsed_hours`
   - Sub-threshold pruning, max_entries bounding, auto-save timer thread

4. **Surfacing Monitor (`surfacing.py`):**
   - `SurfacingMonitor` class with bounded priority queue (max 50 entries)
   - `after_step(step_result)` — scans fired nodes, filters by voltage threshold + min confidence
   - Composite scoring: 50% voltage, 30% excitability, 20% hyperedge membership
   - Per-step decay: `score *= decay_rate` with automatic cleanup of sub-threshold items
   - `format_context()` — renders surfaced items as `[NeuroGraph Surfaced Knowledge]` block

5. **CES Monitoring (`ces_monitoring.py`):**
   - `health_context(ng_memory)` — natural language status string for prompt injection
   - `CESLogger` — rotating file handler to `~/.neurograph/logs/ces.log` (size-based rotation)
   - `MonitoringDashboard` — HTTP server on port 8847 with `/health`, `/stats`, `/surfaced` endpoints (stdlib `http.server`, daemon thread)
   - `CESMonitor` coordinator: starts/stops dashboard + periodic health check logging

6. **OpenClaw integration (`openclaw_hook.py` modifications):**
   - CES modules initialized in `__init__` after peer bridge, all imports guarded by try/except
   - `on_message()`: feeds stream parser, calls surfacing monitor after step, includes `ces_surfaced` in return
   - `save()`: writes activation sidecar alongside checkpoint
   - `stats()`: includes CES subsystem status (stream_parser, surfacing, persistence, monitor)
   - `config={"ces": {"enabled": False}}` disables CES entirely

**Key design decisions:**
- **All CES imports guarded** — `try/except ImportError` around all CES module imports in `openclaw_hook.py`. Core NeuroGraph works unchanged if CES files are absent. This makes CES truly optional
- **Ollama primary, fallback secondary** — stream parser tries Ollama first for embeddings (fast, local, no GPU needed for nomic-embed-text), falls back to the ingestor's `embed_text()` method (sentence-transformers or hash). Ollama availability cached with 60s re-check
- **Direct voltage injection, not current** — `_nudge_nodes()` adds to `node.voltage` directly rather than going through the synapse/current pathway. This is intentional: stream parsing is a continuous attention signal, not a synaptic event. It warms nodes without triggering STDP
- **Sidecar file, not checkpoint modification** — activation persistence writes a separate `.activations.json` file rather than modifying the msgpack checkpoint. This avoids version/schema conflicts and makes activation state independently inspectable
- **Temporal decay on restore, not save** — decay is applied at restore time based on elapsed wall-clock hours. This means the sidecar always stores raw activation state, and the decay amount depends on how long the system was offline
- **Queue-based surfacing with decay** — surfacing queue decays every step (`score *= 0.95`), so stale concepts naturally fade. This prevents old activations from permanently occupying the surfacing window
- **HTTP dashboard in daemon thread** — the monitoring server runs in a daemon thread so it dies automatically when the process exits. No cleanup needed, no port leaks

**Tests added (64):**
- `test_ces.py`:
  - CESConfig: defaults (4), overrides (4), JSON file (3)
  - StreamParser: chunking (4), nudging (3), lifecycle (4), embedding (3), completions (1)
  - ActivationPersistence: capture (3), save/restore (4), decay (3), stats (2), auto-save (1)
  - SurfacingMonitor: after_step (3), scoring (3), queue (3), formatting (2), stats (1)
  - CESMonitoring: health_context (2), logger (1), coordinator (3), dashboard (2)
  - Integration: CES initialized (1), CES disabled (1), on_message (1), save sidecar (1), stats (1)

**Files created:**
- `ces_config.py`, `stream_parser.py`, `activation_persistence.py`, `surfacing.py`, `ces_monitoring.py`
- `tests/test_ces.py`

**Files modified:**
- `openclaw_hook.py` (CES integration: init, on_message, save, stats)
- `deploy.sh` (CES file deployment)
- `CLAUDE.md` (Phase 9 documentation)
