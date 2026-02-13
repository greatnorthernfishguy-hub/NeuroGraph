# NeuroGraph Foundation

## Project Overview
The NeuroGraph Foundation is a reusable, project-agnostic cognitive architecture that provides AI systems with dynamic learning, causal reasoning, and self-optimizing knowledge management through the integration of:

1. **Semantic Memory Layer** (Vector Database) - Content storage and fuzzy similarity retrieval
2. **Structural Memory Layer** (Hypergraph Engine) - N-ary relationships and concept clusters  
3. **Temporal Dynamics Layer** (Spiking Neural Network with STDP) - Real-time learning and causal inference

## Documentation
Complete product requirements and technical specifications are in `/docs/`:
- `NeuroGraph_Foundation_PRD_v1_0_docx.pdf` - Core architecture (v1.0)
- `NeuroGraph_Foundation__Addendums.pdf` - Universal Ingestor & Memory Hierarchy (v1.1-1.2)

**CRITICAL**: Read both PRD documents in `/docs/` before starting any implementation.

## Implementation Priority

### Phase 1: Core Foundation -- COMPLETE
Build the base `neuro_foundation.py` implementing:
- Node, Synapse, Hyperedge, Graph classes (see PRD Section 2)
- Sparse SNN simulation loop (Section 3.1)
- STDP plasticity with LTP/LTD (Section 3.1)
- Homeostatic regulation (Section 3.2)
- Structural plasticity - pruning/sprouting (Section 3.3)
- Basic serialization (Section 6)

### Phase 1.5: Stability Enhancements -- COMPLETE
- Changelog metadata in Graph class
- Refractory period for Hyperedge class

### Phase 2: Hypergraph Engine -- COMPLETE
- Hyperedge activation dynamics — all 4 modes with GRADED output scaling (Section 4)
- Pattern completion — inactive member pre-charging
- Hyperedge plasticity — member weights, threshold learning, member evolution
- Hierarchical hyperedges — multi-level composition
- Hyperedge discovery from co-activation
- Consolidation of overlapping hyperedges (Jaccard merge)

### Phase 2.5: Predictive Infrastructure Enhancements -- COMPLETE
- Prediction error events — PredictionState, SurpriseEvent, full lifecycle
- Dynamic pattern completion — experience-scaled strength
- Cross-level consistency pruning — subsumption archival
- Telemetry — prediction_accuracy, surprise_rate, experience distribution

### Phase 3: Predictive Coding
- Prediction tracking and error events (Section 5)
- Surprise-driven exploration

### Phase 4: Vector DB Integration
- Adapter interface (Section 7)
- Bidirectional flow

## File Structure
```
NeuroGraph/
├── neuro_foundation.py              # Core implementation (Phase 1 + 1.5 + 2 + 2.5)
├── tests/
│   ├── __init__.py
│   ├── test_snn.py                  # SNN dynamics tests (12)
│   ├── test_stdp.py                 # STDP learning tests (9)
│   ├── test_hypergraph.py           # Phase 1 + 1.5 hyperedge tests (18)
│   ├── test_hypergraph_phase2.py    # Phase 2 hypergraph engine tests (26)
│   ├── test_phase25.py              # Phase 2.5 predictive infrastructure tests (26)
│   └── test_integration.py          # End-to-end tests (18)
├── examples/
│   ├── simple_usage.py              # Basic example
│   ├── project_configs.py           # OpenClaw, DSM, Consciousness configs
│   └── hypergraph_demo.py           # Phase 2 hypergraph features demo
├── docs/                            # PRD documentation
├── requirements.txt
├── .gitignore
└── CLAUDE.md                        # This file
```

## Architecture Notes

### Key Design Principles
1. **Sparse by default** - No dense matrices, use sparse representations
2. **Dynamic topology** - Nodes/edges created/destroyed at runtime
3. **Pluggable plasticity** - Learning rules are swappable strategy objects
4. **Persistence-native** - All state is serializable

### Critical Implementation Details

**STDP Asymmetry** (Section 3.1.1):
- A_minus MUST be > A_plus (ratio 1.05-1.2) for stability
- Without this, network saturates

**Weight-dependent STDP**:
- LTP scaled by `(max_weight - w)/max_weight`
- Prevents runaway potentiation

**Homeostatic Plasticity** (Section 3.2):
- Use multiplicative synaptic scaling, NOT normalization
- Normalization destroys learned structure
- Scaling preserves relative weight ratios

**Refractory Periods**:
- Mandatory 2-step rest after firing
- Prevents unrealistic rapid re-firing

## Configuration Defaults (Section 9)
```python
DEFAULT_CONFIG = {
    # Phase 1: Core SNN
    'decay_rate': 0.95,
    'default_threshold': 1.0,
    'refractory_period': 2,
    'tau_plus': 20.0,
    'tau_minus': 20.0,
    'A_plus': 1.0,
    'A_minus': 1.2,
    'learning_rate': 0.01,
    'max_weight': 5.0,
    'target_firing_rate': 0.05,
    'scaling_interval': 100,
    'weight_threshold': 0.01,
    'grace_period': 500,
    'inactivity_threshold': 1000,
    'co_activation_window': 5,
    'initial_sprouting_weight': 0.1,
    # Phase 2: Hypergraph Engine
    'he_pattern_completion_strength': 0.3,
    'he_member_weight_lr': 0.05,
    'he_threshold_lr': 0.01,
    'he_discovery_window': 10,
    'he_discovery_min_co_fires': 5,
    'he_discovery_min_nodes': 3,
    'he_consolidation_overlap': 0.8,
    'he_member_evolution_window': 50,
    'he_member_evolution_min_co_fires': 10,
    'he_member_evolution_initial_weight': 0.3,
    # Phase 2.5: Predictive Infrastructure
    'prediction_window': 5,
    'prediction_ema_alpha': 0.01,
    'he_experience_threshold': 100,
}
```

## Testing Requirements (Section 9)

### Acceptance Criteria - Phase 1
1. 1K-node graph runs 10K steps without explosion or silent death
2. STDP correctly strengthens causal sequences (A→B strengthens when A fires before B)
3. STDP correctly weakens acausal pairs (A→B weakens when A fires after B)
4. Firing rates stabilize within 2× target after homeostatic regulation
5. ≥30% of speculative synapses are pruned within grace period
6. No memory leaks over 100K steps

## Implementation Guidelines

### What Claude Code Should Build

1. **Start with the core data structures** (Section 2.2):
   - Node class with all properties from PRD Table 2.2.1
   - Synapse class with all properties from Table 2.2.2
   - Hyperedge class with all properties from Table 2.2.3
   - Graph container

2. **Implement SNN simulation** (Section 2.2.4):
   - Voltage decay
   - Current injection
   - Spike propagation with delays
   - Refractory period enforcement

3. **Add STDP plasticity** (Section 3.1):
   - Follow mathematical specification exactly
   - Implement weight-dependent scaling
   - Add temporal aliasing handling

4. **Add homeostatic mechanisms** (Section 3.2):
   - Synaptic scaling (multiplicative)
   - Intrinsic excitability adjustment
   - Threshold adaptation

5. **Add structural plasticity** (Section 3.3):
   - Weight-based pruning
   - Co-activation sprouting
   - Age-based cleanup

6. **Comprehensive docstrings**:
   - Reference PRD sections
   - Explain failure modes and mitigations
   - Include example usage

### Code Style
- Use NumPy for vectorized operations where possible
- Sparse representations (dicts, sets) for topology
- Type hints throughout
- Comprehensive error handling
- Performance-critical sections should be optimized

## Dependencies
```
numpy>=1.24.0
scipy>=1.10.0  # For sparse matrices if needed
msgpack>=1.0.0  # For efficient serialization
```

## Notes for Claude Code

- **Read the PRDs first** - They contain critical implementation details
- **Follow the phased approach** - Don't try to build everything at once
- **Pay attention to failure modes** - PRD Section 3.1.2 lists common issues
- **Test as you go** - Each phase has acceptance criteria
- **Ask questions** - If anything in the PRDs is unclear, ask before implementing

## Current Status
- [x] Repository created
- [x] PRDs uploaded to `/docs/`
- [x] Phase 1: Core Foundation
- [x] Phase 1.5: Stability Enhancements
- [x] Phase 2: Hypergraph Engine
- [x] Phase 2.5: Predictive Infrastructure Enhancements
- [x] Phase 3: Predictive Coding
- [ ] Phase 4: Vector DB Integration

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
