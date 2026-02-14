# NeuroGraph Integration User Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [Building Your Graph](#building-your-graph)
6. [Running the Simulation](#running-the-simulation)
7. [Hyperedges — Group Relationships](#hyperedges)
8. [Learning and Plasticity](#learning-and-plasticity)
9. [Predictions and Causal Inference](#predictions-and-causal-inference)
10. [Event System](#event-system)
11. [Telemetry](#telemetry)
12. [Checkpointing](#checkpointing)
13. [Universal Ingestor — Feeding Real Data](#universal-ingestor)
14. [Configuration Reference](#configuration-reference)
15. [Integration Patterns](#integration-patterns)
16. [Troubleshooting](#troubleshooting)

---

## Overview

NeuroGraph is a **project-agnostic cognitive architecture** that gives any application a self-organizing, learning knowledge graph. It combines three complementary memory layers:

| Layer | Mechanism | What it does |
|-------|-----------|-------------|
| Semantic memory | Vector database | Store and retrieve content by meaning |
| Structural memory | Hypergraph engine | Encode N-ary relationships between concepts |
| Temporal dynamics | Spiking neural network (SNN) with STDP | Learn causal sequences, predict future states |

The result is a graph that **learns from experience**: synapses strengthen when A reliably precedes B, weaken when they are unrelated, self-prune noise, sprout connections to surprising co-activations, and generate predictions about what is likely to fire next.

NeuroGraph has no domain assumptions. It can serve as a reasoning backend for a robotics controller, a document understanding system, a recommendation engine, a game AI, or anything else that benefits from causal, associative memory.

---

## Installation

```bash
# Core dependencies
pip install numpy>=1.24.0 scipy>=1.10.0 msgpack>=1.0.0

# Phase 4 Universal Ingestor (optional but recommended)
pip install sentence-transformers>=2.2.0 beautifulsoup4>=4.12.0 PyPDF2>=3.0.0
```

Copy `neuro_foundation.py` (and optionally `universal_ingestor.py`) into your project, or add the NeuroGraph directory to your `PYTHONPATH`.

```python
from neuro_foundation import Graph, SynapseType, ActivationMode, CheckpointMode
from universal_ingestor import UniversalIngestor, SimpleVectorDB, get_ingestor_config
```

If `sentence-transformers` is not available, the ingestor automatically falls back to deterministic hash-based embeddings so you can develop and test without the heavy ML dependency.

---

## Core Concepts

### Nodes

A **node** represents any discrete concept — a sensor reading, a word, a code symbol, an event. Each node has a **voltage** that accumulates incoming charge, a **threshold** it must exceed to fire (spike), and a **refractory period** during which it cannot fire again. Nodes are identified by string IDs you choose, or UUIDs if you omit the ID.

### Synapses

A **synapse** is a directed, weighted connection between two nodes. When the pre-synaptic node fires, its voltage propagates to the post-synaptic node after a configurable **delay** (in steps). Weights are learned by STDP and are bounded by `max_weight`. Synapses can be excitatory (positive current), inhibitory (negative current), or modulatory (no sign assumption).

### Hyperedges

A **hyperedge** groups multiple nodes into a named relationship. When enough of its members fire (determined by an activation mode and threshold), the hyperedge itself activates and injects current into its output targets. Hyperedges are the "concept cluster" layer — they encode facts like "nodes A, B, and C together mean disease D."

### The Step Loop

`graph.step()` is one discrete timestep. The full pipeline per step:

1. Decay voltages toward resting potential
2. Deliver delayed spikes from the propagation buffer
3. Detect which nodes exceed threshold (and are not refractory)
4. Reset fired nodes; start refractory countdown
5. Propagate spikes through outgoing synapses
6. Evaluate hyperedges; generate and validate predictions
7. Apply plasticity rules (STDP, homeostatic, hyperedge)
8. Structural plasticity — prune weak synapses, sprout new ones
9. Decrement refractory counters

The timestep counter increments by 1 on every call. All learning and timing is relative to this counter.

---

## Quick Start

```python
from neuro_foundation import Graph

# Create the graph with optional config overrides
g = Graph({"learning_rate": 0.02, "three_factor_enabled": False})

# Build topology
a = g.create_node("sensor_A")
b = g.create_node("concept_B")
c = g.create_node("action_C")

g.create_synapse("sensor_A", "concept_B", weight=0.8)
g.create_synapse("concept_B", "action_C", weight=0.5)

# Inject external stimulus and step
g.stimulate("sensor_A", current=2.0)
result = g.step()

print(result.fired_node_ids)   # e.g. ["sensor_A", "concept_B"]
print(g.get_telemetry().global_firing_rate)
```

---

## Building Your Graph

### Creating Nodes

```python
# Explicit ID (recommended for readability)
node = g.create_node("my_node")

# Auto UUID
node = g.create_node()

# With metadata (stored, not used by the SNN itself)
node = g.create_node("pain_signal", metadata={"domain": "sensory", "priority": "high"})

# Inhibitory node — its outgoing spikes subtract voltage from targets
node = g.create_node("suppressor", is_inhibitory=True)
```

`create_node()` returns the `Node` object. You can access it later via `g.nodes["my_node"]`.

### Creating Synapses

```python
from neuro_foundation import SynapseType

syn = g.create_synapse(
    pre_node_id="sensor_A",
    post_node_id="concept_B",
    weight=0.5,          # Initial connection strength
    delay=1,             # Propagation delay in steps (default 1)
    synapse_type=SynapseType.EXCITATORY,  # EXCITATORY | INHIBITORY | MODULATORY
    max_weight=5.0,      # Weight ceiling (overrides config default)
)
```

The returned `Synapse` object has a `synapse_id` UUID. To remove a synapse:

```python
g.remove_synapse(syn.synapse_id)
```

### Removing Nodes

Removing a node automatically cascades — all synapses in or out, and memberships in all hyperedges, are cleaned up.

```python
g.remove_node("old_concept")
```

### Accessing Graph State

```python
g.nodes           # Dict[str, Node]
g.synapses        # Dict[str, Synapse]
g.hyperedges      # Dict[str, Hyperedge]
g.timestep        # int — current simulation clock

# All nodes with voltage above 0.5
active = g.get_active_nodes(threshold=0.5)   # [(node_id, voltage), ...]
```

---

## Running the Simulation

### Single Step

```python
result = g.step()   # Returns StepResult

result.fired_node_ids        # List[str]
result.fired_hyperedge_ids   # List[str]
result.synapses_pruned       # int
result.synapses_sprouted     # int
result.predictions_confirmed # int (HE-level)
result.predictions_surprised # int (HE-level)
```

### Multiple Steps

```python
results = g.step_n(1000)   # Returns List[StepResult]
```

### Stimulating Nodes

```python
# Single stimulus
g.stimulate("sensor_A", current=2.0)

# Batch (applied before the next step call)
g.stimulate_batch([
    ("sensor_A", 2.0),
    ("sensor_B", 1.5),
    ("suppressor", -0.5),
])
```

Current values accumulate across `stimulate()` calls before being consumed by the next `step()`. A current of ~1.0 is typically enough to approach the default firing threshold of 1.0; you will usually need 1.5–2.5 to guarantee a spike, depending on decayed voltage.

### Driving a Real-time System

```python
while True:
    # Read sensors / compute inputs
    readings = my_sensor_module.read()
    for node_id, intensity in readings.items():
        g.stimulate(node_id, intensity)

    result = g.step()

    # React to fired nodes
    for node_id in result.fired_node_ids:
        my_actuator.trigger(node_id)

    # Periodically checkpoint
    if g.timestep % 10_000 == 0:
        g.checkpoint(f"checkpoint_{g.timestep}.msgpack")
```

---

## Hyperedges

Hyperedges let you encode multi-node relationships ("when A, B, and C all fire together, that means X").

### Creating Hyperedges

```python
from neuro_foundation import ActivationMode

he = g.create_hyperedge(
    member_node_ids=["fever", "cough", "fatigue"],
    activation_threshold=0.6,          # Fraction of weighted members needed
    activation_mode=ActivationMode.WEIGHTED_THRESHOLD,
    output_targets=["flu_diagnosis"],  # Nodes to inject when this HE fires
    output_weight=1.5,                 # Current injected per target
    metadata={"label": "flu syndrome", "domain": "medical"},
    is_learnable=True,                 # Allow plasticity
)
```

### Activation Modes

| Mode | Fires when... |
|------|--------------|
| `WEIGHTED_THRESHOLD` | Sum of active member weights / total weights >= threshold |
| `K_OF_N` | At least `threshold × N` members fired (integer ceiling) |
| `ALL_OR_NONE` | Every member fired this step |
| `GRADED` | Like WEIGHTED_THRESHOLD, but output current scales with activation fraction |

### Pattern Completion

When a hyperedge fires from **partial** activation, it pre-charges inactive members — nudging them toward threshold. This models concept completion ("given partial cues, activate the rest of the pattern"). The strength scales with the hyperedge's experience:

- New hyperedges (< 100 activations): zero completion strength
- Experienced hyperedges (>= 100 activations): full `pattern_completion_strength` (default 0.3)

This prevents untested concepts from hallucinating too aggressively.

### Hierarchical Hyperedges

Group hyperedges into higher-order concepts:

```python
flu_he = g.create_hyperedge(["fever", "cough"], ...)
asthma_he = g.create_hyperedge(["wheeze", "breathless"], ...)

respiratory_meta = g.create_hierarchical_hyperedge(
    child_hyperedge_ids=[flu_he.hyperedge_id, asthma_he.hyperedge_id],
    activation_threshold=0.5,
    output_targets=["respiratory_alert"],
)
```

Level-0 hyperedges are evaluated first each step; higher levels use the outputs of lower levels as inputs.

### Automatic Discovery and Consolidation

```python
# Discover new hyperedges from co-activation patterns
# (called automatically inside step(), but can be triggered manually)
new_hes = g.discover_hyperedges(fired_node_ids)

# Merge overlapping hyperedges (Jaccard similarity >= he_consolidation_overlap)
merged_count = g.consolidate_hyperedges()
```

Discovery watches for node groups that co-fire `>= he_discovery_min_co_fires` times within a `he_discovery_window`-step window. Consolidation merges duplicate/overlapping HEs and archives subsumed redundant ones.

---

## Learning and Plasticity

### STDP (Default — Active by Default)

Spike-Timing-Dependent Plasticity strengthens synapses where the pre-synaptic node fires *before* the post-synaptic node (causal), and weakens them when the order is reversed (anti-causal).

No configuration needed — STDP runs automatically inside `step()`. To train a causal sequence:

```python
for _ in range(50):
    g.stimulate("A", 2.0)
    g.step()
    g.stimulate("B", 2.0)
    g.step()
    # ... the A→B synapse weight will strengthen over iterations
```

### Homeostatic Regulation

Homeostatic scaling runs every `scaling_interval` steps (default 100). If a node's firing rate drifts from `target_firing_rate` (default 5%), it scales all incoming synapse weights up or down multiplicatively and adjusts its own threshold. This keeps the network from going silent or saturating.

This is automatic — you do not need to call anything.

### Three-Factor Learning (Reward-Modulated STDP)

Enable three-factor mode to gate STDP with an external reward signal:

```python
g = Graph({"three_factor_enabled": True})

# Weights accumulate STDP changes in eligibility traces instead of applying immediately
g.stimulate("A", 2.0); g.step()
g.stimulate("B", 2.0); g.step()

# Commit traces with a reward signal
g.inject_reward(strength=1.0)           # Positive: strengthen recent patterns
g.inject_reward(strength=-0.5)          # Negative: weaken recent patterns

# Scope the reward to a subset of nodes
g.inject_reward(strength=0.8, scope={"A", "B"})
```

Eligibility traces decay exponentially with time constant `eligibility_trace_tau` (default 100 steps). Without a reward signal, traces decay without being applied.

### Custom Plasticity Rules

```python
from neuro_foundation import PlasticityRule, STDPRule, HomeostaticRule, HyperedgePlasticityRule

# Replace default rules entirely
class MyRule(PlasticityRule):
    def apply(self, graph, fired_node_ids, timestep):
        for nid in fired_node_ids:
            for sid in graph._outgoing.get(nid, set()):
                syn = graph.synapses[sid]
                syn.weight = min(syn.weight * 1.001, syn.max_weight)

g.set_plasticity_rules([
    STDPRule(learning_rate=0.05),   # Keep STDP
    HomeostaticRule(),              # Keep homeostatic
    MyRule(),                       # Add custom rule
])
```

---

## Predictions and Causal Inference

NeuroGraph has two prediction systems that run automatically during `step()`.

### Hyperedge-Level Predictions (Phase 2.5)

When a hyperedge with `output_targets` fires, it registers a prediction that those targets should fire within `prediction_window` steps. If they do: `prediction_confirmed` event. If they don't: a `SurpriseEvent` is emitted.

Access active HE predictions:

```python
preds = g.get_active_predictions()   # Dict[str, PredictionState]
for pid, ps in preds.items():
    print(ps.hyperedge_id, ps.predicted_targets, ps.prediction_strength)
```

### Synapse-Level Predictions (Phase 3)

When a node fires with a strong outgoing synapse (weight > `prediction_threshold`, default 3.0), it:

1. Pre-charges the target node (nudges its voltage toward threshold)
2. Registers a `Prediction` in `graph.active_predictions`
3. Optionally cascades predictions along learned chains (A→B→C)

When the prediction resolves:
- **Confirmed**: synapse weight increases by `prediction_confirm_bonus × confidence`
- **Error**: synapse weight decreases by `prediction_error_penalty × confidence`; surprise-driven sprouting may create a speculative synapse toward what actually fired instead

```python
preds = g.get_predictions()   # List[Prediction]
for p in preds:
    print(p.source_node_id, "->", p.target_node_id,
          f"strength={p.strength:.2f} confidence={p.confidence:.2f}")
```

### Causal Chain Tracing

```python
chain = g.get_causal_chain("action_C", depth=3)
# Returns a DAG dict showing which nodes causally lead to action_C
# via the strongest synapse paths
```

---

## Event System

Register callbacks to react to internal events without polling:

```python
def on_spike(node_ids, timestep):
    print(f"Step {timestep}: fired {node_ids}")

g.register_event_handler("spikes", on_spike)
```

Callbacks receive keyword arguments — use `**kwargs` for forward compatibility:

```python
g.register_event_handler("prediction_error", lambda **kw: log_error(kw))
```

### All Available Events

| Event | Fires when | Key kwargs |
|-------|-----------|-----------|
| `spikes` | Nodes fired this step | `node_ids`, `timestep` |
| `pruned` | Synapses pruned | `count`, `timestep` |
| `sprouted` | Synapses auto-created (co-activation) | `count`, `timestep` |
| `hyperedge_fired` | Hyperedge activated | `hid`, `activation` |
| `hyperedge_discovered` | New HE auto-discovered | `hid` |
| `hyperedges_consolidated` | HEs merged | `count` |
| `hyperedge_archived` | HE archived (subsumed) | `archived_id`, `subsumed_by`, `reason` |
| `prediction_created` | HE-level prediction registered | `prediction_id`, `hyperedge_id`, `targets` |
| `prediction_confirmed` | Prediction confirmed (either level) | varies |
| `surprise` | HE-level prediction window expired | `surprise` (SurpriseEvent), `timestep` |
| `prediction_generated` | Phase 3 synapse prediction created | `source`, `target`, `strength`, `confidence`, `chain_depth`, `timestep` |
| `prediction_error` | Phase 3 prediction expired without confirmation | `source`, `expected_target`, `strength`, `confidence`, `actual_fired`, `timestep` |
| `surprise_sprouted` | Surprise-driven alternative synapse created | `source`, `expected`, `alternatives`, `count`, `timestep` |
| `novel_sequence` | Firing pattern with no strong prior connections | `source`, `firing_nodes`, `timestep` |
| `reward_injected` | Reward signal applied | `strength`, `scope_size`, `timestep` |

Multiple handlers can be registered for the same event. They are called in registration order.

---

## Telemetry

```python
tel = g.get_telemetry()

# Topology
tel.total_nodes, tel.total_synapses, tel.total_hyperedges

# Activity
tel.global_firing_rate     # Mean EMA firing rate across all nodes
tel.mean_weight            # Mean synapse weight
tel.std_weight             # Standard deviation of weights

# Plasticity lifecycle
tel.total_pruned           # Cumulative synapses removed
tel.total_sprouted         # Cumulative synapses auto-created
tel.total_he_discovered    # Cumulative hyperedges auto-discovered
tel.total_he_consolidated  # Cumulative hyperedge merges

# Prediction quality
tel.prediction_accuracy    # confirmed / (confirmed + errors)
tel.surprise_rate          # errors / total resolved
tel.active_predictions_count
tel.total_predictions_made
tel.total_novel_sequences

# Hyperedge experience histogram
tel.hyperedge_experience_distribution
# {"0": n, "1-9": n, "10-99": n, "100+": n}
```

A healthy, running graph typically shows:
- `global_firing_rate` near `target_firing_rate` (default 0.05) after homeostatic regulation
- `prediction_accuracy` climbing over time as the network learns causal sequences
- `total_pruned > total_sprouted` in early training as noise is cleared; then stabilizing

---

## Checkpointing

Save and restore the complete graph state, including all active predictions, confirmation histories, and plasticity traces.

```python
# Save (JSON, human-readable)
g.checkpoint("state.json")

# Save (MessagePack, compact binary — recommended for production)
g.checkpoint("state.msgpack")

# Incremental — only records changes since last FULL checkpoint
g.checkpoint("delta.msgpack", CheckpointMode.INCREMENTAL)

# Fork — like FULL but marks the checkpoint as a branch point
g.checkpoint("fork.msgpack", CheckpointMode.FORK)

# Restore into a new Graph instance
g2 = Graph()
g2.restore("state.msgpack")
```

**What is preserved:**
- All nodes, synapses, hyperedges, and their weights
- Current voltages, refractory counters, spike history
- Active predictions (Phase 3 and Phase 2.5) — validated and cleaned on restore
- Prediction outcomes, confirmation histories, novel sequence log, reward history
- Simulation timestep and telemetry counters

**On restore, stale predictions are automatically dropped:**
- Expired predictions (window already passed)
- Predictions referencing nodes or hyperedges that were removed between checkpoint and restore

Checkpoints are backward-compatible: a v0.2.5 checkpoint loads cleanly into the current engine with empty prediction state.

---

## Universal Ingestor

The Universal Ingestor provides a five-stage pipeline that takes raw content (text, code, Markdown, URLs, PDFs) and automatically populates your NeuroGraph with semantically embedded nodes, similarity synapses, and concept cluster hyperedges.

### Setup

```python
from neuro_foundation import Graph
from universal_ingestor import UniversalIngestor, SimpleVectorDB, get_ingestor_config

graph = Graph()
vector_db = SimpleVectorDB()

# Use a predefined project config (see below), or build your own dict
config = get_ingestor_config("openclaw")   # or "dsm" or "consciousness"

ingestor = UniversalIngestor(graph, vector_db, config)
```

### Ingesting Content

```python
# Plain text
result = ingestor.ingest("The capital of France is Paris.", source_type=None)

# Markdown document
result = ingestor.ingest(markdown_string)

# Python/JS source code
result = ingestor.ingest(python_source_code)

# URL (fetches and extracts HTML)
result = ingestor.ingest("https://example.com/article")

# PDF file path
result = ingestor.ingest("/path/to/document.pdf")

# Multiple sources in sequence
results = ingestor.ingest_batch([
    (code_string, None),
    (markdown_string, None),
    ("https://docs.example.com", None),
])
```

`IngestionResult` fields:

```python
result.chunks_created    # Number of semantic chunks produced
result.nodes_created     # Node IDs added to the graph
result.synapses_created  # Synapse IDs created by association
result.hyperedges_created # HE IDs created by clustering
```

### Novelty Dampening

Newly ingested nodes start with **reduced excitability** and a **raised threshold**. This prevents fresh, unvalidated knowledge from dominating the network. Call `update_probation()` each simulation step to gradually restore their full influence:

```python
for step in range(1000):
    graph.step()
    graduated = ingestor.update_probation()
    if graduated:
        print(f"Step {step}: nodes graduated from probation: {graduated}")
```

The dampening curve (`LINEAR`, `EXPONENTIAL`, or `LOGARITHMIC`) controls how quickly nodes earn their full weight over the `probation_period` (e.g., 10 steps for OpenClaw code, 500 steps for Consciousness slow-burn exploration).

### Semantic Query

```python
matches = ingestor.query_similar("neural plasticity", k=5, threshold=0.7)
for m in matches:
    print(m["content"], f"sim={m['similarity']:.3f}")
    print(m["metadata"])
```

Returns a list of dicts sorted by similarity descending.

### Project Configurations

Three built-in profiles:

| Profile | Use case | Chunking | Novelty dampening | Similarity threshold |
|---------|---------|---------|------------------|---------------------|
| `openclaw` | Source code / software projects | Code-aware (function/class boundaries) | 0.3 (moderate) | 0.70 |
| `dsm` | Clinical/regulatory documents | Hierarchical (heading structure) | 0.05 (conservative) | 0.80 |
| `consciousness` | Exploratory / research content | Semantic (paragraph boundaries) | 0.01 (very permissive) | 0.65 |

```python
config = get_ingestor_config("dsm")   # or "openclaw", "consciousness"
```

### Custom Configuration

Build a config dict with any subset of these keys:

```python
config = {
    "chunking": {
        "strategy": "semantic",       # semantic | code_aware | hierarchical | fixed_size
        "min_chunk_tokens": 100,
        "max_chunk_tokens": 400,
        "overlap_tokens": 32,
    },
    "embedding": {
        "model_name": "all-mpnet-base-v2",
        "use_model": True,            # False → deterministic hash fallback
    },
    "registration": {
        "novelty_dampening": 0.1,     # Initial excitability multiplier [0, 1]
        "probation_period": 50,       # Steps until full excitability
        "dampening_curve": "exponential",  # linear | exponential | logarithmic
        "initial_threshold_boost": 0.2,   # Added to node threshold at creation
    },
    "association": {
        "similarity_threshold": 0.75,      # Min cosine sim for synapse creation
        "similarity_weight_scale": 1.0,    # Scale factor for similarity synapses
        "sequential_weight": 0.3,          # Weight for sequential chunk synapses
        "parent_child_weight": 0.5,        # Weight for heading-parent synapses
        "min_cluster_size": 3,             # Min nodes per cluster hyperedge
        "cluster_similarity_threshold": 0.65,
    },
}
```

### Using SimpleVectorDB Directly

The vector database can be used independently of the ingestor:

```python
import numpy as np
vdb = SimpleVectorDB()

vec = np.random.randn(768).astype(np.float32)  # any embedding
vdb.insert("doc_42", vec, content="Full text here", metadata={"source": "wiki"})

results = vdb.search(query_vec, k=10, threshold=0.6)  # [(id, cosine_sim), ...]
entry = vdb.get("doc_42")   # {"id", "embedding", "content", "metadata"}

vdb.delete("doc_42")
vdb.count()
vdb.all_ids()
```

Vectors are L2-normalized at insert time; similarity is computed as a dot product (equivalent to cosine similarity for unit vectors).

---

## Configuration Reference

Pass any of these keys to the `Graph()` constructor to override defaults:

```python
g = Graph({
    "learning_rate": 0.02,
    "three_factor_enabled": True,
    "target_firing_rate": 0.1,
})
```

### Core SNN

| Key | Default | Description |
|-----|---------|-------------|
| `decay_rate` | 0.95 | Voltage decay factor per step |
| `default_threshold` | 1.0 | Initial firing threshold for new nodes |
| `refractory_period` | 2 | Mandatory rest steps after firing |
| `max_weight` | 5.0 | Global synapse weight ceiling |
| `learning_rate` | 0.01 | STDP learning rate |
| `tau_plus` | 20.0 | STDP LTP time constant (steps) |
| `tau_minus` | 20.0 | STDP LTD time constant (steps) |
| `A_plus` | 1.0 | STDP LTP amplitude |
| `A_minus` | 1.2 | STDP LTD amplitude (must be > A_plus) |

### Homeostatic Regulation

| Key | Default | Description |
|-----|---------|-------------|
| `target_firing_rate` | 0.05 | Target EMA firing rate per node |
| `scaling_interval` | 100 | Steps between homeostatic scaling passes |

### Structural Plasticity

| Key | Default | Description |
|-----|---------|-------------|
| `weight_threshold` | 0.01 | Weights below this for `low_weight_steps` → prune |
| `grace_period` | 500 | Steps before newly created synapses are eligible for pruning |
| `inactivity_threshold` | 1000 | Steps of inactivity before a synapse is pruned |
| `co_activation_window` | 5 | Steps window for co-activation sprouting detection |
| `initial_sprouting_weight` | 0.1 | Weight for auto-sprouted synapses |

### Hypergraph Engine (Phase 2)

| Key | Default | Description |
|-----|---------|-------------|
| `he_pattern_completion_strength` | 0.3 | Voltage pre-charge to inactive members |
| `he_member_weight_lr` | 0.05 | Member weight adaptation rate |
| `he_threshold_lr` | 0.01 | Threshold adaptation rate (with reward) |
| `he_discovery_window` | 10 | Co-activation window for HE discovery |
| `he_discovery_min_co_fires` | 5 | Co-fires needed before auto-discovery |
| `he_discovery_min_nodes` | 3 | Minimum group size for discovery |
| `he_consolidation_overlap` | 0.8 | Jaccard threshold for HE merging |
| `he_experience_threshold` | 100 | Activations needed for full pattern completion |

### Predictive Coding (Phase 3)

| Key | Default | Description |
|-----|---------|-------------|
| `prediction_threshold` | 3.0 | Min synapse weight to trigger prediction |
| `prediction_window` | 10 | Steps before prediction expires |
| `prediction_pre_charge_factor` | 0.3 | Pre-charge fraction applied to predicted target |
| `prediction_chain_decay` | 0.7 | Strength decay per chain hop |
| `prediction_max_chain_depth` | 3 | Maximum prediction chain depth |
| `prediction_confirm_bonus` | 0.01 | Weight bonus per confirmed prediction |
| `prediction_error_penalty` | 0.02 | Weight penalty per failed prediction |
| `surprise_sprouting_weight` | 0.1 | Weight for surprise-driven synapses |
| `three_factor_enabled` | False | Enable reward-gated STDP |
| `eligibility_trace_tau` | 100 | Eligibility trace decay time constant |

---

## Integration Patterns

### Pattern 1 — Sensor Fusion

Map multiple sensor channels to nodes; create a hyperedge per meaningful combination:

```python
g = Graph()
for sensor in ["temperature", "pressure", "humidity", "vibration"]:
    g.create_node(sensor)

# Overheating alarm pattern
g.create_hyperedge(
    ["temperature", "pressure"],
    activation_mode=ActivationMode.ALL_OR_NONE,
    output_targets=["alarm_overheat"],
    metadata={"label": "overheat"}
)

# In control loop
while True:
    readings = sensors.read()
    for sensor, value in readings.items():
        if value > threshold[sensor]:
            g.stimulate(sensor, value * scale_factor)
    g.step()
```

### Pattern 2 — Document Understanding

Use the ingestor to load a document corpus, then query it with new text:

```python
ingestor = UniversalIngestor(graph, vdb, get_ingestor_config("dsm"))

for doc_path in document_directory:
    ingestor.ingest(open(doc_path).read())

# Run the graph for a while to stabilize associations
for _ in range(500):
    graph.step()
    ingestor.update_probation()

# Query
results = ingestor.query_similar("contraindications for medication X", k=5)
```

### Pattern 3 — Causal Sequence Learning

Train the graph on observed event sequences and then use predictions for anticipation:

```python
g = Graph({"learning_rate": 0.05})

# Create nodes for events
for event in all_events:
    g.create_node(event)

# Training: present sequences
for sequence in training_data:
    for event in sequence:
        g.stimulate(event, 2.0)
        g.step()

# Inference: stimulate partial sequence, read predictions
g.stimulate("event_A", 2.0)
g.step()

for pred in g.get_predictions():
    print(f"Expecting {pred.target_node_id} with confidence {pred.confidence:.2f}")
```

### Pattern 4 — Reinforcement Learning Integration

Use three-factor learning to gate weight updates with environment rewards:

```python
g = Graph({"three_factor_enabled": True, "eligibility_trace_tau": 50})

# ... run episodes ...
for step in episode:
    action_taken = choose_action(g)
    reward = environment.step(action_taken)

    g.stimulate(action_taken, 2.0)
    g.step()

    if reward != 0:
        g.inject_reward(reward, scope={action_taken})
```

### Pattern 5 — Knowledge Base with Semantic Search

Combine the vector database with graph reasoning:

```python
ingestor = UniversalIngestor(graph, vdb, get_ingestor_config("consciousness"))
ingestor.ingest(knowledge_base_text)

def answer_query(question):
    # Semantic retrieval
    candidates = ingestor.query_similar(question, k=10, threshold=0.6)

    # Activate matching nodes in graph
    for c in candidates:
        node_id = c["metadata"].get("node_id")
        if node_id:
            g.stimulate(node_id, c["similarity"] * 2.0)

    # Let the graph propagate and complete patterns
    for _ in range(5):
        result = g.step()

    # Read what the graph concluded
    return [
        (nid, g.nodes[nid].metadata)
        for nid in g.get_active_nodes(threshold=0.8)
    ]
```

---

## Troubleshooting

### The network goes silent (no nodes ever fire)

- Stimulus current is too low. Try 2.0–3.0 for a default-threshold (1.0) node.
- `decay_rate` is too high. Lower it toward 0.8 to retain voltage between stimulations.
- Homeostatic scaling may have reduced all weights aggressively. Check `tel.mean_weight`.
- Too many inhibitory nodes suppressing activity.

### The network saturates (everything fires every step)

- `A_minus` must be > `A_plus` (default 1.2 vs 1.0). If you have custom values, verify this ratio.
- `target_firing_rate` may be too high. The default 0.05 (5%) is appropriate for most uses.
- Reduce `learning_rate` if weights are growing too fast.

### Predictions are never confirmed

- Predictions only generate when synapse weight > `prediction_threshold` (default 3.0). Weights need time to grow via STDP or reward learning.
- Increase the number of training iterations before relying on predictions.
- Lower `prediction_threshold` to start seeing predictions with weaker links.

### Memory usage grows unboundedly

- Novelty sprouting can create many synapses. Tune `co_activation_window` and `initial_sprouting_weight`.
- Ensure pruning is running: `tel.total_pruned` should grow over time after the `grace_period` (default 500 steps).
- Active prediction lists are bounded to `prediction_max_active` (default 1000).

### Ingestor creates too many/too few nodes

- Adjust `max_chunk_tokens` and `min_chunk_tokens` in the chunking config.
- Switch chunking strategy: `code_aware` for code, `hierarchical` for structured docs, `semantic` for prose.

### Ingested nodes never become active in the graph

- Check `novelty_dampening` — a value of 0.01 means nodes start at 1% excitability. Call `update_probation()` each step to graduate them.
- Increase stimulus current proportionally to the dampening factor during probation.

### Checkpoint/restore loses active predictions

- Make sure you are on version 0.3.5 or later (check `g.metadata.get("version")`). Earlier versions did not serialize predictions.
- Old checkpoints (v0.2.5) restore cleanly but with empty prediction state — this is expected.
