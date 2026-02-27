# Phase 7: Auto-Knowledge — Spreading Activation Harvest

## Problem
NeuroGraph has two disconnected retrieval systems:
1. **Vector DB** — explicit `recall(query)` → cosine similarity search
2. **SNN** — learned synaptic connections with spreading activation, pattern completion, predictions

When content arrives via `on_message()`, the SNN fires nodes through learned connections, but this activation never surfaces as retrievable knowledge. The user must explicitly call `recall()` to get anything back. This is "searching your memory" instead of "just knowing."

## Goal
When content arrives, the network should **automatically surface relevant knowledge** — concepts that the SNN's learned structure considers associated with the incoming stimulus. No explicit `recall()` needed. The result is returned alongside the normal `on_message()` response.

## Design: Three-Layer Associative Recall

Combines approaches 1 (Activation-Triggered Retrieval), 2 (Semantic Priming via SNN), and 3 (Spreading Activation Harvest).

### Flow

```
Input text arrives
    │
    ├─ Stage 1-5: Normal ingestion (extract, chunk, embed, register, associate)
    │   └─ Produces: new node IDs, embeddings
    │
    ├─ PRIME: Embed input → vector search top-k similar existing nodes
    │   └─ Inject current into those nodes (semantic priming)
    │
    ├─ PROPAGATE: Run N SNN steps (spreading activation)
    │   └─ Collect ALL nodes that fire across all N steps
    │   └─ Track firing order and voltage-at-fire for ranking
    │
    ├─ HARVEST: Look up fired nodes' content in vector DB
    │   └─ Exclude newly-created nodes (they're the input, not the recall)
    │   └─ Rank by: (1) firing latency (sooner = stronger association)
    │              (2) prediction confidence (if node was predicted)
    │              (3) connection strength to stimulus
    │
    └─ RETURN: Surfaced knowledge alongside normal ingestion result
```

### Architecture

#### 1. New method: `Graph.prime_and_propagate(node_ids, currents, steps)`

Added to `neuro_foundation.py`. This is the SNN-level primitive:

- **Prime**: Inject specified currents into specified nodes (the semantic priming step)
- **Propagate**: Run `steps` SNN steps, collecting all fired node IDs with metadata:
  - `firing_step`: which step they fired on (latency from stimulus)
  - `voltage_at_fire`: how strongly they activated
  - `was_predicted`: whether they were a prediction target
  - `source_distance`: hop count from primed nodes (tracked via breadth-first from primed set through synapses)
- Returns a `PropagationResult` dataclass with the harvest

Why a separate method instead of modifying `step()`: The existing `step()` is the learning loop — it applies STDP, structural plasticity, predictions, etc. Associative recall should be **read-only** — it should observe the network's activation patterns without modifying weights or structure. A separate method keeps the two concerns clean and avoids side effects from recall-driven activation.

Actually, on reflection, we DON'T want a fully read-only mode. The priming and propagation should still allow:
- Pattern completion (hyperedge pre-charging of inactive members)
- Prediction generation (pre-charging predicted targets)

But it should NOT apply:
- STDP weight changes
- Structural plasticity (pruning/sprouting)
- Prediction confirmation/error (these are artificial firings, not real events)

So: `prime_and_propagate()` runs a modified step loop that includes activation dynamics (voltage decay, spike propagation, hyperedge evaluation, pattern completion) but skips plasticity rules and structural changes.

#### 2. New method: `NeuroGraphMemory.on_message()` enhancement

The existing `on_message()` return value gains a new `"surfaced"` key containing automatically recalled knowledge. The flow:

```python
def on_message(self, text, source_type=None):
    # Existing: ingest
    result = self.ingestor.ingest(text, source_type=source_type)
    new_node_ids = set(result.nodes_created)

    # NEW: Semantic priming
    query_vec = self.ingestor.embedder.embed_text(text)
    similar = self.vector_db.search(query_vec, k=prime_k, threshold=prime_threshold)
    prime_ids = [id for id, sim in similar if id not in new_node_ids]
    prime_currents = [sim * prime_strength for id, sim in similar if id not in new_node_ids]

    # NEW: Spreading activation harvest
    propagation = self.graph.prime_and_propagate(
        node_ids=prime_ids,
        currents=prime_currents,
        steps=propagation_steps,  # default 3
    )

    # NEW: Harvest content from fired nodes
    surfaced = []
    for entry in propagation.fired_entries:
        if entry.node_id in new_node_ids:
            continue  # Skip input nodes
        db_entry = self.vector_db.get(entry.node_id)
        if db_entry:
            surfaced.append({
                "node_id": entry.node_id,
                "content": db_entry["content"],
                "metadata": db_entry["metadata"],
                "latency": entry.firing_step,
                "strength": entry.activation_strength,
                "was_predicted": entry.was_predicted,
            })

    # Sort: lower latency first, then higher strength
    surfaced.sort(key=lambda x: (x["latency"], -x["strength"]))

    # Existing: normal learning step (separate from propagation)
    step_result = self.graph.step()

    # ... rest of existing logic ...

    event_data["surfaced"] = surfaced[:max_surfaced]  # default 10
    return event_data
```

#### 3. New method: `NeuroGraphMemory.associate(text)` — standalone associative recall

For cases where you want to get associated knowledge without ingesting new content:

```python
def associate(self, text, k=10, steps=3):
    """Associative recall: surface knowledge the network connects to this input.

    Unlike recall() which does pure vector similarity, this routes through
    the SNN's learned structure — surfacing knowledge based on causal
    connections, pattern completion, and prediction chains.
    """
    # Embed and find similar nodes
    # Prime them
    # Propagate
    # Harvest and return
```

### Configuration

New config keys added to `OPENCLAW_SNN_CONFIG`:

```python
# Auto-knowledge / Associative recall
"prime_k": 10,                    # Number of nodes to prime via vector similarity
"prime_threshold": 0.4,           # Min similarity to prime (lower than recall's 0.5)
"prime_strength": 0.8,            # Current multiplier for primed nodes
"propagation_steps": 3,           # SNN steps for spreading activation
"max_surfaced": 10,               # Max knowledge items to surface
"auto_knowledge_enabled": True,   # Toggle for the whole system
```

### Latency Considerations

The user's concern: perceived lag.

**Mitigations:**
1. **3 steps, not 30**: The default `propagation_steps=3` is tiny. Each step iterates over active nodes (sparse), not the full graph. For a 1K-node graph, this is sub-millisecond.
2. **Prime only top-10**: Vector search is O(n) but the DB is in-memory numpy dot products. For 10K vectors, ~1ms.
3. **Embedding is the bottleneck**: The `embed_text()` call is already happening for ingestion. We reuse that same embedding for priming — no extra model call.
4. **Configurable depth**: Users can reduce `propagation_steps` to 1 for speed or increase to 5 for depth.
5. **Gating**: `auto_knowledge_enabled` flag lets users disable it entirely if latency matters more than associative recall.

### Ranking & Drift Mitigation

To prevent surfacing irrelevant associations while still allowing "epiphany" connections:

- **Latency rank**: Nodes that fire on step 1 (direct synaptic connection) rank highest. Step 2 (two-hop) next. Step 3 (three-hop or pattern-completed) last. Direct associations beat remote ones.
- **Strength rank within latency tier**: Among nodes firing at the same step, higher voltage-at-fire ranks higher.
- **Prediction bonus**: Nodes that fired because they were prediction targets get a small ranking boost — the network specifically anticipated them.
- **No hard cutoff on hops**: A node that fires at step 3 with high strength IS the "epiphany surprise connection." The ranking ensures it surfaces, just below the obvious direct hits.

### What Changes

| File | Change |
|------|--------|
| `neuro_foundation.py` | Add `PropagationResult` dataclass, `FiredEntry` dataclass, `prime_and_propagate()` method |
| `openclaw_hook.py` | Enhance `on_message()` with priming + harvest, add `associate()` method, new config keys |
| `tests/test_auto_knowledge.py` | New test file: priming, propagation, harvest, ranking, drift mitigation, latency |
| `CLAUDE.md` | Phase 7 documentation |

### Test Plan

1. **Priming injects current**: Prime a node, verify voltage increases
2. **Propagation spreads activation**: Prime A, verify B fires (A→B synapse exists)
3. **Multi-step wavefront**: A→B→C chain, verify C fires on step 2 with higher latency
4. **Pattern completion surfaces**: Prime 2/3 of hyperedge members, verify 3rd member fires via completion
5. **Prediction targets surface**: Node with strong prediction target, verify target appears in results with `was_predicted=True`
6. **New nodes excluded**: Newly ingested nodes don't appear in surfaced results
7. **Ranking correctness**: Step-1 fires rank above step-2, higher voltage ranks above lower within same step
8. **No plasticity side effects**: Weights unchanged after prime_and_propagate
9. **Empty graph returns empty**: No crash, no results
10. **on_message returns surfaced**: End-to-end: ingest content, verify surfaced key in result
11. **associate() standalone**: Get associations without ingestion
12. **Disabled mode**: `auto_knowledge_enabled=False` → no surfaced results
13. **Configurable depth**: More steps → more nodes surface
