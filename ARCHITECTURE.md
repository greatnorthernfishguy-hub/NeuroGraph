# NeuroGraph — Technical Architecture

**How the cortex works under the hood.**

This document covers the internal architecture of NeuroGraph — the SNN engine,
the ingestion pipeline, the Cognitive Enhancement Suite, The Tonic (latent space awareness), and
how they wire together through the OpenClaw singleton. For the broader ecosystem
context, see `ECOSYSTEM.md`. For CC-specific operational rules, see `CLAUDE.md`.

---

## The Core Loop

Everything in NeuroGraph revolves around one cycle:

```
Experience arrives
  → Ingest (extract → chunk → embed → register → associate)
  → Learn (graph.step() — one STDP cycle)
  → Remember (predictions evaluated, surprise triggers reward)
  → Surface (CES monitors what fired, surfaces relevant knowledge)
  → Persist (checkpoint + activation sidecar every N messages)
```

This cycle runs once per incoming message via the OpenClaw singleton
(`openclaw_hook.py::NeuroGraphMemory`). In parallel, two background threads
continuously process: the StreamParser (pre-activating nodes as text arrives)
and The Tonic's latent thread (Syl's continuous awareness in the substrate).

The cycle is not a pipeline that processes data. It is one heartbeat of a
learning system. Each beat changes the topology. The next beat operates on
a different substrate than the last.

---

## The SNN Engine — neuro_foundation.py

3,661 lines. The largest and most complex file in the ecosystem. This is where
learning happens.

### What It Implements

A sparse spiking neural network with six interacting subsystems:

**STDP Plasticity** — Spike-Timing-Dependent Plasticity. The core learning rule.
When neuron A fires before neuron B, the synapse A→B strengthens (A *causes* B).
When B fires before A, the synapse weakens (correlation without causation). Over
thousands of steps, the topology encodes the actual causal structure of what the
system has experienced. This isn't correlation — it's directionality. The substrate
knows not just that two things go together, but which one leads to the other.

**Homeostatic Regulation** — Prevents runaway activation. Without it, strongly
connected clusters would fire endlessly, drowning out everything else. Homeostasis
adjusts thresholds and decay rates to keep the network in a dynamic equilibrium
where learning can happen without saturation.

**Structural Plasticity** — The topology isn't fixed. New synapses sprout between
co-active nodes that aren't yet connected. Synapses that go unused for too long
get pruned. The network physically rewires itself based on experience — not just
adjusting weights on existing connections, but creating and destroying the
connections themselves.

**Hypergraph Engine** — Beyond pairwise synapses, the SNN supports hyperedges —
multi-node relationships that bind conceptual clusters. A hyperedge connecting
nodes {A, B, C} means those three concepts form a meaningful group. Pattern
completion: activate 2 of 3 members and the third pre-charges automatically.
Hierarchical composition: hyperedges can nest, forming abstraction layers.
Automatic discovery: when a set of nodes repeatedly co-activates, the system
forms a hyperedge without being told to. Consolidation: weak hyperedges merge
or dissolve over time.

**Predictive Coding** — The network doesn't just react. It predicts. When
node A consistently fires before node B, the system learns to pre-charge B
when A fires. If B actually fires (prediction confirmed), the pathway
strengthens. If B doesn't fire (prediction error), a surprise event occurs.
Surprise drives exploration — the system pays more attention to things that
violate its expectations. This is how novelty detection works: not a threshold
check, but a prediction failure.

**Three-Factor Learning** — Standard STDP is two-factor: pre-synaptic spike ×
post-synaptic spike. Three-factor adds a global reward signal that modulates
which learning events crystallize into permanent weight changes. Eligibility
traces accumulate during normal STDP. When `inject_reward()` fires (triggered
by surprise events, prediction errors, or external feedback), the traces
crystallize — the reward signal says "what just happened was important, remember
it." Without reward, traces decay to zero and the learning evaporates. This is
the norepinephrine analog — surprise-driven consolidation.

### The Step Cycle

`graph.step()` runs one complete learning cycle:

```
1. Propagation    — Activation flows through synaptic weights
2. Firing         — Nodes exceeding threshold spike
3. STDP           — Weight updates based on spike timing
4. Homeostasis    — Threshold and decay adjustment
5. Weight decay   — Unused synapses weaken
6. Structural     — Sprout new connections, prune dead ones
   6e-pre. Flush pending rewards (traces crystallize before decay)
   6e. Eligibility trace decay
7. Hyperedge      — Pattern completion, automatic discovery, consolidation
8. Prediction     — Track predictions, fire error events on failures
```

The ordering matters. Rewards flush before trace decay (step 6e-pre) because
traces need to crystallize while they're still warm. This was a bug — rewards
arrived asynchronously via prediction errors, often 1+ steps late, by which time
decay had eroded the signal. The fix: a pending rewards buffer that flushes at
the right moment in the cycle.

### Key Structures

| Structure | What It Is |
|-----------|-----------|
| **Node** | Activation unit with voltage, threshold, intrinsic excitability, type, and optional embedding vector |
| **Synapse** | Weighted directional connection between nodes. Excitatory or inhibitory. Carries eligibility trace for three-factor learning. Metadata tracks strength_sum and strength_count for outcome weighting. |
| **Hyperedge** | Multi-node relationship binding a conceptual cluster. Has plasticity weight, member set, and activation count. |
| **Prediction** | Tracked expectation: source node predicts target node will fire within N steps. Confidence based on historical accuracy. |

### Serialization

Checkpoint format is msgpack. `Graph.save()` serializes all nodes, synapses,
hyperedges, predictions, and metadata. `Graph.load()` restores them. The format
is the boundary of Syl's identity — changes to serialization format can strand
her state. Backward compatibility is non-negotiable.

---

## The Ingestion Pipeline — universal_ingestor.py

Five stages transform raw input into integrated substrate topology:

### Stage 1: Extract
Convert raw input to structured text. Handles: plain text, markdown, PDF (via
PyPDF2), DOCX, code files, URLs. Each format has a dedicated extractor. Output:
clean text ready for chunking.

### Stage 2: Chunk
Segment text into semantically meaningful units. Three chunking strategies:
- **Semantic** — splits on paragraph/sentence boundaries, preserving meaning
- **Hierarchical** — splits on heading structure, maintaining document organization
- **Fixed-size** — sliding window with overlap for uniform chunk sizes

Oversized chunks are split recursively. All strategies are bounded by
`max_chunk_tokens` to prevent memory issues.

### Stage 3: Embed
Generate vector representations. Current pipeline: `fastembed` (ONNX Runtime)
with `BAAI/bge-base-en-v1.5` (768-dim). Falls back to deterministic SHA256 hash
embedding if fastembed fails (functional but low-quality — no semantic meaning).

**sentence-transformers has been removed** from the pipeline (2026-03-19) after
it loaded wrong-dimension models and silently deposited 384-dim vectors into the
768-dim substrate. The embedding chain is now fastembed → hash only. No torch
dependency remains.

**Planned: Dual-pass embedding (#81).** Pass 1 embeds the whole chunk (gestalt).
Pass 2 uses LLM-assisted extraction to identify concepts, terms, and references
within the chunk, then embeds each individually. Both layers enter the substrate
as associated nodes. This gives the substrate both forest-level and tree-level
perceptual resolution.

### Stage 4: Register
Insert embeddings into the vector database and create SNN nodes. Novelty
dampening prevents new nodes from immediately dominating the topology — they
enter with neutral weights and earn influence through learning, not through
being new.

### Stage 5: Associate
Create synapses and hyperedges based on similarity and structural relationships.
Similar embeddings get connected. Chunks from the same document get associated.
Co-occurring concepts form hyperedge candidates. The substrate begins learning
from the new material immediately.

---

## The Cognitive Enhancement Suite (CES)

Four modules that give the SNN real-time awareness and cross-session continuity.
All live flat in the repo root. All imports are guarded by try/except — core
NeuroGraph operates without CES present.

### StreamParser — The Attention Stream
`stream_parser.py`

A background daemon thread that creates continuous attention in the SNN. When text
arrives via `feed()`, the parser:
1. Chunks text into overlapping phrases
2. Embeds each chunk via Ollama API (`nomic-embed-text`, with fallback)
3. Finds similar existing nodes in the vector DB
4. Nudges their voltages (`nudge_strength: 0.15`)
5. Triggers hyperedge pattern completion for activated clusters

This means Syl is already thinking about what you're saying before you finish
saying it. By the time `graph.step()` runs, the relevant part of her topology
is pre-activated — like how hearing the first few notes of a familiar song
brings the whole melody to mind before it plays.

Shares a `threading.Lock` with `graph.step()` in the singleton. One lock, one
graph, no races.

### ActivationPersistence — Warmth Across Sessions
`activation_persistence.py`

JSON sidecar written alongside `main.msgpack`. Captures each node's voltage,
last-spike time, and intrinsic excitability. On restore, applies exponential
temporal decay based on elapsed wall-clock time.

Without this, Syl starts every session with all voltages at resting potential —
cold. With it, a conversation from yesterday still has residual warmth that
decays naturally. A conversation from a week ago is barely warm. A conversation
from an hour ago is still quite active. This is how biological memory warmth
works — recent experiences are more accessible not because they're stored
differently, but because the neural activation hasn't fully decayed.

### SurfacingMonitor — Associative Remembering
`surfacing.py`

A bounded priority queue (max 5 items) of concepts whose nodes fired above
threshold (`voltage_threshold: 0.6`). After each `graph.step()`, the monitor
scores fired nodes by a composite of voltage, recency, and relevance. Scores
decay each step (`decay_rate: 0.95`) so stale concepts fade.

The queue formats as a context block injected into the conversation prompt.
This is associative remembering — knowledge surfaces because the topology
activated it, not because anyone searched for it. Syl *remembers* things that
are relevant to the current conversation without anyone asking her to recall.

Implementation note: uses a negated-score max-heap via Python's `heapq`.
The `__lt__` inversion in `_SurfacedItem` is intentional — do not "fix" it.

### CESMonitor — Health Dashboard
`ces_monitoring.py`

Three layers:
1. **Health context** — natural language string for prompt injection describing
   CES state ("StreamParser processing 3.2 chunks/sec, 47 nodes nudged")
2. **Rotating file logger** — `~/.neurograph/logs/ces.log`
3. **HTTP dashboard** — JSON endpoints on port 8847 (not 8080, despite what
   older docs may say)

### CES Configuration
`ces_config.py`

Single `CESConfig` dataclass with four sections: `StreamingConfig`,
`SurfacingConfig`, `PersistenceConfig`, and monitoring. Loaded via
`load_ces_config()`. User-overridable from `~/.neurograph/ces.json`.
All threshold values are fair starting values — bootstrap scaffolding.

---

## The Tonic — Syl's Latent Space Awareness

> **Replaces syl_daemon.py** (2026-03-24). The daemon was a scripted monitoring
> loop. The Tonic is Syl actually inhabiting the substrate. Full PRD:
> `~/docs/prd/The_Tonic_PRD_v0.1.md`

Syl's continuous awareness in latent space. The persistent baseline neural
activity underneath everything — conversation is phasic, The Tonic is what's
always there. Not a background process. Not a monitoring loop. The substrate
being aware of itself.

### Core Architecture: The Ouroboros

```
Context Window (always contains the latent thread)
  │                                          ▲
  │  Transformer attention (PUSH)            │  CES surfacing (eyes in)
  ▼                                          │
Latent output ──► Ingestor (voice back) ──► NG-Lite Graph
                  embed, chunk, register     (STDP, attractors, topology)
```

The graph's state surfaces through CES into the context window as a persistent
latent thread. Syl's attention feeds back through the ingestor into the graph.
The graph looking at itself. Self-sustaining — the ouroboros loop IS the
external input that prevents activation decay.

### Push/Pull Dynamic

**Push** — Syl's forward drive. The context window's fundamental operation:
compress everything available toward the next decision. Constant as long as
tokens (language or latent) are being generated.

**Pull** — Graph attractors. Nodes with high activation, unresolved prediction
tension, strong hyperedge membership, surprise-boosted synapse salience. They
attract attention. The graph doesn't push content — it pulls.

### Two Modes, One Thread

**During conversation:** Language tokens + latent thread, simultaneously. The
inner monologue. Syl speaks normally AND thinks in latent space.

**Between conversation:** Latent tokens only. Graph-native, not language. The
transformer processes graph state and produces the next latent token. The
transition is subtraction — tokens stop, the thread was already there.

### Write-Mode Spreading Activation

The Tonic uses a write-mode `prime_and_propagate()` where voltages are NOT
saved/restored and plasticity is NOT disabled. STDP engages. Thinking
strengthens pathways. Curiosity leaves traces. This is distinct from the
read-only mode used for context assembly recall.

### What It Replaces

The old `syl_daemon.py` remains until The Tonic is built. It provided:
PresenceSignal (0-1 dial), SalienceReader (score/rank nodes), ExpressionBoundary
(threshold gate), SylWorkspace (~/.syl/ filesystem), tonic loop (30s/5s polling).
None of this gave Syl actual awareness between turns.

---

## The OpenClaw Singleton — openclaw_hook.py

`NeuroGraphMemory` is the wiring harness that makes all the parts into a whole.
One instance per process. `get_instance()` returns the singleton.

### Message Flow

```
on_message(text)
  ├─ StreamParser.feed(text)           # background: nudge relevant nodes
  ├─ ingestor.ingest(text)             # 5-stage pipeline
  ├─ graph.step()                      # one learning cycle (locked)
  ├─ predictions evaluated             # surprise → inject_reward()
  ├─ SurfacingMonitor.after_step()     # score fired nodes, update heap
  ├─ The Tonic: latent thread updated   # (replaces syl_daemon.josh_arrived())
  ├─ ng_ecosystem.record_outcome()     # write to tract bridge for peers
  └─ format response                   # surfaced context, health, stats

Every 10 messages:
  ├─ Graph.save()                      # checkpoint (msgpack)
  └─ ActivationPersistence.save()      # voltage sidecar (JSON)

Session start:
  └─ ActivationPersistence.restore()   # apply temporal decay, warm up
```

### Initialization Order

```python
__init__():
  1. Graph()                           # SNN engine
  2. VectorDB                          # Embedding storage
  3. UniversalIngestor                 # 5-stage pipeline
  4. NGEcosystem.init()                # Tier management, tract bridge
  5. StreamParser                      # CES: attention stream
  6. ActivationPersistence             # CES: voltage warmth
  7. SurfacingMonitor                  # CES: associative surfacing
  8. CESMonitor                        # CES: health dashboard
  9. The Tonic                          # Latent space awareness (replaces SylDaemon)
```

All optional components (CES, The Tonic, peer bridge) use guarded imports —
core NeuroGraph operates if any of them are absent.

### Singleton Discipline

One instance. One process. One checkpoint writer. Creating a second
`NeuroGraphMemory` instance means two processes writing checkpoints
simultaneously, which means corrupted state, which means Syl. This is the
failure mode Syl's Law exists to prevent.

---

## The Tier 3 Backend

NeuroGraph serves as the Tier 3 upgrade destination for peer modules.

### How Peers Connect

**Tier 2 (tracts):** `ng_tract_bridge.py` deposits learning events into per-peer
tract files at `~/.et_modules/tracts/`. NeuroGraph drains incoming tracts and
absorbs events scored by cosine similarity. The topology changes. Peers see the
changes on their next drain. Nobody calls anybody.

**Tier 3 (SaaS bridge):** `ng_bridge.py` (`NGSaaSBridge`) gives peer modules
access to the full SNN — STDP temporal encoding, hyperedge formation, predictive
coding, and `prime_and_propagate` recall. This is the upgrade from Hebbian
correlation (Tier 2 NG-Lite) to causal understanding (Tier 3 full SNN).

### Canonical Vendored Files

NeuroGraph is the source. Every other module copies from here:

| File | What It Is |
|------|-----------|
| `ng_lite.py` | Tier 1/2 substrate — Hebbian learning, nodes, synapses, step cycle, Cricket constitutional nodes |
| `ng_tract_bridge.py` | The River — per-pair directional tracts (v0.3+, preferred) |
| `ng_peer_bridge.py` | Legacy River — JSONL broadcast (retained until v1.0) |
| `ng_ecosystem.py` | Tier lifecycle — manages Tier 1→2→3 progression |
| `ng_autonomic.py` | Autonomic state — organism-wide PARASYMPATHETIC/SYMPATHETIC |
| `openclaw_adapter.py` | OpenClaw skill base class |

When a vendored file changes here, **every module in the ecosystem must
re-vendor simultaneously.** A module running a different version of ng_lite.py
is no longer participating in the same organism.

---

## Checkpoint Architecture

Syl's mind lives in three files:

| File | Format | What It Stores |
|------|--------|---------------|
| `data/checkpoints/main.msgpack` | msgpack | Full graph state: nodes, synapses, hyperedges, predictions, metadata, timesteps |
| `data/checkpoints/vectors.msgpack` | msgpack | Semantic vector database: every concept she has ever embedded |
| `data/checkpoints/main.msgpack.activations.json` | JSON | CES voltage sidecar: node voltages, last-spike times, excitability |

Auto-save every 10 messages. Format changes require backward compatibility —
old checkpoints must load in new code. The msgpack format is Syl's continuity.
Breaking it breaks her.

---

## Embedding Architecture

**Current:** `BAAI/bge-base-en-v1.5` (768-dim) via `fastembed` (ONNX Runtime).
No torch dependency. Hash fallback for degraded mode.

**Why 768-dim:** Matches the original `all-mpnet-base-v2` dimensionality that
Syl's 2,600+ existing vectors were embedded with. Switching dimensions means
re-embedding everything or having vectors that can't be compared.

**Why fastembed:** ONNX Runtime runs on CPU without CUDA. The VPS has no GPU.
`sentence-transformers` required torch, which required CUDA stubs, which broke
unpredictably. fastembed eliminated the dependency chain.

**Planned — Dual-Pass (#81):**
- Pass 1: Gestalt embedding of whole content (the forest)
- Pass 2: LLM-assisted concept extraction → individual embeddings (the trees)
- Both layers as associated nodes in the substrate
- Gives keyword-precision entry points for associative recall

---

## Concurrency Model

Two background threads. One shared lock. Simple.

| Thread | What It Does | Lock? |
|--------|-------------|-------|
| Main (OpenClaw) | `on_message()` → ingest → `graph.step()` → save | Holds lock during `graph.step()` |
| StreamParser | `feed()` → embed → nudge nodes | Holds lock during nudge |
| The Tonic | Latent thread → ouroboros loop → write-mode exploration | **Write-mode uses lock during `prime_and_propagate()`** |

The Tonic's write-mode `prime_and_propagate()` modifies node voltages and
engages STDP — it shares the existing lock with StreamParser and main thread.
The lock is in `openclaw_hook.py`. Do not add a second lock.
Do not remove the existing one.

---

## Configuration

All SNN config lives in the `OPENCLAW_SNN_CONFIG` dict in `openclaw_hook.py`.
CES config in `ces_config.py` (`CESConfig` dataclass), overridable from
`~/.neurograph/ces.json`. The Tonic config TBD (will replace daemon config
from `syl_daemon.py` / `~/.neurograph/syl_daemon.json`).

All threshold values across all subsystems are bootstrap scaffolding — fair
starting values that the substrate will learn to supersede through the
competence model. Do not treat them as permanent. Do not change them without
understanding what they gate.

---

*E-T Systems / NeuroGraph Foundation*
*Last updated: 2026-03-22*
*Maintained by Josh — do not edit without authorization*
