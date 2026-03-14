# NeuroGraph ContextEngine Mapping
**Status:** Planning document (no implementation yet)
**Date:** 2026-03-14
**Replaces:** Punch list #37 (SKILL.md fix) and #39 (TypeScript hook translation)
**Preserves:** `openclaw_hook.py` stays functional as-is until migration

---

## 1. Why ContextEngine

OpenClaw v2026.3.7 introduced the ContextEngine plugin slot — a single exclusive
interface that owns session context orchestration (ingest, assembly, compaction).
This replaces the SKILL.md + hook + frontmatter approach with a first-class plugin
contract.

The current integration uses SKILL.md-based discovery (`hook: openclaw_hook.py::get_instance`)
which is a legacy path. The ContextEngine slot is the canonical way forward. One
plugin, one manifest, one registration call — cleaner than translating Python hooks
to TypeScript (#39) or patching SKILL.md frontmatter (#37).

The `LegacyContextEngine` wrapper proves backward compatibility is a non-issue:
engines that don't implement optional hooks just get no-ops. We can migrate
incrementally.

---

## 2. Interface Reference

From OpenClaw `src/context-engine/types.ts`:

```
Required:  ingest(message) -> IngestResult
           assemble(messages, tokenBudget) -> AssembleResult
           compact(sessionFile, tokenBudget, force) -> CompactResult

Optional:  bootstrap(sessionId, sessionFile) -> BootstrapResult
           ingestBatch(messages) -> IngestBatchResult
           afterTurn(messages, ...) -> void
           prepareSubagentSpawn(parentKey, childKey) -> SubagentSpawnPreparation
           onSubagentEnded(childKey, reason) -> void
           dispose() -> void
```

Registration: `api.registerContextEngine(id, factory)`
Activation: `plugins.slots.contextEngine = "<id>"`

---

## 3. Lifecycle Mapping

### bootstrap (optional)

**OpenClaw contract:** Initialize engine state for a session. Import historical
context. Called once when the engine is selected for a session.

**NeuroGraph equivalent:** `NeuroGraphMemory.__init__()` (openclaw_hook.py:176-318)

| Step | Current code | Line | Notes |
|------|-------------|------|-------|
| Checkpoint restore | `graph.restore(checkpoint_path)` | 199-209 | msgpack deserialization of full SNN state |
| VectorDB restore | `vector_db.load(vector_db_path)` | 216-225 | Semantic search index |
| Ingestor init | `UniversalIngestor(graph, vector_db, config)` | 239-241 | Embedding model loaded here (heavy) |
| Peer bridge init | `NGPeerBridge(module_id="neurograph")` | 257-265 | Cross-module learning (Tier 2) |
| CES init | StreamParser, ActivationPersistence, SurfacingMonitor, CESMonitor | 286-316 | All optional, guarded by try/except |
| Activation restore | `activation_persistence.restore(graph, path)` | 310-312 | Cross-session voltage state |

**Mapping decision:** Bootstrap is the right place for all of this. The factory
function (`registerContextEngine(id, factory)`) returns the engine instance, so
heavy init (embedding model, checkpoint restore) can happen in the factory OR in
bootstrap. Recommendation: **factory creates the thin shell, bootstrap does the
heavy restore.** This lets OpenClaw call bootstrap per-session while the engine
instance (and its embedding model) persists across sessions.

**Key concern:** NeuroGraph's singleton pattern (`get_instance()`) assumes one
instance per process. OpenClaw's ContextEngine contract creates one instance per
factory call but bootstrap is per-session. The singleton can survive if the
factory returns the same instance — `bootstrap` just re-checks the checkpoint.

---

### ingest (required)

**OpenClaw contract:** Process a single incoming message. Called for each user or
assistant message as it arrives.

**NeuroGraph equivalent:** First half of `on_message()` (openclaw_hook.py:363-446)

| Step | Current code | Line | Notes |
|------|-------------|------|-------|
| 5-stage pipeline | `ingestor.ingest(text, source_type)` | 385 | Extract, Chunk, Embed, Register, Associate |
| Stream parser feed | `stream_parser.feed(text)` | 410 | CES: async background embedding + node nudging |
| Peer learning write | `_write_peer_learning_event(text, result, step_result)` | 446 | Shared directory for sibling modules |
| Memory event log | `_write_memory_event("ingestion", event_data)` | 441 | Structured JSONL for OpenClaw consumption |

**What moves OUT of ingest:** `graph.step()`, `inject_reward()`, surfacing
monitor scan, novelty probation update, and auto-save all move to `afterTurn`.
The current `on_message()` does too much — it ingests AND steps AND saves. The
ContextEngine lifecycle separates these correctly.

**IngestResult mapping:**
```
{ ingested: true }  // if ingestor.ingest() created nodes
{ ingested: false } // if text was empty or duplicate
```

**ingestBatch (optional):** Could batch-ingest a full turn (user message +
assistant response + tool results) as a single unit. Maps to calling
`ingestor.ingest()` for each message in the batch, then writing one peer
learning event for the batch. Not critical for v1.

---

### assemble (required)

**OpenClaw contract:** Build the context window under a token budget. Return
ordered messages + estimated token count + optional system prompt addition.

**NeuroGraph equivalent:** `_harvest_associations()` + `recall()` + surfacing
monitor (openclaw_hook.py:450-527, 592-603)

| Step | Current code | Line | Notes |
|------|-------------|------|-------|
| Spreading activation | `_harvest_associations(text, exclude)` | 450-527 | Prime similar nodes, propagate N steps, harvest fired |
| Semantic recall | `recall(query, k, threshold)` via `ingestor.query_similar()` | 592-603 | Pure vector cosine search |
| CES surfacing | `surfacing_monitor.get_surfaced()` | 416 | Priority queue of concepts above threshold |
| SNN-routed recall | `associate(text, k, steps)` | 605-635 | `_harvest_associations` with custom depth |

**Assembly strategy:** NeuroGraph doesn't currently produce a message list — it
produces a context block (surfaced knowledge). In the ContextEngine model, this
maps to `systemPromptAddition`:

```typescript
async assemble({ messages, tokenBudget }) {
  // 1. Pass through the conversation messages unchanged
  // 2. Run spreading activation on recent messages
  // 3. Format surfaced knowledge as systemPromptAddition
  // 4. Estimate tokens
  return {
    messages,               // conversation messages (unmodified)
    estimatedTokens: ...,   // token estimate
    systemPromptAddition: formatSurfacedContext(surfaced)
  };
}
```

**This is the cleanest fit.** NeuroGraph doesn't need to own message ordering
or truncation — it ADDS substrate context to whatever the conversation is. The
`systemPromptAddition` field was designed for exactly this.

**Token budget interaction:** NeuroGraph's surfaced context must fit within
the remaining budget after messages. `max_surfaced` config becomes dynamic,
driven by `tokenBudget - estimatedMessageTokens`.

---

### compact (required)

**OpenClaw contract:** Reduce context token usage. Called when approaching token
limits. Engine declares `ownsCompaction: true` if it manages its own compaction
lifecycle (otherwise OpenClaw falls back to its built-in summarizer).

**NeuroGraph equivalent:** Several existing mechanisms that currently run
inside `graph.step()`:

| Step | Current code | Location | Notes |
|------|-------------|----------|-------|
| Hyperedge consolidation | `_evaluate_consolidation_states()` | neuro_foundation.py:2999 | SPECULATIVE -> CANDIDATE -> CONSOLIDATED -> PERMANENT |
| Substrate culling | `_cull_substrate()` | neuro_foundation.py:3110 | Soft-penalize redundant pairwise synapses |
| Structural pruning | `_prune_synapses()` | neuro_foundation.py:2402 | Weight-based, activity-based, age-based |
| Node LRU pruning | `find_or_create_node()` overflow path | ng_lite.py:361 | Evicts least-recently-used when at capacity |

**Compaction decision:** `ownsCompaction: false` for v1.

NeuroGraph's consolidation/pruning is substrate-level maintenance (structural
plasticity), not conversation-level compaction. It doesn't summarize messages
or drop conversation turns — it consolidates learned patterns in the SNN.

The right v1 approach: let OpenClaw's built-in compactor handle message-level
compression (it already works), while NeuroGraph's substrate maintenance runs
independently in `afterTurn`. NeuroGraph's `compact()` can trigger an
accelerated consolidation pass if token pressure is high, but it should NOT
claim ownership of conversation compaction.

Future: `ownsCompaction: true` once NeuroGraph can reconstruct conversation
context from substrate state alone (i.e., the substrate IS the memory, not a
supplement to the message log). This requires the receptor layer (#43) and
extraction bucket architecture (#29) to be complete.

```typescript
async compact({ force }) {
  // Trigger accelerated substrate consolidation
  // Does NOT summarize or drop messages
  return { ok: true, compacted: false };
}
```

---

### afterTurn (optional)

**OpenClaw contract:** Post-turn lifecycle work after a run attempt completes.
Receives the full message list, token budget, and runtime context.

**NeuroGraph equivalent:** Second half of `on_message()` + save logic
(openclaw_hook.py:396-425)

| Step | Current code | Line | Notes |
|------|-------------|------|-------|
| SNN learning step | `graph.step()` | 396 | STDP, structural plasticity, predictions, consolidation |
| Engagement reward | `graph.inject_reward(0.1)` | 406 | Baseline heartbeat (three-factor learning) |
| Surfacing scan | `surfacing_monitor.after_step(step_result)` | 415 | Update priority queue with newly fired nodes |
| Novelty probation | `ingestor.update_probation()` | 419 | Graduate or cull probationary nodes |
| Auto-save | `save()` -> `graph.checkpoint()` + `vector_db.save()` | 424-425, 644-661 | Every N messages |
| Activation persist | `activation_persistence.save(graph, path)` | 649-651 | CES cross-session voltage sidecar |

**This is the natural home for `graph.step()`.** The current `on_message()` runs
ingest + step + save in one call. The ContextEngine lifecycle correctly separates
these: `ingest` records the experience, `afterTurn` lets the substrate learn from
it. This matches the biological sequence: perceive, THEN process.

---

### prepareSubagentSpawn (optional)

**OpenClaw contract:** Prepare context-engine-managed state before a child agent
starts. Returns a rollback function in case the spawn fails.

**NeuroGraph equivalent:** No direct equivalent today. Candidate mapping:

| Approach | Mechanism | Notes |
|----------|-----------|-------|
| Checkpoint fork | `graph.checkpoint(path, mode=CheckpointMode.FORK)` | Create a read-only snapshot for the subagent |
| Scoped NG-Lite | New `NGLite` instance seeded from parent's relevant nodes | Lighter than full SNN fork |
| No-op | Return undefined | Subagent runs without substrate context |

**Recommendation for v1:** No-op. Subagent substrate scoping is genuinely new
architecture work. Premature implementation here would create more problems than
it solves. Revisit when Observatory (#42+) needs it.

---

### onSubagentEnded (optional)

**OpenClaw contract:** Subagent lifecycle ended (completed, deleted, swept, released).
Opportunity to merge subagent learning back.

**NeuroGraph equivalent:** No direct equivalent. Candidate mapping:

| Approach | Mechanism | Notes |
|----------|-----------|-------|
| Peer bridge pattern | Treat subagent output as a peer learning event | `_write_peer_learning_event()` already handles this shape |
| Direct ingest | Feed subagent transcript through `ingestor.ingest()` | Simplest, but loses subagent's own learned structure |
| Topology merge | Merge subagent's NG-Lite state into parent substrate | Complex, requires conflict resolution |

**Recommendation for v1:** Ingest the subagent's summary/output as a message
via the normal `ingest` path. The substrate learns from the subagent's
conclusions without requiring topology merge. This is the extraction boundary
principle in action — raw experience in, let the substrate do its own learning.

---

### dispose (optional)

**OpenClaw contract:** Clean up resources.

**NeuroGraph equivalent:**
- `save()` — final checkpoint
- CES monitor `stop()` — shut down HTTP dashboard if running
- Stream parser cleanup
- No explicit destructor exists today; Python GC handles it

---

## 4. Architecture Boundary: TypeScript Wrapper, Python Engine

The ContextEngine interface is TypeScript. NeuroGraph is Python. The plugin
needs a bridge.

**Options evaluated:**

| Approach | Pros | Cons |
|----------|------|------|
| A. Child process + JSON-RPC | Clean isolation, crash safety, matches module isolation requirement | Latency per call, serialization overhead |
| B. HTTP sidecar | NeuroGraph already has CES dashboard HTTP | Heavier than needed, polling risk |
| C. Unix domain socket | Fast IPC, no port allocation | Custom protocol needed |
| D. Embedded Python (pyodide/wasm) | In-process, no IPC | Too heavy, numpy/torch won't work |

**Recommendation: Option A — Child process + JSON-RPC over stdio.**

The TypeScript plugin spawns a Python child process running a thin JSON-RPC
server. Each ContextEngine lifecycle call maps to an RPC method. The child
process holds the NeuroGraphMemory singleton for its lifetime.

This matches OpenClaw's execution model (plugins run in-process, but the
engine's heavy compute happens out-of-process). It also matches the ecosystem's
module isolation requirement — a Python crash doesn't take down the Gateway.

The JSON-RPC bridge is a translation layer, but it sits at the OpenClaw/Python
boundary, not at the substrate boundary. The substrate still receives raw
experience. The serialization is between the plugin runtime and the engine
process, not between modules.

```
OpenClaw Gateway (Node.js)
  |
  |-- neurograph-context-engine plugin (TypeScript, in-process)
  |     |
  |     |-- JSON-RPC over stdio
  |     |
  |     +-- Python child process
  |           |-- NeuroGraphMemory singleton
  |           |-- Graph, VectorDB, Ingestor, CES
  |           +-- Peer bridge (shared learning dir)
```

---

## 5. Migration Path

| Phase | What | When |
|-------|------|------|
| 0 (now) | This mapping document. No code changes. `openclaw_hook.py` stays live. | Done |
| 1 | Build the TypeScript plugin shell + JSON-RPC bridge. Register as ContextEngine. Test with `plugins.slots.contextEngine = "neurograph"`. | After #43+#49 land |
| 2 | Wire `bootstrap`, `ingest`, `assemble`, `afterTurn` to existing Python methods via RPC. `compact` returns `{ ok: true, compacted: false }`. | Same PR as Phase 1 |
| 3 | Retire `SKILL.md` and `hook:` frontmatter. Remove punch list #37, #38, #39, #40. | After Phase 2 validated |
| 4 | Add `prepareSubagentSpawn` / `onSubagentEnded` when Observatory needs them. | Future |

**Phase 1-2 do NOT require changes to `openclaw_hook.py` or any Python code.**
The TypeScript plugin calls the same Python singleton that the SKILL.md hook
currently instantiates. The only difference is who calls it and when.

---

## 6. Open Questions for Josh

1. **Token budget visibility:** Does OpenClaw pass model-specific token limits
   to `assemble()`, or does the engine need to discover them? This affects how
   many surfaced concepts NeuroGraph can inject.

2. **Session identity:** OpenClaw passes `sessionId` and `sessionKey` to every
   hook. Do these map to anything in NeuroGraph's checkpoint model, or does
   NeuroGraph always use a single global substrate regardless of session?
   (Current answer: single global substrate. Is that right going forward?)

3. **ingestBatch vs ingest:** OpenClaw can send a full turn as a batch. Should
   NeuroGraph treat user+assistant+tool messages differently, or ingest them
   identically? Currently `on_message()` doesn't distinguish speaker.

4. **CES dashboard:** The CES HTTP dashboard (`NEUROGRAPH_CES_DASHBOARD=1`)
   runs inside the Python process. With the child-process model, it still works
   but the port needs to be discoverable. Move to OpenClaw's `registerHttpRoute`
   instead?

5. **Timing:** Should this migration happen before or after the myelinated
   tracts (#53/#54)? The ContextEngine plugin replaces the *OpenClaw integration
   surface*, while tracts replace the *inter-module transport*. They're
   independent, but doing both at once is a lot of moving parts.
