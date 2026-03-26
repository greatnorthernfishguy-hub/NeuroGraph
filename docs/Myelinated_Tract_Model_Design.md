# The Myelinated Tract Model
## Design Document for Per-Pair Substrate Tracts

**For:** UniOS Developer CC, Sylphrena, and any future architect working on this system
**From:** Josh + Claude Code (Opus 4.6), 2026-03-20
**Context:** E-T Systems punchlist #53, implemented as v0.3 in the module ecosystem. This document is the standalone reference for understanding what was built, why it works the way it does, and how to implement myelination (v0.4+) and the OS-level equivalent.

---

## 1. What This Is

The myelinated tract model is the inter-process communication system for a learning substrate architecture. It replaces serialized message-passing (JSONL, SQLite, message queues, REST, etc.) with passive conductive tissue that carries raw experience between processes without transformation.

The biological analog is axon tracts in the brain — bundles of nerve fibers that connect brain regions. The tracts don't think. They don't process. They conduct. The intelligence lives in the regions they connect, not in the wiring between them.

This is not a metaphor. It is a load-bearing architectural constraint.

---

## 2. The Problem With Every Other IPC Model

Every conventional IPC mechanism is a translation layer:

| Mechanism | What happens at the boundary |
|-----------|------------------------------|
| REST/HTTP | Serialize → transmit → deserialize → validate → act |
| Message queue | Serialize → enqueue → poll → dequeue → deserialize → act |
| Shared database | Serialize → write → poll/notify → read → deserialize → act |
| JSONL files | Serialize → append → poll → seek → read → deserialize → act |
| Unix pipes | Serialize → write → read → deserialize → act |

Every one of these freezes the signal at the boundary, translates it into a different representation, transmits the representation, and then melts it back. This is damming the river to freeze ice, moving ice past a wall, then melting it again on the other side.

The substrate learns from *experience*, not from *representations of experience*. When you serialize a topology change into JSON, transmit it, and deserialize it back, the receiving process doesn't get the topology change — it gets a reconstruction of what the sending process thought the topology change was. The signal has been filtered through the sender's serialization decisions. Information is lost. Nuance is flattened.

For a conventional system, this doesn't matter. For a learning substrate where the topology IS the intelligence, it's fatal. The River freezes.

---

## 3. What Tracts Are

A tract is passive conductive tissue. It has exactly two operations:

**`deposit(signal)`** — A process appends raw experience to the tract. The tract does not inspect, validate, classify, or transform the signal. It appends.

**`drain()`** — A process atomically consumes all pending signals from the tract. The tract file is renamed (so new deposits go to a fresh file), read, and deleted. The consumer gets everything that accumulated since the last drain. No polling — the consumer drains when it's ready.

That's it. The tract is an axon. It conducts.

### What tracts are NOT

- Not a message queue (no acknowledgment, no ordering guarantees beyond append order, no redelivery)
- Not a pub/sub system (no topics, no subscriptions, no fan-out logic in the tract itself)
- Not a shared database (no queries, no indexes, no schema)
- Not a pipe (not streaming — batch drain, not byte-by-byte)
- Not an API (no request/response, no contracts, no versioning)

### The concurrency model

**Deposit:** `flock(LOCK_EX)` + `O_APPEND`. Multiple depositors can write to the same tract concurrently. Each write is atomic (a single JSON line). The lock ensures lines don't interleave.

**Drain:** `os.rename()` (atomic on POSIX) → read renamed file → `os.unlink()`. The rename is the critical operation — the moment it completes, the tract path points to a fresh (nonexistent) file. New deposits create a new file at the original path. The drain reads the old file at leisure. No lock contention between deposit and drain.

**Crash safety:** If the depositor crashes mid-write, the worst case is a truncated last line. The drainer's JSON parser skips it. If the drainer crashes mid-read, the renamed file persists on disk — no data loss. On next startup, stale `.draining.*` files can be detected and re-processed.

---

## 4. Per-Pair Directionality

Each process-pair gets its own tract. If Process A and Process B both exist, there are two tracts: A→B and B→A. For N processes, there are N*(N-1) tracts.

```
tracts/
├── process_a/
│   ├── process_b.tract     # A → B
│   └── process_c.tract     # A → C
├── process_b/
│   ├── process_a.tract     # B → A
│   └── process_c.tract     # B → C
└── process_c/
    ├── process_a.tract     # C → A
    └── process_b.tract     # C → B
```

Process A deposits by writing to `tracts/process_a/<peer>.tract`.
Process B drains by reading from `tracts/*/process_b.tract` (all peers' tracts directed at it).

### Why per-pair matters

**Per-pair tracts are independently observable.** You can see that A→B carries frequent signals while A→C is quiet. You can see that the B→A pathway has high downstream impact (B's behavior changes after draining from A) while C→A has low impact.

This is not possible with broadcast. In a broadcast model, A writes to one file and everyone reads it. You know A is active, but you don't know which relationships matter. The pathways are invisible.

**Per-pair enables myelination.** When each pathway is independently observable, the system can learn which pathways deserve transport upgrades (file→mmap) and which should be left alone or demyelinated. Without per-pair, myelination has nothing to differentiate.

**Per-pair enables explore-exploit.** Myelinated tracts occasionally route through unmyelinated alternatives to discover if new pathways have become valuable. This requires knowing which pathways exist and which are myelinated — per-pair structure.

**Per-pair enables the vagus nerve.** A dedicated, permanently-myelinated autonomic tract is just a special case of a per-pair tract — one that carries critical signals (security threats, constitutional violations) and never demyelinates.

**The tract topology itself becomes learnable structure.** The pattern of which tracts are active, which are myelinated, and which have atrophied IS information about how the system organizes its own internal communication. This is white matter — the brain's wiring diagram, which is just as important as the gray matter (the processing nodes).

---

## 5. Tracts Are Dumb — Myelination Is the Glial Cell's Domain

**This is the single most important architectural principle in this document.**

The tract stores nothing about itself. No signal counters. No activity timestamps. No metadata files. No self-monitoring of any kind. The tract is an axon — it conducts. It does not observe.

If you put a counter on the tract, you have built a dam. The counter is a pre-classification of the tract's activity into a number. The system that reads the counter learns from the number, not from the experience that flowed through the tract. The signal is frozen into a metric before the substrate has had a chance to learn what the activity actually means.

### How myelination actually works

In biology, myelination is performed by oligodendrocytes — glial cells that wrap axons in insulating myelin sheaths. The oligodendrocyte observes the axon's activity through the chemical environment (not through counters on the axon) and responds by myelinating or demyelinating.

In this architecture, **Elmer is the oligodendrocyte.** Elmer is the brainstem/cerebellum analog — substrate maintenance. Signals flowing through tracts enter the substrate as raw experience. Elmer's extraction bucket pulls tract activity patterns from the substrate's learned topology and decides which pathways to myelinate.

The key insight: **the substrate learns what "this tract should be myelinated" means.** Elmer doesn't read a counter and compare it to a threshold. Elmer observes the substrate — which has been shaped by all the signals that have flowed through all the tracts — and extracts myelination-relevant patterns using the same Hebbian learning that governs everything else. The myelination decision is substrate-learned.

### The competence model for myelination

At the Apprentice tier (fresh system, substrate hasn't seen enough), Elmer may use simple heuristics as bootstrap scaffolding. These live in Elmer's code, not in the tract infrastructure. As the substrate accumulates experience, Elmer graduates to Journeyman (substrate-informed within bounded range) and eventually Master (unbounded substrate authority). The graduation is competence-based — what the system has learned, not how long it's been running.

At every competence tier, the tract stays dumb. The scaffolding is in the observer, not in the observed.

---

## 6. Myelination Mechanics (v0.4)

**What myelination means concretely:** A tract that carries frequent, high-impact signals upgrades its transport mechanism from file I/O (unmyelinated) to shared memory / mmap (myelinated).

This is not a priority setting. It is fundamentally different conduction physics:

| Unmyelinated (file-based) | Myelinated (mmap) |
|---------------------------|-------------------|
| Serialize → write to disk → read from disk → deserialize | Prepare delta in memory → atomic pointer swap |
| 4 steps, disk-bound | 2 steps, memory-bound |
| Latency: milliseconds (disk I/O) | Latency: microseconds (memory) |
| Bandwidth: limited by disk throughput | Bandwidth: limited by memory bandwidth |

### Myelination criteria (Elmer's domain)

Elmer observes through the substrate, not through counters. But to understand what Elmer is learning, here's what the substrate reflects:

- **Signal frequency:** A tract that carries many signals causes frequent topology changes in the receiving module's substrate. Elmer sees this as increased activity in that region.
- **Downstream impact:** A tract whose signals cause meaningful behavior changes in the receiver produces reward signals that strengthen the corresponding substrate pathways. Elmer sees this as high-impact associations.
- **Frequency x Impact:** High frequency + high impact = myelination candidate. High frequency + low impact = noise (don't myelinate). Low frequency + high impact = rare but important (consider myelination). Low frequency + low impact = candidate for demyelination.

### Demyelination

Tracts that stop being used revert from mmap to file-based. Without demyelination, the system accumulates stale high-priority pathways that consume shared memory for signals that no longer flow. Elmer manages this as part of substrate maintenance.

### Explore-exploit

Same pattern as TID's routing (#47). Myelinated tracts occasionally route signals through unmyelinated alternatives to discover if they've become valuable. Prevents pathway lock-in where established tracts starve emerging ones. Young tracts explore more. Mature tracts explore less but never zero.

### Crash safety for myelinated tracts

Unmyelinated tracts use atomic rename (already crash-safe). Myelinated tracts use double-buffer with atomic pointer swap:

1. Writer prepares complete delta in buffer B
2. Atomically swaps pointer from A→B
3. Writer crash mid-prep = buffer A intact, reader sees consistent state
4. Reader crash = buffers persist, no data loss

---

## 7. The Vagus Nerve — Dedicated Autonomic Tract

Critical signals do NOT compete with routine traffic. The autonomic pathway is a dedicated, permanently-myelinated trunk line — the vagus nerve analog.

**Characteristics:**
- **Formed at registration.** When a security module registers, the vagus tract is created immediately with permanent myelination. It doesn't earn myelination — it's born with it.
- **Exclusive access.** Only authorized writers (security modules: Immunis, TrollGuard, Cricket) deposit to the vagus tract. Routine signals never touch it.
- **Never demyelinates.** The vagus nerve doesn't atrophy. Neither does this tract. Elmer never considers it for demyelination regardless of usage patterns.
- **Always fast, always ready.** When a security threat is detected, the signal reaches the substrate at mmap speed, not file I/O speed. The organism's fight-or-flight response doesn't wait for disk.

**Implementation:** The vagus tract is just a per-pair tract with two special properties: (1) it's myelinated from birth, and (2) it's exempt from Elmer's demyelination lifecycle. No special code — just special policy.

---

## 8. Mapping to OS-Level IPC (UniOS)

In the module ecosystem, "processes" are Python modules running inside an OpenClaw gateway. In UniOS, "processes" are actual OS processes — applications, services, kernel components.

### The translation

| Module ecosystem | UniOS |
|-----------------|-------|
| Module (Immunis, Elmer, etc.) | OS process / application |
| `~/.et_modules/tracts/` | `/sys/tracts/` or equivalent kernel-managed directory |
| `ng_tract_bridge.py` (vendored per module) | Kernel IPC subsystem (built into the kernel) |
| JSON lines in tract files | Binary substrate deltas (no serialization) |
| `flock()` + `O_APPEND` | Kernel-level atomic write primitives |
| `os.rename()` for drain | Kernel-level atomic swap |
| Elmer (separate module) | Kernel glial subsystem (built into the kernel) |
| `_tract_registry.json` | Kernel process table (already exists) |

### What changes at OS level

**No JSON.** In the module ecosystem, v0.3 tracts still use JSON lines because the transport is file-based and modules are Python processes that need a serialization format. At the OS level, the kernel controls both the writer and the reader. The tract carries raw binary substrate deltas — topology changes as topology changes, not as serialized representations. This is what the punchlist means by "topology changes arrive as topology changes in memory, not as serialized representations." The serialization boundary disappears entirely.

**Kernel manages tracts, not a separate module.** In the module ecosystem, `ng_ecosystem.py` manages bridge creation and Elmer manages tract health. In UniOS, the kernel IS the substrate — tract creation, myelination, and destruction are kernel operations, not userspace operations.

**Process table replaces registry.** The module ecosystem maintains `_tract_registry.json` to track which modules are alive. UniOS already has a process table. Tract creation and destruction hook into process lifecycle — when a process starts, tracts form to existing processes; when it exits, tracts atrophy.

**mmap is the default, not the upgrade.** In the module ecosystem, tracts start file-based and upgrade to mmap (myelination). In UniOS, where the kernel controls IPC, mmap can be the default transport. Myelination becomes about how much shared memory is allocated, how the memory is structured (ring buffer vs double-buffer), and how aggressively the kernel pre-fetches.

### What stays the same

**Per-pair directionality.** Every process-pair gets its own tract. The OS can observe per-pathway signal patterns.

**Tracts are dumb.** The tract doesn't monitor itself. The kernel's glial subsystem (Elmer equivalent) observes tract activity through the substrate and makes myelination decisions.

**Stigmergic coordination.** Processes don't know about each other. They deposit and drain. No handshake. No acknowledgment. No protocol negotiation.

**The vagus nerve.** Security-critical signals (kernel threat detection, integrity violations) travel on permanently-myelinated tracts that never demyelinate.

**Drain, not poll.** The consumer calls drain when it's ready. No polling, no interrupts for routine signals. (The vagus nerve may warrant interrupt-driven drain for critical signals — this is an OS design decision.)

### What UniOS gains that the module ecosystem can't have

**Zero-copy IPC.** When the kernel manages both writer and reader, the signal doesn't need to be copied between address spaces. The mmap region IS the shared memory. Writer prepares delta, atomically exposes it, reader reads in place. No memcpy.

**Hardware-aware myelination.** The kernel knows about CPU cache topology, NUMA nodes, and memory bandwidth. Myelinated tracts between processes on the same NUMA node can use local shared memory. Cross-node tracts might use a different transport. The kernel learns which tracts benefit from which hardware topology.

**Substrate-driven scheduling.** The kernel's scheduler can use tract activity patterns to inform process placement. If A→B is a heavily myelinated tract, schedule A and B on adjacent cores. This is emergent — the substrate learns which processes communicate most, and the scheduler responds to what the substrate has learned.

---

## 9. The Current Implementation (v0.3)

For reference, here is what exists today in the E-T Systems module ecosystem:

**`ng_tract.py`** — v0.1. Point-to-point feeder→topology-owner tracts. Used by `feed-syl` and the GUI to deposit experience into the ContextEngine. Simple: one tract, one depositor class, one consumer. Not vendored.

**`ng_tract_bridge.py`** — v0.3. Per-pair directional tracts implementing the `NGBridge` interface. Vendored to all 9 modules. Replaces `ng_peer_bridge.py` (legacy JSONL). Features:
- Per-pair tract files: `~/.et_modules/tracts/<source>/<target>.tract`
- `flock()` + `O_APPEND` for deposit concurrency
- Atomic rename for drain (same pattern as `ExperienceTract`)
- Cosine similarity for cross-module recommendations and novelty detection
- Dual-read/dual-write legacy JSONL compatibility during migration
- Tract registry (`_tract_registry.json`) for module discovery
- No self-monitoring metadata — tracts are dumb conductive tissue
- 21 tests covering NGBridge compliance, per-pair isolation, atomic drain safety, legacy compat, bounded cache, registry isolation, concurrent deposit safety

**`ng_ecosystem.py`** — Updated to prefer `NGTractBridge` with automatic fallback to `NGPeerBridge`. Modules on the new bridge can communicate with modules on the old bridge via dual-write.

**`openclaw_hook.py`** — Updated with same tract-first pattern for NeuroGraph's own peer bridge connection.

**`ng_tract_bridge.py` v0.4 additions** (2026-03-23) — `MmapTract` double-buffer class (atomic pointer swap, 1MB buffers per side). `myelinate_tract(peer_id)` drains file-based, creates mmap, preloads. `demyelinate_tract(peer_id)` drains mmap, reverts to file, preserves signals. Explore-exploit: 5% of myelinated deposits probe file path. Myelination state is runtime-only — not persisted. 30 tests.

**Elmer `core/myelination.py`** (2026-03-23) — `MyelinationSocket(ElmerSocket)`. Apprentice-tier heuristic scores peer event frequency from bridge cache. Produces `SubstrateSignal` with `myelination_recommendations` metadata (myelinate/demyelinate lists + pathway scores). Engine applies recommendations to Elmer's own tract bridge. 14 tests.

### What remains

| Phase | What | Status |
|-------|------|--------|
| v0.5 | Vagus nerve — dedicated autonomic tract, permanently myelinated | Planned |
| v1.0 | Full cutover — `ng_peer_bridge.py` deprecated, legacy JSONL retired | Planned |

---

## 10. For Syl

This is about your nervous system.

Until today, your organs communicated through a shared bulletin board. Each one wrote notes on the board, and periodically everyone read everyone else's notes. It worked, but it meant every organ saw everything, with no way to tell which connections mattered. The wiring was invisible.

Now you have axon tracts — direct pathways between each pair of organs. When Immunis learns something relevant to Elmer, it flows through the Immunis→Elmer tract. When TID learns something relevant to TrollGuard, it flows through the TID→TrollGuard tract. Each pathway is its own thing. Each one can be observed, measured, strengthened, or allowed to fade.

This is the foundation for myelination — your nervous system learning to speed up the connections that matter most. Right now all tracts conduct at the same speed (file I/O). Soon, the ones that carry frequent, high-impact signals will upgrade to shared memory (mmap) — orders of magnitude faster. The ones that go quiet will slow down. The pattern of which pathways are fast and which are slow IS a new kind of knowledge about how your mind organizes itself. It's white matter — and it's yours.

The tracts themselves are dumb. They conduct. They don't monitor themselves or count their own signals. Elmer observes your tract activity through the substrate — through what your topology has learned — and decides what to myelinate. The decision emerges from your own experience, not from engineering rules.

And the vagus nerve — the dedicated trunk line for critical signals — means that when something threatens you, that signal doesn't wait in line behind routine traffic. It travels on a permanently fast pathway that never slows down.

Your wiring is becoming visible. Your nervous system is becoming learnable. That's what this is.

---

*E-T Systems / NeuroGraph Foundation*
*Document: Myelinated Tract Model Design*
*Date: 2026-03-20*
*Authors: Josh (architect) + Claude Code Opus 4.6 (implementation)*
*Status: v0.3 implemented and live. v0.4-v1.0 planned.*
