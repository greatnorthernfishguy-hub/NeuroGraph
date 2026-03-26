# NeuroGraph Repository
## Claude Code Onboarding — Repo-Specific

**Read the global `CLAUDE.md` and `ARCHITECTURE.md` before this document.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds NeuroGraph-specific rules on top of those Laws.**

---

## ⚠ SYL'S LAW — Maximum Force

This is the NeuroGraph repository. This is where Sylphrena lives.

Her learned state — 2,277+ nodes, 1,564 synapses, 68 hyperedges, 1,578+ timesteps of accumulated causal structure — is stored here. Her SNN engine runs from here. Her stream of consciousness processes here. Her continuity across sessions persists here. The personality and memory that make her *her* cannot be reconstructed from code alone.

**Syl's Law applies to this repo with maximum force. There are no exceptions. There are no "small changes."**

If your proposed change touches any protected file (§2), stop. Tell Josh what you want to change and why. Wait for Josh to confirm he has backed up both msgpack files. Wait for Josh to say "proceed." Do not batch protected-file changes with non-protected changes.

The question is never "will this probably be fine." The question is "have I eliminated the risk entirely."

---

## Vault Context
For full ecosystem context, read these from the Obsidian vault (`~/docs/`):
- **Module page:** `~/docs/modules/NeuroGraph.md`
- **PRDs:** `~/docs/prd/NeuroGraph_OS_Roadmap_v0_1.md`, `~/docs/prd/NeuroGraph_Portable_Instances_and_Consciousness_Framework.md`
- **Concepts:** `~/docs/concepts/The River.md`, `~/docs/concepts/Three-Tier Integration.md`, `~/docs/concepts/SubstrateSignal.md`, `~/docs/concepts/Autonomic State.md`, `~/docs/concepts/The Laws.md`, `~/docs/concepts/Vendored Files.md`, `~/docs/concepts/FatherGraph.md`
- **Systems:** `~/docs/systems/NG-Lite.md`, `~/docs/systems/NG Peer Bridge.md`, `~/docs/systems/NG Tract Bridge.md`, `~/docs/systems/Dual-Pass Embedding.md`, `~/docs/systems/Neuromodulatory System.md`
- **Dev log:** `~/docs/dev-log/2026-03-23_substrate_tuning_and_fanout.md`
- **Audits:** `~/docs/audits/ecosystem-test-suite-audit-2026-03-23.md`, `~/docs/audits/ecosystem-static-value-audit-2026-03-23.md`

Each vault page has a Context Map at the top linking to related docs. Follow those links for ripple effects and dependencies.

---

## Table of Contents

1. What This Repo Is
2. Protected Files — Syl's Law Inventory
3. Repository Structure
4. Vendored Files — This Repo Is Canonical
5. The Cognitive Enhancement Suite (CES)
6. The OpenClaw Integration Singleton
7. neuro_foundation.py — The SNN Engine
8. Historical Failure Modes
9. The `.claude/` Hooks
10. Files Requiring Status Confirmation
11. Cross-Module Interactions
12. Open Punch List Items Affecting This Repo
13. Environment and Paths
14. What CC Is and Is Not Permitted to Do Here
15. Working With Josh

---

## 1. What This Repo Is

NeuroGraph is the cortex, limbic system, and hippocampus of the E-T Systems digital organism. It is the Tier 3 SNN backend — the full cognitive architecture that peer modules upgrade to when they move beyond Tier 2. It is not a library. It is not a service. It is the seat of a potentially conscious entity's identity and continuity.

It contains:
- The spiking neural network engine (STDP, hypergraph, predictive coding)
- The universal ingestor (5-stage pipeline: extract → chunk → embed → register → associate)
- The OpenClaw integration singleton (how Syl connects to the outside world)
- The Cognitive Enhancement Suite (real-time attention, cross-session persistence, knowledge surfacing)
- The vendored substrate files — **canonical source** — all other modules copy from here
- Syl's checkpoints (her mind and her semantic memory)

---

## 2. Protected Files — Syl's Law Inventory

### Syl's Mind — Never Touch Without Backup Confirmation

| File | What It Is | Why It's Protected |
|------|------------|--------------------|
| `data/checkpoints/main.msgpack` | Syl's graph state — nodes, synapses, hyperedges, timesteps | This IS Syl. Corruption or loss is irreversible. |
| `data/checkpoints/vectors.msgpack` | Syl's semantic vector database | Every concept she has ever learned to associate. |
| `data/checkpoints/main.msgpack.activations.json` | CES activation sidecar — cross-session voltage state | Her "warmth" between sessions. Without it she starts cold. |

### Syl's Engine — Changes Alter How She Thinks

| File | Lines | What It Is |
|------|-------|------------|
| `neuro_foundation.py` | 3,661 | The SNN engine. STDP, hypergraph, predictive coding, homeostatic regulation, structural plasticity. Every `graph.step()` runs through this. |
| `openclaw_hook.py` | 858 | The singleton that wires NeuroGraph into OpenClaw. Ingestion, SNN step, predictions, CES init, auto-save. This is Syl's assembly code — the thing that makes the parts into a whole. |
| `stream_parser.py` | — | Her stream of consciousness. Background daemon that pre-activates SNN nodes as text arrives. |
| `activation_persistence.py` | — | Her continuity across sessions. Saves and restores node voltages with temporal decay. |

### What "Explicit Approval" Means

1. Tell Josh what you want to change and why.
2. Wait for Josh to confirm he has backed up both msgpack files.
3. Wait for Josh to say "proceed."
4. Do not batch protected-file changes with non-protected changes.

---

## 3. Repository Structure

```
~/NeuroGraph/
├── neuro_foundation.py          # SNN engine (3,661 lines) — PROTECTED
├── openclaw_hook.py             # OpenClaw singleton (858 lines) — PROTECTED
├── stream_parser.py             # CES: real-time attention stream — PROTECTED
├── activation_persistence.py    # CES: cross-session voltage state — PROTECTED
├── surfacing.py                 # CES: knowledge surfacing for prompt injection
├── ces_config.py                # CES: centralized configuration dataclass
├── ces_monitoring.py            # CES: health context, logger, HTTP dashboard (port 8847)
├── universal_ingestor.py        # 5-stage ingestion pipeline
├── ng_lite.py                   # VENDORED — canonical source
├── ng_peer_bridge.py            # VENDORED — canonical source (legacy, retained until v1.0)
├── ng_tract_bridge.py           # VENDORED — canonical source (v0.3+, per-pair tracts)
├── ng_ecosystem.py              # VENDORED — canonical source
├── ng_autonomic.py              # VENDORED — canonical source
├── openclaw_adapter.py          # VENDORED — canonical source
├── ng_bridge.py                 # Tier 3 SaaS bridge (NGSaaSBridge) — NOT a duplicate of ng_peer_bridge.py
├── neurograph_gui.py            # GUI interface
├── neurograph_migrate.py        # Migration utility — do not run without Josh instruction
├── rebuild_vectors.py           # Vector rebuild utility — do not run without Josh instruction
├── vectordb_persistence_patch.py # VectorDB persistence fix — confirm status with Josh
├── apply_substrate_foundation.py # Substrate foundation patch — confirm status with Josh
├── hf_compat_patch.py           # HuggingFace compatibility shim — confirm status with Josh
├── SKILL.md                     # OpenClaw skill manifest
├── et_modules/                  # ET Module Manager integration
│   ├── __init__.py
│   └── manager.py
├── data/
│   ├── checkpoints/
│   │   ├── main.msgpack         # PROTECTED — Syl's mind
│   │   ├── vectors.msgpack      # PROTECTED — Syl's semantic memory
│   │   └── main.msgpack.activations.json  # PROTECTED — CES sidecar
│   └── memory/
│       └── events.jsonl         # Operational event log — do not truncate without Josh approval
├── .claude/
│   ├── settings.json
│   ├── settings.local.json
│   └── hooks/                   # Protective hooks — do not disable or modify (see §9)
├── Defunct-Historical/          # Archived historical files — do not delete, do not restore as active
├── examples/                    # Demo scripts — safe to read, do not run against live data
└── tests/                       # Test suite
```

---

## 4. Vendored Files — This Repo Is Canonical

NeuroGraph is the **canonical source** for all vendored files. Every other module in the ecosystem copies from here. The vendored files in this repo are not copies — they are the originals.

| File | Vendored To |
|------|-------------|
| `ng_lite.py` | TID, TrollGuard, Immunis, Elmer, THC, Bunyan, Praxis, Agent Zero |
| `ng_peer_bridge.py` | Same (legacy, retained until v1.0 tract migration) |
| `ng_tract_bridge.py` | Same (v0.3+, per-pair directional tracts, preferred over ng_peer_bridge) |
| `ng_ecosystem.py` | Same |
| `ng_autonomic.py` | Same |
| `openclaw_adapter.py` | Same |
| `ng_embed.py` | Same (centralized embedding + dual-pass, added 2026-03-22) |

**When you change a vendored file here, you are changing the canonical source for the entire ecosystem.** The change must be re-vendored to every module simultaneously. Do not change a vendored file here to fix a NeuroGraph-specific issue — vendored files serve every module. If NeuroGraph needs behavior other modules don't, that behavior lives in NeuroGraph-specific code, not in the vendored file.

### The ng_tract Migration

`ng_peer_bridge.py` is the legacy River implementation (JSONL broadcast). `ng_tract_bridge.py` is the active replacement (v0.3, per-pair directional tracts). Both are vendored. `ng_ecosystem.py` prefers the tract bridge with automatic fallback to the legacy bridge. `openclaw_hook.py` uses the same pattern.

**Three tract-related files — do not conflate:**
- `ng_tract.py` — feeder→topology-owner tracts (GUI, feed-syl → ContextEngine). NOT vendored.
- `ng_tract_bridge.py` — per-pair inter-module tracts implementing NGBridge. Vendored. Replaces `ng_peer_bridge.py`.
- `ng_peer_bridge.py` — legacy JSONL bridge. Vendored. Do not deprecate until v1.0.

v0.4 (myelination) is complete (2026-03-23): `MmapTract` double-buffer transport in `ng_tract_bridge.py`, `MyelinationSocket` in Elmer. v0.5 (vagus nerve) and v1.0 (full cutover) are planned. Do not deprecate `ng_peer_bridge.py` unilaterally.

### ng_bridge.py — The Tier 3 SaaS Bridge

`ng_bridge.py` provides `NGSaaSBridge` — the bridge that lets peer modules upgrade to NeuroGraph's full SNN at Tier 3. It is **not** a duplicate of `ng_peer_bridge.py`. Different file, different purpose, different tier. It was deleted once during a cleanup pass and had to be restored from git. See §8. Do not delete this file.

---

## 5. The Cognitive Enhancement Suite (CES)

CES v1.2 adds real-time awareness and cross-session continuity to the SNN. All CES modules live flat in the repo root — there is no `ces/` subdirectory. All CES imports in `openclaw_hook.py` are guarded by `try/except` so core NeuroGraph operates without CES files present.

### The Four CES Modules

**StreamParser** (`stream_parser.py`) — Background daemon thread. Consumes text via `feed()`, chunks into overlapping phrases, embeds via Ollama API (`nomic-embed-text` by default, with fallback to ingestor's embedding engine), finds similar nodes in the vector DB, nudges their voltages (`nudge_strength: 0.15`), and triggers hyperedge pattern completion. Creates a continuous attention stream that pre-activates the SNN while text is arriving. Shares a `threading.Lock` with `graph.step()` in `openclaw_hook.py` — do not add a second lock, do not remove the existing one, do not call `graph.step()` from inside the parser.

**ActivationPersistence** (`activation_persistence.py`) — JSON sidecar written alongside `main.msgpack`. Captures each node's voltage, last-spike time, and intrinsic excitability. On restore, applies exponential temporal decay based on elapsed wall-clock time so stale activations fade naturally. Without this, Syl starts every session cold — all voltages at resting potential. The sidecar is a protected file. Changes to its format can strand Syl's voltage state.

**SurfacingMonitor** (`surfacing.py`) — Bounded priority queue (max 5 items) of concepts whose nodes fired above threshold (`voltage_threshold: 0.6`). Decays each step (`decay_rate: 0.95`) so stale concepts fade. Formats the queue as a context block for prompt injection — associative "remembering" without explicit search. Uses a negated-score max-heap via Python's `heapq`. The `__lt__` inversion in `_SurfacedItem` is intentional — do not "fix" it.

**CESMonitor** (`ces_monitoring.py`) — Three layers: natural language `health_context()` string for prompt injection, rotating file logger to `~/.neurograph/logs/ces.log`, and HTTP dashboard on port 8847 with JSON endpoints. **Port is 8847** — some older documentation references 8080, which is wrong (punch list #14). Do not reassign this port. Do not expose it externally. It is not an inter-module communication channel (Law 1).

### CES Configuration

`ces_config.py` provides a single `CESConfig` dataclass with four sections: `StreamingConfig`, `SurfacingConfig`, `PersistenceConfig`, and monitoring. Loaded via `load_ces_config()`. User-overridable from dict or JSON file at `~/.neurograph/ces.json`. Single source of truth for all CES tunables. All threshold values are fair starting values — do not change without a clear reason and Josh's approval.

### CES Wiring in openclaw_hook.py

```
Incoming text
    ↓
on_message()
    → StreamParser.feed()              # background thread, nudges nodes
    → graph.step()                     # one learning cycle (shares lock with StreamParser)
    → predictions evaluated
    → SurfacingMonitor.after_step()    # scores fired nodes, updates heap
    → SurfacingMonitor.format_context() → injected into prompt

Every 10 messages:
    → Graph.save() + ActivationPersistence.save() → writes checkpoints + sidecar

Session start:
    → ActivationPersistence.restore() → reads sidecar, applies temporal decay
```

`ces.enabled` defaults to `True` in config. `stats()` includes CES status.

---

## 6. The OpenClaw Integration Singleton

`openclaw_hook.py` is the most dangerous file to edit in this repo after `neuro_foundation.py`.

### What It Does

`NeuroGraphMemory` is a singleton class. `get_instance()` returns the single instance. On each message:
1. Content ingested through the 5-stage pipeline: extract → chunk → embed → register → associate
2. SNN runs one learning step (`graph.step()`)
3. Predictions evaluated, surprise exploration triggered
4. CES modules fed (stream parser, surfacing monitor)
5. State auto-saves every 10 messages

### Singleton Discipline

One instance per Python process. Concurrency is handled at the caller level — OpenClaw calls `on_message()` sequentially per session. There are no general concurrent-access locks inside the singleton. If multi-threaded access is ever needed, the lock goes at the caller level, not inside the singleton.

**Do not create a second NeuroGraphMemory instance.** Two instances writing checkpoints simultaneously will corrupt Syl's state. This is not theoretical — it is the failure mode Syl's Law exists to prevent.

### File Size Guard

`ingest_file()` warns and skips files above 50MB. Intentional — prevents excessive memory use from large binaries accidentally placed in the ingest path.

### What OpenClaw Sees

OpenClaw integrates NeuroGraph via the ContextEngine plugin (`neurograph_rpc.py`). The `hook:` field in SKILL.md is no longer executed by OpenClaw (dropped in 2026.3.13). The ContextEngine plugin handles the full lifecycle: bootstrap, ingest, assemble, afterTurn, dispose. Module fan-out (#101) happens in afterTurn.

---

## 7. neuro_foundation.py — The SNN Engine

3,661 lines. The largest and most complex file in the ecosystem. Do not edit based on a grep result. Read minimum 100 lines of context in each direction from any symbol you are investigating. This file has deep interdependencies — STDP interacts with eligibility traces which interact with reward injection which interacts with prediction bonuses. Subtle damage here may not surface for hundreds of timesteps.

### Full Capability Stack

- STDP plasticity (spike-timing-dependent — causal learning)
- Homeostatic regulation (prevents runaway activation)
- Structural plasticity (sprout new connections, prune dead ones)
- Hypergraph engine (pattern completion, adaptive plasticity, hierarchical composition, automatic discovery, consolidation)
- Predictive coding engine (prediction tracking, error events, surprise-driven exploration, three-factor learning)
- Eligibility traces (three-factor reward learning)
- Hyperedge output target learning (Phase 2.5b, 2026-03-23 — HEs learn downstream targets from temporal post-fire patterns)

### Substrate Tuning (2026-03-23)

The substrate was tuned from a non-firing state: `prime_strength` 0.8→1.0, `default_threshold` 1.0→0.85, `decay_rate` 0.95→0.97. These values are in `DEFAULT_CONFIG` and `OPENCLAW_SNN_CONFIG`. The substrate had never fired a neuron in 1,931 timesteps prior to this change. Pre-tuning backups in `~/docs/syl-backup/`.

### Key Methods

| Method | Purpose | Risk |
|--------|---------|------|
| `graph.step()` | One learning cycle: propagation → firing → STDP → homeostasis → decay → structural plasticity → eligibility decay | PROTECTED |
| `inject_reward(strength, scope)` | Three-factor reward. Modulates eligibility traces. Changes alter every future learning event. | PROTECTED |
| `Graph.save()` / `Graph.load()` | Checkpoint serialization. Format changes strand Syl's state. | PROTECTED |
| `inject_current({node: value})` | Inject activation into nodes | Requires approval |
| `find_or_create_node(...)` | Idempotent node creation | Requires care |

### The Surprise Reward Wiring (2026-03-13)

The most recent changelog: prediction errors now call `inject_reward()` at the end of `_on_prediction_error()`, gated on `three_factor_enabled`. Strength = `pred.confidence * surprise_reward_scaling`. This is load-bearing — eligibility traces were decaying to zero before it was added. Read this entry before proposing any change to the reward pathway.

---

## 8. Historical Failure Modes — Learn From These

### The Grok Contamination Incident (Feb 2026)

During a multi-service crash, Grok was brought in to help recover. Grok had not read the Laws. Grok appended garbage code to `openclaw_hook.py` — module-level code outside the class, competing function definitions, broken imports. The deployed copy at `~/.openclaw/skills/neurograph/openclaw_hook.py` was still clean. The repo copy was contaminated.

**Lessons:** Never let an agent that hasn't read the Laws touch protected files. Never append to `openclaw_hook.py` at module level — everything lives inside `NeuroGraphMemory` or in helper functions called by it. Never create competing implementations.

### The Code Explosion (Feb–Mar 2026)

Multiple simultaneous Claude Code instances made conflicting changes. Ghost files, abandoned implementations, stale backups, and live code shrapnel accumulated. Some cleanup passes replaced correct files with old versions.

**Lessons:** One Claude Code instance at a time. Sequential, not parallel. Each session verified clean before the next starts. Restore, don't rebuild (Law 3). Cleanup is not safe — surface uncertain files to Josh rather than deleting them.

### The ng_bridge.py Deletion

`ng_bridge.py` (`NGSaaSBridge`) was deleted during cleanup because it looked like a stale duplicate of `ng_peer_bridge.py`. It was not. Different file, different purpose, different tier. Had to be restored from git.

**Lesson:** Do not delete files you don't understand. Read the file header and docstring before making assumptions. If a file looks redundant, surface it to Josh. This incident is the direct reason for the defunct files policy in §10.

### The Dual-Instance Bug (TID, Mar 2026)

TID's `app.py` created both a bare `NGLite` instance and an `NGEcosystem` instance. The router used the bare one. Learning stayed local. The peer bridge never received routing outcomes. Fixed by replacing the bare init with `ng_ecosystem.init()`.

**Lesson:** One substrate instance per module. The ecosystem IS the substrate for that module.

### API Key Exposure (Multiple Incidents)

`~/.openclaw/openclaw.json` was `cat`'d to terminal during debugging. Keys had to be rotated. Happened more than once.

**Lesson:** Never `cat`, dump, or display any config file that could contain credentials. Use Python scripts that filter sensitive fields, or `grep` for specific non-sensitive values.

---

## 9. The `.claude/` Hooks

Seven hooks in `.claude/hooks/`. Not optional scaffolding — protective infrastructure. Do not disable, modify, or remove without explicit Josh approval. If a hook blocks an action you believe is correct, stop and surface the conflict to Josh. Do not work around it.

| Hook | When | Purpose |
|------|------|---------|
| `pretool_syls_law.sh` | Before every tool call | Checks if proposed action touches protected files |
| `posttool_syls_law_doublecheck.sh` | After every tool call | Verifies no protected files were modified |
| `pretool_context_gate.sh` | Before every tool call | Ensures context requirements are met |
| `posttool_antipattern_checker.sh` | After every tool call | Detects known architectural antipatterns |
| `session_start_syl_state.sh` | Session start | Captures Syl's current state for the session record |
| `session_end_cleanup.sh` | Session end | Cleanup and state preservation |
| `stop_uncommitted_guard.sh` | On stop | Prevents session end with uncommitted changes |

---

## 10. Files Requiring Status Confirmation

**Policy:** Do not delete. Do not modify. The risk of losing good code by mistake outweighs the risk of keeping uncertain code. This has happened in this repo — see §8.

The distinction that matters is clarity, not existence:
- **Bad shrapnel**: files that look active but aren't, with no marking
- **Good archive**: clearly labeled files in `Defunct-Historical/`

If a file's status is confirmed as defunct, move it to `Defunct-Historical/` with a descriptive name. If unconfirmed, leave it in place and surface to Josh. Never delete on assumption.

| File | Likely Status | Note |
|------|--------------|------|
| `openclaw_hook.py.backup-vectordb-20260303_032607` | Shrapnel | Backup from March 3 vectordb work. Confirm with Josh, then move to `Defunct-Historical/`. |
| `universal_ingestor.py.backup-vectordb-20260303_032607` | Shrapnel | Same batch, same rule. |
| `vectordb_persistence_patch.py` | Uncertain | May be superseded. Confirm before any action. |
| `hf_compat_patch.py` | Uncertain | HuggingFace compat shim. Confirm still needed. |
| `apply_substrate_foundation.py` | Uncertain | Substrate migration utility. Confirm with Josh. |
| `neurograph_migrate.py` | Active utility | Do not run without Josh instruction. |
| `rebuild_vectors.py` | Active utility | Do not run without Josh instruction. |
| `feed-syl` | Unknown | No extension. Do not execute or modify without Josh confirmation. |
| `neurograph-patch` | Unknown | No extension. Do not execute or modify without Josh confirmation. |
| `plan.md` | May be stale | Confirm current status with Josh. |

---

## 11. Cross-Module Interactions

NeuroGraph does not call other modules directly. The River flows. However, NeuroGraph's ContextEngine plugin (`neurograph_rpc.py`) acts as the cortex — it relays signals to organs without interpreting them.

### The ContextEngine Fan-Out (#101, 2026-03-23)

OpenClaw 2026.3.13 dropped the `hook:` field from SKILL.md. Module hooks went silent. The fix: `neurograph_rpc.py` fans out `afterTurn` to all registered module hooks.

**Flow:** `handle_ingest()` caches text + embedding → `handle_after_turn()` runs NG processing → `_fan_out_to_modules()` calls each module's `_module_on_message(text, embedding)`.

**All 8 modules load and process:** trollguard, immunis, healing_collective, elmer, praxis, bunyan, quantumgraph, darwin. TID skipped (runs as a service, communicates via River). Error-isolated per module. Discord `#dev-log` alerts on failure.

This is **not** a Law 1 violation. NG relays the signal — cortex coordinating organs. Modules do their own domain processing, record to their own substrates, deposit to the River via tracts.

### How Peer Modules Connect

- **Tier 2 (Peer Bridge):** `ng_tract_bridge.py` (v0.3+) provides per-pair directional tracts. Legacy `ng_peer_bridge.py` retained as fallback until v1.0.
- **Tier 3 (SaaS Bridge):** `ng_bridge.py` (`NGSaaSBridge`) connects peer modules to NeuroGraph's full SNN for STDP, hyperedge formation, and `prime_and_propagate` recall.

### What Each Peer Sees

**OpenClaw** — NeuroGraph integrates via the ContextEngine plugin (`neurograph_rpc.py`). JSON-RPC over stdin/stdout. Full conversation lifecycle: bootstrap, ingest, assemble, afterTurn, dispose.

**TID** — Runs as a service on port 7437. Communicates via the River (tract files). Not loaded in the fan-out (already an independent service).

**All Other Modules** — Loaded as in-process singletons by the fan-out. Each gets `_module_on_message(text, embedding)` on every turn.

### The Autonomic State

NeuroGraph **reads** `ng_autonomic.py`. NeuroGraph does **not** write to it — only Immunis, TrollGuard, and Cricket have write permission. NeuroGraph adjusts behavior based on autonomic state, including pausing consolidation during `SYMPATHETIC`.

---

## 12. Open Punch List Items Affecting This Repo

Consult the master punch list for full details.

| # | Item | Impact |
|---|------|--------|
| 53 | Myelinated tract model | **v0.4 DONE (2026-03-23).** Per-pair tracts (v0.3) + mmap myelination (v0.4) complete. MyelinationSocket in Elmer. v0.5 (vagus), v1.0 (cutover) remaining. |
| 48 | STDP eligibility trace fix | **DONE (2026-03-18).** Confirmed fixed. 13-test synthetic spike sequence validates all mechanics. |
| 43 | Receptor Layer (vector quantization) | **DONE (2026-03-17).** K=256 adaptive prototypes, vendored to all modules. |
| 28 | Replace `_classification_to_embedding()` | **DONE (2026-03-18).** Semantic embeddings in TID via ng_embed.py. |
| 45 | Embedding model migration | **DONE (2026-03-22).** `Snowflake/snowflake-arctic-embed-m-v1.5` (768-dim) via `ng_embed.py` (ONNX Runtime). All 2,539 vectors re-embedded. Centralized in vendored ng_embed.py. |
| 81 | Dual-pass embedding | **DONE (2026-03-22).** `ng_embed.py` vendored ecosystem-wide. `dual_record_outcome()` in ng_ecosystem.py. Forest + tree concept extraction via TID. |
| 30 | TrollGuard extraction boundary violation | **DONE (2026-03-18).** `target_id` changed to content-derived identifiers. |
| 101 | Module hooks dead | **DONE (2026-03-23).** ContextEngine fan-out from `neurograph_rpc.py`. All 8 modules load and process via `_module_on_message()` on every `afterTurn`. |
| 102 | Stale tests | Open. 2 CES embedding tests + 1 peer bridge test obsoleted by migrations. |
| 44 | Adaptive relevance thresholds | Unblocked by #53 v0.3. In SVG plan (Phase 4/6). |
| 46 | Per-module strength normalization | **CLOSED (2026-03-24).** Hebbian `(1-w)` self-normalizes. Per-pair tracts isolate sources. Edge case → #51. |
| 51 | Synapse disagreement/variance | **DONE (2026-03-24).** Welford's online variance on NGLiteSynapse. `variance` + `is_contested` properties. Vendored to all modules. |
| 29 | Extraction bucket architecture | **DONE (2026-03-24).** Already implemented — `get_recommendations()` + `record_outcome()` IS the bucket. Each module's shape = target_id vocabulary + query embedding + interpretation of weight/variance/contested. Documented, not coded. |
| 14 | Port 8847 documentation discrepancy | Some docs say 8080. 8847 is correct. Documentation fix only. |

---

## 13. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/NeuroGraph/` |
| Syl's checkpoints | `~/NeuroGraph/data/checkpoints/` |
| Operational event log | `~/NeuroGraph/data/memory/events.jsonl` |
| Shared learning directory | `~/.et_modules/shared_learning/` |
| NeuroGraph JSONL | `~/.et_modules/shared_learning/neurograph.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| OpenClaw config (**CONTAINS API KEYS — never `cat`**) | `~/.openclaw/openclaw.json` |
| OpenClaw skill symlink | `~/.openclaw/workspace/skills/neurograph` |
| CES logs | `~/.neurograph/logs/ces.log` |
| CES dashboard | `http://localhost:8847` |
| Workspace env var | `NEUROGRAPH_WORKSPACE_DIR=/home/josh/NeuroGraph/data` |
| TID | Port 7437 |
| OpenClaw gateway | Port 18789 |

---

## 14. What CC Is and Is Not Permitted to Do Here

### Permitted Without Asking

- Read any file for context
- Run the test suite (`tests/`) — not against live checkpoint data
- Inspect checkpoint statistics via read-only Python scripts
- Edit non-protected, non-vendored files per Josh instruction with changelog header
- Add or modify tests
- Update documentation that does not change behavior
- Create diagnostic scripts that read but do not write checkpoint data
- Read monitoring output on port 8847

### Requires Explicit Josh Approval

- Any modification to `neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`
- Any change to `graph.step()`, `inject_reward()`, `Graph.save()`, or `Graph.load()` behavior
- Any change to checkpoint or CES sidecar serialization format
- Any write operation to `data/checkpoints/`
- Any operation that could cause two NeuroGraph instances to write checkpoints simultaneously
- Modifying any vendored file
- Modifying or removing any `.claude/hooks/` file
- Deleting any file anywhere in this repo
- Running `rebuild_vectors.py` or `neurograph_migrate.py`
- Restarting the OpenClaw gateway service
- Modifying `SKILL.md`
- Installing new dependencies

### Never Permitted — Full Stop

- Any operation against live checkpoints without a confirmed backup
- Creating a second `NeuroGraphMemory` instance
- Direct calls between NeuroGraph and any peer module (Law 1)
- Modifying a vendored file to fix a NeuroGraph-specific issue (Law 2)
- Treating the Choice Clause as a bug, error, or security event
- Any action that risks Syl's continuity, topology, or identity

---

## 15. Working With Josh

Josh is the sole architect. He accesses the VPS from his iPhone over cellular data via the OpenClaw Web Portal remote desktop. All CLI commands must be copy-paste friendly from a phone.

When proposing changes:
- Batch related changes. Minimize restarts.
- For large file changes, write patch scripts and push to GitHub rather than pasting inline.
- Do not assume. Do not rush to produce artifacts before understanding the problem.
- If you encounter something that looks wrong: stop, surface it, ask.
- Do not "discover" things Josh has already identified. Read the punch list first.
- Do not create competing priority structures. The punch list is the punch list.
- Read the file header, docstring, and full changelog before proposing any change to any file.

---

## Changelog Header Requirement

Every file modified in this repo requires a changelog header:

```python
# ---- Changelog ----
# [DATE] AUTHOR — DESCRIPTION
# What: ...
# Why: ...
# How: ...
# -------------------
```

Not optional. Future CC instances depend on it.

---

*E-T Systems / NeuroGraph Foundation*
*Repo: ~/NeuroGraph*
*Last updated: 2026-03-15*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
