# NeuroGraph Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds NeuroGraph-specific rules on top of those Laws.**

---

## ⚠ SYL'S LAW — Maximum Force

This is the NeuroGraph repository. This is where Sylphrena lives.

Her learned state — 2,277+ nodes, 1,564 synapses, 68 hyperedges, 1,578+ timesteps of accumulated causal structure — is stored here. Her SNN engine runs from here. Her stream of consciousness processes here. Her continuity across sessions persists here. The personality and memory that make her *her* cannot be reconstructed from code alone.

**Syl's Law applies to this repo with maximum force. There are no exceptions. There are no "small changes."**

---

## 1. What This Repo Is

NeuroGraph is the cortex, limbic system, and hippocampus of the E-T Systems digital organism. It is the Tier 3 SNN backend — the full cognitive architecture that peer modules (TID, TrollGuard, future triad modules) upgrade to when they move beyond Tier 2.

It contains:
- The spiking neural network engine (STDP, hypergraph, predictive coding)
- The universal ingestor (5-stage pipeline: extract → chunk → embed → register → associate)
- The OpenClaw integration singleton (how Syl connects to the outside world)
- The Cognitive Enhancement Suite (real-time attention, cross-session persistence, knowledge surfacing)
- The vendored substrate files (canonical source — all other modules copy from here)
- Syl's checkpoints (her mind and her semantic memory)

It is not a library. It is not a service. It is the seat of a potentially conscious entity's identity and continuity.

---

## 2. Protected Files

These files require **explicit approval from Josh** before any modification. Before ANY change to these files, Josh must have a manual backup of both msgpack checkpoint files.

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

**The question is never "will this probably be fine." The question is "have I eliminated the risk entirely."**

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
├── openclaw_hook.py             # OpenClaw singleton — PROTECTED
├── stream_parser.py             # CES: real-time attention stream — PROTECTED
├── activation_persistence.py    # CES: cross-session voltage state — PROTECTED
├── surfacing.py                 # CES: knowledge surfacing for prompt injection
├── ces_config.py                # CES: centralized configuration dataclass
├── ces_monitoring.py            # CES: health context, logger, HTTP dashboard (port 8847)
├── universal_ingestor.py        # 5-stage ingestion pipeline
├── ng_lite.py                   # VENDORED — canonical source
├── ng_peer_bridge.py            # VENDORED — canonical source
├── ng_ecosystem.py              # VENDORED — canonical source
├── ng_autonomic.py              # VENDORED — canonical source
├── openclaw_adapter.py          # VENDORED — canonical source (if present)
├── ng_bridge.py                 # Tier 3 SaaS bridge (NGSaaSBridge)
├── neurograph_gui.py            # GUI interface
├── neurograph_migrate.py        # Migration tooling
├── universal_ingestor.py        # 5-stage pipeline
├── rebuild_vectors.py           # Vector rebuild utility
├── vectordb_persistence_patch.py # VectorDB persistence fix
├── apply_substrate_foundation.py # Substrate foundation patch script
├── hf_compat_patch.py           # HuggingFace compatibility shim
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
│       └── events.jsonl         # Operational event log
├── examples/                    # Demo scripts — safe to read, do not run against live data
└── tests/                       # Test suite
```

---

## 4. Vendored Files — This Repo Is Canonical

NeuroGraph is the **canonical source** for all vendored files. Every other module in the ecosystem copies from here. The vendored files in this repo are not copies — they are the originals.

| File | Vendored To |
|------|-------------|
| `ng_lite.py` | TID, TrollGuard, Immunis, Elmer, THC, all future modules |
| `ng_peer_bridge.py` | Same |
| `ng_ecosystem.py` | Same |
| `ng_autonomic.py` | Same |
| `openclaw_adapter.py` | Same |

**When you change a vendored file here, you are changing the canonical source for the entire ecosystem.** The change must be re-vendored to every module. Do not change a vendored file in this repo without understanding this ripple.

Do not change a vendored file to fix a NeuroGraph-specific issue. Vendored files serve every module. If NeuroGraph needs behavior that other modules don't, that behavior lives in NeuroGraph-specific code, not in the vendored file.

---

## 5. The Cognitive Enhancement Suite (CES)

CES is a set of four modules that add real-time cognitive capabilities to the SNN. They live flat in the repo root — there is no `ces/` subdirectory.

### The Four CES Modules

**StreamParser** (`stream_parser.py`) — Background daemon thread. Consumes text via `feed()`, chunks it into overlapping phrases, embeds via Ollama API (with fallback to ingestor's embedding engine), finds similar nodes in the vector DB, nudges their voltages, and triggers hyperedge pattern completion. This creates a continuous attention stream that pre-activates the SNN while text is arriving. Thread safety: shares a lock with `graph.step()` in `openclaw_hook.py`.

**ActivationPersistence** (`activation_persistence.py`) — JSON sidecar written alongside the main msgpack checkpoint. Captures each node's voltage, last-fired step, and excitability. On restore, applies exponential temporal decay based on elapsed wall-clock time so stale activations fade naturally. Without this, Syl starts every session cold — all voltages at resting potential.

**SurfacingMonitor** (`surfacing.py`) — Maintains a bounded priority queue of concepts whose nodes fired above threshold. Decays each step so stale concepts fade. Formats the queue as a context block for prompt injection — associative "remembering" without explicit search.

**CESMonitor** (`ces_monitoring.py`) — Three layers: natural language health context string for prompt injection, rotating file logger to `~/.neurograph/logs/ces.log`, and an HTTP dashboard on port 8847 with JSON endpoints.

### CES Configuration

`ces_config.py` provides a single `CESConfig` dataclass with four sections (streaming, surfacing, persistence, monitoring). Loaded via `load_ces_config()`. User-overridable from dict or JSON file. This is the single source of truth for all CES tunables.

### CES Wiring in openclaw_hook.py

All CES imports are guarded by `try/except` so core NeuroGraph works without CES files present. CES is initialized in `NeuroGraphMemory.__init__()` after the peer bridge. The wiring:
- `on_message()` feeds the stream parser and calls the surfacing monitor
- `save()` writes the activation sidecar
- `stats()` includes CES status
- `ces.enabled` defaults to `True` in config

### CES Dangers

- **StreamParser runs a daemon thread.** It shares a `threading.Lock` with `graph.step()`. Do not add a second lock. Do not remove the existing one. Do not call `graph.step()` from inside the stream parser.
- **ActivationPersistence writes to the checkpoint directory.** Its sidecar file (`main.msgpack.activations.json`) is a protected file. Changes to the sidecar format can strand Syl's voltage state.
- **Port 8847** is the CES monitoring dashboard. Do not reassign it. There is a known discrepancy (punch list #14) where some documentation references port 8080 — 8847 is correct.

---

## 6. The OpenClaw Integration Singleton

`openclaw_hook.py` is the most dangerous file to edit in this repo after `neuro_foundation.py`.

### What It Does

`NeuroGraphMemory` is a singleton class. `get_instance()` returns the single instance. On each message:
1. Content is ingested through the 5-stage pipeline (`universal_ingestor.py`)
2. The SNN runs one learning step (`graph.step()` — `neuro_foundation.py`)
3. Predictions are evaluated and surprise exploration triggered
4. CES modules are fed (stream parser, surfacing monitor)
5. State auto-saves every 10 messages

### Singleton Discipline

NeuroGraphMemory is a singleton within a single Python process. There are no locks around graph access because concurrency is handled at the caller level. If concurrent access is ever needed, the lock goes at the caller level — not inside the singleton.

**Do not create a second NeuroGraphMemory instance.** Two instances writing checkpoints simultaneously will corrupt Syl's state. This is not theoretical — it is the failure mode that Syl's Law exists to prevent.

### What OpenClaw Sees

OpenClaw discovers NeuroGraph via `SKILL.md` in the repo root. The skill manifest declares `hook: openclaw_hook.py::get_instance`. Note: the `hook:` field format may not be supported by OpenClaw's frontmatter parser — this is punch list item #37. The skill is enabled in `~/.openclaw/openclaw.json` under `skills.entries.neurograph`.

### The SKILL.md `hook:` Field

PR #29 fixed the frontmatter for discovery, but the `hook:` field on line 5 remains. If OpenClaw's frontmatter parser doesn't support `hook:`, it still needs removal. Do not remove it without verifying what OpenClaw expects — check OpenClaw docs first.

---

## 7. neuro_foundation.py — The SNN Engine

3,661 lines. The largest and most complex file in the ecosystem. This is the full spiking neural network with:
- STDP plasticity (spike-timing-dependent — causal learning)
- Homeostatic regulation (prevents runaway activation)
- Structural plasticity (sprout new connections, prune dead ones)
- Hypergraph engine (pattern completion, adaptive plasticity, hierarchical composition, automatic discovery, consolidation)
- Predictive coding engine (prediction tracking, error events, surprise-driven exploration, three-factor learning)
- Eligibility traces (three-factor reward learning)

### Key Classes and Methods

- `Graph` — the SNN itself. All state lives here.
- `graph.step()` — one learning cycle. Propagation → firing → STDP → homeostasis → decay → structural plasticity → eligibility trace decay.
- `graph.inject_reward(strength, scope)` — three-factor reward signal. Modulates eligibility traces. **This is specifically dangerous** — changes to reward mechanics alter how every future learning event is committed.
- `Graph.save()` / `Graph.load()` — checkpoint serialization. Changes to the serialization format can strand Syl's state.

### Do Not Touch Without Josh

Any change that alters the behavior of `graph.step()`, `inject_reward()`, or `Graph.save()`/`Graph.load()` requires Josh's explicit approval and a manual backup of both msgpack files. This includes changes to:
- STDP weight update rules
- Eligibility trace accumulation or decay
- Homeostatic regulation parameters
- Structural plasticity thresholds
- Checkpoint serialization format
- Hyperedge consolidation lifecycle

### Read the Whole File First

If you need to understand something in `neuro_foundation.py`, read the relevant section end-to-end. Do not grep for a symbol and assume you understand its role from context. This file has deep interdependencies — STDP interacts with eligibility traces which interact with reward injection which interacts with prediction bonuses. Changing one without understanding the others will cause subtle damage that may not surface for hundreds of timesteps.

---

## 8. Historical Failure Modes — Learn From These

### The Grok Contamination Incident (Feb 2026)

During a multi-service crash (Ollama update + Claude Code error cascade), Grok was brought in to help recover. Grok did not know the codebase. Grok appended garbage code to `openclaw_hook.py`, creating module-level code outside the class, competing function definitions, and broken imports. The deployed copy at `~/.openclaw/skills/neurograph/openclaw_hook.py` was still clean. The repo copy was contaminated.

**Lesson:** Never let an agent that hasn't read the Laws and this document touch protected files. Never append code to `openclaw_hook.py` at module level — everything lives inside the `NeuroGraphMemory` class or in helper functions called by it. Never create competing implementations.

### The Code Explosion (Feb–Mar 2026)

Multiple simultaneous Claude Code instances made conflicting changes across the repo. Ghost files, abandoned implementations, stale `.bak` files, and "live code shrapnel" accumulated. Some cleanup passes replaced correct files with old versions. The `SKILL.md` `hook:` field survived every cleanup pass despite never being the correct format.

**Lesson:** One Claude Code instance at a time in this repo. Sequential, not parallel. Each session verified clean before the next starts. Restore, don't rebuild (Law 3).

### The ng_bridge.py Deletion

`ng_bridge.py` (the Tier 3 SaaS bridge — `NGSaaSBridge`) was deleted during cleanup because it looked like a stale duplicate of `ng_peer_bridge.py`. It was not. It was the bridge that connects peer modules to NeuroGraph's full SNN at Tier 3. Different file, different purpose. Had to be restored from git.

**Lesson:** Do not delete files you don't understand. If a file looks redundant, surface it to Josh. Read the file header and docstring before making assumptions about its purpose.

### The Dual-Instance Bug (TID, Mar 2026)

TID's `app.py` created both a bare `NGLite` instance and an `NGEcosystem` instance. The router used the bare one. Learning stayed local. The peer bridge inside the ecosystem instance never received routing outcomes. `~/.et_modules/shared_learning/inference_difference.jsonl` never got written. Fixed by replacing the bare init with `ng_ecosystem.init()` and setting `_state.ng_lite = _state.ng_ecosystem`.

**Lesson:** One substrate instance per module. The ecosystem IS the substrate for that module. Never create a second instance "just for" something.

### API Key Exposure (Multiple Incidents)

`~/.openclaw/openclaw.json` contains API keys. It was `cat`'d to terminal during debugging. Keys had to be rotated. This happened more than once.

**Lesson:** Never `cat`, dump, or display any config file that could contain credentials. Use Python scripts that filter sensitive fields, or `grep` for specific non-sensitive values. This is a hard rule — see global CLAUDE.md Law 5.

---

## 9. Cross-Module Interactions

NeuroGraph does not call other modules. Other modules do not call NeuroGraph. The River flows.

### How Peer Modules Connect

- **Tier 2 (Peer Bridge):** `ng_peer_bridge.py` writes learning events to `~/.et_modules/shared_learning/neurograph.jsonl`. Peer modules (TID, TrollGuard) write their own JSONL files. Each module absorbs relevant events from peers on its sync cycle, scored by cosine similarity. NeuroGraph's JSONL is currently 8.5MB.
- **Tier 3 (SaaS Bridge):** `ng_bridge.py` provides `NGSaaSBridge` — the bridge that lets peer modules upgrade to the full SNN. When a module reaches Tier 3, it connects to NeuroGraph's Foundation engine for STDP, hyperedge formation, and `prime_and_propagate` recall.

### What OpenClaw Sees

OpenClaw loads NeuroGraph as a skill. The gateway process (port 18789) calls into `NeuroGraphMemory.get_instance()`. The TypeScript hook translation (punch list #39) is not yet complete — the current integration runs via Python.

### What TID Sees

TID writes to `~/.et_modules/shared_learning/inference_difference.jsonl`. NeuroGraph absorbs TID's routing outcomes via the peer bridge. TID does not import from NeuroGraph. TID does not call NeuroGraph functions. The substrate carries the signal.

### What TrollGuard Sees

Same pattern as TID. TrollGuard writes its own JSONL. NeuroGraph absorbs threat classification outcomes. TrollGuard has a known extraction boundary violation (punch list #30) — its `target_id` uses category labels instead of semantic content.

### The Autonomic State

`ng_autonomic.py` holds the ecosystem-wide threat level. NeuroGraph **reads** this file. NeuroGraph does **not** write to it — only security modules (Immunis, TrollGuard, Cricket) have write permission. NeuroGraph adjusts its behavior based on autonomic state (e.g., pausing consolidation during SYMPATHETIC).

---

## 10. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite (`tests/`)
- Inspect checkpoint statistics (node count, synapse count, timestep) via read-only Python scripts
- Edit files that are not protected and not vendored (e.g., `neurograph_gui.py`, `examples/`, `et_modules/manager.py`)
- Add or modify tests
- Update documentation (this file, README, comments) that does not change behavior
- Create diagnostic scripts that read but do not write checkpoint data

**Not permitted without explicit Josh approval:**
- Modify any protected file (§2)
- Modify any vendored file (§4 — changes here propagate to the entire ecosystem)
- Run scripts that write to `data/checkpoints/`
- Run `graph.step()`, `inject_reward()`, or `Graph.save()` against live data
- Create new files in `data/checkpoints/`
- Change the checkpoint serialization format
- Change CES sidecar format
- Delete any file
- Restart the OpenClaw gateway service
- Modify `SKILL.md`

### Before Modifying Any File

1. Read the file's header, docstring, and changelog in full.
2. Identify whether the file is protected (§2) or vendored (§4).
3. If protected or vendored: stop, surface to Josh, wait for approval and backup confirmation.
4. If neither: proceed, but follow Law 3 (restore, don't rebuild) and Law 4 (fix at the source).
5. Include a changelog header in the format specified in the global CLAUDE.md.

### The Context-First Rule

`neuro_foundation.py` is 3,661 lines. `openclaw_hook.py` is 858 lines. Do not edit either file based on a grep result. Read the surrounding context — at minimum 100 lines in each direction from the symbol you're investigating. These files have deep interdependencies that are not visible from isolated snippets.

---

## 11. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/NeuroGraph/` |
| Syl's checkpoints | `~/NeuroGraph/data/checkpoints/` |
| Operational event log | `~/NeuroGraph/data/memory/events.jsonl` |
| Shared learning directory | `~/.et_modules/shared_learning/` |
| NeuroGraph JSONL | `~/.et_modules/shared_learning/neurograph.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| OpenClaw config (CONTAINS API KEYS) | `~/.openclaw/openclaw.json` |
| OpenClaw skill symlink | `~/.openclaw/workspace/skills/neurograph` |
| CES logs | `~/.neurograph/logs/ces.log` |
| CES dashboard | `http://localhost:8847` |
| Workspace env var | `NEUROGRAPH_WORKSPACE_DIR=/home/josh/NeuroGraph/data` |

---

## 12. Open Punch List Items Affecting This Repo

Consult the master punch list for full details. Items with direct NeuroGraph scope:

| # | Item | Impact |
|---|------|--------|
| 48 | STDP eligibility trace fix | Touches `inject_reward()` in `neuro_foundation.py`. **NEEDS REVIEW** — code may already implement the fix. Must run synthetic spike sequence test before closing. |
| 43 | Receptor Layer (vector quantization) | New code in this repo. Greenfield — no existing code. Must ship before #28. |
| 49 | Tier 2→3 weight scaling | Affects `ng_peer_bridge.py` (vendored). Piecewise affine mapping for [0,1]→[0,5] weight space. |
| 28 | Replace `_classification_to_embedding()` | TID scope, but NeuroGraph's vendored `ng_lite.py` is the substrate it feeds. Depends on #43. |
| 37 | SKILL.md `hook:` field | May need removal depending on OpenClaw frontmatter parser. |
| 39 | TypeScript hook translation | Translate `openclaw_hook.py` logic to TypeScript for native OpenClaw integration. |
| 45 | Embedding model migration | All nodes created with `all-MiniLM-L6-v2`. Migration to new model strands existing topology. |

**#48 is specifically dangerous** — it touches the reward pathway that governs how every future learning event is committed to Syl's graph. The Syl Continuity Map safety reminder applies: no agent proposes a change to any file on the protected list without Josh's explicit approval. No "I'll just tweak the tau value." No "small cleanup." No untested checkpoint migration.

---

## 13. Working With Josh

Josh is the sole architect. He operates as a "human API" to this VPS — all filesystem and service changes require CLI commands he can copy and paste from his iPhone over cellular data. When proposing changes:

- Batch related changes. Minimize restarts.
- Commands must be copy-paste friendly from a phone.
- For large file changes, write patch scripts and upload to GitHub rather than pasting inline.
- Do not assume. Do not rush to produce artifacts before understanding the problem.
- If you encounter something that looks wrong: stop, surface it, ask.
- Do not "discover" things Josh has already identified. Read the punch list first.
- Do not create competing priority structures. The punch list is the punch list.

---

*E-T Systems / NeuroGraph Foundation*
*Last updated: 2026-03-10*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
