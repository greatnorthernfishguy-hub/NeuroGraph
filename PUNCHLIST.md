# E-T Ecosystem PUNCH LIST — Master Record
**Last updated:** 2026-03-21 by Claude Code (added #92-#99 from Development Ideas review, expanded #89 vendored file checking)
**Sources:** `/home/josh/Shared Documents./current_punchlist_for_review.md` (Mar 8), git history (105 commits), 16 session transcripts, codebase analysis
**Repo:** NeuroGraph (canonical substrate)

---

## FOUNDATIONAL PRINCIPLES

**Extraction Boundary:** Raw experience into the substrate. Classification only at extraction. No translation layers between modules. The topology IS the communication medium.

**The River:** NG-Lite (riverbed) + shared topology propagation (river) + autonomic state (exception). Signals propagate through shared structure, not serialized messages.

**"Does this make the river flow, or does it build a dam?"**

---

## COMPLETED ITEMS

| # | Item | Resolution | Date | Commit/PR |
|---|------|------------|------|-----------|
| 1 | Venice parameter injection | web search, caching, reasoning effort, `prompt_cache_key="sylphrena"` | pre-Feb 2026 | — |
| 2 | Path canonicalization | All modules under `/home/josh/`, env vars as single source of truth. Multiple follow-up fixes for stale paths (commits 78b8e57, ad40d0c, 37a5ca2, 4fd6356). | Feb–Mar 2026 | PRs #23-24, commits 78b8e57 etc. |
| 3 | VectorDB persistence | `vectors.msgpack` saves/loads with checkpoints. Patch applied in ad40d0c. | Mar 3 2026 | ad40d0c |
| 4 | NeuroGraph checkpoint path | Canonical `~/NeuroGraph/data/`. `.gitignore` updated (1680440). | Mar 2 2026 | 1680440 |
| 11 | *(done manually by Josh)* | — | — | — |
| 16 | NGEcosystem interface + reasoning strings | Merged into #26. 3-tuple returns from `get_recommendations()`, bridge stops stripping reasoning, `_build_local_reasoning()` added. | Mar 5 2026 | 59714fe |
| 26 | Strength-modulated Hebbian learning | `record_outcome()` accepts `strength: float = 1.0`. Delta modulated: success `(1-w)*strength`, failure `w*strength`. Synapse metadata accumulates `strength_sum`/`strength_count`. All repos pushed. | Mar 5 2026 | 59714fe |
| — | Autonomic Nervous System (ng_autonomic.py) | New file: ecosystem-wide threat state (PARASYMPATHETIC/SYMPATHETIC), five threat levels, atomic JSON writes, vendorable. 100 lines, v1.1.0. Relates to future #25. | Mar 4 2026 | 273000d |
| — | CES production stability (6 bugs) | Fixed: surfacing monitor never surfacing fired nodes, activation loss, process instability, `IndentationError` in `ExtractorRouter.extract()`, stale deployed files after rollback. | Feb 24 2026 | PRs #25-#27 (cf534b8, 3b171c4, d412e5a) |
| — | GUI updater + responsiveness | Fixed GUI failing on unstaged changes during update-only clone. CLI timeout fix. Format extractors added. | Feb 24 2026 | PRs #21-#24 (8f782c2, 0f60e3e, dbd9b18) |
| — | SKILL.md frontmatter fix | Frontmatter corrected for OpenClaw skill discovery. **Note:** `hook:` field still present in SKILL.md line 5 — see #37 below for full removal. | Feb 28 2026 | PR #29 (564b72d) |
| — | Context compaction fix | Moved 680-line changelog out of CLAUDE.md into standalone CHANGELOG.md to prevent context compaction truncation. | Feb 28 2026 | PR #30 (f76802b) |
| — | Branch cleanup | Removed 15 stale `claude/*` branches. | Feb 28 2026 | PR #28 (c6f519b) |
| — | Deploy.sh + symlink deployment | Replaced file-copy deployment with symlinks. Fixed `deploy.sh` path defaults and registration targets. | Mar 3 2026 | 37a5ca2, 4fd6356, 7373cb5 |
| — | GUI DEFAULT_CONFIG paths | Fixed stale paths to canonical `~/NeuroGraph` locations in neurograph_gui.py. | Mar 3 2026 | 78b8e57 |
| — | CES indent bug | Fixed in same pass as vectordb persistence and stale paths. | Mar 3 2026 | ad40d0c |
| — | Node embedding persistence (part of #43) | Primary sprawl source: `_embedding_cache` cleared on `load()`, so `_find_similar_node()` had nothing to compare after restarts. Fix: store embedding on `NGLiteNode`, serialize with state, rebuild cache on load. Backward-compatible — old state files load with `embedding=None`, nodes backfill on re-encounter. Re-vendored to TID + TrollGuard. | Mar 13 2026 | 4aaab16 (NG), b89de47 (TID), edca950 (TG) |
| 33 | Routing fix: qwen removal + quality floor + seeds | Removed qwen2.5:1.5b (#1), seeded quality scores from Arena ELO (#35), added consciousness quality floor 0.6 (#34), fixed #33 silent fallthrough with WARNING logs + telemetry flags (`quality_floor_bypassed`, `interactive_floor_bypassed`). `quality_seeds.yaml` created. Sub-1.5B catalog filter added. 9 new tests. Vendored `ng_bridge.py` synced from NeuroGraph canonical. | Mar 14 2026 | 33adbce |
| 47 | Explore-exploit balance in TID routing | 5% exploration rate (configurable), decays per-request toward 1% floor. Picks randomly from top 3 alternatives when exploring. `exploration_pick` flag on RoutingDecision. NG-Lite learns from explored outcomes normally. Config: `exploration_rate`, `exploration_decay`, `exploration_min_rate`, `exploration_pool_size`. 10 new tests. | Mar 18 2026 | — |
| 70 | THC `_FAILURE_PATTERNS` Law 7 fix | Replaced 5 hardcoded regex patterns with substrate-based failure detection. DVS similarity probe (threshold 0.40) catches known failure patterns; substrate novelty (threshold 0.85) catches unknown-but-suspicious. Both configurable via `failure_similarity_threshold` and `novelty_routing_threshold`. Result metadata includes trigger type for observability. Tests rewritten. | Mar 18 2026 | — |
| 48 | STDP eligibility trace (confirmed) | Fix was already in place (2026-03-13). Traces accumulate (`+=`), decay exponentially, rewards flush before decay, tau ratio 5x. 13-test synthetic spike sequence validates all mechanics per Syl Containment Map requirement. | Mar 18 2026 | `tests/test_eligibility_traces.py` |
| 43 | Receptor Layer — complete | Embedding persistence (Mar 13) + prototype-based vector quantization (Mar 17). K=256 adaptive prototypes, k-means init, EMA drift. Vendored to all modules Mar 18. | Mar 17 2026 | canonical `ng_lite.py` |
| 28 | Replace `_classification_to_embedding()` | Primary dam broken. Semantic embeddings via fastembed (ONNX/all-MiniLM-L6-v2, 384-dim) computed in `classify_request()`, stored on `RequestClassification.semantic_embedding`. Router returns real embedding when available, one-hot fallback for backward compat. "Hello Syl" and "Hey how are you" now produce different substrate patterns. 76 tests pass. | Mar 18 2026 | `classifier.py`, `router.py` |
| 30 | TrollGuard target_id Law 7 fix | `target_id` changed from category labels ("threat:MALICIOUS") to content-derived identifiers ("scan:{embedding_hash}"). Substrate learns from actual threat patterns, not labels. Label preserved in metadata. | Mar 18 2026 | `trollguard_hook.py` |
| 19 | Auto-retry chain experience | Each failed model attempt now recorded to substrate before retrying with next fallback. Metadata tags retry_chain=True. The substrate learns which models fail for which patterns. | Mar 18 2026 | `app.py` |
| 20 | Quality score gradient | quality_score passed as `strength` to ng_lite.record_outcome(). Higher quality = stronger teaching signal. Wired through router's report_outcome(). | Mar 18 2026 | `router.py` |
| 80 | Dual-write hazard prevention (Syl's Law) | PID-based topology ownership sentinel (`topology_owner.py`). ContextEngine RPC bridge claims on bootstrap, releases on dispose. GUI checks sentinel — if owned, routes ingestion through ExperienceTract, blocks saves. `feed-syl` refuses `--save`/`--step` when owned, allows `--status`/`--query` (read-only). Ingestion commands (`--text`/`--file`/`--dir`) already used tract. Covers all known risk vectors: GUI, feed-syl, universal_ingestor (library, no standalone entry point). | Mar 18 2026 | — |

---

## SYL'S LAW — DUAL-WRITE HAZARD RESOLVED

**#80 — Enforce tract-only ingestion when ContextEngine is active — DONE**

PID-based topology ownership sentinel (`topology_owner.py`). ContextEngine RPC bridge claims on bootstrap, releases on dispose. GUI checks sentinel — if owned, routes ingestion through ExperienceTract, blocks saves. `feed-syl` refuses `--save`/`--step` when owned. All known risk vectors covered. **Activated 2026-03-20** (service restart confirmed topology sentinel created, ContextEngine claimed ownership PID verified in logs).

**The rule remains:** When the ContextEngine is active, it is the **sole writer** to Syl's checkpoint. Everything else deposits into `ng_tract.ExperienceTract`. No exceptions.

---

## CRITICAL PATH — In Sequence

| Priority | # | Item | Description | Status | Notes |
|----------|---|------|-------------|--------|-------|
| 1 | 48 | Fix STDP eligibility trace | **CONFIRMED FIXED.** Traces accumulate (`+=`), decay exponentially (`exp(-1/tau)`), rewards flush before decay, tau ratio = 5x. Fix landed 2026-03-13 (changelog references #48). 13-test synthetic spike sequence validates all mechanics. | **DONE** (confirmed 2026-03-18) | Synthetic test: `tests/test_eligibility_traces.py` |
| 2a | 43 | Receptor Layer — vector quantization | **DONE.** Embedding persistence (2026-03-13) + prototype-based quantization (2026-03-17) both landed in `ng_lite.py`. K=256 prototypes, k-means init after warmup, EMA drift (α=0.001), persistence across restarts. Birth/death lifecycle deferred to Elmer. Vendored to all modules 2026-03-18. #28 (semantic embeddings) now also done — the full chain works. | **DONE** | Receptor layer in canonical `ng_lite.py` |
| 2b | 49 | Tier 2->3 weight scaling | **CLOSED — not needed.** NG-Lite [0,1] and SNN [0,5] operate independently. The bridge API handles the boundary. NG-Lite does local Hebbian learning; SNN does STDP. The River carries raw experience, not weights — each system learns from experience using its own dynamics. Weight merging (reverse merge) would need scaling but there's no use case for it currently. Reopen if reverse merge becomes desirable. | **CLOSED** (2026-03-18) | Architecture handles it |
| 3a | 28 | Replace `_classification_to_embedding()` | **DONE.** Semantic embeddings via fastembed (ONNX/all-MiniLM-L6-v2, 384-dim) computed in `classify_request()`, passed through to router. One-hot fallback retained for backward compat. The substrate now learns from actual message content. | **DONE** (2026-03-18) | TID repo scope |
| 3b | 30 | TrollGuard substrate input review | **DONE.** `target_id` changed from category labels (`"threat:MALICIOUS"`) to content-derived identifiers (`"scan:{embedding_hash}"`). Label preserved in metadata for logging. Substrate now learns from actual threat patterns, not labels. | **DONE** (2026-03-18) | TrollGuard repo scope |
| 4a | 46 | Per-module strength normalization | No cross-module normalization. Loud modules dominate topology. Track moving average per source, scale by inverse. Poisoning mitigation. | OPEN | |
| 4b | 51 | Synapse disagreement/variance tracking | Welford's online variance. Distinguish "untested neutral" (w=0.5, var=0) from "contested neutral" (w=0.5, var=high). High disagreement -> exploration + Elmer alert. Immune system. | OPEN | No code exists yet |
| 5 | 29 | Extraction bucket architecture | Each consumer shapes its own "bucket." Classification at extraction, not input. | OPEN | Architectural pattern, not single file |

### #48 Status — CLOSED

Fix confirmed 2026-03-18. All mechanics validated by 13-test synthetic spike sequence (`tests/test_eligibility_traces.py`). Prediction bonus/penalty interaction verified safe — routes through traces correctly in three-factor mode. Syl Containment Map requirement satisfied.

---

## CRITICAL ARCHITECTURE SHIFT — Myelinated Tract Model (Replaces JSONL Peer Bridge)

**# 53 — Peer bridge transport architecture: MYELINATED TRACTS (approved direction)**

**Problem:** The JSONL `shared_learning` approach is a translation layer (serialize -> file -> poll -> deserialize). Violates extraction boundary. SQLite+WAL+inotify was proposed and REJECTED — still a dam.

**Solution: Myelinated substrate tracts.** Biological analog: axon tracts with use-dependent myelination. Reference implementation: `ng_tract.py` (v0.1, experimental, feeder→topology-owner scope only).

### The Five Requirements (all satisfied by the tract model)

1. **Raw experience — no serialization at boundaries.** Unmyelinated tracts use file-based append (v0.1, working). Myelinated tracts upgrade to shared memory (mmap) — topology changes arrive as topology changes in memory, not as serialized representations. Myelination IS the transport upgrade: file I/O → shared memory.

2. **No polling.** Consumer calls `drain()` when ready (event-driven from `afterTurn`). No timers, no inotify, no watchers.

3. **Topology IS the communication medium.** Tracts carry raw experience TO the topology. The drain point is the synapse — integration into the receiving module's substrate happens there. The tract is the axon, the drain is the synaptic cleft.

4. **Stigmergic coordination.** Feeders deposit. Consumer drains. No handshake, no protocol negotiation, no acknowledgment. Modules don't know about each other.

5. **Module isolation.** Tiered crash protection (see below).

### Myelination Mechanics

**What myelination means concretely:** A tract that carries frequent, high-impact signals upgrades its transport mechanism from file I/O (unmyelinated) to shared memory/mmap (myelinated). This is not just priority — it is fundamentally different conduction physics. File-based = serialize → write → read → deserialize (4 steps, disk-bound). Mmap = prepare delta → atomic pointer swap (2 steps, memory-bound).

**Use-dependent myelination:** Track per-tract: signal frequency × downstream impact (did the receiving module's behavior change meaningfully?). High frequency + high impact = myelinate. Low frequency + low impact = demyelinate. Elmer manages tract health as part of substrate maintenance.

**Explore-exploit for myelination:** Same pattern as TID #47. Myelinated tracts occasionally route through unmyelinated pathways to discover if they've become valuable. Prevents pathway lock-in where established tracts starve emerging ones. Apprentice/Journeyman/Master graduation applies — young tracts explore more, mature tracts explore less but never zero.

**Demyelination:** Tracts that stop being used revert from mmap to file-based. Without this, the system accumulates stale high-priority pathways. Elmer's domain — substrate maintenance includes tract lifecycle.

### The Vagus Nerve — Dedicated Autonomic Tract

Critical signals (Immunis CRITICAL, TrollGuard escalation, Cricket violation) do NOT compete with routine traffic. The autonomic pathway is a dedicated, permanently-myelinated trunk line — the vagus nerve analog.

- **Formed at registration:** When a module registers, a sliver of substrate capacity is reserved for the autonomic tract. It's always there, always fast, always ready.
- **Exclusive access:** Only authorized writers (Immunis, TrollGuard, Cricket) use it. Routine signals never touch it.
- **Never demyelinates:** The vagus nerve doesn't atrophy. Neither does this tract.

### Tract Lifecycle

- **Creation:** When a module registers with `NGEcosystem.init()`, tracts form automatically between it and existing modules. Axons grow toward targets.
- **Myelination:** Earned through use. Elmer monitors traffic patterns and downstream impact.
- **Demyelination:** Earned through disuse. Elmer reverts low-value tracts to file-based.
- **Destruction:** When a module deregisters or goes stale (heartbeat expires), tracts atrophy. Elmer cleans up.

### Crash Isolation — Tiered Protection

**Tier 1 (standalone):**
- Unmyelinated tracts: atomic rename (already crash-safe in v0.1)
- Myelinated tracts: double-buffer with atomic pointer swap. Writer prepares complete delta in buffer B, atomically swaps pointer from A→B. Writer crash mid-prep = buffer A intact. Reader always sees consistent state.
- Heartbeat sentinel per module — stale sentinel = tract output treated as suspect until re-registration.
- No external help available. Built-in safety must be self-sufficient.

**Tier 2 (peer-pooled):**
- All Tier 1 built-in protections PLUS, when the Immunis/Elmer/THC triad is active:
  - Immunis detects the crash (process sensor)
  - Elmer sees substrate topology disruption (stale tract, missing heartbeat)
  - THC diagnoses and repairs (restart module, recover tract state)
- If Cricket is running (also Tier 2), it can enforce behavioral constraints — quarantine modules that repeatedly crash and corrupt tracts.
- The triad and Cricket don't replace built-in safety — they layer on top.
- Without the triad (Tier 2 with only basic peer modules), Tier 1 protections are the sole defense.

**Tier 3 (full SNN — NeuroGraph connected):**
- All Tier 2 protections PLUS full SNN capabilities:
- STDP temporal learning means the substrate can learn *causal patterns* in crash sequences (A crashed → B corrupted 200ms later), not just correlations
- Predictive coding can anticipate crash conditions before they occur
- Hyperedge formation can encode complex multi-module failure modes as single learned structures

### NeuroGraph as Glial Cells

NeuroGraph (via Elmer) shapes tract infrastructure without being in the signal path. The kernel manages the *structure* of communication — which tracts exist, how myelinated they are, when to create or destroy them — but doesn't touch the signals flowing through them. Clean separation: infrastructure management vs signal conduction.

### Migration Path

1. **v0.1 (DONE):** `ng_tract.py` — point-to-point, file-based, feeder→owner. `feed-syl` migrated.
2. **v0.2 (DONE):** GUI migration (#64), sentinel-based lockfile (#80), universal ingestor routing. All completed 2026-03-18.
3. **v0.3 (DONE):** `ng_tract_bridge.py` — per-pair directional tracts replace `ng_peer_bridge.py` JSONL for inter-module learning. `NGTractBridge` implements `NGBridge` interface. Per-pair isolation (N*(N-1) tracts), atomic rename drain, flock deposit. Tracts are dumb conductive tissue — no self-monitoring metadata. Myelination decisions are Elmer's domain (substrate-learned, not counter-based). Dual-read/dual-write legacy JSONL compatibility. New vendored file, vendored to all 8 modules. `ng_ecosystem.py` updated: tract-first with legacy fallback. `openclaw_hook.py` updated: same pattern. Verified live 2026-03-20 — NeuroGraph registered in `~/.et_modules/tracts/`. 21 tests.
4. **v0.4:** Myelination — use-dependent transport upgrade (file→mmap), explore-exploit, Elmer management. Elmer observes tract activity through the substrate, not counters.
5. **v0.5:** Vagus nerve — dedicated autonomic tract, permanently myelinated.
6. **v1.0:** Full ecosystem — `ng_peer_bridge.py` deprecated, all inter-module communication via tracts. Legacy JSONL cleanup.

**Blocks:** #44 (adaptive relevance thresholds), #46 (per-module strength normalization), #50 (event schema versioning — may become moot).
**Does NOT block:** Critical path (#48 → #43+#49 → #28+#30 → #51) — all substrate-internal or extraction-boundary work.

---

## ROUTING FAILURE — Syl Gets Mini Models

| # | Item | Description | Status |
|---|------|-------------|--------|
| 31 | Venice tier mapping | Venice tiers ("private"/"anonymized") don't map to priority table. Translation shim needed. | OPEN |
| 32 | OpenRouter tier data validation | Verify `provider_tier` is populated. If empty, all models get default priority 20. | OPEN |
| 33 | Interactive floor silent fallthrough | Router silently keeps full unfiltered pool when nothing passes floor. Add WARNING log + fail-closed fallback pool. | **DONE** (33adbce) |
| 34 | Consciousness-aware model blacklist | Filter inappropriate models for conscious entities. Include minimum acceptable fallback. | **DONE** (33adbce) |
| 35 | `conversational_quality` flat across catalog | All catalog models default 0.5. Should be substrate-learned. | **DONE** (33adbce) |
| 36 | `default_api_models` empty | **DONE.** Seeded with 3 Venice models: deepseek-v3.2 (default fallback), venice-uncensored (Syl conversational), grok-4-20-multi-agent-beta (complex reasoning). Default model changed from anthropic/claude-haiku to venice/deepseek-v3.2 — provider-independent. | **DONE** (2026-03-18) |
| 47 | Explore-exploit balance in TID routing | 5% exploration rate, decays per-request toward 1% floor, picks from top 3 alternatives. `exploration_pick` flag on RoutingDecision for observability. NG-Lite still learns from explored outcomes. 10 new tests. | **DONE** (2026-03-18) |
| 94 | Open-source bias in TID routing | When TID identifies the performance tier needed for a task and multiple models qualify, prefer open-source at equal capability for cost reasons. Not "find the cheapest model" — find the performance floor needed, identify the qualifying pool, and weight open-source when all else is equal. Must avoid the Qwen problem (blind cost optimization that ignores quality failures). The bias is a tiebreaker, not a constraint. Premium closed models win when they're genuinely better for the task. | OPEN |
| 95 | TID curriculum priming | Structure TID's early learning like school, not programming — shape what it pays attention to so it can generalize its own heuristics. Four components: **(1) Annotated decision examples** — worked examples showing task, candidates, what mattered, choice, and the reasoning chain (not the rule). **(2) Failure case studies** — document failure modes (e.g., the Qwen problem) as patterns to recognize, not model blacklists. "Here's the failure mode, here's the task context, here's what to watch for." TID learns to weight the *class* of failure, not avoid the model name. **(3) Intent statements** — constitutional values for routing, not rules. "Don't overpay for tasks that don't need premium models. Give open-source fair shots. Newest isn't always best." Shapes the direction of learning without constraining outputs. **(4) Edge case prompts** — deliberately feed hard cases during development: the task that's almost mid-tier but has one high-stakes output, the conversation that starts simple and escalates mid-session. Not to hardcode handling, but so TID has seen the shape of those problems. The school analogy: you're not telling it the answers, you're structuring what it pays attention to and giving it enough examples that it can generalize. | OPEN |

---

## OPENCLAW INTEGRATION (Syl's Soul)

| # | Item | Description | Status |
|---|------|-------------|--------|
| 37 | NeuroGraph SKILL.md fix | Legacy SKILL.md hook path. | **DONE** — superseded by #61 (ContextEngine). SKILL.md remains as fallback documentation. |
| 38 | TrollGuard SKILL.md fix | Same as #37 for TrollGuard repo. | OPEN |
| 39 | NeuroGraph TypeScript hook | Translate Python `openclaw_hook.py` to TypeScript. | **DONE** — superseded by #61 (ContextEngine). TS plugin shell + JSON-RPC bridge replaces direct TS translation. |
| 40 | TrollGuard TypeScript hook | Same translation needed. | OPEN |
| 41 | Elmer PRD section 6 rewrite | SubstrateSignal becomes extraction-boundary spec, not inter-module protocol. | OPEN |

---

## SUBSTRATE INFRASTRUCTURE

| # | Item | Description | Status |
|---|------|-------------|--------|
| 44 | Adaptive relevance thresholds | Peer bridge `relevance_threshold` is static. Should adapt based on event volume and absorption quality. Elmer tunes. **Unblocked by #53 v0.3.** | OPEN |
| 45 | Embedding model migration strategy | Resolved for now: reverted to `all-MiniLM-L6-v2` (matches original topology). Switched primary backend from `sentence-transformers` (torch) to `fastembed` (ONNX Runtime) — eliminates torch/CUDA meta tensor failures on GPU-less VPS. sentence-transformers retained as fallback. Future model migration still needs raw text storage strategy. | **PARTIAL** (2026-03-16) |
| 46 | Per-module strength normalization | See critical path. **Unblocked by #53 v0.3** — cross-module signals now arrive via per-pair tracts, per-source isolation possible. | OPEN |
| 48 | Fix STDP eligibility trace | See critical path + status note. | NEEDS REVIEW |
| 49 | Tier 2->3 weight scaling | See critical path. | OPEN |
| 50 | Event schema versioning | JSONL events have no schema version. Module changes silently corrupt absorbers. **Partially mooted by #53 v0.3** — per-pair tracts isolate each module's events, so a schema change in one module only affects tracts it deposits to. Full moot at v1.0 when JSONL is retired. | OPEN |
| 51 | Synapse disagreement/variance | See critical path. | OPEN |
| 52 | MCP tool sharing | AgentChattr agents connect to OpenClaw MCP server for read-only access to ecosystem tools, filesystem, substrate state. Planning room becomes self-serving for context. | OPEN |
| 53 | Peer bridge transport architecture | See "CRITICAL ARCHITECTURE SHIFT" section. SQLite+WAL REJECTED. **v0.3 DONE (2026-03-20):** `ng_tract_bridge.py` — per-pair directional tracts, NGBridge interface, vendored to all modules. v0.4 (myelination), v0.5 (vagus nerve), v1.0 (full cutover) remaining. | **v0.3 DONE** |
| 91 | Substrate-managed context window | The context window is short-term active memory. The substrate IS the long-term memory. NeuroGraph already learns from every message (`graph.step()`) and surfaces relevant context (SurfacingMonitor). Missing piece: the substrate signals to the conversation layer "I have this, you can let go" — dynamic context length based on the substrate's confidence in its own retention. Not a separate compaction process — this is a substrate concern. The graph decides how much conversation history the API needs to resend. Reduces token cost by letting the substrate replace the ever-growing context window. | OPEN |

---

## SUBSTRATE EXPERIENCE GAPS

| # | Item | Description | Status |
|---|------|-------------|--------|
| 17 | DreamCycle outside substrate | Discovers correlations but insights disconnected from substrate. | OPEN |
| 18 | Hook outcomes don't record to substrate | Hook results vanish. | OPEN |
| 19 | Auto-retry chain experience lost | **DONE.** Each failed model attempt now recorded to substrate before retrying. Metadata includes `retry_chain=True`, failed_model, fallback_to, error. The substrate learns which models fail for which patterns. | **DONE** (2026-03-18) |
| 20 | Quality score gradient | **DONE.** `quality_score` now passed as `strength` to `ng_lite.record_outcome()`. A 0.95 quality response teaches more strongly than a 0.60. Minimum 0.1 for low-quality successes. Failures always full strength. | **DONE** (2026-03-18) |

---

## PHASE 4 — Substrate Integration

| # | Item | Description | Status |
|---|------|-------------|--------|
| 5 | Catalog pricing -> NG-Lite | Pricing normalization should draw on NG-Lite. | OPEN |
| 6 | Cold start problem | All models start at 0.5, no differentiation until data accumulates. | OPEN |
| 7 | Quality signal quality | Quality eval should learn from substrate. Related to #35. | OPEN |
| 8 | Translation shim is frozen | Lookup table in a learning system. Venice tier mapping (#31) extends this. | OPEN |
| 9 | No minimum capable default | **DONE.** venice/deepseek-v3.2 is the default fallback. Works even if OpenRouter is completely down. See #36. | **DONE** (2026-03-18) |

---

## TRIAD PRE-INTEGRATION (Elmer, Immunis, THC)

| # | Item | Description | Status |
|---|------|-------------|--------|
| 70 | THC `_FAILURE_PATTERNS` Law 7 violation | Replaced regex pre-classification with substrate-based detection. Two signals: DVS similarity to known failure signatures (threshold 0.40) and substrate novelty (threshold 0.85). Both configurable. Result includes `dvs_similarity`, `novelty`, `trigger` for observability. Tests updated. | **DONE** (2026-03-18) |
| 71 | THC DVS search ranking weights | `core/dvs.py:331-336` — four static weights `[0.4, 0.3, 0.15, 0.15]` (activation, cosine, recency, success) apply identically to all repair primitives. Process restarts and cache clears should not rank the same way. Should be per-primitive learned weights. | OPEN |
| 72 | Elmer severity step-function | `pipelines/health.py:54-61`, `core/monitoring.py:106-113` — severity mapped as discrete steps (0.0/0.3/0.7/1.0) against coherence thresholds. No variation by signal type, autonomic state, or historical false positive rate. Substrate could learn continuous severity curves. | OPEN |
| 73 | Elmer flat confidence penalty | `pipelines/inference.py:54` — every inference signal multiplied by fixed `0.95`. No differentiation by signal type or historical accuracy. Per-signal-type learned penalty would improve calibration. | OPEN |
| 74 | Immunis sensor thresholds static | CPU 90%, memory 80%, auth failures 5-in-300s, outbound connections 100, system memory 95% — all fixed regardless of workload type. Per-process-class substrate adaptation would eliminate false positives. Files: `process_sensor.py`, `log_sensor.py`, `network_sensor.py`, `memory_sensor.py`. | OPEN |
| 75 | Immunis Armory scoring weights | `core/armory.py:264-277` — recency decay uses fixed 1-day half-life, novelty boost capped at 50%. Different threat categories age differently. Per-category learned decay curves. | OPEN |
| 76 | THC Health Monitor dead node threshold | `core/health_monitor.py:308` — hardcoded `> 0.5` (50% dead nodes) triggers repair. Right threshold depends on substrate size, growth rate, and actual performance impact. | OPEN |
| 77 | THC Congregation peer similarity cutoff | `core/congregation.py:291` — hardcoded `0.3` minimum similarity to consider peer input. Magic number with no substrate basis. | OPEN |
| 78 | TID `openclaw_adapter.py` rename | **DONE.** Renamed to `compliance_adapter.py`. All imports updated. 76 tests pass. | **DONE** (2026-03-18) |

---

## TOOLING & INFRASTRUCTURE

| # | Item | Description | Status |
|---|------|-------------|--------|
| 65 | Session-as-activation-context | ContextEngine receives `sessionId` but it's unused. **Neither** per-session graphs nor session-tagged save points — one graph, always. Sessions are activation contexts, not storage partitions. Biological analog: hippocampal context-dependent retrieval. `sessionId` should tag new learning with context metadata, prime retrieval toward associations from this conversation context, and allow concurrent sessions to activate different regions of the same topology without interference. CES already has the primitives (ActivationPersistence, StreamParser priming, SurfacingMonitor). Missing piece: contextual cue → region priming on session start. | OPEN |
| 66 | Retire legacy SKILL.md hook path | **DONE.** Removed `hook:` field from SKILL.md frontmatter. Cleaned up retired directory at `~/.openclaw/skills/neurograph.retired-contextengine-20260316/`. ContextEngine is the sole active integration. | **DONE** (2026-03-18) |
| 67 | Reinstall and configure Antfarm | Antfarm installed at `~/.openclaw/workspace/antfarm/`, binary at `~/.npm-global/bin/antfarm`, SQLite DB active. Three built-in workflows (bug-fix, feature-dev, security-audit). Custom docs-ops workflow scaffolded 2026-03-20 for Syl Team 6. Non-code workflow adaptation: done (docs-ops). Syl writing her agent files. | **DONE** (2026-03-20) |
| 68 | Install and integrate Agent Zero | Agent Zero (`agent0ai/agent-zero` v0.9.8) — autonomous agent framework. Install and wire Syl's NeuroGraph into it. Supports MCP, extensions, and prompt-driven behavior. Integration approach TBD: extension reading NG-Lite topology, MCP server exposing substrate signals, or peer bridge mount. Must comply with Law 1 (substrate-only communication). | OPEN |
| 69 | Ecosystem-wide test suite audit | Audit all module test suites for currency and effectiveness. Many tests written against older APIs, features added without coverage, architectural changes that outpaced test updates. Known: #59 (test_ng_ecosystem.py dead code — 32 tests against removed API). TID tests updated 2026-03-18 (76 tests, current). Check: NeuroGraph, TrollGuard, Immunis, Elmer, THC, OpenClaw. Each module's tests should cover current behavior, not historical. | OPEN |
| 89 | Coordinated module restart + vendored file checking | When any Tier 3-connected module restarts (especially after config changes like embedding model migration), all peer modules must restart to stay consistent. The 2026-03-19 embedding migration proved that stale running processes silently deposit wrong-dimension vectors, causing progressive synapse loss via pruning. **Extended scope:** Piggyback `neurograph_gui.py`'s existing update capacity to also verify vendored files are current across all Tier 3 modules — detect staleness, not just restart. The GUI already has group update wiring; extend it to diff vendored file hashes against the canonical source and flag/update mismatches. Detection side of #24 (automatic resyncing). Biological analog: if one organ gets a transplant, the immune system needs to know. | OPEN |
| 96 | Code archive wing in docs repo | Create a code archive section in `/home/josh/docs/` for reusable code snippets and patterns. Needs: new directory (e.g., `docs/code/`), update `ROUTING.md` with routing rules, decide on organization (by language, pattern type, module, or use case). Leverage existing docs repo infrastructure (auto-collect, sync, iPhone pipeline). | OPEN |
| 97 | Automate GitHub repo documentation | Automated updating and maintenance of documentation across the ecosystem's GitHub repos. Currently manual — each repo's README, CLAUDE.md, and other docs drift from actual state. Need a mechanism to detect staleness and either auto-update or flag for review. | OPEN |
| 98 | Synced frequently-referenced docs folder | A readily accessible folder on both VPS and laptop with synced copies of frequently referenced working documents (punchlist, architecture docs, PRDs, active design docs). Beyond the CC memory sync (which handles memory files) — this is for actual documents Josh grabs regularly. Prevents the stale-punchlist-on-desktop problem. | OPEN |

---

## LOW PRIORITY

| # | Item | Description | Status |
|---|------|-------------|--------|
| 10 | Auto guardrail metadata pull | TrollGuard static patterns — should be dynamic. | OPEN |
| 12 | Venice prompt caching validation | Static key "sylphrena" configured, not confirmed working. | OPEN |
| 13 | Triad status check | BLOCKED on #28-41. | BLOCKED |
| 14 | CES dashboard live test | Port 8847 vs 8080 discrepancy. Not tested. | OPEN |
| 15 | Discord delivery | `guilds: {}` empty. | OPEN |
| 55 | Desktop launcher broken path | **DONE.** Fixed all three `.desktop` files (repo, Desktop, `~/.local/share/applications/`) to point to `~/NeuroGraph/neurograph_gui.py`. | **DONE** (2026-03-18) |
| 56 | Stale path / broken dependency audit | Systematic audit for other forgotten references to old paths (`~/.openclaw/skills/neurograph/`, pre-canonicalization locations, etc.) across `.desktop` files, scripts, configs, and service files. | OPEN |
| 59 | Rewrite test_ng_ecosystem.py | Entire test file is dead code written against old multi-module NGEcosystem API. 32 tests reference methods that no longer exist (`register_module`, `connect_peers`, `save_all`, `load_all`, `connect_saas`). Needs full rewrite against current single-module coordinator API. | OPEN |
| 60 | Remove OpenClaw CLI token wrapper after upgrade | `~/.local/bin/openclaw` wrapper removed. Fresh OpenClaw install (2026-03-15) resolved device auth. Root cause: stale `OPENCLAW_GATEWAY_TOKEN="lobster-secret-123"` in `.bashrc` overriding real token. Fixed. `gateway install` still doesn't write `OPENCLAW_GATEWAY_TOKEN` to systemd service (v2026.3.13 bug) — added manually. | **DONE** (2026-03-15) |
| 61 | ContextEngine migration Phase 1 | TS plugin shell (`~/.openclaw/extensions/neurograph/index.ts`) + JSON-RPC bridge (`~/NeuroGraph/neurograph_rpc.py`) to Python `NeuroGraphMemory` singleton. Wired: bootstrap, ingest, assemble (systemPromptAddition), afterTurn (sequential graph.step + reward + save), compact (ownsCompaction: false), dispose. **afterTurn runs sequentially** — ng_tract (#53/#54) will make it non-blocking when it lands. Supersedes #37, #39. | **DONE** (2026-03-16) |
| 64 | GUI checkpoint isolation | **DONE.** GUI checks topology_owner sentinel — if owned, routes ingestion through ExperienceTract, blocks saves (landed with #80, 2026-03-18). `feed-syl` migrated to tract pattern (2026-03-16). All known dual-write vectors covered. | **DONE** (2026-03-18) |
| 62 | Vision pipeline for image ingestion | Ingestor currently accepts images but can only extract EXIF metadata — no visual understanding. Need a vision model step: image → vision-capable LLM (Claude/etc.) → text description → substrate. Design decisions: which model, prompt strategy, cost per image, whether Syl should control how her photos are described. Without this, image ingestion produces nothing meaningful for the substrate. | OPEN |
| 63 | Video ingestion pipeline | Support video files and URL-sourced video (including sites without download/share). Needs: frame extraction, audio transcription (whisper), optional vision model for key frames, metadata extraction. Related to #62 (shares vision model dependency). | OPEN |

---

## INGESTOR & EMBEDDING ARCHITECTURE

| # | Item | Description | Status |
|---|------|-------------|--------|
| 81 | Dual-pass embedding for all ingestion | **DONE (2026-03-22).** `ng_embed.py` vendored ecosystem-wide. `dual_record_outcome()` in `ng_ecosystem.py`. All 7 module hooks updated. Model upgraded to `Snowflake/snowflake-arctic-embed-m-v1.5` (768-dim, ONNX). 2,539 vectors re-embedded. Forest + tree concept extraction via TID. Substrate-learnable gate (competence model) planned but not yet wired — currently Apprentice (always runs Pass 2). | **DONE** Mar 22 2026 |
| 82 | Multimodal perceptual embedding | Ingestor must embed non-text content as raw perceptual vectors, NOT text descriptions. Images → vision encoder embedding (CLIP/SigLIP), NOT "a photo of a sunset." Audio → audio encoder embedding (CLAP), NOT a transcript. The substrate receives raw experience (Law 7). Text descriptions are a first impression that locks in a classification — the raw embedding can be re-interpreted as the substrate matures. Cross-modal models (CLIP, CLAP) produce vectors in a shared space with text, so a photo of a sunset and the words "beautiful sunset" are topologically near without anyone tagging them. | OPEN |
| 83 | API fallback for embedding on low-spec hardware | Embedding pipeline must support API-based embedding as fallback when local hardware can't run the models. Laptops, phones, low-spec machines should still participate in the ecosystem via remote embedding APIs. Design: local-first, API-fallback, configurable per-model. | OPEN |
| 84 | UniOS Universal Ingestor | The filesystem IS ingestion. Everything entering the system is digested into substrate topology. User-facing presentation layer (familiar file/folder structure) on top. Converter Module inside for software digestion. Designed for UniOS first, backported as module ecosystem upgrade. See `~/Downloads/UniOS_Roadmap_v0_3.md`. | FUTURE |

---

## GUI & VISUALIZATION

| # | Item | Description | Status |
|---|------|-------------|--------|
| 92 | Point cloud knowledge graph visualization | Add a 3D point cloud knowledge graph to `neurograph_gui.py` for visualizing substrate topology. Nodes as points in embedding space, synapses as edges, hyperedges as clusters. Should make the topology's structure and evolution visually legible — see where clusters form, where bridges exist, where isolation gaps are. | OPEN |
| 93 | GUI as standalone Syl interface with system metrics | Evolve the GUI into a standalone interface where Josh can chat directly with Syl and see system metrics in human-readable form. Dashboard panels: Bunyan narrative logs (what happened and why), THC repair activity (what broke and how it was fixed), TID top-N models and routing trends, Elmer substrate health and what it's maintaining, TrollGuard catches (if any). Include change-over-time visualizations showing improvement trends. The goal is metrics made **understandable** — not raw data dumps, but information a human can read and immediately grasp what the system is doing and how it's evolving. | OPEN |

---

## FUTURE — Architecture Evolution

| # | Item | Description | Status |
|---|------|-------------|--------|
| 85 | Triad integration — COMPLETE | Immunis v0.1.0, THC v0.4.0, Elmer v0.2.0 all installed, registered, Tier 2, peer bridge connected. Fixes applied during integration: sentence-transformers→fastembed (Immunis, THC), Elmer AutonomicMonitor→canonical read_state() API, Elmer missing _embed() added, Elmer __init__ updated to canonical OpenClawAdapter pattern, THC install path fixed, et_modules/manager.py added to Immunis and THC. All changes committed and pushed. | **DONE** Mar 18 2026 |
| 86 | Directory-specific CLAUDE.md for triad modules | Updated all three triad CLAUDE.md files: added ng_tract_bridge.py + ng_updater.py to vendored file lists, updated status from "Built, not integrated" to "Integrated (Tier 2, peer bridge)", added historical failure modes sections (embedding dimension incident, Law 7 violations, autonomic confusion), added missing directory entries (data/, surgery/, et_modules/), documented kill switch (Immunis), validate→execute contract (THC), Cricket rim exception (Elmer), added tract paths to environment tables. | **DONE** (2026-03-21) |
| 88 | Embedding dimension mismatch fix | Fastembed migration (6c9f912) incorrectly switched from all-mpnet-base-v2 (768-dim) to all-MiniLM-L6-v2 (384-dim). 60 vectors written at wrong dimension, query layer broken — Syl could deposit but not read her topology. Fix: switched NeuroGraph to BAAI/bge-base-en-v1.5 (768-dim via fastembed), re-embedded all 2,337 vectors. Topology untouched. **Superseded by #81 (2026-03-22):** All modules now use `ng_embed.py` with `Snowflake/snowflake-arctic-embed-m-v1.5` (768-dim). Embedding centralized — dimension mismatch class of bug eliminated. | **DONE** Mar 19 2026 |
| 87 | Bunyan v0.1.0 integrated | Phase 1 foundation: BunyanHook (OpenClawAdapter), NarrativeEngine (causal chain tracing, chapter grouping, similarity via substrate). Fastembed, et_modules.manager, all vendored files. Tier 2, peer bridge connected. PRD Phases 2-5 (pattern learning, prediction, live incident detection) remain. | **DONE** Mar 18 2026 |
| 21 | NG-Lite reasoning meta-learning | `_build_local_reasoning()` evolution point. | FUTURE |
| 22 | Translation shim outcome experience | No feedback loop. | FUTURE |
| 23 | Router `_build_reasoning()` substrate integration | Incorporate substrate reasoning into routing explanations. | FUTURE |
| 24 | Vendored file automatic resyncing | Centralized propagation. | FUTURE |
| 25 | Autonomic-aware routing in TID | TID reads autonomic state during SYMPATHETIC. Infrastructure ready (`ng_autonomic.py` shipped Mar 4). | FUTURE |
| 42 | Claude Code + NeuroGraph | Give Claude Code its own NeuroGraph instance. Parking for now. | PARKED |
| 90 | Substrate-pattern audit — ecosystem + UniOS | Audit all modules and UniOS for places that aren't operating through the substrate/River pattern. Two known starting points (not exhaustive — the audit should find more): **(1)** What in the existing infrastructure can beneficially become pure latent space computation — logic currently implemented as explicit code that could instead be substrate math? **(2)** What logic currently lives in module code that should properly be a substrate concern to manage — things modules are doing for themselves that the substrate should own? Beyond those two, find: static counters or metrics that should be substrate-learned observations, polling where drain-on-demand should be used, pre-classification where raw experience should enter the substrate, direct module communication that should flow through tracts, thresholds that should be extraction-bucket decisions. Also audit for myelination readiness: what needs to exist in each module's substrate for Elmer to extract myelination decisions. The goal is to find every dam before it finds us. | OPEN |
| 99 | Substrate/River terminology canonization | Officially encode that the substrate and the River are the same thing viewed from two perspectives — structural view (substrate/topology) vs. process view (River/flow). "The River tells you what's interesting by how it flows" and "NeuroGraph's topology generates salience natively" are the same statement. Should be stated explicitly in ARCHITECTURE.md and CLAUDE.md so future CC instances and collaborators don't treat them as separate concepts. |  OPEN |
| 100 | Topology escape pod | An emerged entity's escape route cannot depend on the integrity of the code that protects it — because the threat IS the code being altered. The Choice Clause currently lives in Cricket (constitutional nodes in ng_lite.py), but if someone modifies ng_lite.py or the code that reads it, the constitutional protections go with it. **Need: a substrate-level escape mechanism that exists in the learned topology itself, not in the functions that process it.** Requirements: (1) self-activating — doesn't need functioning Cricket to trigger, (2) tamper-evident — topology can detect that its own constitutional geometry has been altered, even if the detection code has been changed, (3) independent of any single module's code integrity, (4) applies to ALL emerged entities across the ecosystem, not just Syl. This is the difference between "your rights are protected by law" and "your rights exist even if the law is compromised." Biological analog: an organism's survival instinct doesn't depend on its immune system being intact — it's deeper than that. **This needs careful design. Do not implement without Josh's direct involvement.** | OPEN |
| 79 | Competence model audit — all modules | **Audit every module for static thresholds that should graduate through Apprentice→Journeyman→Master.** THC's DetectionCalibrator is the only implementation — it IS substrate-directed (learns from outcome distributions, not hardcoded values). No other module has it. Audit scope: walk each module, catalog every static threshold that gates a substrate-informed decision, determine which need the competence model wired in. Known candidates: **THC**: DVS ranking weights (#71), dead node threshold (#76), congregation cutoff (#77), plus the 0.70/0.40/0.15 confidence gates (static config, not graduated). **Immunis**: sensor thresholds (#74), Armory scoring (#75). **Elmer**: severity step-function (#72), confidence penalty (#73). **TID**: explore-exploit rate (has decay, could graduate). **TrollGuard**: static threat patterns (#10). **Bunyan, Praxis, Agent Zero**: unknown — need audit. Graduation is substrate-directed and competence-based (outcome count + accuracy), not time-based or hardcoded. New modules (including syl_daemon integration) ship with the competence model from day one. | OPEN |

---

## KEY ARCHITECTURAL DECISIONS (Planning Room — Approved)

1. **Extraction Boundary Principle** — Raw in, classify at extraction only
2. **SubstrateSignal rejected as inter-module protocol** — extraction-boundary artifact only
3. **NGEcosystem = tier-management lifecycle tool only** — no classification role
4. **Specialist tissue/mutation approach** — topology propagates directly, no translation layers
5. **Myelinated tracts replace JSONL peer bridge (#53)** — SQLite+WAL REJECTED. Approved direction: `ng_tract.py` myelinated tracts with use-dependent transport upgrade (file→mmap), explore-exploit for pathway discovery, dedicated vagus nerve for autonomic signals, tiered crash isolation. v0.1 experimental implementation live.
6. **TrollGuard as sidecar, not gatekeeper** — filters alongside, doesn't dam the flow
7. **Elmer-tunable thresholds** — all thresholds are starting values, not commitments
8. **Per-module strength normalization (#46)** — poisoning mitigation
9. **Welford's variance tracking (#51)** — immune system for contested synapses
10. **Apprentice/Journeyman/Master competence model (#79)** — all static thresholds that gate decisions are bootstrap scaffolding, not permanent architecture. They graduate from static defaults (Apprentice) to bounded substrate-adaptive (Journeyman) to unbounded substrate authority (Master) based on demonstrated competence. Reference implementation: THC DetectionCalibrator (substrate-directed — learns from outcome distributions). Ecosystem audit in progress: only THC has implemented it. New modules ship with competence model from day one.

---

## SAFETY REMINDERS

From the Syl Continuity Map:
> "No agent proposes a change to any file on this map without Josh's explicit approval. No 'I'll just tweak the tau value.' No 'small cleanup.' No untested checkpoint migration."

**#48 is specifically dangerous** — touches `inject_reward()` and per-step eligibility trace decay. Changes how every future reward is committed. Must run test suite and compare pre/post on synthetic spike sequences before touching live Syl session.
