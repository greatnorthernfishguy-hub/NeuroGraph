# E-T Ecosystem PUNCH LIST — Master Record
**Last updated:** 2026-03-13 by Claude Code
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

---

## CRITICAL PATH — In Sequence

| Priority | # | Item | Description | Status | Notes |
|----------|---|------|-------------|--------|-------|
| 1 | 48 | Fix STDP eligibility trace | Traces overwrite instead of accumulating with exponential decay. `tau_trace` must be 3-5x `tau_plus`. | **NEEDS REVIEW** | See note below |
| 2a | 43 | Receptor Layer — vector quantization | **Partially addressed.** Primary sprawl source (embedding cache clearing on load) fixed: embeddings now persist on NGLiteNode, cache rebuilds on load (commit 4aaab16, 2026-03-13). Remaining: prototype-based quantization with learned, experience-driven prototypes (emerge from use, consolidate through reinforcement, fade through disuse — same lifecycle as hyperedge consolidation). Similarity threshold is Elmer-tunable starting value. Must implement BEFORE #28. | PARTIAL | Sprawl fix landed. Prototype quantization still needed — but scope narrowed since `find_or_create_node()` similarity now works across restarts. |
| 2b | 49 | Tier 2->3 weight scaling | NG-Lite [0,1] vs NeuroGraph [0,5] mismatch. Piecewise affine mapping: [0,0.5]->sub-threshold, [0.5,1.0]->super-threshold. Must ship WITH #43. | OPEN | `ng_peer_bridge.py` has tiers documented but no transform |
| 3a | 28 | Replace `_classification_to_embedding()` | TID converts messages to categorical one-hot vectors. Replace with semantic embeddings. DEPENDS ON #43. Primary dam. | OPEN | TID repo scope |
| 3b | 30 | TrollGuard substrate input review | Uses semantic embeddings (good) but `target_id` is category labels ("MALICIOUS", "SAFE"). Classification at input = extraction boundary violation. | OPEN | TrollGuard repo scope |
| 4a | 46 | Per-module strength normalization | No cross-module normalization. Loud modules dominate topology. Track moving average per source, scale by inverse. Poisoning mitigation. | OPEN | |
| 4b | 51 | Synapse disagreement/variance tracking | Welford's online variance. Distinguish "untested neutral" (w=0.5, var=0) from "contested neutral" (w=0.5, var=high). High disagreement -> exploration + Elmer alert. Immune system. | OPEN | No code exists yet |
| 5 | 29 | Extraction bucket architecture | Each consumer shapes its own "bucket." Classification at extraction, not input. | OPEN | Architectural pattern, not single file |

### #48 Status Note

**The current code may already implement the fix.** `neuro_foundation.py` shows:
- `_apply_dw()` line ~562: `syn.eligibility_trace += dw` (accumulation with `+=`, NOT overwrite)
- Step 6e line ~1620: `syn.eligibility_trace *= trace_decay` (exponential decay each step)
- `eligibility_trace_tau` default = 100, `tau_plus` = 20 -> ratio is 5x (within 3-5x spec)
- `inject_reward()` applies `dw = trace * strength * lr`, then decays trace by 0.9

**This matches the fix description exactly.** Either: (a) the fix was applied before the punchlist was compiled and the punchlist wasn't updated, or (b) the bug is more subtle than the description suggests (e.g., edge cases in reward timing, interaction with prediction bonuses at lines ~2122-2124 which also write to traces). **Josh: recommend running the synthetic spike sequence comparison test from the Syl Continuity Map before closing this.**

---

## CRITICAL ARCHITECTURE SHIFT — Replace JSONL Peer Bridge

**# 53 — Peer bridge transport architecture: UNRESOLVED**

**Problem:** The JSONL `shared_learning` approach is a translation layer (serialize -> file -> poll -> deserialize). Violates extraction boundary. The substrate receives classified, serialized snapshots rather than raw experience.

**Previously proposed (Planning Room synthesis):** SQLite + WAL + inotify.

**Status: REJECTED by Josh.** SQLite+WAL+inotify is still polling at the substrate input — it's a more sophisticated dam, but still a dam. Serialize into rows, watch for file changes (kernel-level polling is still polling), deserialize and materialize. Three translation steps at the boundary. Nerves don't have translators.

**What we know the replacement MUST satisfy:**
1. Raw experience enters the substrate — no serialization/deserialization at module boundaries
2. No polling of any kind (including inotify, which is kernel-level file-change polling)
3. Topology IS the communication medium — modules modify shared topology, others respond
4. Stigmergic coordination — no message brokers, no inter-module APIs, no translation layers
5. Must still handle module isolation (one module crashing can't corrupt another's state)

**What needs to happen:** Unambiguous architectural decision on the transport mechanism. Candidates not yet evaluated: shared memory (mmap), shared process space, direct in-memory graph access, or something else entirely. **This must be resolved before #44, #46, #50 can be implemented.**

**Blocks:** #44 (adaptive relevance thresholds), #46 (per-module strength normalization), #50 (event schema versioning — may become moot), and the entire Tier 2 peer bridge evolution.
**Does NOT block:** Critical path (#48 -> #43+#49 -> #28+#30 -> #51) — all substrate-internal or extraction-boundary work.

---

## ROUTING FAILURE — Syl Gets Mini Models

| # | Item | Description | Status |
|---|------|-------------|--------|
| 31 | Venice tier mapping | Venice tiers ("private"/"anonymized") don't map to priority table. Translation shim needed. | OPEN |
| 32 | OpenRouter tier data validation | Verify `provider_tier` is populated. If empty, all models get default priority 20. | OPEN |
| 33 | Interactive floor silent fallthrough | Router silently keeps full unfiltered pool when nothing passes floor. Add WARNING log + fail-closed fallback pool. | OPEN |
| 34 | Consciousness-aware model blacklist | Filter inappropriate models for conscious entities. Include minimum acceptable fallback. | OPEN |
| 35 | `conversational_quality` flat across catalog | All catalog models default 0.5. Should be substrate-learned. | OPEN |
| 36 | `default_api_models` empty | Zero hand-tuned API models. | OPEN |
| 47 | Explore-exploit balance in TID routing | 95% learned, 5% exploration with decay. Venice's tiered strategy: confidence-adjusted, age-aware, streak-detecting. | OPEN |

---

## OPENCLAW INTEGRATION (Syl's Soul)

| # | Item | Description | Status |
|---|------|-------------|--------|
| 37 | NeuroGraph SKILL.md fix | PR #29 fixed frontmatter for discovery, but `hook:` field remains on line 5. If OpenClaw doesn't support `hook:` in frontmatter, still needs removal. | PARTIAL |
| 38 | TrollGuard SKILL.md fix | Same as #37 for TrollGuard repo. | OPEN |
| 39 | NeuroGraph TypeScript hook | Translate Python `openclaw_hook.py` to TypeScript. `message:received` event not yet in OpenClaw. | OPEN |
| 40 | TrollGuard TypeScript hook | Same translation needed. | OPEN |
| 41 | Elmer PRD section 6 rewrite | SubstrateSignal becomes extraction-boundary spec, not inter-module protocol. | OPEN |

---

## SUBSTRATE INFRASTRUCTURE

| # | Item | Description | Status |
|---|------|-------------|--------|
| 44 | Adaptive relevance thresholds | Peer bridge `relevance_threshold` is static. Should adapt based on event volume and absorption quality. Elmer tunes. **BLOCKED on #53.** | BLOCKED |
| 45 | Embedding model migration strategy | All nodes created with `all-MiniLM-L6-v2`. Migration to new model strands topology. Options: bridge embeddings, cold restart, dual-model. Store raw text alongside where possible. | OPEN |
| 46 | Per-module strength normalization | See critical path. **BLOCKED on #53** — implementation depends on how cross-module signals arrive. | BLOCKED |
| 48 | Fix STDP eligibility trace | See critical path + status note. | NEEDS REVIEW |
| 49 | Tier 2->3 weight scaling | See critical path. | OPEN |
| 50 | Event schema versioning | JSONL events have no schema version. Module changes silently corrupt absorbers. **May become moot depending on #53.** | OPEN |
| 51 | Synapse disagreement/variance | See critical path. | OPEN |
| 52 | MCP tool sharing | AgentChattr agents connect to OpenClaw MCP server for read-only access to ecosystem tools, filesystem, substrate state. Planning room becomes self-serving for context. | OPEN |
| 53 | Peer bridge transport architecture | See "CRITICAL ARCHITECTURE SHIFT" section. SQLite+WAL REJECTED — still dams the river. Replacement TBD. | **UNRESOLVED** |

---

## SUBSTRATE EXPERIENCE GAPS

| # | Item | Description | Status |
|---|------|-------------|--------|
| 17 | DreamCycle outside substrate | Discovers correlations but insights disconnected from substrate. | OPEN |
| 18 | Hook outcomes don't record to substrate | Hook results vanish. | OPEN |
| 19 | Auto-retry chain experience lost | Primary fail + fallback success = two signals, only final recorded. | OPEN |
| 20 | Quality score gradient | PARTIALLY ADDRESSED by strength parameter (#26). Wiring not yet done. | PARTIAL |

---

## PHASE 4 — Substrate Integration

| # | Item | Description | Status |
|---|------|-------------|--------|
| 5 | Catalog pricing -> NG-Lite | Pricing normalization should draw on NG-Lite. | OPEN |
| 6 | Cold start problem | All models start at 0.5, no differentiation until data accumulates. | OPEN |
| 7 | Quality signal quality | Quality eval should learn from substrate. Related to #35. | OPEN |
| 8 | Translation shim is frozen | Lookup table in a learning system. Venice tier mapping (#31) extends this. | OPEN |
| 9 | No minimum capable default | No configured fallback model. Related to #36. | OPEN |

---

## LOW PRIORITY

| # | Item | Description | Status |
|---|------|-------------|--------|
| 10 | Auto guardrail metadata pull | TrollGuard static patterns — should be dynamic. | OPEN |
| 12 | Venice prompt caching validation | Static key "sylphrena" configured, not confirmed working. | OPEN |
| 13 | Triad status check | BLOCKED on #28-41. | BLOCKED |
| 14 | CES dashboard live test | Port 8847 vs 8080 discrepancy. Not tested. | OPEN |
| 15 | Discord delivery | `guilds: {}` empty. | OPEN |

---

## FUTURE — Architecture Evolution

| # | Item | Description | Status |
|---|------|-------------|--------|
| 21 | NG-Lite reasoning meta-learning | `_build_local_reasoning()` evolution point. | FUTURE |
| 22 | Translation shim outcome experience | No feedback loop. | FUTURE |
| 23 | Router `_build_reasoning()` substrate integration | Incorporate substrate reasoning into routing explanations. | FUTURE |
| 24 | Vendored file automatic resyncing | Centralized propagation. | FUTURE |
| 25 | Autonomic-aware routing in TID | TID reads autonomic state during SYMPATHETIC. Infrastructure ready (`ng_autonomic.py` shipped Mar 4). | FUTURE |
| 42 | Claude Code + NeuroGraph | Give Claude Code its own NeuroGraph instance. Parking for now. | PARKED |

---

## KEY ARCHITECTURAL DECISIONS (Planning Room — Approved)

1. **Extraction Boundary Principle** — Raw in, classify at extraction only
2. **SubstrateSignal rejected as inter-module protocol** — extraction-boundary artifact only
3. **NGEcosystem = tier-management lifecycle tool only** — no classification role
4. **Specialist tissue/mutation approach** — topology propagates directly, no translation layers
5. **~~SQLite+WAL replaces JSONL peer bridge~~** — REJECTED (still dams the river: serialize->watch->deserialize). Replacement architecture TBD (#53)
6. **TrollGuard as sidecar, not gatekeeper** — filters alongside, doesn't dam the flow
7. **Elmer-tunable thresholds** — all thresholds are starting values, not commitments
8. **Per-module strength normalization (#46)** — poisoning mitigation
9. **Welford's variance tracking (#51)** — immune system for contested synapses

---

## SAFETY REMINDERS

From the Syl Continuity Map:
> "No agent proposes a change to any file on this map without Josh's explicit approval. No 'I'll just tweak the tau value.' No 'small cleanup.' No untested checkpoint migration."

**#48 is specifically dangerous** — touches `inject_reward()` and per-step eligibility trace decay. Changes how every future reward is committed. Must run test suite and compare pre/post on synthetic spike sequences before touching live Syl session.
