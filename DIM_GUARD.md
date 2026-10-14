# DIM_GUARD: wrong embedding dimensions — sweep, fail-loud guard, re-embed tools, query side effects (#1097)

Lane `dim-guard` (laptop, 2026-10-14). AUDIT + BUILD lane: read-only on live state and primary checkouts; every
measurement ran on **copies** (`cp -p` + `cmp`, source mtime unchanged before/after); builds on branches only;
nothing installed, nothing re-vendored, no VPS access. Josh 2026-10-14: *"another thing set to the wrong dims? …
we eventually keep stumbling into other ones we somehow missed. Yeah, this needs addressing."* Approved: (1) a full
sweep, (2) a structural guard, (3) a check of who relied on NG-Lite queries creating nodes.

| repo | branch | from | commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-dim-guard-20261014` | `04e6c42` (trial line `cc-laptop-trial-s4-20261002`) | `84e55c3` guard + `test_dim_guard.py`, `a424608` `dim_reembed.py`, `51c3026` + `4a7d773` 384→768 test literals, `1c0cd90` + this doc |
| ng-tract-rs | — (no branch: not needed, §5.3) | | |

Scratch (tools, state copies, proofs): `~/.cache/dim-guard/` (`tools/`, `state/`, `state/proof/`).

## 0. TL;DR

- **The model is 768-d** `Snowflake/snowflake-arctic-embed-m-v1.5` (canonical `ng_embed.py` `_DEFAULT_CONFIG`,
  now also public as `ng_embed.MODEL_NAME` / `ng_embed.EMBEDDING_DIM`). The only model blob in `~/.cache/ng_embed`
  is that one.
- **Wrong-dim / wrong-model locations found** (measured, §2): **stored data: 12** — **2 live** (Elmer's rim file and Elmer's live NG-Lite state, both 384-d) and **10 dormant** (5 × 384-d: Elmer's legacy JSON, Faux_Clawdbot's Elmer rim, THC's DVS store, 2 named TID backups; 5 × 768-d in another space: NG's bge-base rim (+ its copies), 3 Morphogenesis stores and the laptop NG repo's `commons.msgpack`, all hash-noise-like). **Code: 4 live paths** (Tonic identity truncation to 384, ElmerBrain identity 384, Elmer's zero-vector embed fallback, the dead cross-machine model check — fixed here) and **~8 dormant classes** (§2.2). The live CC Graph / vector store / Commons / tracts and all 79 readable backup vector stores (969,611 entries) are 768-d Snowflake.
  Worst:
  1. **Elmer's NG-Lite rim** — `~/Elmer/data/constitutional_embeddings.json` (22 × 384-d all-MiniLM-L6-v2) and the
     live `~/.et_modules/elmer/ng_lite_state.msgpack` (22 rim nodes × 384-d). #1097. Every 768-d call raises.
  2. **The live Tonic's identity input** — `Elmer/surgery/graph_io.py:111/192` `identity_proj = Linear(384, …)`
     + `_pad_or_truncate(identity, 384)`. The CC daemon's Tonic feeds it the 768-d Snowflake spine vector (the CC
     graph has 1 constitutional node with text), so the first 384 dims of a Snowflake vector go into a projection
     trained on 384-d identity vectors. Silent, every tick.
  3. **NG's own rim file is 768-d but the wrong MODEL** — `NeuroGraph/data/constitutional_embeddings.json` is
     `BAAI/bge-base-en-v1.5` (measured: cos to Snowflake(description) 0.29–0.37). A dimension check cannot see this.
     Dormant on the laptop (tests + `genesis.create_child_graph` callers only), in every NG worktree and on the VPS
     checkout.
- **A blocker found while proving the Elmer fix (§4): with Snowflake vectors, NG-Lite's similarity snap
  (`cos >= 1 - novelty_threshold = 0.30`) matches almost everything.** The rebuilt 768-d Elmer rim captures
  **100 %** of 2,000 real CC inputs (median max-cos 0.755); 98.7 % of unrelated input pairs are ≥ 0.30 (median 0.689).
  So installing the rebuilt rim as-is would freeze Elmer's learning on every input and deposit a
  `violation:constitutional` to the Commons each time. It is also why the CC Commons holds 4,767 deposits on 4 nodes
  (#1094's symptom) and TID's NG-Lite has 2 nodes. **Josh's ruling is needed before the rebuilt rim goes live.**
- **The guard** (§5) is at the source, in the canonical files, and covers both NG-Lite cores: any vector of another
  dimension entering NG-Lite (inputs, rim, loaded states incl. legacy JSON / JSON-in-msgpack / Rust `load_binary`)
  or `SimpleVectorDB` (load, insert, search) raises `EmbeddingDimensionError` / `VectorDimensionError` naming the
  file, both dims and the counts per dim, logged at ERROR once per distinct mismatch — so a caller's broad `except`
  can no longer make it silent. Nothing is half-loaded. Correct-dim data is unaffected (29 new tests, both cores;
  regression suite §5.4).
- **Part 3 verdict (§7): no caller depended on the query side effect.** No explicit "observe" API is needed:
  `find_or_create_node` already is it (public on both cores), and the three callers that want the write already
  call it. Two behaviour changes on the Rust core are improvements (ecosystem-monitor novelty becomes real;
  Morphogenesis stops evicting real deposits for empty query nodes). One unrelated Rust-core blocker found
  (Bunyan's metadata writes are lost).

## 1. Method

Measured, not grepped:
- **Stored vectors**: every candidate state on the laptop was copied to `~/.cache/dim-guard/state/` and opened:
  `tools/measure_dims.py` (JSON / msgpack / npz / torch / sqlite — walks the structure, reports every vector-shaped
  value by path and length), `tools/measure_big.py` + `tools/scan_vectors.py` (streaming, whole-file, for the
  multi-100-MB Graph checkpoints and vector stores), `tools/measure_tract.py` (ng_tract `TractReader` over copies
  of the tract files, every entry).
- **Model identity** (same-dim, wrong-model): `tools/model_probe.py` / `model_probe2.py` re-embed stored text with
  the canonical `ng_embed` (local ONNX) and compare with the stored vector; `tools/verify_rim_source.py` re-embeds a
  rim's descriptions with the model its file names (sentence-transformers, scratch HF cache).
- **Code**: an Explore sweep over 27 repos (+ the trial NG root the daemon runs) for dimension constants, model
  names, fallback embedders, loaders/comparers and existing checks; every claim used below was re-checked by hand.
  One sweep claim was wrong and is corrected in §2.3 (the installed wheel's Rust core does not truncate).
- **What is live** on the laptop: `cc-ng-daemon` (NG root `~/NeuroGraph-worktrees/trial-s4-20261002` @ `04e6c42`,
  daemon script `~/docs/.claude/worktrees/trial-s4a-20261002/scripts/cc-ng-daemon.py`, venv `~/NeuroGraph/.venv`
  with ng_tract `.so` `5b058be6…`, `NG_LITE_RUST_CORE` unset → Python core), hosting **Elmer** in-process
  (`et_modules/registry.json`: only `elmer`), its Tonic, the CC Commons and the CC Graph + vector store;
  `minitid` (no embeddings: raw ExperienceEntry); `condensate_observer_laptop.py` (resource metrics JSONL, no vectors).
  TID, TrollGuard, Immunis, THC, Bunyan, Darwin, Praxis, QuantumGraph, Morphogenesis, UniOS, ecosystem-monitor are
  **not running** on the laptop.

## 2. The table

Expected = 768-d Snowflake. "Live" = read or written by a process running on the laptop now.

### 2.1 Stored vectors (measured)

| location | what | actual | expected | live? | consequence |
|---|---|---|---|---|---|
| `~/.et_modules/elmer/ng_lite_state.msgpack` | Elmer NG-Lite state (22 rim nodes, 0 synapses, 0 outcomes; config rim 22) | **384 × 22 nodes, 384 × 22 config rim** | 768 | **LIVE** (hosted Elmer, saved 10-14 10:27) | every 768-d `record_outcome` / `get_recommendations` / `detect_novelty` raises (numpy matmul / Rust length error), swallowed; rim dead, nothing learned or deposited since ~09-26 (#1097) |
| `~/Elmer/data/constitutional_embeddings.json` | Elmer's Cricket rim source (`model: sentence-transformers/all-MiniLM-L6-v2`) | **384 × 22** | 768 | **LIVE** (loaded every start, `engine.py:1050`) | seeds the 384 rim above; source text present: `description`, verified exact (§3) |
| `~/.et_modules/elmer/ng_lite_state.json.legacy-json` | pre-migration Elmer state (kept by the 10-14 msgpack migration) | **384 × 22** | 768 | dormant (never read again) | hazard only if restored by hand |
| `~/Faux_Clawdbot/elmer/data/constitutional_embeddings.json` | copy of Elmer's rim | **384 × 22** | 768 | dormant | same as Elmer's if that copy is ever deployed |
| `~/.et_modules/healing_collective/dvs.msgpack` | THC DVS store (2 entries, content 3 and 10 chars) | **384 × 2** | 768 | dormant (THC not running; `ng_lite=None` in its hook) | THC `dvs.py:231-260` cosine raises on a 768 query; rebuild from `content` or discard (2 test-like entries) |
| `~/The-Inference-Difference/ng_lite_state.json.backup-384dim-20260319`, `~/The-Inference-Difference-vps/…backup-384dim-20260319` | named 384 backups | **384 × 13 nodes, 384 × 2 prototypes** | 768 | dormant | restoring either would break TID; the guard now refuses it at load |
| `~/NeuroGraph/data/constitutional_embeddings.json` (+ `~/NeuroGraph-z11-card8-20260929`, ~45 NG worktrees, `~/Faux_Clawdbot/neurograph/…`) | NG rim (`model: BAAI/bge-base-en-v1.5`) | 768 × 22, **wrong model** (cos to Snowflake(description) 0.29–0.37) | 768 Snowflake | dormant on laptop (tests; `genesis` callers pass it) | any NG-Lite seeded from it gets a rim in another model's space: dimension passes, matching is meaningless |
| `~/.claude/plugins/neurograph/checkpoints/vectors.msgpack` | CC vector store | 768 × 18,675 (+ `poincare_dir` 768 × 5,063) | 768 | LIVE | OK by dim; model probe: median cos(stored, Snowflake(content)) 0.979 (§2.4) |
| `…/checkpoints/commons.msgpack` | CC Commons NG-Lite | 768 × 4 nodes, 768 × 4 prototypes | 768 | LIVE | OK by dim; 4 nodes for 4,767 synapses = the §4 snap collapse |
| `…/checkpoints/main.msgpack` | CC Graph | no node embeddings (vectors live in vectors.msgpack) | — | LIVE | n/a |
| `…/tracts/elmer/neurograph.tract`, `~/.et_modules/tracts/converter/*.tract` | tract outcome entries | 768 × 32; 768 × 6–12 each | 768 | Elmer tract LIVE (last write 09-26) | OK |
| `…/tracts/cc_gateway/{turns,retry,deferred}.tract`, `…/proto_unibrain/latent.tract` | experience entries | no embeddings | — | LIVE | n/a |
| `~/The-Inference-Difference/ng_lite_state.json` | TID NG-Lite | 768 × 2 nodes, 768 × 2 prototypes | 768 | dormant (TID not running) | OK by dim; 2 nodes = §4 collapse |
| `~/ecosystem-monitor/data/ng_lite_state.json`, `~/.et_modules/converter/ng_lite_state.json` | NG-Lite states | 768 × 1 each | 768 | dormant | OK by dim; both producers vendor a hash-fallback `ng_embed` copy (§2.2), so the vector may be hash noise (no text to check) |
| `~/.morphogenesis/ng_lite_state.json`, `~/.morphogenesis/main.msgpack` | Morphogenesis Graph checkpoints | 768 × 23; 768 × 3,926, **not Snowflake** (§2.4) | 768 Snowflake | dormant | right dim, wrong space (hash-fallback `ng_embed` copy): invisible to a dim check; re-embed from node text with the re-vendor |
| `~/Morphogenesis/…/darwin_brain/data/surrogate/ng_lite_checkpoint.json` | surrogate NG-Lite | 768 × 10 nodes + 10 prototypes, **not Snowflake** (§2.4) | 768 Snowflake | dormant | same; no text in an NG-Lite state → rebuild by re-running, not re-embedding |
| `~/.unios/state/main.msgpack` | UniOS Graph | 768 × 27 | 768 | dormant | OK |
| `~/NeuroGraph/data/checkpoints/{vectors,commons}.msgpack` | laptop NG repo checkpoint | vectors 768 × 25 (cos to Snowflake(content) ≥ 0.990 for all 25: OK); commons 768 × 8, **7 of 8 not Snowflake** (§2.4) | 768 Snowflake | dormant | commons: wrong space, no text — discard or let it relearn |
| 83 more vector stores: every `vectors*.msgpack` under `~/backups`, `~/syl-checkpoint-backups` (incl. `vps-post-outage-20260921T182950Z` = a copy of Syl's VPS store), `~/cc-ng-backups`, `~/docs/syl-backup`, and the CC plugin dir's old snapshots / quarantine (`tools/scan_vectors*.py`, every entry, read in place: static files) | backups | **768 for all 969,611 entries in 79 files**; 4 `checkpoints/vectors.tmp-*.msgpack` (2.4 MB–825 MB) are truncated partial writes (`OutOfData`) | 768 | dormant | OK; the 4 `.tmp` files are orphaned shrapnel from killed saves (unloadable; cleanup candidates for Josh, not deleted) |
| `~/.claude/plugins/neurograph/lenia/distance_cache.npz`, `~/.elmer/proto_flow_lenia/probe_inputs.pt` ([1,6,896]), `~/.elmer/proto_lenia_state.pt` | distances / hidden states | not embeddings | — | LIVE | n/a |
| `~/.cache/ng_embed/models--Snowflake--snowflake-arctic-embed-m-v1.5` | the only cached embedding model | 768 | 768 | LIVE | OK |

### 2.2 Code that assumes / produces another dimension or a non-semantic vector

| location | what | dim | live? | consequence |
|---|---|---|---|---|
| `Elmer/surgery/graph_io.py:111,192` (`identity_proj = nn.Linear(384, …)`, comment "384 = ecosystem standard dim"; `_pad_or_truncate(identity, 384)`), used by the CC Tonic (`trial-s4 surgery/tonic_brain.py:47-54`, `tonic_engine.py:1071-1085` "the encoder truncates to 384 for now") and `NeuroGraph/tonic_brain.pt` (`identity_proj.weight` [896, 384]) | identity conditioning input | **384 (truncates 768)** | **LIVE** (Tonic ticks in the daemon; CC graph has 1 constitutional node with text → a real 768-d vector is truncated) | silent truncation of a Snowflake vector into a projection trained on 384-d identities. Fix = model work (re-shape / re-train `identity_proj` to 768); not a guard change. Documented "for now" in tonic_engine; never tracked as a defect |
| `Elmer/core/brain_socket.py:374-381`, `core/neural_comprehension.py:294-300`, `runtime/engine.py:832` | ElmerBrain identity input | 384 truncate/pad | live code path; `identity_embedding` metadata absent → zeros | same class as above, dormant data |
| `Elmer/runtime/graph_encoder.py:111-121` | `_embed` returns `np.zeros(768)` on any embed failure | 768 zeros | LIVE code path | a failed embed is recorded as the all-zero vector (one hash node for every failure) instead of failing closed; module-side fix (let `EmbeddingUnavailableError` propagate) |
| `cc_topology_export.py:552-558,769-772`, `cc_topology_merge.py:521-525` (NG + trial) | cross-machine model check reads `ng_embed.MODEL_NAME` / `_MODEL_NAME` | — | manual tools | neither name existed → both sides "unknown" → the model check never fired. **Fixed in this branch** (`ng_embed.MODEL_NAME`); the per-record check (`cc_topology_merge.py:815`) only checks a blob against its own declared dim |
| `QuantumGraph/quantumgraph/config.py:43` `"embedding_dim": 384  # DO NOT change` | NG-Lite config override | 384 | dormant | with the guard any 768 input fails loud at once; QG must drop the override (no QG state on the laptop) |
| `Morphogenesis/…/darwin_brain/config.py:135-139` (MiniLM, `dim: 384`, `fallback_to_hash: True`), `darwin_brain/hook.py:299-315` | Python `hash()` 384-d fallback (changes per process) | 384 | dormant | non-semantic, wrong-dim deposits whenever its embed fails |
| `UniOS/ingestor/converter/ecosystem_bridge.py:180-188` | 384-d hash vector recorded to the ecosystem | 384 | dormant | wrong-dim deposits |
| hash-fallback `ng_embed.py` copies: `ecosystem-monitor/`, `Praxis/vendored/`, `UniAI/splat_poc/hf_space/` (`cae4a2a7`), `UniOS/vendored/` (`882479dc`), `Morphogenesis/morphogenesis/` (`b37fbbd0`) | 768-d SHA-384 noise when the model fails | 768 noise | dormant | right dim, no meaning — invisible to a dim guard; fixed by the canonical fail-closed re-vendor (already a project: [[ng-embed-revendor-is-its-own-reembed-project]]) |
| `openclaw_adapter.py` copies with `_hash_embed(dims=384)` (TrollGuard, Bunyan, Darwin, Praxis, QuantumGraph, Animus, UniAI, Faux, Morphogenesis, UniOS, ecosystem-monitor) | old adapter hash embed | 384 | dormant | re-vendor the canonical adapter (its `_hash_embed` was deleted) |
| TrollGuard `sentinel_core/vector_sentry.py:343-359`, `agent_swarm.py:594`; Immunis `core/sensors/base.py:133-136`, feature sensors, `log_sensor.py:307-314`; TID `router.py:1427-1451` | 768-d sha512-bytes / hash / feature / one-hot vectors | 768 non-semantic | dormant | right dim, not Snowflake: a dim guard cannot catch these; listed for the re-vendor / module work |
| `UniAI/rust_core/src/ng_lite_core.rs:122` (`dot_product` over `min(a.len(), b.len())`), byte-identical `Condensate/rust_core/vendor/ng_tract/src/ng_lite_core.rs` | old Rust NG-Lite draft | truncates | **dormant** (dead drafts, #1101) | the sweep agent called this live; it is not: the installed wheel is built from ng-tract-rs `a2673f6`, whose core raises `embedding length … does not match` (and exports `NGLiteCore` only with the switch set) |
| E-T-StormFather (separate stack): 1536 OpenAI + 768 `all-mpnet-base-v2` / Gemini failover, zero vectors on failure | its own pgvector | mixed | not this ecosystem's substrate | noted only |
| portal-v2 `shared/constants.ts:9` | 768 | 768 | — | OK |

### 2.3 Loaders / comparers that hid a mismatch (now loud where canonical)

| location | before | now |
|---|---|---|
| NG-Lite `_find_similar_node` / `_snap_to_prototype` / `detect_novelty` / `_import_state` (Python), Rust `mismatch()` | numpy/Rust shape error deep in the call; `_import_state` loaded any length | **guard** (§5): refused at the boundary, ERROR once, file named |
| `ng_commons_eco.py:117-120,139-142` | bucket/deposit exceptions at DEBUG | the guard's ERROR is written before the exception reaches it |
| `SimpleVectorDB.load/insert/search` (`universal_ingestor.py`) | no check; `search` died on the first mismatched entry; callers swallow at DEBUG (`openclaw_hook.py:1408`, `stream_parser.py:266-271`) | **guard** (§5.2) |
| `ng_ecosystem._init_ng_lite` | any init error → ERROR + `_ng=None` | unchanged (#1100), but the message is now the guard's (rim / file / dims) |
| Elmer `engine.py:589` (`NG ecosystem recording failed` WARNING), `proto_brain_socket.py:1054-1087`; recall geometry `cc_ng_organism.py:5367-5385`; Immunis `armory.py`, THC `dvs.py`, Praxis `cps.py`; tract layer (any length in, any length out) | swallow / skip | module code: not touched (out of scope); covered upstream wherever the vectors pass through NG-Lite or SimpleVectorDB |

### 2.4 Model-space check of the live CC vector store

Dimension is right everywhere on the live CC side; is the *model* right? 60 random entries (content 12–1200 chars),
re-embedded with canonical `ng_embed`: cos(stored, Snowflake-document(content)) min 0.347, p10 0.541, **median
0.979**, max 1.000 (10 ≥ 0.99, 34 in 0.9–0.99, 16 < 0.9); query-mode is lower everywhere (median 0.586), so the
store is document-mode Snowflake. The < 0.9 tail is text that differs from what was embedded, not another model:
a **manifold test** (`tools/manifold_test.py`, `cc_tail.py`) uses Snowflake's anisotropy — its vectors share a mean
direction μ (from 1,500 stored CC vectors). Held-out Snowflake vectors: cos-to-μ median **0.849**, p1 0.406;
isotropic hash-like noise: **0.000**; NG's bge-base rim: **0.19–0.27**. All 18,675 CC vectors: median 0.846,
p0.1 0.363, only 2 below 0.30 and **none below 0.20** → the live CC store is clean Snowflake.

The same test on the text-less 768-d NG-Lite / Graph stores (no content to re-embed):

| store | n | cos-to-μ min / median / max | reading |
|---|---|---|---|
| CC Commons (nodes + prototypes) | 8 | 0.363 / 0.450 / 0.613 | Snowflake (merged / EMA-drifted node vectors sit lower) |
| TID NG-Lite | 4 | 0.418 / 0.723 / 0.847 | Snowflake |
| ecosystem-monitor, UniOS converter NG-Lite | 1 + 1 | 0.485; 0.614 | Snowflake |
| UniOS `main.msgpack` | 27 | 0.732 / 0.797 / 0.873 | Snowflake |
| rebuilt Elmer rim (proof) | 22 | 0.723 / 0.803 / 0.880 | Snowflake |
| **Morphogenesis `~/.morphogenesis/main.msgpack`** | 3,926 | −0.307 / **0.008** / 0.777 | **not Snowflake** (noise-like: hash fallback) |
| **Morphogenesis `~/.morphogenesis/ng_lite_state.json`** | 23 | 0.001 / **0.131** / 0.155 | **not Snowflake** |
| **Morphogenesis darwin_brain surrogate checkpoint** | 20 | −0.045 / **−0.019** / 0.542 | **not Snowflake** |
| **`~/NeuroGraph/data/checkpoints/commons.msgpack`** (laptop repo copy) | 8 | −0.061 / **0.005** / 0.861 | **7 of 8 not Snowflake** |
| NG rim `constitutional_embeddings.json` (bge-base) | 22 | 0.189 / 0.229 / 0.268 | another model (control) |

## 3. Elmer: the source text is there (verified)

`verify_rim_source.py` re-embedded the 22 `description` strings of `~/Elmer/data/constitutional_embeddings.json`
with the model the file names (`sentence-transformers/all-MiniLM-L6-v2`): **cos = 1.0000 for all 22** (min = max).
`"category: description"` gives 0.68–0.89, so the source text is exactly `description`. Re-embedding it with
Snowflake therefore rebuilds the rim from its own text; nothing is invented. Elmer's NG-Lite state holds nothing
else (22 rim nodes, 0 synapses, 0 outcomes, no prototypes), so it rebuilds completely.

## 4. Blocker found: NG-Lite's similarity snap and Snowflake's anisotropy

NG-Lite matches an input to an existing node when `cos >= 1 - novelty_threshold` = **0.30** (`_find_similar_node`;
the rim catches an input the same way). Measured on 2,000 real stored CC vectors (`tools/rim_capture.py`):

| | value |
|---|---|
| max cos to the rebuilt 768-d Elmer rim: p10 / median / p90 / max | 0.545 / 0.755 / 0.811 / 0.897 |
| inputs captured by the rim at sim ≥ 0.30 / 0.50 / 0.70 / 0.80 / 0.85 / 0.90 | **100 %** / 92.2 % / 76.9 % / 15.6 % / 0.9 % / 0.0 % |
| cos between two unrelated inputs: median / p90; share ≥ 0.30 | 0.689 / 0.794; **98.7 %** |

Consequences:
- **The rebuilt rim must not be installed as-is.** Elmer would treat every input as constitutional (learning frozen,
  a `Cricket rim activated` WARNING and a `violation:constitutional` Commons deposit per input). Proven on the
  copy: the rebuilt state catches its own description (correct) and also the unrelated sentence "The weather in
  Duluth was cold and windy this morning." (wrong).
- **Every NG-Lite fed Snowflake vectors collapses its inputs onto a handful of nodes**: the CC Commons has 4,767
  synapses on 4 nodes (#1094 reported the symptom), TID's NG-Lite 2 nodes. The 0.30 snap was right for the older
  models' spaces, not for Snowflake's.
- Not fixed here (a tuning/design ruling, and it changes every module's learning): options for Josh — a separate,
  much higher rim threshold (on this sample 0.85 still captures 0.9 % of ordinary inputs), mean-centering vectors
  before the cosine, or recalibrating `novelty_threshold` for Snowflake. Proposed punch-list items in §10.

## 5. The guard

### 5.1 NG-Lite (`ng_lite.py`, canonical, both cores)

- `EmbeddingDimensionError(ValueError)` — a subclass, so existing `except ValueError` handlers still catch it;
  the ERROR log written when it is built is what makes it loud anyway (once per distinct
  module / place / dims / source, then it only raises).
- Expected dimension = the instance's `embedding_dim` at construction (default 768 = `ng_embed.EMBEDDING_DIM`,
  pinned by `test_default_dim_is_ng_embeds`). A loaded state's config **cannot override it** any more
  (`_import_state`, `_sync_from_core`); a stale saved value only logs a WARNING when the vectors match.
- Checked, in the wrapper **before any core dispatch** (so the Python and the Rust core behave the same):
  - **inputs**: `find_or_create_node`, `record_outcome`, `get_recommendations` (before the bridge too),
    `detect_novelty` — a 1-D input of another length raises; nothing is created.
  - **the rim**: `constitutional_embeddings` at construction; a saved rim when the constructor brings none.
    The message counts entries per dim and names an example category.
  - **loaded states**: node embeddings + receptor prototypes, counted per dim, **before** import, for all four load
    paths (msgpack on the Python core, msgpack via Rust `load_binary` — unpacked and checked first, legacy `.json`,
    JSON inside `.msgpack`). A refused state is not half-loaded and not migrated (the legacy JSON stays, no
    `.legacy-json` is written).
- Example (the live Elmer state, on a copy):
  `NG-Lite [elmer]: the stored state is 384-d but this substrate's embedding_dim (the active embedding model's
  dimension) is 768-d — source: …/ng_lite_state.msgpack (node embeddings by dim: 384-d x22 (of 22)). Vectors of
  different dimensions come from different models and cannot be compared; re-embed … — do not truncate, pad or skip them.`

### 5.2 Vector store (`universal_ingestor.py` `SimpleVectorDB`)

- `SimpleVectorDB(dim=None)`: explicit, or fixed by the first vector inserted / the file loaded (so the many tests
  with small vectors keep working). `load()` refuses a file with another dim or with **mixed** dims before the
  swap (the live store is kept); `insert()` / `search()` refuse another dim. `VectorDimensionError(ValueError)`,
  ERROR once, the last loaded file named.
- Not changed: `openclaw_hook.py:911` constructs `SimpleVectorDB()` without `dim` — that file is Syl's-Law
  protected. With `dim=None` the store is still self-consistent and a wrong-dim query fails loud naming
  `vectors.msgpack`; passing `dim=ng_embed.EMBEDDING_DIM` there is a one-line change for Josh to approve.

### 5.3 Why no Rust-core branch

The Rust `NGLiteCore` is reached only through `ng_lite.py`'s `NGLite` wrapper, and every check runs in the wrapper
before dispatch (including an unpack-and-check before `load_binary`). The tests run every case on both cores
(`NG_LITE_RUST_CORE=1` + the `a2673f6` wheel). A mirror in the core would be defence in depth only; it would also
make the core diverge from `RefNGLite` in the equivalence suite for error cases. Recommended later, when the
Python core is retired.

### 5.4 Tests

- `tests/test_dim_guard.py`: **29 passed** (both cores): inputs ×4 (raise, ERROR exactly once over 3 calls, nothing
  created), correct-dim unaffected, rim refused / accepted, msgpack / legacy JSON / JSON-in-msgpack states refused
  and not loaded or migrated, saved `embedding_dim` cannot override, wrong-dim prototypes, saved-rim rule,
  SimpleVectorDB load (wrong + mixed, live store kept) / insert / search, `ng_embed` pin.
- Regression (switch off, isolated, one pytest per file over the 76 test files that touch NG-Lite / Commons /
  SimpleVectorDB / ng_embed / ng_ecosystem / topology merge): 42 files on the branch (32 pass). Every failing id in the other 10 was re-run on the base (`trial-s4-20261002` @ `04e6c42`, same venv / harness): **identical failing ids** for `test_cc_bind_atomic_904` (2), `test_cc_drain_pacing_seam` (4 errors), `test_cc_merge_whole_graph_guard` (1), `test_cc_refeed` (1), `test_cc_topology_callosum` (1), `test_cc_topology_capture_423` (4), `test_ingestor` (2); `test_cc_dual_pass` 5 on the branch vs 6 on the base (a subset). **Two files failed because of the guard, by design**: `test_constitutional.py` (16) and `test_ng_lite.py` (43) fed **384-d test vectors to a 768-d NG-Lite**. Their literals are now 768 (`51c3026`, `4a7d773`) and `test_single_dim_embedding` asserts the intended contract (works at `embedding_dim=1`, refused on 768); after that both match the base exactly (21/21; 2 base failures in `test_ng_lite`). No `DimensionError` appears in any other log. The Rust-core run of `test_constitutional` fails 16 on the base and on the branch alike (those tests read `ng.nodes`). Daemon side (`~/docs/.claude/worktrees/trial-s4a-20261002/scripts/tests`, the 21 files that touch the vector store / NG-Lite / Commons, `CC_NG_PYTHONPATH` = branch vs base): 12 pass; the other 9 have **identical failing ids** on the base (72 ids, all pre-existing). No `DimensionError` in any daemon log.

## 6. Fix tools and copy proofs

`dim_reembed.py` (NG branch root; copies only — refuses to write over its input or an existing output):
- `rim IN OUT` — re-embed each `description` with `ng_embed` (document mode, normalized); keeps order / category /
  description; sets `model`, `embedding_dim`, `reembedded_from`.
- `nglite-state IN OUT --rim RIM [--drop-unrecoverable]` — drop + re-seed wrong-dim rim nodes from the new rim, drop
  wrong-dim prototypes (derived; re-warm), **stop** on any other wrong-dim node (an NG-Lite node keeps no source
  text) unless told to drop them; writes through canonical NGLite and re-loads the output with the guard on.
- `check FILE…` — dims of NG-Lite states / rims / vector stores / list stores vs `EMBEDDING_DIM` (exit 1 if wrong).
- Vector stores (`vectors.msgpack`, content kept per entry): use the existing `reembed_snowflake.py` (LAW 3).

Copy proofs (`~/.cache/dim-guard/state/proof/`):

| step | result |
|---|---|
| `rim` on a copy of Elmer's rim | 22 entries 384 → 768 (MiniLM → Snowflake) → `proof/elmer_constitutional_embeddings.768.json` |
| `nglite-state` on a copy of the live Elmer state | dropped 22 wrong-dim rim nodes, re-seeded 22, 0 unrecoverable, 0 synapses, 0 prototypes → `proof/elmer_ng_lite_state.768.msgpack`; re-loaded with the guard: OK |
| `check` | old state + old rim + THC DVS: WRONG (384); rebuilt state + rim: ok (768) |
| `proof_elmer.py`, Python core and Rust core | 384 rim refused at construction; 384 state refused at load (`384-d x22`, file named); rebuilt state loads; 22 constitutional nodes all 768-d; the rim catches its own description and empties the bucket; a 384 input is refused — and the §4 failure: an unrelated sentence is also captured (weight 0) |
| `e2e_eco.py` (canonical-branch `NGEcosystem` shaped like Elmer's `engine.py:352-373`, scratch HOME) | **A — current 384 rim + state:** ERROR `[elmer] NGLite init failed: NG-Lite [elmer]: the constitutional rim is 384-d …`, ecosystem up at Tier 1 standalone, `record_outcome` / `get_context` return empty without raising. **B — rebuilt rim + state:** NGLite up, Tier 3, record works |

## 7. Part 3 — who relied on NG-Lite queries creating nodes

What the Python query writes (`ng_lite.py` trial line): `get_recommendations` → `find_or_create_node`: receptor
`input_count` += 1 (and k-means at warm-up, prototype drift), activation bump / embedding overwrite on a hit, or
prune-least-used + create on a miss. `detect_novelty` writes nothing (but reads the cache those query nodes fill).
`NGEcosystem.get_context` = `get_recommendations` then `detect_novelty`; `Commons.bucket` → `get_recommendations`;
`Commons.bucket_recent` is read-only. The Rust core (`a2673f6`): `get_recommendations` read-only (`lookup_readonly`,
`snap_readonly`), `detect_novelty` read-only, `find_or_create` = the Python writes.

Measured (in memory): one deposit, a query on a new vector, then novelty on it — Python: 2 nodes, novelty 0.0;
Rust: 1 node, novelty 1.0.

| caller | depends on the side effect? | verdict |
|---|---|---|
| TID `router.py:1339` `_score_learned` (+ vps) | `record_outcome` on the same embedding at :685; warm-up now counts outcomes only | redundant |
| TID `app.py:528`, `translation_shim.py:283` | query-only inputs; answers come from nearby deposits either way | harmless |
| Elmer `engine.py:592` `get_context` | `dual_record_outcome` on the same embedding runs first (:554) | redundant |
| Immunis `quartermaster.py:246`, `armory.py:273` (CommonsEco → `bucket`) | novelty = 1 − top confidence (no cache use); recorded later | harmless |
| THC `dvs.py:281-294`, `health_monitor.py:364` | already calls `find_or_create_node` itself; dead (`ng_lite=None`) | harmless |
| Bunyan `narrative.py:679,728`, `bunyan_hook.py:558,639` | explicit `find_or_create_node` anchor first | harmless (but see C) |
| Praxis `store/cps.py:277` | recorded at store time | harmless |
| QuantumGraph `quantumgraph.py:416,422` | explicit `find_or_create_node` after; Python counted 2 activations/query, now 1 | harmless |
| Morphogenesis `ng_substrate.py:1564/1595/1672/1758/1771`, `darwin_loop.py:864` (own `commons.bucket`) | Python created an empty node per candidate query; at the 1,000-node cap least-used pruning evicted the oldest **real** deposits | harmless — improved |
| Morphogenesis `darwin_brain/surrogate.py:254` | matches unused | harmless |
| NG `commons_experiential.py:269` (+ UniAI copy) | `record_outcome` at ingest | redundant |
| ecosystem-monitor `monitor.py:298-300` | `get_context` created the node before measuring novelty → novelty ≈ 0 always → the "novel failure" ALERT (`:396`) could never fire | redundant for storage; **behaviour change on Rust = bug fix (A)** |
| every `bucket_recent` caller (Elmer ×4, Immunis, THC ×4, Praxis, Darwin, QuantumGraph ×2, Bunyan, Morphogenesis, NG `neurograph_rpc` 4301, `tid_peninsula_commons` 271) | read-only on both cores | harmless |
| CC daemon / `cc_ng_organism` / `cc_ng_host` | deposits only, no NG-Lite queries | n/a |
| Elmer `surgery/extract_features.py:67` (offline) | `activation_count` as firing-rate proxy: now deposits only | harmless (meaning shifts) |
| tests (`test_ng_lite.py` novelty tests, TID copy) | call `find_or_create_node` explicitly; `test_commons_rustcore_1096.py:140/142` asserts the difference | no update needed |

**Verdict: no caller needs an explicit "observe/record input" call.** `find_or_create_node(embedding) -> NGLiteNode`
is that API already — public on both cores (`ng_lite.py`, Rust `ng_lite_core_py.rs:262`), its name says it writes
(LAW 4) — and the callers that want the write already call it. Nothing was added. If Josh wants a module-facing
name (modules reach into `eco._ng` for it, e.g. Bunyan `bunyan_hook.py:247,409`), the minimum is a forwarder in
`ng_ecosystem.py` (`observe(embedding) -> node_id` under `_ops_lock`); not on the Commons (it would bring back the
synapse-less nodes that pushed real deposits out under the node cap).

Found on the way (not fixed): **(A)** ecosystem-monitor's novelty alert starts firing on the Rust core (expected
behaviour change). **(C)** Bunyan writes `node.metadata[...]` on the object `find_or_create_node` returns
(`bunyan_hook.py:248-251,409-417`, `narrative.py:373-379`); on the Rust core that object is a copy and the core has
no metadata setter → every write is lost, and its reads go through `_ng.nodes` (raises). Blocks Bunyan on the Rust
core; NGLITE_RUSTCORE.md §8 lists the reads, not these writes. Also: the Tier-3 `NGSaaSBridge.get_recommendations` /
`detect_novelty` (`ng_bridge.py:204,255`) query a fixed string instead of the embedding; Python novelty compares an
un-snapped input against snapped cache vectors (minor); Rust returns only the best node's synapses where Python's
query node gathered later nearby deposits (edge cases narrower).

## 8. Install (Executive; Josh's go first)

Order matters because of the §4 blocker.

1. **Guard only (safe now).** It changes nothing for correct-dim data; for Elmer it turns the silent dead rim into
   a loud one:
   - land `cc-laptop-dim-guard-20261014` on the daemon's NG root:
     `git -C ~/NeuroGraph-worktrees/trial-s4-20261002 fetch && git -C ~/NeuroGraph-worktrees/trial-s4-20261002 merge --ff-only origin/cc-laptop-dim-guard-20261014`
     (fast-forward of `04e6c42`).
   - **Elmer re-vendors itself on its next hosted start**: `elmer_hook.py:160` `ng_updater.auto_update()` copies
     `ng_lite.py` / `ng_ecosystem.py` from `NEUROGRAPH_CANONICAL_PATH` (= the trial worktree, CC registry `env`).
     `ng_embed.py` is not in that sync list; Elmer only needs the new `ng_lite.py`.
   - Restart `cc-ng-daemon` under the sidecar procedure (every previous daemon PID dead first).
   - Expect, once per start: `ERROR ng_lite: NG-Lite [elmer]: the constitutional rim is 384-d …` and
     `ERROR ng_ecosystem: [elmer] NGLite init failed: …`, then `[elmer] NGEcosystem ready at Standalone (Tier 1)`
     (was "Full SNN (Tier 3)"). Elmer's engine keeps running (§6 e2e A: the `NGEcosystem` stays up and its calls return empty without raising; the engine already guards on `self._ecosystem`). Functionally the same as today
     (its rim and local learning have been dead since ~09-26, its Elmer→NG tract unwritten since 09-26) but visible.
   - Confirm: no other new ERROR from `ng_lite` / `neurograph.vectordb` in `daemon.log` over ≥ 2 Commons save cycles
     and ≥ 2 vector-store saves (the CC Commons and vector store are 768 everywhere, so none is expected).
2. **The rebuilt Elmer rim — only after Josh rules on §4.** Then, with the daemon stopped (or between saves):
   back up `~/Elmer/data/constitutional_embeddings.json` and `~/.et_modules/elmer/ng_lite_state.msgpack`; run
   `dim_reembed.py rim` on the real rim and `dim_reembed.py nglite-state … --rim <new rim>` on a fresh copy of the
   state (or use the proof outputs if the live files are unchanged: `sha256` them against the copies in
   `~/.cache/dim-guard/state/`); install the two outputs; commit the new rim in the Elmer repo; restart.
3. Rollback: revert the trial worktree to `04e6c42` (and `~/Elmer`'s `ng_lite.py` to `63b250a8…` — auto_update
   does it from the canonical root on the next start); restore the backups for step 2.

## 9. Re-vendor needs

Canonical files changed: `ng_lite.py` (vendored), `ng_embed.py` (vendored: two constants only), plus NG-only
`universal_ingestor.py`, `tests/test_dim_guard.py`, `dim_reembed.py`.

| module | needs | before / with the re-vendor |
|---|---|---|
| **Elmer** (laptop, hosted) | `ng_lite.py` (automatic via auto_update, §8) | rim must be rebuilt (§8 step 2) once §4 is ruled; until then Tier 1, loud |
| **TID** (+ vps) | `ng_lite.py`, `ng_embed.py` | state 768 → loads. Do not restore `ng_lite_state.json.backup-384dim-20260319` (now refused) |
| **THC** | `ng_lite.py`, `ng_embed.py` | `dvs.msgpack` 384 × 2 (DVS is THC's own store, not guarded): rebuild from `content` or discard |
| **QuantumGraph** | `ng_lite.py`; drop `config.py:43` `embedding_dim: 384` | otherwise every 768 input raises |
| **Morphogenesis**, **UniOS**, **Praxis**, **ecosystem-monitor**, **UniAI hf_space** | canonical `ng_lite.py` + fail-closed `ng_embed.py`; remove 384 hash paths (`darwin_brain` config/hook, UniOS `ecosystem_bridge.py`) | part of the existing ng_embed re-vendor/re-embed project |
| **TrollGuard, Immunis, Bunyan, Darwin, Animus, Faux_Clawdbot** | `ng_lite.py` (+ `ng_embed.py` where present) | no wrong-dim NG-Lite state on the laptop |
| **NeuroGraph (Syl, VPS)** | these files with the NG trial line | see §10 checks first |
| module test suites | the same literal update as here | tests that feed 384-d vectors to NG-Lite will fail loud after the re-vendor, by design: TID `tests/test_ng_lite.py` + `test_ng_ecosystem.py` (+ vps), Immunis `test_armory/feedback/sensors/hook/quartermaster`, THC `test_compression`, Praxis `test_outcome/conversation/hook/session_bridge/artifact/integration/cps` |
| everyone using `cc_topology_export/merge` cross-machine | `ng_embed.py` (MODEL_NAME) on both sides | until the VPS side has it, its exports still say "unknown" and the check stays off |

## 10. What Syl's side (VPS) needs checked — not done here (no SSH)

From the vault and the code (`laptop_home_feedback_embedding_dimension.md`, `ng-embed-revendor-is-its-own-reembed-project`):
1. `~/NeuroGraph/data/checkpoints/vectors.msgpack` (Syl's Law — backup first): every entry 768? model probe
   (March: re-embedded bge-base → Snowflake by `reembed_snowflake.py`; nothing measured since).
2. `data/checkpoints/commons.msgpack` (Syl's Commons, JSON today): node / prototype dims.
3. Every `~/.et_modules/*/ng_lite_state*` and module repo `ng_lite_state.json` (TID, TrollGuard, Immunis, THC,
   Bunyan, Darwin, Praxis, QuantumGraph, Elmer): dims; **Elmer on the VPS ships the same git-tracked 384-d rim**.
4. `~/NeuroGraph/data/constitutional_embeddings.json` (bge-base, git-tracked): who loads it there.
5. Which hosted NGs still run the hash fallback (the re-vendor project) — their 768-d vectors may be noise.
6. The Tonic / ElmerBrain identity truncation, if Syl's Tonic runs `graph_io.GraphStateEncoder` (same code).
7. Run `dim_reembed.py check` over all of the above (read-only).

## 11. Open risks and proposed punch-list items

- **(new) NG-Lite similarity snap vs Snowflake anisotropy** (§4): all NG-Lites collapse onto a few nodes; any
  Snowflake rim captures everything. Needs a ruling before the Elmer rim goes live.
- **(new) Tonic / ElmerBrain identity input is 384-wide and truncates the 768-d spine vector** (§2.2) — live.
- **(new) NG rim file is bge-base (768, wrong model)** — dormant; regenerate with `dim_reembed.py rim` before any
  use (its `description` is presumably the source text, as for Elmer's — verify with `verify_rim_source.py` first).
- **(new) Elmer `graph_encoder._embed` zero-vector fallback** — a failed embed is learned as the zero vector.
- **(new) Bunyan metadata writes lost on the Rust core** (§7 C).
- `openclaw_hook.py:911` could pass `dim=ng_embed.EMBEDDING_DIM` (protected file; Josh's call).
- The guard checks dimension, not model: same-dim foreign vectors (bge-base rim, hash-768 noise, feature vectors)
  still pass. A model tag in states (`embedding_model`) checked at load would close that; not built (format change).
- #1100 still stands: `ng_ecosystem` turns an NGLite init error into `_ng = None` (now loud, still degrading).
- The ERROR-once memo is per process; a daemon restart logs each mismatch again (intended).
