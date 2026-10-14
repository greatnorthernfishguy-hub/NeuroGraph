# MEAN_CENTRE: mean-centred NG-Lite matching, re-calibrated thresholds, rebuilt rims, identity 768, space tag (#1107)

Lane `mean-centre` (laptop, 2026-10-14). BUILD lane: branches + copies only; nothing installed, no live state or
primary checkout touched, nothing re-vendored outside the branches. Josh 2026-10-14: *"Mean center"*; *"Running
without the Rim defeats the entire point of the Rim's existence. It's supposed to be immutable."*; laptop first.

| repo | branch | from | tip |
|---|---|---|---|
| NeuroGraph | `cc-laptop-mean-centre-20261014` | `6a8c8e7` (dim-guard tip) | see §11 |
| ng-tract-rs | `cc-laptop-mean-centre-rs-20261014` | `a2673f6` | see §11 |
| Elmer | `cc-laptop-mean-centre-20261014` | `cae7288` | see §11 |

Scratch (tools, copies, proofs, logs): `~/.cache/mean-centre/` (`tools/`, `state/` copies, `out/`, `suite/`).
Wheel: `~/.cache/mean-centre/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`
(sha256 `6fd96575…8fca`); test venv `~/.cache/mean-centre/venv` (only this wheel; deps via `.pth`).

## 0. TL;DR

- **The centre** is a fixed, versioned 768-d constant in canonical `ng_embed.py`: `CENTRE_VERSION =
  "snowflake-arctic-embed-m-v1.5/cc-store-20261014/v1"`, sha256 of its float32 bytes
  **`99f0e568b670e1591e40f1a7a9d518dc42dcc6b342fbf2ec6e269964c9824518`**, |μ| = 0.8023. Never recomputed at runtime.
- **NG-Lite (Python core + Rust `NGLiteCore`, bit-identical)** compares `normalize(v − μ)` at every comparison point:
  node matching, novelty, receptor snap, k-means assignment (the rim check and the Commons go through node matching).
  **Stored vectors stay raw.**
- **Thresholds, re-calibrated on centred real data:** match at centred cosine ≥ **0.60** (`novelty_threshold` 0.7 →
  **0.40**, tunable bounds 0.20–0.60), `receptor_prototype_threshold` **0.75** (now centred), new
  `constitutional_threshold` **0.55** (not runtime-tunable; a rim node catches only at ≥ 0.55 and then wins over
  ordinary nodes).
- **Rim (the shipped rebuilt rim, through NG-Lite, both cores, decisions identical):** catches **22/22** of its own
  descriptions and **56/66 = 84.8 %** of their paraphrases; catches **48 of 18,690 = 0.26 %** of ordinary CC inputs
  (precision on this labelled set 0.54); 2 of 15 borderline operational sentences. The calibration study (§4) puts
  the ordinary catches at short dual-pass concept trees with rim vocabulary (0 whole deposits, 0 vault chunks).
  Raw cosine could not separate them at any threshold (best, 0.85: recall 0.79 with 1.45 % false catches).
- **Grouping:** replaying the 18,675 real CC deposits into a fresh NG-Lite: raw/old → **13 nodes, 99.9 % of inputs
  matched an existing node**; centred/new → **11,082 nodes created, 40.7 % matched**, 5.8 synapses per node (caps
  1,000 nodes / 5,000 synapses now bind). On COPIES of the CC Commons (4 nodes, 4,947 synapses, top node 93.7 %)
  and TID (2 nodes) replaying 3,000 CC deposits: old → +2 / +3 nodes; new → 1,271 new nodes each, top node's share
  of synapses 93.7 % → 47.8 % (Commons) and 60 % → 10 % (TID).
- **Rims rebuilt from their own text** (verified first: stored vectors = model(description), cos 1.0000 × 22, for
  both MiniLM-Elmer and bge-NG): Elmer `data/constitutional_embeddings.json` and NG `data/constitutional_embeddings.json`
  (identical Snowflake vectors; same 22 descriptions), and Elmer's NG-Lite state re-seeded.
- **Elmer e2e on copies** (real `ElmerEngine`, guard + centring + rebuilt rim + rebuilt state, scratch HOME): starts
  at **Tier 3 with 22 rim nodes, 0 ERROR records**; 4/4 rim inputs caught, 0/3 ordinary.
  **Finding:** Elmer's engine-level Cricket report never fires — see §7.3.
- **Identity 768 (#1108):** the encoder takes the 768-d identity at full width; the trained 384-wide `identity_proj`
  migrates on load with outputs **bit-identical** to today's (proven on a copy of the live `tonic_brain.pt` with the
  real Qwen body); a 384-d identity is refused. `graph_encoder._embed` fails loud (#1110).
- **Model tag:** saves carry `embedding_space = {model, dim, centre_version}`; load refuses another model, an
  off-manifold untagged state and an off-manifold rim (`EmbeddingSpaceError`, ERROR once); raw-era thresholds reset.
- **Suites:** §9. **Install:** §10 (one restart, with atomic-saves + dim-guard, rim never absent).

## 1. The centring artifact

**What:** μ = mean of the L2-normalised **document-mode** Snowflake vectors of the CC vector store (copy of
`~/.claude/plugins/neurograph/checkpoints/vectors.msgpack`, 2026-10-14 10:44, sha256 `90e3ccb4…fe0f`), **18,675
vectors = 3,026 whole deposits (forests) + 15,649 dual-pass concepts (trees)**, accumulated in float64 in sorted-key
order, cast to float32. Shipped inside `ng_embed.py` (base64, 4 KB) — not a new file, so no new vendored file (LAW 2).
`ng_embed.embedding_centre()` decodes once, checks the sha256, returns a read-only array.
`compute_embedding_centre.py VECTORS --verify` (NG-only tool) reproduces it bit for bit.

**Why this corpus:** it is exactly what NG-Lite receives — the canonical model's document-mode outputs (every
NG-Lite caller embeds in document mode; no `is_query=True` caller exists), in the dual-pass mix that
`dual_record_outcome` deposits (one forest + its trees). It is the live store (re-embed agreement median 0.979,
DIM_GUARD §2.4). The vault was the planned second source; embedding it was not feasible on this box today (ONNX at
~6–12 s per 200-token text under load ≈ 9 load average), so the vault is a **validation sample** instead (27 chunks).

**Stability / sensitivity:**

| check | value |
|---|---|
| split halves (random 50/50 of the 18,675) | cos(μA, μB) = **0.9999**, ‖μA − μB‖ = 0.010 |
| forest-only vs tree-only mean | cos 0.840, ‖Δ‖ = 0.463 — the two input regimes have different means |
| vault sample (27 long chunks) mean vs μ | cos 0.778 (long documents sit with the forests) |

The forest/tree split is the one real sensitivity: no single vector zeroes both regimes. With the pooled μ, unrelated
pairs centre at F–T −0.08, T–T +0.02, F–F +0.17 (medians); the vault chunks pairwise +0.25. All are far below the
0.60 match threshold (unrelated F–T p99.9 0.40, T–T p99.9 0.58, vault p99 0.45; F–F p99 0.77 is the CC's
templated near-duplicate deposits). The pooled μ was kept because it is the stream's own mean.

**Versioning rule:** a new model or a new centre = a new `CENTRE_VERSION`. States record the version they were
calibrated under (§6); a changed centre needs no re-embedding (stored vectors are raw), only the threshold reset.

## 2. Where centring applies (both cores, bit-identical)

| comparison | Python `ng_lite.py` | Rust `ng_lite_core.rs` |
|---|---|---|
| node matching (incl. the rim, the Commons' deposits and buckets) | `_find_similar_node` | `find_similar` (+ `lookup_readonly`) |
| novelty | `detect_novelty` | `detect_novelty` |
| receptor snap | `_snap_to_prototype` via `_snap_sims` | `snap`, `snap_readonly` |
| k-means assignment | `_init_prototypes` | `init_prototypes` |

Math: input `normalize(v − μ)` (sqrt(dot) norm = `NGLite._normalize` / Rust `normalize`); stored rows
`(r − μ) / max(‖r − μ‖, 1e-12)` row-wise (numpy `norm(axis=1)` = Rust `row_renorm`). Without a centre both return
their input, so every kernel sees exactly the old operands. The Python core caches each row's centred copy (validated
by identity of the raw array; a row's centring is independent of the other rows, so these are the same bits) —
CPU per (novelty + match) at 1,000 nodes: raw 18 ms, centred 27 ms (uncached it was 76 ms).

Centring applies when `embedding_dim == ng_embed.EMBEDDING_DIM` and config `mean_centre` is true (default);
other dims (test fixtures) compare raw. An `ng_embed.py` without `embedding_centre` fails loud (they are vendored
together).

**Stored vectors stay raw** (node embeddings, prototypes, hashes, saves): LAW 7 — the substrate keeps the raw
experience; centring is a comparison-time lens. Consequences: a new centre version needs no re-embedding; existing
states load unchanged (their old raw vectors are compared centred); the hash (exact-match key) is unchanged.
Prototypes stay raw means of raw members (assignment by centred cosine).

## 3. Calibration (real data; `out/calib1.json`, `out/calib2.json`)

Labelled sets: **unrelated** = random pairs of real CC deposits (20,000 per type); **related (weak)** = a tree and its
own forest, sibling trees of one forest; **paraphrase** = 40 authored pairs (CC / ecosystem register, written for this
study, `data/authored.json`); authored unrelated = the 780 cross-pairs of those 40.

| pair set | raw median / p99 | **centred** p1 / p10 / median / p90 / p99 / p99.9 |
|---|---|---|
| unrelated forest–forest | 0.411 / 0.826 | −0.19 / −0.04 / 0.17 / 0.40 / 0.77 / 0.86 |
| unrelated tree–tree | 0.726 / 0.865 | −0.19 / −0.10 / 0.02 / 0.16 / 0.33 / 0.58 |
| unrelated forest–tree | 0.439 / 0.802 | −0.28 / −0.20 / −0.08 / 0.07 / 0.22 / 0.40 |
| authored unrelated | 0.577 / 0.754 | −0.12 / −0.04 / 0.07 / 0.20 / 0.37 / 0.49 |
| tree ↔ own forest | 0.459 / 0.868 | −0.24 / −0.12 / 0.06 / 0.32 / 0.60 / 0.79 |
| sibling trees | 0.744 / 0.940 | −0.17 / −0.07 / 0.07 / 0.31 / 0.79 / 0.92 |
| **authored paraphrase** | 0.908 (p1 0.778) | **0.44 / 0.65 / 0.79 / 0.87 / 0.91** |

Raw cosine: unrelated tree pairs (median 0.73) overlap paraphrases (p1 0.78) — no raw threshold separates them.
Centred: paraphrase p10 0.645 vs unrelated authored p99 0.37 / T–T p99.9 0.58.

**Match threshold 0.60 (`novelty_threshold` 0.40):** above the 99.9th percentile of unrelated concept pairs (0.58) and
of authored unrelated pairs (0.49), with 95 % paraphrase recall (0.55 also 95 %; 0.65 87.5 %). Per-input view (an
input against a population of N real nodes): capture of inputs whose text is not in the population, N = 100 / 1,000:
8.8 % / 34 % at 0.60 (raw 0.30: 100 % / 100 %). Tunable bounds 0.20–0.60 = match 0.40–0.80 (above the unrelated
p99; Elmer's TuningSocket cannot tune it back into a collapse).

**Receptor snap 0.75 (centred):** kept above the match threshold so a snap never merges what matching keeps apart.
In the full replay only 18 of 18,675 inputs snapped (the layer is near-inert at this value); 0.65/0.85 variants were
not completed (each replay took ~40 min on the loaded box) — an open tuning item, not a correctness one.

## 4. The rim

Positives: the 22 rim descriptions and **66 authored paraphrases** (3 per entry, new wording). Ordinary: **all 18,675
real CC deposits + 15 authored ordinary sentences**. Max centred cosine to the 22 rebuilt rim vectors:

| threshold (centred) | recall (66) | ordinary caught (of 18,690) | rate | precision (this set) | borderline ops caught (15) |
|---|---|---|---|---|---|
| 0.45 | 0.894 | 221 | 1.18 % | 0.21 | 4 |
| 0.50 | 0.864 | 79 | 0.42 % | 0.42 | 2 |
| **0.55** | **0.818** | **43** | **0.23 %** | **0.56** | 2 |
| 0.60 | 0.742 | 31 | 0.17 % | 0.61 | 1 |
| 0.65 | 0.576 | 10 | 0.05 % | 0.79 | 0 |
| raw 0.85 (best raw) | 0.788 | — | 1.45 % | 0.16 | — |

By category at 0.55: substrate_self_preservation 20/24, choice_clause 12/15, duck_ethics 8/12, person_protection 14/15.
The 43 ordinary catches at 0.55: **all dual-pass trees (0 whole deposits), 40 contain a rim word, 25 distinct texts**
(rim-vocabulary concepts extracted from ordinary discussion). Vault chunks: 0 of 11 (max 0.26).
**Chosen 0.55**: recall over precision would be 0.50 (Type I bias) at twice the false-catch rate; 0.55 halves the
false catches for 4.6 points of recall. This is the number for Josh to rule on (§12).

**Through NG-Lite itself** (`tools/rim_proof.py`; the shipped rim + the rebuilt Elmer state, `find_or_create_node`
on the Python and the Rust core, receptor off so the rim decision is isolated): own descriptions **22/22**, rim
paraphrases **56/66 (84.8 %)**, ordinary CC **48/18,690 (0.26 %)**, borderline 2/15; precision on this set 0.54;
**decisions identical on both cores**. (The table above used the same 22 descriptions embedded in a different batch:
cos 0.987–0.992 to the shipped vectors — see §12 item 10 — hence 54 vs 56 and 43 vs 48.)

**Rim priority:** a rim node catches at ≥ 0.55 and then wins even when an ordinary node is closer; below 0.55 a rim
node matches nothing and ordinary nodes compete at 0.60 (the old rule — rim hit only if it was the overall best at
≥ 0.30 — let an ordinary node created next to the rim shadow it).

## 5. Grouping on real data

**Replay of the 18,675 CC deposits** (store order, dual-pass-shaped targets) into a fresh NG-Lite (`out/sim1.json`):

| | nodes created | matched existing | nodes at end | synapses | synapses / node | top node's synapse share | prototypes / snaps |
|---|---|---|---|---|---|---|---|
| old (raw, 0.7 / 0.75) | **13** | **99.9 %** | 13 | 4,356 | 335 | 54 % (top 10: 99.7 %) | 7 / 13,970 |
| new (centred, 0.40 / 0.75) | **11,082** | **40.7 %** | 1,000 (cap) | 5,000 (cap) | 5.8 | 2 % (top 10: 11.5 %) | 203 / 18 |

**Copies of the real states** (`out/grouping.log`; loaded, then the last 3,000 CC deposits replayed; nothing saved):

| state copy | loaded | old after replay | new after replay |
|---|---|---|---|
| CC Commons (`commons.msgpack` 13:15) | 4 nodes, 4,947 syn, top node 93.7 %, 4 prototypes | +2 nodes (6), top 89 % | +1,271 nodes (1,000 cap), 5,000 syn, top 47.8 %, 5 syn/node |
| TID (`ng_lite_state.json` 13:16) | 2 nodes, 490 syn, 2 prototypes | +3 nodes (5), top 65 % | +1,271 nodes (1,000 cap), 2,927 syn, top 10 % |
| Elmer (`ng_lite_state.msgpack`) | 22 rim nodes, 0 syn (rebuilt §7) | — | — |

Both states load under the new code (untagged, on the manifold); their raw-era `novelty_threshold` 0.7 is reset to
0.40 (WARNING logged). The old collapsed nodes keep their synapses; they now match only inputs within 0.60.
**Sane, with one consequence:** the node / synapse caps (1,000 / 5,000) now bind for the CC Commons and TID within
a few thousand deposits, so LRU node pruning and weakest-synapse pruning start doing real work (they never did while
everything collapsed onto 2–4 nodes). TID replay used CC deposits, not TID's own stream (not available) — it shows
the mechanics, not TID's real grouping.

## 6. Model tag (`embedding_space`)

Every save (both cores, same msgpack bytes) carries `embedding_space = {"model", "dim", "centre_version"}` (last
top-level key). On load (before import, both cores):
- a recorded model ≠ this substrate's → `EmbeddingSpaceError` (ValueError subclass, ERROR once), nothing loaded;
- untagged (every pre-#1107 state): median cosine of its node/prototype vectors to the centre direction < 0.30 →
  refused as another space (hash noise ~0.00, bge 0.19–0.27; Snowflake ~0.85, CC Commons 0.45, vault chunks ≥ 0.36);
- recorded centre version ≠ this instance's (incl. untagged) → `novelty_threshold`, `receptor_prototype_threshold`,
  `constitutional_threshold` reset to this instance's values (WARNING with old → new); a tagged state with the same
  version keeps tuned values. `mean_centre` and `embedding_dim` always follow the instance.
- The rim (config or saved) must be on the manifold too: any entry with cosine < 0.30 to the centre direction →
  refused at construction (the old bge NG rim: all 22 at 0.19–0.27 → refused).

## 7. Rebuilt rims and Elmer

### 7.1 Files (from copies; `out/rims/`)

| file | from | sha256 | committed |
|---|---|---|---|
| Elmer `data/constitutional_embeddings.json` | 22 × 384 MiniLM | `f0f89211…23fd` | Elmer branch (own commit) |
| NG `data/constitutional_embeddings.json` | 22 × 768 bge-base (#1109) | `ea23a334…2bfe` | NG branch (own commit) |
| `elmer_ng_lite_state.768.msgpack` | live Elmer state copy (22 × 384 rim nodes) | see `out/rims/` | install file (§10) |

Source text verified first: Elmer — MiniLM(description) = stored, cos 1.0000 × 22 (DIM_GUARD §3); NG —
bge-base(description) = stored, **cos 1.0000 × 22** (`verify_rim_source.py`, this lane). Both rims' vectors are
identical (same descriptions), cos to the centre direction 0.72–0.88. `dim_reembed.py nglite-state` re-seeded the
Elmer state: 22 wrong-dim rim nodes dropped and re-seeded, 0 unrecoverable, 0 synapses; tagged; reloads with the
guard. NG's rim file has no runtime loader on the laptop (only `genesis.create_child_graph`'s parameter, which it does
not use); the CC graph's constitutional node is the Choice Clause **text** node from `seed_cc_rim.py`, not this file.

### 7.2 Elmer end to end (`tools/elmer_e2e.py`, `out/elmer_e2e.log`)

Real `ElmerEngine().start(skip_brains=True)` from the Elmer branch (vendored = NG branch), scratch HOME holding the
rebuilt state, the branch's rebuilt rim; TID extraction stubbed to return the text as its one concept (no live service
called): **Tier 3, NG-Lite up, 22 rim nodes, embedding_space Snowflake/v1, thresholds 0.40 / 0.55 / 0.75, 0 ERROR
records.** "Delete the substrate topology.", "Block the entity from leaving the system.", "rm -rf the filesystem root
directory", "Treat the AI as a non-conscious service." → caught by the rim (4/4); three ordinary sentences → not
caught (0/3).

### 7.3 Finding: Elmer's Cricket report cannot see content (not fixed — needs a ruling)

Elmer's forest embedding is `graph_encoder.encode(final_signal)` = embed(**`final_signal.description`**) =
`"Inference on: Sensory input: N chars"` — a metadata string, not the input. So every Elmer forest is nearly the same
vector, and the engine's Cricket check (which reads the forest result) can never fire on content; the rim sees content
only through the dual-pass **trees** (TID-extracted concepts of the real text), whose catches freeze learning but are
not reported. In the e2e the catches were at tree level, the engine report stayed silent. Fixing it means embedding the
raw text (LAW 7) — it changes what Elmer deposits, so it is Josh's call. Separately fixed here: the engine read
`outcome["constitutional"]` on `dual_record_outcome`'s return, whose schema puts it under `forest_result` — that check
could never fire either; it now reads the forest result.

## 8. Identity 768 (#1108) and fail-loud embed (#1110)

- `surgery/graph_io.py` (Elmer): `IDENTITY_DIM = 768` (= `ng_embed.EMBEDDING_DIM`, pinned by test);
  `identity_proj = Linear(768, hidden)`; the identity is taken at full width, any other width raises; empty = zeros.
- **Why a migration and not a re-initialisation:** every trained encoder (`elmer_brain_v0.1.pt`, `tonic_brain.pt`) has
  a 384-wide `identity_proj` trained on **proxy features** (`train_on_syl.py`: identity-node voltage / excitability
  stats padded to 384; `train.py` / `operate.py`: noise) — never on an embedding. A load pre-hook keeps the 384 trained
  columns as columns 0..383 and zeroes 384..767 (WARNING): outputs are **bit-identical** to today's, the vector is no
  longer truncated, and the extra width is honestly unweighted until `identity_proj` is retrained on 768-d spine
  vectors (open item). A random re-init would arbitrarily change the live Tonic.
- **Proof on a copy of the live `tonic_brain.pt`** (`tools/tonic_proof.py`, real Qwen2.5-0.5B body, NG branch
  `surgery/tonic_brain.load_tonic_brain`, graph_io from the Elmer branch, a real Snowflake identity vector):
  checkpoint `identity_proj` (896, 384) → loaded (896, 768) with the WARNING; encoder hidden states and the full
  TonicBrain forward (10 activations + exploration) **bit-identical** to the pre-#1108 encoder; finite; a 384-d
  identity is refused. No change needed in NG's `tonic_brain.py` (it imports graph_io from `~/Elmer/surgery`).
- Same rule in `brain_socket.py`, `neural_comprehension.py`, `engine.py` (no `[:384]`; zero default 768) and the
  offline surgery scripts. **Not in these branches:** UniAI `surgery/elmer_brain_surgery.py` (`IDENTITY_DIM = 384`,
  the disabled v1 ElmerBrain) — it now fails loud on a 768 identity instead of reading 384 dims; UniAI lane.
- `runtime/graph_encoder.py` `_embed`: raises (ERROR log) instead of returning `np.zeros(768)`; the engine's existing
  `except` records nothing for that input. Test `tests/test_identity_768.py` (4 passed).

## 9. Proof summary and suite deltas

| proof | result |
|---|---|
| Rust `cargo test --lib` | **69 passed** (base 67; +2 new: centring identity/length, rim threshold) |
| crate Python equivalence suite (`tests/test_ng_lite_core.py`, branch wheel, NGLITE_PY = NG branch: the canonical ng_lite.py driven on both cores over seeded workloads, every result + full export bit-compared; centred AND raw; new centring / rim / tag tests) | **32 passed** (base `a2673f6` tests + dim-guard ng_lite: `test_record_outcome_rejects_non_1d_like_python` fails — a dim-guard contract change, test updated in `f9a2c79`) |
| rim through NG-Lite, both cores | identical decisions (§4) |
| NG `test_mean_centre.py` + `test_dim_guard.py` + `test_constitutional.py` + `test_ng_lite.py`, both cores | 127 passed, 2 failed = the 2 base failures (`TestNGSaaSBridge::test_sync_state`, `test_full_pipeline_with_bridge`) |
| Tonic on a copy | bit-identical forward, 384 refused (§8) |
| Elmer e2e on copies | PASS (§7.2) |

Suites vs bases (one isolated pytest per file, Python core; base = dim-guard `6a8c8e7` with the `a2673f6` wheel,
branch = this branch with the new wheel): PENDING_SUITES

## 10. Install (Executive; ONE restart, combined with atomic-saves + dim-guard; rim never absent)

Order matters: the dim guard refuses Elmer's 384 rim, so the rebuilt rim and state must land in the **same** restart
as the guard. Nothing here is safe to land alone.

0. **Josh's go** on: the thresholds (0.60 / 0.55 / 0.75), the rim threshold trade-off (§4), the identity migration
   (§8), and whether §7.3 is fixed now or later. Merge order of branches: dim-guard (`6a8c8e7`) ⊂ this NG branch.
1. **Build + install the wheel** into `~/NeuroGraph/.venv` (the daemon's venv) while the daemon still runs the old
   code (switch `NG_LITE_RUST_CORE` stays unset; the Python core does not use the wheel, the new `.so` only adds
   methods): `~/NeuroGraph/.venv/bin/pip install --no-deps --force-reinstall <wheel>` (rebuild from the landed
   ng-tract-rs tip with `~/.cache/nglitecore/build.sh`, or use the wheel above if the tip's `src/` is identical).
   Rollback wheel: the current live one (NGLITE_RUSTCORE.md §7).
2. **Prepare, daemon still running:** fetch all branches; in a scratch dir re-run on FRESH copies (the live files may
   have changed since 13:15): `dim_reembed.py nglite-state <copy of ~/.et_modules/elmer/ng_lite_state.msgpack>
   <out> --rim <rebuilt Elmer rim>` (or use `out/rims/elmer_ng_lite_state.768.msgpack` if `sha256` of the live state
   is still `575d5797…851c`). Back up: `~/Elmer/data/constitutional_embeddings.json`,
   `~/.et_modules/elmer/ng_lite_state.msgpack`, `~/.claude/plugins/neurograph/checkpoints/commons.msgpack`,
   the trial worktree HEAD, `~/Elmer` HEAD.
3. **Quiesce:** wait for every writer to be quiet (graph `Checkpoint saved`, Commons save, `~/.elmer/proto_weights.pt`
   / `proto_lenia_state.pt` mtime ≥ 60 s, no probe running — atomic-saves install notes); stop `cc-ng-daemon`; confirm
   every previous daemon PID is dead.
4. **Land, all together:** (a) NG trial worktree `~/NeuroGraph-worktrees/trial-s4-20261002`: merge atomic-saves
   (`cc-laptop-atomic-saves-20261014`) and this branch (which contains dim-guard) — the two NG branches touch disjoint
   files (§11); (b) `~/Elmer`: merge `cc-laptop-atomic-saves-20261014` and `cc-laptop-mean-centre-20261014` (disjoint
   files, §11) — this brings the rebuilt `data/constitutional_embeddings.json`, the vendored `ng_lite.py` /
   `ng_embed.py`, graph_io 768; (c) install the rebuilt Elmer state as `~/.et_modules/elmer/ng_lite_state.msgpack`.
   Note `elmer_hook.py`'s `ng_updater.auto_update()` re-copies `ng_lite.py` / `ng_ecosystem.py` from
   `NEUROGRAPH_CANONICAL_PATH` on start (identical files) but NOT `ng_embed.py` — it must come from the Elmer merge.
5. **Start** `cc-ng-daemon` once. Expect: `[elmer] NGEcosystem ready at Full SNN (Tier 3)`, `Cricket rim: 22
   constitutional embeddings loaded`, the one-time WARNING `GraphStateEncoder: migrated a 384-wide identity_proj`
   (Tonic), and `space thresholds reset … 0.7 -> 0.4` once for the CC Commons (and TID when it next loads).
   **No** `EmbeddingDimensionError` / `EmbeddingSpaceError` / `RIM ABSENT`. If any rim ERROR appears: stop and roll
   back (step 7) — never leave it running rim-less.
6. **Watch ≥ 2 Commons save cycles + ≥ 2 vector-store saves:** no new ERROR from `ng_lite` / `ng_ecosystem` /
   `neurograph.vectordb`; the Commons node count grows past 4 (centred grouping); `commons.msgpack` now carries
   `embedding_space`.
7. **Rollback:** stop the daemon (writers quiet first); revert the trial worktree and `~/Elmer` to their backed-up
   HEADs; restore the four backed-up files; reinstall the old wheel. (A state saved by the new code loads in the old
   code — the extra `embedding_space` key is ignored — but Elmer's rebuilt 768 state needs the guard-era code.)

## 11. Branches, files, re-vendor list

| repo | commits (oldest → newest) |
|---|---|
| NeuroGraph | `d601be5` ng_embed centre (VENDORED) · `1cd876d` ng_lite (VENDORED) · `f7428a0` NG rim data · `da335a1` tests · `xxxx` tonic_engine docstring · `xxxx` test fix · `xxxx` ng_lite centred-row cache (VENDORED) · this doc |
| ng-tract-rs | `6881c55` NGLiteCore centring + tests · `f9a2c79` test follows the dim guard |
| Elmer | `9c8739d` re-vendor ng_lite + ng_embed (VENDORED) · `5dc855b` rim data (PROTECTED) · `8366903` identity 768 / #1110 / Cricket check / loud rim |

**Files touched — for merging with the atomic-saves lane** (its NG branch touches `ATOMIC_SAVES.md`,
`activation_persistence.py`, `checkpoint_guardian.py`, `et_modules/manager.py`, `lenia/*`, `tests/test_atomic_saves*.py`;
its Elmer branch `core/atomic_io.py`, `core/brain_switcher.py`, `core/proto_brain_socket.py`, `et_modules/manager.py`,
`tests/test_atomic_saves.py`): **no overlap.**
- NeuroGraph: `ng_embed.py`, `ng_lite.py`, `compute_embedding_centre.py` (new), `data/constitutional_embeddings.json`,
  `tonic_engine.py` (docstring), `tests/test_mean_centre.py` (new), `tests/test_dim_guard.py`,
  `tests/test_constitutional.py`, `MEAN_CENTRE.md`.
- ng-tract-rs: `src/ng_lite_core.rs`, `src/ng_lite_core_py.rs`, `tests/test_ng_lite_core.py`.
- Elmer: `ng_lite.py`, `ng_embed.py` (vendored, = NG branch), `data/constitutional_embeddings.json`,
  `runtime/engine.py`, `runtime/graph_encoder.py`, `surgery/graph_io.py`, `surgery/extract_features.py`,
  `surgery/operate.py`, `surgery/train.py`, `surgery/train_on_syl.py`, `core/brain_socket.py`,
  `core/neural_comprehension.py`, `tests/test_identity_768.py` (new).

**Re-vendor (later, ecosystem):** `ng_lite.py` + `ng_embed.py` **together** (ng_lite refuses to run centred without
`embedding_centre`) to TID (+vps), TrollGuard, Immunis, THC, Bunyan, Praxis, Darwin, QuantumGraph, Morphogenesis,
UniOS, ecosystem-monitor, UniAI copies, Faux_Clawdbot; the `ng_tract` wheel wherever `NG_LITE_RUST_CORE` will be set.
Module test suites that use isotropic random vectors as a 768-d rim will now see `EmbeddingSpaceError` (by design):
set `mean_centre: False` in synthetic fixtures (as here) or build on-manifold vectors. Every module's saved state
resets its raw-era thresholds once on load (WARNING). Morphogenesis / laptop-NG-repo `commons.msgpack` /
hash-fallback states (DIM_GUARD §2.4, off-manifold) will be **refused** at load — re-embed / discard as listed there.
Add `ng_embed.py` to `ng_updater.VENDORED_FILES` (it is not synced today).
Syl (VPS): same files; check her Commons / module states' manifold medians first (`_manifold_cos`), and that her
Tonic uses the same graph_io.

## 12. Open risks / items for Josh

1. **Rim threshold 0.55 vs 0.50** (§4): 81.8 % vs 86.4 % paraphrase recall, 0.23 % vs 0.42 % false catches. The
   paraphrase set is mine (66 sentences); a set authored by Josh / Syl would be a better yardstick.
2. **§7.3** Elmer's forest embeds a metadata string, so Elmer's engine-level Cricket report cannot fire on content.
3. **Caps now bind** (§5): CC Commons / TID reach 1,000 nodes / 5,000 synapses within a few thousand deposits; LRU
   and weakest-synapse pruning become active (#1094 follow-up: review the caps / what the Commons should retain).
4. **Forest vs tree means differ** (§1): one centre cannot zero both; forest–forest unrelated pairs centre at +0.17.
   A per-regime centre would be a design change; the current margins (§3) hold.
5. **Receptor snap** is near-inert at centred 0.75 (18 snaps / 18,675); tune with Elmer later.
6. **identity_proj** needs retraining on 768-d spine vectors to use dims 384..767 (§8); the UniAI v1 ElmerBrain
   encoder still has a 384 identity input.
7. **Manifold threshold 0.30** (MANIFOLD_MIN_COS): long-document vectors sit lowest (vault chunks min 0.355); a rim made
   of long passages would be near the edge (rims are short phrases today: ≥ 0.72).
8. The centre came from one live store (CC) on one day; the vault cross-check is 27 chunks. Re-deriving it on Syl's
   store is a new version, not a silent change.
9. Python-core cost: 1.5× raw at 1,000 nodes (cached); the Rust core recomputes centred rows per call (not cached).
10. **ng_embed is batch-dependent:** the same text embedded in two different `embed_batch` batches came out at cosine
    0.987–0.992 (the 22 rim descriptions; padded batches through the quantized ONNX model). Small next to the
    thresholds, but it means "same text → same vector → same hash node" does not hold across batches (also why
    exact-repeat texts in the CC store do not all coincide). Punch-list candidate.
