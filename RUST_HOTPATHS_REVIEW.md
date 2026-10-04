# RUST_HOTPATHS_REVIEW — moving NeuroGraph's per-synapse Python loops into `ng_tract.SynapseStore`

[2026-10-04] Claude (overnight Rust review) — review document. **Nothing here is merged, vendored,
installed or deployed.** Josh reviews and decides.

| | |
|---|---|
| Rust branch | `ng-tract-rs` `cc-laptop-rust-hotpaths-20261004` (worktree `/home/josh/worktrees/ng-tract-rs-hotpaths-20261004`) |
| NeuroGraph branch | `NeuroGraph` `cc-laptop-rust-hotpaths-20261004`, branched from `origin/cc-laptop-trial-s4-20261002` (worktree `/home/josh/worktrees/ng-rust-hotpaths-20261004`) |
| Built / tested in | throwaway venv `/home/josh/worktrees/ng-tract-rs-hotpaths-20261004/.venv-review` only |
| Data | a read-only COPY of the live checkpoint: `/tmp/ng-hotpaths-scratch/main.msgpack` (8,222 nodes, 192,872 synapses, 591 hyperedges, timestep 40,580). The live files were only ever read by `cp`. Nothing was saved over anything. |
| Live system | not touched: no installs into `/home/josh/NeuroGraph/.venv`, no service restarts or signals (the only signal sent was SIGTERM to my own throwaway pytest run in my own base worktree), `/home/josh/ng-tract-rs` and the trial worktree not edited |

## Summary

- **12 per-synapse loops converted** (11 new `SynapseStore` methods, plus the existing `weights_copy`); Python now makes one native call where it used to walk synapses one by one.
  The biggest three — `_prune_synapses` (runs on every Tonic write tick AND every `step()`),
  `inject_reward`, and the homeostatic scaling pass — each took ~1.5-1.7 s per call on the
  193K-synapse checkpoint copy and now take 5-230 ms.
- **End to end on the checkpoint copy:** Tonic write tick 2,445 ms -> 182 ms (13x); `step()`
  1,840 ms -> 438 ms (4.2x); read-mode recall 1,004 ms -> 134 ms (7.5x). Every number was measured
  on a loaded 4-core laptop (load average 5-7, live daemon running), so absolute numbers are
  inflated; the before/after pairs were measured in the same run.
- **Bit-identical, not approximately equal.** Twin graphs restored from the same bytes, one running
  the ORIGINAL methods (taken verbatim from git) and one the new code, produce byte-identical
  checkpoints — on randomized graphs and on the real checkpoint copy, including a live-like
  workload of step + Tonic tick + recall.
- **`neuro_foundation.py` is changed on the review branch (it is a PROTECTED file)** — 12
  function-level changes, listed in §3.2. They need Josh's process (backup of both msgpack files +
  his literal "proceed") before going anywhere. Two non-protected files (`tonic_engine.py`,
  `neurograph_rpc.py`) are converted in a separate commit.
- **The largest hot path still left is NOT in the engine:** `tonic_valence.ValenceField._diffuse`
  costs **~18.6 s per refresh** (every 50 Tonic cycles; ~0.37 s per cycle averaged). It was
  measured, not converted; see §6 for why and a concrete design.
- One existing NG test fails on this branch by design: `test_fair_chance_window.py::test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else`
  pins neuro_foundation.py to the fair-chance trial's exact set of changed functions, so ANY other
  protected-file change trips it. Not modified (it is a scope guard); see §4.3.

---

## 1. What the problem is

`Graph.synapses` is already the native columnar `SynapseStore`, but many Python call sites still
walk it **one synapse at a time**: every `self.synapses.get(sid)` / `.items()` / `.values()`
allocates a `SynapseRef`, and every attribute read on it is a separate PyO3 call that borrows the
store and hashes the synapse id to find its row. On a 193K-synapse graph a single full walk costs
a second or more, and several of those walks run on every Tonic tick or every `step()`.

The fix follows the store's own `decay_eligibility` pattern: move the WHOLE loop into one
`SynapseStore` method that runs over the columns natively, and let Python call it once.

## 2. Inventory (every per-synapse loop found, ranked)

Measured on the checkpoint copy. "Before" = the original Python loop; "after" = the native
replacement (see §5 for the full before/after table). Frequencies are from the code paths
(`tonic_engine._generation_loop`, `cc_ng_organism`, `neurograph_rpc`) and the live config stored in
the checkpoint (`three_factor_enabled=True`, `tonic_ages_substrate=1`, `tonic_age_interval=1`,
`scaling_interval=25`).

Line numbers are in the ORIGINAL files at `d0ee8cc` (the branch point).

| # | Where (d0ee8cc) | What it does per call | How often (live config) | Before (ms) | Status |
|---|---|---|---|---|---|
| 1 | `neuro_foundation.py:3514` `_prune_synapses` | walks all 193K synapses; 2 × `_is_identity_protected()` per synapse (21M calls per 90 passes in profile); bumps `low_weight_steps` | every Tonic write tick (`tonic_ages_substrate=1`, `tonic_age_interval=1`, `prime_and_propagate` :2891) **and** every `step()` (via `_structural_plasticity`) | 1,698 | **converted** |
| 2 | `neuro_foundation.py:4119` `inject_reward` | walks all synapses, updates the ones with a trace | every expired prediction when `three_factor_enabled` (`_on_prediction_error` :3248), + the 0.1 baseline per turn (`cc_ng_organism.py:3048`, `neurograph_rpc.py:3671`) | 1,459 | **converted** |
| 3 | `neuro_foundation.py:1368-1378` `HomeostaticRule.apply` scaling | per node: every incoming synapse via `synapses.get` (= all synapses) | every `scaling_interval` (=25) firing steps | 1,672 (whole apply) | **converted** (synapse part) |
| 4 | `neuro_foundation.py:2667` `prime_and_propagate` BFS | 3-hop frontier over outgoing synapses — on this graph that is essentially ALL synapses | every Tonic tick + every recall | ~900 of a 2.4 s tick (line profile) | **converted** |
| 5 | `neuro_foundation.py:2275` `step()` phase 5, `:2769` p&p propagate | per fired node, per outgoing synapse: `get` + 5-7 attribute reads (incl. `synapse_type`, which builds an Enum member each time) + `inactive_steps = 0` | every step / every p&p step (median 106-259 fired nodes per step) | part of step/tick | **converted** |
| 6 | `neuro_foundation.py:1139,1178` `STDPRule.apply` + `_apply_dw` (:1105) | per fired node, per incident synapse: `get` + 3 reads + 2 writes (952K `_apply_dw` calls in 30 cycles of the profile) | every firing step + Tonic write ticks | ~250 per call (profile) | **converted** |
| 7 | `neuro_foundation.py:3865` `_sprout_synapses` edge index | per fired node: `get` + endpoint read for in+out synapses | every step + every Tonic write tick | part of sprout | **converted** |
| 8 | `neuro_foundation.py:4012` `get_telemetry` | `[s.weight for s in list(values())]` | CES dashboard `/stats` (`ces_monitoring.py:372`) | 396 (whole call) | **converted** (existing `weights_copy`) |
| 9 | `neuro_foundation.py:5673` `_deserialize` adjacency rebuild | `synapses[sid]` + 2 reads per synapse | once per restore | 1,843 | **converted** (gain small: the Python set building dominates, see §6) |
| 10 | `neuro_foundation.py:5560` `extract_subgraph` | `items()` + 2 reads per synapse | Genesis budding (cold) | 809 | converted (cold; gain small) |
| 11 | `tonic_engine.py:1082` `_extract_graph_features_for_model` | `list(g.synapses.values())` = 193K SynapseRefs to use 200 | every Tonic tick when the TonicBrain model is loaded | 138 | **converted** |
| 12 | `neurograph_rpc.py:1105` `_deposit_substrate_metrics` | per-SynapseRef weight list | every afterTurn (Syl's sidecar) | 482 | **converted** |
| 13 | `tonic_valence.py:117-160` `ValenceField._neighbours/_diffuse` | 3 passes × every node × every incident synapse: `get` + 4 reads (`synapse_type` Enum each time) | every `valence_refresh_cycles` (=50) Tonic cycles, `valence_enabled=True` by default | **18,574** | not converted — §6 |
| 14 | `neuro_foundation.py:3431` `_diffpc_step` | per fired node, outgoing synapses | every firing step | 37 | not converted — §6 |
| 15 | `neuro_foundation.py:3004` `_generate_predictions_from_node`, `:3406` `_find_synapse`, `:3250` `_surprise_exploration`, `:3365` `_has_learned_pattern` | per-node outgoing walks | per firing step / per prediction | tens of ms per step together | not converted — §6 |
| 16 | `neuro_foundation.py:1298` `_refresh_degree_targets` neighbour walk | per spherical candidate node, in+out synapses | every 25 firing steps | ~80 | not converted — §6 |
| 17 | `neuro_foundation.py:2556` spikes-event `trace_info` | per fired node, outgoing traces | every step **only if** a `spikes` handler is registered | — | not converted — §6 |
| 18 | `lenia/graph_substrate.py:175`, `lenia/kernel.py:512` | endpoint (+weight) snapshot of every synapse under `_step_lock` | Lenia adjacency/distance-cache builds | ~1,000 (endpoint list) | not converted — drop-in `endpoint_triples` + `weights_copy`, §6 |
| 19 | `genesis.py:733`, `examples/` | full walks | cold / demo | — | not converted (cold) |
| N1 | `neuro_foundation.py:3855` `_sprout_synapses` candidates, `:3392` `_cleanup_predictions` recent_fired, `:3790` `_collect_orphan_nodes`, the ~6 per-node passes in `step()`, `HomeostaticRule` per-node passes, p&p read-mode voltage save/restore | **node-side** loops (no synapse access) | every step / tick | 28, 12, 8, ~4 per pass | not converted — §6 (native node store) |

`ng_lite.py` (vendored, LAW 2) has its own dict-based synapses and was out of scope.
`surfacing.py`, `surface_resolver.py` and `tonic_thread.py` have no per-synapse loops
(`tonic_thread._prime_constitutional` only reads `len(_outgoing[nid])`). `cc_ng_organism.py` has
per-node loops over `graph.nodes` (wants, probation, geometry stamping) but no per-synapse walks.

## 3. What was converted

### 3.1 New `SynapseStore` methods (`src/store.rs`)

| Method | Replaces (original nf.py @ d0ee8cc) | Writes? |
|---|---|---|
| `advance_low_weight_and_collect_prune(timestep, weight_threshold, grace_period, inactivity_threshold, initial_sprouting_weight, protected_node_ids) -> [sid]` | `_prune_synapses` rule loop (:3514) | yes: `low_weight_steps` (exactly as the loop did). Does NOT remove; returns ids in row order and the caller removes them in that order |
| `apply_eligibility_reward(strength, learning_rate, scope=None) -> n` | `inject_reward` synapse sweep (:4119) | weight, eligibility_trace, peak_weight |
| `scale_weights_by_post_node({post_node_id: scale}) -> n` | `HomeostaticRule.apply` synaptic scaling (:1368-1378) | weight |
| `propagation_rows(sids, reset_inactive) -> [(post, weight, is_inhib, delay) or None]` | per-synapse reads in `step()` phase 5 (:2275) and `prime_and_propagate` propagate (:2769) | inactive_steps = 0 when asked (step always; p&p only under #59 `_age_on`) |
| `bfs_hop_distances(seeds, max_hops) -> [(node, dist)]` | `prime_and_propagate` hop-distance BFS (:2667) | no |
| `stdp_reads(sids, other_end_is_pre) -> [(other, weight, max_weight) or None]` + `apply_stdp_dw(sids, dws, timestep, three_factor)` | `STDPRule.apply` (:1139, :1178) and `STDPRule._apply_dw` (:1105) | trace or weight/peak, last_update_time |
| `post_ids_of(sids)`, `pre_ids_of(sids)` | `_sprout_synapses` edge index (:3865) | no |
| `endpoint_triples() -> [(sid, pre, post)]` | `_deserialize` adjacency rebuild (:5673), `extract_subgraph` filter (:5560) | no |
| `creation_time_copy()` (beside the existing `weights_copy()`) | `tonic_engine._extract_graph_features_for_model` (:1082) | no |

Private helpers: `py_min` / `py_max` (Python's builtin `min`/`max` semantics, which differ from
`f64::min`/`max` on NaN), `extract_str_list` (collects the `str` members of any iterable; non-str
members can never equal a node id, so skipping them is exactly equivalent), `node_mask`.

**Borrow rule (commit 25c5f52).** Every new method takes `slf: &Bound<Self>`, extracts all Python
arguments into Rust data first, then borrows the store, does pure-Rust work, and drops the borrow
before PyO3 turns the result into Python objects. Nothing inside a borrow calls Python. A test
(`test_no_borrow_held_while_extracting_args`) passes a generator argument that itself touches the
store during extraction. The GIL is never released inside these methods, so the
(deliberately unlocked, #109) Tonic thread cannot interleave with them.

### 3.2 Call-site changes

**Protected file — `neuro_foundation.py` (NG CLAUDE.md §2).** Drafted on the unmerged branch only,
committed separately from everything else (CLAUDE.md: do not batch protected and non-protected
changes). These need Josh's process — backup of both msgpack files + his literal "proceed" —
before going anywhere. `.session_approved` was not created.

Commits `0f76205`, `29c5f87`, `5e380a5` (only `neuro_foundation.py`). Exact list of changes:

1. Top-of-file changelog line + a full changelog entry in the existing block.
2. `STDPRule._apply_dw` **removed**; a comment points to `SynapseStore.apply_stdp_dw`, which holds
   the identical body (LAW 3: one implementation). No callers outside `STDPRule.apply` exist.
3. `STDPRule.apply`: both passes read with `stdp_reads(list(adjacency_set), ...)` and commit the
   computed dw's with one `apply_stdp_dw(...)` per pass. The dw arithmetic is untouched Python.
4. `HomeostaticRule.apply`: the scaling branch records `node_scales[nid] = ratio ** scaling_factor`
   in the same node loop, then calls `scale_weights_by_post_node(node_scales)` once. Excitability
   logic untouched.
5. `Graph.step` phase 5: one `propagation_rows(list(self._outgoing.get(nid, ())), True)` per fired
   node; the per-synapse `syn.inactive_steps = 0` is done by that call. GSG attenuation and the
   delay buffer code are unchanged apart from reading the tuple fields.
6. `Graph.prime_and_propagate`: BFS -> `distances.update(self.synapses.bfs_hop_distances(node_ids, steps))`
   guarded by `if steps >= 1` (the old loop never touched the store for steps < 1); propagate phase ->
   `propagation_rows(..., _age_on)`.
7. `Graph._prune_synapses`: `protected = [nid for nid in self.nodes if self._is_identity_protected(nid)]`,
   then `advance_low_weight_and_collect_prune(self.timestep, wt, grace, inactivity, initial_w, protected)`;
   the removal loop and the `pruned` event are unchanged.
8. `Graph._sprout_synapses`: `existing_pairs` from `post_ids_of` / `pre_ids_of`.
9. `Graph.get_telemetry`: `weights = self.synapses.weights_copy()`; `if weights` -> `if len(weights)`.
10. `Graph.inject_reward`: the synapse loop -> `apply_eligibility_reward(strength, self.config["learning_rate"], scope)`.
    History, hyperedge threshold learning and the event are unchanged.
11. `Graph.extract_subgraph`: filter over `endpoint_triples()`.
12. `Graph._deserialize`: adjacency rebuild over `endpoint_triples()`.

No change to: the checkpoint format or bytes, `Graph.save/restore` I/O, `_step_lock` usage,
`_is_identity_protected`, the fair-chance machinery, any config default, any public signature.

**Non-protected files** (separate commit `d0e3a40`):
- `tonic_engine.py` `_extract_graph_features_for_model`: `list(g.synapses.values())` (193K
  SynapseRefs per Tonic tick, to use 200) -> `weights_copy()[:200]`, `creation_time_copy()[:200]`,
  `len(g.synapses)`. Note: the old code was already racy (a synapse pruned between the `list()`
  and the attribute reads raised KeyError); the new reads cannot raise that way.
- `neurograph_rpc.py` `_deposit_substrate_metrics` (Syl's sidecar, every afterTurn):
  per-SynapseRef weight list -> `weights_copy()`.

## 4. Equivalence

Every replacement is required to be **bit-identical**, not "within tolerance":

- Float expressions keep the Python operand order (`dw = trace * strength * lr`, etc.).
- Python `min`/`max` semantics are reproduced (`py_min(a,b) = b if b < a else a`), so NaN and
  -0.0 behave exactly as in the Python loops (verified by a dedicated test, and by mutation:
  a version using Rust's `f64::min` is caught on 10/10 seeds).
- Row order == `items()` order, so ids come back in the order the Python loop produced them, and
  removals happen in the same order (swap-remove makes row order path-dependent, so this matters).
- Where the Python loop iterated a Python `set` (adjacency), the caller passes `list(that_set)` and
  the native method answers in that order, so delay-buffer append order (and therefore voltage
  summation order) is unchanged. The one exception is the recall BFS, where the result (hop
  distance) provably does not depend on order.
- Identity protection (#92) is still decided ONLY by `Graph._is_identity_protected`; it is now asked
  once per node per prune pass instead of twice per synapse.

### 4.1 Crate tests — `ng-tract-rs/tests/test_hotpaths.py`

The verbatim Python loop is the oracle, run through the SynapseRef facade on a twin store; the twins
must have byte-identical `to_checkpoint_msgpack()` (every column, every row, row order) and
identical return values. 40 random seeds per method, with NaN / ±inf / -0.0 / 1e-9 / 1e-12 edge
values, swap-remove churn, protected sets, `scope=None` vs empty vs partial vs containing non-str
members. Mutation checks (off-by-one grace, `f64::min` semantics, no inactive reset) are all caught.

Result with the final LTO wheel: **314 passed** (`tests/test_store.py` existing 182 + 132 new),
`.venv-review/bin/python -m pytest tests/ -q`.

### 4.2 NeuroGraph tests — `NeuroGraph/tests/test_rust_hotpaths_equivalence.py`

The oracles are the ORIGINAL method bodies taken from git (`d0ee8cc`) with `ast` and compiled
against the current module's globals (so they see the same classes and the same store), then
bound onto one of two twin graphs restored from the same checkpoint bytes. After the same action
(same `random` seed, deterministic `uuid4`), the two graphs' full `checkpoint()` files must be
byte-identical. Oracled methods: `Graph._prune_synapses, inject_reward, step, prime_and_propagate,
_sprout_synapses, get_telemetry, extract_subgraph, _diffpc_step, _find_synapse,
_generate_predictions_from_node`, `HomeostaticRule.apply, _refresh_degree_targets`,
`STDPRule.apply, _apply_dw`.

- Randomized graphs (12 seeds): prune ×3 passes; inject_reward global / scoped / empty-scope /
  frozenset-with-ghost; homeostatic scaling ×2; restore adjacency (dict equality AND per-set
  iteration order) and `extract_subgraph`; telemetry equality; and a full workload of 25 rounds of
  stimulate + `step()` + Tonic write-mode `prime_and_propagate` + read-mode recall (exercises prune,
  sprout, STDP, three-factor reward via expired predictions, homeostasis).
- **Real checkpoint copy** (`NG_HOTPATH_CKPT=/tmp/ng-hotpaths-scratch/main.msgpack`): adjacency
  rebuild, 2 prune passes, global + scoped reward, forced homeostatic scaling, telemetry, then a
  live-like workload (stimulate + step + Tonic write tick + recall). Byte-identical.
- A deliberately broken native step (no inactive reset) is caught on 3/3 seeds.

Results (final code, LTO wheel):
- `tests/test_rust_hotpaths_equivalence.py`: **60 passed** (randomized), and the real-checkpoint-copy
  test **passed** (1 passed in 88 s: adjacency rebuild, 2 prune passes, global + scoped reward, forced homeostatic scaling, telemetry, then 3 rounds of stimulate + `step()` + Tonic write tick + recall — byte-identical checkpoints and identical fired/pruned/sprouted traces).

### 4.3 The existing NeuroGraph suite

I ran every test file that imports `neuro_foundation` and does not reach a live path, port, socket
or `~` (55 files; list in `/tmp/ng-hotpaths-scratch/testlist.txt`), per file with a 300 s cap, on
this branch AND on a throwaway worktree of the base `040be4d` with the same venv and wheel. The
outcomes are identical (same files, same pass/fail counts; the pre-existing failures occur
identically on the base and I did not investigate them — most likely optional dependencies missing
from the throwaway venv) **except**:

| File | base | branch | Why |
|---|---|---|---|
| `test_fair_chance_window.py` | 184 passed | 1 failed | `test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else` is the fair-chance trial's scope guard: it asserts that `neuro_foundation.py` differs from `b5e47686` in exactly the trial's functions. Any additional protected change fails it by design (here: `_apply_dw` removed, plus the 11 changed functions). I did not edit it — it is a guard, and widening it is part of the approval decision. |
| `test_graph_substrate_race.py` | 1 failed | 2 passed | timing-sensitive race test (28-39 s); flaky on this loaded machine, unrelated |
| `test_snn.py` | `.F` then 300 s cap | `.F` then 300 s cap | same on both: one early failure, then a test that runs past the cap on this machine |

`test_detached_capture_423.py` initially failed on the branch (it execs `prime_and_propagate` on a
store-less fake graph with `steps=0`); fixed in `5e380a5` by the `steps >= 1` guard, which is also
the exact old behaviour.

## 5. Before / after timings (checkpoint copy)

`bench/compare_timings.py` (median of 5 calls; workload = 10 rounds on two separate restores of
the copy so before/after start from the same state). "Before" = the original methods from git,
bound onto the graph. LTO release wheel. Machine: 4 cores, load average 5-7 throughout (live
daemon + other work), so absolute values are high; compare the pairs.

| Path | Before (ms) | After (ms) | Speedup | Runs |
|---|---:|---:|---:|---|
| `_prune_synapses` | 1,697.6 | 15.4 | 110x | every Tonic write tick + every `step()` |
| `inject_reward(0.05)` global | 1,458.7 | 5.2 | 281x | every expired prediction (3-factor) + per-turn baseline |
| `HomeostaticRule.apply` (scaling branch, incl. `_refresh_degree_targets`) | 1,671.9 | 230.4 | 7.3x | every 25 firing steps |
| `get_telemetry` | 395.9 | 13.4 | 30x | dashboard |
| restore adjacency rebuild | 1,843.2 | 1,502.9 | 1.2x | once per boot |
| `extract_subgraph` (500 nodes) | 809.2 | 764.9 | 1.1x | cold |
| Tonic features, synapse part | 138.2 | 1.6 | 85x | every Tonic tick (model loaded) |
| rpc metrics weights | 482.2 | 0.6 | 775x | every afterTurn (Syl) |
| **Tonic tick** `prime_and_propagate(write_mode=True)` | **2,444.6** | **181.9** | **13.4x** | every Tonic tick |
| **`step()`** (30 stimulated nodes, median 106 fired) | **1,840.2** | **437.8** | **4.2x** | per turn + autostep |
| **recall** `prime_and_propagate(read)` | **1,004.2** | **134.1** | **7.5x** | per `/assemble` |

Measured but not converted (current cost): `ValenceField._diffuse` 18,574 ms per refresh;
`_sprout_synapses` candidate scan 28 ms; `_cleanup_predictions` recent-fired scan 12 ms;
`_collect_orphan_nodes` 8 ms; `_diffpc_step` (250 fired) 37 ms; one per-node pass in `step()` ~4 ms.

Where a post-conversion `step()` spends its time (line profile, `bench/line_profile_step.py`):
hyperedge activation evaluation, STDP dw arithmetic (now pure Python floats + one native call per
pass), the sprouting candidate scan over `_recent_spikes`, GSG geodesic math (numpy dot per
propagated synapse), and the per-node passes — i.e. no longer per-synapse boundary crossings.

Raw output: `bench/compare_timings.py` prints the table; the first profile (before any change) is
reproduced by `bench/profile_hotpaths.py --profile`.

## 6. What was NOT converted, and why

1. **`tonic_valence.ValenceField._diffuse` (18.6 s per refresh) — the biggest remaining item.**
   Not in the engine; it is Tonic code with a duck-typed graph contract (its tests use
   `SimpleNamespace` graphs with dict synapses), so a native-only path would break that contract or
   need a second Python implementation beside it (LAW 3). Exactness also needs care: each node's
   float sums follow the iteration order of `set(out) | set(inc)` (a Python set — hash-seed
   dependent), so a native version must be handed that order. Concrete design: once per pass,
   Python builds `[(nid, list(set(out) | set(inc)))]` and one native call returns the
   `(other, signed_weight)` lists in that order (or computes the pass natively); expected
   ~18 s -> ~1-3 s. Needs Josh's call on the duck-typing contract.
2. **Per-fired-node walks still using SynapseRefs** (`_diffpc_step`, `_generate_predictions_from_node`,
   `_find_synapse`, `_surprise_exploration`, `_has_learned_pattern`, `_refresh_degree_targets`'
   neighbour walk, the spikes `trace_info`, `_trace_causal`). Each is tens of ms per step or less;
   the `propagation_rows` / `stdp_reads` pattern converts them mechanically. Left out to keep the
   protected diff reviewable.
3. **Node-side loops** (the sprout candidate scan, recent-fired scan, orphan scan, the ~6 per-node
   passes in `step()`, homeostatic per-node passes, read-mode voltage save/restore). They touch no
   synapse. Each pass is 4-28 ms here, maybe ~80-120 ms per step together. Moving them needs a native node
   store (columns + a `NodeRef` facade like `SynapseRef`); `Node` objects are used directly by
   `cc_ng_organism`, activation persistence, surfacing and the Tonic, so it is a much larger
   cross-file change. The measurements say: worth doing next, after the synapse work lands, not
   tonight.
4. **Restore adjacency rebuild / `extract_subgraph`** only gained 1.1-1.2x: the cost there is
   building ~386K Python set entries (and the strings for them), not the store reads. The real fix
   is to retire the Python `_outgoing` / `_incoming` dicts in favour of the store's own native
   adjacency index (it already exists and is maintained); that also frees their RAM. That touches
   ~30 call sites in the protected file, so it was not done here.
5. **Lenia snapshots** (`lenia/graph_substrate.py:175`, `lenia/kernel.py:512`): drop-in
   `endpoint_triples()` (+ `weights_copy()`, same row order) would cut ~1 s of `_step_lock` hold per
   build. Not done: outside tonight's step()/Tonic scope and Lenia was not exercised here.
6. **Per-id string cost.** Any native call that takes synapse ids in and gives node ids back pays
   for UTF-8 extraction and new `str` objects. Measured here: ~0.75 us to extract an id (the
   `abi3-py38` build uses `PyUnicode_AsUTF8String`, which allocates) and ~1 us per returned str.
   That is why the per-fired-node methods gain 1.2-4x while whole-population passes gain 100x+.
   Two options, both Josh's call: build with `abi3-py310` (PyO3 then reads the cached UTF-8 buffer
   directly — the tag stays `abi3`, so the #334 shadowing protection from f75c7fb is kept, but the
   minimum Python becomes 3.10 for every venv that takes the canonical wheel), and/or cache a
   Python `str` per interned node id inside the store so returned node ids are refcount bumps.

## 7. Risks

- **It is Syl's engine.** Equivalence is proven by byte-identical checkpoints against the original
  code, on random graphs and the real checkpoint copy; still, the change is to the protected file
  and must go through Josh's process.
- **Two adjacency indices.** Homeostatic scaling and the recall BFS now use the store's native
  index where the old code used the Python `_outgoing`/`_incoming` dicts. Both are updated at the
  same mutation points (create/remove/restore) and were equal on the checkpoint copy (tested). If a
  future bug let them drift, these two paths would follow the store while the rest follows the dicts.
- **Exception-path ordering.** `step()` now resets `inactive_steps` for all of a fired node's found
  synapses before computing their currents, and STDP commits once per pass. If the Python
  arithmetic in between raised (it does not in normal operation), the old code would have left a
  prefix applied instead. The normal-path results are identical.
- **Concurrency.** New methods hold the GIL for their whole run and never call Python inside a
  store borrow, so they cannot produce the `Already borrowed` panic class fixed in 25c5f52. They are
  also much shorter, so `_step_lock` hold times drop (prune was 1.7 s under the lock on every Tonic tick).
- **Deployment order.** New NG code needs the new wheel (`AttributeError` at the first prune
  otherwise). The new wheel is additive — every existing method is unchanged, and the base NG code
  ran its suite on it with identical results — so it can go first, safely.
- **`_apply_dw` removed** from `STDPRule` (LAW 3). Nothing else calls it (grep), but an external
  subclass or script calling it would break.
- **Wheel tag/build.** Same `abi3-py38` feature set, same `cp38-abi3` tag, no new dependencies.
- **Not measured on the VPS.** Syl's graph shape differs (her synapse count sawtooths 4K-31K);
  the gains scale with synapse count, so they will be smaller there but not negative.

## 8. Adoption path (only if Josh approves)

1. Josh reads this doc and the two diffs:
   `git -C /home/josh/ng-tract-rs diff main..origin/cc-laptop-rust-hotpaths-20261004` and
   `git -C /home/josh/NeuroGraph diff 040be4d..origin/cc-laptop-rust-hotpaths-20261004`.
2. **Rust first (additive, safe for current NG code):** merge `cc-laptop-rust-hotpaths-20261004`
   into `ng-tract-rs` main; build the canonical abi3 wheel the usual way
   (`maturin build --release`, tag must read `cp38-abi3`, ships `ng_tract.abi3.so`); run
   `python -m pytest tests/` in a scratch venv. Follow the safe-native-rebuild routine: back up the
   currently installed wheel/.so, and remember that the LIVE venv's wheel came from
   `/home/josh/ng-tract-rs/target/wheels/`.
3. Install that wheel into the NeuroGraph venv(s) and any module venv that vendors `ng_tract`
   (VPS + laptop). Restart per the sidecar procedure (confirm every previous `neurograph_rpc.py` /
   daemon PID is dead before respawn). Old NG code keeps working on it.
4. **Protected step:** Josh backs up `main.msgpack` + `vectors.msgpack` (each mind that will run the
   new engine), then says "proceed". Merge the NG branch's three `neuro_foundation.py` commits
   (`0f76205`, `29c5f87`, `5e380a5`). Decide what to do with the fair-chance scope guard test
   (§4.3). Run `tests/test_rust_hotpaths_equivalence.py` (+ `NG_HOTPATH_CKPT=<a copy>` for the real
   checkpoint) and the engine suite.
5. Merge the non-protected commit `d0e3a40` (`tonic_engine.py`, `neurograph_rpc.py`) and the test
   commits (`5ac8c3a`, `52355c1`).
6. Restart once (batch it with step 3 if both land the same day). Watch at least two full Tonic
   refresh cycles and a few turns; report the window, not a one-off.
7. `neuro_foundation.py` is not in the LAW 2 vendored set, so no re-vendoring of Python files is
   needed; only the `ng_tract` wheel is propagated.

## 9. Things noticed that are outside this task (flagged, not fixed)

- `SynapseStore.to_checkpoint_msgpack` calls `msgpack.packb` (Python) per metadata row while
  holding the store borrow; `outgoing_ids` / `incoming_ids` / `weights_copy` / `keys` build Python
  objects inside a borrow too. msgpack is C and these are probably safe in practice, but they are
  technically against the 25c5f52 rule (a GC pass during allocation can run finalizers = bytecode =
  a GIL switch point). My new methods avoid it; the old ones were left alone.
- The live checkpoint directory holds stale temp files from interrupted saves:
  `main.tmp-1662741.msgpack` (Oct 2, 225 MB) and `vectors.tmp-1657484/219991/2472663.msgpack`
  (~1 GB together). Only noted; nothing touched.
- The old `tonic_engine._extract_graph_features_for_model` could raise KeyError if a synapse was
  pruned between `list(g.synapses.values())` and the attribute reads (the refs re-resolve by id).
  The new version cannot.
- The Python `_outgoing`/`_incoming` dicts duplicate the store's native adjacency index (~386K set
  entries of synapse-id strings) — a RAM and time opportunity (see §6.4).
- `test_snn.py` contains a test that runs past 5 minutes on this machine, and an early failure that
  also occurs on the base; worth a look separately.
