# RUST_HOTPATHS_ONTO_S4 — the overnight native hot-path conversion, rebased onto trial s4

[2026-10-05] Claude (lane rust-hotpaths-onto-s4) — review document. **REVIEW branch only: nothing merged,
installed, restarted or deployed.** The Executive reviews and integrates.

| | |
|---|---|
| Branch | `cc-laptop-rust-hotpaths-onto-s4-20261005` (worktree `/home/josh/worktrees/ng-rust-hotpaths-onto-s4-20261005`) |
| Base | `origin/cc-laptop-trial-s4-20261002` @ `5246b63` (the trial tip when this lane started) |
| Source lane | `origin/cc-laptop-rust-hotpaths-20261004` (built on `040be4d`): protected `0f76205` `29c5f87` `5e380a5`, non-protected `d0e3a40`, tests `5ac8c3a` `52355c1`, doc `8d648c1` (`RUST_HOTPATHS_REVIEW.md`, read end to end) |
| Approval | Josh 2026-10-05: "you have my approval for the overnight Rust review as soon as resources permit it" — covers the protected `neuro_foundation.py` changes of THIS conversion |
| Native methods | already in the canonical wheel (ng-tract-rs `cc-laptop-canonical-wheel-20261004` @ `79be810`, wheel sha256 `8d5c4322…`). No Rust change, no wheel built or installed |
| Test interpreter | throwaway venv `/tmp/rhs4/venv` (system python 3.12.3, `pip install pytest` only) + a `.pth` pointing READ-ONLY at `/home/josh/NeuroGraph/.venv/lib/python3.12/site-packages` for numpy / msgpack / torch / `ng_tract`. That venv's `ng_tract.abi3.so` sha256 `9179f2dc…` == the `.so` inside the canonical wheel file (checked). Nothing was installed into the NG venv; `PYTHONDONTWRITEBYTECODE=1` throughout |
| Data | a read-only COPY of the live CC checkpoint: `/tmp/rhs4/ckpt/main.msgpack` (sha256 `3c2b6995…`, 251,948,729 bytes, chmod 444; 10,158 nodes, **67,968 synapses**, 891 hyperedges, timestep 60,452). The live file was only read by `cp` |

## Commits (on top of `5246b63`)

| Commit | Files | Kind |
|---|---|---|
| `1570a18` | `neuro_foundation.py` only | **PROTECTED** — the conversion (own commit) |
| `d1fc5dd` | `neuro_foundation.py` only | **PROTECTED** — keeps `STDPRule._apply_dw` byte-identical to the trial (a docstring note moved to a comment). The suites below ran at `7a3e372`+tests, before this docstring-only commit; the equivalence file + `test_stdp.py` were re-run after it (323 passed, 1 skipped) |
| `d391856` | `tonic_engine.py`, `neurograph_rpc.py` | non-protected sites (overnight `d0e3a40`) |
| `16f3365`, `a74012f` | `tests/test_rust_hotpaths_onto_s4.py`, `tests/rust_hotpaths_onto_s4_timings.py` | equivalence tests vs the trial tip; timing script |
| `7a3e372` | `tests/test_rust_hotpaths_equivalence.py` | the overnight lane's test, carried over (d0ee8cc oracles) |
| (this doc) | `RUST_HOTPATHS_ONTO_S4.md` | doc |

No other protected file and no vendored file changed (`git diff 5246b63 --stat` lists only the files above).

## How it was ported

`0f76205` conflicted in `_prune_synapses` (the trial rewrote it: want-hub keyword params, lifeline
pruning, last-link grace) and in the changelog block; `29c5f87` and `5e380a5` then applied cleanly.
The 11 non-prune sites are **byte-for-byte the overnight lane's edits** (checked: the +/- lines of
`git diff 040be4d 5e380a5` and of this port are identical apart from `_prune_synapses`). On top of that,
two deliberate additions:

1. **Python fallback at every site** (brief: "keep a Python fallback when ng_tract lacks a method"; same
   `getattr(self.synapses, name, None)` pattern the trial already uses for `normalize_strength` /
   `scale_all`). The fallbacks are the trial's ORIGINAL loops, reshaped as module helpers that return what
   the native method returns (`_stdp_reads_python`, `_apply_stdp_dw_python`,
   `_scale_weights_by_post_node_python`, `_propagation_rows_python`, `_bfs_hop_distances_python`,
   `_endpoint_ids_python`, `_endpoint_triples_python`, `_apply_eligibility_reward_python`). They are also
   proven bit-identical to the trial tip (below), via a store proxy that hides the native methods.
2. **`STDPRule._apply_dw` is kept** (the overnight lane deleted it). It is the fallback commit, so there is
   still exactly one Python body of it, and any external caller keeps working.

## Per site

"Equivalence" = twin graphs restored from the same checkpoint bytes; one runs the trial tip's WHOLE
`neuro_foundation.py` (read from git at `5246b63`, imported as a separate module), the other this branch;
same `random` seed, deterministic `uuid4`; full checkpoint bytes and every return value / trace must be
identical. "fallback" = the same with the native methods hidden. 8 random seeds unless noted
(120 nodes, ~810 synapses after swap-remove churn, NaN-free edge values, ±0.0, 1e-10 traces, pre-existing
last-link stamps, constitutional / `*_authored` / `*_emergent` nodes).

| # | Site | Status | Why | Equivalence evidence |
|---|---|---|---|---|
| 1 | `_prune_synapses` default path, lifeline **off** (also absent key) | **converted** — `advance_low_weight_and_collect_prune(..., protected)`, protected = `[nid for nid in self.nodes if self._is_identity_protected(nid)]` | the rules are unchanged on the trial; `_is_identity_protected` is a pure read and False for a non-node id | `test_prune_site` × {absent, off, budget} × native/fallback: 6 passes each, timestep advancing, `report=` on alternate passes |
| 2 | `_prune_synapses` default path, lifeline **on** (`prune_protected_faint_links`) | **converted (hybrid)** — the same native sweep with NO protected nodes; then each lifeline gets back its pre-pass `low_weight_steps` and the lifelines are filtered out of the result (order kept); the last-link stamps are read from synapse metadata with the SAME per-synapse `items()` + `.metadata` walk the loop did; `_last_link_grace` untouched | the trial loop skips only lifelines (no identity skip) and collects stamps from every synapse; the native method can skip by node only and has no metadata accessor. The native sweep writes only `low_weight_steps`, so restoring the ≤2-per-protected-node lifelines is exact | `test_prune_site` × {lifeline, lifeline_grace3 (expiry), lifeline_grace0 (hold disabled), lifeline_budget} × native/fallback. Mutations caught: no counter restore (32 fail), lifelines not dropped (32 fail), stamps not collected (24 fail; the 8 grace0 cases rightly pass) |
| 3 | `_prune_synapses` **competing** mode (`compete_protected_links`) | **kept Python** | the native method sweeps every row in row order; competing mode visits only the caller's ids in sorted order with no identity skip — not expressible without a new native method. Cold path (once per dream cycle, a few hundred ids) | `test_competing_mode_unchanged` × {absent, lifeline}: 3 × (`compete_protected_links(2, 15)` + a default prune), records + bytes identical |
| 4 | `inject_reward` synapse sweep | converted — `apply_eligibility_reward` | unchanged on the trial | global, scoped set, empty set, frozenset with a ghost id, list scope; native + fallback |
| 5 | `HomeostaticRule.apply` scaling | converted — `scale_weights_by_post_node` | unchanged on the trial (the new `StrengthBudgetRule` is a separate rule) | 7 applications at `scaling_interval=3`; native + fallback |
| 6 | `STDPRule.apply` (+ `_apply_dw`) | converted — `stdp_reads` + one `apply_stdp_dw` per pass; `_apply_dw` kept as the fallback | unchanged on the trial | 4 rounds × 20 fired nodes, 2-factor and 3-factor; native + fallback |
| 7 | `Graph.step` phase 5 propagation | converted — `propagation_rows(ids, True)` | unchanged | 12 stimulate+step rounds (+ sprouting); native + fallback |
| 8 | `prime_and_propagate` hop-distance BFS | converted — `bfs_hop_distances` (guard `steps >= 1`) | unchanged | steps ∈ {0,1,2,3,5} × read / write / write+`tonic_ages_substrate`; fired entries incl. `source_distance`; native + fallback |
| 9 | `prime_and_propagate` propagate | converted — `propagation_rows(ids, _age_on)` | unchanged | as 8 |
| 10 | `_sprout_synapses` edge index | converted — `post_ids_of` / `pre_ids_of` | unchanged | via 7, plus a direct `_sprout_synapses` call |
| 11 | `get_telemetry` weights | converted — `weights_copy()` | unchanged | `repr(Telemetry)` equal, empty-graph branch equal |
| 12 | `extract_subgraph` filter | converted — `endpoint_triples()` | unchanged | 50-node subgraphs equal |
| 13 | `_deserialize` adjacency rebuild | converted — `endpoint_triples()` | unchanged | key order AND per-set iteration order of `_outgoing`/`_incoming` equal to the base restore; fallback helper == native |
| 14 | `tonic_engine._extract_graph_features_for_model` (non-protected) | converted — `weights_copy()[:200]`, `creation_time_copy()[:200]`, `len()`; fallback to the old list | unchanged on the trial | every `GraphFeatures` tensor `torch.equal` + same dtype/shape vs the base function from git, with ≥200 and <200 synapses; native + fallback |
| 15 | `neurograph_rpc._deposit_substrate_metrics` weights (non-protected) | converted — `weights_copy()` (exists since the store landed; no fallback added) | unchanged | values, order, `np.mean`/`np.std` identical |

Trial-only code NOT converted (no change): `apply_strength_budget` / `sleep_downscale` (already native
`normalize_strength` / `scale_all` with fallbacks), `_protected_lifelines` (already `get_weight`),
`_last_link_grace` (works on the small `to_prune` set), `compete_protected_links` (dream-time).
The overnight lane's own not-converted list (§6 of `RUST_HOTPATHS_REVIEW.md`: `ValenceField._diffuse`,
per-fired-node walks, node-side loops, Lenia snapshots) still stands.

## Whole runs

`test_whole_run`: 30 rounds of stimulate + `step()` + Tonic tick (`prime_and_propagate(write_mode=True)`,
`tonic_ages_substrate=1`) + recall (`write_mode=False`), `inject_reward` (global + scoped) every 5
rounds, `compete_protected_links(2, 10)` + `sleep_downscale(0.9)` + a checkpoint snapshot every 10
rounds. Flags ∈ {absent, off, lifeline, budget, lifeline+budget} × 6 seeds × {native, fallback}:
per-step traces, the snapshots and the final checkpoint bytes identical — **60/60**.

Full file: `tests/test_rust_hotpaths_onto_s4.py` — **313 passed, 1 skipped** (the skip is the opt-in
checkpoint-copy test). The overnight lane's `tests/test_rust_hotpaths_equivalence.py` (d0ee8cc oracles)
also passes on this branch: **60 passed, 1 skipped**.

## Checkpoint copy

`NG_ONTO_S4_CKPT=/tmp/rhs4/ckpt/main.msgpack` (run under `ulimit -v 3500000`, `nice -n 10`, base and
branch sequentially in one process, each graph dropped before the next): 2 prune passes, global +
scoped reward, forced homeostatic scaling, telemetry, then 5 × (30 stimulated + `step()` + Tonic tick +
recall). **The copy's own config has `prune_protected_faint_links=True`, `strength_budget_enabled=True`,
`last_link_grace_steps=2000`, `tonic_ages_substrate=1`, `three_factor_enabled=True`** — i.e. the live CC
graph runs the lifeline + budget paths — so every combination was run with explicit values:

| flags | identical | checkpoint out (base = branch) | bytes | prunes | steps (fired, pruned, sprouted) |
|---|---|---|---|---|---|
| absent | yes | `80d16fb285dc…` | 247,110,622 | [0, 0] | (635,0,10) (561,6,10) (553,10,10) (513,9,10) (412,0,10) |
| off | yes | `13401c6f1bbb…` | 246,967,772 | [25, 0] | (635,0,10) (562,6,10) (560,10,10) (522,9,10) (410,0,10) |
| lifeline | yes | `14f1e46a400f…` | 246,184,804 | [0, 0] | (635,0,10) (561,6,10) (552,10,10) (514,9,10) (421,0,10) |
| budget | yes | `5be01fe651da…` | 245,286,556 | [25, 0] | (635,0,10) (527,6,10) (386,10,10) (287,9,10) (252,0,10) |
| lifeline_budget | yes | `bd4f9773afbb…` | 215,436,292 | [841, 0] | (635,0,10) (526,6,10) (385,10,10) (279,9,10) (262,0,10) |

(Absolute digests differ between processes because set iteration order follows the per-process hash
seed; base and branch run in the same process and agree.)

## Timings (checkpoint copy, 67,968 synapses)

`tests/rust_hotpaths_onto_s4_timings.py`, median (min) of 5 calls, one process per side, `nice -n 10`,
`PYTHONHASHSEED=0`, two alternating rounds (base, branch, base, branch). The laptop was loaded
(load average 4-9, live daemon and other lanes running), so compare pairs, not absolutes. `step`,
Tonic tick and recall run with the checkpoint's OWN config (lifeline + budget ON = live CC), and each
includes its internal prune.

| Path | base r1 | branch r1 | base r2 | branch r2 | speedup (median of rounds) |
|---|---:|---:|---:|---:|---:|
| `_prune_synapses`, lifeline OFF | 1248.8 (1102.3) | 17.4 (16.7) | 882.7 (862.8) | 20.2 (18.9) | 56.7x |
| `_prune_synapses`, lifeline ON | 772.1 (725.6) | 216.3 (208.5) | 548.5 (511.9) | 515.5 (301.2) | 1.8x |
| `inject_reward(0.05)` global | 612.0 (503.7) | 1.1 (1.0) | 428.1 (387.3) | 1.4 (1.2) | 416.0x |
| `step()` (30 stimulated) | 2295.4 (2093.5) | 1375.4 (1282.5) | 2537.4 (2334.3) | 1616.4 (1477.0) | 1.6x |
| Tonic tick (`prime_and_propagate` write, 12 seeds, 3 steps) | 835.1 (780.4) | 579.7 (400.4) | 936.3 (828.0) | 466.5 (457.2) | 1.7x |
| recall (`prime_and_propagate` read, 6 seeds, 3 steps) | 165.6 (150.4) | 132.6 (120.4) | 191.0 (150.4) | 120.1 (115.0) | 1.4x |
| `get_telemetry` | 142.8 (127.5) | 12.1 (10.1) | 129.1 (125.4) | 10.9 (10.0) | 11.8x |
| restore (whole `Graph.restore`) | 13868.1 | 5057.5 | 5852.6 | 6103.5 | — |

## Suites (one pytest process per file, base = clean worktree of `5246b63`, same interpreter)

Runner `/tmp/rhs4/suite.sh`: 83 files = every `tests/test_cc_*.py` (40, incl. `test_cc_embed_outside_lock_922.py`),
`test_prune_lifeline.py`, `test_strength_budget.py`, `test_want_hub_competition.py`, `test_fair_chance_window.py`,
the two hot-path equivalence files, and every `tests/test_*.py` that calls a converted function
(`_prune_synapses`, `inject_reward`, `step`, `prime_and_propagate`, `_sprout_synapses`, `get_telemetry`,
`extract_subgraph`, `restore`, `HomeostaticRule`, `STDPRule`, `_extract_graph_features_for_model`,
`_deposit_substrate_metrics`, `compete_protected_links`, `apply_strength_budget`, `sleep_downscale`).
One pytest process per file, base then branch, 600 s cap per file, `PYTHONHASHSEED=0`, gated on load.
Base checkouts for the golden tests: `WANT_HUB_BASE_CHECKOUT=/home/josh/worktrees/ng-want-hub-base-26a0a11-canon-20261004`
(clean, at the pinned `26a0a11`; the test's default path does not exist), `PRUNE_LIFELINE_BASE_CHECKOUT=/home/josh/worktrees/ng-prune-lifeline-base-20261004`
(clean, `39c0422`), `STRENGTH_BUDGET_BASE_CHECKOUT=/home/josh/worktrees/ng-s4-base-20261004` (clean, `45f0812`) — read only.

**Result: no new FAILED/ERROR id.** For all 81 files present on both sides the FAILED/ERROR id sets and exit
codes are identical; 55 are clean on both. The two branch-only files pass (`test_rust_hotpaths_onto_s4.py`
313 passed / 1 skipped; `test_rust_hotpaths_equivalence.py` 60 passed / 1 skipped). `test_prune_lifeline.py`
(31), `test_strength_budget.py`, `test_want_hub_competition.py` (43), `test_cc_embed_outside_lock_922.py`,
`test_stdp.py`, `test_eligibility_traces.py`, `test_identity_protection.py` all pass on both.

Pre-existing on BOTH sides (identical ids; not investigated — outside this lane): `test_cc_bind_atomic_904` (2),
`test_cc_drain_pacing_seam` (4 errors), `test_cc_dual_pass` (6), `test_cc_merge_whole_graph_guard` (1),
`test_cc_pith_off_budget_812` (1), `test_cc_recall_reporting` (collection error), `test_cc_refeed` (1),
`test_cc_retrieval_enrichment` (1), `test_cc_swallows_915` (1), `test_cc_topology_callosum` (1),
`test_cc_topology_capture_423` (4), `test_ces` (4), `test_checkpoint_enforcer` (1), `test_conversational_recall` (2),
`test_fair_chance_window` (1, below), `test_graph_substrate_race` (1, timing race), `test_integration` (1),
`test_migration` (6), `test_prediction` (2), `test_reach_teaching` (1), `test_tonic_bridge` (1),
`test_tonic_habituation` (8), `test_tonic_spine_anchoring` (1); `test_snn` runs past the 600 s cap on both
(`.F` then timeout, as the overnight lane saw); `test_auto_knowledge` and `test_openclaw_hook` end with rc 143
(SIGTERM) on both after the same number of passing tests — no signal code in either file; the live daemon was
not touched (same PID before and after). Per-file logs: `/tmp/rhs4/suite/{base,branch}/`.

### `test_fair_chance_window.py`

`test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else` fails **identically on the trial
tip and on this branch**, at its FIRST assertion: `classes_added == []` gets `['StrengthBudgetRule#class']`
(the trial's strength-budget lane). So on the trial it already never reaches the function-set check. Run by
hand against its pin `b5e47686`, that check would ALSO list this branch's changes: 8 added module helpers
(`_stdp_reads_python` … `_apply_eligibility_reward_python`) and 9 more differing functions
(`Graph._sprout_synapses, extract_subgraph, get_telemetry, inject_reward, prime_and_propagate, step`,
`HomeostaticRule.apply`, `STDPRule.apply`) on top of the trial's own 25 added / 5 differing.
**Not edited**: it is the fair-chance trial's scope guard and the trial itself has outgrown it; re-pinning it
is not a one-line scope update (it would have to enumerate the want-hub, budget, lifeline and this lane's
changes), so it is the Executive's call.


## Worries / things to know

1. **The live CC config runs the slow prune path.** The checkpoint copy has `prune_protected_faint_links=True`
   and `strength_budget_enabled=True`. With the lifeline ON, `_prune_synapses` must still collect the last-link
   stamps from EVERY synapse's metadata (the trial loop does), and `ng_tract` has no native way to list rows
   with a metadata key. That walk (`items()` + `.metadata` per synapse) is now most of the cost: prune 1.8x
   (vs 57x with the lifeline off). It is why `step()` / Tonic tick gain only 1.6-1.7x here, not the overnight
   lane's 4-13x (which was also a 193K-synapse graph; today's copy has 67,968). Fix needs a Rust change
   (e.g. a `rows_with_metadata_key("last_link_since")` method) — not done (no Rust/wheel changes in this lane).
2. **Trial-side finding (not changed, flagged):** `SynapseRef.metadata` LAZILY CREATES and stores an empty dict
   for a metadata-less row. The trial's lifeline-on loop reads `syn.metadata` on every synapse every pass,
   so after the first pass every row holds a live empty dict (read from `store.rs` `SynapseRef::metadata`; not measured): extra RAM (one dict per synapse, ~68K here), and
   `to_checkpoint_msgpack` then delegates that row to Python's msgpack instead of writing 0x80 itself (same
   bytes, slower saves). I reproduced it exactly (equivalence required it). A `metadata`-peek that does not
   materialize would remove it — trial/Rust follow-up.
3. **Competing mode stays Python** (dream-time, small). Fine for cost; noted so nobody assumes it was converted.
4. **Two adjacency indices** (unchanged from the overnight risk): homeostatic scaling and the recall BFS follow
   the store's native index, everything else the Python `_outgoing`/`_incoming` dicts. They are maintained at the
   same mutation points and matched on every test here; a future drift bug would split them.
5. **Lifeline counter restore** writes `low_weight_steps` back on ≤2 synapses per protected node after the
   native sweep. Exact (the sweep writes nothing else, and the restore value is the pre-pass value), but it is
   "write then undo" — a native skip-by-synapse-id parameter would be cleaner (Rust change).
6. **Fallback helpers** are a second Python body of each native method (asked for by the brief, same pattern as
   the trial's `normalize_strength` fallback). They are proven equal to the trial tip by the `fallback` variants;
   they will rot if nobody runs those variants — they live in `test_rust_hotpaths_onto_s4.py`.
7. **Exception-path ordering** (overnight §7, still true): `step()` resets `inactive_steps` for a fired node's
   synapses before computing currents, STDP commits once per pass. Normal-path results identical.
8. **Process-level determinism:** checkpoint digests differ between processes (hash-seeded set order), so the
   equivalence is always base vs branch inside ONE process. Compare digests across processes only with a
   fixed `PYTHONHASHSEED`.
9. **Deployment order** (unchanged): the code needs the canonical wheel with the 11 methods; without it every site
   falls back to the trial's loops (proven equal), so an old wheel is slower, not broken.
10. Where a post-conversion `step()+Tonic tick` still spends its time (cProfile, 4 rounds on the copy, live config):
   `step()` own body (node loops), `STDPRule.apply` (Python dw arithmetic, 0.15 s per call), the lifeline-on
   prune walk, `_sprout_synapses`' candidate scan — the overnight lane's §6 items.

