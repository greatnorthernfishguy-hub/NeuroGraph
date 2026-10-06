# Native node store — P2b (review branch)

*2026-10-06 · lane nodestore-p2b (bounded build lane for the Executive) · review branches only: nothing merged,
nothing deployed, nothing installed into the NG venv, the live daemon untouched.*

Spec: `~/docs/superpowers/specs/2026-10-05-native-node-store-design.md` §4 "P2b", §1.5, §5.2 items 4–5, §6, §9, §10 item 7.
Josh approved the order (P2b next) on 2026-10-06. Builds on P2a (`NODESTORE_P2A.md`; NG `ef78c67`, Rust `55df6a1`).
The native store **stays OFF by default**. `neuro_foundation.py` is PROTECTED: edited on this branch only, in its own
commit; merging needs Josh's "proceed" after his backup check. No vendored file is touched (LAW 2). The wheel is not
re-vendored.

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-nodestore-p2b-20261006` | trial tip `ef78c67` | **PROTECTED** `2c25139` (`neuro_foundation.py` only) · tests `d4a0fa4` · this doc |
| ng-tract-rs | `cc-laptop-nodestore-p2b-rs-20261006` | P2a `55df6a1` | `4b00431` `SynapseStore.stdp_pass` + NodeStore read accessors + 2 Rust unit tests |

**Wheel** (built from `4b00431`'s tree, `maturin build --release -j 2`, abi3 ≥ 3.8; installed ONLY into the throwaway
venvs `~/.cache/p2b-venv` and `~/.cache/p2b-venv-on`, each with the venv's `.pth` link to the NG venv held aside during
the install and `~/NeuroGraph` bind-mounted read-only; the NG venv's `site-packages/ng_tract` mtime is still
2026-10-04 18:48; a copy is at `~/.cache/p2b/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`):

    sha256 6281bf7d79c80713238b49d85c3b0fd6a0886db936757dbed123886737350d7b

---

## 1. Exactness gate (spec §4, §10 item 7): **PASS**

The question: does Rust `f64::exp` return CPython `math.exp`'s exact bits for every argument STDP produces?

- **Structurally the same function.** `objdump -T` on the built `.so` and on `/usr/bin/python3.12` (whose `math`
  module is built in): both import `exp` with version `GLIBC_2.29` from `libm.so.6`, so in one process both call the
  same glibc entry point. CPython's `math.exp` adds only error checks (OverflowError / ValueError), which STDP can
  never trigger with `tau > 0` (the native pass declines otherwise, §2).
- **Empirically, through the shipped binary and the real code path.** `~/.cache/p2b/exp_gate.py` drives
  `SynapseStore.stdp_pass` itself with A± = lr = 1.0, a curvature table of 1.0, w = 0.0, max_weight = 1.0 and
  three-factor from eligibility 0.0, so each row's eligibility is exactly `0.0 + exp(-dt/tau)` (incoming, LTP) or
  `0.0 + (-exp(dt/tau))` (outgoing, LTD). Expected values use the trial loop's own Python expression. Compared with
  `struct.pack('<d')`.
  - τ ∈ {10, 15, 20, 25, 30}: every τ configured in the tree (CC host/daemon 10, openclaw 15, default 20, examples 25/30).
  - Per τ: **every integer |dt| in 1 … 2^18** on both passes (exp underflows to exactly 0.0 past 745.13·τ = 22,354
    for τ = 30, so this covers all non-zero results with 11× margin); later-firing partners (prime_and_propagate's
    write mode stamps `prop_timestep` > `timestep`) for |dt| in 1 … 2^17 on both passes; then 32,768 random integer
    dt up to 2^31, 16 dt = 0, 32,736 random non-integer dt in (0, 800·τ), and 16 dt spread over 10^-300 … 10^9.
  - In production every dt is an integer-valued float (`last_spike_time` is only ever `float(timestep)` /
    `float(prop_timestep)` or −inf), so the integer sweeps are the exhaustive set; the random sweeps are extra.
  - **Result: 4,587,520 results compared, 0 mismatches** (log `~/.cache/p2b/logs/exp_gate.txt`). About 300K of the
    integer-dt results and about 300K of the random ones are non-zero; the rest are exact underflow zeros.
- `three_factor` and the curvature table add only multiplications/additions, no transcendental.
- An in-suite slice of the gate runs in `tests/test_nodestore_p2b.py::test_stdp_pass_exp_is_math_exp_bitwise`
  (every integer dt up to past underflow for each τ, through the binary).

So `exp` runs natively. The fallback plan in the spec (compute arguments natively, `exp` in Python) was not needed.

## 2. What changed

### Rust (`ng-tract-rs`, additive)

`SynapseStore.stdp_pass(node_store, fired_ids, timestep, A+, A-, tau+, tau-, lr, three_factor, curvature_table,
incoming, outgoing) -> bool` is the whole `STDPRule.apply` for one call:
- For each fired id in order: the incoming pass, then the outgoing pass. Each pass reads `(other node, weight,
  max_weight)` per synapse, computes dw with the loop's own expressions in their operand order
  (`A+ * exp(-dt/tau+) * lr * max((mw-w)/mw, 0.0)`, `-A- * exp(dt/tau-) * lr`, `A+ * 0.5 * lr * max(...)` for
  dt == 0 / NaN on the incoming pass; the outgoing pass skips dt == 0 / NaN), multiplies by
  `table[clamp(pre layer)][clamp(post layer)]`, and commits before the next pass reads. Python's `max(a, b)`
  NaN / −0.0 semantics are kept (`py_max`). No FMA.
- `last_spike_time` and `diffpc_layer` come from the NodeStore columns. Node ids are resolved through a cached
  interner-index → (node row, generation) map. It is keyed by (NodeStore `uid`, `layout_epoch`, interner epoch)
  and re-validated per entry: a cached row only while its generation matches (a removed node's row has generation 0;
  a re-created id gets a new row and generation), a cached "absent" only while no node has been inserted since
  (`next_gen`). Compaction, `NodeStore.clear()` and `SynapseStore.clear()` void it.
- The synapse sets are the **graph's own** `_incoming` / `_outgoing` sets, read in their iteration order, not the
  store's native adjacency index. That keeps the equivalence exact by construction (a stale id in a set is skipped
  exactly as `stdp_reads` returns `None` for it). It costs string extraction (see §4).
- Locking (spec §6): all Python inputs are read before any borrow; then `SynapseStore` `borrow_mut`, then
  `NodeStore` shared borrow (synapses, then nodes; both `try_borrow`, a failure declines); no Python call and no
  `allow_threads` inside.
- **Declines** (returns `False`, touches nothing; the caller runs the original loop, which then reproduces the
  partial commit and the exception exactly): a parameter that is not an exact `float`; `tau <= 0` or NaN; a
  non-`bool` three_factor; a table that is not 3×3 exact floats; `fired_ids` not an exact `list`, or a non-`str` id;
  `incoming` / `outgoing` not exact dicts, or a non-`str` member; `node_store` not a NodeStore; a fired id with no
  node (the loop raises `KeyError`); `max_weight == 0` on an LTP / dt == 0 row (`ZeroDivisionError`); any
  exact-Python overflow value (P1's D4 cell) in `last_spike_time` / `diffpc_layer`. Planning (which decides every
  decline) runs before any write; nothing it reads (node presence, spike times, layers, max_weight) is changed by
  STDP commits.
- `apply_stdp_dw`'s per-row body moved verbatim into `stdp_commit_row`, shared by both, so the native pass and the
  fallback's commit cannot drift. Behaviour unchanged (crate `test_hotpaths` passes).
- `node_store.rs`: a process-unique `uid` per NodeStore and three `pub(crate)` read accessors. No method changes.
- Rust unit tests: 30 passed (2 new: dw branches incl. NaN / −0.0 / saturation; node-row cache across add /
  remove / re-add).

### Python (`neuro_foundation.py`, PROTECTED commit `2c25139`)

`STDPRule.apply` calls `getattr(graph.synapses, "stdp_pass", None)` with `graph.nodes`, the rule's parameters,
`graph.config.get("three_factor_enabled", False)`, `_GSG_CURVATURE_TABLE` and `getattr(graph, "_incoming" /
"_outgoing", None)`, and returns only when the result `is True` (so a `MagicMock` graph's truthy auto-return never
skips the loop). Otherwise `_stdp_python(rule, graph, fired_node_ids, timestep)` runs: the trial tip's loop moved
verbatim to module level (`self` → `rule`; checked by string comparison against `git show ef78c67`).
- The default path (dict of `Node`) always declines at the NodeStore check and runs the identical loop.
- An older wheel (no `stdp_pass`) runs the identical loop.
- No other file changed. `activation_persistence.py`, `stream_parser.py`, `openclaw_hook.py`, vendored files: untouched.

### What stayed in Python, and why

- **Step 5 propagation (`propagate_fired`): not built.** On the checkpoint copy **10,979 of 10,997 nodes carry
  `poincare_dir`**, so the GSG geodesic path (which stays Python per spec §4 until the P4 `poincare_dir` decision)
  covers essentially every fired node; a native plain path would serve the other 0.16%. It is not "clean" enough to
  be worth a second code path for no measurable gain. The full step-5 conversion still waits on P4 (D5).
- **`_GSG_CURVATURE_TABLE`** is still the Python table; it is read per call (9 floats).
- Everything P2a left in Python is unchanged (`ratio ** scaling_factor`, the threshold-pass dict lookups, readers).

## 3. Equivalence (the readiness bar)

`tests/test_nodestore_p2b.py`, BASE = `git show ef78c67:{neuro_foundation,tonic_thread,activation_persistence}.py`
imported as their own modules. Floats compared by `struct.pack('<d')`.

| Run | Result |
|---|---|
| whole file, P2b wheel (`~/.cache/p2b-venv`) | **110 passed** (`~/.cache/p2b/logs/test_p2b_full.txt`) |
| — of which whole runs (§5.2 item 5) | **72/72**: 6 seeds × (5 trial flag sets + a `pre_fire`-handler set) × {ON native, OFF fallback}, each against the trial tip in every log entry (incl. **per-step synapse id / weight / eligibility / peak / last_update / max_weight bitwise after every `step()` and every write-mode `prime_and_propagate`**), every snapshot and the final checkpoint bytes. ON asserts `_stdp_python` never ran; OFF asserts it ran ≥ 30×. |
| — per-apply dw on edge graphs vs trial tip | 18/18: 6 seeds × {OFF, ON native, ON fallback over NodeRefs}, 4 applies each (with duplicate fired ids), every synapse bitwise after each apply, final checkpoint bytes; non-vacuous (every apply changes state); ON asserts native engaged |
| — exceptions at the same point | 9/9: `max_weight == 0` (ZeroDivisionError) and a fired id with no node (KeyError), same exception and same partial commit as the trial tip, all modes |
| — native vs fallback on the same ON graph, `True` contract | 3/3 |
| — declines touch nothing (18 bad inputs, a non-str set member, an int `last_spike_time` overflow cell) | pass |
| — node-row cache under churn (tombstones, re-creation, > 1/4 removal → compaction, `SynapseStore.clear()` + reload) vs fallback twin | pass |
| — native engages inside `step()` | pass |
| — exp slice through the binary, 5 τ | 5/5 |
| crate Python tests on the P2b wheel (`test_store`, `test_hotpaths`, `test_btf`, `test_node_store`) | **337 passed** |
| Rust `cargo test --lib` | **30 passed** |

Edge rows covered: dt == 0 (half-strength LTP), dt < 0 and > 0 on both passes (incl. later-firing partners),
soft saturation (w at, one ulp under, and above max_weight; max_weight 1e-300 / inf / NaN), NaN / ±inf
`last_spike_time`, NaN / inf / −0.0 weights, `diffpc_layer` ∈ {−1, 3, 7, 2^40} (clamped), missing pre and post
nodes (nodes deleted with their synapses and adjacency entries left), tombstoned and re-added ids, stale synapse
ids in the adjacency sets, three_factor on and off, three τ / A± sets.

**Checkpoint COPY** (`~/.cache/p2a/ckpt/main.msgpack`, mode 444, sha256 `70d59bbb45d5c39e…`, the P2a lane's
read-only copy of `~/.claude/plugins/neurograph/checkpoints/main.msgpack`; 10,997 nodes, 56,750 synapses after the
run). `vectors.msgpack` is not needed by `Graph.restore` and was not copied. `~/.cache/p2b/ckpt_equiv_p2b.py`, one
isolated process per mode: seeded `random` and `uuid4`; 12 steps with 30 stimulated nodes each, a forced
homeostatic scaling pass, and a write-mode `prime_and_propagate` every 3rd step (the second STDP call site); 994
fired. Hashed after every step and every prime: fired ids, every node's voltage / threshold / EMA / excitability /
last_spike_time / Ca / refractory, and every synapse id / weight / eligibility.

| Mode | trace sha256 | final checkpoint sha256 |
|---|---|---|
| trial tip `ef78c67` (dict) | `a607fd71c500b055…` | `b7095a269cf4ae79…` |
| branch OFF (dict → `_stdp_python`) | `a607fd71c500b055…` | `b7095a269cf4ae79…` |
| branch ON native (`stdp_pass`) | `a607fd71c500b055…` | `b7095a269cf4ae79…` |
| branch ON fallback (`_stdp_python` over NodeRefs) | `a607fd71c500b055…` | `b7095a269cf4ae79…` |

**4/4 identical.**

## 4. Timings vs the §9 estimate

Checkpoint copy, both graphs (dict OFF and native ON) in ONE process, alternating every measurement
(`~/.cache/p2b/bench_p2b.py`, two runs). The live daemon and two suite lanes were running; load 5–10.
**Compare ratios; absolute numbers are ±50%.** ms, median.

| `STDPRule.apply` | python, dict (= trial tip) | python, NodeRef (P1/P2a ON) | **native `stdp_pass`** |
|---|---:|---:|---:|
| inside 12 real `step()`s (37–135 fired) | 15.2 / 18.7 | — | **2.4 / 2.7** |
| alone, the last step's 134 fired | 20.1 / 43.6 | 25.9 / 58.5 | **3.2 / 6.4** |
| alone, 550 fired (the spec's §1.4 step fired 499–646) | 88.2 / 76.9 | 114.1 / 97.7 | **14.8 / 11.7** |

| full `step()` (12 steps) | OFF (dict) | ON (native) |
|---|---:|---:|
| run 1: median / sum | 390.6 ms / 5.70 s | 366.6 ms / 5.19 s (ON faster in 8 of 12 steps) |
| run 2: median / sum | 422.7 ms / 8.14 s | 464.8 ms / 6.97 s (ON faster in 5 of 12 steps) |
| cProfile, 8 steps (overhead inflates Python-heavy paths) | 995 ms/step | 716 ms/step |

Read honestly:
- **STDP itself: about 6–7× faster than the dict loop, about 8× faster than the P1/P2a NodeRef loop.**
- **The §9 baseline did not reproduce.** The spec's "0.77 s/step" came from a cProfile (which inflates per-call
  Python 3–5×) on the 7f6b7e2 tip with 67,968 synapses. Unprofiled, on today's copy (56,750 synapses), the dict
  loop costs 15–19 ms per real step and 77–88 ms at 550 fired. Native is 2.4–2.7 ms per real step and 12–15 ms at
  550 fired: inside the 0.05–0.15 s target, but the saving is ~13–16 ms per real step, not ~0.7 s.
- **About half of the native time is reading the graph's `_incoming` / `_outgoing` sets into Rust strings**
  (550 fired, 5,284 synapse ids: 16.2 of 28.1 ms in a loaded micro-run, `~/.cache/p2b/micro_p2b.py`; abi3-py38
  string extraction, spec D6). Using the store's native adjacency instead would remove it, but equivalence would
  then rest on `Graph._incoming/_outgoing` ≡ the native index (review §6.4 wants those dicts retired anyway). Left
  for a ruling.
- **Is `step()` ON now net faster than OFF? Not resolvable from these runs.** The STDP saving (~13–16 ms) is
  3–4% of a ~400 ms step, below the run-to-run noise at this load: run 1 ON median −6%, run 2 ON median +10%;
  ON's 12-step sums were lower both times (−9%, −14%). Under cProfile ON is 28% faster, but the profiler
  overstates Python-loop savings. P2a measured ON ≈ OFF; P2b removes STDP's NodeRef regression and adds a small
  gain, so ON is at least at parity.
- **Where `step()` time actually is now** (cProfile, both modes): `_structural_plasticity` 0.52–0.57 s/step
  profiled (`_prune_synapses` 0.37 s/step, `_sprout_synapses` 0.10–0.15), then `_compute_hyperedge_activation`,
  `_strength_protected_ids`. ON still pays a NodeRef regression in the prune path:
  `_is_identity_protected` 1.03 s vs 0.56 s (8 steps, profiled; ~98K `NodeStore.get` calls in those 8 steps). That is the
  "pruning into sleep" item (#1046) next in Josh's order, and a candidate for a native protected-id read.

## 5. Suites

Every `tests/test_*.py`, one isolated pytest process per file (`~/.cache/p2b/suite.sh`, a copy of the P2a script
pointed at this lane's harness copy `~/.cache/p2b/run-isolated.sh`), 600 s cap, 3 GB memory cap, `PYTHONHASHSEED=0`,
the same golden checkouts P2a used (read-only). Logs: `~/.cache/p2b/suite/{base,branch-off,branch-on,base-rerun}/`;
compare script `~/.cache/p2b/compare_suites.py`.
- **base**: a clean detached worktree of `ef78c67` (`/home/josh/worktrees/ng-nodestore-p2b-base-ef78c67`, 154 files).
- **OFF**: this branch, default (155 files incl. the new one).
- **ON**: this branch with P1's host-switch hook loaded through a `.pth` in `~/.cache/p2b-venv-on`, so every Graph
  is a native NodeStore and every STDP pass in the suite (e.g. `test_stdp`, `test_eligibility_traces`,
  `test_rust_hotpaths_*`) runs `stdp_pass`.
- Same interpreter and the same P2b wheel for all three (base code never calls `stdp_pass`).

| Pass | passed | failed | errors | new FAILED/ERROR ids vs base |
|---|---:|---:|---:|---|
| base | 3,110 (+351 in the rerun below = 3,461) | 67 | 6 | — |
| OFF | 3,571 | 67 | 6 | **none** |
| ON | 3,570 | 68 | 6 | **1**: `test_graph_substrate_race.py::test_build_adjacency_no_race_under_concurrent_step_mutation` |

- **The one ON difference is the same load-dependent test P2a reported**, and it is not from this change: a 30 s
  wall-clock assertion (a reader must finish 5,000 `_build_adjacency` calls while a writer churns the synapse store);
  it never calls `step()` or STDP. Re-run twice each at load 6–7: base failed 2/2, ON failed 1/2.
- **Base lost three files to the load** (600 s timeout / memory kill while two passes ran): `test_pith_stage4`,
  `test_rust_hotpaths_onto_s4` (559 s on OFF), `test_tonic_no_heuristic`. Re-run alone on base with a 1,500 s cap:
  20, 313 (+1 skipped) and 18 passed, identical to OFF and ON. OFF − base(adjusted) = 110 = the new test file.
- **Not covered by any pass, the same on all three**: `test_auto_knowledge`, `test_ces`, `test_coordinator`,
  `test_et_modules`, `test_openclaw_hook`, `test_tonic_prefetch` exceed the 3 GB cap or the 600 s cap, and
  `test_snn` hits the 600 s cap in its 10K-step `test_1k_nodes_10k_steps` (#754; its other 13 tests ran, with the
  same 1 failure on every pass). Raising the caps next to the live daemon would recreate the 2026-10-05 18:54
  earlyoom kill (P2a §7), so it was not done. `test_snn`'s long test does run STDP every step; it remains unrun.
- **Pre-existing on base, identical ids on OFF and ON**: 67 failed + 6 errors in 29 files (`test_cc_*`,
  `test_migration`, `test_ingestor`, `test_gui`, …), the same set P2a listed. Not investigated (out of lane).

**Older-wheel check** (the live NG venv's wheel has no NodeStore at all; the P2a wheel has NodeStore but no
`stdp_pass`): `tests/test_nodestore_p2b.py` under `~/.cache/p2a-venv` (P2a wheel `0f08699c…`): **99 passed,
11 skipped** (the `stdp_pass`-only tests), including all 72 whole runs, so ON graphs on a wheel without
`stdp_pass` take `_stdp_python` and still equal the trial tip.

## 6. Open risks / worries

1. Protected-file merge is Josh-gated (D7). `STDPRule.apply`'s default (OFF) path now calls the moved-verbatim loop
   as a function; the whole runs and the checkpoint copy prove it equal to the trial tip.
2. The §9 STDP target was set against a profiled baseline; the real gain on this graph is ~13–16 ms per step, and
   `step()` ON vs OFF is within noise (§4). The enable gate's "net faster" criterion is not demonstrated by P2b alone.
3. Native STDP reads the graph's Python adjacency sets (exact, but half its cost). Switching to the native index
   needs a ruling and an equivalence proof that the two cannot diverge.
4. Exactness rests on CPython and the wheel calling the same glibc `exp@GLIBC_2.29` in one process. A different
   libc / libm (e.g. a musl build, a statically linked wheel, a future glibc changing `exp`'s result) would need
   the gate re-run; `~/.cache/p2b/exp_gate.py` is the re-runnable proof. On the VPS (Syl) the wheel has no
   NodeStore, so nothing there changes.
5. Behaviour on concurrency improves rather than regresses: the native pass is atomic under the GIL (no Python
   inside the borrows), where the loop could interleave with an unlocked Tonic thread between synapses.
6. `ZeroDivisionError` / `KeyError` paths decline to the loop, which raises; the plan pass costs a full read before
   it can know, so an exception-path step pays for both (only on malformed state).
7. Vault docs (`~/docs/modules/NeuroGraph.md`, the spec's progress log) are not updated from this lane — for the
   Executive at integration.
8. Off-task, noticed: the P2a harness left ~500 `~/.cache/p2a-home.*` scratch HOMEs (and this lane's
   `~/.cache/p2b-home.*`); harmless, but worth a cleanup line in the harness.
