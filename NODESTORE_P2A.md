# Native node store — P2a (review branch)

*2026-10-06 · lane nodestore-p2a (bounded build lane for the Executive) · review branches only: nothing merged,
nothing deployed, nothing installed into the NG venv, the live daemon untouched.*

Spec: `~/docs/superpowers/specs/2026-10-05-native-node-store-design.md` §4 "P2a" table, §5.2 items 4–5, §6, §9;
Josh's decisions (D1 tombstones, D3 KeyError, D6 abi3-py38, D7 one "proceed" per phase, P1 merged switched OFF).
Builds on P1 (`NODESTORE_P1.md`; NG `6a85357`, Rust `71a9d2f`). The native store **stays OFF by default**.
`neuro_foundation.py` and `activation_persistence.py` are PROTECTED: edited on this branch only; merging needs
Josh's "proceed" after his backup check. No vendored file is touched (LAW 2). The wheel is not re-vendored.

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-nodestore-p2a-20261006` | trial tip `6a85357` | **PROTECTED** `58762e2` (`neuro_foundation.py`, `activation_persistence.py`) · `fb49257` Tonic (`tonic_thread.py`, `tonic_engine.py`) · tests `8fd0090`, `2a9d726`, `da2d4bb` · this doc |
| ng-tract-rs | `cc-laptop-nodestore-p2a-rs-20261006` | P1 `71a9d2f` | `014b13b` P2a methods + kernels · `55df6a1` per-row degree-target lookup |

**Wheel** (built from `55df6a1`, `maturin build --release -j 2`, abi3 ≥ 3.8; installed ONLY into the throwaway
venvs `~/.cache/p2a-venv` and `~/.cache/p2a-venv-on`; a copy is at
`~/.cache/p2a/wheel-v2/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`):

    sha256 0f08699c33ffd9e3c4449f5a54274934e5391331bb69a2631cbf54af91dd970a

(An earlier build from `014b13b`, sha256 `0dce2ba2…94b7`, is superseded; every number below is from the final wheel
unless marked "v1".) The NG venv's `ng_tract` was never written: each install ran with the venv's `.pth` link held
aside and `~/NeuroGraph` bind-mounted read-only (its `site-packages/ng_tract` mtime is still 2026-10-04 18:48).

---

## 1. What changed, per method

All on `ng_tract.NodeStore` (`src/node_store.rs`, additive; no existing method changes). Each method:
- reads its Python inputs before borrowing the store, runs one pure-Rust kernel on `Core` under one borrow
  with no Python call, and builds its Python result after the borrow ends (the 25c5f52 rule);
- keeps the loop's float expression in its operand order (Rust does not contract to FMA), with Python's
  builtin `min`/`max` argument-order and NaN semantics (`py_min`/`py_max`) and rows in insertion (dict) order;
- **declines** (returns `False`/`None`, touches nothing) when a parameter is not an exact `float` / `int`, or
  when a row it reads or writes holds an exact-Python overflow value (P1's D4 cell, e.g. an `int` voltage).
  The caller then runs the Python fallback. On the live data there are 0 overflow cells (P1 measured this).

| Method | Replaces | Kernel (same expression as the loop) | Declines on |
|---|---|---|---|
| `decay_voltages(decay)` | `step` 1 | `v = v*decay + (1.0-decay)*rest` | non-float decay; overflow in voltage/resting |
| `calcium_currents(g_net, Ca_decay)` | `step` 3a | `if Ca > 1e-9: v = v + g_net*Ca; Ca = Ca*Ca_decay` | non-float args; overflow in Ca/voltage |
| `detect_fired()` → ids | `step` 3 | `refractory_remaining > 0` skips; `voltage >= threshold` | overflow in refractory/threshold/voltage. **Called only when no `pre_fire` handler is registered** |
| `fire(ids, timestep, delta_Ca=None)` | `step` 4 node writes | voltage=rest; refractory=period; `last_spike_time = float(ts)`; ring append; `Ca = min(Ca+δ, 5.0)` | non-int timestep, non-float δ, a non-str or missing id (the loop's `nodes[nid]` then raises midway, as before); overflow in a touched field of a fired row. `_recent_spikes` stays Python (P3) |
| `decrement_refractory(exclude_ids)` | `step` 9 | `if rr > 0 and not excluded: rr -= 1` | overflow in refractory |
| `update_firing_ema(fired_ids, alpha)` | `HomeostaticRule` pass 1 | `ema = (1.0-α)*ema + α*fired` | non-float α; overflow in ema |
| `adapt_thresholds(targets, default, rate, ceiling)` | pass 2 | `> t*1.2 → min(th+rate, ceil)`, `< t*0.8 → max(0.01, th-rate)` | non-float args; not an exact dict; a non-float target **of an existing node**; the row layout changed during the dict lookups |
| `adapt_excitability(targets, default, exc_rate)` → `{nid: ratio}` | scaling pass | silent: `min(x*(1.0+er*5), 5.0)`; else `ratio = t/ema`, `min(x*(1+er), 5)` / `max(x*(1-er), 0.1)` | as above; **`ratio ** scaling_factor` stays in Python** (libm `pow` not proven) |
| `columns(names, ids=None, with_ids=True)` → `(ids, arr…)` | readers | numpy f64 / i64 copies in row order, one borrow (atomic snapshot) | a requested cell holds an overflow value (→ `None`); unknown/non-numeric name → `ValueError`; unknown id → `KeyError` |

Degree targets (`HomeostaticRule._degree_targets`, ~11K keys): the first build copied the dict out as Rust Strings,
16–20 ms per call. The final build looks each live row's target up with the row's cached id `str` (P1's
one-`NodeRef`-per-row cache), with no borrow held, then re-validates the `(row, generation)` snapshot and
`layout_epoch` under the pass's borrow and declines if anything was inserted, removed or compacted meanwhile
(tested with a dict key whose `__eq__` creates a node mid-lookup).

## 2. Call-site wiring

Each site: `getattr(graph.nodes, "<method>", None)`. When that is `None` (the dict of `Node`, which is the default;
an older wheel; a duck-typed test graph), or the method declines, the module-level `_<pass>_python` runs. That
function is the trial's original loop moved verbatim (`neuro_foundation.py`, after `_apply_eligibility_reward_python`).

| File (function) | Site | Fallback |
|---|---|---|
| `neuro_foundation.py` `Graph.step` 1 | `decay_voltages` | `_decay_voltages_python` |
| `step` 3a | `calcium_currents` | `_calcium_currents_python` |
| `step` 3 | `detect_fired` only if `not self._event_handlers.get("pre_fire")` | `_detect_fired_python` (the per-node handler loop) |
| `step` 4 | `fire`, then the `_recent_spikes` appends in Python | `_fire_python` |
| `step` 9 | `decrement_refractory(fired_this_step)` | `_decrement_refractory_python` |
| `HomeostaticRule.apply` | `update_firing_ema`, `adapt_thresholds`, `adapt_excitability` (+ Python `**`) | `_update_firing_ema_python`, `_adapt_thresholds_python`, `_adapt_excitability_python` |
| `Graph.get_telemetry` | `columns(["firing_rate_ema"], with_ids=False)` | the old list comprehension |
| `activation_persistence.py` `capture` | `columns([voltage, resting, last_spike, excitability])` | the old loop |
| `tonic_thread.py` `_read_active_nodes`, `_read_recent_spikes` | `_node_scan_rows()` → `columns()` | node attribute reads |
| `tonic_engine.py` `_extract_graph_features_for_model` | `columns()[:100]` | `list(g.nodes.values())` |

The reader sites also accept only a `tuple` result, so a `MagicMock` graph (whose `columns` would be an auto-mock)
still takes the original reads. `openclaw_hook.py`, `stream_parser.py` and all vendored files: unchanged.

## 3. Equivalence (the readiness bar)

`tests/test_nodestore_p2a.py`. BASE = `git show 6a85357:{neuro_foundation,tonic_thread,activation_persistence,tonic_engine}.py`,
each imported as its own module. Floats are compared by `struct.pack('<d')`, so NaN, ±0.0 and ±inf compare exactly.

- **Per method**: native vs its fallback over the same store's `NodeRef`s vs the fallback over a dict of `Node`,
  on edge graphs. The graphs are built by the trial-tip module, checkpointed, restored into each mode, then
  tombstoned and re-added in the live store.
  - Rows covered: voltages/thresholds over {0.0, −0.0, NaN, ±inf, 5e-324, 1e-300, 1e-9 ± 1 ulp, 0.85 ± 1 ulp, 5.0 ± 1 ulp, 1e308};
    EMA over {0, −0.0, 1e-10, 1e-9 ± 1 ulp, NaN, inf, negatives}; Ca over {0, 1e-9 ± 1 ulp, NaN, 4.9, 5, 6, −1};
    refractory and period ints at the msgpack width boundaries (127/128, 255/256, 65535/65536, −129, 2^40, 2^62).
  - Spike rings: capacity 0/1/7/100, empty, full, over-full.
  - Order and ids: tombstoned-then-re-added ids, duplicate fired ids, a `float(2**53+1)` timestep, degree-target
    dicts with NaN/0/tiny targets plus a ghost key.
  - Also tested: the operand-order cases, declines (non-float params, an `int` voltage, an `int` target, a mapping
    proxy), the `columns()` contract, reentrant Python input iterables, and degree-lookup re-validation.
- **`step()` and `HomeostaticRule`** on those edge graphs vs the trial tip, with and without a `pre_fire` handler.
  Compared: fired ids, per-step voltages bitwise, the final node state and checkpoint bytes.
- **Readers** vs the trial tip's files: the Tonic scans, the Tonic features (torch tensors via `~/Elmer/surgery`),
  the activation capture and `get_telemetry`.
- **Whole runs (§5.2 item 5)**: 30 rounds. Each round: stimulate + `step()` + per-step voltages/threshold/Ca/
  refractory bitwise + Tonic tick + Tonic scans + recall + rewards + node create / remove / orphan sweep + a
  > 1/4 mass removal (compaction) + activation capture + telemetry + `compete_protected_links` + `sleep_downscale`
  + a snapshot every 10 rounds. Run over the trial's flag matrix (absent, off, lifeline, budget, lifeline+budget)
  plus a `pre_fire`-handler set, × 6 seeds × {ON native, OFF fallback}, each against the trial tip. ON runs also
  assert that no fallback ran except the 30 pre_fire detections.

| Run | Result |
|---|---|
| final wheel, whole file | **148 passed** |
| whole runs, final wheel | **72/72** (36 ON native + 36 OFF fallback; each = trial tip in every log entry, snapshot and final checkpoint byte) |
| whole file under the **P1** wheel (ON = the fallbacks over `NodeRef`s) | **106 passed, 42 skipped** (native-only asserts), incl. whole runs 72/72 |
| Rust `cargo test --lib` | 28 passed (2 new: min/max semantics; kernels skip tombstones, row order, ring capacity 0) |
| crate Python tests on the final wheel (`test_store`, `test_hotpaths`, `test_btf`, `test_node_store`) | 337 passed |

**Checkpoint COPY** (`~/.cache/p2a/ckpt/main.msgpack`, mode 444, sha256 `70d59bbb45d5c39e…`, 238,121,111 bytes,
10,997 nodes, copied read-only from `~/.claude/plugins/neurograph/checkpoints/` at 00:15):
- Setup: one process per mode; seeded `random` and `uuid4`; 12 steps with 30 stimulated nodes each and a forced
  homeostatic scaling pass; 1,062 fired.
- Hashed after every step: the fired ids, then every node's voltage, threshold, EMA, excitability, Ca and
  refractory (bitwise).
- Result: trace sha256 `8d82e02e8679f8f3…` and final checkpoint sha256 `49dd6fb0e014a46a…` are **identical** for
  the trial tip, branch OFF, branch ON (final wheel) and branch ON under the P1 wheel. That is 4/4, the same on v1.

## 4. Timings vs the §9 targets

Measured on the checkpoint copy, both graphs (dict OFF and native ON) in ONE process, alternating every
measurement (median of 9; `HomeostaticRule` of 3). The laptop ran the live daemon (≈4.8 GB, 2 cores) and the suite
lanes, at load 5–7, so **compare ratios; absolute numbers are ±50%**. Two runs agreed. ms:

| Pass | dict (trial-tip loop) | NodeRef (P1, no P2a) | **native** |
|---|---:|---:|---:|
| 1 decay | 5.3–5.4 | 9.5–10.3 | **0.21** |
| 3a Ca | 4.9–5.0 | 10.6–11.1 | **0.07** |
| 3 detect | 6.2–6.6 | 15.1–16.6 | **0.34–0.36** |
| 4 fire (≈130 fired) | 1.2 | 1.9–2.0 | **0.25** |
| 9 refractory | 2.7–2.9 | 8.8–12.2 | **0.16** |
| H EMA | 6.0–6.2 | 15.0–17.4 | **0.18** |
| H thresholds | 17.4–18.4 | 32.0–35.4 | **6.3** |
| H excitability (+ Python `**`) | 22.4 | 35.9–36.8 | **10.1–10.3** |
| **per firing step (spec's 28 ms subtotal: 1, 3, 3a, 4, 9, EMA, thresholds)** | **≈ 44** | **≈ 95** | **≈ 7.5** (1.2 without thresholds) |

| Path | dict | native | §9 target |
|---|---:|---:|---|
| `HomeostaticRule.apply`, non-scaling | 25.7–26.4 | **8.3–9.5** | ~25 → ~1 ms: **not met** (thresholds pass) |
| `HomeostaticRule.apply`, scaling step (incl. `_refresh_degree_targets`) | 178–191 | **222–254 (slower)** | — |
| Tonic `_read_active_nodes` | 95–99 | 92–93 (NodeRef path 110–114) | 12 → < 1 ms: **not met** |
| Tonic `_read_recent_spikes` | 39 | 35–39 (NodeRef 46–51) | |
| `activation_persistence.capture` | 23–24 | 27–28 (NodeRef 26) | 33 → ~2 ms: **not met** |
| `get_telemetry` | 13.3–13.7 | 10.8–11.1 | |
| **full `step()`** (12 steps, 37–137 fired) | median 0.552 / 0.361 s | **0.492 / 0.358 s** | P1 was +18–22% ON; now **at parity** |

Read honestly:
- The six step and EMA passes meet the target: about 21 ms → about 1.2 ms.
- The **threshold pass misses it.** The native time is ~11K Python dict lookups (one per node, for that node's
  degree target). `list(dict.items())` on the same dict costs 4.5 ms on this box under this load.
  - Going lower needs a cached, row-aligned target vector. Its validity would rest on an invariant ("the
    rule's `_degree_targets` is never mutated in place") rather than hold by construction, so it is left for
    the Executive / Josh to rule on (§6).
- **The readers barely move.** Their cost is the per-node Python work: the Tonic scan computes a
  `hash((nid, cycle))` noise term per node, fatigue lookups and tuple building, and capture builds an entry dict
  per active node. The attribute reads were not the cost.
  - The column copy does remove P1's proxy regression on these paths, and it is now one atomic snapshot (D3).
  - Meeting the spec's numbers would mean vectorising that arithmetic. That is not attempted here (§6).
- **The scaling step is slower ON**, because `_refresh_degree_targets` (degree computation, `diffpc_layer` and
  `manifold_type` assignment), which is not a P2a pass, still runs through `NodeRef`s. It runs once per
  `scaling_interval`.
- **`step()` ON ≈ OFF.** STDP and propagation (P2b) are still Python over `NodeRef`s, which is why ON is not
  yet net faster.

Equality during the timing runs: `bench2.py` leaves `uuid4` unseeded, so the two graphs sprout different synapse
ids. Fired ids were identical every step; voltages differed in one of the two runs. That is explained by different
set iteration, hence float summation order, over different random ids. The seeded checkpoint-copy runs in §3 are
the equality proof. Scripts: `~/.cache/p2a/{bench.py,bench2.py,ckpt_equiv.py}`, outputs in `~/.cache/p2a/{bench,ceq}/`.

## 5. Suites

Every `tests/test_*.py`, one isolated pytest process per file (`~/.cache/p2a/suite.sh`), 600 s cap, 3 GB memory cap,
`PYTHONHASHSEED=0`, golden checkouts as P1 used them (read-only). Logs and summaries are in `~/.cache/p2a/suite/{base,branch-off,branch-on}/`.
- **base**: a clean worktree of `6a85357` (`/home/josh/worktrees/ng-nodestore-p2a-base-6a85357`, 153 files).
- **OFF**: this branch, default (154 files incl. the new one).
- **ON**: this branch with P1's host-switch hook loaded through a `.pth` in `~/.cache/p2a-venv-on`, so every Graph is a native store.
- The interpreter and wheel are the same for all three. Base began on the first build: P2a is additive, and base code never calls it.

| Pass | passed | failed | errors | new FAILED/ERROR ids vs base |
|---|---:|---:|---:|---|
| base | 3,294 | 68 | 6 | — |
| OFF | 3,461 | 67 | 6 | **none** |
| ON | 3,460 | 68 | 6 | **1**: `test_graph_substrate_race.py::test_build_adjacency_no_race_under_concurrent_step_mutation` |

- **The one ON difference is not from this change.** It is a 30 s wall-clock assertion (a reader thread must finish
  5,000 `_build_adjacency` calls while a writer churns the synapse store under `_step_lock`) and touches no P2a code.
  Re-run twice on base and twice ON at load 9–11, it **failed all four times on both**; P1 also recorded it failing
  in all three of its passes. So it is load-dependent and pre-existing, and passed on base once by chance.
- **Other differences, all in the branch's favour:**
  - `test_cc_host_pith_telemetry::test_fsynced_samples_survive_sigkill` failed once on base only (a SIGKILL timing test).
  - `test_tonic_no_heuristic.py` was memory-killed on base and passed (18) OFF and ON.
  - The extra passes are the new file (148) plus those 18.
- **Not covered by any pass — the same on all three.** Seven files exceed the 3 GB cap and were OOM-killed inside
  their own scopes on every pass: `test_auto_knowledge`, `test_ces`, `test_coordinator`, `test_et_modules`,
  `test_openclaw_hook`, `test_tonic_prefetch`, and `test_tonic_no_heuristic` (base only). `test_snn` hit the
  600 s cap (#754).
  - Raising the cap next to the 4.8 GB live daemon would recreate the 18:54 earlyoom kill (§7), so it was not done.
  - `test_ces.py` covers `ActivationPersistence`, which this branch changes. Its lighter classes (`-k "Activation or
    Persistence or Surfacing or Monitor or Config"`, 44 tests incl. every `TestActivation*`) were run separately under
    the cap: base and ON give the identical 42 passed / 2 failed (`TestSurfacingQueue::test_decay_removes_weak_items`,
    `TestCESMonitorCoordinator::test_get_health`, pre-existing).
- **Pre-existing on base, identical ids on OFF and ON** (not investigated, outside this lane): 29 files, 74 ids. They
  match P1's list (`test_cc_bind_atomic_904`, `test_cc_drain_pacing_seam`, `test_cc_dual_pass`, `test_ces`,
  `test_tonic_habituation`, `test_surface_resolver`, `test_migration`, …), plus `test_cc_host_pith_telemetry` (flaky).

## 6. Not proven / left in Python, with reasons

- **`ratio ** scaling_factor`** stays in Python (libm `pow` vs CPython `float.__pow__` unproven). Consequence on an
  exception path only: if `**` raises (an `OverflowError`, which needs `scaling_factor ≫ 1`), the native path has
  already updated every node's excitability, while the loop stopped at that node. Nothing else differs.
- **Threshold-pass speed** (§4): a cached row-aligned target vector would be ~1 ms, but its exactness would depend
  on nobody mutating `_degree_targets` in place. Needs a ruling.
- **Reader vectorisation** (Tonic scan, capture): not attempted; the per-node Python arithmetic would have to move
  to numpy with Python's `max(0, x)` NaN semantics and the per-node `hash()` noise kept.
- **`_refresh_degree_targets`**, the `prime_and_propagate` working-set loops, and `tonic_engine.py:384`'s
  budget-sampled scan are not P2a sites and still read `NodeRef`s.
- **D4**: P2a keeps P1's exact overflow cell. A non-canonical value disables the native pass for that whole pass
  (correct, but silently slower). The live data has none.

## 7. Test isolation — punchlist #1037 audit

**What actually killed the daemon on 2026-10-05 18:54:29 was earlyoom, not a test signal.** Journal
(`journalctl`, system):

    18:54:28 earlyoom: mem avail 1307 of 14929 MiB (8.76%), swap free 9.83% — at or below SIGTERM limits
    18:54:28 earlyoom: sending SIGTERM to process 343907 "python": badness 961, VmRSS 5574 MiB
    18:54:29 earlyoom: sending SIGTERM to process 338727 "python3": badness 925, VmRSS 1988 MiB   <- the daemon
    18:56:50, 18:58:51 cc-ng-service: "CC service refused: startup memory precondition not met" (working as designed)
    19:00:52 daemon started again

- **The process earlyoom killed first was a test.** The P1 lane's base-pass log for `tests/test_openclaw_hook.py`
  (`/tmp/nsp1/suite/base/test_openclaw_hook.py.log`) was last written 18:54:35 and ends `rc=143` (SIGTERM).
- P1 recorded `test_openclaw_hook` and `test_auto_knowledge` ending rc 143 in all three of its passes.
- In this lane's harness, **seven test files** grew past the 3 GB per-process cap on every pass: `test_auto_knowledge`,
  `test_ces`, `test_coordinator`, `test_et_modules`, `test_openclaw_hook`, `test_tonic_prefetch`, and sometimes
  `test_tonic_no_heuristic`. Each was OOM-killed **inside its own cgroup scope**: 17 kernel memcg kills between
  00:32 and 02:52, each at anon-rss ≈ 3.05 GB. earlyoom fired **0** times over the same window, with the live daemon
  running throughout.
- **Cause:** those files grow to several GB. Under machine-wide memory pressure earlyoom picks the largest process,
  then the next: the daemon (2 GB at the time) was second.

Static audit (`os.kill`, `signal`, `daemon.pid`, `pidfile`, `~/.claude/plugins/neurograph`, `expanduser`,
`Path.home()`, `systemctl`, `pkill`, `.kill()`, sockets), tests at `6a85357`:
- `tests/test_cc_host_pith_telemetry.py:240,244`: SIGKILLs **its own** `Popen` child. Cannot reach the daemon.
- `neurograph_rpc.py:5664 _start_http_sidecar` **SIGTERMs whatever process holds TCP 8850** (`_find_pid_on_port`
  via `ss`).
  - Reachable only from `_handle_bootstrap_once` after a successful `topology_owner.claim()`.
  - The two test files that call bootstrap (`test_coordinator.py:88/102`, `test_bootstrap_singleflight_430.py`)
    either return early (`_memory` set), mock `topology_owner` to refuse, or execute a fake body.
  - So **no test reaches it today**, but any future test that bootstraps for real would SIGTERM a live 8850
    holder. Nothing listens on 8850 on the laptop now (the daemon uses `daemon.sock`). Flagged, not fixed:
    production code, out of lane.
- `test_cc_deposit_step`, `test_cc_drain_pacing_seam`, `test_cc_recall_unification`, `test_cc_pith_off_budget_812`
  import `~/docs/scripts/cc-ng-daemon.py` (via `expanduser`).
  - Its `main()` is `__main__`-guarded, so no pidfile or socket is touched.
  - But importing it inserts **`~/NeuroGraph` (the main checkout) at `sys.path[0]`**, so later imports in that
    pytest process can resolve to the live checkout's modules instead of the worktree's. Cross-checkout
    contamination, not a signal. Flagged.
- `test_cc_drain_hold_on_failure.py:110`, `test_cc_bind_atomic_904.py:434`, `test_nodestore_p1.py`,
  `test_rust_hotpaths_*`: explicit live-path refusals. Good.
- `topology_owner.py` writes `~/NeuroGraph/data/checkpoints/.topology_owner.pid`. Mocked wherever tests reach it.

**What this lane did** (no test signals the real daemon, so there was no test to fix on the branch). Every suite
and bench ran through `~/.cache/p2a/run-isolated.sh`:
- a fresh scratch `HOME` (with read-only links for `~/docs`, `~/NeuroGraph`, `~/Elmer`) and a scratch `ET_TRACTS_DIR`;
- `bwrap` with a **private PID namespace and `/proc`** (verified: `os.kill(<daemon pid>, 0)` → `ProcessLookupError`);
- **tmpfs over `~/.claude/plugins/neurograph` and `~/.et_modules`** (the live socket, pidfile and checkpoints are
  invisible), and read-only binds over `~/NeuroGraph`, `~/docs`, `~/Elmer` and the live trial worktree;
- a `systemd-run --user --scope -p MemoryMax=3000M -p MemorySwapMax=0` cap per process, so a runaway test dies in its
  own scope instead of pushing earlyoom onto the daemon;
- a wait while `MemAvailable` < 3 GB; `nice -n 10`; at most 2 workers.

`207fc1f` (commons persist guard, `cc-laptop-commons-testiso-20260930`) is **not in the trial**. It guards only
`Commons.persist` to `~/NeuroGraph/data/checkpoints/commons.msgpack`. Under the scratch HOME plus the read-only bind
that path is already unreachable, so it was not cherry-picked here: adding it to one side would also skew the
base/branch suite comparison. Landing it on the trial is a separate item. Recommended follow-ups (punchlist):
(a) find out why those seven files (above all `test_openclaw_hook.py`) reach several GB, since they are untested on
this laptop while the daemon runs;
(b) a repo-level per-test memory cap or this harness as the standard way to run suites on the laptop;
(c) make `_start_http_sidecar`'s reclaim refuse under pytest / a non-live HOME.

## 8. Open risks / worries

1. Protected-file merge is Josh-gated (D7). P2a changes `step()` and `HomeostaticRule.apply` code shape on BOTH
   paths. The default OFF path now calls the moved-verbatim loops as functions; the whole runs prove it equal to
   the trial tip.
2. The threshold pass and the readers miss their §9 targets (§4). P2b (STDP) is still what makes ON net faster.
3. The scaling step is slower ON (`_refresh_degree_targets` through `NodeRef`s), once per `scaling_interval`.
4. `adapt_thresholds` / `adapt_excitability` now create a `NodeRef` for every row on first use (P1's cache; ≈11K small
   objects, kept until the node is removed).
5. The exception-path non-identity of the `**` split (§6).
6. Timings were taken under load 5–7 with the live daemon running. Ratios are solid; absolutes are not.
7. Vault docs (`~/docs/modules/NeuroGraph.md`, the spec's status line) are not updated from this lane — for the
   Executive at integration.
