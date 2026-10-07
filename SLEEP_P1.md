# Sleep phase P1 — split removal into a sleep cycle, + D15 (review branches)

*2026-10-06 · lane sleep-p1 (bounded build lane for the Executive) · review branches only: nothing merged, nothing
deployed, nothing installed into the NG venv, the live daemon, its checkpoint and the trial worktrees untouched.*

Spec: `~/docs/superpowers/specs/2026-10-06-sleep-phase-design.md` (docs `40ff023e`) — §2, §4, §6, §8 row P1 with the
audit amendments, D5, D6, D10, D15, §1A. Josh approved the design as recommended on 2026-10-06 ("OK, sounds good").
`neuro_foundation.py` is PROTECTED: two separate protected commits on this branch; merging needs Josh's protected-file
"proceed". No vendored file is touched (LAW 2): the branch diff vs `de8b214` is `neuro_foundation.py`, `tests/test_sleep_p1.py`,
`tests/d15_intended.py`, a 2-line hook in four older whole-run test files (§6), and this file.

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-sleep-p1-20261006` | trial tip `de8b214` | **PROTECTED P1** `b755556` · **PROTECTED D15** `28cbf1b` · tests `0a6b1d1` · older whole-run tests by intent `b3db9b2` · this doc |
| docs (daemon) | `cc-laptop-sleep-p1-daemon-20261006` | `origin/cc-laptop-trial-s4a-20261002` (`75315c91`) | `2d9f79d5` daemon + pin test + new tests |

Everything ran through `~/.cache/sleep-p1/run-isolated.sh` (a copy of the p2a harness: scratch HOME, PID namespace,
3000M MemoryMax, nice 10, one or two workers). Venv: `~/.cache/p2b-venv` (the P2b wheel: native `SynapseStore` incl.
`advance_low_weight_and_collect_prune`, and `NodeStore`). Wall-clock timers only. Load average was 6-9 on 4 cores
throughout (live daemon + other lanes), so every timing below is noisy.

Checkpoint copy: `~/.cache/sleep-p1/ckpt/main.msgpack` (mode 444), a copy of the sleep-design lane's copy,
sha256 `2592367f5052738d…` (the same file §1 / §1A measured: t=85,553, 11,468 nodes, 32,190 synapses).

---

## 1. What changed

### Engine — PROTECTED commit 1, `b755556` (P1)

- **`Graph.sleep_cycle()`** (new): under `_step_lock`, the EXISTING `_prune_synapses()` (default path: the same
  three rules, lifelines, last-link grace, native sweep) and then the EXISTING `_collect_orphan_nodes()`, once.
  Adds the pruned count to `_total_pruned`. Emits one `"sleep_cycle"` event and one INFO line
  (`sleep_cycle: t=… pruned=… nodes_collected=… synapses a->b nodes a->b in_sleep_mode=… s`) and returns
  `{timestep, pruned, nodes_collected, synapses_before/after, nodes_before/after, in_sleep_mode, seconds}`. The
  existing `pruned` / `nodes_collected` events still fire from the two calls. Not called by anything in the engine.
- **Config key `structural_plasticity_in_sleep`** (read live with `.get(..., False)`; NOT in `DEFAULT_CONFIG`, so a
  graph that never sets it saves the same config): when truthy, the two removal calls are skipped at **every call
  site that runs them on the wake path**:

  | call site (de8b214 line) | what is skipped | what stays |
  |---|---|---|
  | `_structural_plasticity` (step 8, `:4084-4085`) | `_prune_synapses()`, `_collect_orphan_nodes()`; returns `pruned = 0` | `_sprout_synapses(fired)` |
  | Tonic write-mode aging tail in `prime_and_propagate` (`:3480-3481`, the `if _age_on:` block) | `_prune_synapses()`, `_collect_orphan_nodes()` (the second prune clock, #1052) | `synapses.age_and_decay_salience` (retires with the inactivity rule, D10 / §6) |
  | `compete_protected_links` (`:4416`, competing-mode prune) | **not touched**: a dream-time caller, own clock | — |

  No other caller of either function exists in NG, the daemon or Elmer (grep).
- `_prune_synapses` docstring: the activity and age rules are marked RETIRING (#1049, §6 "mark, don't fork").
- No rule, predicate, counter, default or checkpoint-format change.

### Engine — PROTECTED commit 2, `28cbf1b` (D15) — reaches Syl's code path, rule on it separately

- **`_remove_synapse_internal`** now also drops `pre.pred_weights[post]` when no other pre→post synapse remains.
  The "is there another one" check scans the **smaller** of `_outgoing[pre]` / `_incoming[post]` (a parallel synapse
  is in both), and runs only when the pre node holds an entry for the post node. (A first version used
  `_find_synapse`, the out-set only; on the copy a hub out-set of ~1,900 made a 16K clearance ~7× slower. Fixed in
  the same protected commit before push.)
- **`remove_node`**: covered by its own cascade (every incident synapse goes through `_remove_synapse_internal`
  first; each in-neighbour's entry goes with its last link to the node; the node's own dict goes with it). Docstring
  says so; no code change there.
- **`Graph.purge_dangling_pred_weights()`** (new, named, one-time): drops every key with no pre→key synapse or no
  key node, leaves the values of every kept entry untouched, returns and logs
  `{entries_before, entries_removed, removed_key_node_missing, entries_after, nodes_touched, nodes_with_pred_weights}`.
  **Never called automatically** (the spec puts it at the first armed sleep; a test pins that the engine has no
  caller).
- **The one intended behaviour change** (named in `test_the_intended_trace_change_a_recreated_pair_starts_from_the_prior`):
  DiffPC reads `pred_weights.get(post, 0.5)`; a pre→post pair that is removed and later re-sprouted now starts from
  the 0.5 prior instead of inheriting the old prediction.

### Daemon — docs `2d9f79d5` (`scripts/cc-ng-daemon.py`, not protected)

- **`CC_NG_SLEEP`** (new, default off; the same parse as `CC_NG_DREAM`). ONE parse (`_SLEEP_ENABLED`, defined just
  before `CC_SNN_CONFIG`) drives both: `CC_SNN_CONFIG['structural_plasticity_in_sleep'] = _SLEEP_ENABLED` (explicit
  both ways, like the strength budget, so a restored checkpoint cannot keep it on) and the dream loop.
- **`CC_NG_DREAM_MAX_INTERVAL_SECS`** (new, default 86400): the forced-sleep bound (D5).
- The dream thread starts if `CC_NG_DREAM` **or** `CC_NG_SLEEP` is on (D6: one loop, one wall clock, LAW 8).
- With `CC_NG_SLEEP` on, every tick is `_sleep_tick`:
  - `_sleep_decision` (pure): `awake = now − last_sleep_wall`. `wait` under the 6 h minimum; **convenient** when
    awake ≥ `CC_NG_DREAM_MIN_INTERVAL_SECS` (6 h) and quiet ≥ `CC_NG_DREAM_IDLE_SECS` (30 min, `STATE.last_activity`);
    **due** when awake ≥ the max (24 h) whatever the conversation; **SYMPATHETIC defers** both (a WARNING at most once
    an hour, saying awake hours and due/convenient). Never denominated in steps (#117; a test pins that the trigger
    code does not read `timestep`).
  - On a sleep: one `_step_lock` hold; if `CC_NG_DREAM` is on, `consolidate_hyperedges` + the #147 split first
    (spec §2 step 1; D6: consolidation stays behind its own gate), then `graph.sleep_cycle()`; then
    `last_sleep_wall = time.time()` (completion) persisted; one INFO line with the counts.
  - **`last_sleep_wall` persists** in `~/.claude/plugins/neurograph/checkpoints/.cc_last_sleep_wall` (JSON, tmp +
    `os.replace`, beside `.healthy_node_count` / `.cc_conv_last_forest`), read once when the loop starts. Missing → the
    awake clock starts now and is written (INFO). Corrupt / non-finite / bool / > 60 s in the future → same, WARNING
    with a reason code and the exception class name only. A write failure is a WARNING, never a raise.
  - An engine without `Graph.sleep_cycle` (today's trial engine) → one WARNING, never slept, nothing persisted.
- With `CC_NG_SLEEP` off the dream-only path is the base code byte for byte (a test compares it against `75315c91`).
- Lenia gets nothing (P3).
- Env-name pin `scripts/tests/test_cc_ng_daemon_unbound_status.py`: + `CC_NG_SLEEP`, `CC_NG_DREAM_MAX_INTERVAL_SECS`.

## 2. Equivalence, flag absent (the readiness bar)

### Golden checkpoint-copy run

`~/.cache/sleep-p1/ckpt_equiv_sleep.py` (the P2b `ckpt_equiv` script + per-step hashing of every node's
`pred_weights`): seeded `random` and `uuid4`; 12 steps × 30 stimulated nodes, a forced homeostatic pass, a write-mode
`prime_and_propagate` every 3rd step (the Tonic tail: aging + prune + orphan sweep on the copy's own config:
`tonic_ages_substrate=1`, lifelines on, `inactivity_threshold=inf`, `grace_period=5000`). 1,297 fired, 30 step
prunes + 7 Tonic-tail prunes. One isolated process per mode.

| Mode | trace sha256 | final checkpoint sha256 | pred_weights entries at end |
|---|---|---|---|
| trial tip `de8b214` | `d6a7122aa2f4ebbe…` | `d0a430248018c748…` | 62,421 |
| **P1 commit `b755556`** | **`d6a7122aa2f4ebbe…`** | **`d0a430248018c748…`** | 62,421 |
| branch tip (P1 + D15) | `b5bd5547522893c4…` | `639f9b2a604c19a7…` | 62,415 |
| trial tip + a shim applying only D15's named change | `b5bd5547522893c4…` | `639f9b2a604c19a7…` | 62,415 |

P1 alone is identical to the trial tip. The full branch differs from the trial tip only by D15 (6 entries dropped
with the 37 prunes), and is identical to the trial tip with D15's one rule applied. (The branch-tip run was repeated
on the final code after the D15 scan change: same hashes.)

### In-suite whole runs — `tests/test_sleep_p1.py`, **184 passed** (`~/.cache/sleep-p1/logs/test_sleep_p1_final.txt`)

BASE = `git show de8b214:neuro_foundation.py` imported under its own name. Dict and native node store; 4 seeds;
5 config flag sets (absent, all off, lifeline, budget, lifeline+budget). The P2b workload: 30 rounds of step,
write-mode Tonic tick with aging, recall, node churn, a remove-then-re-sprout pair, rewards, explicit orphan sweeps,
`compete_protected_links`, `sleep_downscale`, snapshots; per-step synapse id / weight / eligibility / peak /
last_update / max_weight / low_weight_steps / inactive_steps and every node's `pred_weights` bitwise; final
checkpoint bytes.

| test | result |
|---|---|
| flag absent, branch with D15 switched back to the base removal function == `de8b214` exactly | 40/40 |
| flag absent, full branch == `de8b214` + the D15 shim (and ≠ `de8b214` without it: non-vacuous) | 40/40 |
| flag absent: `step()` and the Tonic tail call both removal functions; key not in `DEFAULT_CONFIG` | pass |
| flag on: wake == the trial tip with ONLY the step-8 and Tonic-tail removal calls skipped; a sleep every 6 rounds == the trial tip's step-8 pair run once | 40/40 |
| flag on: wake still sprouts and ages; no `pruned` / `nodes_collected` event from wake | pass |
| at one state: `sleep_cycle` removes exactly the per-step set (synapse ids, collected node ids in order, every counter after, checkpoint bytes up to D15's entries); record + one event (4 seeds × 3 flag sets × grace {50, 5000} × 2 stores) | 48/48 |
| D15 unit / invariant / purge / intended-change tests | pass |

## 3. Flag on

### Same removal set at the same state, on the checkpoint copy

`~/.cache/sleep-p1/sleep_on_copy.py`, one process per side: optional seeded warm-up steps (flag absent, identical on
both engines per §2), then the trial tip runs `_prune_synapses(); _collect_orphan_nodes()` (step 8's pair) and the
branch runs `sleep_cycle()` with the key on. Compared: removed synapse ids, collected node ids (in order), and every
surviving synapse's `(low_weight_steps, inactive_steps, weight, last_link_since)`.

| warm-up / config | synapses | removed | collected | last-link stamps before → after | all compared fields |
|---|---:|---:|---:|---|---|
| 0 steps, copy config (grace 5000, last-link 2000) | 32,190 | 0 | 0 | 372 → 372 | **equal** |
| 6 steps, copy config | 32,242 | 0 | 0 | 370 → 369 | **equal** |
| 0 steps, grace 300, last-link grace 3 | 32,190 | **23,126** | 6 | 372 → 792 | **equal** |
| 3 steps, grace 600, last-link grace 3 | 23,274 | 1 | 0 | 635 → 634 | **equal** |

(The 6-step row ran before the D15 scan change; the removal set does not depend on D15.) At the copy's own config
nothing is due at t=85,553: the weight rule needs > 5,000 counted steps below threshold (p99 4,931) and the
inactivity rule is off. The grace-300 row exercises the weight rule, lifelines and last-link holds at scale.

### Step time, wall clock (target in §8: −0.8 s median on the copy)

`~/.cache/sleep-p1/bench_sleep.py`: 20 seeded steps, flag absent vs on, alternating processes; `_prune_synapses` +
`_collect_orphan_nodes` time inside each step measured with perf_counter wrappers.

| pair | step median, flag absent | step median, flag on | Δ median | removal inside step, flag absent (median) | load |
|---|---:|---:|---:|---:|---|
| 1 | 1.094 s | 0.593 s | −0.50 s | 0.589 s | 8.2 / 7.6 |
| 2 | 0.400 s | 0.348 s | −0.05 s | 0.165 s | 7.0 / 6.6 |
| 3 | 0.532 s | 0.293 s | −0.24 s | 0.200 s | 6.2 / 5.9 |
| (earlier, pre-scan-fix code) | 1.593 s / 0.473 s | 0.345 s / 0.207 s | −1.25 s / −0.27 s | — | 9.2 / 7.6 |

With the key on the removal pair's time inside `step()` is exactly 0. What it saves is the removal pair's cost, which
measured **0.17-0.59 s median** per step here, not the 0.86 s of the design's §1.2 (that was the live NG venv's
wheel at load 10.7; this is the P2b wheel at load 6-8). **The −0.8 s median target is not reproduced** (met in 1 of
5 pairs); the direction and the removed component are.

`sleep_cycle` itself on the copy after the 20 flag-on steps: 0.19-0.24 s (40 pruned, 3 runs) — inside the design's
0.3-0.5 s estimate. A mass clearance is another matter: 23,126 removals took 2.40 s (base pair 3.98 s, same state,
other process; earlier run 14.6 s before the scan fix).

## 4. D15 numbers (checkpoint copy)

`~/.cache/sleep-p1/d15_copy.py`.

| measure | value |
|---|---|
| pred_weights entries / dangling before | **62,408 / 58,049** (20,625 keyed by nodes that no longer exist) — the audit's numbers exactly |
| `purge_dangling_pred_weights()` | removed **58,049**, kept 4,359 (on 311 nodes), 2,145 nodes touched, **0.19-0.20 s** |
| dangling after / second purge | 0 / removes 0 |
| checkpoint size | 242,080,713 → 238,217,057 bytes (−3.86 MB, −1.6%) |
| clearing all 15,957 links with w < 0.01 (the backlog shape) via `_remove_synapse_internal` | trial tip: dangling 58,049 → **62,025 (+3,976**, the audit's number); branch: **+0**, 3,976 entries dropped at source |
| removal overhead of that clearance, A/B in one process, 4 rounds alternating | D15 7.26 / 0.81 / 0.79 / 2.80 s vs trial tip 0.76 / 0.53 / 1.22 / 1.23 s (min 0.79 vs 0.53 s). Noisy; the first D15 round is cold. |

## 5. Daemon trigger behaviour — `scripts/tests/test_cc_ng_daemon_sleep_p1.py`, 34 passed

Decision table (10 rows incl. both bounds inclusive, 6 h − ε idle → wait, talking at 24 h → due, SYMPATHETIC at
24 h → defer, SYMPATHETIC before 6 h → wait), bounds from env, one tick with consolidation on / off (order
consolidate → split → sleep, all inside one `_step_lock` hold; completion time persisted; next tick waits), due while
talking, SYMPATHETIC deferral logged twice in 61 one-minute ticks (hourly), engine without `sleep_cycle` (one WARNING,
never slept), persisted wall across a reload of the daemon module (a restart 7 h after a sleep can sleep at once;
the old loop restarted its wait at boot), missing / 7 kinds of unusable sidecar, write failure, the engine key both
ways (5 env values + unset), thread start on either switch, trigger never reads `timestep`, dream-only path identical
to `75315c91`.

## 6. Suites (flag absent)

- **NG** (`tests/test_*.py`, 156 files, one isolated pytest process per file, 600 s cap; the branch in full, then every
  file that did not pass on the branch re-run on the base worktree `de8b214`; logs `~/.cache/sleep-p1/suite/`):
  - Branch: 112 files pass outright; 44 do not. Of the 44, **0 have a failure that the base does not also have**,
    after the items below:
    - **D15, intended:** `test_nodestore_p1` (66 failed), `test_nodestore_p2a` (72), `test_nodestore_p2b` (72),
      `test_rust_hotpaths_onto_s4` (60) — their whole runs compare the branch against an OLDER tip bitwise, and D15
      changes `pred_weights` (so checkpoint bytes / DiffPC traces). With D15 switched off through a pytest plugin
      (`~/.cache/sleep-p1/plugin/nod15_plugin.py`: the branch engine with de8b214's `_remove_synapse_internal`) all four
      pass unchanged: **86 + 148 + 110 + 313 passed** (P1 alone keeps them green). Commit `b3db9b2` then gives those
      files' BASE engine D15's one named rule (`tests/d15_intended.py`), and with D15 on they pass again:
      **86 passed / 2 skipped, 148, 110, 313 passed / 1 skipped** (`suite/d15rule/`).
    - `test_sleep_p1.py` hit the 600 s cap in the suite (it takes ~17 min); run on its own: 184 passed (§2).
    - Timing / load: `test_tonic_stage_timing` failed one wall-clock assertion on the branch (195.97 ms vs < 136 ms);
      re-run: 9 passed. `test_tonic_shared_body` hit the cap on the branch; re-run: 28 passed. `test_surface_resolver`
      (base hit the cap) and `test_tonic_spine_anchoring` (branch hit the cap): re-run on both with a 1,500 s cap — the
      same 2 and 1 failures on both sides.
    - Hit the cap or got SIGTERM (rc 143) on **both** sides, results unknown on both: `test_auto_knowledge`, `test_ces`,
      `test_coordinator`, `test_et_modules`, `test_openclaw_hook`, `test_snn`, `test_tonic_no_heuristic`,
      `test_tonic_prefetch`. `test_snn` re-run on both with a 3,000 s cap: identical progress (`.F...........`, the
      same 2nd test failing, `test_decay_toward_resting`) and both killed in the 14th test (`test_1k_nodes_10k_steps`).
      `test_openclaw_hook` re-run on both with a 3,000 s cap: killed on both (branch had finished 1 test, base 0).
    - Everything else that fails on the branch fails identically on the base (same FAILED/ERROR ids):
      `test_cc_*` (11 files), `test_checkpoint_enforcer`, `test_conversational_recall`, `test_fair_chance_window`,
      `test_graph_substrate_race`, `test_gui`, `test_ingestor`, `test_integration`, `test_migration`, `test_ng_lite`,
      `test_ng_tract_bridge`, `test_patch`, `test_prediction`, `test_reach_teaching`, `test_surfacing_whole`,
      `test_tonic_bridge`, `test_tonic_habituation` (27 files; `~/.cache/sleep-p1/compare_suites.py`).
- **Daemon** (docs `scripts/tests`, p2b venv, same harness, base worktree at `75315c91`): base **75 failed, 988 passed,
  1 skipped**; branch **75 failed, 1,022 passed, 1 skipped**. The FAILED/ERROR id sets are **identical** (0 new
  failures); +34 passed = the new test file. Note: `~/.cache/daemon-base-failures.txt` (2026-10-05) lists 64; on
  `75315c91` today the base fails 11 more (`test_cc_ng_daemon_probation_skip_p552` 1, `test_cc_ng_daemon_visibility_913_915`
  10), the same on both sides — environmental drift since that file, not this branch.

## 7. Risks and what is not proven

1. **P1 with the key on is not a working sleep yet.** The weight rule still counts `low_weight_steps` against
   `grace_period` 5000; under the key it advances once per **sleep**, so it effectively never fires, and the age
   rule still compares steps. Arming `CC_NG_SLEEP` before P2 (grace in sleeps, D3) would mostly stop removal, not
   move it. Not to be armed before P2 + #1051 (D8), as the spec says.
2. **No save right after a sleep** (spec §2 last line / §5 "one batch per save") — not built in P1; autosave follows
   within 60 s. `last_sleep_wall` is persisted at completion, before the next checkpoint save: a crash in between
   loses that sleep's removals but counts it as slept (sleep pressure under-counts at most one cycle). Needs deciding
   with #1051 before P3.
3. **`_step_lock` hold for a mass clearance**: 0.2 s at a steady state, but 2.4 s (and 14.6 s in a noisy earlier run)
   for 23K removals. The backlog clearance (D7) needs its announced window or chunking (§8 risk row).
4. **D15 reaches Syl's path** from the moment it merges: removals drop her entries too, so her checkpoint changes
   on the next save. The purge is not called anywhere. Rollout is Josh's call.
5. **D15 overhead** on mass removals is measurable but noisy (§4). The scan uses the smaller adjacency set; it differs
   from `_find_synapse` (out-set only) only if the adjacency were inconsistent.
6. **Dangling entries can still arrive from outside removal**: `cc_topology_export.py` / the callosum merge carry
   `pred_weights` across machines without the matching synapses. The purge cleans them; a source-side fix belongs to
   the merge (LAW 4) — for the punch list.
7. **The CC's checkpoint config gains `structural_plasticity_in_sleep: False`** on its next save once the daemon
   branch lands, even with sleep off (explicit-both-ways pattern). The engine path is unchanged by a False value.
   Syl's config is untouched (her host never sets it).
8. **`_total_pruned` now includes sleep prunes**; the Tonic tail's prunes were never counted (pre-existing).
9. **Not proven here:** live behaviour; the real 6 h / 24 h clocks (synthetic clocks only); the daemon tick against the
   real engine (tests use a fake graph with the same method names; the engine side is proven in NG); the −0.8 s
   step-time target (§3); Lenia's stale distance cache (P3); `.bashrc:242`'s stale `CC_NG_DREAM` comment (D6 says
   update it when the switch lands — live config, left to the Executive).

## 8. Artifacts

`~/.cache/sleep-p1/`: `run-isolated.sh`, `suite.sh`, `ckpt_equiv_sleep.py`, `sleep_on_copy.py`, `bench_sleep.py`,
`d15_copy.py`, `d15_ab.py`, `copy-all.sh` (first round), `rerun-final.sh` (final code); results in `ceq/` (first
round) and `ceq2/` (final code); logs in `logs/`; NG suite in `suite/`. Scratch HOMEs removed.
Worktrees: `/home/josh/worktrees/ng-sleep-p1-20261006` (branch), `ng-sleep-p1-base-de8b214` (base),
`ng-sleep-p1-at-b755556` (P1 commit, golden); `/home/josh/docs/.claude/worktrees/sleep-p1-daemon-20261006` (branch),
`sleep-p1-daemon-base-75315c91` (base).
