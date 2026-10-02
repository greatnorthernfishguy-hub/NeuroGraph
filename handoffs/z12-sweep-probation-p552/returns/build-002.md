# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, dispatch #15035; re-scoped by Exec P563 / Chief-003 Addenda 3-4) — RETURN build-002 (docs-only)
#   What: the return for the RE-SCOPED round 2 (covers BOTH repos): final commit hashes in order, the framing sentence, the generic names and the exact two `.bashrc` export lines,
#   the env-read design answer with exact lines, the AST proof for the protected diff, the unregistered-sweep and graduation differentials, the five-stall table, the independence of the
#   two knobs, every run (START loads, env lines SEEN), the seed table, the mutants, what ran / did not, not verified, follow-ups, and what content of NG-3 / NG-3b / NG-4 / D-2 is superseded.
#   Why: Exec P563 re-framed P556/P562: the window is canonical, host-neutral machinery in `neuro_foundation.py`; the host keeps only the switch and the heartbeat stamp.
#   How: written from the committed trees and the recorded run outputs (`/tmp/p561_results.jsonl`, `/tmp/p561_s_run.txt`); nothing here is a claim the run outputs do not back.
#   Framing: shared machinery being TESTED FIRST on the CC, not CC-specific code; rollout to other NeuroGraphs (Syl's) is Josh's call, LAW 8 gate per host.
# -------------------

# build-002 — SWEEP-PROBATION round 2, RE-SCOPED (Exec P563): the fair-chance window is CANONICAL, HOST-NEUTRAL machinery

**FRAMING (Josh):** the fair-chance window is **shared machinery being TESTED FIRST on the CC, not CC-specific code** — the pioneer implementation of canonical §8.13 arrival protection. Rolling it out to other NeuroGraphs (Syl's) is **Josh's call, with a LAW 8 gate per host**.

Nothing is pushed or merged. Both branches are local. Z12 holds the fresh-leg FULL PAIR until this is read.

## 1. Final commit hashes, in order

NG worktree `z12-sweep-probation-p552-20261002` (branch `cc-laptop-sweep-probation-p552-20261002`), base `origin/main` `b5e476863cc069a29ec482959b4f9465f2ea4ccf`:

| # | hash | what | status |
|---|---|---|---|
| NG-1 | `c98d06283dc73948c90e851cb10c274d1e143c72` | round 1: `probation_advances` in the organism | history (round 1, superseded by NG-3 then NG-3c; see §13) |
| NG-2 | `0b26f61426f161d3f1057a9931d4c0a240947fe0` | round 1: sweep body spares while probation open | history (superseded) |
| ret-1 | `5e2d2e031361ec83ff3c9f917bd7ba3e8cfa5951` | `build-001.md` | history (its fact (ii)/F6/§3(ii) about arrivals is WRONG, see §12) |
| NG-3 | `3ccc74915e97912dabe6fddfd9af73f27aef52b8` | organism owns the window, step unit, heartbeat | SUPERSEDED in part (§13) |
| NG-3b | `7d3438112d9bbff0a584ebd401fc24643ac5b04c` | `CC_PROBATION_STEP_WINDOW` dedicated knob | SUPERSEDED (§13) |
| NG-4 | `cdbdbe22e77902d951cc811f95c6b4a479dc769d` | sweep reads a host-registered predicate | **SUPERSEDED by NG-4'** (kept in history, NOT amended) |
| **NG-4'** | `8a76bed312dcb63624e8de1d452dfa6a3aa706b8` | **PROTECTED, ALONE**: canonical window inside `neuro_foundation.py` | **FINAL** |
| **NG-3c** | `81489b0bf85210767e0379d7bc33a3bf79427901` | organism keeps only the host's share; calls the canonical helpers; export bans the generic keys; tests | **FINAL** |
| **build-002** | (this file's own commit, on top of NG-3c; `git log -1` in §14) | docs only | **FINAL** |

Daemon worktree `daemon-probation-warn-20261002` (branch `cc-laptop-daemon-probation-warn-20261002`):

| # | hash | what | status |
|---|---|---|---|
| D-1 | `905618ee29d479e07e9b86b18999c87d787b12c9` | round 1: register the predicate; make a skipped decrement visible | history (the visibility part stands) |
| D-2 | `4baa731fbad80183672494d2b1c7e7d824eedf6c` | arm the organism heartbeat, register the organism predicate, K knob `CC_PROBATION_HEARTBEAT_CYCLES` | **SUPERSEDED by D-3** (§13) |
| **D-3** | `ca0a8330bcddf9cdc38fd932ba0db0716dec21db` | **FINAL**: ONE canonical `enable_fair_chance_window(...)` call; the host reads the generic env knobs | **FINAL** |

NEW commits only: nothing above was amended, rebased or rewritten. NG-4' is the only commit that touches `neuro_foundation.py` and it touches nothing else (§4).

## 2. The generic names (nothing `cc`/`CC` in canonical executable code)

| thing | name |
|---|---|
| node field: the step counter | `fair_chance_steps_remaining` |
| node field: the clock value at the node's last decrement | `fair_chance_last_timestep` |
| per-graph config (instance attribute, NOT serialized, NOT in `__init__`, NOT a `DEFAULT_CONFIG` key) | `Graph._fair_chance_cfg`, a dict with keys `window_steps`, `max_age_s`, `excluded`, `clock`, `stamp`, `stale_logged` |
| host registration | `Graph.enable_fair_chance_window(window_steps, heartbeat_max_age_s=None, excluded_creation_modes=(), clock=None)` |
| host deposit helper | `Graph.fair_chance_stamp(node)` |
| host advancer helper | `Graph.fair_chance_advance(node)` |
| host completion stamp | `Graph.fair_chance_heartbeat_stamp()` |
| canonical internals | `Graph._in_fair_chance_window(node)`, `Graph._fair_chance_heartbeat_fresh(cfg)`, `Graph._is_finite_number(v)` (staticmethod) |
| env var: window size (graph STEPS) | `NG_FAIR_CHANCE_WINDOW_STEPS` (positive int, default **10**) — replaces `CC_PROBATION_STEP_WINDOW` (NG-3b) |
| env var: heartbeat tolerance (autosave CYCLES) | `NG_FAIR_CHANCE_HEARTBEAT_CYCLES` (positive int, default **5**) — replaces `CC_PROBATION_HEARTBEAT_CYCLES` (D-2) |
| daemon module constants | `FAIR_CHANCE_WINDOW_STEPS`, `FAIR_CHANCE_HEARTBEAT_CYCLES`, `*_DEFAULT`, `_FAIR_CHANCE_*_PROBLEM`, `_read_positive_int_env` |
| host (organism, CC code, a CC name is allowed there) | `PROBATION_UNADVANCED_CREATION_MODES = ("ingested",)` and `probation_population(node)` |

I reconsidered `CC_PROBATION_STEP_WINDOW` as asked: it is renamed to `NG_FAIR_CHANCE_WINDOW_STEPS` (the `NG_` prefix is the one `scripts/cc-ng-service.py::canonical_exports` reads; the quantity is the window's, not the CC's, not "probation"). `CC_CONV_PROBATION_PERIOD` stays GRADUATION-ONLY and is untouched.

### The exact two `.bashrc` export lines (S4)

```
export NG_FAIR_CHANCE_WINDOW_STEPS=10
export NG_FAIR_CHANCE_HEARTBEAT_CYCLES=5
```

Single-line literals (the reader rejects computed values). They are also in the D-3 changelog and pinned against the code defaults by `test_the_s4_export_lines_in_the_changelog_match_the_code_defaults` (mutant `d-s4-line-wrong` KILLED). **I did not read or write `~/.bashrc`** (not checked), `scripts/cc-ng-service.py` is untouched, `canonical_exports` was not executed. Absent variables fall back to the same defaults silently, so the lines are a documentation/consistency step, not a prerequisite.

## 3. Where the environment is READ (the design answer, with exact lines)

**The HOST reads it; the canonical code reads none.** Daemon `scripts/cc-ng-daemon.py` at `ca0a8330`:

- `:1021` `def _read_positive_int_env(name, default, environ=None)` — pure: absent -> `(default, None)` silently; set but invalid (empty, non-integer, bool-ish word, `<= 0`) -> `(default, "<which rule failed>")`; never raises, never logs.
- `:1039` `FAIR_CHANCE_WINDOW_STEPS, _FAIR_CHANCE_WINDOW_STEPS_PROBLEM = _read_positive_int_env('NG_FAIR_CHANCE_WINDOW_STEPS', FAIR_CHANCE_WINDOW_STEPS_DEFAULT)` (once, at import)
- `:1040` `FAIR_CHANCE_HEARTBEAT_CYCLES, _FAIR_CHANCE_HEARTBEAT_CYCLES_PROBLEM = _read_positive_int_env('NG_FAIR_CHANCE_HEARTBEAT_CYCLES', FAIR_CHANCE_HEARTBEAT_CYCLES_DEFAULT)` (once, at import)
- `:1228-1232` in `init_ng`: one WARNING per invalid knob naming the variable and the rule (logging is configured by then; no import-time side effect).
- `:1234-1238` the hand-over, in its own try OUTSIDE the organism-layer bootstrap:
  `from cc_ng_organism import PROBATION_UNADVANCED_CREATION_MODES as _unadvanced_modes`; `_heartbeat_max_age_s = FAIR_CHANCE_HEARTBEAT_CYCLES * AUTOSAVE_INTERVAL` (`AUTOSAVE_INTERVAL = 60.0` at `:817`);
  `self.ng.graph.enable_fair_chance_window(FAIR_CHANCE_WINDOW_STEPS, heartbeat_max_age_s=_heartbeat_max_age_s, excluded_creation_modes=_unadvanced_modes, clock=time.monotonic)`.

Canonical: `neuro_foundation.py` has no `os.environ` / `getenv` / `environ` anywhere in its diff, no `import os` added, no `time` import (the clock is passed in), **no module-level addition** (§4), no canonical DEFAULT window-size constant (the registration requires a value). Pinned by `tests/test_fair_chance_window.py::test_static_the_canonical_helper_reads_no_environment_anywhere_in_its_executable_code` and `...::test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else`. Mutant `n-env-read-added` (an `os.environ` read inside `enable_fair_chance_window`) KILLED; the organism reads no window variable either (`test_static_the_organism_reads_no_environment_for_the_window_and_names_no_window_knob`; mutant `o-window-env-in-organism` KILLED).

Registration order and atomicity: `enable_fair_chance_window` validates FIRST (raises `ValueError`, never partially applies), stamps the heartbeat NOW with the passed clock, and attaches `_fair_chance_cfg` LAST (mutant `n-attach-before-stamp` KILLED by `test_a_clock_that_raises_registers_nothing`). The daemon registers because its advance runs on its OWN autonomic 60 s `_autosave_loop` (LAW 8, P555, #971); Syl's sidecar does not and registers nothing, so her sweep is today's. An NG that predates the method, an organism lacking the constant, or a raising registration: nothing registered, ONE WARNING with the exception CLASS NAME only (never `str(exc)`), the daemon starts (tests in `test_cc_ng_daemon_probation_skip_p552.py`; mutant `d-registration-warning-str-exc` KILLED).

## 4. AST proof for the protected diff (NG-4', alone)

```
$ git diff --stat b5e47686 8a76bed3 -- neuro_foundation.py
 neuro_foundation.py | 187 +++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 187 insertions(+)           (0 deletions; 3 hunks, all inside `class Graph`)
$ git show --stat 8a76bed3                    -> 4 files: neuro_foundation.py (+187) and THREE TEST files (test_fair_chance_window.py added, test_cc_sweep_probation_p552.py removed,
                                                 test_cc_fair_chance_window_p561.py edited); the ONLY non-test file is neuro_foundation.py (no organism / daemon / export change)
```

The recorded `-s` output (`/tmp/p561_s_run.txt`, line printed by `test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else`, gated, seed 0, START load 1m=3.13 5m=2.99):

```
[ast-proof] base-functions=89 added=7 ['Graph._fair_chance_heartbeat_fresh', 'Graph._in_fair_chance_window', 'Graph._is_finite_number', 'Graph.enable_fair_chance_window',
 'Graph.fair_chance_advance', 'Graph.fair_chance_heartbeat_stamp', 'Graph.fair_chance_stamp'] differing=1 ['Graph._collect_orphan_nodes']
 module-level-non-def-statements=23 identical=True module-level-defs-added=0
```

Read plainly: **7 methods ADDED** (the list above); **exactly ONE existing function differs, `Graph._collect_orphan_nodes`** (the docstring note + the sweep body, below); **every one of the 23 module-level non-def statements is identical to base; no module-level def/class/assignment/import added.** I added **no module-level statement**, so there was nothing to STOP and name before committing (Addendum 4 item 1). The test fails on any other function differing (mutant `n-other-function-changed`, `n-module-level-added` KILLED; see §11 for which test killed which).

The changed sweep body, in order: the structural orphan list is built exactly as before (no synapses, no hyperedge membership, past grace, not identity-protected). Then, only if `_fair_chance_cfg` is set and orphans exist, each orphan is passed to `_in_fair_chance_window` (consulted LAST); a node it spares is skipped; a check that RAISES is NOT spared (swept as without the exemption) and ONE WARNING carries the count and exception class names only. `_unbound_nodes`, `held_unbound_nodes`, `whole_graph_guard`, `step()`, `inject_reward`, `Graph.save/load`, `_prune_synapses`, `_is_identity_protected`, `__init__`, `DEFAULT_CONFIG` are untouched. No `cc`/`CC` token in the executable lines (`test_static_no_cc_token_in_the_executable_code_of_the_new_and_changed_functions`; mutant `n-cc-token-added` KILLED). The other protected files and the six vendored files are untouched.

## 5. UNREGISTERED graph == today's sweep (differential vs base)

`tests/test_fair_chance_window.py`:
- `test_an_UNREGISTERED_graph_sweeps_EXACTLY_as_the_base_does_over_a_seeded_family_even_when_nodes_carry_window_fields`: the BASE `_collect_orphan_nodes` (extracted from `git show b5e47686:neuro_foundation.py` through `ast`) and the new one are run over 150 seeded graphs (random creation modes, random `probation_remaining` shapes including junk, random window-field values on half the nodes, identity-protected nodes, synapses, hyperedges, ages straddling grace); the removed count and the surviving node/synapse/hyperedge snapshot must be equal for every seed; the test asserts the family removes more than 100 nodes in total (not vacuous).
- `test_the_unregistered_sweep_also_emits_the_same_events_as_the_base`: the event/log emissions of the sweep are identical too.
- `test_an_unregistered_graph_never_has_an_open_window`, `test_the_helpers_are_no_ops_on_an_unregistered_graph_and_touch_nothing_else`.
- Through the host: `test_cc_fair_chance_host.py::test_int_an_UNREGISTERED_graph_driven_by_the_same_host_calls_sweeps_exactly_as_today`.
- Mutant `n-sweep-unconditional-window-read` (the sweep implicitly registers an unregistered graph) KILLED.

## 6. Graduation byte-identical (differential vs base)

`tests/test_cc_fair_chance_host.py`: the BASE `cc_update_probation` (extracted from `git show b5e47686:cc_ng_organism.py` through `ast`) against the new one, every pulse, over seeded graphs; the result (or the raised exception class) and the full node snapshot must be equal after every pulse (the clock held, stepping or going BACKWARDS): unregistered, the snapshot is compared WHOLE, so not one key may be added (`..._NOT_ONE_KEY_ADDED_...` tests); registered, the two new window fields are excluded from the comparison and every OLD field (`probation_remaining`, `probation_total`, `novelty_dampening`, `intrinsic_excitability`, the release, the #93 stamp) must still be byte-identical. Recorded `-s` lines (gated, seed 0):

```
[differential] graphs=250 odd=False registered=False old-fields-identical-every-pulse=True graphs-that-raised-identically=0   graduated-nodes=1778
[differential] graphs=250 odd=True  registered=False old-fields-identical-every-pulse=True graphs-that-raised-identically=135 graduated-nodes=595
[differential] graphs=500 odd=False registered=True  old-fields-identical-every-pulse=True graphs-that-raised-identically=0   graduated-nodes=3600
[differential] graphs=500 odd=True  registered=True  old-fields-identical-every-pulse=True graphs-that-raised-identically=265 graduated-nodes=1208
```

That is **1,500 graphs (>= 400 required)**; "odd" includes the shapes that make the base RAISE: the new advancer raises identically (the same exception class) on exactly those graphs. Mutants `o-grad-total-changed` (graduation total changed) KILLED. Also `test_population_is_byte_identical_to_the_round_1_skip_over_odd_values` for `probation_population`.

## 7. Behaviour (the rules the canonical helpers implement; each pinned in `tests/test_fair_chance_window.py`)

- Unit = graph STEPS. One decrement only if `graph.timestep` advanced since the node's `fair_chance_last_timestep`; many pulses with no step count ONCE; a clock regression (restore from an older checkpoint) resets `last` and does NOT decrement. A node that LACKS the counter is SEEDED fresh at its next advancer pass (seeding A, Z12's design call); an excluded `creation_mode` is never stamped, advanced or protected.
- Fail-safe parsing: a counter is honoured only if it is a finite number, not bool, `0 < v < inf`, with a finite `last`; anything else (None, str, negative, zero, NaN, +/-inf, bool, non-dict metadata) is today's sweep. Never protects forever.
- Heartbeat: armed with a max age and a clock; stale is a STRICT `>`; stale for EVERY node (not per node); ONE WARNING per stale episode (age and limit only); a completed pass re-arms the latch with ONE INFO. The advancer stamps ONLY as the last statement of a non-raising pass (not in a `finally`, not before the loop; mutants `o-heartbeat-before-loop`, `o-heartbeat-in-finally`, `o-heartbeat-removed`, `n-stale-ge` KILLED).
- Seeding (A) effect, said plainly: after a daemon restart every non-excluded unbound node that lacks the counter gets a fresh window of `NG_FAIR_CHANCE_WINDOW_STEPS` steps, so the first sweeps after deploy cull fewer old orphans than today until those windows close. One-condition narrowing option if wanted: seed only nodes younger than some age (not done).
- Strict-`>` boundary: with cycles of exactly the interval the exemption closes one cycle later than with `>=`; the stall tests use cycles of interval + 1 s (real cycles are sleep + work).

## 8. The five-stall table (le-062 C2 / P2), through the REAL daemon `_autosave_loop`, the REAL organism advancer and the REAL canonical `Graph`

`scripts/tests/test_cc_ng_daemon_heartbeat_p561.py` (K = 3 in the test; the registration is the host's real call shape; only the loop's collaborators are faked). In each stall: just before K cycles the node SURVIVES the real sweep; after K stalled cycles the real sweep CULLS it as today with exactly ONE heartbeat WARNING (from the canonical logger `neuro_foundation`) for the episode; a further stalled cycle adds no second WARNING; the stamp is never refreshed.

| stall | what stalls the advancer | test | before limit | after K cycles | WARNINGs |
|---|---|---|---|---|---|
| L1 | the tract drain raises every cycle | `test_a_stall_keeps_the_exemption_on_until_K_cycles_then_closes_it_and_the_real_sweep_culls_as_today[L1_drain_raises]` | spared | culled | 1 |
| L2 | graph-busy `_concurrent_lock` try-acquire refused every cycle (`continue`) | `...[L2_graph_busy]` | spared | culled | 1 |
| L3 | a `str` `probation_remaining` poison on a BOUND node aborts the real `cc_update_probation` every cycle | `...[L3_str_poison_on_a_bound_node]` | spared | culled (the bound poison persists, never self-heals) | 1 |
| L4 | a raise in `_guarded_save` (the OUTER try) | `...[L4_save_raises]` | spared | culled | 1 |
| L5 | a wedged save (a real thread blocked inside it; the injected clock advances) | `test_L5_a_wedged_save_thread_is_silent_and_after_K_cycles_of_time_the_exemption_closes` | spared | culled (the wedged thread logs nothing) | 1 |

Also: `test_the_no_fault_control_keeps_the_exemption_on_for_K_plus_three_cycles`; `test_a_recovered_advancer_reopens_the_exemption_with_one_info` (and a SECOND episode warns again); `test_a_host_registered_without_a_heartbeat_is_not_affected_by_any_stall`; `test_the_real_init_ng_against_the_real_graph_registers_the_canonical_window_with_the_hosts_values` (the real `init_ng` against the real Graph: `window_steps`, `max_age_s == K x AUTOSAVE_INTERVAL`, `excluded == ("ingested",)`, `clock is time.monotonic`, stamped NOW). Note (D-3 fix, disclosed): the stall tests had to register BEFORE depositing (as `init_ng` does), because the canonical deposit helper stamps only on a registered graph; the first run after the re-key failed 6 tests for exactly that reason and was fixed in the tests, not in the product code.

## 9. Independence of the two knobs

Window size = `NG_FAIR_CHANCE_WINDOW_STEPS` (daemon, handed to the canonical registration). Graduation = `CC_CONV_PROBATION_PERIOD` (organism `_CC_CONV_PROBATION_PERIOD`). Neither reads the other's variable; the window size never reaches the organism at all.

- Both directions and a swap, at the **host data level** (`test_cc_fair_chance_host.py::test_graduation_and_the_window_size_are_independent_in_both_directions_and_under_a_swap`, ids `window=X,graduation=Y` and `SWAPPED window=Y,graduation=X`; the values are the named constants `STEP_X`, `GRAD_Y`, no literal 10).
- Both directions at the **environment level**, in CLEAN subprocesses importing the real organism and the daemon (`test_cc_ng_daemon_heartbeat_p561.py::test_the_window_knob_and_the_graduation_knob_are_independent_in_both_directions_and_swapped`): only the window variable set -> the window moves, the organism's graduation period stays; only `CC_CONV_PROBATION_PERIOD` set -> the graduation period moves, the window stays; both set with the window LARGER than the graduation period -> each follows its own.
- Import-time, both knobs, absent / valid / invalid: `test_both_knobs_are_read_once_at_import_absent_valid_and_invalid`, `test_each_knob_is_independent_of_the_other_and_of_the_graduation_period_at_import`.
- Mutants, one per direction, KILLED: `d-env-name-window-wrong` (the WINDOW knob reads `CC_CONV_PROBATION_PERIOD`; graduation -> window) killed by `test_both_knobs_are_read_once_at_import_absent_valid_and_invalid`; `o-grad-reads-window-knob` (GRADUATION reads `NG_FAIR_CHANCE_WINDOW_STEPS`; window -> graduation) killed by `test_static_the_organism_reads_no_environment_for_the_window_and_names_no_window_knob`; plus `d-env-name-k-wrong`, `o-window-env-in-organism`, `d-window-literal`. (With `-x` a mutant is reported at its FIRST failing test, so the subprocess/data-level tests above were not each individually shown to fail on these two mutants; they ran green on the committed trees in every seed.)

## 10. Runs (every START load recorded; gate: refuse at 1m >= 6 or 5m >= 5.0, 25 s spacing, never retried while closed)

Every job went through the same wrapper: `sleep 25`, record `START load`, refuse at the gate, `env -u CC_NG_BATCH_SIZE -u CC_NG_IDLE_STEPS -u CC_NG_DRAIN_HOLD_ON_FAILURE -u NG_EMBED_REMOTE -u CC_NG_IN_TRANSIT_IDS_PATH PYTHONPATH= `, and printed the effective values and the resolved module paths before pytest. SEEN on every run (verbatim from the recorded output):

```
[effective] CC_NG_BATCH_SIZE=<unset> CC_NG_IDLE_STEPS=<unset> CC_NG_DRAIN_HOLD_ON_FAILURE=<unset> NG_EMBED_REMOTE=<unset> CC_NG_IN_TRANSIT_IDS_PATH=<unset> PYTHONPATH=[]
[module] cc_ng_organism /home/josh/NeuroGraph-worktrees/z12-sweep-probation-p552-20261002/cc_ng_organism.py
[module] neuro_foundation /home/josh/NeuroGraph-worktrees/z12-sweep-probation-p552-20261002/neuro_foundation.py
```

(`NG_FAIR_CHANCE_*` and `CC_CONV_PROBATION_PERIOD` were not set in the runner environment; the env-level tests set them themselves in subprocesses / `monkeypatch`.) For the daemon runs the daemon-under-test line printed by the files' own guards was `/home/josh/docs/.claude/worktrees/daemon-probation-warn-20261002/scripts/cc-ng-daemon.py`, and the real-NG modules came from the NG worktree path above (`[P561] real organism under test` / `real Graph under test`; `real NeuroGraphMemory loaded: no`). Campaign HEADs recorded in the results file meta: NG `81489b0bf85210767e0379d7bc33a3bf79427901`, daemon `ca0a8330bcddf9cdc38fd932ba0db0716dec21db`, both with 0 dirty paths. START loads over the whole campaign: min 1m=0.32, max 1m=5.45 (< 6), max 5m=3.95 (< 5.0). **0 gate refusals; 0 new batches; one batch.**

### 10.1 Single-session runs

- Daemon, ONE gated session over the three daemon files (`test_cc_ng_daemon_drain_pacing.py` UNCHANGED D24 file + `test_cc_ng_daemon_heartbeat_p561.py` + `test_cc_ng_daemon_probation_skip_p552.py`), seed 0: **123 passed** in 5.22 s, START 1m=2.01 5m=2.24; the files' session guards printed `real NG modules in sys.modules at session end: none`. (The first attempt, START not recorded here, failed 6 stall tests for the registration-order reason in §8 and was fixed before the D-3 commit.)
- NG `-s` run (AST proof and differential lines): three NG files, **238 passed** in 39.39 s, START 1m=3.13 5m=2.99 15m=2.46 at 2026-10-02T07:01:19Z.
- An earlier gated run of the NG-3c set (before it was committed): 311 passed, 1 failed (`tests/test_cc_topology_callosum.py::test_poincare_dir_is_rederived_locally_not_transmitted`, a `bytes` -> `float` ValueError: **pre-existing, not in my files, not touched** — see §12), START 1m=2.53; a gated run of the older sweep-related suites on the NG-4' tree: 73 passed, the same 1 failure.

### 10.2 SEED TABLE — `PYTHONHASHSEED` 0..31, one gated invocation per seed, over EVERY changed/new test file

NG: `tests/test_fair_chance_window.py` (new), `tests/test_cc_fair_chance_host.py` (new), `tests/test_cc_capture_mutations_423.py` (changed) = **238 tests per run**, on NG-3c `81489b0b` (clean). **32 / 32 PASS.**

| seed | START load 1m / 5m / 15m | UTC | result |
|---|---|---|---|
| 0 | 3.18 / 2.57 / 2.92 | 05:31:48 | PASS: 238 passed in 34.83s |
| 1 | 2.33 / 2.48 / 2.87 | 05:32:50 | PASS: 238 passed in 31.30s |
| 2 | 1.53 / 2.25 / 2.76 | 05:33:48 | PASS: 238 passed in 31.56s |
| 3 | 1.17 / 2.04 / 2.66 | 05:34:46 | PASS: 238 passed in 33.56s |
| 4 | 1.20 / 1.92 / 2.58 | 05:35:46 | PASS: 238 passed in 33.97s |
| 5 | 1.86 / 1.97 / 2.55 | 05:36:46 | PASS: 238 passed in 34.07s |
| 6 | 1.67 / 1.91 / 2.49 | 05:37:47 | PASS: 238 passed in 30.13s |
| 7 | 0.99 / 1.69 / 2.38 | 05:38:44 | PASS: 238 passed in 31.57s |
| 8 | 1.05 / 1.59 / 2.31 | 05:39:42 | PASS: 238 passed in 36.35s |
| 9 | 1.85 / 1.75 / 2.32 | 05:40:45 | PASS: 238 passed in 31.20s |
| 10 | 2.13 / 1.85 / 2.31 | 05:41:42 | PASS: 238 passed in 33.56s |
| 11 | 1.97 / 1.83 / 2.27 | 05:42:43 | PASS: 238 passed in 30.66s |
| 12 | 1.25 / 1.67 / 2.19 | 05:43:41 | PASS: 238 passed in 32.35s |
| 13 | 0.93 / 1.53 / 2.11 | 05:44:39 | PASS: 238 passed in 33.86s |
| 14 | 2.27 / 1.77 / 2.15 | 05:45:40 | PASS: 238 passed in 32.88s |
| 15 | 1.92 / 1.81 / 2.14 | 05:46:40 | PASS: 238 passed in 34.04s |
| 16 | 2.06 / 1.91 / 2.16 | 05:47:41 | PASS: 238 passed in 32.39s |
| 17 | 1.29 / 1.74 / 2.09 | 05:48:40 | PASS: 238 passed in 30.80s |
| 18 | 1.45 / 1.72 / 2.06 | 05:49:37 | PASS: 238 passed in 30.08s |
| 19 | 1.29 / 1.63 / 2.01 | 05:50:34 | PASS: 238 passed in 30.09s |
| 20 | 1.19 / 1.56 / 1.96 | 05:51:30 | PASS: 238 passed in 33.34s |
| 21 | 2.14 / 1.79 / 2.02 | 05:52:31 | PASS: 238 passed in 30.61s |
| 22 | 1.40 / 1.67 / 1.96 | 05:53:28 | PASS: 238 passed in 30.85s |
| 23 | 1.66 / 1.70 / 1.96 | 05:54:25 | PASS: 238 passed in 30.58s |
| 24 | 1.37 / 1.62 / 1.91 | 05:55:22 | PASS: 238 passed in 33.34s |
| 25 | 1.54 / 1.69 / 1.92 | 05:56:22 | PASS: 238 passed in 32.23s |
| 26 | 1.24 / 1.58 / 1.87 | 05:57:21 | PASS: 238 passed in 32.54s |
| 27 | 1.45 / 1.61 / 1.86 | 05:58:20 | PASS: 238 passed in 30.48s |
| 28 | 1.33 / 1.57 / 1.84 | 05:59:17 | PASS: 238 passed in 31.02s |
| 29 | 2.17 / 1.70 / 1.86 | 06:00:15 | PASS: 238 passed in 46.78s |
| 30 | 2.69 / 2.10 / 2.00 | 06:01:30 | PASS: 238 passed in 33.25s |
| 31 | 2.24 / 2.10 / 2.00 | 06:02:30 | PASS: 238 passed in 33.81s |

Daemon: `scripts/tests/test_cc_ng_daemon_probation_skip_p552.py` (changed) + `scripts/tests/test_cc_ng_daemon_heartbeat_p561.py` (changed) = **59 tests per run**, on D-3 `ca0a8330` (clean). **32 / 32 PASS.** (The unchanged D24 file is not in the seed set; it ran in the 123-test session above.)

| seed | START load 1m / 5m / 15m | UTC | result |
|---|---|---|---|
| 0 | 1.33 / 1.91 / 1.94 | 06:03:30 | PASS: 59 passed in 3.16s |
| 1 | 1.29 / 1.83 / 1.91 | 06:04:00 | PASS: 59 passed in 3.13s |
| 2 | 1.55 / 1.84 / 1.91 | 06:04:29 | PASS: 59 passed in 3.09s |
| 3 | 1.37 / 1.77 / 1.89 | 06:04:59 | PASS: 59 passed in 3.13s |
| 4 | 1.02 / 1.65 / 1.84 | 06:05:29 | PASS: 59 passed in 3.28s |
| 5 | 1.23 / 1.63 / 1.83 | 06:05:59 | PASS: 59 passed in 3.13s |
| 6 | 0.80 / 1.49 / 1.78 | 06:06:28 | PASS: 59 passed in 3.31s |
| 7 | 0.97 / 1.47 / 1.76 | 06:06:58 | PASS: 59 passed in 4.46s |
| 8 | 1.44 / 1.55 / 1.78 | 06:07:29 | PASS: 59 passed in 3.39s |
| 9 | 2.42 / 1.79 / 1.85 | 06:07:59 | PASS: 59 passed in 3.22s |
| 10 | 3.31 / 2.09 / 1.95 | 06:08:29 | PASS: 59 passed in 4.27s |
| 11 | 2.45 / 2.00 / 1.92 | 06:09:00 | PASS: 59 passed in 3.10s |
| 12 | 1.80 / 1.88 / 1.89 | 06:09:30 | PASS: 59 passed in 3.12s |
| 13 | 1.29 / 1.75 / 1.84 | 06:10:00 | PASS: 59 passed in 3.14s |
| 14 | 0.96 / 1.63 / 1.80 | 06:10:29 | PASS: 59 passed in 3.10s |
| 15 | 1.12 / 1.60 / 1.78 | 06:10:59 | PASS: 59 passed in 3.16s |
| 16 | 1.05 / 1.54 / 1.76 | 06:11:29 | PASS: 59 passed in 3.10s |
| 17 | 1.25 / 1.54 / 1.75 | 06:11:59 | PASS: 59 passed in 4.48s |
| 18 | 1.51 / 1.57 / 1.75 | 06:12:30 | PASS: 59 passed in 3.12s |
| 19 | 1.12 / 1.46 / 1.71 | 06:13:00 | PASS: 59 passed in 3.34s |
| 20 | 0.86 / 1.37 / 1.67 | 06:13:30 | PASS: 59 passed in 3.52s |
| 21 | 1.11 / 1.40 / 1.67 | 06:14:00 | PASS: 59 passed in 3.15s |
| 22 | 0.79 / 1.29 / 1.63 | 06:14:30 | PASS: 59 passed in 3.33s |
| 23 | 0.58 / 1.20 / 1.58 | 06:15:00 | PASS: 59 passed in 3.76s |
| 24 | 0.85 / 1.21 / 1.57 | 06:15:30 | PASS: 59 passed in 4.02s |
| 25 | 1.41 / 1.31 / 1.60 | 06:16:01 | PASS: 59 passed in 4.24s |
| 26 | 1.67 / 1.41 / 1.62 | 06:16:32 | PASS: 59 passed in 3.10s |
| 27 | 1.64 / 1.43 / 1.62 | 06:17:02 | PASS: 59 passed in 4.16s |
| 28 | 2.16 / 1.59 / 1.67 | 06:17:33 | PASS: 59 passed in 3.10s |
| 29 | 2.21 / 1.63 / 1.68 | 06:18:03 | PASS: 59 passed in 4.03s |
| 30 | 1.78 / 1.60 / 1.67 | 06:18:34 | PASS: 59 passed in 3.07s |
| 31 | 1.57 / 1.56 / 1.65 | 06:19:03 | PASS: 59 passed in 3.04s |

## 11. Mutants (each a scratch copy of the COMMITTED file with an asserted single change — every anchor occurs exactly once, every mutant parse-checked — ONE gated run each, `-x`, seed 0)

**76 mutants: 74 KILLED, 2 SURVIVED, 0 errors, 0 refused.** NG mutants run the three NG files from a scratch tree (`GIT_DIR` pointed at the real repo so the base-extraction `git show` works); daemon mutants run the two daemon files through the `*_DAEMON_UNDER_TEST` overrides. The runner never touched the worktrees.

| mutant | group | what changed | START load 1m/5m | result | first failing test |
|---|---|---|---|---|---|
| n-time-keyed | step | decrement per PULSE (time-keyed), not per advanced step | 1.01/1.42 | KILLED | test_held_clock_100_pulses_zero_steps_leave_the_counter_untouched_and_the_node_survives |
| n-no-gt | step | `t >= last`: a held clock still decrements | 0.73/1.32 | KILLED | test_held_clock_100_pulses_zero_steps_leave_the_counter_untouched_and_the_node_survives |
| n-last-never-recorded | step | `last` not recorded at a decrement | 1.62/1.51 | KILLED | test_many_steps_in_one_pulse_count_once |
| n-regression-decrements | step | a clock that goes BACKWARDS decrements | 1.10/1.39 | KILLED | test_timestep_regression_resets_last_and_never_decrements_or_goes_negative |
| n-regression-ignored | step | a clock regression leaves `last` stale (no re-baseline) | 1.19/1.40 | KILLED | test_timestep_regression_resets_last_and_never_decrements_or_goes_negative |
| n-below-zero | step | the count can go below zero | 1.12/1.36 | SURVIVED | (none) |
| n-seeding-dropped | step | seeding of a node lacking the counter dropped | 1.17/1.37 | KILLED | test_legacy_node_is_seeded_once_and_its_window_is_bounded |
| n-seed-every-visit | step | seeding on EVERY visit (a reset each pass) | 1.26/1.38 | KILLED | test_exactly_N_stepped_pulses_close_the_window_then_the_next_sweep_takes_the_unwired_node |
| n-advance-ignores-exclusion | population | advance touches an excluded creation_mode | 0.92/1.29 | KILLED | test_seeding_scope_is_every_non_excluded_node_that_lacks_the_counter_and_never_an_excluded_one |
| n-stamp-ignores-exclusion | population | stamp touches an excluded creation_mode | 0.82/1.24 | KILLED | test_an_excluded_creation_mode_is_never_stamped |
| n-stamp-last-zero | step | stamp records last=0 instead of the graph's timestep | 0.83/1.20 | KILLED | test_held_clock_100_pulses_zero_steps_leave_the_counter_untouched_and_the_node_survives |
| n-advance-no-clock-guard | step | advance does not guard an unreadable clock | 0.86/1.17 | KILLED | test_an_unreadable_clock_makes_advance_a_no_op_and_stamp_falls_back_to_zero[None] |
| n-advance-drops-shape-guard | step | advance raises / mutates on a bad shape (guard dropped) | 0.80/1.14 | KILLED | test_odd_shapes_of_the_fields_are_left_untouched_and_never_raise[0-None] |
| n-stale-ge | heartbeat | staleness `>=` instead of strict `>` | 0.76/1.11 | KILLED | test_fresh_means_the_window_applies_and_staleness_is_a_strict_greater_than |
| n-window-ignores-heartbeat | heartbeat | the window check ignores the heartbeat | 0.82/1.09 | KILLED | test_fresh_means_the_window_applies_and_staleness_is_a_strict_greater_than |
| n-warning-every-call | heartbeat | the stale WARNING on every consult (latch missing) | 0.60/1.02 | KILLED | test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers |
| n-latch-never-reset | heartbeat | a completed pass does not re-arm the latch | 0.82/1.03 | KILLED | test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers |
| n-no-info-on-recovery | heartbeat | the recovery INFO missing | 0.61/0.96 | KILLED | test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers |
| n-stale-once-never | heartbeat | the stale latch never set (warns every consult) | 0.71/0.96 | KILLED | test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers |
| n-heartbeat-stamp-noop | heartbeat | the completion stamp never written | 0.60/0.92 | KILLED | test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers |
| n-stamp-not-at-registration | heartbeat | the heartbeat not stamped at registration (stamp None) | 0.41/0.84 | KILLED | test_a_clock_that_raises_registers_nothing |
| n-attach-before-stamp | registration | the registration attached BEFORE the clock is read (a raising clock leaves it registered) | 0.86/0.92 | KILLED | test_a_clock_that_raises_registers_nothing |
| n-arm-accepts-bad-age | registration | registration accepts any max age | 1.34/1.03 | KILLED | test_a_bad_heartbeat_age_raises_and_registers_nothing[0] |
| n-clock-not-required | registration | a heartbeat without a clock accepted | 1.29/1.04 | KILLED | test_a_heartbeat_needs_a_callable_clock[None] |
| n-window-zero-accepted | registration | window_steps == 0 accepted | 0.91/0.97 | KILLED | test_a_bad_window_size_raises_and_registers_nothing[0] |
| n-window-bool-accepted | registration | window_steps True accepted | 1.01/0.99 | KILLED | test_a_bad_window_size_raises_and_registers_nothing[True] |
| n-excluded-string-accepted | registration | a bare string accepted as the excluded modes | 0.66/0.91 | KILLED | test_excluded_modes_must_be_a_collection_not_a_string[ingested0] |
| n-excluded-not-stored | population | the excluded modes dropped at registration | 0.45/0.84 | KILLED | test_a_valid_registration_attaches_everything_at_once_and_arms_the_heartbeat_now |
| n-window-ignores-exclusion | population | the window check protects an excluded creation_mode | 0.61/0.85 | KILLED | test_an_excluded_creation_mode_is_never_in_the_window_even_with_an_open_counter |
| n-window-non-dict-raises | predicate | the window check raises on a non-dict metadata | 0.47/0.80 | KILLED | test_none_or_non_dict_metadata_is_closed_without_raising[None] |
| n-window-drops-finite-guard | predicate | the check drops the finite / bool guard | 0.60/0.80 | KILLED | test_the_window_is_closed_for_every_other_shape_and_never_raises[True] |
| n-window-drops-last-guard | predicate | the check drops the `last` guard | 0.48/0.75 | KILLED | test_the_window_is_open_for_a_finite_number_greater_than_zero[5.0] |
| n-window-zero-protected | predicate | a count of 0 still protects | 0.75/0.79 | KILLED | test_the_window_is_closed_for_every_other_shape_and_never_raises[0] |
| n-finite-accepts-bool | predicate | _is_finite_number accepts bool | 0.58/0.75 | KILLED | test_a_bad_heartbeat_age_raises_and_registers_nothing[True] |
| n-sweep-no-try | sweep | a raising window check escapes the sweep | 0.75/0.77 | KILLED | test_a_check_that_raises_node_not_spared_ONE_warning_with_a_count_and_class_names_only |
| n-sweep-failed-node-spared | sweep | a node whose window check raised is SPARED (fails toward protecting) | 0.57/0.73 | KILLED | test_a_check_that_raises_node_not_spared_ONE_warning_with_a_count_and_class_names_only |
| n-sweep-warning-carries-str | sweep | the sweep WARNING carries the exception string | 0.54/0.71 | KILLED | test_a_check_that_raises_node_not_spared_ONE_warning_with_a_count_and_class_names_only |
| n-sweep-no-warning | sweep | the sweep WARNING missing | 0.44/0.67 | KILLED | test_a_check_that_raises_node_not_spared_ONE_warning_with_a_count_and_class_names_only |
| n-sweep-ignores-grace-order | sweep | grace dropped from the structural orphan test | 0.45/0.65 | KILLED | test_timestep_regression_resets_last_and_never_decrements_or_goes_negative |
| n-sweep-unconditional-window-read | sweep | an UNREGISTERED graph is implicitly registered by the sweep | 0.46/0.64 | KILLED | test_an_UNREGISTERED_graph_sweeps_EXACTLY_as_the_base_does_over_a_seeded_family_even_when_nodes_carry_window_fields |
| n-env-read-added | purity | an environment read added to the canonical helper | 0.33/0.59 | KILLED | test_a_bad_window_size_raises_and_registers_nothing[True] |
| n-cc-token-added | purity | a cc token added to canonical executable code | 0.43/0.59 | KILLED | test_static_no_cc_token_in_the_executable_code_of_the_new_and_changed_functions |
| n-module-level-added | purity | a module-level statement added | 0.73/0.65 | KILLED | test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else |
| n-other-function-changed | purity | an unrelated function renamed (AST proof must notice) | 0.73/0.67 | KILLED | test_held_clock_100_pulses_zero_steps_leave_the_counter_untouched_and_the_node_survives |
| o-advance-after-continue | host | the advance call placed AFTER an early `continue` | 0.79/0.70 | KILLED | test_the_advancer_calls_the_canonical_advance_once_per_population_node_before_any_continue_and_never_for_an_excluded_one |
| o-advance-removed | host | the advancer never calls the canonical advance | 0.59/0.66 | KILLED | test_a_registered_deposit_opens_the_window_beside_the_graduation_fields_and_an_exact_repeat_reopens_it |
| o-advance-before-population | host | advance called for the excluded population too | 0.41/0.61 | KILLED | test_the_advancer_calls_the_canonical_advance_once_per_population_node_before_any_continue_and_never_for_an_excluded_one |
| o-heartbeat-removed | host | the completion stamp never written | 0.41/0.59 | KILLED | test_the_completion_heartbeat_is_stamped_only_by_a_non_raising_pass_never_before_the_loop_never_in_a_finally |
| o-heartbeat-before-loop | host | the completion stamp written BEFORE the loop | 0.44/0.58 | KILLED | test_the_completion_heartbeat_is_stamped_only_by_a_non_raising_pass_never_before_the_loop_never_in_a_finally |
| o-heartbeat-in-finally | host | the stamp written in a `finally` (so also on the exception path) | 0.32/0.54 | KILLED | test_the_completion_heartbeat_is_stamped_only_by_a_non_raising_pass_never_before_the_loop_never_in_a_finally |
| o-advance-no-getattr | host | the advancer requires the canonical helper (no tolerance for a graph without it) | 0.63/0.60 | KILLED | test_static_the_organism_holds_no_window_logic_only_the_three_canonical_calls |
| o-deposit-stamp-removed | host | a deposit never opens the window | 1.47/0.84 | KILLED | test_a_registered_deposit_opens_the_window_beside_the_graduation_fields_and_an_exact_repeat_reopens_it |
| o-deposit-stamp-no-getattr | host | the deposit requires the canonical stamp (no tolerance) | 1.48/0.92 | KILLED | test_the_deposit_never_raises_on_a_graph_that_predates_the_helpers |
| o-population-includes-ingested | host | the excluded-modes constant emptied | 1.90/1.09 | KILLED | test_population_false_for_ingested_true_for_everything_else |
| o-population-wrong-key | host | the population reads the wrong field | 2.40/1.31 | KILLED | test_population_false_for_ingested_true_for_everything_else |
| o-grad-reads-window-knob | knobs | GRADUATION reads the window knob (direction: window -> graduation) | 3.58/1.78 | KILLED | test_static_the_organism_reads_no_environment_for_the_window_and_names_no_window_knob |
| o-grad-total-changed | knobs | graduation total changed (not byte-identical) | 4.86/2.58 | KILLED | test_a_registered_deposit_opens_the_window_beside_the_graduation_fields_and_an_exact_repeat_reopens_it |
| o-window-env-in-organism | knobs | the organism reads the window env (the host-only rule) | 3.82/2.60 | KILLED | test_static_the_organism_reads_no_environment_for_the_window_and_names_no_window_knob |
| e-banned-steps-dropped | export | the step-count key no longer banned on the wire | 5.45/3.64 | KILLED | test_banned_meta_carries_both_final_field_names_and_no_old_name |
| e-banned-last-dropped | export | the last-timestep key no longer banned on the wire | 5.06/3.95 | KILLED | test_banned_meta_carries_both_final_field_names_and_no_old_name |
| d-window-default-wrong | daemon | window default 11 | 3.63/3.81 | KILLED | test_the_defaults_are_the_documented_ones |
| d-k-default-wrong | daemon | K default 6 | 2.85/3.63 | KILLED | test_the_defaults_are_the_documented_ones |
| d-reader-accepts-zero | daemon | the reader accepts 0 | 2.15/3.37 | KILLED | test_reader_rejects_everything_else_with_the_default_and_names_which_rule_failed[0-<= 0] |
| d-reader-no-boolish | daemon | the reader no longer names bool-ish words | 1.87/3.21 | KILLED | test_reader_rejects_everything_else_with_the_default_and_names_which_rule_failed[true-bool-ish] |
| d-reader-no-strip | daemon | the reader does not strip whitespace | 2.05/3.13 | SURVIVED | (none) |
| d-env-name-window-wrong | knobs | the WINDOW knob reads the GRADUATION variable (direction: graduation -> window) | 2.19/3.07 | KILLED | test_both_knobs_are_read_once_at_import_absent_valid_and_invalid |
| d-env-name-k-wrong | knobs | the heartbeat knob reads the WINDOW variable | 2.24/3.00 | KILLED | test_both_knobs_are_read_once_at_import_absent_valid_and_invalid |
| d-invalid-warning-missing | daemon | no WARNING for an invalid knob | 2.36/2.96 | KILLED | test_an_invalid_knob_logs_exactly_one_warning_from_init_ng_naming_the_variable_and_the_rule[window-NG_FAIR_CHANCE_WINDOW_STEPS] |
| d-registration-warning-str-exc | daemon | the registration WARNING carries str(exc) | 2.55/2.97 | KILLED | test_an_ng_that_predates_the_canonical_registration_registers_nothing_warns_once_class_only_and_the_daemon_starts |
| d-clock-wrong | daemon | the clock handed over is wall time | 2.45/2.90 | KILLED | test_init_registers_through_the_ONE_canonical_call_with_exactly_what_the_host_owns |
| d-age-not-times-interval | daemon | the max age is K, not K x interval | 2.19/2.81 | KILLED | test_init_registers_through_the_ONE_canonical_call_with_exactly_what_the_host_owns |
| d-excluded-dropped | daemon | no creation_mode excluded | 2.21/2.75 | KILLED | test_init_registers_through_the_ONE_canonical_call_with_exactly_what_the_host_owns |
| d-window-literal | daemon | the window size handed over is a literal 10 | 2.35/2.76 | KILLED | test_the_values_handed_over_follow_the_module_constants_the_host_read_from_its_environment |
| d-no-registration | daemon | the registration call replaced by a no-op | 2.16/2.68 | KILLED | test_init_registers_through_the_ONE_canonical_call_with_exactly_what_the_host_owns |
| d-info-missing | daemon | the success INFO missing | 2.31/2.67 | KILLED | test_init_registers_through_the_ONE_canonical_call_with_exactly_what_the_host_owns |
| d-s4-line-wrong | daemon | the S4 export line disagrees with the default | 2.35/2.64 | KILLED | test_the_s4_export_lines_in_the_changelog_match_the_code_defaults |

### 11.1 The two survivors, analysed (not hidden, not "fixed" by weakening anything)

- `n-below-zero` (`max(0, steps - 1)` -> `steps - 1` in `fair_chance_advance`): for the only counters the advancer decrements — finite numbers `> 0` — and integer window sizes (`enable_fair_chance_window` accepts only a positive int, and the advance decrements by one), `steps - 1 >= 0` always, so the `max` is **equivalent for every value the system produces**. It differs only for a hand-edited FRACTIONAL counter in `(0, 1)` (stored `-0.5` instead of `0`; the window is closed either way because `-0.5 > 0` is false). I did not add a test for it because that would change a test file after the seed table was recorded; it is a defensive clamp. Follow-up if wanted: one test with `0.5`.
- `d-reader-no-strip` (drop `.strip()` on the raw env string): `int(" 7 ")` already tolerates padding, so every accepted value is unchanged; the only difference is the REASON text for a whitespace-padded bool-ish word (`" true "` reports "not an integer" instead of "a bool-ish word"). Behaviourally equivalent for the value; the message wording is the only observable. Same follow-up shape.

Two kills worth reading carefully: `n-other-function-changed` (an unrelated function renamed) was KILLED first by a behavioural test (`test_held_clock_100_pulses_...`) because the rename broke the sweep itself, not by the AST test (with `-x` only the first failing test is recorded); and `n-env-read-added` was killed first by `test_a_bad_window_size_raises_and_registers_nothing[True]`. The AST/grep tests named in §3-§4 are pinned on the committed tree (green in every seed) but I did not run a mutant that only they can see (e.g. an `os.environ` read in a helper the behaviour never reaches).

## 12. What ran, what did not, not verified

**Ran:** everything in §10-§11, on the committed trees only (no dirty paths).

**Disclosure — the aborted campaign.** Before Exec P563 I was running a mutation campaign on the SUPERSEDED HEAD `cdbdbe22` (NG-4). After P563 I killed ONLY my own runner by exact PID, kept its results as `/tmp/p561_results_SUPERSEDED_HEAD_cdbdbe22.jsonl`, and used NONE of them: 26 NG seeds had passed and 2 gate refusals had been recorded on that HEAD. Nothing in this return rests on it; everything above was re-run on the final HEADs.

**Did NOT run / not verified:**
- No full NG suite, no `test_cc_deposit_step.py` (standing clause #944); no run against any checkpoint, no live graph, no real `NeuroGraphMemory`.
- `~/.bashrc` not read or written; `cc-ng-service.py` untouched; `canonical_exports` not executed; the two export lines are not verified to be present in `.bashrc` (not checked).
- Nothing deployed, nothing restarted, no VPS. The window is registered only by the laptop daemon; the VPS host (`cc_ng_host.py`) neither registers nor stamps, so there the sweep is today's.
- Whether 10 steps / K = 5 are the right values is a tuning call, not tested against live traffic.
- The behaviour of seeding (A) on a REAL restored checkpoint (§7) is reasoned and unit-tested, not observed live.
- Pre-existing and unrelated: `tests/test_cc_topology_callosum.py::test_poincare_dir_is_rederived_locally_not_transmitted` fails (a `bytes` -> `float` ValueError) in the broader NG run; it is not in any file I changed. Follow-up, not mine.

**Corrections to build-001 (recorded here, build-001 itself is NOT edited by me):** build-001's R5 / F6 / §3(ii) said arrivals carrying an embedding get NO window; that is wrong. Arrivals with an embedding DO get a window (the deposit helper stamps them). The Z12 / vault notes that repeat it are the ZM's to correct. Also: the `probation_cycles_skipped` counter covers the INNER try only (le-062 C2); the heartbeat covers the rest.

## 13. What content is SUPERSEDED (nothing was rewritten; the history keeps it)

| commit | superseded content | by |
|---|---|---|
| NG-3 `3ccc749` | the organism-resident window machinery: `fair_chance_window_open`, the organism heartbeat state / arm / stamp / fresh, `_probation_step_window_tick`, `_probation_clock`, `_probation_finite_number`; the node fields `probation_steps_remaining` / `probation_last_timestep` (now `fair_chance_*`); its reuse of `CC_CONV_PROBATION_PERIOD` as the window size | NG-4' (canonical) + NG-3c |
| NG-3b `7d34381` | the knob `CC_PROBATION_STEP_WINDOW` and the organism-side env reader / constant | `NG_FAIR_CHANCE_WINDOW_STEPS`, read by the daemon |
| NG-4 `cdbdbe2` | the sweep body that read a host-registered PREDICATE (`_fair_chance_window_open` attribute) — the "one function body only, host-agnostic predicate" shape | NG-4' (the whole window is canonical; the sweep reads `_fair_chance_cfg` and calls `_in_fair_chance_window`) |
| D-2 `4baa731` | arming the ORGANISM heartbeat and registering the ORGANISM predicate; the knob `CC_PROBATION_HEARTBEAT_CYCLES` / `PROBATION_HEARTBEAT_CYCLES`; the tests keyed to them | D-3 (one canonical registration; `NG_FAIR_CHANCE_HEARTBEAT_CYCLES`) |
| NG-1 / NG-2 / D-1 | round-1 `probation_advances`, the round-1 sweep filter | already superseded by NG-3/NG-4, now by the above; D-1's skip-VISIBILITY (the counter, the limiter WARNING, the C2 note) STANDS |

Tests: `tests/test_cc_probation_advances_p552.py` (NG-3), `tests/test_cc_sweep_probation_p552.py` (NG-4'-era), `tests/test_cc_fair_chance_window_p561.py` (NG-3c-era) were `git rm`'d and replaced by `tests/test_fair_chance_window.py` (canonical) and `tests/test_cc_fair_chance_host.py` (host share). The 423 harness (`tests/test_cc_capture_mutations_423.py`) keeps its assertions untouched; only its namespace shrank to what the reworked organism needs.

## 14. Follow-up list (for the Chief; none acted on)

1. **Rollout to Syl's NG and the VPS host** — Josh's call; LAW 8 gate per host. Her sidecar `_update_probation` is conversation-gated, so she must NOT register until it runs on its own clock. Registering would also change her sweep (canonical code).
2. le-062 L4: `str(exc)` handling and an uncounted save raise in the daemon's autosave path (not changed here).
3. DEBUG-level swallows, including the VPS host `cc_ng_host.py:1533`.
4. The `_unbound_nodes` docstring says "minus one term" (noticed in round 1); I did not re-check it against the final sweep and did not touch it (protected file, out of scope).
5. `cc_update_probation` is not per-node fail-soft (one poisoned node aborts the pass; the heartbeat now makes that visible and bounded, it does not make the pass tolerant).
6. Exact-repeat reset: `fair_chance_stamp` on an exact repeat deposit re-opens the window to the full size (by design, tested as such); whether a repeating source should be able to keep re-opening it is a decision for the Chief, not made here.
7. NG `CLAUDE.md` still names `inject_current`; the real API is `stimulate` (found while writing tests).
8. Optional: the seeding (A) narrowing condition (§7), and the two survivor tests (§11.1).
9. build-001 / vault notes still say arrivals get no window (§12) — the ZM's to correct; enumeration rows #988 / #989 / #990 / #1000 were not in my context and were not re-derived.
10. A canonical DEFAULT window size, if the Chief wants the helper callable without a host value: not added (Addendum 4 allows it as a module constant; adding one is a module-level addition, which I would have had to STOP and name).

## 15. Closing check

```
$ git -C <NG worktree> status -sb && git -C <NG worktree> log --oneline b5e47686..HEAD     # recorded at commit time, see the commit
$ git -C <docs worktree> status -sb && git -C <docs worktree> log --oneline -3
## cc-laptop-daemon-probation-warn-20261002
ca0a8330 D-3: register the CANONICAL fair-chance window; host reads generic env knobs (Exec P563 / Addenda 3-4)
4baa731f D-2 (P561/P562): ...
905618ee daemon (P552): register the probation predicate on the CC graph at init; make a skipped probation decrement VISIBLE
```

Nothing is pushed or merged: both branches are local only.
