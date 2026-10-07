# #1051 — the checkpoint guardian tells plasticity from damage

*2026-10-07 · lane guardian-1051 (build lane for the Executive) · design agreed by Josh 2026-10-06 ("Yeah, I like it, DudeMan") · branch `cc-laptop-guardian-1051-20261007` off the NG trial tip `149fa1f`; protected commit `ca5f41a` · nothing merged, nothing deployed, the live daemon, its venv, checkpoints, quarantine and the trial worktrees untouched.*

Related: punch list #1051 · sleep-phase spec `superpowers/specs/2026-10-06-sleep-phase-design.md` §5, D7, D8 · #373 / #83 / #105 / #423 (the guardian's history, in `checkpoint_guardian.py`'s changelog).

## 1. What changed, in one paragraph

The synapse "50% rule" (refuse when live synapses < 50% of the last accepted save) is replaced, **only when `NG_GUARDIAN_RECONCILE=1`**, by a reconciled gate. The guardian now counts every removal the engine itself logs (`pruned`, `nodes_collected`; `sleep_cycle` as attribution). Its reference is the **median of the last N (10) accepted saves**, each adjusted down by the removals logged since that save. **One lane addition, which needs Josh's ruling:** the reference is the higher of that median and the newest accepted save minus the removals logged since it (§3.3, §7). A drop the logged removals explain is plasticity: the save proceeds. Loss that nothing explains, beyond a tolerance learned from the typical save-to-save motion, is damage: refuse and quarantine, exactly as today. A separate **UNUSUAL CHURN** WARNING fires when the net change is far outside the history's median/spread; it never blocks a save. Every other protection is unchanged.

## 2. Where the code lives (and what is protected)

| file | protected? | change | commit |
|---|---|---|---|
| `checkpoint_guardian.py` | no | `RemovalLedger`, guard history I/O, `evaluate_synapse_reconciliation()` (pure), `SaveGate.attach_removal_ledger / record_accepted`, `permit(..., removals=)`, `evaluate_save_health(..., synapse_reconciliation=)` | `bfe788e`, review fixes `b1563f9` |
| `openclaw_hook.py` | **YES** (Syl's Law) | 4 small host-wiring additions, no decision logic (the #83/#105 pattern) | **`ca5f41a`** (own commit) |
| `tests/test_guardian_reconcile_1051.py` + `tests/fixtures/guardian_1051_daemon_log_saves.json` | no | replays + concurrency proofs | `2cc3d65`, `b1563f9` |

**The daemon needs no change.** The guardian lives in NG; the laptop daemon reaches it through `STATE.ng.save()`. `cc-ng-service.py prepare` already forwards every `NG_*` export from `.bashrc` into the daemon's environment, so arming it is one line in `.bashrc` (see §8). No daemon branch was created.

The protected commit's merge needs Josh's separate protected-file "proceed" (Executive handles), with the usual msgpack backup.

### The protected diff (`ca5f41a`), all inert when the flag is off
1. `__init__`, right after the boot restore and `record_restore`: `self._removal_ledger = self._save_gate.attach_removal_ledger(self.graph)` (returns `None` unless `NG_GUARDIAN_RECONCILE` is on).
2. `_capture_checkpoint_state`, **inside the same `_step_lock` hold** that captures the counts: `counts["removals"] = ledger.snapshot()`.
3. `save()`: `permit(..., removals=counts["removals"])`, passed only when the snapshot exists (an unarmed host calls `permit()` byte-for-byte as before).
4. `save()`: `self._save_gate.record_accepted(counts)` right after `write_manifest` of a **primary** save. Never on quarantine or a failed write.

## 3. The mechanism

### 3.1 The removal ledger (exact counters)
`RemovalLedger` subscribes to the graph's public `register_event_handler` (a pure observer, like the daemon's reap logging and the vdb drop handler):

- `pruned(count)` → `synapses += count`. Every engine synapse removal goes through `_prune_synapses`, which emits it: `step()`, the Tonic write-mode tail, the strength budget (competing mode) and `sleep_cycle`.
- `nodes_collected(count)` → `nodes += count`. Orphans have no synapses, so this never changes the synapse count.
- `sleep_cycle(pruned, nodes_collected)` → `sleep_cycles += 1`, `sleep_synapses += pruned`. **Attribution only.** `sleep_cycle` runs the two calls that already emitted `pruned`/`nodes_collected`; adding its totals again would double count. Mutant M2 (count it) is killed by 7 tests.
- A handler that cannot count an event never raises into `step()`: it bumps `handler_errors` and logs once. Under-counting makes the gate **stricter**, never laxer.

**Not counted, by design:** host/operator removals through `remove_synapse` / `remove_node`: hub-prune, remove-false-wants, anneal_core, deposit rollbacks. These are not the engine's plasticity. The 2026-10-04 false-want removal was exactly this shape, and the brief requires it refused.

**Why the count is exact at the save boundary.** The engine emits all of these while holding `graph._step_lock`. The host snapshots the ledger inside the same lock hold that takes the #423 detached capture and its counts. So every removal is either before the capture (in the counts **and** the snapshot) or after it (in **neither**; it falls into the next save's interval). Proven:
- **Racing removals** (`test_real_removals_racing_saves_count_exactly`): a thread runs real `sleep_cycle` prunes while 25 saves run (both return modes). For every pair of accepted saves, `S_prev − explained − S_cur == 0` exactly, and removals really interleaved (≥ 5 intervals with removals).
- **Mid-pass save** (`test_real_save_requested_mid_sleep_waits_for_the_whole_pass`): a save started from inside `sleep_cycle` (in its `pruned` handler) blocks at capture until the pass ends. Its history entry holds the whole sleep: explained 300, 1 sleep_cycle, the post-sleep count.
- **Detached capture (#423)** (`test_real_removal_after_detached_capture_counts_in_the_next_save`): 250 synapses removed after the capture released the lock, before permit/write. They are **not** in that save (explained 100) and are in the next (250). Residuals 0, 0. Mutant M1 moves the snapshot to permit time (outside the lock). This test kills it every time; the race test killed it in one of two runs, because it depends on timing.
- **Restart:** the ledger is per process and starts at 0 when the graph is restored. Removals made after the last accepted save, if never saved, die with the process together with the state they removed (`test_restart_with_lost_removals_is_not_unexplained`, `test_real_restart_continues_history`).

### 3.2 The history (persisted)
`<checkpoint>.guard_history.json`, beside `.manifest.json` and `.guard_state.json`:

```json
{"version": 1, "entries": [
  {"saved_at": "<the manifest's saved_at>", "recorded_at": "...", "synapses": 64327, "nodes": 11281,
   "hyperedges": 0, "timestep": 85038, "explained_synapses": 1800, "explained_nodes": 0,
   "sleep_cycles": 0, "sleep_synapses": 0, "handler_errors": 0, "source": "save"}]}
```

Each entry is one accepted primary save. `explained_*` = the removals the engine logged between the previous accepted save and this one. The file holds the last `NG_GUARDIAN_HISTORY_N` entries.

**Why JSON, not SQLite (Format-for-Purpose).** The access pattern is: one small document (≤ 10 records, ~3 KB), read whole once per save, rewritten whole once per accepted save, with a single writer (the publication lock already serializes saves in a process). There are no queries, no appends from several writers, and no growth. The file also needs to be readable by a human mid-incident with `cat`. That is a whole-snapshot rewrite of a tiny document. SQLite would add a binary file, a schema and WAL/journal sidecars for no query and no concurrency benefit. The two sibling guard files already use JSON with atomic `tmp + os.replace`. This one follows the same pattern.

**When the disk changes outside the save path** (Josh promotes a quarantine with a truthful `write_manifest`, an offline tool rewrites the checkpoint, a generation is restored, or the flag is turned on for the first time), the manifest's `saved_at` no longer matches the newest entry. The history is **re-seeded** from the manifest: one entry, `source: out_of_band` (or `seed`), `explained: null`. This is logged as a WARNING, and tolerance bootstraps again. It never invents explained removals. At boot the restored synapse count is compared with what the next save will be measured against: the manifest if it was rewritten outside the save path (`test_boot_check_reads_the_manifest_after_an_offline_rewrite`), otherwise the newest entry.
- A shortfall is logged at **ERROR** and names the operator step. It counts **against** the next save and is never silently adopted (`test_boot_mismatch_is_warned_not_laundered`).
- A ledger that cannot attach logs ERROR, and that process keeps the pre-#1051 rule. The guardian never stops a boot (`test_attach_failure_never_stops_a_boot`).

### 3.3 The decision (`evaluate_synapse_reconciliation`, pure)
For the live synapse count `L` and the explained count `E` since the newest entry:

- **Reference.** For each of the last N entries, `A_i = S_i − (all removals logged since save i)`. That is what save i would hold now if only the logged removals had happened; sprouting only adds. `ref = median(A_i)`.
  - A sprout burst in a minority of the window does not move the median. A burst lasting more than half the window becomes the norm (`test_median_reference_vs_a_sprout_burst`).
  - A staged collapse cannot walk the median down. Refused saves never enter the window, and accepted unexplained steps pile up against the lagging median. Mutant M3 (newest save only, median ignored) is killed by the slow-walk test.
- **Newest-save floor (lane addition, from the law review; `NG_GUARDIAN_LAST_SAVE_FLOOR`, default on).** `ref = max(median, A_last)`.
  - The median alone lags a growing graph. Ten saves of +5% put it at ~0.8 of the newest save, so one sudden unexplained ~20% loss would pass with only the churn alarm (`test_sudden_unexplained_loss_during_growth`).
  - Sprouting only adds, so `A_last` (the newest save minus everything logged since it) is an exact floor for a healthy graph.
  - It does not bring back Josh's max-anchor objection. A sprout burst that is then pruned back is logged, so `A_last` follows it down. A burst that vanishes with nothing logged is refused, which is the lost-vs-pruned rule itself.
  - `=0` gives the agreed text verbatim (median alone). **Josh to rule.**
- **Unexplained loss** = `ref − L`.
- **Tolerance** = `median + k·1.4826·MAD` of the history's residual fractions `|S_{i−1} − E_i − S_i| / S_{i−1}`. That is the motion the logged removals do not account for: on the CC almost entirely sprouting, measured 2-6% per save in the daemon log. It is clamped to `[TOL_MIN 2%, TOL_MAX 20%]`, times `ref`, and never below 50 synapses. With fewer than 3 residuals the bootstrap value is used (10%).
- **Refuse** when unexplained loss > tolerance. Also refuse when `L < 10%` of the last save **even if explained**: a near-empty graph is the clobber shape whatever the logs say. This is a retention floor, not a 50% rule, and it is far from any real event (worst real explained drop: −52%).
- **Churn alarm** (separate, never blocks): `|L − S_last| / S_last` > `median + 6·1.4826·MAD` of the history's net-change fractions, floor 10% (bootstrap 25%). It logs a WARNING with the counts, the explained/unexplained split and the sleep_cycle attribution.

Every reconciled save also logs one INFO line, the "explained" record the sleep-phase P3 watch reads:
`Guardian #1051 reconcile: synapses 32190 -> 16353 (-49.2%), explained 15957 synapses + 0 nodes logged by the engine; 1 sleep_cycle(s) removed 15957, reference 16233 (newest accepted save; median 15783), unexplained -120, tolerance 325 (2.0%) -> permit`

### 3.4 What is kept exactly
The absolute node floor, the #105 content-collapse rule, the hyperedge ratio, the node EMA gate and MELT path, provisional mode, quarantine (`quarantine_save`, unchanged; the caller still logs "Guardian REFUSED…" and returns the quarantine path), the manifest (unchanged content, still truthful), the generation ring, and the daemon's outer node-count guard (`CC_NG_COLLAPSE_RATIO`). A refused or failed save never advances the history or the ledger baseline (`test_failed_or_refused_save_never_advances`).

### 3.5 Laptop-only by default
`NG_GUARDIAN_RECONCILE` defaults to `0`. With it unset (Syl's host, and every host until someone sets it):
- `attach_removal_ledger()` returns `None`;
- the counts carry no `removals`;
- `permit()` is called with the same arguments as before and `record_accepted()` is not called;
- no history file is written;
- the 50% rule refuses exactly as before.

`test_default_off_is_the_legacy_gate` and `test_real_default_off_parity` prove it. Turning it on for Syl would be a separate decision.

## 4. Replay results

All numbers are produced by `tests/test_guardian_reconcile_1051.py`'s scripted host (`~/.cache/guardian-1051/replay_table.py`). "Pre" history = the 10 accepted saves before the 10-06 cascade from the daemon log, ending at the punch list's 64,327.

| replay | verdict | synapses | explained | reference | unexplained | tolerance | churn alarm |
|---|---|---|---|---|---|---|---|
| 10-06 punch-list step 1, **no** events | **REFUSE** | 64,327 → 34,652 | 0 | 64,327 | 29,675 | 1,287 (2.0%) | yes |
| 10-06 punch-list step 2, no events | **REFUSE** | 64,327 → 21,618 | 0 | 64,327 | 42,709 | 1,287 | yes |
| 10-06 punch-list step 1, events 29,675 | PERMIT | 64,327 → 34,652 | 29,675 | 34,652 | 0 | 693 | yes |
| 10-06 punch-list step 2, events 13,034 | PERMIT | 34,652 → 21,618 | 13,034 | 21,618 | 0 | 504 | yes |
| 10-06 log-derived 10:11, no events | **REFUSE** | 55,260 → 28,691 | 0 | 55,260 | 26,569 | 1,226 | yes |
| 10-06 log-derived 10:11, events 26,976 | PERMIT | 55,260 → 28,691 | 26,976 | 28,284 | −407 | 628 | yes |
| 10-06 log-derived 10:23, events 7,871 | PERMIT | 28,691 → 21,274 | 7,871 | 20,820 | −454 | 468 | yes |
| 10-04 false-want removal, no events | **REFUSE** | 231,934 → 31,925 | 0 | 231,934 | 200,009 | 4,639 | yes |
| sleep clearance 15,957 of 32,190 (one `sleep_cycle`) | PERMIT | 32,190 → 16,353 | 15,957 | 16,233 | −120 | 325 | yes |
| slow unexplained walk, −1.5%/save: step 1 / 2 / **3** | PERMIT / PERMIT / **REFUSE** | 64,327 → 61,474 | 0 | 63,199 (median) at step 3 | 1,725 at step 3 | 1,515 | no |
| growth +5%/save ×10, then −22% unexplained | **REFUSE** (newest-save floor); PERMIT + alarm with the floor off | | 0 | | | | yes |
| every synapse gone, all "explained" | **REFUSE** (retention floor) | 64,327 → 0 | 64,327 | — | — | — | yes |
| empty graph (3 nodes) | **REFUSE** (node floor, unchanged) | — | — | — | — | — | — |
| restart mid-history | same reference, tolerance and verdict as the uninterrupted run; history continues | | | | | | |
| out-of-band manifest (promoted quarantine) | re-seeded, WARNING, bootstrap tolerance, next save PERMIT | | | | | | |

**The daemon log's whole save history** (2026-10-05 12:35 → 2026-10-06 23:52: 11 process segments, 250 primary saves; `test_daemon_log_save_history_replayed`), replayed with the removals the log shows:
- **0 refusals**;
- the churn alarm fires on exactly 5 saves, the two cascades: 10-05 evening (63,849 → 56,043 → 45,039 → 33,737) and 10-06 morning (55,260 → 28,691 → 21,274);
- every other save, including the growth bursts of +5-10% since the inactivity rule went off, is accepted silently.

With the removals hidden, both cascades are refused at their first step.

**A discrepancy with the punch-list row (for the record, not a code issue).** The row says 34,652 was accepted at 10:11. In the log, 34,652 is the count at the 10:08:33 REAP WARNING (line 636315). Pruning continued to 28,691 (the last prune line before the 10:11:32 save, line 636387), so the 10:11 save held about 28.7K plus a few sprouts, against about 55.3K at 10:03. The legacy rule still passed it (52%). The 10:23 figure (21,618) is confirmed by the next boot (line 636566). Both versions are replayed above.

## 5. Mutation check
Four mutants, each run against the final test file (`~/.cache/guardian-1051/mut/`, logs `logs/mutant_m*.txt`). All were **killed**:

| mutant | killed by |
|---|---|
| M1: ledger snapshot taken at permit time (outside the capture lock) | the detached-capture test. The race test also killed it in the first round, but it depends on timing. |
| M2: `sleep_cycle` removals added to the explained total (double count) | 7 tests |
| M3: reference = the newest save only (median ignored) | the slow-walk, growth and sprout-burst tests |
| M4: reference = the median only (newest-save floor ignored) | the growth and sprout-burst tests |

## 6. Test suites (base `149fa1f` vs branch)
Harness: `~/.cache/guardian-1051/run-isolated.sh`. Every file ran in its own isolated pytest process: scratch HOME (removed after), private PID namespace, `MemoryMax=3000M`, no swap, `nice 10`, one worker, 600 s cap, MemAvailable ≥ 4 GB gate. Venv `~/.cache/p2b-venv`, the one the P1 lane used. Base = detached worktree `/home/josh/worktrees/ng-guardian-1051-base-149fa1f`. Logs: `~/.cache/guardian-1051/suite/{branch,base,branch_rerun,base_rerun}/`; comparison: `logs/compare.txt`.

- **NG, branch (all 157 files):** 3,587 passed, 69 failed, 6 errors, 17 skipped; 39 files not green. `tests/test_guardian_reconcile_1051.py`: **28 passed**. The existing guardian files are green: `test_checkpoint_guardian` 24, `test_checkpoint_guardian_integration` 5, `test_checkpoint_capture_set_423` 18, and `test_save_guard_structural` 27 and `test_save_receipt_423` 37 all pass.
- **Base:** every one of the 39 non-green files was re-run on base. **36 have identical FAILED/ERROR id sets** (including the 7 known > 3 GB files killed by the cap, #1045, and `test_sleep_p1` / `test_snn` at the 600 s cap on both).
- **The 3 differences are load-dependent timing, not this change:**
  - `test_cc_embed_outside_lock_922` hit the 600 s cap on the branch (base took 423 s). The branch re-run passed: 10 passed in 465 s.
  - `test_cc_host_pith_telemetry::test_fsynced_samples_survive_sigkill` fails on the branch, and on the base re-run too.
  - `test_graph_substrate_race::test_build_adjacency_no_race_under_concurrent_step_mutation` is a 30 s wall-clock join on a pure engine + Lenia path that this diff does not touch. In alternating runs at load ~8 it **failed on base 2 of 2 and on the branch 2 of 2**.
- **New failures: 0.**
- **Daemon suite: not run, on purpose.** There is no daemon change: the guardian lives in NG and the daemon reaches it through `STATE.ng.save()`. The daemon tests (docs `scripts/tests`) use fakes or import the **live** `~/NeuroGraph` checkout (#1045), never this branch. So base and branch would be the same code and the comparison would carry no information.

## 7. Risks and what is NOT proven
- **Explained ≠ correct.** A buggy prune rule that removes half the graph is "explained" and is accepted, with a loud alarm. The guardian now protects against damage nobody logged, not against wrong plasticity. The only rail is the 10% retention floor. This is the agreed design ("explained = plasticity"); the alarm is the signal to read.
- **Operator surgery is unexplained.** hub-prune, remove-false-wants and anneal_core remove through `remove_synapse` / `remove_node` and will be refused if they exceed tolerance. This is the right default (10-04), and the quarantine-promotion path handles it. A "declared removal" API for host tools was deliberately not built (scope); flag it if wanted.
- **A slow leak can hide under the tolerance.** Unexplained loss slower than about tolerance / (N/2) per save (≈0.4%/save at a 2% tolerance) passes indefinitely, because by counts alone it looks the same as less sprouting. Each accepted unexplained residual also widens the tolerance a little (bounded by `NG_GUARDIAN_TOL_MAX` = 20%).
- **Additions are not reconciled.** Sprouting masks loss of the same size within one interval. The count-based guard cannot see a swap (that was also true of the 50% rule).
- **Crash window → stuck refusal until an operator acts.** Suppose the process dies after the atomic primary write but before `write_manifest` (milliseconds). The restored graph then differs from the newest history entry, the difference is unexplained, and refusals never advance the reference. So **every later save is refused and quarantined until an operator** checks the on-disk checkpoint and records it with a truthful `write_manifest` (the quarantine-promotion procedure).
  - The legacy rule had the same failure class, but with a 50% margin. The new margin is 2-10%, and one sleep can change half the graph in a single save. This is therefore **more reachable** than before.
  - Boot now logs it at ERROR, naming the operator step.
  - Lane recommendation: the daemon saves right after a sleep (spec §5.2), which also shrinks the exposure. A future fix is to write the manifest before the primary is renamed into place. That is protected code (the save order in `openclaw_hook.py`) and is out of scope here.
- **Median lag (law-review finding):** addressed by the newest-save floor. See §3.3; Josh rules on keeping it.
- **Handler order.** `Graph._emit` stops at the first handler that raises. The ledger is registered right after the restore, before the vector-drop and the daemon's reap handlers, so it runs first for both events. If it were ever registered later, a raising earlier handler would make it under-count, which errs toward refusing.
- **Nodes and hyperedges are not reconciled.** Node collection is counted and logged but does not decide anything. The node gates, including the daemon's outer 50% node guard and #105, are unchanged and can still refuse a node-heavy sleep. With `NG_HOST_WIRES_OWN_DEPOSITS=true` on the laptop the engine's node gate is the #83 MELT path. Watch this in P3.
- **Thresholds come from the log, not a long live run.** The tolerance (2-6%) and alarm (10% floor) defaults are calibrated on 250 saves over about 36 hours, not watched live. P3 should watch ≥ 2 sleeps plus the days between and report a window.
- **Not proven live.** Nothing ran against the live daemon or a checkpoint copy; the real-engine tests use a real `NeuroGraphMemory` + `Graph` in tmp workspaces (4-8K synapses).

## 8. To arm it on the laptop (Executive, after Josh's proceed on `ca5f41a`)
1. Merge the NG branch into the NG trial branch (protected-file process: msgpack backup, Josh's literal "proceed").
2. Add `export NG_GUARDIAN_RECONCILE=1  # #1051 reconciled synapse gate (laptop only)` to `~/.bashrc`. `cc-ng-service.py prepare` forwards `NG_*`. Other knobs stay at their code defaults unless P3 says otherwise:

| knob | default | meaning |
|---|---|---|
| `NG_GUARDIAN_HISTORY_N` | 10 | accepted saves in the window / history file |
| `NG_GUARDIAN_HISTORY_MIN` | 3 | residuals / changes needed before the learned tolerance / alarm replace the bootstrap values |
| `NG_GUARDIAN_TOL_K` | 4 | tolerance = median + k·1.4826·MAD of the residual fractions |
| `NG_GUARDIAN_TOL_MIN` / `_TOL_MAX` | 0.02 / 0.20 | clamp on the tolerance fraction |
| `NG_GUARDIAN_TOL_MIN_ABS` | 50 | tolerance floor in synapses |
| `NG_GUARDIAN_BOOTSTRAP_TOLERANCE` | 0.10 | tolerance until HISTORY_MIN residuals exist |
| `NG_GUARDIAN_MIN_SYNAPSE_RETENTION` | 0.10 | refuse below this fraction of the newest save even when explained |
| `NG_GUARDIAN_LAST_SAVE_FLOOR` | 1 | reference = max(median, newest save − logged removals); 0 = median alone (**Josh to rule**) |
| `NG_GUARDIAN_CHURN_K` / `_CHURN_MIN` / `_CHURN_BOOTSTRAP` | 6 / 0.10 / 0.25 | the non-blocking alarm level |
| `NG_GUARDIAN_MIN_REF_SYNAPSES` (existing) | 100 | below this the synapse gate does not apply |
3. One daemon restart. The first save seeds the history from the manifest (INFO); the first few saves use the bootstrap tolerance (10%) until 3 residuals exist.
4. Before arming sleep removal (D8), check the INFO "reconcile" lines on ordinary saves for a day.

**Optional, from spec §5.2, not built (it is sleep-lane code):** the daemon could save right after `sleep_cycle` so one save = one sleep batch. Reconciliation does not need it; it would only make the "explained" lines map 1:1 to sleeps.

## 9. Off-task findings (for the punch list)
- `neuro_foundation.py:3580`: the Tonic write-mode tail calls `_prune_synapses()` and discards the count, so `_total_pruned` / `StepResult` telemetry under-counts those removals. (`pruned` is still emitted, so the ledger and the daemon's reap logging see them.) Protected file; telemetry only.
- The punch-list #1051 row's 10:11 figure (§4).

## 10. Artifacts
`~/.cache/guardian-1051/`: `run-isolated.sh` (harness copy: 4 GB gate, scratch HOME removed after), `suite.sh`, `extract_saves.py` (the log → fixture extractor; reads the log by line order only), `replay_table.py`, `mut/` (mutants), `logs/`, `suite/{base,branch}/`.
