# NGLITE_SLEEP: NG-Lite sleeps the way the mind does (punch list #1114, Phase 1)

Lane report, 2026-10-15. Design: `~/docs/superpowers/specs/2026-10-15-ng-lite-sleep-design.md` (Josh, approved
direction). The rule comes from the mind's sleep spec, `2026-10-06-sleep-phase-design.md` §3.1 (rule) and §3.3
(clearance). Systems: [[NG-Lite]], [[The Commons]]. Laws: LAW 2, 5, 7, 8.

**Nothing is installed.** No live checkout was edited, no service was touched, and the wheel is not in
`~/NeuroGraph/.venv`. The Executive reviews; Josh decides.

## 0. TL;DR

- **What it is.** `NGLite.sleep()` exists on both cores and they are bit-identical. It does the strength-aware
  downscale with evidence armor, then clears links after `grace_sleeps` sleeps below `pruning_threshold`, removes
  orphan nodes after the same grace, and never touches the rim. It returns a report and logs one INFO line every
  sleep. `sleep_epoch` and `last_sleep_at` are saved with the state and can be read with `sleep_state()` and
  `get_stats()`.
- **Caps.** The caps are unchanged. At-cap eviction now uses the sleep's rule, so a link is never evicted for being
  newest.
- **The daemon.** It sleeps the CC Commons right after each real CC sleep and then persists it. This sits behind
  `CC_NG_COMMONS_SLEEP`, which is off by default.
- **Harness result: the mind's values (d0 0.1, h 0.05) do nothing useful for the Commons.**
  - On a copy of the live Commons, 60 wake/sleep cycles (about 24 days) cleared **0** links.
  - The count stays pinned at 4,999 / 5,000, so the cap still decides what the River keeps (about 155 evictions per
    wake).
  - The one gain is that the cap now takes the oldest, most-downscaled links. The youngest link it evicted was 29
    cycles old.
  - Why: NG-Lite links are born at 0.5 and reach 0.575 after their first success, on a [0, 1] scale. A one-off
    deposit needs about 155 sleeps (about 62 days) to fall under 0.01, and the cap removes it after about 31 cycles.
- **Recommendation (decision for Josh): `sleep_h = 0.5`, `sleep_d0 = 0.2`, grace 2.**
  - A one-off deposit lives about 27 sleeps (about 11 days). That is close to today's FIFO horizon of about 12 days.
  - The count settles at about 4,360, and the cap stops evicting after cycle 27.
  - Reinforced links live longer: 2 outcomes about 47 sleeps, 7 outcomes about 90.
  - `d0 = 0.3` gives about 7 days and about 2,920 links.
  - **The values are sticky** (§5). Decide before install.
- **Tests.** No new failures on any suite, base vs branch (§4).

## 1. What changed (file:line on the branches)

### NeuroGraph `cc-laptop-nglite-sleep-20261015` (on `8121409`, the #1115 ng_embed lane, which sits on the live trial `fe67f88`)

**Commits:** `592ae59`, `541cf1b`, `4288822`.

**`ng_lite.py`** (vendored, canonical; sha256 `24c868a2…`)

- `:312` — `import math`. `math.log` is libm `log`, the same function the Rust `f64::ln` calls.
- `:633-640` — `DEFAULT_CONFIG` gains five keys:

  | key | value |
  |---|---|
  | `sleep_d0` | 0.1 |
  | `sleep_h` | 0.05 |
  | `sleep_grace_sleeps` | 2 |
  | `sleep_armor_gain` | 1.0 |
  | `sleep_armor_max` | 5.0 |

  - These are the mind's values. `he_salience_max` is 5.0.
  - The weight threshold is the existing `pruning_threshold` (0.01). Its comment already said "Synapses below this
    weight get pruned", but until now nothing pruned by it.
- `:2080-2084` — `TUNABLE_PARAMS` gains the same five keys with these bounds. `update_tunable` clamps them.

  | key | bounds |
  |---|---|
  | `sleep_d0` | 0–0.5 |
  | `sleep_h` | 0.001–1 |
  | `sleep_grace_sleeps` | 0–10 |
  | `sleep_armor_gain` | 0–5 |
  | `sleep_armor_max` | 1–10 |

- `:960` — `_sleep_epoch` and `_last_sleep_at` are added to the Python core.
- `:1891` — `sleep(now=None)`.
  - It validates the parameters (`:1964`, ValueError) and dispatches to the Rust or the Python core.
  - It adds `core` and `seconds` to the report and logs one INFO line:
    `NG-Lite sleep [<module>] epoch=N: synapses a->b (downscaled, faint, cleared, frozen), nodes … (orphans waiting, collected, frozen)`.
  - A pre-#1114 wheel raises RuntimeError naming the wheel, logged at ERROR once (`4288822`).
- `:1945` — `sleep_state()` returns `{"epoch", "last_sleep_at"}` without a scan.
- `:1972` — `_sleep_armor`: `min(armor_max, 1 + gain·ln(success + failure))`. One outcome gives armor 1.
  - Both sides count, so a **contested** link is armored by all of its evidence, never weakened by the
    disagreement.
- `:1982` — `_sleep_python`, the rule. It works on dict snapshots and removes with `pop`.
  1. Every non-rim synapse is downscaled: `d = d0·h/(h+w)/armor`, `w ← w·(1−d)`. Weight only:
     `last_updated`, the counters and Welford are untouched, so recency buckets do not move.
  2. Below threshold after the downscale: `metadata["sleep_faint"] += 1`. The link is cleared once the count is
     > grace. At or above the threshold, the key is dropped.
  3. A non-constitutional node with no synapse: `metadata["sleep_orphan"] += 1`. It is removed (node and embedding
     cache) once the count is > grace. A node with a link drops the key.
  4. Constitutional nodes, and synapses whose source is one, are counted as frozen and never touched.
  5. The epoch goes up by 1 and the stamp is set.
- `:1803`, `:2047`, `:1862`, `:2055` — the state gets `"sleep": {"epoch", "last_sleep_at"}`. The key is absent
  until the first sleep, so a never-slept state saves byte-identically to before. Import restores it.
- `:2176`, `:2201` — `get_stats()` gains `sleep_epoch` and `last_sleep_at` on both cores.
- `:2547` and `:2592` — the at-cap eviction uses the same rule:
  - **synapses:** longest-faint first (counted only while still below the threshold), then lowest `w·armor`;
  - **nodes:** longest-orphaned first, then the weakest strongest-link `w·armor`, then the lowest activation count;
  - ties keep storage order, so the oldest-inserted goes first. The rim is never evicted.
- Changelog entry at the top: `[2026-10-15] … #1114 Phase 1`.

**Why sleep bookkeeping lives in the metadata dicts (like `strength_sum`).** An older `ng_lite.py` or wheel still
loads a slept state:
- `NGLiteSynapse(**d)` and the Rust `check_fields` refuse unknown fields.
- A new dataclass field would therefore make a rollback lose the whole Commons, because the restore fails and the
  Commons starts fresh.
- Unknown top-level keys such as `"sleep"` are ignored by both old cores.

**`commons.py:516`, `:525`** — `Commons.sleep(now=None)` and `Commons.sleep_state()` pass through to the NG-Lite.
The daemon never reaches into `Commons._ng`. Changelog at the top.

**`cc_ng_organism.py:1703`, `:1777`** — `persist_cc_commons` serializes its writers with `_cc_commons_persist_lock`.
`NGLite.save`'s temp file is `<path>.tmp.<pid>` on both cores. The autosave thread and the new post-sleep persist
are two threads of one process, so they would write the same temp file.

**`tests/test_ng_lite_sleep.py`** (new, 23 tests). **`tests/test_commons.py`**: the axiom guard's allow-list admits
`sleep` and `sleep_state` (§6, item 7).

### ng-tract-rs `cc-laptop-nglite-sleep-rs-20261015` (on `f9a2c79`, the installed wheel's commit)

**Commit `97a241e`.**

**`src/ng_lite_core.rs`**
- `:716-720` and `:741-745` — the `DEFAULT_CONFIG` and `TUNABLE_PARAMS` mirrors.
- `:797-801` — `Params` fields.
- `:125` — `VMap::remove` (dict `del`).
- `:914` — `meta_count`.
- `:1200` — `armor`.
- `:1211` and `:1252` — eviction.
- `:1282` — `Core::sleep`, op for op the Python: the same float operand order, `retain` and `shift_remove` keep dict
  order.
- `:1733` and `:1830` — export and import of `"sleep"`.
- `:1893` — `stats`.
- Three Rust unit tests.

**`src/ng_lite_core_py.rs:486`, `:506`** — `NGLiteCore.sleep(now)` (the report dict) and `sleep_state()`.

**`tests/test_ng_lite_core.py`**
- The tunable pin was updated by intent.
- `RefNGLite._import_state` restores the epoch.
- Seven new parity tests: interleaved sleeps under 3 configs, reload and cross-core reload, a rule population that
  exercises every branch, the eviction victim, epoch save/load, bad parameters, and the never-slept state.

### docs `cc-laptop-nglite-sleep-daemon-20261015` (on the live `d58cf0057`)

**Commit `bfd1d3729`.**

**`scripts/cc-ng-daemon.py`**
- `:1442` — `_COMMONS_SLEEP_ENABLED = CC_NG_COMMONS_SLEEP` in (1, true, yes). Default off.
- `:5321` — `_commons_sleep_after_cc()`.
- `:5407` — the call at the end of `_sleep_tick`, after `graph.sleep_cycle()`, its post-sleep save, the wall-time
  persist and the CC sleep log lines.
- It is not called in observe mode, or on wait or deferred ticks.
- It calls `STATE.commons.sleep()` and then `persist_cc_commons()`. One INFO line follows:
  `CC Commons sleep: epoch=N synapses a->b cleared=… persisted=…`.
- Every failure is a WARNING, wrapped twice. The CC's own sleep, save and last-sleep time are never affected.

**`scripts/tests/test_cc_ng_daemon_commons_sleep.py`** (new, 10 tests).

## 2. Harness (copy of the live Commons, both cores in lockstep)

### Setup

**Script and data.**
- Script: `~/.cache/nglite-sleep/tools/harness.py`. Results: `~/.cache/nglite-sleep/out/h-*.json`.
- Source: `cp` of `~/.claude/plugins/neurograph/checkpoints/commons.msgpack` at 20:57 (sha256 `0e2a0f98…`,
  `cmp`-verified). The live file was only read.
- The copy has 97 nodes, 5,000 synapses and 0 rim nodes. 4,995 of the 5,000 links have exactly one success, at
  w = 0.575.

**Isolation.**
- Runs went through `run-isolated.sh` (scratch HOME, tmpfs over `~/.claude/plugins/neurograph` and `~/.et_modules`,
  3.5 GB scope).
- Each (config, core) restores its own second copy in a temp dir.

**The wake model.** Each cycle has 160 experience deposits plus 14 metric deposits, then one sleep. The rates come
from the copy:
- the `last_updated` histogram shows about 400 deposits a day;
- `daemon.log` shows 2–3 CC sleeps a day, so a sleep every 9.6 h.

Each deposit lands on the node of a random recent deposit (its raw embedding plus 5e-4 noise) under a new
content-style target. 2% of deposits re-hit a target learned in the last 3 wakes, so that armor matters. "none"
means sleeps only.

**Parity.**
- Both cores start from the Python-loaded state, which the Rust core imports exactly. The canonical Python core
  re-normalises embeddings on load by ulps; that is the known quirk (b), fixed only in the Rust core.
- The same deposits and the same `now` go to both cores.
- **Every cycle of every run: the Python report equals the Rust report** (minus seconds / core), and the full
  states are equal (timestamps masked).
- The harness ran on `592ae59` + `541cf1b`. `4288822` changes only the pre-#1114-wheel path.

### Replay wake, 60 cycles (about 24 days)

| config | first clearance | synapses: start → cycle 30 → end | cleared total | cap evictions total / last cycle with one | youngest harness link the cap evicted | fewest sleeps a cleared harness link lived |
|---|---|---|---|---|---|---|
| **mind d0 0.1, h 0.05** | never | 5,000 → 4,999 → **4,999 (pinned)** | 0 | 9,347 / cycle 60 | 29 cycles | — |
| d0 0.1, h 0.5 | never | 5,000 → 4,999 → 4,999 (pinned) | 0 | 9,347 / cycle 60 | 29 cycles | — |
| **d0 0.2, h 0.5** | cycle 27 (662 cleared: the original cohort) | 5,000 → 4,343 → **4,360 (steady, about 158 cleared per sleep)** | 5,888 | 4,224 / cycle 27 | none | 26 |
| d0 0.3, h 0.5 | cycle 18 (2,090 cleared) | 5,000 → 2,920 → 2,927 (steady) | 8,749 | 2,820 / cycle 18 | none | 17 |

### Other findings

- **Recently learned links survive their grace by a wide margin.** The fewest sleeps a cleared harness link lived
  was 26 (d0 0.2) and 17 (d0 0.3). Grace only matters at the threshold. No link born in a wake was cleared or
  cap-evicted within its first 17 sleeps under any config.
- **Never the newest.** With the cap active, the youngest harness-born link it evicted was 29–30 cycles old, under
  the mind's values and under d0 0.1 / h 0.5.
- **Nothing frozen moves.** In the `--rim` run (40 cycles), the copy's second-busiest node `n_3` was given as a
  constitutional entry: the real node plus its 149 real synapses.
  - The 149 weights were bit-unchanged after every cycle, under all four configs.
  - Every report counted `frozen_synapses` 149 and `frozen_nodes` 1.
  - The other results matched the non-rim runs: d0 0.2 first clearance cycle 27, end 4,504; d0 0.3 cycle 18, end
    3,068.
- **No wake, 12 cycles.** Nothing reaches the threshold in 12 sleeps under any config. Median weight 0.575 →
  0.520 (mind) / 0.297 / 0.117 / 0.033. Under d0 0.3, 4,995 links are below 0.1 at cycle 12.
- **Nodes.** 97 → 93–96, from a few orphans collected after their links went. Frozen nodes were never counted as
  orphans.
- **Sleep time on a 5,000-link Commons.** Python core about 0.02–0.04 s, Rust core about 0.003–0.009 s (one outlier
  0.25 s under load).

### Closed-form lifetimes (computed from the rule, armor = 1 + ln n)

Sleeps until clearance (below 0.01, plus grace 2). Days at 2.5 sleeps a day are in parentheses. This is the armor
evidence.

| config | one-off (0.575, n=1) | n=2 (0.639) | n=3 (0.693) | n=7 (0.84) | n=20 (0.95) |
|---|---|---|---|---|---|
| mind 0.1 / 0.05 | 155 (62 d) | 285 | 377 | 621 | 935 (374 d) |
| 0.1 / 0.5 | 53 (21 d) | 93 | 118 | 180 | 258 |
| **0.2 / 0.5** | **27 (11 d)** | 47 (19 d) | 60 | 90 (36 d) | 129 (52 d) |
| 0.3 / 0.5 | 18 (7 d) | 31 | 40 | 61 | 86 (34 d) |

## 3. Parameter recommendation

**The mind's values do not transfer, because of scale.**
- The mind's sprouts are born at 0.1 of a 5.0 maximum (2%), so h = 0.05 sits at the faint end.
- NG-Lite links are born at 0.5 of 1.0 (50%), so under h = 0.05 every NG-Lite link is "established" from birth.
- The harness confirms it: 0 clearances in 24 days.

**Recommendation: `sleep_h = 0.5` (NG-Lite's neutral birth weight is the half-established point), `sleep_d0 = 0.2`,
`sleep_grace_sleeps = 2`, armor gain 1 / max 5.**
- A one-off experience fades in about 11 days. That is about the River's current memory horizon, which today is
  set by FIFO at the cap.
- Reinforced links are kept 2–5× longer.
- The Commons settles at about 4,360 synapses (87% of the cap) instead of pinning.
- The first sleeps clear nothing. The first clearance arrives about 27 sleeps after install (about 11 days), when the
  original cohort crosses together (662 links). After that it is about 158 per sleep, which is the deposit rate.
- Choose **d0 0.3** for a shorter, about 7-day horizon (about 2,920 links). That first clearance is bigger: 2,090 at
  sleep 18.

**These values are global defaults.** TID's and Elmer's NG-Lites get the same values when they start sleeping in
Phase 2. Their populations (routing outcomes with failures) are different and should get their own harness then.

## 4. Test deltas (base = branch points, same harness, isolated)

**Venvs.**
- base: `~/.cache/mean-centre/venv`, the installed wheel, `.so` `f5a79724…`.
- branch: `~/.cache/nglite-sleep/venv`, the new wheel only, deps through `.pth`.

| suite | base | branch | new failures |
|---|---|---|---|
| ng-tract-rs `cargo test --lib` | (mean-centre lane) | **72 passed** (incl. 3 new) | — |
| ng-tract-rs `tests/` (NGLITE_PY = base / branch `ng_lite.py`) | 404 passed | **411 passed** (+7 new) | none |
| NG `test_ng_lite.py`, `test_dim_guard`, `test_mean_centre`, `test_commons*` (14 files), `test_cc_commons_persist`, `test_ng_commons_eco`, `test_constitutional` | 2 failed / 243 passed | **2 failed / 266 passed** on the final head (243 + the 23 new `test_ng_lite_sleep.py`, Python and Rust core); `test_commons` axiom guard updated by intent (§6, item 7) | none. The same 2 NGSaaSBridge `_sync_count` failures as base. |
| daemon `scripts/tests -k "sleep or commons"` (`CC_NG_PYTHONPATH` = NG base / branch) | 86 passed | **96 passed** (+10 new) | none |

**Re-runs on the final heads (after `4288822`).** The rs `test_ng_lite_core.py` (39 passed) and the NG subset above
were both re-run.

**Install-order check.** The old wheel with the new `ng_lite.py` and the switch on: `Commons.stats()` /
`sleep_state()` work (epoch 0), and `sleep()` raises a clear RuntimeError. The new wheel with the old `ng_lite.py`
fails 18 parity tests. The old `ng_lite.py` lacks the sleep keys and the eviction rule, so it is not a supported
pairing. **Install the code and the wheel together.**

## 5. Install (for the Executive; Josh's go first)

1. **Decide the parameters first. They are sticky.**
   - NG-Lite saves its config in its state, and the saved config wins over `DEFAULT_CONFIG` on load.
   - The live `commons.msgpack` has no sleep keys yet. The first load after install takes the `DEFAULT_CONFIG`
     values, and the first save writes them into the Commons for good. After that only `update_tunable` changes
     them.
   - There is no env or admin path to it today. Elmer's tuner is broken (#1106).
   - Options:
     - **(a)** Keep the mind's values. Sleep is then a near no-op for the Commons, but cap eviction is fair.
     - **(b, recommended)** Change `DEFAULT_CONFIG` to `sleep_h 0.5`, `sleep_d0 0.2` on this branch (both cores'
       mirrors and the tunable test) before install.
     - **(c)** Add a daemon env override (`CC_NG_COMMONS_SLEEP_D0` / `_H`, applied with `update_tunable` at Commons
       init; LAW 5). Not built.
2. **Back up.** `~/.claude/plugins/neurograph/checkpoints/commons.msgpack`, `~/.et_modules/elmer/ng_lite_state.*`,
   `/home/josh/The-Inference-Difference-vps/ng_lite_state.*` (or wherever TID's lives).
3. **NG code into the trial worktree.** `git -C ~/NeuroGraph-worktrees/trial-s4-20261002 fetch origin && git -C
   ~/NeuroGraph-worktrees/trial-s4-20261002 merge --ff-only origin/cc-laptop-nglite-sleep-20261015`.
   - The trial is at `fe67f88`. The branch is `fe67f88` + `8121409` (#1115) + the 3 commits, so **this also installs
     #1115's `ng_embed.py`**.
4. **Daemon.** `git -C ~/docs/.claude/worktrees/trial-s4a-20261002 merge --ff-only
   origin/cc-laptop-nglite-sleep-daemon-20261015`.
   - The live branch was at `d58cf0057`, which is 1 commit ahead of its origin. This branch contains that commit, so
     pushing it published `d58cf0057` too.
5. **Wheel.** `~/NeuroGraph/.venv/bin/pip install --no-deps --force-reinstall
   ~/.cache/nglite-sleep/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`.
   - wheel sha256 `dc136f0f488d09737e60762de71fc4b2935d3a7d6d1adb8b91d337852db88ad8`
   - `.so` sha256 `1c1229da61d4dd2662f1b153dafc68df5cd4b700f3d7a29b45f84dd7ffa8a98a`
   - Built from `97a241e` with `~/.cache/nglitecore/build.sh` (`cargo test --lib` + `maturin build --release`,
     `-j 1`, offline).
   - Confirm: `~/NeuroGraph/.venv/bin/python -I -c "import ng_tract; print(hasattr(ng_tract._NGLiteCore,'sleep'))"`
     → `True`.
6. **Re-vendor to Elmer and TID, byte-identical.** Copy `ng_lite.py` (sha `24c868a2…`) **and** `ng_embed.py` (sha
   `a097b443…`) from the trial worktree into `~/Elmer/` and `~/The-Inference-Difference-vps/` (TID's live process
   runs that checkout), then commit there on a branch.
   - Both modules run `ng_updater.auto_update()` at start with `NEUROGRAPH_CANONICAL_PATH` = the trial worktree:
     - Elmer: from the registry `env`.
     - TID: from its process env.
   - So they would copy `ng_lite.py` themselves. They would **not** copy `ng_embed.py`, which is not in
     `VENDORED_FILES`. Copy both by hand.
   - In the daemon process, Elmer already imports the trial's `ng_lite` from `sys.modules`.
   - Phase 1 effect on them: the new at-cap eviction rule only. They do not sleep until Phase 2.
7. **Env (LAW 5).** Add `export CC_NG_COMMONS_SLEEP=1` to `~/.bashrc`.
   - It needs `CC_NG_SLEEP=1` (on) and observe off (it is).
   - `cc-ng-service.py prepare` passes `CC_*` exports through.
8. **Restart once.** Use the sidecar procedure: every previous daemon PID dead first. Restart TID only for its
   re-vendored copy; that can wait for Phase 2.
9. **Watch at least 2 full CC sleep cycles.**
   - Each `CC sleep (…)` line in `daemon.log` should be followed by `NG-Lite sleep [commons] epoch=N …` and
     `CC Commons sleep: epoch=N … persisted=True`.
   - The epoch should go up by 1 per CC sleep.
   - The `commons.msgpack` mtime should advance, and its `"sleep"` key should be present.
   - Expect clearance only after about 27 sleeps (recommended values) or never (mind's values). "The synapse count
     stops pinning" is a multi-week observation, not a 2-cycle one.

## 6. Risks / found, not fixed

1. **Sticky parameters** (§5, step 1).
2. **The first clearance is a cohort.** With d0 0.2 it is 662 links at about sleep 27; with d0 0.3, 2,090 at sleep
   18. Expected, not a fault (cf. the mind's backlog clearance).
3. **Python core only: the sleep is not atomic against a concurrent deposit.** It works on snapshots, so a deposit
   that lifts a link in the same instant the sleep clears it can be lost. The live daemon runs the **Rust core**
   (`CC_NG_LITE_RUST_CORE=1`), where the call is atomic under the core's mutex.
4. **Install order.** The wheel and the code go together (§4). The old wheel with the new code fails loud on
   `sleep()` only.
5. **Rollback.**
   - Switch: unset `CC_NG_COMMONS_SLEEP` and restart. The Commons stops sleeping; the new cap eviction stays.
   - Full: reset the trial and daemon worktrees to `fe67f88` / `d58cf0057` and reinstall the mean-centre wheel
     (`~/.cache/mean-centre/wheel/…`, `.so` `f5a79724…`).
   - Slept states load in the old code: the metadata counters stay as inert metadata and the `"sleep"` key is
     ignored.
   - **Downscaled weights are not reverted.** Restore the step-2 backup for that.
6. **Design §1 fault 3 is inaccurate for NG-Lite.**
   - A new link is born at 0.5, which is 0.575 after its first success. It is not born at the lowest weight.
   - On the Commons, 4,995 of 5,000 links tie at 0.575, so the old weight-only eviction was in effect **FIFO**:
     `min` returns the first in insertion order. The newest was not being cut.
   - The real faults stand: deposit-gated, weight-only, and the cap decides everything.
7. **The Commons axiom guard** (`tests/test_commons.py::test_no_send_surface_exists`) needed `sleep` and
   `sleep_state` added to its allow-list. The forbidden send/route check is unchanged. Rule on it.
8. **Shared temp name (punch list candidate).** `ng_lite._atomic_write` and the Rust `save_binary` both use
   `<path>.tmp.<pid>`. Any two threads of one process saving the same NG-Lite file can collide. This is fixed here
   only for `persist_cc_commons`, with a lock. Syl's `neurograph_rpc.py` Commons persist paths are worth a look.
9. **Not in scope.**
   - Syl's VPS Commons riding her NeuroGraph's sleep (design §2.2, last bullet).
   - Phase 2 (Elmer / TID bucket the epoch; #1106).
   - The env override (§5, step 1, option c).
   - A per-module harness for TID / Elmer.

## 7. Branches and heads

| repo | branch | head | base |
|---|---|---|---|
| NeuroGraph | `cc-laptop-nglite-sleep-20261015` | (this doc's commit) | `8121409` |
| ng-tract-rs | `cc-laptop-nglite-sleep-rs-20261015` | `97a241e` | `f9a2c79` |
| docs (daemon) | `cc-laptop-nglite-sleep-daemon-20261015` | `bfd1d3729` | `d58cf0057` |

Base worktrees for the suite comparison: `~/worktrees/nglite-sleep-{ng,rs,daemon}-base` (detached).
