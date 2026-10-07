# Sleep phase P2 — disuse in sleep (review branches)

*2026-10-07 · lane sleep-p2 (bounded build lane for the Executive) · review branches only: nothing merged, nothing
deployed, nothing armed, nothing installed into the NG venv; the live daemon, its venv, its checkpoint and the trial
worktrees untouched.*

Spec: `~/docs/superpowers/specs/2026-10-06-sleep-phase-design.md` — §3 (rule, grace counters), §1A (substrate audit),
§8 row P2 with the audit amendments (its proof bar is the readiness bar), D1-D4, D11, D13, D14. Josh approved the design
as recommended (2026-10-06) and said to start P2 ("Let's do both, as much as the hardware permits").
`neuro_foundation.py` is PROTECTED: one protected commit; merging needs Josh's protected-file "proceed". No vendored file
is touched (LAW 2): the branch diff vs `149fa1f` is `neuro_foundation.py`, `tests/test_sleep_p2.py` and this file.

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-sleep-p2-20261007` | P1 tip `149fa1f` | **PROTECTED** `da23ec6` · tests `812df0b` · this doc |
| ng-tract-rs | `cc-laptop-sleep-p2-rs-20261007` | `4b00431` (P2b wheel line) | `cba74b1` `SynapseStore.scale_strength_aware` |
| docs (daemon) | `cc-laptop-sleep-p2-daemon-20261007` | `2d9f79d5` (trial s4a tip) | `0b42c9c5` env names + tests |

**Verdict in one paragraph.** With every new key absent the engine is byte-identical to `149fa1f` (golden copy run,
whole-run tests). With disuse on, on a fresh copy of the live checkpoint (93,085 synapses), 12 wake/sleep cycles with the
chosen parameters: no protected node lost a lifeline; the synapse count settled at **~54-57K** (55.9K after sleep 12,
no trend over sleeps 3-12) instead of growing; the first clearance removed 45,692 (the backlog). **The "established links
lose < 10%" bar FAILS as written** (peak >= 0.5: -30.2%; excluding the 504 that were already below 0.01 at the start:
-12.4%); links that are strong now (w >= 0.5 at the start) lost 7.1%. Today's per-step rules lose 8.5% / 3.2% / 2.0%
over the same 3,000 wake steps. The cause is measured, not guessed: wake competition pushes some once-strong links under
0.01, and the sleep clears a link after G sleeps under 0.01 where today's weight rule waits 5,000 steps; the downscale is
not the driver (the no-downscale control loses at the same rate, §3; the 12-cycle comparison is §4.3). The first big clearance holds `_step_lock` for
**2-16 s** on this machine (steady state 0.3-2.3 s) — too long to arm without chunking or an announced window.
Parameters chosen from the sweep: **d0 = 0.1, h = 0.05, G = 2 sleeps, kappa = 0.015** (last-link grace 2 sleeps, D4,
not swept). G and kappa differ from the design's D3 / D14 recommendations (1 / 0.15); the sweep chose them (§3).

---

## 1. What changed

### Engine — PROTECTED commit `da23ec6` (`neuro_foundation.py`)

All new behaviour sits behind config keys that are **absent by default** and NOT in `DEFAULT_CONFIG` (read live with
absent defaults), so a graph that never sets them runs and checkpoints exactly as `149fa1f` (Syl unchanged).

- **`sleep_disuse_enabled`** (switch) + five REQUIRED parameters: `sleep_downscale_d0`, `sleep_downscale_h`,
  `sleep_weight_grace_sleeps` (G), `sleep_last_link_grace_sleeps`, `sleep_credit_shield_kappa`. With the switch on,
  `Graph.sleep_cycle()` runs the new `_sleep_cycle_disuse` instead of the P1 cycle. It **refuses** (ValueError, before
  anything is touched) if a parameter is missing / out of range, or if `structural_plasticity_in_sleep` is off (step()
  would otherwise keep advancing the counter per step and the grace would no longer be in sleeps).
- **One `_step_lock` hold**, in the spec's §2 order:
  0. **Migration** (`_sleep_migrate_counters`) on the first disuse sleep: every `low_weight_steps` -> 0 and every
     last-link stamp dropped (step stamps `last_link_since` and sleep stamps `last_link_since_sleep`), so the first sleep
     only tags. Marks config `sleep_low_weight_unit = "sleeps"`, `sleep_cycles_completed = 0`.
  1. **Downscale** (`sleep_downscale_strength_aware(d0, h)`, public, D1 + D11): every synapse
     `d = d0·h/(h+w)/max(salience, 1)`, `w <- w·(1-d)`. **Weight only** (eligibility trace, salience, peak, counters,
     delay untouched), then the existing strongest-link guarantee of every protected node (pre-pass strongest out / in
     link ends >= min(pre-pass w, 2·wt)), as `sleep_downscale`. Native `SynapseStore.scale_strength_aware` when the
     wheel has it, else `_scale_strength_aware_python` — **bit-identical** (30 parametrized tests). d0 = 0 skips it.
  2. **Clearance**: the EXISTING `_prune_synapses` with new keyword-only, default-None sleep-unit arguments (the want-hub
     pattern; the default path is untouched): the native rule sweep with `grace = G` (counted once per sleep),
     `inactivity = inf` (activity clause off, §6) and `initial_w = 0` (age clause off, D2: `peak < 0` is never true);
     the lifelines exactly as before; then the **D14 pending-credit shield** (a rule-chosen id is kept while
     `w + kappa·max(trace, 0) >= wt`; it keeps its count and is re-tested next sleep); then the **last-link grace in
     sleeps** (D4: `_last_link_grace` gains keyword-only `now` / `grace` / `stamp_key`; the sleep path stamps
     `last_link_since_sleep` with the sleep index).
  3. **Orphan collection** — the existing function, unchanged.
  4. `sleep_cycles_completed += 1`; one `"sleep_cycle"` event + one INFO line with: sleep index, migrated, counters
     reset, downscaled / clamped, eligible, shield held (+ ids in the record, never in the log), last-link held, cleared,
     collected, synapses / nodes before and after, below-threshold count after, per-part timings, total seconds.
- **Marker drop on the step path**: a default-path (step-unit) `_prune_synapses` deletes config
  `sleep_low_weight_unit` if present, so a counter that step() advanced (e.g. after a rollback of `CC_NG_SLEEP`) is
  re-migrated by the next disuse sleep instead of being read as sleeps. With the key absent this is one dict lookup.
- Nothing else: no rule, predicate or default of the existing step path changes; `compete_protected_links` untouched.

### Rust — `cba74b1` (`ng-tract-rs`, `src/store.rs`, pure addition)

`SynapseStore.scale_strength_aware(d0, h, protected, floor)`: one column pass beside `scale_all` (same
`strength_guard` helper, args extracted before the borrow, dict built after — the 25c5f52 rule). 3 new Rust unit tests
(`cargo test --lib`: 10/10 strength tests pass). Wheel built with `cargo -j 1` into `~/.cache/sleep-p2/wheel/`
(installed only into the throwaway `~/.cache/sleep-p2/venv`, never into the NG venv). Measured: 0.0006 s native vs
0.27 s Python fallback at 45K synapses (isolated); inside sleep cycles under load the native downscale took 0.02-0.24 s
(including the protected-node probe), the fallback 0.45-2.5 s at 93K.

### Daemon — `0b42c9c5` (`scripts/cc-ng-daemon.py`, not protected; NOTHING armed)

`CC_SNN_CONFIG` gains, explicit both ways like the strength budget: `sleep_disuse_enabled` = `CC_NG_SLEEP` **and**
`CC_NG_SLEEP_DISUSE` (default off), `sleep_downscale_d0` (`CC_NG_SLEEP_D0`, 0.1), `sleep_downscale_h`
(`CC_NG_SLEEP_H`, 0.05), `sleep_weight_grace_sleeps` (`CC_NG_SLEEP_WEAK_GRACE_SLEEPS`, 2),
`sleep_last_link_grace_sleeps` (`CC_NG_SLEEP_LAST_LINK_GRACE_SLEEPS`, 2), `sleep_credit_shield_kappa`
(`CC_NG_SLEEP_CREDIT_KAPPA`, 0.015). The sleep tick logs one more INFO line with the disuse counts (never ids). Env-name
pin test updated; `scripts/tests/test_cc_ng_daemon_sleep_p2.py` (14 tests, incl. one tick of the REAL branch engine
configured from `CC_SNN_CONFIG`: sleep 1 migrates and only tags, sleep 2 keeps a faint link under G = 2).

## 2. Keys absent: byte-identical to `149fa1f` (readiness bar, part 1)

Fresh copy: `~/.cache/sleep-p2/ckpt/main.msgpack` + `vectors.msgpack` (mode 444), copied from the live checkpoint saved
2026-10-07 07:52:27Z (manifest: 11,783 nodes, 93,085 synapses, 1,710 hyperedges, t = 89,965, engine `149fa1f`),
sha256 `0b2ca0e563e390bf…` / `bcca35e044220bb5…`. Every run below went through `~/.cache/sleep-p2/run-isolated.sh` (the
p2a harness: scratch HOME, PID namespace, tmpfs over the live plugin dir, read-only binds, MemoryMax 3-4.5 GB, no swap,
`nice 10`, waits for MemAvailable >= 4 GB), one worker, wall-clock timers only.

| check | base `149fa1f` | branch | result |
|---|---|---|---|
| golden copy run (`scripts/ckpt_equiv_p2.py`: seeded random + uuid4; 12 steps × 30 stimulated; write-mode Tonic tick every 3rd step; per-step fired ids, node state, synapse id/weight/eligibility, pred_weights hashed; 147 step prunes) | trace `259b82efb715cb19…`, checkpoint `f90576e583195393…` | identical | **equal** |
| P1 `sleep_cycle` with `structural_plasticity_in_sleep` on, disuse absent, grace 300 / last-link 3 (`sleep_on_copy.py`): removed set, collected nodes, every survivor's counters / weight / stamp | 80,856 removed, ids `51de16b9…`, survivors `0a6eec59…` | identical | **equal** |
| same, after 6 warm steps | 0 removed, survivors `cf0be031…` | identical | **equal** |
| `tests/test_sleep_p2.py::test_keys_absent_is_the_p1_tip_exactly` — the P1 whole-run workload (step, Tonic ticks with aging, recall, node churn, rewards, want-hub competition, `sleep_downscale`, P1 `sleep_cycle` every 6 rounds, snapshots; per-step synapse state + pred_weights bitwise; final checkpoint bytes), 3 flag sets × 4 seeds × dict / native node store × with / without `structural_plasticity_in_sleep` | — | — | **48/48** |

## 3. The parameter sweep (wake on)

**Harness** (`~/.cache/sleep-p2/scripts/drywake.py`, one isolated process per run, same seeds in every run so wake 1
is identical everywhere and runs diverge only through the sleep): restore the copy; `structural_plasticity_in_sleep` on;
per cycle a **250-step drain-like wake** — 10 deposits, each a fresh deposit node plus the 20 existing nodes nearest
(cosine) to a **real deposit's stored embedding** (a seeded sample of 600 `cc:conv` entries from the vector-store copy;
only ids, embeddings and text sha256s were extracted) stimulated at 1.5, one step, reward 0.1 (a landed deposit turn),
then 24 idle steps — then one `sleep_cycle()`. Wake re-lifts are attributed by wrapping (harness side, per instance)
`HomeostaticRule.apply`, `StrengthBudgetRule.apply` and `Graph.inject_reward` with native `weights_copy()` before/after.
Engine code is the branch, venv `~/.cache/sleep-p2/venv` (new wheel) except `A-d0.1-h0.05` (P2b wheel: Python-fallback
downscale; same results by bit-identity, slower lock). 4 cycles per run (6 for the first), set by the hardware: a wake
took 5-12 min at load 6-17.

| run | d0 | h | G | kappa | synapses after sleep 1 / 2 / 3 / 4 | cleared at sleep 2 / 3 / 4 | shield held at sleep 2 | of those: crossed / cleared | est. (peak >= 0.5) lost | non-backlog est. lost | w >= 0.5 after sleep 4 | lock hold s, sleeps 1-4 | lifelines lost |
|---|---:|---:|---:|---:|---|---|---:|---|---:|---:|---:|---|---|
| A-d0.1-h0.05 | 0.1 | 0.05 | 1 | 0.15 | 95,639 / 71,734 / 53,803 / 53,470 | 26,446 / 20,526 / 3,105 | 19602 | 776 / 18990 | 23.7% | - | 653 | 1.7 / 9.0 / 16.7 / 1.5 | none |
| A-d0.05-h0.05 | 0.05 | 0.05 | 1 | 0.15 | 95,639 / 72,361 / 55,227 / 51,834 | 25,819 / 19,719 / 6,089 | 18668 | 495 / 18259 | 24.3% | 5.7% | 519 | 2.3 / 9.1 / 2.3 / 1.8 | none |
| A-d0.2-h0.05 | 0.2 | 0.05 | 1 | 0.15 | 95,639 / 69,143 / 50,803 / 50,143 | 29,046 / 20,983 / 3,505 | 20219 | 1611 / 18991 | 25.4% | 6.5% | 700 | 1.2 / 4.0 / 6.4 / 0.5 | none |
| A-d0.1-h0.02 | 0.1 | 0.02 | 1 | 0.15 | 95,639 / 71,554 / 54,713 / 55,464 | 26,626 / 19,416 / 2,023 | 18325 | 625 / 17770 | 22.8% | 4.6% | 733 | 0.9 / 4.1 / 2.0 / 0.5 | none |
| A-d0.1-h0.1 | 0.1 | 0.1 | 1 | 0.15 | 95,639 / 71,186 / 53,630 / 51,218 | 26,953 / 20,239 / 5,027 | 19592 | 1600 / 18049 | 24.7% | 6.0% | 504 | 1.5 / 2.1 / 1.7 / 3.4 | none |
| C-d0-none | 0.0 | 0.05 | 1 | 0.15 | 95,639 / 73,437 / 56,688 / 56,495 | 24,766 / 19,307 / 3,009 | 18287 | 488 / 17809 | 22.8% | 4.8% | 579 | 0.9 / 6.2 / 2.8 / 4.1 | none |
| B-G2 | 0.1 | 0.05 | 2 | 0.15 | 95,639 / 98,180 / 55,170 / 56,495 | 0 / 45,510 / 1,393 | 0 | 130 / 49 | 21.3% | 3.0% | 527 | 1.6 / 0.3 / 7.6 / 0.8 | none |
| B-k0.015 | 0.1 | 0.05 | 1 | 0.015 | 95,639 / 53,778 / 53,317 / 53,615 | 44,402 / 3,191 / 2,379 | 1634 | 350 / 1288 | 24.0% | 5.7% | 708 | 2.4 / 10.5 / 6.1 / 0.5 | none |
| B-k0 | 0.1 | 0.05 | 1 | 0.0 | 95,639 / 52,152 / 53,195 / 51,799 | 46,028 / 1,696 / 4,153 | 0 | 0 / 0 | 25.3% | 6.5% | 810 | 1.0 / 16.0 / 0.6 / 0.5 | none |
| baseline-sq | - | - | - | - | 82,413 / 81,918 / 75,158 / 74,608 | 3,275 / 9,339 / 3,216 | None | 0 / 0 | 4.4% | 3.2% | 832 | 0.0 / 0.0 / 0.0 / 0.0 | none |

(`baseline-sq` has no sleep: its "cleared" column is what today's per-step rules removed during wakes 2-4 (13,205 in wake 1), and it has no lock column. `A-d0.1-h0.05` ran with the Python-fallback downscale and predates the non-backlog metric. All runs: same seeds, same copy; wake 1 identical in every run.)

Reading the sweep:
- **Lifelines**: none lost in any run (every protected node's strongest in / out link survived every sleep).
- **The first clearance is the backlog** (~45K of 93K, 48.7% below 0.01 at the copy): 44-47K are cleared by the first
  clearing sleep whatever d0 / h are; with kappa = 0.15 about 19K of them are held one sleep longer.
- **d0 / h move the result less than the wake does.** At sleep 4 the synapse count spans 50.1K (d0 0.2) to 56.5K (no
  downscale) and the non-backlog established loss 4.6-6.5% vs **4.8% with no downscale at all** (`C-d0-none`). Run-to-run
  wake variation (firing bursts of 600-900 nodes in some 25-step windows) is of the same size. A finer d0 / h choice is
  not resolvable on this hardware in 4-cycle runs. **Chosen: d0 = 0.1, h = 0.05** — the middle of a flat region, bounded
  on both sides by measured runs; it keeps the design's intended unused-link timescales (table below) and its
  established-link loss is within ~1 point of the no-downscale control.
- **G**: G = 2 halved the non-backlog established loss at sleep 4 (3.0% vs 4.6-6.5% for every G = 1 run) and matched
  today's per-step rules (3.2%, `baseline-sq`), at the cost of a faint link living one more sleep (synapses at sleep 4:
  56.5K vs ~53.5K). The 12-cycle pair confirms it (§4.3). **Chosen: G = 2** (D3 recommended 1).
- **kappa (D14)**: wake 1 and 2 are identical in `A-d0.1-h0.05` (kappa 0.15) and `B-k0.015`, so their first clearances
  are directly comparable. kappa 0.15 held **19,602** links (42% of the 46,080 eligible); in the next wake **504** of
  them crossed back over 0.01 (2.6%) and 18,847 were cleared one sleep later. kappa 0.015 held **1,634**; **345** crossed
  (21%). The extra ~18K links held by 0.15 yielded ~160 more crossings (<1%) and a second large clearance (20.5K, lock
  9-17 s) one sleep later. kappa 0 (`B-k0`) shields nothing. **Chosen: kappa = 0.015** (one maximum reward; D14
  recommended 0.15 as the ceiling and named 0.015 as the alternative).
- **Status quo** (`baseline-sq`: today's per-step rules, grace 5000 steps + age clause, no sleep): on this copy it also
  shrinks over 1,000 steps (82.4K -> 74.6K), because old sprout cohorts cross the 5000-step deadlines — the deadline
  pattern Josh ruled out.

**Timescales of an UNUSED link at the chosen d0 = 0.1, h = 0.05** (sleep-only closed iteration
`w <- w(1 - d0·h/(h+w))` until w < 0.01, then G = 2 sleeps to clear; wake re-strengthening / competition not included;
days at 2-4 sleeps a day):

| starting weight | sleeps to fall under 0.01 | + G = 2 to clear | days (4 / 2 sleeps a day) |
|---:|---:|---:|---|
| 0.011 | 2 | 4 | 1 - 2 |
| 0.02 | 9 | 11 | 3 - 6 |
| 0.1 (a nascent sprout; the retired age rule's target, D2) | 41 | 43 | 11 - 22 |
| 0.5 | 137 | 139 | 35 - 70 |
| 1.0 | 244 | 246 | 62 - 123 |
| 5.0 (max_weight) | 1,060 | 1,062 | 266 - 531 |

(Other points of the sweep: 0.1 -> under 0.01 in 82 sleeps at d0 0.05 / h 0.05, 20 at d0 0.2, 68 at h 0.02, 32 at
h 0.1; `~/.cache/sleep-p2/scripts/timescale.py`.) In the dry run, links fall under 0.01 much faster than this because
wake competition (strength budget, homeostatic down-scaling) pushes them; the downscale is the gentle pressure on links
that are simply unused, as §3.1 intends.

## 4. The proof run: 12 wake/sleep cycles with the chosen parameters (`F-chosen`)

d0 0.1 · h 0.05 · G 2 · last-link 2 · kappa 0.015 · 250-step wakes · 3,000 wake steps · total 60 min wall clock.

Start (copy, t=89,965): 93,085 synapses; faint (< 0.01) 45,328 (42,341 never potentiated); w >= 0.5: 451; established (peak >= 0.5): 2,338.

| cycle | wake step median / p90 s | fired median per 25 steps | sprouted | faint re-lifted by wake (homeostatic / reward / other) | fell under | shield held | eligible | cleared | last-link held | orphans collected | synapses after | w >= 0.5 | est. alive (all / non-backlog) | lock hold s (migrate / downscale / clearance / orphans) | lifelines lost |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1.78 / 6.20 | 38 58 117 744 767 628 303 164 121 83 | 2500 | 704 (486 / 218 / 0) | 7766 | 0 | 0 | 0 | 0 | 5 | 95639 | 367 | 2338 / 1834 | 3.41 (1.38 / 0.03 / 1.98 / 0.02) | none |
| 2 | 0.49 / 2.88 | 88 88 83 96 88 93 83 124 777 832 | 2500 | 9913 (9831 / 82 / 0) | 1811 | 0 | 0 | 0 | 0 | 9 | 98180 | 574 | 2338 / 1834 | 0.44 (0.00 / 0.02 / 0.40 / 0.01) | none |
| 3 | 0.87 / 3.05 | 708 380 84 35 27 39 61 75 72 56 | 2500 | 1032 (529 / 503 / 0) | 6267 | 7 | 45748 | 45692 | 49 | 5 | 54988 | 472 | 1846 / 1787 | 8.62 (0.00 / 0.03 / 8.57 / 0.03) | none |
| 4 | 0.69 / 1.48 | 38 29 71 178 479 613 461 201 87 95 | 2500 | 3245 (2950 / 295 / 0) | 2244 | 12 | 1699 | 1654 | 33 | 10 | 56099 | 531 | 1834 / 1775 | 0.43 (0.00 / 0.04 / 0.38 / 0.01) | none |
| 5 | 0.83 / 2.04 | 75 55 65 86 223 252 129 58 68 101 | 2500 | 3812 (3779 / 33 / 0) | 2193 | 19 | 3244 | 3225 | 0 | 11 | 55506 | 719 | 1785 / 1742 | 1.24 (0.00 / 0.06 / 1.17 / 0.02) | none |
| 6 | 0.89 / 1.99 | 493 618 566 332 54 54 68 71 54 54 | 2500 | 382 (363 / 19 / 0) | 6342 | 1 | 1755 | 1754 | 0 | 10 | 56452 | 709 | 1774 / 1731 | 0.74 (0.00 / 0.06 / 0.65 / 0.03) | none |
| 7 | 1.18 / 2.18 | 59 62 124 166 94 316 796 770 626 373 | 2500 | 1819 (1223 / 596 / 0) | 2915 | 48 | 2249 | 2197 | 4 | 8 | 57051 | 737 | 1759 / 1716 | 2.05 (0.00 / 0.18 / 1.85 / 0.01) | none |
| 8 | 0.30 / 0.71 | 110 73 51 43 45 47 53 52 62 66 | 2500 | 2766 (2412 / 354 / 0) | 2349 | 4 | 5186 | 5167 | 15 | 9 | 54436 | 779 | 1709 / 1674 | 1.41 (0.00 / 0.05 / 1.32 / 0.04) | none |
| 9 | 0.52 / 1.16 | 69 151 291 447 571 273 58 60 67 60 | 2500 | 1524 (1431 / 93 / 0) | 2282 | 0 | 2401 | 2389 | 12 | 8 | 54752 | 812 | 1689 / 1657 | 0.61 (0.00 / 0.14 / 0.46 / 0.01) | none |
| 10 | 0.51 / 2.94 | 71 103 80 40 35 46 635 940 914 866 | 2500 | 796 (758 / 38 / 0) | 2797 | 82 | 1769 | 1684 | 3 | 10 | 55660 | 855 | 1681 / 1650 | 2.21 (0.00 / 0.24 / 1.95 / 0.02) | none |
| 11 | 0.34 / 2.75 | 676 488 233 35 39 42 46 42 58 70 | 2500 | 403 (384 / 19 / 0) | 7470 | 0 | 2271 | 2257 | 14 | 9 | 55903 | 612 | 1658 / 1631 | 0.57 (0.00 / 0.02 / 0.54 / 0.01) | none |
| 12 | 0.52 / 2.46 | 56 33 25 41 50 90 890 925 783 523 | 2500 | 5393 (5311 / 82 / 0) | 1793 | 85 | 2686 | 2583 | 18 | 8 | 55897 | 819 | 1632 / 1606 | 0.43 (0.00 / 0.02 / 0.39 / 0.01) | none |

Weight histogram after each sleep (=0, (0,1e-4), [1e-4,1e-3), [1e-3,0.01), [0.01,0.05), [0.05,0.1), [0.1,0.5), [0.5,1), [1,2), >=2):

| cycle | =0 | (0,1e-4) | [1e-4,1e-3) | [1e-3,0.01) | [0.01,0.05) | [0.05,0.1) | [0.1,0.5) | [0.5,1) | [1,2) | >=2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14,623 | 1,766 | 6,675 | 31,486 | 34,398 | 4,620 | 1,704 | 165 | 68 | 134 |
| 2 | 6,957 | 3,684 | 10,507 | 27,017 | 38,345 | 7,996 | 3,100 | 254 | 138 | 182 |
| 3 | 86 | 9 | 98 | 9,564 | 36,677 | 5,942 | 2,140 | 193 | 127 | 152 |
| 4 | 145 | 39 | 366 | 7,925 | 37,702 | 6,959 | 2,432 | 233 | 120 | 178 |
| 5 | 50 | 51 | 454 | 3,830 | 34,138 | 11,869 | 4,395 | 297 | 190 | 232 |
| 6 | 249 | 212 | 398 | 9,062 | 36,343 | 6,782 | 2,697 | 299 | 200 | 210 |
| 7 | 129 | 169 | 426 | 9,497 | 36,826 | 6,766 | 2,501 | 304 | 169 | 264 |
| 8 | 56 | 136 | 291 | 5,243 | 37,047 | 7,799 | 3,085 | 290 | 168 | 321 |
| 9 | 160 | 85 | 243 | 4,281 | 35,485 | 9,931 | 3,755 | 316 | 186 | 310 |
| 10 | 106 | 35 | 248 | 5,321 | 32,618 | 11,406 | 5,071 | 354 | 185 | 316 |
| 11 | 94 | 21 | 264 | 11,830 | 35,945 | 5,024 | 2,113 | 228 | 158 | 226 |
| 12 | 65 | 11 | 231 | 6,861 | 37,017 | 7,710 | 3,183 | 345 | 225 | 249 |

Protected nodes after each sleep (out / in degree; lifeline out / in weight):

| cycle | cc:want::372ea08cab33e71c | cc:want::4625485108d2e9be | cc:want::b73925f91509aadf | constitutional::rim::choice_clause |
|---:|---|---|---|---|
| 1 | 12 / 12; 2.724 / 1.150 | 2163 / 16; 0.247 / 0.037 | 308 / 34; 0.506 / 0.051 | 1671 / 35; 0.093 / 0.076 |
| 2 | 12 / 12; 2.412 / 1.692 | 2163 / 16; 4.570 / 0.051 | 308 / 35; 0.720 / 0.086 | 1781 / 35; 0.230 / 0.191 |
| 3 | 4 / 10; 2.511 / 1.478 | 143 / 6; 4.995 / 0.079 | 248 / 17; 0.785 / 0.065 | 318 / 14; 0.143 / 0.081 |
| 4 | 4 / 10; 1.641 / 1.248 | 104 / 6; 3.950 / 0.125 | 242 / 17; 0.895 / 0.073 | 274 / 14; 0.319 / 0.100 |
| 5 | 4 / 10; 1.672 / 2.839 | 75 / 6; 3.990 / 0.176 | 238 / 14; 0.977 / 0.060 | 263 / 12; 0.349 / 0.188 |
| 6 | 4 / 10; 1.648 / 2.052 | 126 / 6; 1.068 / 0.110 | 235 / 14; 0.887 / 0.083 | 337 / 12; 0.119 / 0.090 |
| 7 | 4 / 10; 1.366 / 1.774 | 288 / 6; 0.380 / 0.092 | 233 / 14; 0.662 / 0.069 | 405 / 12; 0.234 / 0.111 |
| 8 | 4 / 10; 1.787 / 2.458 | 275 / 6; 4.143 / 0.053 | 216 / 13; 1.029 / 0.053 | 337 / 11; 0.387 / 0.127 |
| 9 | 4 / 10; 1.686 / 2.793 | 206 / 6; 4.995 / 0.116 | 213 / 9; 0.989 / 0.100 | 273 / 10; 0.291 / 0.192 |
| 10 | 4 / 10; 1.281 / 2.287 | 326 / 6; 1.803 / 0.200 | 210 / 10; 1.043 / 0.127 | 468 / 11; 0.272 / 0.178 |
| 11 | 4 / 10; 0.765 / 1.349 | 290 / 6; 0.935 / 0.039 | 202 / 10; 0.678 / 0.047 | 451 / 11; 0.149 / 0.052 |
| 12 | 4 / 9; 1.236 / 2.380 | 243 / 6; 1.524 / 0.048 | 168 / 10; 1.013 / 0.058 | 370 / 11; 0.166 / 0.069 |

Hubs (top-3 out-degree; top-3 in-degree) after each sleep, and degree-derived churn:

| cycle | top out-degree | top in-degree | churn after the clearance, to wake step 26 (DiffPC layer / manifold / DAS target > 10%) | wake-only churn, steps 26 -> 51 (same) | shield: crossed / cleared / pending |
|---:|---|---|---|---|---|
| 1 | 2163, 1671, 1289 | 1289, 622, 552 | 5 / 1 / 6 | 16 / 0 / 16 | 0 / 0 / 0 |
| 2 | 2163, 1781, 1289 | 1289, 622, 552 | 9 / 0 / 6 | 6 / 0 / 7 | 0 / 0 / 0 |
| 3 | 318, 248, 145 | 134, 100, 100 | 3 / 49 / 1925 | 5 / 0 / 4 | 0 / 0 / 7 |
| 4 | 274, 242, 143 | 134, 100, 100 | 4 / 3 / 55 | 6 / 0 / 8 | 7 / 0 / 12 |
| 5 | 263, 238, 135 | 134, 100, 100 | 26 / 18 / 116 | 3 / 0 / 8 | 18 / 1 / 19 |
| 6 | 337, 235, 135 | 100, 100, 100 | 3 / 8 / 70 | 1 / 0 / 5 | 21 / 17 / 1 |
| 7 | 405, 288, 233 | 100, 100, 100 | 2 / 4 / 75 | 2 / 1 / 5 | 21 / 18 / 48 |
| 8 | 337, 275, 216 | 100, 100, 100 | 5 / 26 / 71 | 4 / 1 / 9 | 42 / 45 / 4 |
| 9 | 273, 213, 206 | 100, 100, 100 | 6 / 14 / 58 | 0 / 0 / 6 | 44 / 47 / 0 |
| 10 | 468, 326, 210 | 100, 100, 100 | 5 / 12 / 39 | 0 / 0 / 5 | 44 / 47 / 82 |
| 11 | 451, 290, 202 | 100, 100, 100 | 4 / 7 / 54 | 3 / 0 / 3 | 46 / 127 / 0 |
| 12 | 370, 243, 168 | 100, 100, 99 | - | 1 / 0 / 5 | 46 / 127 / 85 |

### 4.1 What the tables say

- **Synapse count**: 93,085 -> 95,639 (sleep 1 only tags) -> 98,180 (sleep 2: G = 2, tags again) -> **54,988** (sleep 3,
  the backlog: 45,692 cleared) -> 56,099 / 55,506 / 56,452 / 57,051 / 54,436 / 54,752 / 55,660 / 55,903 / **55,897**.
  Each wake sprouts 2,500 (the 10-per-step cap × 250 steps); each sleep from 4 on clears 1,654-5,167. **Steady state
  ~54-57K, no trend over sleeps 3-12** — the bar's "steady state instead of growing" holds.
- **Lifelines**: 0 lost over 12 sleeps. The protected hubs shrink as ruling (a) foresaw: `cc:want::4625…` 2,163 -> 75-326
  out-links, the Choice Clause node 1,671 -> 263-468; their lifelines stay and mostly strengthen (Choice Clause out
  lifeline 0.09 -> 0.12-0.39).
- **Faint links re-lifted by the wake**: 382-9,913 per wake, overwhelmingly **homeostatic** up-scaling (e.g. 9,831 of
  9,913 in wake 2; reward commits 19-596). The wake also pushes 1,793-7,766 links under 0.01 each time. Both are §1A
  (e) measured: the two-sided Turrigiano counterweight.
- **D14 shield** (kappa 0.015): held 0-85 per sleep after the first clearance (7 at the first clearance under G = 2);
  over the run 46 of them later crossed back over 0.01, 127 were cleared, 85 still pending at the end.
- **DiffPC / DAS / manifold churn**: the first clearance changed 1,925 nodes' DAS firing target by > 10%, 49 manifold
  types and 3 DiffPC layers by wake step 26 (the spec's backlog estimate was 483 targets / 371 layers); later sleeps
  39-116 targets, 3-26 manifolds, 2-26 layers; a wake-only window of the same length changes 3-16 targets. Self-correcting
  at the 25-step refresh, as §1A says — reported so it is not read as a regression.
- **Orphans collected**: 5-11 per sleep (nodes held by hyperedges almost never orphan).
- **Wake step time**: median 0.30-1.78 s, p90 0.71-6.2 s on a loaded machine (load 6-12); it follows the firing bursts.
- **Weight histogram**: the mass below 0.001 (30K at the start) is gone after the first clearance; the 0.01-0.05 band
  (33-38K) is the working population; w >= 0.5 grows 451 -> 819.

### 4.2 Established links

| measure | start | after sleep 12 | change |
|---|---:|---:|---:|
| links with peak >= 0.5 (the bar as written) | 2,338 | 1,632 | **-30.2%** |
| … of which below 0.01 already at the start (the backlog; cleared by any rule, incl. the old weight rule) | 504 | — | — |
| peak >= 0.5 AND w >= 0.01 at the start (non-backlog) | 1,834 | 1,606 | **-12.4%** |
| w >= 0.5 at the start (strong now) | 451 | 419 | **-7.1%** |
| count of links with w >= 0.5 (any) | 451 | 819 | +82% |
| median weight of surviving peak >= 0.5 links, relative to the start | 1.00 | 0.73-1.35 by cycle (no trend) | — |

### 4.3 Twelve cycles: G = 2 (chosen) vs G = 1 vs today's per-step rules

Same copy, seeds and wakes (3,000 wake steps each). `F-G1` = the chosen parameters with G = 1; `baseline-sq-12` = today's
CC config (per-step removal, weight grace 5,000 steps + age clause, inactivity off), no sleep at all.

| after cycle 12 | `F-chosen` (G 2) | `F-G1` (G 1) | `baseline-sq-12` (today) |
|---|---:|---:|---:|
| synapses | 55,897 | 51,773 | 71,930 |
| synapses over cycles 3-12 (min - max) | 54,436 - 57,051 | 51,276 - 53,615 | 70,992 - 75,158 (still drifting down) |
| peak >= 0.5 (2,338 at start) | 1,632 (-30.2%) | 1,571 (-32.8%) | 2,140 (-8.5%) |
| non-backlog peak >= 0.5 (1,834) | 1,606 (-12.4%) | 1,551 (-15.4%) | 1,775 (-3.2%; flat after cycle 3) |
| w >= 0.5 at the start (451) | 419 (-7.1%) | 415 (-8.0%) | 442 (-2.0%) |
| lifelines lost | 0 | 0 | 0 |
| lock hold, first clearance / steady state (s) | 8.6 / 0.43-2.21 | 4.3 / 0.42-2.31 | — (removal is inside every step) |

**What this shows.** G = 2 keeps more established links than G = 1 at every cycle (non-backlog loss 12.4% vs 15.4%),
which is why the sweep's G choice holds over the long run. But **both sleep runs lose established links faster than
today's rules over this window**. The mechanism: wake competition (strength budget + homeostatic down-scaling; 1.8-7.8K
links fall under 0.01 per wake, §4.1) pushes some once-strong links under 0.01; the sleep clears a link that stays under
0.01 for G sleeps, while today's weight rule needs 5,000 steps below 0.01 — longer than this whole 3,000-step run, so
today's rules have not yet reached those links (today's loss is mostly the at-start backlog and the age clause). In other
words the sleep is removing what the weight rule would remove later; the bar's "< 10%" is not met because G sleeps is a
much shorter "below threshold" window than 5,000 steps at 250 steps per sleep. The downscale itself is not the driver:
at sleep 4 the no-downscale control lost 4.8% (non-backlog) vs 4.6-6.5% with downscale (§3). Whether that is the
intended behaviour (D2's "same duplicative pattern" ruling suggests the deadline length was never a design value) or the
bar should change, or once-strong links need a longer G, is Josh's call (§8 item 1).

## 5. Pass / fail against the P2 bar (spec §8 row P2)

| bar item | result |
|---|---|
| Keys absent: byte-identical golden run vs `149fa1f` | **PASS** (§2: trace + checkpoint sha equal; P1-sleep removal sets equal; 48 whole-run tests) |
| No protected node loses a lifeline | **PASS** — 0 lost in every sleep of every disuse run (11 dry runs + the lock-time run: 67 sleeps) |
| Synapse count reaches a steady state instead of growing | **PASS** — ~54-57K over sleeps 3-12, no trend; sprouting (2,500/wake) balanced by clearance |
| Established links (peak >= 0.5) lose < 10% over the run | **FAIL** — -30.2% as written; -12.4% excluding the at-start backlog; strong-now links -7.1% (today's rules over the same window: -8.5% / -3.2% / -2.0%; §4.2-4.3) |
| d0, h, G, kappa chosen from a sweep with wake on (shown) | **PASS with a caveat** — 10 runs shown (§3); d0 / h differences are within wake noise, so the choice is the centre of a flat region; G and kappa were resolvable and differ from D3 / D14 |
| Per-cycle report (synapses, histogram, links >= 0.5, hubs, protected lifelines + out-degree, orphans, step time, re-lifts split, shield fates, churn) | **DONE** (§4) |
| Timescales in sleeps and days | **DONE** (§3) |
| Unit + whole-run tests; NG suite and daemon suite vs base, new failures only | **PASS** (§7) |
| Lock hold: steady state and first big clearance | **MEASURED** (§6) — steady state 0.3-2.2 s; the first clearance 2-16 s: not acceptable to arm as is |

## 6. `_step_lock` hold (wall clock; `sleep_cycle` holds the lock for its whole body)

| sleep | where | synapses cleared | hold s | parts (migrate / downscale / clearance / orphans) | load |
|---|---|---:|---:|---|---|
| migration + tag (sleep 1) | dry runs (10) | 0 | 0.9-3.4 | migration 0.44-1.38 s (Python pass over 93K synapses) | 6-17 |
| migration + tag (sleep 1) | `locktime.py` | 0 | 8.6 | 5.65 / 0.45 / 2.45 / 0.02 | ~9 |
| **first big clearance** | `locktime.py` (no wake between sleeps 1 and 2) | 46,917 | **11.8** | 0.00 / 0.03 / **11.71** / 0.02 | ~9 |
| first big clearance | dry runs (9 native, 1 fallback) | 24,766-46,028 | **2.1-16.0** | clearance part 2.0-15.9 s | 6-17 |
| first big clearance | `F-chosen` (sleep 3) | 45,692 | 8.6 | 0.00 / 0.03 / 8.57 / 0.03 | 6-9 |
| steady state | `F-chosen` sleeps 4-12 | 1,654-5,167 | **0.43-2.21** | clearance 0.38-1.95 s | 6-9 |
| steady state | `locktime.py` sleeps 3-5 | 864-1,157 | 0.33-0.55 | clearance 0.30-0.46 s | ~9 |

The clearance part is dominated by removal (`_remove_synapse_internal` per synapse, incl. D15's adjacency scan) and the
per-sleep metadata read of every synapse for last-link stamps; the native rule sweep and the downscale are milliseconds.
A turn that collides with a 2-16 s hold waits that long (or takes its "recall unavailable" path). **The first clearance
needs the spec's mitigation before arming**: chunked removal with the lock released between chunks (the sweep's row
order is stable) or D7's announced window. Not built here (P3 scope; flagged).

## 7. Suites (keys absent)

- **`tests/test_sleep_p2.py`: 127 passed** (whole file, isolated; 490-736 s). Includes the 48 keys-absent whole runs,
  30 native-vs-fallback bitwise cases, validation (15), migration, G in sleeps (G 0-3), activity / age clauses off,
  downscale formula / salience / protected floor, the D14 shield (kappa 0.15 / 0.015 / 0, negative traces, re-test),
  lifelines, last-link grace in sleeps (and grace 0), record / event / log line, D15 consistency, and 8 + 2 whole runs
  with disuse on (determinism, every protected node keeps its lifelines through every sleep, no dangling pred_weights).
- **NG suite** (157 files, one isolated pytest process per file, 600 s cap; every file that did not pass on the branch
  re-run on the base worktree `149fa1f`; `~/.cache/sleep-p2/suite/`): branch 119 files pass outright, 38 do not. **0 new
  failures**: 37 fail or time out identically on the base (same FAILED / ERROR ids; the P1 report's list, e.g.
  `test_cc_*`, `test_ng_tract_bridge`, `test_migration`; timeouts on both: `test_auto_knowledge`, `test_ces`,
  `test_coordinator`, `test_et_modules`, `test_openclaw_hook`, `test_snn`, `test_tonic_prefetch`). One timed out on the
  branch only (`test_cc_embed_outside_lock_922`, 601 s; the base passed in under 600 s) and passes on the branch alone:
  **10 passed** (412 s — load). Files that time out on both and matter here, run alone on the branch with a 3,000 s cap:
  **`test_sleep_p1.py` 184 passed** (D15 + P1 whole runs vs `de8b214` still green), **`test_rust_hotpaths_onto_s4.py`
  313 passed / 1 skipped**.
- **Daemon suite** (docs `scripts/tests`, one process per side, venv `~/.cache/sleep-p2/venv`): base `2d9f79d5`
  **75 failed, 1,022 passed, 1 skipped**; branch **75 failed, 1,035 passed, 2 skipped**. The FAILED / ERROR id sets are
  **identical** (one log-record id's line number shifts with the added lines). +13 passed / +1 skipped = the new file:
  its real-engine test skips inside the full suite because an earlier test drops `ng_tract` from `sys.modules` (PyO3
  allows one init per process); run alone the file is **14 passed**.

## 8. Risks and what is not proven

1. **The established-link bar fails** (§4.2-4.3). Once-strong links that wake competition (strength budget,
   homeostatic down-scaling) pushes under 0.01 are cleared after G sleeps; today's weight rule would clear the same links
   only after 5,000 steps under 0.01, which this 3,000-step run never reaches. The spec rejects a peak-based shield
   ("would rebuild immortality", §1A). Options for Josh, none taken here: accept it (the bar then reads "strong now":
   -7.1%), a longer G (the G = 1 -> 2 step cut the loss from 15.4% to 12.4%; G was swept only at 1 and 2), or treating
   once-strong links differently. Not a lane decision.
2. **Lock hold of the first clearance: 2-16 s** (§6). Needs chunking or D7's announced window before P3 arms anything.
3. **G = 2 and kappa = 0.015 differ from D3 / D14's recommendations** (1 / 0.15). Chosen by the sweep (§3); Josh should
   confirm or overrule.
4. **d0 / h are weakly determined**: 4-cycle runs cannot separate them from wake noise. The chosen centre keeps the
   designed timescales; a longer comparison (or live observe mode, P3) is the place to refine.
5. **Not simulated**: the Tonic (write-mode STDP adds to eligibility traces that only step() decays — the D14 shield's
   wall-clock behaviour with Tonic traces is untested), the real dual-pass deposit pipeline (the wake uses kNN of real
   deposit embeddings + stimulation, not `run_conversational_dual_pass`), Leg 2 / callosum-merged arrivals (spec risk row:
   "P2's dry run should include a callosum-merged sample" — not done), HE consolidation in the sleep (CC_NG_DREAM off),
   Lenia's distance cache (P3).
6. **Migration** is one Python pass over every synapse under the lock (0.44-5.65 s measured). Once per enable.
7. **`compete_protected_links`** (want-hub (d), not armed) still ranks by the STEP stamp `last_link_since` and advances
   `low_weight_steps` for its competing ids; if it is ever armed in the same sleep as disuse it would count those ids
   twice per sleep and not see sleep stamps. Flag for whoever arms (d).
8. **Sleep stamps vs step stamps**: after a rollback to per-step removal the sleep stamps (`last_link_since_sleep`)
   linger in metadata (harmless; the next migration drops them) and the step path starts last-link episodes fresh.
9. **Weight-only passes are not in `_dirty_synapses`** (pre-existing for `sleep_downscale` / the budget): an INCREMENTAL
   checkpoint would miss the downscale. Nothing in NG or the daemon uses INCREMENTAL today (grep).
10. P1's open items stand: no save right after a sleep; `last_sleep_wall` persisted before the next save; the stale
    `.bashrc:242` comment; the one-time `purge_dangling_pred_weights()` has not been run.
11. Every timing here is from a shared, loaded laptop (load 6-17: live daemon + the guardian lane + this lane); absolute
    times are noisy, ratios less so.

## 9. Artifacts

`~/.cache/sleep-p2/`: `ckpt/` (the copy + `cues.npz`), `run-isolated.sh`, `run-dry.sh`, `queue.sh`, `golden.sh`,
`suite.sh`, `suite-base.sh`, `dsuite.sh`, `build-wheel*.sh`; `scripts/` (`drywake.py`, `prep_vecs.py`, `locktime.py`,
`ckpt_equiv_p2.py`, `static_est.py`, `timescale.py`, `analyze.py`, `report_tables.py`, `summ.py`, `compare_suites.py`);
`out/` (one JSON per dry run, `locktime-1.jsonl`, `static_est.json`); `golden/`; `suite/` (branch, base, rerun);
`dsuite/`; `logs/`; `wheel/`; `venv/` (throwaway: new wheel + `.pth` to the P2b / NG venv packages, read-only).
Worktrees: `/home/josh/worktrees/ng-sleep-p2-20261007` (branch), `ng-sleep-p2-base-149fa1f` (base),
`/home/josh/worktrees/ng-tract-rs-sleep-p2-20261007`, `/home/josh/docs/.claude/worktrees/sleep-p2-daemon-20261007`
(branch), `sleep-p2-daemon-base-2d9f79d5` (base).
