<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane poison-quarantine-920, dispatch #13417+) — build-002 addendum 001: the CARRIED mutants
#   What: results of the 16 mutants that did not run in build-002 (NG M8-M14, daemon D1-D9) against the build-002 tests: 15 ran, ALL 15 KILLED; 1 (D9) was
#     REFUSED by the load gate and stays CARRIED.
#   Why: Z12's carried-mutants nudge after build-002 was accepted. No code or test file was edited and no NG/daemon commit was made (tips unmoved).
#   How: scratch mutant copies under /tmp/z12-920-scratch/mut/, the gated harness build-002-mutants.py (identical to the committed one, checked with cmp), run ONE
#     mutant per invocation by the spaced driver build-002-addendum-001-spaced.py (20 s pause between mutants; the pause is not a retry). Raw output:
#     build-002-addendum-001.mutants.out.txt.
# -------------------
-->

# #920 build-002 addendum 001: the carried mutants

Related: [[NeuroGraph]] · [[Dual-Pass Embedding]] · [[The Laws]]

## READ FIRST

- **15 of 16 ran; all 15 KILLED.** **D9 did NOT run: the load gate refused at START LOAD 6.43 / 4.78 (1-minute / 5-minute). D9 stays CARRIED.** I did not retry it.
- Tips unchanged and trees clean, verified before and after: NG code `44e034fb4d1491696a5cdcc064133b3c24a9a70b` (NG HEAD is the return commit `f0a4e439` until this docs commit), daemon `2e882d2f948b549a9223ba14e5e36f26e5cfcf56`. No file edited, no code commit, nothing pushed or merged.
- `quarantined_total` (your answer received): it exists from build-001 (accepted); I leave it, and #928 is not built.

## Gate and spacing

Gate on every mutant (the harness's own, recorded per run): refuse at 1-minute >= 6 OR 5-minute >= 5.0. One mutant per harness invocation, 20 s pause between mutants. The box was 3.11 / 2.81 at the start. The 1-minute load still climbed to 5.86 by D7 and the gate refused at D9 (6.43 / 4.78), so spacing helped but did not prevent the climb; the load afterwards read 6.59 / 4.87 / 3.63, so I cannot say how much of it was my own runs and how much other sessions.

## Results (START LOAD = 1-minute / 5-minute)

| Mutant (seeded fault) | Result | Failing tests | START LOAD |
|---|---|---|---|
| M8 no idempotency (always append) | **KILLED** | 5 | 2.94 / 2.79 |
| M9 `already_present` skips the fsync (the build-001 hole) | **KILLED** | 2 | 3.29 / 2.88 |
| M10 retry key from the text, not the bytes | **KILLED** | 12 | 4.21 / 3.13 |
| M11 count not cleared on success | **KILLED** | 5 | 4.50 / 3.31 |
| M12 quarantined but tract not advanced | **KILLED** | 19 | 4.69 / 3.44 |
| M13 default #794 WARNING text changed | **KILLED** | 2 | 5.03 / 3.63 |
| M14 counter also incremented on `already_present` | **KILLED** | 2 | 3.89 / 3.48 |
| D1 `max_retries` kwarg passed with the hold flag off | **KILLED** | 5 | 3.53 / 3.42 |
| D2 default 5 even with the hold flag off | **KILLED** | 11 | 4.39 / 3.63 |
| D3 bound read per call from the environment | **KILLED** | 1 | 5.25 / 3.89 |
| D4 counter not mirrored into `status` | **KILLED** | 4 | 4.15 / 3.76 |
| D5 garbage warning logged at import | **KILLED** | 6 | 4.14 / 3.78 |
| D6 `main()` never emits the garbage warning | **KILLED** | 2 | 4.50 / 3.88 |
| D7 zero accepted as a bound | **KILLED** | 2 | 5.86 / 4.28 |
| D8 mirror runs inside the graph lock | **KILLED** | 1 | 5.01 / 4.23 |
| **D9 `quarantined_total` seeded even with the flag off** | **NOT RUN (CARRIED)** | n/a | **6.43 / 4.78: REFUSED** |

NG mutants (M8 to M14) ran against `tests/test_cc_poison_quarantine_920.py` (53 tests, with the daemon env set); daemon mutants (D1 to D9) against `scripts/tests/test_cc_ng_daemon_poison_quarantine_920.py` (59 tests). D6 shows 2 failing tests (build-001 showed 1): the build-002 test of the truthful garbage warning also goes through `main()`.

## Combined mutant status for #920 (build-001 + build-002 + this addendum)

Against the build-002 tests: 27 of 28 mutants have now run and all 27 were killed (12 in build-002, 15 here). **The one remaining unrun mutant is D9** (the stats seed with the hold flag off); it was killed in build-001 against the earlier tests and the test that kills it (`test_the_stat_is_seeded_only_when_the_bound_is_on`) is unchanged since, but I have **not** re-run it against build-002, so that is unverified.
