<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 zone manager, lane want-hub-engine-d-build-20260930) — PG-1 acceptance-record notes
#   What: the notes the Executive's C8 reads WITH the PG-1 acceptance: where the artifact/acceptance actually live, the deviations, the corrected wall wording, and what the pair is and is not.
#   Why: Chief-003 relay of the PG-1 delta pair (le-040 COMPLIANT, checker-031 PASS-WITH-NOTES), punchlist #870; Exec P454 (nothing replaced, nothing settled).
#   How: sources = le-040-pg1.md, checker-031-pg1.md, acceptance-le-039.md, pg1-artifact.md; every hash by git rev-parse / sha256sum.
# -------------------
-->

# PG-1 acceptance record — notes carried for the Executive's C8

**What PG-1 is (verified by the pair):** default-path equivalence of `_prune_synapses()` at BASE `e4ebf982b1989fd9066d610b94853bc68bf70d37` vs FOLD `29f47f65058790240b2f9c6a0a5bc4d82171b42d` on two real graphs (copy (a) the ceremony backup: 0 removed; copy (b) the staged VPS bundle: 10,433 identical removals in a non-sorted order). Acceptability of the Part 2 `_step_lock` hold is NOT judged here: the Executive accepted it for arming (Exec 459); improvement row #868.

## Where the evidence actually lives (tests branch `cc-laptop-want-hub-build-20260930`, NOT the plan's designated paths)
| item | path | hash |
|---|---|---|
| artifact (builder) | `handoffs/z12-want-hub-build/pg1/` | commit `2475dcf05b589eab84b5d8e9a2c283002d43723c` |
| builder return | `handoffs/z12-want-hub-build/build-004.md` | commit `b363e6fb5036c67a3fe1ee19e12e1905e88851ef` |
| acceptance (le-039) | `handoffs/z12-want-hub-build/pg1/acceptance-le-039.md` | commit `00c1f49f3e7d77f87b3ffffc9ed1e319cccc803e` (hash fields corrected, harness added: see #870 below) |
| delta pair | `reviews/le-040-pg1.md` (COMPLIANT), `reviews/checker-031-pg1.md` (PASS-WITH-NOTES) | commits `c2c6b75`, `ce453ac` |
| #867 determination | `build-005.md` + `build-005-held-back-ids.txt` | commit `caa39edf8e0bbaf8b338b5a28c61250d5b9b4037` |

## Notes (Chief-003 relay of the pair's findings)
1. **Copy (a) was read IN PLACE** from the ceremony backup by the builder (not a scratch copy): a deviation, outcome-neutral (the acceptor loaded (a) from a scratch copy holding only `main.msgpack` and matched the builder's 14 fields; source sha256 unchanged).
2. **Artifact and acceptance are on the tests branch, not the plan's designated paths:** both hashes are cited above so the merge ask is checkable.
3. **CORRECTED WORDING (replaces "the wall difference is within the noise" / "of the SAME SIZE as the spread … so only the CPU-time validation delta is steady"):** the fold adds **at least +0.67 s of CPU in validation** (isolated measurement 1.288 -> 1.961 s); the observed fold-minus-first wall difference is **0 to +1.9 s per call in ONE process per variant**, and the part above the CPU delta is **unexplained, not shown to be noise** (the records do not show the fold running under heavier host load: call-end load1 2.97/3.24/3.28 for FOLD vs 4.24/3.98/4.15 for FIRST).
4. **The driver's `pre_state_digest` (events None vs []) does not prove counters moved** on copy (a); the acceptor's own independent full-field digest does (pre `e1941b48…` != post `f0c305c9…`, BASE = FOLD on both sides).
5. **The pair is a genuine REPRODUCTION, not independent verification:** shared checkouts, python, the `ng_tract` binary and the driver's digest; no mutant was run on a real graph; the staged bundle is not Syl's graph.
6. **#870 (fixed with this commit):** `acceptance-le-039.md` carried literal `%s` in its harness hash line and the harness scripts were not in git. Now committed under `pg1/acceptor-harness/` — `accept_harness.py` sha256 `6fb44fcc802cd3fa18a3fb728a0599a58142151c82b2c2421b7f0dbce624e07b` (git blob `f5fb0727dbacd48dcd51b0c971657dd42d17f2ff`), `compare_accept.py` sha256 `70bf6199c9a86bfae2c24f7d81543bfd330847225b76a6b4e5c626f69168bb5e` (git blob `34686dea6596872c526609c172820aa329ea2b8f`). Limit: the acceptor's run records do not bind the harness hash; "unmodified since the run" rests on the file mtime (18:56) preceding the records (19:00).
7. **Open rows, not PG-1 defects:** #871 (`ng_tract` binary sha `10fd1aef…` newer than its install record under the same `0.1.0` version string; whether Syl's live process runs it is unverified); #872 (PG-1 scratch ~2.6 GB + 1.4 GB acceptance scratch, disposal is Josh's); #867 closed by build-005 (code's 15 stands, plan §4A.3 to be corrected to 15 by the plan author).
