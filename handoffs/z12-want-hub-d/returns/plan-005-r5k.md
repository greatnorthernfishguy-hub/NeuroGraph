# plan-005 [R5k] return — the #867 plan-sync: the engine's 15 held-back last-link partners (Exec P459, build-005 verdict A)

Lane `want-hub-competition-d` · dispatch #12532 · author thread `097da447` · PLAN ONLY: `neuro_foundation.py` untouched, no data load, no probe rerun. Plan head at dispatch `24bcd335a9c1e265f4c9d562c5b9c68a7bd92711` (branch `cc-laptop-want-hub-d-20260930`).

**Commits:** the `[R5k·#867]` commit (plan-005.md + this file) and ONE pin-line follow-up; a commit cannot contain its own hash, so both full hashes are in plan-005's changelog pin line and in my report. Cite the hash, never the file name.

**A repo note, stated as fact:** at the start of this turn `git pull --rebase` (the branch tracked `origin/main`) rebased the local branch onto a moved `main` (20 new commits). I checked the local plan file was identical to the pushed one, then restored the local branch to the pushed head `24bcd335` (`git reset --hard origin/cc-laptop-want-hub-d-20260930`) and pointed its upstream at its own remote branch. Nothing was pushed from the rebased state; the remote branch was never rewritten.

## Sites touched (line numbers before → after; `git diff --stat`: plan-005.md + this return)
| before → after | site | first words of the old line |
|---|---|---|
| 363 → 380 | Lead line | Lane `want-hub-competition-d` · Zone manager Z12 (`52d39aba-db92-4bf2- |
| 393 → 410 | Legend | **Legend.** `[R5·X7]` = Exec Packet 409; `[R5·C15-1…3]` = checker-015' |
| 402 → 419 | (see text) | - **X5** `wires_own_deposits=false` + the ≤ 16 partner nodes that coul |
| 414 → 431 | §0 item 6 | 6. **The staged schedule, K = 50 + 50 (RULED), B = 5,000 (RULED):** co |
| 621 → 638 | §4A.3 | **Numbers re-run [R4·X]:** 16 partners at K=50 have all links competin |
| 667 → 684 | §4A.7 inputs | Inputs (laptop probe, `timestep 33,637`; graph 138,753 synapses / 7,25 |
| 703 → 720 | §4A.7 sensitivity note | **[R3·X] Correction of plan-002.** plan-002 §4A.7 and §0 gave the K=50 |
| 764 → 781 | §5.1 K=50 row | \| **50** \| 17,391 \| **106,841** \| 51,856 \| 94,646 – 106,841 \| **94,646 |
| 769 → 786 | §5.1 note | **[R4·X] The K=50 row is BEFORE the caller-side last-link exclusion (§ |
| 776 → 793 | §5.3 | Model (upper bound, every competitor eligible, §4A.7): after the full  |
| 838 → 855 | R5 | - **R5 — partner orphaning:** zero by construction via the last-link r |
| 903 → 920 | §9 sequencing | **Sequencing and rollout order [R4·C8][R4·P399 cond. 3][R5·L1][R5c·LE1 |
| 1118 → 1145 | §12 pointer row | \| `returns/plan-005.md` \| *(edited in place — **cite a COMMIT HASH, ne |
| 1126 → 1153 | (see text) | **Rev-4 additions to the method [R4·X]:** (17) caller-side last-link:  |
Added lines (new text, not edits): the changelog entry `[R5k·#867]` (top), the §9 "#867 RESOLVED" sentence (inside the sequencing line above), the §10 ruling row, the new §11.10 table.

Corrected: held-back partners/links **16 → 15** (§0 item 6 and X5, §4A.3, §4A.7 inputs and sensitivity note, §5.3, R5, appendix (17)); K=50 **|G| 17,391 → 17,392** and **competing0 106,841 → 106,840** (§5.1 row and note, §4A.3, §0 item 6); **106,825 kept everywhere**. The provenance sentence is at §4A.3 and cites build-005 (commit `caa39edf8e0bbaf8b338b5a28c61250d5b9b4037`) and the id list sha256 `2a90ef643bd354006f047841748edada44c5a552e943c0520c1cc5544062960d` (ids not pasted).

## Ambiguous / not touched (listed, as instructed)
- **Eligible bracket cells quoting 106,841 as the upper end:** the §4A.7 sensitivity table K=50 row (94,646 – 106,841), the §5.1 K=50 row's bracket/prune-eligible cells, and §5.4 "94,646–106,841 (94,630–106,825 …)". These are probe-model bracket figures (lower end depends on E, not recomputed; the upper end equals the probe's competing0). Left as they stand; the clause "probe-model figure, not recomputed against the real G; the post-merge dry run (§4A.5/N5) supplies the exact values" is attached at the nearby touched sites (§0 item 6, §4A.3, §4A.7, the §5.1 note). The consequence — the §5.1 row now shows competing 106,840 beside a bracket upper end of 106,841 — is the clause's point and is disclosed in the §5.1 note.
- **"9 orphan-collectable" and the K=100 / K=200 partner counts (12 (5), 8 (3)):** not recomputed by build-005; left, with the §4A.3 / §5.3 wording saying so.
- **§5.3 "16 → 15":** I treated it as the same quantity (partners that would lose all synapses = the held-back partners). Flagged here in case the Chief reads it as a distinct figure.
- **Changelog history (earlier entries) and plan-003/-004 evidence tables:** untouched (history).

## Byte-for-byte proof (string compare + sha256; and a diff-by-lines check)
For each protected string the SET OF LINES containing it is identical before and after (compared as sorted line lists), and **no changed line contains any protected string**.
| string | chars | count before | count after | lines containing it | sha256 |
|---|---|---|---|---|---|
| census sentence (1) (P419) | 231 | 1 | 1 | identical | `3dc376905dc8910794c8ce0d918f8a42615d401200655affa5707c165d8e1f93` |
| census sentence (2) (P418) | 620 | 1 | 1 | identical | `efd2fe581e6fdeb930b52c7a7bcd64557696ea716ab3cb8b0a6556902bdd4dcd` |
| P420 D2 session sentence | 113 | 3 | 3 | identical | `ac5846cefaacda2bf1220b3d1f042ee64d9b8cd2426e1d1cdda269b0d2b38ec7` |
| P424 session-facing decline sentence | 154 | 3 | 3 | identical | `86c1ba6527baf43686c931c6c29313e31988f19e376ba89f6af20c06bdaa06f5` |
| P436 line | 52 | 16 | 16 | identical | `e8fc06076aa60cdbca7f246277e0efacde6bb7776df98799f14bb19aeb3f5425` |
| Josh's clause | 65 | 6 | 6 | identical | `497471702795ec72b1d76b29659f936a45034a6fce90f243801e2bed7ad7da9f` |
| Executive's one-line mapping | 67 | 3 | 3 | identical | `d974ae4b59a2b3e7edbe592b0ce6bb23c6068b61e1b0e395951c8530cfb2c852` |

The P436 line sits once in each frame (ARMING and CHECK-IN), as delivered. The consent frames, every arming precondition, the go-record pointer and every other section are untouched.

## What I did NOT verify
- build-005's numbers (|G| 17,392, competing0 106,840, 15 held, 106,825) — taken from the file and its id list as cited; I loaded nothing, re-derived nothing, and did not open the id list.
- The eligible bracket's lower end, the "9 orphan-collectable" sub-count and the other K rows — not recomputed (stated in the plan).
- Exec P459 / the #867 determination as primaries (quoted as supplied in the dispatch and the brief).
- The engine branch and `neuro_foundation.py` — not read.
