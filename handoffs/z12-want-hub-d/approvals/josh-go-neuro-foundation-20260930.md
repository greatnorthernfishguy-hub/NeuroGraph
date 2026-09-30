<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #11852) — NEW FILE: the record of Josh's "proceed" for the protected-file work [R5i·P440]
#   What: a HANDOFF-FILE RECORD ONLY. It records, before the first commit that touches `neuro_foundation.py`: Josh's go (his words VERBATIM), the backup confirmation naming BOTH msgpack files with their
#     sha256, the exact scope of the go, and one flag. It changes no code and loads no data.
#   Why: NeuroGraph `CLAUDE.md` §2 "What Explicit Approval Means" (tell Josh; backup of both msgpack files; "proceed"; no batching) and le-026 F7 / le-028 G6 (plan-005 §9) require this record, dated,
#     on the plan branch, BEFORE the first `neuro_foundation.py` commit. The engine builds are dispatched only after this record lands and Z12 has checked it.
#   How: transcribed from `assignments/record-josh-go-engine-825-p440.md` (docs worktree branch `cc-laptop-daemon-recall-756-20260930`, dispatch #11852). The quoted words, sizes and sha256 values are
#     copied from that brief — NOT re-hashed or re-measured by me (the brief says they were supplied in the dispatch). No secret involved.
# -------------------
-->

# Josh's go — the protected-file work on `neuro_foundation.py` (Exec Packet 440)

Lane `want-hub-competition-d` · branch `cc-laptop-want-hub-d-20260930` · cite the COMMIT HASH of the commit that adds this file, never only the file name. **Per plan-005 §9 [R5g·LE28-G6]: the first `neuro_foundation.py` commit message on each engine branch quotes this record's commit hash.** This file is the plan-005 §9 "record of Josh's approval" — the brief names this file instead of the `branch-build-approval.md` that §9 designates; no second file is created.

## 1. Source
**Exec Packet 440 (Josh direct), relayed by Chief-003.** The packet states the date **2026-10-01** for Josh's words — "stated as 2026-10-01 in Exec Packet 440". The real box date when this record was written (`date -u`): **2026-09-30** (see the commit). Nothing is reconciled. I did not open Exec Packet 440 as a primary; its text is quoted as supplied in the dispatch and the brief.

## 2. Josh's words, VERBATIM (two fragments, in this order)
> *"That copy you already approved is fine, I guess? Unless there is some reason it shouldn't be..."*

> *"So, proceed."*

## 3. The backup confirmation — naming BOTH msgpack files
Josh accepted the copy as the backup (his first fragment). Directory: `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc/`

| file | bytes | sha256 |
|---|---|---|
| `main.msgpack` | 230,539,966 | `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` |
| `vectors.msgpack` | 1,000,876,671 | `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e` |

Sidecars present in that directory: `main.msgpack.activations.json`, `main.msgpack.guard_state.json`, `main.msgpack.manifest.json`, `commons.msgpack`.

**As supplied in the dispatch (NOT re-hashed or re-measured by me):** these values equal the LIVE CC-laptop checkpoint's files — sha256 recorded by the TURN B-1 copy manifest at 18:31 UTC and re-hashed by Z12 at the dispatch of this record: same values — and the live files are unchanged since 2026-09-23 (the daemon is down).

## 4. Scope of the go — EXACTLY TWO changes to `neuro_foundation.py`, EACH on its OWN branch, EACH UNBATCHED (`CLAUDE.md:95`)
1. **The (d) engine change per plan-005:** additive keyword-only default-`None` parameters on `_prune_synapses` (`max_removals`, the static order key, the removed-ids out-param, the competing-set assertion) + the orchestrator as plan-005 specifies + the P399 golden equivalence test. Branch **`cc-laptop-want-hub-engine-20260930`**, containing ONLY `neuro_foundation.py`; the tests stay on the tests branch `cc-laptop-want-hub-build-20260930`.
2. **#825:** prove, and if needed fix, that incremental saves persist deletions so pruned links stay pruned across save and restore (the arming gate). Its own branch **`cc-laptop-incsave-825-engine-20260930`**, containing ONLY `neuro_foundation.py`; its proof/tests/return are on a separate plan branch **`cc-laptop-incsave-825-plan-20260930`**.

**NOT covered:** any merge (the rollout call); arming (needs the S4 Tonic check, the L7 consent and #825 proven); #740; #768; any other protected file. **Each change gets its own delta pair and a PG-1 artifact BEFORE ANYTHING ELSE.**

## 5. A FLAG, stated as fact (for the Executive; not resolved here)
The backup Josh accepted is the **CC LAPTOP checkpoint copy**; it is **NOT Syl's own checkpoint** (`~/NeuroGraph/data/checkpoints`). Nothing in these two BRANCH builds loads, reads or writes Syl's files or any live checkpoint (synthetic graphs only), and no merge is covered; but `_prune_synapses` and the save path are a shared hot path executed by Syl's process after a merge (plan-005 R13), so the Executive should say whether Josh's backup confirmation for Syl's own two msgpack files is to be re-stated before the MERGE / rollout.

## 6. Where else recorded
This file is also the record for the `#825` lane: the lane `incremental-save-deletions-825-20260930` and its plan branch cite this file by path and commit; Z12 copies it there.
