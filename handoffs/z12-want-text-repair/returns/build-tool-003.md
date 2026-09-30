```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11525, TURN A2b) - build-tool-003: the SECOND hardening
             follow-up (TOOL-ONLY): the three Executive rulings of Packet 432 (the Choice Clause marker set, struck, the peer-hold
             H2 rule) and le-031's R-1..R-8. RETURNED, not self-accepted.
-------------------
```

# build-tool-003 - TURN A2b, the second hardening follow-up - RETURNED

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11525 - tool branch `cc-laptop-want-repair-tool-20260930` - base = the A2 return `c099089503b275b39e9d2c93d7b4a08d3ae01194` (le-031 on top).
Related: [[NeuroGraph]] - [[The Laws]] - [[The Choice Clause]] - [[Duck Ethics]]

**Status.** TURN A2b only, tool-only. **Nothing applied, merged, deployed or restarted.** Synthetic data only; no checkpoint directory was listed or opened, no real checkpoint read, the live tract and Syl's directories untouched, `~/.bashrc` untouched. No protected or vendored file in any commit (my diff = the tool file and the test file, plus this return). No raw want text anywhere. Every hard-link or file-identity statement is a measured `(st_dev, st_ino, st_nlink)` on synthetic files - never inferred from a link count.

## 1. Commits (all pushed to `origin/cc-laptop-want-repair-tool-20260930`, in order; le-031's three review commits sit before them)

| Commit | What |
|---|---|
| `813a1b0548ec2aef63ad369299703caeb07c3c01` | tests(A2b): the failing-first tests for all items - committed BEFORE any tool change |
| `2dcd03577d33c68363773ba3e89f75f2e6854e10` | tests: the R-4 real-`Probes` case stubs `subprocess.run` (so it never queries the real host) |
| `d391bb1c5c1563d06e863ae4c764fd0a4ec230a2` | the tool change (P432 marker set / struck / H2, R-1..R-8) |
| `9721e6fc95a54855c4615bd83ccb5486824360bd` | tests: the R-4 stubs were unscoped and leaked into the CLI's own `git` query (my test bug) - now scoped |
| `a300bf1be437c6ce996b6185801bb20751110d2b` | tool: `linked` was still in the "enabled" word family, so H2 read `linked` as enabled (a real bug the new test found) |
| this file's commit | the branch tip after it |

`git diff --stat c099089..a300bf1` on my paths: `oneshot-tool/want_text_repair_oneshot.py | 199 ++++--` and `tests/test_want_text_repair_oneshot.py | 521 ++++---` (627 insertions, 93 deletions; the two `reviews/` files are le-031's). File sha256 at `a300bf1`: tool `955eee35bb32da2b32a27db5fdb271b320a3c55374adf296c281e99d07fa7893`, tests `c8271431c59701c987a9a3781cc7159ac9a0efadb4ca57af9c8ffe52076a31d2`. The tool stays under `handoffs/z12-want-text-repair/oneshot-tool/`, tests in their own commits.

## 2. Pin worktree, recomputed

| Value | Recomputed now | Equals the pin |
|---|---|---|
| PIN `git rev-parse HEAD` | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` | yes (code commit) |
| `cc_ng_organism.py` sha256 | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` | yes |
| blob at HEAD | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` | yes |
| `tests/test_cc_want_legitimacy_810.py` sha256 | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` | yes |
| `git status --porcelain --ignored` | 0 lines (never edited, never committed) | - |

P379 preamble of the final run (both heads printed and the file sha256 asserted, as in A2): `P1 pin/stack head (frozen) c7921b84... ; actual pin-worktree HEAD ae798b94... ; both recorded` / `P1 cc_ng_organism.py sha256 asserted equal to the pin: 8ad0f69e...` / `P379 cc_ng_organism.__file__ .../z12-want-repair-pin-ae798b9/cc_ng_organism.py` / `P379 PYTHONPATH None ; NG_EMBED_* names none`.

## 3. Failing-then-passing evidence, per item

Failing-first = the tests committed and pushed BEFORE the tool change and run against the A2 tool: **35 failed / 153 passed**. The 153 that passed include the tests that pin behaviour A2 already had and must keep (a struck Choice Clause id or rim still STOPs, an unapproved id still STOPs, a live |S| mismatch still STOPs, `enabled`/`active` still fail H2, an unproven-absent unit still fails closed, no marker is ever written) - stated so nobody counts them as new coverage.

| Item | Failed pre-fix | Pre-fix failure (from the log) | What now holds |
|---|---|---|---|
| **P432 (1)** marker set | 10 | `0 == 3` (a `source == cricket_rim`, `creation_mode == constitutional`, `rim_source` node passed classify); `DID NOT RAISE Stop` in the build and V15; the dropped names WERE refused (`Stop ... carry the constitutional or Choice Clause flag`); `TypeError: is_choice_clause_marked() takes 1 positional argument` (no id-based identity) | `is_choice_clause_marked(md, nid)` refuses: the rim id, either literal want id (no metadata marker needed - the id IS the identity), `constitutional` truthy, `source == "cricket_rim"`, `creation_mode == "constitutional"`, a `rim_source` key present. **My reading of "present": the KEY exists, whatever its value (even `None`/empty)** - tested with `"seed_cc_rim.py"` and `""`. The speculative `tag/tags/kind/category/choice_clause` are DROPPED: a node with only those, or an ordinary want, is NOT refused (tested). The check runs over S, the mapping AND every approval id, in classify, the build and V15 (each marker tested in all three). No marker is ever written (the carried nodes keep exactly their input keys - tested). |
| **P432 (2)** struck | 1 (+1 adapted) | `STOP: any deviation from the approved list` on a legitimately struck id | `gate_write_set` gives `struck` its own reason. At `--apply`: a `struck` id is a recorded **deviation** (`struck_deviations` in the FINAL receipt, ids and hashes only), the node stays byte-identical (tested: the raw node entry is equal before and after), `id-map` and its inverse exclude it, and the apply COMPLETES with the other two repaired; the frozen map must equal mapping + struck pairs. Still STOP (all tested): a struck Choice Clause id or the rim (`deny-check`), an unapproved/absent id, an approval whose hashes differ, a live `|S|` mismatch (`P9`). |
| **P432 (3)** H2 with the REAL `Probes`, `subprocess.run` stubbed at the run boundary | 5 | `linked` + `inactive` failed (`['h2_leg2_timer_enabled']`); an unit absent per `list-unit-files` failed (`probe_error`); the old positive characterisation did not know the witness | `is-enabled` **`linked` / `disabled`** pass, `enabled` (and its family, `static`, `linked-runtime`) FAIL, `masked` / unknown words / empty FAIL CLOSED; a unit counts as off when absent ONLY if `list-unit-files --no-legend <unit>` lists nothing on a bus that just answered - a listed unit with a `not-found` answer, an empty listing without the bus witness, an unreachable bus and an unknown answer all FAIL. Tested: linked+inactive passes, disabled+inactive passes, enabled fails, active fails, absent-per-list (rc 0 and rc 1) passes, unreachable bus fails, unknown answer fails. **R-6 / le-031 N-b positive bus witness: added, not just recorded** - every `down` (`is-active`), `linked`/`disabled` and `absent` verdict first requires `systemctl --user is-system-running` to print a known manager state; without it the verdict is a `ProbeError` (tested: `offline` witness + `linked` fails). (I left `is-active` `failed` counting as down, as in A2/C2; the Chief's H2 wording expects `inactive`, which passes.) |
| **R-1** bind the receipt | 2 (+6 rollback flows, feature absent) | `0 == 2` (no receipt flag needed) | the rollback REQUIRES `--josh-go-receipt-sha256` (repeatable); each quoted hash must name a post-apply receipt in the run directory (else refused before any write); an edited receipt no longer matches its quoted hash (tested: refused, live untouched); an unquoted (forged) extra receipt is IGNORED and cannot widen the identity check (tested: live in a third state is refused as `identity`). |
| **R-2** more than one receipt | 1 | feature absent (`SystemExit: 2`; the A2 code refused two drafts) | the identity check trusts the UNION of the quoted receipts' after-hashes. Tested end to end with the tool's own `stage_apply`: attempt 1 dies before any replace, attempt 2 dies between the two replaces -> two drafts, no FINAL, torn state -> the rollback quoting the torn draft COMPLETES (all six files equal the pre-apply bytes), and quoting both drafts also works. |
| **R-3** | 1 | feature absent | `manifest["target_realpath"]` must equal the target directory at `--apply` and rollback (STOP; tested with an edited-and-re-hashed manifest). |
| **R-4** | 1 | `gate_p6` accepted `--conduit-dir` = `<ckpt>/generations` and would have walked it | `conduit_path_refused`: a conduit path that IS or CONTAINS the recorded checkpoint directory (realpaths, symlinks included) is refused in `gate_p6` and never passed to `stat_snapshot` (tested: a recording probe and a wrapped `os.walk` see no call under `generations`; the CLI refuses at `phase2-backup`). |
| **R-5** | 2 | `3 == 2` (the writer wrote through a stale hard-linked tmp) | both writers call `refuse_inplace_write(tmp)` first; tested for `--apply` and rollback with a stale tmp of the exact `atomic_file_write` name hard-linked to a decoy: refused, the decoy's content untouched, live untouched. |
| **R-7** | 2 | feature absent | before ANY replace the rollback preserves a hash-equal copy of every file it displaces: `<run>/stage/<name>` if that still holds exactly those bytes, else a new copy in `<run>/displaced-<UTC>/`; recorded in the rollback receipt. Tested: stage intact -> recorded `stage`, no displaced dir; stage removed -> `displaced` copies whose sha256 equals the post-apply bytes; a WRONG stage copy (one byte appended) -> falls back to `displaced`. |
| **R-8** | 3 | a future `--code-placed-at` was accepted; no `foreign_unreadable` key; P3(b) had no timeout (`TimeoutExpired` escaped) | a `--code-placed-at` later than now is refused at backup, apply and rollback; `foreign_unreadable` is in the `gate_p4` and `gate_p6` results (tested with an other-uid stand-in: counted, gate still ok); P3(b) runs with `P3B_TIMEOUT_S = 1800` and a timeout fails the gate (`timed_out: true`). **P3(b) is still never executed by this tool or its tests (it is stubbed in the flow tests): it must be run ONCE, read-only, before Phase 2** - now also in the `--apply` help text. |

## 4. Tests - how many runs, honestly

New file, targeted, `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider`, every run AFTER a push, from the tool worktree root. **Three runs**:

| Run | At | Result |
|---|---|---|
| 1 | `813a1b0` tests only, tool at the A2 state | **35 failed / 153 passed** (failing-first, section 3) |
| 2 | `d391bb1` tool changed | 3 failed / 185 passed: two = ONE tool bug (`linked` still in the enabled word family, checked first), one = my test bug (the R-4 stub leaked into the CLI's `git` query) |
| 3 | `a300bf1` | **188 passed in 71.50 s**, exit 0 |

(A2 had 153; now 188 collected. Existing tests were adapted (I did not count them): past placement times because a future one is now refused, the `struck` reason, the receipt hash in the rollback argv, and the three superseded tests - the A2 marker guess, "struck blocks the apply", and the pre-witness Probes characterisation - replaced by the P432 ones.) No scratch runs this turn; no sanity script.

## 5. Readings and flags (nothing here was decided silently)

1. **"a refusal touching a protected want ... STOPs" (P432 (2)).** Every member of S is a `cc_authored` want and therefore identity-protected, so a literal reading would make `struck` impossible. I read the STOP clause as the **Choice Clause marker set** (the two wants, the rim / constitutional node, anything matching the set): a struck ordinary S want is the legitimate deviation; a struck marker-set id STOPs. Please confirm.
2. **`rim_source` "present"** = key present, any value (above).
3. **The frozen-list review "still FLAGS leaving/exiting/refusal/consent excerpts" (P432 (1)).** The tool does NOT do this and never did: it flags structural things (regions, marker-bearing mentions, mention shapes) and has no semantic/keyword flag, by design (plan 4.2-5 "no proxy decides a node"). That flagging is the Executive's reading of the off-repo excerpts. If a keyword hint in the review file is wanted, it is a small advisory addition; I did not add it because no test/item asked for one.
4. **The `struck` deviation and the frozen id-map.** The frozen `reports/id-map.json` is the CANDIDATE mapping; with a struck id the apply accepts it because the receipt reconciles frozen = mapping + struck. It still STOPs if the frozen map differs any other way.
5. **The witness is on down/off/absent verdicts, not on `active`.** An `active`/`activating` answer already fails the gate, so it needs no witness. The residual N-b case (a translated unreachable-bus message that also prints `inactive` rc 3) is now closed by the witness, not merely recorded.
6. **`static`, `linked-runtime`, `alias`, `indirect`, `generated`, `transient`** count as "could start on its own" (H2 fails); `masked` and any unknown word fail closed. Only `linked` and `disabled` (and a proven-absent unit) pass.
7. **What TURN B needs (unchanged for Phase 1):** Phase 1 on a COPY calls none of this (no probes, no rollback, no P3(b)). For the later Phase 2 the operator additionally supplies `--josh-go-receipt-sha256` to any rollback (the Chief relays it with the manifest hash) and a past `--code-placed-at`.

## 6. What I did NOT verify

The hardened tool has still never touched a real checkpoint or a real host: the real `Probes` against a live systemd bus with the real Leg-2 timer (`linked`/`inactive`, OnCalendar `*:0/15` per the Executive) and against the real `list-unit-files` output format of this host's systemd; the real `is-system-running` words the daemon user's manager prints; P3(b) (never executed); the rollback, receipts and inode assertions on a real hard-linked live file; the real generation directory and its partners (never listed or opened); whether `list-unit-files --no-legend <unit>` prints nothing (rc 0 or 1) on every systemd version for an absent unit (the tests stub the two rc shapes I know); real-file V13, peak memory and the outcome split (TURN B); `atomic_file_write` has no `fsync` (le-029 N5, unchanged).

## 7. Closing check

`git status -sb && git log -1` on the tool worktree: clean, tip pushed. The PIN worktree was not touched. A tiny fresh law-enforcer re-look follows; TURN B (Phase 1 on a COPY, 6 GB cap, both heads recorded, before/after inodes and link counts) is a separate message. I have stopped.
