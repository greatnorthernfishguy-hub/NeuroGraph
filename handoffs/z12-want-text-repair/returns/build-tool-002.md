```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11395, TURN A2) - build-tool-002: the HARDENING
             follow-up to the 118-want text repair one-shot tool (le-029 C1-C8 + N1, checker-026 c026-C3, the P1 heads).
             RETURNED, not self-accepted.
-------------------
```

# build-tool-002 - TURN A2, the hardening follow-up - RETURNED

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11395 - tool branch `cc-laptop-want-repair-tool-20260930` - base = the TURN A return `ab24e85c06a0b524828db895574e7544ca01ed09` with le-029 and checker-026 on top.
Related: [[NeuroGraph]] - [[The Laws]] - [[The Choice Clause]] - [[Duck Ethics]]

**Status.** TURN A2 only. **Nothing applied, merged, deployed or restarted.** Synthetic data only: no checkpoint directory was listed or opened, no real checkpoint read, the live tract and Syl's directories untouched, `~/.bashrc` untouched. No protected or vendored file is in any commit (my diff is exactly the tool file and the test file plus this return). **No raw want text anywhere.** The rule "check INODES, never infer from link counts" applies to every claim below: every hard-link statement about the synthetic world is a measured `(st_dev, st_ino, st_nlink)` from `os.stat`, and the tool itself decides "same file" by the `(st_dev, st_ino)` pair only.

## 1. Commits (all pushed to `origin/cc-laptop-want-repair-tool-20260930`, in order; the reviewers' commits sit between)

| Commit | What |
|---|---|
| `4d88f14b9112a983ee791e34f36f6fe05e8a6896` | tests(A2): the failing-first tests for every item (committed BEFORE any tool change) |
| `3f7cb85c9d011c5c0071ec33fcb6d950b99cbe74` | tests(A2): my test helper fell back to the old `id-map` name so the first failing-first run's Phase-2 tests died on a `FileNotFoundError` in the helper and MASKED the behavioural failures; fixed and re-run |
| `e94b3ce3c9652ad5ea30f3c27e81046991ce35f0` | the tool change (C1-C8, N1, c026-C3, P1 heads) |
| `e7bd634a70679dffef1eb65bb7d8c4df8490869d` | one tool bug the tests found: the new ripple text was 296 characters and the report text guard (<= 200) refused it, failing every backup; text shortened, guard unchanged |
| this file's commit | the branch tip after it |

`git diff --stat ab24e85c..e7bd634a` on my paths: `oneshot-tool/want_text_repair_oneshot.py | 631 +++++--` and `tests/test_want_text_repair_oneshot.py | 667 +++++--` (the range also holds the two review files, 371 lines, not mine). File sha256 at `e7bd634a`: tool `ac0bf1f5ef43e459d48b745dfe59ee52bf38bab92f315a66afb5cb5b6bf477b7`, tests `6c08f3e771cc0cb6d716aeebe481785a0899bb5895014fe845f4e3916c30d323`. The tool stays under `handoffs/z12-want-text-repair/oneshot-tool/` (never merged; removed by a commit before any merge of the handoffs); the tests are a separate commit from the tool.

## 2. Pin worktree, recomputed

| Value | Recomputed now | Equals the frozen pin |
|---|---|---|
| PIN `git rev-parse HEAD` | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` | code commit - yes |
| `cc_ng_organism.py` sha256 | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` | yes |
| blob at HEAD | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` | yes |
| `tests/test_cc_want_legitimacy_810.py` sha256 | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` | yes |
| `git status --porcelain --ignored` | 0 lines (clean; never edited, never committed) | - |

**P1 records BOTH heads and asserts the file sha256** (Chief's ruling on flag 1). The run's preamble now prints, and the run record already carried, both:
```
P1 pin/stack head (frozen) c7921b8436fb174c3f70fcf02827f16bb16deff0 ; actual pin-worktree HEAD ae798b94cb14740d200fc3f4fd8d36eef8b86c6a ; both recorded
P1 cc_ng_organism.py sha256 asserted equal to the pin: 8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2
P379 cc_ng_organism.__file__ .../z12-want-repair-pin-ae798b9/cc_ng_organism.py ; PYTHONPATH None ; NG_EMBED_* names none
```
The assertion is real: `load_pinned` hashes the file from DISK before importing and re-hashes the LOADED file after (tests: `test_p1_prints_both_heads_and_asserts_the_loaded_file_sha256` monkeypatches the second hash and expects the refusal; `test_p1_the_run_record_carries_both_heads`).

## 3. Failing-then-passing evidence, per item

Failing-first = the new tests committed and pushed BEFORE the tool change and run against the unfixed TURN A tool. Two such runs (run 1: 56 failed / 97 passed, run 2: 50 failed / 103 passed after the helper fix - see 6). The pre-fix failure reasons below are from run 2's log; all 153 pass after the fix (run 4).

| Item | Tests (pre-fix failed / all pass after) | Pre-fix failure, in the words of the log | What now holds |
|---|---|---|---|
| **C1** required `--code-placed-at` + readable `daemon.log`; no leg omitted | 3 + the 2 adapted P4 tests | `gate_p4` `AssertionError` (the leg was absent and `ok` True); `phase2-backup` accepted a run without the flag (`assert 0 == 2`) | `gate_p4(..., *, code_placed_at, daemon_log)` ALWAYS reports `no_pulse_since_code_placement` and `six_files_equal_the_start_of_phase2_backup` (False when its input is missing/unreadable/absent; the rollback step records the second under `skipped_legs` with the reason); `phase2-backup`, `--apply` and rollback REFUSE without a parseable `--code-placed-at`; a missing or unreadable log makes P4 fail (the old tests `:1269/:1274` that encoded the omission as a pass are rewritten) |
| **C2** probes fail closed, incl. the ERRORING REAL-`Probes` case | 21 | `DID NOT RAISE ProbeError` for `unit_active`, `unit_enabled`, `crontab_text` under all seven stubbed host failures; `FileNotFoundError: systemctl` escaped the whole gate; the proc-scan case did not raise | real `Probes._sysctl` / `crontab_text` / `/proc` scans with **`subprocess.run` stubbed at the run boundary (NOT by overriding the query methods)**: bus unreachable, empty stdout rc 0, non-zero rc, unknown word, timeout, missing binary, OSError -> `ProbeError`; `inactive`/`failed` (rc 3/4) is the ONLY "down"; `crontab` rc 0 or rc 1 + `no crontab for` = empty, else error; an unreadable own-uid `/proc` entry raises, a vanished one is skipped; the gates record `probe_errors` and FAIL; end-to-end `phase2-backup` with the REAL `Probes()` and systemctl/crontab failing refuses (`P4/P6`), creates no run directory and leaves the target untouched |
| **C3** P2 identity + refuse pin/tool worktree | 2 | `AttributeError: no gate_p2`; the pin's own file (and a symlink into it) was ACCEPTED at `--apply` | `gate_p2` records realpath / inode / mtime / sha256 and refuses a path whose realpath is inside the pin worktree or the tool worktree (or not a file); the help text says the Chief names the unit's import root (Q9) |
| **C4** `--expect-*` pinned | 1 | `--expect-wants` differing from the constant was accepted at backup and apply | at `phase2-backup`, `--apply` and rollback the three flags must equal the module constants or are refused (Phase 1 still takes other values for a COPY - tested) |
| **C5** dry-run packet structurally not an approval | 2 | `KeyError: provisional` (no flag, decisions `approved`); a packet with ONE string edited and re-hashed was ACCEPTED | dry-run packet = `decision: "provisional"` + top-level `provisional: true`; ONLY `stage_rewrite` accepts it (`provisional_ok`); `load_approvals` refuses it for any edit tried (packet string; flag alone; decisions flipped to `approved` with the flag left) |
| **C6** inode evidence, write guard, ripple, rollback | 11 | the write guard `DID NOT RAISE`; the ripple table missing; every partner/rollback/inode flow failed `SystemExit: 2` (the new flags and step did not exist) - **so for the flow tests the pre-fix failure shows the FEATURE was absent, not a wrong answer** | see 4 |
| **C7** presence asserted | 2 | a world with no Choice Clause wants and no rim passed V15/V16 (`0 == 3`); no presence evidence (`KeyError: present_in_input`) | V15 requires both Choice Clause wants AND the constitutional node PRESENT in the input and the output; V16 requires the rim present; the detail lists `present_in_input/output` and `missing`; a missing one is a STOP |
| **C8** metadata-flag deny-check | 2 | `TypeError: unexpected keyword nodes_meta`; a `constitutional`-flagged want inside S under an ordinary id passed classify (`0 == 3`) | `deny_check(..., nodes_meta=)` refuses any member of S / the mapping with a truthy `constitutional` or `choice_clause`, or a tag/tags/kind/category mentioning choice_clause; wired into classify, the build and V15 |
| **N1** | 1 | V13 detail had no "writer-enforced" | the V13 detail says `writer-enforced` (it records the writer's counts; it is not a second pass) |
| **c026-C3** V18 precedence | 1 | `V18` stayed True with an EMPTY mapping and a tampered artifact hash | the artifact-hash check is unconditional: `bool(art) and all(...)` |
| **P1 heads** | 1 (+1 record test that already held) | the preamble did not print the heads | both heads printed and recorded; file sha256 asserted after import |
| **Naming (N-procedural)** | 1 | Phase 1 wrote `candidate-id-map.json` while Phase 2 needs `id-map.json` | Phase 1 now writes `reports/id-map.json`; `--frozen-dir` help names exactly the three files to freeze UNEDITED: `reports/repair-list.json`, `reports/scope-ids.json`, `reports/id-map.json` |

## 4. C6 - the DECISION, and its cost

**Decision: I ADDED a gated `--step rollback` (option i).** Why: a torn apply (main replaced, sidecar not) is a documented Phase-2 failure mode, and after it P4's backup-equality leg REFUSES a re-run (checker-026 reproduced this), so without the step the only recovery is a hand-typed restore of six files under pressure on a live identity substrate. A tested, hash-verified step from the tool's own backup is safer than an improvised one.

**Cost, stated plainly:** (1) it is a **second write path into the live checkpoint** (still only `checkpoint_guardian.atomic_file_write`, tmp + `os.replace`, never in place), 83 more lines (`stage_rollback`) the delta pair must read; (2) it can never be exercised on real data before it is needed - it is tested on synthetic directories only; (3) a buggy rollback could damage the state it is meant to protect, so it carries the same gate stack as the apply and more: Josh's go quoting the backup-manifest sha256 (refused first, before anything loads), `--code-placed-at`, the pinned `--expect-*`, P4 (daemon down; its equality leg is replaced by the identity check below) and P6, all re-checked **immediately before each `os.replace`**; (4) it is **not** a general undo: it **refuses unless every live file is either the pre-apply backup's bytes or the post-apply receipt's bytes** (an apply that died between the replaces, or a finished apply); anything else (someone wrote after; S4 authored something) is refused with "identity ... a post-S4 restore needs the P391 export first" - so it cannot lose S4-authored content silently, it simply does not run; (5) its source is **ONLY** `<run>/backup/` verified against `backup-manifest-<UTC>.json` - **every** sha256 is verified before any write, the run directory must lie under the backups root, so a generation directory or `last_good/` is refused (tested for three bad `--run-dir` values); (6) the RETIRED receipt is left in place (the tool stays one-shot; a re-apply after a rollback is refused and is Josh's), the receipt records `host_stays_down` until Chief's resume gate (P398) and the Executive's parser ruling (P423-C2). If the delta pair prefers option (ii), deleting `stage_rollback` and its tests is self-contained.

**The INODE evidence (both cases, Exec P428), as built:**
- **BEFORE**, in the backup manifest Josh's go quotes: `st_dev / st_ino / st_nlink` of the six live files (`copy_six` stats the source before copying); P4's equality leg now also requires the same `(st_dev, st_ino)`, so an equal-bytes but REPLACED file is caught.
- **Generation partners**, read-only: `--generation-partner NAME=PATH` (repeatable) names the KNOWN partner paths; each must lie under `<target>/generations/`, be a regular file, and is only `stat`ed and hashed - **the directory is never listed or opened** (test: `os.listdir`/`os.scandir`/`glob` are wrapped and no call touches `generations`). The record holds path, `(dev, ino, nlink)`, sha256 and `same_file_as_live` decided by `(dev, ino)`; a same-bytes copy with another inode is recorded `false` (tested), never assumed.
- **AFTER**, in the post-apply receipt: `live_inodes_before`, `live_inodes_after`, `generation_partners_after`. The tool **asserts** post-apply main and sidecar are **NEW inodes on the same device with link count 1**, every non-rewritten file is the SAME inode (STOP otherwise). In the synthetic world (where the generation partners are real `os.link`s, so main/vectors measure `st_nlink` 2 before): after the apply main and sidecar are new inodes with `st_nlink` 1; vectors keeps its inode and `st_nlink` 2; the main partner keeps the OLD inode (`st_nlink` 1) with the pre-repair sha256 (`same_file_as_live` false); the vectors partner is still the same file as live. The rollback records the same before/after and asserts the restored files are new inodes with link count 1.
- **Write guard:** `refuse_inplace_write` refuses any existing destination with `st_nlink > 1` in `out_write_bytes` (so every report/artifact), `copy_six` and `rewrite_main`; nothing in the tool opens a live file for write - the atomic tmp + `os.replace` path is the only writer.
- **Ripple:** `RIPPLE_TABLE` (in every backup manifest, receipt and rollback receipt): `generations/` = "incidental, expiring (rotation prunes it), never a rollback source; live files may be hard-linked into it (Exec P428, by inode); only stat/hash of recorded partner paths, never listed"; `last_good/` = NOT a link partner, never a rollback source; the rollback source is only the tool's own named backup.

## 5. Tests - how many runs, honestly

New file, targeted, `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider` from the tool worktree root, each run AFTER a push. **Four runs, not one** (the failing-first discipline needs a before-run; details):

| Run | At | Result | Why |
|---|---|---|---|
| 1 | `4d88f14` tests only, tool unfixed | 56 failed / 97 passed | failing-first; but my helper masked the Phase-2 failures (below) |
| 2 | `3f7cb85` helper fixed, tool unfixed | **50 failed / 103 passed** | the real failing-first evidence in 3 |
| 3 | `e94b3ce` tool fixed | 18 failed / 135 passed | all 18 = ONE real tool bug: the ripple text (296 chars) tripped the report text guard in every `phase2-backup` |
| 4 | `e7bd634` | **153 passed in 40.07 s**, exit 0 | final |

(TURN A had 103 tests; there are now 153 collected: 50 more, 2 rewritten, the rest adapted only for the required keyword arguments / patched constants.) The P379 preamble of run 4 is quoted in 2; `cc_ng_organism` was the PIN copy and the tests fail if it is not. I also ran ONE non-pytest sanity script (import, `--help` text, one stubbed `Probes.unit_active` call); no scratch pipeline runs this turn.

## 6. What I chose or could not settle (flags)

1. **"A Choice-Clause tag" (C8).** The plan names no tag. I read it as: a truthy `constitutional` or `choice_clause`, or a tag/tags/kind/category containing `choice_clause`. If the real nodes carry another marker, the deny-check does not see it. **Needs the Chief/Executive to name the real marker.**
2. **`struck` at Phase 2 STOPs the whole run.** In `stage_apply` any refused candidate (a `struck` decision included) is treated as "deviation from the approved list" and STOPs (Exec P423-C1 defines `struck` = leave the node as-is; plan 4.2-6 says a node that "fails approval" at Phase 2 is a STOP). A legitimate `struck` id therefore blocks the apply rather than being skipped. Unchanged from TURN A and not in the A2 list: **flagging it**, not deciding it. If Exec means a struck id to be skipped, that is a small change plus a frozen-id-map rule (the frozen map would be the approved subset).
3. **`is-enabled` of a NOT INSTALLED unit fails closed.** `systemctl --user is-enabled` on a unit that does not exist prints nothing, so H2's Leg-2 timer check reports a probe error and P6 fails. That is the specified behaviour ("`unit_enabled` likewise") but it is a **TURN B cost**: if `cc-callosum-leg2.timer` is not installed on this laptop, the hold cannot pass as written; the operator (or a Chief ruling that an absent unit counts as off, proven by `is-active` answering `inactive` rc 4 from a reachable bus) has to decide. I did not add that extension.
4. **`foreign_unreadable`** (other-uid `/proc` entries that could not be read) is counted on the `Probes` object but not surfaced in the gate result. Advisory; not done.
5. **The Phase-2 flow tests still stub P3(b)** (checker-026 c026-C4): TURN B's `--apply` on a COPY must run the real `gate_p3(..., run_pinned_tests=True)`; do not take the stubbed tests as proof of it.
6. **Advisory items N2-N8 (le-029) - not done, by instruction:** N2 (the text guard is a backstop; the guarantee is by construction), N3, N4 (real-file V13 / memory unmeasured), N5 (`atomic_file_write` has no `fsync`; the backup is the answer), N6 (flag rulings - the Chief has ruled them), N7, N8 (`oneshot-tool/` must be removed before any merge of the handoffs; nothing enforces it). checker-026's c026-C5 (flag 1) is closed by the Chief's ruling and item 2 of the P1 heads above.
7. **How TURN B calls it now (changes since TURN A):** `phase2-backup`, `--apply` and `--step rollback` all need `--code-placed-at`, a readable `<workspace>/daemon.log`, and `--expect-*` left at the defaults; `phase2-backup` should be given `--generation-partner main.msgpack=<target>/generations/<stamp>/main.msgpack` and `--generation-partner vectors.msgpack=<...>/vectors.msgpack` (the two paths Exec P428 verified); `--daemon-organism-file` must be the unit's own import root, named by the Chief (Q9), NOT a file in the pin or tool worktree; the operator freezes the three named files of the classify run. TURN B itself (Phase 1 on a COPY) needs none of the Phase-2 flags.

## 7. What I did NOT verify

The hardened tool has still never touched a real checkpoint or a real host state: the real `Probes` against a running daemon, a real systemd bus, a real crontab and the real `/proc` of the daemon user (the fail-closed shapes are proven only through stubbed `subprocess.run` and synthetic `/proc`-like directories, plus the real `/proc` scans run by two of the tests on this host); the gated rollback and the inode assertions against a real hard-linked live file that the daemon's guardian may also touch (the synthetic hard links are made with `os.link`); the real generation directory and its partners (never listed or opened - the two paths are cited from Exec P428, not re-checked); P3(b); the real-file V13, peak memory and outcome split (TURN B); whether the rotation of `generations/` can race the recorded partner sha256 (a partner that vanished is recorded `present: false`, not asserted); the `systemd-run` invocation. Also not verified: that `Probes` unknown-word handling covers every `systemctl` wording of this host's version (it accepts only the exact words above and treats everything else as UNKNOWN by design).

## 8. Closing check

`git status -sb && git log -1` on the tool worktree: clean, tip pushed. The PIN worktree was not touched. A tiny fresh law-enforcer re-look follows; TURN B (the Phase-1 dry run on a COPY, 6 GB cap, recording both heads and the before/after inodes and link counts) is a separate message. I have stopped.
