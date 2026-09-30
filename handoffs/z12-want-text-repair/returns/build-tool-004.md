```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11587, TURN A2c) - build-tool-004: the TINY fold after le-034:
             the Exec P433 confirmations (a)/(b)/(d), the ONE code addition (c) (the advisory review hint), and le-034 C-1..C-4 + N-2.
             RETURNED, not self-accepted.
-------------------
```

# build-tool-004 - TURN A2c, the tiny fold - RETURNED

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11587 - tool branch `cc-laptop-want-repair-tool-20260930` - base = the A2b return `00eb0ae14cc3bbdd51d1fd4b331cc1bf093dc106` (code `a300bf1b...`; le-034 on top).
Related: [[NeuroGraph]] - [[The Laws]] - [[The Choice Clause]] - [[Duck Ethics]]

**Status.** TURN A2c only. **Nothing applied, merged, deployed or restarted.** Synthetic data only; no checkpoint directory was listed or opened, no real checkpoint read, the live tract and Syl's directories untouched, `~/.bashrc` untouched. No protected or vendored file in any commit (my diff = the tool file and the test file plus this return). No raw want text anywhere.

## 1. Commits (all pushed to `origin/cc-laptop-want-repair-tool-20260930`, in order; le-034's commits sit before them)

| Commit | What |
|---|---|
| `1dfc567221c82a7f81e5e89fa803fc616950d5e2` | tests(A2c): failing-first tests for the hint and le-034 C-1..C-4, N-2, **plus the pins for (a)/(b)/(d)** - committed BEFORE any tool change |
| `1988e73d1dfa9ee9394bd564d8b0f33dd89224f8` | the tool change (the advisory hint, C-1..C-4, N-2) |
| `1929c0df5c691c3c213d56712a86ee7b17291a78` | tests: `-inf` checked at the input check (my test bug: `argparse` rejects a leading-dash value before the tool ever sees it) |
| this file's commit | the branch tip after it |

`git diff --stat 00eb0ae..1929c0d` on my paths: `oneshot-tool/want_text_repair_oneshot.py | 95 +++++--` and `tests/test_want_text_repair_oneshot.py | 281 +++++--` (358 insertions, 18 deletions). File sha256 at `1929c0d`: tool `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`, tests `9b89f687764c6ef90fa7df01a34d9c94210b8a170abe73643086fb13901e2ab6`. The tool stays under `handoffs/z12-want-text-repair/oneshot-tool/`; the tests are in their own commits.

## 2. Pin worktree, recomputed

| Value | Recomputed now | Equals the pin |
|---|---|---|
| PIN `git rev-parse HEAD` | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` | yes (code commit; the frozen stack head is `c7921b84...`, both still recorded by P1) |
| `cc_ng_organism.py` sha256 | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` | yes |
| blob at HEAD | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` | yes |
| `tests/test_cc_want_legitimacy_810.py` sha256 | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` | yes |
| `git status --porcelain --ignored` | 0 lines (never edited, never committed) | - |

## 3. (a) THE STOP PREDICATE - verified, stated, pinned (no code change)

**The tool's STOP predicate for the Choice Clause is keyed on the marker set, NOT on "protected" or `*_authored`.**
- The predicate is **`is_choice_clause_marked(md, nid)`** (`want_text_repair_oneshot.py:1141`): the rim id, either literal Choice Clause want id, `constitutional` truthy, `source == "cricket_rim"`, `creation_mode == "constitutional"`, or a `rim_source` key present. It is applied by **`deny_check`** (`:1155`), which reads S, the mapping AND every approval id, and is called from classify (`:2616`), the build (`:2529`) and V15 (`:1975`). Nothing else decides a Choice-Clause STOP.
- `gate_write_set` (`:1519`) decides only `struck` / `not_approved` / `approval_mismatch` per id, and `stage_apply` (`:2851`) turns `struck` into a recorded deviation and STOPs on the other refusals; **none of these reads provenance or `*_authored`**. A struck ordinary `cc_authored` want completes the apply (tested end to end in A2b and again here); a struck marker-set id STOPs at `deny_check`.
- **Where `_authored` DOES appear in the tool, so nobody is surprised** (grep of the whole file): (1) `is_protected` (`:564-567`, a mirror of the canonical `Graph._is_identity_protected`, parity-tested) - used ONLY for the V1 protected counts (`:1809-1810`), the classify step's A0 "the source node is not protected" eligibility test (`:1033`, whose result is the LISTED, UNCHANGED outcome `SOURCE_MISSING`, not a STOP) and the receipt's protected-id list (`:2548`); (2) `derive_scope` (`:1000-1004`), which selects which wants are IN S by `provenance == "cc_authored"` - a membership rule, not a STOP test. No STOP path calls either.
- **Pinned by tests** (these two pass against A2b and are expected to; they are pins, not new behaviour): `test_a2c_a_the_stop_predicate_is_the_marker_set_and_no_stop_path_reads_authored_or_protected` (the source of `deny_check`, `is_choice_clause_marked`, `gate_write_set`, `stage_apply` and `stage_rollback` contains neither `_authored` nor `is_protected`; provenance `cc_authored` / `syl_authored` / `cc_emergent` alone never marks or stops) and `test_a2c_a_a_struck_ordinary_cc_authored_want_completes_the_apply_and_a_struck_marker_id_stops_it`.
- **(b) verified in the code and pinned:** `return "rim_source" in md` (`:1152`) - the key exists, whatever its value. `test_a2c_b_rim_source_present_means_the_key_exists_with_any_value` covers `"seed_cc_rim.py"`, `""`, `None`, `0`, `False`, `[]`, `{}` (all marked) and the absence of the key (not marked). Marker keys are read at the TOP LEVEL of the metadata only (le-034 N-1; the P432 list names top-level keys).
- **(d) verified and pinned:** with a struck id the frozen `id-map.json` is the CANDIDATE mapping and `stage_apply` requires frozen = applied mapping + struck pairs; the FINAL receipt's `struck_deviations` entries carry `id`, `would_have_been`, `class` and the 16-hex `old_sha16` / `new_sha16`; the inverse mapping file covers ONLY the applied set (`test_a2c_d_struck_entries_carry_hashes_and_the_inverse_excludes_them` reconciles `applied pairs + [struck pair] == frozen pairs`).

## 4. (c) The ONE code addition - the advisory review hint

- **What it is.** `HINT_TERMS` (`:1400`) - ONE named constant tuple, no config key: `leave, leaving, exit, quit, refuse, refusal, consent, decline, choice clause, say no`. `review_hint(text)` (`:1405`) returns the matched terms in tuple order, once each. `write_review_files` (`:1410`) writes a `hint: <terms>` (or `hint: none`) line on EVERY entry of the two OFF-REPO review files (`review-excerpts-*.md` for the SEPARATE candidates - matched against their four scrubbed anchors - and `left-list-*.md`, matched against the LEFT node's scrubbed excerpt). **It marks; it classifies, skips, reorders, alters and decides nothing**; every id is still decided by the Executive's signature per id, marked or not.
- **Matching, stated exactly.** Case-insensitive; a letter, digit or underscore on either side is NOT a boundary; multi-word terms accept any whitespace between their words; ONE trailing inflection (`s, es, ed, d, ing`) is allowed - so `exit` marks exit / exits / exited / exiting but NOT `existing`, `exitless` or `preexit`; `leaving` is its own term. **No stemming beyond that: `quitting` (a doubled consonant) is not marked**, and `consenting` / `refused` / `declined` are. (Your example "`exit` in `existing`" is not even a substring - `e-x-i-s-t` - so it is unmarked trivially; the boundary tests use `exitless`, `preexit`, `quitter`, `leavening`, `declination`.)
- **Where the marks go - and where they do not.** Only the off-repo review files (mode `0600`, unchanged scrub and text-free guards). **Counts only** go to `run-record.json` (`review_hint_counts`: `entries_total`, `entries_marked`, `per_term` - keys are always members of `HINT_TERMS`, values ints, no text) and to the classify result printed on stdout. Never in `repair-list.json`, `scope-ids.json`, `id-map.json` or any stamped count/report artifact (tested: no `"hint"`, no term name, in any `reports/*.json`); `excerpt_sha256` is untouched (the hint is not part of the anchors), so no approval hash moves.
- **Byte-equality with the hint off** (tested): the same world classified twice, once with `review_hint` monkeypatched to return `[]` - every `reports/*.json`, the id-map, the repair-list / scope-ids shas, the outcome table and the candidate count are BYTE-identical; the two review files are identical apart from their `hint:` lines; the run record differs only by `review_hint_counts` (zero marks, same total).

## 5. Failing-then-passing evidence

Failing-first = the tests committed and pushed BEFORE the tool change and run against the A2b tool: **24 failed / 199 passed**. The 199 include the pins for (a)/(b)/(d) (10 test items - 2 for (a), 7 for (b), 1 for (d) - plus the "hint changes no excerpt hash / no field of the repair-list" test) - said plainly so nobody counts them as new behaviour.

| Item | Failed pre-fix | Pre-fix reason | Now |
|---|---|---|---|
| **(c) hint** | 14 | `AttributeError: no HINT_TERMS / review_hint`; the review files had no `hint:` line; the run record had no counts | as section 4: each of the 10 terms marks (lower / upper / title case), word boundaries, whitespace, marks only in the review files, counts only in the run record with no text, byte-equal with the hint off |
| **C-1** zero-write apply | 2 | an all-struck packet COMPLETED (`rc 0`, `applied 0`) and WROTE the RETIRED receipt; the RETIRED receipt carried no struck trace | `stage_apply` STOPs **`nothing to apply`** right after the live classification checks, BEFORE any staging (`gate_write_set` returns no write id): rc 3, six files unchanged, no FINAL receipt, **no RETIRED receipt** - and the one-shot is provably not consumed (tested: a real packet applied afterwards in the same run directory completes). A partly-struck packet still completes and the RETIRED receipt now carries `struck_deviations` (the COUNT only) and `post_apply_receipt` (the FINAL receipt's name) + `post_apply_receipt_sha256` (ids and hashes stay in the FINAL receipt; the struck id is not in the RETIRED receipt - tested). |
| **C-2** `foreign_unreadable` persisted | 1 | `KeyError: foreign_unreadable` in the manifest | `{"p4": n, "p6": n}` is in the backup manifest, `hold-start.json`, the FINAL receipt (from the apply's own P4/P6 gate results) and the rollback receipt - tested with a probe that counts other-uid entries (p4 = 3, p6 = 2 in all four artifacts). |
| **C-3** non-finite placement | 5 | `nan` passed the check (`nan > now` is False) and only failed later at P4; `inf` was refused but without saying why | `_require_phase2_inputs` refuses `nan` / `inf` (and `-inf`, `+inf`, `Infinity`, `-Infinity`) with "not a finite time". **`-inf` cannot reach the tool through the CLI at all**: `argparse` reads a leading-dash value as an option and exits 2 first, so it is tested at the input check itself; the four other forms are tested through the CLI (refused before any run directory is created). |
| **C-4** same-second overwrite | 1 | one `displaced-<UTC>` directory reused, its files truncated by `copyfile`; one `rollback-receipt-<UTC>.json` overwritten | `_preserve_displaced` creates a NEW directory each time (`displaced-<UTC>`, then `-2`, `-3`, ...; `makedirs(..., exist_ok=False)`, like `new_run_dir`), and the rollback receipt takes a `-N` name when its name exists. Tested with `utc_stamp` frozen to ONE second: two rollbacks -> two displaced directories, the first copy's inode and sha256 unchanged, two rollback receipts. |
| **N-2** | 1 | `--apply` silently won over `--step rollback` | the two together are refused ("mutually exclusive"), before anything loads; tested (live files unchanged). |

N-1 and N-3 (top-level-only marker keys; operational constants such as `PROBE_TIMEOUT_S`, `P3B_TIMEOUT_S` and the systemd word lists are literals) are stated, not changed.

## 6. Tests - how many runs, honestly

New file only, targeted, `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider` from the tool worktree root, every run after a push. **Three runs**:

| Run | At | Result |
|---|---|---|
| 1 | `1dfc567` tests only, tool at the A2b state | **24 failed / 199 passed** (failing-first, section 5) |
| 2 | `1988e73` tool changed | 1 failed / 222 passed - the `-inf` CLI case: my TEST was wrong (argparse rejects the value before the tool sees it); the tool was right |
| 3 | `1929c0d` | **223 passed in 136.47 s**, exit 0 |

(A2b had 188; now 223 collected.) The final run's P379 preamble prints both heads and asserts the file sha256, as before, and `cc_ng_organism` was the PIN copy (the tests fail if it is not). No scratch runs this turn.

## 7. What I did NOT verify

The hardened tool has still never touched a real checkpoint or a real host: the real `Probes` against this host's systemd and cron; **P3(b)** (never executed by the tool or any test, timeout path only stub-tested; it must be run once, read-only, before Phase 2); the rollback, receipts and inode assertions on a real hard-linked live file and a real generation partner; free-disk-space margin for `backup + stage + displaced + tmp` copies of a real multi-GB main file (no preflight exists); `atomic_file_write` has no `fsync` (le-029 N5). **The hint's coverage of the real excerpts is unmeasured**: it is a word list, so an excerpt that expresses leaving or refusing in other words is unmarked, and one that uses a term innocently is marked; that is why it decides nothing. **Whether the frozen-list marker set covers every real Choice Clause node** is the Executive's fact (P432/P433), not something the tool can prove; marker keys nested inside a sub-dict are not seen (N-1). Real-file V13, peak memory and the outcome split remain TURN B's.

## 8. Closing check

`git status -sb && git log -1` on the tool worktree: clean, tip pushed. The PIN worktree was not touched. TURN B (Phase 1 on a COPY: load gate memory >= 8 GB, load < 6, alone; 6 GB `systemd-run` cap; both heads recorded; `--provisional-approve-all`; the P3(b) run a separate read-only step before Phase 2) is a separate message. I have stopped.
