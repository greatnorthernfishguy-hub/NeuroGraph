```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11228, TURN A) - build-tool-001: the 118-want
             TEXT repair ONE-SHOT TOOL and its synthetic-graph tests, built to plan-004 [R4b]. RETURNED, not self-accepted.
-------------------
```

# build-tool-001 - the one-shot tool + tests (TURN A) - RETURNED

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11228 - tool branch `cc-laptop-want-repair-tool-20260930` (from `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37`) - plan `plan-004` (NG branch `cc-laptop-want-text-repair-20260930`, HEAD `5b539216756bb0f9d731bef929ac0a104ff81912`).
Related: [[NeuroGraph]] - [[The Laws]] (LAW 3/4/5) - [[The Choice Clause]] - [[Duck Ethics]] - [[NeuroGraph Is a Mind, Not a Database]]

**Status.** TURN A only. **Nothing applied, merged, deployed, restarted or wired.** No real checkpoint was read for data, no protected or vendored file is in any commit (diff = exactly two new files, below). The tool refuses `--apply` without Josh's go and cannot pass the Phase-2 gates in this build. **No raw want text and no conversation excerpt is in any pushed file, commit message or reply**; all test text is invented filler (`grep` of both files for the Choice Clause wording: none).

## 1. Commits (all pushed to `origin/cc-laptop-want-repair-tool-20260930`, in order)

| # | Commit | What |
|---|---|---|
| 1 | `6eb1a96c0ce3001cc39fc54214fe79fd8f2ef692` | the tool (`handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py`) |
| 2 | `a2babd42a4334390629a99c833a10a4e4f0c649b` | tool fixes found before the first test run: V14 needle matched its own source; Phase-2 scope pin used a literal 118 as well as `--expect-scope`; writer now deletes its partial output on any STOP; unique run dir; `prepare_outputs` split into `build_outputs` + `run_verifier` (same behaviour) |
| 3 | `ea4b5b4f74300a33b9f24627d32314673f823b51` | the tests (NEW file `tests/test_want_text_repair_oneshot.py`, own commit) |
| 4 | `98564f4fed2d88787625264b7d0d7e6c7df3c74b` | one wrong assertion in MY test (plan 6.8(3) reading) - the tool was right |
| 5 | this file's commit | `git rev-parse HEAD` on the branch after it (a document cannot contain its own hash) |

`git diff --stat e4ebf982..98564f4`: `oneshot-tool/want_text_repair_oneshot.py | 2549 +` and `tests/test_want_text_repair_oneshot.py | 1518 +` - **2 files, 4067 insertions, 0 outside those two paths**. File sha256 at 98564f4: tool `93c13ebd5b1d0192308a924029edace7f7bd8fbc90bd2ea97a5448015c289b47`, tests `f9ed1bafaa8debed33ecf4d7746e18fefa5281e3f1fa79b345e316ab4fb96734`. Location per plan 6.8 / le-024 C8: never under `scripts/`, never at the repo root, never merged; a commit removes `oneshot-tool/` before any merge of the handoffs. Each file carries a changelog header; the tool carries the dated ONE-SHOT notice with a `RETIRED: (not yet ...)` placeholder (stamped only after an apply, which changes its sha256 - see 5.11).

## 2. Worktrees and the pin, recomputed this turn

Created exactly as the assignment gives: tool worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930` (branch above) and the detached, never-committed PIN worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9`.

| Value | Recomputed now | Equals the frozen pin |
|---|---|---|
| PIN `git rev-parse HEAD` | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` | code commit - yes |
| `cc_ng_organism.py` sha256 (file on disk in the PIN worktree) | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` | yes |
| blob (`git rev-parse HEAD:cc_ng_organism.py`) | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` | yes |
| `tests/test_cc_want_legitimacy_810.py` sha256 | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` | yes |
| PIN worktree `git status --porcelain` | 0 lines (clean; nothing ever committed there) | - |
| `ae798b94` ancestor of the frozen branch head `c7921b84...` | yes | - |

## 3. What implements each gate / check (`file:line` are in the tool at 98564f4; tests are in the new test file)

**Gates P1-P10**

| Gate | Implemented by | Proven by (tests) |
|---|---|---|
| P1 function identity | `load_pinned` `:209` - file sha256 + blob from DISK before import, blob at tree HEAD, test-file sha256 EQUAL to the pin, HEAD descends from the code commit, tree not a primary checkout, every loaded NG module equals its HEAD blob | `test_p1_a_wrong_function_sha/blob/test_file_sha_refuses`, `..._non_pinned_file_refuses`, `..._refuses_a_pin_root_that_is_a_primary_checkout` |
| P2 same code the daemon runs | `stage_apply` gate list: sha256 of `--daemon-organism-file` == pin (path is operator-resolved, see 5.4) | `test_apply_refuses_when_the_daemon_organism_file_is_not_the_pin` |
| P3 behavioural fingerprint | `gate_p3` `:1932` - 16-row battery of recorded outputs of the pinned function (row 1 = the closer-after-backtick repro); optional subprocess run of the pinned test file | `test_p3_...passes...carries_the_repro`, `test_p3_refuses_a_parser_that_regresses_the_closer_after_backtick_repro` |
| P4 daemon down, mechanical | `gate_p4` `:2037` - unit inactive, recover timer inactive, no daemon/service/rpc process, no process holds the six files, `daemon.pid` names a dead pid, six files equal the start-of-Phase-2 backup (sha256/size/mtime), no pulse since code placement | `test_p4_...` (each leg fails; pid file; log newer than placement) |
| P5 frozen list pinned / stamps | `pin_stamp` `:180`, `check_stamp` `:435`, `load_artifact` `:441`, `load_approvals` `:1390`; `stage_rewrite` proves the re-derived repair-list/scope-ids hash to the saved ones | `test_the_stamp_is_the_frozen_pin_tuple`, `test_every_report_artifact_is_stamped...`, `test_p5_refuses_an_artifact_whose_stamp_is_not_the_frozen_pin` (5 mutations), `test_the_rewrite_step_stops_when_the_rederived_classification_is_not_the_saved_one` |
| P6 peer hold H1-H3 | `hold_snapshot` `:2071`, `gate_p6` `:2089`; recorded at backup (`hold-start.json`), compared at apply, **repeated immediately before the os.replace** (`stage_apply.recheck`); conduit compared by `stat` only, never opened | `test_p6_...` (each of H1/H2/H3 fails closed; change between start and now caught), `test_apply_rechecks_the_gates_immediately_before_the_replace` |
| P7 target guard | `guard_target` `:343` (+ `guard_out_path` `:364`, `is_primary_checkout_path` `:193`, `daemon_checkpoint_dir` `:329`) - required arg, no default, realpath recorded; HARD REFUSES Syl's checkpoints / a primary checkout / not-the-recorded-CC-dir / daemon-script disagreement; Phase 1 writes only under `<backups>/z12-want-text-repair-*` | `test_the_target_dir_is_a_required_argument...`, `test_hard_refusal_*` (3), `test_the_daemon_script_cross_check_must_agree`, `test_phase_1_may_write_only_under...` |
| P8 Josh's go | `main` refuses `--apply` without `--josh-go` BEFORE loading anything; `stage_apply` requires the go to quote the sha256 of the start-of-Phase-2 backup manifest and verifies it | `test_apply_is_refused_without_josh_go...`, `test_apply_refuses_when_josh_go_does_not_quote_the_backup_manifest` |
| P9 scope pin | `stage_apply` - frozen `scope-ids.json` list, size == `--expect-scope`, live rule-derived set == the frozen list, live repair-list/scope-ids hash == frozen; `--scope-min-len` is a reported cross-check only (`scope_ids_obj`) | `test_the_scope_ids_are_enumerated_and_the_rule_is_only_a_cross_check` |
| P10 (S4's, not the tool's) | the tool PRODUCES its input: `t6_replay` `:1265` writes the stamped would-mint set (ids, lengths, marker flag, `in_url`/`in_json_string`/`in_link_target` flags); `apply` prints that the S4 start waits on it. The tool does not (and per P423 must not) gate on it | `test_t6_...` |

**Verifier V1-V19** (`Verifier.lockstep` `:1661` is one lockstep pass over input+output using the SAME `iter_sections` walk as the writer; `Verifier.run` `:1763`; a failing check = no write). Every check has a positive run and a tampered-output negative:

V1 counts `:1772` - V2 ids/positions/accounting `:1780` - V3 node carry-over incl. `poincare_dir` bytes `:1785` - V4 synapses + per-old/per-new incident counts `:1786` - V5 hyperedges + archived `:1790` - V6 S4/S6-S12 + every other top-level value byte-equal `:1791` - V7 non-want nodes via the independent `diff_modulo` `:1608` `:1793` - V8 sidecar + other files sha256-equal `:1801` - V9 verbatim/untruncated, no length assertion `:1812` - V10 census (raw byte census == writer substitutions == walker substitutions; old ids left = 0) `:1818` - V11 canonical `Graph.restore` (per-id `_outgoing/_incoming/_node_hyperedges` figures, no dangling, render bytes before/after) `:1847` - V12 idempotence (pinned function re-derives every new id; identity rewrite of the output is a byte no-op) `:1860` - V13 fidelity `:1862` (**see 5.9**) - V14 shared-function proof `:1867` - V15 Choice Clause / constitutional / every unrepaired want, ONE reading "byte-equal apart from the key remap of moved ids" + deny-check `:1873` - V16 rim (count equals the count in the mapping, not a literal 131) `:1876` - V17 approvals `written == approved ∩ passed` `:1880` - V18 mapping + INVERSE + post-apply-receipt draft exist and hash-verify `:1886` - V19 all nodes evaluated before any byte, every `assert_failed` listed `:1888`. T6 = `t6_replay`.

**Algorithm 4.2 (`Classifier.classify_node` `:992`)**: A0 source (`creation_mode == "conversational"`, not want/constitutional/`*_authored`, content contains `WANT]`) - A1 anchor (`count(T)==1`, outer opener, closer skipping whitespace, `closer_end = j + len(WANT_CLOSE)` with the imported constant) - parse once per distinct source (cached) - decide by the want closing at `j` (GENUINE / SEPARATE / ANOMALY / NONE / OVERLAP-ANOMALY, no guessing) - post-conditions with `assert_failed` (never a bare assert) - report fields (class A/B/C, reasons, region flags, outer-opener skip reason, excerpt anchors, PRE-node line). `apply_collision_rule` `:1084` drops AND lists; `deny_check` `:1114`; `build_mapping` `:1101` (injective, else Stop).

**Report artifacts** (all stamped; `_assert_text_free` refuses a text-like key or a >200-char string): outcome table, histograms twice with all 11 reasons + hand-review node ids, marker-bearing minted list, residual classes (base vs pinned difference grouped by the pinned skip reason, with `unclassified`), `repair-list.json`, `scope-ids.json`, PRE-node report line, candidate id-map, mapping + inverse, T6 would-mint set. Excerpts / LEFT list: `<run>/review/*.md`, mode `0600`, scrubbed (`scrub` `:466`, rule-named redactions), `excerpt_sha256` covers all five anchors + scrub version.

## 4. Tests - the P379 preamble and results

New file only, targeted, **run twice** (first run failed on my own test error, fixed and pushed BEFORE the second run; both runs were after a push):
`env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider` (cwd = the tool worktree root).

- Run 1 (at `ea4b5b4`): **102 passed, 1 failed** in 15.5 s - `test_a_retired_receipt_for_the_same_checkpoint_directory_refuses`: my test expected a refusal for an unrelated directory; plan 6.8(3) refuses on the SAME directory or this tool's sha256. Test fixed (`98564f4`), tool unchanged.
- Run 2 (at `98564f4`): **103 passed in 13.05 s**, exit 0.

P379 preamble printed by the run (the tests FAIL if `cc_ng_organism` is not the PIN copy - `test_p379_preamble_...` and the autouse fixture):
```
P379 sys.executable /usr/bin/python3
P379 sys.path[0:6] ['.../z12-want-repair-pin-ae798b9', '.../z12-want-repair-tool-20260930', '.../z12-want-repair-tool-20260930', '/usr/lib/python312.zip', ...]
P379 cc_ng_organism.__file__ /home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9/cc_ng_organism.py
P379 cc_ng_organism sha256 8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2
P379 PYTHONPATH None ; NG_EMBED_* names none
P379 checkpoint_guardian / neuro_foundation / universal_ingestor / cc_ng_organism -> all under .../z12-want-repair-pin-ae798b9/
P379 ng_lite, ng_embed, neurograph_rpc, cc_ng_host, activation_persistence: not loaded
```
Coverage requested by the assignment, all in the file: each outcome class (incl. OVERLAP, a GENUINE want containing a masked mention pair, `assert_failed` via a tampered id function); the collision rule (two repairs producing one id, a new id already in the graph, a chain onto another want's id - both parties dropped AND listed); Choice Clause byte-equality of both wants + the constitutional node + every unrepaired want, with the deny-check (neither id in S, the mapping as old or new, or any approval); mapping + inverse; the rim / `pred_weights` assertion (a key remap of a moved id passes; a changed value, an added key, a removed key, another field, an edited text all FAIL V15); id-follows-text; stamp / P1 / P5 / import-isolation refusals (wrong sha, blob, test-file sha, non-pinned file, a preloaded module outside the pin, `__file__` == the primary checkout path); hard target refusals; idempotence; retirement refusals; V1-V19 negatives; approvals refusals (unapproved, struck, three hash mismatches, six binding mismatches, provisional packet); and the WHOLE Phase-2 path on a synthetic dir with fake probes (backup, gated apply, atomic replace, retirement, second apply and second backup REFUSED, deviation = STOP with live files untouched, gates re-checked before the replace).

## 5. What I could NOT implement, where the plan was ambiguous, and what I chose (each is a flag for you or the delta pair; I did not silently settle any)

1. **Stamp "branch head" vs the PIN worktree's HEAD.** The plan's stamp names `c7921b84...` (the docs-only return commit) but the assignment's pin worktree is the code commit `ae798b94` (its parent), so `git rev-parse HEAD` in the pin tree can never equal it. Chosen (least-committal): the stamp carries the FROZEN constants (`c7921b84...`); the P1 record carries the actual tree HEAD (`ae798b94...`); P1 gates on the three hashes (disk AND at HEAD) and requires HEAD to descend from the code commit; it does NOT require HEAD == `c7921b84` (that would refuse the assignment's own pin tree). **Needs your ruling** (or re-pin the worktree at `c7921b84`).
2. **"Reuse the analysis-001 loader, do NOT fork it."** The only thing I found by that name is the off-repo scratch script `analysis-scratch/analyze_pair.py` (top-level argv, not importable). I reused its LOADING MECHANISM - `Graph().restore` + `SimpleVectorDB().load` (its lines 84-86) - in a 6-line `load_pair` and copied none of its statistics code. If you meant a different importable loader, tell me its path.
3. **Order of approvals vs rewrite in Phase 1.** Plan 6.6 lists "the review and its signed approvals" before "rewrite to a tmp file", but approvals exist only after the Executive rules on the classify output, and TURN B is "classify/rewrite-to-tmp/V1-V19" with none yet. I added `--provisional-approve-all` (DRY-RUN ONLY): it self-approves for the tmp rewrite, the packet is `PROVISIONAL-SELF-DRY-RUN-NOT-AN-APPROVAL`, the report says `provisional: true`, `load_approvals` REFUSES that packet, so Phase 2 cannot use it. Please confirm this is what TURN B should use.
4. **P2 "path resolved from the unit at run time".** Q9 (unit/slot) is still open, so the tool takes `--daemon-organism-file` explicitly (operator-resolved) and compares its sha256 to the pin; it cannot resolve the path from the unit by itself.
5. **P4 "deployed but not restarted".** My mechanical realisation: unit inactive AND `daemon.log` mtime <= `--code-placed-at` AND the pid file names a dead process. The plan offers "unit start time precedes placement, or the log shows no pulse since"; I picked a form I could test without parsing systemd timestamps. Interpretation, not a plan quote.
6. **P6 limits.** H1's "flag not `1` in any environment the daemon or sync reads" can only be read from the tool's own environment while the daemon is down; the crontab check is "hash + no uncommented callosum line"; the conduit path (H3) is an operator argument and stays **[unverified]** as the plan says.
7. **P3(b)** (run the pinned test file's ~280 tests as a subprocess) is implemented and REQUIRED at `--apply`, but the tests of this build stub it in the Phase-2 flow tests (it is #810's own suite, run under its own turn). Unexercised here.
8. **V1's 182/183 and scope 118** are the plan's literals as DEFAULTS of `--expect-wants/--expect-protected/--expect-scope`; a Turn-B mismatch on the real copy is a STOP, not a silent continue. At Phase 1 the scope is derived by the `>600` rule on the COPY and written as `scope-ids.json` with `matches_expected_count`; only Phase 2 STOPs on a size other than the expected one.
9. **V13 is enforced in the WRITER, not re-proven by the verifier.** Each re-encoded value is proven `pack(unpack(raw)) == raw` inside `rewrite_main` (a failure raises `Stop`, tested with a non-minimal int) and the sidecar must round-trip through `json` byte-for-byte; the verifier's V13 line records the writer's counts and is `True` because a failure never reaches it. Say so to the delta pair - it is not an independent second check.
10. **Advisory mention-shape flags (FINAL PIN (d)).** `mention_shape_flags` is a best-effort set of named regex hints for the frozen-list review (single-quoted JSON-ish, reference definition, `www.`/`mailto:`/`data:`, YAML/TOML line, blockquote, indented code, HTML comment / `<code>`, emphasis, link text, F4b). It is never a decision input and does not claim to classify every (d) class; a bare prose pair has no flag.
11. **Retirement stamp.** Editing the tool's header after an apply changes its sha256; the receipt records the sha256 at apply time (`tool_sha256`) and the refusal keys on ANY `RETIRED-*.receipt` for the checkpoint directory OR a receipt recording that sha256, so the header edit cannot re-enable it. The header stamp itself is not written (nothing applied).
12. **File-name shorthand.** The plan lists `.guard_state.json` / `.manifest.json`; the real files are `main.msgpack.guard_state.json` / `main.msgpack.manifest.json` (per `checkpoint_guardian`). The tool uses the real names.

## 6. Found while working (not fixed - for the punchlist / Josh)

- **Hard-linked checkpoint files.** `ls -l` of the live CC checkpoint directory shows `main.msgpack` and `vectors.msgpack` with a link count of **2** (mode `0600`) - very likely hard links into `last_good/`. The tool never writes in place (Phase-1 copy = a new inode, `shutil.copyfile`; Phase 2 = `atomic_file_write`'s tmp + `os.replace`), so the link partner keeps the pre-repair bytes; but any in-place writer would change BOTH names, and `last_good/` is not a rollback copy of the post-apply state. The rollback plan (6.7) should state which copy it restores from. **[unverified]** that they are hard links into `last_good/`; I did not open or list that directory.
- **Peak memory of the real classify.** `analyze` frees the vector arrays after `SimpleVectorDB.load` but the PEAK happens during that load of the ~1 GB `vectors.msgpack`; the plan's 3 GB cap for classify/rewrite is **unmeasured** for the real files (the plan measured only V11's restore, ~3.6 GiB). Suggest TURN B run classify under the 6 GB cap or measure first.
- Already tracked: daemon `CC_NG_WORKSPACE` hard-coded literal (row #826).

## 7. What I did NOT verify

The tool has never touched a real checkpoint: real-file behaviour of the value-granular rewrite (in particular V13 on the Rust-emitted synapses map of the real 139k-synapse file - a fidelity failure there would STOP by design), real memory/time under the `systemd-run` caps, the real `Probes` class (`systemctl`, `crontab`, `/proc` scans - only fakes are tested; the class imports and its methods are read-only queries), P3(b), the daemon-script cross-check against the real daemon script (a synthetic script of the recorded two-line shape only), `is_primary_checkout_path` against the real primary checkout (deliberately never referenced - the logic is tested on a synthetic repo + linked worktree), the real outcome split / collisions / histograms (Phase 1 on the COPY = TURN B), any behaviour under a different cwd than the tool worktree root.

## 8. Touches outside the strict "TURN A touches no real checkpoint" line (full disclosure)

- One `ls -la` of the live CC checkpoint directory (names, sizes, mtimes, link counts; **no file opened**), and an `ls` of `analysis-scratch/laptop-copy` + a `cat` of `laptop-copy-hashes.txt` (hashes only) - to learn the six real file names.
- An over-broad `find / -name plan-004.md ...` at the very start (name search / directory walk; **no file content read**, run once, in the background). It may have listed directory entries under directories I am told not to touch; nothing was opened. I should have used the known docs/NG paths.
- Read-only reads of: the docs-branch assignment and the plan and its four reviews; `build-004` and `le-021` in the parser worktree; the analysis scratch scripts and `plan-scratch/protected_census.py` (text); the daemon script by `git show`; the pin worktree's `cc_ng_organism.py`, `neuro_foundation.py`, `activation_persistence.py`, `checkpoint_guardian.py`, `universal_ingestor.py`, the pinned test file (read only, never edited).
- **Exploratory scratch runs (not the suite), all on synthetic data in `/tmp/z12-repair-probe/`:** a `Graph` checkpoint round-trip, the pinned parser's outputs over the fingerprint battery (to record the P3 expectations), the classifier's outcomes over the synthetic corpus (to design the tests), and one run each of the full Phase-1 and Phase-2 pipelines against synthetic directories (before the commits). These are disclosed here because they are executions of the tool code outside the two counted pytest runs.
- No daemon, unit, timer, tract, `~/.bashrc`, Syl directory or primary checkout was opened, written or restarted. No `git stash`. The vault docs / wikilink sync for this work is the zone manager's (outside my authorised paths).

## 9. Closing check

`git status -sb && git log -1` on the tool worktree: clean, tip pushed. TURN B (the Phase-1 dry run on a COPY) is a separate message from you; I have stopped. Suggested shape of the TURN B call: `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B want_text_repair_oneshot.py --pin-root /home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9 --target-dir ~/.claude/plugins/neurograph/checkpoints --daemon-script <the docs-branch cc-ng-daemon.py> --scope-min-len 600 --step classify`, then `--step rewrite --run-dir <that run dir> --provisional-approve-all` (see 5.3).
