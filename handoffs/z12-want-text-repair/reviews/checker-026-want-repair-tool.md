# checker-026 ROLE A (cross-family) — 118-want text repair ONE-SHOT TOOL (TURN A)

STATUS: INCOMPLETE - draft findings (1)-(8) filled; ROLE B file not yet read

- Seat: checker-026 (cross-family, grok-4.6, `report_only`). ROLE A only of ADDENDUM 2; ROLE B not written and not read.
- Lane: `z12-s3-restore-bundle-20260929`. Dispatch #11342. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-text-repair-118.md` LAST section ADDENDUM 2. Docs branch `cc-laptop-daemon-recall-756-20260930` @ `da893559f7c2359a737c89e14f9bf70bdceadfe6` (read, not edited). Packet file sha256 `0bfd95d0e5af439ca7b650d090af53739a8d7095f2d5e63073dc0b7efd82e1a8`.
- Assignment: `assignments/build-want-text-repair-118.md` sha256 `d71dd75d68d5e3b0e9384f43910732e4204a1d2535d3fb8364ba019ff23efa41`.
- Spec plan-004 (READ only, NG worktree `/home/josh/NeuroGraph-worktrees/z12-want-text-repair-20260930`): packet pin `5b539216756bb0f9d731bef929ac0a104ff81912`, blob `8f24c1139a703f9cf99a1f1ba276711aa261482a`, sha256 `013475e822135a41100e8fabb59ccea7319fbb23dc42f862f623a6d89b0f75a5`. After this review started, that worktree moved to `bacdf08bb77805caebfb98c86773ec8ae1f2de4c` (plan-004 `[R4c]` Exec P428 link-count fold, PLAN ONLY, tool unchanged). R4c blob `cefb59bc077f92c40cf0aa444832a477b0f5d12c`, sha256 `22c7137f8e31b4eb4950f1021120c648a80ad3b92c5dc22f5eabd4b2953156af`. Item (7) is judged against P428 as given by the zone manager and as folded in `[R4c]`. Pair: `reviews/checker-024-want-repair-plan004.md` sha256 `e83813daeefc2f2b5a9ec7844f2012785a797bc31358271a415fe30e4ec4b533`; `reviews/le-024-want-repair-plan004.md` sha256 `971a680c2a07cf08d534028409a291246b56c8843a5f13d99e77a9c2517e909d`.
- Tool under review: worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930`, branch `cc-laptop-want-repair-tool-20260930`. Packet tool head `ab24e85c06a0b524828db895574e7544ca01ed09`. `git diff --name-only e4ebf982 ab24e85c` = the three named files. Tool sha256 `93c13ebd5b1d0192308a924029edace7f7bd8fbc90bd2ea97a5448015c289b47`. Tests sha256 `f9ed1bafaa8debed33ecf4d7746e18fefa5281e3f1fa79b345e316ab4fb96734`. Return `build-tool-001.md` sha256 `662358a67470c4ae62589b6ed7c9869363b71d8beb1e822c242acdd50724c854`. Tool and tests blobs are unchanged from `ab24e85c` through current HEAD (later commits are this reviews directory).
- Frozen parser PIN worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9` (detached, never edited, never committed, never checked out into another worktree). `git rev-parse HEAD` = `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`. `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`, blob `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab`. Test-file sha256 `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53`. `ae798b94` is an ancestor of frozen branch head `c7921b8436fb174c3f70fcf02827f16bb16deff0`. Pin `git status --porcelain` empty.
- Authority: report_only. TURN A: tool + synthetic tests; nothing applied. No real checkpoint, no graph load beyond synthetic, live tract never opened, nothing under Syl's directories or `~/NeuroGraph/data/checkpoints`; live CC checkpoint directories were not listed or opened (P428). Real `~/.bashrc` never written (sha256 still `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`). Secrets by NAME only. Hashes from `git rev-parse` / `sha256sum`. NO raw want text or conversation excerpt in this file.

## P379 session start

Inherited parent env had `PYTHONPATH=/home/josh/NeuroGraph:`. All analysis and both targeted runs used `env -u PYTHONPATH -u NG_EMBED_REMOTE -u NG_EMBED_URL PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`.

First (no import) print, cwd = the tool worktree:

```
python: /usr/bin/python3
sys.path (after unset PYTHONPATH):
  (empty)
  /usr/lib/python312.zip
  /usr/lib/python3.12
  /usr/lib/python3.12/lib-dynload
  /home/josh/.local/lib/python3.12/site-packages
  /usr/local/lib/python3.12/dist-packages
  /usr/lib/python3/dist-packages
NG-related sys.modules: NONE
PYTHONPATH: <unset>
NG_EMBED_*: []
cwd: /home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930
```

Targeted pytest (once) and the independent `/tmp` Phase-1 harness both printed:

```
P379 sys.executable /usr/bin/python3
P379 cc_ng_organism.__file__ /home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9/cc_ng_organism.py
P379 cc_ng_organism sha256 8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2
P379 PYTHONPATH None ; NG_EMBED_* names none
P379 checkpoint_guardian / neuro_foundation / universal_ingestor / cc_ng_organism -> all under .../z12-want-repair-pin-ae798b9/
P379 ng_lite, ng_embed, neurograph_rpc, cc_ng_host, activation_persistence: not loaded
```

Tests FAIL if `cc_ng_organism` is not the PIN copy (`test_p379_preamble_resolves_the_pin_worktree_copy` + autouse fixture). Both runs resolved the PIN copy.

Stub first-write: commit `8ad7d2a7da104dba3badf097d86df6fa6cdc5887` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-repair-tool-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

The TURN A tool implements P1–P10 and V1–V19 as written in plan-004 `[R4b]`, with the worker's twelve flags disclosed rather than silently settled. Hard refusals, the stamp/P5, Choice Clause / rim / `pred_weights` assertion, collision drop-and-list, mapping + inverse, and text-free report guard hold on the code and on a synthetic Phase-1 run. Protected, vendored, and `scripts/` files are byte-identical to `e4ebf982`. Exec P428: the tool never writes in place; rollback is the named hash-verified pre-apply backup; it does not yet record before/after `st_nlink` or generation-partner inode+sha256 (plan `[R4c·P428-scope]` already names that as a TURN B / later follow-up). Additional gaps found in the tool (C6–C10): P4's pulse check is skippable by omitting `--code-placed-at`; live probes fail-open; `--expect-*` is CLI-overridable vs the plan's literal 118; V15 does not require the Choice Clause ids to be present in the walk; a torn main-then-sidecar apply has no automatic rollback.

---

## (1) gates P1-P10 and V1-V19 vs plan-004

**Verdict: PASS-WITH-NOTES**

Read from the tool, not from the return's table. File:line at `ab24e85c` / HEAD (tool blob unchanged).

| Gate | Where | Implemented as written |
|---|---|---|
| P1 | `load_pinned` `:209` | Disk sha256 + computed blob of `cc_ng_organism.py` EQUAL the pin; test-file sha256 EQUAL `04b1a494…` (record-only is not enough); HEAD blob EQUAL; HEAD descends from `ae798b94`; pin root is not a primary checkout and not under Syl; every loaded NG module is under the pin tree and matches HEAD. |
| P2 | `stage_apply` `:2422` | sha256 of `--daemon-organism-file` EQUAL the pin. Path is operator-resolved (flag 4 / Q9). |
| P3 | `gate_p3` `:1932` | 16-row in-process battery; row 1 is the closer-after-backtick repro. `--apply` sets `run_pinned_tests=True` (the pinned test file as a subprocess). |
| P4 | `gate_p4` `:2037` | Unit inactive, recover timer inactive, no daemon/service/rpc process, no holder of the six files, `daemon.pid` names a dead pid, six files equal the start-of-Phase-2 backup. The log-mtime / "no pulse" leg is inside `if code_placed_at is not None` (`:2063-2068`); `--code-placed-at` is optional (`:2517`). Omitting it drops that check from `checks`, so `ok` can be True without it (C6). |
| P5 | `pin_stamp` `:180`, `check_stamp` `:435`, `load_artifact` `:441`, `stage_rewrite` `:2329` | Stamp is the frozen tuple; load refuses a mismatch; rewrite STOPS if re-derived repair-list/scope-ids sha256 differ. |
| P6 | `hold_snapshot` `:2071`, `gate_p6` `:2089`, `stage_apply.recheck` `:2457` | H1–H3 recorded at backup, compared at apply, repeated immediately before `os.replace`. Conduit `stat` only. |
| P7 | `guard_target` `:343` | Required arg, no default; realpath recorded; HARD refuse Syl / primary checkout / not-the-recorded-CC-dir / daemon-script disagreement. |
| P8 | `main` `:2524`, `stage_apply` `:2401` | `--apply` without `--josh-go` refuses BEFORE `load_pinned`. Go must quote `--josh-go-manifest-sha256` equal to the backup-manifest file. |
| P9 | `stage_apply` `:2437` | Frozen scope size == `--expect-scope` (CLI default 118, overridable — C8); live rule-derived set == frozen list; live repair-list/scope-ids hash == frozen. `--scope-min-len` is a reported cross-check (`scope_ids_obj` `:1578`). |
| P10 | `t6_replay` `:1265` | Tool PRODUCES the stamped would-mint set. Apply prints that S4 waits on it. Tool does not gate on P10. |

Verifier `Verifier.lockstep` `:1661` / `run` `:1763`. V1 `:1772` · V2 `:1780` · V3 `:1785` · V4 `:1786` · V5 `:1790` · V6 `:1791` · V7 `:1793` (`diff_modulo` `:1608`) · V8 `:1801` · V9 `:1812` (no length assertion) · V10 `:1818` · V11 `:1847` (canonical `Graph.restore`) · V12 `:1860` · V13 `:1862` (see flag 9) · V14 `:1867` · V15 `:1873` · V16 `:1876` (count equals mapped rim-incident, not a literal 131) · V17 `:1880` · V18 `:1886` · V19 `:1888`. Classifier `classify_node` `:992` implements A0 (`creation_mode == "conversational"`), A1 (`closer_end = j + len(WANT_CLOSE)` with the imported constant), GENUINE / SEPARATE / ANOMALY / NONE / OVERLAP, `assert_failed` never a bare assert, collision `apply_collision_rule` `:1084` drops AND lists.

**Worker flags 1–12** (return §§5–6). Choice, and which need a ruling:

| # | Choice | This review |
|---|---|---|
| 1 | Stamp carries frozen `branch_head` `c7921b84…`; P1 record carries actual tree HEAD `ae798b94…`; P1 does not require HEAD == `c7921b84`. | **Right.** The assignment's pin worktree is detached at the code commit; requiring HEAD == the docs-only descendant would refuse that tree. Needs a ruling (or re-pin at `c7921b84`). See (2). |
| 2 | Reused `Graph().restore` + `SimpleVectorDB().load` (`load_pair` `:1445` = `analyze_pair.py:83-86`). | **Right.** The named loader is an off-repo argv script, not an importable module. |
| 3 | `--provisional-approve-all` self-approves for the COPY rewrite; packet `PROVISIONAL-SELF-DRY-RUN-NOT-AN-APPROVAL`; `load_approvals` refuses that packet (`:1408`). `allow_provisional` defaults False and is never passed True. `main` routes `--apply` to `stage_apply`, never `stage_rewrite`. | **Right.** Cannot reach a live write. Confirm for TURN B that Phase 1 uses this flag and Phase 2 uses a Chief-relayed approvals hash. |
| 4 | `--daemon-organism-file` operator-resolved (Q9 still open). | **Right.** |
| 5 | P4 "no pulse" = unit inactive AND log mtime ≤ `--code-placed-at`. | Sound mechanical stand-in; it is an interpretation. |
| 6 | H1 flag read from the tool's env; crontab = hash + no uncommented callosum line; conduit path is an argument. | **Right**; conduit path stays `[unverified]` as the plan says. |
| 7 | P3(b) implemented and required at `--apply`; Phase-2 flow tests stub `gate_p3` (`p3_stub` tests `:1405`). | Disclosed test weakening of the Phase-2 path only. P3(a) is tested (`test_p3_the_fingerprint_battery_passes…`, `test_p3_refuses_a_parser_that_regresses…`). |
| 8 | `--expect-wants/--expect-protected/--expect-scope` default 182/183/118; Phase 1 records `matches_expected_count`; Phase 2 STOPs on size mismatch. | **Right.** |
| 9 | V13 proven in the writer (`fidelity_ok` inside `rewrite_main` `:879`); verifier V13 is `self.check("V13", True, …)` `:1862`. | **Right, and a note:** V13 is not an independent second check. A failure never reaches the verifier. |
| 10 | `mention_shape_flags` `:1177` advisory only. | **Right.** |
| 11 | Header RETIRED stamp not written (nothing applied); refusal keys on any `RETIRED-*.receipt` for the directory OR a receipt recording this tool sha256 (`find_retired_receipts` `:2113`). | **Right.** |
| 12 | Real names `main.msgpack.guard_state.json` / `main.msgpack.manifest.json`. | **Right.** |

---

## (2) the stamp

**Verdict: PASS-WITH-NOTES**

`pin_stamp()` `:180-190` is the tuple (file sha256, blob, branch head `c7921b84…`, test-file sha256, scrub version). `write_artifact` `:426` stamps, then `_assert_text_free`, then writes. `check_stamp` `:435` refuses `pin_stamp` inequality. Independent run: every report under the synthetic run dir carried that stamp (`all_reports_stamped: true`); a mutated `branch_head` was refused (`p5_refuses_mutated_stamp: true`). P1 requires the test-file sha EQUAL (`load_pinned` `:228-232`).

Flag 1 handling is sound: the stamp is the frozen constants so a count cannot detach from the pin tuple; the P1 record (`:280`) carries `tree_head` (actual `ae798b94…`) plus `frozen_branch_head`. P1 gates on the three hashes (disk AND at HEAD) and ancestry, not on HEAD == `c7921b84`. **Needs a ruling** (or re-pin the worktree at `c7921b84`) to close the plan-vs-assignment mismatch. Not a FAIL of the tool.

---

## (3) hard refusals

**Verdict: PASS**

`--target-dir` is `required=True` in `build_parser` `:2498` with no default. `guard_target` `:347` refuses an empty string. HARD refusals `:351-356`: under `SYL_CHECKPOINTS` (`~/NeuroGraph/data/checkpoints`), `is_primary_checkout_path`, not equal to `RECORDED_CC_CHECKPOINT_DIR`. Phase 1 writes only under `BACKUPS_ROOT`/`z12-want-text-repair-*` (`guard_out_path` `:364`). `--apply` without `--josh-go` refuses at `main` `:2524` before load.

Independent synthetic run (all paths under `/tmp`; recorded constants patched the same way the tests do; live checkpoint directories never listed or opened):

| Refusal | Result |
|---|---|
| `--apply` without `--josh-go` (pin/target/daemon all `/nonexistent`) | rc 2, "Josh's go" |
| missing `--target-dir` | argparse required |
| target not the recorded CC dir | `Refusal` "not the recorded" |
| target under a patched Syl path | `Refusal` "Syl" |
| target inside a synthetic primary git checkout | `Refusal` "primary checkout" |
| write outside the backups run dir | `Refusal` |
| `--apply --josh-go` without the rest of the Phase-2 inputs | rc 2 |

Phase 1 classify+rewrite on that synthetic directory: rc 0 / 0; target sha256 unchanged after the run (`target_untouched: true`); copy is a new inode (`copy_new_inode: true`).

---

## (4) the rim / `pred_weights` assertion, Choice Clause, id-follows-text, collision, mapping + inverse, idempotent re-run

**Verdict: PASS**

V15 `:1726-1731` / `:1868-1874`: unrepaired wants, both Choice Clause ids, and the constitutional node allow only `pred_weights` key remap of moved ids; any other field, a value change, an added/removed key, or a text edit is `v15_bad`. Deny-check `deny_check` `:1114` STOPS if either Choice Clause id or the constitutional id is in S, the mapping (old or new), or approvals. Presence of those ids in the output walk is not part of V15's `ok` (C9). Deny-check is by id constant, not by metadata (the synthetic tests mint those ids with filler text). V16 `:1746-1755` / `:1876`: rim-incident synapses identical apart from the mapped want-side endpoint; `rim_changed == rim_incident_mapped`. Tests: `test_choice_clause_wants_and_the_rim_are_byte_equal_apart_from_the_key_remap`, `test_v15_fails_for_any_change_other_than_the_key_remap_of_moved_ids` (value / added key / removed keys / other field / unrepaired text), `test_the_rim_synapses_are_repointed_at_the_want_side_only`, `test_v16_fails_when_a_rim_synapse_field_changes`, `test_the_collision_rule_drops_and_lists_both_parties_never_merges`, `test_the_mapping_and_its_inverse_are_written_hash_verified_and_inverse_composes`, `test_an_empty_mapping_makes_a_byte_identical_output_and_the_rewrite_is_idempotent`, `test_separate_takes_the_nested_pair_verbatim_and_the_id_follows_the_text`.

Three inputs of this seat (synthetic filler only, `/tmp`):

1. **Want containing a masked mention pair** (inline-code nested markers inside a live pair). Outcome `GENUINE`, disposition `unchanged`, `t_has_marker` true, id unchanged.
2. **Collision.** Two SEPARATE repairs producing one `new_id`. Both `collision_dropped` / `collision:new_id_produced_by_two_repairs`; `same_new` true; neither written.
3. **Tampered output.** Choice Clause node's `pred_weights` VALUES incremented in the rewrite bytes. Verifier failed `V15` (and `V11` because the harness passed no restore of the tampered file).

---

## (5) text-free artifacts

**Verdict: PASS-WITH-NOTES**

Write paths in the tool: `out_write_bytes` `:378` (guarded), `write_artifact` `:426` (stamp + `_assert_text_free` `:395`), `rewrite_main` `:834` (guarded tmp under the run dir), `write_review_files` `:1345` (the only excerpt path: `<run>/review/*.md`, mode `0600`), Phase-2 live write only via `checkpoint_guardian.atomic_file_write` after gates (`stage_apply` `:2475`). `_TEXTY_KEYS` = `text`, `want_text`, `head`, `content`, `excerpt`, `x_text`, `t_text`, `prose`. Strings longer than 200 characters are refused. Independent run: all JSON reports stamped; tests `test_no_pushed_class_report_carries_want_text_and_excerpts_are_off_repo_0600` and `test_the_report_text_guard_rejects_text_like_keys_and_long_strings`.

Note: a ≤200-character string under a key outside `_TEXTY_KEYS` would pass the guard. Review excerpts are confined to `review/` and are not pushed. Residual, not a FAIL.

`git grep -c '\[WANT\]'` of the three files at `ab24e85c`: tool 25, tests 24, return 0. Hits are marker names, the fingerprint battery, and invented filler. Choice Clause wording in the tests is the filler phrase `synthetic choice clause want one/two`. The two protected ids appear as named constants only.

---

## (6) tests

**Verdict: PASS-WITH-NOTES**

Run once, after this review's analysis, from the tool worktree:

`env -u PYTHONPATH -u NG_EMBED_REMOTE -u NG_EMBED_URL PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider`

**103 passed in 16.86s**, exit 0. P379 preamble as above; `cc_ng_organism` was the PIN copy.

Coverage includes refusals, every outcome class, collision drop-and-list, V1–V19 negatives on tampered output (`_tampered_verify`), approvals refusals (unapproved, struck, hash/binding mismatches, provisional packet), import isolation, P1/P5 stamp mutations, retirement, and the whole Phase-2 path on a synthetic dir with fake probes. Weakening: `p3_stub` (flag 7) skips P3(b) in five Phase-2 flow tests. P3(a) still has its own tests. Worker run 1 had a wrong assertion in the retired-receipt test; that test was fixed at `98564f4` and the tool was unchanged.

---

## (7) hard-link finding and peak-memory unknown (Exec P428)

**Verdict: PASS-WITH-NOTES**

**P428 as given (verified read-only BY INODE by the Executive; this seat did not list or open any checkpoint directory):** live CC `main.msgpack` inode `19413931`, link count 2, hard-linked to `checkpoints/generations/20260923T104614Z/main.msgpack`; `vectors.msgpack` inode `19413942`, hard-linked to `generations/20260923T104614Z/vectors.msgpack`. `last_good/` is not a link partner. The return's "probably `last_good/`" was an inference from the link count. Rule: check inodes; never infer partners from link counts. The tool file itself never names `last_good/` (only the return does).

**Does the tool ever write in place?** No.

- Phase 1 `copy_six` `:2162` uses `shutil.copyfile` onto a new path and STOPS if `os.stat(d).st_ino == before[n]["inode"]` (`:2180`). Independent run: `copy_new_inode: true`.
- Phase 2 live write is `pinned.cg.atomic_file_write` (`checkpoint_guardian.py:343-356`: write to a tmp that preserves the suffix, then `os.replace`). The writer callback copies the staged file to that tmp (`stage_apply.writer` `:2464`). `os.replace` of a new tmp onto a hard-linked live name leaves the generation directory entry holding the old inode and therefore the PRE-repair bytes.
- Independent hard-link simulation under `/tmp` (two names, `os.link`, then `copyfile`+`os.replace`): before nlink 2 and same inode; after, live nlink 1 and a new inode; partner kept the old inode, nlink 1, bytes equal to the pre-replace file. This is the P428 shape. The generation partner is an incidental pre-repair copy, not a rollback source.
- Only `main.msgpack` and the sidecar are replaced at apply (`:2475-2476`). `vectors.msgpack` is not rewritten, so it remains the same inode as its generation partner (still nlink 2, same unrepaired bytes). That is the plan's rewrite set.

**Does it record before/after inodes and link counts?** Partially, and not yet to the P428 receipt bar.

- `copy_six.stat_row` `:2165-2167` records `size`, `mtime_ns`, `inode` (`st_ino`) of the source at copy time. Independent run: `copy_records_inode: true`, `copy_records_nlink: false`. No `st_nlink`.
- `backup-manifest` records sha256, size, mtime_ns. No inode, no nlink.
- Post-apply receipt (`:2482`) records six-file sha256, mapping hashes, tool sha256, packet. No before/after live inode/nlink, no generation-partner inode+sha256.

Plan `[R4c·P428-scope]` (folded after TURN A built the tool; no tool edit requested there): recording those inodes is a TURN B / Phase 2 runbook item and, if the delta pair finds the tool does not already record them, a later small tool follow-up. This review finds that follow-up is still open.

**Rollback / TURN B:**

- Restore only from the tool's own named, hash-verified pre-apply backup, identified by path + sha256 (`backup-manifest-<UTC>.json` and the six files it lists). Never from `generations/` or `last_good/`.
- At apply, record before and after `(st_dev, st_ino, st_nlink)` of the six live files in the post-apply receipt; STOP if a rewritten live file's after inode equals its before inode or its after nlink is not 1. Generation-partner inode+sha256, if recorded, is read-only evidence and is not a restore source.
- Peak memory of the real `SimpleVectorDB.load` of `vectors.msgpack` is unmeasured against the plan's 3 GB classify cap (the plan measured V11 restore ≈ 3.6 GiB). TURN B: run classify under the 6 GB cap, or measure first. The tool stays safe under that unknown because it never writes in place.

---

## (8) no protected/vendored/shared file, no `scripts/`, no raw want text in the diff

**Verdict: PASS**

`git diff --name-only e4ebf982 ab24e85c`:

- `handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py`
- `tests/test_want_text_repair_oneshot.py`
- `handoffs/z12-want-text-repair/returns/build-tool-001.md`

HEAD additionally has this reviews directory (`checker-026` stub + `le-029`). No `scripts/`. Blobs of `neuro_foundation.py`, `cc_ng_organism.py` (this worktree's copy), and the six vendored files equal `e4ebf982` (`neuro_foundation.py` blob `53494b7c56896d25040f3e7fd7c4046da7d0ab05`). The tool lives in `oneshot-tool/` on its own branch. Marker hits in the diff are synthetic. No conversation excerpt.

---

## Numbered corrections

**C1 (NOTE — TURN B / later tool follow-up; P428-scope already names it).** Record before/after `(st_dev, st_ino, st_nlink)` of the six live files in the post-apply receipt, and the generation-partner inode+sha256 read-only. STOP if a rewritten live file is not a new inode with nlink 1. Do not infer partners from link counts. Rollback remains the named backup.

**C2 (NOTE — flag 9).** Verifier V13 is `True` by construction (`:1862`). Encoder fidelity is enforced in the writer. State that in the TURN B receipt so a reader does not treat V13 as a second independent pass.

**C3 (NOTE).** V18 `:1885`: `all(...) and bool(art) or (not m and bool(art))` — because `or` binds looser than `and`, an empty mapping with a non-empty artifacts dict passes V18 even if a listed file's hash fails. The empty-mapping identity-rewrite path is the one this hits. Tighten to require the `all(...)` hash check unconditionally.

**C4 (NOTE — flag 7).** Phase-2 flow tests stub P3(b). A TURN B `--apply` on a copy still has to run the real `gate_p3(..., run_pinned_tests=True)` (the tool already requires it). Do not take the stubbed tests as proof of P3(b).

**C5 (NOTE — flag 1; needs a ruling).** Stamp `branch_head` is the frozen `c7921b84…`; P1 tree HEAD is `ae798b94…`. The split is sound for the assignment's pin worktree. Close it by ruling or by re-pinning the worktree at `c7921b84` with the same organism blob.

**C6 (HIGH).** P4's "no pulse since code placement" check is skippable by omission: `gate_p4` `:2063` only adds `no_pulse_since_code_placement` when `code_placed_at is not None`. `--code-placed-at` is optional. A Phase-2 apply that omits it can pass P4 with a live pulse window untested.

**C7 (HIGH).** Live `Probes` fail-open. `_sysctl` `:1969` returns stdout and ignores returncode; a failed `systemctl is-active` (empty stdout) makes `unit_active` False, so `unit_inactive` is True. `crontab_text` `:2014` returns `""` when `crontab -l` fails, so H1 sees no callosum line and passes. Phase-2 "daemon down" / peer-hold can go green when the probes cannot actually query the host.

**C8 (HIGH).** `--expect-wants/--expect-protected/--expect-scope` are CLI-overridable (`:2507-2509`, defaults 182/183/118). P9 compares live size to `--expect-scope`, not to the plan's literal 118. A run that passes `--expect-scope N` matching a shrunk frozen list satisfies the tool's P9.

**C9 (HIGH).** V15's `ok` is `not W["v15_bad"] and bool(dc.get("clean"))` (`:1873`). `choice_clause_present` is recorded in the detail only. If both Choice Clause ids (and the rim) are absent from the graph, the lockstep never appends them to `v15_bad`, deny-check stays clean (they are not in S/mapping/approvals), and V15 passes. Presence is not asserted.

**C10 (HIGH for apply, NOTE for TURN A).** `stage_apply` `:2475-2476` `os.replace`s main then the sidecar. A death between them is documented as "idempotent re-run from the mapping"; there is no automatic restore of a partial live write. Combined with C1 (no before/after inode record), a torn apply is not mechanically proven. RETIRED is written after both files verify, so a re-run is not blocked by 6.8 — that part matches the plan.

---

## Not-verified

- Real-file behaviour of the value-granular rewrite and V13 on the live 139k-synapse checkpoint (TURN B).
- Peak memory / time under `systemd-run` caps on the real files (TURN B).
- Real `Probes` (`systemctl`, `crontab`, `/proc`) — only fakes were exercised.
- P3(b) subprocess of the pinned ~280 tests.
- Daemon-script cross-check against the real daemon script (synthetic two-line shape only).
- `is_primary_checkout_path` against the real primary checkout (deliberately never referenced).
- Real outcome split / histograms / collisions (Phase 1 on the COPY = TURN B).
- Generation-directory partners' inodes: cited from Exec P428; this seat did not list or open checkpoint directories to re-check them.
- Conduit path (H3) still `[unverified]` as the plan says.
- Q9 unit/slot/retention.

---

## Isolation / closing check

Pin worktree remained detached at `ae798b94`, porcelain empty. Real `~/.bashrc` sha256 unchanged. No live tract open. No Syl path, no `~/NeuroGraph/data/checkpoints`, no primary checkout write. No PR, merge, settle, or dispatch. ROLE B not written. Commit of this file only; push `origin cc-laptop-want-repair-tool-20260930` by name.
