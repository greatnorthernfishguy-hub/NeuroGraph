# checker-024 ROLE A (cross-family) — 118-want text repair plan-004 SHORT RE-PAIR

STATUS: COMPLETE

- Seat: checker-024 (cross-family, grok-4.6, `report_only`). ROLE A only of ADDENDUM 1; ROLE B not written.
- Lane: `z12-s3-restore-bundle-20260929`. Dispatch #11101. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-text-repair-118.md` LAST section ADDENDUM 1 (docs branch `cc-laptop-daemon-recall-756-20260930` @ `89812224fe50814641d3a3892c0899a1c14ed9f4`, not edited). Packet file sha256 `9e0d95a909820157559ca2fca08fed0529e8710be71505d73c7422a832d824bd`.
- Request: `assignments/plan-want-text-repair-118-rev4.md` sha256 `1c32ec8da3a3a6fd6e39d14c93941850239070fa8b79f864f0d692a237fc407f`.
- Plan under review: `handoffs/z12-want-text-repair/returns/plan-004.md` at `13a8fd6beca75f813bccd5df0b99557d2985bed8` (213 lines). Blob `git rev-parse 13a8fd6:handoffs/z12-want-text-repair/returns/plan-004.md` = `684f27ce782ec46199869cc2669e1c79c7dcf0a9`. File sha256 (`git show 13a8fd6:… | sha256sum`) = `f8b36b7919e9f7a805b479db0bdee56f407dc4c16d88f766e99cf608bd503b32` (matches packet).
- Prior pair read in full: `reviews/checker-017-want-repair-plan003.md`, `reviews/le-015-want-repair-plan003.md`. Parser `build-004.md` §§2–3 and `reviews/le-021-810-turn4.md` read in the 810 worktree (read-only).
- Frozen parser: branch `cc-laptop-want-legitimacy-810-20260930`, code `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`, read via `git show` only (810 never checked out into this worktree). Blob `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab`. File sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`. Test-file sha256 `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53`. Review head `c7921b8436fb174c3f70fcf02827f16bb16deff0` (ancestor of current 810 worktree `d8f99c3435e0a3209b02707e1314032391374724`; organism blob unchanged).
- Line cites at `ae798b94:cc_ng_organism.py` (`git show | awk` on `^def`/`^class`/constants): `WANT_OPEN`/`WANT_CLOSE` `:1627-1628`, `WANT_SKIP_REASONS` `:1632` (11 strings), `WantSpan` `:1664`, `SkippedMarker` `:1673`, `WantParse` `:1681`, `want_id_for_text` `:1686`, `parse_wants` `:2084`.
- Code pin for base cites: `e4ebf982b1989fd9066d610b94853bc68bf70d37`. `neuro_foundation.py` blob `53494b7c56896d25040f3e7fd7c4046da7d0ab05` identical to base. This worktree's `cc_ng_organism.py` blob `6f6749c976a1af69c0faa86d18ec6108c330ee0b` identical to base. `git merge-base --is-ancestor e4ebf982 HEAD` holds.
- Derived JSON only (stdlib `json`; no graph/msgpack load): `summary-laptop.json`, `probe-laptop.json`, `want-rows-laptop.json`. Ids, lengths, flags, counts only; `head` never printed.
- Authority: report_only. Plan only. No build, no PR, no merge, no settle, no dispatch. Live tract, `~/NeuroGraph/data/checkpoints`, primary checkout never opened or edited. Secrets by NAME only. No raw want text or conversation excerpt in this file.

## P379 session start

Inherited parent env had `NG_EMBED_REMOTE=hf` and `PYTHONPATH=/home/josh/NeuroGraph:`. All analysis commands ran as `env -u NG_EMBED_REMOTE -u NG_EMBED_URL -u PYTHONPATH`. No NG module was imported (`parse_wants` was `git show` text only; never called).

```
python: /usr/bin/python3
sys.path (after unset PYTHONPATH, cwd = this worktree):
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
cwd: /home/josh/NeuroGraph-worktrees/z12-want-text-repair-20260930
```

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-text-repair-20260930` on `cc-laptop-want-text-repair-20260930`. `git pull --rebase origin cc-laptop-want-text-repair-20260930` was already up to date at plan pin `13a8fd6` before the stub.

Stub first-write: commit `1e1fb2e18f72a1ac2c444d3d7fac6346bd145838` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-text-repair-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

plan-004 applies every Exec P416 item as ruled and every Chief-required G2/G3/scope/excerpts/C1 item. The FINAL PIN table equals the frozen parser values recomputed from `git show ae798b94:cc_ng_organism.py`. The recount split, le-021 (a)–(d) outputs, apply order, and “no dry-run count as fact” rule hold. Commit `13a8fd6` touches only `handoffs/z12-want-text-repair/returns/plan-004.md`; no protected file is in the branch diff vs `e4ebf982`.

One correction before any Phase-1 count is treated as pinned (zone manager / Chief stamp requirement):

**C1 (HIGH for freeze, not for the P416 fold).** P1 checks the loaded function’s sha256 + blob, and P5 stamps the mapping / `repair-list.json` / `scope-ids.json` / approvals with P1–P3 values, but the plan does **not** say that every dry-run count/report artifact is stamped with the pin sha256 + blob + branch head. Histogram, marker-bearing list, residual counts, and outcome split can therefore be detached from the pinned function. Exact one-line fix (add to §4.4 and to §6.1 P5): `Every dry-run count/report artifact (histogram, marker-bearing list, residual counts, repair-list, outcome split) is stamped with the pin file sha256 + blob + branch head.`

Phase 2 remains gated on Josh’s backup + proceed and on Q9 (unit/slot/retention).

---

## (1) Exec P416 items applied as RULED

**Verdict: PASS**

| P416 item | Where | Check |
|---|---|---|
| Q5/G4 peer-hold Phase 2 gate, mechanical, recorded | §0.1 item 4; §6.1 P6; §6.5 H1–H3 | H1: 03:00 crontab hashed before/after and `CC_CALLOSUM_LEG1_ENABLED` not `1`. H2: `cc-callosum-leg2.timer` disabled/inactive, no `leg2-tick`/merge process. H3: `ng_topology/` conduit compared by `stat` only, never opened. Hold released only by the post-track VPS/Leg 1 lane; frozen mapping handed to #802/#803. Any P1–P9 failure = STOP. |
| Q6 frozen-list REQUIRED per id, SIGNED by the Executive; tool REFUSES an unapproved id | §5, §5.2, V17 | Writes only if `decision == approved`, excerpt/`x`/`t` hashes recompute, `function_pin` / `repair_list_sha256` / `scope_ids_sha256` match, and `--approvals-sha256` equals the Chief-relayed file. Else `not_approved` / `approval_mismatch`, node unchanged. Limit stated: the tool cannot authenticate a person; the signature is the relayed hash. |
| Q7 131 rim↔want re-points; rim-side fields unchanged apart from the endpoint id | §3.3, V16, V15 | Want-side endpoint remapped; synapse weights/peaks/ages/counters byte-identical. Rim node/text/flag byte-equal. A rim-node `pred_weights` key remap STOPs (would edit the rim node). |
| C8 one-shot tool RETIRED after apply; no canonical re-key in the protected file | §6.8, §3.4 | ONE-SHOT header + RETIRED stamp; lives only on this feature branch; never merged to `main`; never placed under `scripts/`; `--apply` refuses an existing `RETIRED-<mapping_sha256>.receipt`; archive off-repo. Explicitly no `Graph` re-key in `neuro_foundation.py`. |

---

## (2) Chief-required items

**Verdict: PASS**

| Item | Where | Check |
|---|---|---|
| G2 byte-equality of both Choice Clause wants, the constitutional node, and every unrepaired want; deny-check | §7 V15 | Ids named: `cc:want::7bd0f5fdca6eb404`, `cc:want::3eecfa18710e3b6b`, `constitutional::rim::choice_clause`. Permitted difference on unrepaired wants: `pred_weights` key remap shown by the diff walker. Deny-check: neither Choice Clause id nor the constitutional id is in `S`, the mapping (old or new), or any approval entry. Derived JSON (flags/ids/lens only): both CC ids present, lengths 22 and 8, `prov=cc_authored`, not in S (S ∩ CC = ∅). `7bd0f5fd…` shares a source node with S (collision path is live; the drop-and-list rule is the backstop). |
| G3 rollback per P391/#799 | §6.7, §6.4, V18 | Export every protected node authored after apply before restore; census before/after; host DOWN if export fails; inverse mapping written at apply (V18 before `os.replace`); non-empty export waits on Josh (P398). Pre-S4 path restores the two files under §6.4 mechanical daemon-down. |
| Scope pin to the enumerated 118 | §4.3, P9 | `S` = the COPY-proven id list (`scope-ids.json`). Phase 2 STOPs if live set or size ≠ 118. `--scope-min-len 600` is a reported cross-check only. |
| Off-repo excerpts; no raw want text in any pushed file | §5.3, §8 | Excerpts / LEFT list / text-bearing reports go only to `/home/josh/backups/z12-want-text-repair-<UTC>/review/` mode `0600`. Scrub replaces token-like strings with `[REDACTED:<rule-name>]`. LEFT list reaches Exec as path + sha256. `grep` of plan-004 for `[WANT]` hits marker names, the assignment-named synthetic closer-after-backtick repro, and the `\"[WANT]\"` hug pattern — no derived head and no conversation excerpt. Disclosure of plan-001:105 / plan-003:107 does not quote those fragments. |
| C1 target path / HARD-REFUSE Syl’s checkpoints | §6.3 P7 | Target named: daemon `CHECKPOINT_DIR` `~/.claude/plugins/neurograph/checkpoints` (cite `scripts/cc-ng-daemon.py:588`, docs HEAD `155343e4`). Required argument, no default; `realpath` recorded. HARD REFUSAL (no override): under `~/NeuroGraph/data/checkpoints`, under any primary checkout, or not equal to the recorded CC directory. Phase 1 writes only under `/home/josh/backups/z12-want-text-repair-*`. Phase 2 go: Josh, `--josh-go <reference>` required. |

---

## (3) The pin

**Verdict: PASS-WITH-NOTES**

FINAL PIN table (§1) vs recomputed values (`git rev-parse` / `git show … \| sha256sum` from this worktree; 810 branch never checked out here):

| Field | Plan §1 | Recomputed | Match |
|---|---|---|---|
| Branch | `cc-laptop-want-legitimacy-810-20260930` | 810 worktree branch (read-only) | yes |
| Code commit | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` | `git rev-parse ae798b94…` | yes |
| Blob | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` | `git rev-parse ae798b94:cc_ng_organism.py` | yes |
| File sha256 | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` | `git show ae798b94:cc_ng_organism.py \| sha256sum` | yes |
| Test-file sha256 | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` (plan: not re-hashed by author) | `git show ae798b94:tests/test_cc_want_legitimacy_810.py \| sha256sum` | yes (this seat) |
| Review head | `c7921b8436fb174c3f70fcf02827f16bb16deff0` + le-021 `d8f99c34` | `c7921b8` is an ancestor of 810 HEAD `d8f99c3435e0a3209b02707e1314032391374724`; organism blob at HEAD still `a3aa8a0d` | yes |

P1 (§6.1): loaded `cc_ng_organism.py` sha256 `8ad0f69e…` and blob `a3aa8a0d…`. P5: mapping, `repair-list.json`, `scope-ids.json`, approvals carry P1–P3 values; Phase 2 refuses any difference. Line cites in §1 match the `awk` defs above. This **replaces** `2ab4a851` / `c9fe56d` / `07eeaaaa`.

**Stamp gap (C1), confirmed missing.** The only “stamp” word in plan-004 is the tool’s RETIRED stamp (§6.8). `repair-list.json` carries “the function pin” as a field (§5.1) and P5 stamps four frozen-list files, but **histogram, marker-bearing list, residual counts, and outcome split** are produced in §4.4 / §6.6 without a pin tuple. P1 also does not require the branch head in the loaded-function check. Exact one-line fix: `Every dry-run count/report artifact (histogram, marker-bearing list, residual counts, repair-list, outcome split) is stamped with the pin file sha256 + blob + branch head.`

LOW: P1 “records” the test-file sha256 rather than requiring equality to `04b1a494…`.

---

## (4) Recount split and le-021 requirements

**Verdict: PASS**

§4.4 split is correct. Independent recompute from the three derived JSON files (stdlib `json`; `head` unread):

| Claim | Plan | This seat |
|---|---|---|
| 182 wants, all `cc_authored`, kind `want` | yes | 182 / `{cc_authored}` / `{want}` |
| protected 183 (182 + 1 constitutional) | yes | `summary-laptop.json` `protected_nodes` 183, `constitutional` 1 |
| \|S\| = 118 (`len>600`); 64 shorter | yes | 118 / 64 |
| A 67 / B 16 / C 35 | yes | 67 / 16 / 35; D = 0; 67+16+35 = 118 |
| 123 marker / 91 backtick-led among 182 | yes | 123 / 91 |
| 49 distinct S sources, in-graph, conversational, unprotected | yes | 49, all in `probe` nodes, `creation_mode=conversational`, `constitutional_flag=0`, prefix `cc_gateway`, ids `cc:conv::` |
| 81,999 synapses touch S; 13,203 S↔any-want; 131 rim↔S | yes | 81999 / 13203 / 131 (`constitutional::rim::choice_clause` in nodes, flag 1) |
| 48 hyperedges / 145 memberships (S) | yes | 48 / 145 (50 HEs touch any of the 182) |
| 28 of newest-40 in S | yes | 28 |
| SEPARATE ≤ 102; GENUINE ≤ 118; no derived lower bound | yes | class B has no nested marker so cannot be SEPARATE; A+C = 102 is the derived upper bound; split itself not computable without `parse_wants(C)` |

What can ONLY come from the Phase-1 dry run on a COPY with the pinned function is named and not quoted: GENUINE / SEPARATE / NONE / OVERLAP / ANOMALY split, collisions, per-reason histogram, marker-bearing list, residual-class counts, S2/S4/S6–S12 occurrence counts.

le-021 / FINAL PIN (a)–(d) are present in §4.4:

- **(a)** per-reason histogram over all 11 reasons (`WANT_SKIP_REASONS` `:1632`: `in_fence`, `in_code_span`, `code_adjacent`, `escaped`, `quoted`, `in_json_string`, `in_url`, `in_link_target`, `opener_unclosed`, `closer_without_opener`, `empty_pair`), twice (49 S sources and every conversational node whose content contains a marker); `in_url` / `in_json_string` / `in_link_target` hand-reviewed.
- **(b)** marker-bearing minted-want list (F3 nested mention pairs + N1); each id to the frozen-list review. OVERLAP-ANOMALY added in §4.2 step 4 for N1.
- **(c)** named residual classes that still DROP a well-formed want (build-004 §3 items 1–7, including N2’s wider `"key":` wording and N3’s `quoted` note) listed with counts, **not fixed**.
- **(d)** “still MINTS a mention” classes as frozen-list material.

Plan-003 §3.4–3.5 assumptions that the FINAL reasons / closer fix could change are listed as dry-run checks, not assertions (JSON-first labelling, SEPARATE may fall, closer-`code_adjacent` NONE route gone, F3/N1, F4a). Tooling: reuse analysis-001 loader; one load; `MemAvailable ≥ ~8 GB`.

---

## (5) Apply order

**Verdict: PASS**

§0.1 item 3: #810 is fixed, paired, **and the code the laptop daemon runs**, then #801 is applied live (daemon down, full backup), then S4. §6.1 P2 defines “deployed” as code placed with the daemon **not** restarted (or proof no pulse ran). §6.3 / §10: Phase 2 needs Josh’s backup + proceed (`--josh-go`); `--apply` refused without it. P2’s path still waits on Q9 (unit/slot) — carried, not a missing order statement.

---

## (6) Plan quotes NO dry-run count as fact

**Verdict: PASS**

§4.4 and §11: “No dry-run count is quoted anywhere before it is taken on the pinned function on a COPY (and none is quoted here).” Grep of GENUINE/SEPARATE/NONE against digits hits only derived bounds (SEPARATE ≤ 102, GENUINE ≤ 118, 11 reasons) and qualitative “may fall / more nodes may end NONE”. No realized split, histogram bucket, residual count, or collision count is stated as fact. V11 “restore ≈ 3.6 GiB” is a prior RSS measurement, not a parse-outcome count.

---

## (7) Nothing in the diff touches a protected file (plan only)

**Verdict: PASS**

`git show --name-only --format= 13a8fd6` = `handoffs/z12-want-text-repair/returns/plan-004.md` only. `git diff --name-only e4ebf982 13a8fd6 | grep -v '^handoffs/'` is empty. `neuro_foundation.py` and this worktree’s `cc_ng_organism.py` are byte-identical to `e4ebf982`. Protected-file mentions in the plan are read-only cites and the C8 “do not add a re-key” sentence.

---

## Numbered corrections

1. **HIGH for freeze (C1).** Add to §4.4 and §6.1 P5: `Every dry-run count/report artifact (histogram, marker-bearing list, residual counts, repair-list, outcome split) is stamped with the pin file sha256 + blob + branch head.` Until that lands, a Phase-1 count is not unambiguously tied to the frozen function. Does not undo the P416 / Chief-required fold.

2. **LOW.** P1 records the test-file sha256 rather than requiring equality to `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53`.

---

## Not verified

- Realized GENUINE / SEPARATE / NONE / OVERLAP / ANOMALY split, collisions, skip-reason histogram, marker-bearing list, residual counts (need Phase-1 on a COPY with the pinned function and full `C`; forbidden here).
- S2 `pred_weights` node count 1,735; S4 `expected_target` hits; S6–S12 occurrence counts; S13 “all 182 sidecar keys present” (packet files have no sidecar key set).
- Whether the live laptop checkpoint still equals the 09-23 copy.
- Installed unit files, crontab, autosave interval, daemon import path (Q9). `scripts/systemd/` repo copies were not re-read this turn.
- Runtime behaviour of `parse_wants` (paper/line cites of `ae798b94` only; no import, no pytest).
- Two of 60 source nodes among the full 182 are absent from `probe` nodes; all 49 S sources are present. Outside ADDENDUM 1 except as a leftover of the 182.

No graph, checkpoint, or live tract was opened. Primary checkout untouched. Only this reviews file is in scope to commit.

STATUS: COMPLETE
