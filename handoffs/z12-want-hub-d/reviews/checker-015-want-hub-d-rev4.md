# checker-015 ROLE A (cross-family) — want-hub-competition-d plan REVISION 4

STATUS: COMPLETE

- Seat: checker-015 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `want-hub-competition-d`. Dispatch #10735. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-d-rev2.md` including ADDENDUM and ADDENDUM 2 (docs branch `cc-laptop-daemon-recall-756-20260930`, not edited). Also read `assignments/plan-want-hub-d-rev4.md`.
- Plan: `handoffs/z12-want-hub-d/returns/plan-004.md` on `cc-laptop-want-hub-d-20260930` at `618980ea5536232c0255c8cbfc3e9e685e5164fc` (604 lines).
- Plan sha256 (`sha256sum`): `9ff61ac3ba3d24d7d1979b94663897db730110049d2a71bd4032924f9f36f2e8` (matches packet).
- Inputs folded: `reviews/le-012-want-hub-d-rev3.md` (C1–C11) and Exec Packets 397/399/404 as transcribed in the packet addenda and the rev-4 assignment. Packets themselves were not opened as primary documents.
- Code pin: `e4ebf982b1989fd9066d610b94853bc68bf70d37`. `neuro_foundation.py` read only via `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:neuro_foundation.py`. Never edited. `git diff e4ebf982..HEAD -- neuro_foundation.py` is empty.
- Derived JSON only: `summary-laptop.json`, `probe-laptop.json` under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`. No graph, msgpack, checkpoint, or tract load. Want text / VDB prefixes not printed.
- Authority: report_only. No build, no PR, no merge, no settle, no dispatch. Primary `/home/josh/NeuroGraph` not edited.

## P379 session start

```
python: /usr/bin/python3
sys.path[0:8]:
  ''
  /home/josh/NeuroGraph
  /home/josh
  /usr/lib/python312.zip
  /usr/lib/python3.12
  /usr/lib/python3.12/lib-dynload
  /home/josh/.local/lib/python3.12/site-packages
  /usr/local/lib/python3.12/dist-packages
NG-related in sys.modules: NONE
CC_NG_PYTHONPATH: <unset>
PYTHONPATH: /home/josh/NeuroGraph:
NG_EMBED_*: none set
cwd: /home/josh
```

`PYTHONPATH` would resolve every NG import to the **primary** checkout `/home/josh/NeuroGraph`. No NG module was imported (no `neuro_foundation`, `checkpoint_guardian`, `openclaw_hook`, `cc_ng_organism`, `neurograph_rpc`). The JSON recompute used stdlib `json` only. Guardian ratios were recomputed as arithmetic; `evaluate_save_health` was **not** called.

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930` at plan pin `618980e`; `git pull --rebase origin cc-laptop-want-hub-d-20260930` was up to date before the stub and before this complete file.

Stub first-write: commit `f604e40098e59a6b0b9e1b274130497c5968279a` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-hub-d-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

plan-004 is a faithful fold of le-012 C1–C11 and of P399/P404 as transcribed. The additive surface is implementable on the real `_prune_synapses` at `e4ebf982` with the default path behaviour-identical. The K=50+50 / B=5,000 cycle table and the worst-cycle guardian arithmetic still hold under a static key. Three numbered corrections, none HIGH. Do not treat the author's **height** formula (X7) as ruled.

---

## (1) every le-012 correction C1–C11 is really applied

**Verdict: PASS-WITH-NOTES**

Checked against plan-004 **body text**, not the §11 table. The table is accurate as a map; the notes below are the text-level residues.

| # | Sev | Applied in the text? | Where |
|---|---|---|---|
| C1 | HIGH | Yes | TOP NOTICE mutation paragraph; §4.4 (one call per cycle, no suppressing branch); §7 LAW 4 + R7; test A; dry-run item 8. plan-003's "does not advance" survives only as a quoted falsehood being corrected. |
| C2 | HIGH | Yes | Q-C **RESOLVED = (i)** in the TOP NOTICE and §0.3 / §4.1 / §10. §4.3 forbids predicate copy, `_remove_synapse_internal`, and a removal loop in the orchestrator. (ii)/(iii) appear only as REJECTED / "gone". §11's "appears nowhere" overclaims: rejection mentions remain, which is the correct residue. |
| C3 | HIGH | Yes | §2.6 test G: two checkouts (not an in-test copy), full post-state field list, both Door A (`:3495`) and Door B (`:2891`) callers, boundary seeds, all-defaults case, PYTHONPATH pinning + printed `__file__` and git rev, green on both revisions. Dry-run item 4 repeats it. |
| C4 | HIGH | Yes | §4.2 (a)–(h): kw-only default-`None`; competing set as ids; loop iterates only `competing_ids`; validation before the loop; explicit `raise` (`ValueError`); assertion set; test K includes `python -O`. |
| C5 | HIGH→ruled | Yes, with X7 flagged | §4.2(e)–(g), §4A.2: `max_removals` truncates the function's own `to_prune` after a static caller-supplied `order_key`; caller owns the INFO record from `removed_ids`. Dynamic greedy declared not implementable. Height vs naive vs greedy table is in the text. |
| C6 | HIGH→ruled | Yes | §3 rewritten: env → call arguments; nothing in `graph.config` / `DEFAULT_CONFIG` / `OPENCLAW_SNN_CONFIG` / `CC_SNN_CONFIG`; OFF at every restart by construction; plan-003 `CC_SNN_CONFIG` invariant withdrawn. |
| C7 | MED | Yes | §4.5: own try/except, fail-closed, WARNING+, `last_pass` still updated; daemon pinned by commit (`039a3bf4…` at `:2156-2202`, `cdcf8ce2…` at `:2343`). |
| C8 | MED | Yes | §4.6 / §9: NG engine first (P329), then daemon slice on its own docs branch; no `hasattr` shim; unset env ⇒ skip + INFO "not armed". |
| C9 | MED | Yes | §4.1 table and §8.2: default-path *behaviour* identical, source of `_prune_synapses` IS edited. §2.6.1 now says `_prune_synapses` IS edited. §3.2.4 / §6 / §7 Syl's Law re-scoped to checkpoint *content* unchanged when OFF, guarded by test G. |
| C10 | MED | Yes | §7 LAW 8 row: input-clock dependency, #117 family, no-widening if §4B fails; §8 punchlist item 5. |
| C11 | LOW | Yes, per the rev-4 assignment | Changelog SUPERSEDED banner (assignment: do not edit plan-001..003). §9.1 drops the `:78/:162/:194/:3286` comment corrections. C11's "§8 item 1 wording" landed as §8 item 2 after the checklist was rewritten. |

**Note (not a miss of C1's substance):** TOP NOTICE and the ADDENDUM 2 prompt still cite the mutation as `:3524-3529`. The reset assignment is at `:3530`. See item (4). plan-004 §11 itself says the author re-read `:3521-3530`.

---

## (2) NO contradiction of P399/P404

**Verdict: PASS-WITH-NOTES**

Against the packet addenda (P399) and `assignments/plan-want-hub-d-rev4.md` (P404 C5/C6):

| Ruling | Plan-004 |
|---|---|
| Q-C option **(i)** only; (ii)/(iii) REJECTED | Held. No predicate helper, no second removal loop, no orchestrator copy of the three predicates (§4.3). |
| Additive kw-only default-`None` | `competing_ids`, `excluded_ids`, `max_removals`, `order_key`, `report` (§4.2). Names illustrative, surface matches P404 C5. |
| Competing set passed as ids, NEVER inferred | Caller builds F ∪ G ∪ last-link as `excluded_ids`; function does not call `_is_identity_protected` to decide membership (§4.2(c)). |
| Function asserts none of F/G/last-link/constitutional appear | Explicit `raise` (C4 refinement of P399's "ASSERTS", so it survives `python -O`); validation **before** the loop (§4.2(d), §4A.4). |
| Eligibility + removal stay inside the one function | Held. Orchestrator builds sets/key/INFO only. |
| Q-A: same criteria as they behave at runtime; **no weight path** | Held. No new predicate; no suppressing branch on `low_weight_steps` (C1). |
| Q-B: Door B liveness a hard S4 gate | §4B unchanged as a hard gate; idle-window requirement added (X10 / Chief LE-9). |
| P399 cond. 1 golden test | §2.6 G, including P404 C5 all-defaults case. |
| P399 cond. 3 NG-first / P329 | §4.6, §9. |
| P399 cond. 4 OFF = checkpoint byte-identical, **NO new config keys** | §3: K/B never enter `CC_SNN_CONFIG` / `DEFAULT_CONFIG` / `OPENCLAW_SNN_CONFIG`. `openclaw_hook.py:861` `dict.update` hazard cannot arise because no key is stored. |
| P404 C6: K and B env-read and passed as **call arguments** | §3.1 / §4A.1. Unset / non-int / `< 1` ⇒ skip + INFO "not armed". |
| P404 C5: static sort key on the function's own `to_prune` **before** truncation; removed-ids out-param; caller owns budget policy and INFO | §4.2(e)–(g), §4A.2, §4A.6. Return stays `int`. |

**X7 is a note, not a contradiction.** P404 names the ingredients (competing-degree, `inactive_steps`, `weight`, `synapse_id`) and asks whether a static key preserves tallest-first intent. The plan answers: a **height** key (`c_w − r_w`) does; a naive competing-degree-desc key does not (one want loses 3,052 links in cycle 1). That formula is the author's and is flagged for Executive confirmation. BUILD of the *surface* can proceed; BUILD of that *formula* waits on X7.

X8 (`report['eligible']`) is an extra field on the out-param, droppable; P404 named removed ids.

---

## (3) implementability against the real `_prune_synapses` at `e4ebf982`

**Verdict: PASS-WITH-NOTES**

Read at pin via `git show` (line numbers from `nl -ba`):

```
3500    def _prune_synapses(self) -> int:
3513        to_prune: List[str] = []
3514        for sid, syn in self.synapses.items():
3517            if (self._is_identity_protected(syn.pre_node_id) or
3518                    self._is_identity_protected(syn.post_node_id)):
3519                continue
3524            if syn.weight < wt:
3525                syn.low_weight_steps += 1
3526                if syn.low_weight_steps > grace:
3527                    to_prune.append(sid)
3528                    continue
3529            else:
3530                syn.low_weight_steps = 0
3535            if syn.inactive_steps > effective_inactivity:
3540            if age > grace and syn.peak_weight < 2.0 * initial_w:
3543        for sid in to_prune:
3544            self._remove_synapse_internal(sid)
3546        if to_prune:
3547            self._emit("pruned", count=len(to_prune), timestep=self.timestep)
3549        return len(to_prune)
```

Callers at pin, both no-arg: Door A `_structural_plasticity` `:3495`; Door B tail `:2891`. `git grep _prune_synapses` on tests at this pin: **0 files**.

**Default path can stay behaviour-identical.** Keyword-only default-`None` parameters do not change `_prune_synapses()`. With all new params `None`: same `self.synapses.items()` iteration, same `:3515-3519` `continue` before any counter, same three predicates, same full `to_prune` removal, same `pruned` event, same `int` return. That is the all-defaults / both-callers golden case (P399 cond. 1 + P404 C5).

**Static sort + `max_removals` + removed-ids out-param fit the existing `to_prune` list.** The function already accumulates ids, then removes in a second loop. Additive, after the collection loop and only when the new params are not `None`:

- capture `eligible = len(to_prune)`
- `to_prune.sort(key=lambda sid: order_key[sid])` if `order_key` is given
- `to_prune = to_prune[:max_removals]` if `max_removals` is given
- fill `report['removed_ids']` / `report['eligible']` if `report` is a dict
- then the existing removal / emit / `return len(to_prune)`

An **unconditional** sort or slice would change the default path (today's removal order is dict-iteration order, and every eligible id is removed). The plan's "all defaults ⇒ today's function" requires the sort/slice/report to be gated on the new params. Competing mode should also **require** `max_removals >= 1` (correction C2 below): as written, §4.2(d) treats `max_removals` as optional ("if given"), and omitting it on a competing call is the one-pass cliff.

**Competing mode is a second iterator, not a patch inside the identity `continue`.** If the loop still walked the whole table, ordinary synapses would be evaluated outside B and rim/G counters could move (C4). The plan is right: iterate **only** `competing_ids`, and do **not** apply the `:3515-3519` `continue` to those ids (otherwise every competitor is still exempt and the pass removes nothing). Validation runs first so a `raise` mutates nothing. `_step_lock` is `threading.RLock` (`:1751`), so the daemon already holding it (§4.5) plus an orchestrator acquire (§4.3) does not deadlock; document that rather than taking a second lock "because the plan said so".

Native `SynapseStore` (`:1618-1624`) is Mapping-shaped and today's loop already uses `items()` / `SynapseRef` write-through. Competing-mode `self.synapses[sid]` and `syn.low_weight_steps += 1` ride that same contract. The store itself is in the plan's UNVERIFIED list; this review did not load it.

---

## (4) C1 statement of the real `low_weight_steps` mutation at `:3524-3529`

**Verdict: PASS-WITH-NOTES**

The **mutation statement is TRUE** of the code at pin:

- if `syn.weight < wt`: `syn.low_weight_steps += 1` (`:3525`); if then `> grace`, append and `continue` (`:3526-3528`)
- else: `syn.low_weight_steps = 0` (`:3530`)

This runs on every synapse the loop evaluates, **after** the identity `continue` (`:3517-3519`). Wake-time protected synapses never reach it. Under option (i) competing mode they will, once per call, eligible or not. Dormancy of the *weight criterion* is then `increments ≤ 22 ≪ grace_period 5,000`, not "the pass does no bookkeeping". No suppressing branch belongs here.

The **cited range `:3524-3529` is short by one line.** `:3529` is the `else:`; the reset is `:3530`. le-012 C1 and ADDENDUM 2 item (4) repeat `:3524-3529`. plan-004 TOP NOTICE copies that range; §11's "re-read `:3521-3530`" is the accurate window.

`low_weight_steps > 0` is 913 graph-wide (`summary-laptop.json`). The competing-set **maximum** is still unknown (probe has no `low_weight_steps`). Dry-run item 8 / TOP NOTICE correctly make that a pre-arming read.

---

## (5) numbers — static key vs K=50+50 / B=5,000 cycle table

**Verdict: PASS**

Independent recompute from **one JSON at a time** (stdlib `json`; no NG import). Probe synapses are `[pre, post, weight, peak, creation_time]` (5 fields). Ranking used probe index as the `synapse_id` stand-in. Node ids only (`constitutional::rim::choice_clause`, `cc:want::` prefix); no want text.

**From `summary-laptop.json`:** nodes 7,253; synapses 138,753; hyperedges 517; timestep 33,637; protected nodes 183 (1 constitutional + 182 `cc_authored`); protected-touching 128,359; `inactive_gt_threshold(x salience)` **E = 116,164**; `low_weight_steps_gt_0` = 913; `last_spike_time_gt_timestep` = 475; saved `weight_threshold 0.01`, `grace_period 5000`, `inactivity_threshold 1000`, `initial_sprouting_weight 0.1`, `sprout_degree_cap 100`, `tonic_ages_substrate 1`.

**From `probe-laptop.json` at K=50 per direction, rank `(−weight, −peak, index)`:**

| quantity | plan-004 | this recompute |
|---|---|---|
| F (rim-touching) | 4,127 | 4,127 |
| want-touching | 124,437 | 124,437 |
| arena (want-touching non-F) | 124,232 | 124,232 |
| rim-only / rim↔want | 3,922 / 205 | 3,922 / 205 |
| unprotected synapses | 10,394 | 10,394 |
| unique G kept | 17,391 | 17,391 |
| competing (before last-link) | 106,841 | 106,841 |
| lists ≤ K out / in | 8 / 8 | 8 / 8 |
| K-th silent out / in | 137 / 174 | 137 / 174 |
| level-1 ties out / in | 12 / 65 | 12 / 65 |
| level-2 residual | 7 / 8 | 7 / 8 |
| arena `weight == 0.0` | 20,381 | 20,381 |
| age-eligible on competing | 51,856 | 51,856 |
| inactivity bracket | 94,646 – 106,841 | 94,646 – 106,841 |
| last-link partners / orphanable | 16 / 9 | 16 / 9 |
| competing after last-link | **106,825** | **106,825** |
| eligible after last-link | **94,630 – 106,825** | **94,630 – 106,825** |
| cycles | 19 – 22 | 19 – 22 |
| last cycle (upper) | 1,825 → live 31,928 | 1,825 → 31,928 |
| last cycle (lower) | 4,630 → live 44,123 | 4,630 → 44,123 |

Inactivity bracket formula confirmed: `lo = max(0, E − (128,359 − |competing|))`, `hi = min(|competing|, E)`.

**The static key does not change the cycle table.** Each cycle still removes `min(B, remaining eligible)`. Per-cycle live/reference (ref updated after each permitted save; synapse gate 0.5 at `checkpoint_guardian.py:275-283`):

| cycle | live after | live/ref | margin over 50% |
|---|---|---|---|
| 1 | 133,753 | 96.396% | +46.40 pts |
| … | … | … | … |
| **21 (worst)** | **33,753** | **87.098%** | **+37.10 pts** (drop 12.90%) |
| 22 | 31,928 | 94.593% | +44.59 pts |

Lower bound worst = cycle 19 at 44,123 / 48,753 = **90.503%**. Every cycle `live >= 0.5 * ref`. Node-loss gate (`:261-271`) does not fire if nodes are unchanged (`wires_own_deposits=False` refuses only `live_nodes < ref_nodes`). One-pass: 31,928 (23.0%) or 44,123 (31.8%) of 138,753 — refused. Adverse bound cycle 21 plus all 10,394 unprotected: 23,359 / 38,753 = 60.3%.

Plan-004's 1-decimal rounding (96.4%, 87.1%, +37.1 pts, 12.9%, 94.6%, 90.5%) matches.

**Not recomputed, and why:**

- Height-key vs greedy p50/p90/max table (§4A.2): probe has **no `inactive_steps`**; the author's table used a weakest-weight proxy. Cycle *counts* do not depend on it. Trajectory of want-degree under the height formula is left to the dry run (plan item 9).
- Real `evaluate_save_health` return strings: not called (would import an NG module via `PYTHONPATH=/home/josh/NeuroGraph:`). The 50% synapse-gate arithmetic is the load-bearing part and was recomputed.
- Stored `low_weight_steps` maximum on the competing set: not in the probe.
- Un-rounded engine weights / real `synapse_id` tie-break.

---

## (6) the dream-loop change

**Verdict: PASS**

Pinned slices re-read via `git show` (not the live primary working tree as an editor):

- docs `039a3bf4f39da8a2024b65724e31509e69f3119c` `scripts/cc-ng-daemon.py`: `_DREAM_ENABLED` `:1963`, idle/interval `:1964-1965`, `_dream_loop` `:2156-2202`, start `:2440-2441`.
- `cdcf8ce26f2e8c375384866262e03c2965d2c051`: `_dream_loop` `:2343-2389`, start `:2627-2628`. Same control flow; lines moved (parallel-edit residue, R15).

At both pins:

- `last_pass = time.time()` is set at boot (`:2163` / `:2350`) and **only on the success path** after consolidate + seam-split (`:2196` / `:2383`).
- `except Exception` (`:2200-2201` / `:2387-2388`) logs a WARNING and **does not** update `last_pass`. A raising competition call in that same `try` would re-run `consolidate_hyperedges` every 60 s tick. C7's diagnosis is true of the code.
- The plan's hook is after `dedup_and_split_oversized_hyperedges`, inside the existing `with lock:` on `graph._step_lock` (`RLock` `:1751`): unset/invalid env ⇒ skip + INFO `"not armed"`; armed ⇒ orchestrator call in **its own** try/except, fail-closed, WARNING+, then fall through so `last_pass` still updates. Test D-2 specifies that. No `hasattr` shim (§4.6).

This is implementable on the existing loop with no new thread. Env constants next to `_DREAM_*` match `_DREAM_ENABLED`'s import-time pattern (arming = planned restart).

---

## (7) anything that fails if built as written

**Verdict: PASS-WITH-NOTES** (no HIGH; three numbered corrections)

What holds if built as written:

- Wake-time callers keep today's behaviour, **if** sort/slice/report are gated on non-`None` params and competing mode is a separate iterator (item 3).
- Rim / G / last-link stay out of `competing_ids`; function `raise`s if they appear; `_collect_orphan_nodes` still skips `_is_identity_protected` (`:3602`).
- K/B absent ⇒ OFF at every restart; no new checkpoint config keys.
- Staged B=5,000 stays above the 50% synapse gate on these numbers; last-link keeps the node-loss gate quiet.
- Dream-loop isolation prevents the 60 s consolidate storm.

What would fail or silently recreate a rejected shape:

1. **Competing call without `max_removals`** — §4.2(d) currently allows it; the function would remove every eligible competitor in one pass (the cliff Exec 397 removed). The orchestrator passes B, but the function contract should refuse competing mode without a budget.
2. **Naive `order_key` (competing-degree descending)** — implementable from P404's ingredient list, and the plan's own table shows it drains one want of 3,052 links in cycle 1. The **height** formula is the author's (X7) and is not yet an Executive ruling. Building a key without X7 confirmation picks a policy the pair has not been given.
3. **Unconditional `to_prune.sort` / slice on the default path** — would change Door A/B removal order or count; test G would (correctly) fail. The plan's prose says all-defaults ≡ today; the BUILD must keep that branch.

Not failures of this plan, already disclosed: Door B liveness UNVERIFIED (§4B hard gate); competing-set `low_weight_steps` max unknown; native store; PYTHONPATH vacuous golden test if `sys.path[0]` is not pinned; daemon `_dream_loop` under parallel edit (rebase + re-pin).

---

## Numbered corrections

**C1 — LOW.** Cite the `low_weight_steps` mutation as `:3524-3530` (reset is `:3530`). The statement is true; the ADDENDUM 2 / TOP NOTICE range `:3524-3529` is not.

**C2 — MEDIUM.** In competing mode (`competing_ids is not None`) require `max_removals` as an `int >= 1` in the pre-loop `raise` set. Optional `max_removals` on the additive surface recreates the one-pass cliff if a caller omits it. Default path keeps `max_removals is None` (no truncation).

**C3 — MEDIUM.** Do not treat the §4A.2 **height** formula as ruled. P404 C5 ruled the *surface* (static caller-supplied key on `to_prune` before truncation). X7 is still an Executive confirmation. The naive competing-degree key is a legal reading of the named ingredients and does **not** preserve tallest-first intent.

No HIGH. No FAIL item.

---

## Numbered not-verified

1. Executive Packets 388/392/397/399/404 as primary documents (assignment + packet addenda + le-012 only).
2. Door B liveness on a running laptop daemon (§4B; extra code in §1.5 / [R3·A6] suggests it may be dormant).
3. Test G / dry run / armed-path counter test — designed, not executed.
4. Real-engine ranking (un-rounded weights, real `synapse_id`, real `inactive_steps` in the height key).
5. Stored `low_weight_steps` maximum on the competing set (dormancy premise of the weight criterion).
6. Daemon process environment for `NG_GUARDIAN_*` (simulation uses code defaults; `.bashrc` sets none).
7. Idle-window evidence path (X10).
8. Native/Rust `SynapseStore` behaviour under a competing-ids iterator.
9. `cleanup_cc_tool_noise.py`; current S3/S4 plan text; other docs-branch edits to `_dream_loop` beyond the two pins.
10. Height-key want-degree trajectory with real `inactive_steps` (probe lacks the field).

---

## Verdict per ADDENDUM 2 item

| Item | Verdict |
|---|---|
| (1) C1–C11 applied in the text | PASS-WITH-NOTES |
| (2) no P399/P404 contradiction | PASS-WITH-NOTES (X7 unruled formula) |
| (3) implementability vs `_prune_synapses` at `e4ebf982` | PASS-WITH-NOTES |
| (4) C1 `low_weight_steps` mutation | PASS-WITH-NOTES (true; range `:3524-3530`) |
| (5) numbers / cycle table under static key | PASS |
| (6) dream-loop isolation / unset ⇒ skip + INFO | PASS |
| (7) fails if built as written | PASS-WITH-NOTES |
| **Overall** | **PASS-WITH-NOTES** |

ROLE B is a separate turn. Nothing built, merged, armed, settled, or dispatched.
