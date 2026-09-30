# checker-021 ROLE A (cross-family) — SHORT re-check of plan-005 after the [R5c] fold

STATUS: COMPLETE

- Seat: checker-021 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `want-hub-competition-d`. Dispatch #10996. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-d-rev2.md` including ADDENDUM 4 (docs worktree branch `cc-laptop-daemon-recall-756-20260930`, not edited). Also read `assignments/plan-want-hub-d-p418.md` and `assignments/plan-want-hub-d-p415.md` (ruling text). Packets themselves were not opened as primary documents.
- Plan: `handoffs/z12-want-hub-d/returns/plan-005.md` on `cc-laptop-want-hub-d-20260930` at `1800d28ad6f82f8f503e45ef787041b52cdf4df1` (fold `25ae7a2a72346db0091c3a2d0d25b48e15cd0821` + pin-line `1800d28`). 784 lines.
- Plan sha256 (`sha256sum`): `e556874d07928ad74cd9dc736fec2ef168d8cb17cdcb0bcbbb8b082f6286e60c` (matches packet).
- Inputs: `reviews/le-018-want-hub-d-rev5.md` (N1–N8), `reviews/checker-020-want-hub-d-rev5.md` (N1/N2), Exec P418 (docs 5285f260 as transcribed).
- Code pin: `e4ebf982b1989fd9066d610b94853bc68bf70d37`. `neuro_foundation.py` read only via `git show`. Never edited. `git diff e4ebf982 HEAD -- neuro_foundation.py` is empty.
- Daemon: `git -C /home/josh/docs show 039a3bf4:scripts/cc-ng-daemon.py` (`039a3bf4f39da8a2024b65724e31509e69f3119c`), `_guarded_save` `:970-1011`, autosave caller `:1902`.
- Derived JSON only: `summary-laptop.json`, `probe-laptop.json` under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`. No graph, msgpack, checkpoint, or tract load. Want text not printed. Node ids only.
- Authority: report_only. No build, no PR, no merge, no settle, no dispatch. Primary `/home/josh/NeuroGraph` not edited. ROLE B is a separate turn.

## P379 session start

Targeted runs used `env -u PYTHONPATH -u NG_EMBED_REMOTE -u NG_EMBED_MODEL`. No NG module was imported (no `neuro_foundation`, `checkpoint_guardian`, `openclaw_hook`, `cc_ng_organism`, `neurograph_rpc`, `ng_embed`, `ng_lite`). Guardian ratios were recomputed as arithmetic; `evaluate_save_health` was **not** called. JSON recompute used stdlib `json` only.

```
python: /usr/bin/python3
sys.path[0:7]:
  ''
  /usr/lib/python312.zip
  /usr/lib/python3.12
  /usr/lib/python3.12/lib-dynload
  /home/josh/.local/lib/python3.12/site-packages
  /usr/local/lib/python3.12/dist-packages
  /usr/lib/python3/dist-packages
NG-related in sys.modules: NONE
PYTHONPATH: unset in the run
NG_EMBED_*: unset in the run
cwd: /home/josh
neuro_foundation in sys.modules: False
cc_ng_organism in sys.modules: False
ng_embed in sys.modules: False
ng_lite in sys.modules: False
```

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930` at plan pin `1800d28`; `git pull --rebase origin cc-laptop-want-hub-d-20260930` was up to date before the stub and before this complete file.

Stub first-write: commit `6810a34513f4efa520f65f0cbe4fa98bb2ca1dca` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-hub-d-20260930`.

---

## Overall verdict

**PASS**

The [R5c] fold of le-018 N1–N8, checker-020 N1/N2, and Exec P418 is present in the TEXT of the changed sections. Both census sentences match the assignment files byte-for-byte. The daemon's two `_guarded_save` refusal paths at `039a3bf4` `:970-1011` are described accurately; #824 is an ARMING precondition and the plan states that until it lands the operator-level check is the only stop. End-state p50/p90/max **122 / 151 / 191**, the **1,662** residual, and **−82 / −90 / −94%** recompute from the derived JSON. No HIGH. One LOW note on §10 map-row lag (operative sections are current). No contradiction of P399/P404/P409/P412/P415/P418 that would fail a build as written.

---

## (1) N1 / #824 — automatic-refusal text vs `_guarded_save`

**Verdict: PASS**

Re-read `git -C /home/josh/docs show 039a3bf4:scripts/cc-ng-daemon.py` at `_guarded_save` `:970-1011` and the autosave caller `_guarded_save("autosave")` `:1902`.

**Both refusal paths, as the plan states at §4A.4:**

| Path | Code | Plan's description | Accurate? |
|---|---|---|---|
| **Outer** node-only guard | `:987-1000`: `_SAVE_GUARD_ENABLED and _save_would_collapse(current, ref)` → `logger.critical(...)`; `with_receipt` returns a failed receipt, else **`return False`**. No write to `quarantine/` and no `last_permit_ts`. | Outer refusal writes **no quarantine file and no `last_permit_ts`**, so signal (1) cannot see it. | **Yes.** |
| **Inner** SaveGate, non-receipt | `:1006-1011`: `STATE.ng.save()` then, if `current >= _SAVE_GUARD_MIN`, `_write_healthy_ref(current)` `:1009` and `_maybe_snapshot_last_good`; **`return True`**. Autosave uses this path (`:1902`). | Non-receipt path returns `True` `:1011` and advances `_write_healthy_ref` `:1009` even if the inner gate refused; no truthful in-process boolean at this pin. Receipt path `:1003-1005` does return the receipt; autosave does not use it. | **Yes.** |

The plan no longer treats quarantine/`last_permit_ts` as a sufficient automatic signal. It names the #824 fix as the truthful source (autosave records receipt outcome + timestamp for **both** paths, WARNING+; dream slice fail-closed if no accepted autosave since the previous pass, or the latest attempt was refused by either guard). Signals (1)/(2) are a **second, independent check**. **#824 is S4-GATING and an ARMING precondition** (§4A.4, §4B arming order, §7, §9). **Until it lands: "the automatic check cannot promise what it claims, and the operator-level check below is the only stop."** That is the P418 LE18-N1 alternative, stated plainly.

Operator-level (§4A.4): after-each-cycle, including "any autosave refused or quarantined after the pass (the guardian logs every refusal, `checkpoint_guardian.py:69-70`)" and the #799 regime. Test D-3 covers both refusal paths.

Note (not a miss of the fold): the named operator-level inner-gate log still does not see an **outer** refusal; that path is already logged `CRITICAL` at `:988`. The plan does not claim otherwise. #824 remains the truthful in-process boolean.

---

## (2) N2 / N3 — both census sentences VERBATIM; consent protocol

**Verdict: PASS**

Byte-for-byte compare of the two quoted sentences in §4A.6 (plan lines 427–428) against:

- P415 assignment sentence in `assignments/plan-want-hub-d-p415.md` line 6
- P418 item 4 sentence in `assignments/plan-want-hub-d-p418.md`

Both **equal** (`plan_sents[0] == s415`; `plan_sents[1] == s418`; lengths 120 and 620). Wording is not paraphrased. §4A.6, §4A.9, §7, §0.9, and the §4B arming order all require **both** sentences, side by side.

**Consent protocol, present as ruled (P418 / CH):**

| Required | Where |
|---|---|
| "does not decline" = explicit **UNQUALIFIED** non-decline | §4A.6, §7(b) |
| qualified assent / concerns / ambiguity / silence / timeout → Josh, arming stops | §4A.6, §7(b), §4B |
| exact census text shown recorded VERBATIM beside the verbatim response | §4A.6, §7(a)(b) |
| operator/Executive act, **never** a daemon→session relay (LAW 1 corollary) | §4A.6, LAW 1 row, §7(b) |
| CC-session request to stop = a DECLINE | §4A.4 Q-R runbook, §4A.6, §7(d) |
| N3(vi) RULED: session non-decline is the **first** informed consent; P392 proxy is necessary, not sufficient | §4A.6, §7 consent record |
| Packet 392 sha256 held as the 16-hex prefix `f4bca117bdcd034c`; full digest to be carried into the record | §7 |

"Stale" is not added to the Executive's sentences (§4A.6, OPEN in §10) — that matches P418's "do not paraphrase them."

---

## (3) N4 — PG-1 artifact, named acceptor, fresh process, memory gate

**Verdict: PASS**

§4A.5 PG-1 and §9 sequencing:

- **Committed ARTIFACT** per copy × checkout: printed `neuro_foundation.__file__` + git rev, source/copy sha256 before/after, removed-id hash, return value, full state hash, serialized-bytes hash, `PYTHONHASHSEED`, `MemAvailable`.
- **Who accepts:** Chief / the Executive together with the delta pair; PASS recorded in the artifact and the review record **BEFORE Josh is asked to approve the merge**.
- **Each of the four loads in its OWN fresh process/scope** (one per copy × checkout); reason stated (`sys.modules` / Graph release).
- **Load gate `MemAvailable ≥ ~8 GB`, ONE load at a time (P373); the gate WAITS** (checker-020's ~5.35 GiB reading is recorded).
- §9 order: delta pair on the diff → PG-1 RUNS → artifact committed + PASS recorded → **then** Josh is asked → NG-first merge.

---

## (4) N5 — pre-accepted BAND as reporting thresholds; off the operator list

**Verdict: PASS**

Exact band in §4A.5 item 11, restated in §4A.9 condition (2) and §10's 418 row:

- single-want loss **≤ 55% per cycle**
- guardian margin **≥ +30 points at EVERY save**
- end-state per-want losses (p50 / p90 / max) **within ±5 points of −82% / −90% / −94%**

Inside = reported "within the modelled envelope" + census updated, **no fresh ruling**. Outside → the Executive **before arming**. Nothing is adjusted (no ramp, no cap, no new parameter). **REPORTING thresholds, not tunables:** no env knob, no code constant that changes behaviour; literals live **ONLY in the dry-run/census TOOLING — never in `neuro_foundation.py` or the daemon**.

**Removed from the §4A.4 operator after-each-cycle list** (C20-N2 / LE18-N6): the operator-level list no longer contains ~50% / +30. A separate **PRE-ARMING-ONLY** paragraph points at item 11. They are an arming stop, never a live cap on the key.

---

## (5) #825 — pruned links stay pruned across SAVE AND RESTORE

**Verdict: PASS**

§4A.5 item 12: ARMING precondition, proved **ON A COPY**. Cites `_serialize_incremental` at base `:5287` (`if sid in self.synapses` at `:5301`) — **TRUE** at `e4ebf982` (read via `git show`). Check: armed competing pass in memory → `checkpoint()` in the daemon's actual mode (and INCREMENTAL if any path can select it) to a scratch temp → restore into a fresh `Graph` → assert every removed id is absent and synapse count equals the post-pass count. **#825 GATES ARMING only (not S4).** Also in §4B arming order, §7, §9 (post-merge dry run includes item 12). Optional recording with the PG-1 artifact is extra; PG-1 itself is the all-defaults golden, so the load-bearing check belongs on the dry run.

---

## (6) reporting column, tie-break, telemetry note, pin

**Verdict: PASS**

| Item | Present |
|---|---|
| **Reporting column** "conducting links lost" (`weight ≥ weight_threshold`) per cycle and per want | §4A.5 item 9; §4A.6 INFO record + census; §4A.9. Silent-K-th fact 137/174 out, 174/174 non-short in wherever "strongest" is used. Probe upper bound 10,399 of 106,825 — **recomputed 10,399**. |
| **Tie-break node-id ASCENDING** | §4A.5 item 9, §4A.6, §4A.7 table re-derived with a "wants tied at that max" column. Cycle 2 names `101738dd7219c463` (4-way tie at 300). **Independent recompute matches the whole which-want column** under that rule. |
| **Telemetry note** | §4A.6: armed `pruned` event ~5,000/cycle; dashboard `total_pruned` is `step()`-only (X1), so observers see the synapse-count drop; arming record notes ~5,000/cycle is expected. Cite `openclaw_hook.py:1920` / `neurograph_gui.py:1642` not re-opened this round. |
| **Pin** | Changelog PIN line names fold commit `25ae7a2a72346db0091c3a2d0d25b48e15cd0821` (`git rev-parse` agrees). This file's live pin is `1800d28ad6f82f8f503e45ef787041b52cdf4df1`. §12: cite a COMMIT HASH, never the file name. |

---

## (7) end-state p50/p90/max, −82/−90/−94%, 1,662; contradictions

**Verdict: PASS**

Independent HEIGHT-key recompute (method (20); K=50/50; last-link 16; every competitor eligible; weakest-weight staleness proxy; stdlib `json`; one file at a time; no NG import):

| quantity | plan-005 | this recompute |
|---|---|---|
| F / arena / unique G / competing after last-link | 4,127 / 124,232 / 17,391 / **106,825** | **same** |
| silent K-th out / in | 137/174 / 174/174 | **same** |
| cycle-1 MAX / want / share | 1,527 / `ac4d8c6a7f50852c` / 48% | **1,527 / same / 47.88% → 48%** |
| cycle-1 residual | **1,662** (3,189 − 1,527) | **1,662** |
| cycles (upper) / last removed / live after | 22 / 1,825 / 31,928 | **same** |
| cycle-2 which-want (ascending) | `101738dd7219c463` (4-way tie at 300) | **same** (tied with `8e9856140b42e65f`, `ac4d…`, `c0ee…`) |
| **end-state p50 / p90 / max** | **122 / 151 / 191** (§5.3, §4A.2 done) | **122 / 151 / 191** |
| start max | 3,196 `c0ee…` / 3,189 `ac4d…` | **same** |
| start p90 | 1,447 (§5.3) | **1,447** (index `int(0.9·(n−1))`) |
| start p50 | 669 (§5.3) | **664** at `int(0.5·(n−1))`; **669** at `xs[n//2]` (n=182 even). Same two adjacent ranks already named as a definition difference for p90 in §4A.5 item 9. |

**−82 / −90 / −94%:** from §5.3's pairing 669 → 122 (**−81.8%**), 1,447 → 151 (**−89.6%**), 3,196 → 191 (**−94.0%**). Using index-based start p50 664 → 122 is **−81.6%**. All three sit inside the ±5-point band. The 1,662 figure reproduces.

Cycle-1 conducting links lost under the weakest-weight proxy: **0 of 5,000** (expected: the proxy understates real conducting loss; the dry run reports the real column).

Guardian arithmetic (not `evaluate_save_health`): cycle 1 133,753/138,753 = **96.4%** (+46.4 pts); worst cycle 21 33,753/38,753 = **87.1%** (+37.1 pts). Inside the +30-point floor.

**No contradiction of P399 / P404 / P409 / P412 / P415 / P418** in the changed sections: option (i) only; no weight path; ids never inferred; K/B call arguments; no config key; HEIGHT key; X8 keep; Q-E accept-as-is with no ramp/cap; both census sentences verbatim; Q-R stop-PID-then-unset; N5 band as reporting thresholds; #824 S4-gating / #825 arming-only.

**LOW note (does not require another fold):** §10's compact 412 L7(d) and 415 map rows still quote the pre-P418 wording ("unset the two env values"; "~50% … REPORTED BEFORE ARMING"). The operative text (§4A.4, §4A.5 item 11, §4A.9, §7, and the 418 row) carries the ruled runbook and the band. Not a ruling contradiction.

---

## Numbered corrections

None that block BUILD of the additive surface, the HEIGHT key, or this fold.

**N1 — LOW.** §10's 412 L7(d) and 415 summary rows still carry pre-P418 wording. Operative sections are current. Optional housekeeping on a later pin; not a miss of N1–N8 in the changed sections.

---

## Numbered not-verified

1. Executive Packets 388/392/397/399/404/409/412/415/418 as primary documents (assignment + packet addenda + plan transcription only). Packet 392 full sha256 not re-hashed.
2. Door B liveness on a running laptop daemon (§4B).
3. Test G / PG-1 / dry run / #825 persistence check / armed-path counter test — designed, not executed. Graphs were not loaded.
4. Real-engine HEIGHT order (`inactive_steps`, un-rounded weights, real `synapse_id`).
5. Stored `low_weight_steps` maximum on the competing set.
6. Daemon process environment for `NG_GUARDIAN_*` / `CC_NG_TONIC_IDLE*` (`.bashrc` by name only).
7. `openclaw_hook.py:1920` / `neurograph_gui.py:1642` line cites for the telemetry note — not re-opened this round.
8. Bodies of `_save_would_collapse` / `_read_healthy_ref` / `_SAVE_GUARD_*` values — call site only, as le-018.
9. Native/Rust store; `cleanup_cc_tool_noise.py`; other docs-branch edits to `_dream_loop`.

---

## Verdict per ADDENDUM 4 item

| Item | Verdict |
|---|---|
| (1) N1/#824 both refusal paths; operator-level until #824; #824 ARMING | **PASS** |
| (2) N2/N3 both sentences VERBATIM; consent protocol | **PASS** |
| (3) N4 PG-1 artifact + acceptor + fresh process + 8 GB gate | **PASS** |
| (4) N5 band exact; reporting-only; off the operator list | **PASS** |
| (5) #825 save-and-restore on a copy, ARMING | **PASS** |
| (6) reporting column, tie-break, telemetry, pin | **PASS** |
| (7) end-state 122/151/191; 1,662; −82/−90/−94%; contradictions | **PASS** (LOW N1 = §10 map-row lag) |
| **Overall** | **PASS** |

ROLE B is a separate turn. Nothing built, merged, armed, settled, or dispatched.
