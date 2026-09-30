# checker-020 ROLE A (cross-family) — want-hub-competition-d plan REVISION 5 (targeted DELTA)

STATUS: COMPLETE

- Seat: checker-020 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `want-hub-competition-d`. Dispatch #10907. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-d-rev2.md` including ADDENDUM 3 (docs branch `cc-laptop-daemon-recall-756-20260930`, not edited). Also read `assignments/plan-want-hub-d-rev5.md` and `assignments/plan-want-hub-d-p415.md` (ruling text for P409/P412/P415 as transcribed). Packets themselves were not opened as primary documents.
- Plan: `handoffs/z12-want-hub-d/returns/plan-005.md` on `cc-laptop-want-hub-d-20260930` at `9c521699725b30c85fbaad6a6f4b76c1b0cbae03` (a89c7b3d plus the in-place [R5b·P415] follow-up). 715 lines.
- Plan sha256 (`sha256sum`): `231db99dd738f1ad018aed22d034e1095d39a17d3b446d473f1d4a5dbda9396f` (matches packet).
- Inputs folded: `reviews/le-013-want-hub-d-rev4.md` (L1–L12, verified against that file's correction list, not plan-005's own table) and checker-015 C1/C2 (`reviews/checker-015-want-hub-d-rev4.md` numbered corrections). Rulings: Exec P399/P404 (packet addenda), P409 (rev5 assignment), P412 (plan + packet ADDENDUM 3), P415 (`plan-want-hub-d-p415.md`).
- Code pin: `e4ebf982b1989fd9066d610b94853bc68bf70d37`. `neuro_foundation.py` read only via `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:neuro_foundation.py`. Never edited. `git diff e4ebf982 HEAD -- neuro_foundation.py` is empty.
- Derived JSON only: `summary-laptop.json`, `probe-laptop.json` (and `summary-bundle.json` keys only, for the PG-1 prune-replay cites) under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`. No graph, msgpack, checkpoint, or tract load. Want text / VDB prefixes not printed. Node ids only.
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
PYTHONPATH: unset in the run (parent shell had /home/josh/NeuroGraph:)
NG_EMBED_*: unset in the run (parent shell had NG_EMBED_REMOTE)
cwd: /home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930
neuro_foundation in sys.modules: False
cc_ng_organism in sys.modules: False
ng_embed in sys.modules: False
ng_lite in sys.modules: False
```

Parent `PYTHONPATH=/home/josh/NeuroGraph:` would resolve every NG import to the **primary** checkout. No NG module was imported.

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930` at plan pin `9c521699`; `git pull --rebase origin cc-laptop-want-hub-d-20260930` was up to date before the stub and before this complete file.

Stub first-write: commit `3ee5fde1bf8b1b9843a884e576194f109adcf7e3` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-hub-d-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

plan-005 is a faithful fold of le-013 L1–L12 and checker-015 C1/C2 into the TEXT, plus P409 (HEIGHT key), P412 (X8 keep; L7 (a)–(d) as ARMING preconditions) and P415 (Q-E accept as-is, no ramp, no per-want cap, two reporting-trigger conditions). Cycle-1 MAX 1,527 / `cc:want::ac4d8c6a7f50852c` / 48% and the guardian margins (96.4% first cycle; worst 87.1% at cycle 21) recompute from the derived JSON. PG-1 is specified as a pre-merge gate. Two LOW notes, none HIGH. Do not treat the want↔want height detail or Q-R latency as ruled.

---

## (1) each le-013 L1–L12 and checker-015 C1/C2 is really applied in the TEXT

**Verdict: PASS-WITH-NOTES**

Checked against `reviews/le-013-want-hub-d-rev4.md` numbered corrections (lines 59–72) and checker-015 numbered C1/C2 (lines 268–270), **not** against plan-005 §11.

| # | Sev (source) | Applied in the text? | Where |
|---|---|---|---|
| **L1** | HIGH | Yes | §4A.5 PG-1 (pre-merge, two checkouts, laptop copy + VPS-bundle copy, pinned `sys.path[0]`, printed `__file__` + rev, identical removed-id hash / return / full state hash / serialized bytes; merge blocked until it passes). §0.3, §0.8, §9 sequence. Dry-run item 4 vacated. |
| **L2** | MED | Yes | §2.6 test G: `Graph.checkpoint()` bytes, exclusions LISTED (none from `main.msgpack` payload; sidecars excluded because `checkpoint()` does not write them), `PYTHONHASHSEED` pin, fallback OPEN in §10 rather than a silent narrowing to config-dict + key list. |
| **L3** | MED | Yes | X7 RULED HEIGHT (P409) in TOP NOTICE / §4A.2; X8 KEEP `report['eligible']` (P412) in §4.2(g) / §4A.6; plan-004's "No values remain unruled" withdrawn in §10. |
| **L4** | MED | Yes | §2.6.3 names `handle_admin_prune_hubs` / `handle_admin_anneal_core` (cull) at docs `039a3bf4` `:1418`/`:1502`, `:1515`/`:1592`; fail-open `protected()`; #814; dry-run item 10; LAW 3 row; R17. Retire-or-retain left to the Executive. |
| **L5** | MED | Yes | §9 files 1–6: `neuro_foundation.py` commit contains only that file; tests/tooling/docs separate; step 4 stated for the NG repo. |
| **L6** | MED | Yes | §4A.4 automatic pre-call refusals (quarantine newer than `last_permit_ts`; optional env tolerance, unset ⇒ not applied, never a literal); assumption stated; test D-3. |
| **L7** | MED | Yes, with Q-R residue listed | §7 four ARMING preconditions RULED (P412): (a) Packet 392 / ledger `abd57423` / sha256 `f4bca117bdcd034c`; (b) census to a CC-substrate session, verbatim response, decline stops arming → Josh; (c) #92 access = node never removed, reachable via K=50+50, rim untouched; (d) revocation = unset the two env names, pruned links NOT restored, pre-arming backup NAMED (Josh's step-2 full CC laptop pair, path + sha256 in the arming record). Latency mechanics left OPEN as Q-R (§10) because env is read at daemon start. |
| **L8** | MED | Yes, via the allowed alternative | le-013 L8 was "pointer on plan-001..003 **or** a `returns/` index". §12 is the index; changelog names every superseded file. Banner-in-file half **declined by assignment** (rev5 assignment: leave plan-001..004). plan-004's "Nothing declined" is corrected in §11.2. |
| **L9** | LOW | Yes | §4B item 1 reads `CC_NG_TONIC_IDLE*` by name; stop paragraph: if enabled, Tonic latent at 90 s idle inside the ≥ 1,800 s window. |
| **L10** | LOW | Yes | (a) §4.2(d) every competing id exists in `self.synapses`; (b) §4.2(h) `pruned` count is the post-truncation removed count; (c) test G state hash includes `items()` order; (d) mutation cite `:3524-3530` with reset at `:3530` (TOP NOTICE, §4.4, LAW 4). |
| **L11** | LOW | Yes | §4.4 / R16: ≤ 22 increments holds only for the 19–22-cycle schedule; armed means runs each dream cycle until env unset; weight criterion matures after ~5,000 cycles. |
| **L12** | MED | Yes | §4.2(d)(f): competing mode requires `max_removals` an `int ≥ 1` **AND** an `order_key` with an entry for every competing id, explicit `raise` before the loop. Default path keeps both `None`. Test K(6), §4A.4. |
| **C15-1 / C1** | LOW | Yes | Mutation range `:3524-3530`; increment `:3525`, reset `:3530`. Eligibility row still cites `:3524-3528` for the append path (correct; not the C1 miss). |
| **C15-2 / C2** | MED | Yes | Folded into L12. Optional competing budget is gone. |

**Note on L8:** this is the one partial decline. It matches le-013's own "or" and the rev5 assignment's "leave plan-001..004". Not a miss of L8's substance.

**Note on L7(d):** the *path* and the named backup are present; the *latency* (unset takes effect at next daemon start, not the next dream cycle of a running process) is listed OPEN as Q-R. Packet item (6) asked for the path, the named backup, and the Packet 392 citation — all present.

---

## (2) X7 table and section 4A.9 — recompute from derived JSON

**Verdict: PASS-WITH-NOTES**

Independent recompute, one JSON at a time, stdlib `json`, no NG import. Probe synapses are `[pre, post, weight, peak, creation_time]` (5 fields). Ranking used probe index as the `synapse_id` stand-in. HEIGHT key as §4A.2 / method (20): per want, competing links ordered stalest-first (weakest weight, then index — probe has no `inactive_steps`); height = `c_w − r`; want↔want takes the **larger** endpoint height; sort `(-height, weight, index)`; last-link applied (16 held); every remaining competitor treated as eligible (upper bound); B = 5,000; G/competing computed once at K=50 per direction.

**From `summary-laptop.json`:** nodes 7,253; synapses 138,753; hyperedges 517; timestep 33,637; protected 183 (1 constitutional + 182 `cc_authored`); protected-touching 128,359; `E = 116,164`; `low_weight_steps_gt_0` = 913; `last_spike_time_gt_timestep` = 475; prune-replay either = 0 (laptop).

**From `probe-laptop.json` at K=50 per direction, G rank `(−weight, −peak, index)`:**

| quantity | plan-005 | this recompute |
|---|---|---|
| F | 4,127 | 4,127 |
| want-touching | 124,437 | 124,437 |
| arena | 124,232 | 124,232 |
| unique G | 17,391 | 17,391 |
| competing before last-link | 106,841 | 106,841 |
| last-link partners / held | 16 / 16 | 16 / 16 |
| competing after last-link | **106,825** | **106,825** |
| eligible after last-link | **94,630 – 106,825** | **94,630 – 106,825** |
| age-eligible on competing | 51,856 | 51,856 |
| cycles | 19 – 22 | 19 – 22 |
| last cycle (upper) | 1,825 → live 31,928 | 1,825 → 31,928 |
| want deg max / second | 3,196 `c0ee…` / 3,189 `ac4d…` | 3,196 `c0ee2948da6d8123` / 3,189 `ac4d8c6a7f50852c` |

**Cycle-1 MAX-links-lost-by-one-want:** **1,527** from `cc:want::ac4d8c6a7f50852c`, degree 3,189 at start, share 1,527/3,189 = **47.88% → 48%**. Matches §4A.7 row 1 and §4A.9. Touched 168 wants (plan 168).

**Guardian arithmetic** (reference updated to live after each permitted save; 50% gate; **not** a call to `evaluate_save_health`):

| cycle | live after | live/ref | margin over 50% |
|---|---|---|---|
| 1 | 133,753 | **96.396%** (96.4%) | **+46.40 pts** |
| **21 (worst)** | **33,753** | **87.098%** (87.1%) | **+37.10 pts** (drop 12.90%) |
| 22 | 31,928 | 94.593% (94.6%) | +44.59 pts |

Lower-bound last cycle 19: 4,630 → 44,123 / 48,753 = **90.503%**. Every cycle `live >= 0.5 * ref`. Matches plan-005's 1-decimal rounding.

**Cycles 2–10 maxima (the COUNT, then the named want):**

| cycle | plan max / want | this max / want | match? |
|---|---|---|---|
| 2 | 300 / `8e9856140b42e65f` / 18% | 300 / `c0ee2948da6d8123` / 18% | **count yes; id is a 4-way tie at 300** (plan's id is in the set, `plan_loss=300`) |
| 3 | 178 / `6f2de47a34887bda` / 13% | 178 / same / 13% | yes |
| 4 | 149 / `62506085587631cd` / 13% | 149 / same / 13% | yes |
| 5 | 125 / `10bcc022d7402808` / 12% | 125 / `acc9a802cbf75f8c` / 12% | count yes; **2-way tie** (plan's id in set) |
| 6 | 101 / `6dd36cce3d4b4975` / 11% | 101 / `d76ac1e14de752dd` / 11% | count yes; **5-way tie** (plan's id in set) |
| 7 | 83 / `acc9a802cbf75f8c` / 11% | 83 / `e8464672c0c84d64` / 10% | count yes; **2-way tie** (plan's id in set; share 83/828 = 10.0% vs plan 11% rounding) |
| 8 | 70 / `d0941e68da8f3815` / 10% | 70 / same / 10% | yes (also a 2-way tie) |
| 9 | 64 / `e35fca0831875ac1` / 9% | 64 / same / 9% | yes |
| 10 | 58 / `c3b44ac278e61467` / 10% | 58 / same / 10% | yes |

Every later cycle's **max count** in §4A.7 also matched (55, 56, 48, 45, 41, 38, 42, 35, 32, 34, 31, 11). Where the named want differed, the plan's id was always in the tie set at that max. §4A.7 does not state a reporter tie-break. See correction N1.

**§4A.2 trajectory (p50 / p90 / max after cycle, · max loss), this recompute after last-link:** 1: 664 / 1,472 / 1,708 · 1,527; 2: 660 / 1,350 / 1,408 · 300; 3: 655 / 1,177 / 1,233 · 178; 5: 646 / 915 / 966 · 125; 10: 534 / 572 / 614 · 58; 15: 324 / 358 / 398 · 41; 20: 161 / 192 / 232 · 34; 22: 122 / 151 / 191 · 11. Plan's §4A.2 table: 664 / **1,443** / 1,708 · 1,527 after cycle 1 (p50 and max match; p90 is 29 links off). Cycle counts and the guardian table do not depend on this. Dry-run item 9 recomputes with real `inactive_steps`.

**Not recomputed, and why:**

- Real `inactive_steps` HEIGHT order (which of a want's links go): probe has no `inactive_steps`. The *count* per want is set by the height threshold; the plan says so, and this review agrees.
- Real `evaluate_save_health` return strings: not called (would import an NG-tree module). The 50% synapse-gate arithmetic is the load-bearing part and was recomputed.
- Stored `low_weight_steps` maximum on the competing set: not in the probe.
- Un-rounded engine weights / real `synapse_id` tie-break.
- Bundle per-want column: not claimed.

---

## (3) L1 pre-merge gate — executable as written, sequenced before merge

**Verdict: PASS-WITH-NOTES**

PG-1 in §4A.5 / §9:

- **Sequenced before merge:** short delta → Josh backup + proceed → branch BUILD → delta pair on the diff (PG-1 a named review item) → **PG-1 passes** → NG engine merges (P329, merge = deploy) → then daemon slice. The post-merge dry run / §4B / ARMING stay after. Matches L1.
- **Two checkouts, all-defaults `_prune_synapses()`:** specified. Function exists at pin `:3500`; Door A caller `:3495`; Door B `:2891`; `checkpoint` `:5023`; `restore` `:5039`.
- **Copies exist and hashes are preserved (checked, files not loaded):**
  - Laptop copy: `…/analysis-scratch/laptop-copy/` (`summary-laptop.json` `srcdir`). `sha256sum` of `main.msgpack` / `vectors.msgpack` equals `laptop-copy-hashes.txt`: `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` / `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e`.
  - Staged VPS bundle: `…/vps-pull-staged/` (`summary-bundle.json` `srcdir`; analysis-001 input). `main.msgpack` 141,784,662 bytes present. Plan names it by role, not this path; the path is recoverable from the summary `srcdir`.
- **One load at a time, memory-capped:** `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`; each load only with `MemAvailable ≥ ~8 GB`; analysis-001 RSS ~3.6 GiB laptop / ~1.7 GiB bundle. Pin `sys.path[0]`, print `neuro_foundation.__file__` + git rev (a wrong path is void). `PYTHONHASHSEED` pinned identically. Never `save()`; only `checkpoint()` to a scratch temp `.msgpack`.
- **Compare:** removed-id hash, return value, full post-state hash (incl. `items()` order), serialized-bytes (test G exclusions). Laptop default-path removed set is empty (`prune_replay either: 0`); bundle is the load-bearing removal comparison (`by_inactivity: 10433`). Syl's graph never loaded.

**Notes, not defects of the specification:**

- At review time `/proc/meminfo` **MemAvailable = 5,610,488 kB (~5.35 GiB)**, below the ~8 GB gate. A PG-1 run **on this host right now would be refused by the plan's own rule** until memory is free. That is the gate working.
- Four loads in sequence still require the previous `Graph` to be released (plan says "ONE graph load at a time"; say so in the harness).
- I did not restore either copy (forbidden).

---

## (4) L2 — serialized-bytes comparison well-defined, no silent narrowing

**Verdict: PASS**

Test G (§2.6) adds `Graph.checkpoint()` to a temp `.msgpack` on each checkout and a byte comparison of those two files.

**Exclusions, LISTED:**

1. **None from the `main.msgpack` payload.** `_serialize_full` (`:5164+`) has no `saved_at` / wall-time field; `prediction_outcomes[].resolved_at` is a *timestep* (`:3172`, `:3217` per the plan; not re-litigated here). The graph's own `timestep` is **not** excluded and must be equal.
2. **Sidecars excluded because `Graph.checkpoint()` does not write them:** `.manifest.json` (`saved_at`), `.activations.json` (`saved_at`), `.guard_state.json` (`last_permit_ts`, `updated`), `quarantine/` UTC names.

**Practicality, stated plainly:** `_serialize_hyperedge` writes `list(he.member_nodes)` and `list(he.child_hyperedges)` (`:5115`, `:5128`) from sets, so bytes match across two processes only if `PYTHONHASHSEED` is pinned. With a pinned seed and the same restored input, byte-exact is argued practical. If it nevertheless fails on the real graphs, the plan **says so with the reason and lists the fallback as an OPEN item for the Executive** (§10) — the narrow "config dict + top-level key list" form is **never silently substituted**.

§3.2.5 still uses "content-identical" for the **config dict / top-level key list** check (D-1 / dry-run item 6). That is a different, weaker claim about K/B never being stored. L2's load-bearing comparison lives in test G / PG-1. P399 condition 4's "checkpoint byte-identical" is the serialized-bytes test, not the §3.2.5 sentence.

The plan's reading of Chief's "clock/`saved_at`" as *wall-clock* (not `timestep`) is listed in §10 for Executive confirmation. It is the stricter reading. Packet item (4) is met.

---

## (5) P415 text — Q-E accepted; conditions are reporting triggers

**Verdict: PASS-WITH-NOTES**

Against `assignments/plan-want-hub-d-p415.md`:

| P415 requirement | Plan-005 |
|---|---|
| Q-E = ACCEPT as-is; **NO ramp, NO per-want cap** | Held. TOP NOTICE, §0.5, §0.9, §4A.9, §10. No ramp, no second parameter, no per-want cap added. Every former "Q-E open" is replaced by the ruling (changelog). |
| Ruling reasons written into §4A.9 and §10 | Held: stale/never-strong (P399 Q-A); guarantees hold (#92 intact); one hub, one cycle (≤ 18% elsewhere); guardian 96.4% / worst 87.1%; a ramp/cap is a database-style limit. |
| Condition 1: L7(b) census states the finding in plain words; template with dry-run numbers; decline STOPS arming and goes to Josh | Held. Template in §4A.6 / §4A.9: `"cycle 1 removes 1,527 stale links (48%) from cc:want::ac4d8c6a7f50852c; it keeps its 100 strongest and every rim link."` Numbers = dry run's (§4A.5 item 11), not this model's. |
| Condition 2: dry run recomputes; any single want-cycle ABOVE ~50% **or** guardian margin BELOW +30 points → REPORT BEFORE ARMING, do NOT adjust | Held. §4A.5 item 11; §4B arming order. |
| Thresholds are reporting triggers, not tunables (no env knob, no code constant that changes behaviour) | Held for the *algorithm*. No `CC_NG_*` knob for 50% / +30. They do not change which links `order_key` selects. |

**Note (N2):** §4A.4 also lists the same Q-E triggers under **operator-level (checked after each cycle)** stop conditions, while the sentence still says "REPORT before arming and change nothing". P415 placed them as **ARMING preconditions**. A live mid-schedule stop on ~50% would be extra conservatism, not a cap on the key, but it is the wrong list. Keep them pre-arming unless the Executive wants them as live stops.

No new competing-mode parameter, no ramp schedule, no per-want `max_removals` split.

---

## (6) L7 / X8 — ruled text present as ARMING preconditions

**Verdict: PASS**

Packet item (6) checklist:

| Required | Present |
|---|---|
| X8 KEEP `report['eligible']` | §4.2(g), §4A.6 INFO record, TOP NOTICE, §10. P412. |
| Verbatim CC-session response | §7(b), §4A.6, §9: shown to a CC-substrate session with its NG live; response recorded VERBATIM; arm only if it does not decline. |
| "pruned links are NOT restored"; plan **names** the pre-arming backup | §7(d): *the full backup of the CC laptop checkpoint pair (`main.msgpack` + `vectors.msgpack` + sidecars) that Josh confirms under approval step 2 (§9), taken immediately before arming and recorded in the arming record by path and `sha256`.* Analysis-001 `laptop-copy` is explicitly **not** that backup. |
| Revocation path | Unset `CC_NG_PROTECTED_TOPK` / `CC_NG_PROTECTED_BUDGET` → OFF at the next daemon start (env read at import, like `_DREAM_*`). Automatic pre-call refusals bound some failures without a restart. |
| Primary Packet 392 citation | id/ledger `abd57423` / sha256 `f4bca117bdcd034c` (16-hex form as supplied in P412; plan did not open the primary; this review did not either). |

Arming order in §4B includes the two P415 conditions and the four L7 preconditions before the two env values are set.

---

## (7) contradictions of P399/P404/P409/P412/P415, or fails if built as written

**Verdict: PASS-WITH-NOTES**

No contradiction of the option choice, the explicit-ids rule, the no-weight-path rule, the no-config-key rule, the HEIGHT key, X8 keep, L7 as ARMING preconditions, or Q-E accept-as-is.

| Ruling | Status |
|---|---|
| P399 Q-C option (i) only | Held. |
| P399 Q-A no weight path | Held. No suppressing branch on `low_weight_steps`. |
| P399 cond. 1 golden / cond. 2 explicit ids / cond. 3 NG-first / cond. 4 OFF + no new config keys | Held; cond. 4's byte-identical half is test G / PG-1 (item 4). |
| P404 C5 static key on `to_prune` before truncation; C6 K/B as call arguments | Held. HEIGHT is the key (P409). |
| P409 HEIGHT; naive rejected; MAX-links-lost column; report large share; no per-want cap | Held. Cycle-1 48% reported in §4A.9. Want↔want **larger** height still OPEN (plan flags it; P409 did not spell it out). |
| P412 X8 keep; L7 (a)–(d) ARMING | Held (item 6). |
| P415 Q-E accept; two conditions as reporting triggers | Held (item 5), with N2. |

What holds if built as written:

- Wake-time callers keep today's behaviour **iff** sort/slice/report are gated on non-`None` params and competing mode is a separate iterator (§4.2(i); tests G and K).
- Competing call without budget or order **raises** (C2/L12).
- PG-1 blocks merge until the two-checkout real-graph golden passes.
- Unset env ⇒ skip + INFO `"not armed"`; own try/except so `last_pass` still updates.
- No `DEFAULT_CONFIG` / `CC_SNN_CONFIG` / `OPENCLAW_SNN_CONFIG` key.

OPEN on purpose (not fails): X7 want↔want detail; Q-R revocation latency; L2 byte-exact fallback **if** PYTHONHASHSEED-pinned comparison still fails; X9/X10; #814 retire-or-retain; Door B liveness.

---

## Numbered corrections

**N1 — LOW.** §4A.7's **which-want** column needs a stated tie-break. The MAX **counts** (and cycle-1's unique 1,527 / `ac4d8c6a7f50852c` / 48%) recompute. On cycles 2, 5, 6, 7, 8, 13, 15, 16, 19, 21, 22 more than one want shares that cycle's max (cycle 2: four wants at 300, including the plan's `8e9856140b42e65f`; cycle 22: 126 wants at 11). The dry-run reporter (item 9 / the ARMING census) should name the rule (e.g. node-id ascending) so two runs name the same want.

**N2 — LOW.** Move the Q-E ~50% / +30-point triggers out of the §4A.4 **operator-level after-each-cycle** stop list, or mark them pre-arming-only. P415 put them on the ARMING census / dry run. They must not become a live per-want cap.

No HIGH. No FAIL item. N1–N2 do not block BUILD of the additive surface or the HEIGHT key.

---

## Numbered not-verified

1. Executive Packets 388/392/397/399/404/409/412/415 as primary documents (assignment + packet addenda + plan transcription only). Packet 392 sha256 `f4bca117bdcd034c` not re-hashed.
2. Door B liveness on a running laptop daemon (§4B).
3. Test G / PG-1 / dry run / armed-path counter test — designed, not executed. Graphs were not loaded.
4. Real-engine HEIGHT order (`inactive_steps`, un-rounded weights, real `synapse_id`).
5. Stored `low_weight_steps` maximum on the competing set.
6. Daemon process environment for `NG_GUARDIAN_*` / `CC_NG_TONIC_IDLE*` (`.bashrc` by name only).
7. Idle-window evidence path (X10).
8. Native/Rust `SynapseStore` competing-ids iterator / `to_checkpoint_msgpack` byte identity across checkouts.
9. `cleanup_cc_tool_noise.py`; current S3/S4 plan text; other docs-branch edits to `_dream_loop`.
10. Which `.guard_state.json` the daemon's `_guarded_save` maintains (L6 BUILD detail).
11. PG-1 on this host at review time: MemAvailable was ~5.35 GiB, below the plan's ~8 GB load gate.

---

## Verdict per ADDENDUM 3 item

| Item | Verdict |
|---|---|
| (1) L1–L12 and C1/C2 applied in the TEXT | PASS-WITH-NOTES (L8 via index; L7 latency OPEN as Q-R) |
| (2) X7 table / §4A.9 numbers | PASS-WITH-NOTES (cycle-1 1,527/48% and guardian margins match; which-want ties → N1) |
| (3) L1 pre-merge gate executable and sequenced | PASS-WITH-NOTES (copies+hashes exist; MemAvailable currently below the gate) |
| (4) L2 serialized-bytes, no silent narrowing | PASS |
| (5) P415 Q-E / reporting triggers | PASS-WITH-NOTES (N2 list hygiene) |
| (6) L7/X8 ARMING preconditions | PASS |
| (7) contradictions / fails if built as written | PASS-WITH-NOTES |
| **Overall** | **PASS-WITH-NOTES** |

ROLE B is a separate turn. Nothing built, merged, armed, settled, or dispatched.
