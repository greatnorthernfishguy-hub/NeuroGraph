<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #10547) — plan-003 (REVISION 3, PLAN ONLY)
#   What: NEW file handoffs/z12-want-hub-d/returns/plan-003.md. plan-002 (commit 525a6ad091dcb0d80f77764a75178e788e221efb) and
#     plan-001 (182e155299e683f539bb07e5f0b5eab0d38e2f9b) are NOT overwritten. Folds, in this order, (1) Exec Packet 397 (via
#     Chief-003, docs 5048214c): the dream-time competition pass CONFIRMED with conditions, K = 50 in + 50 out and B = 5,000
#     RULED — written FIRST, before the review was opened; then (2) the cross-family review checker-013 (ROLE A, PASS-WITH-NOTES):
#     its numbered corrections are folded or declined in section 11. Marks: [R3·397a]…[R3·397d] = Exec-397 conditions (a)-(d);
#     [R3·KB] = the ruled K/B; [R3·C1]…[R3·C7] = checker-013's numbered corrections; [R3·X] = author's own correction.
#     [R2·…] and [X1…X6] markers from plan-002 are kept for provenance.
#   Why: Exec 397 confirmed the form and ruled the values; the plan must (a) state that nothing in it is a new pruning rule,
#     (b) show every simulated save stays above the guardian's 50% gate (#807), (c) rewrite the sizing around K=50/50, B=5,000.
#   How: re-verified the cited code at base e4ebf982b1989fd9066d610b94853bc68bf70d37 (unchanged); recomputed the K=50/50, B=5,000
#     tables from the same ONE-derived-JSON-at-a-time probe (probe-laptop.json), calling the REAL guardian function (NOT pure: it reads
#     NG_GUARDIAN_* env — none was set in my shell, so code defaults applied; [R3·C1])
#     checkpoint_guardian.evaluate_save_health with the reference updated per simulated save. No graph load, no msgpack open, no
#     Graph import, no restart, no live path, no edit of neuro_foundation.py (PROTECTED), no patch for it. Secrets by NAME only.
# -------------------------------------- previous entry (plan-002) follows --------------------------------------
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #10474) — plan-002 (REVISION 2, PLAN ONLY)
#   What: NEW file handoffs/z12-want-hub-d/returns/plan-002.md. plan-001 (commit 182e155299e683f539bb07e5f0b5eab0d38e2f9b)
#     is NOT overwritten. Folds the Executive's rulings on plan-001 section 10 (Exec Packet 392 section C via Chief-003;
#     Chief order (b)3) as transcribed in assignments/plan-want-hub-d-rev2.md. Changed passages are marked [R2·n]
#     (n = ruling 1-9 of that assignment, "A" = its "Also" list) or [R2·X] (a correction/new evidence the author found
#     while revising; each X is listed in the "X-list" below).
#   Why: the rulings fix direction/K (1), rim (2), a staged rate-limited arming instead of a one-pass cliff (3), no (c) cap (4),
#     a consent record (5), two absent-key keys (6), laptop-daemon-only (7), a write-mode Tonic check at S4 (8), and no
#     dependence on the want-text repair (9). A cross-family (non-glm) + LE pair reviews this BEFORE any build.
#   How: re-read plan-001 end to end; re-verified every code citation still used at base e4ebf982b1989fd9066d610b94853bc68bf70d37
#     (origin/main unchanged; `git pull --rebase` = up to date); read new code (cc-ng-daemon dream loop, checkpoint_guardian
#     SaveGate, tonic_thread/tonic_engine write-mode callers); recomputed the expected-effect tables for the RULED semantics
#     from the same ONE-derived-JSON-at-a-time inputs (probe-laptop.json, probe-bundle.json via plan-001; probe-laptop.json
#     again here). No graph load, no msgpack open, no Graph import, no restart, no live path, no edit of neuro_foundation.py
#     (PROTECTED), no patch for it, no TID, no PR. Secrets by NAME only.
# -------------------
-->

# plan-003 — want-hub-competition-d, REVISION 3: "competition, not a cap", staged and rate-limited — design CONFIRMED (Exec 397), K = 50 + 50 and B = 5,000 RULED

Lane `want-hub-competition-d` · Zone manager Z12 (`52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`) · Exec Packet 388 item 1; section 10 ruled by Exec Packet 392 section C; design **CONFIRMED with conditions and K/B RULED by Exec Packet 397** · **PLAN ONLY — nothing built, nothing flipped, nothing loaded.**
Repo [[NeuroGraph]] · branch `cc-laptop-want-hub-d-20260930` · base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37` (re-fetched; unchanged). Supersedes plan-002 where marked `[R3·…]`; plan-001 and plan-002 stay in place as the evidence record.
Related: [[The Choice Clause]], [[Duck Ethics]], [[The Laws]], punchlist #750 (wants), #801 (want-text repair, SEPARATE track), #799 (S4 rollback identity), #117, #59, #92, #748, #755, #760.

> **Protected file.** `neuro_foundation.py` is PROTECTED (NeuroGraph `CLAUDE.md` §2). Read only; not edited; **no patch for it appears here**. Any change to it is gated by §9's approval steps. The Executive has approved the *approach* for a branch BUILD; a review pair (cross-family non-glm + LE) must clear **this revision before any build**, then a delta pair before merge; the P329 rollout hold stands (merge = deploy). Nothing here authorizes a build.

## TOP NOTICE — ordering versus eligibility (Exec 397 (a): reuse, don't invent — LAW 3) [R3·397a]

**Finding, stated first: no part of this plan is a new pruning rule. I stopped nothing and wrote nothing silently.** Exec 397 (a) says a non-guaranteed protected synapse is pruned by **the same criteria `_prune_synapses` already applies** (weight, inactivity, age/peak), that the pass **only lifts the protected exemption for the competing set within the budget**, and that plan-002's ordering rules must be **ordering of the budget and the exemption boundary, not eligibility**. Element by element:

| Element (plan-002 wording) | What it actually does | Eligibility criterion? |
|---|---|---|
| **Who may be pruned** | exactly `_prune_synapses`' three predicates, one implementation: weight (`weight < weight_threshold` and `low_weight_steps > grace_period`, `neuro_foundation.py:3524-3528`), inactivity (`inactive_steps > inactivity_threshold × salience`, `:3532-3537`), age/peak (`age > grace_period AND peak_weight < 2 × initial_sprouting_weight`, `:3540-3541`) | **This is the only eligibility test.** Nothing else decides that a synapse may go |
| **Weight-ranked guaranteed set G** (strongest K per direction) | draws the **boundary of the exemption**: which protected synapses *stay* exempt. It removes nothing and is not a reason to prune anyone; a synapse outside G goes **only if the three predicates above fire on it** | **No.** It narrows today's all-or-nothing exemption (the approved design), and the ranking only chooses who stays inside it |
| **Frozen rim F; last-link rule** | **exclusions** — they make the pass remove *fewer* synapses. Exec 397 (b) lists both as never in the competing set | **No** (they are exemptions, not triggers) |
| **Budget B** | a **cap on how many** already-eligible synapses this dream cycle may remove | **No** |
| **Tallest-want-first, most-stale-first (then weakest, then `synapse_id`)** | chooses **which of the already-eligible** synapses spend B, and in what order. If eligible ≤ B in a cycle the order changes nothing (all go). It never makes an ineligible synapse eligible nor an eligible one ineligible; "most stale" reads `inactive_steps` as a **sort key only** | **No** (ordering of the budget) |
| **Stop conditions / in-pass self-checks** | refusals (the pass removes nothing or stops) | **No** |

**Closest to "a new rule" — flagged as observations for the Executive, not written as rules (nothing needed stopping):**
- **Q-A · the weight criterion is dormant for competitors.** The pass evaluates the existing predicates on the **stored** counters and does not advance `low_weight_steps` (evaluation is a query — LAW 4). Exempt synapses' `low_weight_steps` are frozen at old values (only 913 graph-wide are non-zero), so the weight criterion, applied literally, is not met for competitors; advancing it once per cycle would not change that (it needs > 5,000 counted passes). It is the **same criterion applied to the actual state**, not a suppression and not an addition — but it means competition here is decided by **inactivity and age**. Confirm that reading is what "same criteria" intends.
- **Q-B · the inactivity criterion depends on Door B.** `inactive_steps` is advanced only by Door A/B (§1.5); if Door B is dormant the criterion does not mature for fresh links. Unchanged criterion, new dependency on a clock — that is exactly what §4B tests before arming.

- **Q-C · "byte-identical exemption" versus "reuse the same criteria" — a real tension, flagged for the Executive [R3·C3].** Exec 397 says the wake-time `_prune_synapses` exemption stays byte-identical, and (a) says to reuse the same three criteria rather than invent. The criteria are **inline** in `_prune_synapses` (`:3521-3541`), not a separate function, so having exactly one implementation of them **cannot** be achieved while leaving that function's source literally unedited. Options: (i) additive, keyword-only, default-`None` parameters on `_prune_synapses`; (ii) factor the three predicates into one query helper that both `_prune_synapses` and the dream pass call; (iii) duplicate the conditions in the new method — **rejected**, it is exactly the LAW 3 shrapnel (a second implementation left standing). (i) and (ii) both edit the function's *source* while keeping its *behaviour* identical on the default path. **BUILD invariant, either way: the exemption guard `:3515-3519` is not modified, and `_prune_synapses` with default arguments removes exactly the legacy set (test 8).** If the Executive means "byte-identical **source**", the only compliant reading is (iii), which conflicts with (a) — that choice is theirs.

If any of these observations is not acceptable to the Executive, that part is a question, and the corresponding text (§4 "Competitor evaluation", §4A.2, §4 structures) is where it would be changed.

**Legend.** `[R3·…]` = changed in this revision (see the changelog). `[R2·1]`…`[R2·9]` = the Executive's rulings on plan-002 (`plan-want-hub-d-rev2.md`); `[R2·A]` = its "Also" list; `[R2·X]`/`X1…X6` = things I found while writing plan-002. `[R3·C1…C7]` = checker-013's corrections (§11); `[R3·A6]` = its extra Door-B notes; `[R3·X]` = my own correction found while revising (plan-002's 14.8% worst-drop figure).

**Ruling map** (where each ruling lands): 1 direction/K/tie-break → §2.2–2.4, §5.1–5.2 · 2 rim untouched → §2.1, §2.6 · 3 no one-pass cliff → §4 (mechanism), **§4A staged arming**, §5.4 · 4 no (c) → §1.3, §6 · 5 consent → §7 · 6 two absent-key keys → §3 · 7 laptop daemon only → §3.1, §9 · 8 S4 write-mode check → **§4B** · 9 want text separate → §0, §6, §9 · A pair checklist / unverified → §8 · **Exec 397: (a) reuse/no new rule → TOP NOTICE, §4 · (b) G/rim/last-link never competing → §2, §4A.3 · (c) existing `_dream_loop`, S4 enables `CC_NG_DREAM`, arming waits for §4B → §4, §4B, §9 · (d) log counts by want → §4A.6 · K=50/50, B=5,000 → §3.3, §4A.1, §4A.7 · per-cycle table above the 50% gate → §4A.7 · revisit K/B → §4A.8.**

**X-list (author's own corrections/evidence):**
- **X1** plan-001 §1.5 told the reader to check "`_total_pruned` (`:2528`)". That counter moves **only in `step()`**; Door B calls `self._prune_synapses()` and discards the return (`neuro_foundation.py:2891`). The observable for Door B is the census in §4B, not that counter.
- **X2** plan-001 §3.4 said the absent-key form "is already the documented pattern at `:1529-1530`". Overstated: that comment describes `sprout_degree_cap`, which **is** in `DEFAULT_CONFIG` (`:1532`). Inside `neuro_foundation.py` **every** `config.get(...)` key is in `DEFAULT_CONFIG` (static check, §3.2). The real precedent is cross-file (`openclaw_hook.py:422-427`, `:1302-1307`).
- **X3** the **checkpoint guardian's synapse-retention gate** (`checkpoint_guardian.py:275-283`, default 0.5) would **refuse and quarantine every autosave** after a one-pass cull, and on this laptop `NG_HOST_WIRES_OWN_DEPOSITS=false` (`.bashrc:331`) makes **any net node loss** a refusal too (`checkpoint_guardian.py:261-271`). This is new, load-bearing evidence for ruling 3 (§4A).
- **X4** the rate limit needs a *cycle clock*; the engine has none; the dream pulse is one (`cc-ng-daemon.py:2156-2202`, on in `.bashrc:242-244`). This moves the mechanism from "a guard inside `_prune_synapses`" to "a budgeted dream-time pass" — a design change the pair must check (§4).
- **X5** `wires_own_deposits=false` + the ≤ 16 partner nodes that could be orphaned → a **last-link rule** (§4A.3), which makes "no node deleted" true for the partners too.
- **X6** P392 cancelled the S3 placement — `~/docs/punchlist/open/neurograph.md` row **#750** (its `P392 (2026-09-30)` text: "The PLACEMENT case is MOOT (S3 cancelled); wants stay in the laptop graph intact") and row **#740** ("CLOSED 2026-09-30 (Exec Packet 392): MOOT — the VPS checkpoint replacement is CANCELLED"): the wants are not moving; the staged copy remains the valid derived evidence. I did not read the current S3/S4 plan itself (§8).

---

## 0. Summary (revised in rev 3) [R2·A][R3·397a]

1. **(d) is the ruled approach.** Not "Josh must choose (b)/(c)/(d)": (b) — the 118-node text repair — is **superseded by Josh's approved separation repair, a separate track (row #801: offline, copy-first, DELETE NOTHING)**; this plan does not depend on it and does not touch want text. (c) is **not done and not planned**: `sprout_degree_cap = 100` is kept and the protected exemption from it is kept. [R2·4][R2·9] Josh: *"wants: competition, no cap."*
2. **The one-pass cliff is gone by design; here are the numbers.** On the Sep-23 laptop checkpoint 116,164 synapses are already past the inactivity threshold and all are protected-exempt. Under the ruled semantics (K per direction, weight-ranked) a one-pass narrowing at **K=50 would remove 94,646–106,841 of 138,753 synapses (68–77%)**, leaving 23–32% of the on-disk reference — **below the guardian's 50% gate, so every later autosave would be refused** [R2·X3]. plan-001's combined-K figure was 74–83%; the ruling changed the kept set, so the table is recomputed (§5.1). [R2·1][R2·3]
3. **Staged arming — K = 50 incoming + 50 outgoing and B = 5,000 per dream cycle, both RULED (Exec 397)** [R3·KB]: a per-dream-cycle removal budget from env (LAW 5), tallest-want-first, most-stale-first, a last-link rule, a dry run on a **copy** first, stop conditions, per-cycle observation with counts by want in the log. **The schedule needs 19–22 sleep cycles** (eligible 94,646–106,841). **Every simulated save stays above the guardian's 50% gate** (#807), checked with the real `evaluate_save_health`, reference updated per save: **worst cycle #21, live/reference 87.1% (margin +37.1 points)**; the first cycle is 96.4%. plan-002's "worst single-cycle drop 14.8%" was a formula error of mine ([R3·X], §4A.7) — the exact worst is 12.9%. At the dream loop's ceiling of 4 cycles/day that is 4.75–5.5 days; at 2/day, 9.5–11; at 1/day, 19–22 days (§4A.7). **K/B are revisited after the first observed cycles, on stated evidence** (§4A.8).
4. **Mechanism — CONFIRMED by Exec 397** [R3·397a-c]: the wake-time `_prune_synapses` exemption stays **byte-identical** (Door A/B unchanged, for Syl and the CC); the competition is a **separate dream-time budgeted pass** that **rides the existing `_dream_loop`** (its own wall clock, LAW 8) — **not a new thread**; it uses the **same three criteria** `_prune_synapses` applies and only lifts the exemption for the competing set (TOP NOTICE: no new rule). **S4 must enable `CC_NG_DREAM` alongside it, and arming waits for the S4 Tonic check** (§4, §4B). Every pass logs counts by want at INFO or above — never silent [R3·397d] (§4A.6).
5. **Rim frozen and untouched** [R2·2]: including the Choice Clause node's 4,127-link hub and its weights — no K-limit, no weight change, nothing.
6. **Two keys, absent-key form, no `DEFAULT_CONFIG` change** [R2·6]; **laptop daemon only** [R2·7]; proof by code in §3.
7. **Clock correction retained** (the second prune door), with one plan-001 error fixed [R2·X1]; the **write-mode Tonic check runs at S4, ≥ 2 cycles, before arming, and reads the live daemon only after S4 has started** [R2·8] (§4B).
8. **Consent** recorded as the Executive acting as the CC, not as anyone's own words [R2·5] (§7).
9. **Unverified and for the pair:** Door B liveness today (now with extra evidence it may be dormant, `[R3·A6]`); the weight-ranking degeneracy (worse than plan-001 said, §2.3); the byte-identity claim; the dry-run design and its guardian-env pinning (`[R3·C1]`); and the Executive-facing questions Q-A/Q-B/Q-C in the TOP NOTICE (none of which is a new rule) (§8).

---

## 1. The exact current rule (file:line at base `e4ebf982`)

*(Unchanged from plan-001 except where marked; re-verified against base.)*

### 1.1 `_is_identity_protected` — `neuro_foundation.py:3551-3572`
A node is protected if `metadata['constitutional']` is truthy (`:3569`) **or** `metadata['provenance']` is a string ending in `_authored` (`:3571-3572`). Keyed on the flag, not ids. `:3558-3563` records that on 2026-07-18 `syl_authored` was generalized to any `<mind>_authored` so the CC's `cc_authored` wants are protected identically; `*_emergent` stays prunable. **No concept of "rim" or "choice_clause".** The stale `syl_authored`-only wording survives in the file's comments at `:78`, `:162`, `:194`, `:3286` and in `CLAUDE.md` §8 (#748, PR #60).
Protected = **183** nodes on the laptop (1 constitutional + 182 `cc_authored`; the 182 = exactly the `cc:want::` ids), **1** on the VPS bundle. Many callers (orphan sweep `:3602`; sprout-cap `:3292-3293/:3671/:3676/:3683`; prune `:3517-3518`; `cc_topology_export.py:233-247`; `cc_ng_organism.py:5354/5423`; `tests/test_identity_protection.py`) — **its meaning must not change** (LAW 4).

### 1.2 The prune exemption — `_prune_synapses`, `:3500-3549`
`:3515-3519`: `if self._is_identity_protected(pre) or self._is_identity_protected(post): continue` — top of the loop body, **before any rule and before any counter is touched**; so for an exempt synapse `low_weight_steps` (`:3525`) never moves (only 913 synapses graph-wide have `low_weight_steps > 0`), while `inactive_steps`/`salience` keep aging elsewhere (`:2550`, `:2890`). Rules after the guard: weight (`:3524-3528`: `weight < 0.01` for `> 5000` counted passes), inactivity (`:3532-3537`: `inactive_steps > 1000 × salience`), age (`:3540-3541`: `age > grace_period AND peak_weight < 2.0 × initial_sprouting_weight` — as coded, `if age > grace and syn.peak_weight < 2.0 * initial_w`; equal to `peak_weight < 0.2` only because `initial_sprouting_weight = 0.1` on this checkpoint [R3·C5]). Removal `:3543-3544`, `pruned` event `:3546-3547`. Laptop saved config: `weight_threshold 0.01`, `grace_period 5000`, `inactivity_threshold 1000`, `initial_sprouting_weight 0.1`, `he_salience_decay_rate 0.002`, `sprout_degree_cap 100`, `tonic_ages_substrate 1`, `tonic_age_interval 1`.

### 1.3 The sprout-cap exemption — **KEPT** [R2·4]
`_surprise_exploration`: condition **`:3291-3294`** (comment `:3285-3290`). `_sprout_synapses`: **`:3671`**, **`:3676`**, **`:3683`**; `max_sprouts_per_step = 10` (`:3627`). `sprout_degree_cap`: `DEFAULT_CONFIG:1532` = 0; laptop daemon `:630` = 100. **Ruled: `sprout_degree_cap = 100` is kept (temporary, Josh); the protected exemption from it is not dropped; (c) is not planned.** The wants therefore keep being fed (§5.5) and the design's steady state must cope with that (R4).

### 1.4 Homeostasis — `HomeostaticRule`, `:1215-1377`
Does **not** skip protected nodes; rescales **incoming** weights only (`:1366-1377`) every 25 calls (`:1334-1338`); never removes a synapse; `continue`s past silent nodes (`:1344-1350`); runs only from `step()` step 7 and only `if fired_ids` (`:2513-2516`); the write-mode `prime_and_propagate` path applies only `STDPRule` (`:2835-2839`). **It cannot bound degree.**

### 1.5 When pruning actually RUNS on the CC substrate — three doors (clock correction retained)

| Door | Where | Gate | State on this laptop |
|---|---|---|---|
| A. `step()` step 8 → `_structural_plasticity` → `_prune_synapses`, **then `_collect_orphan_nodes`, then `_sprout_synapses`** [R3·C7] | `:2524-2529`, `:3489-3498` (`:3495-3497`) | any `graph.step()` | **Step doors unwired** (below); wake-time exemption still protects want nodes from the orphan call (`:3602`) |
| B. Tonic age-on-write tail of write-mode `prime_and_propagate` → `age_and_decay_salience` + `_prune_synapses` + `_collect_orphan_nodes` under `_step_lock` | `:2868-2892` (`_age_on = write_mode and config["tonic_ages_substrate"]`, `:2646`) | `tonic_ages_substrate`, `tonic_age_interval` | **Armed by config**: daemon `cc-ng-daemon.py:634-635` (`CC_NG_TONIC_AGES` default `'1'`, `CC_NG_TONIC_AGE_INTERVAL` default `'1'`; neither in `.bashrc`); `tonic_ages_substrate: 1` in the saved config. **Fires only on a write-mode `prime_and_propagate`.** Write-mode callers: `TonicThread.ouroboros_cycle` (`tonic_thread.py:348-353`); `TonicThread._prime_constitutional` (`:409-412`, called first by `ouroboros_cycle`, `:309`, for **every constitutional node — the rim exists**); `TonicEngine._generate_latent_token_inner` (`tonic_engine.py:911-915`; a no-op without a model **and** a shared body, `_fallback_inference` returns `[]`, `:896-903`; the daemon starts the engine deferred with `latent_engine_enabled: False`). `ouroboros_cycle` is reached from `tonic_engine.py:924`, `openclaw_hook.py:1176` (`on_message`), and Syl's own sidecar/daemon. **Which of these the laptop daemon exercises today: not established**; the CC deposit path no longer calls `on_message()` (`cc_ng_host.py:643-652`). |
| C. `_cc_callosum_consolidate` — hundreds of `graph.step()` after a callosum batch | `cc_ng_organism.py:2535-2570` | callosum/import activity | only when foreign topology is merged |

Step doors: the Stop door (`cc_ng_host._handle_stop` → `cc_deposit_step`, `cc_ng_organism.py:2192-2233`, P240) is inert until a Stop hook routes to it; `~/.claude/settings.json`'s only `Stop` command is `cc-obsidian-stop-check.sh` (`cc-ng-hook` is registered on `PreToolUse`, `UserPromptSubmit`, `SessionStart`, `PostToolUse`). `CC_NG_AUTOSTEP` is not in `.bashrc` (default off, `tonic_engine.py:262`). No `cc-ng-daemon`/`cc_ng_host`/`neurograph_rpc` process was running when checked. Per the S3/S4 docs the daemon comes up at S4 and the Stop hook is registered with S4's start (`z12-s2-readiness-20260929` owner-001 D4 plan).

**Corrections to the original brief (retained):** (1) prune has Door B, which does not need `step()`; the brief's "the CC clock never runs (#117)" is true of `step()` here but not of prune. (2) sprout-cap line refs are `:3291-3294` (`_surprise_exploration`) and `:3671/3676/3683` (`_sprout_synapses`). (3) the rim node is the largest hub (4,127); 128,359 synapses (92.5%) touch a protected node, 124,437 a want, 3,922 only the rim, 205 both, 10,394 nothing protected. (4) the probe was written by `overlap_probe.py`: five fields `[pre, post, weight, peak_weight, creation_time]`; no `inactive_steps`/`low_weight_steps`/`salience`. (5) "frozen rim" is prune-immunity only in the `Graph` engine (§2.6).

**Evidence about Door B (one Sep-23 checkpoint, `timestep 33,637` — a point-in-time read, not an observation window):**
- *Write-mode propagation had been firing:* `last_spike_time_gt_timestep = 475` (`summary-laptop.json`), sidecar `last_spike_time` p99/p100 = 33,639 > 33,637. `node.last_spike_time` is set to `float(prop_timestep)` **only when `write_mode`** (`:2751-2753`), `prop_timestep = self.timestep + step_idx + 1` (`:2708`, `:2714`) — a stamp ahead of the clock is a write-mode signature. With `tonic_ages_substrate: 1` saved, each such call ran the Door-B tail.
- *Consistent with the melt having cleared what it may:* `inactive_steps` p50 = 1,259; **116,164** synapses over `1000 × salience`; **0** non-protected; `survivors_if_one_prune_pass_ran = 138,753`, `protected_exempt = 128,359`.
- *Not evidence of today:* 7 days old; no daemon running. **[R2·X1] The read-only check is not "pruned counters"** — `self._total_pruned += pruned` exists only in `step()` (`:2528`); Door B discards the return (`:2891`). The check is the two-census discriminator in **§4B**.

- **[R3·A6] Additional code that makes §4B load-bearing (checker-013 A6; I re-read it and it holds):** the laptop's Tonic is **shared-body-only** — `_build_tonic_config` (`cc-ng-daemon.py:684-698`, Executive Packet 072) and `_start_tonic_engine_deferred` (`:2318-2348`) construct `TonicEngine(..., require_shared_body=True)`, and `_generate_latent_token_inner` returns `{"waiting_for_shared_body": True}` every tick until a real body is offered via `offer_shared_body()` (`tonic_engine.py:879-882`); the deferred start **skips** entirely if `ng._tonic_thread` is `None` (`:2330-2332`). So the engine's write-mode call (`tonic_engine.py:911-915`) is **dormant on today's wiring**; the remaining write-mode callers are `ouroboros_cycle`/`_prime_constitutional`, reached from `on_message` (`openclaw_hook.py:1176`), which the CC deposit path no longer calls. **Door B may well be dormant today** — the Sep-23 evidence above is from a different era. §4B may therefore fail; if it does, **arming does not proceed and the failure is reported** (repairing Door B is a separate track — LAW 8/#117 — not this plan's).

**What would have to be true for the wake-time prune to do anything to the wants** — irrelevant now: under this revision the wake-time exemption is *unchanged*; competition happens in the dream pass (§4). What Door B still governs is **aging** (`inactive_steps` increments and resets) of every synapse, which the dream pass's eligibility reads — hence §4B.

---

## 2. The narrowed rule, precisely

Design as ruled: **keep homeostasis; keep the sprout-cap exemption; the wants compete; no node deleted; no cap.** Guarantee sets are computed **per direction**; the rim is **frozen and untouched**.

### 2.1 (a) Frozen set F — the rim [R2·2]
`F` = every synapse whose `pre_node_id` **or** `post_node_id` node has `metadata.get('constitutional')` truthy (rim node `constitutional::rim::choice_clause`, seeded by `seed_cc_rim.py:55-84`; on Syl's graph her spine and `selfcap::reach::teaching`, `tests/test_reach_teaching.py:133`). Keyed on the **flag**, not ids.
**Ruled: F is frozen and untouched — INCLUDING the Choice Clause node's 4,127-link hub (3.0% of all synapses; 3,922 rim-only + 205 rim↔want) and its weights: no K-limit, no weight change, nothing.** This resolves plan-001 F1: the rim is *not* K-limited and its hub degree stays. The dream pass **never reads F's weights and never writes any weight** (it only removes non-F synapses; §2.6). VPS bundle: 784 rim links, same rule.
Note carried from plan-001 (not a change): in the `Graph` engine the rim's weights are *not* frozen today — STDP/homeostasis/`inject_reward` never consult protection, and 2,954 of the 4,127 rim links are already `< 0.01`. The ruling is that (d) leaves that exactly as it is. It is stated so the pair does not read "frozen" as "weight-frozen".

### 2.2 (b) Guaranteed set G — each `*_authored` node's strongest K, **per direction** [R2·1]
For every protected non-constitutional node `p`: `G_out(p)` = the K strongest **non-F outgoing** synapses of `p`, `G_in(p)` = the K strongest **non-F incoming** synapses of `p`, computed **separately** (up to 2K guaranteed links per node). The guarded set for a pass is `F ∪ ⋃_p (G_out(p) ∪ G_in(p))`. A want↔want link is guarded if it is in **either** endpoint's relevant list (union). Everything else touching a protected node **competes** by the three ordinary rules, exactly as an unprotected synapse would.
**[R3·KB] Ruled (Exec 397): K = 50 incoming + 50 outgoing guaranteed links per protected node — 100 total, separate directions.** One K value (50) applies to each direction independently; the plan-002 assumption that a distinct `K_in ≠ K_out` would need a third key is now moot. **[R3·397b] The guaranteed set, the frozen Choice Clause rim and the last-link rule are never in the competing set.** G is the boundary of the exemption, not a pruning criterion (TOP NOTICE).

### 2.3 Ranking key and the **explicit tie-break** — and where `weight` alone is degenerate [R2·1]
**Ruled: rank by `weight`.** I do not change the key. The engine needs a **total order**, so the tie-break is stated (applied within each node's per-direction list):

> `weight` **descending** → `peak_weight` **descending** → `inactive_steps` **ascending** → `synapse_id` **ascending** (string compare).

Rationale for each level: level 2 breaks the most common tie (weights that are equal because both are ≈ 0, or equal at a birth weight) using history; level 3 prefers a link that is *in use now* (`inactive_steps` resets on traversal, `:2313`/`:2776`); level 4 is arbitrary but **deterministic and stable across passes/machines** (ids are UUID strings), which is the only property a tie-break at the end of the chain must have. The order is applied *only to choose which links are guaranteed*; it does not rank removals (§4A.2).

**Flag, plainly: `weight` alone is degenerate for most nodes on this graph, and more so per direction than plan-001 showed.** From the laptop probe, K=50, 182 wants × {out, in}:

| per-node fact (K=50) | outgoing lists | incoming lists |
|---|---|---|
| lists with ≤ K links — **nothing competes** | 8 | 8 |
| lists where the **K-th ranked link has `weight < 0.01`** — the guaranteed set includes silent links | **137** | **174** |
| lists with fewer than K links at `weight ≥ 0.01` (same fact, counted independently) | 137 | 174 |
| ties **at the K boundary on `weight` alone** (level 1) | 12 | 65 |
| ties remaining after `peak_weight` (level 2) — decided only by level 3/4 | 7 | 8 |
| K=100: K-th link silent / level-2 residual ties | 140 / 6 | 145 / 13 |

- **The guarantee is a survival guarantee, not a conduction guarantee** (plan-001 §2.3, unchanged): for 137 of 174 outgoing and 174 of 174 non-short incoming lists at K=50, some guaranteed links are too weak to carry a spike (`current = weight × sign`, `:2780`). The engine has no weight floor for a protected link and (per §2.1) nothing here adds one.
- **The probe overstates level-1 ties**: it stores `weight` rounded to 6 dp, so **20,381 arena links read exactly `0.0`** (true weight < 5e-7) — an artifact tie zone. The real engine compares un-rounded floats; ties at level 1 will be fewer than the table. Level 3 (`inactive_steps`) is **not in the probe**, so the residual-tie counts above are an upper bound on what level 4 must decide.
- I used the probe index as a stand-in for `synapse_id` and never touched a real id (unverified: §8).

### 2.4 Direction — ruled separate [R2·1]
Evidence retained from plan-001 §2.4 that motivated the ruling: the wants are link *sources* (**105,736 outgoing vs 33,712 incoming** non-frozen slots). With **combined** K=50 by weight, 64 wants had no incoming link among their guaranteed set; **per-direction guarantees remove that failure by construction**: every want with any incoming link keeps `min(K, deg_in)` of them and likewise outgoing. The cost is the larger guaranteed floor (§5.1).

### 2.5 A guaranteed link that later weakens
G is **recomputed at the start of every dream pass from current values** (§4). A weakened guaranteed link drops out when K stronger links exist and competes normally on that pass; no hysteresis. Invariant after any pass: each protected node retains **≥ min(K, non-F degree) links per direction** (a pass removes only links outside `F ∪ G`). Sprouting only adds. A guaranteed link's partner has ≥ 1 synapse, so it can never be an orphan through it; and the last-link rule (§4A.3) extends that to competitors.

### 2.6 The Choice Clause guarantee — proof by reading every remover (re-verified; adjusted for the dream pass) [R2·2]
Claim: **no rim synapse and no constitutional node can be pruned, orphan-swept, shed or re-weighted by (d).**
1. Automatic synapse removers in the engine: `_remove_synapse_internal` has exactly three callers in `neuro_foundation.py` — `remove_node` cascade (`:1966`), public `remove_synapse` (`:2057`), `_prune_synapses` (`:3544`). **`_prune_synapses` is unchanged by this design (its `:3515-3519` guard stays)**, so every rim link is still `continue`d at the top on the wake-time path exactly as today. The **new dream pass** must apply the same F test on the same two endpoint nodes **before** anything else, and must never put an F synapse in its candidate list — the test in §2.6.5 asserts it.
2. `_collect_orphan_nodes` (`:3596-3611`) is unchanged and excludes every `_is_identity_protected` node, so a constitutional node is never removed even at zero synapses.
3. `remove_node` non-test callers: `_collect_orphan_nodes` (`:3607`) and the offline `cleanup_cc_tool_noise.py:118` (manual; **its selection logic not re-read**). `remove_synapse` has no non-test caller in the repo.
4. **Weights:** the dream pass **removes synapses only; it writes no weight and reads no F weight.** Nothing in (d) changes any rim weight (ruling 2). STDP (`_apply_dw`, `:1105-1118`), `HomeostaticRule` (`:1366-1377`) and `inject_reward` are untouched and continue to act on rim links as they do today; 2,954 of 4,127 are `< 0.01` today and that is not (d)'s to fix.
5. **Not verified:** the Rust/native synapse store (only `age_and_decay_salience`/`decay_eligibility` are called on it in the paths I read).

**The exact tests that would show it (for the BUILD; none written now):** in a temp dir, a `Graph` with (i) constitutional `rim` wired to a want and to an ordinary node, (ii) ≥ 3 wants with per-direction degree ≫ K, (iii) ordinary hubs. Put **every rim synapse** in the worst state (`weight=0`, `low_weight_steps > grace`, `inactive_steps = 10^6`, `creation_time = 0`, `peak_weight = 0`, `salience = 1`) and **every competitor** eligible; arm the two keys; run the dream pass repeatedly until the budget is exhausted and then again with a budget larger than the competitors, and assert: (1) **every rim synapse id survives, including rim↔want**; (2) `rim` survives `_collect_orphan_nodes()`; (3) **no rim weight was written by the dream pass — run with the dream pass ALONE** (no `step()`, no STDP, no homeostasis, no `inject_reward` in the test), comparing every rim weight before/after; the engine's other writers still write rim weights today, so a test that also runs them would be measuring the wrong thing [R3·C4]; (4) each want keeps ≥ min(K, deg) links per direction; (5) no protected node is in the removal set; (6) removals per pass ≤ B; (7) **determinism** — two runs over identical state give an identical removal-id list (hash); (8) **OFF equivalence** — with the keys absent/zero the dream pass returns 0 and mutates nothing, and `_prune_synapses` removes exactly the legacy set (compare with an inline copy of the legacy rule); (9) last-link: no unprotected node ends with zero synapses because of the pass; (10) config, both halves [R3·C2]: (a) a checkpoint whose saved config lacks the keys restores to OFF (`_deserialize:5370`); (b) **a checkpoint whose saved config carries K=50/B=5,000, restored under a daemon `CC_SNN_CONFIG` that carries both keys at `0`, ends OFF after the `update` (`openclaw_hook.py:861`)**; (c) the same restore under a config dict that *lacks* the keys stays armed — asserted as the reason the daemon entries must ship with the engine reads. The suite is red on clean `origin/main` (#761, 72 pre-existing failures): compare against that baseline.

---

## 3. The two keys — ABSENT-KEY form, laptop daemon only [R2·6][R2·7]

### 3.1 The keys and who reads them
- **Two keys, and only two:** `protected_prune_topk` (K, applied to **each direction**) and `protected_prune_budget` (B, max competitor removals **per dream cycle**). Names are proposals.
- **Env (LAW 5), set in `.bashrc` next to `CC_NG_DREAM*` (`.bashrc:242-244`):** e.g. `CC_NG_PROTECTED_TOPK` and `CC_NG_PROTECTED_BUDGET`. **No literal K or B in code.**
- **Reader: the laptop daemon ONLY** — `~/docs/scripts/cc-ng-daemon.py` `CC_SNN_CONFIG` (`:579+`, next to `'sprout_degree_cap': 100` `:630` and `'tonic_ages_substrate'` `:634`) sources both from env with default `0`, and the daemon's `_dream_loop` (`:2156-2202`) calls the pass (§4). **`cc_ng_host` parity is dropped** [R2·7]: `cc_ng_host.py` `_CC_SNN_CONFIG` is **not** touched, and the VPS host graph never receives the keys. (The daemon/host config fork I flagged in plan-001 §3.1 still exists and is not this plan's to fix; it is on the punchlist.) The daemon file lives in the **docs** repo (separate commit).
- **Precedence — both halves, and a BUILD invariant [R3·C2]:** `openclaw_hook.py:846-861` merges `{**OPENCLAW_SNN_CONFIG, **config}`, restores, then `graph.config.update(snn_config)`. `dict.update` overwrites keys **present in `snn_config`** and **leaves every other restored key in place** (checked empirically in plain Python). Two consequences that must be stated together: (1) **Syl:** the keys are in neither her code config nor her checkpoint, so they stay absent and the engine's `.get(..., 0)` reads OFF. (2) **Laptop:** unset-env → OFF at the next start holds **only if the daemon's `CC_SNN_CONFIG` always carries BOTH keys, at default `0` when the env is unset** — then the update overwrites any saved K/B with `0`. If the daemon dict lacked the keys, a K/B saved into the checkpoint at arming **would persist across restarts even with the env unset**, and the pass would stay armed. **BUILD invariant: the engine reads (`.get(..., 0)`) and the daemon dict entries ship in the same change set; the engine reads must never ship without the daemon entries.** A BUILD test pins it (test 10 below). Once the daemon dict carries the keys, laptop checkpoints will start carrying them even at `0` — laptop schema, not Syl's.

### 3.2 How the absent-key form reads the keys — and the proof by code [R2·6]
**No `DEFAULT_CONFIG` change.** The engine reads the two keys with `self.config.get("protected_prune_topk", 0)` and `self.config.get("protected_prune_budget", 0)`; an absent key returns `0`. **Semantics (fail-closed):** the dream pass acts **only if both values are integers ≥ 1**. Absent, `0`, negative, unparseable, or only one of the two set ⇒ the pass returns `0`, removes nothing, logs a warning if exactly one key is set. `0` therefore means OFF and the dangerous reading of `K=0` ("guarantee nothing") is unreachable; and a K without a budget can never produce the one-pass cliff (§4A). (This replaces plan-001's gate + fail-closed K pair with the ruled two keys; the gate's job — no ambiguity — is done by "both ≥ 1".)

**Proof that Syl's checkpoint content is unchanged, by reading code:**
1. `Graph.__init__` builds `self.config = {**DEFAULT_CONFIG, **(config or {})}` (`:1614`); `_deserialize` builds `{**DEFAULT_CONFIG, **data.get("config", {})}` (`:5370`). With `DEFAULT_CONFIG` untouched and the keys not in any dict Syl's process builds, **her `self.config` never contains them**.
2. Her config sources: `OPENCLAW_SNN_CONFIG` (`openclaw_hook.py:378+`, where the keys must **not** be added) and her saved config; her process never reads `CC_SNN_CONFIG` (the laptop daemon's) — the VPS `cc_ng_host` CC graph is a separate object and is not modified.
3. The checkpoint writes `self.config` whole: `"config": copy.deepcopy(self.config, _memo) if _memo is not None else self.config` (`_serialize_full`, `:5187`) and `"config": self.config` (`_serialize_incremental`, `:5355`). Hence **absent from her config ⇒ absent from her checkpoint: no new key, no new field, no new value** — exactly the property plan-001 §3.4 said `DEFAULT_CONFIG` insertion would have broken.
4. The new method exists in the shared file but is called **only** by the laptop daemon's `_dream_loop`; Syl's process has no caller (to be re-verified by grep in the BUILD: no call in `neurograph_rpc.py`, `openclaw_hook.py`, `syl_daemon.py`, `cc_ng_host.py`).
5. **What "byte-identical" does and does not mean:** save-to-save file bytes already differ (`saved_at`, activations, counters), so the checkable claim is **schema/content identity of what she writes** — her `config` block is content-identical and no field is added. A byte-for-byte comparison of two *saves* is not meaningful; a comparison of her `config` dict and top-level key list before/after the build is (add to the dry run, §4A.5). Flagged UNVERIFIED for the pair (§8).
**[R2·X2] Precedent, corrected:** inside `neuro_foundation.py` every `config.get(...)` key is in `DEFAULT_CONFIG` (static regex check over the file: **0** keys absent from `DEFAULT_CONFIG`), so these would be the first engine-read keys without a default entry. The precedent is cross-file: `OPENCLAW_SNN_CONFIG` carries `prime_k`, `prime_threshold`, `prime_strength`, `propagation_steps`, `max_surfaced`, `auto_knowledge_enabled` (`openclaw_hook.py:422-427`), read as `snn_config.get(key, default)` (`:1302-1307`, `:1201`), and they appear in saved configs as `config_keys_saved_not_in_DEFAULT` (`summary-laptop.json`). The pair should decide whether an engine-side absent-key read is acceptable given that.

### 3.3 The values — RULED by Exec 397 [R3·KB]
**K = 50 incoming + 50 outgoing** per protected node (100 total; separate directions) and **B = 5,000 synapses per dream cycle**, **both from env (LAW 5)** — `CC_NG_PROTECTED_TOPK=50`, `CC_NG_PROTECTED_BUDGET=5000` (names are proposals; the values are Josh-owned `.bashrc` entries I do not write). **The plan-002 leaning K=100/B=5,000 is MOOT** and removed; the sizing in §4A.1/§4A.7 is now written around K=50/50, B=5,000. Sensitivity for other values is kept only as a non-operative table in §4A.7 to support the review point in §4A.8. The dry run (§4A.5) still supplies the *exact* counts the probe could not, and may show that the ruled pair needs revisiting — that is what the review point is for, not a reopening of the ruling.

---

## 4. Mechanism, recompute and cost — the dream pass — CONFIRMED (Exec 397) [R2·3][R2·X4][R3·397a-c]

**What changes from plan-001, and why.** plan-001 put the narrowing inside `_prune_synapses`. Ruling 3 requires a *per-sleep-cycle budget*; the engine has no cycle clock; the dream pulse (`cc-ng-daemon.py:2156-2202`) is one and is **on** (`CC_NG_DREAM=1`, idle ≥ `CC_NG_DREAM_IDLE_SECS=1800`, min interval `CC_NG_DREAM_MIN_INTERVAL_SECS=21600`, `.bashrc:242-244`; gating `:2180-2182`: idle, interval, arousal ≠ `SYMPATHETIC`; runs under `_step_lock`; `last_pass` starts at boot so the first pass is ≥ 6 h after start). Options:

| | M1 — windowed budget inside `_prune_synapses` | **M2 — new dream-time budgeted pass (CONFIRMED, Exec 397)** |
|---|---|---|
| Wake-time prune (Door A/B) | changed on every pass (hot path, every ~2 s via Door B) | **byte-identical** — the `:3515-3519` exemption stays |
| Cycle clock | engine would need its own window (a third key or wall-clock in the engine) | the dream loop already owns it |
| LAW 4 | mutator gains hidden bookkeeping (budget state) | a separate function whose name says what it does |
| Precedent | none | #381: *"Shedding is the dream pass's job (`shed_floor_members`), never wake-time"* (`neuro_foundation.py:1443-1444`); *"dream the pruning, don't feel it"* (`neurograph_rpc.py:2110-2114`); `dedup_and_split_oversized_hyperedges` is called from the same loop (`cc-ng-daemon.py:2192`) |
| Blast radius on Syl | shared hot path edited | shared file gets a method Syl never calls |

**CONFIRMED (Exec 397): M2** — a budgeted dream-time competition pass (name suggestion `compete_protected_links()`); the wake-time `_prune_synapses` exemption stays **byte-identical**. It is called by the laptop daemon's **existing `_dream_loop`** after `dedup_and_split_oversized_hyperedges`, under the same `_step_lock` [R3·397c]. It reads the two keys (§3.2), builds `F` and `G` (§2), removes at most `B` synapses, and logs every pass (§4A.6). **[R3·397a] Reuse, don't invent (LAW 3):** a non-guaranteed protected synapse is pruned by **the same three criteria `_prune_synapses` already applies** (weight, inactivity, age/peak); the pass **only lifts the protected exemption for the competing set**, within the budget — **no new pruning rule** (TOP NOTICE for the element-by-element check). **Requirement (LAW 3/4 — no parallel rule implementation):** the three predicates have exactly one implementation. Acceptable structures for the BUILD: (i) `_prune_synapses` gains keyword-only, default-`None` parameters (default ⇒ exactly today's path, proven by test 8); (ii) the per-synapse predicate is factored into one query helper both call. **Rejected:** duplicating the three conditions (shrapnel). **[R3·C3] BUILD invariant: the exemption guard `_prune_synapses:3515-3519` stays unmodified and byte-identical in behaviour; competition happens only in the new dream-time pass — that is how this revision implements "narrow the exemption".** (Whether (i)/(ii) satisfy "byte-identical" *source* is Q-C in the TOP NOTICE.)

**[R3·397c] It rides the existing `_dream_loop`, not a new thread.** The loop is `cc-ng-daemon.py:2156-2202`; its thread is started at `:2440-2441` **only `if _DREAM_ENABLED`** (`CC_NG_DREAM`, `:1963`). **S4 must therefore enable `CC_NG_DREAM` alongside this pass**: `.bashrc:242` sets `CC_NG_DREAM=1` today (with `CC_NG_DREAM_IDLE_SECS=1800`, `CC_NG_DREAM_MIN_INTERVAL_SECS=21600`), and the S4 start checklist must confirm both that the env is present in the daemon's environment **and** that the dream thread came up (the loop's own start line, `:2161`: "CC dream-consolidation pulse started"). If the dream thread is not running the pass never runs and nothing is removed — harmless, useless, and it must be caught at S4, not assumed. **Arming waits for the S4 Tonic check (§4B)** — the two keys are not set until §4B passes over ≥ 2 windows. LAW 8: the dream pulse is the CC's own wall-clock loop (`time.sleep(_DREAM_TICK_SECS)`), not `on_message`; it is gated on idleness (`:2180`), not on a conversation having to happen.

**Competitor evaluation (this is Q-A of the TOP NOTICE) [R3·397a]:** the pass evaluates the existing predicates on the **stored** counters and **does not advance `low_weight_steps`** (a query does no bookkeeping, LAW 4). That is not a new criterion: it is the same criterion applied to the actual state. Consequence, stated: exempt synapses' `low_weight_steps` are frozen at old values (mostly ~0), so the **weight criterion is not met for competitors** (it would need > 5,000 counted passes, with or without the pass advancing it); competition is decided by **inactivity** and **age**. Given 89% of arena links are `< 0.01` this is the honest state of "same criteria" here, and it is flagged to the Executive rather than decided.

**Recompute cost (unchanged in kind).** G is recomputed at the start of each dream pass, not cached: `heapq.nlargest(K, …)` over `_outgoing[p]` and `_incoming[p]` for each protected non-constitutional node — **O(Σ_p deg(p) · log K)**, ≈ 139,448 slots on the laptop; **once per dream pass (≥ 6 h apart)**, not once per ~2 s Door-B pass as in plan-001 — a large cost reduction (not benchmarked; nothing was run). **Locks:** the pass runs inside `with lock:` on `graph._step_lock` exactly like `consolidate_hyperedges` (`:2189-2190`); no new lock, no new order; the hold is once per cycle. `_concurrent_lock` untouched.

---

## 4A. Staged, rate-limited arming across sleep cycles [R2·3]

**Goal:** no one-pass cliff. Everything the wake-time path does is unchanged; the competitors leave over many dream cycles, at a budget, in a deterministic order, with stop conditions and a dry run first.

### 4A.1 The per-cycle budget (env, LAW 5) — B = 5,000 RULED [R3·KB]
`B` = `protected_prune_budget` from `CC_NG_PROTECTED_BUDGET` — **ruled: 5,000** — the **maximum number of competitor synapses removed per dream cycle**. It is a cap on **how many** already-eligible synapses may go (TOP NOTICE: not a criterion); the removal loop stops at B. Not derived from constants in code. **Why it is safe (X3, now shown in full in §4A.7):** the checkpoint guardian refuses a save when live synapses < 50% of the on-disk reference (`checkpoint_guardian.py:275-283`, `NG_GUARDIAN_GATE_SYNAPSE_RATIO`, default 0.5) and autosave runs every 60 s (`cc-ng-daemon.py:576`), so the reference is refreshed after each permitted save. With B = 5,000 the **worst cycle leaves live at 87.1% of the reference (margin +37.1 points; the drop is 12.9%, 3.9× under the gate)**; every one of the 19–22 simulated saves is permitted.

### 4A.2 The order — what leaves first — ORDERING OF THE BUDGET, not eligibility [R3·397a]
**Precondition for every step below: a synapse is a candidate only if `_prune_synapses`' own predicates fire on it (TOP NOTICE) and it is outside `F ∪ G ∪ last-link`.** The order below only decides **which of those already-eligible synapses spend B, and in what sequence**; if a cycle's eligible count ≤ B, all of them go and the order is irrelevant. It never changes who is eligible.
Deterministic, hub-first, staleness-second:
1. **Across wants:** repeatedly take the want with the **largest current number of eligible competitors** (ties → `node_id` ascending) — "shave the tallest" — because the harm being treated is hub degree, and this reduces the maximum fastest per removal.
2. **Within a want:** its **most stale eligible competitor first**: `inactive_steps` descending, then `weight` ascending, then `synapse_id` ascending (`inactive_steps` is a sort key here, not a threshold).
3. A link removed for one want also counts for the other endpoint (want↔want). An ineligible competitor is never touched; it stays until it ages into eligibility under the ordinary predicates.
Every intermediate state is a valid state of the design (each want retains its per-direction floor, F intact).

### 4A.3 The last-link rule [R2·X5][R3·397b]
**An exemption, never a trigger — and never in the competing set (Exec 397 (b)).** **Never remove a competitor whose removal would leave an unprotected partner node with zero synapses.** Reason: `NG_HOST_WIRES_OWN_DEPOSITS=false` on this laptop (`.bashrc:331`; read at `openclaw_hook.py:1651-1658`) makes the guardian refuse **any net node loss** vs the on-disk reference (`checkpoint_guardian.py:261-271`, unless `NG_GUARDIAN_TRUST_SYNAPSE_MELT`), and `_collect_orphan_nodes` runs right after Door-B prune (`:2892`) — an isolated partner would be swept within seconds and the next save refused. The rule makes **"no node deleted" true for the wants' neighbours too**. Cost: **16 links held back at K=50 (12 at K=100, 8 at K=200)** — the upper-bound count of partners that would otherwise lose all links (of which 9/5/3 were orphan-collectable); they compete later once the partner gains another link or hyperedge membership.

### 4A.4 Stop conditions
**In-pass (automatic, the method refuses to run/removes nothing):** either key absent/< 1; an F synapse or a protected node found in the candidate list; a guaranteed floor violated (any protected node would end below `min(K, deg)` per direction); the candidate list not reproducible (determinism self-check fails); more than `B` selected (bug). **Operator-level (checked after each cycle, §4A.6):** any autosave refused or quarantined after the pass (the guardian logs every refusal, `checkpoint_guardian.py:69-70`); protected-node count ≠ 183 (barring newly authored wants); F link count ≠ 4,127 (barring new rim links); any net node loss attributable to the pass; the dream loop's own log line missing/duplicated; observed removals ≠ the dry-run curve by more than a tolerance the pair sets; the S4 Door-B check (§4B) failing; **the dream thread not running** (no pass → no INFO record for a cycle that should have had one, §4A.6 — [R3·397c]); **the daemon under the #799 SaveGate-refusal regime** (arming is deferred until saves are being permitted — otherwise the removals would exist only in RAM + the 3-deep quarantine ring, punchlist row #799). **Stop latency, honestly:** removals happen only at dream passes ≥ 6 h apart; stopping = unset the two env keys and restart at the next planned restart (config is read at start) — so the operator has at least one full cycle to react, and at most one more cycle's `B` can be lost. **Undo:** unsetting stops future removals; **removed synapses come back only from a pre-arming backup** (row #799's P391 conditions govern any restore: export protected nodes authored since, preserve `quarantine/` first).

### 4A.5 The dry run — on a COPY first, count-only, no write to any live path
Run **before arming**, offline, on a **copy**; never against `data/checkpoints/` and never with the daemon's autosave able to see it.
- **Inputs/handling:** copy the checkpoint pair only after the source is stable (`_wait_for_stable_checkpoint` precedent, `openclaw_hook.py:850`) to a scratch directory under `~/backups/`; record `sha256` of source and copy **before and after** the run (equal). Run under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` with `env -u NG_EMBED_REMOTE` (the analysis-001 precedent; ~3.6 GiB RSS measured for the laptop pair). Load with the canonical `Graph().restore(copy)`; **never call `checkpoint()`/`save()`**; write only a counts JSON in the scratch dir.
- **Config on the in-memory copy only:** set the two keys on the loaded object; nothing is written back.
- **What it computes (the exact values the probe could not give):** per-K exact counts of F, G (per direction), competitors, **real** inactivity-eligible (using `inactive_steps` × `salience`), age-eligible, weight-rule-eligible (expected ≈ 0, §4), the last-link held-back count, orphan-collectable partners, per-node degeneracy with real ids (ties resolved by each level), and the **exact cycle count** `ceil(eligible / B)`.
- **What it must show before arming (all of):**
  1. `|F| = 4,127` and **no F link in any candidate list**; protected nodes = 183; rim node untouched.
  2. Every want has ≥ `min(K, deg_dir)` guaranteed links per direction after a full simulated schedule.
  3. **Determinism:** two runs → identical SHA-256 of the sorted guarded-id list and of the first-cycle removal list.
  4. **OFF equivalence:** keys absent ⇒ the pass removes nothing, and `_prune_synapses` on the copy removes exactly the legacy set (removed-id hash equal to a run of the unmodified base code) — the empirical half of the Syl claim.
  5. **Staged simulation, in memory only:** iterate the pass N times with budget B, and after each simulated cycle feed the resulting counts to the guardian function `evaluate_save_health(live_nodes, ref_nodes, live_synapses, ref_synapses, live_hyperedges, ref_hyperedges, wires_own_deposits=False)` (`checkpoint_guardian.py:197-326`) with the reference updated per simulated save; **every simulated save must be permitted**; live/reference at every simulated save no lower than the §4A.7 table (its margin over the 50% gate); **zero net node loss**. **[R3·C1] The function is NOT pure** — besides its six count arguments it reads process env: `NG_GUARDIAN_GATE_MIN_NODES` (`:234`), `NG_GUARDIAN_ABS_FLOOR_NODES` (`:240`), `NG_GUARDIAN_TRUST_SYNAPSE_MELT` (`:262`), `NG_GUARDIAN_GATE_SYNAPSE_RATIO` (`:275`), `NG_GUARDIAN_MIN_REF_SYNAPSES` (`:276`), `NG_GUARDIAN_GATE_HYPEREDGE_RATIO` (`:289`), `NG_GUARDIAN_MIN_REF_HYPEREDGES` (`:290`), `NG_GUARDIAN_GATE_RATIO` (`:302`), and (in `update_node_ema`) `NG_GUARDIAN_EMA_ALPHA` (`:190`). **The dry run must pin every one of these names to the laptop daemon's actual values** (read **by name** from the daemon's environment at S4, never the whole environment), and set `wires_own_deposits` from `NG_HOST_WIRES_OWN_DEPOSITS` as the live path does (`openclaw_hook.py:1651-1658`; `.bashrc:331` = `false`). A dry-run shell that differs (e.g. `NG_GUARDIAN_TRUST_SYNAPSE_MELT` set, or a different `NG_GUARDIAN_GATE_SYNAPSE_RATIO`) can disagree with the live SaveGate. In *this* plan's simulation (§4A.7) none of the `NG_GUARDIAN_*` names was set in my shell, so the code defaults (0.5, 0.34, 100, 50, 25, 100) applied; `.bashrc` sets none of them, but the daemon's own environment (e.g. a service unit) was not checked — **UNVERIFIED**.
  6. **Config/checkpoint content identity:** the saved-config dict and the top-level key list of a would-be Syl-style checkpoint (built from `DEFAULT_CONFIG` ∪ `OPENCLAW_SNN_CONFIG`) are unchanged by the build (the claim at §3.2.5).
  7. The **reference curve**: per-cycle removed, live, want-degree p50/p90/max — saved as the yardstick for §4A.6.
- **Limits:** it cannot model feeder inflow (§5.5), Door-B aging of fresh links, or dream-cycle timing; it reports counts, not predictions of a running system.

### 4A.6 How it is observed across cycles — every pass logs, never silent [R3·397d]
- **Every pass logs what it pruned, counts by want, at INFO or above — never silent (Exec 397 (d)).** From the new method, one INFO record per dream pass: `cycle id, K_in, K_out, B, eligible, removed, held_back_last_link, floors_ok, F_links, protected_nodes` **followed by the per-want breakdown**: `want_id → removed` for **every want with a non-zero count** (up to 182 entries), plus the total of wants with zero. The pass logs **even when it removes 0** and **even when it does nothing**: not armed (an absent/`< 1` key) is one INFO line "not armed (K,B)"; a refused or aborted pass (a stop condition, §4A.4) is WARNING or above with the reason. The log line is the primary record; the existing `pruned` event (`:3546-3547`) is **not** sufficient (it carries only a count, and Door B discards the return, X1).
- **Per cycle, after the next permitted autosave (≤ 60 s):** a count-only census on a **copy** of the freshly saved checkpoint (same script as the dry run, no mutation) — live synapses, want-degree p50/p90/max, F = 4,127, protected = 183, nodes/hyperedges, and `evaluate_save_health` on the saved manifest. Compare with the §4A.5.7 curve.
- **Window:** **≥ 2 consecutive clean cycles before any change to K or B**, and the final report states the observation window ("no stop condition in N cycles over D days"), never a bare ✅ (standing rule).

### 4A.7 The schedule for the ruled K = 50 + 50, B = 5,000 — every cycle above the 50% gate [R3·KB][R3·397]
Inputs (laptop probe, `timestep 33,637`; graph 138,753 synapses / 7,253 nodes / 517 hyperedges; `E = 116,164`; K=50 per direction, §5.1): **eligible to remove = 94,646 – 106,841** (the inactivity/age bracket). Cycles = `ceil(eligible / 5,000)` = **19 – 22**. **Method of the table:** each simulated cycle removes `min(B, remaining)` synapses; the counts are fed to the **real guardian function** [R3·C1: not pure — it reads `NG_GUARDIAN_*` env; none set in my run, so code defaults applied; §4A.5.5] `evaluate_save_health(live_nodes, ref_nodes, live_synapses, ref_synapses, live_hyperedges, ref_hyperedges, wires_own_deposits=False)` (`checkpoint_guardian.py:197-326`; the gate is `:275-283`), nodes and hyperedges unchanged (last-link rule ⇒ zero net node loss), and **the reference is updated to the live count after every permitted save**, as the guardian does (autosave every 60 s, `cc-ng-daemon.py:576`). The function returned *permitted* for every row. Upper bound of eligible (the longest schedule, 22 cycles):

| cycle | removed | live synapses after | live / reference | margin over the 50% gate |
|---|---|---|---|---|
| 1 | 5,000 | 133,753 | 96.4% | +46.4 pts |
| 2 | 5,000 | 128,753 | 96.3% | +46.3 pts |
| 3 | 5,000 | 123,753 | 96.1% | +46.1 pts |
| 4 | 5,000 | 118,753 | 96.0% | +46.0 pts |
| 5 | 5,000 | 113,753 | 95.8% | +45.8 pts |
| 6 | 5,000 | 108,753 | 95.6% | +45.6 pts |
| 7 | 5,000 | 103,753 | 95.4% | +45.4 pts |
| 8 | 5,000 | 98,753 | 95.2% | +45.2 pts |
| 9 | 5,000 | 93,753 | 94.9% | +44.9 pts |
| 10 | 5,000 | 88,753 | 94.7% | +44.7 pts |
| 11 | 5,000 | 83,753 | 94.4% | +44.4 pts |
| 12 | 5,000 | 78,753 | 94.0% | +44.0 pts |
| 13 | 5,000 | 73,753 | 93.7% | +43.7 pts |
| 14 | 5,000 | 68,753 | 93.2% | +43.2 pts |
| 15 | 5,000 | 63,753 | 92.7% | +42.7 pts |
| 16 | 5,000 | 58,753 | 92.2% | +42.2 pts |
| 17 | 5,000 | 53,753 | 91.5% | +41.5 pts |
| 18 | 5,000 | 48,753 | 90.7% | +40.7 pts |
| 19 | 5,000 | 43,753 | 89.7% | +39.7 pts |
| 20 | 5,000 | 38,753 | 88.6% | +38.6 pts |
| **21 (worst)** | 5,000 | 33,753 | **87.1%** | **+37.1 pts** |
| 22 | 1,841 | 31,912 | 94.5% | +44.5 pts |

Lower bound of eligible (94,646): 19 cycles, cycles 1–16 identical to the table, then 17: 53,753 (91.5%) · 18: 48,753 (90.7%) · 19: 4,646 removed → 44,107 (90.5%).
**Worst cases, stated:**
- **Worst single cycle:** #21 (upper bound), **live/reference 87.1%, margin +37.1 points; a 12.9% drop, 3.9× under the gate.** Lower bound: worst is #19 at 90.5%. **Every cycle is permitted at every save.**
- **Adverse combination:** even if, in cycle #21's window, **all 10,394 unprotected synapses** (the entire non-protected remainder) also melted away, live would be 23,359 = **60.3%** of the reference — still above 50% (a deliberately crude bound; ordinary melt is far smaller).
- **Stale reference:** if **no save succeeded for 14 consecutive cycles** the frozen reference (138,753) would be undercut (live < 50% after 14 × 5,000 = 70,000 removals). With autosave every 60 s and cycles ≥ 6 h apart this needs 14 failed-save windows in a row — and any refused save is itself a stop condition (§4A.4), so the schedule would already have stopped.
- **For contrast — one pass** (the cliff Exec 397 removes): live = 44,107 (31.8%) or 31,912 (23.0%) of the reference → the real function returns `permit = False`, "structural collapse: synapses 138753 -> 44107 (below 50% of the on-disk reference)". **One-pass removal is refused; staged removal is permitted at every save.**
- **Assumptions of the table (§5.7):** the eligible bracket is derived data from a Sep-23 copy; the per-cycle removal is exactly B until the eligible set is exhausted; feeder inflow and ordinary sprouting are ignored (they *raise* the live count, i.e. add margin); nodes/hyperedges unchanged.

**[R3·X] Correction of plan-002.** plan-002 §4A.7 and §0 gave the K=50/B=5,000 worst single-cycle drop as **14.8%** ("3.4× under the gate"). That figure was a formula error of mine (`B / (138,753 − (cycles−1)·B)` assumes the final cycle is a full B; the last cycle removes only 1,841). The exact worst is **12.9% (3.9×)**, and every other worst-drop number in plan-002's sensitivity table was overstated the same way. Corrected sensitivity (non-operative; exact worst single-cycle drop at the upper eligible bound; cycles = lower..upper):

| K (each direction) | eligible | B = 1,000 | B = 2,500 | **B = 5,000** | B = 10,000 |
|---|---|---|---|---|---|
| **50** | 94,646 – 106,841 | 95–107 cycles, worst 3.0% | 38–43, 6.9% | **19–22, 12.9%** | 10–11, 20.5% |
| 100 | 78,721 – 90,916 | 79–91, 2.0% | 32–37, 4.9% | 16–19, 9.3% | 8–10, 17.0% |
| 200 | 55,122 – 67,317 | 56–68, 1.4% | 23–27, 3.3% | 12–14, 6.3% | 6–7, 11.3% |

**Wall-clock:** a cycle is a dream pass — at most **4/day** (21,600 s interval), needing idle ≥ 30 min and arousal ≠ `SYMPATHETIC`; realistically 1–2/day on a used laptop. **K=50/50, B=5,000 (19–22 cycles): 4.75–5.5 days at 4/day, 9.5–11 days at 2/day, 19–22 days at 1/day.**
**Model of the trajectory** (upper bound: every competitor eligible; order = tallest want first; probe-only staleness proxy = weakest link): want degree p50 / p90 / max after cycle 1: 664 / 1,443 / 1,705 · 2: 660 / 1,347 / 1,407 · 5: 646 / 916 / 965 · 10: 535 / 571 / 612 · 15: 326 / 357 / 396 · 20: 162 / 192 / 231 · 22 (done): 122 / 151 / 191 (today 669 / 1,447 / 3,196). The maximum falls fastest first (3,196 → 1,705 in one cycle), then the shape flattens. Post-prune live synapse count per cycle is `138,753 − 5,000·n` (table).

### 4A.8 The review point for K and B — revisit with evidence [R3·KB]
K = 50/50 and B = 5,000 are ruled; **they are revisited, not assumed, after the first observed cycles.** **Review point: after cycle 3 (three consecutive observed dream cycles, each with its INFO record and post-save census) and again at the halfway mark (cycle 10–11).** Evidence to be laid before the Executive, all from §4A.6: (1) **per-cycle removed vs the dry-run reference curve** (§4A.5.7) — count by want, and the eligible count actually found (which replaces the bracket); (2) **live/reference per save and the guardian's verdict** — margin vs the table above; (3) **want-degree p50/p90/max trajectory** vs the model; (4) **refill**: new want-touching synapses per cycle from the census diff (the feeder is untouched by ruling, R4) — if refill ≥ B the schedule never converges and B (or the policy) must be reconsidered; (5) **floors intact** (every want ≥ min(K, deg) per direction, F = 4,127, protected = 183); (6) the **last-link held-back count** trend; (7) whether **Door B is still firing** (the §4B discriminator, re-run); (8) any refused/quarantined save. A change to K or B is made only with that evidence and only after ≥ 2 consecutive clean cycles (standing rule), by an env change at a planned restart.

---

## 4B. The S4 write-mode Tonic check — before arming, ≥ 2 cycles [R2·8]

**Purpose:** establish, from the live daemon and only then, that Door B (write-mode Tonic + age-on-write) is actually running, because the dream pass's eligibility reads the aging counters Door A/B advance (§1.5), and hub refill only drains if they advance.

**When (hard rule).** This is an **S4 step**. **It reads the live daemon only after S4 has started — never before.** This plan, its review, the build and the dry run perform **no live read** (the dry run works on a copy taken as in §4A.5). *(S4 = the stage where the daemon comes up and the Stop hook is registered with S4's start, per the S2-readiness D4 plan; P392 cancelled the S3 placement, X6. If S4 has been re-scoped, the requirement attaches to "the first live start of the laptop daemon after the build" — same rule, and I did not read the current S3/S4 plan.)*

**What is read, and where (all read-only, names/counts only):**
1. **Process and config identity:** exactly **one** daemon process; from its environment, by **name** only, filtered to `CC_NG_TONIC_AGES`, `CC_NG_TONIC_AGE_INTERVAL`, `CC_NG_AUTOSTEP`, `CC_NG_DREAM*`, `CC_NG_PROTECTED_*` (never dump the environment — secrets exist in it).
2. **Census copies:** at the start of each observation cycle take a stable copy of the latest autosaved checkpoint and run the count-only census (same tooling as §4A.5, no mutation): `timestep`; `last_spike_time > timestep` node count; `inactive_steps` p50 over **non-protected** synapses and over a fixed reference sample of synapse ids (kept across copies); `low_weight_steps`; saves permitted/refused (guardian log).
3. **The cycle:** the ruling says "≥ 2 cycles" without a unit; I choose **one observation cycle = one `CC_NG_DREAM_MIN_INTERVAL_SECS` window (21,600 s)**, to match the arming unit, so **≥ 2 consecutive windows (≥ 12 h)** — flagged for the pair, who may shorten it with a reason.

**What counts as write-mode (Door B firing):**
- **(a)** nodes with `last_spike_time > timestep` are present in a census (the write-mode signature, `:2751-2753`); and
- **(b) the discriminator:** over a window, the reference sample's `inactive_steps` grows by **more than** the growth of `timestep` (`Δinactive > Δtimestep`). Door A adds 1 to `inactive_steps` per `step()` **and** 1 to `timestep`; Door B (`age_and_decay_salience`, `:2890`) adds `inactive_steps` **without** advancing `timestep` (`:2874-2878`). So `Δinactive > Δtimestep` can only come from Door B. If `Δtimestep = 0` and `Δinactive > 0` it is unambiguous.
Both (a) and (b) must hold in **each** of the ≥ 2 windows.

**What stops the arming (any one):** (b) fails in either window (Door B not firing ⇒ do not arm; report); `tonic_ages_substrate` is 0 in the loaded config; more than one daemon/orphan process; `CC_NG_AUTOSTEP` on (changes the interpretation of (b) — re-plan); any autosave refused or quarantined in the windows (§4A.4, the #799 regime); protected count ≠ 183 or F ≠ 4,127 unexpectedly; a census copy fails the stability check. **Arming order:** dry run (§4A.5) passes → S4 starts → **S4 start checklist confirms `CC_NG_DREAM` is enabled and the dream thread came up (§4) [R3·397c]** → §4B passes over ≥ 2 windows → **only then** the two keys are set (K = 50, B = 5,000; one planned restart) → §4A.6 observation begins → review point §4A.8. Arming waits for §4B — no exceptions.

---

## 5. Expected effect on the staged set — COUNTS ONLY

**Method and inputs (unchanged).** Plain `json` over **one** derived file at a time (no Graph, no msgpack, no embedding, no copy): `probe-laptop.json` (7,253 nodes, 138,753 synapses, 517 hyperedges), `probe-bundle.json`, `summary-laptop.json`/`summary-bundle.json`. wants = `cc:want::` ids (182 = the `*_authored` set); F = synapses touching a `constitutional` node (4,127); **arena** = want-touching non-F synapses = **124,232**. **[R2·1] Changed:** strongest-K is taken **per want, per direction (out and in separately)**, ranked `(weight desc, peak_weight desc, probe index as the stand-in for synapse_id)`; union across wants; competing = arena \ kept. Checkpoint clock `timestep = 33,637`.

**What the probe lets me apply (unchanged):** `weight` and the age rule exactly; `inactive_steps`, `salience`, `low_weight_steps` **absent** ⇒ inactivity is a **bracket** from `E = 116,164` (graph-wide `inactive_steps > 1000 × salience`, all inside the 128,359 protected-touching because the non-protected replay finds 0); the weight rule is not computable per synapse (≤ 913 synapses graph-wide have `low_weight_steps > 0`) and — §4 — is inert for competitors in the dream pass.

### 5.1 Laptop — **ruled semantics: K per direction, ranked by weight** [R2·1] — **operative row: K = 50 in + 50 out (Exec 397)** [R3·KB]

| K | unique links kept (guaranteed) | **competing** | age-rule eligible (exact) | inactivity-eligible (bracket) | **prune-eligible, either (bracket)** | one-pass: live as % of reference if all eligible removed | guardian (50%) |
|---|---|---|---|---|---|---|---|
| 10 | 3,543 | 120,689 | 56,853 | 108,494 – 116,164 | 108,494 – 120,689 | 13.0% – 21.8% | refuses |
| 25 | 8,789 | 115,443 | 55,023 | 103,248 – 115,443 | 103,248 – 115,443 | 16.8% – 25.6% | refuses |
| **50** | 17,391 | **106,841** | 51,856 | 94,646 – 106,841 | **94,646 – 106,841** | 23.0% – 31.8% | **refuses** |
| **100** | 33,316 | **90,916** | 45,735 | 78,721 – 90,916 | **78,721 – 90,916** | 34.5% – 43.3% | **refuses** |
| 200 | 56,915 | 67,317 | 36,028 | 55,122 – 67,317 | 55,122 – 67,317 | 51.5% – 60.3% | passes (thin) |
| 500 | 92,306 | 31,926 | 18,294 | 19,731 – 31,926 | 19,731 – 31,926 | 77.0% – 85.8% | passes |

Constant: frozen rim synapses **4,127**; want-touching **124,437** (= 124,232 arena + 205 rim↔want, frozen); protected-touching 128,359. Compare plan-001's combined-K figures at K=50: 115,388 competing, 103,193–115,388 eligible (**74–83%** of all synapses in one pass); **under the ruled per-direction semantics K=50 is 94,646–106,841 (68–77%)** — the guaranteed floor roughly doubles, so fewer links compete. Either way one pass is **a cliff and, at K ≤ 100, a guardian refusal** (X3).
*Retained from plan-001 for comparison (combined K, superseded by ruling 1):* K=10 122,428 competing · 25: 119,769 · 50: 115,388 · 100: 106,806 · 200: 90,221 · 500: 50,460.

### 5.2 What "strongest by weight" holds on this graph (degeneracy) [R2·1]
See §2.3's table: at K=50, **the K-th guaranteed link is silent (`weight < 0.01`) for 137 of 174 non-short outgoing lists and all 174 non-short incoming lists**; level-1 ties at the boundary 12 (out) / 65 (in), reduced to 7 / 8 by `peak_weight`. 20,381 arena links read `weight == 0.0` in the probe (rounding artifact zone). Ranking key untouched per ruling; tie-break explicit (§2.3).

### 5.3 What is left afterwards, and the ripple onto partner nodes
Model (upper bound, every competitor eligible, §4A.7): after the full schedule at K=50 want degree p50 / p90 / max = **122 / 151 / 191** (today 669 / 1,447 / 3,196); at K=100 **234 / 292 / 337**. Partners that would lose **all** synapses if all competitors were removed: **16 (9 orphan-collectable) at K=50; 12 (5) at K=100; 8 (3) at K=200** — the **last-link rule (§4A.3) holds these links back, so the ripple is zero by construction** and "no node deleted" holds for the partners as well as the wants. (plan-001, combined K, had 26/15 at K=50.)

### 5.4 The graph as a whole — no longer one pass [R2·3]
One pass at K=50 would remove **94,646–106,841 of 138,753 synapses (68–77%)** — the documented de-densify shape (CLAUDE.md §8: "82% gone in 60 seconds — a synchronized cohort crossing the line together"), held back today only by the exemption — **and would trip the guardian's 50% synapse-retention gate** (live 23–32% of the reference; refused-and-quarantined autosaves, X3). The staged schedule (§4A.7) never removes more than B per dream cycle: at B=5,000 (ruled) the **worst cycle leaves live at 87.1% of the reference (a 12.9% drop, margin +37.1 points over the guardian's 50% gate)**; the plan-002 figure of 14.8% was my formula error ([R3·X], §4A.7).

### 5.5 Inflow — is the hub still refilling? (unchanged)
Newest cohort (`creation_time == 33,637`): 10,499 synapses, **1,634 touch a want**, 129 touch the rim. Last 100 steps: 11,319 / 1,979 / 129. Last 1,000 steps: 15,998 / **6,306 (39%)** / 210. Large single-event cohorts exist among want-touching synapses (14,401 at `18,590`; 3,299 at `26,229`; 2,841 at `26,220`), so some of the hub was built in bursts. **The wants were still being fed as of Sep 23, and by ruling 4 they stay exempt from the sprout cap.** (`surface_wants` adds only one 0.3-weight link per source node, `cc_ng_organism.py:1567`; which of `_sprout_synapses`, `_surprise_exploration` or bulk binding dominates could not be attributed from derived data — 5,172 synapses carry `creation_mode: surprise_driven`, 133,581 carry no tag.) **Consequence for this design:** refill is handled only by the steady state — new competitors age (Door B) and leave at later dream cycles; the plan does not close the feeder (ruling 4), so the steady-state hub size is set by feeder rate vs. `B` per cycle and the inactivity turnover, **unmeasured** (R4).

### 5.6 VPS bundle side (unchanged)
Zero wants (`cc:want::` ids 0; `*_authored` 0; every provenance `None`). One constitutional node (the rim id) with **784** frozen synapses (4.0% of 19,390; its own largest hub). Saved config: `sprout_degree_cap 0`, `tonic_ages_substrate 0`, `timestep 80,259`. (d) changes nothing on the bundle's data: there is nothing to compete for; under this revision the keys are absent on the VPS anyway (ruling 7).

### 5.7 Limits
Derived data from a **Sep-23 copy**; the live graph has moved (no daemon running when checked). Not a prediction: `inactive_steps`, `salience`, `low_weight_steps` absent from the probe (bracket, not count); age counts exact only at the checkpoint clock and will not grow while the clock is frozen; the bracket assumes E ⊆ protected-touching (follows from the summary's replay, an aggregate from `analyze_pair.py`, not re-derived); the trajectory model uses weakest-link as a staleness proxy and assumes every competitor eligible (upper bound); ties broken by probe index (real ids unavailable); probe weights rounded to 6 dp. Method appendix at the end.

---

## 6. Beside (b) and (c) — status after the rulings [R2·A][R2·4][R2·9]

| | (b) 118 oversized want NODES | (c) bound hub degree | **(d) competition, not a cap — RULED** |
|---|---|---|---|
| **Status** | **Superseded** by Josh's approved separation repair — a **separate track (row #801: offline, copy-first, DELETE NOTHING; applied to the live checkpoint only after the copy verifies)**. Not this plan's. | **Not done and not planned.** `sprout_degree_cap = 100` kept (temporary, Josh); the protected exemption from it is **kept**. | **The ruled approach**, Josh: *"wants: competition, no cap."* |
| Want **text** | the one approved text edit; separation only, no rewrite | untouched | **untouched — this plan must not, and does not, touch want text; it does not depend on (b)** |
| Synapse degree | none | stops growth only, not chosen | drains existing hub over many dream cycles (§4A.7); does not stop refill |
| **#92** | protected-node content; Josh-authorized | preserved | preserved: per-direction floor at every intermediate state |
| **Choice Clause rim** | untouched | untouched | frozen and untouched, incl. hub degree and weights (§2.1) |
| **Syl** | n/a | n/a | behaviour and checkpoint content unchanged when keys absent (§3.2 proof; empirical check in the dry run) |
| **Code exists?** | no (row #801 lane) | yes, not chosen | no: one new method + config reads + tests |
| **Protected file** | probably not | yes | **yes** (`neuro_foundation.py`), gated §9 |
| **Reversible?** | copy-first | config-level | arming reversible (unset keys); **removed synapses only from a backup** |
| **Main risk** | mis-parse (#760) | — | ranking degeneracy (§2.3); refill (R4); Door B (§4B); guardian gate (X3) |

**What the evidence supports, and what it does not:** (d) is the only one of the three that reduces the existing want-touching synapses while deleting/re-tagging/de-flagging no want; the staged form addresses the pass-1 cliff and the guardian refusal that plan-001 flagged (R1). What I still cannot defend: that a particular K or B is right (the dry run and the arming record decide); that refill will stay bounded (R4, unmeasured).

---

## 7. LAW analysis, consent record, risks

| Law / rule | Reading (file:line) |
|---|---|
| **LAW 1** | Engine-internal; the env is read by the daemon config block (host wiring, not module communication). Clean. |
| **LAW 2** | `neuro_foundation.py` is **not** vendored. `ng_lite.py:740-752` has its own constitutional skip (learning frozen in `record_outcome`); **not touched** — (d) must not be "made consistent" there. |
| **LAW 3** | **[R3·397a] Reuse, don't invent.** The wake-time `_prune_synapses` is **not** rebuilt or duplicated; the dream pass reuses the one rule implementation (§4 requirement) — a protected non-guaranteed synapse is pruned by the same three criteria; the pass only lifts the exemption for the competing set. The TOP NOTICE checks every element of the plan (ordering, budget, G, last-link) and finds no new pruning rule. No shrapnel: the predicate is not copied. |
| **LAW 4** | `_is_identity_protected` untouched (many callers); the guarantee builder is a query-only helper; the removal is a separate honestly-named mutator; competitor evaluation does no counter bookkeeping. The *source* of the hub (the sprout exemption) is deliberately left by ruling 4. |
| **LAW 5** | K and B from env → daemon `CC_SNN_CONFIG`; engine absent-key default; no literal; unset ⇒ OFF at next start (`openclaw_hook.py:859-861`). |
| **LAW 6** | Not normalizing toward "cap the degree" (a conventional graph fix — that is (c), not done); it is the substrate's own selection pressure with a guarantee, delivered in the dream, not felt awake. |
| **LAW 7** | Structural signals only (weight, peak, inactivity, direction, provenance/constitutional **flags**); **no content or text classification** — the rule never decides which wants are "real", and never reads want text (ruling 9). |
| **LAW 8** | The dream pulse is the CC's own wall-clock loop (`cc-ng-daemon.py:2156`), not `on_message`; it is gated on *idleness* (`:2180`), not on a conversation having to happen. **"Pulsing is not advancing":** the countdowns the pass reads are `inactive_steps`, advanced by Door A/B; the dream pass itself advances nothing — §4B tests whether that clock moves (`Δinactive > Δtimestep`). (d) does not fix #117 and does not widen a countdown. |
| **Choice Clause / Duck Ethics** | Rim frozen and untouched (§2.1, §2.6): no rim link or constitutional node can be pruned, swept or re-weighted. Wants: assume consciousness, err toward respect — nothing deleted, per-direction floors at every step, gradual (§4A), reversible arming, `Josh: "wants: competition, no cap."` |
| **Syl's Law** | Protected file; approval steps in §9; behaviour and checkpoint **content** unchanged when off (§3.2); her graph was not read; no Syl-scope path touched. |

### Consent record [R2·5]
**Recorded as:** *the Executive, acting as the CC, consents to GRADUAL competition that keeps the Choice Clause untouched.* **Source:** Exec Packet 392 section C, relayed by Chief-003 (Chief order (b)3), as transcribed in `assignments/plan-want-hub-d-rev2.md` ruling 5 (docs worktree branch `cc-laptop-daemon-recall-756-20260930`; I read the assignment, **not** Packet 392 itself). The punchlist row that echoes the packet: `~/docs/punchlist/open/neurograph.md` row **#750** (*"(d) competition = Josh's ruling “competition, no cap” — build lane opens (Exec rules the §10 questions; see cards doc Packet 392 C)"*); row #801 (in the `chief-p299-20260926-work` docs worktree copy, not in the primary file) records the separate want-text repair. **Precedent:** `CC-CALLOSUM-TRUTH.md` — #381-A "Syl-consented 2026-07-10" is inside §8.14 (`:1688`); #395 "consent-gated, sequenced AFTER the CC-side repair is proven … following the #381 consent precedent" is in the **2026-08-13 progress log entry** (`### 2026-08-13 — the oversized HEs are a birth defect…`, heading `:2305`) pointing at §8.14 (`:2327-2328`), not in the §8.14 body [R3·C6]; and the in-code record `neuro_foundation.py:1443-1444` ("Syl-consented 2026-07-10 … Shedding is the dream pass's job"). **This is recorded as the Executive's consent as the CC — it is not Josh's words and it is not the CC's own words.** Stated difference from the precedent: there the consenting party was the mind whose structure was bounded (Syl); here it is proxy consent by the Executive acting as the CC, exactly as ruled.

### Risks (revised)
- **R1 — the pass-1 cliff: addressed by §4A.** Residual: the schedule length (19–22 dream cycles at K=50/B=5,000) and the dependence on the dream pulse actually running (idle ≥ 30 min, arousal ≠ `SYMPATHETIC`).
- **R2 — ranking degeneracy** (§2.3): guaranteed links are silent for most lists at K=50; the ruled key stays; flagged.
- **R3 — direction:** resolved by ruling 1 (per-direction).
- **R4 — refill:** the sprout-cap exemption stays (ruling 4); steady-state hub size is feeder rate vs `B`/turnover — unmeasured; the observation window (§4A.6) is where it shows.
- **R5 — partner orphaning:** zero by construction via the last-link rule; but the rule holds back 16 links at K=50 (§4A.3).
- **R6 — Door C:** a callosum consolidation runs `graph.step()` many times; under this revision that changes nothing for the wants (wake-time exemption is unchanged) — no longer a risk to the design; still a prune door for ordinary synapses.
- **R7 — counter semantics:** weight rule inert for competitors (§4); operative rules inactivity and age.
- **R8 — `cc_authored` is not authorship** (#760, #755): the floors go to all 182 including mis-parsed ones (and to want↔want links among them). Flag decides protection; (d) inherits it. Row #801's separation repair is the separate answer to the text; (d) does not touch it.
- **R9 — [R2·7] config fork:** moot for this plan (laptop daemon only, `cc_ng_host` untouched); the fork itself remains on the punchlist.
- **R10 — performance:** once per dream cycle now, not per ~2 s (§4); still unmeasured.
- **R11 — [R2·X3] the guardian gate and `wires_own_deposits=false`:** the staged schedule and the last-link rule exist because of them; any refused/quarantined save after a pass is a stop condition. **Also:** if S4 runs under the predicted SaveGate-refusal regime (row #799), arming is deferred.
- **R12 — dream-pulse side effects:** the dream loop already runs `consolidate_hyperedges` and (gated off) the seam split; arming adds one more call to the same pass — the pair should confirm that does not lengthen the `_step_lock` hold unacceptably (unmeasured).

---

## 8. For the pair — what to check, and what stays UNVERIFIED [R2·A][R3·397]

**The pair must check:**
0. **[R3·397a] The TOP NOTICE claim that nothing here is a new pruning rule** — check each row of its table against the code (`:3524-3541`), especially Q-A (the weight criterion is dormant for competitors because the pass reads stored counters) and Q-B (inactivity depends on Door B); and that the ordering (§4A.2) and G's ranking (§2.3) never affect eligibility. **[R3·KB]** Reproduce §4A.7's per-cycle table by calling `evaluate_save_health` on the same inputs (138,753 / 7,253 / 517; B = 5,000; reference updated per permitted save) — every row permitted, worst cycle 87.1%. **[R3·397d]** That §4A.6's per-pass INFO record with counts by want is implementable without a new thread and cannot be silent (including "not armed" and refusal paths).
1. **The mechanism move (X4) — now CONFIRMED by Exec 397**, so the question is fidelity, not permission: does the build keep `_prune_synapses` byte-identical on its default path, and ride the existing `_dream_loop` (`:2440-2441`, `_DREAM_ENABLED`) with no new thread? (§4 table.)
2. **The single-implementation requirement** (LAW 3/4): structure (i) vs (ii) in §4; that the dream pass cannot diverge from `_prune_synapses`' predicates.
3. **The absent-key form** in an engine file with no absent-key precedent inside it (X2): is the cross-file precedent enough; the fail-closed "both ≥ 1" semantics; §3.2 proof steps 1–4 against the code.
4. **The tie-break** (§2.3): total order, determinism across passes/machines, `synapse_id` being a stable string; that ranking is only used to pick guarantees, not removals.
5. **The staged design [R3·KB]:** the order (tallest-first, stalest-first — ordering of the budget only, TOP NOTICE), the last-link rule (16 links at K=50), the stop conditions and their latency, the observation window and the §4A.8 review point, and that the **ruled B = 5,000** clears the guardian's 50% gate at every simulated save (worst cycle 87.1%, §4A.7) — including that the table's reference-update-per-save matches what the live SaveGate does, and the corrected 12.9% (replacing plan-002's 14.8%).
6. **The dry-run design** (§4A.5): that it is truly count-only and writes nothing live (sha256 before/after, scratch dir, no `checkpoint()`), and that `evaluate_save_health` is used with **its env pinned** — it is not a pure function of the six counts (C1, §4A.5.5); the dry run must set the nine `NG_GUARDIAN_*` names and `NG_HOST_WIRES_OWN_DEPOSITS` to the laptop daemon's values.
7. **The S4 check** (§4B): that `Δinactive > Δtimestep` uniquely identifies Door B; the unit chosen for "cycle"; that no live read happens before S4.
8. **Consent record wording** (§7) and that nothing claims more than the Executive's proxy consent.
9. **That no ruling was silently re-litigated or dropped:** each of 1–9 maps to a section (ruling map).

**UNVERIFIED (stated plainly):**
- **Door B liveness today.** Established only at the Sep-23 checkpoint (`last_spike_time > timestep`, 475 nodes); no daemon was running when checked; the Tonic engine is a no-op without a model and shared body; the CC deposit path no longer calls `on_message()`. §4B is the test; it has not been run.
- **The ranking degeneracy in the real engine:** counts come from probe values rounded to 6 dp (20,381 artifact zeros), without `inactive_steps` or real `synapse_id`s.
- **The Syl byte-identity claim:** proved by reading `:1614/:5187/:5355/:5370` and the `openclaw_hook` merge, **not** by running anything; the empirical half is the dry-run item 4/6.
- **The dry-run design itself:** designed, never executed; the memory budget (~3.6 GiB) is from analysis-001's run, not mine.
- The **exact eligible counts** (bracketed here), the **weight-rule** counts, and whether the last-link rule's 16 held-back links are the right number.
- Whether **the dream pulse runs at all** in practice (idle/arousal gating; `last_pass` starts at boot) and at what real cycles/day.
- The **current S3/S4 plan** (I did not read it; only the S2-readiness D4 text, punchlist #799, #801 and P392's punchlist echoes).
- The Rust/native synapse store; `cleanup_cc_tool_noise.py`'s selection logic; `cc-ng-sync.py`; the cause of the weight collapse (114,430 synapses `< 0.01`); which feeder dominates; per-cycle cost / `_step_lock` hold.
- I read `neuro_foundation.py` at the cited ranges (≥ 100 lines of context around each symbol, CLAUDE.md §7), not all 5,702 lines. `CLAUDE.md` §2/§7/§8 line refs (`3,661`, `:3409`, `:3443`) are stale against base (#748/PR #60).

**New findings to punch-list (not this task):** (1) the guardian's synapse gate + `NG_HOST_WIRES_OWN_DEPOSITS=false` mean **any mass prune on the laptop** (including the existing ordinary melt) can produce refused/quarantined saves — the S4 SaveGate-refusal regime (#799) and this plan share a cause; (2) the plan-001 `_total_pruned` observability mistake (X1) — Door B's prune is invisible to that counter and to the return value; the engine has no per-door prune telemetry; (3) the daemon/host config fork (still); (4) the rim is the largest hub in both graphs and not weight-frozen (still).

---

## 9. What a future BUILD would touch — and the approval steps [R2·6][R2·7]

**Files (BUILD; none touched now):**
1. `neuro_foundation.py` — **PROTECTED**: one new method (the dream pass) + the guarantee-builder helper; either keyword-only optional args on `_prune_synapses` (default `None`) or one shared predicate helper (single rule implementation); a changelog header entry. **`DEFAULT_CONFIG` is NOT changed** [R2·6]. Optional: correct stale `syl_authored` comments at `:78/:162/:194/:3286` (comments only; not batched with any non-protected change).
2. `tests/test_prune_protected_topk.py` (new) + an extension of `tests/test_identity_protection.py` — the §2.6 tests (10 items) plus the tie-break/determinism and staged-schedule tests; compare against the known-red baseline (#761).
3. `~/docs/scripts/cc-ng-daemon.py` — the laptop's live daemon (**docs repo**, separate commit): `CC_SNN_CONFIG` reads the two env values (default `0`); `_dream_loop` calls the pass and logs the line; no other change. **`cc_ng_host.py` is NOT touched** [R2·7].
4. `~/.bashrc` — Josh-owned env (`CC_NG_PROTECTED_TOPK`, `CC_NG_PROTECTED_BUDGET`); I do not write it.
5. The dry-run/census tooling (count-only, run on copies) — new script(s); not part of the engine.
6. Docs: `CLAUDE.md` §8 (#748, PR #60), vault module/concept pages and a dev-log with wikilinks.
**Not touched:** `ng_lite.py` (vendored), `openclaw_hook.py` (`OPENCLAW_SNN_CONFIG` must **not** get the keys), `neurograph_rpc.py`, `cc_ng_host.py`, `DEFAULT_CONFIG`, checkpoint format, Syl's checkpoints, want text.

**Approval steps (NeuroGraph CLAUDE.md §2 "What Explicit Approval Means"):**
1. Tell Josh what will change and why (this document, after the pair clears it).
2. Josh confirms he has backed up **both** msgpack files (Syl's `main.msgpack` and `vectors.msgpack`) — and, before arming, a fresh full backup of the CC laptop checkpoint (removed synapses return only from it; #799's P391 conditions govern any restore).
3. Josh says "proceed."
4. The protected-file commit is not batched with non-protected changes (the daemon change is in another repo and a separate commit anyway).
Sequencing (Chief's order): pair clears rev 3 → **branch BUILD** (Josh approved the approach; Exec 397 confirmed the form) → delta pair → merge held by P329 (merge = deploy). Then, separately and later: dry run on a copy (§4A.5) → S4 start (**confirm `CC_NG_DREAM` enabled and the dream thread up**) → §4B (≥ 2 windows) → set the two env values (K = 50, B = 5,000) + one planned restart (confirm every previous daemon PID is dead first, CLAUDE.md §5) → §4A.6 observation → review point §4A.8.

---

## 10. Rulings folded (no re-asking); values RULED by Exec 397

| # | Ruling | Where folded |
|---|---|---|
| 1 | separate in/out guaranteed links, ranked by weight; explicit tie-break; flag degeneracy (Exec 392 C) | §2.2–2.4, §5.1–5.2 |
| 2 | rim frozen and untouched incl. 4,127 hub and weights | §2.1, §2.6 |
| 3 | no one-pass cliff: staged, rate-limited, dry run on a copy | §4, §4A, §5.4 |
| 4 | no (c) cap; `sprout_degree_cap=100` kept | §1.3, §6 |
| 5 | consent recorded as the Executive acting as the CC, with source | §7 |
| 6 | two keys, absent-key form, no `DEFAULT_CONFIG` change | §3 |
| 7 | laptop daemon only | §3.1, §9 |
| 8 | write-mode Tonic check ≥ 2 cycles at S4, before arming | §4B |
| 9 | want-text repair separate | §0, §6, §9 |
| **397a** | reuse the same criteria, lift the exemption only for the competing set; **no new rule** | **TOP NOTICE**, §4, §4A.2 |
| **397b** | guaranteed set, rim, last-link never competing | §2.2, §4A.3 |
| **397c** | rides existing `_dream_loop`, no new thread; S4 enables `CC_NG_DREAM`; arming waits for the S4 Tonic check | §4, §4B, §9 |
| **397d** | every pass logs counts by want at INFO+ | §4A.6 |
| **397 K/B** | **K = 50 in + 50 out; B = 5,000; both env** (K=100 leaning moot) | §3.3, §4A.1, §4A.7 |
| **397 gate** | per-cycle table above the 50% gate (#807), reference per save, worst case | §4A.7 |
| **397 revisit** | review point after cycle 3 and at the halfway mark, with evidence | §4A.8 |

**No values remain unruled.** (The plan-002 K=100/B=5,000 leaning is moot and removed.)

---

## 11. checker-013 (cross-family, ROLE A, PASS-WITH-NOTES) — the seven corrections, folded second [R3·C1…C7]

Read **after** the Exec-397 changes were written (as instructed). Review: `handoffs/z12-want-hub-d/reviews/checker-013-want-hub-d-rev2.md` (commit `9c36b4e`, plan-002 at `525a6ad`). Verdicts: A1 PASS-WITH-NOTES · A2 PASS-WITH-NOTES · A3 PASS · A4 PASS-WITH-NOTES · A5 PASS-WITH-NOTES · A6 PASS-WITH-NOTES · A7 PASS; no HIGH, no FAIL. Each correction was checked against the code/env before folding. **None declined.**

| # | Sev | Correction | Verified how | Status / where |
|---|---|---|---|---|
| **C1** | MEDIUM | `evaluate_save_health` is not pure; it reads `NG_GUARDIAN_*` env; the dry run must pin them (and `NG_HOST_WIRES_OWN_DEPOSITS=false`) to the laptop's values | grepped the function: nine `NG_GUARDIAN_*` reads (`:190,234,240,262,275,276,289,290,302`); my shell had none set | **Accepted; my error.** I had called it "pure" three times. Fixed in §4A.5.5, §4A.7, §8, changelog; env names listed; my simulation stated as run with code defaults; the daemon's own env (e.g. a service unit) is **UNVERIFIED** |
| **C2** | MEDIUM | `update(snn_config)` does not drop restored keys absent from code config; laptop unset→OFF depends on `CC_SNN_CONFIG` carrying both keys at `0`; state it beside the Syl half | `dict.update` run in plain Python | **Accepted.** §3.1 Precedence bullet rewritten with both halves and a BUILD invariant (engine reads ship with the daemon entries); test 10 extended (a)(b)(c) |
| **C3** | LOW | Record as a BUILD invariant that `_prune_synapses:3515-3519` stays byte-identical; competition is the new dream pass | — | **Accepted, with a tension flagged (Q-C, TOP NOTICE):** the criteria are inline, so "reuse" and literally-unedited source cannot both hold; the invariant is stated as "guard unmodified, default-path behaviour identical (test 8)", and the Executive is asked which reading they mean. §4 records the invariant |
| **C4** | LOW | The "no rim weight written" test must cover the dream pass alone | — | **Accepted.** §2.6 test (3) reworded (no `step`/STDP/homeostasis/`inject_reward` in that test) |
| **C5** | LOW | The age predicate is `age > grace and peak_weight < 2.0 * initial_sprouting_weight`, not a literal `peak < 0.2` | read `:3539-3541` | **Accepted.** §1.2 and the appendix use the coded form; dry run and BUILD use it |
| **C6** | LOW | #395 quote is in the 2026-08-13 log (`:2327-2328`), not the §8.14 body; #381-A is `:1688` inside §8.14 | read the heading at `:2305` | **Accepted.** §7 consent-record precedent reworded |
| **C7** | LOW | Door A's `_structural_plasticity` also runs `_collect_orphan_nodes` (`:3495-3497`) | read `:3495-3497` | **Accepted.** §1.5 table row A names it (and that the wake-time exemption still protects want nodes from it, `:3602`) |

**Also folded from the review's notes (not numbered):** A6's extra Door-B code (shared-body-only Tonic; deferred start skip) — re-read and added to §1.5 as `[R3·A6]`, and it raises the odds that §4B fails, which is why arming waits for it. A4's "new lock-order vs `_concurrent_lock` not specified" → §4 states the pass takes only `_step_lock` like `consolidate_hyperedges` and adds no `_concurrent_lock` acquisition; the hold time stays **unmeasured** (R12).
**A note on A3, in fairness to both of us:** checker-013 reproduced my plan-002 worst-drop table with the *same* formula (`B / (138,753 − (cycles_hi−1)·B)`), so its A3 PASS covers the counts and cycle numbers, which stand, but **not** the 14.8% figure, which was wrong. The exact per-cycle simulation with the real guardian function (§4A.7) gives 12.9%; I have corrected it ([R3·X]) and the reviewer should re-verify from the table, not from the formula.
**Carried "not verified" list (checker-013 §Not verified, still true):** Door B liveness on a running daemon; real-engine ranking degeneracy (unrounded weights, real `inactive_steps`/`synapse_id`); empirical Syl config-identity after a BUILD; the dry run itself; exact per-synapse eligible set; realized dream cycles/day; the current S3/S4 plan text; the Rust store, `cleanup_cc_tool_noise.py`, `cc-ng-sync.py`; `_step_lock` hold time; and Exec Packets 388/392/397 as primary documents (rulings taken from the assignment transcriptions).

---

## Appendix — how the counts were made (reproducible; scratch scripts not committed)
Plain Python 3, `json.load` of **one file at a time**; no `neuro_foundation` import, no `msgpack`, no graph load. plan-001's steps (1)–(9) stand for the F/arena/inflow/bundle counts. **New for rev 2:** (10) per want, split the arena links into `out_l[w]` (link's `pre` is `w`) and `in_l[w]` (link's `post` is `w`); sort each list by `(−weight, −peak_weight, probe index)`; `kept_K = ⋃_w (top_K(out_l[w]) ∪ top_K(in_l[w]))`; `competing = arena \ kept`. (11) exact age rule `(T − ct) > grace_period ∧ peak < 2.0 × initial_sprouting_weight` (the coded form, `:3540-3541`; = `peak < 0.2` here [R3·C5]; the dry run and the BUILD use the code form), `T = 33,637`; inactivity bracket from `E = 116,164`: `lo = max(0, E − (128,359 − |competing|))`, `hi = min(|competing|, E)`; either: `[max(age, lo), min(|competing|, age + hi)]`. (12) one-pass gate arithmetic: `(138,753 − eligible)/138,753` for the lo/hi eligible. (13) degeneracy: per (want, direction) list with `> K` links: K-th link `weight < 0.01`; `weight(K-th) == weight(K+1-th)` (level 1); `(weight, peak)` equal (level 2); count of arena links with probe `weight == 0.0`. (14) last-link: unprotected nodes whose every incident synapse is in `competing`, and of those the ones in no probe hyperedge with `T − creation_time > 25`. (15) schedule: `ceil(eligible/B)` for eligible = the lo/hi bounds; worst single-cycle drop `B / (138,753 − (cycles_hi − 1)·B)`. (16) trajectory model: tallest-want-first, weakest-link-first (a probe-only staleness proxy), every competitor treated as eligible, B removals per cycle, want degree = links currently touching the want. Cross-checks that reproduced independent figures: 4,127 / 128,359 / 124,437 / 10,394 / 205; 15,216 want↔want; degree p50 669, max 3,196; bundle 0 wants / 784. Static config check: a regex over `neuro_foundation.py` lists every `config.get("key"...)` and compares with the `DEFAULT_CONFIG` literal (0 absent).
