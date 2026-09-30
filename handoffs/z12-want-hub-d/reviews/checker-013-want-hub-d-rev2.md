# checker-013 ROLE A — want-hub-competition-d plan REVISION 2

STATUS: COMPLETE

- Seat: checker-013 (cross-family, report_only). ROLE A only; ROLE B not read, not written.
- Lane: `want-hub-competition-d`. Dispatch #10525. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-d-rev2.md`
- Plan: `handoffs/z12-want-hub-d/returns/plan-002.md` on `cc-laptop-want-hub-d-20260930` at `525a6ad091dcb0d80f77764a75178e788e221efb`
- Plan sha256 (sha256sum): `cb2fb4b98801f2b369763b2a5e58d9ca5fea6cb6a15af0edb8e232b24784c520` (matches packet)
- Code pin: `e4ebf982b1989fd9066d610b94853bc68bf70d37`. `neuro_foundation.py` read only via `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:neuro_foundation.py`. sha256 `7080d57a6a0a16cc070eb39b4e788f8a02701b383c914cb46d3d1afd2344eaae`. Never edited.
- Derived JSON only: `summary-laptop.json`, `probe-laptop.json` under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`. No graph, msgpack, checkpoint, or tract load. VDB text not printed.
- Rulings source actually read: `assignments/plan-want-hub-d-rev2.md` (Exec Packet 392 C as transcribed). Packets 388/392 themselves were not opened.
- Authority: report_only. No build, no PR, no merge, no settle, no dispatch. Primary `/home/josh/NeuroGraph` left on `main`.

## P379 session start (printed before analysis)

```
python /usr/bin/python3
sys.path[0:8]= ['', '/home/josh/NeuroGraph', '/home/josh', ...]
module=neuro_foundation in_sys.modules=False origin=/home/josh/NeuroGraph/neuro_foundation.py
module=cc_ng_organism in_sys.modules=False origin=/home/josh/NeuroGraph/cc_ng_organism.py
module=ng_embed in_sys.modules=False origin=/home/josh/NeuroGraph/ng_embed.py
module=ng_lite in_sys.modules=False origin=/home/josh/NeuroGraph/ng_lite.py
module=neurograph_rpc in_sys.modules=False origin=/home/josh/NeuroGraph/neurograph_rpc.py
module=openclaw_hook in_sys.modules=False origin=/home/josh/NeuroGraph/openclaw_hook.py
module=ng_tract_bridge in_sys.modules=False origin=/home/josh/NeuroGraph/ng_tract_bridge.py
module=ng_ecosystem in_sys.modules=False origin=/home/josh/NeuroGraph/ng_ecosystem.py
module=ng_autonomic in_sys.modules=False origin=/home/josh/NeuroGraph/ng_autonomic.py
module=cc_ng_host in_sys.modules=False origin=/home/josh/NeuroGraph/cc_ng_host.py
CC_NG_WORKSPACE None
CC_NG_PYTHONPATH None
PYTHONPATH /home/josh/NeuroGraph:
NG_EMBED_*: unset
```

`find_spec` resolved to the primary checkout because `PYTHONPATH` contains `/home/josh/NeuroGraph`. No NG module was imported. The JSON recompute used stdlib `json` only.

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930` was already at `525a6ad`; `git pull --rebase origin cc-laptop-want-hub-d-20260930` was up to date. Primary NeuroGraph: `main`, clean.

---

## A1 The mechanism reading

**Verdict: PASS-WITH-NOTES**

The plan's reading is faithful to Exec Packet 392 C ruling 3 as transcribed (staged, rate-limited arming across sleep cycles) and to Josh's "competition, no cap". It is a locus change relative to a literal "narrow the `#92` `continue` inside `_prune_synapses`".

Traced at pin `e4ebf982`:

- `#92` exemption is the top-of-loop `continue` in `_prune_synapses` (`neuro_foundation.py:3515-3519`) before any counter or rule. Protected-touching synapses never enter weight / inactivity / age removal on this path.
- Door A: `Graph.step()` step 8 (`:2524-2528`) → `_structural_plasticity` (`:3489-3498`) → `_prune_synapses` then `_collect_orphan_nodes` then `_sprout_synapses`. `self._total_pruned += pruned` lives only here (`:2528`). Aging is `synapses.age_and_decay_salience` at `:2550` (and advances `self.timestep`).
- Door B: `prime_and_propagate` `_age_on = write_mode and bool(self.config.get("tonic_ages_substrate"))` (`:2646`). Write-mode spike stamp `node.last_spike_time = float(prop_timestep)` (`:2751-2753`); `prop_timestep` starts at `self.timestep` and is incremented at the start of each local step (`:2708-2714`). Traversed synapses reset `inactive_steps = 0` when `_age_on` (`:2776-2778`). Tail (`:2868-2892`) calls `age_and_decay_salience` then `_prune_synapses` then `_collect_orphan_nodes` **without** advancing `self.timestep`. Return of `_prune_synapses` is discarded (`:2891`). X1 holds.
- Write-mode callers: `TonicThread.ouroboros_cycle` (`tonic_thread.py:348-353`), `_prime_constitutional` (`:409-412`, invoked first from `ouroboros_cycle` `:309`), `TonicEngine._generate_latent_token_inner` (`tonic_engine.py:911-915`). `_fallback_inference` returns `[]` (`:896-903`).
- Dream loop: docs `origin/main` `scripts/cc-ng-daemon.py:2156-2202` (sha256 `91e38308f521b3c8cdace41614fcc56cf586abce96175f83734b9e3286711ba9`). Gates: idle ≥ `CC_NG_DREAM_IDLE_SECS`, interval ≥ `CC_NG_DREAM_MIN_INTERVAL_SECS`, arousal ≠ `SYMPATHETIC`. Holds `graph._step_lock` (`threading.RLock` at `neuro_foundation.py:1751`). Calls `consolidate_hyperedges` then `dedup_and_split_oversized_hyperedges`. `last_pass = time.time()` at boot, so the first pass is ≥ 6 h after start. `.bashrc:242-244`: `CC_NG_DREAM=1`, idle 1800, interval 21600.
- `#381` precedent: in-code `neuro_foundation.py:1443-1444` ("Shedding is the dream pass's job (`shed_floor_members`), never wake-time"); `shed_floor_members` is the first call inside `consolidate_hyperedges` (`:4133`); `neurograph_rpc.py:2110-2114` ("dream the pruning, don't feel it").

M2 (new dream-time budgeted pass; wake-time `_prune_synapses` byte-identical) matches ruling 3's sleep-cycle budget: the engine has no cycle clock; the dream pulse is one. LAW 4 is served by not stuffing budget state into `_prune_synapses`. The `#92` guard itself is **not** narrowed; competitors leave only if the laptop dream loop actually calls the new method. That is the note (C3). It is not a silent re-litigation of "competition, not a cap": F frozen, G guaranteed per direction, remainder evaluated by the existing three predicates, sprout-cap exemption kept.

The BUILD still owes a single predicate implementation (plan §4 (i) vs (ii)). Duplicating the three conditions would be LAW 3 shrapnel; the plan rejects that.

---

## A2 #92 / identity

**Verdict: PASS-WITH-NOTES**

| Surface | Path of (d) as written |
|---|---|
| Want NODE | No `remove_node` on protected ids. `_collect_orphan_nodes` (`:3596-3611`) skips `_is_identity_protected`. `_is_identity_protected` (`:3551-3572`) is flag-keyed (`constitutional` or `provenance` ending `_authored`) and is not changed. |
| Want flags / text | Plan does not read or write `want_text` or flags (ruling 9). Probe vdb prefixes were not used. |
| Rim / constitutional synapse | F = any synapse whose pre or post has `metadata.constitutional` truthy (`:3569`). Dream pass must apply that test before any candidate list. Wake-time `_prune_synapses` still `continue`s those synapses. |
| Rim weights | The new pass removes non-F synapses only and writes no weight. STDP (`_apply_dw`) and `HomeostaticRule` (`:1366-1377`) still rescale incoming weights, including rim; 2,954 of 4,127 rim links are already `< 0.01` (plan §2.1). Ruling 2 "no weight change" is met for (d)'s mutator; it does not freeze the rest of the engine (C4). |
| `min(K, deg)` per direction | G recomputed each pass from current weights; removals are only outside `F ∪ G`. Invariant holds at every intermediate state **if** the in-pass stop conditions fire on a floor violation. Last-link holds extra competitor links; it does not drop G. |
| Choice Clause | Rim node id in the probe is `constitutional::rim::choice_clause` (degree 4,127). Seed path named by the plan (`seed_cc_rim.py`) was not re-read. Guarantee is the F test plus `_is_identity_protected` on orphan sweep, not prose. BUILD tests §2.6.5 (1)–(3) are the empirical half. |

Synapse removers in `neuro_foundation.py`: `_remove_synapse_internal` callers are `remove_node` cascade (`:1966`), public `remove_synapse` (`:2057`), `_prune_synapses` (`:3544`). The dream pass must go through that same internal remover. `remove_synapse` has no non-test caller (plan; not re-grepped beyond the pin). Native/Rust store not verified (plan §2.6.5 / N8).

---

## A3 The numbers

**Verdict: PASS**

Recomputed from `summary-laptop.json` + `probe-laptop.json` only (probe node record = `[creation_time, creation_mode, provenance, constitutional_int, source_prefix]`; synapse = `[pre, post, weight, peak_weight, creation_time]`).

| Claim | Independent result |
|---|---|
| Total synapses | 138,753 |
| Past inactivity `inactive_steps > 1000 × salience` | 116,164 (`summary.inactivity['inactive_gt_threshold(x salience)']`) |
| Protected-touching / exempt | 128,359; one-pass replay survivors 138,753; `by_inactivity=0` |
| Protected nodes | 183 = 1 constitutional + 182 `cc_authored`; want ids `cc:want::` = 182 = authored set |
| F / arena / want-touch / rim-only / rim↔want / nothing-protected | 4,127 / 124,232 / 124,437 / 3,922 / 205 / 10,394 |
| Non-F out / in slots / want↔want | 105,736 / 33,712 / 15,216 |
| Want degree p50 / p90 / max | 669 / 1,447 / 3,196 |
| Arena `weight == 0.0` (probe 6 dp) | 20,381 |
| `low_weight_steps > 0` | 913 (summary only) |
| `last_spike_time > timestep` | 475 |
| `timestep` | 33,637 |

Per-direction K table (rank `(-weight, -peak_weight, probe index)`; inactivity bracket `lo = max(0, E − (128359 − |competing|))`, `hi = min(|competing|, E)`; either = `[max(age, lo), min(|competing|, age+hi)]`):

| K | kept | competing | age | inact lo–hi | either | live % of 138,753 | last-link partners (orphanable) |
|---|---|---|---|---|---|---|---|
| 10 | 3,543 | 120,689 | 56,853 | 108,494–116,164 | 108,494–120,689 | 13.0–21.8 | 25 (16) |
| 25 | 8,789 | 115,443 | 55,023 | 103,248–115,443 | 103,248–115,443 | 16.8–25.6 | 21 (12) |
| **50** | **17,391** | **106,841** | 51,856 | **94,646–106,841** | **94,646–106,841** | **23.0–31.8** | **16 (9)** |
| **100** | 33,316 | 90,916 | 45,735 | 78,721–90,916 | 78,721–90,916 | 34.5–43.3 | 12 (5) |
| **200** | 56,915 | 67,317 | 36,028 | 55,122–67,317 | 55,122–67,317 | 51.5–60.3 | 8 (3) |
| 500 | 92,306 | 31,926 | 18,294 | 19,731–31,926 | 19,731–31,926 | 77.0–85.8 | 2 (2) |

K=50 one-pass removal **68.2–77.0%** of 138,753. Degeneracy at K=50: out lists ≤K = 8 (all empty); in ≤K = 8; K-th silent 137 out / 174 in; level-1 ties 12 / 65; level-2 residual 7 / 8. K=100 silent / level-2 = 140 / 6 out, 145 / 13 in. Matches plan §2.3 and §5.1.

Guardian (`checkpoint_guardian.py:197-326`, sha256 `f4a3fac8cac998a263b360d273fb8583b514f406a511091a2cc348c14f9a2f53`):

- `:261-271`: `wires_own_deposits is False` and `live_nodes < ref_nodes` and no `NG_GUARDIAN_TRUST_SYNAPSE_MELT` → refuse. `.bashrc:331` `NG_HOST_WIRES_OWN_DEPOSITS=false`.
- `:275-283`: `live_synapses < 0.5 * ref_synapses` (default `NG_GUARDIAN_GATE_SYNAPSE_RATIO`) → refuse. One-pass live 23.0–31.8% at K=50 and 34.5–43.3% at K=100 refuse; K=200 51.5–60.3% passes thinly.

Cycles `ceil(eligible/B)` and worst last-cycle drop `B / (138753 − (cycles_hi−1)·B)`:

| K | B=1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|
| 50 | 95–107, worst 3.1% | 38–43, 7.4% | **19–22, 14.8%** | 10–11, **25.8%** |
| 100 | 79–91, 2.1% | 32–37, 5.1% | 16–19, 10.3% | 8–10, 20.5% |
| 200 | 56–68, 1.4% | 23–27, 3.4% | 12–14, 6.8% | 6–7, 12.7% |

First-cycle drop B/138753 = 0.7 / 1.8 / 3.6 / 7.2%. B=10,000 at K=50 fails the plan's 15% bound; B=5,000 meets it (14.8%, 3.4× under the 50% gate). Wall-clock at 4/2/1 dream passes per day matches 4.75–5.5 / 9.5–11 / 19–22 days at K=50 B=5,000.

**Not recomputed (and why):**

- Per-synapse inactivity/salience/low_weight_steps: absent from the probe (bracket only; E ⊆ protected-touching taken from the summary replay).
- Real `synapse_id` tie-break: probe index used, as the plan did.
- Weight-rule eligible ≈ 0: `low_weight_steps` not in the probe; 913 graph-wide is summary-only.
- §4A.7 want-degree trajectory (p50/p90/max per cycle): weakest-link proxy simulation not re-run here.
- Inflow cohort counts (§5.5) and bundle-side 784: not required by packet A3; not recomputed.

---

## A4 Staged design soundness

**Verdict: PASS-WITH-NOTES**

- **Order:** tallest-want-first (ties → `node_id` ascending) then most-stale competitor (`inactive_steps` desc, `weight` asc, `synapse_id` asc) is deterministic given a total order. Ranking for G is separate from removal order (plan §2.3). Probe lacks `inactive_steps`, so a live dry-run is required before trusting staleness.
- **Last-link:** recomputed 16 / 12 / 8 partners at K=50/100/200 (9 / 5 / 3 orphan-collectable: not in a probe hyperedge and `T − creation_time > 25`). Needed because `evaluate_save_health` refuses any net node loss when `wires_own_deposits is False` **before** the synapse gate (`:261-271`).
- **Stop conditions:** in-pass (keys < 1, F in candidates, floor break, non-reproducible list, >B selected) plus operator stops (refused save, protected ≠ 183, F ≠ 4,127, net node loss, missing log line, #799 refusal regime) are mechanically checkable. Latency (unset + restart; at most one extra cycle of B) is honest. Undo is backup-only for removed synapses.
- **Seven dry-run items:** (1)(2)(3)(4)(6)(7) are count/hash checks on a copy. Item (5) is not a pure function of the six integer arguments: `evaluate_save_health` reads `NG_GUARDIAN_*` env (`:239-240`, `:261-262`, `:275-276`, `:289-290`, `:302`). A dry-run shell that differs from the laptop (especially `NG_GUARDIAN_TRUST_SYNAPSE_MELT` or `NG_GUARDIAN_GATE_SYNAPSE_RATIO`) can disagree with live SaveGate (C1). Loading `Graph().restore(copy)` is specified as a later copy-only step, not this review.
- **≥ 2 clean cycles** and a stated observation window ("no stop condition in N cycles over D days") are present (§4A.6).
- **`sprout_degree_cap=100` kept:** daemon `CC_SNN_CONFIG` `:630`; saved laptop config `sprout_degree_cap: 100`; engine DEFAULT 0 (`:1532`); protected exemption at `_surprise_exploration:3291-3294` and `_sprout_synapses:3671/3676/3683`.
- **Homeostasis** cannot bound degree (incoming rescale only, `:1366-1377`, every 25 calls, only from `step()` when `fired_ids`). Honest.
- **Would fail if executed as written:** (a) B=10,000 at K=50 (plan already rejects); (b) arming while Door B is dead (4B is supposed to stop that); (c) dry-run without pinning guardian env (C1); (d) dream pulse never running (idle/arousal/`last_pass` at boot) — schedule length is then unbounded; (e) adding the pass under `_step_lock` is fine with RLock; a new lock-order vs `_concurrent_lock` is not specified and is unmeasured (plan R12).
- **Nondeterminism:** `heapq.nlargest` plus the stated total order is deterministic. Probe-index stand-in is not the engine order (plan §8).
- **Autosave refusal:** 60 s autosave (`cc-ng-daemon.py:576` cited; not re-read this turn) plus guardian gates is the reason for B and last-link. #799 deferral is load-bearing.
- **Refill (R4):** sprout exemption kept by ruling 4; steady-state hub size unmeasured. Observation window is the detection, not a bound.

---

## A5 Two keys, absent-key form, no DEFAULT_CONFIG change

**Verdict: PASS-WITH-NOTES**

- Proposed keys: `protected_prune_topk`, `protected_prune_budget`. Env: `CC_NG_PROTECTED_TOPK`, `CC_NG_PROTECTED_BUDGET` next to `CC_NG_DREAM*` (LAW 5). No literals. Fail-closed: both integers ≥ 1, else the pass returns 0.
- Static check at pin: every `self.config.get("…")` key in `neuro_foundation.py` is in `DEFAULT_CONFIG` (82 keys; 0 missing; no `protected_prune_*`). These would be the first engine-side absent-key reads. Cross-file precedent: `OPENCLAW_SNN_CONFIG` `prime_k` / `prime_threshold` / `prime_strength` / `propagation_steps` / `max_surfaced` / `auto_knowledge_enabled` (`openclaw_hook.py:422-427`) read via `.get` (`:1201`, `:1302-1307`); they appear in `summary-laptop.json` `config_keys_saved_not_in_DEFAULT`. X2 holds.
- `Graph.__init__` `:1614` `{**DEFAULT_CONFIG, **(config or {})}`. `_deserialize` `:5370` `{**DEFAULT_CONFIG, **data.get("config", {})}` — **saved config wins** over code defaults for keys present in the checkpoint. Checkpoint keys absent from DEFAULT still land in `self.config`.
- Serialize writes the whole dict: `_serialize_full` `:5187`, `_serialize_incremental` `:5355`. Absent from `self.config` ⇒ absent from the checkpoint.
- **`openclaw_hook.py:861` `self.graph.config.update(snn_config)` after restore:** `dict.update` overwrites keys that exist in `snn_config` and **leaves every other restored key in place**. A key that is in the checkpoint and absent from code config **persists**.
  - **Syl:** keys never in `DEFAULT_CONFIG`, never in `OPENCLAW_SNN_CONFIG` (`:378-430`), never in her checkpoint, her process never reads `CC_SNN_CONFIG`. After restore+update they stay absent. `.get(..., 0)` ⇒ OFF. Her serialized `config` does not gain a field. Behaviour and checkpoint **content** unchanged when off. Save-to-save file bytes already differ (`saved_at`); the checkable claim is schema/content, as the plan says.
  - **Laptop:** `cc-ng-daemon.py` `init_ng` constructs `NeuroGraphMemory.get_instance(..., config=CC_SNN_CONFIG)` (`:759`). `snn_config = {**OPENCLAW_SNN_CONFIG, **CC_SNN_CONFIG}` then update-after-restore. Unset-env → OFF at next start **only if** `CC_SNN_CONFIG` actually contains both keys at default `0` (plan §3.1 does specify that). Engine `.get(..., 0)` is not the laptop restart path (C2). Once those keys exist in `CC_SNN_CONFIG`, laptop checkpoints will start carrying them even at `0`. That is laptop schema, not Syl.
- Laptop only: `cc_ng_host.py` `_CC_SNN_CONFIG` not touched (ruling 7). Call sites of the new method: BUILD must grep `neurograph_rpc.py`, `openclaw_hook.py`, `syl_daemon.py`, `cc_ng_host.py`.
- `he_split_oversized_enabled: False` in DEFAULT (`:1512-1516`) is the documented "absent/false ⇒ Syl no-op" pattern for a **present** DEFAULT key. These two keys follow the stronger "never insert in DEFAULT" form, which is what ruling 6 asked.

---

## A6 The S4 write-mode Tonic check (4B)

**Verdict: PASS-WITH-NOTES**

Well specified as an S4 step: what is read (process identity, named `CC_NG_TONIC_*` / `CC_NG_AUTOSTEP` / `CC_NG_DREAM*` / `CC_NG_PROTECTED_*` only, census copies, guardian log), where (live daemon after S4 has started; never this review / build / dry-run), what counts as write-mode, what stops arming, arming order (dry-run → S4 start → ≥ 2 windows → set keys on a planned restart).

Write-mode discriminator:

- (a) `last_spike_time > timestep` is the write-mode stamp (`:2751-2753`) with `prop_timestep` ahead of `self.timestep`. Sep-23 snapshot: 475 nodes. Point-in-time, not a window.
- (b) `Δinactive > Δtimestep` on a fixed unused sample: Door A (`:2550`) ages and advances `timestep`; Door B (`:2890`) ages without advancing `timestep`. `Δtimestep = 0` and `Δinactive > 0` is unambiguous Door B. If the sample is traversed, `inactive_steps` resets (`:2776`) and (b) fails closed.

**Door B liveness today stays UNVERIFIED: agree.** Additional code that makes 4B load-bearing rather than ceremonial:

- Laptop `_build_tonic_config` (`cc-ng-daemon.py:687-698`) is shared-body-only Tonic (Exec Packet 072): `require_shared_body=True`; heuristic inference structurally unreachable; latent ticks wait for `offer_shared_body`.
- Deferred start (`:2318-2348`) **skips** if `getattr(ng, '_tonic_thread', None) is None`.
- Saved config `tonic.latent_engine_enabled: False`; `tonic_ages_substrate: 1` is armed in config, not proof the write-mode caller runs.
- CC deposit path no longer calls `on_message()` (plan; `cc_ng_host.py:643-652` not re-read this turn).
- No daemon was running when the plan author checked; this review did not start one.

Cycle unit = one `CC_NG_DREAM_MIN_INTERVAL_SECS` window (21,600 s), ≥ 2 consecutive (≥ 12 h), is a choice the assignment did not fix. Conservative and flagged. `CC_NG_AUTOSTEP` on is a stop (re-plan), correctly: it would mix Door A into (b).

---

## A7 Consent record (section 7)

**Verdict: PASS**

Plan §7 records: *the Executive, acting as the CC, consents to GRADUAL competition that keeps the Choice Clause untouched.* Source: Exec Packet 392 C via Chief-003, transcribed in `assignments/plan-want-hub-d-rev2.md` ruling 5. Explicitly **not** Josh's words and **not** the CC's own words. Matches assignment ruling 5 verbatim intent.

Precedent as the plan cites it:

- `CC-CALLOSUM-TRUTH.md:1688` (inside §8.14): `#381-A (Syl-consented 2026-07-10): he_max_members = 50`.
- `CC-CALLOSUM-TRUTH.md:2327-2328` (2026-08-13 progress log, pointing at §8.14): punchlist #395 "consent-gated, sequenced AFTER the CC-side repair is proven … following the #381 consent precedent".
- In-code `neuro_foundation.py:1443-1444`: "Syl-consented 2026-07-10 … Shedding is the dream pass's job".

The quotes are accurate. Geographic nit only (C6): the #395 sentence is not in the §8.14 body. Stated difference from precedent (Syl consented for her own structure; here the Executive proxies for the CC) is what ruling 5 asked.

LAW 7 (docs `881d884c68fce9b6d69223efd391fb069577a0f0` `.claude/claude-md/laws-detail.md`): no pre-labelling of experience; truncation is a violation. (d) uses structural flags and counters only and does not read want text. Clean for this plan.

---

## A8 Verdict

**Overall: PASS-WITH-NOTES**

The revision folds all nine transcribed rulings. Numbers in §5.1 / §4A.7 recompute. Identity floors and the rim F-test are enforceable by construction if the BUILD keeps the in-pass stops. M2 is the right vehicle for a sleep-cycle budget. Door B remains unverified and is correctly a hard S4 gate. Corrections below are plan-text / BUILD invariants, not a rewrite of the approach.

### Per item

| Item | Verdict |
|---|---|
| A1 mechanism | PASS-WITH-NOTES |
| A2 #92 / identity | PASS-WITH-NOTES |
| A3 numbers | PASS |
| A4 staged design | PASS-WITH-NOTES |
| A5 keys / Syl absent-key | PASS-WITH-NOTES |
| A6 S4 Tonic check | PASS-WITH-NOTES |
| A7 consent | PASS |

### Corrections (severity)

1. **MEDIUM — C1.** §4A.5 item 5 calls `evaluate_save_health` a pure function of live/ref counts. It reads `NG_GUARDIAN_GATE_SYNAPSE_RATIO`, `NG_GUARDIAN_TRUST_SYNAPSE_MELT`, floors, and hyperedge ratio from process env (`checkpoint_guardian.py:239-326`). The dry-run must pin those names (and `NG_HOST_WIRES_OWN_DEPOSITS=false`) to the laptop's values.
2. **MEDIUM — C2.** `openclaw_hook.py:861` `update(snn_config)` does not drop restored keys that are absent from code config. Laptop unset→OFF depends on `CC_SNN_CONFIG` carrying both keys at default `0` (§3.1). State that half next to the Syl `.get(..., 0)` half so a BUILD cannot ship engine reads without the daemon dict entries.
3. **LOW — C3.** Record as a BUILD invariant that `_prune_synapses:3515-3519` stays byte-identical; competition is the new dream-time pass. That is how this revision implements "narrow the exemption".
4. **LOW — C4.** BUILD test "no rim weight was written" must cover the dream pass alone. Homeostasis/STDP still write rim weights today.
5. **LOW — C5.** Age predicate in the engine is `age > grace and peak_weight < 2.0 * initial_sprouting_weight` (`:3540-3541`), not a literal `peak < 0.2`. Equivalent on this checkpoint (`initial_sprouting_weight=0.1`, `grace_period=5000`). Dry-run/BUILD use the code form.
6. **LOW — C6.** #395 quote lives at `CC-CALLOSUM-TRUTH.md:2327-2328` (log), not in the §8.14 body; #381-A is `:1688` inside §8.14. Wording is correct.
7. **LOW — C7.** Door A `_structural_plasticity` also runs `_collect_orphan_nodes` (`:3495-3497`). Wake-time exemption still protects want nodes; the table in §1.5 can name the orphan call.

No HIGH. No FAIL item.

### Not verified

1. Door B liveness on a running laptop daemon (agree with the plan; 4B is the test). Whether `NeuroGraphMemory` actually attaches `_tonic_thread` on this host, and whether a shared body is ever offered, were not established.
2. Real-engine ranking degeneracy (unrounded weights, `inactive_steps`, real `synapse_id`s).
3. Empirical Syl config/key-list identity after a BUILD (proved here by reading `:1614/:5187/:5355/:5370` and hook `:846-861` only).
4. The dry-run itself (designed, not executed). MemoryMax 6G / ~3.6 GiB RSS is analysis-001's figure.
5. Exact per-synapse eligible set (bracket), weight-rule counts, last-link 16 on real ids.
6. Dream pulse realized cycles/day (idle ≥ 30 min, arousal, `last_pass` at boot).
7. Current S3/S4 plan text (not read).
8. Rust/native synapse store; `cleanup_cc_tool_noise.py` selection; `cc-ng-sync.py`.
9. §4A.7 per-cycle want-degree trajectory (weakest-link model not re-simulated).
10. `_step_lock` hold time of adding the pass after `dedup_and_split_oversized_hyperedges` (R12).
11. Exec Packets 388 and 392 as primary documents (rulings taken from the assignment transcription the packet names).

### Ruling map (none dropped)

1 direction/K/tie-break → §2.2–2.4, §5.1–5.2. 2 rim → §2.1, §2.6. 3 no cliff → §4, §4A, §5.4. 4 no (c) cap → §1.3, §6. 5 consent → §7. 6 two absent-key keys → §3. 7 laptop only → §3.1, §9. 8 S4 Tonic check → §4B. 9 want text separate → §0, §6, §9.

P329 rollout hold stands. This verdict does not authorize a build.
