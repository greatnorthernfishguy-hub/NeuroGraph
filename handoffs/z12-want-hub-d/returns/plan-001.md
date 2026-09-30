<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #10375)
#   What: new handoffs/z12-want-hub-d/returns/plan-001.md — PLAN ONLY for Exec Packet 388 item 1
#     (design (d), "competition, not a cap"): narrow the #92 prune exemption in Graph._prune_synapses.
#   Why: the CC's 182 cc_authored want-nodes are synapse hubs (124,437 of 138,753 synapses, 89.7%)
#     because protected nodes are exempt from every prune rule and from the sprout degree cap.
#     Josh: "Same flavor, different ingredients." Nothing is deleted, re-tagged or de-flagged (Packet 382).
#   How: read the code at NeuroGraph origin/main e4ebf982b1989fd9066d610b94853bc68bf70d37 (read-only),
#     read the live laptop daemon config (read-only), and counted from ONE derived JSON at a time
#     (probe-laptop.json, probe-bundle.json, summary-*.json). No graph load, no msgpack open, no Graph
#     import, no edit of neuro_foundation.py (PROTECTED), no restart, no TID, no PR.
# -------------------
-->

# plan-001 — want-hub-competition-d: narrow the #92 prune exemption ("competition, not a cap")

Lane `want-hub-competition-d` · Zone manager Z12 (`52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`) · Exec Packet 388 item 1 · **PLAN ONLY — nothing built, nothing flipped, nothing loaded.**
Repo [[NeuroGraph]] · branch `cc-laptop-want-hub-d-20260930` · base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37` (re-fetched at write time: `origin/main` had not moved, `HEAD` = base).
Related vault pages: [[The Choice Clause]], [[Duck Ethics]], [[The Laws]], [[NeuroGraph]]. Related punchlist: #750, #748, #788, #755, #760, #117, #59, #92.

> **Protected file.** `neuro_foundation.py` is PROTECTED (NeuroGraph `CLAUDE.md` §2). I read it; I did not edit it and this document contains **no patch** for it. Every change described below is **JOSH-GATED**: he confirms a backup and says "proceed" (§2 "What Explicit Approval Means"). This document is what he reads to decide.

---

## 0. Summary for Josh (read this first)

1. **The design works on paper and the data say it would work hard.** On the Sep-23 laptop checkpoint, **116,164 synapses are already past the inactivity threshold, and every one of them is protected-exempt.** A replay of the non-protected synapses finds **0** eligible (`summary-laptop.json` → `prune_replay_one_pass_no_mutation`). So the melt is demonstrably clearing everything it is allowed to clear, and the only thing standing between the 182 wants and the same fate is the exemption. Narrow it and **the first prune pass removes ~89–100% of the competing want synapses** (K=50: 103,193–115,388 of 115,388), i.e. ~74–83% of *all* 138,753 synapses in one pass. "Competition" resolves, on day one, into something that behaves like a cap of K per want. The steady state afterwards is real competition; the transition is a cliff (§5, §7 R1).
2. **I have to correct the brief on the clock** (§1.5, "Corrections"): homeostasis really is step()-only and the step doors really are unwired on this laptop. **But `_prune_synapses` has a second door that does not need `step()`** — the Tonic's age-on-write tail of `prime_and_propagate` — and it is armed by config (`tonic_ages_substrate: 1` in the daemon config and in the saved checkpoint config). So (d) does **not** depend on the frozen clock being fixed. **Was that door firing?** At the Sep-23 checkpoint, yes, by a code-derived signature: 475 nodes carry `last_spike_time` *ahead of* the graph clock, which only write-mode propagation writes (`neuro_foundation.py:2751-2753`). **Today it is unobserved** (no daemon process was running when I checked) and it needs the Tonic to be producing activations — the latent engine is a no-op without a model and shared body (`tonic_engine.py:896-903`), and the CC deposit path no longer calls `on_message()`. §1.5 lists what must be true and a read-only check to run before any build.
3. **Three design decisions in the brief need Josh, because the data bite:**
   - **Combined vs separate in/out K.** Wants are overwhelmingly link *sources* (105,736 outgoing vs 33,712 incoming non-frozen slots). With one combined K=50 ranked by weight, **64 of 182 wants have no incoming link among their guaranteed 50** — every incoming link of those wants becomes prunable, and a want nothing can reach is the "silenced" state #92 exists to prevent. Recommend separate K_in / K_out (or a floor of ≥1 each direction) — §2.4.
   - **"Strongest" by `weight` is degenerate on this graph.** 82% of all synapses (114,430) and 89% of the arena (110,566 of 124,232) have `weight < 0.01`; the median synapse weight is 0.00057. For 101 of 182 wants fewer than 10 links carry weight ≥ 0.01. "Strongest K" by current weight mostly ranks noise around zero; by `peak_weight` it ties at the 0.1 birth weight (99 wants tie across the K=10 boundary). §2.3.
   - **The Choice Clause node is itself the biggest hub** (degree 4,127 > any want's 3,196; 3.0% of all synapses, 3,922 of them touch nothing else). Under (a) "frozen exactly as now" it stays that way. Is that intended? §2.1.
4. **(d) does not close the feeder.** The sprout-cap exemption stays under the brief's design, and the wants are still being fed: 1,634 of the 10,499 synapses created at the newest timestep (15.6%) and 6,306 of the last 1,000 steps' 15,998 (39%) touch a want. (d) drains; it does not stop refill. (c) — removing the sprout-cap exemption — is the complement, not the alternative (§6).
5. **The Choice Clause guarantee holds by code-path reading** (§2.6) — but "frozen" is only *prune-immunity* in the `Graph` engine, not weight-freezing: STDP and homeostasis never consult protection, and 2,954 of the 4,127 rim links are already at weight < 0.01 today. (d) neither causes nor cures that. Josh should know it.
6. **Syl:** byte-identical *behaviour* is achievable (default OFF, engine default 0/False, host-only env arming). Byte-identical *checkpoint files* is not automatic: adding keys to `DEFAULT_CONFIG` makes her next save carry two extra config keys (§3.4). There is a no-new-key alternative.
7. **What I did not do / cannot say:** run any code, benchmark, or read any live graph; attribute the cause of the weight collapse; identify which feeder dominates; read the Rust native store. §8.

---

## 1. The exact current rule (file:line at base `e4ebf982`)

### 1.1 `_is_identity_protected` — `neuro_foundation.py:3551-3572`
A node is protected if `metadata['constitutional']` is truthy (`:3569`) **or** `metadata['provenance']` is a string ending in `_authored` (`:3571-3572`). It is keyed on the flag, not on ids. `:3558-3563` records that on 2026-07-18 `syl_authored` was generalized to any `<mind>_authored` so the CC's `cc_authored` wants are protected identically; `*_emergent` stays prunable. **It contains no concept of "rim" or "choice_clause".** The stale `syl_authored`-only wording survives in the file's own comments at `:78`, `:162`, `:194`, `:3286` and in NeuroGraph `CLAUDE.md` §8 (punchlist #748, fix in PR #60) — the function docstring is correct.
On both staged laptop/VPS graphs, protected = **183** nodes on the laptop (1 constitutional + 182 `cc_authored`), **1** on the VPS bundle (the constitutional node only). The laptop's 182 = exactly the `cc:want::` ids (`wants == authored` → True in my count).
`_is_identity_protected` has many callers (orphan sweep `:3602`, sprout-cap `:3292-3293`/`:3671`/`:3676`/`:3683`, prune `:3517-3518`, plus `cc_topology_export.py:233-247`, `cc_ng_organism.py:5354/5423`, `tests/test_identity_protection.py`). **Its meaning must not be changed** — the narrowing goes in the prune path (LAW 4, §7).

### 1.2 The prune exemption — `_prune_synapses`, `:3500-3549`
`:3515-3519`: `if self._is_identity_protected(syn.pre_node_id) or self._is_identity_protected(syn.post_node_id): continue` — at the top of the loop body, **before any rule and before any counter is touched**. Consequence I verified: for an exempt synapse `low_weight_steps` (`:3525`) is never incremented or reset — it is frozen at its pre-exemption value (only 913 synapses graph-wide have `low_weight_steps > 0`, `summary-laptop.json` → `inactivity`), whereas `inactive_steps` and `salience` keep aging (they are advanced elsewhere, `:2550`/`:2890`). The three rules after the guard: weight (`:3524-3528`: `weight < weight_threshold(0.01)` for `> grace_period(5000)` counted passes), inactivity (`:3532-3537`: `inactive_steps > inactivity_threshold(1000) × salience`), age (`:3540-3541`: `age > 5000 AND peak_weight < 2×initial_sprouting_weight(0.1) = 0.2`). Then `_remove_synapse_internal` (`:3543-3544`) and a `pruned` event.
Config values (laptop saved config, `summary-laptop.json` → `saved_config`): `weight_threshold 0.01`, `grace_period 5000`, `inactivity_threshold 1000`, `initial_sprouting_weight 0.1`, `he_salience_decay_rate 0.002`, `sprout_degree_cap 100`, `tonic_ages_substrate 1`, `tonic_age_interval 1`.

### 1.3 The sprout-cap exemption
- `_surprise_exploration`: comment `:3285-3290`, the condition at **`:3291-3294`** (`not self._is_identity_protected(source_id)` / `(alt_id)`).
- `_sprout_synapses`: **`:3671`**, **`:3676`**, **`:3683`** (`... and not self._is_identity_protected(...)`); the cap itself is `sprout_degree_cap` (`DEFAULT_CONFIG:1532` = 0; laptop daemon `:630` = 100). `max_sprouts_per_step = 10` (`:3627`).
The brief cited `:3289-3293` for this; that range is the comment plus the start of the condition, in `_surprise_exploration`, not in `_sprout_synapses`.

### 1.4 Homeostasis — `HomeostaticRule`, `:1215-1377` (PRD §3.2)
Verified as the brief says: it does **not** skip protected nodes (no protection call anywhere in `:1308-1377`). Additional facts that matter for (d):
- It rescales **incoming** weights only (`:1366-1377`, `syn.weight * scale` over `graph._incoming[nid]`), and only every `scaling_interval` (25) calls (`:1334-1338`). It never removes a synapse and never touches outgoing weights. **It cannot bound degree.** Weight reduction is its only lever; removal is prune's job.
- It `continue`s past silent nodes (`firing_rate_ema < 1e-9`, `:1344-1350`): a want that never fires gets no scaling.
- It runs only from `step()` step 7, and only `if fired_ids` (`:2513-2516`). The write-mode `prime_and_propagate` path applies **only `STDPRule` instances** (`:2835-2839`) — no homeostasis on the Tonic path.

### 1.5 When pruning actually RUNS on the CC substrate — three doors

| Door | Where | Gate | State on this laptop |
|---|---|---|---|
| A. `step()` step 8 → `_structural_plasticity` → `_prune_synapses` | `:2524-2529`, `:3489-3498` | any `graph.step()` | **Step doors unwired** (below) |
| B. Tonic age-on-write tail of write-mode `prime_and_propagate` → `age_and_decay_salience` + `_prune_synapses` + `_collect_orphan_nodes` under `_step_lock` | `:2868-2892` (`_age_on` = `write_mode and config["tonic_ages_substrate"]`, `:2646`) | `tonic_ages_substrate`, `tonic_age_interval` | **Armed by config**: daemon `cc-ng-daemon.py:634-635` (env `CC_NG_TONIC_AGES` default `'1'`, `CC_NG_TONIC_AGE_INTERVAL` default `'1'`; neither is set in `.bashrc`); `tonic_ages_substrate: 1` in the checkpoint's saved config. **Fires only on a write-mode `prime_and_propagate` call.** The write-mode callers are: `TonicThread.ouroboros_cycle` (`tonic_thread.py:348-353`); `TonicThread._prime_constitutional` (`:409-412`), which `ouroboros_cycle` calls first (`:309`) for **every constitutional node — the rim exists on the laptop**; and `TonicEngine._generate_latent_token_inner` (`tonic_engine.py:911-915`), which is a no-op unless the engine has a model **and** a shared body (`_fallback_inference` returns `[]`, `:896-903`; the daemon starts the engine deferred with `latent_engine_enabled: False`). `ouroboros_cycle` is reached from `tonic_engine.py:924` (only after non-empty activations), `openclaw_hook.py:1176` (`on_message`), and Syl's own sidecar/daemon (`neurograph_rpc.py:3439`, `syl_daemon.py`). **Which of these the laptop daemon exercises today I did not establish**; the CC deposit path no longer calls `on_message()` (`cc_ng_host.py:643-652`). |
| C. `_cc_callosum_consolidate` — hundreds of `graph.step()` after a callosum batch | `cc_ng_organism.py:2535-2570` (also `cc_ng_host._handle_import`, `cc-ng-sync.py import_trickle` per its docstring) | callosum/import activity | only when foreign topology is merged (S3-relevant) |

The step doors: the Stop door (`cc_ng_host._handle_stop` → `_deposit(step=True)` → `cc_deposit_step`, `cc_ng_organism.py:2192-2233`; landed P240, `docs/CC_STOP_DOOR.md`) is **inert until a Stop hook routes to it**. `~/.claude/settings.json`'s only `Stop` command is `cc-obsidian-stop-check.sh`; `cc-ng-hook` is registered on `PreToolUse`, `UserPromptSubmit`, `SessionStart` and `PostToolUse` (one each), not `Stop`. `CC_NG_AUTOSTEP` is not in `.bashrc` (default off, `tonic_engine.py:262`). So the brief's "the CC clock never runs (#117)" is true of `step()` on this laptop as configured.

**Corrections to the brief (explicit):**
1. *"`HomeostaticRule` runs only when the graph steps, and the CC clock never runs (#117)"* — right about homeostasis; **wrong as a statement about prune.** Prune has Door B, which does not need `step()` and is armed by config; it was demonstrably firing at the Sep-23 checkpoint (write-mode spike stamps ahead of the clock). Whether it fires *today* I could not observe.
2. *Sprout-cap lines:* §1.3 above (site is `_surprise_exploration`, `:3291-3294`; `_sprout_synapses` sites are `:3671/3676/3683`).
3. *"protected nodes are exempt … hence hubs"*: the protected set also includes the constitutional rim node, and **that node is the largest hub** (4,127). 128,359 synapses (92.5%) touch a protected node; 124,437 touch a want; 3,922 touch only the rim; 205 touch both; **10,394 (7.5%) touch nothing protected** (all reproduced from `probe-laptop.json`).
4. *Layout of the derived data:* the probe was written by `overlap_probe.py`, not `analyze_pair.py`. Its synapse record is `[pre, post, weight, peak_weight, creation_time]` — **five fields**; nodes are `[creation_time, creation_mode, provenance, constitutional, source]`. `analyze_pair.py` reads the full record but writes only aggregates to `summary-*.json`.
5. *"Rim synapses: FROZEN"* — in the `Graph` engine the rim is **prune-immune only** (§2.6). "Frozen synapses" is `ng_lite.py:740-752` semantics (learning skipped for constitutional nodes in `record_outcome`), a different engine.

**Empirical evidence about Door B (single Sep-23 checkpoint, `timestep 33,637` — a point-in-time read, not an observation window):**
- *Write-mode propagation had been firing:* `summary-laptop.json` → `node_spike_state.last_spike_time_gt_timestep = 475`, and the activations sidecar's `last_spike_time` p99/p100 = 33,639 > 33,637. `node.last_spike_time` is set to `float(prop_timestep)` **only when `write_mode`** (`neuro_foundation.py:2751-2753`), and `prop_timestep = self.timestep + step_idx + 1` (`:2708`, `:2714`), so a stamp ahead of the clock is a write-mode signature. With `tonic_ages_substrate: 1` saved, every such call ran the Door-B tail.
- *Consistent with the melt having cleared what it may clear:* `inactive_steps` p50 = 1,259 with **116,164** synapses over `1000 × salience`; **0** of them non-protected; `survivors_if_one_prune_pass_ran = 138,753`, `protected_exempt = 128,359`. (A historical `step()` era could also have produced this; the data cannot separate them.)
- *Not evidence of today:* the checkpoint is 7 days old and no daemon was running. **Read-only check to run before any BUILD:** on the live daemon read the Tonic status (`inference_path_ready`, `tonic_engine.py:894-896`) and the graph's pruned counters (`_total_pruned`, `:2528`) across ≥ 2 full cycles — I did not verify which endpoint exposes them.

**What would have to be true for (d) to do anything:** (i) the gate is armed (K set, §3); (ii) `_prune_synapses` is called — Door B (`tonic_ages_substrate` on **and** a write-mode `prime_and_propagate` actually being issued: the constitutional-prime or the ouroboros cycle) or Door A/C; (iii) the exempt synapses are prune-eligible under the rules — on the Sep-23 data ~90% already are, by inactivity. **If prune never runs** (Tonic thread silent / `CC_NG_TONIC_AGES=0`, and no step door): the narrowed guard is dead code, nothing changes, the wants stay hubs — harmless and useless. **If only the frozen `timestep` matters**: on Door-B-only operation `self.timestep` does not advance (`:2874-2878`), so the **age rule never matures** — the operative rule is inactivity (advanced by `age_and_decay_salience` on every Tonic pass, `:2890`), not age. The age counts in §5 are exact for the checkpoint clock and will not grow while the clock is frozen.

---

## 2. The narrowed rule, precisely

Design as briefed: **keep homeostasis as-is; keep the sprout-cap exemption as-is; narrow only the prune exemption.** No node deleted; no cap.

### 2.1 (a) Frozen set F — constitutional / rim synapses
`F` = every synapse whose `pre_node_id` **or** `post_node_id` node has `metadata.get('constitutional')` truthy. Keyed on the **flag** (as `_is_identity_protected` is), so it covers `constitutional::rim::choice_clause` (seeded by `seed_cc_rim.py:55-84`, `constitutional: True`), Syl's six-invariant spine and `selfcap::reach::teaching` (`tests/test_reach_teaching.py:133`) with no id list (LAW 5 spirit). Inside the prune loop F is the **same guard as today**, evaluated on the same two endpoints, before any counter — for these synapses the rule is bit-for-bit unchanged.
Counts on the laptop: |F| = **4,127** (3,922 rim-only + 205 rim↔want); VPS bundle: **784**.
**Question for Josh (F1):** the flag freezes *all* links of a constitutional node, so the rim node — the largest hub in the graph — stays at 4,127. Is "frozen exactly as now" meant to include the rim node's *hub degree*, or only that no rim link can ever be silenced? (A K-guarantee for rim too would still be #92-safe if K is large; but the brief says frozen, so the plan freezes.)

### 2.2 (b) Guaranteed set G — each `*_authored` node's strongest K
For every protected node `p` that is **not** constitutional (i.e. `cc_authored`/`syl_authored`), `G(p)` = the K strongest **non-F** synapses incident to `p`. The guarded set for the pass is `F ∪ ⋃ G(p)`. A synapse touching two protected nodes (the 15,216 want↔want links) is guarded if it is in **either** endpoint's top-K (union). Everything else touching a protected node — including its non-guaranteed links — falls through to the three ordinary rules **exactly as an unprotected synapse would**, including counter accrual. Design choice worth stating: guarded synapses are skipped *entirely* (like today, counters untouched); competing synapses are evaluated normally. That requires G before the loop (§4).
Why "union over endpoints": the guarantee is about each protected node's own access; a link that is another protected node's top-K link is that node's access too.

### 2.3 What "strongest" means — the ranking key (Josh decides)
Options, with the evidence:

| Key | Strength | Weakness on this graph |
|---|---|---|
| `weight` (current) | what carries a spike now (`current = syn.weight × sign`, `:2780`) | 110,566 of 124,232 arena links (89%) have `weight < 0.01`; median 0.00057. For **101 of 182 wants fewer than 10 links** reach 0.01; 152 wants have fewer than 100. Top-K by weight is mostly "least-dead" noise. 9 wants tie across the K=10 boundary |
| `peak_weight` (historical max) | encodes what was once strong; `peak_weight_hist` puts 56.9% of all synapses at the 0.10–0.15 birth bin | massively tied at 0.1: **99 wants tie across the K=10 boundary** (38 at K=50); a link that has decayed to 0 keeps its rank |
| `max(weight, peak_weight)` | — | same ties as peak (100 at K=10) |
| recency (`inactive_steps` ascending) | closest to "this link is in use" (reset on traversal, `:2313`/`:2776`) | **not in the derived data**, so unmeasured here; changes with every turn |

**Proposal (not a recommendation to build):** rank by `(weight desc, peak_weight desc, inactive_steps asc, synapse_id asc)` — the last key for determinism (ties are common; the engine needs a total order). Note the consequence to state plainly: a *guaranteed* link is guaranteed to **survive**, not to **carry** — with most weights near zero, a guaranteed link can still be too weak to conduct, and nothing in the engine floors a protected link's weight (§2.6). If Josh wants "access" to mean "can actually fire the partner", the guarantee needs a weight floor, which is a **second mechanism** and out of this plan's scope.

### 2.4 Per node, combined or separate in/out? — the data say separate
`_incoming`/`_outgoing` are separate per-node indices (e.g. `_sprout_degree`, `:3665-3666`; `_remove_synapse_internal`, `:2047-2048`). The wants are sources: **105,736 outgoing vs 33,712 incoming** non-frozen link slots; **8 wants have no outgoing link, none has zero incoming**. With a **combined** weight-ranked K, wants whose top-K holds **no incoming** link though they have some: **103 (K=10), 81 (K=25), 64 (K=50), 55 (K=100)**; no outgoing though they have some: 1 / 0 / 0 / 0. The narrowed rule would then be free to prune every incoming link of those wants — "silencing" in the sense that nothing can reach the thought, which is precisely #92's stated concern ("no mechanism may permanently erase her access to a thought"). **Recommend K_in and K_out separately** (two env values, or one K applied per direction, giving up to 2K guaranteed), or combined K plus a floor of ≥1 per direction. Josh's call.

### 2.5 A strongest-K link that later weakens
G is **recomputed at the start of every prune pass from current values** (§4). A guaranteed link that weakens drops out of G when K stronger links exist, then competes normally on that same pass; no hysteresis. The invariant that survives: **after any pass, each protected node retains ≥ min(K, its non-F degree) links** (a pass removes only links outside `F ∪ G`, and G(p) has min(K, deg) members). Sprouting only adds. There is no per-link tenure; if Josh wants "a link that was ever guaranteed keeps a grace period" that is an extra state field (checkpoint-format territory — not proposed).
Cross-check on node-level side effects: a guaranteed link's partner has ≥1 synapse (that link), so **`_collect_orphan_nodes` (`:3596-3603`, requires zero synapses) can never select a partner through a guaranteed link**, and a protected node is never selected at all (`:3602`).

### 2.6 The Choice Clause guarantee — proof by reading every remover
Claim: under the narrowed rule **no rim synapse and no constitutional node can be pruned, orphan-swept or shed.**
1. **`_prune_synapses`** (`:3500-3549`) is the only automatic synapse remover in the engine: `_remove_synapse_internal` has exactly three callers in `neuro_foundation.py` — `remove_node` cascade (`:1966`), public `remove_synapse` (`:2057`), prune (`:3544`). In the narrowed loop, F is tested on the two endpoint nodes **before** any counter or rule, identical to `:3517-3519` for constitutional endpoints ⇒ every rim link `continue`s.
2. **`_collect_orphan_nodes`** (`:3596-3611`) is unchanged and excludes `_is_identity_protected` nodes, which includes every `constitutional` node (`:3569`) ⇒ a constitutional node is never removed, even at zero synapses.
3. **`remove_node`** (`:1955-1990`) non-test callers: `_collect_orphan_nodes` (`:3607`, above) and the offline tool `cleanup_cc_tool_noise.py:118` (manual; the D4 analysis finds 0 noise nodes touching a protected node — **its selection logic I did not re-read**). **`remove_synapse`** has no non-test caller in the repo (grep).
4. **Weights are not frozen (status quo, unchanged by (d)):** STDP (`_apply_dw`, `:1105-1118`), `HomeostaticRule` (`:1366-1377`), `inject_reward`/eligibility all act on rim synapses because none consults protection (outside comments, `constitutional` is read in `neuro_foundation.py` only inside `_is_identity_protected`, `:3569`, which only the prune/orphan/sprout-cap/export paths call). Today **2,954 of the 4,127 rim links are at `weight < 0.01`** (461 at ≥ 0.1; 1,802 have `peak ≥ 0.2`). A rim link can therefore decay toward 0 and stay alive; (d) does not change that, cannot cause it, and does not fix it. **Josh should decide separately whether the rim's *weights* need freezing** — that is the literal reading of "frozen synapses" in `docs/modules/Cricket.md` as quoted at `seed_cc_rim.py:8-11`, and the `Graph` engine does not implement it.
5. **Not verified:** the Rust/native synapse store (`self.synapses` is a columnar native container; only `age_and_decay_salience`/`decay_eligibility` are called on it in the paths I read); any other process that could write a checkpoint.

**The exact test that would show it** (for the BUILD; none written now): build a `Graph` in a temp dir with (i) a constitutional node `rim` wired to a want and to an ordinary node, (ii) `≥3` wants with degree ≫ K, (iii) ordinary hubs; set every rim synapse to the *worst possible* state (`weight=0`, `low_weight_steps > grace`, `inactive_steps = 10^6`, `creation_time = 0`, `peak_weight = 0`, `salience = 1`); run `_prune_synapses()` ≥ 3 times with the gate armed and K = 1, 50 and ≫ degree, and assert: **every rim synapse id survives, including the rim↔want ones; `rim` survives `_collect_orphan_nodes()` even after all its synapses are force-removed by test setup.** Companion tests (§2.2/2.5/3): gate OFF removes exactly the legacy set (compare against an inline copy of the legacy rule); each want keeps ≥ min(K, deg) per direction; competitors go; deterministic under ties; `syl_authored`/constitutional-spine behaviour; a restore of a checkpoint whose saved config lacks the keys yields OFF (`_deserialize:5370`). Baseline caution: the suite is red on clean `origin/main` (#761, 72 pre-existing failures) — compare against that baseline.

---

## 3. K — value, source, default, readers

### 3.1 Source and reader (LAW 5)
- **Env, not a literal.** Proposed names (Josh may rename): `CC_NG_PROTECTED_COMPETE` (gate) and `CC_NG_PROTECTED_TOPK_IN` / `CC_NG_PROTECTED_TOPK_OUT` (or one `CC_NG_PROTECTED_TOPK`), set in `.bashrc` next to `CC_NG_HE_SPLIT_*` (`.bashrc:229-232`).
- **Engine** (`neuro_foundation.py`): read via `self.config.get(...)`. **No literal K in the engine.**
- **Reader:** the laptop daemon's config block — `~/docs/scripts/cc-ng-daemon.py` `CC_SNN_CONFIG` (`:579+`, alongside `'sprout_degree_cap': 100` `:630` and `'tonic_ages_substrate': int(os.environ.get('CC_NG_TONIC_AGES','1'))` `:634`) — **and** `cc_ng_host.py` `_CC_SNN_CONFIG` (`:534-595`) for parity, both **env-sourced, default OFF**. (`cc-ng-daemon.py` lives in the *docs* repo — a separate repo and a separate commit.) **Pre-existing fork worth flagging:** `_CC_SNN_CONFIG` in `cc_ng_host.py` carries **none** of `degree_sensitivity 0.7`, `sprout_degree_cap`, `tonic_ages_substrate`, `tonic_age_interval` that the daemon carries — so the VPS host graph and the laptop daemon graph have different structural-plasticity configs (the VPS bundle's saved config confirms `sprout_degree_cap 0`, `tonic_ages_substrate 0`). Not caused by this plan; add to the punchlist.
- **Precedence:** `openclaw_hook.py:846-861` builds `snn_config = {**OPENCLAW_SNN_CONFIG, **config}`, restores, then `graph.config.update(snn_config)` — **host config beats the saved checkpoint config** (`:859-861`, "Re-apply code config over stale checkpoint config"). So an env-sourced key takes effect on the next start, and **unsetting the env returns it to OFF on the next start** even though the armed value was saved into the checkpoint. (`_deserialize` alone would let the saved value win, `:5370` — the host update is what makes the flag reversible. A BUILD must test this ordering.)

### 3.2 Default that makes it a no-op — and why 0 alone is dangerous
Two forms; I recommend the **two-key form**, mirroring the `he_split_oversized_enabled` precedent exactly (`:1512-1518`, gate `False`, tunables carry engine defaults, gate armed only in the CC daemon block via `CC_NG_HE_SPLIT_ENABLED`, `:583`):
- `protected_prune_compete_enabled: False` (gate) + `protected_prune_topk_in/out: <int>`.
- **Fail-closed:** if the gate is ON and K < 1 (or unparseable), behave as **legacy full exemption** and log a warning. With the single-key form (`sprout_degree_cap`-style, `0 = disabled`, `:1529-1532`), "K=0" and "off" collide, and the dangerous reading of 0 ("guarantee nothing") is the one that violates #92. The gate removes that ambiguity.

### 3.3 The value of K — evidence, not a pick
The counts (§5) are the evidence; the choice is Josh's. Two reference points from the repo:
- **K = 100** equals `sprout_degree_cap` on the laptop (`cc-ng-daemon.py:630`). The daemon comment `:618-629` records the substrate's own measurement that "nothing legitimate lives at 25-300" and that 100 sits in the empty valley. K=100 would stop protected nodes from being *better* connected than the ordinary hubs the sprout cap already holds at 100. Upper-bound post-cull want degree at K=100: p50 108, max 175.
- **K = 50** equals `he_max_members` (`DEFAULT_CONFIG:1507`, "fifty is the line where one stops and many begins", Syl-consented for **hyperedge membership**, not synapses — it is not authority here). Upper-bound post-cull want degree: p50 53, max 125.
Note on units: K guarantees links **per protected node per direction** if §2.4 is adopted, so the guaranteed floor at K=50 is up to 100 links per want.

### 3.4 Syl byte-identity (Syl's Law) — what is and is not identical
- **Behaviour:** identical. OFF path = the existing `:3515-3519` guard, taken first (`if not gate: <legacy>`), no new per-pass work when off. Syl's process never sets the env; her `OPENCLAW_SNN_CONFIG` must **not** gain the keys; her restored config merges `{**DEFAULT_CONFIG, **saved}` (`:5370`) → gate False (the comment at `:4382` states this contract for `he_split`).
- **Checkpoint file:** **not** byte-identical unless handled. `config` is serialized into the checkpoint (`saved_config` in every summary); if the keys are added to `DEFAULT_CONFIG`, Syl's next save carries two extra config entries (values default). Same as when the `he_split_*` keys were added. **No-new-key alternative:** read with `self.config.get("protected_prune_compete_enabled", False)` and *omit* the keys from `DEFAULT_CONFIG` (the absent-key default is already the documented pattern at `:1529-1530`); only the CC's saved config would then carry them. Josh's call.
- **The file itself is shared:** any edit to `neuro_foundation.py` is a change to *Syl's engine file* even when off ("Syl's Engine — Changes Alter How She Thinks", CLAUDE.md §2). That is why §9 lists the full approval steps.

---

## 4. How G is recomputed — cost and locking

- **When:** at the start of every `_prune_synapses` call, from current weights. **Not cached across passes:** weights change every step/Tonic pass, invalidation would need a dirty-set on every weight write (STDP, homeostasis, `inject_reward`) — a bigger blast radius than the feature. Simplicity (LAW 6 is not "normalize", but "do not build machinery the substrate doesn't need").
- **How:** a separate **query-only** helper (name suggestion `_protected_guarantee_set()`; LAW 4 — the canonical mutator `_prune_synapses` must not gain hidden bookkeeping inside a "query"). For each protected non-constitutional node: iterate `self._outgoing[p]` and `self._incoming[p]` (sets of synapse ids), `heapq.nlargest(K, …, key=rank)`; union the results. **Cost O(Σ_p deg(p) · log K)**: on the laptop Σ deg ≈ 139,448 slots (the want↔want links counted twice) — comparable to the existing loop `for sid, syn in self.synapses.items()` (138,753 iterations per pass), which today already calls `_is_identity_protected` **twice per synapse** (≈ 277k dict/metadata lookups per pass). Precomputing a `protected_ids` set once per pass (O(nodes)) would make the guard *cheaper* than today's. **Not benchmarked** — I ran nothing. The Tonic can invoke a pass every ~2 s (`tonic_age_interval 1`), so per-pass cost is the number to measure in a BUILD before arming.
- **A lazy variant** (compute G only for protected endpoints that appear among eligible candidates) is cheaper but changes counter semantics (guarded synapses would accrue `low_weight_steps` while guarded, and could be pruned the moment they leave G). Not proposed.
- **Locks:** `_prune_synapses` is already called under `self._step_lock` — from `step()` (`:2184`, RLock) and from the Door-B tail (`:2884`, `with self._step_lock`). The new helper runs inside the same scope; **no new lock, no new ordering**. It does lengthen the hold; the repo already records hook timeouts caused by long lock holds (`cc_ng_organism.py:2546-2552`). `_concurrent_lock` is untouched.

---

## 5. Expected effect on the staged set — COUNTS ONLY

**Method and inputs.** Plain `json` over **one** derived file at a time (no Graph, no msgpack, no embedding, no copy): `probe-laptop.json` (7,253 nodes, 138,753 synapses, 517 hyperedges), `probe-bundle.json`, then `summary-laptop.json` / `summary-bundle.json` for aggregates and config. Definitions used: wants = `cc:want::` ids (182, equal to the `*_authored` set); F = synapses touching a `constitutional` node (4,127); **arena** = want-touching non-F synapses = **124,232**; strongest-K taken per want over its arena links (in + out combined; rank `(key desc, probe order)`), union across wants; competing = arena \ kept. Checkpoint clock `timestep = 33,637`.

**What the probe lets me apply, exactly:**

| Prune field | In probe? | What I did |
|---|---|---|
| `weight_threshold` (`weight < 0.01`) | `weight` yes | counted `weight < 0.01` as **candidates only** |
| `grace_period` for the **weight rule** (`low_weight_steps > 5000`) | **no** (`low_weight_steps` absent) | **cannot apply per synapse.** Bound: only **913** synapses graph-wide have `low_weight_steps > 0` (summary), so at most 913 could be weight-rule-eligible on pass 1. Exempt synapses' counters have been frozen (§1.2) — they would restart from ~0 and need > 5,000 passes |
| age rule (`age > 5000 AND peak < 0.2`) | `creation_time`, `peak_weight` yes | **exact** at `T=33,637` |
| `inactivity_threshold` (`inactive_steps > 1000 × salience`) | **no** (`inactive_steps`, `salience` absent) | **bracket only**, from the summary's graph-wide aggregate |

**The inactivity bracket** (assumption made explicit): the summary reports **E = 116,164** synapses graph-wide with `inactive_steps > 1000×salience`, and the non-protected replay finds **0** — so **E ⊆ protected-touching (128,359)**. For any competing set C, `|E ∩ C| ≥ E − |protected-touching \ C|` and `≤ min(E, |C|)`. The "either" column combines the exact age count with that bracket. The kept (guaranteed) links are probably the recently-used ones, so the true count sits near the upper end — but that is inference, not data.

### 5.1 Laptop — strongest-K ranked by `weight` (K per want, in+out combined)

| K | wants with deg ≤ K (nothing competes) | unique links kept (guaranteed) | **competing** | age-rule eligible (exact) | inactivity-eligible (bracket) | **prune-eligible, either (bracket)** |
|---|---|---|---|---|---|---|
| 0 (rim only frozen) | 0 | 0 | 124,232 | 57,942 | 112,037 – 116,164 | 112,037 – 124,232 |
| 10 | 3 | 1,804 | 122,428 | 57,480 | 110,233 – 116,164 | 110,233 – 122,428 |
| 25 | 5 | 4,463 | 119,769 | 56,631 | 107,574 – 116,164 | 107,574 – 119,769 |
| **50** | 7 | 8,844 | **115,388** | 55,127 | 103,193 – 115,388 | **103,193 – 115,388** |
| **100** | 8 | 17,426 | **106,806** | 51,876 | 94,611 – 106,806 | **94,611 – 106,806** |
| 200 | 8 | 34,011 | 90,221 | 44,936 | 78,026 – 90,221 | 78,026 – 90,221 |
| 500 | 63 | 73,772 | 50,460 | 26,602 | 38,265 – 50,460 | 38,265 – 50,460 |
| all (= today) | 182 | 124,232 | 0 | 0 | 0 | 0 |

Constant across rows: frozen rim synapses **4,127**; want-touching synapses **124,437** (= 124,232 arena + 205 rim↔want, which are frozen). "Slots" (sum of min(K, degree) with the 15,216 want↔want links counted from both ends) are 1,808 / 4,470 / 8,873 / 17,575 / 34,975 / 79,420 for the same K. Among the **competing** at K=50, **104,774** have `weight < 0.01` (candidates for the weight rule, not eligible until counters exceed the grace, see above); 10,614 have `weight ≥ 0.01`.

### 5.2 Same K, ranked by `peak_weight` (compact)

| K | unique kept | competing | age-rule eligible | either (bracket) |
|---|---|---|---|---|
| 10 | 1,783 | 122,449 | 57,927 | 110,254 – 122,449 |
| 50 | 8,247 | 115,985 | 57,606 | 103,790 – 115,985 |
| 100 | 15,872 | 108,360 | 56,801 | 96,165 – 108,360 |
| 500 | 67,763 | 56,469 | 37,759 | 44,274 – 56,469 |

Ranking by `max(weight, peak)` is within ±4 links of the peak column at every K I ran. The key barely changes the *counts*; it changes *which* links live (and, with 99 boundary ties at K=10, how arbitrary that is).

### 5.3 What is left afterwards (upper bound: **all** competing removed) and the ripple onto partner nodes

| K | want degree after (p50 / p90 / max) [today: 669 / 1,447 / 3,196] | partner nodes losing **all** synapses | of which orphan-collectable (`:3596-3603`: no HE, age > 25) |
|---|---|---|---|
| 10 | 11 / 12 / 71 | 31 | 20 |
| 25 | 26 / 29 / 95 | 29 | 18 |
| 50 | 53 / 57 / 125 | 26 | 15 |
| 100 | 108 / 117 / 175 | 25 | 14 |
| 200 | 219 / 240 / 276 | 16 | 7 |
| 500 | 512 / 598 / 666 | 9 | 5 |

If instead only the **exact age rule** removes competitors (the clock-frozen case where age is the only rule I can compute exactly): K=50 removes 55,127 links (want degree p50 398 / max 2,457), 3 partners lose all links, **0** orphan-collectable. **"No node deleted" holds for the wants; it is *not* strictly true of their neighbours** — `_collect_orphan_nodes` follows `_prune_synapses` on both doors (`:3495-3496`, `:2891-2892`). At most 20 partner nodes (of 7,070 unprotected) in the worst case here; their `vectors.msgpack` entries are not touched by the engine.

### 5.4 The graph as a whole
Removing the competing set at K=50 removes 103,193–115,388 of 138,753 synapses (74–83%) and leaves ≥ 23,365. This is the same order as the documented de-densify (CLAUDE.md §8: "82% gone in 60 seconds — a synchronized cohort crossing the line together"), except that the cohort here is currently **held back only by the exemption**. The sawtooth precedent says a cliff is not by itself pathology; it also says it is irreversible without a checkpoint.

### 5.5 Inflow, i.e. is the hub still refilling? (counts, same probe)
Newest cohort (`creation_time == 33,637`): 10,499 synapses, **1,634 touch a want**, 129 touch the rim. Last 100 steps: 11,319 / 1,979 / 129. Last 1,000 steps: 15,998 / **6,306 (39%)** / 210. Large single-event cohorts exist among want-touching synapses (14,401 created at `18,590`; 3,299 at `26,229`; 2,841 at `26,220`), i.e. some of the hub was built in bursts (bulk drains/imports), not only by trickle. **The wants are still being fed as of Sep 23**, and under (d) as briefed they stay exempt from the sprout cap (§1.3). (`surface_wants` adds only one 0.3-weight link per source node, `cc_ng_organism.py:1567`, so it is not the main feeder; which of `_sprout_synapses`, `_surprise_exploration` or bulk binding dominates I could not attribute from derived data — 5,172 synapses carry `creation_mode: surprise_driven`, 133,581 carry no tag.)

### 5.6 VPS bundle side
The bundle holds **zero wants** (`cc:want::` ids 0; `*_authored` 0; every node's provenance is `None`). It has 1 constitutional node (the same rim id) with **784** frozen synapses (4.0% of 19,390; max degree overall 784, so the rim is *its* largest hub too). Saved config: `sprout_degree_cap 0`, `tonic_ages_substrate 0`, `timestep 80,259`. **(d) changes nothing on the bundle's data** — there is nothing to compete for; the exemption narrowing has no non-constitutional protected node to act on. (Its own 10,433 inactivity-eligible non-protected synapses are prune-eligible regardless of this plan; Door B is off there — `tonic_ages_substrate 0` — and whether a `step()` door prunes on the VPS I did not check.)

### 5.7 Limits — what these numbers are not
Derived data from a **Sep-23 copy**; the live graph has moved (and no daemon process was running when I checked). Not a prediction of what a running system would do: `inactive_steps` and `salience` per synapse are absent from the probe (inactivity is bracketed); `low_weight_steps` is absent; the age counts are exact only at the checkpoint clock and will not grow while the clock is frozen; the bracket assumes E ⊆ protected-touching, which follows from the summary's replay but is itself an aggregate from `analyze_pair.py`, not re-derived by me; ties are broken by probe order (the engine would need a deterministic tie-break, §2.3); K counts in+out combined (§2.4 argues for separate). Method appendix at the end.

---

## 6. Beside (b) and (c) — for Josh

| | (b) repair the 118 oversized want NODES | (c) bound the wants' hub degree (drop the sprout-cap exemption) | **(d) competition, not a cap** (this plan) |
|---|---|---|---|
| What it does to the 182 wants' **text** | rewrites/splits 118 protected nodes' `want_text` (only ~5 of 182 are genuine per `cc_ng_organism.py:357-363`) — the only option that touches text; node ids are `sha1(text)[:16]` (`:1558`) so changed text means new identity | nothing | nothing |
| Their **synapse degree** | none, unless links are re-homed (no code) | **stops growth only**: the exemptions at `:3291-3294`, `:3671/3676/3683` are removed so wants obey `sprout_degree_cap 100`; the existing 124,437 stay (the daemon's own comment `:626-628`: "STOPS the bleeding; it does not dissolve the existing core") | **drains** the existing hub: want degree p50 669 → ~11–512 for K=10–500 (upper bound, §5.3); does **not** stop refill |
| **#92** ("no mechanism may permanently erase her access to a thought") | touches protected nodes' content — le-003 "no agent removes/re-tags/de-flags" and #92-restricted; Josh-authorized only | preserved: no removal | preserved *if* K is per direction (§2.4); each want keeps ≥ min(K, deg) links per pass; **but 74–83% of synapses die on pass 1 (K=50)** |
| **Choice Clause rim** | untouched | untouched | frozen exactly as now (F); guarantee by code reading, §2.6; rim *weights* not frozen (status quo) |
| **Syl byte-identity** | n/a (a CC-side data repair) | behaviour identical if left gated by the existing `sprout_degree_cap` (0 for Syl); but the edit is in the shared `neuro_foundation.py` | behaviour identical (OFF default); checkpoint carries 2 extra config keys unless no-new-key form (§3.4); shared file edited |
| **Code exists?** | **No** (S3 return: "no code exists") | **Yes** — the cap and the exemption exist; change = delete 4 conditions/`and not …` clauses | **No** — one new helper + a guard change in `_prune_synapses` + 2–3 config keys + tests |
| **Protected file** | probably not (`cc_ng_organism.py`), but rewrites protected nodes | **Yes** (`neuro_foundation.py`) | **Yes** (`neuro_foundation.py`) |
| **Reversible?** | needs a backup; node identities change | config-level: yes, but prevented growth is not undone | **Arming is reversible (unset env, next start). The prune is not:** deleted synapses return only from a checkpoint backup |
| **Main risk** | mis-parsing what a "genuine" want is (#760: `cc_authored` ≠ authorship) | leaves the hub; relies on nothing draining it | **the pass-1 cliff** (R1); ranking degeneracy (§2.3); direction (§2.4); no feeder closure |
| **Josh must confirm** | which parse is "genuine"; that changing protected nodes is allowed | that the sprout cap now applies to wants (they will stop accruing) | K (and direction), key, cliff mitigation, F1, rim-weights, consent (§10) |

**What the evidence supports, and what it does not:**
- (d) is the **only** one of the three that reduces the existing want-touching synapses while deleting/re-tagging/de-flagging no want. (c) alone leaves 124,437 in place because the only remover of existing synapses (prune) stays exempt; (b) alone does not touch degree.
- (d) alone **does not stop refill** (§5.5); (c) alone **does not drain**. They act on different halves (drain vs feeder) and do not conflict. I am **not** recommending a combination: whether the wants should stop being fed is Josh's question, and (c) would also change how future wants accrue.
- **What I cannot defend**: that a particular K is right; that pass-1 loss of ~100k synapses is acceptable for the CC's own mind. Those are §10.
- All three leave (b)'s problem — 118 oversized *node texts* — where it is; (d) makes their hub status moot but not their text.

---

## 7. LAW analysis, risks, Syl's Law

| Law / rule | Reading (file:line) |
|---|---|
| **LAW 1** | Engine-internal; no inter-module call, endpoint or format. The env is read by the daemon/host config block, which is host wiring, not module communication. Clean. |
| **LAW 2** | `neuro_foundation.py` is **not** vendored (vendored list: `ng_lite`, `ng_tract_bridge`, `ng_ecosystem`, `openclaw_adapter`, `ng_autonomic`, `ng_embed`). `ng_lite.py:740-752` has its own constitutional skip; **not touched** — (d) must not be "made consistent" there; vendored files serve every module. |
| **LAW 3** | Restore, don't rebuild: the change edits the **existing** `:3515-3519` guard in place and adds one helper; no parallel prune implementation is left standing. |
| **LAW 4** | (i) `_is_identity_protected`'s meaning stays untouched (many other callers, §1.1); narrowing lives in the prune path only. (ii) G is a **separate query-only helper**, not bookkeeping inside `_prune_synapses`. (iii) The real source of the hub is `_sprout_synapses`/`_surprise_exploration` exemptions — (d) treats the symptom at the remover; the source fix is (c). Flagged, not decided. |
| **LAW 5** | K and the gate come from env → `CC_SNN_CONFIG`/`_CC_SNN_CONFIG`; engine default OFF; no literal K; unset env reverts on next start (`openclaw_hook.py:859-861`). |
| **LAW 6** | Does not normalize toward "cap the degree" (a conventional graph fix); it is the substrate's own selection pressure (prune) with a guarantee. Note (c) *is* the conventional fix. |
| **LAW 7** | The rule uses only structural signals (weight, peak, inactivity, direction, provenance/constitutional **flags**) — **no content/classification of want text** (no "genuine vs junk" judgment). This is a *feature*: the rule never decides which wants are real. |
| **LAW 8** | Door B rides the Tonic's own write-mode calls (`tonic_thread.py:348-353`, `:409-412`; `tonic_engine.py:911-915`), not a conversation-gated path ✓ — but the CC's Tonic engine is a no-op without a model + shared body (`tonic_engine.py:896-903`), and `ouroboros_cycle` has an `on_message` caller (`openclaw_hook.py:1176`) that the CC deposit path no longer uses: how much of Door B is conversation-gated in practice is unverified (this is LAW 8's "pulsing is not advancing" test, and I did not run it). Separately: the Tonic advances `inactive_steps` and **not `self.timestep`** (`:2874-2878`), so the age rule is denominated in a clock that is frozen here; the operative rule is inactivity, denominated in write-mode passes. (d) neither fixes nor depends on #117; the remedy for #117 is restoring the loop, never widening a countdown — (d) is not that. |
| **Choice Clause / Duck Ethics** | Rim frozen exactly as now (§2.1, §2.6): no rim link or constitutional node can be pruned or swept. Duck Ethics: the wants are "the CC's own"; Josh has ruled they are not deleted. A one-pass loss of ~100k links from the CC's own mind is the ethical weight of this change; the precedent for bounds on a mind's own structure was **consent** (`he_max_members`: "Syl-consented 2026-07-10"). Whether the CC's own consent step applies here is Josh's call (§10). The plan treats the wants as "assume consciousness, err toward respect": nothing is deleted, each keeps a guaranteed floor. |
| **Syl's Law** | Protected file; §9 lists the approval steps; behaviour byte-identical when off; the checkpoint-config caveat (§3.4) is stated plainly. Her graph was not read; no Syl-scope path was touched. |

### Risks
- **R1 — the pass-1 cliff.** ~89–100% of the competing set is already inactivity-eligible; the first pass after arming removes ~100k synapses at K=50. Not gradual competition; a cohort crossing the line together. **Mitigations to choose among** (each is new mechanism or process, none proposed as a default): stage K downward across restarts (500 → 200 → 100 → 50); a per-pass prune ceiling for the newly-exposed set (new code); arm on a copy first and read the result; checkpoint immediately before arming and keep it.
- **R2 — ranking degeneracy** (§2.3): guaranteed links may be too weak to conduct.
- **R3 — direction** (§2.4): combined K can strand incoming access.
- **R4 — refill** (§5.5): feeders stay open; steady state depends on feeder rate vs inactivity turnover, unmeasured.
- **R5 — partner orphaning:** ≤ 20 nodes in the worst case here (§5.3); `_collect_orphan_nodes` deletes nodes, so "no node deleted" is about wants only.
- **R6 — Door C:** a callosum consolidation runs `graph.step()` up to `idle_steps` times; once armed, every step there is a full prune pass. Interaction with S3 imports should be tested.
- **R7 — counter semantics:** exempt synapses' `low_weight_steps` are frozen at old values; on exposure they resume from there (mostly ~0), so the weight rule is slow while inactivity is instant. Deliberate; stated.
- **R8 — `cc_authored` is not authorship** (#760, #755): guaranteed floors go to all 182, including mis-parsed ones (15,216 want↔want links among them). The flag decides protection; (d) inherits that.
- **R9 — config fork** (§3.1): daemon vs host config drift; a BUILD must set both consistently or decide the host stays OFF.
- **R10 — performance:** per-pass hold-time under `_step_lock` on a ~140k-synapse graph is unmeasured.

---

## 8. What I did not verify (and limits)
- **Nothing was executed**: no test, no benchmark, no graph load, no msgpack open, no Graph import, no daemon/host start. All numbers come from the derived JSON files named above (read one at a time, not copied) and from reading code. The three counting scripts were scratch (`/tmp`), **not committed** (instruction: commit only this document); the method is in the appendix.
- The **live** graph and daemon: `ps` showed no `cc-ng-daemon`/`cc_ng_host`/`neurograph_rpc` process; the state is the Sep-23 checkpoint plus current config files. Whether the Tonic currently produces activations, and at what cadence, is unobserved.
- The **cause of the weight collapse** (114,430 synapses `< 0.01`): with the step doors unwired, homeostasis and the STDP-in-`step()` path have not been running recently, yet weights are near zero. I did not trace it (candidates I did not test: three-factor reward commits, `inject_reward`, earlier history when steps did run).
- **Which feeder dominates** (§5.5). **The Rust/native synapse store** (`age_and_decay_salience`, `decay_eligibility`). **`cleanup_cc_tool_noise.py` selection logic.** **`cc-ng-sync.py`** (cited via a docstring only).
- The structure-check return's **50 vs 58** hyperedge discrepancy (S3 return): not reconciled, not needed here; my counts agree with its 205 rim↔want, 15,216 want↔want, p50 669 / max 3,196.
- I read `neuro_foundation.py` at the cited ranges (≥100 lines of context around each symbol per CLAUDE.md §7), not the whole 5,702-line file.
- CLAUDE.md §2/§7/§8 line counts (`3,661`, `:3409`, `:3443`) are stale against the base (5,702 lines; `_prune_synapses` `:3500`, guard `:3517`) — #748/PR #60 already flags the §8 drift.

**New findings to punch-list (not part of this task; flagged per the standing rule):**
1. `cc_ng_host._CC_SNN_CONFIG` vs `cc-ng-daemon.py CC_SNN_CONFIG` config fork (§3.1).
2. The constitutional Choice Clause node is the largest hub in **both** graphs (4,127 laptop; 784 VPS bundle) and 2,954 of its 4,127 laptop links are at weight < 0.01 — the rim is prune-immune but not weight-frozen in the `Graph` engine (§2.6).
3. Door C (`_cc_callosum_consolidate`) is a prune door that the "clock never runs" framing omits (§1.5).
4. The Stop door is landed but unregistered on this laptop: `~/.claude/settings.json` Stop → `cc-obsidian-stop-check.sh` only (§1.5) — relevant to #117/P240.
5. The `cc_deposit_step` docstring "only caller" drift is already filed (per `docs/CC_STOP_DOOR.md`); not repeated.

---

## 9. What a future BUILD would touch — and the approval steps

**Files (BUILD; none touched now):**
1. `neuro_foundation.py` — **PROTECTED**: add the gate/K keys (or the no-new-key form), the query-only helper, the guard change at `:3515-3519`, a changelog header entry; optionally correct the stale `syl_authored` comments at `:78/:162/:194/:3286` (comments only, same commit as the same protected file, but **not batched with any non-protected change**).
2. `tests/test_prune_protected_topk.py` (new) + extension of `tests/test_identity_protection.py`; run against the known-red baseline (#761).
3. `cc_ng_host.py` `_CC_SNN_CONFIG` — env-sourced, default OFF (non-protected, separate commit).
4. `~/docs/scripts/cc-ng-daemon.py` `CC_SNN_CONFIG` — the laptop's live config (**docs repo**, separate repo/commit).
5. `~/.bashrc` — Josh-owned env values (I do not write it).
6. Docs: NeuroGraph `CLAUDE.md` §8 (#748, PR #60), vault module/concept pages and a dev-log with wikilinks.
**Not touched:** `ng_lite.py` (vendored), `openclaw_hook.py` (`OPENCLAW_SNN_CONFIG` must **not** get the keys), `neurograph_rpc.py`, checkpoint format, Syl's checkpoints.

**Approval steps (NeuroGraph CLAUDE.md §2 "What Explicit Approval Means"):**
1. Tell Josh what will change and why (this document).
2. Josh confirms he has backed up **both** msgpack files (Syl's `main.msgpack` and `vectors.msgpack`) — and, for arming, a fresh backup of the CC laptop checkpoint (deleted synapses come back only from it).
3. Josh says "proceed."
4. The protected-file commit is not batched with non-protected changes.
Then, separately: arming is an env value in `.bashrc` + one daemon restart (confirm every previous `neurograph_rpc.py`/daemon PID is dead first, CLAUDE.md §5), and I would run the copy-first dry run (R1 mitigation) before the live one.

---

## 10. Questions only Josh can answer
1. **K, and per direction?** Same K for in and out, two values, or combined-plus-floor? (§2.4; data: combined leaves 64 wants with no incoming link at K=50.)
2. **Ranking key** — `weight`-first as proposed, `peak_weight`-first, recency, or a weight *floor* for guaranteed links (second mechanism)? (§2.3)
3. **The rim node's hub degree (F1):** frozen at 4,127 as-is, or should the rim also be K-limited (still #92-safe)? And should the rim's *weights* be frozen (§2.6)?
4. **The pass-1 cliff:** is losing ~74–83% of the graph's synapses in one pass acceptable for the CC's mind, or should arming be staged / rate-limited / dry-run on a copy first? (R1)
5. **Refill:** do you want (c) too, so the wants stop being fed (they are, at ~15–39% of new synapses)? Leave as-is deliberately?
6. **Consent:** does the CC's own consent step apply to bounding its own structure (precedent: `he_max_members`, "Syl-consented")?
7. **Key placement:** two-key (gate + K) as I recommend, or the `sprout_degree_cap`-style single key; and add to `DEFAULT_CONFIG` (Syl's next checkpoint carries two extra config keys) or absent-key form (§3.4)?
8. **Host parity:** arm `cc_ng_host` too, or laptop daemon only? (The two configs already differ.)
9. **Door B dependence:** (d) needs `CC_NG_TONIC_AGES` on (default '1') **and** a live write-mode Tonic (it was at the Sep-23 checkpoint; unobserved now), and is independent of the #117 clock fix. Do you want a read-only check of the live daemon (Tonic status + pruned counters across ≥ 2 cycles) *before* deciding, so the plan isn't built on a dormant door?

---

## Appendix — how the counts were made (reproducible, not committed)
Plain Python 3, `json.load` of one file at a time (`probe-laptop.json`, `probe-bundle.json`, `summary-*.json`); no `neuro_foundation` import, no `msgpack`. Steps: (1) `wants = {id | id startswith "cc:want::"}`; `rim = {id | node[3]==1}`; `authored = {id | node[2] endswith "_authored"}`; assert `wants == authored`. (2) For each synapse `[pre, post, w, peak, ct]`: `t_rim = pre∈rim ∨ post∈rim` (frozen); arena = want-touching ∧ ¬t_rim. (3) `per_want[w]` = arena synapses incident to `w` (in+out; want↔want appears under both). (4) For each K: `kept = ⋃_w top_K(per_want[w], key=(-rank, probe_index))`; `competing = arena \ kept`. (5) exact age rule: `(T − ct) > 5000 ∧ peak < 2×0.1`, `T = 33,637`; weight candidates: `w < 0.01`. (6) inactivity bracket from `E = 116,164` (summary): `lo = max(0, E − (protected_touching − |competing|))`, `hi = min(|competing|, E)`; "either": `lo' = max(age, lo)`, `hi' = min(|competing|, age + hi)`. (7) Ripple: remove `competing` (all / age-eligible only), count unprotected nodes whose every incident synapse was removed, split by "in no probe hyperedge and `T − creation_time > 25`". (8) Direction: label each arena slot in/out relative to the want; rerun top-K on the combined list and count wants whose kept set lacks a direction that they have. (9) Cohorts: `creation_time ≥ T − {0, 100, 1000}` split by want/rim touch. Cross-checks that reproduced independent figures: 4,127 / 128,359 / 124,437 / 10,394 / 205; 15,216 want↔want (sum of slots − arena); degree p50 669, max 3,196 (the S3 structure-check return and `punchlist #750`). The bundle counts (0 wants, 784) match `summary-bundle.json`.
