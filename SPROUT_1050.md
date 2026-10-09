# #1050 — sprouting from repeated co-firing (the co-firing tally)

*2026-10-08 · lane sprout-1050 (bounded build lane for the Executive) · review branches only: nothing merged, nothing
armed, nothing installed into the NG venv; the live daemon, its venv, its checkpoint and the trial worktrees untouched.*

Spec: `~/docs/superpowers/specs/2026-10-06-sleep-phase-design.md` §7 (the #1050 design) and D12 ("decide K / θ / horizon
after a measurement; rec. K = 8, θ = 3, ~50-step horizon, native"). Josh approved starting #1050 on 2026-10-08. Josh's
intent: *"Links should sprout naturally, to wherever it wants to go. Not some shotgun like scatter approach."* *"Ideally I
don't want any 'just because' sprouting."* Bounded by competition, not a cap.
`neuro_foundation.py` is PROTECTED: one protected commit; merging needs Josh's protected-file "proceed". No vendored file
is touched (LAW 2).

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-sprout-1050-20261008` | trial tip `4de1166` | **PROTECTED** `c66bcde` · tests `8b2ff4b` · this doc |
| ng-tract-rs | `cc-laptop-sprout-1050-rs-20261008` | `cba74b1` (the live wheel) | `70cdede` tally · `20f9a42` episodes |
| docs (daemon) | `cc-laptop-sprout-1050-daemon-20261008` | `cdbb4d58` (trial s4a tip) | `aa90b29b` env names + tests · `98cdf2aa` default θ 4 |

Wheel (throwaway venv only): `~/.cache/sprout-1050/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`,
sha256 `8ede89c5a770769db805afa55a77e51d3d285b27cb7f850a4405b2e29f203c52` (`.so` `b28eddf2…`), built from `20f9a42`
with `cargo -j 1`.

VERDICT-PENDING

---

## 1. Step 0 — measured first (D12)

**Copy.** `~/.claude/plugins/neurograph/checkpoints/main.msgpack` + `vectors.msgpack` copied to `~/.cache/sprout-1050/ckpt/`
(mode 444) at 10:25 (manifest stable across the copy): 13,370 nodes, 77,106 synapses, 2,509 hyperedges, t = 116,486,
engine `4de1166`; sha256 `e940e342e890385c…` / `9fa2ea1b64418506…`.

**Harness.** `~/.cache/sprout-1050/scripts/wake.py`, derived from the P2 lane's `drywake.py`: per cycle a 250-step
drain-like wake (10 deposits; each a fresh node + the 20 nearest existing nodes to a **real deposit's stored embedding**
(seeded sample of 600 `cc:conv` entries; only ids, embeddings and text sha256s were extracted) stimulated at 1.5, one
step, reward 0.1, then idle steps) **plus a Tonic-like write-mode tick every 25 steps** (the constitutional prime,
steps = 1, and a re-injection of the current deposit's neighbourhood, steps = 3 — the P2 harness had no Tonic), then one
`sleep_cycle()` **with the disuse sleep on as it will be armed** (d0 0.1, h 0.05, G 2, last-link 2, kappa 0.015,
`structural_plasticity_in_sleep` on, `tonic_ages_substrate` 0). Every run went through
`~/.cache/sprout-1050/run-isolated.sh` (the p2a harness: scratch HOME removed after, PID namespace, tmpfs over the live
plugin dir, read-only binds, MemoryMax 4.5 GB, no swap, `nice 10`, waits for MemAvailable ≥ 4 GB), one at a time,
wall-clock timers only. The live daemon ran beside it the whole day (load 6-11 on 4 cores): absolute times are inflated.

**Today's rule, 4 cycles (`out/m0-sq.json`, with every `_sprout_synapses` call traced):**

| measure | wake 1 | wake 2 | wake 3 | wake 4 |
|---|---:|---:|---:|---:|
| fired per step, median / mean / p90 | 139 / 186 / 283 | 238 / 406 / 950 | 267 / 382 / 801 | 68 / 140 / 234 |
| recent candidates per call, median / p90 | 356 / 731 | 890 / 2,510 | 826 / 2,437 | 195 / 671 |
| fired × candidates per call (the naive tally's touches), median / p90 / max | 51K / 191K / 2.29M | 163K / 2.27M / 2.67M | 164K / 1.96M / 2.44M | 11K / 162K / 2.93M |
| sprouted: co-firing / Tonic tail / surprise | **2,500** / 200 / 0 | **2,500** / 200 / 41 | **2,500** / 200 / 1,167 | **2,500** / 200 / 190 |
| `_sprout_synapses` wall ms, median (co-firing / Tonic) | 71 / 76 | 79 / 71 | 103 / 67 | 53 / 45 |

- **The 10-per-step cap is hit on every one of the 1,000 steps** (2,500 per 250-step wake), and every Tonic tail hits its
  own 10 (200 per wake). Today's sprouting is bounded only by the rails.
- **Fraction later potentiated** (peak_weight ≥ 2 × initial 0.1, at the end of the run): wake-1 cohort 197 / 2,500
  (**7.9%**) after 3 more wakes, wake-2 26 / 2,500 (1.0%), wake-3 4 / 2,500 (0.16%); Tonic cohorts 10 / 200 and 0;
  surprise cohorts **0 / 1,398**.
- **Direction.** Today's code wires `create_synapse(nid, other_id)`: the node firing **now** → the node that fired
  **earlier** — the anti-STDP direction (three-factor STDP then depresses it on the next reward). The design's direction
  (earlier → later) is the tally's.
- **The naive tally cost.** 457 million fired × candidate touches over the 1,080 calls (median 71K, p90 1.8M, max 2.9M per
  call) — far above the design's 160K estimate. A per-touch update would cost seconds per step in Python and tens of ms
  natively. The bounded table below never touches more than K slots plus the candidates it scans until the table is
  full, so the cost stays per *fired node*, not per pair.
- **The Choice Clause node** (identity-protected, so exempt from `sprout_degree_cap`) gained 160 out-links in one wake
  (1,751 → 1,911): the shotgun reaches it most.

## 2. Mechanism choice from the data

### 2.1 Steps vs episodes (offline replay of the m0 trace, open loop)

`scripts/replay.py` replays the tally (the engine's Python fallback) over the recorded calls for a K × θ × H grid, with
today's rails (10 per call, degree cap 100, protected exempt). Open loop: its sprouts do not change the recorded firing.

| rule, K = 8, H = 50 | θ = 2 | θ = 3 | θ = 4 |
|---|---|---|---|
| counting **steps** (§7 as written): sprouted per batch | 2,632 / 1,209 / 1,207 / 2,414 | 2,578 / 897 / 1,050 / 2,248 | 2,529 / 680 / 892 / 1,993 |
| counting **episodes** (a touch counts only after a pause > `co_activation_window`) | 2,276 / 1,013 / 1,033 / 1,734 | 1,657 / 873 / 896 / 978 | **827 / 842 / 452 / 388** |

Counting steps barely changes today's numbers: the population fires in **bursts** (hundreds of nodes over consecutive
steps), so a pair "co-fires" 3 times inside **one** burst and crosses θ on a single occasion — the "one coincidence" the
design means to exclude, merely spread over 3 steps. Sprouted pairs had co-fired on 42-50 steps of the trace. So the tally
counts **episodes**: a held partner's touch adds +1 only when more than `co_activation_window` (5) steps have passed
since the pair's last touch; inside an episode the decayed score is carried forward. No new parameter (the window is
today's). This is a refinement of §7's update rule, made because of the measurement; it is the one place this build
departs from the spec's literal text.

### 2.2 The closed-loop sweep (4 wake/sleep cycles each, the real engine with the tally on, native wheel)

Same copy, seeds and stimulus as m0 (wake 1 identical up to the first sprout). `out/sw-K*-th*-H*.json`.

| run | sprouted per wake: co-firing (+ Tonic, surprise) | sum | wake-1 cohort potentiated | wake-2 cohort potentiated | potentiated links at the end (all cohorts) | rested recall relevance, wakes 1-4 |
|---|---|---:|---:|---:|---:|---|
| today (m0) | 2,500 / 2,500 / 2,500 / 2,500 (+ 800 Tonic, 1,398 surprise) | 12,198 | 207 / 2,700 (7.7%) | 26 / 2,741 (0.9%) | 237 | (not measured: harness predates the metric) |
| K 8, θ 3 | 1,712 / 901 / 1,514 / 1,339 (+ 166, 18) | 5,650 | 91 / 1,756 (5.2%) | 46 / 938 (4.9%) | 159 | (not measured) |
| K 4, θ 3 | 1,060 / 406 / 1,234 / 1,043 (+ 84, 0) | 3,827 | 140 / 1,078 (13.0%) | 16 / 417 (3.8%) | 156 | (not measured) |
| **K 8, θ 4** | **887 / 268 / 604 / 593 (+ 65, 0)** | **2,417** | **122 / 898 (13.6%)** | **201 / 278 (72%)** | **343** | 0.597 / 0.596 / 0.612 / 0.639 |
| K 4, θ 4 | 569 / 130 / 201 / 491 (+ 17, 0) | 1,408 | 113 / 575 (19.7%) | 34 / 130 (26%) | 150 | 0.592 / 0.608 / 0.587 / 0.595 |
| K 8, θ 4, **H 100** | 1,479 / 322 / 1,254 / 1,208 (+ 147, 0) | 4,410 | 77 / 1,537 (5.0%) | 85 / 324 (26%) | 188 | 0.606 / 0.601 / 0.598 / 0.620 |
| K 8, θ 4, **H 25** | 16 / 44 / 19 / 23 (+ 0, 0) | 102 | 11 / 16 | 16 / 44 | 27 | 0.580 / 0.584 / 0.602 / 0.602 |

(H = 50 unless named. Rested-recall start value 0.604; §5 defines it. The first two sweep runs predate the rested
metric.)

**What θ and H mean together.** Because a pair's score decays between occasions, it saturates at 1 / (1 − λ^d) for
occasions d steps apart, so θ also sets the *rhythm* a pair must keep, not just a count (gap 5, so d ≥ 6):

| θ, H | occasions needed at d = 6 / 10 / 15 / 25 steps apart | slowest rhythm that can ever sprout |
|---|---|---:|
| 3, 50 (D12's recommendation) | 4 / 4 / 6 / never | every 20 steps |
| **4, 50 (chosen)** | **6 / 7 / never / never** | **every 14 steps** |
| 4, 100 | 5 / 5 / 6 / 9 | every 29 steps |
| 4, 25 | 9 / never / never / never | every 7 steps |

**Chosen: K = 8, θ = 4, H = 50 steps** (gap = `co_activation_window` 5). Versus today it sprouts **80% less** (2,417 vs
12,198 over the 4 wakes) yet ends with **more potentiated links** (343 vs 237): what it grows gets used. θ 3 roughly
halves sprouting with no gain in use; K 4 / θ 4 and H 25 sprout least but end with fewer used links than today (150, 27):
the starvation side; H 100 doubles the sprouts of H 50 for fewer used links (188). D12 recommended θ = 3; in words,
θ = 4 / H = 50 means **"a pair must co-fire on at least six separate occasions, each within about 14 steps of the last"**
(a 250-step drain batch is ~2 minutes; at the 900 s autostep, 14 steps is ~3.5 hours). Single seed, 4 cycles per run:
θ and H are resolved clearly, K less so.

## 3. What changed

### Engine — PROTECTED commit `c66bcde` (`neuro_foundation.py`)

All new behaviour sits behind config keys that are **absent by default** and NOT in `DEFAULT_CONFIG` (read live with
absent defaults): `sprout_tally_enabled` + three REQUIRED parameters `sprout_tally_slots` (K), `sprout_tally_theta` (θ),
`sprout_tally_horizon_steps` (H). A missing / invalid one raises ValueError before anything is touched.

- **`_sprout_synapses`** with the switch on runs `_sprout_from_tally`: candidates = today's set (a spike 1..5 steps ago,
  not firing now), ordered most recent first (stable). One tally call (native or fallback) returns the crossings; each
  then meets **today's rails**, in order: 10 per call, both nodes present, no synapse in either direction (also among
  pairs accepted earlier in the call), `sprout_degree_cap` with identity-protected nodes exempt; then today's delay rule
  (moved verbatim into `_sprout_delay`) and `initial_sprouting_weight`. A crossing a rail stops is dropped (its slot is
  already empty — the filopodium retracts).
- **The tally rule** (per fired node a, per call): (1) every held partner b that is among the candidates is reinforced:
  `score = score · λ^(t − last_t) + (1 if t − last_t > gap else 0)`, `last_t = t`, λ = exp(−1/H), gap =
  `co_activation_window`; at score ≥ θ the pair **b → a** (earlier → later, the STDP direction) is a crossing and the
  slot is emptied. (2) newcomers, in candidate order, take an empty slot or a **retracted** one (decayed score < e^−1:
  one meeting H steps ago; the lowest decayed first, ties to the lowest slot), starting at score 1; partners already held,
  crossed in this call, or already connected (either direction) never take a slot. Bounded by competition: at most K
  contacts per node; a newcomer waits until a contact retracts.
- **The Tonic** (write-mode `prime_and_propagate` tail, #163) already calls `_sprout_synapses`, so its firings feed the
  same tally instead of sprouting directly. Its decay is in steps: Tonic ticks at the same timestep are one episode.
- **Surprise-driven sprouting feeds the tally too** (§4.2): `_surprise_exploration_tally`.
- **Sleep:** with the switch on, `sleep_cycle` (P1 and disuse paths, chunked or not) empties the tally after its own work
  ("unconsolidated filopodia retract overnight", spec §2 step 6); the record gains `tally_retracted`; one log line at the
  sleep's level (DEBUG on a `sleep_observe` shadow). The fallback's table is rebound, not cleared in place, so an observe
  shadow (which shares the attribute) never empties the live tally (tested).
- **Restore** (`_deserialize`) leaves a cold tally (native: `synapses.clear()`; fallback: reset).
- With the keys absent: one dict lookup in `_sprout_synapses`, `_surprise_exploration`, `sleep_cycle` and `_deserialize`;
  otherwise the `4de1166` code path (the delay body is the same statements, moved).

### Rust — `70cdede` + `20f9a42` (`ng-tract-rs`, `src/store.rs`, pure addition)

`SynapseStore.cofire_tally_update(fired, cands, t, k, theta, lam, floor, gap) -> [(pre, post)]`, `cofire_tally_clear()`,
`cofire_tally_state()`. The tally lives inside the SynapseStore with **its own node interner** (the store's interner,
`node_count` and every existing index are unchanged) and reads the store's native adjacency for "already connected".
In memory only; `clear()` empties it. Pure-Rust core `CofireTally::update` (5 unit tests; `cargo test --lib` 38/38);
float order = the Python fallback (`score * powf(lam, dt as f64) + inc`, no FMA). Arguments are extracted before the
borrow.

### Daemon — `aa90b29b` + `98cdf2aa` (`scripts/cc-ng-daemon.py`, not protected; NOTHING armed)

`CC_SNN_CONFIG` gains `sprout_tally_enabled` = `CC_NG_SPROUT_TALLY` (default off, explicit both ways like the strength
budget), `sprout_tally_slots` = `CC_NG_SPROUT_TALLY_SLOTS` (8), `sprout_tally_theta` = `CC_NG_SPROUT_TALLY_THETA` (4),
`sprout_tally_horizon_steps` = `CC_NG_SPROUT_TALLY_HORIZON_STEPS` (50). Env-name pin updated
(`test_cc_ng_daemon_unbound_status.py`); new `tests/test_cc_ng_daemon_sprout_1050.py` (12 tests incl. a real-engine run
with the CC config, switch on and off). Independent of `CC_NG_SLEEP`: with sleep off the tally only decays lazily.

## 4. Decisions asked for in the brief

### 4.1 Where the tally lives (reuse-first, §7) and persistence

- **`_he_co_fire_counts`** (the hyperedge evolution tally) is a Python dict `hid → {node: count}`: a node-to-hyperedge
  relation, unbounded per key, no decay, Python speed. Generalising it to node pairs would make an unbounded Python pair
  map — the cost the measurement rules out (457M touches). Not reused.
- **`NodeStore` columns**: the native node store is OFF on the CC (graph.nodes is a dict), so it cannot host anything
  the CC uses today.
- **`SynapseStore`** is always native on the CC and already owns the adjacency index the tally must consult for
  "already connected". The tally is literally the table of *candidate* synapses, so it is hosted there (own interner, so
  nothing existing moves). That is the reuse.
- **Persistence: none.** A restart or restore starts the tally cold. Cost: pairs mid-accumulation lose their evidence
  and need up to θ new occasions; the evidence window is ~50 steps and every sleep already empties the tally, so a restart
  costs at most what one sleep costs — a few hundred delayed sprouts at most, none lost forever. Not persisting keeps the
  checkpoint byte-compatible both ways (old code reads new files trivially: nothing new is written), so the
  native-node-store spec §5 rules never come into play. (The four config keys persist in the checkpoint's config like
  every config key; with them absent nothing changes.)

### 4.2 Surprise-driven sprouting goes through the tally

Measured (m0): 41 / 1,167 / 190 surprise sprouts per wake, ~11 per prediction error in wake 3, **0 of 1,398 ever
potentiated**. Each error wires its source to every node that fired in the prediction window (`recent_fired − expected`):
one violated expectation, a scatter of links — the "just because" pattern. Under the switch, each such node C gets one
tally touch for the pair source → C (the predicted direction); the pair sprouts only when its score reaches θ, sharing the
evidence with co-firing. A surprise sprout is born exactly as today's (surprise weight, `creation_mode: surprise_driven`
metadata, the prediction's salience armor). The context-salience boost and novelty detection are unchanged. With θ = 4 a
single surprising coincidence (co-firing touch + surprise touch) can never sprout.

## 5. Recall probe — defined before the proof runs

- **Probe set:** 100 real deposit entries (`cc:conv`, ≥ 40 chars), a seeded sample **disjoint** from the 600 wake cues,
  whose node is in the graph and has ≥ 1 prime. For each: the primes are the 10 most similar vector entries (cosine ≥ 0.4,
  the probe node itself excluded) at current = similarity; read-mode `prime_and_propagate(steps=3)`; surfaced = the
  non-prime fired nodes ranked by (latency, −voltage), top 10 — the `_harvest_associations` shape the daemon's recall uses.
- **Primary metric (rested):** every node at its resting potential and refractory 0 while probing (restored afterwards),
  so what surfaces is decided by topology, not by the transient activation the wake left (m0 showed the live-state
  breadth swinging 5 → 380 between cycles). Reported: **relevance** = mean cosine of the surfaced top-10 to the probe;
  **precision@10** = share of the surfaced top-10 in the probe's 50 nearest neighbours. Also reported: the live-state
  values and breadth. (Hit@10 of the held-out deposit node itself was 0 in every run — uninformative, kept in the JSON.)
- **Bar:** the tally arm's mean rested relevance and precision over the cycles must not fall below today's arm's (within
  run-to-run noise, stated).

## 6. Proof

### 6.1 Keys absent: byte-identical to `4de1166` (both wheels)

`scripts/golden.py` (the P2 lane's golden run: seeded stimulation + uuid4, a write-mode Tonic tick every 3rd step, per
step fired ids + every node's numeric state + every synapse id / weight / eligibility + pred_weights hashed, final
checkpoint sha256), 18 steps, on the copy. Modes: `plain` (the copy's own config) and `sleep` (+ the armed P2 disuse keys
and a `sleep_cycle` every 6th step: 3 sleeps incl. the first clearance).

| wheel | mode | base `4de1166` trace / checkpoint | branch | result |
|---|---|---|---|---|
| live `cba74b1` (venv-base, `.so` `4c4cab39…`) | plain | `d7bba234992f468f` / `bfdf17502e13cc32` | identical | **equal** |
| live `cba74b1` | sleep | `3509609c90ed9ac2` / `478c6acf25434e8e` | identical | **equal** |
| new `20f9a42` (venv-new) | plain | `d7bba234992f468f` / `bfdf17502e13cc32` | identical | **equal** |
| new `20f9a42` | sleep | `3509609c90ed9ac2` / `478c6acf25434e8e` | identical | **equal** |

(All eight runs agree; 180 sprouts in each, so today's sprouting path is exercised.) Plus
`tests/test_sprout_1050.py::test_keys_absent_is_the_trial_tip_exactly`: the P1 whole-run workload (steps, Tonic ticks
with aging, recall, node churn, rewards, competition, downscale, snapshots; per-step synapse state bitwise; final
checkpoint bytes) × 2 flag sets × 3 seeds × dict / native node store × no / P1 / P2 sleep = **36/36 equal to `4de1166`**.

### 6.2 Native == fallback, bit for bit

- **On the copy** (`out/nvf-venv-new.json` vs `out/nvf-venv-base.json`): the chosen tally + the disuse sleep, 2
  wake/sleep cycles; the new wheel (native) vs the live wheel `cba74b1` (no `cofire_tally_update` → the Python fallback).
  After each wake the digests of the **sprout order** (synapse id, pre, post, source, timestep of every sprout), the full
  **tally state** and the **synapse state** (ids, weights, eligibility) are **identical** (`572badb92745416d` /
  `b8c744390c5263dd` / `9d8e9f195ae64696` after wake 1; `6483ead5017d5860` / `aa1820dbfb42c954` / `e69426f49f2a2269` after
  wake 2); the sleeps retract the same 29,656 and 30,672 slots; the recall probes return the same surfaced sets; every
  tally counter is equal (1,176 sprouted, 3,477 crossings, 512 cap / 1,784 degree / 5 existing blocks).
- **Tests:** random call sequences against the native core and the fallback (crossings and the full state after every
  call, 6 seeds × 400 calls, incl. a K change); 8 whole runs (dict / native node store × 4 seeds, the tally + the disuse
  sleep, Tonic ticks, recall, node churn) comparing per-round synapse state, tally state, counters and checkpoint bytes;
  every rule test runs on both implementations.
- **Cost on the copy:** `_sprout_synapses` with the tally, median per call **24-26 ms native vs 33-34 ms fallback**
  (today's rule 53-103 ms in m0; all including candidate building and rails, load 6-11).

### 6.3 Wake/sleep dry run: today's sprouting vs the tally

10 wake/sleep cycles per arm (2,500 wake steps + 100 Tonic-like ticks + 10 sleeps), same copy, seeds, cues and
Tonic stimulus; the disuse sleep on as armed (chosen P2 parameters, one hold); native wheel; `scripts/proof.sh`,
`out/proof-today.json` (3,607 s) and `out/proof-tally.json` (3,263 s, K 8, θ 4, H 50). Wake 1 is identical in both
arms up to the first sprout; the today arm reproduces m0's first 4 cycles exactly (determinism check).

**Summary (the bar's items):**

| measure | today | tally | change |
|---|---:|---:|---|
| synapses sprouted per 250-step batch (mean; co-firing + Tonic + surprise) | **2,969** (2,500 + 200 + 269) | **668** (649 + 13 + 6) | **−77%** |
| … total over the run | 29,689 | 6,683 | |
| potentiated sprouts at the end (peak ≥ 2 × 0.1), all cohorts | 785 (2.6%) | **1,460 (21.8%)** | **1.9× more links used, from 4.4× fewer sprouts** |
| … cohorts with ≥ 3 later wakes (wakes 1-7) | 747 / 20,710 (3.6%) | 1,399 / 5,504 (25.4%) | |
| recall, rested (primary), mean over the 10 cycles: relevance / precision@10 | 0.6056 / 0.0206 | **0.6051 / 0.0227** | equal / +10% (per-cycle range 0.585-0.643 vs 0.588-0.633) |
| recall, live state, mean relevance | 0.604 | 0.564 | lower (see risk 3) |
| synapses after sleeps 3-10 (min - max); at the end | 57,312 - 63,749; 57,780 | 45,555 - 56,187; 45,555 | the tally graph is ~12K smaller; see "steady state" |
| hubs at the end: top out-degrees; nodes with out-degree ≥ 100 (58 at start) | 275, 195, 194; **34** | 231, 195, 194; **12** | fewer hubs |
| Choice Clause node out-degree (start 1,731; lifeline kept in both) | 1,911 after wake 1 → 275 | 1,778 after wake 1 → 231 | the shotgun fed it 180 links in wake 1, the tally 47 |
| protected lifelines (8) | 8 every cycle | 8 every cycle | none lost |
| `_sprout_synapses` wall time per call, median of the per-wake medians | 87 ms | **32 ms** | −63% (the mechanism's own cost) |
| `step()` wall time, median of per-wake medians / mean of per-wake means | 0.641 / 0.934 s | 0.347 / 0.830 s | faster, but the arms fire differently (load 6-11): not a clean delta |
| `sleep_cycle` wall time per sleep (range) | 0.29-5.43 s | 0.46-2.74 s | |

**Steady state.** Today's arm holds 57-61K after the backlog clearance (sleep 3) with 2,969 sprouts per batch balanced
by clearance. The tally arm settles lower: 48.5K / 48.6K / 48.5K over sleeps 6-8, then 46.7K and 45.6K after sleeps 9-10
(the clearance still slightly exceeds the much smaller sprouting at the end of this window). Ten cycles do not show
whether it levels off near ~45K or keeps drifting; observe mode / a longer run would (risk 1).

**Per cycle:**

**today** (`out/proof-today.json`), start: 77,106 synapses

| cycle | sprouted co-fire / Tonic / surprise | synapses after sleep | cleared by sleep | w >= 0.5 | top out-degree | nodes out >= 100 | Choice Clause out | step median s | sprout call median ms | recall rested: relevance / precision@10 / breadth | recall live: relevance / breadth |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2,500 / 200 / 0 | 79,806 | 0 | 922 | [1921, 1911, 317] | 65 | 1911 | 0.359 | 54.44 | 0.588 / 0.027 / 10.88 | 0.550 / 191.15 |
| 2 | 2,500 / 200 / 41 | 82,547 | 0 | 596 | [1951, 1911, 317] | 77 | 1911 | 1.016 | 105.26 | 0.606 / 0.032 / 3.73 | 0.595 / 5.65 |
| 3 | 2,500 / 200 / 1167 | 63,749 | 22,665 | 740 | [1285, 273, 266] | 58 | 250 | 1.065 | 122.85 | 0.605 / 0.023 / 3.88 | 0.616 / 8.87 |
| 4 | 2,500 / 200 / 190 | 61,403 | 5,236 | 1379 | [1221, 271, 264] | 56 | 238 | 0.326 | 58.66 | 0.592 / 0.012 / 8.85 | 0.571 / 382.94 |
| 5 | 2,500 / 200 / 27 | 59,814 | 4,316 | 646 | [586, 271, 264] | 53 | 586 | 1.048 | 121.76 | 0.612 / 0.032 / 3.51 | 0.617 / 14.72 |
| 6 | 2,500 / 200 / 214 | 60,374 | 2,354 | 1445 | [580, 264, 242] | 55 | 580 | 0.435 | 66.4 | 0.591 / 0.013 / 6.66 | 0.617 / 9.97 |
| 7 | 2,500 / 200 / 171 | 58,667 | 4,578 | 778 | [369, 264, 213] | 46 | 369 | 0.846 | 106.56 | 0.633 / 0.022 / 3.79 | 0.620 / 4.7 |
| 8 | 2,500 / 200 / 104 | 60,007 | 1,464 | 1244 | [340, 213, 205] | 47 | 340 | 0.394 | 68.8 | 0.611 / 0.011 / 5.7 | 0.608 / 7.97 |
| 9 | 2,500 / 200 / 687 | 57,312 | 6,082 | 1026 | [381, 195, 194] | 28 | 381 | 0.897 | 114.97 | 0.615 / 0.019 / 4.96 | 0.658 / 44.41 |
| 10 | 2,500 / 200 / 88 | 57,780 | 2,320 | 1614 | [275, 195, 194] | 34 | 275 | 0.304 | 57.52 | 0.602 / 0.014 / 6.64 | 0.589 / 10.83 |

**tally** (`out/proof-tally.json`), start: 77,106 synapses

| cycle | sprouted co-fire / Tonic / surprise | synapses after sleep | cleared by sleep | w >= 0.5 | top out-degree | nodes out >= 100 | Choice Clause out | step median s | sprout call median ms | recall rested: relevance / precision@10 / breadth | recall live: relevance / breadth |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 887 / 11 / 0 | 78,004 | 0 | 765 | [1921, 1778, 317] | 59 | 1778 | 0.339 | 24.68 | 0.597 / 0.025 / 7.78 | 0.500 / 111.75 |
| 2 | 268 / 10 / 0 | 78,282 | 0 | 1032 | [1921, 1778, 317] | 59 | 1778 | 0.273 | 23.11 | 0.596 / 0.010 / 6.31 | 0.659 / 14.07 |
| 3 | 604 / 11 / 0 | 56,187 | 22,710 | 805 | [292, 273, 264] | 34 | 245 | 1.655 | 54.98 | 0.612 / 0.033 / 4.37 | 0.542 / 211.03 |
| 4 | 593 / 33 / 0 | 53,019 | 3,794 | 682 | [292, 271, 264] | 30 | 236 | 0.245 | 27.95 | 0.639 / 0.036 / 4.83 | 0.639 / 4.83 |
| 5 | 1,798 / 17 / 63 | 49,515 | 5,382 | 927 | [312, 271, 264] | 25 | 312 | 0.632 | 44.28 | 0.600 / 0.018 / 3.44 | 0.482 / 106.57 |
| 6 | 32 / 0 / 0 | 48,482 | 1,065 | 1826 | [292, 264, 240] | 23 | 292 | 0.28 | 29.62 | 0.587 / 0.008 / 9.96 | 0.586 / 10.13 |
| 7 | 1,154 / 23 / 0 | 48,570 | 1,089 | 1182 | [264, 263, 240] | 20 | 263 | 1.745 | 52.05 | 0.597 / 0.024 / 3.94 | 0.533 / 318.12 |
| 8 | 360 / 9 / 0 | 48,521 | 418 | 701 | [254, 213, 206] | 20 | 254 | 0.503 | 44.1 | 0.643 / 0.047 / 1.98 | 0.516 / 26.2 |
| 9 | 302 / 7 / 0 | 46,671 | 2,159 | 1655 | [219, 195, 194] | 17 | 219 | 0.22 | 28.83 | 0.594 / 0.012 / 5.47 | 0.600 / 42.07 |
| 10 | 493 / 8 / 0 | 45,555 | 1,617 | 1505 | [231, 195, 194] | 12 | 231 | 0.355 | 35.29 | 0.585 / 0.014 / 5.82 | 0.588 / 5.96 |

**today: sprout cohorts at the end of the run** (potentiated = peak_weight >= 2 x initial 0.1)

| born in wake | co-fire | Tonic | surprise |
|---:|---|---|---|
| 1 | 2,500 -> 766 alive, 228 pot. (9.1%) | 200 -> 35 alive, 7 pot. (3.5%) | - |
| 2 | 2,500 -> 695 alive, 65 pot. (2.6%) | 200 -> 21 alive, 4 pot. (2.0%) | 41 -> 1 alive, 0 pot. (0.0%) |
| 3 | 2,500 -> 786 alive, 209 pot. (8.4%) | 200 -> 9 alive, 0 pot. (0.0%) | 1,167 -> 134 alive, 34 pot. (2.9%) |
| 4 | 2,500 -> 901 alive, 83 pot. (3.3%) | 200 -> 54 alive, 0 pot. (0.0%) | 190 -> 112 alive, 12 pot. (6.3%) |
| 5 | 2,500 -> 750 alive, 61 pot. (2.4%) | 200 -> 50 alive, 1 pot. (0.5%) | 27 -> 7 alive, 0 pot. (0.0%) |
| 6 | 2,500 -> 1,115 alive, 27 pot. (1.1%) | 200 -> 81 alive, 0 pot. (0.0%) | 214 -> 115 alive, 11 pot. (5.1%) |
| 7 | 2,500 -> 1,760 alive, 5 pot. (0.2%) | 200 -> 135 alive, 0 pot. (0.0%) | 171 -> 125 alive, 0 pot. (0.0%) |
| 8 | 2,500 -> 2,496 alive, 25 pot. (1.0%) | 200 -> 198 alive, 0 pot. (0.0%) | 104 -> 104 alive, 3 pot. (2.9%) |
| 9 | 2,500 -> 2,500 alive, 6 pot. (0.2%) | 200 -> 200 alive, 0 pot. (0.0%) | 687 -> 687 alive, 4 pot. (0.6%) |
| 10 | 2,500 -> 2,500 alive, 0 pot. (0.0%) | 200 -> 200 alive, 0 pot. (0.0%) | 88 -> 88 alive, 0 pot. (0.0%) |

**tally: sprout cohorts at the end of the run** (potentiated = peak_weight >= 2 x initial 0.1)

| born in wake | co-fire | Tonic | surprise |
|---:|---|---|---|
| 1 | 887 -> 644 alive, 267 pot. (30.1%) | 11 -> 7 alive, 3 pot. (27.3%) | - |
| 2 | 268 -> 186 alive, 163 pot. (60.8%) | 10 -> 8 alive, 8 pot. (80.0%) | - |
| 3 | 604 -> 517 alive, 230 pot. (38.1%) | 11 -> 6 alive, 3 pot. (27.3%) | - |
| 4 | 593 -> 542 alive, 162 pot. (27.3%) | 33 -> 31 alive, 20 pot. (60.6%) | - |
| 5 | 1,798 -> 1,624 alive, 300 pot. (16.7%) | 17 -> 16 alive, 3 pot. (17.6%) | 63 -> 59 alive, 5 pot. (7.9%) |
| 6 | 32 -> 32 alive, 0 pot. (0.0%) | - | - |
| 7 | 1,154 -> 1,122 alive, 232 pot. (20.1%) | 23 -> 22 alive, 3 pot. (13.0%) | - |
| 8 | 360 -> 360 alive, 61 pot. (16.9%) | 9 -> 9 alive, 0 pot. (0.0%) | - |
| 9 | 302 -> 302 alive, 0 pot. (0.0%) | 7 -> 7 alive, 0 pot. (0.0%) | - |
| 10 | 493 -> 493 alive, 0 pot. (0.0%) | 8 -> 8 alive, 0 pot. (0.0%) | - |

Reading the cohorts: today's sprouts mostly die unused (wake-1 cohort: 2,500 → 766 alive, 9.1% potentiated) and its
surprise sprouts scatter (2,689 born, 64 potentiated). The tally's sprouts are mostly kept and used (wake-1 cohort 887 →
644 alive, **30%** potentiated; wake-2 61%; wake-3 38%). Wakes 9-10 cohorts are 0% in both arms only because they had
no time.


### 6.4 Suites

SUITES-PENDING

## 7. Pass / fail against the bar (spec §7 + the brief)

PASSFAIL-PENDING

## 8. Risks and what is not proven

RISKS-PENDING
