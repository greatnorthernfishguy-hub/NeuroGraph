# Sleep phase P3 — observe mode (review branches)

*2026-10-07 · lane sleep-observe (bounded build lane for the Executive; Josh approved the lane 2026-10-07: "yep") ·
review branches only: nothing merged, nothing armed, nothing installed; the live daemon, its venv, its checkpoint files
and the trial worktrees untouched.*

Spec: `~/docs/superpowers/specs/2026-10-06-sleep-phase-design.md` §8 P3 — *"`CC_NG_SLEEP=1` in **observe** mode first
(downscale + clearance computed and logged, nothing written) for ≥ 2 sleeps. Then the one-time backlog clearance (D7) …
Then armed."* The Progress log (2026-10-07 19:38) records that no earlier lane built it. Josh accepted the P2
established-links result on condition it is confirmed live in this mode.

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-sleep-observe-20261007` | trial tip `1cb9706` | **PROTECTED** `0050f4b` (`neuro_foundation.py`) · tests `545ec4a` · this doc |
| docs (daemon) | `cc-laptop-sleep-observe-daemon-20261007` | trial tip `3f015f07` | `cdbb4d58` daemon + tests + env pin |

No vendored file is touched (LAW 2). `neuro_foundation.py` is PROTECTED: its change is one commit of its own; merging
needs Josh's protected-file "proceed" (Executive).

---

## 1. Mechanism

### 1.1 Engine — `Graph.sleep_observe(sleeps=None, config_overrides=None, sample=20, detail=False)`

1. **One short `_step_lock` hold** on the live graph (`_sleep_observe_shadow_graph`) captures what the sleep reads,
   with the cheapest consistent copy that exists:
   - the synapse store as its own checkpoint bytes (`SynapseStore.to_checkpoint_msgpack`, the save path's capture:
     weight, salience, low_weight_steps, peak, trace, creation time, metadata stamps, endpoints);
   - every node's metadata contents (copied dicts: the protection flags and the fair-chance counters the orphan sweep
     reads). With the native node store on, its own checkpoint bytes instead;
   - C-level shallow copies of the small maps (hyperedge membership, recent spikes, confirmation history, hyperedges,
     dirty sets, config, the fair-chance registration) and a shallow copy of the `Graph` object (timestep etc.).
2. **After the hold, with the live lock free:** the shadow `SynapseStore` is bulk-loaded from those bytes. The Node
   shells are copied with the captured metadata and an **empty** `pred_weights`: the sleep's only use of `pred_weights` is
   the D15 deletion on removal, which decides nothing. `creation_time`, the only other node field the sleep reads, never
   changes after creation. The adjacency is rebuilt from the shadow store's endpoints, as `restore()` does. Every sleep
   decision that walks it breaks ties by synapse id, so set order cannot matter.
3. **The shadow is isolated.** It has a fresh `_step_lock` and **no event handlers**: the #1051 guardian ledger, the
   vector-store drop on `nodes_collected` and Lenia never see it. Any instance attribute that overrides a `Graph` method
   (bound to the live graph, e.g. a test or host patch) is dropped, so the shadow runs the class's own methods. Its
   fair-chance latch is its own and is pre-set, so the stale WARNING stays the live sweep's. Its `sleep_cycle` log line
   is DEBUG (`_sleep_log_level()`).
4. **The unchanged `Graph.sleep_cycle()` runs on the shadow** 1..16 times (consecutive sleeps, no wake between). It is
   whatever path the shadow's config selects: P1, the P2 disuse sleep, chunked or one-hold. A chunked config runs chunked
   with the pause set to 0, because nothing waits on the private lock. A removal recorder on the shadow instance captures
   each removed synapse's endpoints, its weight at removal (= post-downscale), peak and trace, then calls the unchanged
   `Graph._remove_synapse_internal`.
5. `config_overrides` apply to the **shadow's config only**. The daemon passes the switches an armed sleep would set
   (§1.2).
6. `sleeps=None` (auto): on the disuse path, while the live counters are not yet in sleep units (no disuse sleep has run),
   **G + 1** sleeps. The migration sleep only tags, and a tagged link clears once its count exceeds G, so G + 1 reaches the
   first clearance (3 with the laptop's G = 2). Otherwise 1, which is exactly the next real sleep.
7. One INFO line (counts only). Validation refuses bad arguments before anything is touched. The shadow's own config
   validation (`_sleep_disuse_params`, `_sleep_chunk_params`) runs before any projection.

**Why this cannot drift:** the projection *is* the real sleep. No predicate, rule, shield, lifeline or last-link
computation is re-implemented anywhere. If a later phase changes `sleep_cycle`, observe follows it.

### 1.2 Daemon — `CC_NG_SLEEP_OBSERVE`

| env (LAW 5) | default | meaning |
|---|---|---|
| `CC_NG_SLEEP_OBSERVE` | off | with `CC_NG_SLEEP=1`: the sleep trigger runs an observe pass instead of `sleep_cycle()`. Without `CC_NG_SLEEP` it is inert. |
| `CC_NG_SLEEP_OBSERVE_DIR` | `~/.claude/plugins/neurograph/sleep_observe/` | report files |
| `CC_NG_SLEEP_OBSERVE_SLEEPS` | 0 (engine auto) | projected sleeps per pass |
| `CC_NG_SLEEP_OBSERVE_SAMPLE` | 20 | strongest would-be-forgotten links listed per projected sleep |
| `CC_NG_SLEEP_OBSERVE_KEEP` | 60 | newest report files kept |
| `CC_NG_SLEEP_OBSERVE_CONTENT` | on | ≤ 160-char node excerpts in the report **file** only |
| `CC_NG_SLEEP_OBSERVE_MIN_AVAILABLE_MB` | 1500 | below this, the pass is skipped (WARNING) and retried at the next tick |

All seven are added to the env-name pin (`test_cc_ng_daemon_unbound_status.py`).

**The live substrate is unchanged while observing.** With observe on, `CC_SNN_CONFIG['structural_plasticity_in_sleep']`
and `['sleep_disuse_enabled']` stay False. That is exactly the config with `CC_NG_SLEEP` off, so wake removal goes on as
today (proven byte-identical on the copy, §3.1). The projection gets `{structural_plasticity_in_sleep: True,
sleep_disuse_enabled: CC_NG_SLEEP_DISUSE}` as overrides; every disuse parameter and chunk key is already in the live
config, where it is inert. The alternative was to switch wake removal off for the observe days, as plain `CC_NG_SLEEP=1`
does. I rejected it: with nothing removing anything, the graph would grow unchecked, and "observe" would change the live
organism. **This is a design choice for the Executive / Josh to confirm.**

**Each observe tick** uses the same trigger as a real sleep (`_sleep_decision`), on its own clock (§1.3). It does:
- no HE consolidation (it writes), even with `CC_NG_DREAM` on;
- no save (nothing changed);
- no `last_sleep_wall` write.

**Admin command `admin_sleep_observe`** (`DISPATCH`; optional `{"sleeps": 1..16}`):
- one pass on demand, at any time (with or without `CC_NG_SLEEP`; it previews the sleep the env would arm: disuse iff
  `CC_NG_SLEEP_DISUSE`);
- writes its report file and INFO line, and returns the counts plus the report path (never content);
- does not move the observe clock;
- one pass at a time (`_SLEEP_OBSERVE_LOCK`, shared with the dream loop); the same memory gate.

Invocation (daemon socket):
```bash
python3 -c 'import json,os,socket; s=socket.socket(socket.AF_UNIX); s.connect(os.path.expanduser("~/.claude/plugins/neurograph/daemon.sock")); s.sendall(json.dumps({"event":"admin_sleep_observe","data":{}}).encode()+b"\n"); print(s.makefile().read())'
```

### 1.3 Pacing decision

- Observe passes have **their own persisted clock**, `.cc_last_sleep_observe_wall`, beside `.cc_last_sleep_wall`
  (tmp + `os.replace`). Its first value is the real last-sleep time.
- The **same trigger** applies to that clock: ≥ 6 h since the last observe and quiet ≥ 30 min, or ≥ 24 h whatever the
  conversation; SYMPATHETIC defers (WARNING at most hourly).
- So observe passes come no more often than real sleeps would, survive restarts, and never run every 60 s tick.
- **Real sleep pressure is neither faked nor starved.** `last_sleep_wall` is only read; no observe pass counts as a
  sleep. When observe is switched off and the sleep armed, the first real sleep is due by its own (real) clock. It is
  usually already due, so it runs at the first convenient tick, or within 24 h.
- The observe clock advances after a pass that ran, ok or failed (a config error does not retry every minute). A skipped
  pass (low memory, another pass running) retries at the next tick.
- To get the spec's "≥ 2 observe sleeps" without waiting 6 h idle, the Executive runs `admin_sleep_observe` (tonight)
  and the dream loop adds its own on the sleep schedule.

---

## 2. Report format

**INFO line** (counts only, never ids, never content), e.g. from the copy run:
`CC sleep observe (auto-convenient, nothing written): 3 projected sleep(s) on the disuse path; next sleep would remove 0
(shield 0, last-link held 0), collect 0; whole projection removes 21018 (peak>=0.5: 497, w>=0.5: 0), collects 0, first
clearance at sleep 3; protected lifelines removed 0; lock hold …s, …s; report <path>`. The engine also logs one
`sleep_observe:` INFO line.

**Report file** (`<dir>/<UTC ts>-<kind>.json`, mode 0600; kind `auto-convenient` / `auto-due` / `admin`; 27.9 KB on the
copy):
- `kind`, `written_utc`, `nothing_written_to_the_graph: true`, `config_overrides`, `mem_available_mb_before`.
- `engine`: `timestep`, `sleeps_projected`, `no_wake_between_projections`, `path`, `chunked`, `params`, `start`
  (synapses, nodes, weight bands, established links (peak ≥ 0.5), w ≥ 0.5, migrated?), `protected_nodes`,
  `lifelines_at_start`, totals, `first_clearance`, `removal_digest`, `lock_holds`, `seconds_parts`, and `projections[]`.
- Each projection has:
  - the engine's own sleep record (migrated, downscaled, clamped, eligible, last_link_held, below_threshold_after, …);
  - would_remove / would_collect, plus collected node ids (≤ 500);
  - weight bands before / after (`[0,.001) [.001,.01) [.01,.05) [.05,.1) [.1,.5) [.5,1) [1,∞)`);
  - removals by weight band before, at removal, and by peak band;
  - removed with peak ≥ 0.5, removed with w ≥ 0.5, removed touching a protected node;
  - established links after;
  - `low_weight_counts_after` (sleeps-below-threshold histogram: who clears next);
  - shield-held count + sample;
  - `strongest_removed` (synapse id, pre, post, w_before, w_at_removal, peak, trace);
  - per protected node: out / in degree before → after, lifelines before / after, intact;
  - `lifelines_removed`.
- `node_excerpts`: `{node_id: ≤ 160 chars}` for the endpoints of the sampled strongest removals
  (`_forest_content` → `want_text` → `core_text`, the organism's order). File only.

`detail=True` (proofs only) adds every projection's removal order, collected ids, shield-held ids and the post-downscale
weight of every synapse.

---

## 3. Proof

Fresh copy of `~/.claude/plugins/neurograph/checkpoints/main.msgpack` + `vectors.msgpack` → `~/.cache/sleep-observe/ckpt/`
(mode 444; main sha256 `af8ad4e00843ff16…`): t = 99,459, 12,442 nodes, 79,700 synapses. Every run:
- `~/.cache/sleep-observe/run-isolated.sh`, a copy of the pre-arming harness: scratch HOME (removed after), bwrap PID
  namespace, tmpfs over the live plugin dir, read-only binds, systemd scope MemoryMax 3 GB (tests) / 4.5 GB (copy runs),
  no swap, nice 10, one process at a time, MemAvailable ≥ 4 GB gate;
- a throwaway venv `~/.cache/sleep-observe/venv` with the live wheel (`ng_tract-0.1.0…whl` sha256 `37c0726d…`, `.so`
  `4c4cab39…`, identical to the live venv's; deps read from the live venv's site-packages via `.pth`, never written).

The graph is built the daemon's way (the pre-arming `eqv.py` builder): the daemon module is imported with `.bashrc`'s
`CC_NG_*` / `NG_*` export lines, its `CC_SNN_CONFIG` is merged over `OPENCLAW_SNN_CONFIG`, then `restore(copy)` and
`config.update`. Node store OFF. Load average was 6-11 on 4 cores throughout (live daemon + other lanes), so absolute
times are inflated.

### 3.1 Observe never called / observe off = byte-identical to `1cb9706` — **PASS**

Copy (`~/.cache/prearm/scripts/eqv.py`: 12 seeded steps, Tonic-like ticks, per-step state hashes, sleep records, final
checkpoint sha256):

| run | trace sha256 | checkpoint sha256 |
|---|---|---|
| base `1cb9706` + daemon `3f015f07`, plain (sleep off, one P1 sleep) | `85a825e983b191b6…` | `f51877f5275670f4…` |
| branch + branch daemon, plain | `85a825e983b191b6…` | `f51877f5275670f4…` |
| branch + branch daemon, **observe mode** env (`CC_NG_SLEEP=1 CC_NG_SLEEP_DISUSE=1 CC_NG_SLEEP_OBSERVE=1`), plain | `85a825e983b191b6…` | `f51877f5275670f4…` |
| base, sleepkeys (`CC_NG_SLEEP=1 CC_NG_SLEEP_DISUSE=1`, 3 chunked disuse sleeps; sleep 3 removes 22,927) | `0a1b1039d9859ea4…` | `b4a39bba4ac59e69…` |
| branch, sleepkeys | `0a1b1039d9859ea4…` | `b4a39bba4ac59e69…` |

The observe-mode row is the live-config claim of §1.2: with observe on, the live substrate behaves and checkpoints
exactly as with sleep off.

Tests (`tests/test_sleep_observe.py::test_without_observe_is_1cb9706_exactly`, `BASE = git show 1cb9706`): the P1/P2
whole-run workload with P1 sleeps, with one-hold disuse sleeps and with chunked disuse sleeps, both node stores, all
seeds. Per-step state is bitwise and the checkpoint bytes are equal. **24/24 pass.**

### 3.2 Observe == real — **PASS** (copy: 6/6 sleeps; tests: all cases)

Copy (`~/.cache/sleep-observe/scripts/observe_copy.py` → `out/observe-copy.json`). The live config is unarmed (the
daemon's observe mode). The observe pass uses the daemon's overrides. Then the config is armed and the real chunked
sleep runs (chunk 0.25 s / gap 0.05 s, the daemon defaults). Compared per sleep:
- the removal set **and order**;
- the collected nodes;
- the D14 shield-held ids;
- the engine record (minus timing);
- the post-downscale weight of **every** synapse, bitwise.

| sleep | how reached | removed (observe / real) | shield held | collected | all equal |
|---|---|---|---|---|---|
| 1 (migration, tags) | projection 1 of the auto 3-sleep pass | 0 / 0 | 0 / 0 | 0 / 0 | **yes** (79,700 weights) |
| 2 (tags) | projection 2 (no wake) | 0 / 0 | 0 / 0 | 0 / 0 | **yes** |
| 3 (**first big clearance**) | projection 3 (no wake) | **21,018 / 21,018** | 0 / 0 | 0 / 0 | **yes** |
| 4 (steady state) | 6 wake steps + Tonic tick, then observe(sleeps=1) | 1,250 / 1,250 | 0 / 0 | 0 / 0 | **yes** (58,762) |
| 5 | same | 1,436 / 1,436 | 20 / 20 | 0 / 0 | **yes** |
| 6 | same | 2,941 / 2,941 | 368 / 368 | 0 / 0 | **yes** |

Tests (`tests/test_sleep_observe.py`):
- projection 1 == the following real sleep over 7 wake/sleep cycles: G = 0 and G = 2, chunked and one-hold, both node
  stores. With G = 2: migration, two tag-only sleeps, the big first clearance, then steady state.
- a k-sleep projection == k consecutive real sleeps; the auto pass is 3 sleeps while unmigrated and 1 after.
- an unarmed graph + overrides == the armed graph's real sleep, chunked and one-hold.

### 3.3 Observe writes nothing — **PASS**

- **Copy:** around the daemon's own pass and the proof pass, these were all equal before / after:
  - the checkpoint sha256 of a save (`f1639119722693e0…` both);
  - a state digest over every synapse's weight, trace, peak, salience, low_weight_steps, inactive_steps and metadata,
    every node's `pred_weights`, metadata and creation_time, the config, the dirty sets, `_total_pruned` and the three
    adjacency maps;
  - the #1051 `RemovalLedger` snapshot.

  No `pruned` / `nodes_collected` / `sleep_cycle` event fired.
- **Tests** (`test_observe_writes_nothing`, 10 cases): checkpoint bytes, every synapse column, stamps, `pred_weights`,
  node metadata, config, dirty sets, adjacency, hyperedge membership, confirmation history, `_total_pruned`, the
  fair-chance latch (a stale heartbeat, latch open), handlers, timestep. No events; ledger unchanged. Cases: P1, disuse
  migrating / migrated / chunked, overrides on an unarmed graph, both node stores.
- **Patched live graph** (`test_observe_ignores_a_live_instance_patch…`): the patched method is never called and the
  live state is unchanged.
- **Vector drop** (`test_observe_never_touches_the_vector_drop_handler`): the `nodes_collected` handler never fires
  although the projection collects nodes.

### 3.4 Lock holds (copy, wall clock, load 6-11)

The observe pass holds the live `_step_lock` **once**. Everything else runs with the live lock free: the shadow build is
3.3-5.6 s, the projections 0.6-3.4 s, 4.6-11.2 s in total. The test
`test_live_lock_held_once_briefly_and_free_during_the_projection` shows another thread taking the lock repeatedly during
the projection, and `test_concurrent_steps_during_the_projection_are_harmless` runs steps concurrently.

| same process, same load (`out/holdcmp.json`, `out/observe-copy.json`) | seconds |
|---|---|
| **observe snapshot hold** (11 passes) | **0.23 · 0.29 · 0.56 · 0.59 · 0.59 · 0.65 · 0.71 · 0.85 · 1.23 · 1.52 · 1.70** (median 0.65) |
| real chunked sleep, max hold per sleep (decide hold; 9 sleeps) | 0.31 · 0.31 · 0.31 · 0.40 · 0.40 · 0.41 · 0.51 · 0.52 · 0.72 |
| `capture_checkpoint` hold (every autosave, every 60 s) | 4.21 · 4.68 · 9.94 |

**Against the ~0.7 s target: met at the median, missed in 5 of 11 passes (0.71-1.70 s).** The hold is one native
`to_checkpoint_msgpack` of 79,700 rows (38 MB; 0.76-3.3 s alone under load 10, `out/snapparts.json`) plus about 12K
metadata dict copies. It is a fraction of the save capture's hold, which happens every minute. A first version that also
copied Node objects and adjacency sets inside the hold measured 1.5-5.2 s; those copies were moved out of the hold. It
cannot be chunked without losing a consistent snapshot. The remaining lever is native: a `SynapseStore` clone method (a
column memcpy) would make the hold tens of ms. That needs a Rust change and a wheel rebuild (§6).

**Memory:** the daemon's pass took the process from 671 MB to a peak of 850 MB; a second pass added about 40 MB to the
peak. The shadow is freed at return (the shadow ↔ recorder cycle is broken explicitly).

### 3.5 What the live first observe pass would show (copy, t = 99,459; projection, no wake between sleeps)

- Sleeps 1-2 tag only: 21,074, then 22,340 links counted below threshold.
- Sleep 3, the backlog clearance, removes **21,018 of 79,700 (26%)**:
  - **0** with w ≥ 0.5 before the sleep;
  - **497** with peak ≥ 0.5 (266 in [0.5, 1), 231 ≥ 1);
  - established links (peak ≥ 0.5) go 5,251 → 4,754 (**−9.5%**);
  - every removed link had w < 0.01 before the sleep (10,965 under 0.001, 10,053 in [0.001, 0.01));
  - 1,006 removals touch a protected node;
  - 0 nodes collected; last-link holds 56.
- **Every protected lifeline is intact.** Degrees:
  - Choice Clause node out 1,864 → 1,383, in 48 → 10;
  - want `4625…` out 1,413 → 1,106, in 189 → 13;
  - the two small wants 2 → 2 / 8 → 5 out.
- The eqv sleepkeys run (12 wake steps between the sleeps) removed 22,927 at its sleep 3, the same order of magnitude.

That is what Josh asked to see before the announced clearance. **These are the forgetting numbers, not a verdict.**

---

## 4. Suites

- **Daemon:** full suite, base `3f015f07` vs branch, same harness and venv:
  - base: **75 failed** / 1,048 passed / 3 skipped;
  - branch: **75 failed** / 1,065 passed / 4 skipped;
  - the identical 75 failures (diffed), the known 75. The +17 passed / +1 skipped are the new file. Its real-engine test
    skips inside the whole suite, because `ng_tract` cannot be re-imported in one process, the same as the pre-arming
    test. Alone, the new file is **18/18 passed**.
- **Engine:** `tests/test_sleep_observe.py` **65/65 passed** (571 s). Full suite, per file, branch vs base `1cb9706` on
  the branch's failing files: *running at the time of this commit; the result is added in a follow-up commit.*

---

## 5. Pass / fail

| item | result |
|---|---|
| keys absent / observe off == `1cb9706` (copy: plain + sleepkeys; tests: whole runs) | **PASS** |
| observe-mode daemon config == sleep-off config (copy, byte-identical) | **PASS** |
| observe == real (removal set + order, collected, shield, record, every post-downscale weight): migration, tags, first big clearance, 3 steady-state sleeps on the copy; tests | **PASS** |
| observe writes nothing (checkpoint bytes, state digest, ledger, no events; tests incl. patched graph, vector drop) | **PASS** |
| lock hold ≤ ~0.7 s | **partial**: one hold, median 0.65 s, 5 of 11 above (max 1.70 s) under load 6-11 (real sleep max holds 0.31-0.72 s, save capture 4.2-9.9 s) |
| daemon: switch, pacing, report file, admin command (18 tests) | **PASS** |
| daemon suite vs `3f015f07` | **PASS** (same 75) |
| NG suite vs `1cb9706` | pending (follow-up commit) |

## 6. Risks and open items

1. **The live config while observing (§1.2) is a design choice:** wake removal stays exactly as with sleep off. Confirm
   it, or ask for the alternative (`structural_plasticity_in_sleep` on while observing: nothing removed at all for the
   observe days).
2. **Snapshot hold above target in 5 of 11 passes** (§3.4). A native `SynapseStore` clone would fix it (Rust change +
   wheel). With the dream loop's pacing (≤ ~4 passes a day) it is well below the existing autosave capture.
3. **A projection beyond sleep 1 assumes no wake between the sleeps:** an upper bound on forgetting (spec §3.2). With G =
   2 and unmigrated counters, a *real* armed sleep from today's state removes nothing for two sleeps. The auto pass
   therefore projects 3 to show the clearance, which is what the report's `no_wake_between_projections` says.
4. **CPU:** a pass is 5-11 s of Python + native work in the daemon process, with the live lock free; GIL contention
   slows concurrent steps somewhat. Gated by MemAvailable ≥ 1.5 GB and one pass at a time.
5. **Node metadata is copied, Node shells after the hold.** Only metadata contents and creation_time feed decisions;
   both are consistent with the snapshot. Other node fields are copied after the hold and never read by the sleep.
6. **Not run on the live graph.** Wiring it live (`CC_NG_SLEEP=1 CC_NG_SLEEP_DISUSE=1 CC_NG_SLEEP_OBSERVE=1` + a
   restart, or the admin command after deploying) is the Executive's step after the protected-file "proceed".
7. Not changed and still pending from the spec: the D15 purge (admin command exists, not run), the announced backlog
   clearance (D7), `CC_NG_TONIC_AGES=0` at arming (D10), the stale `.bashrc:242` comment (D6).
