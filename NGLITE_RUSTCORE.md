# NGLITE_RUSTCORE: the Rust `NGLiteCore` made usable through the canonical `ng_lite.py` (punch list #1096, #1098)

Lane `nglite-rustcore` (laptop, 2026-10-13). BUILD lane: branches, copies and a throwaway venv only. Nothing was
installed into `~/NeuroGraph/.venv`, nothing live was touched, nothing was re-vendored outside the three branches below.
Josh's rulings this lane implements (2026-10-13, #1096 row): dial it in on the laptop first and re-vendor later;
**saves are msgpack, never JSON** (a legacy JSON state is converted once on load); `NG_LITE_RUST_CORE` stays only for
the rollout; **fix the three quirks in the Rust core**.

| repo | branch | from | commit |
|---|---|---|---|
| ng-tract-rs | `cc-laptop-nglitecore-quirks-20261013` | `04fc211` | `fef26f5` (quirks + accessors), `a2673f6` (`list_synapses` newest_first / limit; the wheel is built from this) |
| NeuroGraph | `cc-laptop-nglite-rustcore-20261013` | `88576dc` (the live trial root) | `6d76b9c` (code), `0e8a6a9` (newest-rows bucket_recent + test), then this doc |
| Elmer | `cc-laptop-nglite-rustcore-20261013` | `13dde60` (live) | `52aaec5`, `cae7288` (vendored `ng_lite.py` = NG `0e8a6a9`, sha256 `63b250a8…`) |

## 0. TL;DR

- Canonical `ng_lite.py` now has **public read methods** that the Rust core serves when it is on and the Python
  structures serve otherwise. The Commons, Elmer's surgery and the NG tests now read through them, not through
  `ng.nodes` / `ng.synapses`.
- **Saves are msgpack on both cores**, and the bytes are the same: the Python writer's output equals the Rust core's
  `save_binary` for the same state, except the top-level timestamp. `load()` converts a legacy JSON state once. The
  Commons' JSON-in-`commons.msgpack` (#1098) is fixed by this.
- **The three quirks are fixed in the Rust core** (warm-up count persisted, lossless load, read-only
  `get_recommendations`). Against a Python reference with exactly those three corrections, the Rust core is still
  bit-identical everywhere else.
- **Switch off:** the laptop behaves exactly as before except the save format. Golden runs over copies of the real
  CC Commons and Elmer states: 1,400 records, and the only 6 that differ are the save-format / migration records.
- **Switch on:** the CC Commons (through `cc_ng_organism`) and the hosted Elmer NGEcosystem work end to end on copies,
  across a restart. `bucket_recent` returns the same rows as the Python core (276 records, 0 diffs, time masked;
  unmasked on the loaded state). Metric eviction works, the autonomic drop works, and saves go through msgpack.
  118 Commons tests pass on the Rust core.
- **Suites vs bases (switch off):** 0 new failures in the NG suite (163 files, per file, isolated) or the daemon suite
  (1,093 passed; identical FAILED/ERROR id sets). Elmer: 115 passed; one test there is flaky on base too.
- **Install trap found (§7.2):** Elmer's `elmer_hook.py` runs `ng_updater.auto_update()` on import. That re-syncs
  `ng_lite.py` / `ng_ecosystem.py` from `~/NeuroGraph` on every hosted start, which would silently undo the Elmer
  re-vendor unless the canonical source it reads is the new one.

## 1. What was added

### 1.1 `ng_lite.py` public reads (both cores; every row is a snapshot copy)

| method | returns | replaces (consumer side) |
|---|---|---|
| `node_count()` / `synapse_count()` | int | `len(ng.nodes)` / `len(ng.synapses)` |
| `list_nodes(with_embedding=True)` | `[NGLiteNode]` in storage order (embedding = float32 copy or None) | `ng.nodes.values()` |
| `get_node(node_id)` | `NGLiteNode` or None (by `node_id`, i.e. a synapse's `source_id`) | the node_id→node scans |
| `list_synapses(*, source_id, target_id, target_prefix, since, with_metadata=True, newest_first=False, limit=None)` | `[NGLiteSynapse]` in storage order; filters AND; `since` = `last_updated > since`; `with_metadata=False` → `metadata={}`; `newest_first` = stable sort by `last_updated` desc; `limit` = only that many rows are built | `ng.synapses.values()` / `.items()` |
| `get_synapse(source_id, target_id)` | `NGLiteSynapse` or None | `ng.synapses[(s, t)]` |
| `remove_synapses(keys)` | count removed (dict `del` order semantics) | `del ng.synapses[k]` (the only write: Commons windows / autonomic drop) |
| `get_history()` | list copy | `ng._history` |
| `export_state()` | the persisted dict | `ng._export_state()` |
| `state_paths(path)` / `state_exists(path)` | (msgpack path, legacy JSON path) / bool | `.json` existence checks |
| `save(path)` → str | the path written (msgpack) | — |
| `load(path)` → bool | True if a state was loaded | — |

Mutating a returned row does not touch the substrate (tested on both cores).

### 1.2 The switch and the Rust path in `ng_lite.py`

- The Rust core is used **only when `NG_LITE_RUST_CORE` is truthy** (`1/true/yes/on`, the same rule the wheel's export
  gate uses). If the switch is set but `ng_tract.NGLiteCore` cannot be imported, a WARNING is logged and the Python
  core runs. It is no longer a silent `except ImportError: pass`.
- On the Rust path, `ng.nodes` / `ng.synapses` are a **loud stand-in**: any use raises
  `RuntimeError("NGLite.nodes is held by the Rust core ...; use the public reads: ...")`. The old Rust path left them
  as silently empty dicts, so readers saw an empty substrate.
- These all read the core: `get_stats()` (same keys and values as the Python one), `export_state` / `_export_state` /
  `sync_with_bridge`, and `update_tunable` (the core validates and clamps). `config` / `module_id` are refreshed from
  the core after a load.
- `find_or_create_node` returns a fresh node built from the core's answer. This removes the stale stub cached by hash
  (NGLITECORE.md §6.3 item 5).
- Outcomes are counted once, in the core (removes item 6). The in-memory history is kept on both paths.
- The #119 changelog line ("3-5x speedup", "zero API changes") is corrected in the new changelog entry.

### 1.3 New `ng_tract.NGLiteCore` / `_NGLiteCore` methods (additions)

`list_synapses(source_id=None, target_id=None, target_prefix=None, since=None, with_metadata=True)`,
`get_synapse(source_id, target_id)`, `list_nodes(with_embedding=True)`, `get_node(node_id)`, `remove_synapses(keys)`,
(`list_synapses` also takes `newest_first=False, limit=None`),
`get_config()`, `get_module_id()`, `counts()`.

They follow the same three-phase pattern as every other method: Python → Rust values, then `detach` + lock + pure
Rust, then Rust → Python. Rows are tuples in dataclass field order, and `ng_lite.py` wraps them.

## 2. Persistence: msgpack only, with a one-time migration

- **`save(path)`** writes msgpack on both cores, through a temp file + fsync + rename:
  - `x.json` → `x.msgpack`
  - `x.msgpack` (the Commons' name) → itself
  - any other name → as given
- **Content:** the `_export_state()` dict. Embeddings and prototypes are msgpack float32. Map / array headers, int
  widths, str8 and dict-key coercion follow json-style rules, so the Python writer (`_pack_state`) emits the same
  bytes as the Rust core's `save_binary` (tested: equal after masking the top-level timestamp).
- **The Python core needs the `msgpack` package.** There is no JSON fallback, by ruling. Without it, `save` raises
  ImportError naming the package.
- **`load(path)`:**
  - The msgpack save is read on either core. A state written by one core loads into the other: tested both ways
    through `load()`, plus in the e2e runs.
  - **A legacy JSON is converted once** in either of two cases: `x.json` exists and there is no `x.msgpack` (or the
    JSON is newer, with a WARNING); or the `.msgpack` file actually contains JSON (#1098, today's
    `commons.msgpack`). The conversion imports the JSON, writes the msgpack and keeps the JSON as
    `<file>.legacy-json`. That file is never read again.
  - If there is no state, `load` returns False.
- **`ng_ecosystem.py`** (canonical, changed because the API change requires it): `_init_ng_lite()` now calls `load()`
  unconditionally instead of only when the `.json` exists. Otherwise every module would restart empty after its first
  msgpack save.
- **#368** is still open: the Commons suppression map is not persisted. When it is, it goes into this one msgpack
  artifact (noted in `commons.py`).

## 3. The quirk fixes (Rust core, `ng-tract-rs` `fef26f5`)

| quirk | fix | proof |
|---|---|---|
| (a) the warm-up count is lost on a restart before prototypes exist | export writes `receptor_layer = {"input_count": n}` while warming up (n > 0); import restores it with or without prototypes. n = 0 still exports `{}`, like Python. | `test_quirk_a_*` (crate + Rust unit): canonical Python resets to 0; Rust keeps the count; a module restarting every ~40 counted inputs reaches warm-up 60 on its 2nd run |
| (b) embedding drift on every load | import keeps a vector whose norm is within 1e-5 of 1 bit-for-bit, and still normalizes a foreign or damaged one | `test_quirk_b_load_is_lossless`: 4 save/load cycles leave the full state identical on Rust (0 changed values); the canonical Python changes values on the same data |
| (c) `get_recommendations` creates / activates nodes | read-only: snap to the current prototype without drift, count or warm-up bump; look up by exact hash then similarity; an unknown input returns []; nothing is written | `test_quirk_c_*`: full state unchanged (unmasked) after 100+ queries on Rust; canonical Python writes. e2e: Elmer queries created 0 nodes on Rust and 5 on Python |

- **Everything else is bit-identical.** The equivalence suite now compares the Rust core against `RefNGLite`, which is
  the canonical NGLite plus exactly these three corrections, implemented in `tests/test_ng_lite_core.py`.
- The comparison is op by op and on the full state, including saves / reloads / cross-loads, the rim, LRU / weakest
  pruning, k-means, EMA, the bridge paths and threads.
- **The canonical Python core keeps its quirks** (switch off = unchanged behaviour, per the brief). It dies with the
  switch.

## 4. Consumer changes

**NeuroGraph branch:**
- **`commons.py`:**
  - The metric / error / wire windows share one `_evict_to_recent_window`: `list_synapses(target_prefix)`, a stable
    sort by `last_updated`, then `remove_synapses`. The evictions are the same as before.
  - `bucket_recent` asks for the newest rows only (`newest_first` + `limit`). It widens the window when dedup or
    suppression needs more rows, and fetches metadata (`get_synapse`) and the embedding (`get_node`) only for the
    rows it returns. The rows and order are the same as before (tested with 300 duplicates + a suppression on both
    cores).
  - `arousal` uses `list_synapses(target_id=...)`; `_drop_autonomic_synapses` uses `remove_synapses`.
  - `persist` now writes msgpack (#1098).
  - Every read is a snapshot, which also closes #1093's iterate-while-mutated window.
- **`ng_ecosystem.py`:** the unconditional `load()` (§2).
- **`neurograph_gui.py`:** the Elmer panel reads the msgpack save (and falls back to an unmigrated `.json`).
- **Tests:**
  - 10 Commons test files: `._ng.synapses.values()` → `._ng.list_synapses()`.
  - `test_ng_lite.py` TestPersistence follows the msgpack contract.
  - New `tests/test_commons_rustcore_1096.py`: the Commons verbs on both cores.
- No other NG-tree code read NGLite internals (audit). The CC daemon's own code (docs trial `scripts/`) reads none: its
  `g.nodes` / `g.synapses` are the NeuroGraph Graph.

**Elmer branch:**
- `ng_lite.py` + `ng_ecosystem.py` are byte copies of the NG branch. **This is the one laptop re-vendor the live CC
  needs:** the daemon hosts Elmer with namespace isolation (`cc_ng_organism.bootstrap_cc_modules` stashes `ng_lite`
  etc.), so the hosted Elmer loads `~/Elmer/ng_lite.py`, not NG's copy.
- `surgery/extract_features.py` uses the public reads. It is orphan code: nothing calls it.

## 5. Proof results

- **Isolation:** every run went through `~/.cache/nglite-rustcore/run-isolated.sh`. That gives a scratch HOME, a
  private PID namespace, tmpfs over the live `~/.claude/plugins/neurograph` and `~/.et_modules`, read-only live trees,
  a MemoryMax 3–4 GB scope, nice 10, a 4 GB MemAvailable gate, cargo `-j 1`, and one job at a time
  (`~/.cache/nglite-rustcore/queue.sh`).
- **Venv:** `~/.cache/nglite-rustcore/venv` contains only the new wheel; deps come via `.pth`.
- **State copies:** `~/.cache/nglite-rustcore/state/` (cp + cmp):
  - `cc_commons.msgpack`: 4 nodes / 4,615 synapses, JSON inside, sha `4f9f497c…`
  - `elmer_ng_lite_state.json`: 22 rim nodes, 384-d, sha `61a9d19d…`

| item | result |
|---|---|
| crate `cargo test --lib` | **67 passed** (64 + 3 new quirk unit tests), on both builds |
| crate pytests `tests/` (with `NGLITE_PY` = the NG branch `ng_lite.py`; run on both builds, last on the `a2673f6` wheel) | **394 passed** (372 existing + 22 in `test_ng_lite_core.py`: 15 updated + 7 new: quirks a/b/c, public reads on both cores, Python msgpack bytes == `save_binary`, legacy JSON migration, switch-set-but-missing warning) |
| **switch OFF golden** (`drive.py`, old trial root vs NG branch, Python core, fake clock, real-state copies: NG-Lite workloads on the Commons + Elmer states, then a Commons workload: restore, 260 mixed deposits, 230 metric deposits, all `bucket_recent` modes, arousal, persist, restore) | **1,400 records, 6 differ, all expected** (run on both builds, same result): the saved-file first byte (`7b` JSON → `88` msgpack) ×3 and the directory listings (`.msgpack` + `.legacy-json`) ×3. Every result, every state (timestamps included) and every reload is identical. |
| **switch ON Commons rows** (`drive.py` NG branch: Python core vs Rust core, real clock) | Commons workload: **276 records, 0 differ** (time masked). The loaded-state `bucket_recent` (all modes), stats and arousal are equal **unmasked**. The NG-Lite workloads diverge by design: quirk (c) means a query no longer creates a node, so node ids shift. That equivalence is proven against `RefNGLite` instead. |
| **switch ON end to end** (`e2e.py`, two processes = a restart; re-run on the final code; CC Commons via `cc_ng_organism.get_cc_commons` / `persist_cc_commons`; hosted Elmer via Elmer's own `ng_ecosystem.NGEcosystem` with its rim) | **19/19 + 18/18 checks pass.** Commons JSON converted once; `bucket_recent` incl. embeddings; metric window = 200; arousal; persist = msgpack; reload after restart; autonomic dropped on restore. Elmer `ng_lite_state.json` migrated to `.msgpack`; 22 rim nodes; 384-d record works; queries create 0 nodes; tunable survives save → reload. The same e2e with the switch OFF also passes (19/18 checks; queries create 5 nodes there). |
| Commons tests on the Rust core (16 files incl. legs 1–3, bunyan stage 1, ng_commons_eco, cc_commons_persist, the new file) | **118 passed** with `NG_LITE_RUST_CORE=1`. Switch off, the same files + `test_ng_lite.py` + `test_constitutional.py`: 198 passed, plus the 2 `test_ng_lite.py` NGSaaSBridge failures that also fail on base |
| NG suite / daemon suite / Elmer tests vs bases | §5.1 |
| wheel surface vs the live wheel (`.so` `6f03dca0…`) | `only-live: []`, `only-new: ['_NGLiteCore']`, no changed member; with the switch set, `NGLiteCore` is added too. Against the 10-12 lane wheel: the 8 methods above are additions, and the deliberate change is `get_recommendations` being read-only. |

### 5.1 Suites vs bases (switch OFF)

Same venv and harness for both sides. The bases are detached worktrees at the branch points:
`~/worktrees/neurograph-base-88576dc-20261013`, `~/worktrees/elmer-base-13dde60-20261013`, and
`~/worktrees/docs-base-trial-s4a-20261013` (the daemon's scripts at `e4d5dc40`).

- **NG suite** (`~/.cache/nglite-rustcore/suite/`): every `tests/test_*.py` on the branch, one isolated pytest per file,
  600 s cap. Each of the 39 files that did not pass was re-run on the base. **2 ids were only on the branch:**
  - `test_constitutional::test_load_old_state_without_constitutional_field`: it json-loaded the save. The test was
    updated to the msgpack contract and now passes.
  - `test_graph_substrate_race::test_build_adjacency_no_race_under_concurrent_step_mutation`: flaky on the base too.
    Re-runs went branch fail/fail, base fail/pass. It is a Graph race test that does not touch NG-Lite.
  - `test_tonic_no_heuristic` was killed (rc 143) under load on the first branch run. On re-run: 18 passed, like the
    base.
  - Every other failing id is identical on both sides. Examples: `test_ng_tract_bridge` 17/17, `test_cc_dual_pass`
    10/10, `test_migration` 6/6, and `test_ng_lite` 2/2 (the NGSaaSBridge `_sync_count` failures). The
    `test_auto_knowledge` / `test_ces` / `test_coordinator` / `test_et_modules` / `test_openclaw_hook` / `test_snn`
    timeouts happen on both sides.
  - After the final `ng_lite.py` / `commons.py` change, the 42 test files that touch `commons` / NG-Lite /
    `ng_ecosystem` were re-run on the branch: **0 ids beyond the base**.
- **Daemon suite** (docs trial `scripts/tests` + `test_cc_ng_daemon_guard.py`, `CC_NG_PYTHONPATH` = NG base vs NG branch):
  - base: 75 failed / 1,093 passed / 6 skipped
  - branch: 75 failed / 1,093 passed / 6 skipped
  - **identical FAILED/ERROR id sets (77 ids)**
- **Elmer tests:**
  - base: 115 passed in one run; `test_hook::test_start_stop` failed in 2 of 3 runs (flaky: `already_started`)
  - branch: 115 passed in one run; the same test failed in 2 of 4 runs
  - branch with the switch ON: 114 passed + the same flaky test
  - one branch run aborted at interpreter exit ("terminate called without an active exception", after 115 passed).
    It did not recur in 2 re-runs.
  - The Elmer branch runs set `NEUROGRAPH_CANONICAL_PATH` to the NG branch (§7.2). Without it the suite itself
    reverted the worktree's vendored files, which is how the trap was found.

### 5.2 Timings (real CC Commons copy, 4,615 synapses; interleaved Python / Rust in one process, load ~6, nice 10; medians)

| | old code (Python) | new code, Python core | new code, Rust core |
|---|---|---|---|
| `bucket_recent(limit=50)` | ~35 ms | ~7 ms | ~3–4 ms |
| `bucket_recent(limit=20, with_metadata=True)` | ~35 ms | ~7 ms | ~2–3 ms |
| `arousal()` | ~30 ms | ~2.3 ms | ~0.03 ms |
| metric deposit (incl. its window) | ~4.5 ms | ~4.5 ms | ~0.5–0.9 ms |
| `persist` | 7.9 s (7.76 MB JSON) | 0.6–0.7 s (5.99 MB msgpack) | 0.1–0.4 s (same bytes) |

A full `list_synapses()` of 4,615 rows costs ~30 ms (Python) / ~50 ms (Rust: 15 ms in the core, the rest building
Python objects). That is why `bucket_recent` asks only for its newest rows.

## 6. Wheel

`/home/josh/.cache/nglite-rustcore/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`

- sha256 `0c4c59f2aae36ec45bdb3c5fc544c85b2d6d752737522a3317f058b38c59c14d`
- inner `ng_tract/ng_tract.abi3.so` sha256 `5b058be61692e031f147e9f3382c9e08fb5ab51d6b10cb284a12e6d7e6dbd520`
- The first build of this lane (`fef26f5`, `.so` `4a19e8ad…`) is superseded; it is kept in `wheel-prev/`.
- Built from `ng-tract-rs` `a2673f6` `src/` with `~/.cache/nglite-rustcore/build.sh` (`cargo test --lib` +
  `maturin build --release`, abi3-py38, `-j 1`, offline).
- **Rollback wheel = the live one:**
  `/home/josh/.cache/flow-lenia-f1/wheel/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`
  (inner `.so` `6f03dca0…ef50`).

## 7. Install on the laptop (for the Executive; Josh's go first)

### 7.1 Back up first

Back up both states while the daemon is stopped or between saves:
- `~/.claude/plugins/neurograph/checkpoints/commons.msgpack` (JSON today, 7.0 MB)
- `~/.et_modules/elmer/ng_lite_state.json`

### 7.2 Code and wheel, switch still OFF

1. **NG code.** The daemon runs NG from the trial worktree `~/NeuroGraph-worktrees/trial-s4-20261002` (branch
   `cc-laptop-trial-s4-20261002` @ `88576dc`). The branch sits directly on `88576dc`:
   `git -C ~/NeuroGraph-worktrees/trial-s4-20261002 fetch && git -C ~/NeuroGraph-worktrees/trial-s4-20261002 merge --ff-only origin/cc-laptop-nglite-rustcore-20261013`
   (or check the branch out there).
2. **Elmer code (the laptop re-vendor).** `~/Elmer` is on `cc-revendor-ng-20260921` @ `13dde60`:
   `git -C ~/Elmer fetch origin cc-laptop-nglite-rustcore-20261013 && git -C ~/Elmer merge --ff-only FETCH_HEAD`
   (Elmer's remote fetches only `main` by refspec, so name the branch; tip `cae7288`).
   - **TRAP, decide before restarting.** `elmer_hook.py:160` runs `ng_updater.auto_update()` on import: a
     `git pull --ff-only` in `~/Elmer`, plus copying `ng_lite.py`, `ng_ecosystem.py`, `ng_autonomic.py`,
     `openclaw_adapter.py` and `ng_updater.py` from the canonical source over any copy that differs.
   - The canonical source is `$NEUROGRAPH_CANONICAL_PATH`, else `~/NeuroGraph`.
   - `~/NeuroGraph` main still has the old `ng_lite.py` (`b477c1cf`). So the first hosted Elmer start would put the
     old file back. The lane's own Elmer suite did exactly that to the branch worktree.
   - Pick one:
     - **(B, recommended for "laptop first")** Add `"env": {"NEUROGRAPH_CANONICAL_PATH":
       "~/NeuroGraph-worktrees/trial-s4-20261002"}` to the `elmer` entry of the CC registry
       `~/.claude/plugins/neurograph/et_modules/registry.json`.
       - `cc_ng_organism.bootstrap_cc_modules` applies a module's `env` before loading its hook. Elmer then syncs from
         the daemon's own NG root, which has the new files.
       - Tested: with the variable set, the Elmer suite left the vendored files byte-identical.
       - The other three synced files (`ng_autonomic.py`, `openclaw_adapter.py`, `ng_updater.py`) are byte-identical
         across `~/NeuroGraph`, the trial root and `~/Elmer`, so B changes nothing else.
       - Caveat: the registry is JSON config outside `.bashrc` (LAW 5 reading is Josh's call), and `os.environ` is set
         process-wide in the daemon.
     - **(A)** Put the new `ng_lite.py` + `ng_ecosystem.py` on `~/NeuroGraph` main as well. Cherry-pick only those two
       files: main is not on the trial line, and its copies are byte-identical to `88576dc`'s.
       - **This starts the ecosystem re-vendor on the laptop.** TID (live; `app.py` runs `auto_update()`), TrollGuard,
         Immunis, THC and Bunyan copy the new files at their next start.
       - TID would then migrate `/home/josh/The-Inference-Difference/ng_lite_state.json` to msgpack.
       - TID's `reset_receptor_prototypes.py`, `install.sh` and tests expect the `.json` (§8).
3. **Wheel:** `~/NeuroGraph/.venv/bin/pip install --no-deps --force-reinstall <wheel above>`
4. **Confirm:**
   - `~/NeuroGraph/.venv/bin/python -I -c "import ng_tract; print(hasattr(ng_tract,'_NGLiteCore'), hasattr(ng_tract,'NGLiteCore'))"` → `True False`
   - `sha256sum` of the `.so` → `5b058be6…`
5. **Restart `cc-ng-daemon`** under the sidecar procedure: every previous daemon PID dead first.
6. **What happens on that first start, even with the switch OFF:**
   - `get_cc_commons` restore converts the JSON inside `commons.msgpack` to real msgpack and keeps
     `commons.msgpack.legacy-json`.
   - The hosted Elmer's NGEcosystem migrates `ng_lite_state.json` → `ng_lite_state.msgpack` and keeps
     `ng_lite_state.json.legacy-json`.
   - Expect the log lines `NG-Lite: converted the JSON inside …` and `NG-Lite: migrated legacy JSON …`.
7. **Watch ≥ 2 Commons save cycles.** `commons.msgpack` mtime should advance, its first byte should not be `{`, and
   `CC Commons restored from …` should report the same synapse count after the next restart.

### 7.3 Turning the switch ON

1. **Add `export NG_LITE_RUST_CORE=1` to `~/.bashrc`** (LAW 5).
   - `cc-ng-service.py prepare` passes every `CC_|NG_|ET_` export into the daemon's launch environment; a systemd
     `Environment=` line would be dropped by `launch_environment`.
   - **Caution:** with the export in `.bashrc`, *every* shell-started Python on `~/NeuroGraph/.venv` sees it.
   - The wheel then exports `NGLiteCore`, and any **old** `ng_lite.py` imported there (the `~/NeuroGraph` main
     checkout, other worktrees) takes the broken old Rust path: empty `nodes` / `synapses`, msgpack-only save.
   - So land this NG branch on `~/NeuroGraph` main too before setting it, or confine such runs.
   - Nothing else on the laptop runs on that venv except `cc-callosum-leg2`, which builds no NGLite.
2. **Restart the daemon** (sidecar procedure). The CC Commons and the hosted Elmer switch together.
3. **Confirm:**
   - no `runs the Python core` WARNING in `daemon.log`
   - `CC Commons restored … synapse_count` equals the pre-switch count
   - `bucket_recent` consumers (Elmer proto River feed, `tid_peninsula_commons`) keep getting rows
4. **Watch ≥ 2 save cycles.**

### 7.4 Rollback

- **Switch:** remove the export and restart. The Python core reads the Rust core's msgpack directly, so no conversion
  is needed.
- **Wheel:** `pip install --no-deps --force-reinstall` the F1 wheel above. With the switch off the new `ng_lite.py`
  never touches `ng_tract`.
- **Code:** reset the trial worktree and `~/Elmer` back to `88576dc` / `13dde60`, and remove the registry `env`
  (option B). `auto_update` then keeps Elmer on main's old copy. **This needs a state conversion first.** The old code
  reads only JSON:
  - the old Commons would fail to parse the msgpack and start fresh
  - the old Elmer would find no `.json` and start with only its rim

  Convert with:

```python
import json, msgpack, os
def back(p_msgpack, p_json):
    st = msgpack.unpackb(open(p_msgpack, "rb").read(), raw=False, strict_map_key=False)
    tmp = p_json + ".tmp"; json.dump(st, open(tmp, "w"), indent=2); os.replace(tmp, p_json)
back(CC + "/checkpoints/commons.msgpack", CC + "/checkpoints/commons.msgpack")   # the Commons: same name
back(EM + "/ng_lite_state.msgpack", EM + "/ng_lite_state.json"); os.remove(EM + "/ng_lite_state.msgpack")
```

  The `.legacy-json` files are the pre-migration states. They are older than anything learned since the switch.

## 8. The later re-vendor (per module; from the consumer audit of 2026-10-13)

For every module: copy the canonical `ng_lite.py` + `ng_ecosystem.py` from this NG branch once it lands. Then:

1. Make sure the module's interpreter has **`msgpack`**. The Python core has no JSON fallback.
2. Replace every reach into `ng.nodes` / `ng.synapses` / `_history` / `_total_outcomes` / `_export_state` with the
   public reads.
3. Replace `if <x>.json exists: load()` with an unconditional `load()` (or `NGLite.state_exists(path)`).
4. Retarget anything that opens an NG-Lite `.json` state file directly to the `.msgpack` (or to `NGLite.load` /
   `export_state`).

`ng_ecosystem.py` itself exists in 4 drifted variants (A `17ca9ac1`, B `d78689af`, Morphogenesis `1efdf8e8`, UniOS
`41fe79f2`). All of them gate `load()` on `.exists()`, so all need the canonical copy.

| module | NGLite internals to move to public reads | load/save path work | interpreter |
|---|---|---|---|
| **TID** (`The-Inference-Difference`; live process runs `The-Inference-Difference-vps` code) | none | `app.py` fallback `os.path.exists(state_path)` → `load` (TID `:864-868`, vps `:883-886`); **`reset_receptor_prototypes.py` edits `ng_lite_state.json` directly**; `install.sh` sizes / copies the `.json`; vps `app.py:857` hard-codes TID's `.json` path (fine after: `load` maps it); tests `test_ng_lite.py` json-loads the save | TID venv: msgpack yes, ng_tract no (Python core) |
| **TrollGuard** | `main.py:178` takes `eco._ng` (public use only) | `main.py:489-492` `save("ng_lite_state.json")` → writes `.msgpack` (fine) | none found |
| **Immunis** | none (tests read `commons._ng.synapses`) | none active | none found |
| **THC** | `health_monitor.py:351,357` reads `_eco._ng.config` (public attr, fine) | `healing_collective_hook.py:860-864` checkpoints the `.json` if it exists → must copy the `.msgpack`; `repair_primitives.py` `NGLiteRebalance` / `CheckpointRestore` json-load / dump a peer state → msgpack | none found |
| **Bunyan** | `core/narrative.py:805` `_ng.synapses.items()` (backward synapses by target) → `list_synapses(target_id=...)`; `:910` `nodes.values()` scan by node_id → `get_node()` | adapter `.json` path (fine via the canonical ng_ecosystem) | none found |
| **Darwin** | `darwin/surrogate.py:572` `len(nodes) if hasattr(...)` → `node_count()` (on the Rust path `hasattr` is True and `len` raises, which is loud, not silently 0) | `surrogate.py:561-564` `.exists()` on `ng_lite_checkpoint.json` → unconditional `load`; `:521` save is fine | none found |
| **QuantumGraph** | `quantumgraph.py:134-135` `len(nodes/synapses)` → counts; **`:266` the interference loop over `synapses.items()`** → `list_synapses()` | `:128-131` `.exists()` gate → unconditional `load` | none found |
| **Praxis** | none | none active; `vendored/ng_lite.py` is the drifted `69c19ed0` (lacks #92) | none found |
| **Morphogenesis** | `ng_substrate.py:620` `_commons._ng.nodes` (rim count; **raises if low**) → `list_nodes(False)`; `darwin_brain/surrogate.py:494` `synapses.values()`, `:689` `len(nodes)`; its **own `commons.py` copy** (`82fcd8d3`, older than canonical) reads / deletes `_ng.synapses` / `_ng.nodes` → take the canonical `commons.py` changes; tests assign into `g.nodes[...]` | `surrogate.py:678-681` `.exists()` gate; `~/.morphogenesis/ng_lite_state.json` is a full Graph checkpoint (#1099) | `.venv` has **no msgpack** (install it) |
| **ecosystem-monitor** | none | via ng_ecosystem (canonical copy fixes the gate) | system python (msgpack yes) |
| **Animus** | none | **`anima_gui.py:1008-1009` opens `~/.et_modules/elmer/ng_lite_state.json` directly** → read the `.msgpack`. **This one matters on the laptop as soon as the Elmer branch is installed:** its Elmer panel goes "file not found" after the migration. | Rust binary + Python GUI |
| **Faux_Clawdbot** (`elmer/`, `neurograph/` copies) | same lines as Elmer surgery / NG `commons.py` | `neurograph_gui.py:944` opens the Elmer `.json` | venv: msgpack yes |
| **UniAI** (`splat_poc/hf_space/`) | own `commons.py` (`12704af4`) reads / deletes `_ng.synapses`; `ng_lite.py` drifted `69c19ed0` | `neurograph_gui.py:944` Elmer `.json` | `splat_poc/.venv`: **no msgpack**, old ng_tract |
| **UniOS** | none | converter `~/.et_modules/converter/ng_lite_state.json` via its own drifted `ng_ecosystem` (`41fe79f2`); `vendored/ng_lite.py` drifted `3f239f21` (#92, #55 missing) | none found |
| **Condensate** | no Python NG-Lite; carries a dead `ng_lite_core.rs` draft (#1101) | — | — |
| **NeuroGraph (Syl, VPS)** | done in this branch | Syl's `neurograph_rpc.py` restore / persist of `data/checkpoints/commons.msgpack` converts her JSON Commons once on the first start with this code | VPS venv: check msgpack |

The VPS wheel is a separate build / install (AVX-512 host; NGLITECORE.md §3: bitwise vs the laptop's numpy does not
carry over there, though the Rust core is deterministic).

## 9. Risks, found-not-fixed

- **The `.bashrc` export reaches every shell on the NG venv** (§7.3). Any old `ng_lite.py` imported there flips to the
  broken old Rust path. Land the canonical change on `~/NeuroGraph` main before switching.
- **#1097 (reported, not fixed): Elmer's rim is 384-d.**
  - Every 768-d `record_outcome` / `get_recommendations` / `detect_novelty` against Elmer's local NG-Lite raises
    ValueError on both cores. e2e: Rust `embedding length 768 does not match a stored embedding of length 384`;
    Python `matmul … mismatch`.
  - The fix is re-embedding `~/Elmer/data/constitutional_embeddings.json` with the 768-d model (+ a rim migration).
    That is not trivial, so it is out of scope.
- **`auto_update` re-vendors from `~/NeuroGraph` on every module start** (Elmer, TID, TrollGuard, Immunis, THC,
  Bunyan). Whatever lands on `~/NeuroGraph` main reaches those modules at their next start, with no per-module gate.
  That is the real re-vendor mechanism on this laptop (§7.2).
- **New, found here, not fixed (punch list #1106):** `Elmer/runtime/engine.py:380-381, 501-502, 654-656` read
  `self._ecosystem._graph`,
  but NGEcosystem has no `_graph` (it is `_ng`).
  - So the TuningSocket is never wired to NG-Lite, and `context["live_graph"]` is always None. The `hasattr` checks
    hide this.
  - Pre-existing. Punch-list candidate.
- **Queries no longer write (quirk c).** With the switch on, `Commons.bucket()` / `get_recommendations()` stop creating
  nodes and bumping activation counts. That is the ruling, but the medium's node-LRU now sees only deposits.
- **Bitwise "same rows"** is against the laptop's numpy / OpenBLAS (NGLITECORE.md §3), as before.
- **`list_synapses` cost.** A full listing builds one Python object per synapse (~30–50 ms at 4,615). The Commons
  now asks only for the rows it needs (§5.2). Other future consumers should use `newest_first` / `limit` / the filters
  the same way.
- Still open from NGLITECORE.md §8: #1093 (now mitigated by snapshot reads), #1099 (Morphogenesis state is a Graph),
  #1100 (ng_ecosystem swallows load errors: a corrupt msgpack still leaves `_ng = None`), #1101 (dead drafts),
  #1102 (drifted vendored copies).
