# Native node store — P1 (review branch)

*2026-10-05 · lane nodestore-p1 (bounded worker) · review branches only, nothing merged, nothing deployed.*

Spec: `~/docs/superpowers/specs/2026-10-05-native-node-store-design.md` (P1 row of §9, Josh's decisions at the end:
build order P4a→P1→P2a→P2b; D1 order-preserving removal with tombstones; D3 a `NodeRef` to a removed node raises
`KeyError`; D6 wheel ABI floor stays Python 3.8; D7 one "proceed" per phase, P1 merges switched OFF; D2/D4/D5 deferred).

| Repo | Branch | Base | Commits |
|---|---|---|---|
| ng-tract-rs | `cc-laptop-nodestore-p1-rs-20261005` | `origin/cc-laptop-canonical-wheel-20261004` @ `79be810` | `480d7f3` NodeStore + tests · `c20aeb5` chunked pack (reverted, `fd49c16`) · `12d72a2` pack buffer reserve · `be4ba0f` changelog · `71a9d2f` cached NodeRef per row (identity), mapping `==`, GC traverse |
| NeuroGraph | `cc-laptop-nodestore-p1-20261005` | `origin/cc-laptop-trial-s4-20261002` @ `0595221` | **PROTECTED** `neuro_foundation.py`: `0784a9f`, `dd6b42a`, `39100b7` · tests `2a53ed7`, `d27713f` · this doc |

Wheel (throwaway, installed only into `/tmp/nsp1/venv`):
`/home/josh/worktrees/ng-tract-rs-nodestore-p1-20261005/target/wheels/ng_tract-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`,
sha256 `a0d495280a8bb6d9d681ef88e284b50b93563dd78a4e79586abbcb3b32a17170` (abi3, Python ≥ 3.8). Not installed into
`~/NeuroGraph/.venv` or any shared site-packages.

---

## 1. What was built

### Rust: `ng_tract.NodeStore`, `NodeRef`, `SpikeHistoryView` (`src/node_store.rs`, additive)

- **Columns** (parallel Vecs, row order = insertion order): 8 × f64 (`voltage`, `threshold`, `resting_potential`,
  `last_spike_time`, `firing_rate_ema`, `intrinsic_excitability`, `Ca_i`, `pred_error_ema`), 4 × i64
  (`refractory_remaining`, `refractory_period`, `diffpc_layer`, `creation_time`), `is_inhibitory` u8, `manifold_type`
  u32 index into an interned name table (any string, never rejected), spike ring `Option<Box<VecDeque<f64>>>` allocated
  on first spike + per-row capacity, live Python handles for `metadata` and `pred_weights` (an empty `pred_weights`
  is materialised — created outside the borrow — on first read).
- **Identity**: `ids`, `id → row` map, a per-row **generation** (unique per insertion; 0 = tombstone).
- **D1 removal**: `del` tombstones the row; iteration skips tombstones; re-inserting a removed id appends at the end;
  update of an existing id stays in place (all CPython dict semantics). When tombstones exceed 1/4 of rows a stable
  compaction runs and bumps `layout_epoch` (exposed for P2 row caches). Property-tested in Rust against a dict oracle
  (39 seeds × 600 random ops).
- **D3 / NodeRef**: holds `(store, node_id, generation, cached row)`. Every access checks the cached row's generation
  (no string hash on the fast path); after a compaction it re-resolves by id. A removed node — or a different node
  later created under the same id — raises `KeyError(node_id)`.
- **Exact overflow cell (D4 left open)**: a value that is not the field's canonical type (an `int` written to a float
  field, a non-int / > i64 value in an int field, a non-`bool` `is_inhibitory`, a non-`str` `manifold_type`, a
  non-float spike history, a history object assigned by a caller) is kept as **the exact Python object** in a sparse
  `(row, field) → object` map. Nothing is coerced, so reads and checkpoint bytes are exactly the dict path's. The
  census found none of these in the live data (`stats()` reports 0 overflow cells on the checkpoint copy); the cell
  exists so the byte-identity bar holds by construction rather than by audit. Josh's D4 choice (coerce vs exact) is
  still his: this implements the exact option because it is the one that cannot change behaviour.
- **Identity**: one `NodeRef` object per live row, cached and returned by every `[]`, `.get`, `.values()`,
  `.items()` — so `graph.nodes.get(id) is node` holds exactly as with the dict of Node objects (`cc_ng_organism.py`'s
  #904 rollback uses it at :3605/:5617/:5655; the spec's census missed these). The store ↔ ref cycle is collectable
  (`__traverse__`/`__clear__`). `nodes == {}` / `nodes == dict` compare like a dict.
- **Mapping API** (what `graph.nodes` callers use): `in`, `len`, `bool`, `[]`, `.get(k[, d])`, `[k] = node`
  (Node / NodeRef / duck-typed, every attribute read before borrowing), `del`, `.pop(k[, d])` (detached `Node`),
  `.clear()`, `.keys()/.values()/.items()/iter()` as insertion-order snapshots. Non-str keys behave like a dict
  (`5 in nodes` is False, `nodes.get(None)` is the default, unhashable keys raise `TypeError`).
- **SpikeHistoryView**: `append`, `len`, `iter`, `to_list`, `capacity`, `repr` (same text as `RingBuffer`). A
  non-float append moves that node's history to a real Python `RingBuffer` (exact).
- **Serializer** `to_checkpoint_msgpack()`: packs `{node_id: {19 keys}}` in `_serialize_node`'s key order with
  msgpack-python's exact type/width choices (float64, minimal ints, str8 with `use_bin_type`, bin, `-inf`/`+inf`
  `last_spike_time` → nil). `metadata` / `pred_weights` / overflow values are packed by a native walker over exact
  `dict/list/tuple/str/bytes/int/float/bool/None`; anything else (numpy scalars, dict subclasses, ExtType …) goes
  to `msgpack.packb` for that one value. Strings are encoded through `PyUnicode_AsUTF8String` (abi3-py38), so packing
  leaves **no cached UTF-8 copy** on the `str` (D6, verified: `sys.getsizeof` unchanged; msgpack-python grows it).
- **Loader** `bulk_load_msgpack(raw)`: a direct msgpack → Python decoder with `Unpacker(raw=False)` semantics
  (str keys interned, `strict_map_key`, float32 → float, ext via `msgpack.unpackb`), `_deserialize`'s field defaults,
  `-inf` for a nil `last_spike_time`, `RingBuffer.from_list` semantics (keeps the last `capacity` values) and **P4a
  text sharing** (top-level metadata `str` values of ≥ 256 code points shared across the load when equal).
- **25c5f52 rule**: no Python code runs while the store's RefCell is borrowed. Arguments are read before borrowing;
  every Python handle that must be released (old metadata, removed rows, overflow values, `clear()`) is moved out
  and dropped after the borrow; the serializer snapshots rows into byte pieces + handles under a shared borrow and
  walks the handles afterwards. Tested with Python code that re-enters the store mid-pack (a dict subclass whose
  `items()` writes to the store) and during release (`__del__` that inserts/deletes), plus a 6-thread hammer at a
  1 µs switch interval (readers, writers, churn, pack + reload): no `Already borrowed`.

### NeuroGraph: `neuro_foundation.py` (PROTECTED, commit `0784a9f`, the only NG code change)

- `Graph(config=None, *, native_node_store=None)`. `_native_node_store_wanted()`: the installed `ng_tract` must have
  `NodeStore` **and** the caller opts in — the keyword when given, else a process default that only a **host** sets,
  through `set_native_node_store_default(True)` (returns the previous value). Default OFF → `self.nodes` is today's
  dict and every line of that path runs unchanged.
- **Why not a config key, and why no env read in the engine** (the lane asked for config + an `NG_*` env read at the
  host boundary): `graph.config` is saved inside the checkpoint — a key in `DEFAULT_CONFIG` would change every OFF
  checkpoint vs the trial tip, and a key set only when ON would make ON and OFF checkpoints differ; both fail the
  proof bar. And `neuro_foundation.py` reads no environment by its own rule ("the host reads its own configuration
  (LAW 5) and hands it over", enforced by `test_fair_chance_window`'s static check, which caught my first draft's env
  read). So the host reads `NG_NATIVE_NODE_STORE` and calls the setter before constructing its graph. **No host calls
  it yet** — the laptop daemon (`~/docs/.claude/worktrees/trial-s4a-20261002/scripts/cc-ng-daemon.py`) is live and out
  of scope; wiring it is the enable gate (D2). The setting is process-wide: a host that shares its process with
  another graph (the VPS runs Syl and CC in one process) must set, construct, and restore.
- Reads of `self._native_nodes` outside `__init__` use `getattr(self, "_native_nodes", False)`: duck-typed fakes that
  call Graph methods with their own `self` (`tests/test_capture_borrowed_state_423.py`) take the dict path.
- When ON: `create_node` returns the live `NodeRef` (callers mutate the returned node); `_serialize_full` puts
  `nodes.to_checkpoint_msgpack()` bytes under `"nodes"` (no `_serialize_node` loop, no deep copy — the bytes are
  detached by construction); `write_checkpoint` splices pre-packed `bytes` for `"nodes"` exactly as for `"synapses"`;
  `restore` slices the raw nodes bytes; `_deserialize` bulk-loads them and then sets up `_outgoing/_incoming/
  _node_hyperedges/_recent_spikes` per node in the same order as the loop. Dict-form nodes (legacy JSON) keep the
  existing loop (which inserts through `NodeStore.__setitem__`). INCREMENTAL and `extract_subgraph` keep calling
  `_serialize_node`, which works unchanged on a `NodeRef`.
- `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py` and every vendored file: **unchanged**
  (`git diff 0595221 --stat` lists `neuro_foundation.py`, `tests/test_nodestore_p1.py`, `NODESTORE_P1.md` only).

---

## 2. Proofs

### 2.1 OFF is byte-identical to the trial tip `0595221`

- `tests/test_nodestore_p1.py` (BASE = `git show 0595221:neuro_foundation.py` imported as a second module):
  graphs **built** through `create_node`/`remove_node`/re-add (6 seeds) and **30-round whole runs** (6 seeds × 5 flag
  sets: absent, off, lifeline, budget, lifeline+budget) — stimulate + `step()` + write-mode `prime_and_propagate`
  (Tonic tick) + recall + rewards + node create (+ wiring) + `remove_node` + isolated old nodes swept by the orphan
  sweep inside `step()` + a mass removal (> 1/4 → compaction when ON) + `activation_persistence.capture` +
  `compete_protected_links` (want-hub competition) + `sleep_downscale` + a checkpoint every 10 rounds. Per-step fired
  ids, fired entries, return values, captures, snapshots and final checkpoint bytes all equal. INCREMENTAL capture,
  `extract_subgraph`, `get_telemetry` equal.
  - final wheel installed (`/tmp/nsp1/venv`): **86 passed, 2 skipped** (the two optional real-checkpoint cases);
  - canonical wheel without `NodeStore` (`/tmp/nsp1/venv_base`): **40 passed, 8 skipped** (ON cases skip).
- **Cross-process** (`PYTHONHASHSEED=0`, separate processes, final code): trial-tip worktree + canonical wheel,
  branch OFF + canonical wheel, branch OFF + new wheel, branch ON (the script acting as host) — 2 seeds ×
  {defaults, lifeline+budget}: identical trace sha and checkpoint sha in all four, and restore→re-save identical.

### 2.2 ON behaves the same

- Same file, ON column: every built graph, every whole run (30/30 flag × seed cases) and every capture path equals
  the trial tip — per-step outputs and checkpoint bytes (ON == base == OFF).
- **Both directions**: a checkpoint written ON is restored by the trial-tip module, by OFF and by ON, and each
  re-saves identical bytes; OFF/base-written files restored ON re-save identically (6 seeds).
- NodeRef contract inside the engine: `create_node` returns a live view, D3 `KeyError` after `remove_node` and after
  re-creating the id, D1 order (re-added id at the end), opt-in resolution (the host switch needs the wheel; keyword wins;
  nothing lands in `config`).

### 2.3 Checkpoint COPY (`/tmp/rhs4/ckpt/main.msgpack`, mode 444, sha256 `3c2b6995…`, 251,948,729 bytes, 10,158 nodes)

One process per mode, `ulimit -v 3500000`, `nice -n 10`, `PYTHONHASHSEED=0`, load-gated (load 4–8 throughout:
**compare ratios, absolute times are ±50%**). Base = clean `0595221` worktree.

- **Bytes**: restore → re-save sha256 `39ea9737…` for base, OFF and ON (every run); after 3 stimulated steps the
  checkpoint sha `0603a668…` and the fired-id trace (645/990/689 fired) are identical in all three.

| | base | OFF | ON | ON − OFF |
|---|---:|---:|---:|---:|
| RSS after restore (MB) | 623 | 622–623 | 590–591 | **−32** |
| RSS after 1st save | 655 | 654–655 | 589–591 | **−64** |
| RSS after 2nd save | 672 | 671–672 | 589–591 | **−81** |
| large metadata text heap after restore → after 1st save (MB) | 246 → 351 | 246 → 351 | 246 → 246 | **−105** (UTF-8 copies not attached) |
| peak RSS (MB) | 1062 | 1061–1062 | 1040–1041 | −21 |
| Rust node columns (`stats()`) | | | 6.7 MB, 0 overflow cells | |

Spec prediction: −24 MB node objects (measured −32 MB at restore) and −139 MB UTF-8 copies (measured −105 MB of
text heap: P4a, already merged, had cut the number of distinct texts the copies attach to). Same numbers on every
run and every wheel revision.

Times (s; final wheel; ranges over runs; "interleaved" = OFF and ON alternated in one process):

| | OFF | ON |
|---|---:|---:|
| whole `restore` (interleaved, 3 each) | 4.7–6.8 (median 4.8) | 4.3–6.0 (median 5.3) — a wash |
| nodes section only: decode + build (interleaved) | 1.5–3.3 (msgpack + `_deserialize`) | 0.85–1.6 (`bulk_load_msgpack`) |
| `capture_checkpoint` FULL (held under `_step_lock`) | 1.6–2.9 (base 1.6–1.9) | 2.3–2.5 (one cold run 5.5–7.8) |
| `write_checkpoint` | 2.1–3.1 (base 1.8–2.2) | 0.56–1.4 |
| capture + write per save | ≈ 3.6–5.7 | ≈ 3.1–3.5 (warm) |
| `step()` (interleaved, 10 steps, same stimulation, identical fired ids) | median 1.69 / 2.20 (two sessions) | median 2.06 / 2.60: **+0.38–0.40 s, +18–22%** |

Read honestly:
- **`step()` is slower ON** (+22%), as the spec predicted for P1 (proxy reads on every Python node loop). P1 must
  stay off until P2a/P2b convert those loops (D7).
- **Save total is equal or a little faster, but more of it is under the lock**: the whole nodes pack now runs
  inside the capture (the coherent point), +0.3–1 s of `_step_lock` hold, and it **re-encodes ~145 MB of text to
  UTF-8 on every save** — the price of D6's saving. OFF pays the encoding once (first save) and then packs from the
  cached copies it keeps (the +105 MB). The spec's "nodes pack 0.1–0.3 s" did not include that encoding
  (measured: native nodes pack 1.0–2.0 s; msgpack-python's first metadata pack 4.8 s, later ones 0.8 s from cache).
- Restore: the nodes section is ~2× faster ON; the whole restore is a wash (other sections dominate).

### 2.4 Rust and Python store tests

- `cargo test`: 26 passed (6 new: ring = `deque(maxlen)`, `from_list` tail, dict-oracle churn, compaction /
  generations, msgpack int widths, manifold interning).
- `tests/test_node_store.py` (throwaway venv, final wheel): 17 passed — dict semantics, random churn vs dict + refs across
  compactions, write-through/types, exact overflow cell, byte identity on edge rows (NaN/±0.0/±inf, empty/full ring,
  capacity 0 and 7, int width boundaries incl. ±2^63 and 2^64−1, non-ASCII + astral text, `bytes`, nested lists and
  tuples, str/bin width boundaries, unknown manifold, numpy scalars and `OrderedDict` via fallback, a tombstoned then
  re-added id), round trip vs a `_deserialize` replica, defaults / legacy shapes / float32, P4a sharing + interned
  keys, no UTF-8 cache, spike-history view + non-float migration, `pop`, borrow rule, concurrency hammer, NodeRef
  identity across compaction / update / re-creation, GC of the store ↔ ref cycle, mapping equality.
- Existing crate tests on the new wheel: `test_store.py` 39, `test_hotpaths.py` 265, `test_btf.py` 16 — all pass.

### 2.5 Suites

One pytest process per file, `nice -n 10`, gated on load1 ≤ 7, `PYTHONHASHSEED=0`, 600 s cap, same interpreter
(`/tmp/nsp1/venv`, final wheel) for all three passes. 112 files: every `tests/test_cc_*.py`, the previous lane's 83-file
list (`test_rust_hotpaths_onto_s4.py`, `test_rust_hotpaths_equivalence.py`, `test_prune_lifeline.py`,
`test_strength_budget.py`, `test_want_hub_competition.py`, `test_fair_chance_window.py`, …), every test file that
touches `.nodes` / `create_node` / `checkpoint` / `restore` / `capture_checkpoint` / `_deserialize` /
`extract_subgraph` (101), `test_p4a_text_share.py`, `test_vdb_lock_leak.py`, `test_nodestore_p1.py`.
Base = clean worktree of `0595221` (`/home/josh/worktrees/ng-nodestore-p1-base-0595221`). OFF = branch, default.
ON = branch with the host switch set at import (`/tmp/nsp1/venv_on`: a `.pth` hook calls
`set_native_node_store_default(True)` when `neuro_foundation` loads — test harness only).
Golden checkouts: `WANT_HUB_BASE_CHECKOUT=/home/josh/worktrees/ng-want-hub-base-26a0a11-canon-20261004`,
`PRUNE_LIFELINE_BASE_CHECKOUT=/home/josh/worktrees/ng-prune-lifeline-base-20261004`,
`STRENGTH_BUDGET_BASE_CHECKOUT=/home/josh/worktrees/ng-s4-base-20261004` (read only). Logs: `/tmp/nsp1/suite/{base,off,on}/`.

**Result: no new FAILED/ERROR id, OFF or ON.** 81 files clean on all three. Pre-existing on base, identical ids on
OFF and ON (not investigated — outside this lane): `test_cc_bind_atomic_904` (2), `test_cc_drain_pacing_seam` (4
errors), `test_cc_dual_pass` (6), `test_cc_merge_whole_graph_guard` (1), `test_cc_pith_off_budget_812` (1),
`test_cc_recall_reporting` (1), `test_cc_refeed` (1), `test_cc_retrieval_enrichment` (1), `test_cc_swallows_915` (1),
`test_cc_topology_callosum` (1), `test_cc_topology_capture_423` (4), `test_ces` (4), `test_checkpoint_enforcer` (1),
`test_conversational_recall` (2), `test_fair_chance_window` (1), `test_graph_substrate_race` (1), `test_ingestor` (2),
`test_integration` (1), `test_migration` (6), `test_ng_lite` (2), `test_prediction` (2), `test_reach_teaching` (1),
`test_surface_resolver` (2), `test_tonic_bridge` (1), `test_tonic_habituation` (8), `test_tonic_spine_anchoring` (1);
`test_snn` hits the 600 s cap and `test_auto_knowledge` / `test_openclaw_hook` end rc 143 on all three, as before.

Two exit-code differences, both re-run and cleared: `test_harvest_orphan_seeds.py` OFF timed out once under load
(re-run: 2 passed, same as base); `test_surfacing_whole.py` ON errored at collection under the first ON harness
(a `PYTHONPATH` sitecustomize, which changed where that file's module-origin guard found `cc_ng_organism`); with the
`.pth` harness it passes (47 passed, 2 xfailed, = base).

Found and fixed during the suites (so they now pass ON): `cc_ng_organism`'s `graph.nodes.get(id) is node` (→ the
NodeRef cache), `graph.nodes == {}` in `test_cc_bind_atomic_904` (→ mapping equality), duck-typed `self` in
`test_capture_borrowed_state_423` (→ `getattr` default), and the engine env read (→ host switch).

---

## 3. Deferred to P2 (and later)

- **P2a**: whole-population node passes as `NodeStore` methods (decay, fire detect, Ca, fire, refractory, firing
  EMA / thresholds, excitability, `columns()` for Tonic / persistence / telemetry). Rows holding an overflow cell must
  take the Python path in those methods (none on the live data).
- **P2b**: `SynapseStore.stdp_pass(node_store, …)` + plain propagation rows; exp exactness gate.
- **Enable gate** (D2): only after P2a+P2b, laptop daemon only, ≥ 2 daemon cycles with `memory_report`, a dict-path
  shadow save sha, and a timing gate. The save-lock hold (above) is part of that gate.
- `serialize_one` (native single-row dict) was not added: INCREMENTAL and `extract_subgraph` use `_serialize_node`
  through `NodeRef`, already proven equal; one dict builder.
- Optional: encode each shared (P4a) text once per pack (≈ 25% of the per-save encoding); a native text table is P4b.
- D4 (coerce vs exact overflow) and D5 remain Josh's.

## 4. Worries

0. **The opt-in deviates from the lane's wording** (no config key, no env read in the engine; host setter instead) —
   forced by the byte bar and by the engine's no-environment rule; see §1. The host wiring is not done.
1. **Outside-lock readers and D3.** Tonic scans, surfacing and `activation_persistence.capture` iterate
   `graph.nodes.items()` without `_step_lock`; ON, a node removed meanwhile raises `KeyError` on its stale ref where
   the dict path returned the detached object. Audit before enabling (spec §10.2).
2. **Lock hold during save** grows ON (the nodes pack + per-save UTF-8 encoding run inside the capture, see 2.3).
5. The spec's identity census ("no caller uses `is`") was wrong; the cache fixes it, but P2 work should assume
   callers rely on dict-of-Node semantics beyond the attribute contract and keep running the suites ON.
6. Vault docs (`~/docs/modules/NeuroGraph.md`, the spec's status line) are not updated from this lane — for the
   Executive at integration.
3. **Pre-existing, outside this lane — flagged, not fixed:** `SynapseStore.to_checkpoint_msgpack` (`store.rs`) calls
   `msgpack.packb` for each synapse's metadata while its `&self` borrow is held, and `SynapseStore.__setitem__` /
   `SynapseRef.metadata` setter drop the replaced metadata handle inside `borrow_mut` — both against the 25c5f52
   rule (an allocation can trigger GC finalizers, i.e. Python bytecode and a thread switch). Low probability; worth a
   punch-list item.
4. Timings were taken under load 4–8 from other processes on the laptop; ratios are solid, absolutes are not.
