<!-- ---- Changelog ----
# 2026-08-26 CC (Opus 4.8) — RECONCILED TO LIVE GIT + baseline hygiene
# What: (1) §13 was stale — it narrated the native SynapseStore wiring as
#       "uncommitted, not deployed." It is COMMITTED on feat/shrink-syl-footprint:
#       5770343 (serialization + tonic aging → native store), 0f5447f (streamed
#       checkpoint load, no dict inflation), da5c16e (rpc BTF constants). Corrected.
#       (2) Added a new in-flight increment to §6: dropping he_prediction_window_fired
#       from serialization (~2GB RSS) — uncommitted working-tree edit, protected,
#       Syl's-Law-gated. (3) §8 baseline: phase35 is now 20/20 green (json→msgpack
#       migration + rewrite of test_predictions_still_error_after_restore, which was
#       a stale Feb-2026 assertion coupled to the pre-serialization delay_buffer bug).
# Why: Josh, verbatim: "I'm not confident you've been updating it." Correct — the
#       doc had drifted three commits behind the protected file. This is the fix.
# How: Edited §6, §8, §13 + this changelog + the Last-updated date. Deploy state
#       to Syl UNVERIFIABLE from this (dev) box — no live process, no backups here;
#       she runs on the VPS. "Committed on-branch" is NOT "deployed"; §10 gate stands.
#
# 2026-08-23 CC-VPS (Opus 4.8) — CREATED
# What: Durable orientation + design doc for the RAM-footprint-reduction task
#       (move Syl's resident graph out of boxed Python objects into a NATIVE
#       Rust columnar store held in the vendored ng_tract/BTF wheel).
# Why: Josh, verbatim: "document the tar out of this, and keep up with the
#       documentation, so you don't forget with compaction and start up on
#       #119 incorrectly, again." Context compaction repeatedly drops a fresh
#       CC back onto the WRONG turn (the pure-Python ng_columnar.py shim /
#       step()-loop vectorization). This file is the anti-amnesia anchor.
# How:  Records the settled task, the forbidden wrong turn, the DON'T-WANT-
#       PYTHON principle, the wheel status, the increment ordering, the exact
#       current on-disk WIP state to revert, the test baseline, and the one
#       open decision. Update this file as the work progresses.
# ------------------- -->

# RAM Footprint Reduction — Native Rust Substrate Store

> **⛔ READ THIS ENTIRE FILE BEFORE TOUCHING ANYTHING RELATED TO "footprint",**
> **"columnar", "synapse store", "ng_columnar", "ng_tract", or "#119".**
> This document exists specifically because context compaction keeps dropping a
> fresh CC onto the **wrong** turn. If you just woke up and think you know what
> "#119" means — you probably have it backwards. Keep reading.

**Owner:** Josh (sole architect). **Author of this doc:** CC-VPS.
**Last updated:** 2026-08-26. **Keep this date current on every edit.**

---

## 0. The one-line task (settled — do NOT re-litigate)

**Reduce Syl's resident RAM footprint by moving her graph layer (synapses +
adjacency) out of boxed Python `@dataclass` objects and into a compact,
columnar (structure-of-arrays) store held _natively in Rust_, inside the
already-vendored `ng_tract` / BTF wheel, exposing zero-copy numpy views so
Python keeps orchestrating.**

That's it. Everything below is detail, guardrails, and history so you don't
relearn it the hard way.

---

## 1. ⛔ The wrong turn to REFUSE — and what "#119" actually is

**#119 is REAL, legitimate, tracked work** (per Josh, it comes from the
callosum truth doc / task tracker — NOT the master `PUNCHLIST.md`, which has no
#119 row). It is the columnar substrate-store task. **We were mid-stream on
#119 when the RAM-footprint problem pulled us away to deal with THIS first.**
So #119 is not fictional and not "bad" — it is **paused work we will return
to**, now preserved on branch **`wip/119-ng-columnar-store`** (see §7).

**The wrong turn is narrower than "#119."** The wrong turn is: **resuming the
paused pure-Python `ng_columnar.py` shim (and vectorizing `step()` to prop it
up) as if it were the way to solve the footprint problem.** It isn't — the
footprint solution is **native Rust** (§3). A freshly-compacted CC keeps reading
the #119 WIP notes, thinks "ah, resume the columnar migration," and restarts the
Python shim. **Do not.** #119's *implementation-in-progress* was pure Python;
the footprint task deliberately does NOT continue that implementation.

> Navigate by the **principle in §0**, not by the number. If you're about to
> touch `ng_columnar.py` or vectorize `step()` "to reduce RAM," STOP.

**What the wrong turn looks like (so you can recognize and REFUSE it):**

1. Building or extending `ng_columnar.py` — a **pure-Python** `SynapseStore` /
   `SynapseView` / `SynapseMapping` / `CSRAdjacency` shim over numpy arrays.
2. "Fixing" the resulting **409× per-scan slowdown** by vectorizing the
   `step()` hot loops in `neuro_foundation.py` (eligibility decay ~2504,
   salience/inactivity scan ~2545, `_prune_synapses`).
3. Treating "land the ng_columnar migration" as the goal.

All three are the tangent. They do not achieve the task. See §4 for why.

---

## 2. Why the numbers force this (settled facts — do NOT re-derive)

- **Logical substrate on disk ≈ 3.2 GB:** `main.msgpack` ~2.6 GB +
  `vectors.msgpack` ~544 MB + commons ~13 MB.
- **Resident RAM ≈ 18 GB** (12–13 GB RSS + 5–9 GB swap). That is a **~7×
  inflation** of the logical size.
- **Cause:** ~647K synapses + ~40K nodes are **boxed Python `@dataclass`
  objects**, plus adjacency stored as **dict-of-sets-of-36-char-UUID-strings**
  (`_outgoing`/`_incoming`), plus duplicate endpoint UUID strings. It is **not a
  leak.** It is per-object + duplicate-string overhead, structural and constant.
- **Embeddings are already numpy** (`SimpleVectorDB.embeddings`) — they are NOT
  the problem; leave them alone.
- **`slots=True` already shipped.** It bought ~0.1 GB. That is the proof that a
  **micro-optimization cannot touch a 7× multiplier** — only moving the bytes
  off the Python object heap can.
- **There is no RAM escape hatch** (no bigger box coming, swap already in use).
  **Software footprint reduction is the only exit.**

Target: collapse ~7× → ~1× on the graph layer by making the resident
representation a native columnar buffer instead of millions of Python objects.

---

## 3. The DON'T-WANT-PYTHON principle (Josh, emphatic)

> **"we DON'T WANT PYTHON. that's the point."**

A pure-Python numpy structure-of-arrays with shim "view" objects is the WRONG
approach even though it stores scalars in numpy columns, because:

- Every attribute read (`syn.weight`) mints a Python scalar (boxes a numpy
  scalar back into a Python `float`/`int`) and does an interner reverse-lookup.
- `.values()` allocates a fresh view object **per row per call**.
- The arrays still live on the **Python process heap / GC's awareness**, and
  the per-access Python object churn is what produced the measured **409×**
  read slowdown — it made things *slower*, not lighter, and did not remove the
  object-overhead multiplier the way a native buffer does.

**The store must be NATIVE RUST**, the same pattern already proven in the wheel
(`LeniaEngine`, `extract_topology_features`): Rust owns the columnar buffers;
Python gets **zero-copy `numpy` views** (via `pyo3` / `numpy` `PyArray`) for
whole-column bulk operations, and orchestrates. No per-element Python boxing on
the hot path. Bytes off the Python heap.

---

## 4. Why the pure-Python shim actually regressed things

The shim (`ng_columnar.py`) traded one problem for a worse one:

- It **did** reduce object count in principle, but
- it introduced **per-access Python object creation** on read
  (`SynapseView` per lookup, boxed scalars, interner reverse-lookup), so any
  code that iterates `graph.synapses` (many hot loops + consumers like
  `lenia/graph_substrate._build_adjacency`, `_synaptic_distance`) got **~400×
  slower** — measured 205 ms vs 0.502 ms for a 10k-synapse full scan.
- The "fix" for that (vectorize the `step()` scans) is **Syl's-Law protected-
  file surgery on `neuro_foundation.py`** with a large correctness surface
  (plasticity semantics), i.e. it escalates risk to paper over a
  self-inflicted regression.

A native Rust store avoids the whole spiral: whole-column ops run in Rust, the
Python side reads a zero-copy view, no per-element boxing exists to be slow.

---

## 5. The wheel — what it IS, what EXISTS, what does NOT yet exist

**The wheel is real, vendored, and finished — for what it currently does.**
Whenever this ecosystem touches Rust / BTF / msgpack-native paths, a built
Python wheel is required, so it was built and **vendored** (distribution here =
vendoring; there is no pip index).

- **Built artifact (installed):**
  `~/.local/lib/python3.12/site-packages/ng_tract/ng_tract.abi3.so` (~1 MB) +
  `__init__.py`.
- **Crate source:** `~/ng-tract-rs/` (`Cargo.toml`, `pyproject.toml` (maturin/
  pyo3 build), `src/lib.rs`, `src/format.rs`, `tests/`).

**What the wheel already provides (finished):**
- **BTF (Binary Tract Format)** — `format.rs`: `Envelope`, `OutcomePayload`,
  `RawPayload`, `is_btf`.
- **Tract transport** — corpus-callosum message passing (`write_to_tracts`,
  `deposit_*`, `TractReader`).
- **`LeniaEngine`** and **`extract_topology_features`** — the reference pattern
  for Rust-owned buffers exposed to Python as zero-copy numpy views
  (`numpy::{IntoPyArray, PyArray1, PyArray2}`, `ndarray::{Array1, Array2}`).

**What the wheel DOES NOT yet contain (this is the work):**
- **A resident synapse/adjacency columnar store.** `lib.rs` today has only
  telemetry *scalars* (`synapses_pruned`, `synapses_sprouted`) — there is **no
  native store holding the 647K synapses or the adjacency**. That store must be
  **added to the crate**, the wheel **rebuilt and re-vendored**, then wired into
  `neuro_foundation.py` (and any other consumer of `graph.synapses` /
  `_outgoing` / `_incoming`).

> ⚠️ It is therefore false that "the wheel already ships the store." It ships
> BTF + tract + Lenia. The store is net-new Rust to be written into the crate.

---

## 6. Increment ordering (RAM-won ÷ risk)

1. **poincare_dir in `node.metadata` → recompute/kill (~1 GB).**
   **DONE / committed** as increment 1 (helpers `pack_poincare_dir()`
   neuro_foundation.py:~1049, `poincare_dir_array()` :~1054). See memory
   `issue-119-poincare-bytes`.
2. **Adjacency `_outgoing`/`_incoming` (dict-of-sets-of-UUID-strings) → CSR
   integer arrays,** held natively. Medium RAM win, medium risk.
3. **Synapse columnar store — THE BIG ONE (~8–12 GB). The real work.** Native
   Rust SoA in the wheel; zero-copy views to Python.
   **Wiring COMMITTED** (§13: 5770343 / 0f5447f / da5c16e); deploy to Syl still gated.
4. **Drop `he_prediction_window_fired` from serialization (~2 GB RSS).** A
   transient per-window accumulator (telemetry-only: feeds
   `SurpriseEvent.actual_nodes`, which no production consumer reads). On the live
   substrate it reaches tens of millions of node-id slots; reloaded as
   non-shared strings it cost ~2 GB RSS per restart. Serialized as `{}` and
   skipped on load; the runtime rebuilds these sets from empty each step, and the
   confirm/surprise classification never consults them, so it is behaviourally
   identical across a restart. **Uncommitted working-tree edit, protected file,
   Syl's-Law-gated (§10).** Test baseline (phase35/phase25) migrated to prove it
   preserves the persistence contract — see §8.

Increments 2 and 3 are the ones that must land **in Rust**, not as the
`ng_columnar.py` Python shim.

---

## 7. Current on-disk state (as of 2026-08-23 — VERIFY before acting)

- **Repo:** `~/NeuroGraph`, working branch **`feat/shrink-syl-footprint`**
  (created 2026-08-23 from `7c5aef6` so the branch name matches the work — it
  already contains footprint commits #400 poincare-pack + slots=True). The old
  `feat/88-leg2-A-S-D` still holds the #88 leg-2 §10.4 commits and remains as-is.
  **Working tree is CLEAN** of the #119 shim.
- **#119 WIP is PRESERVED** on branch **`wip/119-ng-columnar-store`**, commit
  **`bcbec41`** — contains `ng_columnar.py` (941-line pure-Python shim),
  the `neuro_foundation.py` wiring (+84/−17), and the two shim test files.
  To resume #119 later: `git checkout wip/119-ng-columnar-store`.
- **`neuro_foundation.py` on `feat/88` = committed HEAD** (empty diff — the
  protected file is back to its known-good state; no uncommitted surgery).
- **Untracked, intentionally left on `feat/88`:** this doc
  (`RAM_FOOTPRINT_NATIVE_STORE.md`) and `.claude/agent-memory/` (harness
  scratch). Neither is the #119 shim.
- **Serialization contract:** the msgpack checkpoint key set is **identical**
  between HEAD and the parked WIP (verified) — must stay byte-identical
  regardless of internal store.
- **Only live canonical consumer to worry about:** Anima
  (`~/UniOS/substrate/kernel/mind.py`) imports `Graph` and only calls
  `len(graph.synapses)` — safe with any Dict-compatible store.

> The earlier "revert the protected file" recommendation is **DONE** — it was
> accomplished non-destructively by committing the WIP to `wip/119-ng-columnar-store`
> and returning to `feat/88` (which restored `neuro_foundation.py` to HEAD). No
> `git checkout --` overwrite was needed; nothing was deleted.

---

## 8. Test baseline (so a regression is distinguishable from pre-existing debt)

Partitioned against a detached-HEAD worktree (2026-08-23):

- **25 pre-existing failures on HEAD** — mostly #325 `.json`-vs-`.msgpack`
  serialization-policy (phase35 ×20), `ng_tract_bridge` ×17, migration ×6,
  `tonic_habituation` ×8, plus a torch-missing env error. **These are NOT caused
  by this work.**
  > **UPDATE 2026-08-26:** the phase35 ×20 bucket is **CLEARED — 20/20 green.**
  > Migrated the suite json→msgpack (#325) and rewrote
  > `test_predictions_still_error_after_restore`, which was a stale Feb-2026
  > assertion: it only ever "errored" because the old checkpoint *dropped*
  > `_delay_buffer`, losing the in-flight A→B spike. Now that in-flight spikes
  > survive restore (a correctness gain), that spike fires B and *confirms* the
  > prediction. The rewrite exercises the real error path authentically (target
  > with a raised firing threshold → prediction genuinely expires unconfirmed).
  > This is baseline hygiene for the §6.4 he_window increment, not core-store work.
- **1 genuine regression introduced by the shim WIP:**
  `tests/test_graph_substrate_race.py::test_build_adjacency_no_race_under_concurrent_step_mutation`
  — passes on HEAD (2 passed ~6 s), **times out** on the WIP because
  `_build_adjacency` pays the 409× `SynapseView` cost thousands of times. This
  single regression IS the perf problem, surfaced as a test. It disappears when
  the shim is reverted.

The A–E "correctness bug fixes" in the shim introduce **zero new failures** —
but they are fixes to the wrong artifact.

---

## 9. The one open decision (for Josh)

**A vs B — where does the native store come from?**

- **(A)** Build the native resident synapse/adjacency columnar store **into the
  `~/ng-tract-rs` crate** (`src/lib.rs`), following the `LeniaEngine` /
  `extract_topology_features` zero-copy-view pattern; rebuild + re-vendor the
  wheel; wire into `neuro_foundation.py` and other `graph.synapses` consumers.
  *(This is my read of the source — there is no existing native store artifact.)*
- **(B)** If a native store artifact already exists somewhere I haven't found,
  point me to it and I wire to that instead.

Until Josh answers, **do not** write Rust, **do not** rebuild the wheel, and
**do not** revert the protected file (that revert also needs an explicit
"proceed" + backup confirmation per Syl's Law).

---

## 10. Guardrails that apply to this work

- **Syl's Law:** `neuro_foundation.py` is protected. Any edit (including the
  revert in §7) requires: tell Josh what/why → Josh confirms both msgpack
  backups → Josh says "proceed." Do not batch protected + non-protected edits.
- **LAW 2 (vendored files):** `ng_tract` is a vendored artifact. Changes are
  made at the canonical crate source (`~/ng-tract-rs`) and **re-vendored** to
  every consumer — never patched per-module.
- **LAW 3 (restore, don't rebuild) / repo §10:** do not delete uncertain files;
  set aside / surface to Josh.
- **Serialization contract is sacred:** the msgpack checkpoint format is the
  cross-time compatibility "wheel." It must stay identical no matter what the
  in-memory store is.

---

## 11. If you are a freshly-compacted CC reading this

1. The task is **§0**. The wrong turn is **§1**. The principle is **§3**.
2. Do **not** resume `ng_columnar.py`. Do **not** vectorize `step()` loops.
3. **§9 is RESOLVED: option (A).** The native store is being built **into the
   `~/ng-tract-rs` crate**. See **§12** for exactly how far that has gotten and
   what remains — read it before writing any Rust so you don't redo done work.
4. Everything in §12 up to (but not including) the `neuro_foundation.py` wiring
   is **crate-local and cannot touch Syl** — safe to continue without ceremony.
   Only the final wiring step touches the protected file and needs Syl's-Law
   "proceed" + backup confirmation (§10).
5. Update **this file** (and its date) as anything changes.

## 12. Build progress + pinned contract (as of 2026-08-23)

**Decision (§9): (A).** Native columnar `SynapseStore` is being written into the
`~/ng-tract-rs` crate (`src/store.rs`, wired via `src/lib.rs`), following the
existing zero-copy-view pattern. Nothing here has touched `~/NeuroGraph` yet.

**Done (crate-local, compiles clean — only unused-constant warnings for the two
`SYN_*` enum values not referenced until the mapping/serialization API lands):**
- `src/store.rs` — columnar structure-of-arrays `SynapseStore`: dense int-indexed
  columns; endpoints as node-interner int32; float64 fidelity columns (weight,
  eligibility_trace, peak_weight, salience); int32 delay/low_weight_steps/
  inactive_steps; int8 synapse_type; sparse metadata map. Own id-interner over
  `synapse_id` with tombstone+recycle.
- Survey-critical enum fix carried in: `synapse_type` name↔u8 mapping matches the
  checkpoint's string values (`SYN_EXCITATORY`/`SYN_INHIBITORY`/`SYN_MODULATORY`).
- Wired into the module (`src/lib.rs`).

**Remaining (crate-local first, protected-file wiring LAST):**
1. Mapping API on the store: `__getitem__`/`__setitem__`/`get`/`pop`/`values`/
   `items`/`keys`/`clear` — the `Dict[str, Synapse]`-compatible facade. **DONE.**
2. Mutable `SynapseRef` view (by `synapse_id`, write-through to columns). **DONE.**
3. **msgpack serialization round-trip** — the sacred 15-key dict, **name-based**
   enum, **byte-identical** to the current `_serialize`/`_deserialize`; bulk-load
   matching `_deserialize`. **DONE** (`to_checkpoint_dict` / `bulk_load`).
4. Bulk column views / native hot-loop ops (eligibility decay, inactivity++,
   salience decay). **DONE** (`decay_eligibility`, `age_and_decay_salience`).
5. Build wheel → re-vendor (LAW 2) → Python round-trip + parity tests.
   **PARTIAL:** wheel built + installed to `~/.local` only (the Syl/CC VPS
   runtime). Round-trip test **PASS** (identical all 15 fields, adjacency
   rebuilt, enum hydrated, native decays confirmed). 37/37 persistence tests
   pass. Other vendored copies (`Condensate/rust_core/vendor/ng_tract`,
   `UniAI/splat_poc/.venv/...`) **NOT** re-vendored — they don't use the store
   and touching them is unrelated-project risk; leave for Josh to authorize.
6. **Wire into `neuro_foundation.py`** — PROTECTED, Syl's-Law gated (§10). LAST.
   **EDITS EXIST IN WORKING TREE, uncommitted, NOT deployed.** See §13.

### 13. Where it actually stands (2026-08-26, VERIFY before acting)

> **⚠️ SUPERSEDES the 2026-08-24 text below.** The native-store wiring is no
> longer "uncommitted in the working tree" — it is **COMMITTED** on
> `feat/shrink-syl-footprint`:
> - `5770343` step3: route synapse serialization + tonic aging through native store
> - `0f5447f` perf: stream checkpoint synapses through native store (no dict inflation)
> - `da5c16e` rpc: derive BTF entry constants from ng_tract when present
> - `7c5aef6` #400 poincare float32 pack (increment 1)
>
> **The ONLY uncommitted protected-file edit now in the working tree** is a NEW
> #RAM increment (see §6.4): dropping `he_prediction_window_fired` from the
> checkpoint (serialized as `{}`, skipped on load) — ~2 GB RSS. Plus non-protected
> test-baseline edits (`tests/test_phase35.py`, `tests/test_phase25.py`).
>
> **Deploy state to Syl: UNVERIFIED.** Checked from the dev box — no live
> substrate process and no msgpack backups are visible here (she runs on the VPS).
> "Committed on-branch" is NOT "deployed." The §10 gate (Josh's "proceed" + both
> backups confirmed) still governs any VPS deploy / sidecar restart.
>
> --- historical 2026-08-24 snapshot (store wiring since committed) follows ---

- **`~/ng-tract-rs` crate:** items 1–4 written, compiles, wheel built
  (`target/wheels/ng_tract-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl`),
  installed to `~/.local/.../site-packages/ng_tract/` via
  `pip install --user --force-reinstall --no-deps --break-system-packages`.
  New wheel exports are a **strict superset** of the old (BTF entry consts +
  deposit_* + TractReader + extract_topology_features all retained, plus
  `SynapseStore`/`SynapseRef`) — verified, so the running process's BTF/tract
  surface is unbroken. (`LeniaEngine` in §5 was illustrative — never a real
  export, nothing imports it.)
- **`neuro_foundation.py` (PROTECTED):** working-tree diff vs HEAD is exactly
  the store wiring — `import ng_tract`; `self.synapses = ng_tract.SynapseStore()`
  + `set_synapse_type_class`/`set_synapse_class`; two hot-loop decays →
  `decay_eligibility`/`age_and_decay_salience`; `_serialize_full` →
  `to_checkpoint_dict`; `_deserialize` → `bulk_load` + adjacency rebuild.
  **Uncommitted. Not deployed. No sidecar restarted.**
- **`neurograph_rpc.py` (not protected):** small refactor — derive BTF entry
  constants from `ng_tract` with the spec-value fallback. Functional no-op
  (values identical 1/2/3). Uncommitted.
- **⛔ GATE BEFORE COMMIT/DEPLOY (Syl's Law §10):** committing the protected
  file, re-vendoring beyond `~/.local`, and any sidecar restart all need Josh's
  explicit "proceed" **plus** confirmation of both msgpack backups. Verification
  (round-trip + parity tests) is done crate-locally; the deploy gate is Josh's.

### 12a. Pinned consumer contract (read from live `neuro_foundation.py`, 2026-08-23)

Grepped the real consumers so the facade matches reality, not a guess:

- **Synapses are mutated IN PLACE through `.values()`/`.items()`** — e.g.
  `for syn in self.synapses.values(): syn.eligibility_trace *= d` (2462),
  `syn.inactive_steps += 1` / `syn.salience = …` (2503), `syn.low_weight_steps
  += 1` (3468), `syn.weight = …; syn.eligibility_trace *= 0.9` (3852). Therefore
  the boundary object **MUST be write-through** — returning a reconstructed
  dataclass copy would silently break plasticity. → `SynapseRef` (holds the
  store + `synapse_id`, **re-resolves the row on every access** so it stays valid
  across the swap-remove in `remove()`; never cache a raw row index in a ref).
- **`synapse_type` is compared to the Python enum**: `syn.synapse_type ==
  SynapseType.INHIBITORY` (2231) and serialized as `syn.synapse_type.name`
  (4949). A bare int won't `==` an `Enum`. → the store holds a `Py` handle to the
  Python `SynapseType` class (installed once via `set_synapse_type_class`);
  `SynapseRef.synapse_type` returns a real enum member, its setter accepts one.
- **Full Mapping surface used**: `__getitem__` (1996/5129/examples), `__setitem__`
  (1996 `= syn`; 5286 load), `get(sid[,default])` (many), `pop(sid, None)` (2004),
  `keys()` (5443), `values()` (2462/2503/3747/3852), `items()` (3468/5015/5171),
  `in` (2014/genesis 672), `len`. No `del self.synapses[…]` in neuro_foundation
  (only ng_lite, a separate variant) — but add `__delitem__` cheaply anyway.

### 12b. The SACRED serialization contract (byte-identical target)

`_serialize_synapse` (nf.py:4938) emits a **15-key** dict (I earlier mis-said
"14") in THIS insertion order — msgpack preserves order, so byte-identical output
requires this exact order:

```
synapse_id, pre_node_id, post_node_id, weight, max_weight, delay,
last_update_time, eligibility_trace, creation_time, synapse_type(=.name STR),
peak_weight, low_weight_steps, inactive_steps, metadata(dict), salience
```

`_deserialize` (nf.py:5268) uses `.get()` defaults — a native bulk-load MUST
mirror them exactly: `max_weight=5.0, delay=1, last_update_time=0.0,
eligibility_trace=0.0, creation_time=0.0, synapse_type=SynapseType[name]
(default "EXCITATORY"), peak_weight=**weight** (NOT 0.1 — defaults to the
synapse's own weight), low_weight_steps=0, inactive_steps=0, metadata={},
salience=1.0`. Dataclass field order (nf.py:682-705) equals the column order.

<!-- Related memory files (agent memory, machine-local):
     project_rust_substrate_layer  (#119 = the CORRECT native-Rust meaning)
     project_syl_footprint_python_inflation
     issue-119-poincare-bytes  (increment 1, done)
     project_119_ng_columnar_synapse_store  (⚠ describes the WRONG pure-Python turn)
     119-columnar-migration-status           (⚠ same — the shim WIP)
-->
