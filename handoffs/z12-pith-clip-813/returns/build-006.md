<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 6 return build-006
# What: le-025 C-2 (FIX, failing-first), C-5 (FIX, failing-first), C-1, C-3, C-4; C-6 nothing; shared-path
#   fail-soft recorded for row #829.
# Why: dispatch #11238; Chief ruling docs e19962de on le-025 (PASS-WITH-NOTES, turn 5 only).
# How: NG repo only. Every number below is from this session; the "before" values are from runs made on the
#   turn-5 head BEFORE the fix existed.
# -------------------
-->

# #813 TURN 6 — RETURN build-006 (small)

Lane `pith-clip-removal-813` · dispatch #11238 · worker seat · returned **unreviewed**.
Related: [[NeuroGraph]] · [[Pith]] · [[Duck Ethics]] · previous `build-005.md` · review `../reviews/le-025-813-turn5.md` (read in full).

**Nothing merged, wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a`, unchanged). Docs repo, `cc_ng_host.py`, `surfacing.py`, `surface_resolver.py` and every protected/vendored file not touched. Condensate not touched.**

> **CORRECTION (dated 2026-09-30) to `build-005.md` and to the turn-5 code comment:** they said the measured provider node limit is "deliberately conservative" and that a node between it and the true budget "is still caught, loudly, as a never-fit assembly at admit". **The opposite happened.** The limit measured the *worst-case* shell, so it fell below what fits in a tight budget and the node was dropped or referenced instead (§2). `build-005.md` is annotated at its top; its history is otherwise left as written.

## 1. The items

| Item | Result |
|---|---|
| **C-2 (MEDIUM) — FIX, failing-first** | **Done.** `_pith_provider_node_limit` (`cc_ng_organism.py:4644`, used at `:5600`) is now the **optimistic bound**: it renders the **smallest** envelope the renderer can wrap around one node — ONE ordinary connected line (an alert-free coherence, no correction, no sources line, no anchors, no relations) inside the "Learned Situation" section, empty text — and returns `budget − len(core) − that overhead`, floored at 1. It can therefore never be smaller than what fits. **Approach chosen (the brief let me choose): optimistic bound + the existing loud NEVER-FIT backstop at admit, not a two-pass.** Why: a two-pass needs the real candidate shell, which exists only after the basins are built, so I would either build the basins twice (the first pass would run anchor regexes over giant raw texts it is about to replace with references) or restructure `pith_connected_activation_basins`; the backstop already exists, is whole-or-absent and is loud by id. The two errors are not symmetric: a limit too large costs a loud drop of a node that could not fit anyway; a limit too small (turn 5) turned a node that did fit into a reference or nothing. **Price, stated:** a node at or below the limit whose real assembly needs more overhead (alerts, sources, anchors, relations) is now dropped whole and loudly as a never-fit assembly *without* its trees, where turn 5 would have shown a reference. See §2 for the proof. |
| **C-1 (MED-LOW) — test** | **Done.** `test_c1_the_pin_probe_is_the_base_closure_except_the_deliberate_c5_change` compares the `_pinned` inside `_cc_pin_probe` with the closure in `git show e4ebf982:cc_ng_organism.py` by `ast`: parameter and the guarded call `bool(ng.graph._is_identity_protected(node_id))` are **`ast.dump`-identical**; the `except` handler is the one asserted difference (base `return False` at DEBUG, head returns via `_cc_pin_guard_failed` → `True`). `test_c1_executed_on_the_le025_cases_identical_except_when_the_guard_raises` **executes both** on the guard cases (true, false, truthy str, None, `[]`, `0`, raising `RuntimeError`, raising `KeyError`, no `.graph`, `graph=None`): identical to base everywhere except the raising/missing-guard cases, where base `False` → head `True`, stated in the test. |
| **C-3 (LOW) — truth** | **Done.** The docstring (`:4644`) now says what the code does (optimistic bound; the between-band is caught at admit as a loud never-fit drop; the asymmetry). The INFO line (`_pith_log_reference`, `:4721`) says **"N node(s) above the reference limit L chars (C chars) surfaced through their trees + a whole-node reference"** instead of "over-budget" — on the provider path L is the measured per-node limit, on the recall paths it is the L1 budget. Dated CORRECTION lines: above, and at the top of `build-005.md`. The contract's #819 paragraph now states the threshold rule. |
| **C-4 (LOW)** | **Done, with one honest deviation.** The alert vocabulary is defined **once** — `_PITH_COHERENCE_STATES` and `_PITH_ALERT_COHERENCE = ("conflict","stale","uncertain","unknown")` (`:5476-5477`) — and drives **both** `_pith_provider_sections` (`:5489`) and `_pith_provider_node_limit` (which builds its *ordinary* line from the states not in the alert set). A test flips the constant: the renderer emits `shared_material` **and** the measured limit shrinks; and asserts the literal tuple appears once in the source. **Deviation:** the brief also asked to share the `"failure -> correction"` entry. After C-2 the limit no longer probes a correction line at all (the optimistic bound excludes it), so there is no second copy left to share; I did not invent one. |
| **C-5 — FIX (Chief ruling), failing-first** | **Done.** `_cc_pin_probe` (`:5739`) **fails closed**: when `ng.graph._is_identity_protected` raises, or the guard does not exist (no `.graph`, no method), `_pinned` returns **True** via `_cc_pin_guard_failed` (`:5724`), which logs a **WARNING** naming the node id and the exception **type** only (no exception message, no node text), **first-seen per id** (the same bounded tracker as the drop lines — no flood). It applies to **both** users of the probe: Stage 3 (Pith-ON, `cc_assemble_recall`) and the un-Pithed renderer. Identity fails toward keeping content (#92 "nothing protected dies"; Duck Ethics). **Found by its own test:** my first version passed a bare `str` to `_pith_note_ids`, which iterates it character by character, so "first-seen" tracked letters and warned twice — fixed by passing a list. |
| **C-6** | Nothing to do (the #812 row is written). |

## 2. Failing-first evidence

**Tests written first** (in `tests/test_cc_pith_clip_813.py`) and run on the **turn-5 head `3a053e1`** (whose code is identical to `2b098e9`) before any fix existed: **17 failed, 12 passed**. The failures were: **all 7** tight-regime cases, the sweep, the 4 C-5 tests, the 2 C-1 tests, C-4, and 2 C-3 tests.

**C-2 — le-025's probe reproduced on head, then after** (`/tmp/le025_probe_c.py`, one node, fake graph; `len` = context length):
```
                                BEFORE (3a053e1)                          AFTER (77a18ce)
core=800 budget=1200 node=30    state=empty capacity_empty  whole=False    state=ok whole=True  len=1130
core=800 budget=1200 node=60    (dropped, capacity_empty)                   state=ok whole=True  len=1160
core=800 budget=1500 node=60    state=ok whole=False ref=True  len=1217     state=ok whole=True  len=1160
core=400 budget=1000 node=30    state=ok whole=False ref=True  len=817      state=ok whole=True  len=730
```
The after-lengths (1130 / 1160) are exactly what the turn-5 parent produced. Sweep on head, first failing grid point: `(core 0, budget 500, node 30) old-whole but head is not`.

**The invariant, proven on a grid:** `test_c2_sweep_old_whole_implies_new_whole_and_nothing_is_ever_cut` renders one node on **BASE `e4ebf982`**, on the **turn-5 parent `b2f3d18`** (loaded from `git show`; a missing commit FAILS the test) and on head, over core ∈ {0, 200, 400, 800} × budget ∈ {500, 1000, 1200, 1500, 4000} × node ∈ {30, 60, 120, 600}: **80 grid points, 57 rendered the node whole under BASE or the parent, and 0 of those 57 are not whole at head.** It also asserts, at every point, no `⋯`/` …` marker, no partial node text, and `len(context) ≤ budget`. The 7 named tight cases are parametrized separately.

**C-5 — before/after:** a raising guard (message `SECRET-GUARD-TEXT-DO-NOT-LOG…`) on head `3a053e1` returned `False` (unpinned, DEBUG only) so the identity item was budget-droppable; after: `True`, one WARNING `Pith identity-pin guard failed for node ident-1 (RuntimeError); treated as PINNED (fail closed …)`, the secret text appears nowhere in the log, a second call for the same id is silent, a missing guard also pins, and on **both** the Pith-ON and the gate-off path all three items of the N1 world survive when the guard is down.

## 3. Commits, diff, proofs

| | |
|---|---|
| base of this turn | `3a053e123f75842cd7e6c6ae36f154c915533407` (turn-5 return `2b098e9` + le-025's review commits only; `git diff 2b098e9 3a053e1 --stat` is that one review file) |
| **code commit** (the commit before this return file, **pushed before the definitive test run**) | **`77a18ce1c4bc83e5689c38839fa385662ce99b8e`** |
| **return commit** = final NG branch head | the commit that adds **this file plus the build-005 correction note** (parent `77a18ce`); a file cannot name its own commit — the exact hash is in the reply and equals `git rev-parse HEAD` / `git ls-remote origin refs/heads/cc-laptop-pith-clip-813-20260930` |
| docs branch head (untouched) | `fd0ccc9c08b965e6b9c68baebe6f78a09ed36931` |

`git diff 3a053e1 HEAD --stat` at the code commit: `cc_ng_organism.py | 114 ++++++-----` · `docs/PITH_HOST_CONTRACT.md | 20 ++-` · `tests/pith_clip_813_scenarios.py | 6 +-` · `tests/test_cc_pith_clip_813.py | 262 +++++++++++++++++++++++-` → **4 files, 355 insertions, 47 deletions** (the return and the build-005 note add on top). `tests/pith_clip_813_scenarios.py`: `FakeGraph._is_identity_protected` now mirrors the real guard (unknown node → `False`, never raises; `neuro_foundation.py:3551-3572`), because a fake that raised `KeyError` would have pinned every unknown node under C-5.

* **`git diff e4ebf982 HEAD --stat -- cc_ng_host.py` → empty** (byte-identical to base).
* Protected / vendored / shared (`surfacing, surface_resolver, cc_ng_host, neurograph_rpc, kiss_filter, tonic_thread, neuro_foundation, openclaw_hook, stream_parser, activation_persistence, ng_*`) touched vs base: **none**. I only *read* `neuro_foundation._is_identity_protected` to confirm it never raises on an unknown node.
* Golden vs BASE `e4ebf982` unchanged and passing; P379 preamble passes (`env -u NG_EMBED_REMOTE -u PYTHONPATH`).

## 4. Tests — once, on the pushed commit
**(A)** the turn-5 union (`test_cc_pith_clip_813, test_pith_provider_context, test_pith_stage1…5, test_pith_l1_provenance, test_pith_metrics_concurrency, test_cc_host_pith_telemetry, test_cc_recall_dedup, test_cc_recall_unification, test_cc_region_confidence, test_pith_history_metrics, test_cc_host_compress_history`; worktree code only, no daemon env) at `77a18ce` → **302 passed, 3 skipped** (282 at turn 5 + 20 new; the 3 skips are the host/daemon parity tests). **(B)** parity tests **alone** with `CC_DAEMON_UNDER_TEST` → the docs worktree daemon → **3 passed**.
**Honest note on "push before each test run":** the definitive run above was made on the pushed commit. My *iterative* runs of the new file while developing (the RED run on the turn-5 head, and the runs while fixing) were made on uncommitted work, as in earlier turns.
**Not verified:** anything live (no service, socket, sidecar, checkpoint, tract, real graph); the **real** constitutional-core size and `cc_l1_budget` breathing range — so how often the tight regime occurs in production is still unmeasured (the fix makes it irrelevant for correctness, not for incidence); the real `Graph._is_identity_protected` (tests use a fake that mirrors its contract).

## 5. RECORDED: fail-soft on the identity guard in shared / Syl-adjacent code (Chief row #829) — NOT edited
Searched for every use of `_is_identity_protected` outside `cc_ng_organism.py`:
* `neuro_foundation.py` `:3292-3293`, `:3517-3518`, `:3602`, `:3671`, `:3676`, `:3683` — the engine calls the guard **directly, with no `try/except`**: an exception would propagate, so there is no soft-fail there (protected file; read only).
* **`cc_topology_export.py:233-255`** — `_is_identity_protected(graph, node_id, node)` does `getattr(graph, "_is_identity_protected", None)`, and on an exception `except Exception: pass`, then **falls back to evaluating the same metadata fields** (`constitutional`, `provenance` ending `_authored`). It fails soft to *the field check*, not to "unprotected" — a much weaker hazard than the organism's old closure — and its own docstring says it is **deliberately no longer called** on either export path since #147 (retained as a mirror).
* **`surfacing.py:194`** — `except Exception:  # noqa: BLE001 - fail-safe: old vdb path for this node` around `resolve_surface_item(node, db_entry)`: not the pin guard, but a fail-soft on the **monitor's content resolve** that keeps/falls back to the vdb content (this is the shared side of the T1/#812 fork; my CC-side `_cc_monitor_items_whole` deliberately does the opposite — drop — per turn 4 C2). `surface_resolver.py`: no swallow.
* No other soft-fail on the identity guard found in the repo. **Shared code is Josh's; nothing edited.**

## 6. Open notes carried forward (unchanged)
C4 pins off-budget (Exec) · #817 deferred (checklist in `plan-001-audit.md` §11) · T1/#812 (source fix deletes the fork; row written) · D3 wants · D4 deposit-path clips (#741) · D7 `MAX_QUEST_CHARS` until card-7 (`88ddfec`) and both hosts move together · #821 · N5 (`on_monitor_error` / an `ImportError` of `surface_resolver` drops the whole monitor stream, loudly) · a side effect of C-5 worth knowing: an `ng` with **no graph at all** now pins every item (off-budget, whole) and warns once per id, where the old closure treated them as ordinary — consistent with "identity fails toward keeping content", and not reachable in production where `ng.graph` exists.
