<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, dispatch #14794) — RETURN build-001 (docs-only)
#   What: the return for BUILD SWEEP-PROBATION (covers BOTH repos): the three commits, the one-function proof for the protected diff,
#   facts (i)-(iii), the FOREVER-LEAK ENUMERATION TABLE, tests (1)-(14), runs/seed/mutant tables, what ran and did not, not verified,
#   follow-ups. Josh's ruling (Exec P550 / P552, amended P554, placement P556); CC-CALLOSUM-TRUTH §8.13 (read §0, §0.2, §7, §8.12, §8.13).
#   Why: the pair counts runs on the committed hashes from SEPARATE detached worktrees; the Chief/Exec rule on the enumeration before any merge.
#   How: written from the committed trees and the recorded run outputs; nothing here is a claim the transcript does not back.
# -------------------
-->

# build-001 — SWEEP-PROBATION (P552): an unbound node is NOT swept while its probation window is open AND advancing

**Nothing pushed, nothing merged, no VPS, no live daemon/graph/checkpoint/socket touched.** NG merge-hold stands. The ceremony backup under
`~/backups/exec-p550-sweep-ceremony-*` was never opened. The forbidden tests (`tests/test_cc_deposit_step.py`, the NG full suite) were never run.

## 1. The three commits (and how to read each)

| # | repo / branch | hash | scope (source) | read it with |
|---|---|---|---|---|
| NG-1 | NeuroGraph `cc-laptop-sweep-probation-p552-20261002` | `c98d06283dc73948c90e851cb10c274d1e143c72` | `cc_ng_organism.py` ONLY (+ its tests) | `git -C /home/josh/NeuroGraph show --stat c98d062` |
| NG-2 | same branch, on NG-1 | `0b26f61426f161d3f1057a9931d4c0a240947fe0` | `neuro_foundation.py` ONLY, ONE function (+ its tests) | `git -C /home/josh/NeuroGraph diff b5e476863cc069a29ec482959b4f9465f2ea4ccf 0b26f61 -- neuro_foundation.py` |
| daemon | docs `cc-laptop-daemon-probation-warn-20261002` (base D24 tip `228100795`) | `905618ee29d479e07e9b86b18999c87d787b12c9` | `scripts/cc-ng-daemon.py` (+ its tests) | `git -C /home/josh/docs show --stat 905618ee` |
| return | NeuroGraph branch, on NG-2 | this file's own commit (docs-only; see `git log -1` below) | `handoffs/z12-sweep-probation-p552/returns/build-001.md` | `git -C /home/josh/NeuroGraph show --stat <tip>` |

Bases: NG `origin/main` `b5e476863cc069a29ec482959b4f9465f2ea4ccf` (confirmed unchanged: local ref and `git ls-remote` agreed at start); docs `228100795a63a35a1c01196720d2ae9778a481af`.
The NG-1/NG-2 order is as ruled (the protected diff stands alone). I read each commit with `git show --stat` and, for NG-2, `git diff -U0` against the base.

### 1.1 PROOF: the `neuro_foundation.py` diff is ONE function (NG-2)

```
$ git diff --stat b5e4768 0b26f61 -- neuro_foundation.py
 neuro_foundation.py | 43 +++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 43 insertions(+)
$ git diff -U0 b5e4768 0b26f61 -- neuro_foundation.py | grep '^@@'
@@ -3593,0 +3594,18 @@ class Graph:      # the docstring note, inside _collect_orphan_nodes (def at :3574)
@@ -3603,0 +3622,25 @@ class Graph:      # the getattr read + probation filter, inside the same body (base body :3595-3611)
```
43 insertions, **0 deletions**, **two hunks, both inside `Graph._collect_orphan_nodes`** (base `:3574-3611`). No module-level addition, no import, no `cc_`-named reference
(`tests/test_cc_sweep_probation_p552.py::test_static_the_sweep_reads_the_registered_predicate_via_getattr_and_names_no_cc_module` pins it by `ast`). `_unbound_nodes`, `held_unbound_nodes`,
`whole_graph_guard`, `step()`, `inject_reward`, `Graph.save/load`, `_prune_synapses`, `_is_identity_protected` are untouched. The other protected files and the six vendored files are untouched.

**Graph has no `__slots__`/`__setattr__`/`__getattr__`/property that blocks `graph._probation_advances = ...`** (`test_graph_accepts_a_plain_attribute_and_has_none_by_default`).

## 2. What changed (file:line, on the committed trees)

* **NG-1 `cc_ng_organism.py`** — `probation_advances(node)` at `:2077-2092` (`return (node.metadata or {}).get("creation_mode") != "ingested"`), the ONE definition. `cc_update_probation` (`:2095`)
  skip at `:2132` is now `if not probation_advances(node): continue` (was the inline `== "ingested"` at base `:2089`). Byte-identical (test 14). The writer `:1873` (base `:1851`) is unchanged.
* **NG-2 `neuro_foundation.py`** — `_collect_orphan_nodes` (`:3574`): the docstring carries the §8.13 sentence and a dated note (see 8.2: the changelog entry lives in the docstring, not in a header hunk);
  `pred = getattr(self, "_probation_advances", None)` at `:3622`; the structural comprehension is **byte-for-byte unchanged** and still builds `orphans`; the probation term is a loop over those
  candidates only (`:3627`...): spared iff `pred(node)` is truthy AND `probation_remaining` is a finite number `> 0` (`isinstance(prob, (int, float)) and not isinstance(prob, bool) and 0 < prob < float("inf")`).
  `pred` raises => that node is NOT spared; ONE WARNING per sweep (`:3642`) with a count and exception CLASS names only. `:3648-3654` (`removed` loop, `_emit`, `return removed`) is unchanged.
  The attribute absent (`None`) => the loop is skipped: exactly today's sweep.
* **daemon `scripts/cc-ng-daemon.py`** (base D24 tip lines in brackets): `DaemonState.stats` gains `probation_cycles_skipped: 0` (`:1007`); `init_ng` (`:1011`) registers
  `graph._probation_advances = cc_ng_organism.probation_advances` (`:1147-1149`, own `try`, AFTER the organism-layer bootstrap and OUTSIDE its `except`; ONE INFO naming `module.qualname`; any failure =>
  ONE WARNING, class name only); `_note_probation_skip` (`:2409`); `_autosave_loop` (`:2637`): `_probation_ran = False` before the inner `try` (`:2667`), `= True` right after
  `cc_update_probation` returns (`:2712`), the `except` (`:2716-2722`) calls `_note_probation_skip(exc)` when it was skipped, and keeps the old DEBUG line otherwise.
  **The block on the D24 base:** the inner `try` at `:2601`, `cc_update_probation(STATE.ng.graph)` at `:2644`, the DEBUG `except` at `:2647-2648`.
  **Graph-creation sites in the daemon: exactly ONE** (`init_ng`, `:989/:993` on the base; `NeuroGraphMemory.get_instance`, a singleton). No `reset_instance`, `Graph(`, `.restore(`, or `.graph =` anywhere in the daemon.
  In the real NG (pinned rev) the only `self.graph =` is `NeuroGraphMemory.__init__` (`openclaw_hook.py:847` on the base), `get_instance` constructs only when `_instance is None`, and `Graph.restore` (`neuro_foundation.py:5082` here)
  delegates to `_deserialize` (`:5411`) which `clear()`s the graph's OWN containers in place and never rebinds `self`. So there is no "graph replaced after init" path; the proof test is `test_the_daemon_has_exactly_one_graph_creation_site_and_rebinds_no_graph` + `test_the_real_ng_creates_its_graph_once_and_restore_works_in_place`.

## 3. Facts (i)-(iii), cited at the base `origin/main` `b5e47686` (daemon cites at the D24 tip `228100795`)

**(i) `cc_ng_organism.cc_update_probation` (`:2055-2134`)**: per call, for every node that HAS a non-None `probation_remaining`: `prob <= 0` => late-graduation branch only (no decrement); else `prob -= 1` and the dampening fade / graduation. It skips `creation_mode == "ingested"` (`:2089`; the Ingestor's own sweep owns those, `universal_ingestor.py:2261-2316`, read `:2298`, decrement `:2303`). No other skip exists. Its drivers: the laptop daemon's `_autosave_loop` (D24 `:2644`, `AUTOSAVE_INTERVAL = 60.0` s at `:751`: a wall-clock pulse, not conversation-gated) and the VPS CC host `cc_ng_host.py:1530` (same call, same loop shape).

**(ii) where nodes get the key**: CC conversational deposits `cc_ng_organism.py:1851` (`_cc_deposit_memory_node`, `_CC_CONV_PROBATION_PERIOD = int(env CC_CONV_PROBATION_PERIOD, "10")` `:1783`; every CC deposit that reaches it originates in `run_conversational_dual_pass`, whose `meta` (base `:2158`) carries `creation_mode: "conversational"` into the forest and tree-concept nodes (`_CCConversationalDualPassEco.record_outcome`; `ng_embed.dual_record_outcome` copies it: `tree_meta = dict(metadata or {})`) and, explicitly, the window nodes; the eco adapter is constructed ONLY there. Verified by `ast` + grep in this session; an earlier draft of this paragraph named four line numbers as "callers", which was imprecise and is corrected here); Syl's conversational deposits `neurograph_rpc.py:2498` (`_CONV_PROBATION_PERIOD`, env `ANIMA_CONV_PROBATION_PERIOD`, default 10, `:2469`); the Ingestor's `NodeRegistrar` `universal_ingestor.py:2187` (`creation_mode: "ingested"`, `probation_period` default 10, profiles 10/100/500 at `:2613/:2640/:2667`). Nothing else writes the key (a grep of the whole NG tree, `*.py` and non-py: four files). `cc_topology_export.py:279` STRIPS `probation_remaining`/`probation_total` from what crosses the callosum, and `cc_topology_merge.py` stamps nothing on receipt (see follow-up F6).

**(iii) can any population be spared forever? — answered in §4 (the table).** Short form: **by classification, no.** The one population that would be spared forever without this predicate (`ingested`) is exactly what the predicate excludes (test 9). **By liveness, conditionally yes** (rows R8, R8b, R9): a node the predicate classifies correctly as "advanced by `cc_update_probation`" stays spared for as long as `cc_update_probation` is not actually completing. The predicate is per-node and cannot see that; the daemon commit makes the common case VISIBLE (WARNING + count) but does not bound it. **No cap added, no exemption added** — reported; the Chief/Exec rule.

## 4. THE ENUMERATION TABLE (every writer and decrementer of `probation_remaining`, over the NG tree)

Processes that run a decrementer: **(L)** the laptop CC daemon (`scripts/cc-ng-daemon.py`, registers the predicate); **(V)** the VPS CC host `cc_ng_host.py` (registers nothing in this build); **(S)** Syl's `neurograph_rpc.py` sidecar (registers nothing); **(I)** the Ingestor sweep.

| # | population (writer; creation_mode / shape) | decrementer (file:line) | process that runs it, cadence | spared by the sweep after this change? | **STILL spared forever AFTER the predicate? (yes/no + why)** |
|---|---|---|---|---|---|
| R1 | CC conversational deposits (the forest node, its tree-concept nodes AND its window nodes: all go through `_cc_deposit_memory_node`) — `cc_ng_organism.py:1851`; `creation_mode="conversational"`; window `CC_CONV_PROBATION_PERIOD` (10) | `cc_update_probation` `:2055-2134` | (L) `_autosave_loop` every 60 s (D24 `:2644`); (V) `cc_ng_host.py:1530` | (L) yes while `probation_remaining > 0` (predicate True); (V) no (nothing registered) | **No (bounded) — closes after 10 completed pulses ≈ 10 min wall clock — PROVIDED the pulse reaches `cc_update_probation` (conditional-yes: see R8/R8b/R9).** |
| R2 | CC nodes with the key but NO `creation_mode` (older checkpoints / seeds; no current writer produces one: every CC deposit goes through `run_conversational_dual_pass`, which stamps `conversational`) | same `cc_update_probation` (the skip is an EXCLUSION, `:2079-2082`, so they ARE decremented) | (L) as R1 | (L) yes (predicate True) | **No (bounded), same condition as R1.** Shape cannot be measured on the live checkpoint (never opened); the predicate keeps them in the decremented population, which is the truth. |
| R3 | **Ingested** — `universal_ingestor.py:2187` `NodeRegistrar`; `creation_mode="ingested"`; `probation_period` default 10 (profiles 10/100/500) | `NodeRegistrar.update_probation` `:2261-2316` via `UniversalIngestor.update_probation` `:2892` <- `openclaw_hook.py:1240` (inside `on_message`, success path); `cc_update_probation` SKIPS them (`:2089`) | (I) **on the laptop only inside `on_message`, whose ONLY caller is `handle_import` (`cc-ng-daemon.py:1494`, a manual socket verb, never the pulse)**: the last <period nodes of an import stay `> 0` until the next import, forever if none comes. Syl: `on_message` has no callers (comment `cc_ng_organism.py:2084-2086`). | **No — predicate is False** | **No — the predicate returns False so the sweep never spares them (test 9). WITHOUT the predicate this is the unbounded leak: yes, forever.** This is the population the amendment exists for. |
| R4 | Syl's conversational deposits — `neurograph_rpc.py:2498`; `creation_mode="conversational"` (`:2808`, `:4955`); window `ANIMA_CONV_PROBATION_PERIOD` (10) | `_update_probation` `:2598-2655`, called at `:3769` inside `handle_after_turn` (`:3619`): **per TURN, conversation-gated** (LAW 8). It decrements EVERY node that has the key, **`ingested` ones included** (no skip) | (S) Syl's sidecar, once per turn | **No — Syl registers NOTHING; `_probation_advances` is absent; the sweep is today's (test 11)** | **No (nothing is exempt for her).** If Josh later registers there: yes whenever she goes quiet, because her window advances only on conversation. That is the rollout decision, not this trial. |
| R5 | Callosum ARRIVALS (merged VPS/laptop nodes) — `cc_topology_merge.py:465` `create_node(metadata=dict(meta))`; `_portable_metadata` (`cc_topology_export.py:279`) strips `probation_remaining`/`probation_total`, `creation_mode` rides | none (no key) | — | **No — no key => today's sweep** | **No — never has the key.** (See F6: the export comment says "receiver runs its own probation window", but nothing stamps one on receipt, so arrivals get NO window from this change.) |
| R6 | Every other node kind (`emergent` wants `cc_ng_organism.py:1766`, `constitutional`, `sensory`, `discovered`, `seam_split`) | none | — | No — no key | **No — never has the key** (constitutional / `*_authored` are identity-protected anyway). |
| R7 | The VPS CC host `cc_ng_host.py` graph (R1-shaped nodes) | `cc_update_probation` `cc_ng_host.py:1530` | (V) its own `_autosave_loop` | **No — registers nothing in this build (out of scope: no VPS)** | **No (nothing exempt there).** Parity is a Chief question (F2). |
| **R8** | **R1/R2 nodes, when the pulse does not reach `cc_update_probation`**: an earlier call in the held section's `try` raises EVERY cycle (`_drain_precheck_blocked`, `drain_ingest_tract`, `trickle_gateway_conduit`; D24 `:2601-2644`), so the decrement is skipped | (same function, never reached) | (L) | yes (predicate True, window stays open) | **YES, while the failure persists — unbounded in time.** The predicate is per-node and CANNOT express a process-liveness failure. The daemon commit makes it VISIBLE (WARNING + `probation_cycles_skipped`, rate-limited, class name only) but does not bound it. The Exec accepted a skipped decrement on that condition (brief). |
| **R8b** | same nodes, when the graph is busy EVERY cycle: `_concurrent_lock.acquire(blocking=False)` fails => `continue` skips the whole held section INCLUDING probation (D24 `:2591-2594`) | (never reached) | (L) | yes | **YES only if the lock is busy at every 60 s tick, persistently** (heavy deposit/merge). **Not covered by the daemon commit** (DEBUG-only, uncounted; the brief limits that commit to the `except`'s visibility). Reported as F3. |
| **R9** | **A "poison" node**: `probation_remaining` is a `str`, or `metadata` is `None`/a truthy non-dict. `cc_update_probation` raises (`TypeError` at `prob <= 0`, `AttributeError` at `.get`) at that node EVERY pulse; nodes AFTER it in `graph.nodes` order are never decremented. If the poison node is BOUND it is never swept, so it persists | (raises mid-loop) | (L) | later nodes yes | **YES (theoretical): every later R1/R2 node is spared while the poison persists.** No current writer produces the shape (writers stamp `int(env)`; `ingested` nodes are skipped BEFORE the compare). Legacy checkpoints could not be measured (never opened). Visible through the same daemon WARNING (`TypeError`). Fix-at-source (LAW 4) would be a per-node fail-soft in `cc_update_probation`: F4. |
| R10 | `probation_remaining = inf` / huge / NaN | `cc_update_probation` (`inf-1 == inf`; NaN stays NaN) | — | `inf` and NaN: **No — excluded by the sweep's parse (fail toward today)**; a huge finite value is spared for `value x 60 s` | **No** (`inf`/NaN excluded; huge finite is bounded and a config choice; no cap per the Exec). **`inf` exclusion is MY addition to the brief's list (8.2).** |
| R11 | An exact-repeat turn lands on the SAME node (content-hashed id) and `_cc_deposit_memory_node` re-stamps the window (`:1851`, `node.metadata.update(meta)` then `= PERIOD`) | `cc_update_probation` | (L) | yes, restarted on each repeat | **No (bounded by repetition)** — spared only while the same text keeps recurring, i.e. while it is a live node; not an unbounded leak by itself. Noted for completeness. |

**STOP-condition judgement (I did not stop):** every population above CLASSIFIES cleanly as spare / not-spare by the one predicate. R8/R8b/R9 are not populations the predicate fails to classify — they are the runtime-liveness gap between "is in the decremented population" and "is ACTUALLY being decremented now" (the Exec's own P554 phrase). I judged that the Exec's accepted "visible, not bounded" covers R8; **R9 and R8b are NEW facts the brief did not list, so I am putting them in front of the Chief explicitly rather than treating them as covered.** If the Chief reads P554 as requiring the exemption to *stop* when the pulse is not completing, that needs a design decision (a liveness signal the host passes, or decoupling `cc_update_probation` into its own `try`: F3/F4) and is NOT something this build can express with a per-node predicate.

## 5. Canonical scope note (recorded in the changelogs; NOT acted on)

On Syl's process `neurograph_rpc._update_probation` is conversation-gated (`handle_after_turn`, `:3769` inside `:3619`), so the window there does not advance on the autonomic clock (LAW 8). It decrements **every node that carries the key, including `ingested` nodes** (the CC mirror skips them; Syl's Ingestor sweep is dead code because `on_message` has no callers there). Syl registers nothing, so nothing changes for her (test 11: no registration => byte-identical to the base over a seeded family that includes open windows). NG is the LAPTOP's primary canonical checkout, so NG-2 is canonical code: it ALSO changes Syl's sweep at ROLLOUT if her host ever registers. That is Josh's call (row note f01e5219).

## 6. Tests

Files: `tests/test_cc_probation_advances_p552.py` (NG-1, 17 tests), `tests/test_cc_sweep_probation_p552.py` (NG-2, 65 test items), `scripts/tests/test_cc_ng_daemon_probation_skip_p552.py` (daemon, 20). (Deviation noted in 8.2: two NG test files, so each commit carries its own tests beside one source file.)

| # | brief item | test(s) |
|---|---|---|
| 1 | open window survives real steps past grace | `test_1_open_window_survives_real_steps_past_grace` (60+ real `step()`; a keyless control IS swept, so the sweep demonstrably ran) |
| 2 | window closes via the REAL `cc_update_probation`, then swept | `test_2_window_closes_via_real_cc_update_probation_then_swept` |
| 3 | wires during the window, survives after | `test_3_...real_hyperedge...`, `test_3b_...engines_own_cofiring_sprout...` (precondition asserts the engine really sprouted; `Graph.stimulate`) |
| 4 | identity-protected unaffected | `test_4_identity_protected_unaffected` (10 cases: registered/unregistered x 5 probation shapes; the predicate is never reached) |
| 5 | no key => today's sweep, byte-identical to the base | `test_5_...` (120 seeds, registered predicate, vs the base function embedded VERBATIM from `b5e47686`) |
| 6 | odd shapes not spared | `test_6_...` (16 values incl. `None`,`"5"`,`-1`,`0`,`True`,`False`,NaN,`inf`,`-inf`; the predicate is never called), `6b` (numeric >0 spared), `6c` (falsy non-dict metadata), `6d` (truthy non-dict raises exactly as the base) |
| 7 | grace unchanged | `test_7_...` (boundary ages 0, 24, 25, 26, 400), `7b` |
| 8 | existing orphan/grace tests | run, see 7.2 |
| 9 | ingested + open window STILL swept (real predicate) | `test_9_ingested_node_with_an_open_window_is_still_swept_with_the_real_predicate` |
| 10 | conversational, same state, spared | `test_10_conversational_node_in_the_same_state_is_spared` |
| 11 | no registration => today's sweep (Syl) | `test_11_no_registration_is_exactly_todays_sweep_open_windows_included` (120 seeds, probation values incl. open windows) + `test_the_exemption_only_ever_spares_...` (monotone: never removes a node today's sweep keeps) |
| 12 | predicate raises => not spared, ONE WARNING with a count | `test_12_...` (3 nodes, 1 WARNING, count + class, no `str(exc)`), `12b` (one per sweep), `12c` (mixed), `12d` (none when it does not raise), `12e` (non-callable) |
| 13 | defined ONCE and CONSULTED (static `ast`) | `test_probation_advances_defined_exactly_once_in_cc_ng_organism`, `test_cc_update_probation_references_the_predicate_and_holds_no_literal`, `test_the_ingested_literal_lives_only_in_the_predicate_within_cc_ng_organism`, `test_neuro_foundation_holds_no_ingested_literal_and_imports_no_cc_module` |
| 14 | `cc_update_probation` byte-identical to the base | `test_cc_update_probation_byte_identical_clean_family` / `..._odd_family` (150 seeds each x 14 pulses; ingested/conversational/no-mode/None-metadata/`str`/NaN/`True`/`inf`; compares every node's metadata, excitability, threshold, the `graduated` list, and for raising shapes the exception class + partial state) |

**Does any test depend on node order?** The sweep builds `orphans` from `self.nodes` (a dict: insertion order, not hash order); the seeded families use `random.Random(seed)` and lists. Nothing depends on set/dict hash order, and the 0..31 `PYTHONHASHSEED` sweep (7.3) confirms it empirically.


## 7. Runs (every START load recorded; nothing here is a claim the run outputs do not back)

### 7.1 Discipline actually applied

Every run went through one wrapper that: slept 25 s BEFORE the run, read `/proc/loadavg`, **refused at 1-min >= 6 OR 5-min >= 5.0** (exit 99, never retried while closed), then ran under
`env -u CC_NG_BATCH_SIZE -u CC_NG_IDLE_STEPS -u CC_NG_DRAIN_HOLD_ON_FAILURE -u NG_EMBED_REMOTE -u CC_NG_IN_TRANSIT_IDS_PATH` and PRINTED the effective values first. **Every run printed
`CC_NG_BATCH_SIZE=<unset> CC_NG_IDLE_STEPS=<unset> CC_NG_DRAIN_HOLD_ON_FAILURE=<unset> NG_EMBED_REMOTE=<unset> CC_NG_IN_TRANSIT_IDS_PATH=<unset>`** (seen in each output). No real `systemctl`; no kills; deletes by exact path only
(scratch under `/tmp`, listed in 8.1). `PYTHONHASHSEED=0` unless a seed table says otherwise.

**One refusal happened (seed 9), and it was my own doing:** at `2026-10-02T01:30:52Z` the START load was `1m=6.23 5m=4.47`, the batch STOPPED as designed (seeds 9..31 not run at that moment) — the spike coincided with my
own anchor-validation loop, which extracted ~95 MB of scratch trees for the 21 mutants during the sweep. I did not retry while the gate was closed; I resumed as a NEW batch that first waited until the load settled
(`1m<5.0 and 5m<4.5`, it waited 195 s) and then ran seeds 9..31, each under its own full gate. The gate itself was never lowered. Lesson recorded: no heavy file work during a gated batch.

### 7.2 (8) The pre-existing orphan/grace/probation tests (on the committed tree)

| run | START load (1m / 5m / 15m) | files | result |
|---|---|---|---|
| `ng-existing-sweep-tests` | 3.81 / 4.01 / 4.01 @ 01:23:30Z | `tests/test_identity_protection.py`, `tests/test_cc_topology_callosum.py`, `tests/test_save_guard_structural.py` | **70 passed, 1 failed** |
| `baseline-poincare-test-on-b5e4768` | 4.05 / 4.07 / 4.03 @ 01:24:32Z | the one failing test, on a scratch extraction of the BASE `b5e47686` | **fails identically (same `ValueError`)** |
| `ng1-final` | 3.04 / 3.50 / 3.91 @ 01:17:18Z | the NG-1 file + `test_cc_capture_mutations_423.py::test_probation_keeps_existing_clock_semantics` + `test_cc_dual_pass.py::{test_cc_probation_rollback_drains_marker_instead_of_stranding_it, test_cc_update_probation_leaves_ingested_nodes_to_the_ingestor, test_cc_update_probation_still_owns_nodes_with_no_creation_mode}` | 21 passed |
| `daemon-p552-and-d24-r2` | 2.56 / 2.53 / 3.12 @ 01:50:34Z | the daemon file + the UNMODIFIED D24 `scripts/tests/test_cc_ng_daemon_drain_pacing.py` | **84 passed** (20 new + 64 D24) |

**The one failure is PRE-EXISTING and unrelated:** `tests/test_cc_topology_callosum.py::test_poincare_dir_is_rederived_locally_not_transmitted` does `np.asarray(n.metadata["poincare_dir"], dtype=np.float32)` on a value that is packed float32 BYTES since #400; it fails the same way on the base. Not touched (F5).
`tests/test_cc_refeed.py` was NOT run: it mentions "orphan" for its own collector and builds a real `NeuroGraphMemory` (~800 MB, heavy); it is not a sweep test.

**Existing tests whose expectation changed (file, test, why; never weakened): exactly ONE, and it is a harness change, not an expectation change:**
`tests/test_cc_capture_mutations_423.py::test_probation_keeps_existing_clock_semantics` AST-extracts ONLY the named functions into a bare namespace, so after the refactor it hit
`NameError: name 'probation_advances' is not defined` (observed: run `ng1-existing-423-probation`, START 2.04 / 3.41 / 3.90 @ 01:16:29Z). `functions('cc_update_probation')` became
`functions('probation_advances','cc_update_probation')`; every assertion is untouched and it passes. The D24 daemon test file is unchanged and passes.

### 7.3 (SEED SWEEP) `PYTHONHASHSEED` 0..31, ONE gated invocation per seed, both NG test files (17 + 65 = 82 tests), on the committed NG-2 tree (clean: 0 dirty paths)

**Seeds ran: 32 of 32 (0..31); every one `82 passed`.** Seed 9 was REFUSED once at `1m=6.23` (see 7.1) and later ran green under its own gate (the table keeps both rows).

| PYTHONHASHSEED | result | START load (1m / 5m / 15m @ UTC) |
|---|---|---|
| 0 | rc=0, 82 passed in 8.21s | 4.58 / 4.19 / 4.07 @ 01:25:20Z |
| 1 | rc=0, 82 passed in 11.40s | 3.59 / 3.96 / 3.99 @ 01:25:55Z |
| 2 | rc=0, 82 passed in 7.69s | 3.79 / 4.01 / 4.01 @ 01:26:33Z |
| 3 | rc=0, 82 passed in 9.31s | 3.39 / 3.87 / 3.96 @ 01:27:08Z |
| 4 | rc=0, 82 passed in 8.45s | 3.86 / 3.93 / 3.98 @ 01:27:44Z |
| 5 | rc=0, 82 passed in 11.43s | 2.79 / 3.66 / 3.89 @ 01:28:19Z |
| 6 | rc=0, 82 passed in 9.88s | 4.09 / 3.87 / 3.95 @ 01:28:57Z |
| 7 | rc=0, 82 passed in 8.52s | 4.32 / 3.98 / 3.98 @ 01:29:34Z |
| 8 | rc=0, 82 passed in 16.12s | 3.59 / 3.83 / 3.93 @ 01:30:09Z |
| 9 | **REFUSED** (gate closed; not retried while closed) | 6.23 / 4.47 / 4.14 @ 01:30:52Z |
| 9 | rc=0, 82 passed in 10.78s | 3.14 / 4.23 / 4.18 @ 01:35:09Z |
| 10 | rc=0, 82 passed in 9.23s | 2.57 / 4.00 / 4.10 @ 01:35:46Z |
| 11 | rc=0, 82 passed in 8.62s | 2.12 / 3.75 / 4.01 @ 01:36:22Z |
| 12 | rc=0, 82 passed in 8.82s | 2.75 / 3.73 / 4.00 @ 01:36:58Z |
| 13 | rc=0, 82 passed in 10.92s | 2.79 / 3.66 / 3.96 @ 01:37:33Z |
| 14 | rc=0, 82 passed in 8.66s | 2.51 / 3.51 / 3.90 @ 01:38:11Z |
| 15 | rc=0, 82 passed in 8.65s | 2.40 / 3.37 / 3.84 @ 01:38:46Z |
| 16 | rc=0, 82 passed in 11.07s | 3.28 / 3.49 / 3.86 @ 01:39:22Z |
| 17 | rc=0, 82 passed in 12.14s | 2.66 / 3.34 / 3.80 @ 01:40:01Z |
| 18 | rc=0, 82 passed in 9.64s | 3.08 / 3.38 / 3.79 @ 01:40:40Z |
| 19 | rc=0, 82 passed in 7.61s | 3.43 / 3.46 / 3.81 @ 01:41:17Z |
| 20 | rc=0, 82 passed in 7.89s | 2.39 / 3.19 / 3.70 @ 01:41:51Z |
| 21 | rc=0, 82 passed in 7.90s | 1.96 / 3.00 / 3.62 @ 01:42:25Z |
| 22 | rc=0, 82 passed in 8.97s | 1.72 / 2.83 / 3.54 @ 01:42:59Z |
| 23 | rc=0, 82 passed in 8.21s | 1.58 / 2.67 / 3.46 @ 01:43:35Z |
| 24 | rc=0, 82 passed in 11.25s | 1.71 / 2.58 / 3.40 @ 01:44:09Z |
| 25 | rc=0, 82 passed in 8.00s | 2.13 / 2.61 / 3.38 @ 01:44:48Z |
| 26 | rc=0, 82 passed in 10.39s | 2.52 / 2.64 / 3.36 @ 01:45:22Z |
| 27 | rc=0, 82 passed in 10.55s | 3.44 / 2.91 / 3.42 @ 01:46:00Z |
| 28 | rc=0, 82 passed in 10.49s | 3.08 / 2.90 / 3.40 @ 01:46:37Z |
| 29 | rc=0, 82 passed in 8.19s | 3.02 / 2.92 / 3.39 @ 01:47:14Z |
| 30 | rc=0, 82 passed in 8.82s | 2.15 / 2.72 / 3.30 @ 01:47:49Z |
| 31 | rc=0, 82 passed in 8.26s | 2.13 / 2.66 / 3.26 @ 01:48:25Z |

### 7.4 Mutants — NG (each a scratch copy of the COMMITTED file with ONE asserted single-line change; ONE gated run each; run on the NG-2 tree)

**21 of 21 NG mutants KILLED, 0 SURVIVED.** The first campaign stopped after 5 mutants at a gate refusal on `n2-before-structural` (START `1m=6.29`: external load from other sessions; not retried while closed); the resume ran after a bounded settle-wait and the `n2-before-structural` row below is its real run. The first refused attempt is listed as `REFUSED`.

| mutant | what it breaks | verdict | result | START load (1m / 5m / 15m @ UTC) | killed by (first 3) |
|---|---|---|---|---|---|
| `n2-drop-term` | the probation term dropped (nothing is ever spared) | **KILLED** | 20 failed, 45 passed in 3.82s | 2.70 / 2.63 / 3.10 @ 01:52:10Z | `test_1_open_window_survives_real_steps_past_grace`; `test_2_window_closes_via_real_cc_update_probation_then_swept`; `test_3_wired_by_a_real_hyperedge_during_the_window_survives_after_it` (+17 more) |
| `n2-gte` | > replaced by >= (a closed window still spares) | **KILLED** | 7 failed, 58 passed in 4.69s | 2.71 / 2.66 / 3.10 @ 01:52:42Z | `test_2_window_closes_via_real_cc_update_probation_then_swept`; `test_3_wired_by_a_real_hyperedge_during_the_window_survives_after_it`; `test_3b_wired_by_the_engines_own_cofiring_sprout_survives_after_the_window` (+4 more) |
| `n2-bool-accepted` | bool accepted as protecting | **KILLED** | 1 failed, 64 passed in 3.78s | 3.47 / 2.87 / 3.15 @ 01:53:16Z | `test_6_odd_probation_values_are_not_spared_and_never_reach_the_predicate[True]` |
| `n2-str-accepted` | a non-empty str accepted as protecting | **KILLED** | 2 failed, 63 passed in 8.59s | 3.02 / 2.83 / 3.13 @ 01:53:48Z | `test_6_odd_probation_values_are_not_spared_and_never_reach_the_predicate[5]`; `test_6_odd_probation_values_are_not_spared_and_never_reach_the_predicate[abc]` |
| `n2-missing-key-protected` | a missing key treated as protected (the forever-leak mutant) | **KILLED** | 15 failed, 50 passed in 4.23s | 5.88 / 3.60 / 3.38 @ 01:54:26Z | `test_1_open_window_survives_real_steps_past_grace`; `test_4_identity_protected_unaffected[5-True]`; `test_4_identity_protected_unaffected[0-True]` (+12 more) |
| `n2-before-structural` | the term evaluated over ALL nodes, before/instead of the structural checks | **REFUSED** | not run | 6.29 / 3.90 / 3.49 @ 01:54:58Z |  |
| `n2-before-structural` | the term evaluated over ALL nodes, before/instead of the structural checks | **KILLED** | 10 failed, 55 passed in 2.17s | 3.71 / 3.96 / 3.59 @ 01:57:29Z | `test_3_wired_by_a_real_hyperedge_during_the_window_survives_after_it`; `test_3b_wired_by_the_engines_own_cofiring_sprout_survives_after_the_window`; `test_4_identity_protected_unaffected[5-True]` (+7 more) |
| `n2-identity-bypass` | identity protection bypassed | **KILLED** | 15 failed, 50 passed in 1.53s | 3.19 / 3.82 / 3.55 @ 01:57:58Z | `test_4_identity_protected_unaffected[5-True]`; `test_4_identity_protected_unaffected[5-False]`; `test_4_identity_protected_unaffected[0-True]` (+12 more) |
| `n2-grace-bypass` | grace bypassed | **KILLED** | 9 failed, 56 passed in 1.40s | 2.52 / 3.60 / 3.48 @ 01:58:26Z | `test_5_no_probation_key_is_todays_sweep_even_when_the_predicate_is_registered`; `test_11_no_registration_is_exactly_todays_sweep_open_windows_included`; `test_the_exemption_only_ever_spares_and_does_something_when_registered` (+6 more) |
| `n2-disable-sweep-any-open` | the whole sweep disabled while probation>0 on ANY node | **KILLED** | 11 failed, 54 passed in 3.46s | 3.24 / 3.65 / 3.50 @ 01:58:55Z | `test_1_open_window_survives_real_steps_past_grace`; `test_4_identity_protected_unaffected[5-True]`; `test_4_identity_protected_unaffected[5-False]` (+8 more) |
| `n2-exc-no-warning` | a raising predicate swallowed with NO WARNING | **KILLED** | 4 failed, 61 passed in 3.16s | 3.68 / 3.73 / 3.54 @ 01:59:26Z | `test_12_raising_predicate_node_not_spared_one_warning_with_count_and_class_only`; `test_12b_one_warning_per_sweep_not_per_process`; `test_12c_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise` (+1 more) |
| `n2-exc-treated-as-spare` | a raising predicate treated as SPARE | **KILLED** | 3 failed, 62 passed in 5.99s | 3.02 / 3.56 / 3.49 @ 01:59:56Z | `test_12_raising_predicate_node_not_spared_one_warning_with_count_and_class_only`; `test_12c_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise`; `test_12e_a_non_callable_registration_fails_toward_todays_sweep_with_a_warning` |
| `n2-no-registration-spare-all` | no registration treated as spare-all | **KILLED** | 1 failed, 64 passed in 4.41s | 4.50 / 3.92 / 3.61 @ 02:00:32Z | `test_11_no_registration_is_exactly_todays_sweep_open_windows_included` |
| `n2-ignores-pred` | the sweep ignores pred | **KILLED** | 8 failed, 57 passed in 3.35s | 3.73 / 3.81 / 3.59 @ 02:01:05Z | `test_7b_an_open_window_spares_at_every_age_and_grace_still_spares_a_young_node[26]`; `test_7b_an_open_window_spares_at_every_age_and_grace_still_spares_a_young_node[400]`; `test_9_ingested_node_with_an_open_window_is_still_swept_with_the_real_predicate` (+5 more) |
| `n2-inf-accepted` | inf accepted (a window that never closes) | **KILLED** | 1 failed, 64 passed in 5.12s | 3.82 / 3.81 / 3.60 @ 02:01:36Z | `test_6_odd_probation_values_are_not_spared_and_never_reach_the_predicate[inf]` |
| `n2-warning-per-node` | one WARNING per raising node (not one per sweep) | **KILLED** | 4 failed, 61 passed in 4.26s | 2.91 / 3.61 / 3.53 @ 02:02:08Z | `test_12_raising_predicate_node_not_spared_one_warning_with_count_and_class_only`; `test_12b_one_warning_per_sweep_not_per_process`; `test_12c_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise` (+1 more) |
| `n2-str-exc` | str(exc) put into the WARNING | **KILLED** | 2 failed, 63 passed in 5.16s | 3.08 / 3.57 / 3.52 @ 02:02:40Z | `test_12_raising_predicate_node_not_spared_one_warning_with_count_and_class_only`; `test_12c_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise` |
| `n1-pred-true-for-ingested` | the predicate True for ingested | **KILLED** | 6 failed, 76 passed in 6.39s | 3.47 / 3.63 / 3.54 @ 02:03:13Z | `test_predicate_false_for_ingested`; `test_cc_update_probation_consults_the_module_level_predicate`; `test_cc_update_probation_byte_identical_clean_family` (+3 more) |
| `n1-pred-false-for-all` | the predicate False for everything | **KILLED** | 20 failed, 62 passed in 7.39s | 2.74 / 3.44 / 3.49 @ 02:03:47Z | `test_predicate_true_for_everything_else[meta0]`; `test_predicate_true_for_everything_else[meta1]`; `test_predicate_true_for_everything_else[meta2]` (+17 more) |
| `n1-update-ignores-pred` | cc_update_probation ignores the predicate (ingested decremented) | **KILLED** | 4 failed, 78 passed in 6.08s | 2.47 / 3.30 / 3.44 @ 02:04:21Z | `test_cc_update_probation_consults_the_module_level_predicate`; `test_cc_update_probation_byte_identical_clean_family`; `test_cc_update_probation_byte_identical_odd_family` (+1 more) |
| `n1-pred-duplicated` | the predicate DEFINED TWICE | **KILLED** | 2 failed, 80 passed in 8.05s | 2.05 / 3.13 / 3.38 @ 02:04:54Z | `test_probation_advances_defined_exactly_once_in_cc_ng_organism`; `test_the_ingested_literal_lives_only_in_the_predicate_within_cc_ng_organism` |
| `n1-inline-literal` | cc_update_probation holds its own inline 'ingested' literal (predicate not consulted) | **KILLED** | 3 failed, 79 passed in 9.65s | 2.29 / 3.08 / 3.35 @ 02:05:29Z | `test_cc_update_probation_consults_the_module_level_predicate`; `test_cc_update_probation_references_the_predicate_and_holds_no_literal`; `test_the_ingested_literal_lives_only_in_the_predicate_within_cc_ng_organism` |

### 7.5 Mutants — daemon (scratch copy of the COMMITTED `cc-ng-daemon.py`, loaded through the test file's `Z12_P552_DAEMON_UNDER_TEST`; ONE gated run each)

**10 of 10 daemon mutants KILLED, 0 SURVIVED.**

| mutant | what it breaks | verdict | result | START load (1m / 5m / 15m @ UTC) | killed by (first 3) |
|---|---|---|---|---|---|
| `d-back-to-debug` | the skip back to DEBUG | **KILLED** | 5 failed, 15 passed in 1.31s | 1.76 / 2.86 / 3.27 @ 02:06:05Z | `test_a_drain_that_raises_warns_once_per_window_with_the_class_name_and_counts_every_skipped_cycle`; `test_the_warning_repeats_when_the_window_has_elapsed`; `test_with_the_limiter_open_every_skipped_cycle_warns` (+2 more) |
| `d-no-counter` | no counter | **KILLED** | 3 failed, 17 passed in 1.13s | 3.44 / 3.18 / 3.36 @ 02:06:33Z | `test_a_drain_that_raises_warns_once_per_window_with_the_class_name_and_counts_every_skipped_cycle`; `test_the_warning_repeats_when_the_window_has_elapsed`; `test_cc_update_probation_itself_raising_counts_as_a_skip` |
| `d-str-exc` | str(exc) in the WARNING text | **KILLED** | 1 failed, 19 passed in 1.03s | 2.38 / 2.95 / 3.28 @ 02:07:01Z | `test_the_message_never_carries_str_exc_at_any_level` |
| `d-not-rate-limited` | not rate-limited | **KILLED** | 2 failed, 18 passed in 1.04s | 1.78 / 2.76 / 3.21 @ 02:07:29Z | `test_a_drain_that_raises_warns_once_per_window_with_the_class_name_and_counts_every_skipped_cycle`; `test_the_warning_repeats_when_the_window_has_elapsed` |
| `d-counter-on-clean-cycle` | the counter incremented on a CLEAN cycle | **KILLED** | 3 failed, 17 passed in 0.94s | 1.38 / 2.58 / 3.13 @ 02:07:56Z | `test_a_clean_cycle_logs_no_warning_and_leaves_the_counter_unchanged`; `test_a_raise_after_the_decrement_is_not_a_probation_skip[wants]`; `test_a_raise_after_the_decrement_is_not_a_probation_skip[emergent]` |
| `d-skip-flag-ignored` | a raise AFTER the decrement counted as a skip | **KILLED** | 2 failed, 18 passed in 0.95s | 1.47 / 2.50 / 3.09 @ 02:08:24Z | `test_a_raise_after_the_decrement_is_not_a_probation_skip[wants]`; `test_a_raise_after_the_decrement_is_not_a_probation_skip[emergent]` |
| `d-reg-missing` | the predicate never registered | **KILLED** | 3 failed, 17 passed in 0.97s | 1.20 / 2.34 / 3.01 @ 02:08:51Z | `test_init_registers_the_very_function_object_from_cc_ng_organism`; `test_a_graph_that_refuses_the_attribute_warns_once_class_name_only_and_the_daemon_starts`; `test_a_failing_organism_bootstrap_does_not_skip_the_registration` |
| `d-reg-wrapper-copy` | a wrapper registered instead of the function object itself | **KILLED** | 2 failed, 18 passed in 0.93s | 0.93 / 2.18 / 2.94 @ 02:09:18Z | `test_init_registers_the_very_function_object_from_cc_ng_organism`; `test_a_failing_organism_bootstrap_does_not_skip_the_registration` |
| `d-reg-failure-raises` | a registration failure propagates (the daemon would not start) | **KILLED** | 2 failed, 18 passed in 0.95s | 0.98 / 2.07 / 2.88 @ 02:09:46Z | `test_an_old_ng_without_probation_advances_warns_once_sets_nothing_and_the_daemon_starts`; `test_a_graph_that_refuses_the_attribute_warns_once_class_name_only_and_the_daemon_starts` |
| `d-reg-str-exc` | str(exc) in the registration-failure WARNING | **KILLED** | 1 failed, 19 passed in 0.98s | 1.05 / 2.00 / 2.83 @ 02:10:13Z | `test_a_graph_that_refuses_the_attribute_warns_once_class_name_only_and_the_daemon_starts` |

### 7.5b Daemon-test seed sweep (`PYTHONHASHSEED` 0..31; the new daemon file + the D24 file = 84 tests), one gated invocation per seed, on the committed daemon hash `905618ee`

**Seeds ran: 32 of 32.** All `84 passed`.

A gate refusal occurred (`1m=8.85`); the batch stopped there and the table below lists exactly what ran and what did not.

| PYTHONHASHSEED | result | START load (1m / 5m / 15m @ UTC) |
|---|---|---|
| 0 | rc=0, 84 passed in 2.18s | 1.50 / 2.01 / 2.81 @ 02:10:47Z |
| 1 | rc=0, 84 passed in 1.80s | 1.35 / 1.93 / 2.75 @ 02:11:16Z |
| 2 | rc=0, 84 passed in 3.15s | 1.10 / 1.82 / 2.70 @ 02:11:44Z |
| 3 | rc=0, 84 passed in 2.22s | 1.28 / 1.81 / 2.66 @ 02:12:14Z |
| 4 | rc=0, 84 passed in 1.81s | 0.95 / 1.68 / 2.59 @ 02:12:42Z |
| 5 | rc=0, 84 passed in 2.16s | 1.07 / 1.63 / 2.55 @ 02:13:11Z |
| 6 | rc=0, 84 passed in 2.27s | 1.61 / 1.73 / 2.56 @ 02:13:39Z |
| 7 | rc=0, 84 passed in 2.64s | 4.62 / 2.45 / 2.77 @ 02:14:08Z |
| 8 | rc=0, 84 passed in 4.25s | 4.08 / 2.53 / 2.79 @ 02:14:39Z |
| 9 | rc=0, 84 passed in 2.87s | 5.12 / 2.90 / 2.90 @ 02:15:10Z |
| 10 | rc=0, 84 passed in 4.21s | 4.04 / 2.85 / 2.88 @ 02:15:41Z |
| 11 | rc=0, 84 passed in 4.38s | 3.18 / 2.77 / 2.86 @ 02:16:13Z |
| 12 | rc=0, 84 passed in 5.33s | 2.48 / 2.65 / 2.81 @ 02:16:45Z |
| 13 | rc=0, 84 passed in 12.78s | 5.89 / 3.39 / 3.05 @ 02:17:19Z |
| 14 | **REFUSED** (gate closed; not retried while closed) | 8.85 / 4.62 / 3.49 @ 02:18:08Z |
| 14 | rc=0, 84 passed in 5.02s | 3.28 / 4.25 / 4.50 @ 02:31:48Z |
| 15 | rc=0, 84 passed in 3.84s | 3.87 / 4.34 / 4.52 @ 02:32:21Z |
| 16 | rc=0, 84 passed in 5.10s | 2.73 / 4.02 / 4.40 @ 02:32:52Z |
| 17 | rc=0, 84 passed in 3.83s | 2.58 / 3.88 / 4.35 @ 02:33:26Z |
| 18 | rc=0, 84 passed in 4.13s | 2.02 / 3.62 / 4.24 @ 02:33:57Z |
| 19 | rc=0, 84 passed in 3.89s | 1.94 / 3.46 / 4.17 @ 02:34:29Z |
| 20 | rc=0, 84 passed in 3.83s | 1.70 / 3.24 / 4.07 @ 02:35:00Z |
| 21 | rc=0, 84 passed in 3.89s | 1.88 / 3.15 / 4.02 @ 02:35:31Z |
| 22 | rc=0, 84 passed in 3.83s | 1.40 / 2.91 / 3.91 @ 02:36:03Z |
| 23 | rc=0, 84 passed in 1.78s | 1.56 / 2.81 / 3.84 @ 02:36:34Z |
| 24 | rc=0, 84 passed in 1.79s | 1.13 / 2.59 / 3.73 @ 02:37:02Z |
| 25 | rc=0, 84 passed in 2.30s | 1.00 / 2.42 / 3.64 @ 02:37:30Z |
| 26 | rc=0, 84 passed in 1.78s | 0.95 / 2.29 / 3.56 @ 02:37:59Z |
| 27 | rc=0, 84 passed in 2.62s | 0.68 / 2.10 / 3.46 @ 02:38:27Z |
| 28 | rc=0, 84 passed in 1.82s | 0.90 / 2.02 / 3.39 @ 02:38:56Z |
| 29 | rc=0, 84 passed in 1.77s | 0.66 / 1.88 / 3.30 @ 02:39:24Z |
| 30 | rc=0, 84 passed in 1.83s | 0.79 / 1.79 / 3.23 @ 02:39:52Z |
| 31 | rc=0, 84 passed in 2.45s | 0.82 / 1.70 / 3.15 @ 02:40:21Z |

### 7.6 Every other run, in order (so nothing is hidden)

| run | START load | outcome |
|---|---|---|
| `ng1-tests-seed0` (`-x`, piped through `tail`) | **line lost: my `tail -30` cut it** (the run executed, so the gate had passed) | 1 failed / 10 passed: a bug in MY test (`conv` was also decremented by the "predicate True for all" step) — fixed |
| `ng1-tests-seed0-r2` | 3.21 / 4.03 / 4.14 @ 01:14:49Z | 1 failed / 16 passed: a non-vacuity floor I had guessed (`graduated > 50`) vs the measured 24 — floor set to 10 with the measurement stated in the test |
| `ng1-tests-seed0-r3` | 2.19 / 3.68 / 4.01 @ 01:15:36Z | 17 passed |
| `ng1-existing-423-probation` | 2.04 / 3.41 / 3.90 @ 01:16:29Z | 1 failed (`NameError`, the predicted pre-fix failure) |
| `ng1-final` | 3.04 / 3.50 / 3.91 @ 01:17:18Z | 21 passed (then NG-1 committed) |
| `ng2-tests-seed0` | 4.84 / 3.96 / 3.98 @ 01:21:16Z | 1 failed / 64 passed: my test called `Graph.inject_current`, which does not exist (the real API is `stimulate`) — fixed |
| `ng2-tests-seed0-r2` | 3.89 / 3.88 / 3.96 @ 01:22:07Z | 65 passed (then NG-2 committed) |
| `daemon-p552-and-d24` | 2.11 / 2.51 / 3.16 @ 01:49:34Z | 1 failed / 83 passed: my pin asserted `self.nodes.clear()` sits directly in `Graph.restore`; it is in `_deserialize`, which `restore` delegates to — pin rewritten to follow the real chain |

NG-1 and NG-2 test runs before each commit were on the working tree, whose content equals the commit (verified clean at commit); the existing-test, seed, mutant and daemon-mutant runs were on the committed hashes.

## 8. What ran, what did not, and the calls I made

### 8.1 What ran
NG-1 / NG-2 / daemon tests; the pre-existing sweep tests named above; the baseline check of the one pre-existing failure; the 0..31 seed sweep of both NG files; 21 NG mutants; 10 daemon mutants; the daemon-test `PYTHONHASHSEED` sweep (32 of 32 seeds).
Scratch created and removed by exact path at the end: `/tmp/p552-mut/<id>/`, `/tmp/p552-dmut/<id>/`, `/tmp/p552-base/`, and the helper scripts/tables under `/tmp/` (`gated_run_p552.sh`, `mk_mut_p552.py`, `mk_dmut_p552.py`, `mutant_batch_p552.sh`, `all_mutants_p552.sh`, `seed_sweep*_p552.sh`, `p552_*.txt`). Nothing under `/tmp` is part of the commits.

### 8.2 Calls I made that you may veto (each stated, none hidden)

1. **Two NG test files, not one.** The brief names `tests/test_cc_sweep_probation_p552.py` and also says NG-1 carries tests (13)/(14). To keep each commit's SOURCE diff to one file with its own tests beside it, NG-1's tests are `tests/test_cc_probation_advances_p552.py` and NG-2's are the named file.
2. **The `neuro_foundation.py` changelog entry lives in the function's docstring** (dated note, the style `_is_identity_protected` already uses), NOT in the file's changelog header. Standing clause 4 asks for a header in every touched file; "ONE function, no other hunk, no module-level addition" is the stricter protected-file rule, so I followed it and kept the diff to the one function. If a header hunk is wanted, it is a second hunk outside the authorised function and needs Josh's word.
3. **`inf` is excluded in the sweep's numeric parse** (`0 < prob < float("inf")`). The brief's list names NaN/bool/negative/str; `inf` is a window that `inf - 1` never closes (R10), i.e. protecting forever, which the brief's own principle says to fail away from. A one-token addition inside the authorised body; `test_6_...[inf]` and the mutant `n2-inf-accepted` pin it.
4. **The daemon counts only REAL skips.** `_probation_ran` is set right after `cc_update_probation` returns; a raise in `surface_wants`/`generate_emergent_want` (which run AFTER the decrement) is not a probation skip and keeps its old DEBUG line. The brief said "any exception earlier in that block"; counting a later raise would have reported a skip that did not happen.
5. **Beyond the brief's mutant list** I added: str-accepted, before-structural (as "the term over ALL nodes"), warning-per-node, str(exc)-in-WARNING (NG); skip-flag-ignored, registration-missing, wrapper-copy-registered, registration-failure-raises, str(exc)-in-registration-WARNING (daemon).
6. The non-vacuity floor in the odd-family byte-identity test was lowered from a guessed 50 to 10 after measuring 24 (stated in the test); the identity assertions themselves never changed.

## 9. Not verified (said plainly)

* **No integration run.** No real `NeuroGraphMemory`, no real `cc_ng_organism` inside the daemon tests (fakes only; the real predicate is pinned by `git show` + `ast` at the NG-2 hash), no live daemon/sidecar/graph/checkpoint/socket, no VPS. The exemption has NOT been observed over wall-clock time on a real graph; "no recurrence over N cycles" claims are out of reach and not made.
* **Legacy populations on the live laptop checkpoint (R2, R9) are unmeasured:** a checkpoint was never opened (standing clause). Whether any `str`/`None`-shaped `probation_remaining` or metadata exists there is unknown.
* `tests/test_cc_deposit_step.py` and the NG full suite were NOT run (forbidden). `tests/test_conversational_recall.py` and `tests/test_ingestor.py` (they reference `probation_remaining` on the CANONICAL `_update_probation` / Registrar, which this build does not touch) were NOT run. `tests/test_cc_dual_pass.py` was run ONLY for the three bare-`Graph` probation tests by node id; its embedder-backed tests were not.
* `tests/test_cc_refeed.py` not run (heavy, not a sweep test). `scripts/tests/` other than the D24 file and the new file were not run.
* The NG tests exercise the sweep on tiny scratch graphs (<= 30 nodes); interaction with the full homeostasis/Lenia/Tonic machinery of a 6,000-node graph is not exercised.
* Whether Syl's host should ever register is not examined (Josh's rollout call).

## 10. Follow-up list (for the Chief; none acted on)

* **F1 — `cc_topology_merge.py` docstring (doc-only, FORBIDDEN here):** `whole_graph_guard`'s docstring claim "the sweep's own test MINUS ONE TERM (age)" becomes "minus TWO terms (age, probation)" once a host registers the predicate. `_unbound_nodes`/`held_unbound_nodes`/`whole_graph_guard` were NOT touched (LAW 4).
* **F2 — VPS CC host parity:** `cc_ng_host.py` drives the same `cc_update_probation` (`:1530`) and registers nothing in this build (no VPS). Its sweep is today's. Whether it should register is a Chief/Josh question.
* **F3 — decoupling idea (REPORTED, not done):** `cc_update_probation` sits in the same `try` as the drain/trickle, so a persistent drain failure stalls the probation decrement and, with the exemption, stretches protection without bound (R8). Moving it into its own `try` would make the window's advance independent of the drain's health; separately, the graph-busy `continue` that skips the whole held section (R8b, D24 `:2591-2594`) is DEBUG-only and uncounted. Both are separate changes.
* **F4 — `cc_update_probation` is not per-node fail-soft (R9):** one malformed node (`str` probation_remaining, `None`/non-dict metadata) aborts the whole pulse mid-loop, every pulse, silently stopping the decrement for every node after it. The fix belongs at the source (LAW 4) and is separate from this build.
* **F5 — pre-existing failing test (punchlist):** `tests/test_cc_topology_callosum.py::test_poincare_dir_is_rederived_locally_not_transmitted` fails on the base `b5e47686` (`np.asarray` on packed-bytes `poincare_dir`, #400).
* **F6 — callosum arrivals get NO probation window:** `cc_topology_export._BANNED_META` strips `probation_remaining` with the comment "receiver runs its own probation window", but `cc_topology_merge.py` stamps no window on receipt (no "probation" in it). So this build protects the laptop's OWN conversational nodes (the P550 case) and does not protect merged arrivals. If §8.13's "firing-keyed arrival exemption" is meant to cover arrivals, that needs its own design.
* **F7 — doc drift:** the NG `CLAUDE.md` "Key Methods" table lists `Graph.inject_current(...)`; the method does not exist (the API is `stimulate` / `stimulate_batch`). I hit it while writing test 3b.
* **F8 — decisions for the Chief/Exec before any merge:** (a) the R8/R8b/R9 liveness residuals (§4); (b) the docstring-as-changelog call (8.2 #2); (c) the `inf` addition (8.2 #3); (d) whether the NG-2 cherry-pick onto the trial branch is clean (base = `origin/main`; no chain commit touches `neuro_foundation.py`, per the brief's assumption, which I did not re-verify against the trial branch because it is not cut).
* **F9 — a loose class name in committed NG-1 text:** the `probation_advances` docstring (`cc_ng_organism.py`) and the NG-1 changelog entry say "`universal_ingestor.py Registrar.update_probation`"; the real class is `NodeRegistrar` (`universal_ingestor.py:2112`, method `:2261`; reached through `UniversalIngestor.update_probation` `:2892`). Left as committed (rewriting NG-1 would change every hash the runs above cite); fix in any later commit that touches the file.
* **Docs-vault edits (CC-CALLOSUM-TRUTH §7/§8.13 notes, plan notes) are the ZONE MANAGER'S** and were not made.

## 11. Closing check

```
$ git -C <NG worktree> status -sb && git -C <NG worktree> log --oneline b5e4768..HEAD   # BEFORE this return file is added
## cc-laptop-sweep-probation-p552-20261002
0b26f61 NG-2 (P552): orphan sweep spares an unbound node while its probation window is open and advancing
c98d062 NG-1 (P552): probation_advances(node) defined once in cc_ng_organism; cc_update_probation consults it

$ git -C <docs worktree> status -sb && git -C <docs worktree> log --oneline 228100795..HEAD
## cc-laptop-daemon-probation-warn-20261002
905618ee daemon (P552): register the probation predicate on the CC graph at init; make a skipped probation decrement VISIBLE

$ NG origin/main still:
b5e476863cc069a29ec482959b4f9465f2ea4ccf
```

Nothing is pushed or merged: both branches are local only (`git branch -r --contains` finds neither commit on any remote ref). The return file itself is committed in the NG worktree on top of NG-2 (its hash is in the final report message: a file cannot name its own commit).
