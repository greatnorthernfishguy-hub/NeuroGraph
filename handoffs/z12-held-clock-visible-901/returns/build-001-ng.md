<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane held-clock-visible-901, dispatch #13220) — return build-001-ng (NG half): #915
#   What: the builder return for the NG half of #915 (Exec Packet 495): three swallows in cc_ng_organism.py made loud, class-name only.
#   Why: assignment handoffs/z12-silent-failure-lanes-20260930/assignments/build-901-held-clock-visible.md PART C item 2 (docs commit 08b9ad42).
#   How: ONE code commit 78810c1b + this docs-only return. Local commits only; no upstream; nothing merged. The docs half is
#        handoffs/z12-held-clock-visible-901/returns/build-001.md in the docs worktree.
# -------------------
-->

# Return build-001-ng — #915 (NG half): three swallows made loud, class-name only

**Status: BUILT, returned for review (not self-accepted).** FULL PAIR still owed; it spans both repos (docs half: `build-001.md`).

## Hashes (real `git rev-parse`)
| repo / branch | base | code commit |
|---|---|---|
| NG `cc-laptop-swallows-915-20261001` (`/home/josh/NeuroGraph-worktrees/z12-swallows-915-20261001`) | `895a809a60ab86afacd8ef193b95c4ab39f6893f` (the #904 NG tip) | `78810c1b158fb64ce4e3f65dd51b2a7ecb61e9d9` |
| docs `cc-laptop-daemon-visibility-901-20261001` | `15cf82783bc84777890d924b8a0ead332aa68206` | `b49cab56b7bb7ea1a3e8e9545fc43c3549afdf07` |

The hash of THIS return commit is reported in the closing message. One code commit per repo; no upstream; nothing pushed or merged.

## READ first (verified, not assumed)
The worktree already existed, clean, at the base (`git status -sb`, `git rev-parse HEAD`). I read, in the worktree: `surface_wants_for_graph` (`:1215` at base), `surface_wants` (`:1611`), `_cc_deposit_memory_node` (`:1938`), `_cc_report_want_synapse_failures` (the #904 reporter I mirror), `tests/test_cc_capture_mutations_423.py` (the AST-extraction pin: `functions(...)` execs the named functions into a namespace holding only `logger`, `time`, a few constants), `tests/test_cc_bind_atomic_904.py` (the want-world fixtures), `tests/test_cc_want_bounds.py`. Files touched: `cc_ng_organism.py` and ONE new test file `tests/test_cc_swallows_915.py`. `ng_embed.py` and every vendored / protected file untouched.

## What changed (`cc_ng_organism.py` at `78810c1b`)
- **(a) `_cc_deposit_memory_node`** (`:2021`): `logger.warning("CC recall insert failed: reason=recall_insert_failed exc_type=%s", type(exc).__name__)`; the `raise` is unchanged, there is still no rollback. The node KIND is NOT logged (I would have had to take it from a substring of the id; the brief says nothing then). The function body is otherwise byte-identical and uses only `logger` and builtins, so `tests/test_cc_capture_mutations_423.py`'s AST extraction still works: **that file passes UNMODIFIED (10 passed)**.
- **(b) `surface_wants`** (`:1710`, `:1724`): `graph.create_node(...)` is guarded PER WANT. A raise is counted (`create_failed`, distinct class names), the loop `continue`s to the next want, and ONE WARNING per call is emitted by the new `_cc_report_want_create_failures` (`:1219`): `CC want create FAILED: reason=want_create_failed fn=<name> attempted=<n> failed=<n> exc_types=<Class,...>`: fixed code, function name, counts, CLASS names only, never `str(exc)`, want text or an id. Counter: module-level `_CC_WANT_CREATE_FAILURES` (`:1216`, per function, in-process).
- **(c) `surface_wants_for_graph`** (`:1325`, `:1328`): its outer `except Exception: logger.debug("Failed to create want node: %s", exc)` is now the same counted, class-only WARNING (it was already non-fatal per want; it was silent and printed `str(exc)`).
- **H-1:** nothing about the 182 `cc_authored` wants or the constitutional node is deleted, re-tagged, edited or de-flagged. A want whose create raised is NOT created and is NOT rolled back or half-removed; because its id is not in the graph it is retried on the next pulse. The success path is golden-identical (return value, node metadata, the one 0.3 source->want synapse, no record above DEBUG): asserted. `generate_emergent_want` is untouched (a test compares its source to the base).

## Tests (REAL `neuro_foundation.Graph` + REAL `SimpleVectorDB`; only `create_node` / `insert` wrapped to raise for chosen ids; no model, network or live graph)
`tests/test_cc_swallows_915.py`: 16 tests (the want tests parametrized over both functions): one raising create does not abort the loop; ONE counted WARNING with the code, counts and class only; every want failing reports once per call with the total; a failed want is retried next pulse; H-1 (pre-existing authored want + constitutional node byte-identical after a failing pulse); golden success; the #904 seed-synapse swallow unchanged; the deposit warning is class-only AND still re-raises, node retained (no fabricated rollback); deposit success unchanged and quiet; `generate_emergent_want` untouched. No `str(exc)`, secret, tree words, want text or id appears in ANY record at ANY level (all `caplog` records and their args).

## Runs (scratch `HOME=/tmp/z12h`, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `-p no:cacheprovider`; one run per file; the new file printed the P379/#770 preamble with the resolved paths)
| file | result on the branch |
|---|---|
| NEW `tests/test_cc_swallows_915.py` | **16 passed** |
| same, against the BASE `895a809a` scratch copy of `cc_ng_organism.py` (`Z12_915_ORG_UNDER_TEST`) | **7 failed, 9 passed** (`surface_wants` aborts on the first raise: 4 tests; `surface_wants_for_graph` logs DEBUG + `str(exc)`, no WARNING: 2 tests; the deposit warning prints `str(exc)`: 1). The 9 passes are the golden / H-1 / re-raise / untouched-pin tests |
| `tests/test_cc_capture_mutations_423.py` (UNMODIFIED) | 10 passed |
| `tests/test_cc_want_bounds.py` | 8 passed |
| `tests/test_cc_bind_atomic_904.py` | 40 passed, 4 skipped (the 4 daemon-loop tests are skipped when `Z12_904_DAEMON_UNDER_TEST` is unset: I did not run them) |
| `tests/test_cc_topology_capture_423.py`, `tests/test_cc_host_stop_door.py` | 4, 4 passed |
| `tests/test_cc_dual_pass.py` | 12 failed, 9 passed: **identical on the BASE** (untouched `git archive 895a809a` export in `/tmp/z12h/ngbase`): the failures are `EmbeddingUnavailableError: tokenizer load failed` (offline model) |
| `tests/test_cc_topology_callosum.py` | 1 failed (`test_poincare_dir_is_rederived_locally_not_transmitted`), 33 passed: identical on the base |
| `tests/test_conversational_recall.py` | 2 failed, 17 passed: identical on the base |
| `tests/test_cc_pattern_completion_recall.py` | 1 failed (offline tokenizer), 7 passed: identical on the base |
| `tests/test_tonic_bridge.py` | 1 failed (`test_handle_after_turn_skips_outbound_check_for_autonomous_source`), 28 passed: identical on the base |
| `tests/test_cc_deposit_step.py` | **TIMED OUT at 280 s** on the branch (it did not finish: probably waiting on the offline embedder); NOT compared on the base, so I make NO claim about it (see Not verified) |

## Mutations (REAL mutants on SCRATCH copies of `cc_ng_organism.py` under `/tmp`, each replacement matching exactly once; the real file never written) — 8 of 8 KILLED
N1 `surface_wants_for_graph` swallow silent again (no report call): 2 failed. N2 `surface_wants` guard re-raises, i.e. the loop aborts on the first raise again: 4 failed. N3 deposit warning back to `str(exc)`: 1 failed. N4 `str(exc)` smuggled into the `surface_wants` report: 2 failed. N5 failure counter never bumped: 2 failed. N6 `surface_wants` guard swallows without reporting: 2 failed. N7 the deposit no longer re-raises: 1 failed. N8 the graph-fn swallow back at DEBUG: 2 failed. Survivors: none. (The runner is a throwaway in `/tmp/z12h/ng_mutants.py`; the brief allows only ONE new file in this worktree, so it is not committed. The `(old -> new)` pairs are the N1-N8 lines above, each one exact string at the sites named in "What changed".)

## Ripple / dependencies checked
- Callers of the changed functions: `cc_ng_host.py:698` (`surface_wants_for_graph`), `cc_ng_host.py:1519-1531` (`surface_wants`), `cc_topology_merge.py:281,450` (`_cc_deposit_memory_node`). Only failure paths changed: a `surface_wants` caller that previously saw an exception now gets a normal return (the wants it could create, in order), and log text changed. `neurograph_rpc.py` (Syl's) has its own `_surface_wants`; it does not import these.
- **Not touched, same class (punchlist):** `cc_ng_host.py:1534` `logger.debug("ingest-tract/probation/surface_wants/emergent_want failed (non-fatal): %s", exc)`: the second host has the SAME invisible autosave swallow with `str(exc)` that the daemon half fixes. Outside this worktree's allowed files; not changed.

## Consequences to state (not decided)
- **Volume:** a want that keeps failing to create is retried every pulse (its id never lands in the graph), so it re-reports ONE WARNING per call, i.e. per autosave pulse, while it keeps failing. The existing #904 seed-synapse reporter does not have this because a created want is skipped on later pulses. If that is too chatty, a rate limit (like the daemon's `RECALL_WARN_INTERVAL_S`) is a follow-up; there is no existing interval helper in `cc_ng_organism.py`, so I did not invent one.
- `_CC_WANT_CREATE_FAILURES` is in-process only and not surfaced by any `status` verb yet (the daemon's `status` shows its own `STATE.stats`); wiring it there would be a daemon + NG change I did not make.

## Not verified
- The 4 daemon-loop tests in `test_cc_bind_atomic_904.py` (skipped without `Z12_904_DAEMON_UNDER_TEST`), and `test_cc_deposit_step.py` (timed out at 280 s, not compared on the base).
- Nothing was run against the real embedder, a live graph, the live tract or the live daemon.
- **Load:** one batch of the NG regression runs above (the `test_cc_deposit_step.py`, `test_cc_dual_pass.py`, ... pass) STARTED at one-minute load 7.27, above the brief's `< 6`; the load was from other processes. The base-comparison runs waited until it was < 6 (5.3-5.7). The file's results are the same on the base, so the load did not change an outcome, but the rule was exceeded once and I am saying so.
