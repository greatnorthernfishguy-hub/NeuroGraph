# build-794 — return: opt-in `hold_on_failure` on `drain_ingest_tract` (lane `ingest-tract-swallow-781`, #794 ONLY)

```
# ---- Changelog ----
# [2026-09-30] Claude Code (Sonnet 5.5), Z12 worker — build-794 return
# What: return for dispatch #10349 (Chief-003 ruling / Exec P386): hold-on-failure for the ingest-tract drain.
# Why: #794 — an entry whose absorb returned False/raised was truncated out of the tract and lost (LAW 7).
# How: NeuroGraph code commit c625623ebcbdde894a8f9e36d5013aa76fc3a702 (this doc is the separate docs-only commit).
# -------------------
```

Worktree `/home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930`, branch `cc-laptop-ingest-tract-781-20260930`,
base `e4ebf982b1989fd9066d610b94853bc68bf70d37`, plan commit `2b8f5dc2669e9c290135630ae2ebbeca844ed41b`.
**Code commit: `c625623ebcbdde894a8f9e36d5013aa76fc3a702`** (`cc_ng_organism.py` + `tests/test_cc_drain_hold_on_failure.py` only).
Nothing was rebased, force-pushed, merged, restarted, wired or connected. No PR.

## 1. What changed (file:line at tip `c625623`)

`cc_ng_organism.py` (not protected, not vendored):
- Signature `:2323-2325`:
  ```python
  def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                          return_consumed: bool = False, max_entries: int = 0,
                          hold_on_failure: bool = False):
  ```
  `hold_on_failure` is the **last** keyword; the earlier parameter order is unchanged (pinned by a test).
- Docstring paragraph `:2354-2369` documents the flag; changelog header entry (2026-09-30, #794, lane, Chief-003 / Exec P386) at the top of the file (`:6`).
- `safe_offset = 0` `:2424`: offset just past the last entry that was **absorbed (True)** or **legitimately filter-skipped**
  (wrong type / wrong source / empty text). Each of the three filter `continue`s now also sets `safe_offset = consumed_offset`;
  a True absorb sets it too.
- Per entry `:2443-2464`: `hold_reason` is `absorb_returned_false` (dual pass returned False) or `absorb_raised`
  (exception; `hold_exc_type = type(exc).__name__`). The existing per-entry `logger.debug(... %s, exc)` is unchanged.
  `if hold_on_failure and hold_reason is not None:` (`:2455`) emits **one** `logger.warning` (`:2459`) and `break`s.
- After the loop `:2467-2469`: `if hold_on_failure: consumed_offset = safe_offset`. The existing code then truncates
  `data[:consumed_offset]`, so under the flag only the prefix before the held entry is removed; `safe_offset == 0` takes the
  **existing** early return (`if not consumed_prefix`, `:2503`), i.e. the file is not rewritten at all.
- **Untouched:** the parse-failure handler (`:2470-2476`, the WARNING and early return), `consumed_prefix`/`consumed_actual`/`return_consumed`
  logic, the truncate step, `max_entries` accounting (`taken` still counts entries *attempted*; the hold breaks before the cap check),
  `cc_ng_host.py` (Path B), every vendored/protected file.

Warning text (hardcoded; fixed reason code + exception **class name** only, never `str(exc)`, entry text, a path or a secret):
```
CC ingest-tract hold: reason=<absorb_returned_false|absorb_raised> exc_type=<ClassName|-> -- failed entry and everything after it kept in the tract for the next cycle
```
It is reachable **only** when `hold_on_failure=True`, and at most once per call (the hold breaks the loop).

## 2. Semantics as built, with the ambiguities I hit and how I resolved them
1. **Filter-skips before the held entry are consumed; entries (skips included) after it stay** — as ruled ("legitimately filter-skipped" advances the offset). Test `skips_around_fail`.
2. **`run_conversational_dual_pass` returns False in more places than a "real" failure** (`:2146-2147`: `graph is None or embedding is None`). Under the flag those hold too. That is the desired direction (a None embedding = the embedder failed) and consistent with the brief ("False or an exception IS a failure").
3. **`max_entries`**: a held entry counts as attempted; the hold breaks the loop. If the cap is reached *before* the failing entry, nothing is attempted, nothing holds, no warning (`cap_fail_after`).
4. **Trailing partial frame** (reader returns None mid-frame): `safe_offset` equals the offset after the last complete entry, same as `consumed_offset` today, so the partial tail stays exactly as before. Reasoned from the code and the reader source; **not exercised by a test**.
5. **Parse failure with the flag on** is byte-for-byte the default behaviour (whole file untouched, the existing WARNING, no hold warning). Entries absorbed before the poison frame are still re-absorbed each cycle — the plan's case D, **not** fixed here.
6. **Truncate failure / file changed underneath** keep their existing behaviour under the flag (plan cases E). Not touched.

## 3. Tests: `tests/test_cc_drain_hold_on_failure.py` (49 tests)

**Fakes only.** `ng_tract` is the **real installed** writer/reader (`~/.local/lib/python3.12/site-packages/ng_tract`, imported via `PYTHONPATH=$US`); tract bytes are built with `deposit_experience`/`deposit_topology` into tmp files. `ng_embed` is a **faked module** injected into `sys.modules` (the real one is never imported: no model, no network). `run_conversational_dual_pass` is monkeypatched per entry text (`OK…` True, `FALSE…` False, `RAISE…` raises `RuntimeError("SECRET-EXC-TEXT …")`); graph/vector_db/state are plain fakes.

**Live-tract safety (asserted, not just intended).** The `env` fixture (a) `_check_safe()` asserts every tract path is under a pytest tmp dir and **not** under the real `~/.claude`, `~/.et_modules`, any `data/` dir or a `plugins/neurograph` path (the real home comes from `pwd`, not `$HOME`, so a scratch `HOME` cannot mask it); (b) monkeypatches `cc_ng_organism.cc_gateway_tract_path` to raise `AssertionError`, so a default-path call dies **before any `open()`**; (c) points `CC_GATEWAY_TRACT_PATH` at a never-created tmp path (covers the loaded base module too). Every drain call passes an explicit tmp `tract_path`. **I never opened, read, copied or `stat`ed the live `turns.tract` this turn.**

**Cases** (each with `return_consumed` False *and* True): all-success · filter-skips-only · False in the middle · raising entry · failure at entry 1 (file untouched **and never reopened for writing**, via an `open` spy) · failure at the last entry · `max_entries` with the failure inside the cap · cap reached before the failure · skips around a failure · parse failure (poison `{…}\n` line, reader precondition asserted: it yields `bytes`) · empty file · retry-next-call (the held entry and the one after it land once the cause is fixed; a still-failing second call re-holds and changes nothing) · signature/order guard · existing caller shapes still bind. With `return_consumed=True` every hold test asserts `consumed` is exactly the bytes removed and `original == consumed + remaining`. The WARNING tests assert: exactly one when the hold engages; none with the default; none for cap-before-failure; contains the reason code and class name; contains **no** `str(exc)` (`SECRET-EXC-TEXT`), entry text (`ENTRYTEXT`), tmp path or `.tract`.

**Why each hold test FAILS on the base:** the base `drain_ingest_tract` has no `hold_on_failure` parameter, so each call raises `TypeError: unexpected keyword argument` (the signature test raises `KeyError`). **I reasoned this; I did not execute the file against the base** (the brief allows one run of the file; see the run log below). Two groups deliberately pass on the base and are labelled as such: `test_default_still_consumes_failed_entries…` (characterisation of today's loss) and the byte-identity matrix / call-shape guards (preservation).

**Byte-identity proof for default callers.** `test_default_is_byte_identical_to_the_base_module` (11 scenarios × `return_consumed` False/True = 22 cases) loads the BASE module from `git show e4ebf982…:cc_ng_organism.py` into a tmp dir (registered in `sys.modules` under a unique name), builds each scenario's frames **once**, writes the *identical bytes* for both modules, calls both with `hold_on_failure` **unset**, and asserts equal: return value, resulting file bytes, dual-pass call sequence, whether/how the file was reopened for writing, and every captured log record `(level, message)` at DEBUG. It covers success, filters, mid/first/last failure, raise, cap, parse failure and empty file. The proof requires `git show` to work; if it cannot obtain the base it **fails** rather than skips.

### Run log (honest, three executions of the new file)
| Run | Result | Cause |
|---|---|---|
| 1 | 27 passed, 22 failed | **my harness**: the base module uses dataclasses, which resolve their module via `sys.modules`; I had not registered it. Also the P379 write went to fd 2 and was swallowed by pytest capture. |
| 2 | 30 passed, 19 failed | **my harness**: `deposit_*` *appends*, so re-building a frame file doubled the frames, and rebuilt frames carry new timestamps so the two modules were not fed identical bytes. |
| 3 (final) | **49 passed in 12.00s**, exit 0 | fixed: build once, feed the same bytes, remove stale frame files. |

Plus one `--collect-only` (no test bodies) and one probe of the installed `ng_tract` API in a scratch dir. No production code changed between runs; only the test file. Each run used `HOME=$(mktemp -d) PYTHONPATH=$US PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q … -p no:cacheprovider`, `US` taken before overriding `HOME`.

### P379 / #770 preamble as printed by run 3 (via the terminal reporter; the import-time guard raises if the path is wrong)
```
[P379/#770] cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930/cc_ng_organism.py
[P379/#770] worktree root           = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930
[P379/#770] ng_tract.__file__       = /home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py (REAL installed writer/reader; tract bytes are tmp-file only)
[P379/#770] ng_embed                = FAKED (types.ModuleType injected per test; the real one is never imported)
[P379/#770] neurograph_rpc in sys.modules: False
[P379/#770] other NG modules in sys.modules: []
[P379/#770] HOME = /tmp/tmp.AXWDDTQdGJ
[P379/#770] worktree check: PASSED (cc_ng_organism is under the worktree)
.................................................                        [100%]
49 passed in 12.00s
```

### Existing drain tests (optional run, ONE run)
`tests/test_cc_callosum_leg1.py -k drain_ingest_tract` → **10 passed, 6 deselected** (unchanged file). It loads the worktree module (`sys.path.insert(0, worktree)`; the wrapper printed the same guard and passed). Its `cc_ng` fixture is a pure fake and every drain call there passes an explicit `tract_path`.
**Not run, and why:** `tests/test_cc_dual_pass.py` and `tests/test_cc_refeed.py` build a real `NeuroGraphMemory` (graph load) and use the real embedder; `tests/test_cc_deposit_step.py` loads the daemon from the **primary** `~/docs/scripts/cc-ng-daemon.py` in some tests (`:336`) and its drain test uses a fixture I did not confirm to be fake. The byte-identity matrix is the compensating proof for those call shapes.

## 4. What the daemon slice must pass, and what else it needs (NOT built here)
- Pass `hold_on_failure=True` at the daemon's call `D:1924-1926` (`scripts/cc-ng-daemon.py`, docs `origin/main`); keep `return_consumed=True`.
- **NG merges first.** An older organism raises `TypeError: unexpected keyword argument`, which the loop's catch-all (`D:1931-1932`, `debug`) swallows — silently killing all five steps every cycle (the plan's §5.3 item 4 hazard).
- The held-entry signal is **one WARNING line per cycle** and nothing else: `status` (`D:1022-1055`) still shows nothing about the drain. Getting it into `stats` needs the parked `on_degraded` slice.
- `cc_ng_host.py:1526` (VPS half, Path B, Syl's process) keeps default behaviour and stays lossy; Josh-gated, not touched.

## 5. Findings and adjacent items (listed, not fixed)
1. **Head-of-line blocking is the accepted cost of (E).** An entry that fails *every* time (e.g. text that can never be embedded, or a persistent embed outage) now blocks every entry behind it, the tract grows, and the same WARNING repeats **every autosave cycle (~60 s, ~1,440 lines/day)**. This WARNING is emitted inside the organism and is **not rate-limited**, so the daemon's future rate limiter cannot suppress it. Worth a Chief decision together with the parked frame policy (an unfixable held entry has no exit today).
2. **The hold does not cover degraded embeddings:** during a remote-embed outage (the 09-23 run logged 811 HF HTTP 401 → hash fallbacks) `ng_embed` returned *hash vectors*, not None, so the dual pass reports True and **nothing holds**. The hold protects against False/raise only (plan A3, vendored `ng_embed`, unchanged).
3. **A 70 MB backlog still drains uncapped** the first time a hold clears (the parked cap, plan Q3) — under `_concurrent_lock`.
4. The truncate-window race without `flock` (plan A1) is unchanged and unfixed.
5. Entries absorbed before a poison frame are still re-absorbed each cycle (plan case D); unchanged.
6. Test-suite note: an `open` spy replaces `builtins.open` for the duration of one drain call (the same technique as `tests/test_cc_callosum_leg1.py:392-419`).

## 6. Closing state
`git rev-parse HEAD` of the code commit: `c625623ebcbdde894a8f9e36d5013aa76fc3a702`. This return is a separate docs-only commit on the same branch; its own hash is reported in the dispatch report (a file cannot contain the hash of the commit that adds it).
