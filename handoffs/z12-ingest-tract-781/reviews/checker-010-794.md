# checker-010 ROLE A review of #794 (`hold_on_failure`)

STATUS: COMPLETE

Reviewer: checker-010 (cross-family, report_only, ROLE A only)
Lane: ingest-tract-swallow-781 (#794 ONLY)
Dispatch: #10401
Zone manager: Z12 (session 52d39aba-db92-4bf2-b3b1-0e4c13f77d8c)
Worktree: `/home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930`
Branch: `cc-laptop-ingest-tract-781-20260930`

Pins (`git rev-parse`):
- worktree HEAD at review start: `41c03b136e9e6a72bad3ad9c9aba421267ed0ab3`
- code commit: `c625623ebcbdde894a8f9e36d5013aa76fc3a702`
- docs-only return: `41c03b136e9e6a72bad3ad9c9aba421267ed0ab3`
- plan-001: `2b8f5dc2669e9c290135630ae2ebbeca844ed41b`
- base `origin/main`: `e4ebf982b1989fd9066d610b94853bc68bf70d37`

`sha256sum` of reviewed blobs in the worktree:
- `cc_ng_organism.py` `5443f50c41229219152f6262bee73cdb44983a0de2519fd3333265d6d98ac9cd`
- `tests/test_cc_drain_hold_on_failure.py` `1c192f140a6b26e70058ca2320e2f44fc860dd982c7d91ff62eb8681f4d8cb49`
- `ng_embed.py` (unchanged, identity evidence for item 6) `f8e0592af08c16a4f7318ffd8f2ecbfd016938980ccea6b61ad0c215b2a962ee`

Safety: every probe used an explicit `/tmp/...` `tract_path`. The live tract path was never opened, read, hashed, or passed to any function. `NG_EMBED_*` names were unset before targeted runs. ROLE B is a later turn. This file is the only edit.

## 1. Semantics vs the ruling

Verdict: **PASS**

`git diff e4ebf982b1989fd9066d610b94853bc68bf70d37 c625623ebcbdde894a8f9e36d5013aa76fc3a702 -- cc_ng_organism.py` is the changelog header, a last-kwarg `hold_on_failure=False`, a docstring paragraph, `safe_offset = 0`, three `safe_offset = consumed_offset` writes on the type/source/empty `continue`s, the per-entry hold reason/class-name capture, one `logger.warning` + `break` under the flag, and `if hold_on_failure: consumed_offset = safe_offset` before the existing parse `except`. The parse-failure handler, `consumed_prefix` / `consumed_actual` / `return_consumed` logic, truncate step, and `max_entries` increment (`taken += 1` before absorb) are the same control flow as the base aside from that post-loop assignment.

Hand trace (file bytes and return). Confirmed by reading `drain_ingest_tract` at `cc_ng_organism.py:2323-2528` and by an independent `/tmp` probe against both the new module and `git show e4ebf982:cc_ng_organism.py`.

| Case | `hold_on_failure=True` file | return | `hold_on_failure` default / False |
|---|---|---|---|
| all-success | truncated to empty | absorbed=3; with `return_consumed=True`, consumed = original | same (identity) |
| filter-skips only | truncated to empty (skips advance `safe_offset`) | absorbed=0, no dual-pass calls, no hold warning | same |
| False in the middle (ok, false, ok) | prefix before the false entry removed; false + everything after kept | absorbed=1; `return_consumed=True` consumed = first frame only; original = consumed + remaining | false entry consumed; file empty; absorbed=2; no WARNING |
| raising entry | same hold as False; remainder starts at the raising frame | absorbed=1; one WARNING `reason=absorb_raised exc_type=RuntimeError` | raising entry consumed; existing DEBUG `str(exc)` still fires |
| failure at entry 1 | file bytes unchanged; `open(..., "wb")` never called (`safe_offset==0` takes the existing `if not consumed_prefix` return) | absorbed=0; consumed=`b""` | first entry consumed |
| failure at last | prefix of successes removed; last frame remains | absorbed=2 | last frame consumed too |
| `max_entries` reached before the failure (4 frames, cap=2, fail is 3rd) | first two successes removed; fail never attempted; no hold warning | absorbed=2, calls=2 | same (cap stops first) |
| `max_entries` with failure inside the cap (ok, false, ok, cap=2) | hold at the false; remainder = false+ok; taken counts the failed attempt | absorbed=1, calls=2, one WARNING | cap would consume the false as well |
| parse failure (valid frame + JSONL `{…}\n`; reader yields `bytes`) | whole file untouched, existing parse WARNING, no hold WARNING | absorbed=1 (the valid frame was applied), consumed=`b""` | same bytes/return/log |
| `return_consumed=True` on a hold | consumed is exactly the truncated prefix; `original == consumed + remaining` | pair `(absorbed, consumed_bytes)` | consumed is the walked span including failed absorbs |

Ambiguities the worker listed match the ruling: filter-skips before a hold are consumed; a `False` from `run_conversational_dual_pass` (including `graph is None or embedding is None` at `:2170-2171`) holds; a held entry counts as attempted; parse failure stays case D (re-absorb of the prefix next cycle). The function returns normally after a hold (D). `on_degraded` is absent. `cc_ng_host.py` is not in the code commit.

## 2. The default path is byte-identical

Verdict: **PASS-WITH-NOTES**

Independent proof (not the worker's test file): load BASE from `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:cc_ng_organism.py` into `/tmp/c010-794-*`, fake `ng_embed`, poison `cc_gateway_tract_path` on both modules, build frames once with the real `ng_tract` writer into tmp files, copy the same bytes into two files, call both with `hold_on_failure` unset.

20/20 independent identity cases matched on return value, resulting file bytes, dual-pass call sequence, write-open spy, and every captured log record `(level, message)` at DEBUG. Scenarios: all-success, filter-skips, mid False, mid raise, fail-first, fail-last, skip-then-fail, two consecutive fails, empty, parse-poison × `return_consumed` False/True.

The worker matrix (`test_default_is_byte_identical_to_the_base_module`, 11 names × 2 = 22) also passed in this reviewer's one run of `tests/test_cc_drain_hold_on_failure.py` (49 passed, 6.16s). Explicit `hold_on_failure=False` matched unset on the fail-middle case.

Note (non-observable): with the flag off the new function still *assigns* `safe_offset`, `hold_reason`, and `hold_exc_type`. Those names are unread when `hold_on_failure` is false (`if hold_on_failure:` is the only reader). No extra log line, exception, file rewrite, or return change showed up in the DEBUG identity comparison. The extra statements are the whole of the default-path difference.

## 3. Other callers

Verdict: **PASS**

`git grep drain_ingest_tract(` at NeuroGraph `e4ebf982` and docs `origin/main` `scripts/`:

Production:
- docs `origin/main` `scripts/cc-ng-daemon.py:1924-1926`: three positionals + `return_consumed=True`. New last kwarg stays default False. No positional shift.
- NeuroGraph `e4ebf982` `cc_ng_host.py:1526`: three positionals only (`graph, vector_db, conv_state`). Default False. Path B / VPS / Syl's process unchanged.

Tests at `e4ebf982` (`tests/test_cc_callosum_leg1.py`, `test_cc_deposit_step.py:509-511`, `test_cc_dual_pass.py`, `test_cc_refeed.py:118`): extra arguments are keywords (`tract_path=`, `return_consumed=`, `max_entries=`). `inspect.signature(…).bind(object(), None, {})` and `bind(..., return_consumed=True)` succeed on the new signature.

No caller passes a 4th positional that would land on `hold_on_failure`. The code commit does not touch `cc_ng_host.py`.

## 4. The untested cases

Verdict: **PASS-WITH-NOTES**

The worker file does not contain a trailing-partial-frame test, a two-consecutive-failure test, or a skip-then-fail case beyond `skips_around_fail`. This reviewer ran those on tmp files.

Trailing partial (complete frames + half of a third BTF frame): `TractReader` (`ng-tract-rs/src/lib.rs:272-280`) yields the complete entries then Python `None` (`StopIteration`) without advancing `position()` into the tail. Final pos = end of last complete frame. Default new, default base, explicit False, and hold True all left the partial tail as the remaining file bytes (33-byte half-frame; also a 2-byte `BT` sub-envelope tail). Hold True on all-success+partial emitted no hold warning. A False complete entry *before* a partial tail: hold keeps `false_frame + partial`; default consumes the false frame and keeps the partial. Partial tail preservation matches today.

Skip then fail (skip_source, ok, false, ok): hold consumed skip+ok (`return_consumed` prefix), kept false+ok, one WARNING.

Two consecutive failures: hold stopped at the first False (calls=2, one WARNING); the second False was never attempted; remainder = fail1+fail2+ok.

No reporter exists; a raising reporter is not applicable.

Note: these cases are reviewer-probed, not added to the suite (report_only).

## 5. Loud signal and leak-safety

Verdict: **PASS**

Hold WARNING is inside `if hold_on_failure and hold_reason is not None` (`cc_ng_organism.py:2455-2463`). Format string interpolates only `hold_reason` (`absorb_returned_false` / `absorb_raised`) and `hold_exc_type` (`type(exc).__name__` or `"-"`). The loop `break`s, so at most one WARNING per call.

Independent probe of a `RuntimeError(SECRET-EXC-TEXT + entry text)`: WARNING text was

`CC ingest-tract hold: reason=absorb_raised exc_type=RuntimeError -- failed entry and everything after it kept in the tract for the next cycle`

No `SECRET-EXC-TEXT`, no `ENTRYTEXT`, no tmp path, no `.tract`. Default / cap-before-failure / parse-failure paths emitted zero hold WARNINGs. Parse-failure still emits the existing parse WARNING (includes `str(exc)`; unchanged handler). The pre-existing per-entry DEBUG line still interpolates `%s, exc` (SECRET appeared in DEBUG only, as on the base).

Worker suite WARNING assertions also passed on this run.

## 6. Head-of-line / S4 implications

Verdict: **PASS-WITH-NOTES**

S4 FIRST-DRAIN RULE (Executive Packet 386 / `kiss-pith-zero-replay-cards.md`): the daemon does not drain the tract on first start until all of (1) laptop TID up+verified, (2) local embed healthy (no remote, no hash), (3) #794 landed on the trial stack, (4) drain capped and paced (#792). #794 as built is condition (3) only.

| Worker-listed implication | Ruling for the trial stack |
|---|---|
| Head-of-line blocking | Accepted cost of Chief ruling (E). A permanently-unabsorbable entry blocks everything behind it and retries every autosave cycle. Not a #794 merge blocker. An unfixable held entry still has no exit (parked frame policy / Q1, Josh's). |
| ~1,440 unrate-limited WARNING lines/day | Acceptable as the loud signal for a trial once a drain is *allowed*. The line is inside the organism; a later daemon rate limiter cannot suppress it. Not a #794 merge blocker. |
| 70 MB backlog still draining uncapped once a hold clears | **Must be closed before the first drain.** This is S4 condition (4) / #792. `max_entries` still defaults to 0. Neither production caller passes a cap. #794 does not add one. NG merge of this slice is a gate, not permission to start the daemon against the live tract. |
| Hash-embedding path | See reconciliation below. |

**Reconcile hash-fallback vs fail-closed.** The worker return §5.2 says a remote-embed outage returns hash vectors so nothing holds. That describes the 2026-09-23 daemon log (811 HF HTTP 401 → hash fallbacks) and an older binary. It does **not** describe `ng_embed.py` at `e4ebf982`.

Evidence on this pin:
- `ng_embed.py` changelog 2026-09-21: "Fail-closed embed: no hash fallback"; `class EmbeddingUnavailableError`; "There is no hash fallback and no env var that re-enables one."
- `embed()` (`:421-439` / module wrapper `:1285-1294`) raises `EmbeddingUnavailableError("embedding model unavailable")` when `_ensure_model()` is false.
- Zero `def _hash_embed` / `def hash_embed`; zero `Falling back to hash` in the file.
- `sha256sum` `f8e0592af08c16a4f7318ffd8f2ecbfd016938980ccea6b61ad0c215b2a962ee` is identical for worktree `ng_embed.py` and `/home/josh/NeuroGraph/ng_embed.py`. Primary NeuroGraph HEAD is `e4ebf982b1989fd9066d610b94853bc68bf70d37`.

`_apply_gateway_experience` (`:2316-2320`) does `from ng_embed import embed` then `embed(entry.content)` *before* `run_conversational_dual_pass`. A raise there is the drain's per-entry `except` (`:2451-2454`) → `hold_reason = "absorb_raised"` → hold under the flag. A None embedding would make dual-pass return False (`:2170-2171`) and also hold. Checker-006's fail-closed reading of this pin is the one that matches the source.

**Which `ng_embed` the daemon imports.** `scripts/cc-ng-daemon.py:541-543` (docs `origin/main` `79eb80a50bb3d09d69686fae1462fa2b98bbbdc3`, same insert in the primary file) does `sys.path.insert(0, os.path.expanduser('~/NeuroGraph'))`. A subsequent `from ng_embed import embed` therefore resolves to `/home/josh/NeuroGraph/ng_embed.py` as that file exists on disk today (fail-closed blob above), unless something else is already at `sys.path[0]` *after* that insert or a running process has an older module in `sys.modules`.

Established: source identity of `~/NeuroGraph/ng_embed.py` with this worktree / `e4ebf982`; daemon import order as written; hold would engage on `EmbeddingUnavailableError` if that source is the one called. Not established: a live daemon process's `sys.modules` (this review did not start, attach to, or restart any daemon); embedder health at runtime; a `CC_NG_PYTHONPATH` shadow in some other launch wrapper.

## 7. Test quality

Verdict: **PASS-WITH-NOTES**

This reviewer ran `tests/test_cc_drain_hold_on_failure.py` **once** under scratch HOME. P379/#770 preamble (terminal reporter):

```
[P379/#770] cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930/cc_ng_organism.py
[P379/#770] worktree root           = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930
[P379/#770] ng_tract.__file__       = /home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py (REAL installed writer/reader; tract bytes are tmp-file only)
[P379/#770] ng_embed                = FAKED (types.ModuleType injected per test; the real one is never imported)
[P379/#770] neurograph_rpc in sys.modules: False
[P379/#770] other NG modules in sys.modules: []
[P379/#770] HOME = /tmp/c010-home-N9yWOE
[P379/#770] worktree check: PASSED (cc_ng_organism is under the worktree)
49 passed in 6.16s
```

`US=/home/josh/.local/lib/python3.12/site-packages`; `NG_EMBED_*` names remaining `[]`; `PYTHONPATH=$US`; `-p no:cacheprovider`. Full NG suite not run.

Hold tests vs base (worker reasoned TypeError, did not execute; this review executed):
- BASE `drain_ingest_tract(..., hold_on_failure=True)` → `TypeError: drain_ingest_tract() got an unexpected keyword argument 'hold_on_failure'`.
- Wrapper that strips the kwarg and calls BASE: fail-middle absorbed=2, file emptied, 3 dual-pass calls. The hold assertions would fail on behavior, not only on the missing parameter.

Safety guards are real code:
- Import-time worktree prefix check raised nothing; `__file__` was the worktree path (printed).
- `env` fixture poisons `cc_gateway_tract_path` with `AssertionError` and sets `CC_GATEWAY_TRACT_PATH` to a never-created tmp path (covers the loaded BASE module too).
- `_check_safe` uses `pwd.getpwuid` for the real home, so a scratch `HOME` cannot mask `~/.claude` / `~/.et_modules`; it also rejects `data/` and `plugins/neurograph` substrings.
- Every `_run` / drain call in the file passes an explicit tmp `tract_path`.

Notes:
- No test *omits* `tract_path` to trip the poison on purpose; it is a fixture net, not an asserted negative test.
- `_check_safe` also accepts a path that merely contains the substring `pytest` even if it is outside `/tmp`; this run's paths were under `/tmp`.
- `drain_ingest_tract` still does `from ng_embed import embed as ng_embed_fn` as an availability import (`:2399-2400`). Tests inject a fake `sys.modules['ng_embed']` in `env`. A drain call without that fixture would import the real module (model load is lazy; still the wrong import for this suite).
- Worker did not run `tests/test_cc_dual_pass.py` / `test_cc_refeed.py` / `test_cc_deposit_step.py` (real graph/embedder or primary daemon path). This review also skipped them; byte-identity + call-shape bind are the compensating evidence.

Nothing in the targeted run pointed a path at `~/.claude`, `~/.et_modules`, or a `data/` checkpoint.

## 8. Verdict

**Overall: PASS-WITH-NOTES**

Per-item:
1. Semantics vs the ruling: **PASS**
2. The default path is byte-identical: **PASS-WITH-NOTES**
3. Other callers: **PASS**
4. The untested cases: **PASS-WITH-NOTES**
5. Loud signal and leak-safety: **PASS**
6. Head-of-line / S4 implications: **PASS-WITH-NOTES**
7. Test quality: **PASS-WITH-NOTES**

The organism slice matches Chief-003 A–E: last-kwarg opt-in, default observably identical to `e4ebf982`, hold truncates only the prefix before the first False/raise, one class-name WARNING, normal return. It is fit to merge to NeuroGraph *as the #794 gate*. It does not by itself make a first drain of the live tract lawful.

Numbered corrections (before merge of this organism slice):
1. None. No code correction is required on `c625623ebcbdde894a8f9e36d5013aa76fc3a702` for ROLE A.

Numbered notes (do not block this NG merge; do block or shape later slices):
1. Default path still executes unused local assignments (`safe_offset` / `hold_reason` / `hold_exc_type`). Observables matched the base at DEBUG.
2. Trailing-partial-frame and two-consecutive-failure are absent from `tests/test_cc_drain_hold_on_failure.py`. Reviewer-probed; behavior matched the ruling.
3. Worker return §5.2 (hash vectors so nothing holds) is false for `ng_embed.py` at `e4ebf982` / `~/NeuroGraph`. Fail-closed raise holds. Correct that claim on any later daemon-slice brief.
4. S4 first drain still requires (1)(2)(4) in addition to this merge: laptop TID verified, local embed healthy, drain capped+paced. Do not start `cc-ng-daemon` against the live tract on the strength of #794 alone.
5. Head-of-line + unrate-limited organism WARNING are accepted trial costs of (E); Q1 (unframable/unabsorbable entry) remains Josh's.

Numbered "not verified":
1. Live daemon `sys.modules` / in-process `ng_embed` object (no process was started or attached).
2. Runtime embedder health (local model vs remote vs error).
3. `tests/test_cc_dual_pass.py`, `tests/test_cc_refeed.py`, `tests/test_cc_deposit_step.py`, and `tests/test_cc_callosum_leg1.py` were not re-run here.
4. Import-time filesystem side effects of `import cc_ng_organism` beyond the P379 print and the targeted drain calls.
5. Live tract inode/mtime/size (this review did not `stat` it).
6. ROLE B (law enforcer) — separate later turn.

Independent probe P379/#770 (scratch HOME `/tmp/c010-home-4jwFNt`; fake `ng_embed` already in `sys.modules` as a `types.ModuleType`):

```
[P379/#770] cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930/cc_ng_organism.py
[P379/#770] worktree root           = /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930
[P379/#770] ng_tract.__file__       = /home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py
[P379/#770] ng_embed                = <module 'ng_embed'>
[P379/#770] neurograph_rpc in sys.modules: False
[P379/#770] other NG modules in sys.modules: ['ng_embed']
[P379/#770] HOME = /tmp/c010-home-4jwFNt
[P379/#770] NG_EMBED names set: []
[P379/#770] module cc_ng_organism -> /home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930/cc_ng_organism.py
[P379/#770] module ng_tract -> /home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py
[P379/#770] worktree check: PASSED
```

Probe result: all independent traces passed (identity 20/20, hold traces, partial tail, skip-then-fail, two-fails, leak check, base TypeError + tolerated consume-on-fail).
