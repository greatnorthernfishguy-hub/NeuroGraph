# plan-001 — lane `ingest-tract-swallow-781` (#781 / B3, HIGH): PLAN ONLY

```
# ---- Changelog ----
# [2026-09-30] Claude Code (Sonnet 5.5) — plan-001 for row #781 (ingest-tract swallow)
# What: read-only plan for the drain_ingest_tract / _autosave_loop silent-failure family.
# Why: Chief-003 / Exec P381 item 1; standing policy P370 (silent failures fixed on sight, LAW 4).
# How: code read at NG base e4ebf982, docs scripts/cc-ng-daemon.py at origin/main (c5b7552c == c622e16c
#      for that file), miniTID + ng-tract-rs source, and the STOPPED CC daemon's own log (grep/stat only).
#      No code edited, no test run, nothing started, no live data/checkpoint/tract CONTENT read.
# -------------------
```

Zone manager Z12 (session `52d39aba-…`), dispatch #10275. Worker seat. Worktree
`/home/josh/NeuroGraph-worktrees/z12-ingest-tract-781-20260930`, branch `cc-laptop-ingest-tract-781-20260930`,
base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37` (verified: `HEAD == origin/main`).
NeuroGraph `CLAUDE.md` read in full: `cc_ng_organism.py` and `cc_ng_host.py` are **not** protected
(§2 lists `neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`) and **not**
vendored (LAW 2 list). Syl's Law is not engaged by this plan.

**Evidence tags used below:** **[V]** read at the cited ref; **[L]** read from the stopped CC daemon's own log/stat
(read-only); **[I]** inferred from [V] by reasoning, not executed; **[U]** not verified. Refs:
NG = `e4ebf982` unless stated; **D** = docs `scripts/cc-ng-daemon.py` at `origin/main` (I read it at `c5b7552c`;
`origin/main` moved to `c622e16c` while I worked and the file is byte-identical at both, sha256 prefix
`91e38308f521b3c8`); at the #756 branch tip `27d362f5` the same code sits ~+187 lines (loop `:2077`, root INFO `:952`).

---

## 0. Summary (read this first)

1. **The claim is right in impact and partly wrong in mechanism.** Absorption from the miniTID tract *can* stop with
   no signal, and it *has*: the CC daemon's own log shows **no absorption from 2026-08-24 through 2026-09-02** (6,132
   consecutive `parse failed … 'bytes' object has no attribute 'entry_type'` WARNINGs, first day 08-23) **[L]**. But that
   stop went through a *different* handler (`cc_ng_organism.py:2402-2408`, already a WARNING), not through the
   `logger.debug` swallows the brief names. The named swallows are real and silent, but the one that actually fired
   is silent everywhere except a log line: not in `status`, not in the return value, not to the caller.
2. **Root cause of that documented stop is a poison frame, by reading [V][I]:** `ng_tract.TractReader` hands back a raw
   `bytes` object for anything it cannot frame (`ng-tract-rs/src/lib.rs:257-268`), and the drain dereferences
   `entry.entry_type` without a type check (`cc_ng_organism.py:2387`). One such frame anywhere in the file makes the
   drain raise before it truncates, **every cycle, forever**, and nothing after it is ever reached. A docs memory note
   (2026-09-07) documents the same reader behaviour for byte-swapped `TB` vs canonical `BT` magic.
3. **Truncation does run without a successful absorb** [V]: an entry whose embed/dual-pass fails is consumed and gone
   (`:2385` advances the offset before the attempt; the failure is `debug` at `:2399` or an uncounted `False` from
   `:2187-2189`). A *persistent* failure of this kind is the most dangerous mode because it looks healthy (tract stays
   small, no log line, no counter).
4. **#117 hypothesis, tested by reading: refuted as stated, supported only in a weaker sense.** The drain never steps
   (`:2205-2210`, test `tests/test_cc_deposit_step.py:489-518`) and runs on the autosave loop's own wall clock, so a drain
   failure cannot freeze `graph.timestep`, and a frozen timestep cannot stop the drain. They share a *surface* (a
   substrate that stops taking in conversation) but not a mechanism. Details §4.
5. **Design:** optional default-`None` `on_degraded=None` appended as the LAST kwarg of `drain_ingest_tract`, same
   vocabulary/shape as the #756b lane; eight hardcoded codes; **no organism logging change** (Chief's byte-identity
   constraint forbids raising the debug lines in the shared file). Daemon slice (docs repo, later, after NG merges) folds
   codes into flat `STATE.stats` counters through the sibling lane's existing `_report_recall(..., kind=…)`, adds
   outcome gauges (backlog bytes, last-absorb age) and makes the loop's catch-all visible. **Two decisions are surfaced,
   not settled:** per-step `try` in `_autosave_loop` (§5.4) and what to do with a poison frame (§8 Q1).
6. **A 70,563,518-byte tract file exists right now** (`stat` only, mtime 2026-09-29 20:40) **[L]** and no daemon is
   draining it. A naive daemon restart would run an **uncapped** drain of that backlog in one `_concurrent_lock` hold
   (§2.4 case A, §8 Q3). This is an operational hazard to flag to Josh, independent of this lane's code.

---

## 1. Corrections to the zone manager's read (verified, not assumed)

| # | Brief said | What the source shows |
|---|---|---|
| C1 | one `try` wraps `drain_ingest_tract`, `cc_update_probation`, `surface_wants`, `generate_emergent_want` (`:1917-1940`, except `:1931-1932`) | **Five** calls plus the import: `from cc_ng_organism import (…)` (`D:1908-1911`), `drain_ingest_tract` (`D:1924-1926`), **`trickle_gateway_conduit(_leg1_consumed)` (`D:1927`)**, `cc_update_probation` (`D:1928`), `surface_wants` (`D:1929`), `generate_emergent_want` (`D:1930`); `try` at `D:1907`, `except` `D:1931`, `logger.debug` `D:1932`. Loop is `D:1890-1944`. **[V]** |
| C2 | "an exception in the first aborts the rest of that cycle" | True but nearly moot for the drain: it is *first* and internally fail-soft (every I/O step is in its own `try`, `:2357-2371`, `:2380-2408`, `:2440-2456`), so it essentially cannot raise into `D:1931`. The starvation runs the other way: a failure in step *k* skips steps *k+1…5* (e.g. `cc_update_probation` raising every cycle silently kills `surface_wants` and `generate_emergent_want`), and an `ImportError` on `D:1908` kills **all five**. **[V][I]** |
| C3 | four swallows at `cc_ng_organism.py:2360, 2367, 2398, 2455` | Those are the `except` lines; the `logger.debug` is the next line (`:2361, :2368, :2399, :2456`). Correct as a count of four. But there is a **fifth handler at `:2402-2408`**, already `logger.warning` (comment `:2403-2406` says it was raised from debug for exactly this skew case), and it is **the one that fired 6,133 times** **[L]**. Also **silent branches with no log at all** that the #785 AST pass cannot list: `:2355-2356`, `:2387-2393` (filters), `:2396-2397` (falsy result), `:2448-2454`. **[V]** |
| C4 | the debug line is never written because root is INFO (`:895`) | Confirmed: `D:895`, hard-coded, no env override; no `setLevel`/`DEBUG` anywhere in `cc_ng_organism.py`, `cc_ng_host.py`, `ng_tract_bridge.py`, `ng_embed.py` (grep, no matches); `logger = logging.getLogger("cc_ng_organism")` (`:1021`) inherits INFO. **[V]** |
| C5 | title: "silently stop ALL conversation absorption" | Over-scoped. On the laptop there are **two doors** into `run_conversational_dual_pass`: the hook door `_deposit` (`D:1746-1775`; spawned from `D:1088` Stop and `D:1108` UserPromptSubmit, counted in `stats['deposits']` `D:1748-1749`) and the tract door (miniTID-captured turns). A dead drain stops the tract door only. Impact is still HIGH (the tract door carries what hooks do not), but "ALL" should read "all miniTID-captured turns". **[V]** |
| C6 | the swallow list is the #785 rows 12-15 and 47 | Also outside #785 because it is a `debug` **not inside an `except`**: `D:1897-1900` (`Autosave skipped — graph busy` then `continue`). A lock holder that never releases silently disables save, drain, probation and wants every cycle. Evidence the lock can be held long: the last run logged `Shutdown: _concurrent_lock busy after Ns — skipping final save` **[L]**. |
| C7 | (implicit) the VPS half imports this file | Confirmed: `cc_ng_host.py:1506-1546` has the identical five-call `try` (`:1517-1534`) and the same `logger.debug` (`:1534`); it calls `drain_ingest_tract` positionally at `:1526` under `_concurrent_lock` (`:1525`). **Path B, Josh-gated: listed, not touched.** **[V]** |

---

## 2. The whole path, producer → consumer

### 2.1 Producer
- **miniTID** (Rust, repo `~/Condensate`, `master` `4086540`), file `rust_core/src/minitid.rs` **[V]**:
  `deposit_turn` `:1561-1570` (called `:1855`) writes the user text and the assistant text as **two independent
  `ENTRY_EXPERIENCE` frames**, `source="cc_gateway"`; `deposit_pith_failure` `:1575-1580` (called `:1778`) writes a Pith
  failure as one more `cc_gateway` frame. Both go through `deposit_experience_entry` `:1537-1552`
  → `ng_tract::write::deposit_to_file` (`rust_core/vendor/ng_tract/src/write.rs:13-58`: `O_WRONLY|O_CREAT|O_APPEND`,
  `flock(LOCK_EX)`, write loop, unlock). Errors are discarded (`let _ =` at `:1551`, `:1564`, `:1577`): a failed deposit
  never reaches the tract and nothing says so (producer-side sibling silent failure, **out of scope**, §8 A2).
- **Path:** `CC_GATEWAY_TRACT_PATH` else `$HOME/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract`
  (`minitid.rs:1530-1535`, falls back to `/root` if `HOME` is unset); the NG side resolves the same way
  (`cc_ng_organism.py:2279-2289`). Raw text, no classification (`minitid.rs:1554-1560`) → **LAW 7 satisfied on the
  producer**; the drain does no classification either.
- Other writer of the same file: `cc_refeed.py:209` (`source="cc_gateway"`, front door only).
- **Frame format** (`ng-tract-rs` `25c5f52`): magic `0x42 0x54` (`format.rs:63`), 24-byte envelope (`:83`),
  CRC over the payload only.

### 2.2 Consumer chain
`D:1890 _autosave_loop` (every `AUTOSAVE_INTERVAL`, a **hard-coded `60.0`** at `D:576`, see §8 A4) →
non-blocking `_concurrent_lock.acquire` (`D:1897`) → `_guarded_save` (`D:1902`) → the five-call `try` (`D:1907-1932`)
→ `drain_ingest_tract` (`cc_ng_organism.py:2299-2460`; called `D:1924` with `return_consumed=True`, **no `max_entries`**)
→ per entry `_apply_gateway_experience` (`:2292-2296`: `embed()` then `run_conversational_dual_pass`, `:2137-2189`).
The daemon **discards** the absorbed count (`_leg1_absorbed`, `D:1924`, never read) and `handle_status` (`D:1022-1055`)
has no drain field; `stats` holds only `requests_total, deposits, recalls, rewards, errors, started_at`
(`D:744-750`). **No telemetry anywhere mentions the drain**: `pith_metrics.jsonl` has zero `ingest|tract|drain` hits **[L]**.

### 2.3 State the drain keeps
None across cycles except the caller's `state` dict (`last_forest_id`, `D:742`). **The tract file itself is the queue**:
no cursor, no retry queue, no counters. `run_conversational_dual_pass` docstring says "caller decides retry policy
(this function does not enqueue)" (`:2138-2139`); the tract-door caller has none.

### 2.4 What happens to deposits, by failure mode (once vs every cycle)

| Case | Code site | One cycle | Every cycle | Consumed? | File | Log/status today |
|---|---|---|---|---|---|---|
| **A** lib import fails / file unreadable | `:2357-2362`, `:2364-2369` | nothing read | nothing ever absorbed | no | **grows without bound** (no reader- or producer-side cap; 70.5 MB now [L]) | `debug` only; `_ret(0)` = same value as "nothing to do" |
| **B** path missing / desync | `:2355-2356` | 0 | 0 forever | no | n/a | **no log at all** |
| **C** per-entry embed/dual-pass fails (falsy or exception) | `:2396-2399`, dual-pass `:2146-2147`, `:2187-2189` | that turn **lost** | every new turn lost as it arrives | **YES** (offset advanced at `:2385` *before* the attempt) | stays small, looks healthy | `debug` or nothing; not counted anywhere |
| **D** poison frame / parse exception | `:2380-2408` | entries *before* the poison are absorbed, file untouched | those same entries re-absorbed each cycle; everything after the poison never reached | **no** (early return `:2408` before truncate) | grows without bound | WARNING `:2407`, **unlimited rate**; the `absorbed N` INFO (`:2458-2459`) is skipped even when `absorbed>0` |
| **E** truncate fails / file changed underneath | `:2443-2456` | absorbed entries stay | re-absorbed each cycle (idempotent `target_id=cc:conv::sha1(text)` `:2151` makes the forest node coincide; whether `_cc_bind_conversational_topology` `:2183` adds edges on a repeat **[U]**) | reports `b""` correctly (`:2439`) | grows | `debug` `:2456`; the changed-underneath branch `:2448-2454` is silent |
| **F** lock busy | `D:1897-1900` | cycle skipped | drain + save + wants never run | no | grows | `debug` |

**Answers to the brief's questions.** Does the tract grow without bound? Yes in A, B(if producer writes elsewhere),
D, E, F. Is data lost? **Yes in C** (permanently, silently). Is anything retried? Only by re-reading the file next
cycle, i.e. only in the not-consumed modes. Does truncation run without a successful absorb? **Yes** (C, and the
by-design filter skips `:2387-2393`, which count nowhere).

**Unbounded recovery drain [V]:** `max_entries` defaults to 0 = whole file in one call (`:2308-2312`), and the docstring
itself warns that an uncapped drain "gets drained in one unbounded lock hold" (`:2322-2327`). Neither caller passes a cap
(`D:1924`, `cc_ng_host.py:1526`).

### 2.5 LAW check
- **LAW 1:** the tract is a producer→consumer file between the CC's own gateway proxy (I/O-path infrastructure) and
  the CC organism inside one process; whether this file is "a Tract" in the LAW 1 sense or an I/O-path file is not settled
  by anything I read. Either way **this plan adds no module-to-module path**: reporting is a callback within one process.
- **LAW 7:** codes describe *transport/absorb outcomes*, never content classes; the reporter path must never write failure
  state into the substrate. Any poison-frame quarantine (Q1) must keep the bytes raw.
- **LAW 8:** the drain is on the autosave loop's own `time.sleep` clock (`D:1893`), not gated on conversation. It *is*
  gated on the non-blocking `_concurrent_lock` acquire (`D:1897`), which is the silent stop in case F.
- **LAW 4:** the organism owns the knowledge of *why* a drain step failed, so it must report; the daemon counts/logs.
- **LAW 5:** warn interval from the environment; the `60.0` literal (`D:576`) is an adjacent blemish (§8 A4).

---

## 3. Every silent site on the path

"Stops absorption?" distinguishes **entry** (one turn) from **drain** (the whole drain cannot run).

| ID | File:line | Swallowed | Caller / status sees | Stops absorption? |
|---|---|---|---|---|
| N1 | `cc_ng_organism.py:2355-2356` | tract path missing | `_ret(0)`, no log | **drain**, if path desync |
| N2 | `:2360-2362` (debug `:2361`) | `import ng_tract` / `ng_embed` fails | `_ret(0)` | **drain**, every cycle |
| N3 | `:2367-2369` (debug `:2368`) | file read fails | `_ret(0)` | **drain** |
| N4 | `:2387-2393` | wrong type/source/empty text | offset advanced, uncounted | no (by design; uncounted) |
| N5 | `:2396-2397` | dual-pass returned `False` | not counted, **consumed** | **entry** (lost) |
| N6 | `:2398-2399` (debug `:2399`) | exception in embed/wrapper | debug; **consumed** | **entry** (lost) |
| N7 | `:2402-2408` | reader/attr exception (poison) | WARNING; file untouched | **drain, wedge** |
| N8 | `:2448-2454` | file changed underneath | silent rewrite, `b""` | no; duplicates |
| N9 | `:2455-2456` (debug `:2456`) | truncate I/O fails | debug | no; duplicates |
| N10 | `:2146-2147`, `:2187-2189` | `run_conversational_dual_pass` inner failures | `False`, debug `:2188` | **entry** (lost) |
| D1 | `D:1897-1900` | lock busy | `debug`, `continue` | **drain + everything**, while held |
| D2 | `D:1902` | `_guarded_save` raises | outer WARNING `D:1943-1944`; skips this cycle's drain | drain, visibly. A save *refusal* returns `False` + CRITICAL and does **not** skip the drain (`D:987-1000`) |
| D3 | `D:1908-1911` | `ImportError`/`TypeError` (version skew) inside the shared `try` | `debug` `D:1932` | **all five steps** |
| D4 | `D:1931-1932` | any exception in steps 1-5 | `debug` | steps *k+1…5* |
| D5 | `D:1924-1927` | absorbed count discarded; no `status` field | nothing | n/a (blindness) |
| H1 | `cc_ng_host.py:1533-1534` | VPS twin of D4 | `debug` | Path B, Josh-gated |
| P1 | `minitid.rs:1551, 1564, 1577` | producer append fails | nothing | **entry** (never reaches tract); other repo |
| E1 | vendored `ng_embed` (WARNING in log) | remote embed HF **HTTP 401** → hash-embedding fallback, absorbed as success | 811 WARNINGs in one run [L] | not a stop; **degrades** the recall store; vendored → LAW 2 |

`status` shows **none** of these (§2.2).

---

## 4. The #117 hypothesis, tested by reading

**#117 as defined in the repo docs [V]:** the laptop's `graph.timestep` advances only via conversation deposits
(`~/.claude/claude-md/laws-detail.md:74`; docs `CC-CALLOSUM-TRUTH.md` §0.2: 1.5-8 steps/hour, ~80/hour under agentic drive); the intended remedy
is the wall-clock `CC_NG_AUTOSTEP` gate, **default OFF** (`tonic_engine.py:262`, consumed `:1272`). *(I could not find the
current #117 row body: `punchlist/ARCHIVE.md:111` holds an unrelated old #117 (Immunis), and `open/*.md` has no #117 row.
I used `laws-detail.md:74` + §0.2. Number-collision note for the punchlist owner.)*

**Test 1: can a drain failure freeze the step counters? No, by reading.** `drain_ingest_tract` and `drain_gateway_conduit`
"never step at all" (`cc_ng_organism.py:2205-2210`); `test_drain_ingest_tract_never_steps` asserts `timestep` unchanged
across a drain and that no step call is made (`tests/test_cc_deposit_step.py:489-518`); its third turn's dual pass returns
`False` and is simply not counted (`absorbed == 2`), which is the uncounted-failure shape of case C (the test does not assert
the file afterwards). The only step site is `cc_deposit_step` from the Stop hook door (`cc_ng_host.py`
`_deposit(step=True)`; daemon `D:1088`). So a wedged drain leaves the step clock exactly where #117 already has it.

**Test 2: can #117 stop the drain? No.** The drain does not read `timestep` and runs on `D:1893`'s wall clock.

**Test 3: do they share an observable surface? Yes, weakly.** Both read as "the CC substrate is not taking in / evolving
from conversation". A persistent drain failure would show in `status` as: `nodes` flat, `stats.deposits` still rising
(hook door), `timestep` unchanged **or** rising (independent), and no drain field at all. So `status` cannot tell a
wedged drain from a quiet day. That is the real overlap: **the same blindness**, not the same cause.

**What the evidence shows [L] (stopped daemon's `~/.claude/plugins/neurograph/daemon.log`, 134,362,138 bytes,
2026-04-16 → `=== CC-NG daemon stopped ===` 2026-09-23 19:23:58; grep/count/`cut` only):**

| Signal | Observation |
|---|---|
| `CC ingest-tract: absorbed N turn(s)` (INFO) | 5,632 lines. First 2026-07-07 04:07:45. Present on almost every day **07-07 → 08-23** (no lines on 07-09/10/12 and 07-24/25/26; I did not check why). **None 08-24 → 09-02.** Then sparse: 09-03 (4), 09-04 (2), 09-09 (2), 09-10 (3), 09-11 (37), 09-12 (12). **Last: 2026-09-12 16:41:49. None after.** |
| `CC ingest-tract parse failed` (WARNING, `:2407`) | 6,133 lines: one on 08-10 (`entry claims 768 bytes but only 645 available`, an older reader's message), then **6,132 × `'bytes' object has no attribute 'entry_type'`**: 08-23 (38), 08-24 (545), 08-25 (207), 08-26 (779), 08-27 (696), 08-28 (588), 08-29 (794), 08-30 (1117), 08-31 (720), 09-01 (648); last 09-01 22:44:06. Consistent with one persistent poison frame retried every cycle. |
| `SAVE-GUARD: REFUSING` | 11,178 lines, last 2026-08-13 04:10 (does not skip the drain, §3 D2). |
| Last run 09-23 02:41:54 → 19:23:58 (16 h 42 min, 1,042 lines) | **0** absorbed, **0** parse-failed, 0 `Autosave failed`; 811 `ng_embed … HF HTTP 401 … Falling back to hash`; 169 `Broken pipe`; 1 `Shutdown: _concurrent_lock busy`; only one `Checkpoint saved`. |

**What this does and does not establish.**
- **Supported [L]:** absorption *did* stop silently for ~10 days, the alarm existed only as a log line, and status/return
  value carried nothing. This is the kind of loss the row describes; the row's HIGH rating is justified.
- **Not the named swallow [L][V]:** the stop went through `:2402-2408`. The debug swallows (`:2361`, `:2368`, `:2399`, `:2456`,
  `D:1932`) are **invisible by construction**, so their absence from the log proves nothing: any of them may have fired
  during the quiet windows and I cannot tell.
- **Cause of the poison frames [I][U]:** the signature matches `memory/laptop_home_btf_magic_no_translator_silent_drop.md`
  (2026-09-07: unrecognised magic → raw `bytes`; laptop wheel built 2026-08-30 reads `BT`, not legacy `TB`) and
  `ng-tract-rs/src/format.rs:246` names "the 2026-08-23 Leg 1 drain wedge". I did **not** read the tract's content, so I
  cannot confirm which frames are unreadable or why. How the wedge cleared on ~09-03 is **[U]** (no record found).
- **Not establishable without touching something live:** (1) the 09-23 window: was there anything to drain, and did the
  drain return 0 silently? The only discriminator would be the tract's size/content at 02:41 and 19:23, which I have no
  snapshot of; (2) current tract content (frame magics, `{` bytes, entry count) needs a read of the live, still-being-
  appended file (mtime 2026-09-29 20:40 [L], miniTID pid 2619 alive) or a copy of it; (3) the installed
  `ng_tract.so` (mtime 2026-09-15 23:44; built from `ng-tract-rs/target/wheels`) was **not imported or run**, so its
  reader behaviour is inferred from source `25c5f52` plus the log's error string, which matches; (4) any live `status`
  (no daemon running); (5) the VPS half. The 70 MB size alone is **not** evidence of a drain fault: the daemon has
  been down since 09-23 19:23 while miniTID kept appending.

**Verdict for the hypothesis:** *refuted* as "this swallow is part of #117's frozen-step mechanism"; *confirmed but
re-attributed* as "conversation absorption stalled silently on this laptop", by a sibling handler, with the same
observability gap. It should be tracked as its own row's mechanism, not merged into #117.

---

## 5. Design (per Chief's binding constraints)

### 5.1 Constraints restated (all honoured)
Optional default-`None` reporting only, so every existing caller is byte-identical; **second repo (NeuroGraph), own
branch, own review; NG merges BEFORE the daemon**; the VPS/Syl's-process half (Path B) is Josh-gated, listed and untouched
(P294(b)); no vendored file, no protected file, no new module (LAW 3).

### 5.2 NeuroGraph slice (BUILD turn 1; repo NeuroGraph)

**Signature (one new keyword, appended LAST so positional and existing keyword callers are unchanged):**
```python
def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                        return_consumed: bool = False, max_entries: int = 0,
                        on_degraded=None):
```
- `on_degraded(code: str) -> None`, **same shape as the #756b lane's `cc_assemble_recall(..., on_degraded=None)`**
  (`assignments/build-002-organism.md:13`): one positional `str` from a fixed vocabulary; called synchronously on the
  calling thread, i.e. **under the caller's `graph._concurrent_lock`**, so a reporter must not block or take
  `STATE.lock` (daemon discipline `D:1741-1743`; the sibling `_deposit` bumps `errors` only after leaving the lock,
  `D:1772-1774`). A plain `list.append` is the intended reporter (the #756 "collector list" pattern).
- Implemented as a **function-local closure** next to the existing `_ret` (`:2351-2352`), not a new module-level symbol:
  ```python
  def _degraded(code):            # no-op when on_degraded is None
      if on_degraded is None: return
      try: on_degraded(code)
      except Exception: pass      # a reporter bug must never change the drain (Q5)
  ```
  This avoids a duplicate module-level helper if #756b adds its own (§6).
- **Return values, side effects, log output, exceptions: unchanged when `on_degraded is None`.** No control flow change
  at any site; each report is one added line beside an existing statement.

**Codes (hardcoded lowercase `[a-z0-9_]`; never `str(exc)`, a path, content, or a secret):**

| Code | Reported at | Meaning |
|---|---|---|
| `tract_missing` | `:2355` | file does not exist (expected before first deposit; **counter-only**, Q6) |
| `tract_lib_unavailable` | `:2360` | `ng_tract`/`ng_embed` import failed: whole drain cannot run |
| `tract_read_failed` | `:2367` | file exists but read failed: whole drain cannot run |
| `tract_parse_failed` | `:2402` | reader/attribute exception (poison frame): file untouched, wedge |
| `entry_failed` | `:2398` | embed/wrapper raised for one entry: entry **consumed and lost** |
| `entry_not_absorbed` | `:2396` (falsy) | dual pass returned `False` for one entry: entry **consumed and lost** |
| `truncate_failed` | `:2455` | truncate I/O failed: absorbed entries remain, will repeat |
| `tract_changed_underneath` | `:2448` | file no longer starts with our prefix: nothing removed |

Called once **per event**, so a counter's value is the number of entries affected. Filter skips (`:2387-2393`) are
*not* degradation and are not reported. The existing WARNING at `:2407` and every existing `debug` line are left
**as-is**: raising them would change the log output of the VPS half (Syl's process imports this file), which Chief's
byte-identity rule forbids. The organism therefore keeps **zero logging/counters of its own** (as the #756 plan §4 says).

**No new NG API is needed for outcome gauges:** the daemon already receives `absorbed` (`_leg1_absorbed`, `D:1924`) and can
stat `cc_gateway_tract_path()` (`:2284`, already public).

**Docstring/changelog:** document `on_degraded` and the codes in the docstring (`:2301-2350`) and add a changelog entry
(dated 2026-09-30, naming #781/#785 and this lane) in the header block at the top of the file.

### 5.3 Daemon slice (BUILD turn 2; repo docs; separate branch/slice; **after** NG merges **and** after/stacked on the
#756 daemon slice, see §6)

1. **Collector + fold, outside the graph lock.** In `_autosave_loop`, create `_ingest_codes = []` before the inner `try`,
   call `drain_ingest_tract(..., return_consumed=True, on_degraded=_ingest_codes.append)`. After the inner `finally`
   releases `_concurrent_lock` (`D:1941-1942`), for each code call the sibling lane's existing
   `_report_recall(None, code, kind='ingest')` (built on the #756 branch at `cc-ng-daemon.py:1894-1917`): it bumps flat
   `STATE.stats['ingest_fail_<code>']` under `STATE.lock` and emits a per-`(kind, code)` rate-limited WARNING carrying only
   `code=` and the exception **class name**. Reusing it is exactly "reuse the callback pattern so the lanes do not invent
   two vocabularies". Its docstring says never to call it while holding `STATE.lock` (non-reentrant); folding after the
   graph-lock release also respects the loop's own discipline (`D:1741-1743`).
2. **Outcome counters (proof the loop reached the drain):** `ingest_drain_cycles` (+1 each time the drain call returns),
   `ingest_absorbed_total` (+`_leg1_absorbed`, currently discarded), `ingest_cycles_skipped_busy` (+1 at the `D:1899`
   branch, which is today a `debug`).
3. **Outcome gauges in `handle_status`** (computed at request time, guarded, additive keys): `ingest_tract_bytes`
   (`os.path.getsize(cc_gateway_tract_path())`, `None` if absent) and `ingest_last_absorb_age_s`. These measure the *real
   output* (is the backlog growing, when did anything last land), so they catch failure modes nobody has a code for yet,
   including the 08-24→09-02 wedge from the outside.
4. **The catch-all becomes visible.** `D:1931-1932`: report `loop_step_failed` (rate-limited WARNING with the exception
   class name only) instead of `debug`; the import at `D:1908` gets its own `try` reporting `loop_import_failed`. This is
   also the guard for the merge-order hazard: if the daemon ships before NG, `on_degraded=` raises `TypeError`, which today
   the same catch-all would swallow at `debug`, silently killing all five steps every cycle.
5. **Rate-limit interval** from the environment (LAW 5). `_report_recall` reads `CC_NG_RECALL_WARN_INTERVAL_S`
   (default 60, `cc-ng-daemon.py:757` on the #756 branch). Reusing a recall-named knob for ingest is a misnomer; see Q4.

### 5.4 DECISION TO SURFACE (not settled): should each step in `_autosave_loop` get its own `try`?
- **Today:** one `try` (`D:1907-1932`) over five calls + the import; a failure in step *k* skips *k+1…5*.
- **Option A (recommend):** one `try` per step (`drain`+`trickle` together since `_leg1_consumed` feeds `trickle`;
  then `probation`; `surface_wants`; `emergent_want`), each fail-soft and each reporting its own code (`step_<name>_failed`).
- **Option B:** keep one `try`, only make it visible (item 4 above).
- **Why A is not a LAW 6 normalization:** it adds no pattern, class, framework or restructuring; it applies the fail-soft
  discipline *this same function already applies* to the Commons persist block directly below (`D:1933-1940`, comment: "an
  independent file, so a failure here can never touch the save-guarded checkpoint"). The steps are documented independent
  (`D:1904-1906` "Idempotent -- safe every autosave cycle"); their current coupling is an ordering accident of one `try`.
- **Why A is still behaviour-changing:** steps that were previously skipped behind a persistently failing earlier step
  will start to run (strictly more work under `_concurrent_lock`; wants/emergent-wants may begin materialising on a graph
  where they were silently dead). That is the correct outcome, but it is a change of behaviour on the live CC daemon, so
  it needs Chief/Josh's call and a mention in the merge notes. **The twin in `cc_ng_host.py` is not touched (Path B).**

### 5.5 Not in this lane (listed so they are not lost)
Poison-frame *policy* (Q1); a recovery-drain cap (Q3); Path B (`cc_ng_host.py:1506-1546`); the producer-side swallows;
`ng_embed`'s hash fallback; the `AUTOSAVE_INTERVAL` literal.

---

## 6. Collision analysis

- **With #756b (`daemon-recall-organism-756b`, same file `cc_ng_organism.py`)** [V]: its branch
  `cc-laptop-recall-organism-756b-20260930` is at the base with **0 commits ahead**, so I can compare only to its
  assignment text (`build-002-organism.md`). Its regions: the two render functions (`:1608`, `:1642`),
  `cc_pattern_completion_recall` (`:2875-3029`), `cc_assemble_recall` (`:5326-5524`). Mine: `drain_ingest_tract`
  (`:2299-2460`) only. **Disjoint function bodies.** Real overlap: (a) the **changelog header block at the top of the
  file**, where every lane inserts an entry: a guaranteed trivial textual conflict for whichever merges second;
  (b) a duplicate helper *if* both add a module-level reporter: mine is function-local (§5.2) so no second symbol is
  created, and a later hoist of a shared helper is a LAW 3 cleanup once both are merged; (c) tests: my new
  `tests/test_cc_drain_reporting.py` vs its `tests/test_cc_recall_reporting.py`, no collision. Vocabulary is shared
  (`on_degraded(code: str)`, lowercase codes, organism-does-not-log). Order between the two NG branches is free; each
  rebases over the other's header entry.
- **With other unmerged branches** [V, method-limited]: no local or last-fetched remote ref has a hunk whose function
  context is `drain_ingest_tract`, `_apply_gateway_experience`, `cc_gateway_tract_path` or a host `_autosave_loop`
  (scan by hunk-header function context, because `git diff A...B` reports merge-base line numbers, which are not
  comparable to `e4ebf982` for branches with old merge-bases; **no `git fetch` was run**, so remote refs may be stale).
  Nearest: `cc-conduit-retention-20260911` touches `drain_gateway_conduit` (a different function).
- **Daemon side (docs repo)**: both my slice and the #756 daemon slice edit `scripts/cc-ng-daemon.py`. Theirs (branch
  `cc-laptop-daemon-recall-756-20260930`, tip `27d362f5`, unmerged) adds `_report_recall`, `RECALL_WARN_INTERVAL_S`, the
  reply fields and `recall_fail_*` counters; mine edits `_autosave_loop` (`D:1890-1944`) and `handle_status`. Different
  functions but the same header block and the same helper: **stack mine on theirs** (rebase after they merge) so the helper
  is reused, not duplicated. Until then a daemon BUILD for #781 cannot be written against `origin/main` without inventing a
  second helper.
- **Required merge order:** NG (#756b, #781 in any order) → daemon #756 → daemon #781. No NG or daemon merge happens in
  this turn (rollout hold).

---

## 7. Test plan (fakes only: no live `STATE`, graph, checkpoint, tract file or `data/`; no daemon start)

### 7.1 P379 / #770 guard (every targeted run)
Existing tests already put the worktree first (`sys.path.insert(0, dirname(dirname(__file__)))`, e.g.
`tests/test_cc_callosum_leg1.py:3`), but `neurograph_rpc.py:735-738` and the daemon (`D:540-543`, both read) hard-code
`~/NeuroGraph` at `sys.path[0]`, so a narrow run can silently exercise the **primary** checkout. The new test file will,
at import (collection) time:
1. write to `sys.__stderr__` the resolved `__file__` of `cc_ng_organism` and of every NG module under test
   (`ng_embed`, `ng_tract`);
2. **raise** (collection error, the session fails) unless `os.path.realpath(cc_ng_organism.__file__)` starts with the
   test file's own repo root (`dirname(dirname(abspath(__file__)))`).
No `tests/conftest.py` exists (verified), and adding one would change every NG test, so the guard lives in the new file.
Daemon-side tests follow the sibling lane's convention (`scripts/tests/test_cc_ng_daemon_recall_status.py`, override
`Z12_756_DAEMON_UNDER_TEST`, printed `[P379/#770] daemon under test (resolved)` preamble) and inject a **fake**
`cc_ng_organism` into `sys.modules`, so they cannot depend on which checkout the daemon's `sys.path` prefers.

### 7.2 NeuroGraph tests: new `tests/test_cc_drain_reporting.py`
Fixtures reuse the fake pattern already used by `tests/test_cc_callosum_leg1.py:58-66` (a `SimpleNamespace` graph with an
`RLock`, `ng_embed.embed` and `run_conversational_dual_pass` monkeypatched) and `tmp_path` tract files built with
`ng_tract.deposit_experience`. The existing `tests/test_cc_dual_pass.py`, `tests/test_cc_deposit_step.py` fixtures were
**not** fully read; the BUILD turn must confirm each builds only a temp workspace before it is included in a run.

| # | Test | Why it FAILS on the base |
|---|---|---|
| T1 | `on_degraded` is the LAST parameter, default `None`; existing positional order `(graph, vector_db, state, tract_path, return_consumed, max_entries)` unchanged | `KeyError`: no such parameter |
| T2 | `tract_missing` reported for a nonexistent path; return `0` | `TypeError: unexpected keyword` |
| T3 | `tract_lib_unavailable` (`sys.modules['ng_tract']=None`); return `0` | same |
| T4 | `tract_read_failed` (flaky `open` for the path, `"rb"`, the `leg1:392` pattern); return `0` | same |
| T5 | `entry_failed` when the wrapper raises; **entry consumed** (file empty afterwards) and reported once per entry | same |
| T6 | `entry_not_absorbed` when the dual pass returns `False`; consumed; `absorbed` excludes it | same |
| T7 | **poison frame**: valid frame + `b'{"x":1}\n'` → reader yields `bytes` → `tract_parse_failed`; file **untouched**; leading entry absorbed and **re-absorbed on a second call** (dual-pass call count 2); no INFO line. A precondition assertion pins that `list(ng_tract.TractReader(b'{"a":1}\n'))[0]` is `bytes`, so a reader change fails loudly instead of the test silently passing | same; the precondition documents the wedge |
| T8 | `truncate_failed` (flaky `open` for `"wb"`, existing pattern `leg1:392-419`); `consumed == b""` | same |
| T9 | `tract_changed_underneath` (file rewritten between read and truncate, existing pattern `leg1:356`) | same |
| T10 | a **raising reporter** changes nothing: same return, same file bytes, same `consumed` | same |
| T11 | reporter set vs unset over T2-T9 scenarios: identical returns/`consumed`/file bytes | same |
| T12 | **byte-identity proof for default callers:** load the base module from `git show e4ebf982:cc_ng_organism.py` into a temp dir under a unique name and run the same scenarios (empty, missing, ok, entry-fail, poison, truncate-fail) through base and new with **no** callback; assert equal return values, tract bytes and captured `caplog` `(levelname, getMessage())` lists | vacuous-by-design on the base (a preservation guard, labelled as such, like the sibling lane's 4 guards) |
| T13 | call shapes still bind: `cc_ng_host.py:1526` (3 positional) and `D:1924` (`return_consumed=True`) via `inspect.signature(...).bind` | preservation guard |
| T14 | the reporter is invoked while the caller's `RLock` is held and never re-enters it (guards the "no `STATE.lock` under `_concurrent_lock`" discipline) | `TypeError` |

**Existing tests that must pass unchanged:** `tests/test_cc_callosum_leg1.py` (drain cases `:174-420`, fake fixture),
plus, after fixture confirmation, `tests/test_cc_deposit_step.py::test_drain_ingest_tract_never_steps` and
`tests/test_cc_dual_pass.py` drain cases. **Note (a coverage gap I found by reading, not running):** the existing
`test_drain_ingest_tract_return_consumed_is_empty_on_parse_failure` (`leg1:335-353`) feeds `b"not a valid BTF tract, ever"`,
which the reader byte-scans to zero entries **without raising**, so it likely never reaches the `:2402` handler; T7 is
the first test that does. **[I]**

**One targeted run** is requested for the BUILD turn only (same shape the sibling lane was authorised): scratch `HOME`,
`US=$(python3 -c "import site;print(site.getusersitepackages())")` taken BEFORE overriding `HOME`,
`HOME=$(mktemp -d) PYTHONPATH=$US python3 -m pytest -q <new file> tests/test_cc_callosum_leg1.py -p no:cacheprovider`.
**Not** the NG full suite (known-hanging test, #754), no daemon, no graph load, no `data/` or checkpoint path.

### 7.3 Daemon tests (later slice; docs repo `scripts/tests/…`)
Drive `_autosave_loop` without refactoring it: fake `STATE` (fake graph with `RLock`), fake `time.sleep` that flips
`STATE.running=False` after one pass, fake `cc_ng_organism` in `sys.modules`. Each fails on the base for the stated reason:
D-T1 codes appended by a fake drain become `ingest_fail_<code>` counters and one WARNING per interval (fake monotonic);
D-T2 lock held → `ingest_cycles_skipped_busy` and a WARNING (base: `debug`); D-T3 `ImportError` on the names import →
`loop_import_failed` (base: silent); D-T4 fake organism lacking `on_degraded` (version skew) → visible WARNING and the
other steps still handled per the §5.4 decision; D-T5 `handle_status` carries `ingest_tract_bytes`/`ingest_last_absorb_age_s`
against a `tmp_path` file (base: absent); D-T6 (only if Option A) step *k* raising does not skip *k+1…5*; D-T7 reply for a
clean cycle differs from base only by the new keys.

---

## 8. Risks, open questions, and what I did not verify

### Open questions for Chief / Josh
- **Q1 (Josh: data handling).** What should happen to a frame the reader cannot frame? (a) leave it and only report (the wedge
  persists, but is now loud); (b) skip-and-consume (**destroys** bytes; conflicts with Josh's stated *"I do NOT want them
  deleted"* for BTF entries, memory note 2026-09-07); (c) copy the raw bytes to a sibling quarantine file, then consume.
  Precedent for (c) in the same file: `drain_gateway_conduit` retains each file raw in a journal *before* parsing and marks
  a bad one `invalid` (`cc_ng_organism.py:2664-2674`, `:2724-2730`), and the leg1 test docstring anticipates "eventual
  quarantine" (`leg1:337-341`). I recommend (c) as its **own lane after** the observability slice. Not in scope here.
- **Q2 (Chief).** Per-step `try` in `_autosave_loop`: Option A vs B (§5.4).
- **Q3 (Chief/Josh).** Cap the drain (`max_entries` from an env var, LAW 5) so a large backlog cannot hold
  `_concurrent_lock` for the whole drain, and **do not restart the CC daemon casually** while a 70 MB tract is waiting:
  the first drain would be uncapped, under the lock, through remote embeds that were returning HTTP 401 → hash fallback
  on 09-23 [L]. Separate change; default 0 keeps today's behaviour.
- **Q4 (Chief).** Warn-interval knob: reuse the recall-named `CC_NG_RECALL_WARN_INTERVAL_S` via `_report_recall(kind=…)`, or add
  one generic daemon-side name for both lanes.
- **Q5 (Chief).** A reporter that raises is swallowed silently by the closure (§5.2) to keep the drain unchanged. Is one
  extra `logger.warning` (reachable only when a reporter was supplied, so default callers are unaffected) acceptable?
  Whatever is chosen should match #756b.
- **Q6 (Chief).** Should `tract_missing` be counter-only (proposed) or also WARN? It is normal before the first deposit.
- **Q7.** Log volume: the existing `:2407` WARNING is unlimited (≈1/min during a wedge); the daemon's rate-limited line adds a
  second line per interval for `tract_parse_failed`. Acceptable, or suppress the daemon line for that code?

### Risks
- **Merge order** (§6): daemon-before-NG → `TypeError` swallowed at debug → all five steps dead. Mitigated by §5.3 item 4 and by
  the order NG → daemon.
- **Sibling dependency:** the daemon slice needs `_report_recall`, which exists only on the unmerged #756 branch.
- **Behaviour change:** Option A of §5.4 (more work per cycle under the lock).
- **`kind='ingest'` counter names** (`ingest_fail_<code>`) become part of the `status` contract; additive, but the hook
  should not be taught to read them in this lane.
- **VPS half unchanged:** `cc_ng_host.py:1533-1534` stays silent until Path B is wired (Josh-gated). The byte-identical
  organism guarantees the VPS behaviour does not change; it also means the VPS gets **no** benefit yet.

### Adjacent findings (not this task; for the punchlist, per the standing rule)
- **A1.** The drain rewrites the tract with plain `open(path, "wb")` and **takes no `flock`** (`:2441-2447`), while the
  producer appends under `flock(LOCK_EX)` (`write.rs:13-58`): the producer's lock gives no protection against the drain's
  read-then-truncate window (docstring `:2410-2416` says the window is "two file ops", not zero). Potential silent loss of
  bytes appended in that window. **[V] code, [U] whether it has ever happened.**
- **A2.** miniTID discards deposit errors (`let _ =`, `minitid.rs:1551,1564,1577`). Other repo.
- **A3.** `ng_embed` (vendored, LAW 2 → fix at canonical, then re-vendor) falls back to **hash embeddings** on HF HTTP 401 and the
  dual pass absorbs them as real (811 WARNINGs in the last run [L]): the recall store can be polluted with non-semantic vectors.
- **A4.** `AUTOSAVE_INTERVAL = 60.0` is a literal (`D:576`), LAW 5.
- **A5.** The parse-failure test gap in §7.2.
- **A6.** `punchlist` #117 number collision (§4) and the #117 row body I could not locate.
- **A7.** The #785 AST audit misses handlers that swallow with no log and `debug` calls outside `except` (its own stated
  limit 1); the sites in §1 C3/C6 should be appended to it.

### What I did not verify
Anything by execution (no test run, no import of `ng_tract`, no daemon, no reader call); the **content** of the live tract;
whether the installed `ng_tract.so` matches source `25c5f52`; how the 08-24→09-02 wedge was cleared; whether the quiet
windows in the log hid debug-level failures; whether a repeated absorb adds topology at `:2183`; whether the #756b BUILD
will place a module-level helper; the fixtures of `test_cc_dual_pass.py`/`test_cc_deposit_step.py`; anything on the VPS
or Syl's scope (never touched). My log reading was `grep -c`, `grep | cut`, `sort | uniq -c` and `stat`; I did not print
conversation content.

---

## 9. Exact files a BUILD turn would touch

**Turn 1: NeuroGraph** (branch `cc-laptop-ingest-tract-781-20260930`):
- `cc_ng_organism.py`: `drain_ingest_tract` (`:2299-2460`: signature, docstring, one closure, eight one-line reports) and
  the changelog header block. **Nothing else in the file.**
- `tests/test_cc_drain_reporting.py` (new).
- `handoffs/z12-ingest-tract-781/returns/build-001.md` (separate docs-only commit).
- **Not touched:** `cc_ng_host.py` (Path B), any vendored file, any protected file, `tests/test_cc_recall_unification.py`, `data/`.

**Turn 2: docs repo** (own branch, after NG merge and stacked on the #756 daemon slice; not this dispatch):
- `scripts/cc-ng-daemon.py`: `_autosave_loop` (`D:1890-1944`), `handle_status` (`D:1022-1055`), changelog header.
- `scripts/tests/test_cc_ng_daemon_ingest_status.py` (new), plus the handoff return.
