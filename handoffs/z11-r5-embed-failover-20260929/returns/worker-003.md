# R5 C3 return — worker-003

Zone Z11, lane `ng-embed-remaining-work-20260922`. Brief: `assignments/worker-003-c3.md` (delta law-enforcer finding F1).
Scope done: **F1 fixed with remedy 2 (the required shape), tested, mutation-controlled, pushed.** Not touched: #764, #766, #772, F2/#771.

## Commit (on top of `c7c2753`, normal push, no PR, no merge, no history rewrite)

`245d2bd1b865feebdb559ac8fe9e4671ccc3973d` — `R5 C3: require_local is honored by the inference path (delta-LE F1)`. Files: `ng_embed.py`, `reembed_snowflake.py`, `tests/test_ng_embed_r5_failover.py` (+401 / -19). `handoffs/` never staged.

```
$ git status -sb && git log -3 --oneline
## cc-laptop-r5-embed-failover-20260929...origin/cc-laptop-r5-embed-failover-20260929
?? handoffs/
245d2bd R5 C3: require_local is honored by the inference path (delta-LE F1)
c7c2753 ng_embed: R5 correction - narrow failover except (#765), truthful _ensure_model (#763), C2
9d3a284 tests: R5 C1 - cover the three untested inference-failover sites
$ git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929
245d2bd1b865feebdb559ac8fe9e4671ccc3973d	refs/heads/cc-laptop-r5-embed-failover-20260929
```

## Design chosen (remedy 2) and why

Remedy 1 (check `_remote_mode` after each batch) was not used: the failover would already have sent that batch's text to the HF API inside `embed_batch`. Remedy 2 was possible with a small change, so I did it.

`ng_embed.py`
- `embed` (`:574`), `embed_batch` (`:597`), `embed_windows` (`:692`) take `require_local: bool = False`. Default behavior for every existing caller is unchanged.
- New `_require_model(require_local)` (`:566`) replaces the three `if not self._ensure_model(): raise` guards. It calls `_ensure_model(require_local=True)` only when asked, and plain `_ensure_model()` otherwise, so the existing tests that stub `_ensure_model` with `lambda: True/False` still work. **Already-in-remote-mode handling:** `_ensure_model(require_local=True)` returns False when `_remote_mode` is already True (explicit `NG_EMBED_REMOTE=hf`, or an earlier default-caller failover), so the call **refuses** with `EmbeddingUnavailableError` before any embedding or remote work.
- `_fail_over_or_raise(exc, stage, require_local=False)` (`:539`, branch `:551`): with `require_local` it logs one WARNING with the traceback and raises `EmbeddingUnavailableError` (chained `from exc`) instead of failing over. So the `_hf_remote_*` re-issue lines after it are never reached. `_remote_mode` and `_model_loaded` are untouched; `_model_failed` is set (it is a diagnostic; same as the existing `require_local` load branch). A bug-class defect still raises as the original object with ERROR + traceback (the bug-class check runs first). `except EmbeddingUnavailableError: raise` untouched.
- The four inference sites pass the flag (`:641`, `:674`, `:726`, `:760`), and every remote arm is now `if self._remote_mode and not require_local:` (`:629`, `:659`, `:716`, `:742`). That last part closes a race: if another thread's default-caller failover sets `_remote_mode` after this call passed the guard, a `require_local` call still uses the local session and never the remote arm.
- New dated changelog entry. The module-level `embed()`/`embed_batch()` convenience wrappers (`:1483`, `:1495`) were not given the flag (no caller needs it).

`reembed_snowflake.py`
- `:140` `emb.embed_batch(batch_texts, require_local=True)`; `:151` a specific `except EmbeddingUnavailableError` placed **ahead of** the existing broad `except Exception`, logging an error and `sys.exit(1)`. `--dry-run` handling untouched; the load-time abort (`:99`) kept as is. Changelog entry added.
- Why the specific clause is required, not optional: the tool's pre-existing broad `except Exception` swallows any batch error, counts it, continues, and finally prints "Writing anyway (errored entries unchanged)". Without the new clause, my `EmbeddingUnavailableError` would have been swallowed and a half re-embedded file (mixed embedding spaces) written.

## What is on disk after an abort

**Nothing is written.** The tool's only write is `open(VECTORS_PATH, "wb")` at `reembed_snowflake.py:201`, at the very end of a successful run. Per-batch results are placed only in the in-memory `entries` dict (non-dry-run) and are discarded on `sys.exit(1)`. So after an abort, `vectors.msgpack` is byte-identical to before: no partial file, not even batches that had succeeded. With `--dry-run` nothing is ever written. (Line 205 is a read-only verification after a successful write.)

## Tests (`tests/test_ng_embed_r5_failover.py`, +23)

- Env-class inference failure with `require_local=True` raises `EmbeddingUnavailableError`, with **zero** remote calls (`_hf_post`, `_hf_remote_call` and `urllib.request.urlopen` are all counted; `urlopen` raises if touched), `_remote_mode` False, `_model_loaded` True, no failover warning, one not-failing-over WARNING with traceback, no quarantine file: 5 cases (embed_batch short, embed_batch long, embed_windows long, embed_windows short, embed).
- Bug-class (`TypeError`) with `require_local=True`: raised as the original object, ERROR logged, zero remote calls: same 5 cases.
- Already in remote mode (explicit `NG_EMBED_REMOTE=hf`; after a default-caller failover) + `require_local=True` refuses, no added remote call: 6 cases (3 entry points x 2 states).
- Mid-call `_remote_mode` flip never reaches the remote arm: 4 cases. `embed` is excluded and the test says why: it guards twice, so a flip between the guards makes the second guard refuse (safe, zero remote calls, and covered by the already-remote test). This is simulated by wrapping the guard; it is not a real thread race.
- Regression pin: omitted and explicit `require_local=False` still fail over, bit-identical to explicit `NG_EMBED_REMOTE=hf` (also pinned by the earlier C1 parametrized tests, which still pass): 1 test.
- Tool level, `reembed_snowflake.main()`, 130 entries (batches 64/64/2), local inference OK on batch 1 and RuntimeError on batch 2: exit code 1, `session.calls == 2` (batch 3 never attempted), zero remote calls, `_remote_mode` False, vectors file byte-identical. Two variants: `--dry-run`, and **write path enabled** (no `--dry-run`, `VECTORS_PATH` asserted to be a temp file) so a regression that carried on and wrote would change the bytes. In the `--dry-run` variant the bytes assertion is vacuous by design; the other variant is the meaningful one.
- These tests do not stub `_ensure_model`; the real guard is under test.

## Counts

| Point | R5 file | dualpass | Total |
|---|---|---|---|
| Baseline (`c7c2753`, worker-002) | 40 | 31 | 71 passed |
| After C3 (`245d2bd`, committed state) | 63 | 31 | **94 passed, 0 failed** |

I did **not** run the full suite; make no claim about it. Clean `origin/main` is red (72 known failures, #761).

## Commands and NG_EMBED_* names

Every pytest command: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest <files> -p z11_noreal_commons -p no:cacheprovider ...` (guard plugin v4), cwd = the worktree (or my scratch clone for mutation runs). Targeted files only: `tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py`. Never the full suite, `test_coordinator.py`, `test_tonic_bridge.py`, or any afterTurn test. `reembed_snowflake.py` was never run against real data, only via the temp-file test.

`NG_EMBED_REMOTE` is removed from the process environment by `env -u`; each new test also deletes or sets it itself and asserts it before the call under test:
- All C3 unit tests except the explicit-hf half of the already-remote test: `NG_EMBED_REMOTE` deleted via `monkeypatch.delenv`, asserted `is None`.
- `test_c3_require_local_refuses_when_already_in_remote_mode[explicit_hf]`: `NG_EMBED_REMOTE` set to `hf` (asserted `== "hf"`).
- `test_c3_default_and_explicit_false_still_fail_over_bit_identical` and the ground-truth helper: `NG_EMBED_REMOTE` set to `hf` only inside the helper, then deleted and asserted unset.
- Tool-level tests: `NG_EMBED_REMOTE` deleted and asserted unset.
- `HF_TOKEN=tok-test` is set by them (never read: `_hf_post` is always replaced). The autouse fixture also deletes `NG_EMBED_ALLOW_HASH_FALLBACK`, `NG_EMBED_TID_ENDPOINT`, `HF_TOKEN`. No other `NG_EMBED_*` name is used.
- The dualpass file's own autouse fixture deletes `NG_EMBED_REMOTE`, `NG_EMBED_ALLOW_HASH_FALLBACK`, `NG_EMBED_TID_ENDPOINT`; it sets `HF_HUB_OFFLINE=1`.

## `[z11 origin]` lines (final run, on the committed state `245d2bd`, 94 passed)

```
[z11 origin] cwd/worktree = /home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929/
[z11 origin]  neurograph_rpc       (not imported yet)
[z11 origin]  ng_embed             WORKTREE       /home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929/ng_embed.py
[z11 origin]  commons              (not imported yet)
[z11 origin]  neuro_foundation     (not imported yet)
[z11 origin]  cc_ng_host           (not imported yet)
[z11 origin]  cc_ng_organism       (not imported yet)
[z11 origin]  openclaw_hook        (not imported yet)
[z11 origin]  universal_ingestor   (not imported yet)
[z11 origin] neurograph_rpc.__file__ = NOT IMPORTED at collection
[z11 origin] END: neurograph_rpc.__file__ = /home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929/neurograph_rpc.py | repo modules loaded from MAIN checkout: ng_salience_gate
94 passed in 4.82s
```
**Main-checkout note, as the brief asks me to say:** the END line reports `ng_salience_gate` loaded from the MAIN checkout. I re-ran several times (including this final run) and it recurs identically. It is the lazily-imported module the sys.path hazard from worker-002 predicts (imported by `neurograph_rpc` after the plugin's path change); no test here uses it. `ng_embed` and `neurograph_rpc` are both WORKTREE. I did not investigate further.
An earlier run of mine after the last changelog edit was filtered to counts only; I re-ran on the committed state to get the lines above, so that run is not relied on.

## Mutation results (scratch clone under `/tmp`, `git clone --no-hardlinks`, deleted afterwards; worktree `ng_embed.py` hash unchanged: `662f0bc8...`)

Control (final files, unmutated, in scratch): 63 passed. One regression at a time, R5 test file under the guard:

| Regression | Result |
|---|---|
| R1 `reembed_snowflake.py` back to default `embed_batch(batch_texts)` | tool-level test fails, both variants (2) |
| R2 `EmbeddingUnavailableError` clause removed from the tool | tool-level test fails, both variants (2) |
| N1 embed_batch short site stops passing the flag | 1 unit case + both tool-level (3) |
| N2 embed_batch long site stops passing the flag | 1 (the `[embed_batch_long]` unit case) |
| N3 embed_windows short site stops passing the flag | 2 (`[embed_windows_short]`, `[embed]`) |
| N4 embed_windows long site stops passing the flag | 1 (`[embed_windows_long]`) |
| N5 `require_local` branch removed from `_fail_over_or_raise` | 7 (5 unit + both tool-level) |
| N6 `_require_model` ignores `require_local` | 3 (already-in-remote-mode cases) |
| N7a-d one remote arm no longer gated (batch short / batch long / windows short / windows long) | exactly 1 each: the mid-call-flip test for that site |
| N9 `embed()` stops passing the flag to `embed_windows` | 1 (`[embed]`) |

All 13 caught, and each per-site regression fails its own site's test. Not mutation-tested: the load-time guard (covered by the existing `test_763_*` tests from worker-002).

## Real commons file (`~/NeuroGraph/data/checkpoints/commons.msgpack`)

```
BEFORE: 192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
AFTER:  192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
```
Byte-identical (checked after the tests and again after the final committed-state run); plugin trace `BLOCKED` count 0. I read nothing else under `~/NeuroGraph/data/`. Vault (`~/docs`) and every auto-memory directory: untouched by me this turn.

## Surprises and unverified (listed, not acted on)

1. **The tool still swallows other errors.** The pre-existing broad `except Exception` in `reembed_snowflake.py` still swallows any non-`EmbeddingUnavailableError` batch error (including a bug-class `TypeError` that #765 now lets through from `ng_embed`), counts it, continues, and writes a partially re-embedded file ("Writing anyway"). I only added the abort for `EmbeddingUnavailableError`, as scoped. Whether a bug-class error should also abort is a decision for you.
2. `_model_failed` is set True when a `require_local` inference failure raises (diagnostic flag, consistent with the load branch). The brief named only `_remote_mode` and `_model_loaded` as things not to change.
3. A `require_local` failure logs its WARNING every time (not once per process, unlike the failover warning), matching the existing `require_local` load branch; the tool exits after the first.
4. The mid-call race is simulated by wrapping the guard, not by real threads. `embed()` is left out of that one test for the reason given above.
5. The `dualpass` skip guard swapped by worker-002 and the `os.cpu_count()` item (F2) are untouched, as instructed.
6. Not verified: the real ONNX Runtime out-of-memory class (I never induced one); any consumer repo; behavior on a real run of `reembed_snowflake.py` (never run against real data); the full suite (not run).

Stopped here. No self-review, no PR, no merge.
