# R5 correction pass — worker-002 return

Zone Z11, lane `ng-embed-remaining-work-20260922`. Brief: `assignments/worker-002-correction.md` incl. Addendum B (Executive Packet 376).
Scope done: **C1, C2, #765, #763, all four, in one turn.** Nothing dropped, nothing half-finished. Not touched: #764, #766.

## Commits (on top of `0f5ca4c`, normal push, no PR, no merge, no history rewrite)

| Commit | Content |
|---|---|
| `9d3a284` | C1: tests only (`tests/test_ng_embed_r5_failover.py`) |
| `c7c275360c8cf2639e88d3f4f5a531f15a64e5d8` | C2 + #765 + #763: `ng_embed.py`, `reembed_snowflake.py`, `tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py` (1-line swap) |

```
$ git status -sb && git log -3 --oneline
## cc-laptop-r5-embed-failover-20260929...origin/cc-laptop-r5-embed-failover-20260929
?? handoffs/
c7c2753 ng_embed: R5 correction - narrow failover except (#765), truthful _ensure_model (#763), C2
9d3a284 tests: R5 C1 - cover the three untested inference-failover sites
0f5ca4c ng_embed: R5 local-failure failover chain (Executive Packet 139)
$ git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929
c7c275360c8cf2639e88d3f4f5a531f15a64e5d8	refs/heads/cc-laptop-r5-embed-failover-20260929
```
`handoffs/` was never staged. Commit 2 diffstat: 4 files, +568 / -30.

## Files changed, with line refs (at `c7c2753`)

`ng_embed.py`
- `:288` `_BUG_CLASS_EXCEPTIONS` (TypeError, AttributeError, IndexError, KeyError, NameError, ValueError, AssertionError, NotImplementedError).
- `:415-` `_ensure_model(self, require_local=False)`; cached branches `:426`, `:430` return `not (require_local and self._remote_mode)`; explicit-hf branch `:444` returns `not require_local`; load-`except` `:493-501`: `require_local` -> raise-if-bug-class, set `_model_failed`, one WARNING with traceback, return False (no failover, no `_model_loaded`); default -> `_fail_over_or_raise` then `_model_loaded=True`, return True.
- `:505` new `_raise_if_bug_class`: ERROR + `exc_info` + `raise exc` for a bug-class defect.
- `:514` `_fail_over_to_remote` renamed `_fail_over_or_raise` (it can now raise): bug-class check, then `_model_failed = True`, one WARNING with `exc_info` per process, `_remote_mode = True`.
- Five failover sites now call the helper: load `:501`, inference `:593` (embed_batch short), `:626` (embed_batch long), `:676` (embed_windows short), `:710` (embed_windows long). `except EmbeddingUnavailableError: raise` untouched.
- New dated changelog entry `[2026-09-30]` above the 09-29 one. I did not edit the 09-29 entry (file rule: never edit prior entries); the new entry states the rename. The 09-29 entry still names `_fail_over_to_remote`, which is now historical.

`reembed_snowflake.py:87` — `emb._ensure_model(require_local=True)`; clearer abort message; changelog entry added. Only this call site.

`tests/test_ng_embed_dualpass.py:247` — one-line swap to `require_local=True` (see #763), changelog entry added.

`tests/test_ng_embed_r5_failover.py` — 3 -> 40 tests (see below), changelog entries.

## #765 — allowlist vs denylist (decision and evidence)

**Chose a denylist.** Classes I actually verified (onnxruntime 1.24.4, real cached Snowflake model, `HF_HUB_OFFLINE=1`, temp dir; nothing to the network):

| Failure | Raised class | Notes |
|---|---|---|
| ORT load, missing file | `NoSuchFile` | direct subclass of `Exception`, **not** OSError |
| ORT load, corrupt file | `InvalidProtobuf` | direct subclass of `Exception` |
| ORT run, wrong dtype / wrong rank / out-of-range token id | `InvalidArgument` | direct subclass of `Exception`; **same class for our-bug and for bad data** |
| ORT run, wrong/missing input name | builtin `ValueError` | raised by ORT's Python wrapper: a caller defect |
| `hf_hub_download` / `Tokenizer.from_pretrained`, model absent offline | `LocalEntryNotFoundError` (FileNotFoundError -> OSError) | environment |
| `hf_hub_download`, bad repo id | `HFValidationError` (**ValueError**) | config defect |
| ORT, unavailable provider requested | no exception, only a `UserWarning` | |

Why not an allowlist: ORT's own exceptions are bare `Exception` subclasses, so a stdlib allowlist (RuntimeError/OSError/MemoryError/ImportError) would let real ORT failures raise, which is exactly the original R5 bug. An allowlist of the ORT classes needs the private `onnxruntime.capi.onnxruntime_pybind11_state` path, and if that moves in a future ORT the allowlist silently narrows and the bug returns without any test noticing. A denylist fails safe toward failover.
Cost, stated plainly: an ORT `InvalidArgument` caused by our own dtype/rank bug still fails over, because ORT gives it the same class as a bad-data error. It is visible only in the single WARNING traceback. I did not try to classify by message text (brittle).
`ValueError` is in the denylist because both verified ValueError sources are caller/config defects. I found no environment failure that raises ValueError.

Both directions tested (all with real behavior, not only mocks):
- fails over, bit-identical to explicit `NG_EMBED_REMOTE=hf`, no ERROR log: RuntimeError, OSError, MemoryError, bare Exception, **real** ORT `RuntimeException` and `Fail`, at the inference site; RuntimeError, **real** hub `LocalEntryNotFoundError`, bare Exception, ImportError at the load site.
- raised as the original object, one ERROR with traceback, `_hf_post` never called, no `failed_embeds.jsonl`, `_remote_mode`/`_model_failed` stay False: all 8 denylist classes at the `embed()` site; TypeError at **all four** inference sites; a **real** `IndexError` (a session returning one output makes the code's own `outputs[1]` raise); TypeError and real `HFValidationError` at the load site.

## #763 — decisions

`_ensure_model()` callers, all of them: production `ng_embed.py:547` (`embed`), `:569` (`embed_batch`), `:656` (`embed_windows`); `reembed_snowflake.py:87`; tests (`test_ng_embed_dualpass.py` stubs at lines ~99/158/165/172/252/264/289/300/325 and real calls at 337-478; `test_ng_embed_r5_failover.py`). No consumer-repo copy was touched.

- **Guards at `:547/:569/:656`: kept.** The default path can no longer return False, but the guards state the fail-closed contract at the call sites and existing tests (`test_embed_raises_when_model_unavailable`, `test_embed_batch_raises_when_model_unavailable`) stub `_ensure_model` to False and rely on them. Removing them would break those tests for no behavior gain.
- (a) default callers keep R5 failover: tested (default `embed()` after a failed `require_local` call still fails over, bit-identical).
- (b) `require_local=True` with local unavailable: returns False, no failover, no `_model_loaded`, no `_remote_mode`, neither `_hf_post` nor `urllib.request.urlopen` called, no failover warning. Tested for load failure, for `NG_EMBED_REMOTE=hf`, and after a prior failover (no second load attempt). A bug-class defect during that load still raises rather than being reported as "local unavailable".
- (c) `reembed_snowflake.py:87` uses it. (d) `test_763_reembed_snowflake_aborts_...` runs `main()` against a temp vectors file with `--dry-run` (so a regression cannot write): exit code 1, zero remote calls, file bytes unchanged.
- `tests/test_ng_embed_dualpass.py:247` skip guard: **confirmed dead by the same cause** (default `_ensure_model()` returns True right after a load failure; asserted in `test_763_require_local_false_when_remote_selected_or_already_failed_over`), so I made the one-line swap.

## `_model_failed` readers (grepped the whole repo before changing it)

Production code: **none**. Writes only: `ng_embed.py:__init__` (False), the `require_local` load branch, and `_fail_over_or_raise`. The two reads that existed at the base (`if self._model_failed: return False` fast paths) were removed by R5 in `0f5ca4c`. Only reader anywhere is `tests/test_ng_embed_r5_failover.py:220` (my own assertion). `stats` does not expose it. So setting it on inference failover changes no behavior.

## Commands run and the NG_EMBED_* names each ran under

Every pytest command: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest <files> -p z11_noreal_commons -p no:cacheprovider ...`, cwd = the worktree (or my scratch clone for mutation runs). Names in play: **`NG_EMBED_REMOTE`** is removed from the process environment by `env -u`, then each test sets/deletes it itself. Targeted files only: `tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py`. **Never run:** the full suite, `test_coordinator.py`, `test_tonic_bridge.py`, any afterTurn test.

Per test (names only, none of these tests reads any other `NG_EMBED_*` name):
- Every new test in `test_ng_embed_r5_failover.py`: `NG_EMBED_REMOTE` deleted via `monkeypatch.delenv` and `assert os.environ.get("NG_EMBED_REMOTE") is None` immediately before the call under test. Ground-truth helpers set `NG_EMBED_REMOTE=hf` (`monkeypatch.setenv`, asserted `== "hf"`), then delete it again and assert unset. `test_763_..._remote_selected_or_already_failed_over`: `NG_EMBED_REMOTE=hf` first half, unset (asserted) second half. `HF_TOKEN=tok-test` is set by them (unused: `_hf_post` is always replaced).
- The autouse fixture in that file additionally deletes `NG_EMBED_ALLOW_HASH_FALLBACK`, `NG_EMBED_TID_ENDPOINT`, `HF_TOKEN`.
- `test_tokenizer_truncation_is_not_enabled_on_loaded_instance` (dualpass, now `require_local=True`): that file's autouse fixture deletes `NG_EMBED_REMOTE`, `NG_EMBED_ALLOW_HASH_FALLBACK`, `NG_EMBED_TID_ENDPOINT`; module sets `HF_HUB_OFFLINE=1`.
- Exploratory scripts (ORT/hub/tokenizer class verification): `env -u NG_EMBED_REMOTE HF_HUB_OFFLINE=1`.

Plugin trace: `grep -c BLOCKED /tmp/z11_r5/plugin_trace.log` = 0 after all my runs.

## Counts

| Point | R5 file | dualpass | Total |
|---|---|---|---|
| Before (`0f5ca4c`, worker-001) | 3 | 31 | 34 passed |
| After C1 (`9d3a284`) | 9 | 31 | 40 passed |
| After all (`c7c2753`) | 40 | 31 | **71 passed, 0 failed** |

Acceptance 1 reading: C1 = 3 site cases + 3 double-failure cases (parametrized); flag test 1; #765 = 6+8+4+1+1+4+2 cases; #763 = 5 tests. The original 3 R5 tests and the 31 dualpass tests still pass. I did **not** run the full suite and make no claim about it; suite wording per P375: no claim, not run.

## Negative controls (scratch clone under `/tmp`, deleted afterwards; worktree `ng_embed.py` never mutated)

Method: `git clone --no-hardlinks` of the branch into `/tmp/z11_r5_worker002/scratch` (first attempt with hardlinks failed cross-device; see surprises), copy the file(s) under test, delete/regress ONE thing, run the R5 test file under the guard. The module-level assertion (and the 9/9 control) prove the scratch `ng_embed` was the one loaded.

**C1 (wrap deleted, one site at a time; control 9/9 pass):**
| Wrap removed | Result |
|---|---|
| embed_batch short | exactly `..matches_explicit_remote[embed_batch_short]` and `..double_failure..[embed_batch_short]` fail (2 failed, 7 passed) |
| embed_batch long | exactly the two `[embed_batch_long]` cases fail (2 failed, 7 passed) |
| embed_windows long | exactly the two `[embed_windows_long]` cases fail (2 failed, 7 passed) |
| embed_windows short (control that the original tests catch it) | the original inference + double-failure tests fail (2 failed, 7 passed) |

**C2 / #765 / #763 (control 40/40 pass on final files):**
| Regression | Failed |
|---|---|
| M1 `exc_info` removed from failover WARNING | 11 |
| M2 `_model_failed = True` removed from `_fail_over_or_raise` | 6 |
| M3 denylist emptied (no bug-class check) | 16 |
| M4a/M4b/M4d one site made unconditional failover (batch short / batch long / windows long) | 1 each, exactly `test_765_bug_class_raised_at_every_inference_site[<that site>]` |
| M4c same for embed_windows short (entry point for `embed()`) | 17 |
| M5a `require_local` load branch removed | 2 (incl. the reembed test) |
| M5b explicit-hf branch ignores `require_local` | 1 |
| M5c cached branches ignore `require_local` | 1 |
| M5d `require_local` load branch stops re-raising bug-class | 1 |
| M6 `reembed_snowflake.py` reverted to `_ensure_model()` | 1 (`test_763_reembed_snowflake_aborts...`) |

All 16 regressions are caught. Not mutation-tested: the `test_ng_embed_dualpass.py:247` skip guard (it only matters when the model is absent from the cache, which I cannot produce without disturbing the cache).

## Real commons file (`~/NeuroGraph/data/checkpoints/commons.msgpack`), before and after

```
BEFORE: 192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
AFTER:  192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
```
Byte-identical. I read nothing else under `~/NeuroGraph/data/`. Vault (`~/docs`) and every auto-memory directory: untouched by me this turn.

## Surprises and things I did not verify (listed, not acted on)

1. **sys.path hazard under the guard plugin.** Importing `neurograph_rpc` (which the plugin's fixture does on the first test) prepends `/home/josh/NeuroGraph` (the **main checkout**) to `sys.path`. Any repo module first imported after that resolves to the main checkout, not the branch. It bit me once: my first reembed test did a plain `import reembed_snowflake`, got the main checkout's old copy, and (correctly) failed by re-embedding over my fake remote, which also incidentally demonstrates the #763 regression. Fixed in the test (explicit-path load + a module-level assertion that `ng_embed` came from this repo). My earlier results remain valid (test files import `ng_embed` at collection behind their own `sys.path.insert(0, root)`; the 9/9 control proves it). **For the zone manager:** other test modules that import repo modules lazily (for example `openclaw_adapter` inside `test_openclaw_adapter_has_no_hash_embed`) may be testing the main checkout's copy in a guarded full-suite run. I did not investigate further.
2. **`os.cpu_count()` returning None.** `ng_embed.py` load does `max(1, os.cpu_count() // 2)`. On a host where that is None it raises TypeError. Before #765 that failed over silently; now, being bug-class, it raises loudly at load. Latent, pre-existing defect that #765 surfaces. Not fixed (out of scope). Fine on this machine (returns 4).
3. **`InvalidArgument` residual** (see #765 table): our own dtype/rank bug still fails over.
4. **Decision for Josh:** `require_local=True` returns False when `NG_EMBED_REMOTE=hf` selects remote. Base behavior was that `reembed_snowflake.py` under `NG_EMBED_REMOTE=hf` proceeded via remote; it now aborts. Only the opt-in parameter changed; default callers and the `NG_EMBED_REMOTE` semantics (criterion 7) are unchanged. I chose the fail-closed reading of "local or nothing" for a tool that rewrites Syl's `vectors.msgpack`.
5. **"No network call" scope.** Local load includes `hf_hub_download`, a fetch of the local model file from the hub (pre-existing behavior). I read requirement (b) as "no remote embedding API call and no failover", which is what is tested; a `require_local` caller whose model is missing would still try to download the local model first. Tests simulate "local unavailable" by making that download raise.
6. A failed `require_local` load is not cached, so repeated `require_local` calls retry the load. Harmless for `reembed_snowflake.py` (exits).
7. Brief said "update the R5 changelog entry"; file rule says never edit prior entries. I added a new dated entry.
8. Hygiene slips, no effect: (a) my first quest-hint command had backticks inside double quotes; the shell ran an argument-less `git clone`, which failed harmlessly; (b) my first scratch clone failed (cross-device hardlinks) and a following `cp` ran in the worktree, refusing to copy a file onto itself: I verified `ng_embed.py` was unchanged; (c) I briefly created an untracked probe test in `tests/` to print import paths and deleted it immediately; it was never staged and `git status` is clean apart from `handoffs/`.
9. Not verified: any consumer repo; the full suite (72 known failures on clean main); the real ORT out-of-memory exception class (I did not induce a real OOM; the real ORT `RuntimeException`/`Fail` used in tests are constructed, not raised by ORT); behavior of the `dualpass:247` skip when the model is absent.
10. #764 (`_hf_remote_call` with no HF token / tokenizer-load failure) and #766 (permanent one-way failover) are unchanged, as instructed. #764 is visible in my new tests only as the existing behavior.

Stopped here. No self-review, no PR, no merge.
