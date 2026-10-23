# R5 embed failover chain — worker-001 return

Zone: Z11, lane `ng-embed-remaining-work-20260922`. Worker thread `7d470fe8-2737-4831-8ae4-bd438cb7773d`.
Executive Packet 139 R5, with Executive Packet 353 (chief-003) build guards applied.

## Frozen inputs (verified before any action in this turn)

Zone manager supplied these sha256 hashes as frozen inputs; I verified them against the
working tree before doing anything and they matched exactly — **no edits were made to
either file in this turn**:

- `ng_embed.py` = `13129f2bf56dd15abd61314679163613dc25d5bcf92133ba5e07077c48e9b329`
- `tests/test_ng_embed_r5_failover.py` = `c47450de73c31dfdf57b92ae4ff7eb35eba81b10d3f4f4b34d94719818a296f5`

I also noticed my own background full-suite retry (`bgmo9uzwk`) was writing to the same
path the zone manager announced as their authoritative run 3 (`/tmp/r5_full_suite_run3.log`).
I killed my process (PIDs 867639/867641) immediately on reading the continuation message,
before doing anything else, to avoid corrupting the zone manager's log.

## What changed (file:line, as committed)

`ng_embed.py`:
- `_ensure_model` (`ng_embed.py:371`), load-failure branch (`:447-451`): on a local
  ONNX/tokenizer load exception, instead of only setting `_model_failed = True` and
  returning `False`, it now also calls `self._fail_over_to_remote(exc, "model load")`
  and sets `_model_loaded = True`, returning `True`. This routes the caller's existing
  `if self._remote_mode:` branches to the remote path.
- New method `_fail_over_to_remote` (`ng_embed.py:453-469`): idempotent helper that sets
  `self._remote_mode = True` and logs the transition once. Called from the load-failure
  branch above and from the four inference call sites below.
- `embed_batch` (`ng_embed.py:493`): short-idx branch (`:521-539`, try/except at `:534`)
  and long-idx branch (`:542-563`, try/except at `:567`) each wrap their local ONNX call
  in `try / except EmbeddingUnavailableError: raise / except Exception:`, and on the
  latter call `_fail_over_to_remote` then re-issue the identical `_hf_remote_*` call the
  explicit-remote branch immediately above it already uses.
- `embed_windows` (`ng_embed.py:586`): short-path branch (`:607-621`, try/except at
  `:617`) and long-path window loop (`:624-...`, try/except at `:651`) follow the same
  pattern.

`tests/test_ng_embed_r5_failover.py`: new file, 3 tests (see below).

No other file was touched. `git diff --stat` before commit showed only these two files
changed/added (plus the untracked `handoffs/` directory, not committed per instruction).

## Targeted test run (Executive Packet 353 guard #732 compliance)

Command run (exactly, twice — once as part of this turn's step 1, matching the earlier
isolated runs earlier in this thread):

```
env -u NG_EMBED_REMOTE python3 -m pytest tests/test_ng_embed_r5_failover.py tests/test_ng_embed_dualpass.py -p no:cacheprovider -v
```

Result: **34 passed in 17.50s**, zero failures, zero skips.

`test_coordinator.py` and any `afterTurn`-path test were explicitly NOT run, per
instruction (lane `commons-test-isolation-738` not yet landed; those tests write the
real `~/NeuroGraph/data/checkpoints/commons.msgpack`).

**NG_EMBED_* environment variable names each test depends on / manages (names only, not
values, per Packet 353):**

- The `pytest` invocation itself is prefixed `env -u NG_EMBED_REMOTE`, so the ambient
  shell value of `NG_EMBED_REMOTE` (known-present in this T3 worker shell per Packet 353
  #732) is stripped from the process environment before pytest even starts.
- Inside `tests/test_ng_embed_dualpass.py`, the autouse fixture `_reset_singleton_and_env`
  additionally deletes `NG_EMBED_REMOTE`, `NG_EMBED_ALLOW_HASH_FALLBACK`, `HF_TOKEN`, and
  `NG_EMBED_TID_ENDPOINT` via `monkeypatch.delenv` before every test in that file, and
  individual tests in that file use `monkeypatch.setenv` on `NG_EMBED_REMOTE` and
  `HF_TOKEN` where an explicit-remote-mode scenario is being tested.
- Inside `tests/test_ng_embed_r5_failover.py` (my new file), the same autouse fixture is
  present, and — per the #732 guard — every one of the 3 test functions additionally
  calls `monkeypatch.delenv` on `NG_EMBED_REMOTE` itself at its own start (not relying on
  the fixture alone), and asserts `os.environ.get("NG_EMBED_REMOTE") is None`
  immediately before the call under test. The shared `_remote_mode_vector` helper
  (called by 2 of the 3 tests to build the explicit-remote-mode ground-truth vector)
  explicitly `monkeypatch.setenv`s `NG_EMBED_REMOTE` and `HF_TOKEN`, then explicitly
  `monkeypatch.delenv`s `NG_EMBED_REMOTE` again before returning control to the test, and
  asserts unset on the way out. `HF_TOKEN` is set by every test (value irrelevant —
  `_hf_post` is replaced directly in every test, so the token is never actually read).
  `NG_EMBED_TID_ENDPOINT` is deleted by the autouse fixture only; no R5 test depends on
  it (it governs the unrelated TID concept-extraction endpoint, not the embed/failover
  path under test).
- I additionally re-ran `tests/test_ng_embed_r5_failover.py` earlier in this thread with
  `NG_EMBED_REMOTE` confirmed present and set to a non-empty value in the ambient shell
  (i.e. without the `env -u` prefix) specifically to prove the tests do not silently ride
  the wrong code path under that contamination — all 3 passed. That run is not part of
  this turn's step 1 but is why I'm confident the `env -u` prefix in step 1 is
  belt-and-suspenders, not load-bearing, for this specific file.

## Full-suite status

**Own full-suite runs did not complete; the zone manager's run 3
(`/tmp/r5_full_suite_run3.log`) is the reference.**

For the record (not submitted as acceptance evidence): I ran the full
`pytest tests/ -q --ignore=tests/test_tonic_spine_anchoring.py` command twice earlier in
this thread. Run 1 was killed by me after a >13 min hang traced to ambient
`NG_EMBED_REMOTE` contamination causing a real network call elsewhere in the suite. Run 2
(env-clean) ran 6194.71s and reported 60 failed / 1625 passed / 26 skipped / 4 subtests
passed before an external `KeyboardInterrupt` cut it short at ~86% (consistent with the
zone manager's independent finding that it hung in
`tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps`, a PRD §9 stress
test unrelated to `ng_embed.py`). I spot-checked 5 of the 60 failures from run 2's log
and confirmed none touch `ng_embed.py`, `_ensure_model`, `_onnx_embed*`, `_hf_remote*`,
or `EmbeddingUnavailableError` semantics — root causes were in `neurograph_rpc.py`
retry-queue/graph-node wiring, a missing `cc_gsg_backfill` in `cc_ng_organism.py`,
`neuro_foundation.py` SNN voltage-decay math, a msgpack-restore type mismatch, and a CES
dashboard port conflict from a concurrent session. Neither `test_ng_embed_dualpass.py`
nor `test_ng_embed_r5_failover.py` appeared in run 2's failure list. I am not asserting
this as the acceptance record — per instruction, the zone manager's run 3 is the
reference for full-suite status.

## Commit and push

- Commit: `0f5ca4c839db9b69a03408544dd30669fcfc6ad8` on branch
  `cc-laptop-r5-embed-failover-20260929`.
- Staged and committed only `ng_embed.py` and `tests/test_ng_embed_r5_failover.py`
  (confirmed via `git status -sb` before commit: only those two files plus the untracked
  `handoffs/` directory, which was not added).
- Pushed: `git push -u origin cc-laptop-r5-embed-failover-20260929` succeeded (new
  branch on remote, no PR opened).
- Confirmed post-push:
  - `git status -sb` → `## cc-laptop-r5-embed-failover-20260929...origin/cc-laptop-r5-embed-failover-20260929` (up to date, clean except untracked `handoffs/`)
  - `git log -1` → commit `0f5ca4c8` as above
  - `git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929` →
    `0f5ca4c839db9b69a03408544dd30669fcfc6ad8	refs/heads/cc-laptop-r5-embed-failover-20260929`
    (matches the local commit SHA exactly)

## Acceptance items (worker-observed only; full-suite item 9 deferred to zone manager's run 3)

- **1 (no hash embed):** met — unchanged by this diff; not touched.
- **2, 3 (truncation/regression):** met — window/truncation logic not touched.
- **4 (unit-norm):** met — pooling/normalization code not touched.
- **5, 5a, 5b (dual-pass atomicity):** met — dual-pass code not touched; all 31
  pre-existing `test_ng_embed_dualpass.py` tests still pass.
- **6 (existing remote retry/quarantine):** met — reused unchanged; not replaced.
- **6a (R5 failover, the point of this task):** met per the 3 new tests in
  `tests/test_ng_embed_r5_failover.py` (load-failure and inference-failure failover
  producing bit-for-bit identical vectors to explicit `NG_EMBED_REMOTE=hf`; double-
  failure quarantine-to-`failed_embeds.jsonl` + raise).
- **7 (`NG_EMBED_REMOTE` unchanged):** met — explicit-remote-mode branch in
  `_ensure_model` untouched; semantics unchanged.
- **9 (full test suite green):** **not independently confirmed by this worker** — see
  Full-suite status above. Zone manager's run 3 is the reference.

## Things I did not verify

- I did not verify the zone manager's run 3 result myself (instructed not to run another
  full-suite pass).
- I did not run `test_coordinator.py` or any `afterTurn`-path test (instructed not to,
  pending `commons-test-isolation-738`).
- I did not re-vendor to any consumer repo (explicitly out of scope).
- I did not independently re-derive whether the 60 failures I spot-checked in my own run
  2 are identical to whatever run 3 finds — system state may differ between runs, and
  run 3 uses different pytest plugin flags (`-p z11_noreal_commons`, `--timeout=300`)
  than my runs did.

## Stop

Per instruction: stopping here. No self-review performed. No PR opened. No merge
attempted.
