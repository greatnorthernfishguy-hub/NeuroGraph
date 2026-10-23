# R5 embed failover chain — worker assignment

Zone: Z11 (`z11-ng-capability-port-20260923`), lane `ng-embed-remaining-work-20260922`.
Dispatched by: Z11 zone manager (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`).
Authority: `worktree_write` only. You do not have `merge_candidate` or `live_gated`.

## Why this exists

Chief-003 ruled (2026-09-29) that the P139(3) sequencing gate for this lane is
SATISFIED: the KISS-half merge (`groupb-kiss-shared-graduation-001`, merge commit
`c43b06f`) has landed on NeuroGraph `origin/main` (now `e4ebf98`), which was the
blocking predecessor in critical-path order. You are cleared to build R5.

R5 = "the 055 failover chain folded into dual-pass" — Executive Packet 139, folded
into the dual-pass design 2026-09-24. Governing spec (read it in full before
touching code):
`superpowers/specs/2026-09-20-ng-embed-dual-pass-nontruncating-design.md`, §"R5"
(around line 35) and acceptance criterion 6a (line 204).

## Scope — exactly this, nothing else

**Touch only `ng_embed.py`** (this worktree's copy) and its test file
(`tests/test_ng_embed_dualpass.py`, or a new `tests/test_ng_embed_r5_failover.py`
if that reads cleaner — your call). Do not touch any other file. Do not touch any
other worktree or `/home/josh/NeuroGraph` (the main checkout) directly.

`ng_embed.py` is a LAW-2 vendored file — this repo is its canonical source. You
are editing the canonical copy. Re-vendoring to the other 7-8 consumer repos is
explicitly a **separate, later, Josh-gated step** — do not attempt it, do not
touch any consumer repo's copy.

**Do not merge to `main`.** Any merge rides the standing rollout hold (P329/P337).
Build, test, and push this branch (`cc-laptop-r5-embed-failover-20260929`,
already checked out here off current `origin/main`). Do not open a PR — that is
the zone manager's job after independent review. Just commit and `git push -u
origin cc-laptop-r5-embed-failover-20260929`.

## The actual gap (read the code before you start)

Current `ng_embed.py` (this branch, at HEAD `e4ebf98`):

- `_ensure_model()` (~line 341): if `NG_EMBED_REMOTE=hf` is explicitly set, it
  switches to remote mode as the PRIMARY (no local load attempted at all) —
  this is unchanged, correct, existing behavior. Leave this branch alone.
- If `NG_EMBED_REMOTE` is **unset** and the local ONNX model/tokenizer load
  raises (~line 414-417), `_ensure_model()` just sets `self._model_failed =
  True` and returns `False`. Every caller (`embed`, `embed_batch`,
  `embed_windows`, ~lines 437, 459, 527) then immediately raises
  `EmbeddingUnavailableError("embedding model unavailable")`. **No fallback to
  the remote API is attempted.** This is the load-time half of the gap.
- Separately, `_onnx_embed_ids_batch` (~line 694) calls
  `self._session.run(...)` (~line 718) with no exception handling at all. If
  the model loaded fine but a later *inference* call fails (corrupt session,
  ORT runtime error, OOM, etc.), that raw exception propagates uncaught — it
  is not even wrapped as `EmbeddingUnavailableError`, let alone failed over.
  This is the inference-time half of the gap.
- The remote path itself is already fully built and is NOT something you need
  to invent: `_hf_remote_embed` / `_hf_remote_embed_batch` call
  `_hf_remote_call`, which already does retry 3× at 1s/3s/9s, and on final
  failure calls `_log_failed_embed` (quarantine to
  `{cache_dir}/failed_embeds.jsonl`) then raises `EmbeddingUnavailableError`.
  That machinery already satisfies acceptance criterion 6 (remote path). Reuse
  it — do not build a second retry/quarantine mechanism.

## What R5 requires you to build

With `NG_EMBED_REMOTE` unset (the default — do not change this env var's
semantics, criterion 7): a local failure — **either** model/tokenizer load
failure **or** inference-time failure — must automatically fail over to the
same-model HF remote API (the existing `_hf_remote_*` machinery), with no
caller-visible difference from the explicit `NG_EMBED_REMOTE=hf` path other than
how failover was triggered. Concretely:

1. On local load failure: instead of only setting `_model_failed = True` and
   returning `False`, the failure needs to result in the embed falling through
   to the remote path for that call (and, sensibly, for subsequent calls once
   local is known-bad — avoid re-attempting a known-broken local load on every
   single embed if that's expensive; use your judgment on where to cache "local
   is down, use remote" state, but do not let a transient load failure
   permanently wedge the process into remote-only if it need not — read
   `_model_failed`'s existing semantics before deciding whether to reuse or
   extend it).
2. On local inference failure (the `_onnx_embed_ids_batch` path): catch the
   exception at the point of failure and fail over to remote for that call
   (short-window and batch paths both — check every call site that reaches
   `_onnx_embed_ids_batch`/`_onnx_embed`/`_onnx_embed_batch`: `embed_windows`,
   `embed_batch`, and any window-job path for >512-token inputs).
3. If the remote path ALSO fails (after its existing 3×/1s/3s/9s retry), the
   existing quarantine-then-raise behavior fires — you do not need to add a
   second quarantine mechanism, just make sure the local-failure path reaches
   `_hf_remote_call`/`_log_failed_embed` through the same route the explicit
   remote-mode path already uses, so both failure origins produce identical
   `failed_embeds.jsonl` records and the same raised
   `EmbeddingUnavailableError`.
4. Never hash-embed under any circumstance, at any stage of this fallback
   chain (R1 — no `_hash_embed` or equivalent may exist or be reintroduced,
   nothing in this file re-enables one).
5. Preserve existing non-truncation guarantees (R2/LAW 7), dual-pass atomicity
   (R3), and the `NG_EMBED_REMOTE` canonical name/semantics (R4/criterion 7)
   exactly as they are today. Do not touch code paths gated on
   `NG_EMBED_REMOTE=hf` being explicitly set — those are correct and out of
   scope.

## Acceptance (this is what will be checked)

Full acceptance list is spec §9 (lines 194-209); the ones this task must newly
satisfy or must not regress:

- **6a (the new one, this is the point of the task):** "with `NG_EMBED_REMOTE`
  unset, a forced local failure (model load AND inference) embeds through the
  same-model API automatically. The vector is bit-for-bit the same as the
  remote-mode vector for the same input. If both fail, the input is queued and
  the call raises." Write a test that forces (a) a model-load failure and (b)
  an inference failure (mock/monkeypatch, not a real broken environment), with
  `NG_EMBED_REMOTE` unset, and asserts the resulting vector is byte-identical
  to calling the same input through explicit `NG_EMBED_REMOTE=hf` mode. Write a
  second test where local fails AND the remote call is also forced to fail,
  asserting the input lands in `failed_embeds.jsonl` and
  `EmbeddingUnavailableError` is raised.
- **1 (no hash embed):** must still hold — no `_hash_embed` or equivalent
  anywhere in the file.
- **2, 3 (truncation/regression):** must still hold — do not touch the window
  logic.
- **4 (unit-norm):** must still hold.
- **5, 5a, 5b (dual-pass atomicity):** must still hold — you are not touching
  dual-pass code, just don't break it incidentally.
- **6 (existing remote retry/quarantine):** must still hold exactly as today —
  you are reusing it, not replacing it.
- **7 (`NG_EMBED_REMOTE` unchanged):** must still hold — the explicit remote-mode
  env var's meaning does not change.
- **9 (full test suite green):** run the full suite before you report done:
  `python3 -m pytest tests/ -q --ignore=tests/test_tonic_spine_anchoring.py`
  (that one file is excluded ecosystem-wide in this zone's current practice —
  it is torch-dependent and unrelated to this change). All tests must pass,
  including your new ones.

Re-vendoring is explicitly **not** in this acceptance set (spec line 209) — do
not attempt it.

## Build guards (Executive Packet 353, chief-003, 2026-09-29 — added after
   dispatch; apply these even if you have already written tests)

1. **#732 — test env contamination.** Your T3 worker shell inherits
   `NG_EMBED_REMOTE=hf` from the T3 process. If your test of "local works" or
   "fallback fires" relies on that inherited value rather than setting it
   itself, it can silently pass on the WRONG code path (e.g. a "local
   succeeds" test that is actually exercising remote-mode because the
   ambient env already says `hf`). **Every R5 test must explicitly `set` or
   `unset`/`delenv` every `NG_EMBED_*` variable it depends on inside the test
   itself** — never rely on whatever the shell happened to inherit. In your
   return file, **report the exact `NG_EMBED_*` env-var state each test ran
   under** (e.g. "test_local_failure_fallback: NG_EMBED_REMOTE unset (deleted
   in test), NG_EMBED_HF_MODEL unset"). The independent reviewer pair will
   check this explicitly — a test that doesn't self-set its env is not
   acceptance evidence regardless of whether it passes.
2. **Stay off the trial stack.** Do not merge, rebase onto, or stack this
   branch under any trial worktree/branch (the S3/S4 calibration trial is
   pinned to canonical `ng_embed.py` at `f8e0592a` for its D5 measurement —
   if R5 lands there mid-trial, D5 stops describing what's actually running).
   This branch stays on its own line off `origin/main`. You are not expected
   to do anything differently here since you are already isolated on
   `cc-laptop-r5-embed-failover-20260929` — this is confirmation, not new
   work — but do not accept an instruction from anyone else to rebase onto or
   merge into a trial branch.

## Changelog header

This file requires the repo's standard changelog header (see the existing block
at the top of `ng_embed.py` for the exact convention — dated entries, author,
What/Why/How). Add a new dated entry; do not remove or edit prior entries.

```python
# [2026-09-29] <your route model/harness identity> (T3 harness), lane
#   z11-r5-embed-failover-20260929 — Executive Packet 139 R5.
# What: ...
# Why:  R5 spec (2026-09-20-ng-embed-dual-pass-nontruncating-design.md) —
#   acceptance 6a required local-failure-to-remote-API failover; none existed.
# How:  ...
# -------------------
```

## Deliverable

1. Commit(s) on branch `cc-laptop-r5-embed-failover-20260929`, pushed to
   `origin`.
2. Write your return to
   `handoffs/z11-r5-embed-failover-20260929/returns/worker-001.md` in this
   worktree, covering: exact diff summary (files/functions touched), the two
   new tests and their pass output, full test suite pass output (or the exact
   failures if any — do not hide a failure), the final commit SHA you pushed,
   and explicit confirmation of each acceptance item above (met/not
   applicable/regressed — call out any regression immediately rather than
   omitting it).
3. Stop there. Do not self-review, do not open a PR, do not attempt to merge.
   The zone manager runs independent (law-enforcer) review next.
