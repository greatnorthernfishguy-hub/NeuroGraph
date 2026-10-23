# Assignment worker-002 — R5 CORRECTION PASS (lane `ng-embed-remaining-work-20260922`)

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 · **Granted by:** chief-003 (2026-09-30), after the R5 review pair
- **Branch / worktree:** `cc-laptop-r5-embed-failover-20260929` · `/home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929`, currently at `0f5ca4c839db9b69a03408544dd30669fcfc6ad8` (your first delivery). Your correction is a **new commit on top**; do not rewrite history, do not force-push.
- **Authority:** `worktree_write`, exactly **two files: `ng_embed.py` (canonical, LAW 2) and `tests/test_ng_embed_r5_failover.py`**. Commit and `git push origin cc-laptop-r5-embed-failover-20260929` (normal push). **No PR, no merge** (P329/P337 hold and the trial end). Do not commit any `handoffs/` file.
- **Return to:** `handoffs/z11-r5-embed-failover-20260929/returns/worker-002.md` (UNTRACKED, do not commit), then stop. Do not self-review.

## Why (the review pair's verdicts, both read by the zone manager; verified against the code)

- Cross-family (grok-4.6): ACCEPT-WITH-CORRECTIONS. **Confirmed HIGH:** your 3 tests all call `emb.embed(...)`, so only ONE of the four new inference-failover sites is exercised (the `embed_windows` short path). `embed_batch` short (`ng_embed.py:528-538`), `embed_batch` long (`:561-574`) and `embed_windows` long (`:642-658`) can have their wrap deleted and every test stays green.
- Law-enforcer: COMPLIANT / APPROVE-WITH-NOTES. Notes N2: the failover warning has no traceback; cross-family #3: `_model_failed` is set only on load failure (`:448`), not on inference failover, though the new docstring (`:377-379`) calls it a "local has failed at least once" diagnostic flag.
- Full review text: `/home/josh/docs/handoffs/z11-ng-capability-port-20260923/reviews/checker-001-r5-crossfamily.md` and `.../le-001-r5.md`; adjudication `.../returns/zone-return-012-r5-review-pair-adjudication.md`. Read the findings for C1/C2 before you start.

## What to build — exactly this, nothing more

**C1 — tests for the three untested sites** in `tests/test_ng_embed_r5_failover.py`:
- `embed_batch`, short-index branch: local ONNX inference raises → vectors **bit-for-bit identical** to the explicit `NG_EMBED_REMOTE=hf` ground truth (reuse your `_remote_mode_vector` helper pattern).
- `embed_batch`, long-index branch (input needing more than one window, >512 tokens): same assertion.
- `embed_windows`, long path (window loop): same assertion.
- **At least one double-failure test on one of these new sites:** local inference fails AND `_hf_post` fails 3x → `EmbeddingUnavailableError` raised, a `failed_embeds.jsonl` record with `attempts: 3` (same assertions as your existing double-failure test), `time.sleep` patched so it is fast.
- **Per-test `NG_EMBED_*` discipline (Packet 353 / #732):** every new test itself `monkeypatch.delenv/setenv` every `NG_EMBED_*` it depends on and asserts the state immediately before the call under test; do not rely on the autouse fixture alone. **Report the exact NG_EMBED_* names** each test ran under (names only).
- **Negative control for each new test (mutation check):** prove each new test FAILS when its corresponding wrap is removed. Do it in a **scratch copy outside the repo tree** (e.g. `git clone` this worktree into `/tmp/...`, delete the one wrap there, run the test, expect failure, then delete the scratch). **Never leave a mutated `ng_embed.py` in the worktree.** State the result per site.

**C2 — two small edits in `ng_embed.py`:**
1. `exc_info` on the failover warning in `_fail_over_to_remote` (~`:465-468`): the log line must carry the traceback of the triggering exception (`exc_info=` with the exception you already receive). Nothing else about the log changes (still once, still WARNING).
2. Make `_model_failed` consistent with its docstring: set `_model_failed = True` inside `_fail_over_to_remote` (so it is true after ANY failover, load or inference), or correct the docstring if you find that setting it would change behavior. **First grep every read of `_model_failed` in the repo and report them**; do not change any behavior that depends on it. Add one test asserting the flag after an inference failover.
3. Update the R5 changelog entry in `ng_embed.py` to record this correction (repo convention).

## Explicitly NOT in this pass (separate rows; do not touch)

`_hf_remote_call` behavior for a missing HF token / tokenizer-load failure (#764), permanent one-way failover policy (#766, a Josh decision), the always-`True` `_ensure_model()` ripple incl. `reembed_snowflake.py:78` (#763), and the broad `except Exception` masking question (#765). Do **not** narrow the `except`, do **not** edit `reembed_snowflake.py`, `universal_ingestor.py` or any other file. If you believe one of these needs a change to satisfy C1/C2, stop and say so in your return.

## HARD safety rules

1. Every pytest command via the guard: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons -p no:cacheprovider` (it redirects and blocks the real Commons checkpoint path). **Never** run `tests/test_coordinator.py`, `tests/test_tonic_bridge.py`, any `handle_after_turn` test, or the **full suite** (org-wide one-at-a-time; not your slot). Targeted files only: `tests/test_ng_embed_r5_failover.py` and `tests/test_ng_embed_dualpass.py`.
2. The real file `~/NeuroGraph/data/checkpoints/commons.msgpack` must stay byte-identical (size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`, sha256 `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`); show `stat` + `sha256sum` before/after. Do not touch or read `~/NeuroGraph/data/` otherwise.
3. Write scope: the two files, the branch, and your untracked return. **Nothing to the vault (`~/docs`), nothing to any auto-memory directory** (that store is shared across seats). No secrets or tokens printed.
4. Suite wording (P375): the suite is red on clean `origin/main` (72 known failures, row #761). Never write "suite green"; if you mention it, say "no new failures versus the known-red baseline (72)" — and you did not run the suite, so say that plainly.

## Acceptance

1. 3 new site tests + 1 double-failure + 1 flag test pass; the original 3 R5 tests and the 31 dualpass tests still pass (report counts).
2. Each new test proven to fail when its wrap is removed (scratch copy), with the result stated per site.
3. Failover warning carries a traceback (a test asserts `exc_info` present on the log record); `_model_failed` consistent after any failover; all `_model_failed` readers listed.
4. Diff limited to the two files; new commit on top of `0f5ca4c`; pushed; `git status -sb && git log -3` and `git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929` shown in the return.
5. Real commons file byte-identical before/after.

## Return contents (`returns/worker-002.md`)

Files changed with line refs · every command run with the `NG_EMBED_*` names it ran under · counts before/after · negative-control result per site · the `_model_failed` readers · real-file stat+sha256 before/after · commit SHA + push confirmation · anything you did not verify or that surprised you (list it; do not act on it).

---

## ADDENDUM B (Executive Packet 376, chief-003, 2026-09-30) — SUPERSEDES "Explicitly NOT in this pass" for #763 and #765 ONLY

Scope of this one bounded turn is now **C1 + C2 + #763 + #765**. Everything else in this brief is unchanged (hard safety rules, write scope, guard plugin, no PR/merge, untracked return, NG_EMBED_* names). Still NOT in this pass: #764 (`_hf_remote_call` behavior for missing HF token / tokenizer-load failure; its own lane `hf-remote-6a-764`) and #766 (permanent one-way failover; a Josh decision, the failover stays as designed).

**Allowed files are now: `ng_embed.py`, `tests/test_ng_embed_r5_failover.py`, and `reembed_snowflake.py` (the single call site at ~`:78` only).** If `tests/test_ng_embed_dualpass.py:238`'s skip guard is confirmed dead by the same cause, you may make the equivalent one-line swap there and say so; otherwise leave it and report.

**#763 — restore a truthful `_ensure_model()` (R5's own regression).** Since R5, every `return` in `_ensure_model()` is `True`, so a caller that REQUIRES the local model can no longer learn that local is unavailable: `reembed_snowflake.py:78` (`if not emb._ensure_model(): ... sys.exit(1)`) would silently re-embed Syl's `vectors.msgpack` over the network. Fix: give callers an explicit way to ask "local or nothing" (a `require_local`-style parameter or a separate method, or a return that says which path was taken; your choice, keep the smallest change that is honest). Requirements: (a) existing R5 callers (`embed`/`embed_batch`/`embed_windows`) keep the R5 failover behavior; (b) a `require_local` caller that finds local unavailable gets a clear False/raise and NO failover and NO network call; (c) `reembed_snowflake.py:78` uses it and aborts again if local is unavailable; (d) tests for (b) and (c) (for `reembed_snowflake.py` a small test that the abort path is taken, no network). Grep every caller of `_ensure_model` (report them), and treat the guards at `:489`, `:511`, `:598` deliberately (leave, remove, or rewire, and say why).

**#765 — narrow the failover `except`.** Today every failover site catches bare `Exception` (`:447`, `:534`, `:567`, `:617`, `:651`), so a bug-class local defect (`TypeError`, `IndexError` on `outputs[1]`, `AttributeError`, ...) is masked behind a successful remote vector. Change it so **only environment/runtime failures fail over** (ONNX Runtime errors, `MemoryError`/OOM, `OSError`/file-not-found, model/tokenizer load failures, `ImportError` of the runtime) and a **bug-class defect is raised, not swallowed, and logged at ERROR with `exc_info`**. Every actual failover keeps logging one WARNING with `exc_info` (C2). Design tension you must resolve and state in your return: an allowlist that is too narrow re-introduces the original R5 bug (a real environment failure raises instead of failing over), a denylist of bug-class exceptions is safer against that. Pick one, justify it with the exception classes you actually verified (`onnxruntime` raises what? `hf_hub_download`/`tokenizers` load raise what?), and test BOTH directions: an ORT-style/OSError/MemoryError-type failure fails over (bit-identical, as before), and a `TypeError`-style defect is raised and logged and does NOT reach the remote. Keep `except EmbeddingUnavailableError: raise` as is.

**One turn or split, honestly.** This is more than the first brief. Work in this order and commit at a CLEAN boundary: **C1 -> C2 -> #765 -> #763**. If you cannot finish all of it in this turn, commit and push what is complete and tested, and state exactly what remains in your return; do NOT drop any item and do NOT commit half-working code. The zone manager will send the remainder as a second turn.

The delta-only law-enforcer re-check will now cover the whole correction diff (it is small); keep the diff minimal and the changelog entry in `ng_embed.py` accurate.
