# Assignment worker-003 — R5 C3 (F1 from the delta law-enforcer) — lane `ng-embed-remaining-work-20260922`

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 · **Granted by:** chief-003 (2026-09-30): ONE bounded turn, F1 fix + test ONLY.
- **Branch / worktree:** `cc-laptop-r5-embed-failover-20260929` · `/home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929`, now at `c7c275360c8cf2639e88d3f4f5a531f15a64e5d8`. New commit on top; no history rewrite, no force-push; **no PR, no merge**; do not commit any `handoffs/` file.
- **Allowed files:** `reembed_snowflake.py`, `tests/test_ng_embed_r5_failover.py`, and `ng_embed.py` **only if the fix genuinely needs it** (it probably does; see below). Nothing else.
- **Return to:** `handoffs/z11-r5-embed-failover-20260929/returns/worker-003.md` (UNTRACKED), then stop. No self-review.

## The finding (delta LE F1, confirmed by the zone manager against your code)

`reembed_snowflake.py:87` now uses `_ensure_model(require_local=True)` and aborts if local is unavailable **at load**. But `reembed_snowflake.py:128` still calls the **default** `emb.embed_batch(batch_texts)`. If local *inference* hits an environment-class error mid-run, `_fail_over_or_raise` flips `_remote_mode` and `embed_batch` **re-issues that same batch to the HF remote API inside the same call**; every remaining batch then also goes over the network with one WARNING. That contradicts the tool's own abort text ("this tool never re-embeds Syl's vectors over the network") and it is untested. #763 is therefore closed only at load time.

## Two remedies, and which one is acceptable

1. Check `emb._remote_mode` after each `embed_batch` and abort. **Insufficient on its own:** the failover already made the remote call inside `embed_batch`, so that batch's text has already been sent to the network by the time the tool could look.
2. **Make the inference path honor `require_local`** so a `require_local` caller can never fail over: **this is the required shape** unless you demonstrate it cannot be done minimally (then say so plainly and stop rather than ship remedy 1 as if it were enough).

Requirements for remedy 2:
- `embed_batch` (and `embed_windows`, so the two public batch entry points behave the same; `embed()` if it shares the code path) accept `require_local: bool = False`. Default behavior for every existing caller is **unchanged** (R5 failover as built).
- With `require_local=True`, an **environment/runtime-class** local inference failure (the same classification the denylist already makes) **raises `EmbeddingUnavailableError`** (loud: log the traceback via the existing logging, same one-WARNING-or-ERROR convention you used), makes **NO** call to `_hf_remote_*`/`_hf_post`/`urllib`, does **not** set `_remote_mode`, and does not change `_model_loaded`. A **bug-class** defect still raises as before (ERROR + traceback), not masked. `EmbeddingUnavailableError: raise` carve-out unchanged.
- If `_remote_mode` is already True when a `require_local` call arrives (e.g. `NG_EMBED_REMOTE=hf`, or an earlier default-caller failover in the same process), a `require_local` call must **refuse** (raise) rather than use the remote path. State how you handled it.
- `reembed_snowflake.py`: call the batch entry point with `require_local=True`; on `EmbeddingUnavailableError` from a batch, **abort with a non-zero exit BEFORE writing that batch or any later one**. First read how the tool writes (incremental per batch, or once at the end) and state exactly what is and is not on disk after an abort; keep `--dry-run` semantics untouched. Also keep the load-time abort as is.

## Tests (in `tests/test_ng_embed_r5_failover.py`)

- `embed_batch(..., require_local=True)` and `embed_windows(..., require_local=True)`: env-class inference failure -> raises `EmbeddingUnavailableError`; **asserts `_hf_post` and `urllib.request.urlopen` were never called**, `_remote_mode` still False, no failover warning; bug-class still raised as the original object. Default (`require_local=False`) still fails over bit-identical to explicit remote (regression pin).
- `_remote_mode` already True + `require_local=True` -> refuses.
- **The tool-level test:** run `reembed_snowflake.main()` (as your existing #763 test does: temp vectors file, `--dry-run` so a regression cannot write) with local inference made to fail on batch N of M -> exit code non-zero, **zero remote calls**, nothing from batch N or later reaches the output path, file bytes unchanged.
- **Negative controls in a scratch clone outside the repo** (never mutate the worktree): revert `reembed_snowflake.py` to the default `embed_batch` call -> the tool-level test must fail; remove the `require_local` branch in `embed_batch`/`embed_windows` -> the unit tests must fail. Report per mutation.
- Per-test `NG_EMBED_*` discipline exactly as before (set/unset inside each test, assert immediately before the call, report NAMES).

## HARD rules (unchanged, plus the new one)

1. **Interim rule (Packet 379, #770):** every pytest command via the guard plugin v4, which now prints `neurograph_rpc.__file__` and the WORKTREE/MAIN origin of NG modules after collection and again at session end: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons -p no:cacheprovider`. **Paste the `[z11 origin]` lines of your final runs into the return**; a run without them does not count. If any line warns that a module resolved from the MAIN checkout, re-run and say so.
2. Targeted files only: `tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py`. **Never** the full suite, `test_coordinator.py`, `test_tonic_bridge.py`, any afterTurn test.
3. Real `~/NeuroGraph/data/checkpoints/commons.msgpack` byte-identical (size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`, sha256 `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`); show before/after. Do not read `~/NeuroGraph/data/` otherwise. Never run `reembed_snowflake.py` against real data: only via the temp-file test.
4. Write scope: the files above, the branch, your untracked return. Nothing to the vault (`~/docs`), nothing to any auto-memory directory. No secrets printed.
5. Suite wording (P375): never "suite green"; say the suite was not run, and that clean `origin/main` is red (72 known failures, #761).
6. Changelog headers/entries in every file you modify (new dated entry; never edit prior entries).
7. **NOT in this turn:** #764 (`_hf_remote_call` token/tokenizer), #766 (permanent one-way failover; Josh's), #772 (`cc_ng_host` catch), F2/#771 hub-version and `cpu_count` items. Do not fix them; if C3 forces contact with one, stop and say so.

## Acceptance

1. A `require_local` inference failure cannot cause any network call or failover; default callers unchanged (tested both ways).
2. `reembed_snowflake.py` aborts non-zero before writing the failing batch or any later one; what is on disk after an abort is stated.
3. Mutation controls fail as expected for each new test.
4. Counts before/after (baseline now: 71 passed in the two files); diff limited to the allowed files; new commit pushed on top of `c7c2753`; `git status -sb`, `git log -3`, `git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929` shown.
5. Real commons file byte-identical.

## Return contents

Files changed with line refs · the design you chose and why (and if you could not do remedy 2, why) · what is on disk after an abort · every command with the NG_EMBED_* names it ran under · the `[z11 origin]` lines · counts · mutation results · real-file stat+sha256 before/after · commit SHA + push confirmation · anything surprising or unverified.
