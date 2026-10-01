<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane poison-quarantine-920, dispatch #13303) — build-001 return for #920 (Exec Packet 497 via Chief-003)
#   What: the builder return: the poison-turn policy (bounded retry -> quarantine, never discard -> loud) on the NG drain and the docs daemon,
#     plus the LOCK-SEMANTICS verification table and the one part that is STOPPED (it needs a change to the vendored ng_embed.py).
#   Why: Exec Packet 497; gates setting CC_NG_DRAIN_HOLD_ON_FAILURE (with #914). FULL PAIR to follow.
#   How: facts below come from commands run in the two worktrees; every hash is a real `git rev-parse HEAD`; what was NOT run or verified is listed.
# -------------------
-->

# #920 build-001 return: bounded retry -> quarantine -> loud

Related: [[NeuroGraph]] · [[Dual-Pass Embedding]] · [[Vendored Files]] · [[The Laws]] · [[NG Tract Bridge]]

## 0. READ FIRST (what the Chief must know in 60 seconds)

1. **Delivered:** the full policy (parts 1-4 of the brief), in both repos, local commits only, nothing pushed or merged. New tests: 43 (NG) + 56 (daemon), all passing; 23 seeded mutants, all killed.
2. **The LOCK part is NOT delivered: STOPPED, as instructed.** Verified: the slow TID / embed work **does** run under `graph._concurrent_lock` at every caller. The split needs a change to the **vendored `ng_embed.py`**, which I did not touch. The exact canonical change is in section 4.
3. **One real defect found and fixed in my own first draft by a test:** the "frame already at the tail of the quarantine file" shortcut skipped the fsync, so after a failed fsync the next cycle would have advanced the tract on page-cache-only data. Now fsync of the file AND its directory happens on every ok return. (Test `test_a_frame_found_at_the_tail_is_fsynced_again_before_the_tract_advances`; mutant M9 is exactly this bug and is killed.)
4. **Biggest consequence (stated, not decided): a TID outage quarantines live turns.** At the drain a failed turn is just `False`, so "TID down for 5 cycles" is indistinguishable from "poison turn". With default 5 and a 60 s autosave, every 5 minutes of TID outage moves one good turn into the quarantine file. Nothing is lost, but re-ingest is not built (section 6.1).
5. **`redact_node_id` is NOT in this base** (it is slice #905, commit `ee94f7d2`, not an ancestor of `895a809`). At drain time there is no node id anyway, so the ERROR uses `entry:<12 hex of sha256(raw frame bytes)>` only. No dependency on an unmerged helper.
6. **Three existing tests were edited** because the policy intentionally changes what they pin (a signature's last parameter x2, and the set of new env names x1). Listed in section 5.

## 1. Hashes (real `git rev-parse HEAD`)

| Repo | Worktree | Branch | Base | Code commit |
|---|---|---|---|---|
| NG | `/home/josh/NeuroGraph-worktrees/z12-quarantine-920-20261001` | `cc-laptop-quarantine-920-20261001` | `895a809a60ab86afacd8ef193b95c4ab39f6893f` | `8b3225f5bbcdd3b0815ed39e46e513471de7a08e` |
| docs daemon | `/home/josh/docs/.claude/worktrees/daemon-quarantine-920-20261001` | `cc-laptop-daemon-quarantine-920-20261001` | `6b5555022b6a31dcca417a6e24e4fcaa92c2d150` | `45f57f86390ac962949851bfd8a5757179bc517a` |

The NG return commit (this file + `build-001-mutants.py`) is the NG branch HEAD after the code commit; its hash is in the builder's final message (a file cannot contain its own commit hash). Neither branch has an upstream. `git fetch -q` was run once in the NG worktree (allowed by the brief); nothing was pulled, pushed, merged or rebased. `git diff --name-only 895a809a -- ng_embed.py neuro_foundation.py openclaw_hook.py stream_parser.py activation_persistence.py` is empty (also asserted by a test).

## 2. What changed (file:line, at the code commits)

### NG `cc_ng_organism.py` (+208 lines, one deletion = the signature line)
- `:2604` `_DRAIN_RETRY_TABLE_MAX = 256` and the helpers `:2608` `_drain_entry_tag`, `:2615` `_drain_retry_bump`, `:2628` `_drain_retry_forget`, `:2640` `_drain_quarantine_mode`, `:2651` `_drain_quarantine_frame`, `:2693` `_drain_bounded_hold`, all directly above the drain.
- `:2736` `drain_ingest_tract(..., hold_on_failure=False, max_retries=None)`: ONE new LAST keyword. Docstring paragraph added.
- `:2855` the bound is live only with `hold_on_failure and max_retries is not None` and a positive int (a non-positive / non-int value = ONE fixed WARNING per call and behaves as `None`); `:2869` `entry_start` (first byte of this entry's frame); `:2892` count cleared on a successful absorb; `:2904` the bounded branch inside the existing hold block; `:2973` the key is forgotten only after the truncate is confirmed to have removed the frame.
- New imports `hashlib`, `stat` (module level). Changelog header entry added.

**Behaviour, exactly:**
- A held entry's failures are counted in `state["drain_retry_counts"]` (caller-owned dict, **in memory**, key = full sha256 hex of the entry's raw frame bytes, never the text; hard-capped at 256, oldest evicted; a restart resets it, so **the bound is per process lifetime**).
- **Count semantics (my reading, please confirm):** "failed `max_retries` times" = the `max_retries`-th failing attempt quarantines **in that same cycle** (`max_retries=5` is five attempts, one per cycle, i.e. four retries after the first failure). The brief's test line "retried exactly `max_retries` times" can also be read as `max_retries` retries after the first attempt (six attempts); that is the one-character change `failures < max_retries` to `failures <= max_retries` in `_drain_bounded_hold`. I implemented the policy sentence ("When an entry has failed `max_retries` times it is moved").
- Below the bound: the existing #794 hold WARNING plus ` attempt=n/N` (only when the bound is set). At the bound: `_drain_quarantine_frame` appends the exact frame WHOLE to `<tract path>.quarantine` (`O_WRONLY|O_APPEND|O_CREAT`, never `O_TRUNC`, no extra framing; `write`, `flush`, `os.fsync(file)`, `os.fsync(directory)` on **every** ok return), and only then `safe_offset` moves past the entry and the post-loop truncate runs. The drain **stops after a quarantine** (one poison event per call; the entries behind it are drained on the next call, as the brief says).
- File creation mode: the tract's DIRECTORY mode with execute bits dropped, and never wider than the tract FILE's own mode (it holds the same raw turn bytes), 0600 if either is unreadable. (My narrowing of "created with the tract's directory permissions": a 0755 directory around a 0600 tract must not make the copy world-readable.)
- Failed quarantine write (open / write / flush / fsync / dir fsync): the entry stays HELD, the tract is not advanced, **one ERROR** `CC ingest-tract QUARANTINE FAILED`, retried next cycle.
- Idempotency: the file's LAST `len(frame)` bytes are compared with the frame (O(frame), no full-file read). Identical ⇒ not appended again (`stored=already_present`), but still fsynced. Safe because a frame embeds a per-deposit stamp: I wrote the same text twice with the real writer and the bytes differ, so two distinct turns can never collide. The counter increments only for a NEWLY appended frame.
- LOUD: one ERROR per quarantined turn: `entry:<12 hex>`, `reason=` (`absorb_returned_false` | `absorb_raised`), `exc_type=` (the class name when it raised; `-` when the dual pass returned False, because that class is swallowed inside the dual pass and appears in the dual pass's own WARNING), `failures=`, `max_retries=`, `file=<name>.quarantine` (name only), `stored=`. Never the text, never a path, never an id, never `str(exc)`. `state["quarantined_total"]` (int) counts newly appended frames.
- `return_consumed`'s span **includes** a quarantined frame (it did leave the tract; the contract is "EXACTLY what was truncated"). See consequence 6.3.

### Docs daemon `scripts/cc-ng-daemon.py` (+91 lines)
- `:1006` `DRAIN_MAX_RETRIES_DEFAULT = 5`; `:1009` `_read_drain_max_retries_env(environ=None, hold=None)` (pure, never logs; docstring states the default); `:1031` `DRAIN_MAX_RETRIES, DRAIN_MAX_RETRIES_GARBAGE = _read_drain_max_retries_env()` read ONCE; `:1034` `_quarantine_on()` (hold flag on AND a bound set: the one predicate for all three sites); `:1041` `_warn_drain_max_retries_garbage()`.
- Env semantics: hold flag OFF ⇒ no bound whatever the variable says. Hold ON: unset/empty/blank ⇒ 5; positive integer ⇒ itself; non-integer / 0 / negative ⇒ 5 + the garbage flag. **The ONE WARNING (names the variable only, never its value) is emitted from `main()` at `:3165`, right after `_setup_logging()` and the start line**, so it reaches the daemon FILE handler.
- `:1092` `stats['quarantined_total'] = 0` seeded only when `_quarantine_on()`; `:2701-2704` `_hold_kw['max_retries'] = DRAIN_MAX_RETRIES` only when `_quarantine_on()` (both the unpaced and D24-paced call share `_hold_kw`); `:2626` `_mirror_quarantined_total()` and its call at `:2738-2739`, **after** the held section's `finally` has released `_concurrent_lock` (STATE.lock is taken only when this thread holds nothing; a raise is one ERROR with the class name only and never breaks the cycle).
- **Counter path (the smallest clean way, as asked to state):** the drain writes the caller-owned `conv_state['quarantined_total']` (the #904 `bind_site_failures` idiom) and the daemon mirrors it into `STATE.stats` (the existing `status` reply). The drain's return value and D24's `receipt` are untouched: both are on the merge front.

## 3. Tests

**New:** NG `tests/test_cc_poison_quarantine_920.py` (27 test functions, 43 collected), docs `scripts/tests/test_cc_ng_daemon_poison_quarantine_920.py` (17 functions, 56 collected). Real `Graph` + `SimpleVectorDB` + the real vendored `NGEmbed.dual_record_outcome` (model/TID seams only replaced on the instance) + the real drain on real `ng_tract` frames in pytest tmp files. The live tract is never opened (every tract path asserted to be a pytest tmp path). Covered: every-cycle poison (max_retries 1, 3, 5; first-entry poison; a raising dual pass); fsync-before-truncate spy (at every fsync the tract still holds the frame); fail-twice-then-succeed; three injected quarantine faults (target is a directory, `os.open` raises, `os.fsync` raises) held for 3 cycles then recovered exactly once; crash between fsync and truncate (a `BaseException` at the truncate, then a restart with fresh memory); a soft truncate failure; never created / never truncated / append-only across calls; the file parses as a tract; mode; key = sha256 of bytes; 300 distinct failing frames vs the cap; restart resets; invalid bound; `max_entries`; **live differential against the BASE module** (6 scenarios: returns, tract bytes, graph snapshot, every log record, state keys; not vacuous); the real daemon loop on real NG.

### Runs (scratch HOME `/tmp/z12-920-home`, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `NG_EMBED_REMOTE` unset, printed-path preamble each time, one run at a time, every run after the first gated on `/proc/loadavg < 6`)

| File | On the fix | Notes |
|---|---|---|
| NG `tests/test_cc_poison_quarantine_920.py` | **43 passed** (daemon env set); 41 passed + 2 skipped without it | final state after the last code edit |
| NG `tests/test_cc_bind_atomic_904.py` | **44 passed** (daemon env set) | 1 failed before the pin edit (section 5) |
| NG `tests/test_cc_drain_hold_on_failure.py` | **49 passed** | 1 failed before the pin edit |
| docs `test_cc_ng_daemon_poison_quarantine_920.py` | **56 passed** | |
| docs `test_cc_ng_daemon_deposit_904.py` | **62 passed**, pacing UNSET and SET (25/250) | 1 failed before the pin edit |
| docs `test_cc_ng_daemon_drain_pacing.py` | **46 passed** unset, **46** set | unchanged file |
| docs `test_cc_ng_daemon_recall_status.py` | **41 passed** unset, **41** set | unchanged file |

**BASE counts (new files vs the base copies):** NG new file vs the 895a809 `cc_ng_organism.py` (git blob `bf88491f59d6`, scratch copy): **37 failed, 6 passed** of 43. The 6 are the tests designed to pass on both: unedited-vendored-files, four golden differentials, and the flag-off daemon-loop test. Daemon new file vs the 6b555502 daemon (blob `20956fec94c3`): **40 failed, 16 passed** of 56.

### Mutations, on SCRATCH copies under `/tmp/z12-920-scratch/mut/` (harness: `build-001-mutants.py`, asserts each replacement matches exactly once): **23 of 23 killed**

| Mutant | Failing tests |
|---|---|
| M1 no retry bound | 21 |
| M2 quarantine skipped (discard) | 20 |
| M3 tract truncated BEFORE the fsync | 17 (and the ordering test `test_fsync_happens_before_the_tract_is_truncated` on its own: killed) |
| M4 unbounded table | 1 (`test_the_retry_table_is_hard_capped`) |
| M5 text in the ERROR | 8 |
| M6 failed quarantine advances the tract | 7 |
| M7 not append-only (truncating open) | 7 |
| M8 no idempotency | 5 |
| M9 `already_present` skips the fsync (the hole in section 0.3) | 2 |
| M10 key from text, not bytes | 5 |
| M11 count not cleared on success | 2 |
| M12 quarantined but tract not advanced | 16 |
| M13 default #794 WARNING text changed | 2 (the byte-identity differential) |
| M14 counter also on `already_present` | 2 |
| D1 kwarg passed with hold off | 5 |
| D2 default 5 even with hold off | 11 |
| D3 bound read per call from env | 1 |
| D4 counter not mirrored | 4 |
| D5 garbage warning logged at import | 6 |
| D6 `main()` never warns | 1 |
| D7 zero accepted | 2 |
| D8 mirror inside the graph lock | 1 |
| D9 stat seeded with the flag off | 1 |

## 4. LOCK SEMANTICS: verified, table, decision

**Verification (read-only, at the base commits; cite `git show <base>:<file>`):** the Executive's rule is violated today everywhere the dual pass runs. Notably `_cc_mutation_lock` (`cc_ng_organism.py:1925`) returns `graph._step_lock` and its docstring says "Embedding runs between deposits", so the design intent already is "slow work outside, commit inside"; it is the callers' held `_concurrent_lock` that defeats it.

| # | Where | Slow call | Under which lock | Evidence |
|---|---|---|---|---|
| 1 | docs daemon autosave drain | per entry `embed(entry.content)` (ONNX, or HF HTTP if `NG_EMBED_REMOTE=hf`), then row 2/3 | **`_concurrent_lock`**, held from acquire to release | daemon `:2578` acquire, drain calls `:2621`/`:2627`, release `:2649`; NG `_apply_gateway_experience` `:2568-2572` |
| 2 | NG `run_conversational_dual_pass` | `embedder.embed_windows(text)` (tokenise + per-window embed) | caller's `_concurrent_lock` | NG `:2386` (function `:2348`) |
| 3 | same, into vendored `NGEmbed.dual_record_outcome` | `_extract_concepts` = **TID HTTP, `requests.post` timeout 30 s PER extraction window**; then `embed_batch(concepts)`; then the writes in the same call | caller's `_concurrent_lock`; the writes take `_step_lock` inside it (`_concurrent_lock -> _step_lock`, the established order) | NG `:2393`; `ng_embed.py` `:953` def, `:995` extract, `:1147`/`:1153` TID call, `:236` `tid_timeout: 30`, `:1023` `embed_batch`, `:1033`/`:1063` writes |
| 4 | docs daemon `_deposit` (hooks, blocking) | `_embed(text)` evaluated **inside** the lock, plus rows 2-3 | **`_concurrent_lock`** | daemon `:2090` def, `:2117` `with ...lock`, `:2119` `_embed(text)` |
| 5 | VPS host `_deposit` (Path B) | `_embed(text)` is **already hoisted before the lock**; rows 2-3 still inside | `_concurrent_lock` for rows 2-3 only | `cc_ng_host.py` `:655` embed, `:663` lock, `:664` dual pass |
| 6 | VPS host autosave drain | same as row 1 | **`_concurrent_lock`** | `cc_ng_host.py` `:1525` lock, `:1526` drain |

Consequence on the poison path: a TID-slow poison turn holds the lock up to 30 s x (extraction windows) every 60 s cycle. The #920 bound caps the **cumulative** lock time of such a turn at `max_retries` attempts; it does not shorten any single attempt.

**Design of the least invasive split (OPEN, needs a ruling; not built):**
1. **Canonical `ng_embed.py` (NeuroGraph, then re-vendor everywhere)**, default-preserving:
   a. A public `NGEmbed.prepare_dual_pass(content) -> DualPassPrep(concepts: Optional[List[str]], tree_embeddings: List[ndarray])` doing exactly lines `:995` and `:1023` (extract, then `embed_batch`), with no ecosystem and no write. `concepts=None` records the pass-2 failure.
   b. `dual_record_outcome(..., *, windows=None, prep=None)`: with `prep` given it skips `:995`/`:1023` and uses `prep`; **if `prep.concepts is None` it takes the identical existing failure path** (rate-limited warn + `signal_error` + `DualPassIncompleteError`, so R3 atomicity is unchanged). `prep=None` (default) is byte-identical to today.
   c. Re-vendor to TID, TrollGuard, Immunis, Elmer, THC, Bunyan, Praxis, Agent Zero together (CLAUDE.md section 4, LAW 2). I could not do this without editing a vendored file, and a subclass / proxy overriding `_extract_concepts` in `cc_ng_organism` would be "wrapping or patching a vendored file" (LAW 2); re-implementing the loop locally would be a competing implementation (LAW 3). So: STOPPED.
2. **NG `cc_ng_organism.py` (not vendored)**, after (1): split `run_conversational_dual_pass` into `prepare` (outside: `embed(text)`, `embed_windows`, `prepare_dual_pass`) and the commit (inside: the eco writes and bind), with an optional `prep=None` so every current caller is unchanged.
3. **Callers:** the daemon `_deposit` (row 4) hoists `prepare` before `with ...lock:` (the host already does this for `embed(text)`, row 5); the two **drains** (rows 1, 6) are the hard part: they run entirely inside a held lock. Per-entry prepare outside the lock needs the drain to receive the lock (or a two-phase API) instead of assuming the caller holds it. **That conflicts with D24's #896 rule 2** ("NO UNPACED DEPOSITS": the whole-graph guard is evaluated and the drain runs inside ONE hold so nothing changes in between). A split drain must re-evaluate that guard at each commit under the lock. This is an Executive-level design choice, not a builder one.
4. **Vendored-free sub-step available now, NOT delivered** (the brief delivers the lock part only if the whole of it needs no vendored change): hoist `_embed(text)` out of the lock in the daemon `_deposit` (copy the host's `:655` idiom, keeping it inside the same `try` so `dual_pass_raised` reporting is unchanged) and add an optional `windows=` to `run_conversational_dual_pass`. It removes the ONNX embed from the lock but **not** the TID call, which is the 30 s item, so it must not be mistaken for the fix.

## 5. Existing tests edited (3 assertions, each with a changelog line; please look at these)

- NG `tests/test_cc_bind_atomic_904.py::test_signatures_are_pinned`: `[-1] == 'hold_on_failure'` became `[-2:] == ['hold_on_failure','max_retries']`.
- NG `tests/test_cc_drain_hold_on_failure.py::test_signature_hold_on_failure_is_last_and_defaults_false`: the expected parameter list gains `"max_retries"` (name and intent unchanged).
- docs `scripts/tests/test_cc_ng_daemon_deposit_904.py::test_no_default_in_code_and_no_other_new_environment_variable`: the set of new env names vs the D24 base is now `{CC_NG_DRAIN_HOLD_ON_FAILURE, CC_NG_DRAIN_MAX_RETRIES}`.

All three failed first (expected), were read, and were updated minimally. Nothing else in those files changed.

## 6. Consequences (STATE, not decide)

1. **TID outage amplification** (section 0.4): the failure class is invisible at the drain. A permanent-looking outage quarantines one good turn per `max_retries` cycles. Options for the Chief (not chosen): a larger default, a wall-clock bound, or counting only cycles in which the vendored `NGEmbed._failures` (TID failure counter) did not rise. **Re-ingest is a deliberate later act and is NOT built.** Where it is: `<tract path>.quarantine`, i.e. by default `~/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract.quarantine` (from `CC_GATEWAY_TRACT_PATH`; I never opened it). What a re-ingest would need: it parses as-is with `ng_tract.TractReader` (a test proves it is a flat run of the tract's own frames); a tool that runs each frame through the dual pass under the lock after the cause is fixed; progress recorded in a SEPARATE sidecar offset file, because the quarantine file is append-only and is never truncated; idempotent re-runs come free from the content-hashed `target_id`s.
2. **Count off-by-one** (section 2): reading "failed N times" vs "N retries after the first attempt".
3. **Leg 1 conduit:** `return_consumed` includes the quarantined frame, so with `CC_CALLOSUM_LEG1_ENABLED` on, `trickle_gateway_conduit` would send the poison frame to the VPS, whose own drain would then try it. The real-NG daemon test records this (`trickled[2] == frames[1]`). Default Leg 1 is off. One-line option if ruled otherwise: exclude it from the returned span.
4. **Disk growth** of the append-only file is unbounded by design: at worst one frame per `max_retries` cycles per poison head (about 288 a day at 5 and 60 s). Frames seen elsewhere in this repo reach 722 KB, so worst case is the order of 200 MB a day under a sustained outage; typical turns are a few KB.
5. **Bound is per process lifetime:** the counts live in memory (`state`), so a daemon that restarts more often than every `max_retries` cycles never quarantines anything.
6. **Log volume per poison turn** (D24 and #904 interplay): instead of 2 WARNING lines a minute forever, at bound 5: 4 held cycles x (1 dual-pass WARNING + 1 hold WARNING) + the 5th cycle's dual-pass WARNING + 1 ERROR, then silence. A failed quarantine write adds 1 ERROR per cycle until it succeeds.
7. **Torn tail:** a quarantine write that fails midway (ENOSPC) can leave a partial frame at the file's end; nothing is truncated to hide it (the "never truncated" rule wins). The next successful append lands after it, so a re-ingest tool must tolerate one unparseable fragment.
8. **Stack-merge front:** this NG commit touches the drain signature, the docstring tail, the new helper block, the `safe_offset` area, the loop top, the success branch, the hold block and the post-truncate step. Hypothetical merge (`git merge-tree`, no ref touched): the #904 NG base alone already conflicts with the D24 NG commit `c798a8ed` in `cc_ng_organism.py` (**3** conflict hunks); with #920 on top it is **4**, so #920 adds one more. The daemon side already passes `batch_nodes` / `receipt` and shares one `_hold_kw`, tested with and without pacing in fakes; the combination with a real D24 drain was **not** built or run (D24 is not in either base).
9. **Observed, not fixed (punchlist candidate):** the #904 hold-flag garbage WARNING (`_read_drain_hold_env`) fires at import, before the file handler exists, so it reaches only stderr. I verified this against the base daemon (log file had 2 lines, none about the flag). `_read_drain_max_retries_env` avoids it by being pure; the hold flag's reader was left alone (#904's slice).
10. **`redact_node_id`** (#905) is not in this base; when it merges there is nothing to change here, since the drain has no node id to redact.

## 7. NOT verified / NOT done

- **Not run, by constraint:** the NG full suite and `tests/test_cc_deposit_step.py`. I also did **not** run the two existing NG files against the base code (my first "base counts" loop only copied files; I am not claiming base counts for them). The base counts above are for the two NEW files only.
- **Not verified:** that `os.fsync` reaches physical media (power loss), or behaviour on a filesystem other than the laptop's; concurrent miniTID appends *during* the quarantine step (the truncate path is unchanged and re-reads the file as before, but I did not race it); two drainers (the code assumes one); the VPS host `cc_ng_host.py:1526` call (untouched, default arguments, so unchanged, but not run); any combination with the real D24 drain.
- **Load:** the one-run-at-a-time rule was kept; the highest load average I observed was 5.73 (another session, at the start of one run), never at or above 6 at a gate. The first few runs were not gated (readings 2.6 to 4.3).
- **Not done, by brief:** no re-ingest code, no flag default flipped, no `.bashrc` / crontab / unit edit, no daemon start, no live graph / tract / checkpoint opened, nothing under Syl's directories, no VPS, no push, no merge. `ng_embed.py` untouched.
- **Docs vault:** not updated (the brief confines me to the two worktrees); wikilinks above are for the Chief's filing.
