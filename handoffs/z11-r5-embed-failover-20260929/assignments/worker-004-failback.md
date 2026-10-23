# Assignment worker-004 — R5 C4: fail-back (Josh's ruling on #766) — lane `ng-embed-remaining-work-20260922`

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 · **Source:** Executive Packet 382, Josh's answer on #766: failover lasts **only as long as local is down**; the code MUST **re-probe local and FAIL BACK**, **logging loudly BOTH ways**. Scoped by chief-003 into R5 as a small correction turn (not a new lane).
- **Branch / worktree:** `cc-laptop-r5-embed-failover-20260929` · `/home/josh/NeuroGraph-worktrees/z11-r5-embed-failover-20260929`, at `245d2bd1b865feebdb559ac8fe9e4671ccc3973d` (your C3). New commit on top; no history rewrite, no force-push; **no PR, no merge**; commit no `handoffs/` file.
- **Allowed files: `ng_embed.py` and `tests/test_ng_embed_r5_failover.py` ONLY.** Return (UNTRACKED): `handoffs/z11-r5-embed-failover-20260929/returns/worker-004.md`. Then stop; no self-review.

## Why (what Josh ruled and what is wrong today)

Today, once `_fail_over_or_raise` sets `_remote_mode = True` (`ng_embed.py` ~`:564`), it is **one-way for the life of the process**: a transient ORT/OOM error or a briefly missing model pins Syl's sidecar on the HF remote API (her text leaves the machine) until restart. Josh's ruling: failover must last only while local is actually down, must **re-probe local**, and must **fail back**, with a **loud log in both directions**.

## Requirements

1. **Distinguish the two remote modes.** Explicit `NG_EMBED_REMOTE=hf` (set at ~`:463`) is an operator choice: it **never** re-probes or fails back. Failover (set by `_fail_over_or_raise`) is the only mode that fails back. Record it explicitly (e.g. a `_failover_since` timestamp / episode marker) rather than inferring it. `_model_failed` keeps its documented meaning.
2. **Re-probe cadence, env-driven (LAW 5):** new env var **`NG_EMBED_REPROBE_SECS`**, default in code (choose and justify; the probe must be cheap and must not run on every call), `0` disables re-probing (an operator escape; say so in the log at failover time when disabled). Read it the way the file reads its other env vars. Report the exact name.
3. **The probe is LOCAL ONLY, makes NO network call and sends NO caller text.** Two episode kinds:
   - *Load-failure failover* (no local session exists; `_model_loaded` was set True by the failover branch at ~`:527`): the probe re-attempts the local load. R5's LE suggested splitting the load body into a private loader that raises; do that if it keeps the diff small (it also gives you a probe entry point). A failed probe must leave state **exactly as it was**.
   - *Inference-failure failover* (session and tokenizer exist): the probe runs one tiny fixed benign string (NOT caller text) through the **local** session. Success -> local is back.
4. **Fail back:** on a successful probe set `_remote_mode = False` and clear the failover marker, **atomically under `_model_lock`**, then the next default call uses local again. **Log loudly (WARNING)** with: that local recovered, how long the process was on remote, and how many vectors/calls were served remotely during the episode (a cheap counter). Also reset the once-per-episode failover-warning latch so a **later re-failover logs again** (today it is once per process).
5. **Failed probes are visible but not spammy:** log each failed probe at most once per probe interval (level your call, justified), with the exception type and message; never a traceback storm.
6. **Concurrency:** only one thread probes at a time (non-blocking try-lock); other threads keep using remote, never block waiting for a probe. **No new background thread** (ride the calling thread when the interval has elapsed, and state the worst-case latency cost of an inline probe). Do not hold `_model_lock` across the whole probe if avoidable; say what you chose.
7. **C3 interplay (`require_local`):** a `require_local` call while failed over must still never make a network call. It **may** trigger a due local probe; if that succeeds it proceeds locally, otherwise it refuses as today. State and test this.
8. **Bug-class defects stay bug-class:** a bug-class exception during a probe is raised/logged as in #765, not swallowed as "still down".
9. Default behavior of every other path is unchanged (R5 failover, bit-identical to explicit remote while remote; explicit-`hf` semantics; `require_local`).

## Tests (in `tests/test_ng_embed_r5_failover.py`), all with a controllable clock

Patch the module's time source so no test sleeps. Cover: (a) load-failure failover, local restored, after the interval a default call **fails back**: vector equals the local vector, `_remote_mode` False, one WARNING "failing BACK" with duration and served-count; (b) inference-failure failover, same; (c) local still down at probe time: stays remote, **exactly one probe per interval** (assert probe count with a counter across many calls inside and across intervals), failed-probe log rate-limited; (d) explicit `NG_EMBED_REMOTE=hf`: never probes, never fails back; (e) `NG_EMBED_REPROBE_SECS=0`: never probes; (f) fail back then **fail over again**: second failover logs its WARNING again; (g) `require_local` while failed over: no network call, may probe, proceeds locally if recovered else refuses; (h) two threads at the interval boundary: only one probe runs (simulate with the lock, no real races claimed); (i) a probe raising a bug-class exception is raised, not swallowed; (j) probes and the probe input never reach `_hf_post`/`urlopen` (assert zero remote calls attributable to the probe). Per-test `NG_EMBED_*` set/unset with asserts before the call under test; **report the NAMES** (now including `NG_EMBED_REPROBE_SECS`).
- **Negative controls in a scratch clone** (never mutate the worktree): remove the fail-back flip -> tests (a)(b) fail; make explicit-hf fail back -> (d) fails; remove the once-per-interval gate -> (c) fails; remove the latch reset -> (f) fails; probe sends caller text / hits the network -> (j) fails. Report per mutation.

## HARD rules (unchanged)

1. Every pytest command via guard plugin v4: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons -p no:cacheprovider`; **paste the `[z11 origin]` lines** (start and END) of your final runs; a run without them does not count.
2. Targeted files only: `tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py`. Never the full suite, `test_coordinator.py`, `test_tonic_bridge.py`, any afterTurn test. Do not run `reembed_snowflake.py`.
3. Real `~/NeuroGraph/data/checkpoints/commons.msgpack` byte-identical (size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`, sha256 `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`); show before/after.
4. Write scope: the two files, the branch, your untracked return. Nothing to the vault (`~/docs`), nothing to any auto-memory directory. No secrets printed.
5. Suite wording (P375): never "suite green"; the suite was not run and clean `origin/main` is red (72 known failures, #761).
6. Changelog entry (new dated entry; never edit prior entries) and update the docstrings that now describe one-way failover.
7. **NOT in this turn:** #764, #772, F2/#771 (hub versions, `cpu_count`), the `reembed_snowflake.py` broad-`except` question, the `cc_ng_host` catch, anything outside the two files. If C4 forces contact with one, stop and say so.

## Acceptance

1. Failover lasts only while local is down: proven fail-back for both episode kinds, loud WARNING both ways, latch reset, explicit-hf never fails back, `NG_EMBED_REPROBE_SECS` honored (incl. 0).
2. Probe is local-only: zero remote calls and no caller text in probes (tested).
3. One probe per interval, one prober at a time (tested); no new thread; latency cost stated.
4. Mutation controls fail as expected.
5. Counts before/after (baseline now: 94 passed in the two files); diff limited to the two files; new commit on top of `245d2bd`, pushed; `git status -sb`, `git log -3`, `git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929` shown.
6. Real commons file byte-identical.

## Return contents

Files changed with line refs · design (probe cost measured, default interval and why, inline-probe latency, lock strategy) · every command with `NG_EMBED_*` names (incl. `NG_EMBED_REPROBE_SECS`) · the `[z11 origin]` lines · counts · mutation results · real-file stat+sha256 · commit SHA + push confirmation · anything surprising or unverified (list it, do not act on it).
