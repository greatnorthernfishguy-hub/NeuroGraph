# R5 C4 return — worker-004 (fail-back, #766)

Zone Z11, lane `ng-embed-remaining-work-20260922`. Brief: `assignments/worker-004-failback.md` (Josh's ruling via Executive Packet 382).
Scope done: **failover now lasts only while local is down; the process re-probes local and fails back, loud WARNING both ways.** Not touched: #764, #772, F2/#771, the reembed broad-except question.

## Commit (on top of `245d2bd`, normal push, no PR, no merge, no history rewrite)

`c896ca76f4d01680a0b304694710fc251b79ad82` — `R5 C4: fail back - failover lasts only while local is down (#766)`. Files: `ng_embed.py`, `tests/test_ng_embed_r5_failover.py` only (+713 / -48). `handoffs/` never staged.

```
$ git status -sb && git log -3 --oneline
## cc-laptop-r5-embed-failover-20260929...origin/cc-laptop-r5-embed-failover-20260929
?? handoffs/
c896ca7 R5 C4: fail back - failover lasts only while local is down (#766)
245d2bd R5 C3: require_local is honored by the inference path (delta-LE F1)
c7c2753 ng_embed: R5 correction - narrow failover except (#765), truthful _ensure_model (#763), C2
$ git ls-remote --heads origin cc-laptop-r5-embed-failover-20260929
c896ca76f4d01680a0b304694710fc251b79ad82	refs/heads/cc-laptop-r5-embed-failover-20260929
```

## Design

**Episode marker, explicit (req 1).** `_failover_since` (monotonic timestamp, `None` = not failed over), `_failover_kind` (`"load"` or `"inference"`), `_failover_remote_calls/_vectors`, `_next_probe_at` (`ng_embed.py:446-455`). `_fail_over_or_raise` (`:614`) starts an episode under `_model_lock` and logs the failover WARNING once per episode. Explicit `NG_EMBED_REMOTE=hf` never starts one (`_failover_since` stays `None`), so it never probes or fails back. `_model_failed` keeps its documented meaning ("local has failed at least once") and is **not** reset on fail-back.

**Cadence (req 2).** `NG_EMBED_REPROBE_SECS`, read live inside `_reprobe_interval()` (`:654`) like the file's other env reads, default `_REPROBE_SECS_DEFAULT = 60.0` (`:360`), `<= 0` disables, and the failover WARNING then says "re-probing is DISABLED (NG_EMBED_REPROBE_SECS<=0)". First probe is due one interval after the failover, not immediately.
- **Deviation to flag:** a non-numeric value falls back to the default with **one** WARNING instead of raising like the sibling `CC_EXTRACT_WARN_INTERVAL_S` read would. Reason: a config typo must not turn a working failover into hard failures inside an embed call.
- **Why 60 s (measured, below).** Exposure after local recovers is at most one interval, and Josh's concern is text leaving the machine, so shorter is better. At 60 s the inline cost is one caller paying one probe per minute while local stays down: about 4.5% of that caller's time for a load episode and under 1% for an inference episode, worst case, on this loaded machine. An operator can raise it.

**The probe is local only (req 3).** `_maybe_fail_back` (`:670`):
- *Load episode:* calls the new private loader `_load_local(local_only=True)` (`:555`), which resolves the ONNX file and `tokenizer.json` with `hf_hub_download(..., local_files_only=True)` and builds the tokenizer with `Tokenizer.from_file`. No network. `_load_local` builds and returns `(session, tokenizer)` and touches no state, so a failed probe leaves everything exactly as it was. The first load still calls `hf_hub_download`/`Tokenizer.from_pretrained` with the same arguments as before and now assigns session and tokenizer together, only after both succeed.
- *Inference episode:* runs the fixed string `_PROBE_TEXT` (`:361`) through the local session via `_onnx_embed`. Never caller text.
- **Consequence to state plainly:** because the probe never downloads, an episode caused by model files that were never cached cannot recover by probing. It stays on remote (with one failed-probe WARNING per interval, showing the reason) until an operator restores the files or restarts, since a restart is allowed to download. I judged a 110 MB download on a caller thread worse than that, and the brief forbids network in the probe.

**Fail back (req 4).** `_fail_back` (`:708`): under `_model_lock`, installs the probe's session and tokenizer (load episode), sets `_remote_mode = False`, clears the episode (`_failover_since = None`, which is also the failover-warning latch, so the next failover logs again), resets the counters. Then logs a WARNING: `local model RECOVERED (probe ok) — failing BACK to local after 61.0s on HF remote; served remotely during the episode: 3 calls, 3 vectors`. The counter is `_count_served_remotely` (`:1283`), called at the two remote choke points; it is approximate under concurrency and includes keepalive pings made during an episode.

**Failed probes (req 5).** One WARNING per failed probe with exception type and message and **no traceback**: `local re-probe failed (RuntimeError: ...); still on HF remote after 60s, next probe in 60s`. Level justified: while remote, Syl's text is leaving the machine, so a persistent reminder is worth a WARNING; the interval bounds the volume (60 lines/hour at the default) and there is no traceback storm.

**Concurrency and latency (req 6).** `_probe_lock` is a non-blocking try-lock: one prober at a time, other threads skip the probe and keep using remote without waiting. The interval is claimed (`_next_probe_at = now + interval`) when the probe starts, so there is exactly one probe per interval whatever its outcome. **No new thread:** the probe rides the calling thread from `_require_model` (`:729-731`). `_model_lock` is held only for the state flip and the episode start, never across a probe. It became an `RLock` because `_fail_over_or_raise` now takes it and is also called from inside `_ensure_model`, which already holds it. Lock order is `_probe_lock` then `_model_lock`, no inversion. The `_failover_since` read in the fast path is an unlocked, benign racy read and is re-checked under the probe lock.

**`require_local` (req 7).** A `require_local` call goes through the same `_require_model`, so it may trigger a **due** probe. If that recovers local it proceeds locally, otherwise it refuses as before, and it never makes a network call. Tested for both episode kinds.

**Bug-class (req 8).** A bug-class exception during a probe goes through `_raise_if_bug_class` (ERROR + traceback) and is **raised out of the call that happened to run the probe**. That one caller's own embed fails, as ruled. The interval was already consumed, so it does not storm.

**Not in your list but required:** `_keepalive_loop` (`:1317`) now skips its ping while `_remote_mode` is False. Before C4 the process could never leave remote, so a keepalive thread started while remote would, after a fail-back, keep pinging the HF API from a process that is local again. That state was unreachable before, so the change cannot alter any existing behavior. Tested.

**Probe cost, measured** (real cached Snowflake model, `env -u NG_EMBED_REMOTE -u HF_HUB_OFFLINE`, `socket.connect` and `socket.getaddrinfo` patched to raise): load probe (`_load_local(local_only=True)`) succeeded 3 of 3, **0 network attempts**, 4.67 / 2.75 / 2.30 s (median 2.7 s). Warm inference probe, 20 runs: median 544 ms, max 2.13 s, vector shape (768,), 0 network attempts. **The laptop was heavily loaded** (about 10 concurrent Claude Code sessions, load average 8-12 earlier), so these are pessimistic; I did not measure it unloaded. Worst-case inline latency to one caller once per interval is therefore about 2-5 s (load episode) or about 0.5-2 s (inference episode) on this machine right now.

## Tests (+23, `tests/test_ng_embed_r5_failover.py`, controllable clock; nothing sleeps)

`ng_embed._now` is patched with a `_Clock`. `_Scenario` builds either episode kind (load failure, inference failure) with a switchable "local is up" flag and the C3 remote spy. Coverage against the brief: (a)+(b) fail back for both kinds with the exact log text, duration, served-count; (c) still down, exactly one probe per interval across 50-call bursts and two calls at one boundary, one log line each, no traceback, failed load probe leaves state identical; (d) explicit `NG_EMBED_REMOTE=hf` never probes (probe entry points patched to raise) and its counters stay 0; (e) `NG_EMBED_REPROBE_SECS=0` (and the log says so), the unset default, and the non-number fallback with one warning; (f) fail back then fail over again logs OVER, BACK, OVER in order and restarts the episode counters; (g) `require_local` while failed over: not due / due and down / due and recovered, for both kinds, zero added remote calls; (h) one prober at a time, simulated by holding the lock, the call is served remotely without blocking and the skipped call does not consume the interval; (i) a bug-class probe defect is raised (both kinds), ERROR logged, episode untouched, lock released, no storm; (j) the probe made directly with no caller in flight: zero remote calls of any kind, `socket.connect` and `getaddrinfo` raise (0 attempts), load probe used `local_files_only=True` for both files and ran no inference, inference probe sent exactly the fixed probe text; plus keepalive pings only while on remote. Concurrency in (h) is simulated with the lock; **no real thread race is claimed.**

## Counts

| Point | R5 file | dualpass | Total |
|---|---|---|---|
| Baseline (`245d2bd`, worker-003) | 63 | 31 | 94 passed |
| After C4 (`c896ca7`, committed state) | 86 | 31 | **117 passed, 0 failed** |

The 23 new tests: fail-back 2, still-down 2, explicit-hf 1, disabled-by-0 2, default/invalid 1, re-failover 1, require_local 6, one-prober 2, bug-class probe 2, local-only probe 2, keepalive 2. I did **not** run the full suite; make no claim about it. Clean `origin/main` is red (72 known failures, #761).

## Commands and NG_EMBED_* names

Every pytest command: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest <files> -p z11_noreal_commons -p no:cacheprovider ...` (guard plugin v4); mutation runs added `--timeout=60`. Targeted files only (`tests/test_ng_embed_r5_failover.py`, `tests/test_ng_embed_dualpass.py`); never the full suite, `test_coordinator.py`, `test_tonic_bridge.py`, any afterTurn test, or `reembed_snowflake.py`.

NG_EMBED_* names in play: **`NG_EMBED_REMOTE`** (removed by `env -u`, then each test sets/deletes it and asserts it) and **`NG_EMBED_REPROBE_SECS`** (new; each test sets or deletes it and asserts it). The autouse fixture now also deletes `NG_EMBED_REPROBE_SECS`, in addition to `NG_EMBED_ALLOW_HASH_FALLBACK`, `NG_EMBED_TID_ENDPOINT`, `HF_TOKEN`. Per test:
- All `_Scenario` tests (fail-back, still-down, require_local, one-prober, bug-class, local-only, re-failover): `NG_EMBED_REMOTE` deleted (asserted `is None`), `NG_EMBED_REPROBE_SECS=60` (asserted).
- `test_c4_explicit_remote_hf_never_probes_or_fails_back`: `NG_EMBED_REMOTE=hf` (asserted), `NG_EMBED_REPROBE_SECS=1` (asserted).
- `test_c4_reprobe_secs_zero_disables_and_says_so`: `NG_EMBED_REMOTE` deleted, `NG_EMBED_REPROBE_SECS=0`.
- `test_c4_reprobe_secs_default_and_invalid_value`: `NG_EMBED_REMOTE` deleted; `NG_EMBED_REPROBE_SECS` unset (asserted), then `abc` (asserted).
- `test_c4_keepalive_pings_only_while_on_remote`: both deleted (asserted).
- `HF_TOKEN=tok-test` is set by scenario tests (never read: `_hf_post` is always replaced).
- Measurement script: `env -u NG_EMBED_REMOTE -u HF_HUB_OFFLINE`; it never read `NG_EMBED_REPROBE_SECS`. It only read the local model caches, read-only.

## `[z11 origin]` lines (final run on the committed state `c896ca7`; start lines above the counts, END line last)

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
117 passed in 6.64s
```
Main-checkout note: same as the last two returns. The END line reports `ng_salience_gate` from the MAIN checkout, recurring every run; no test here uses it; `ng_embed` and `neurograph_rpc` are both WORKTREE; not investigated.

## Mutation results (scratch clone under `/tmp`, `git clone --no-hardlinks`, deleted afterwards; worktree `ng_embed.py` hash unchanged: `78e1db7e...`)

Control (final files, unmutated, in scratch): 86 passed. One regression at a time, R5 test file under the guard:

| Regression | Failed | Tests that caught it |
|---|---|---|
| M01 remove the fail-back flip (`_remote_mode = False`) | 9 | fail-back x2, re-failover, require_local recovered, one-prober, local-only probe |
| M02 explicit `NG_EMBED_REMOTE=hf` starts an episode (so it would fail back) | 21 | includes `test_c4_explicit_remote_hf_never_probes_or_fails_back`; also many earlier tests because their ground-truth helper uses explicit mode |
| M03 remove the once-per-interval gate | 5 | still-down x2, bug-class probe x2 (storm), default/invalid |
| M04 episode not cleared on fail-back (latch not reset) | 3 | re-failover, fail-back x2 |
| M05 inference probe hits the network (`_hf_remote_embed`) | 8 | local-only probe, fail-back, require_local, still-down, others |
| M06 load probe not local-only (`local_only=False`) | 7 | local-only probe, fail-back, still-down, require_local, others |
| M07 probe text altered | 7 | local-only probe (exact feed), fail-back, still-down, others |
| M08 failed probe mutates state | 2 | still-down (state snapshot) |
| M09 bug-class probe defect swallowed | 2 | bug-class probe x2 |
| M10 load fail-back installs no session/tokenizer | 3 | fail-back (load), require_local, one-prober |
| M11 served-remotely counter dead | 3 | fail-back x2, re-failover |
| M12 keepalive keeps pinging after fail-back | 1 | keepalive |
| M13 `require_local` never triggers a probe | 4 | require_local x4 |
| M14 probe lock not exclusive | 2 | one-prober x2 |

All 14 caught. Brief's five named controls: remove flip -> (a)(b) fail (M01); explicit-hf fails back -> (d) fails (M02); remove once-per-interval gate -> (c) fails (M03); remove latch reset -> (f) fails (M04); probe hits network / sends other text -> (j) fails (M05, M06, M07).

## Real commons file (`~/NeuroGraph/data/checkpoints/commons.msgpack`)

```
BEFORE: 192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
AFTER:  192773  2026-09-29 17:38:16.949928157 -0800  a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
```
Byte-identical; plugin trace `BLOCKED` count 0. I read nothing else under `~/NeuroGraph/data/`. Vault (`~/docs`) and every auto-memory directory: untouched by me this turn.

## Surprises and unverified (listed, not acted on)

1. **Probe cannot fetch missing model files.** Recovery from "model files were never cached" needs an operator or a restart (see design). The probe is deliberately local only.
2. **A bug-class probe defect fails an unrelated caller's embed once**, because the probe rides that caller's thread (as ruled). It is loud and does not repeat within the interval.
3. **Timings are from a heavily loaded machine** (load probe median 2.7 s, inference probe median 0.54 s); not measured unloaded, so the latency figures are pessimistic.
4. **Concurrency is simulated, not raced.** The one-prober tests hold the lock; the served-remotely counter is approximate under concurrency.
5. **The load-probe path was proven on the real model only in my measurement script** (sockets and DNS blocked, 3 of 3 ok). The test suite exercises the probe with stubs. I never induced a real ONNX Runtime failure and recovery.
6. **#764 still applies:** in a load-episode failover the remote-mode windowing still calls `Tokenizer.from_pretrained` via `_ensure_tokenizer`, which can reach the hub (no caller text). Untouched, as instructed.
7. **`os.cpu_count()` (F2)** is untouched; the probe reuses the same loader, so on a host where it returns `None` a load probe would raise `TypeError` (bug-class, raised).
8. `NG_EMBED_REMOTE=hf` set in the environment after an episode began is not consulted (env is only read at first load, as before). `embed()` guards twice but the second guard sees the interval already consumed, so it never probes twice.
9. Not verified: consumer repos; the full suite (not run); any behavior of `reembed_snowflake.py` (never run).

Stopped here. No self-review, no PR, no merge.
