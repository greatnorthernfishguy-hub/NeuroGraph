<!--
# ---- Changelog ----
# [2026-09-15] Cursor Grok 4.6 — operator record for landed Tonic stage timing
# What: new docs/CC_TONIC_STAGE_TIMING.md (docs only)
# Why: PITH_HOST_CONTRACT.md covers hosted Pith (#44/#45/#46). GitHub PR #43
#   last/EMA stage keys were already on main but undocumented, so agents keep
#   rediscovering the ~62s EMA split as missing. ARCHITECTURE.md still reads
#   as if Tonic is unbuilt.
# How: verified against main tonic_engine.py (TonicEngine.status),
#   cc_ng_host.py (_handle_status, _handle_pith_metrics, _pith_snapshot),
#   tests/test_tonic_stage_timing.py, and merged GitHub PR #43
# -------------------
-->

# CC Tonic stage timing

The ~62s Tonic EMA split is already in the live engine. Do not treat a missing
mention in `ARCHITECTURE.md` or `PUNCHLIST.md` as a missing feature. Do not
reopen merged GitHub PR
[#43](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/43) as a
code change.

This file is the operator record for that contract. Hosted Pith JSONL
(`docs/PITH_HOST_CONTRACT.md`, GitHub #44/#45/#46) is a different boundary.

## Two different #43s

| Number | What it is | Status |
|---|---|---|
| **GitHub PR #43** | Bounded per-stage Tonic last-sample + EMA timings | **Merged 2026-09-13** (`ed34a78`) |
| Punch-list #43 | Receptor Layer (vector quantization in `ng_lite.py`) | Done March 2026; unrelated |

This document is about GitHub PR #43 only.

## Probe

Send one JSON line on the hosted CC socket:

```json
{"event":"status"}
```

`cc_ng_host._handle_status` copies `TonicEngine.status` onto `tonic.engine`.
The sixteen stage keys live there:

`tonic.engine.{last,ema}_{feature_extract,model_feature_extract,body_lock_wait,transformer_forward,propagate,ouroboros,latent,autostep}_ms`

The eight names, from `_STAGE_NAMES` in `tonic_engine.py`:

1. `feature_extract` — candidate graph-feature scan
2. `model_feature_extract` — graph tensors for the shared body (outside the body lock)
3. `body_lock_wait` — time to enter the existing body lock (`__enter__` only; not a second acquire)
4. `transformer_forward` — held-lock forward
5. `propagate` — write-mode `prime_and_propagate`
6. `ouroboros` — tonic-thread cycle
7. `latent` — whole latent-token call
8. `autostep` — optional autonomous `graph.step()` when it actually runs

Each name has `last_*_ms` (this tick, milliseconds, rounded to 2 decimals) and
`ema_*_ms` (EMA α=0.2). Schema is constant-size: keys are always present.
Unrun stages report `last_*_ms = 0`. EMAs of stages that have run stay; they
are not wiped on early return.

Always on. No env knob. Timing failures fail-soft (zeros) and cannot stop a
tick or change activations.

## `ema_tick_ms` is not `ema_latent_ms`

`tonic.engine.ema_tick_ms` is the **total-tick** cadence EMA used by adaptive
backoff in `_generation_loop`. `ema_latent_ms` is only the latent-token stage.
Do not diagnose a 62s tick from `ema_latent_ms`.

## Over-budget warnings

`EngineConfig.tick_budget_seconds` defaults to **30s**. Override with
`NEUROGRAPH_TONIC_BUDGET_SECONDS` (invalid values keep 30). When a tick exceeds
the budget, the warning logs the same eight stage **last-samples** plus the
total-tick `ema` (`ema_tick_ms`). Under-budget ticks do not emit that warning.

## Not in `pith_metrics.jsonl`

The durable snapshot writer (`_pith_snapshot`) records gates, resolved config,
and Pith counters only. It never reads `TonicEngine.status`.

The live `{"event":"pith_metrics"}` handler (`_handle_pith_metrics`) peeks
`engine.status` for **`prefetch_seeded` only**. Stage timings will never appear
there. Use `{"event":"status"}`.

## Process must be post-#43

The source is on `main`. A process started before GitHub PR #43 was pulled and
restarted will not emit these keys. Missing keys mean an old process, not a
missing implementation.

## Tonic is built

`TonicEngine` is the idle-loop / latent-tick path. `ARCHITECTURE.md` still
talks as if Tonic is unbuilt; that paragraph is stale. `syl_daemon.py` is
leftover optional wiring in `openclaw_hook.py`. Do not start from the daemon
when looking for stage timings.

Live sources: `tonic_engine.py` (`TonicEngine.status`, `_generation_loop`
over-budget log), `cc_ng_host.py` (`_handle_status`). Tests:
`tests/test_tonic_stage_timing.py`.
