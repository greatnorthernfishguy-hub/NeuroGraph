<!--
# ---- Changelog ----
# [2026-09-13] Cursor Grok 4.6 — operator record for landed hosted Pith/CC work
# What: new docs/PITH_HOST_CONTRACT.md (docs only)
# Why: PUNCHLIST/CHANGELOG/USER_GUIDE/docs never mentioned compress_history,
#   pith_metrics.jsonl, or history_* counters, so agents keep rediscovering them
# How: verified against main cc_ng_host.py / cc_ng_organism.py and PRs #38/#39/#41
# -------------------
-->

# Hosted Pith contract (already on main)

This is an operator record, not a proposal. Hosted Claude Code NeuroGraph already
exposes history compression and durable Pith telemetry. Landed in PRs
[#38](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/38),
[#39](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/39), and
[#41](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/41).

Do not reopen those as code changes. Do not treat a missing mention in
`PUNCHLIST.md` as a missing feature.

Live sources: `cc_ng_host.py`, `cc_ng_organism.py`. Tests:
`tests/test_cc_host_compress_history.py`, `tests/test_cc_host_pith_telemetry.py`,
`tests/test_pith_history_metrics.py`.

---

## Where it lives

The hosted CC process serves a Unix socket at
`~/.claude/plugins/neurograph/daemon.sock` (`CC_NG_WORKSPACE`). Clients send one
JSON line:

```json
{"event": "pith_metrics", "data": {}}
{"event": "compress_history", "data": {"turns": ["..."], "per_turn_chars": 220}}
```

Unknown events return `{"ok": false, "error": "unknown event: ..."}`. That is
what miniTID used to get for `compress_history` before #38, then it fell back
to fixed-length truncation.

---

## `compress_history` → `pith_compress_history`

`cc_ng_host._handle_compress_history` is the socket wrapper.
`cc_ng_organism.pith_compress_history` is the one implementation.

- Empty or non-list `turns` is a no-op. The compressor is not called.
- Optional `per_turn_chars` is forwarded. If omitted, the organism uses
  `CC_PITH_KEYFRAME_CHARS`.
- The host holds `graph._concurrent_lock` when that lock exists. No graph, or
  no lock, still runs (graphless startup is allowed).
- The graph read is read-only. Warmth comes from `cc_thermal` on
  `cc:conv::<sha1(text)>`. Warmer turns keep more characters (about 1x–2x the
  base budget). Cold or missing nodes stay at the base budget.
- Fail-soft. A host exception returns `{"ok": true, "compressed": <original
  turns>}`. A per-turn exception inside the organism keeps that turn unchanged.
  History is never discarded.
- This outbound path does **not** check `CC_PITH_ENABLED`. That gate is for
  inbound L1 assembly in `cc_assemble_recall`. The compressor runs when the
  socket event arrives.

---

## `CC_PITH_KEYFRAME_CHARS`

Resolved at organism import time:

- env default `"220"`
- clamped to `[60, 1000]`
- reported in `pith_effective_config()` as `resolved.CC_PITH_KEYFRAME_CHARS`
  with `authority=cc_ng_organism`
- also on the host snapshot allow-list

`pith_compress_history` uses this as the per-turn base unless the caller
sends `per_turn_chars`. The same constant is the Stage 2 keyframe size for
inbound L1.

Editing the env after the process started does not change the running
organism constants. Snapshots record both raw env (`gates`) and what the
process actually resolved (`config.resolved`).

---

## Durable snapshots: `pith_metrics.jsonl`

D4 port of the laptop daemon contract, in the hosted lifecycle.

| Knob | Default | Floor |
|---|---|---|
| path | `~/.claude/plugins/neurograph/pith_metrics.jsonl` | — |
| `CC_PITH_SNAPSHOT_INTERVAL_SECS` | 300 | 30 |
| `CC_PITH_SNAPSHOT_MAX_BYTES` | 4 MiB | 64 KiB |

Each record is one JSON line, then `flush` + `fsync`. Rotation: if the live
file is over the max, it is renamed to `pith_metrics.jsonl.1` and a new file
starts. One rotated sibling, not an archive chain.

`window_id` is `uuid4().hex[:16]`, assigned at host-module import. Stable for
one Python lifetime. A new process gets a new id.

Reasons:

- `start` — first write of the `cc-pith-telemetry` thread
- `interval` — while `_STATE.running`
- `shutdown` — clean `shutdown_cc_host()`, only if `started_at > 0`

Fields: `ts`, `iso`, `reason`, `pid`, `window_id`, `window_started_ts`,
`gates` (allow-listed raw env only), `config` (or `null`), `config_error`,
`counters`.

The writer uses `_PITH_SNAPSHOT_LOCK` (2s timeout). It never reads or mutates
graph state. Snapshot failure is non-fatal.

`config_error` is a closed vocabulary:
`organism_unavailable`, `symbol_unavailable`, `import_failed`,
`resolve_failed`. Exception text is not written.

---

## Live `pith_metrics` socket read

`_handle_pith_metrics` is a pure read of `cc_ng_organism._PITH_METRICS`. No
reset, no file write.

It adds request-time env flags (not the import-time organism constants):

- `pith_enabled` — `CC_PITH_ENABLED` truthy (`"0"` / `"false"` / `""` are off)
- `prefetch_enabled` — `CC_PITH_PREFETCH_ENABLED`, same truth table
- `gate_enabled` — **alias of `prefetch_enabled`**, kept for pre-D4c clients
- `prefetch_hit_rate` — `prefetch_hits / promoted_predicted`, or `null` if
  nothing was promoted
- `prefetch_seeded` — CC's own Tonic engine only, never Syl's

`gate_enabled` is still the prefetch alias. It is not the Pith L1 gate.

Both Pith and prefetch default **off**.

---

## Outbound `history_*` vs inbound L1 / stage counters

`PithMetrics` keeps two boundaries. A snapshot can prove which one ran.

**Outbound history** (committed together under the metrics lock in
`record_history_compression`):

| Counter | Meaning |
|---|---|
| `history_calls` | one increment per `pith_compress_history` return |
| `history_turns_in` | turns handed in |
| `history_turns_compressed` | turns whose output is shorter than input |
| `history_chars_in` / `history_chars_out` | character totals |
| `history_chars_saved` | `max(0, in - out)` |
| `history_failures` | per-turn exceptions (also added to `pith_failures`) |

A host-wrapper exception that never reaches `pith_compress_history` does
**not** increment `history_*`. The client still gets the original turns.

**Inbound L1 / stage** (gated by `CC_PITH_ENABLED` inside
`cc_assemble_recall`):

- `l1_assemblies` — one increment per real Stage 3 assembly. Sample-size
  denominator. Gate off → stays 0. That is not a 0% prefetch rate.
- `l1_kept_distinct`, `l1_prefetch_distinct`, and the `_promotable` pair —
  coherent with `l1_assemblies`
- Stage leftovers (`ranked_*`, `compressed_count`, `chars_saved`,
  `prefetch_hits`, …) are best-effort. Do not ratio them against the L1
  terms.

`history_* == 0` does not mean L1 is missing. `l1_assemblies == 0` does not
mean history compression is missing.

---

## Healthy VPS vs failed VPS

Read the file first. The live socket is a point-in-time check; the JSONL
survives a restart.

**Healthy host**

- `pith_metrics.jsonl` exists after `init_cc_host`
- first record `reason=start`, later `interval` about every five minutes
- `window_id` is stable across that lifetime
- `config_error` is `null` and `config.resolved` is populated
- socket `{"event":"compress_history",...}` is not `unknown event`
- if miniTID is actually sending older turns: `history_calls` rises;
  `history_failures` stays small vs `history_turns_in`;
  `history_chars_saved` rises when turns are longer than the budget
- if `config.resolved.CC_PITH_ENABLED` is false, `l1_assemblies == 0` is
  expected
- `gate_enabled == false` with `pith_enabled == true` only means prefetch
  is off

**Failed or stale host**

- no JSONL after the host has been up, or no `start` record — telemetry
  thread did not start (init logs `CC Pith telemetry failed to start`)
- `{"ok": false, "error": "unknown event: compress_history"}` — old host
  build; miniTID will truncate
- `history_calls == 0` while the peninsula is sending long turns — event
  is not reaching this process
- `history_calls > 0` but `history_failures` ≈ `history_turns_in` and
  `history_chars_saved == 0` — organism per-turn path is failing
- `config_error` set — resolved settings cannot be proven for that window
- `window_id` changing on every sample — process is restarting
- RPC `pith_enabled` true but `config.resolved.CC_PITH_ENABLED` false —
  env was flipped after import; L1 is still off

Do not judge a window by volume of deleted text or by inbound counters
alone. For history, the question is whether `history_calls` moved and
whether `history_failures` stayed rare.
