<!--
# ---- Changelog ----
# [2026-09-13] Codex — specify the topology-built provider_context boundary
# What: add closed states, ownership, cue rules, epistemic labels, and whole-basin output
# Why: history compression cannot provide fresh model-agnostic situational continuity
# How: shared organism assembler with thin, parity-tested VPS/laptop socket wrappers
# [2026-09-13] Cursor Grok 4.6 — operator record for landed hosted Pith/CC work
# What: new docs/PITH_HOST_CONTRACT.md (docs only)
# Why: PUNCHLIST/CHANGELOG/USER_GUIDE/docs never mentioned compress_history,
#   pith_metrics.jsonl, or history_* counters, so agents keep rediscovering them
# How: verified against main cc_ng_host.py / cc_ng_organism.py and PRs #38/#39/#41
# -------------------
-->

# Hosted Pith contract

Hosted Claude Code NeuroGraph already exposes history compression and durable
Pith telemetry, landed in PRs
[#38](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/38),
[#39](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/39), and
[#41](https://github.com/greatnorthernfishguy-hub/NeuroGraph/pull/41). Slice A
adds `provider_context`: the read-only boundary that turns current SNN topology
and activation into a fresh provider-facing situational model.

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
{"event": "provider_context", "data": {"current_instruction": "...", "quest_focus": "..."}}
```

Unknown events return `{"ok": false, "error": "unknown event: ..."}`. That is
what miniTID used to get for `compress_history` before #38, then it fell back
to fixed-length truncation.

## `provider_context` → `pith_provider_context`

`cc_ng_organism.pith_provider_context` owns request validation, SNN ignition,
connected assembly, admission, epistemic labeling, and model-facing Markdown.
Both socket hosts are thin binders:

- VPS: `cc_ng_host._handle_provider_context`
- laptop: `docs/scripts/cc-ng-daemon.py:handle_provider_context`

Each binder passes only its own CC NeuroGraph, conversation-state bookkeeping,
Commons arousal reader, and existing graph consistency lock. Neither host reads
Quest storage. Neither host nudges, deposits, steps, saves, or changes topology.
The current instruction and latest already-rendered bounded Quest focus arrive
from the harness and are used as attention cues only. `provider_context` does
not echo either value. miniTID keeps their one exact occurrence in the live
message tail when it composes the overall provider prompt.

If an identical stored forest fires, Pith keeps its learned relationships but
renders a structural live-tail placeholder instead of the duplicate text (and
does not repeat that node's live-tail anchors). This preserves a connected
correction without echoing the instruction or Quest focus.

Pattern completion remains topology-native: the vector store seeds the existing
`_harvest_associations` read-mode propagation. Pith treats fired nodes as roots,
then follows learned synapses, co-active hyperedges, and temporal directions to
form connected activation basins. One `CacheLine` carries the root keyframe,
related deltas, source/coherence labels, member identity internally, and exact
operational anchors. Admission moves that line whole.
Current activation, learned topological support, thermal state, coherence, and
the capacity envelope shape competition between whole basins. Edge kind is only
a tie-break/presentation fact. Internal scores and node IDs are never rendered
to the provider.

The provider path suppresses Stage-4 speculative `primed_nodes` promotion.
Predictions that did not fire in this cue's propagation cannot become situation
roots. If the topology fires them normally, they remain eligible like any other
current activation.

This is observational ignition, not teaching. On its normal path, the
underlying `prime_and_propagate` read mode disables plasticity and restores
transient voltages and refractory state. The wrapper also leaves graph
configuration, conversation bookkeeping, topology, and deposits unchanged.
Known adjacent canonical debt: that restoration is not yet protected by a
`finally`, so a mid-propagation exception can leave transient state changed
before `_harvest_associations` degrades to an empty result. Fix belongs in
canonical `neuro_foundation.py`; Slice A does not compensate beside it.

### Request

| Field | Required | Contract |
|---|---:|---|
| `current_instruction` | yes | non-empty string; current human request |
| `quest_focus` | no | already-rendered bounded Quest focus; empty string allowed |
| `budget_chars` | no | total returned `context` envelope, integer 500–40000; defaults to the breathing L1 budget |
| `root_count` | no | bounded ignition-root count, integer 1–24 |

The request never contains transcript history. `provider_context` never opens a
Quest database, chooses a memory shard, or invokes a provider.

### Deployment configuration

Before either host deploys Slice A, its canonical environment source must set
the new controls explicitly (normally `.bashrc`, inherited by the service):

```bash
export CC_PITH_PROVIDER_ROOTS=8
export CC_PITH_PROVIDER_MEMBERS=6
export CC_PITH_PROVIDER_DEPTH=2
export CC_PITH_PROVIDER_NODE_CHARS=700
export CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS=8000
export CC_PITH_PROVIDER_MAX_QUEST_CHARS=8000
```

They appear in `pith_effective_config()` and both hosts' allow-listed telemetry,
so an operator can verify resolved authority after restart. This branch does not
edit the live environment or restart either host.

### Closed response states

Every response has `ok`, `state`, `context`, `source`, `coherence`, `anchors`,
`warnings`, and `assemblies`.

| `state` | `ok` | Meaning |
|---|---:|---|
| `ok` | true | one or more connected topology assemblies were admitted |
| `empty` | true | the graph was available but no learned assembly fired/admitted; the constitutional block remains present |
| `unavailable` | false | the request or substrate path could not produce fresh context; `context` is a short bounded status notice |

An `empty` result warns `topology_empty` when nothing fired and
`capacity_empty` when connected material existed but no whole line fit.

The unavailable warning vocabulary is closed:
`invalid_instruction`, `instruction_too_large`, `invalid_quest_focus`,
`ng_unavailable`, `invalid_budget`, `invalid_root_count`,
`constitutional_core_missing`, `constitutional_core_exceeds_budget`,
`context_bound_failed`, `assembly_failed`, and `symbol_unavailable`
(host cannot import the shared owner).

There is no unavailable fallback to heuristic topology, faux KISS, ranked
database snippets, compressed turns, or full transcript replay.

### Model-facing Markdown

The output has these purpose-shaped blocks when populated:

1. `Who I Am` — constitutional core, exactly once and outside learned-context admission
2. `Learned Situation` — connected topology assemblies, each labeled learned plus coherence and source
3. `Learned Corrections and Failures` — whole causal assemblies containing correction/failure relations
4. `Uncertainty and Conflicts` — explicit stale, uncertain, or conflicting learned material

Empty learned sections disappear. Exact paths, repos, branches, commits, issue
IDs, UUIDs, and URLs remain attached to the assembly that supplied them. Pith
does not call learned material “verified.” Verification belongs to current
request text and authoritative live observations; miniTID's later Slice B owns
the small exact live tool tail.

Missing coherence metadata renders as `unknown`, with an explicit uncertainty
notice. It never defaults to `exclusive` or implies that learned material is
currently verified.

`len(context)` never exceeds `budget_chars`. The constitutional core remains
whole; if it alone cannot fit, the response is closed `unavailable`. Oversized
learned cache lines may shorten each member's prose, but all recorded
relationships, sources, coherence, and exact anchors remain attached. If that
fixed structure cannot fit, the whole line stays out.

Slice A does not add a provider-specific victim cache. Recapture remains owned
by Pith's existing Stage-5/canonical autonomic design; a later slice must use
that owner rather than creating query-time process memory beside it. Therefore
old provider output cannot revive an old quest when current topology is empty.

### Behavioral acceptance

A blank session receiving this fresh context plus the one exact live-tail copy
of the current instruction and Quest focus can identify the active mission,
done condition, next action, constitutional core, exact working anchor,
relevant correction, and unresolved stale/conflict state. After a side quest,
it resumes the parent rather than an older quest.

The same populated NeuroGraph and provider context preserve mission, done
condition, corrections, constitutional values, relationship structure, exact
anchors, and unresolved work when the harness swaps Claude, Grok, or OpenAI.
Style may change; continuity does not require transcript replay.

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
