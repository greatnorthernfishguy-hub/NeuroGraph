# ATOMIC_SAVES — every save that can't risk a failed write is atomic (lane atomic-saves, 2026-10-14)

Josh, 2026-10-14: *"We should have all saves and checkpoints, anything that can't risk a failed save, be atomic."*

Trigger (#1095): the 10:26 CC-daemon restart caught Elmer's `save_evolved_weights()` mid-`torch.save` of
`~/.elmer/proto_weights.pt` (in place, ~1.44 GB). The torn file could not load (`PytorchStreamReader failed reading zip
archive: failed finding central directory`, daemon.log 10:29:31), protoUniBrain started from the pretrained body, and its
next periodic save wrote over the torn file. Mechanism: the daemon's shutdown watchdog `os._exit`s after
`SHUTDOWN_GRACE_SECS` = 45 s (cc-ng-daemon.py:4992), and one proto save takes ~95 s on this laptop (measured below), so any
restart that lands inside a proto save tears the file.

Pattern used everywhere: temp file in the SAME directory -> write -> flush + fsync -> `os.replace` onto the final name ->
fsync the directory. Readers see the previous complete file or the new complete file, never a mix. Final bytes, formats and
paths are unchanged (torch.save / msgpack / JSON exactly as before), so no reader changed. No vendored file changed.

## Branches

| Repo | Branch | Base | Commits |
|---|---|---|---|
| NeuroGraph | `cc-laptop-atomic-saves-20261014` | `04e6c42` | `c92dda5` (non-protected), `fc4dfdf` **PROTECTED** (`activation_persistence.py`), this doc |
| Elmer | `cc-laptop-atomic-saves-20261014` | `cae7288` | `bd5cf15` (proto_weights.pt — the incident), `95ddcc0` (failed restore keeps the files), `2f06958` (et_modules/manager.py, byte-identical to NG `c92dda5`) |
| UniAI | `cc-laptop-atomic-saves-20261014` | `3080e47` | `bcc6c75` |
| docs (daemon) | `cc-laptop-atomic-saves-daemon-20261014` | `origin/cc-laptop-trial-s4a-20261002` (`e4d5dc40`) | `6dbee99f` |
| TID | `cc-laptop-atomic-saves-20261014` | `8fd9cd1` (live `-vps` checkout's branch `cc-laptop-tid-pool-20261003`) | `067f7d1` |

Overlap with the concurrent NG lane `cc-laptop-dim-guard-20261014` (from `04e6c42`): it touches `dim_reembed.py`,
`ng_embed.py`, `ng_lite.py`, `tests/test_constitutional.py`, `tests/test_dim_guard.py`, `universal_ingestor.py`; this
branch touches `activation_persistence.py`, `checkpoint_guardian.py`, `et_modules/manager.py`, `lenia/channels.py`,
`lenia/kernel.py`, `lenia/kill_switch.py`, `tests/test_atomic_saves*.py`, `ATOMIC_SAVES.md`. **No file in common.**

## Inventory — writers of persistent state run by the laptop's live processes

Live processes (2026-10-14 ~11:30): the CC daemon (`docs/.claude/worktrees/trial-s4a-20261002/scripts/cc-ng-daemon.py`,
importing NG from the trial worktree, hosting Elmer from `~/Elmer` and UniAI's `lenia/`), TID uvicorn (cwd
`~/The-Inference-Difference-vps`), Condensate's `condensate_observer_laptop.py`. Line numbers are the BASE commits'.
Cost: **irreplaceable** = learned state lost; **degrades** = falls back / restarts a clock or reference; **rebuildable** =
recomputed (cost noted).

### A. Converted (C) — the write could tear or lose state

| # | File | Writer (base file:line) | Before | Cadence | Torn-write cost | After (branch) |
|---|---|---|---|---|---|---|
| A1 | `~/.elmer/proto_weights.pt` (body + Flow-Lenia F1 state) | Elmer `core/proto_brain_socket.py:958` `save_evolved_weights` | `torch.save` in place | every 5 lenia steps (live 8-17 min) + unload; ~95 s per save | **irreplaceable** (the #1095 loss) | `core/atomic_io.torch_save_atomic` (`proto_brain_socket.py:986`) — Elmer `bd5cf15` |
| A2 | `~/.elmer/proto_lenia_state.pt` | same, `:962` | `torch.save` in place | with A1 | degrades (Phase-1 baselines) | `torch_save_atomic` (`:990`) |
| A3 | (A1/A2 on a failed restore) | `_restore_evolved_weights` except path `:936` | logged "starting fresh"; next save overwrote the unloadable file | on start | irreplaceable | renames both to `<name>.unloadable-<stamp>`, ERROR (`:959`) — Elmer `95ddcc0` |
| A4 | `~/.et_modules/elmer/proto_body_status.json` | Elmer `core/brain_switcher.py:256` | `open('w')` + json.dump | proto load | degrades (Tonic consumer can't parse) | `atomic_write_text` (`:262`) |
| A5 | `~/.et_modules/registry.json` (+ `et_module.json` manifests) | `et_modules/manager.py:373` (`:88`) — NG and Elmer copies identical | `open('w')` in place | module register (live: 10:56) | degrades (registry discarded, modules forgotten until re-register) | local `_atomic_write_text` (`:47`, used `:131`, `:415`) — NG `c92dda5`, Elmer `2f06958` |
| A6 | `checkpoints/main.msgpack`, `vectors.msgpack`, `quarantine/*` | `openclaw_hook.py:1793/1810` -> `checkpoint_guardian.atomic_file_write` `:385` | tmp + `os.replace`, **no fsync before the rename** (fsync came later, in `rotate_generations`) | autosave ~5 min | **irreplaceable** on power loss / hard hang (ecryptfs `~`; laptop GPU freezes): empty/partial file under the final name | `atomic_file_write` fsyncs the tmp, replaces, fsyncs the dir; reports stale tmps (`checkpoint_guardian.py:465`) — NG `c92dda5`. Protected writers untouched. |
| A7 | `main.msgpack.manifest.json`, `.guard_state.json`, `.guard_history.json` | `checkpoint_guardian.py:158`, `:211`, `:520` | tmp + replace, no fsync | every save | degrades (gate reference / #1051 history reset) | `atomic_write_text` (`:168`, `:219`, `:622`) |
| A8 | `checkpoints/main.msgpack.activations.json` | **PROTECTED** `activation_persistence.py:191` `write_state` | `open('w')` + json.dump in place | every save | irreplaceable-ish (all CES warmth; `restore()` rejects a torn file; the generation ring copies it as-is) | `checkpoint_guardian.atomic_write_text` (`:203`) — NG `fc4dfdf`, **needs Josh's "proceed"** |
| A9 | `lenia/channels.json` | `lenia/channels.py:291` | `open('w')` in place | channel lifecycle events | degrades (registry unparseable at load) | `atomic_write_text` (`:296`) |
| A10 | `lenia/enabled` | `lenia/kill_switch.py:82`, `:101` | `write_text` (truncate first) | enable/disable | degrades: empty file = Lenia silently OFF next start | `atomic_write_text` (`:87`, `:106`) |
| A11 | `lenia/distance_cache.npz` | `lenia/kernel.py:355` | tmp + replace (#261), no fsync | Lenia checkpoints | rebuildable but multi-hour | file-object `savez_compressed` + fsync + dir fsync (`:360`) |
| A12 | `checkpoints/.healthy_node_count` | daemon `:2409` | tmp + replace, no fsync | every accepted save | degrades: outer save guard's reference resets to 0 | `_atomic_write_text` (daemon `:2418`, used `:2464`) — docs `6dbee99f` |
| A13 | `checkpoints/.cc_conv_last_forest` | daemon `:2422` | tmp + replace, no fsync | every accepted save | degrades (no predecessor link) | `:2474` |
| A14 | `checkpoints/.cc_last_sleep_wall` | daemon `:5147` | tmp + replace, no fsync | each sleep | degrades (sleep clock) | `:5207` |
| A15 | `checkpoints/.cc_last_sleep_observe_wall` | daemon `:5291` | tmp + replace, no fsync | each observe | degrades | `:5348` |
| A16 | sleep observe reports (0600) | daemon `:5342` | tmp + replace, no fsync | each observe | report only | `:5395` (mode 0600 kept) |
| A17 | `checkpoints/last_good/` (one-slot backup) | daemon `_maybe_snapshot_last_good` `:2486-2495` | copies not fsynced; old slot `rmtree`'d BEFORE the new one was renamed in — a crash there left **no last_good** | 30 min | the backup itself | copies fsynced; old slot renamed to `last_good.old`, new slot renamed in, then `.old` removed (`:2528`, `:2553`) |
| A18 | `~/.claude/plugins/neurograph/refcount`, `daemon.pid` | daemon `:4261`, `:5787` | `open('w')` in place | session refcount / start | degrades | `_atomic_write_text` (`:4323`, `:5838`) |
| A19 | `~/.elmer/proto_flow_lenia/enabled` | UniAI `lenia/engine.py:966` `suspend` | tmp + replace, no fsync | on F1 suspend | degrades: empty = F1 silently off | `atomic_write_text` (`:998`) — UniAI `bcc6c75` |
| A20 | `~/.elmer/proto_flow_lenia/suspended.json` | UniAI `lenia/engine.py:969` | `open('w')` in place | on suspend | loses the suspension reason | `:999` |
| A21 | `~/.elmer/proto_flow_lenia/probe_result.json` | UniAI `lenia/flow_probe.py:131` | tmp + replace, no fsync | each hunger probe (~4 h) | degrades (engine re-reads next probe) | fsync + dir fsync (`:136`) |
| A22 | TID `model_stats_cache.msgpack` (success stats, penalties, TTL-exempt tool_competence) | TID `inference_difference/router.py:1009` `save_stats` | `open('wb')` in place | periodic + shutdown | **irreplaceable** learned routing state (load_stats starts cold) | `inference_difference/atomic_io.atomic_write_bytes` (`router.py:1015`) — TID `067f7d1` |
| A23 | TID `routing_state.msgpack` | TID `app.py:800` | `open('wb')` in place | on mode change | degrades (mode reset) | `app.py:821` |

### B. Already atomic (A) — left as is

| # | File | Writer | Method | Note |
|---|---|---|---|---|
| B1 | `checkpoints/commons.msgpack` | `cc_ng_organism.persist_cc_commons:1766` -> `Commons.persist` -> `ng_lite.save` | Rust `save_binary` (tmp + `sync_all` + rename; ng-tract-rs `src/ng_lite_core_py.rs:496`) or `ng_lite._atomic_write` (tmp + fsync + replace, `ng_lite.py:372`) | VENDORED. Missing only the directory fsync (rename may be lost on power loss -> previous file, never torn). Optional hardening = canonical + re-vendor; not done. |
| B2 | `~/.et_modules/elmer/ng_lite_state.msgpack` | Elmer `ng_ecosystem.save:605` -> same | same | same |
| B3 | `checkpoints/generations/<stamp>/` | `checkpoint_guardian.rotate_generations` | hardlinks (msgpack) / copy2 (json) + fsync of files and dirs before pruning (Codex #423) | durable |
| B4 | `daemon.refused.json`, cc-ng-service marker | daemon `_write_refusal_marker:2011`, cc-ng-service `_write_json_atomic:356` | mkstemp + fsync + replace | already the pattern (no dir fsync) |
| B5 | `checkpoints/<membership>` | `cc_topology_merge.py:511` | tmp + fsync + replace | durable |
| B6 | conduit exports | `cc_topology_export.py:620`, `:994` | `.partial` + replace | export copies (rebuildable) |
| B7 | gateway delivery journal | `cc_ng_organism` Leg-1 receiver | sqlite WAL, `synchronous=FULL`, dir fsyncs | durable |
| B8 | tract cursor sidecars / compaction | `ng_tract_bridge.py:819`, `:845` | tmp + replace, no fsync | VENDORED; rename-atomic against kills |
| B9 | `ng_autonomic` state | `ng_autonomic.py:97` | tmp + replace, no fsync | VENDORED; tiny |
| B10 | Leg-1 conduit write | `cc_ng_organism.trickle_gateway_conduit:4538` | tmp + replace | gate OFF by default |
| B11 | TID blocked-provider / policy caches | TID `model_client.py:431`, `:456` | tmp + replace | caches |

### C. Append-only logs — torn LAST line only

| File | Writer | Reader tolerance |
|---|---|---|
| `pith_metrics.jsonl` | daemon `:5068` (append + fsync per record, rotation by `os.replace`) | analysis only; fsync bounds loss to the record in flight |
| `~/.elmer/competence_delta.jsonl` | Elmer `runtime/engine.py:997`, `proto_brain_socket.py:834`, UniAI `engine._log:987` | **no runtime reader** (writers only). A torn last line glues onto the next append, so an offline parser must skip one unparseable line |
| `~/.elmer/weight_stats.jsonl` | `proto_brain_socket.py:1004` | no runtime reader; same caveat |
| `proto_flow_lenia/probe_history.jsonl`, `probe.log` | `flow_probe.py:134` | no runtime reader |
| `memory/events.jsonl` | `openclaw_hook.py:1176` (protected), `openclaw_adapter.py:398` (vendored) | operational log |
| `*.tract` (BTF, binary appends) | gateway / deposits | the cursor drain resyncs past a partial trailing frame (`ng_tract_bridge.py:917`, #285): tolerated |
| `cc_refeed` records | `cc_refeed.py:133` (append + fsync) | durable per record |
| Condensate `observations` | `condensate_observer_laptop.py:175` | append-only |
| `/tmp/elmer_*.log`, `/tmp/tid_last_usage.json` | debug / ephemeral | n/a |

### D. Left, with reason

| # | File | Writer | Why left |
|---|---|---|---|
| D1 | `lenia/field.bin` | `lenia/field.py:85` (`_create`) + mmap writes | Not a save: the live mmap'd field store, double-buffered with a selector byte; atomic replace would detach the mapping. (Laptop file last modified Jul 5.) |
| D2 | `tracts/cc_gateway/turns.tract` rewrite after drain | `cc_ng_organism.drain_ingest_tract:4457/4465` (`open('wb')` of the remainder) | A queue with concurrent appenders, not a save: tmp + rename would drop appends made to the old inode. Torn rewrite loses only the not-yet-absorbed remainder, but that IS unabsorbed turns — **open risk, needs its own design** (e.g. the #243 cursor-sidecar pattern). Already has a read->truncate race with appenders. |
| D3 | `trisynaptic_handoff_*.json` | `trisynaptic/handoff.py:36` | tmp + rename already (no fsync); ephemeral worker handoff |
| D4 | topology owner sentinel | `topology_owner.py:101` | tmp + replace; a pid |
| D5 | `proto_flow_lenia/probe_inputs.pt`, `probe_tokens.pt` | Elmer `scripts/proto_flow_f0_restore.py:150/154` | offline one-shot tool, not run by a live process; reproducible by re-running F0 restore |
| D6 | TID `credit_notice.json` | `model_client.py:126` | one-shot notice |
| D7 | TID NG-Lite state | TID `ng_lite.py:1035` (python fallback writes JSON in place) | VENDORED and **stale**: the canonical `ng_lite.py` already writes msgpack atomically. Fix = re-vendor (listed below), not an edit. Live TID uses the Rust core path (`save_binary`, atomic) when available. |
| D8 | TID catalog / CTEM sqlite | `catalog_manager.py:220`, `ctem.py:239` | sqlite journaling |
| D9 | Not run on the laptop | `memory_retry_queue.py:74` (tmp+replace, VPS sidecar), `neurograph_rpc.py:5269`, `syl_sync.py`, `cc_ng_host.py:806` (refcount in place; not the live host), `neurograph_migrate.py`, `reembed_snowflake.py`, `vectordb_persistence_patch.py`, `surgery/*` torch.save (training) | outside "live on the laptop"; same pattern applies when they are next touched |

**Counts:** 59 inventory rows (A 23 + B 11 + C 9 + D 16, each D9 file counted). **Converted: 22 writers + 1 restore guard
(A1-A23, 25 call sites).** **Left: 36** — 11 already atomic, 9 append-only log classes, 16 with a stated reason.

## Format notes (not changed here)

Several sidecars are JSON (`activations.json`, manifest, guard state/history, `channels.json`, `registry.json`, the
daemon's small state files). Per [[saves-are-msgpack-never-json]] saves are msgpack; changing those formats is a reader
change and, for the activation sidecar, a protected-format change — out of scope for this lane, flagged for Josh.

## Proof

All runs isolated (`~/.cache/flow-lenia-f1/run-isolated.sh`: bwrap PID namespace, read-only binds over the live trees and
`~/.elmer`, tmpfs over the live plugin / `~/.et_modules` dirs, MemoryMax scope, nice 10, MemAvailable >= 4 GB gate),
ONE run at a time. Logs: `~/.cache/atomic-saves-20261014/`.

### SIGKILL tests per converted writer (forked child runs the REAL writer in a loop, parent SIGKILLs at random points)

| Test | Writers | Result |
|---|---|---|
| Elmer `tests/test_atomic_saves.py` (6) | `save_evolved_weights` (A1+A2), helper failure path, modes, stale-temp report, failed-restore keeps files (A3) | **6 passed**. Stats run (40 kills each): atomic **40/40 loadable, 34 killed mid-write**; in-place control **37/40 torn** |
| NG `tests/test_atomic_saves.py` | guardian helper (fsync order asserted: file fsync -> replace -> dir fsync), stale tmps, `NeuroGraphMemory.save()` set (A6/A7), ChannelRegistry + KillSwitch + ETModuleManager (A5/A9/A10), DistanceCache (A11), in-place control tears | see suite results |
| NG `tests/test_atomic_saves_activation.py` (protected commit) | `ActivationPersistence.write_state` (A8), full `NeuroGraphMemory.save()` incl. sidecar | see suite results |
| daemon `scripts/tests/test_cc_ng_daemon_atomic_saves.py` (4) | A12-A18 incl. last_good never absent (25 kills) | **4 passed** |
| UniAI `lenia/tests/test_atomic_saves.py` (2) | A19-A21 (real `FlowLeniaF1.suspend`, real `write_result`) | **2 passed** |
| TID `tests/test_atomic_saves.py` (2) | A22-A23 (real `save_stats`, real `_save_routing_state`) | **2 passed** |

### Whole-process proof on a copy of `~/.elmer/proto_weights.pt`

Copy taken right after the live 11:58:21 save (cmp-identical; 1,441,068,039 B; 313 tensors; lenia_steps 40).
`~/.cache/atomic-saves-20261014/scripts/elmer_whole_process_proof.py`: the parent loads the copy once, forks a child per
round that runs the REAL `ProtoUniBrainSocket.save_evolved_weights()` (paths -> the copy dir) in a loop, SIGKILLs it at a
random point in 0.1 s .. 1.5 x one save, then `torch.load(mmap=True)` + key set + a sample tensor + lenia state.

- One uninterrupted save of the 1.44 GB body: **94.9 s** (isolated, nice 10, 1 thread, ecryptfs).
- **12/12 rounds loadable, 0 torn, 12/12 kills landed mid-write** (a temp was present at every kill). Kills at 17-128 s;
  lenia_steps progressed 40 -> 4041 -> 6041 -> ... -> 12041, i.e. complete saves replaced the file between kills and
  every interrupted one left the previous file. Log: `~/.cache/atomic-saves-20261014/proof.log`.

### Existing suites, branch vs base (new failures only)

SUITE_RESULTS_PLACEHOLDER

## Protected-file commits needing Josh's "proceed"

- NeuroGraph `fc4dfdf` — `activation_persistence.py` `write_state()` (A8) + `tests/test_atomic_saves_activation.py`.
  Same path, same JSON bytes, errors still raised (#423). `c92dda5` stands alone without it (the activation sidecar then
  stays an in-place write). Josh's Syl's-Law steps apply: confirm the msgpack backups, then "proceed".
- No change to `neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py` (their saves go through the guardian helper,
  which is where the fsync was added).

## Re-vendor list

None required by this lane (no vendored file changed; Elmer's `ng_lite.py`, `ng_ecosystem.py`, `ng_tract_bridge.py`,
`ng_autonomic.py`, `ng_embed.py`, `openclaw_adapter.py` are byte-identical to this NG branch).
Pre-existing drift found: **TID's vendored `ng_lite.py` is stale** (python fallback writes JSON in place, D7) — re-vendor
TID from canonical. Optional hardening at the canonical source (would need a re-vendor everywhere): directory fsync in
`ng_lite._atomic_write`, `ng_autonomic.write_state`, `ng_tract_bridge` cursor/compaction, and the Rust `save_binary`.

## Install order for the Executive

1. **Before anything**: verified backups (sha256) of `~/.elmer/proto_weights.pt`, `proto_lenia_state.pt`, and the CC
   checkpoint set; Josh's "proceed" for `fc4dfdf` if it is to be included.
2. Merge (no restart yet): NG `c92dda5` (+ `fc4dfdf` if approved) into `cc-laptop-trial-s4-20261002`; Elmer `bd5cf15`,
   `95ddcc0`, `2f06958` into `cc-revendor-ng-20260921`; UniAI `bcc6c75` into main (or whichever UniAI checkout the daemon
   imports — confirm with the daemon's UniAI path); daemon `6dbee99f` into `cc-laptop-trial-s4a-20261002`. All four load
   into the ONE daemon process, so one restart covers them.
3. **Restart procedure — wait for EVERY writer to be quiet** ([[atomic-saves-and-quiet-writers]]). The process being
   stopped still runs the OLD, non-atomic code, and its shutdown watchdog `os._exit`s after 45 s:
   - wait for a fresh `ProtoUniBrain: weights saved` line in daemon.log, then confirm `~/.elmer/proto_weights.pt` and
     `proto_lenia_state.pt` mtimes are >= 60 s old and stable (a save rewrites the file for ~1-2 min; the next starts
     5 lenia steps later, ~8+ min);
   - a fresh `Checkpoint saved to` line (autosave ~5 min) and the commons persist after it;
   - no hunger probe running (`pgrep -f flow_probe.py` empty);
   - then stop, confirm every old daemon PID is gone, `torch.load` (or at least `unzip -l`) `proto_weights.pt` OK, then start.
   - After start: expect at most one `atomic_io: ... stale temp` / `Guardian: ... stale temp file(s)` WARNING per target;
     they name leftovers of killed saves (see open risks) — safe to delete once seen.
4. TID: merge `067f7d1` into `cc-laptop-tid-pool-20261003`. NOTE: the live `~/The-Inference-Difference-vps` checkout has
   **uncommitted** changes to `ng_ecosystem.py` and `openclaw_adapter.py` (someone's re-vendor) — resolve before pulling.
   Restart TID uvicorn separately (its writers are its own; its shutdown calls `save_stats` once — small, sub-second).

## Open risks / findings for the Executive

1. **Live proto state is the post-incident fresh body.** daemon.log 10:29:31 `weight restore failed (... central
   directory)`, 10:30:16 `Flow-Lenia F1 selected but not initialised (no engine state)`. The live file copied at 11:58 had
   lenia_steps 40 and **no `flow_lenia_f1` key**; it keeps being saved every ~8-17 min. The pre-E1 archive is at
   `~/.elmer/archive/proto_weights.pt.pre-E1-20261013T161415`. Whatever restore is planned has not taken effect in the
   running process (not touched by this lane).
2. Stale temps of earlier killed saves in the live checkpoints dir (~1.44 GB): `main.tmp-1662741.msgpack` (Oct 2),
   `vectors.tmp-{219991,2472663,534504,1657484}.msgpack` (Sep 7 - Oct 4). The new code reports them once (WARNING), never
   deletes them; deleting is Josh's call. Each future killed proto save leaves a ~1.44 GB `proto_weights.pt.tmp-<pid>-*`
   (reported, not deleted) — repeated hard restarts accumulate disk use.
3. The 45 s shutdown watchdog vs a ~95 s proto save: a restart during a save now loses only that save's progress (the
   previous file stands), but a shutdown-time `unload()` save will usually be killed. Lengthening the grace or skipping
   the shutdown save is Josh's design call.
4. A1/A2 are two files: a kill between them leaves the new body with the previous Lenia baselines (as before; restore
   tolerates it). Single-file pairing would be a format change.
5. D2 (ingest-tract remainder rewrite in place) is the remaining live non-atomic write that can lose unabsorbed turns.
6. Directory fsync on the vendored writers (B1, B8, B9) — optional canonical hardening.
