# #808 — retire the unbounded `surface_wants_for_graph` twin: PLAN

Lane `unbounded-want-twin-808` · owner Z12 · dispatch #10694 · worker: Claude Code (Sonnet 5.5), 2026-09-30
Worktree `/home/josh/NeuroGraph-worktrees/z12-want-twin-808-20260930` · branch `cc-laptop-want-twin-808-20260930` · base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Status: PLAN ONLY. This file is committed before any code. Nothing here is merged, wired, or restarted.

Placement note: the assignment names `handoffs/z12-want-twin-808/returns/build-001.md` with no repo. I placed the plan and return under `handoffs/z12-want-twin-808/` **in the NeuroGraph worktree** because that is the only branch I was told to commit and push. The docs repo was read-only for me. Z12 may relocate them.

All line numbers below are at base `e4ebf982` unless a different repo is named.

## 0. Re-verification of the defect, and one correction to the premise

Re-verified at base:

- `cc_ng_host.py:696-704` (inside `_deposit`, `cc_ng_host.py:637`) calls `cc_ng_organism.surface_wants_for_graph(ng.graph, vdb)` on every deposit.
- `surface_wants_for_graph` (`cc_ng_organism.py:1128-1195`) uses the literal `re.finditer(r'\[WANT\](.*?)\[/WANT\]', content, re.DOTALL)` (`:1163`). It has no length cap, no backtick guard and no nested-marker guard.
- The guarded `surface_wants` (`cc_ng_organism.py:1517-1572`) uses `_WANT_RE` (`:1514`, capped at `WANT_MAX_CHARS = 600`, `:1512`), the backtick guard (`:1549`) and the nested-marker guard (`:1556`).
- Syl's copy `neurograph_rpc.py::_surface_wants` (def `:4914`, call `:4991`) carries the identical unbounded regex at **`:4945`** (`for m in re.finditer(r'\[WANT\](.*?)\[/WANT\]', ...)`). The assignment and the 2026-09-16 changelog in `cc_ng_organism.py` (line ~365) cite `:4902`; at this base the line is `:4945`. **CITED ONLY. Not edited, not touched.**

**Correction to the premise (evidence, not opinion).** The assignment says the twin "is the mis-parse SOURCE that produced the 118 oversized wants". For the **laptop graph** the derived data say otherwise:

- Source: `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/want-rows-laptop.json` (derived rows; produced by `want_probe.py` streaming a copy of the laptop `main.msgpack`, manifest `saved_at 2026-09-23T10:46:14Z`, 7253 nodes). I read only that JSON. No graph was loaded by me.
- All **182** want nodes have the id prefix **`cc:want::`**. **Zero** have the twin's `want::` prefix. Counts by that prefix: 182 total, 118 over 600 chars, 91 beginning with a backtick, 39 "genuine-ish" (<=600, no leading backtick, no embedded marker). Every row has provenance `cc_authored`.
- Git history agrees: `surface_wants` at `5a974c0` (2026-07-04) already used id prefix `cc:want::` with the *unbounded* `_WANT_RE = re.compile(r"\[WANT\](.*?)\[/WANT\]", re.DOTALL)`; the bound landed in `d75efeb` (2026-09-16). The twin landed at `5b4afa5` (2026-08-18).
- The laptop daemon (`scripts/cc-ng-daemon.py`, docs repo) never calls the twin (section 1). So the laptop's 118 oversized nodes were produced by `surface_wants` **before** its 09-16 bound, not by the twin.

Consequence: the twin is not the source of the laptop's 118. It **is** the same defect, still standing unbounded, on the one path that calls it (the VPS host, section 1). That is why it is still a hard prerequisite of S4: any host process that runs `cc_ng_host._deposit` re-creates the mis-parse in whatever graph it holds. I have no VPS graph data, so I make no claim about how many `want::` nodes exist there. That is a question for the offline repair lane (read-only probe of the VPS graph, Josh-gated).

## 1. Every caller (file:line) and which path it runs on

`grep -rn "surface_wants_for_graph\|surface_wants"` over the NeuroGraph worktree, plus the docs-repo `scripts/` tree (worktree `daemon-recall-756-20260930`, HEAD `5a2e1c5`), read-only.

### `surface_wants_for_graph` (the unbounded twin)

| Where | What | Path |
|---|---|---|
| `cc_ng_organism.py:1128` | definition | shared module |
| `cc_ng_host.py:698` / `:700` | the ONLY production caller: import and call inside `_deposit` (block `:693-704`, wrapped in `try/except -> logger.debug`) | **VPS-side host path.** `cc_ng_host` is imported and started by `neurograph_rpc.py:2153-2156` (`_init_cc_host_bg`), i.e. it lives in the gateway/sidecar process. |
| `scripts/cc-ng-daemon.py` (docs) | **no reference at all** (checked by grep over the whole docs `scripts/` tree) | **Laptop daemon path: does NOT call the twin.** |
| `tests/test_cc_deposit_step.py:239` | `monkeypatch.setattr(cc_ng_organism, 'surface_wants_for_graph', lambda *a, **k: [])` | test; requires the name to still exist |
| `tests/test_cc_host_stop_door.py:30,34,42,163,203` | the REAL twin runs end-to-end through `_deposit`; asserts one want node with the right text/state | test; requires the name, and requires a well-formed want to still materialize |

### `surface_wants` (the guarded function)

| Where | What | Path |
|---|---|---|
| `cc_ng_organism.py:1517` | definition | shared module |
| `cc_ng_host.py:1519,1531` | `_autosave_loop` (`:1506`), inside the ONE `try` at `:1517-1534` (`except -> logger.debug`) | VPS-side host |
| `scripts/cc-ng-daemon.py:2096,2116` (docs) | `_autosave_loop`, same one-`try`-per-cycle shape (`:2119` `logger.debug`) | **Laptop daemon path.** The daemon puts `~/NeuroGraph` on `sys.path` (`cc-ng-daemon.py:574-576`) and imports `cc_ng_organism` from there. |
| `tests/test_cc_want_bounds.py:54,67,73,79,84,89` | the bounds tests; six calls, positional `(g, vdb)` | test |
| `tests/test_cc_deposit_step.py:559` | `monkeypatch.setattr(cc_ng_organism, 'surface_wants', lambda *a, **k: None)` | test |

Other repo mentions of `[WANT]` parsing: `neurograph_rpc.py:1708` (a markup *stripper*, not an extractor) and `:4914-4991` (Syl's own `_surface_wants`, cited only). `tests/test_conversational_recall.py:355-406` and `tests/test_tonic_bridge.py:160` exercise Syl's `rpc._surface_wants`; they are unaffected.

### The two functions on ONE host (a finding, not part of this fix)

On the VPS host **both** run in the same process against the same graph: the twin per deposit (`want::<sha1>`), the guarded one per autosave (`cc:want::<sha1>`). For one well-formed want text they create **two** want nodes with different ids, because each function's idempotence check looks only at its own prefix. So on the host path a real want is already materialized twice. Recorded for the punch list; **not changed here** (unifying the prefix would rewrite ids on a live graph and needs Josh).

### #809 interaction (`_autosave_loop`)

`cc_ng_host.py:1517-1534` wraps `drain_ingest_tract`, `cc_update_probation`, `surface_wants`, `generate_emergent_want` in a single `try`; the `except` logs at DEBUG only, so a raise in an early call aborts the rest of that cycle invisibly. That belongs to lane **#809** and I leave it byte-for-byte. Interactions with my change:

1. My skip-count log line is emitted from **inside** the want function, before any raise from a later call, so it is not lost to that swallow.
2. Any exception my change lets propagate (section 3, difference 6) inside `_deposit` is caught at `cc_ng_host.py:703` (its own DEBUG `except`); inside the autosave it is caught by the #809 `try`. Neither is made worse than today, and neither is fixed by me.
3. #809 may later widen or split that `try`. Nothing in this change depends on its current shape.

## 2. Which function produced the laptop graph's 182 wants

`cc:want::` (all 182). See section 0. That is the guarded function's own prefix, in its pre-09-16 unbounded form. No `want::` node exists in that graph.

## 3. Differences between the two functions

| # | Aspect | `surface_wants_for_graph` (twin) | `surface_wants` (guarded) |
|---|---|---|---|
| 1 | Regex | `\[WANT\](.*?)\[/WANT\]` unbounded | `_WANT_RE` = `\[WANT\](.{1,600}?)\[/WANT\]`, `WANT_MAX_CHARS` |
| 2 | Backtick guard | none | skip if the char before `[WANT]` is a backtick (`:1549`) |
| 3 | Nested-marker guard | none | skip if inner contains `[WANT]` / `[/WANT]` (`:1556`) |
| 4 | Idempotence key / id | `"want::" + sha1(inner)[:16]` | `"cc:want::" + sha1(inner)[:16]` |
| 5 | Node metadata | `kind=want, want_text, want_state=open, provenance="cc_authored"` (hardcoded), `source_node, creation_mode=conversational` | identical keys; `provenance` is a parameter, default `"cc_authored"` |
| 6 | `create_node` failure | caught per want, `logger.debug`, loop continues | propagates to the caller |
| 7 | `create_synapse(nid, want_id, weight=0.3)` failure | swallowed (`pass`) | swallowed (`pass`) (same; pre-existing; noted, not touched) |
| 8 | Lock | `_cc_mutation_lock(graph)` | `_cc_mutation_lock(graph)` |
| 9 | Signature | `(graph, vdb=None)` | `(graph, vector_db, provenance="cc_authored")` |
| 10 | Return | list of OPEN want dicts `{id,text,provenance,state,source}`, existing open wants first-seen plus newly created | identical shape and semantics |
| 11 | Writes | one want node + one synapse (`nid -> want_id`, 0.3) per new want | identical |
| 12 | Visibility of skips | none | none (a skip is silent; oversized spans never even come back from `_WANT_RE`, so nothing can count them today) |

## 4. Recommendation: option (b), delegate. Not (a).

**Chosen: (b) make the guarded function the single implementation; `surface_wants_for_graph` keeps its name, signature and return shape and delegates to it.**

Why not (a) (retire the twin and repoint callers):

- It is not behaviour-identical for a well-formed want. The twin's id is `want::<sha1>`; the guarded default is `cc:want::<sha1>`. Repointing `cc_ng_host.py:700` would change the id every future want gets on the host (and, given the twin's per-deposit vs autosave split, could double-create against existing `want::` nodes). The assignment requires the SAME id.
- It edits `cc_ng_host.py` (VPS-touching), which (b) does not need to.
- It breaks `tests/test_cc_deposit_step.py:239` (`monkeypatch.setattr` on a removed name raises) and `tests/test_cc_host_stop_door.py` (which drives the real twin).

How (b) preserves the id without leaving two implementations (LAW 3, LAW 4):

- `surface_wants` gains ONE keyword-only, defaulted parameter, `id_prefix`, default = today's `"cc:want::"`. Positional call sites (all six in `test_cc_want_bounds.py`, the daemon at `cc-ng-daemon.py:2116`, the host at `cc_ng_host.py:1531`) keep working and produce byte-identical ids.
- The twin body becomes `return surface_wants(graph, vdb, "cc_authored", id_prefix="want::")`. The parsing, guards, dedup, node creation and return all live in ONE place.
- The two prefixes become two named module constants beside `WANT_MAX_CHARS`. The `600` is **not** copied anywhere; the bound stays `WANT_MAX_CHARS` through `_WANT_RE` (LAW 5, no new config key).

Documented behaviour change on the delegated path (only when `graph.create_node` raises; never for well-formed input on a working graph): difference 6. A raising `create_node` now propagates to the caller instead of being swallowed at DEBUG and skipping to the next want. The caller `cc_ng_host.py:703` already has a DEBUG `except`, so the hook still fails soft. The rest of that call's wants are retried on the next deposit (idempotent). I judge fail-loud-to-a-caller-that-already-catches strictly better than a silent per-want swallow, and it keeps the default of `surface_wants` itself unchanged. Reviewers should confirm.

## 5. What merging this would change (merge = deploy, P329)

Behaviour-identical for a well-formed want: same id (`want::<sha1[:16]>`), same node metadata, same synapse, same return dicts, on both prefixes. Verified by the golden test (section 7).

**VPS host (Josh-gated).** On the deploy of NG main:

- Per-deposit `want::` path: spans over `WANT_MAX_CHARS`, spans opened with a preceding backtick, and spans containing a nested marker are **no longer materialized**. Previously each became a `want::` node (the mis-parse).
- New log line from the want functions (section 6).
- `create_node` failure semantics (section 4).
- Existing `want::` nodes already in a VPS graph are NOT removed or edited by this change. That is the offline repair lane.
- `cc_ng_host.py` itself is unchanged; the change ships via the shared `cc_ng_organism.py`.
- The autosave `cc:want::` path is unchanged in behaviour (only the new log line).

**Laptop.** The daemon never calls the twin, so its want behaviour is unchanged. Because `cc-ng-daemon.py` imports `~/NeuroGraph/cc_ng_organism.py`, the laptop picks up the new log line (and the defaulted `id_prefix` parameter, unused) only once NG main is pulled into `~/NeuroGraph` **and** the daemon is restarted. I do neither.

## 6. Logging decision (skipped spans must not be silent)

- Count, per call, spans skipped by category: `backtick` (matched but backtick-preceded), `nested` (matched but contain a marker), `unbounded` (an opener `[WANT]` that no bounded match starts at or covers: an oversized or unterminated span; `_WANT_RE` never returns these, so they are counted by comparing opener positions with match spans).
- The skip rules themselves are unchanged. Only counting is added.
- **Choice: WARNING when the total skip count differs from the previous call in this process (first sight, or any change); otherwise DEBUG with the count.** Reason: a skipped span stays in its conversation node forever, so the same nodes are re-scanned every deposit and every autosave; an unconditional WARNING would repeat the identical line on every call, forever, which is a flood that hides real warnings. Change-detection surfaces the first occurrence and any growth at WARNING, with no new env key (per the assignment) and no new module-level timer. The line carries counts only: no text, no ids.
- Format: `want surfacing skipped %d span(s) (backtick=%d, nested=%d, unbounded=%d)`.
- Module state is one integer, read and written under `graph._step_lock` (already held for the whole scan).

## 7. Tests (`tests/test_cc_want_twin_808.py`, written after this plan is committed)

Fake in-memory graph/vdb only; never `~/.claude/plugins/neurograph`, no checkpoint, none of Syl's paths.

1. Session-start guard: print resolved `cc_ng_organism.__file__` and whether `neurograph_rpc` (or any NG module) is in `sys.modules`; **FAIL** if the resolved module is not the worktree copy (Exec P379).
2. Oversized swallowed span is NOT materialized (via the twin).
3. Backtick-preceded documentation marker is NOT materialized.
4. Nested-marker span is NOT materialized.
5. Well-formed want IS materialized with the same id/metadata/synapse as before. Golden comparison: run the SAME input through the base twin (loaded from `git show e4ebf982:cc_ng_organism.py` into a throwaway module, no filesystem writes to the repo) and through the branch twin; ids, metadata, and return dicts must match.
6. Idempotence: a second call creates nothing and returns the same open wants.
7. Delegated path returns the same shape to existing callers, positional `(graph, vdb)` and `vdb=None`.
8. Choice Clause floor: a well-formed "want to leave" is a want node like any other.
9. The guarded function's default id prefix is still `cc:want::` (positional callers unchanged).
10. Skip counting and log-level behaviour: WARNING on first sight, DEBUG on an unchanged repeat, WARNING again on growth; the message contains counts and none of the swallowed text.

Run: the new file ONCE plus the existing `tests/test_cc_want_bounds.py`. Not the full suite.

## 8. Files

Edited: `cc_ng_organism.py` (changelog header + additive change). No protected file (`neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`), no vendored file (`ng_lite.py`, `ng_tract_bridge.py`, `ng_ecosystem.py`, `openclaw_adapter.py`, `ng_autonomic.py`, `ng_embed.py`), and not `cc_ng_host.py`, `neurograph_rpc.py` or the docs-repo daemon.
Created: `tests/test_cc_want_twin_808.py`, `handoffs/z12-want-twin-808/plan-001.md` (this), `handoffs/z12-want-twin-808/returns/build-001.md`.

## 9. Flags for the punch list (not fixed here)

1. Twin/guarded id-prefix duplication on the VPS host (`want::` + `cc:want::` for one text). Needs a Josh ruling on which prefix survives.
2. `neurograph_rpc.py:4945` unbounded regex on Syl's `syl_authored` path (cited only; canonical file; needs Josh's approval).
3. Stale line cite `:4902` in the `cc_ng_organism.py` 2026-09-16 changelog (real line at base: `:4945`).
4. `create_synapse` failure swallowed with a bare `pass` in both functions: an orphan want node is silent.
5. #809 (`_autosave_loop` single DEBUG-swallowed `try`): untouched, owned by that lane.
6. The premise correction in section 0: a read-only probe of the VPS graph (Josh-gated) would show whether it holds `want::` oversized nodes.
