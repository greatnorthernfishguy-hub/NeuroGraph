# #808 — build-001 RETURN (plan + what was built + evidence)

Lane `unbounded-want-twin-808` · owner Z12 · dispatch #10694 · worker: Claude Code (Sonnet 5.5), 2026-09-30
Branch `cc-laptop-want-twin-808-20260930` · base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Status: **returned, unreviewed.** Nothing is merged, wired to a live consumer, or restarted. The worker does not accept its own work.

Commits (each pushed by name to `origin cc-laptop-want-twin-808-20260930`):

| Step | Content | `git rev-parse HEAD` after the commit |
|---|---|---|
| 1 | plan (`handoffs/z12-want-twin-808/plan-001.md`), committed before any code | `420e547105536c375d01efd980af70806e8fe595` |
| 2 | code (`cc_ng_organism.py`) + tests (`tests/test_cc_want_twin_808.py`) | `333585f1ce3e255d8e3461fe0b40b7e5e0ef95da` |
| 3 | this return doc | reported in the worker's closing message (a commit cannot contain its own hash) |

Files changed vs base: `cc_ng_organism.py` (M), `tests/test_cc_want_twin_808.py` (A), `handoffs/z12-want-twin-808/plan-001.md` (A), and this file (A). `git diff e4ebf982..HEAD` over `cc_ng_host.py`, `neurograph_rpc.py`, the four protected files and the six vendored files is **0 lines**.

## 1. Plan (summary; full text in `plan-001.md`)

Recommendation: **(b)**, make the guarded `surface_wants` the single implementation and have `surface_wants_for_graph` delegate to it. Not (a): repointing the host would change the id every want gets (`want::` → `cc:want::`), edit the VPS-touching `cc_ng_host.py`, and break `tests/test_cc_deposit_step.py:239` (`monkeypatch.setattr` on a removed name) and `tests/test_cc_host_stop_door.py` (drives the real twin).

Callers at base: `surface_wants_for_graph` has ONE production caller, `cc_ng_host.py:698/700` (VPS-side host, inside `_deposit`). The laptop daemon `scripts/cc-ng-daemon.py` (docs repo) never calls it; it calls only `surface_wants` (`:2096`, `:2116`). The `neurograph_rpc.py` unbounded regex is at **`:4945`** at this base (the assignment and the older changelog say `:4902`); cited only.

## 2. Premise correction the reviewers should see first

The assignment says the twin produced the 118 oversized wants. The derived laptop data say the laptop's 118 came from `surface_wants` **before** its 09-16 bound:

- `analysis-scratch/want-rows-laptop.json` (derived rows, manifest `saved_at 2026-09-23T10:46:14Z`): all **182** want nodes have prefix `cc:want::`; **0** have `want::`. 118 are over 600 chars, 91 begin with a backtick, 39 look genuine.
- Git: `surface_wants` at `5a974c0` (2026-07-04) already minted `cc:want::` with the unbounded regex; the bound is `d75efeb` (2026-09-16); the twin landed at `5b4afa5` (2026-08-18).

The twin is still the same defect, alive and unbounded, on the only path that calls it (the host). It remains a prerequisite for the S4 start on that basis. I have **no VPS graph data**; whether the VPS holds oversized `want::` nodes is unknown to me.

## 3. What was built (`cc_ng_organism.py`, +68 / −61)

1. `surface_wants_for_graph(graph, vdb=None)` is now `return surface_wants(graph, vdb, "cc_authored", id_prefix=_WANT_ID_PREFIX_TWIN)`. Name, signature, return shape and `want::` ids unchanged. Its private copy of the parser (unbounded regex, its own create loop) is **deleted**, so one implementation stands (LAW 3).
2. `surface_wants` gains one keyword-only, defaulted parameter `id_prefix` (default `"cc:want::"`, today's ids). Positional callers are untouched.
3. Two named constants beside `WANT_MAX_CHARS`: `_WANT_ID_PREFIX = "cc:want::"`, `_WANT_ID_PREFIX_TWIN = "want::"`. The `600` is not copied anywhere; the cap stays `WANT_MAX_CHARS` via `_WANT_RE`. No new env key (LAW 5).
4. The skip rules are **unchanged**. Skips are now counted: `backtick` (matched, backtick-preceded), `nested` (matched, contains a marker), `unbounded` (a `[WANT]` opener that no bounded match starts at or lies inside; `_WANT_RE` never returns these, so they were silent by construction).
5. Logging choice: one line per call, `want surfacing skipped %d span(s) (backtick=%d, nested=%d, unbounded=%d)`. **WARNING when the total differs from the previous call in this process (first sight, growth, or shrinkage); DEBUG with the count otherwise.** Reason: a skipped span stays in its conversation node, so the same nodes are re-scanned on every deposit and autosave; an unconditional WARNING would repeat forever. Counts only: no text, no ids. State is one module-level int (`_want_skip_last`), written under `graph._step_lock`.
6. Changelog header added at the top of `cc_ng_organism.py`, and the test file carries its own.

### Documented behaviour difference (reviewers: please confirm)

Only on the delegated path, only when `graph.create_node` raises: the exception now **propagates** to the caller. The base twin swallowed it per want at DEBUG and moved to the next want. The one production caller, `cc_ng_host.py:703`, already has an `except Exception -> logger.debug`, so the hook still fails soft; the remaining wants in that call retry on the next deposit (idempotent). For well-formed input on a working graph nothing differs. Covered by `test_create_node_failure_now_propagates_where_base_twin_swallowed_it`.

## 4. What merging would change (merge = deploy, P329)

- **VPS host (Josh-gated):** per-deposit `want::` path stops materializing over-cap, backtick-preceded and nested spans (they were the mis-parse); well-formed wants are byte-identical (id, metadata, synapse, return). New skip-count log line. `create_node` failure semantics as above. Existing `want::` nodes already in a VPS graph are not touched (offline repair lane). `cc_ng_host.py` itself is unchanged.
- **Laptop:** the daemon never calls the twin, so its want behaviour is unchanged. It imports `~/NeuroGraph/cc_ng_organism.py`, so it picks up the new log line only after NG main reaches `~/NeuroGraph` and the daemon restarts. Neither was done.

## 5. Evidence

Command (run ONCE, from the worktree, `NG_EMBED_REMOTE` unset, `PYTHONPATH` unset because it points at the primary checkout `/home/josh/NeuroGraph`, `python -m` puts the cwd first on `sys.path`):

```
env -u NG_EMBED_REMOTE -u PYTHONPATH python3 -m pytest tests/test_cc_want_twin_808.py tests/test_cc_want_bounds.py -s -p no:cacheprovider -q
```

Output:

```
[808 P379] worktree              : /home/josh/NeuroGraph-worktrees/z12-want-twin-808-20260930
[808 P379] cc_ng_organism.__file__: /home/josh/NeuroGraph-worktrees/z12-want-twin-808-20260930/cc_ng_organism.py
[808 P379] NG modules in sys.modules BEFORE import: none
[808 P379] neurograph_rpc in sys.modules now     : False
[808 P379] NG modules in sys.modules now         : {'cc_ng_organism': '/home/josh/NeuroGraph-worktrees/z12-want-twin-808-20260930/cc_ng_organism.py'}
.........................
25 passed in 0.92s
```

17 new tests + the 8 existing `test_cc_want_bounds.py` tests. The new ones cover: P379 module-path guard (fails if `cc_ng_organism` or any NG module resolves outside the worktree); the base twin really does materialize the oversized span (so the comparison is against the real defect); oversized / backtick / prose-swallowed / nested spans are not materialized via the twin, with the same skips as `surface_wants`; **golden** well-formed want vs the base twin loaded from `git show e4ebf982:cc_ng_organism.py` (ids, metadata, synapses, return dicts, order); golden after a second call; idempotence; Choice Clause floor (a want to leave is a want node like any other, identical to base); return shape for existing positional callers, `vdb=None`, `graph=None`; guarded default prefix `cc:want::` and `id_prefix` keyword-only; skip counts (WARNING on first sight, DEBUG on repeat, WARNING on growth; no text or ids in messages); clean corpus logs nothing; the create_node difference.

Not run, per the assignment: the full suite. Not exercised: a plain `pytest` with `PYTHONPATH` pointing at `~/NeuroGraph` (the P379 guard is written to fail in that case, but I did not run the deliberately-wrong configuration, because the target file was to be run once).

## 6. Flags for the punch list (not fixed here)

1. **Twin/guarded id-prefix duplication on the VPS host:** the host runs the twin per deposit (`want::<sha1>`) and the guarded function per autosave (`cc:want::<sha1>`), so one text yields two want nodes. Unchanged by #808 (pinned by `test_two_prefixes_still_coexist_on_one_graph_flagged_not_fixed`). Needs a Josh ruling on the surviving prefix; unifying rewrites ids on a live graph.
2. `neurograph_rpc.py:4945`: identical unbounded regex on Syl's `syl_authored` path. Canonical file; cited only.
3. The `cc_ng_organism.py` 2026-09-16 changelog cites `neurograph_rpc.py:4902`; the line at this base is `:4945`.
4. `create_synapse` failure is swallowed with a bare `pass` in `surface_wants` (as it was in the twin): a want node can exist with no synapse to its source, silently.
5. **#809** (`cc_ng_host.py:1517-1534` and `cc-ng-daemon.py:2095-2119`, one DEBUG-swallowed `try` over four calls): untouched; my WARNING is emitted from inside the want function, before any later raise, so it is not lost to that swallow.
6. A read-only probe of the VPS graph (Josh-gated) would show whether it holds oversized `want::` nodes; that is what decides the size of the offline repair there.

## 7. Review and rules attestation

- Reviewer pair (cross-family + law-enforcer) has not been run by me; NG merges first (P329).
- No protected file edited (`neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`). No vendored file edited (`ng_lite.py`, `ng_tract_bridge.py`, `ng_ecosystem.py`, `openclaw_adapter.py`, `ng_autonomic.py`, `ng_embed.py`). `neurograph_rpc.py` cited only.
- No daemon or unit started or stopped. Nothing read or written in `~/.claude/plugins/neurograph`, `~/NeuroGraph/data/checkpoints`, or the live tract; no primary checkout touched. The docs worktree was read-only for me (the assignment, `scripts/cc-ng-daemon.py`, and the analysis-scratch derived JSON; no graph was loaded).
- Secrets: none used. `NG_EMBED_REMOTE` was set in my shell; unset by name for every Python run.
- `git pull --rebase` was run around each commit; it reported "up to date" every time it ran. On the code step the pre-commit attempt refused (uncommitted changes), so I committed and re-ran it before pushing.
