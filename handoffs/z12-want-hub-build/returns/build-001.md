<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #11135, scope per Exec P420 + P421) — build-001
#   What: return for the want-hub (d) BUILD, CUT BACK to TESTS ONLY. NEW file; nothing else in the repo is changed by this commit.
#   Why: the Executive retracted the branch-build gate (P420, corrected by P421): neuro_foundation.py is PROTECTED and needs Josh's
#     backup confirmation + "proceed" BEFORE any change, even on a branch. Josh has not answered. This return says exactly what was
#     and was NOT done.
#   How: every number/hash below is from git rev-parse / sha256sum / the pytest and driver output of this session.
# -------------------
-->

# build-001 — want-hub (d): tests G/A/K/R committed and pushed; `neuro_foundation.py` NOT touched

Lane `want-hub-competition-d` · dispatch #11135 · repo [[NeuroGraph]] · branch `cc-laptop-want-hub-build-20260930` (from `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37`, re-fetched, unchanged) · plan cited by commit: `2e286e5` (last edit of `plan-005.md`), read at branch tip `e6c927a6a01c33337aa982ad85357cf0588b6e9d`. Related: [[The Laws]], [[Syl's Law]] (NeuroGraph `CLAUDE.md` §2), [[The Choice Clause]].

## 1. What was done / what was NOT done

| Item (brief) | Status |
|---|---|
| Two worktrees created | **DONE** — build `/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930` (branch above); base `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` (detached at `e4ebf982…`, `git status --porcelain` = 0 lines, never committed to) |
| Tests G, A, K, R (new file, synthetic seeded graphs only) | **DONE, committed, pushed** (§2) |
| Tests pushed BEFORE any test run | **DONE** — both commits pushed and `git ls-remote` confirmed before each pytest run (§4) |
| `neuro_foundation.py` (PROTECTED) commit | **NOT DONE — by directive (P420/P421).** Not edited, not staged, not committed, not pushed. Waiting for Josh's backup confirmation and "proceed". |
| Orchestrator `compete_protected_links`, additive `_prune_synapses` parameters | **NOT WRITTEN** (they live in the protected file) |
| Tests G/A/K/R **against the new code** | **CANNOT RUN** — the new API does not exist (§4). Nothing here is faked. |
| Merge / arming / daemon slice / PG-1 / delta pair / real graph / restart / install / `~/.bashrc` | **NONE done.** No real checkpoint, no Syl file, no tract, no embed/TID call. |

**Protected-file state (P421 report).** The file was never edited in this session, so there is nothing to leave or revert.
- `git status -sb` (build worktree, after both commits): `## cc-laptop-want-hub-build-20260930` — clean, no modified/untracked files.
- `git diff --stat e4ebf982 -- neuro_foundation.py`: empty (0 lines). `git diff --stat e4ebf982 HEAD`: only the two test files (§2).
- `sha256sum neuro_foundation.py` = `7080d57a6a0a16cc070eb39b4e788f8a02701b383c914cb46d3d1afd2344eaae`.
- Blob at base `e4ebf982:neuro_foundation.py` = `53494b7c56896d25040f3e7fd7c4046da7d0ab05`; blob at branch HEAD = `53494b7c56896d25040f3e7fd7c4046da7d0ab05` (**identical**).
- Primary checkouts (`~/NeuroGraph`, `~/docs`) were not edited; the docs assignment file and `plan-005.md` were read only.

## 2. Commits (full hashes) — tests only, no protected file in either

| # | hash | what |
|---|---|---|
| 1 | `408727408a6367700c2ef306bac1c42fd9fff226` | NEW `tests/test_want_hub_competition.py` + NEW `tests/want_hub_golden_driver.py` (helper, not collected: builder, state hash, reference caller-side sets, script entry for G) |
| 2 | `c17089b6353f742a7063792442fbdd18d96ad3f7` | test-bug fix: the reference floor assertion is `>=` not `==` (a want↔want link is guarded via EITHER endpoint's list, plan §2.2, so a want can hold more than K guaranteed links in one direction). My assertion was wrong, the reference was not. |

Exact diff stat `e4ebf982..c17089b6`: `tests/test_want_hub_competition.py | 514 +`, `tests/want_hub_golden_driver.py | 427 +` — 2 files changed, 941 insertions(+). This return file is a third, separate commit (hash reported in the closing message; a commit cannot contain its own hash). "A NEW test file" was the brief; I used two files because test G's driver must run as a script in each checkout (it cannot import pytest, and the base checkout cannot import the branch's test module) — flagged in case that should be one file.

## 3. P379 preamble (module under test = the worktree copy)

The test file **raises at import** if `neuro_foundation` resolves anywhere but its own worktree, and asserts every NG module already in `sys.modules` resolves inside it (`test_p379_module_under_test_is_this_worktree`). Each G subprocess is pinned to its own checkout and its run is VOID (exit 3) if the printed path is not that checkout. **My login shell has `PYTHONPATH` and `NG_EMBED_REMOTE` set** (the plan's known hazard); every run used `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1`.

```
cwd /home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930
sys.path[0:2] ['.', 'tests']
neuro_foundation.__file__ /home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930/neuro_foundation.py
NG modules in sys.modules: ['neuro_foundation', 'ng_tract', 'ng_tract.ng_tract']   (ng_tract = the native Rust store, site-packages)
PYTHONPATH None
pytest preamble line: P379 neuro_foundation.__file__=…/z12-want-hub-build-20260930/neuro_foundation.py git_rev=408727408a63… base_rev=e4ebf982b198… new_api_present=False
```

## 4. Test results (ONE run of the new file after the push; a first run stopped at my test bug, fixed in commit 2, both runs after a push)

Command: `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 -m pytest tests/test_want_hub_competition.py -p no:cacheprovider -q -rA` → **10 passed, 1 failed, 16 errors in 22.4 s** (MemAvailable ≈ 6.9 GB; tests are small synthetic graphs, well under 1 GB).

**PASS today, on base (harness self-checks + G's default path):** `test_p379_…`, `test_seed_covers_every_prune_boundary`, `test_reference_eligibility_matches_the_base_default_path` (my read-only eligibility reference == what the REAL base function removes), `test_reference_sets_are_consistent`, `test_G0_control_base_vs_base_is_deterministic[door_a|door_b|direct_defaults]` (control: the harness reproduces itself exactly, incl. checkpoint bytes), `test_G_default_path_base_vs_branch[door_a|door_b|direct_defaults]`.

**FAIL today BY DESIGN (need the new API; collected, not skipped, not xfailed):**
- 16 × **ERROR at setup** with the message `NEW API ABSENT at …/neuro_foundation.py: Graph._prune_synapses has no keyword-only ['competing_ids','excluded_ids','max_removals','order_key','report'] (protected-file commit not yet made — Exec P420/P421 …)`: all of A (2), K (8), R (6 incl. the parametrized rim test ×2). (Errors, not failures, because the check is a fixture; they are counted as failing.)
- 1 × **FAILED** `test_G_all_new_parameters_at_defaults_equal_the_base_default_path` — `assert "unavailable: _prune_synapses has no keyword-only […]" == 'ok'`.

### G — read this before relying on it
- **Two SEPARATE checkouts, two processes, same seeded graph** (`PYTHONHASHSEED=0`, counter-minted synapse ids): base `e4ebf982…` vs branch. Compared: return value, removal ORDER and count, `pruned` events (count, timestep), full post-state hash (every synapse field, store `items()` order, `_dirty_synapses`, `_synapse_confirmation_history`, node set, timestep), AND the serialized `Graph.checkpoint()` **bytes** (file `read_bytes()` equality). Exclusions: none from the payload; sidecars are not written by `checkpoint()` (plan §2.6).
- **RESULT: identical on all three scenarios — but VACUOUS as evidence about the new code.** Branch HEAD's `neuro_foundation.py` is byte-identical to base (§1), so branch == base by construction. What it does prove: the harness is exact (the base-vs-base control passes), the seeds are non-vacuous (176 / 189 / 176 synapses removed), and the comparison will be meaningful the moment the protected commit lands. **G's branch side cannot mean anything until then.**

| scenario | checkout | `neuro_foundation.__file__` (dir) | git rev | removed | return | `state_digest` (16) | serialized sha256 |
|---|---|---|---|---|---|---|---|
| door_a (`_structural_plasticity`, :3495) | base | `z12-want-hub-base-e4ebf982` | `e4ebf982b198…` | 176 | `[176, 0]` | `a9a5b2e42ac84610` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` |
| door_a | branch | `z12-want-hub-build-20260930` | `c17089b6353f…` | 176 | `[176, 0]` | `a9a5b2e42ac84610` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` |
| door_b (Door-B tail, :2891: `prime_and_propagate` write_mode + `tonic_ages_substrate`) | base | same | `e4ebf982b198…` | 189 | `[1, 1]` | `65942967f9c55dc0` | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` |
| door_b | branch | same | `c17089b6353f…` | 189 | `[1, 1]` | `65942967f9c55dc0` | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` |
| direct_defaults (`_prune_synapses()`) | base | same | `e4ebf982b198…` | 176 | `176` | `a9a5b2e42ac84610` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` |
| direct_defaults | branch | same | `c17089b6353f…` | 176 | `176` | `a9a5b2e42ac84610` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` |

(`git_rev` for the branch column is the tip when the driver ran; sha256 of each serialized file from `sha256sum` on the scratch files in `/tmp/want_hub_g/`. Full state digests are printed in the driver JSON; 16 hex shown.)

**Limits of G here (state plainly):** (1) the synthetic graph has **no hyperedges**, so the `list(he.member_nodes)` hash-order-dependent serialization path (plan §2.6) is not exercised — byte equality on real graphs is PG-1's job; (2) `PYTHONHASHSEED` was pinned to `0`; (3) the "all-new-parameters-at-defaults" and the gating clause (no unconditional sort/slice — an order that differs from insertion order is observable because seeds are random) become testable only once the API exists; the gating clause is covered by G's removal-ORDER comparison; (4) native-store float precision is the store's — boundary values compare identically on both checkouts by construction.

## 5. Additivity proof
**Not applicable yet** — there is no `_prune_synapses` diff (the protected file is untouched). It will be read line by line at the protected commit.

## 6. What I did NOT verify (numbered)
1. **Tests A, K, R have never run against any implementation.** Their assertions come from plan-005 as written; only the harness helpers (eligibility reference, set reference, seed coverage, control) are validated against the real base function. Expect the first run against the new code to expose test bugs as well as engine bugs — the `>=` fix in commit 2 is an example.
2. Existing suites: **not run.** With the protected file unchanged a comparison against the #761 known-red baseline would be empty; the two or three existing files that exercise `_prune_synapses` were **not identified** beyond a grep (only `neuro_foundation.py` and a comment in `openclaw_hook.py` name it; `test_identity_protection.py` covers the orphan/sprout guards). To do at the protected commit.
3. Tests D (daemon slice) and PG-1 (real-graph golden): out of scope, not written.
4. Reviews read: `le-023` and `checker-023` in full; `le-018`, `le-020`, `checker-020`, `checker-021` were **searched** for engine/build-relevant content (every hit is already folded into `plan-005.md`; none changes the engine surface), **not read end to end** — the brief said in full; I did not.
5. Whether the PreToolUse Syl's-Law hook would gate an edit of the protected file in the worktree (`le-023` D6 says its check is a literal `$HOME/NeuroGraph/…` path). Moot: I edited no protected file, and no hook gated any edit I made.
6. Vault docs/dev-log for this change: **not updated** (docs primary not edited per the brief; the only artifact is this return). Punchlist not edited (location/owner not confirmed) — items in §8 are for the zone manager.

## 7. Plan ambiguities I hit — each test says `READING:` where it chose; nothing chosen silently
1. **Names.** The plan says the parameter names are "illustrative" (§4.2) and the orchestrator name a "suggestion" (§4.3). The tests bind to exactly `competing_ids, excluded_ids, max_removals, order_key, report` (keyword-only) and `compete_protected_links(topk, budget)`. If the build names them differently the tests need a one-line rename.
2. **Types and returns.** §4.2 says "ids" and a "mapping"; the tests pass `set` for both id arguments and a `dict` for `order_key`. The orchestrator's return ("returns the counts", §4.3) has no stated shape — the tests assert **nothing** about it, only graph state and the one internal `_prune_synapses` call. Where the INFO record is emitted (orchestrator "builds", daemon emits) is untested here.
3. **"Arena".** §4.3 says `competing_ids = arena \ (G ∪ last-link)` but I found no one-line definition of "arena"; from §5.1 ("124,437 = 124,232 arena + 205 rim↔want, frozen") I read it as: non-F synapses touching a protected non-constitutional node. Tests use that.
4. **Height key (§4A.2).** (a) A want↔want link takes the **larger** endpoint height — the plan flags this as an open policy point for the Executive; tests use the larger. (b) `c_w` is counted **after** the last-link hold-back and counts all competing links, eligible or not (as §4A.2 says). (c) Whether the key's static tuple is `(-height, -inactive_steps, weight, synapse_id)` is taken from §4A.2 verbatim.
5. **Removal order after truncation** is not stated in words (§4.2(e)/(f) imply key order). `test_R_removed_set_is_the_top_B…` (set) and the order assertion are kept separate so only the order half depends on this reading.
6. **Return value in competing mode** — assumed = removed count after truncation (`len(to_prune)`), as §4.2(h) says of the `pruned` event.
7. **Validation edges the plan leaves open:** a `bool` `max_removals`; an EMPTY competing set in competing mode; a float `5.0` and a string `"5"` — the tests require both to refuse (§4.2(d) "an `int ≥ 1`"), `True`/empty are not tested. "Supplied together" (§4.2(d)) is tested in both directions (excluded without competing also refuses).
8. **Visit-count test (K4)** is stricter than the plan's words: it wraps the store and fails on any whole-table walk **or any read of a non-competing id** — so an implementation that validates by reading `self.synapses[sid]` for *excluded* ids would fail it. The plan says only "iterate ONLY competing_ids". Reviewer to decide if that strictness is wanted.
9. `report['eligible']` is `len(to_prune)` before truncation (§4.2(g)); tested against a read-only reference computed BEFORE the call.
10. **Stale text in the plan itself (not edited, read-only):** `plan-005.md` TOP NOTICE (`:177`, "nothing here authorizes a build") and §9 "BUILD GATE, RULED by Exec P419" are contradicted by P419's own retraction in P420/P421; `le-023` D5/D6 already flag the TOP NOTICE and the §9-vs-`CLAUDE.md` conflict.

## 8. For the zone manager
- **Checkpoint reached.** Everything before the protected commit is done and pushed. To continue: Josh confirms the backup of both msgpack files and says "proceed" → then `neuro_foundation.py` alone in its own commit, pushed, then the same test file is run (expect A/K/R + the explicit-None G test to turn green or expose bugs, and G's default-path comparison to become meaningful).
- Suggested punchlist rows (not filed by me): (i) `plan-005` §9/TOP NOTICE need a P420/P421 note (owner: plan author); (ii) the one-file vs two-file question in §2; (iii) `le-023` D6 (path-literal Syl's-Law hook) stays open with Josh.
- Nothing merged, armed, wired, restarted or installed. The real `~/.bashrc` was never written. No secret was read or printed.
