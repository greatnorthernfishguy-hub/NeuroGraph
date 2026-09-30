<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #11877) — build-002
#   What: return for the want-hub (d) ENGINE build. NEW file, docs only; nothing else on this branch is changed by this commit.
#   Why: the dispatch names `handoffs/z12-want-hub-build/build-002.md` (build-001 sits one level down, in `returns/`; I followed the dispatch
#     path literally — move it if you prefer the old convention).
#   How: every hash/number below is from git rev-parse / sha256sum / the pytest and driver output of this session. No secret involved.
# -------------------
-->

# build-002 — want-hub (d): the ENGINE change is built and pushed on its own branch; 26/27 tests pass, the 1 red is a proven test-side defect I did NOT edit

**Not self-accepted.** A fresh cross-family + law-enforcer delta pair reviews the engine diff next; PG-1 (the real-graph golden) is a separate pre-merge gate and was NOT run. No merge, no arming, no #825, no daemon/unit start or stop, no real checkpoint/vectors/Syl file/tract touched.

## 0. Headline (read this first)

| | |
|---|---|
| Engine branch | `cc-laptop-want-hub-engine-20260930` head **`8e5785322f910aefaf781fa3525bea2831fedd31`** (pushed; `origin/…` == local, verified after `git fetch`) |
| Diff vs base `e4ebf982b1989fd9066d610b94853bc68bf70d37` | **ONE file**, `neuro_foundation.py`, 252 insertions / 3 deletions (`git diff --name-only` = that file only) |
| Protected-file blob at base / at head | `53494b7c56896d25040f3e7fd7c4046da7d0ab05` / `96d12f507746ce168fffe6feba56e097c8832353` (sha256 of the head file `d1641399c031bcb14fd93bef787acd7d11b8a834592ead4cdf42799e8d83123f`) |
| Tests | `tests/test_want_hub_competition.py`: **26 passed, 1 FAILED** |
| The 1 failure | `test_A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets` — **a defect in the test, not the engine** (§5.3, proven). **I did not edit any test**; proposed fix in §5.3 for Z12 to rule on. |
| Test G (golden equivalence) | **PASS on both checkouts, byte-identical serialized checkpoints** (§5.2) |

## 1. Step Zero (done before touching anything)
- `git fetch origin` run. The go record **exists** on `origin/cc-laptop-want-hub-d-20260930` (head `e62fc2d35350694b535136c8976e2e0128fab843`) at `handoffs/z12-want-hub-d/approvals/josh-go-neuro-foundation-20260930.md`, **added by commit `a434525cd3cdf68da5f282aa319a2323715d3938`** (`git log --diff-filter=A`; `a434525` is an ancestor of the branch head).
- Confirmed in it: Josh's two fragments VERBATIM — *"That copy you already approved is fine, I guess? Unless there is some reason it shouldn't be..."* and *"So, proceed."*; the backup directory `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc/` with `main.msgpack` sha256 `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` and `vectors.msgpack` sha256 `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e` (both strings grep-matched in the file); the two-change scope (§4 of the record). Base `e4ebf982…` exists and **`origin/main` == base** (no drift on `neuro_foundation.py`).
- I **did not re-hash the two msgpack files** (they are not mine to read; the record itself says its values were supplied in the dispatch and not re-hashed). The record's own FLAG (§5: the backup Josh accepted is the CC-laptop copy, NOT Syl's own checkpoint; Exec Packet 441 makes the merge a new protected-file event for Syl) is **not mine to resolve and I did nothing about it**.
- **The hook:** the repo's `pretool_syls_law.sh` compares the edited path for equality with the literal `$HOME/NeuroGraph/neuro_foundation.py`, so it stayed **silent on a worktree path** (plan-005 §10 [R5e·LE23-D6] already records this). I did **not** create `.claude/hooks/.session_approved`, did not edit any hook, and did not treat the silence as approval — the safeguard here is the verified go record above.
- First `neuro_foundation.py` commit message quotes the hash: *"Josh's go recorded in a434525cd3cdf68da5f282aa319a2323715d3938"* (plan-005 §9 [R5g·LE28-G6]).

## 2. What was built (one commit, `8e578532`)
1. **`Graph._prune_synapses`** — five keyword-only parameters, all default `None`: `competing_ids`, `excluded_ids`, `max_removals`, `order_key`, `report`. Return stays `int`. Contract §4.2 (a)–(i). Validation = explicit `raise ValueError` (not `assert`) **before** the loop.
2. **`Graph.compete_protected_links(topk, budget)`** — the ONE new method. Builds F, G (per direction, §2.3 ranking), the last-link set, `competing_ids`, `excluded_ids`, the static HEIGHT key (§4A.2); captures `id → (pre, post, conducting)` BEFORE the call; makes **exactly one** `_prune_synapses` call under `self._step_lock`; builds the counts-by-want record from `report['removed_ids']`; logs it at INFO (even at 0 removed) and returns it. **No predicate copy, no `_remove_synapse_internal` call, no removal loop, no `_collect_orphan_nodes` call.** The builder is a nested function, so there is exactly one new method.
3. Changelog header entry at the top of the file's changelog.
4. **Not touched** (as ordered): `DEFAULT_CONFIG`, any config key, `CC_SNN_CONFIG`, the comments at `:78/:162/:194/:3286`, any other file.

## 3. Additivity proof — the `_prune_synapses` diff read line by line
`git diff e4ebf982..8e578532 -- neuro_foundation.py` in the function region has **3 removed lines, all listed here**; everything else is added.

| # | Base line (removed) | Replacement | Why behaviour is identical with all parameters `None` |
|---|---|---|---|
| 1 | `def _prune_synapses(self) -> int:` | same name, `*` then five `=None` keyword-only params, `-> int` | callers pass no arguments (ripple check: the only code callers are `:2891` Door B tail and `:3495` Door A, both in `neuro_foundation.py`; `openclaw_hook.py` mentions it only in a comment; the docs repo's `scripts/cc-ng-daemon.py` mentions it only in a docstring — line 1423 on the primary checkout, 1486 on the `daemon-recall-756` worktree — no call) |
| 2 | `for sid, syn in self.synapses.items():` | `for sid, syn in candidates:` | `candidates = self.synapses.items()` is assigned on the default path immediately before (no mutation between), so the same iterable is walked in the same order |
| 3 | `if (self._is_identity_protected(syn.pre_node_id) or` | `if not competing_mode and (self._is_identity_protected(syn.pre_node_id) or` (2nd line unchanged) | `competing_mode` is `False` ⇒ `not False and X` ≡ `X`; same operands, same short-circuit order |

Added lines, each classed:
- **Gated on a new parameter being non-`None` (never run on the default path):** the whole `if competing_mode:` validation block; the competing-mode `candidates` generator; `report["eligible"] = …`; the `order_key` sort; the `max_removals` slice; `report["removed_ids"] = …`.
- **Evaluated on the default path but inert:** `competing_mode = competing_ids is not None`; the pairing check `competing_mode != (excluded_ids is not None)`; the `max_removals is not None and …` and `report is not None and …` checks (all `False`/skipped with defaults); the `if competing_mode … else …` selection of `candidates`. They read only the new parameters and mutate nothing. **So the honest statement is "every added line is either gated on, or a pure predicate of, the new parameters" — NOT "purely additive", because of rows 1–3 above.**
- **Unchanged, byte-for-byte:** the three predicates and their `low_weight_steps` bookkeeping, `for sid in to_prune: self._remove_synapse_internal(sid)`, `if to_prune: self._emit("pruned", count=len(to_prune), timestep=self.timestep)`, `return len(to_prune)`. After the optional slice, `to_prune` *is* the removed list, so the event count is the post-truncation count (§4.2(h)).
- The competing-mode loop is a **separate iterator** over `competing_ids` with **no identity `continue`** applied to them (§4.2(i)(2)); the sort/slice/report are gated (§4.2(i)(1)).

## 4. P379 preamble (module under test is the intended copy)
Printed by `test_p379_module_under_test_is_this_worktree` (the file also refuses to load if `neuro_foundation` resolves outside its worktree):
`neuro_foundation.__file__=/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930/neuro_foundation.py git_rev=95154e33f299d70e4fdf5945ea1a189bcbd61acc base_rev=e4ebf982b1989fd9066d610b94853bc68bf70d37 new_api_present=True`
- **Read the `git_rev` correctly:** the integration worktree's HEAD is the TESTS head `95154e33`; the engine file is checked out into it uncommitted (`git checkout 8e578532 -- neuro_foundation.py`; its blob `96d12f50…` equals the engine commit's blob, checked with `git hash-object`). The engine commit's own rev is printed in §5.2 (driver run directly against the engine worktree).
- Env: `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1` (`PYTHONHASHSEED=0` set by the test's subprocesses). Python 3.12.3. Before the runs: load 2.55, MemAvailable ≈ 6.1 GiB (bounds: load < 6, ≥ 3 GiB). Synthetic seeded graphs only (≤ ~500 synapses); nothing near 1 GB; no embed/TID call.
- NG-module state: `ng_lite`, `ng_tract_bridge`, … — the P379 test asserts every already-imported NG module resolves inside the worktree; it passed.

## 5. Test results

### 5.1 Summary — `tests/test_want_hub_competition.py`, run from the integration worktree (engine pushed first)
**26 passed, 1 failed.** Run 1 (`-x`) stopped at the failure; run 2 (no `-x`, disclosed iteration run) gave the full picture. Both ran after the engine commit was pushed.

| group | tests | result |
|---|---|---|
| P379 + harness self-checks | `p379…`, `seed_covers_every_prune_boundary`, `reference_eligibility_matches_the_base_default_path`, `reference_sets_are_consistent` | 4/4 PASS |
| **G** golden, default path, base vs branch (two checkouts, two processes) | `G0` control ×3 (base vs base), `G_default_path_base_vs_branch[door_a|door_b|direct_defaults]`, `G_all_new_parameters_at_defaults_equal_the_base_default_path` | 7/7 PASS |
| **A** armed-path counter | `A_only_low_weight_steps_moves…` | PASS |
| **A** (orchestrator) | `A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets` | **FAIL — test defect, §5.3** |
| **K** contract | every refusal is `ValueError` and mutates nothing; missing endpoint; **explicit raises under `python -O`**; **competing mode visits only `competing_ids`**; report matches; eligible count; default path accepts `max_removals=None`+report; orchestrator refuses topk/budget < 1 | 8/8 PASS |
| **R** rim / floors / determinism | worst-state rim+floors+last-link `[B=7]`, `[B=10^6]`; determinism; top-B by the height key (removal order == key order); B > eligible removes all eligible; outside-arena synapses untouched | 6/6 PASS |

### 5.2 Test G on BOTH checkouts — printed module path, git rev, sha256 of each serialized checkpoint
Driver run directly (`tests/want_hub_golden_driver.py --checkout …`, `PYTHONHASHSEED=0`, deterministic uuid counter, `sys.path[0]` pinned; a run is void if the imported file is not inside its checkout — none was void). BASE = `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` (rev `e4ebf982b1989fd9066d610b94853bc68bf70d37`); ENGINE = `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930` (rev `8e5785322f910aefaf781fa3525bea2831fedd31`). Both printed `neuro_foundation.__file__` = `<checkout>/neuro_foundation.py`.

| scenario | BASE `Graph.checkpoint()` sha256 | ENGINE `Graph.checkpoint()` sha256 | compared keys identical | removals / return |
|---|---|---|---|---|
| `door_a` (`_structural_plasticity`, `:3495`) | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | **yes** | 176 / `[176, 0]` |
| `door_b` (Door B tail, `:2891`) | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` | **yes** | 189 / `[1, 1]` |
| `direct_defaults` (`_prune_synapses()`) | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | **yes** | 176 / `176` |
| all-new-params-explicit-`None` call (ENGINE) vs plain call (BASE) | (base `direct_defaults` above) | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | **yes** | `variant=ok` |

"Compared keys" = `return`, `removal_order` (order, not just set), `pruned_events`, the full `state_digest` (every synapse field in store `items()` order, `_dirty_synapses`, `_synapse_confirmation_history`, node set, timestep), `checkpoint_sha256`, before/after counts. Checkpoints were written to `/tmp/z12-wh-scratch/` (scratch, not a live path). Scenarios are **non-vacuous** (176/189 real removals; the test asserts `removal_order` non-empty for `door_a`/`direct_defaults`).

### 5.3 The one failure — `test_A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets` is wrong, the engine is not (PROVEN; I did not touch the test)
- **Trace:** the assertion that fails is not an engine assertion. The test does `s = drv.ref_sets(g, K)` *before* the call, then calls `g.compete_protected_links(K, B_SMALL)` (which removes up to 12 synapses), then — **after** the call — `ok = drv.ref_order_key(g, s.competing)` (line 269). `ref_order_key` does `g.synapses[sid]` for every id in `s.competing`; removed ids are gone ⇒ `KeyError`, inside the test's own reference helper.
- **Proof it is independent of the orchestrator** (scratch `/tmp/z12-wh-scratch/prove_test_A_defect.py`, not committed, no test edited): (1) a **direct** `_prune_synapses(**drv.good_kwargs(...))` removing 12 synapses, then `drv.ref_order_key(g, s.competing)` ⇒ the same `KeyError`; (2) the test's real intent with the reference key computed **before** the call: exactly 1 `_prune_synapses` call, `args == ()`, `competing_ids == reference`, `excluded_ids == reference` (and `F ⊆ excluded`), `max_removals == B`, and **`order_key == reference HEIGHT key` for all 195 keys** — all `True`.
- **Proposed fix (NOT applied; Z12 to rule — the brief says stop and report a provably-wrong test first):** in that test, move the line `ok = drv.ref_order_key(g, s.competing)` to immediately after `s = drv.ref_sets(g, K)` (before `g.compete_protected_links(...)`), and compare `… == ok` after. One-line move; no assertion weakened.
- Until that is ruled, **the record stands at 26 green / 1 red, and this is the only red.**

### 5.4 Existing tests near the prune path (plan §2.6: none call `_prune_synapses` directly; these reach it through `step()`/the exemption)
`tests/test_identity_protection.py`, `tests/test_integration.py`, `tests/test_stdp.py` (chosen by grep of `prune|structural_plasticity|_collect_orphan|identity_protected|grace_period`), run ONCE on BASE (read-only worktree) and ONCE on the integration worktree: **both 34 passed / 1 failed, per-test outcomes diff-identical.** The one red, `tests/test_integration.py::TestStructuralPlasticityPruning::test_speculative_synapses_pruned`, **also fails on clean base `e4ebf982`** ⇒ pre-existing. I did **not** check it against the #761 list of 72 (I did not run the full suite — P373), so I am not claiming it is one of those 72; only that it is red on base and unchanged by this diff. Not run: the full suite.

### 5.5 Extra checks the test file does not pin (scratch `/tmp/z12-wh-scratch/extra_checks.py`, not committed)
- **Determinism across processes/hash seeds:** 3 cycles of the orchestrator, `PYTHONHASHSEED` = 0, 1, 12345 ⇒ identical removal order (sha256 prefix `01e81431f1be7bf1`, 36 removals) and identical post-state digest. (R-7 in the test file only proves same-process.)
- **Re-entrancy:** the orchestrator under a caller-held `g._step_lock` (the daemon holds it) removed 12, no deadlock (RLock).
- **INFO record:** one record per call, emitted even when it removes 0 (eligible=0 removed=0 after exhaustion); record keys: `B, F_links, K_in, K_out, by_want, conducting_links_removed, eligible, floors_ok, held_back_last_link, protected_nodes, removed, timestep, wants_with_removals, wants_zero`.
- **Unspecified combinations (see §6 items 2–4):** default path + `max_removals=0`/`-1` ⇒ `ValueError`, state unchanged; + `report=[]` ⇒ `ValueError`, state unchanged; + `order_key={}` ⇒ `ValueError` **but counters were already advanced** (state NOT unchanged); default path + `max_removals=5, report={}` ⇒ returns 5, `len(removed_ids)`=5, `eligible`=160.

## 6. Plan ambiguities / choices I had to make — LISTED, for the pair and the Executive (none silently chosen)
1. **Want↔want height — the plan's own OPEN item (§4A.2, §10 "X7 detail").** I implemented the plan's stated rule, the **larger** of the two endpoint heights (the plan: "The plan gives it the larger… The numbers below use the larger"; the test reference does the same). If the Executive rules otherwise it is a one-expression change in `compete_protected_links`.
2. **Default path + `order_key` with a missing entry.** §4.2(f) says `order_key` is optional on the default path but is silent when an entry is missing. I raise `ValueError` (from `KeyError`) **after** the predicates ran and **before** any removal — so `low_weight_steps` counters HAVE moved (validation-before-the-loop is impossible here: the eligible list is unknown until the loop). No current caller uses the combination.
3. **Default path + `max_removals` given.** §4.2(d) requires `int ≥ 1` only in competing mode. I validate `max_removals` whenever it is not `None` (a negative slice would silently drop the wrong end). `bool` and non-`int` (including `numpy` ints) are refused; the daemon passes a Python `int` from env.
4. **`report` that is neither `None` nor a dict.** §4.2(g) says "if `report` is a dict the function fills…". I raise `ValueError` rather than silently ignore a list (an ignored report would be a silent failure).
5. **`cycle id` in the §4A.6 record.** The engine has no cycle clock; the record carries `timestep`. The daemon slice (separate lane) must stamp the cycle id. Not invented here.
6. **Orchestrator takes `_step_lock` itself** (re-entrant; harmless under the daemon's own hold). §4.3 says "call `_prune_synapses` once under `_step_lock`"; §4.7 says the daemon holds it. Both hold.
7. **Orchestrator emits the INFO record** (§4A.6: "the orchestrator builds the record … then emits one INFO record per dream pass"); the daemon's separate "not armed" INFO line is the daemon slice's. Level is WARNING instead of INFO if `floors_ok` is false.
8. **Sorted iteration.** Competing mode iterates `sorted(set(competing_ids))`: de-duplicates (an id listed twice must not advance its counter twice) and makes the eligible-list pre-order deterministic even if a caller's `order_key` has ties. Does not change *which* ids are visited.
9. **Determinism self-check by equality, not by hash.** §4A.4 says "compare a hash"; I compare the two builds' competing set, excluded set and key dict directly (a strictly stronger test); mismatch ⇒ `RuntimeError`.
10. **Capture is `id → (pre, post, conducting)`,** not `(pre, post)`: `conducting_links_removed` (§4A.6) needs `weight ≥ weight_threshold` and removed synapses no longer exist afterwards.
11. **Floors.** The pre-call floor assertion (§4A.4) is trivially true by construction (G ⊂ excluded); `floors_ok` in the record is a real **post-call recount** of each want's non-F links vs `min(K, non-F degree)` before.
12. **Self-loops.** Per-want competing count `c_w` counts a link once per distinct endpoint (a set), i.e. "w's number of competing links"; the test reference appends a self-loop twice. The fixtures have no self-loops, so the tests cannot tell the difference.
13. **A want↔want removal is tallied under BOTH wants** in `by_want`, so the per-want totals can exceed `removed` (documented in the docstring).
14. **Last-link partner scan** looks only at the endpoints of competing synapses instead of every node; equivalent (a node with no competing synapse cannot have `inc ⊆ competing`, `inc` non-empty).

## 7. What I did NOT verify (stated plainly)
- **No real graph.** Nothing ran against a real checkpoint, the staged VPS bundle, or any copy — that is **PG-1, a separate pre-merge gate, not run**. Behaviour at ~107k competing ids / 138k synapses on the native Rust `SynapseStore` is **unmeasured** (time, memory, `_step_lock` hold time, the per-id `self.synapses[sid]` cost versus `items()`); the tests use Python-sized synthetic graphs (≈ 500 synapses).
- The store wrapper in `test_K_competing_mode_visits_only_competing_ids` is a Python proxy; I did not instrument the native store's `__getitem__`.
- I did not run the full suite (P373), so "no new failures anywhere" is **not** claimed — only the four files above, compared to base.
- Not run / not mine: the daemon slice (D tests), #824, #825, the S4 Tonic check, the dry run, any arming.
- `ast.parse` and the runs above are the only static checks; I did not run a linter or type-checker.
- Byte-exactness of `Graph.checkpoint()` was shown equal on the synthetic graphs with `PYTHONHASHSEED=0` pinned in both processes; on the real graphs it remains PG-1's job (plan §2.6/L2).
- I did not open Exec Packets 440/441 as primaries; quoted as supplied in the dispatch and the go record.
- The arming-time risks the plan lists (Door B liveness, guardian behaviour, #824/#825) are untouched by this build.

## 8. Findings outside this task (for the punch list / Josh)
1. **The repo's Syl's-Law PreToolUse hook does not fire on worktree paths** (already plan-005 §10 [R5e·LE23-D6]) — confirmed first-hand here: editing `neuro_foundation.py` in a worktree produced no prompt. Only Josh can amend it.
2. `tests/test_integration.py::TestStructuralPlasticityPruning::test_speculative_synapses_pruned` is **red on clean base** (§5.4) — pre-existing; whether it is one of the #761 72 is unchecked.
3. `build-001` lives under `handoffs/z12-want-hub-build/returns/` while the dispatch path for this file is one level up — the lane should pick one convention.

## 9. State left behind
- Engine branch `cc-laptop-want-hub-engine-20260930` @ `8e578532` pushed; worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930` clean.
- **Integration worktree** `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930`: detached at the tests head `95154e33` with the engine `neuro_foundation.py` **staged, never committed, never pushed** (scratch, as ordered). Left in place so the pair can re-run; safe to `git worktree remove --force`.
- Scratch outputs only under `/tmp/z12-wh-scratch/` (driver JSONs, checkpoints, the two diagnostic scripts, the engine diff); nothing written to any live or checkpoint path.
- The tests branch gets this file only; **no test file was changed.**
- Reproduce: `git -C /home/josh/NeuroGraph worktree add --detach <dir> 95154e3 && cd <dir> && git checkout 8e5785322f910aefaf781fa3525bea2831fedd31 -- neuro_foundation.py && env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -m pytest tests/test_want_hub_competition.py -p no:cacheprovider -v`.

## 10. Next (not mine)
1. Z12 rules on the §5.3 test fix (move one line) — the tests branch then goes 27/27, or the pair rules otherwise.
2. Fresh cross-family + law-enforcer delta pair on the engine diff (`git diff e4ebf982..8e578532`); §6 is the list of items they should see first.
3. PG-1 (real-graph golden, read-only copies, `MemAvailable ≥ ~8 GB`, one load at a time) — separate pre-merge gate.
4. Any merge reaching Syl's process is a NEW protected-file event for her (Exec Packet 441; the go-record FLAG).
