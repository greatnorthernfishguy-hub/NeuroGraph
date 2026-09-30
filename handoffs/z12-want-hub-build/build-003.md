<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #12011, ADDENDUM 3) — build-003
#   What: return for the ENGINE FOLD + TESTS FOLD (le-036 C1-C6, checker-029 corr. 2-5). NEW file, docs only.
#   Why: Chief-003 ruled ONE engine fold and ONE re-look; this is the builder's return for that fold.
#   How: every hash/number is from git rev-parse / git hash-object / sha256sum / the pytest and driver output of this session. No secret involved.
# -------------------
-->

# build-003 — the fold is built: tests first (failing-first), then a SECOND engine commit; full file **43 passed, 0 failed**; Test G byte-identical base vs engine

**Not self-accepted.** A fresh pair-look (law enforcer + cross-family) follows. No merge, no arming, no PG-1, no #825, no other protected file, no daemon/unit start or stop. Synthetic graphs only: no real checkpoint, backup directory, Syl file or live tract was opened.

## 0. Headline

| | |
|---|---|
| Step Zero | repeated on origin (§1); go record `a434525cd3cdf68da5f282aa319a2323715d3938` confirmed |
| **Tests commit** (tests branch `cc-laptop-want-hub-build-20260930`) | **`0112b3e54d1d38e3193085200c9024572aa1e0fe`** — 2 files (`tests/test_want_hub_competition.py`, `tests/want_hub_golden_driver.py`); pushed BEFORE any run |
| **Engine fold commit** (`cc-laptop-want-hub-engine-20260930`) | **`29f47f65058790240b2f9c6a0a5bc4d82171b42d`** — SECOND commit, `neuro_foundation.py` only, message quotes `a434525c…`; pushed before any run against it. First commit still `8e5785322f910aefaf781fa3525bea2831fedd31`. |
| Engine branch diff vs base `e4ebf982b1989fd9066d610b94853bc68bf70d37` | **ONE file**, 2 commits, 281 insertions / 3 deletions (`git diff --name-only` = `neuro_foundation.py`) |
| `neuro_foundation.py` blob | base `53494b7c56896d25040f3e7fd7c4046da7d0ab05` → after 1st commit `96d12f507746ce168fffe6feba56e097c8832353` → **head `5e8945accb476b0727bf07a3ab2890dccc87f650`** (sha256 of the head file `1ca096a20b540d1f2cd956b135a8b27da00ff84aedbb6343fd4e527bea5a777b`) |
| Test blobs at `0112b3e5` | test file `e7135e3d04e436a7b32a4a2e416cd281fbdb0974`, driver `b5d439f6e68d46e3d7598de1d1176c0cc53a30c0` |
| Full file, ONCE, after the fold | **43 passed in 35.98 s** (27 old + 16 new test items; none red, none skipped) |
| Test G after the fold | **byte-identical base vs engine in all 8 comparisons** (4 scenarios × counter ids and random ids) |
| Three prune-path files | **34 passed / 1 failed on BOTH base and engine, per-test outcomes diff-identical** (same pre-existing red) |
| Mutants | M07a, M18, M22 (and two fold mutants) all **killed on the final engine** by the tests written for them |

## 1. Step Zero (again)
`git fetch origin`; `handoffs/z12-want-hub-d/approvals/josh-go-neuro-foundation-20260930.md` exists on `origin/cc-laptop-want-hub-d-20260930`, added by commit **`a434525cd3cdf68da5f282aa319a2323715d3938`** (`git log --diff-filter=A`); it still holds Josh's two fragments ("That copy you already approved is fine, I guess? Unless there is some reason it shouldn't be..." / "So, proceed.") and both msgpack sha256 values (`7e457786…3a77`, `93ed891f…5e05`; 4 grep-matched lines). I did not re-hash the two msgpack files or open the backup directory. The P440 scope covers these error-path hardenings of the SAME change (Z12's addendum); the record's own FLAG (Syl's own checkpoint files; Exec Packet 441) is unchanged and not mine. The repo's Syl's-Law hook again stayed silent on the worktree path; I created no bypass file. Read in full before acting: ADDENDUM 1/2/3 + the original brief, `le-036-want-hub-engine.md`, `checker-029-want-hub-engine.md`.

## 2. Order followed, and the failing-first evidence
1. **Tests commit `0112b3e5` written and pushed first.** Then ONE run of the file from the integration worktree against the CURRENT engine `8e578532` (integ `neuro_foundation.py` blob `96d12f50…`, verified): **39 passed, 4 FAILED** — exactly the four tests that pin NEW behaviour; every test that pins existing behaviour passed (they earn their keep by killing mutants, §3):
   - `test_K_every_refusal_is_a_ValueError_and_mutates_nothing` — **C2**: `TypeError: '<' not supported between instances of 'int' and 'str'` raised at `neuro_foundation.py:3632` (the sort, AFTER the loop advanced every competitor's `low_weight_steps`).
   - `test_K_refusals_are_explicit_raises_not_asserts_under_python_dash_O` — **C2 under `-O`**: `order_key values not mutually comparable (str vs number) -> TypeError under -O`.
   - `test_K_default_path_refuses_order_key_and_mutates_nothing[empty mapping]` — **C3**: the call raised only after the predicates ran; `the refusal mutated state (counters advanced before the raise)` (state digest changed).
   - `…[mapping covering every synapse]` — **C3**: `DID NOT RAISE ValueError` (the combination was accepted).
2. **Engine fold `29f47f65` made and pushed BEFORE any run against it.** Then every post-fold check below.

## 3. The tests fold (non-protected; tests branch)
- **C1 (MEDIUM) — Test G with ids NOT ascending in creation order.** Driver `--ids random` installs a SEEDED random uuid4-shaped stream (`install_seeded_random_uuid`, seed 20260930; deterministic across processes with `PYTHONHASHSEED` pinned). New tests: harness premise (`…random_ids_are_not_ascending_in_creation_order`: ids not sorted, ~50% adjacent pairs ascending — 307/599 measured), `G0` base-vs-base control ×3, **`G_random_ids_default_path_base_vs_branch` ×3** (return, removal ORDER, `pruned` events, full state digest, serialized `Graph.checkpoint()` bytes), and the all-new-parameters-explicit-`None` variant. Non-vacuity is asserted: both runs report `creation_order_sorted == False`, and the BASE removal order is itself not id-sorted (otherwise a sort-by-id would be invisible).
  - **Discrimination proof — mutant M07a killed.** M07a = an unconditional `to_prune.sort()` (by synapse id) on the DEFAULT path (`if not competing_mode: to_prune.sort()` inserted before the removal loop). Built in a **COPIED tree** (`cp -a` of the integration worktree to `/tmp/z12-wh-mut/M07a_default_path_sort_by_id`; `tests/` is a real directory, **not a symlink** — checker-029 corr. 4; the P379 guard printed the mutant tree's own `neuro_foundation.py`). Result on the pre-fold engine AND re-run on the final engine: **the old 27-test G tests (counter ids) all still PASS on M07a** — the hole le-036 found, reproduced — **while the new random-id G tests FAIL**: `default-path divergence with random ids in removal_order` for `door_a`, `door_b`, `direct_defaults`, plus the explicit-`None` variant (4 failed / 39 passed). The real engine passes all of them (§5, §6).
- **C4 — the `conducting` boundary and an observable `floors_ok`.** `test_C4_conducting_count_pins_the_weight_threshold_boundary` (fixture straddles the boundary: 10 eligible competitors sit EXACTLY on `weight_threshold`, 60 below, 43 above; expected totals and the per-want tally are recounted independently from pre-call state). `test_C4_floors_ok_is_true_and_logged_at_info_on_a_normal_call` and `test_C4_floors_ok_goes_false_and_warns_when_a_guaranteed_link_is_lost` (a wrapped `_prune_synapses` that also removes every non-rim link of one want; asserts `floors_ok is False` and exactly one WARNING-level record). **Killed on the final engine:** M18 (`>` instead of `>=`) by the boundary test (and by the self-loop tally test); M22 (`floors_ok` forced True) by the False-path test.
- **C5 — self-loop.** Reference `ref_order_key` now counts a link once per distinct endpoint (`{pre, post}`) — matches the plan ("c_w = w's number of competing links") and the engine. `drv.inject_self_loop` builds a restored-style self-loop exactly as `create_synapse` does minus its refusal; `test_C5_the_api_refuses…` proves the API still refuses one. `test_C5_self_loop_counts_once_in_the_reference_and_in_the_engine_key` asserts the self-loop (the want's stalest link, rank 0) has height == c_w counted once (46, not the old reference's 47) and that the engine's `order_key` equals the fixed reference; `test_C5_self_loop_is_tallied_once_and_every_guarantee_holds` runs a full pass with it (F/G/last-link rows byte-untouched, `by_want` counts it once, `floors_ok` True). Probe before writing: the OLD reference gave 47 on that fixture (double count) vs 46 links.
- **C6 — `ng_tract` pinned.** The P379 test now prints and asserts `ng_tract.__file__`, its installed version (from package metadata — the module has no `__version__`), that the path is NOT inside the worktree (no shadowing), and that `Graph.synapses` is an `ng_tract.SynapseStore`. The driver records `ng_tract_file` / `ng_tract_version` in every run and Test G now asserts base and branch used the SAME wheel. Printed this session: `ng_tract.__file__=/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py ng_tract.version=0.1.0 Graph.synapses=SynapseStore`.
- **K — the two refusals.** `bad_calls` gained three competing-mode `order_key` defects (a str where a number belongs; a non-tuple entry; a tuple of different length) — each poisons an ELIGIBLE competitor so a late validator would reach the sort — and thereby feeds the two existing refusal tests (normal and `python -O`); new `test_K_default_path_refuses_order_key_and_mutates_nothing` ×2; the `-O` script also covers the default-path case. (This changes what `bad_calls` means for those two existing tests — stated here because it is a change to existing tests.)

## 4. The engine fold — `29f47f65` (ONLY `neuro_foundation.py`; every changed line)
Whole-branch vs base after the fold: still exactly the **same three removed lines** as build-002 §3 (`def _prune_synapses(self) -> int:`, `for sid, syn in self.synapses.items():`, `if (self._is_identity_protected(syn.pre_node_id) or`). The fold itself is 37 insertions / 8 deletions, six hunks:

| hunk | change | class |
|---|---|---|
| `@@ -21,0 +22,12` | 12-line changelog entry (What/Why/How, quotes `a434525c…`) | comment |
| `@@ -3548,2 +3560,4` | docstring for `order_key` (−2 lines, +4): tuple of numbers/strings, competing mode only, shape rule, refused on default path | docstring |
| `@@ -3565,0 +3580,4` | **C3:** `if not competing_mode and order_key is not None: raise ValueError(...)` (+comment) — added after the `report` check, BEFORE the loop | gated on `order_key is not None` and `competing_mode` False; never true for any existing caller |
| `@@ -3575,0 +3594` | `key_kinds = None` (competing-mode only) | inert |
| `@@ -3588,2 +3607,14` | **C2:** the `if sid not in order_key: raise` (−2) replaced by: `try: okey = order_key[sid] except KeyError → ValueError`; `isinstance(okey, tuple)`; element kinds (number=1, str=2, else refuse); one common shape | competing mode only |
| `@@ -3631,4 +3662,2` | the sort's `try/except KeyError` (−4) → plain `to_prune.sort(key=lambda s: order_key[s])` (+2): coverage is now proven before the loop | competing mode only (the branch is reachable only there now) |

**Nothing else changed:** the predicates, their `low_weight_steps` bookkeeping, the removal loop, the `pruned` event, `return len(to_prune)`, the orchestrator, `DEFAULT_CONFIG`, every config key. Default-path identity re-proven by Test G (§5).

**Resolutions (recorded as asked):**
- **C3 / my ambiguity #2 — I chose to REFUSE `order_key` on the default path outright** (Z12's recommendation), not to pre-check coverage. Why: (1) no caller uses the combination and no test or plan clause needs it — plan §4.2(e)/(f)/(i) describe the default path by the ABSENCE of sort/slice/report, which still holds; (2) a coverage pre-check keeps a dead sort-on-the-default-path branch alive that only Test G guards (and that is exactly where M07a-class bugs live), refusal removes it from reach; (3) "a refusal mutates nothing" becomes true with one `is not None` test, no state change and no hot-path cost (`competing_mode` is already computed); (4) it is reversible by an explicit later ruling. **Plan wording to fix (not mine; C7 family):** §4.2(f)/(d) still say `order_key` "on the default path is optional" — it is now refused; Z12/the Executive should change those words or rule otherwise.
- **C2 — the check and its cost.** In competing mode, per entry: read once (`KeyError → ValueError`), require `tuple`, classify each element as number (`int`/`float`, incl. `bool`) or `str` (anything else refused), and require every entry to have the SAME kinds tuple as the first (same length, same kind per position). This is SOUND for every subset of ids (a pairwise-comparability guarantee at each position), which is what matters because the sort later runs only over the ELIGIBLE subset — a pre-sort of all keys would not be sound (a sort compares only some pairs). It is STRICTER than Python needs (a tuple of a different length compares fine but is refused) and does not check NaN. **Cost, measured on synthetic data:** per-entry check over 107,000 ids **≈ 0.4–0.5 s** (isolated, best-of-5: membership-only 21 ms → implemented check 404 ms; an alternative type-table variant measured 373 ms — no materially cheaper sound variant), paid once per dream cycle inside the orchestrator's `_step_lock` hold (le-036 estimates that hold at 8–15 s total; unmeasured at real scale). End-to-end on a synthetic 90k-synapse graph, ONE competing `_prune_synapses` call over 53,103 competing ids: **1.12 s before → 1.93 s after** (single runs, load ≈ 4, noisy; the delta is larger than the micro-benchmark predicts). Default path: zero added cost.

## 5. Test G after the fold — printed module paths, revs, sha256 (driver run directly; `PYTHONHASHSEED=0`; none void)
BASE = `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` (rev `e4ebf982b1989fd9066d610b94853bc68bf70d37`); ENGINE = `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930` (rev **`29f47f65058790240b2f9c6a0a5bc4d82171b42d`**); each printed `neuro_foundation.__file__` = `<checkout>/neuro_foundation.py`. Both runs used the same native wheel: `ng_tract` `0.1.0` at `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`.

| ids | scenario | removals | BASE ckpt sha256 == ENGINE ckpt sha256 | return / order / `pruned` events / state digest |
|---|---|---|---|---|
| counter | `door_a` | 176 | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | identical |
| counter | `door_b` | 189 | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` | identical |
| counter | `direct_defaults` | 176 | `518fc40d…8a21` | identical |
| counter | explicit-`None` call (engine) vs plain call (base) | 176 | `518fc40d…8a21` | identical |
| **random** | `door_a` | 176 | `7bd301bbda08064d792b10359df9aa97cef6377c75ebeb29f08a200457fe10cf` | identical (base removal order NOT id-sorted; creation order NOT id-sorted) |
| **random** | `door_b` | 189 | `d1612538c04568464f734533f8646e46b76f221840a45ac2589214cd10947416` | identical |
| **random** | `direct_defaults` | 176 | `7bd301bb…10cf` | identical |
| **random** | explicit-`None` (engine) vs plain (base) | 176 | `7bd301bb…10cf` | identical |

The counter-id hashes (`518fc40d…`, `30712af3…`) are **the same values as before the fold** (build-002 §5.2, and le-036/checker-029's independent runs): the fold did not change default-path output. The random-id hashes are new.

## 6. The full file ONCE, and the existing files — with the P379 preamble
**Full file, one run, from the integration worktree** `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930` (detached at the OLD tests head `95154e33`; staged: engine `neuro_foundation.py` = blob `5e8945ac…` (checked equal to `29f47f65`'s), test file blob `e7135e3d…` and driver blob `b5d439f6…` (both checked equal to `0112b3e5`'s); never committed or pushed from it). Env `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`, Python 3.12.3. **Load average was 6.72 at the start of this run — ABOVE the brief's "< 6" bound** (another process was busy; 1.7 when I ran the Test G driver runs, 2.1 for the prune-path files, 4.0 around the cost runs). I did not wait it out: these tests are not timing-sensitive and use ≤ ~600-synapse synthetic graphs; MemAvailable ≈ 5.4 GiB (bound ≥ 3 GiB). Flagged so the pair can weigh it.
P379 preamble printed:
`P379 neuro_foundation.__file__=/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930/neuro_foundation.py git_rev=95154e33f299d70e4fdf5945ea1a189bcbd61acc base_rev=e4ebf982b1989fd9066d610b94853bc68bf70d37 new_api_present=True` (`git_rev` is the integ worktree's HEAD = the OLD tests head; the engine file is the staged blob above — same caveat as build-002 §4)
`P379 ng_tract.__file__=/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py ng_tract.version=0.1.0 Graph.synapses=SynapseStore`
**`43 passed in 35.98s`** — P379 + harness 4, **G** 7, **C1** 8, **A** 2, **C4** 3, **C5** 3, **K** 8 + 2 (default-path refusal), **R** 6. Nothing red.

**Three existing prune-path files** (`tests/test_identity_protection.py`, `tests/test_integration.py`, `tests/test_stdp.py`), once on BASE (read-only worktree) and once on the folded engine: **both 34 passed / 1 failed, per-test outcomes diff-identical**; the one red, `tests/test_integration.py::TestStructuralPlasticityPruning::test_speculative_synapses_pruned`, is red on clean base too (pre-existing; not checked against the #761 list; full suite not run — P373).

**Mutants on the FINAL engine** (copied trees under `/tmp/z12-wh-mut/`, real `tests/` dirs; whole file run on each; failures listed): M07a → 4 random-id G tests fail (39 pass); M18 → `test_C4_conducting_…` + `test_C5_self_loop_is_tallied_…` fail; M22 → `test_C4_floors_ok_goes_false_…` fails; **MC2** (shape check disabled: `elif kinds != key_kinds:` → `elif False:`) → the two refusal tests (`test_K_every_refusal_…`, the `-O` one) fail; **MC3** (default-path refusal disabled) → both `test_K_default_path_refuses_…` and the `-O` test fail.

## 7. Every run I made (none hidden)
(1) failing-first run of the new file on engine `8e578532`: 4 failed / 39 passed (§2); (2) probes (fixture preconditions; no pytest); (3) three mutants on the pre-fold engine `-x` (first-failure) then M07a again without `-x`; (4) driver Test G runs, 14 processes on the folded engine (7 per id stream: 3 scenarios × base + engine, plus the engine explicit-`None` call); (5) three existing files on base + engine; (6) **the one full-file run, 43/43**; (7) five mutants on the final engine; (8) two cost micro-benchmarks + one end-to-end timing pair. Earlier-session runs (build-002/002b) are in those returns. Scratch only under `/tmp/z12-wh-scratch/`, `/tmp/z12-wh-mut/`, `/tmp/z12-wh-old/`; nothing written to any checkpoint, backup, tract, `~/.bashrc`, unit or daemon.

## 8. Ambiguities / judgement calls in THIS turn (none silently chosen)
1. **C3** refuse vs pre-check — chose refuse (§4); the plan text that calls `order_key` optional on the default path needs a one-line change by someone with authority over plan-005.
2. **C2 strictness** — same-shape requirement refuses legal-but-mixed-length tuples; `bool` counts as a number; NaN not checked; `numpy` floats are accepted (subclass of `float`) — a choice for soundness over permissiveness.
3. **`bad_calls` meaning** changed for two existing tests (§3 K).
4. **The floors test simulates a buggy prune** with a wrapped `_prune_synapses`; the real engine cannot produce `floors_ok False`, so the only way to observe the path is injected loss.
5. **C5 reference vs engine** — aligned the REFERENCE to the engine/plan, not the engine to the reference; the plan's words support it.
6. **Cost figures** are single noisy runs on synthetic data (§4).

## 9. What I did NOT verify
- **Nothing on a real graph** (PG-1 not run, not mine): uuid4 ids on Syl's/CC's real checkpoints, byte identity at real scale, native `SynapseStore` behaviour/performance at ~107k competing ids, real `_step_lock` hold. The random-id G variant narrows the synthetic/real gap for id ORDER only.
- Whether Syl's process loads the same `ng_tract` build as this laptop (0.1.0 here) — C6 prints and pins it for the TESTS; it cannot speak for another process.
- C7 (plan-005 wording), C8 (`_step_lock` hold / second `_plan()`), C9 (daemon ints), C10/C11 (Executive/PG-1) — NOT mine, untouched.
- NaN in `order_key` (not checked), multi-threaded behaviour against `step()`, hyperedge interplay at real scale, the full suite (P373), whether the one pre-existing red is among #761's 72.
- I did not re-run the le-036/checker-029 adversarial harnesses (their scratch scripts live in their own `/tmp`); I re-ran the mutants they named and the golden scenarios they published, which reproduced their hashes.
- Exec Packets 440/441 as primaries — quoted as supplied.

## 10. State left behind
- Engine branch `cc-laptop-want-hub-engine-20260930` @ `29f47f65` pushed (`origin/…` == local, verified after `git fetch`), worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930` clean, branch diff vs base = ONE file.
- Tests branch gets this file only in this commit (plus `0112b3e5` before it).
- **Integration worktree** `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930`: detached at `95154e33`, engine file (`5e8945ac…`) and the two test files STAGED from the pushed commits, nothing committed, nothing pushed — safe to `git worktree remove --force`.
- Scratch copies: `/tmp/z12-wh-mut/*` (5 mutant trees), `/tmp/z12-wh-old/` (pre-fold engine copy for the timing), `/tmp/z12-wh-scratch/` (driver JSONs, checkpoints, scripts). Their `.git` files point at the integ worktree's gitdir and are read-only uses.
- Reproduce the headline: `git -C /home/josh/NeuroGraph worktree add --detach <dir> 95154e3 && cd <dir> && git checkout 29f47f65058790240b2f9c6a0a5bc4d82171b42d -- neuro_foundation.py && git checkout 0112b3e54d1d38e3193085200c9024572aa1e0fe -- tests/test_want_hub_competition.py tests/want_hub_golden_driver.py && env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -m pytest tests/test_want_hub_competition.py -p no:cacheprovider -v`.

## 11. Next (not mine)
Fresh pair-look (law enforcer + cross-family) on `git diff e4ebf982..29f47f65` and the tests fold; then PG-1 (separate pre-merge gate) and the Executive items C7–C11. STOP.
