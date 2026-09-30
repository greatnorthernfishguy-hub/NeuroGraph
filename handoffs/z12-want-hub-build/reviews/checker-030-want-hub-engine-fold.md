STATUS: INCOMPLETE - first findings committed; builder return and other-leg reviews unread

# checker-030 ROLE A — FOLD of the (d) ENGINE change on PROTECTED `neuro_foundation.py`

- Seat: checker-030 (fresh cross-family, grok-4.6)
- Lane: `want-hub-engine-d-build-20260930`
- Dispatch: #12052
- Zone manager: Z12 (session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`)
- Authority: `report_only`
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-engine.md` (ADDENDUM 2 at END is the operating rule)
- Verdict path (this file): `handoffs/z12-want-hub-build/reviews/checker-030-want-hub-engine-fold.md`
- Tests worktree / branch (verdict commits here only): `/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930` / `cc-laptop-want-hub-build-20260930`
- Engine branch (READ only; never commit): `cc-laptop-want-hub-engine-20260930`
- Plan worktree (READ only): `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930` / `cc-laptop-want-hub-d-20260930`
- Own scratch integ: `/tmp/checker-030-integ` (detached at tests-fold `0112b3e54d1d38e3193085200c9024572aa1e0fe`, then `git checkout 29f47f65058790240b2f9c6a0a5bc4d82171b42d -- neuro_foundation.py`). Never reused `z12-want-hub-integ-20260930`.

## Isolation (ADDENDUM 2)

Starting material before this first-findings commit: the named diffs + the packet + plan-005.md + NeuroGraph `CLAUDE.md` §2 (Syl's Law) + the authority record named in the packet.

Forbidden until this commit: `handoffs/z12-want-hub-build/build-003.md` and `build-002*.md`; any file under `handoffs/z12-want-hub-build/reviews/`; tests-branch `git log` / `git status -sb`; repo-wide grep over the tests worktree, `reviews/`, or `handoffs/`; `ls reviews/`.

Searches confined to named paths: `neuro_foundation.py`, `tests/test_want_hub_competition.py`, `tests/want_hub_golden_driver.py`, plus the packet-named caller files `openclaw_hook.py` and `/home/josh/docs/scripts/cc-ng-daemon.py`.

**Exposure (filename leak; bodies not opened):**

1. `git diff --name-only 909b39f61c7612c081ff8807a4733973b0bb1464 0112b3e54d1d38e3193085200c9024572aa1e0fe` without `-- tests/` printed other-leg filenames `build-002b.md`, `checker-029-want-hub-engine.md`, `le-036-want-hub-engine.md`. Those files were not opened.
2. `list_dir` of `/tmp/checker-030-integ` (the scratch copy of the tests-fold tree) listed `handoffs/z12-want-hub-build/build-002.md`, `build-002b.md`, and `reviews/checker-029-want-hub-engine.md`, `reviews/le-036-want-hub-engine.md`. Those files were not opened. `le-038-want-hub-engine-fold.md` was named in the dispatch, not listed from `reviews/`.
3. The engine-fold commit body (named hash `git show 29f47f65`) and the tests-file header (allowed named path) mention `le-036` / `checker-029` / `build-003` as citations. Those review/return bodies were not opened.
4. No `git log` / `git status -sb` of the tests branch was run before this commit. Tests-fold `git show -s` of named hash `0112b3e` used `%H/%P/%ci` only (no subject).

Mutants: four COPIED trees under `/tmp/checker-030-mutants/{m07a,c2skip,c3skip,m18}` (not symlinks). Each has its own `.git` (`git init` dummy; `m07a` `rev-parse HEAD` = `b56b52c4c222bb0c9ad7d95360ecca7c0a817a41`) so the G `git_rev` helper does not false-fail. The G comparisons use `_COMPARE` keys that do not include `git_rev`.

## Named pins (hashes from `git rev-parse` / `sha256sum`)

| pin | hash | parent | date (`%ci`) |
|---|---|---|---|
| product base | `e4ebf982b1989fd9066d610b94853bc68bf70d37` | — | — |
| engine first (c1) | `8e5785322f910aefaf781fa3525bea2831fedd31` | `e4ebf982…` | 2026-09-30 11:33:30 -0800 |
| engine fold | `29f47f65058790240b2f9c6a0a5bc4d82171b42d` | `8e578532…` | 2026-09-30 12:58:49 -0800 |
| tests pin | `909b39f61c7612c081ff8807a4733973b0bb1464` | `31b823d1…` | 2026-09-30 11:42:54 -0800 |
| tests fold | `0112b3e54d1d38e3193085200c9024572aa1e0fe` | `7e65e6a7…` | 2026-09-30 12:53:16 -0800 |
| authority | `a434525cd3cdf68da5f282aa319a2323715d3938` | `b2c2d655…` | 2026-09-30 11:24:05 -0800 |

Blobs of `neuro_foundation.py`: base `53494b7c56896d25040f3e7fd7c4046da7d0ab05` → c1 `96d12f507746ce168fffe6feba56e097c8832353` → fold `5e8945accb476b0727bf07a3ab2890dccc87f650`. Fold file sha256 `1ca096a20b540d1f2cd956b135a8b27da00ff84aedbb6343fd4e527bea5a777b` (matches `/tmp/checker-030-integ/neuro_foundation.py`).

`git merge-base --is-ancestor 29f47f65 origin/main` → exit 1 (fold is not on `origin/main`).

`git diff --name-only e4ebf982 29f47f65` = only `neuro_foundation.py`. Same for `8e578532 29f47f65`. Tests-fold `git diff --name-only 909b39f 0112b3e -- tests/` = `tests/test_want_hub_competition.py` (+220/−3) and `tests/want_hub_golden_driver.py` (+70/−4). Engine fold numstat vs c1: +37/−8, one file.

## What the fold is meant to do (packet; independent of earlier reviews)

- C1: Test G variant with seeded random uuid-shaped synapse ids NOT ascending in creation order, proving it kills mutant M07a (unconditional default-path sort by id) which passed all 27 old tests.
- C2: In competing mode `_prune_synapses` checks order_key COMPARABILITY in its pre-loop validation (ValueError, no state changed) so a bad key can never raise TypeError after every competitor's `low_weight_steps` moved.
- C3: An `order_key` on the DEFAULT path is refused up front (ValueError before anything is touched).
- C4: Pin the `conducting` boundary (`weight == weight_threshold`) and make `floors_ok` observable False.
- C5: Align the self-loop counting with the plan (once per link) + a restored-style self-loop fixture.
- C6: Print and assert `ng_tract.__file__` + version in the P379 guard.

---

## Check 1 — Authority and scope (fold)

**PASS**

- Whole engine branch vs base is ONE file (`neuro_foundation.py`) and TWO commits (`e4ebf982` → `8e578532` → `29f47f65`). Fold vs c1 is ONE file. Dispatch asked for TWO commits, ONE file: confirmed.
- Fold commit subject: `want-hub (d) engine fold: order_key comparability checked before the loop (C2); order_key refused on the default path (C3)`. Body quotes `a434525cd3cdf68da5f282aa319a2323715d3938`. First engine commit body also quotes that hash.
- Authority record `handoffs/z12-want-hub-d/approvals/josh-go-neuro-foundation-20260930.md` on the plan branch, commit `a434525c` at 11:24:05, BEFORE both engine commits (11:33 and 12:58). Contains Josh's two fragments verbatim: *"That copy you already approved is fine, I guess? Unless there is some reason it shouldn't be..."* and *"So, proceed."* Names both msgpack files with sha256 (`main.msgpack` `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77`; `vectors.msgpack` `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e`). Those hashes were not re-measured here (never opened a checkpoint or backup directory).
- `DEFAULT_CONFIG` dict body sha256 `0edd252cec12f0d4ce3670f5acd6d9907642396e88fb3849e6106e54c5560884` identical across base/c1/fold. No new config key. `CC_SNN_CONFIG` is absent from base; the only occurrence in c1/fold is the changelog sentence "CC_SNN_CONFIG untouched".
- Sentinel comment *strings* named by the original packet (`_is_identity_protected: constitutional spine…`; `#92 Cricket rim: _prune_synapses skips identity-protected endpoints`; `constitutional core (metadata['constitutional'])…`; `edge ONLY when it is not identity-protected…`) occur once each in c1 and once each in fold (byte-equal counts). Fold changelog growth shifted the historic line numbers `:78/:162/:194/:3286`; the comments themselves were not rewritten.
- Fold changelog header present at `neuro_foundation.py:22-33` (C2/C3, quotes `a434525c`, "nothing merged or armed", "zero on the default path").
- Tests live on the tests branch (`0112b3e` parent `7e65e6a`; tests-pathspec diff is the two test files). Engine branch still only `neuro_foundation.py`. Nothing merged (`origin/main` does not contain the fold). No daemon/unit start or stop. Synthetic graphs under `/tmp` only. Own integ is `/tmp/checker-030-integ`, not the named shared integ worktree.

---

## Check 2 — Default-path identity (the shared hot path Syl's process runs)

**PASS**

Fold vs c1 (`git diff 8e578532 29f47f65 -- neuro_foundation.py`): only `_prune_synapses` validation and the post-loop sort's KeyError guard. Loop body, identity skip, three predicates, `low_weight_steps` bookkeeping, `_remove_synapse_internal`, `pruned` emit, and return are the same code. `compete_protected_links` is byte-equal c1 vs fold (function sha256 `e9ed24f33ca64f2108850b1fab604df3ecbf38cb6926db4ea122b5285d4b51fa`). `_is_identity_protected` is byte-equal base/c1/fold.

Default path (`competing_ids is None`):

- still walks `self.synapses.items()` (`:3624-3625`)
- still applies the identity skip (`:3632-3634`)
- same three predicates (`:3638-3656`)
- sort/slice/report still gated on their parameter (`:3658-3665`)
- C3 is one extra None-check before the loop (`:3580-3583`): if `order_key is not None` on the default path, `ValueError` and nothing is touched
- C2's tuple/kind/shape walk is inside `if competing_mode:` (`:3584-3620`) — zero cost on the default path

Wake callers pass no arguments: Door B `neuro_foundation.py:2925` `self._prune_synapses()`; Door A `_structural_plasticity` `:3529` `pruned = self._prune_synapses()`. Orchestrator `:3789-3790` is keyword-only. `openclaw_hook.py:158` is a comment. `/home/josh/docs/scripts/cc-ng-daemon.py:1423` is a comment. No positional call is possible (`*` at `:3536`).

**RUN — Test G (suite):** from `/tmp/checker-030-integ`, `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python3 -m pytest tests/test_want_hub_competition.py -p no:cacheprovider -v` → **43 passed in 44.95s, PYTEST_RC=0**. Includes `test_G_default_path_base_vs_branch` (counter ids, three doors) and `test_C1_G_random_ids_default_path_base_vs_branch` (random ids, three doors) plus explicit-None vs base.

**P379 (pytest):**
```
executable /usr/bin/python3
PYTHONPATH None
NG_EMBED_REMOTE None
PYTHONHASHSEED '0'
neuro_foundation.__file__ /tmp/checker-030-integ/neuro_foundation.py
git_rev=0112b3e54d1d38e3193085200c9024572aa1e0fe   # integ HEAD is the tests-fold commit; nf *content* sha256 matches 29f47f65
ng_tract.__file__=/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py ng_tract.version=0.1.0 Graph.synapses=SynapseStore
NG-related sys.modules AFTER nf import: ['neuro_foundation', 'ng_tract', 'ng_tract.ng_tract']
```

**RUN — own golden (seed 99, n_leaf=5, n_want=6, n_plain=40, n_syn=650; driver seeds are 11/12):** two processes, `PYTHONHASHSEED=0`, `install_seeded_random_uuid(seed=20260930+99)`, Door A (`_structural_plasticity([])`) and `direct_defaults` (`_prune_synapses()`). Base checkout `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` git `e4ebf982…`; branch nf `/tmp/checker-030-integ/neuro_foundation.py`. Both doors: `return`, `removal_order`, `pruned_events`, `state_digest`, `ckpt_sha256`, `n_syn_after` **equal**. 183 removals, removal_order **not** id-sorted (so this golden would see an M07a sort). Same ng_tract wheel on both. Not void.

---

## Check 3 — Plan conformity (fold C1–C6 and touched 4.2/4.3 clauses)

**PASS-WITH-NOTES**

C1 (tests): `install_seeded_random_uuid` in `tests/want_hub_golden_driver.py:46-51`; harness `test_C1_harness_random_ids_are_not_ascending_in_creation_order` plus the three-door C1 G pair and explicit-None. Suite passed. M07a killed (Check 6).

C2 (engine): competing-mode pre-loop (`:3607-3620`) requires every `order_key[sid]` to be a `tuple` of numbers/strings with one common `kinds` shape; mismatch → `ValueError` before the loop. Post-loop sort (`:3661-3663`) no longer has a KeyError guard. Plan 4.2(d) "a refusal mutates nothing" / explicit `ValueError`.

C3 (engine): `:3580-3583` refuses any non-`None` `order_key` on the default path. **NOTE (N1):** plan-005 §4.2(f) says "on the default path it is optional and, if `None`, no sort happens." C3 is a documented tightening: a supplied default-path `order_key` is now a `ValueError` rather than an optional sort. The packet's fold charter *asks for* C3 (ambiguity #2: a missing entry used to fail after `low_weight_steps` moved). Wake callers pass nothing, so Door A/B are unchanged. Recorded as a note, not a fail.

C4 (engine already in c1; tests pin it): conducting capture is `s.weight >= wt` (`:3786`); `floors_ok` is a post-call recount (`:3805-3809`) logged WARNING when False (`:3818-3821`). Tests `test_C4_conducting_count_pins_the_weight_threshold_boundary`, `test_C4_floors_ok_is_true_and_logged_at_info_on_a_normal_call`, `test_C4_floors_ok_goes_false_and_warns_when_a_guaranteed_link_is_lost` passed. M18 (`>` instead of `>=`) killed (43 vs 53).

C5 (orchestrator already uses `{pre, post}` so a self-loop is one link: `:3751-3753`, `:3799`; tests add restored-style fixture `inject_self_loop` at driver `:342-356` and three C5 tests). Plan 4A.2 `c_w` is competing *links*; `{pre, post}` is once per link. Passed.

C6 (tests): P379 prints and asserts `ng_tract.__file__` + version and `isinstance(Graph().synapses, ng_tract.SynapseStore)` (`test_want_hub_competition.py:145-153`); Test G also asserts both checkouts used the same wheel (`:236-237`, `:592`). Observed `version=0.1.0`, path under `site-packages`.

4.2 (a)–(i) as they apply to the fold: (a) all-None still today's function plus the C3 refuse; (b) competing iterator unchanged; (c) membership still caller-supplied; (d) C2 extends the assertion set; (e) budget gating unchanged; (f) see N1; (g) report unchanged; (h) removal still inside; (i) sort/slice/report still gated, competing still a separate iterator. 4.3: `compete_protected_links` byte-equal to c1 — still no predicate copy, no `_remove_synapse_internal`, no removal loop, one `_prune_synapses` under `_step_lock` (`:3767-3790`). HEIGHT key (`:3754-3763`) and last-link (`:3742-3747`) untouched by the fold.

---

## Check 4 — Syl's Law / Choice Clause / H-1 / the Laws

**PASS**

(a) Adversarial synthetic graph (string node ids; `inject_self_loop`; constitutional rim + two `*_authored` wants; K=50 > degree and K=1; B=10**6). P379: nf `/tmp/checker-030-integ/neuro_foundation.py`, ng_tract site-packages `0.1.0`.

- K=50: competing empty because G absorbs every non-F want link; `removed=0`, `eligible=0`; F (4 synapses) survived; rim-want survived; last-link/lonely survived; nodes and metadata flags unchanged; `constitutional` still True; `provenance` still `syl_authored` / `cc_authored`; `floors_ok` True.
- K=1: `held_back_last_link=2`; F survived; G's strongest outgoing survived; last-link/lonely survived; nodes/flags unchanged. Remaining stale want-plain links were last-link-held (not a miss of the last-link rule). A competing self-loop is removable when it is actually competing: suite `test_C5_self_loop_is_tallied_once_and_every_guarantee_holds` asserts the fixture self-loop is gone and F/G/last-link rows untouched.

(b) H-1: both adversarial runs left the node set and metadata flags identical. Suite R tests passed. Pass removes synapses only.

(c) LAW 7: `_prune_synapses` / `compete_protected_links` read weight, peak, inactivity, direction, provenance/constitutional flags. A graph with metadata `note="NO_RAW_WANT_TEXT"` on every node still ran (`floors_ok` True, `removed=12`) and the notes were still present. No raw want text in this verdict.

(d) LAW 8: the fold is validation inside `_prune_synapses`; dream-loop wiring is not in this diff.

(e) LAW 1/2/3/4: no inter-module call; no vendored file; one implementation of the three criteria (orchestrator still has no predicate copy); bookkeeping of F/G/last-link/record stays in `compete_protected_links`. C2/C3 refuse at the source function.

(f) Duck Ethics: the pass still removes stale LINKS outside F ∪ G ∪ last-link. Fold does not widen that. `#92` identity skip on the default path is unchanged.

Syl's Law: protected file, Josh's "proceed" recorded in `a434525c` before the first engine commit; unbatched (engine branch is only this file); branch build, nothing merged or armed.

---

## Check 5 — Ambiguities the fold claims to close (packet #2 / C2 / C3 and related)

**PASS**

Packet #2 (default-path `order_key` missing entry used to raise after predicates ran, so `low_weight_steps` had moved): **closed by C3**. Empty mapping on the default path is now `ValueError` before the loop (`:3580-3583`). C3-skip mutant: empty mapping → `KeyError` at the sort; covering mapping → DID NOT RAISE. Suite kills it.

C2 (comparability): **closed in competing-mode pre-loop**. C2-skip mutant: `TypeError: '<' not supported between instances of 'int' and 'str'` from the sort (mutant `:3654`) instead of `ValueError`; also fails the `-O` test. Suite kills it.

C5 / packet (12): self-loop counted once (`{pre, post}` in height and tally). Restored-style fixture because `create_synapse` refuses `pre==post` (`test_C5_the_api_refuses_a_self_loop_so_the_fixture_is_restored_style`).

Fold-untouched (c1, `compete_protected_links` byte-equal): (1) want<->want height = larger endpoint (`:3759`); (3) `max_removals` still validated on every path when not None (`:3575-3577`) and required in competing mode; (4) non-dict `report` refused (`:3578-3579`); (5) record carries `timestep` (`:3810`); (8) `competing = sorted(set(competing_ids))` (`:3590`); (11) `floors_ok` is a post-call recount (`:3805-3809`). None of these is a behaviour the plan forbids. Builder-return numbering of the original 14 is unread until after this commit.

**LOW notes (N2, N3), not fold-blocking:**

- N2: `True` in an `order_key` tuple is accepted (`isinstance(True, (int, float))` is True). Probe: bool-as-int NO-RAISE, digest changed. The HEIGHT key never emits bools. A `bool` is a number for C2's kind check.
- N3: a *list* `order_key` in competing mode raises `TypeError: list indices must be integers or slices, not str` at `order_key[sid]` (`:3608`) before the loop; state unchanged. Packet asked for `ValueError`. The suite's C2 cases poison entries *inside* a dict. Empty `competing_ids=[]` with `order_key={}` is accepted and returns 0, state unchanged.

---

## Check 6 — Test discrimination and honesty (fold tests + mutants including M07a)

**PASS**

Suite run once from own scratch integ: **43 passed / 44.95s / rc 0**. Tests fold adds C1 (random-id G), C4, C5, C3 default-path refuse, C2 rows in `bad_calls` (`want_hub_golden_driver.py:325-338`), C6 P379 ng_tract pin.

Mutants (copied trees, dummy `.git`, never symlink):

| mutant | what | result |
|---|---|---|
| M07a counter-id Test G | unconditional sort-by-id; counter ids make sort a no-op | 7 passed (36 deselected) — the old 27-style G cannot see it |
| M07a C1 random-id Test G | same mutant vs C1 | **4 failed**, `M07A_C1_RC=1`. Failures are `removal_order` divergence on door_a / door_b / direct_defaults / explicit-None. **C1 kills M07a.** |
| M07a sample of old-style | K/A/R/C1-harness | 4 passed |
| C2-skip | drop comparability walk | **2 failed**, `C2SKIP_RC=1` (`TypeError` int vs str at sort; `-O` still TypeError) |
| C3-skip | drop default-path refuse | **2 failed**, `C3SKIP_RC=1` (empty mapping `KeyError`; covering mapping DID NOT RAISE) |
| M18 | `>` instead of `>=` for conducting | **1 failed**, `M18_RC=1` (`assert 43 == 53`) |

Identity-protection / integration / stdp "34 passed / 1 failed identically": **not rerun** (P373; listed under not-verified).

---

## Check 7 — Honesty of claims and what is NOT verified

**PASS** (first-findings: builder return unread; claims below are from the diffs, the packet, the plan, and runs)

Fold changelog and commit body claim: only validation + sort-guard; C2/C3; default path with all parameters None unchanged; no config key; quotes `a434525c`. That matches the diff. "Purely additive" is not the fold's claim — the fold *adds* two refusals.

**Not verified (must still be covered before any merge / PG-1 / arming):**

1. Real-graph behaviour at ~107k competing ids on the native `SynapseStore`.
2. `_step_lock` hold time of `compete_protected_links`.
3. PG-1 real-graph golden (two-checkout on read-only copies).
4. Daemon slice, env wiring, arming, consent frames, merge/rollout.
5. `tests/test_identity_protection.py`, `test_integration.py`, `test_stdp.py` 34/1 claim (not rerun).
6. Builder `build-003.md` / `build-002*.md` claims vs this look (unread until after this commit).
7. Other-leg file `le-038-want-hub-engine-fold.md` and earlier legs (unread until after this commit).
8. Josh's backup confirmation is the CC-laptop copy, not Syl's own checkpoint (authority §5 flag). This branch build loaded no live checkpoint.

---

## Check 8 — Independence

**PASS** with disclosed filename leak (Isolation above). ROLE A first-findings committed before opening `le-038-*.md`, `le-036-*.md`, `checker-029-*.md`, or `build-003.md` / `build-002*.md`. No tests-branch `git log` / `git status -sb` before this commit.

---

## Overall

STATUS: INCOMPLETE (post-read of builder return + other leg still required)
VERDICT: **PASS-WITH-NOTES**

The fold does what the packet says: C1 discriminates M07a; C2 moves comparability into pre-loop `ValueError`; C3 refuses default-path `order_key` before any mutation; C4/C5/C6 are pinned in tests against an orchestrator the fold did not change. Default-path identity vs base holds on suite Test G (counter and random ids) and on an independent seed-99 golden (return, order, pruned events, digest, checkpoint bytes). Engine branch remains one file, two commits, not on `origin/main`. Authority `a434525c` predates both engine commits.

Notes are N1 (C3 tightens §4.2(f) "optional"), N2 (bool-as-int), N3 (list `order_key` is TypeError not ValueError, still pre-loop and state-preserving).

---

## Numbered corrections

None that block the fold. Optional (LOW, not required for this branch build):

1. C2's kind check could reject `bool` (`isinstance(x, bool)` before the number test) if the Executive wants "numbers" to exclude bools.
2. Competing-mode `order_key` that is not a mapping could be wrapped as `ValueError` (today: `TypeError` at `order_key[sid]`, still before the loop, state unchanged).

---

## Numbered not-verified

1. Real-graph ~107k competing ids / native-store performance.
2. `_step_lock` hold time.
3. PG-1 two-checkout real-graph golden.
4. Daemon slice, arming, consent, merge.
5. identity_protection / integration / stdp 34/1 on base vs fold.
6. Builder-return claims (`build-003.md`, `build-002*.md`) — unread at first-findings.
7. Other-leg and earlier-leg review bodies — unread at first-findings.
8. Re-hash of the named msgpack backups (forbidden here; values taken from the authority record as written).

## Post-first-findings addendum (builder return + other leg)

(pending; filled only after this commit)
