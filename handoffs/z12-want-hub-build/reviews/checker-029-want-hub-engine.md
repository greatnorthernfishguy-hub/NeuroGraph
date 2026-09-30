# checker-029 ROLE A — want-hub (d) ENGINE delta review

STATUS: COMPLETE

Seat: checker-029 (cross-family, grok-4.6)
Lane: want-hub-engine-d-build-20260930
Dispatch: #11946
Zone manager: Z12 (session 52d39aba-db92-4bf2-b3b1-0e4c13f77d8c)
Authority: report_only
Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-engine.md`
Tests worktree: `/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930`
Tests branch: `cc-laptop-want-hub-build-20260930`
Scratch integ (own, never the shared integ): `/tmp/checker-029-integ` detached `15700944d1e105007ab2f0adaa6f68597200752e` + uncommitted engine blob `96d12f507746ce168fffe6feba56e097c8832353`

Overall: **PASS-WITH-NOTES**
Independence: own findings in this file are committed before opening `handoffs/z12-want-hub-build/reviews/le-036-want-hub-engine.md`. Accidental metadata exposure is disclosed in Check 8.

## Preamble (P379)

- python executable: `/usr/bin/python3` (3.12.3)
- PYTHONPATH at session start: `/home/josh/NeuroGraph:`
- NG_EMBED_REMOTE at session start: `hf`
- NG-related sys.modules at session start (no NG import): none
- MemAvailable at session start: 5777280 kB
- load average at session start: 3.07, 2.19, 2.49
- `~/.bashrc` sha256 (read-only, never written): `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`
- Analysis/tests: `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1` unless a targeted run records otherwise.
- Scratch-integ pytest: `neuro_foundation.__file__=/tmp/checker-029-integ/neuro_foundation.py` `git_rev=15700944d1e105007ab2f0adaa6f68597200752e` (tests HEAD; engine file is uncommitted checkout of `8e578532`, blob `96d12f50`) `new_api_present=True`. NG modules after import: `neuro_foundation`, `ng_tract`, `ng_tract.ng_tract`.

## Check 1 — Authority and scope

**PASS**

- Engine branch `cc-laptop-want-hub-engine-20260930` HEAD `8e5785322f910aefaf781fa3525bea2831fedd31` (2026-09-30T11:33:30-08:00). `git log e4ebf982..HEAD` is that one commit; `git diff --name-only e4ebf982 HEAD` is `neuro_foundation.py` only; `git diff --stat` = 252 insertions / 3 deletions.
- Blobs: base `53494b7c56896d25040f3e7fd7c4046da7d0ab05` → head `96d12f507746ce168fffe6feba56e097c8832353`. sha256 of the head file `d1641399c031bcb14fd93bef787acd7d11b8a834592ead4cdf42799e8d83123f`.
- Commit message quotes `a434525cd3cdf68da5f282aa319a2323715d3938`.
- Go record exists on the plan branch BEFORE the engine commit: `a434525c` at 2026-09-30T11:24:05-08:00, file `handoffs/z12-want-hub-d/approvals/josh-go-neuro-foundation-20260930.md`. Josh's two fragments verbatim: *"That copy you already approved is fine, I guess? Unless there is some reason it shouldn't be..."* and *"So, proceed."* Both msgpack sha256 values are in the record (`main.msgpack` `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77`, `vectors.msgpack` `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e`). I did not re-hash those files.
- Changelog header present at the top of `neuro_foundation.py`.
- `DEFAULT_CONFIG` assignment body sha256 identical on base and engine: `2f145de446ffa68e27bac012f00695ed628cf78b88c8cc280bdc8627d5074f55` (8436 bytes). `CC_SNN_CONFIG` is not assigned in this file; the engine diff mentions it only in changelog prose ("untouched"). No config key added.
- Comment lines at BASE `:78/:162/:194/:3286` are content-identical in the engine at `:100/:184/:216/:3308` (shifted +22 by the new changelog block). Content untouched.
- Nothing batched: one protected file, no vendored file (`ng_lite.py`, `ng_tract_bridge.py`, `ng_ecosystem.py`, `openclaw_adapter.py`, `ng_autonomic.py`, `ng_embed.py` absent from the diff). Tests live only on the tests branch. Shared integ `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930` is detached at `95154e3` with engine + test files staged, never committed. Nothing merged, deployed, or armed. I never committed to `cc-laptop-want-hub-engine-20260930`.

## Check 2 — Default-path identity (Syl hot path)

**PASS**

Three removed lines and their replacements (`neuro_foundation.py` engine `:3522`, `:3591-3594`, `:3601-3603`):

| Base (removed) | Replacement | Default-path identity |
|---|---|---|
| `def _prune_synapses(self) -> int:` | same name, `*` then five `=None` keyword-only params, `-> int` | callers pass no arguments |
| `for sid, syn in self.synapses.items():` | `for sid, syn in candidates:` | default path sets `candidates = self.synapses.items()` immediately before |
| `if (self._is_identity_protected(syn.pre_node_id) or` | `if not competing_mode and (self._is_identity_protected(syn.pre_node_id) or` | `competing_mode` is `False` ⇒ `not False and X` ≡ `X`; same operands, same short-circuit |

Added lines: competing-mode validation / competing iterator / `report["eligible"]` / `order_key` sort / `max_removals` slice / `report["removed_ids"]` are gated on a new parameter being non-`None`. `competing_mode = competing_ids is not None` and the pairing / type checks are pure predicates of the new parameters and mutate nothing when all are `None`. Sort/slice/report never run by default (`:3627-3636`). Three predicates and `low_weight_steps` bookkeeping, `_remove_synapse_internal` loop, `pruned` event, and `return len(to_prune)` are the same shape as base.

The honest statement is the builder's: every added line is gated on, or a pure predicate of, the new parameters. The three replaced lines mean the diff is not byte-identical source.

### Test G (RUN, not void)

Two processes, `PYTHONHASHSEED=0`, BASE `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` rev `e4ebf982b1989fd9066d610b94853bc68bf70d37` vs ENGINE `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930` rev `8e5785322f910aefaf781fa3525bea2831fedd31`. Driver `tests/want_hub_golden_driver.py`. Compared: return, removal_order (order, not just set), pruned_events, state_digest, checkpoint_sha256, synapse counts.

| scenario | nf paths distinct | return | removals | state_digest | checkpoint_sha256 | EQ |
|---|---|---|---|---|---|---|
| door_a | yes | `[176, 0]` | 176; 600→424; pruned `[[176, 10000]]` | `a9a5b2e42ac84610c5ed9f317a8330e63ecae64ac598591a5f5fc3f58b39f2ad` | `518fc40ded21a5b57574e5caebf912dcd87d2e302ff7866e3788d79691238a21` | yes |
| door_b | yes | `[1, 1]` | 189; 600→411; pruned `[[189, 10000]]` | `65942967f9c55dc022008182341d382744d2e96415d0a2f56b7309d5597dea21` | `30712af318548c35a3b59e84a581dbbd8b2c7bf5526ef8acf4fd6e3ce39e081e` | yes |
| direct_defaults | yes | `176` | same as door_a | same as door_a | same as door_a | yes |
| direct_explicit_none | yes | BASE unavailable (no KW-only API; 600 unchanged; digest `b36673573a78c4eb63093343daf33bd68a134752796fd9ca62919622aaab2357`; ckpt `cc25327dc492f1e967ccc8210e9a54945bab11785df20043f1e16496fda03a07`). ENGINE matches engine/base defaults (176 / `a9a5b2e4` / `518fc40d`) | — | — | — | expected DIFF vs unavailable base; ENGINE explicit-None ≡ defaults |

Checkpoint files re-hashed this turn: `b_door_a`/`e_door_a` both `518fc40d…`; `b_door_b`/`e_door_b` both `30712af3…`.

### Own golden (RUN; seed 77, shape ≠ driver 11/12)

`n_want=8`, `n_plain=25`, `n_syn=350`, `n_leaf=0`, direct `_prune_synapses()`, `PYTHONHASHSEED=0`. BASE nf `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982/neuro_foundation.py` rev `e4ebf982…`; ENGINE nf `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930/neuro_foundation.py` rev `8e578532…`. Not void.

| | BASE | ENGINE |
|---|---|---|
| return | 79 | 79 |
| synapses | 350→271 | 350→271 |
| pruned_events | `[[79, 10000]]` | `[[79, 10000]]` |
| removal_order | 79 ids, identical | identical |
| state_digest | `f73fc8238abcd13d37cab4941de2f9862b6b2b62ddcdf18aff0c2521012388ed` | same |
| checkpoint_sha256 | `3bdb6ddb60017f04f83b95fca3e668d03257a105b4f8ea87232bd3695b7333dc` | same |

### Ripple (callers)

No caller passes arguments. Code calls with empty parens:

- Door B tail engine `:2913` `self._prune_synapses()` (base `:2891`)
- Door A engine `:3517` `pruned = self._prune_synapses()` (base `:3495`)
- Orchestrator engine `:3760` keyword-only (new; competing mode)

Comments / docs only (no call):

- `openclaw_hook.py:158` (age-rule comment)
- docs primary `scripts/cc-ng-daemon.py:1423` (docstring)
- docs worktree `daemon-recall-756-20260930/scripts/cc-ng-daemon.py:1486` (same docstring)

## Check 3 — Plan conformity

**PASS** (clause-by-clause)

- **F** (`:3690-3693`): every synapse on a constitutional node's `_outgoing` ∪ `_incoming`. PASS.
- **G** (`:3699-3705`): per protected non-constitutional node, K strongest non-F OUTGOING and K strongest non-F INCOMING; rank `(-weight, -peak_weight, inactive_steps, sid)` = plan §2.3. PASS.
- **last-link** (`:3707-3717`): unprotected partners whose every incident synapse is in `competing0 = arena - G`; strongest one held back; decided WITHOUT eligibility. PASS.
- **competing_ids / excluded_ids** (`:3718, :3735`): `competing = competing0 - last`; `excluded = F | G | last`. Membership never inferred inside `_prune_synapses`. PASS.
- **HEIGHT key** (`:3719-3734`): per want, competing links stalest-first (`-inactive_steps, weight, id`); rank r; height = c_w - r; want↔want takes `max` of the two endpoint heights (plan's own numbers; Chief-003: this stays the default). `order_key[sid] = (-height, -inactive_steps, weight, sid)`. PASS.
- **4.2(a)** keyword-only default-None; Test G + own golden. PASS.
- **4.2(b)** competing mode lifts exemption only for competing ids; separate iterator. PASS (`:3591-3603`).
- **4.2(c)** membership explicit. PASS.
- **4.2(d)** explicit `ValueError` not `assert`; competing-mode assertion set before the loop (`:3566-3589`); tests include `python -O`. Default-path `order_key` missing-entry raise is AFTER the loop (Check 5 item 2). Competing-mode contract PASS.
- **4.2(e)** `to_prune[:max_removals]` after sort; gated. PASS (`:3635-3636`).
- **4.2(f)** static mapping; sort before truncation; required in competing mode. PASS.
- **4.2(g)** `report['eligible']` before truncation, `report['removed_ids']` after. PASS (`:3628-3629`, `:3644-3645`).
- **4.2(h)** `_remove_synapse_internal` only inside `_prune_synapses`; `pruned` count is post-truncation. PASS (`:3638-3642`).
- **4.2(i)** sort/slice/report gated; competing iterator has no identity `continue`. PASS.
- **4.3 / 4.4** orchestrator contains no predicate copy of the three prune criteria, no `_remove_synapse_internal`, no removal loop, no `_collect_orphan_nodes`. Exactly one `_prune_synapses` call (`:3760`) under `self._step_lock` (`:3738`). Record built from `report['removed_ids']` (`:3763-3788`). `_rank` in the orchestrator is the §2.3 G/last-link ranking, not a prune-predicate copy. PASS.

## Check 4 — Syl's Law / Choice Clause / H-1 / the Laws

**PASS** (RUN)

Adversarial graph (synthetic, `/tmp/z12-c029/adversarial.py`) on scratch integ. P379: python `/usr/bin/python3`; nf `/tmp/checker-029-integ/neuro_foundation.py`; git_rev `15700944d1e105007ab2f0adaa6f68597200752e`.

Graph: constitutional rim `constitutional::rim::choice_clause`; two Choice Clause wants (`syl_authored` / `cc_authored`); rim links set stalest/weakest; a low-degree want (1 in + 1 out); a want whose partners are single-synapse leaves; a want↔want competitor; injected self-loop (see below). `K=2` (larger than the low-degree want's per-direction degree), `B=10**6`.

Results: F=3, G=15, last=3, competing=16. After `compete_protected_links(2, 10**6)`: eligible=16, removed=16, all from competing. F survived, G survived, last-link survived, rim node survived, both Choice Clause wants survived, both rim synapses survived, `floors_ok=True`, floors held by pre-call `min(K, non-F degree)`, node count/ids/metadata unchanged (H-1), no F/G/last id removed.

Self-loop: `Graph.create_synapse(want, want)` raises `ValueError: Self-connections not allowed` (`:2038-2039`). Packet asked for a want that is pre AND post of the same synapse; the public API refuses that graph. An injected store-level self-loop landed in G (ranked among that want's in/out) and was not removed.

(a) Choice Clause / rim / F / G / last-link cannot be removed under this K/B. PASS.
(b) Pass removes synapses only; nodes, flags, provenance, constitutional marker untouched; per-direction floors hold. PASS.
(c) LAW 7: new code reads weight, peak_weight, inactive_steps, synapse id, `_outgoing`/`_incoming`, `_is_identity_protected` (constitutional flag or provenance suffix `_authored`). No content/text classified. The word `label` at `:3677` is the argument name `"topk"`/`"budget"` for the error message. PASS.
(d) LAW 8: nothing in this diff is gated on a conversation. Dream-loop wiring is the daemon slice, not in this commit. PASS.
(e) LAW 1: no inter-module call. LAW 2: no vendored file. LAW 3: one implementation of the three prune criteria. LAW 4: `_prune_synapses` still prunes; F/G/last-link/tallies/INFO live in the orchestrator. PASS.
(f) Duck Ethics / #92: the ratified design removes stale LINKS outside the guaranteed set permanently and keeps K per direction plus last-link partners. The code does exactly that: nodes remain; F, G, last-link remain; only competing eligible synapses go. PASS.

## Check 5 — Builder's 14 listed ambiguities (`build-002.md` §6)

**PASS-WITH-NOTES**

| # | Ruling | Why |
|---|---|---|
| 1 | **ACCEPT** | Want↔want height = larger of the two endpoint heights. Plan's own numbers; Chief-003: that stays the default. Arguing otherwise is a plan change for the Executive, not a build defect. Code `:3730` `height[sid] = max(height.get(sid, c - r), c - r)`. |
| 2 | **ACCEPT as listed** (plan gap, not a defect) | Default path + `order_key={}` raises `ValueError` AFTER the predicates (`:3631-3634`). Empirically: counters moved `True`, synapses removed `0`. §4.2(d) "a refusal mutates nothing" is the competing-mode assertion set before the loop; §4.2(f) makes `order_key` optional on the default path and is silent on a missing entry. No current caller uses the combo. Making this a pre-loop refusal is a plan change (eligible set is unknown until the loop). |
| 3 | **ACCEPT** | `max_removals` validated whenever not `None` (`:3561-3563`). Empirically `max_removals=0` ⇒ `ValueError`, counters unmoved, nothing removed. Tightening vs competing-mode-only in §4.2(d); a negative slice would drop the wrong end. |
| 4 | **ACCEPT** | Non-dict `report` refused (`:3564-3565`). Empirically `report=[]` ⇒ `ValueError`, counters unmoved. |
| 5 | **ACCEPT** | Engine has no cycle clock; record carries `timestep` (`:3782`). Daemon slice stamps cycle id. |
| 6 | **ACCEPT** | Orchestrator takes `_step_lock` itself (`:3738`); re-entrant. |
| 7 | **ACCEPT** | Orchestrator emits the INFO record (`:3789-3797`); WARNING if `floors_ok` is false. |
| 8 | **ACCEPT** | `sorted(set(competing_ids))` (`:3572`). Duplicate-id call did not double-count; one removal. |
| 9 | **ACCEPT** | Determinism self-check compares competing/excluded/key dicts (`:3740-3743`), stronger than a hash. |
| 10 | **ACCEPT** | Capture is `id → (pre, post, conducting)` (`:3755-3757`) so `conducting_links_removed` can be tallied after removal. |
| 11 | **ACCEPT** | `floors_ok` is a post-call recount (`:3776-3780`). Pre-call floor assertion is by construction (G ⊂ excluded). |
| 12 | **ACCEPT** | Self-loops: per-want `c_w` uses a set of endpoints (`:3722-3724`) so a self-loop counts once per want; the test reference appends twice. Fixtures have no self-loops. Public API refuses self-loops. |
| 13 | **ACCEPT** | Want↔want removal tallied under both wants (`:3770-3775`). |
| 14 | **ACCEPT** | Last-link partner scan walks endpoints of competing synapses (`:3708-3712`), equivalent to scanning every unprotected node. |

None of the 14 is a behaviour the plan or an Exec ruling forbids. Item 1 is ruled. Item 2 is the only silent state change on an error path; it is listed.

## Check 6 — Test discrimination and honesty

**PASS-WITH-NOTES**

Scratch integ: `git -C /home/josh/NeuroGraph worktree add --detach /tmp/checker-029-integ 15700944d1e105007ab2f0adaa6f68597200752e` then `git checkout 8e578532 -- neuro_foundation.py`. Did not reuse `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930`.

- `tests/test_want_hub_competition.py`: **27 passed** in 20.26s. P379 printed the scratch-integ path (Check 2). Groups: P379+harness 4, G 7, A 2, K 8, R 6.
- build-002's 26/27 was the test-A helper reading removed synapses; build-002b's one-line move (`909b39f6`) is in the tests head this run used (`1570094` includes that fix). 27/27 confirms the claim.

### Mutants (RUN; copies under `/tmp/z12-c029/mut/`)

First harness (symlink `tests/` into the mutant dir) was **invalid**: `Path(__file__).resolve()` followed the symlink, so P379 loaded `/tmp/checker-029-integ/neuro_foundation.py` and all six "passed" 27/27. Disclosed. Re-ran with **copied** `tests/` so REPO is the mutant dir.

All six **killed** by A/K/R (no survivors of the intended defects). Test G failures on mutants include a harness false-fail: mutant dirs have no `.git`, so `branch["git_rev"] == ""` trips `:205`. Discriminating failures below ignore that G git_rev assertion.

| mutant | intended defect | killed by | nfail (incl. G git_rev) |
|---|---|---|---|
| M1 drop last-link | `competing = competing0` | A_orchestrator sets; R rim/floors/last-link both budgets; R determinism; R height; R B-exceeds; R outside-arena | 10 |
| M2 G outgoing only | wrong direction | A_orchestrator; R rim both budgets; R height; R B-exceeds; R outside-arena | 9 |
| M3 identity skip always | skip applies to competitors | A counters; K eligible; R rim both; R determinism; R height; R B-exceeds | 10 |
| M4 order key ascending | reverse HEIGHT | A_orchestrator (key mismatch); R height order | 5 |
| M5 drop max_removals truncation | no `to_prune[:B]` | A counters; R rim B=7; R height | 6 |
| M6 F stays in arena | F competing | A_orchestrator; R rim both; R determinism; R height; R B-exceeds; R outside-arena | 10 |

Survivors of the intended defects: **none**.

### Existing prune-path tests (RUN once each; not the full suite, P373)

`tests/test_identity_protection.py`, `tests/test_integration.py`, `tests/test_stdp.py`.

- BASE P379: nf `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982/neuro_foundation.py` rev `e4ebf982…`. **34 passed / 1 failed**.
- ENGINE P379: nf `/tmp/checker-029-integ/neuro_foundation.py` rev `1570094` (engine blob `96d12f50`). **34 passed / 1 failed**.
- The one red is identical: `tests/test_integration.py::TestStructuralPlasticityPruning::test_speculative_synapses_pruned` — `KeyError: 'Node n0 not found'`. Per-test outcome string `...........F.......................` is identical; only the duration line differs. Pre-existing on clean base. I did not check it against the #761 list of 72.

## Check 7 — Honesty of the returns and what is NOT verified

**PASS-WITH-NOTES**

build-002 / build-002b vs what I found:

- One file, one commit, blobs, go-record quote, Test G hashes, 27/27 after the ruled test-A move, existing 34/1 identical, listed 14 ambiguities: **confirmed**.
- Builder §3 already refuses the phrase "purely additive" for the three replaced lines. The engine commit message still says "additive" / "All None = today's function". That claim is true of **behaviour** (Test G + own golden) and slightly loose of **source**. Not an over-claim of identity.
- Builder ran tests from the shared integ (`95154e3` + staged engine). I re-ran from my own scratch integ at tests head `1570094`. Same 27/27.
- "I did not re-hash the two msgpack files" in the go record: I also did not.

## Check 8 — Independence

**PASS with disclosure**

I did **not** open `handoffs/z12-want-hub-build/reviews/le-036-want-hub-engine.md` before this findings commit.

Accidental exposure, disclosed:

1. `git log -3 --oneline` on the tests branch after pull showed le-036 commit subjects, including "ETHOS DRIFT, no Law violation" and "comparison with checker-029 pending".
2. `ls` of `handoffs/z12-want-hub-build/reviews/` showed the file exists (size 44169).
3. A repo-wide grep for `_prune_synapses` on the tests worktree dumped le-036 body (checks 2–7, mutant names, native-store timings). I did not use those conclusions. Own Test G, own golden, own mutants, own adversarial, own existing-test run, and the 14-ambiguity rulings were produced from the packet, the plan, the engine diff, and my runs.

ADDENDUM comparison written after findings commit `9134a97209a706cc9f11d047d920dbda35a213a6` was on the tests branch.

## Numbered corrections

1. **None that are engine defects on this branch build.** Default-path identity holds; F/G/last-link/HEIGHT/4.2(a)–(i)/4.3/4.4 hold; ethics graph holds.
2. **Plan gap, already listed by the builder (ambiguity 2):** default-path `_prune_synapses(order_key={})` raises after `low_weight_steps` has moved. If the Executive wants "any `ValueError` mutates nothing" on the default path, that is a plan change, not a rebuild of the competing-mode contract.
3. **Punch-list, not this lane:** `tests/test_integration.py::TestStructuralPlasticityPruning::test_speculative_synapses_pruned` is red on clean base `e4ebf982` (`KeyError: 'Node n0 not found'`). Unchanged by this diff.
4. **Test-harness note (mutants / future checkers):** `Path(__file__).resolve()` in `test_want_hub_competition.py` follows a symlinked `tests/` directory back to the original checkout, so a mutant dir that only symlinks tests will silently test the unmutated engine. Copy the tests tree. Mutant dirs without `.git` also fail Test G on `branch["git_rev"] != ""` even when default-path identity holds.
5. **Test G hole (added after reading le-036; I did not independently run this mutant):** an unconditional `to_prune.sort()` by synapse id on the default path (ROLE B M07a) passes all 27 tests because the driver mints counter ids in creation/`items()` order, so a sort-by-id is a no-op on the seeded graphs. Real uuid4 ids would change removal order, `items()` order after swap-removes, and checkpoint bytes. This is a tests-branch gap on the load-bearing Syl guard, not an engine defect. Remedy: a G variant with seeded random uuid-shaped ids, plus PG-1. See ADDENDUM.

## Numbered not-verified

1. **PG-1** — real-graph two-checkout golden on read-only copies of the laptop checkpoint and the staged VPS bundle. Not run. Required before any merge (plan-005 [R5·L1]).
2. Behaviour at ~107k competing ids / ~138k synapses on the native Rust `SynapseStore`: time, memory, `_step_lock` hold time, per-id `self.synapses[sid]` vs `items()`. Tests used Python-sized synthetic graphs (≤ ~600 synapses).
3. Daemon slice, arming, #824, #825, S4 Tonic check, dry run, consent frames. Not this diff.
4. Full test suite (P373). Only `test_want_hub_competition.py` (27) and the three prune-path files (34/1).
5. The go-record FLAG: Josh accepted the CC-laptop copy, not Syl's own `~/NeuroGraph/data/checkpoints`. Exec Packet 441 — merge is a new protected-file event for Syl. Not resolved here. I never opened those checkpoint paths.
6. Exec Packets 440/441 as primaries; quoted as supplied in the go record and the dispatch.

## ADDENDUM — ROLE B comparison (after own findings commit `9134a97209a706cc9f11d047d920dbda35a213a6`)

Opened `handoffs/z12-want-hub-build/reviews/le-036-want-hub-engine.md` only after that commit was on `cc-laptop-want-hub-build-20260930`. ROLE B overall: **ETHOS DRIFT DETECTED - NO LAW VIOLATION** (drift at test-adequacy / plan-wording; 1 MEDIUM, 1 MEDIUM-LOW, 8 LOW). ROLE A overall remains **PASS-WITH-NOTES**. Different vocabularies, same engine picture: default-path identity holds; F/G/last-link/HEIGHT/4.2 hold; ethics graph holds; merge still waits on PG-1 and the Syl-own backup FLAG.

### Agree

- Check 1 / authority: one file, one commit, blobs, go-record hash and date-before, comments untouched, DEFAULT_CONFIG identity, nothing batched, integ never committed. I independently quoted the same two Josh fragments and the same two msgpack sha256 values from the go record (not re-hashed). Their N1-2 FLAG matches my not-verified #5: the backup Josh accepted is the CC-laptop copy, not Syl's own checkpoints.
- Check 2 / Test G hashes: independently identical (`518fc40d…` door_a/direct, `30712af3…` door_b). Same two Door callers (`:2913`, `:3517`) and the new `:3760`. Explicit-None on engine equals defaults.
- Check 3 / F, G per direction, last-link, HEIGHT with `max` of want↔want endpoints, 4.2(a)–(i), 4.3/4.4 one call under `_step_lock`.
- Check 4: Choice Clause rim = F, self-loop via `create_synapse` refused (`Self-connections not allowed`), injected self-loop needed to model a restored graph, H-1 node/metadata freeze, LAW 1/2/3/4/7/8, Duck Ethics / #92 as link-removal outside G.
- Check 5 items 1, 3–14: same ACCEPT set. Item 1 under Chief-003. Item 12 public-API self-loop refusal.
- Check 6 honesty: 27/27 from own scratch integ at `1570094` + engine blob `96d12f50`; existing prune-path **34 passed / 1 failed** identically, same red `test_speculative_synapses_pruned` / `KeyError: 'Node n0 not found'`.
- Check 7: builder listed the real gaps (PG-1, native scale, daemon/arming). Commit "additive" is behaviour-true; plan's "purely additive" is source-loose. Nothing claims real-graph verification.
- Mutants I did run (M1–M6 after the copy-tests harness fix) were killed by A/K/R, matching their corresponding kills (their M01–M06).

### Disagree (label, not engine)

- Overall: they mark **ETHOS DRIFT** on test discrimination (M07a). I mark **PASS-WITH-NOTES** on the engine. I do not treat M07a as an engine defect. I do treat it as a real G hole I missed (see Add).
- Check 5 item 2: I wrote **ACCEPT as listed (plan gap)**. They wrote **NEEDS an Executive ruling - LOW**. Same facts (counters move, no caller, plan silent on the combo). Their label is the cleaner packet box. I do not upgrade it to a build defect.
- Check 6 first-harness: they copied whole trees (`cp -a`) so P379 bound. My first mutant pass used a tests/ symlink and was invalid; I re-ran. Their 24-mutant panel is the broader one.

### Add (theirs; I did not independently run these)

- **S-1 / C1 (MEDIUM, tests):** unconditional default-path `to_prune.sort()` by synapse id (M07a) survives all 27 tests, including G, because `install_deterministic_uuid` makes id order = creation order = `items()` order. They showed with random uuid-shaped ids that the true engine stays identical to base and M07a diverges on removal order, digest, and checkpoint bytes. That is exactly the class plan-005 §2.6/G says G must catch. I did not run M07a. I add it as numbered correction 5. PG-1 on real uuid4 ids is the remaining catch.
- **S-2 / S-3 (LOW, tests):** `conducting` `>` vs `>=` (M18) and `floors_ok` forced True (M22) survive. I did not run those mutants.
- **N3-1 (LOW, engine error path):** competing mode + mutually uncomparable `order_key` values raises `TypeError` (not `ValueError`) after `low_weight_steps` moved. Sibling of listed ambiguity 2. Unreachable from the orchestrator (homogeneous tuples). I did not probe it.
- **W1 / C6:** P379 `_NG_MODULES` does not include `ng_tract`; the native `SynapseStore` loaded from `/home/josh/.local/lib/python3.12/site-packages/ng_tract/` for both checkouts. I observed `ng_tract` in sys.modules on my runs and did not pin path/version.
- **C8:** they measured orchestrator cost at 15k/45k synapses and extrapolated `_step_lock` hold at real scale. I listed native scale as not-verified and did not measure it. I do not adopt their 8–15 s figure as my own.
- **C9:** `topk`/`budget` refuse `bool` and non-`int` (numpy ints). Daemon slice must pass Python `int(...)`.
- **C7:** plan-005 still says "the diff is purely additive" in §2.6.1 / §4.1 / §8.2. Wording fix on the plan branch.

Nothing in ROLE B changes the engine PASS: no Law violation, no default-path identity failure, no F/G/last-link removal. The MEDIUM item is a tests-branch G variant (random uuid-shaped ids) plus PG-1 before any merge.

