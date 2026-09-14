---
name: ruling-93-106-probation-journal
description: #93 (earned graduation) + #106 (merge-journal poison-pill) — post-commit re-review 2026-08-01 at 8237774. Applied fixes verified; the LIVE CC-side #93 bypass (cc_ng_host._deposit -> on_message -> ingestor.update_probation) is the real residual defect; a duplicate #93 sibling commit 6ddd960 is stuck mid-merge in the worktree.
metadata:
  type: project
---

Two review passes. Pass 1 was on the uncommitted diff; pass 2 (2026-08-01) re-reviewed the committed result at `8237774` ("#93 + #106: earned graduation, and un-poison the merge journal", parent `418a9f0`).

## The correction that governs this whole area (Josh, 2026-07-31)

> **The Universal Ingestor has NOTHING to do with the conversational turn/experience path. It is for documents and reference materials. The topology sync is the corpus callosum.**

**This holds for Syl and was re-verified at pass 2:** `grep '\.on_message('` still returns **zero hits in `neurograph_rpc.py`**; Syl's turn path stamps probation inline and sweeps via `_update_probation(_memory.graph)` (`neurograph_rpc.py:3561`). `NGSaaSBridge` (`ng_bridge.py:59`) is **never instantiated in production** — no caller anywhere outside tests — so `ng_bridge.py:168/332/334`'s `_memory.on_message()` calls are dead. `syl_sync.py:175` is still a manual batch tool. **Syl is clean. Do not prescribe unifying the document and conversational probation domains.**

## The residual defect, now located and LIVE — and it is on the CC side, not Syl's

Josh's correction covers Syl's pipeline. **CC's host does not use Syl's turn pipeline** — it deposits CC's conversational turns through the Ingestor-bearing `NeuroGraphMemory.on_message`. Full live chain, each hop verified at pass 2:

`_handle_user_prompt_submit` (`cc_ng_host.py:588`) → `threading.Thread(target=_deposit, ...)` (`:611`) → `_deposit` (`:393`) → `ng.on_message(text)` (`:401`, on `_STATE.cc_ng`) → `NeuroGraphMemory.on_message` → `self.ingestor.update_probation()` (`openclaw_hook.py:951`, **no `node_ids` → unscoped whole-graph sweep**) → `universal_ingestor.py:2039` stamps `graduated = True` on timer expiry, **ungated by firing**.

Meanwhile the CC autosave pulse (`cc_ng_host.py:888`) runs the #93-gated `cc_update_probation` on **the same graph**. Consequences:
- Both sweeps decrement `probation_remaining`, so CC's window burns ~2x fast and two different dampening curves fight over `intrinsic_excitability` (the Ingestor's log/linear `get_dampening_factor` vs `cc_update_probation`'s linear ramp).
- **Whichever sweep lands the final decrement decides whether #93's gate applies.** `_deposit` runs per CC prompt; the pulse is 60s — so in an active session the Ingestor usually wins, stamping `graduated=True` with zero spike evidence and **no** `probation_expired_unfired` marker. `cc_update_probation` then sees `prob <= 0` with no marker and falls straight through. The un-earned stamp is permanent and invisible. **#93 is defeated on CC by a live path.**

**Search gotcha that hid this at pass 1:** `grep '_deposit('` misses `target=_deposit,`. Grep the bare name when checking whether a function is dead.

**Remedy (unchanged in shape from pass 1, now with a target):** scope `UniversalIngestor.update_probation`'s **sweep** to document-provenance nodes so it only graduates what it stamped — `universal_ingestor.py:1965` already stamps `"creation_mode": "ingested"`, a ready-made discriminator. Preserve the domain separation; do not unify. Whether CC's conversational deposit *should* route through `on_message` at all is a LAW 6 question for Josh, not something to change unilaterally.

## Pass-2 verification of the pass-1 applied fixes — all confirmed good

- (c) two-way rollback: gate is INSIDE the `probation_expired_unfired` branch in both `neurograph_rpc.py:2477` and `cc_ng_organism.py:1487`. Traced all four knob/marker states.
- (d) `journal_stale_readmitted` increments at the absorption site (`cc_topology_merge.py:324`), past the provenance gates and the deposit `try/except`.
- `RingBuffer`-vs-`deque` docstring corrected in both files; `except TypeError` now warns before returning False.
- `skipped_journal` → `journal_stale_readmitted` rename has **zero stale consumers** tree-wide.
- `_append_journal(..., [n for n in landed_ids if n not in journal])` correctly avoids duplicate lines.
- **Tests: 45 passed / 2 failed** on `test_cc_topology_callosum.py` + `test_conversational_recall.py`; the 2 are the same pre-existing failures (`test_embedding_failure_routes_to_retry_queue`, `test_tree_concept_lands_in_graph_and_vdb`). `test_cc_dual_pass.py -k "probation or graduat or rollback or kiss"` = 10 passed. Run in a detached worktree at `8237774` because the main tree does not parse (see below).
- No vendored file touched, either pass.
- **SUPERSEDED by #131 (2026-08-04, ruled COMPLIANT):** the pass-2 "non-issue checked and cleared" verdict on `_cc_kiss_reinforce_node` (`cc_ng_organism.py:1326`) was WRONG. It stamped `graduated=True` ungated when reinforcement drove `prob<=0`, producing the exact un-earned state #93/#111 outlawed (graduated + empty spike_history). Live proof: `cc:conv::1028c3d6` graduated with 12 confirmations while 17-18-confirmation siblings did not — the stamp tracked which tick the decrement landed on (timer-race), not confirmation strength. #131 gates this last site on `_cc_has_ever_fired` exactly like `cc_update_probation`'s two paths; un-fired → `graduated=False` + `probation_expired_unfired=True`, re-earns on first real spike. Dampening release stays unconditional. Reinforcement ACCELERATES the timer but is conversation-driven and only additive — base decrement + graduation-earn both still ride the 60s autosave pulse, so LAW 8 is clean (probation_remaining and spike_history both advance autonomically). LAW 7 win: stops conflating an input-boundary cosine near-dup ("content recurred") with evidence of cognition. 3 new tests all `monkeypatch.setattr(cc, "_CC_CONV_PROBATION_REQUIRE_SPIKE", ...)` (module attr, not env) — correct per [[finding-pith-gate-tests-need-monkeypatch]].

## LAW 3 — duplicate #93 implementation stuck mid-merge (found at pass 2)

The working tree is in an **unresolved `git merge`**: `.git/MERGE_HEAD` = `6ddd960` ("#93: graduation must be earned, not aged into", Claude Opus 4.6 1M, also a child of `418a9f0`) — a **second, independent implementation of #93** being merged into `8237774`, which already contains #93. Three files are `UU` with conflict markers committed to the working tree: `neurograph_rpc.py`, `cc_ng_organism.py`, `tests/test_conversational_recall.py`. **Syl's live RPC module does not currently parse.**

`git diff 6ddd960 8237774` proves **HEAD is a strict superset**: the only things HEAD "removes" are exactly the three defects ruled on at pass 1 — the `deque(maxlen=100)` docstring error, the silent `except TypeError`, and the one-way rollback (`if _CONV_PROBATION_REQUIRE_SPIKE and node.metadata.get(...)`). Resolution is therefore mechanical: **take HEAD/ours for every hunk**, nothing to salvage from the sibling. `tests/test_cc_dual_pass.py` auto-merged cleanly with no duplicate test defs.

## Still open (verified still open at pass 2)

- **(b)** `exclude_ids` has **no production wiring at all** — only `cc_topology_export.py:207/352` accept it, nothing passes it. `merge_cc_topology` still has **no production caller** (tests only), so #106 remains a correct fix to not-yet-live code. When wired, `exclude_ids` must be a presence manifest from the receiver's live `graph.nodes`, **not** the journal — `landed_ids` still never includes `skipped_present`, so after any journal loss the file never repopulates.
- **(e)** LAW 5: `ANIMA_CONV_PROBATION_REQUIRE_SPIKE` and `CC_CONV_PROBATION_REQUIRE_SPIKE` are absent from `.bashrc`. Broader: **zero `ANIMA_` knobs are declared there** (`grep -c 'ANIMA_' ~/.bashrc` = 0), and `/home/josh/openclaw.json` **does not exist** at that path. Separately, `.bashrc` exports `CC_NG_HOST_MODULES=1` which is referenced **nowhere in the tree** — a phantom knob.
- From [[ruling-half-brain-wholeness-ring]], untouched: `batch_landed` written-never-read (`cc_topology_merge.py:222/232/318`); `orphan_node_grace_period` still has no env override.
- [[finding-cc-authored-export-protection-collision]] still stands — the receive-side identity gate `str(meta.get("provenance")).endswith("_authored")` (`cc_topology_merge.py:264`) skips `cc_authored`.

## Ethos watch (carry to #93's consumer diff)

`graduated` is a binary flag. It records an event (ever spiked), not a competence tier, and gates no behavior today — acceptable. When #93 consumes it, it must feed a **continuous** protection signal, not a hard protected/unprotected gate.

## How to apply at next review

Require: (a) the Ingestor **sweep** scoped to document-provenance nodes — do NOT unify the domains, do NOT retire the sweep; (b) `exclude_ids` from graph membership; (e) knobs declared in `.bashrc`. Verify the merge was resolved as ours-everywhere. Re-check the CC bypass chain by grepping the **bare name** `_deposit`, not `_deposit(`.
