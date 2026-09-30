<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 7 return build-007
# What: le-027 N-2 (FIX, failing-first), N-4, N-5; N-1 RECORDED ONLY (with a quotable paragraph); the C4 open-note wording.
# Why: dispatch #11293; Chief ruling on le-027 (COMPLIANT, PASS-WITH-NOTES; turn 6 only).
# How: NG repo only. "Before" values come from runs made on the turn-6 code (commit 77a18ce / head efd9179)
#   BEFORE the change existed.
# -------------------
-->

# #813 TURN 7 — RETURN build-007 (tiny)

Lane `pith-clip-removal-813` · dispatch #11293 · worker seat · returned **unreviewed and NOT MERGED**.
Related: [[NeuroGraph]] · [[Pith]] · [[Duck Ethics]] · previous `build-006.md` · review `../reviews/le-027-813-turn6.md` (read in full).

**Nothing merged (the NG-first merge is Josh-gated, P329 — this owner did not merge and will not), wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a`, unchanged). Docs repo, `cc_ng_host.py`, `surfacing.py`, `surface_resolver.py` and every protected/vendored file not touched. Condensate not touched.**

## 1. The items

| Item | Result |
|---|---|
| **N-2 (MEDIUM) — FIX, failing-first** | **Done.** The identity guard **still fails closed exactly as ruled** (raise/missing ⇒ pinned; one WARNING per id, first-seen). New: **ONE count-only WARNING PER CALL** whenever any item of that call was pinned *because the guard failed*, on **both** paths that use the probe. Shape: `pith <where>: K of N items pinned because the identity guard failed (<ExcType>[/<ExcType>…]); L1 X chars vs budget B`. Counts and exception **type** only — no node text, no ids (the existing first-seen WARNING already names them once). A **working guard emits nothing**. Mechanism: one probe lives for one recall call, so `_cc_pin_probe` (`cc_ng_organism.py:5759`) keeps that call's record (`_pinned.guard_failed`, `:5779`; filled by `_cc_pin_guard_failed`, `:5740`), and `_cc_log_guard_pins` (`:5783`) turns it into the line — emitted from the Pith-ON path after Stage 3 (`:6102`) and from `_cc_render_unpithed` after admission (`:5920`). **Level chosen: WARNING**, because a dead identity guard is an operational fault that keeps costing the prompt while it persists, whereas the C3/N4 lines are INFO because they report *designed* behaviour (a drop, a reference, trees left out); the per-id WARNING it accompanies is already WARNING. It repeats every call while the guard stays dead — deliberately: that is the signal — but is one line per call, never per item (tested: 3 calls ⇒ 3 lines, and still only 5 per-id WARNINGs). `L1 X chars` is the summed **item content** of what was kept (le-027's 50,614/50,619 also counts block framing). |
| **N-4 (LOW)** | **Done as a test, by choice.** I did **not** derive `_PITH_COHERENCE_STATES` from one source: the other four sites are the assignment ladder (`pith_connected_activation_basins`, `:5358` onward), the `coherence_support` weight dict (`:5406`), the `coherence_order` tuple (`:5634`) and `_pith_node_coherence`'s returns (`:5125`) — if/elif code, a function-local dict and a local tuple, so deriving would mean restructuring behaviour to fix a LOW note. Instead `test_n4_every_place_that_lists_a_coherence_state_agrees_with_the_shared_vocabulary` (`tests/test_cc_pith_clip_813.py:1527`) parses all four with `ast` and asserts each equals `_PITH_COHERENCE_STATES` (`:5492`), and that `_PITH_ALERT_COHERENCE` (`:5493`) is a strict subset with an ordinary state left — so adding a state in one place fails the test. |
| **N-5 (LOW)** | **Done.** (i) The permissive `or` chain is gone: `_assert_fail_closed_handler` (`tests/…:1339`) asserts the handler's **exact structure** — `except Exception as exc:` with one statement, `return _cc_pin_guard_failed(node_id, exc, failed)`, those three arguments, no keywords. `test_n5_the_exact_handler_assertion_rejects_the_turn5_module_and_a_bare_return_true` (`:1352`) proves it **fails** on the turn-5 module, on BASE's soft-`False` closure, and on a handler that merely says `return True`, and passes on head. (ii) `FakeGraph._is_identity_protected` now uses `isinstance(prov, str) and prov.endswith("_authored")` exactly as the real guard (`neuro_foundation.py:3551-3572`, read only). |
| **N-1 — RECORD ONLY (not implemented)** | See §4 (quotable paragraph). |
| **N-3, `cc_topology_export._is_identity_protected` mirror (#830)** | The Chief's and Josh's; nothing edited, no shared file touched. |

## 2. N-2 — failing first, then passing

**Written first** (10 new tests — `test_n2_*` ×8 counting the dead-guard parametrizations, `test_n4_*`, `test_n5_*` — plus the tightened existing `test_c1_the_pin_probe_is_the_base_closure…`) and run on the **turn-6 head `efd9179`** before any change: **8 failed, 11 passed**. The failures were both dead-guard modes (`raising` = `RuntimeError`, `missing` = `AttributeError`) on both paths, the repeat-per-call test, the partial-count test, and the two exact-handler tests; the dominant assertion was `ValueError: not enough values to unpack (expected 1, got 0)` — i.e. **no per-call line was emitted** — and `assert ['node_id','exc'] == ['node_id','exc','failed']` for the handler structure. (The 11 that passed on the old head are the guards: the working-guard "no line" tests, the N-4 equality test, the golden and the earlier C-tests.)

**le-027's probe world reproduced** (5 non-identity items of 1500/1500/1500/6000/40000 chars, default budget 4000):
```
                                            L1 chars   whole items   per-call line
BEFORE (77a18ce)  Pith-ON  guard=working        3086        2/5            0
BEFORE (77a18ce)  Pith-ON  guard=dead          50619        5/5            0     <- silent
BEFORE (77a18ce)  gate-off guard=dead          50619        5/5            0     <- silent
AFTER  (head)     Pith-ON  guard=working        3086        2/5            0     (unchanged, no line)
AFTER  (head)     Pith-ON  guard=dead          50619        5/5            1
   -> "5 of 5 items pinned because the identity guard failed (RuntimeError); L1 50500 chars vs budget 4000"
AFTER  (head)     gate-off guard=dead          50619        5/5            1     (same line)
```
Fail-closed is intact (5/5 kept whole, byte-for-byte the same L1), the working guard is unchanged, and the only difference is the line. The tests also assert: `n0`/`n3` and the secret exception text appear nowhere in the line; a guard that raises for only `n0, n1` yields `2 of 5 … (KeyError)`.

## 3. Commits, diff, proofs

| | |
|---|---|
| base of this turn | `efd91791eeab9990b5604821d2bab82e7d4e5379` = turn-6 return `babf37b` + le-027's review commits only (`git diff babf37b efd9179 --stat` is that one review file) |
| **code commit** (the commit before this return file, **pushed before the definitive test run**) | **`c1d73864f54aee7fb82c6e0eb511e595cfe5e2ce`** |
| **return commit** = final NG branch head | the commit that adds **this file and nothing else** (parent `c1d7386`); a file cannot name its own commit — the exact hash is in the reply and equals `git rev-parse HEAD` / `git ls-remote origin refs/heads/cc-laptop-pith-clip-813-20260930` |
| docs branch head (untouched) | `fd0ccc9c08b965e6b9c68baebe6f78a09ed36931` |

`git diff efd9179 HEAD --stat` at the code commit: `cc_ng_organism.py | 60 +++++++++-` · `tests/pith_clip_813_scenarios.py | 3 +-` · `tests/test_cc_pith_clip_813.py | 163 ++++++++++++++++++++++-` → **3 files, 218 insertions, 8 deletions** (this return adds one file).

* **`git diff e4ebf982 HEAD --stat -- cc_ng_host.py` → empty** (byte-identical to base).
* Protected / vendored / shared (`surfacing, surface_resolver, cc_ng_host, neurograph_rpc, kiss_filter, tonic_thread, neuro_foundation, openclaw_hook, stream_parser, activation_persistence, ng_*`) touched vs base: **none**.
* Golden vs BASE `e4ebf982` unchanged and passing; P379 preamble passes (`env -u NG_EMBED_REMOTE -u PYTHONPATH`; the probe confirmed the module path is this worktree).

## 4. Tests — once, on the pushed commit
**(A)** the turn-6 union (`test_cc_pith_clip_813, test_pith_provider_context, test_pith_stage1…5, test_pith_l1_provenance, test_pith_metrics_concurrency, test_cc_host_pith_telemetry, test_cc_recall_dedup, test_cc_recall_unification, test_cc_region_confidence, test_pith_history_metrics, test_cc_host_compress_history`; worktree code only, no daemon env) at `c1d7386` → **312 passed, 3 skipped** (302 at turn 6 + 10 new; the 3 skips are the host/daemon parity tests). **(B)** parity tests **alone** with `CC_DAEMON_UNDER_TEST` → the docs worktree daemon → **3 passed**.
**Not verified:** anything live; how the host and prompt tolerate a very large L1 when a guard is dead (le-027 N-2a — this turn makes it *visible*, it does not bound it, and bounding it would need a second copy of the identity predicate, which is Josh's call); the real `Graph._is_identity_protected` under load (tests use a fake that mirrors its contract).

## 5. RECORDS (quotable)

**N-1 — the accepted price of the optimistic node limit (record only; not implemented).** *Turn 6 made the per-node reference limit an optimistic bound so a node that used to render whole never becomes a reference or vanishes in a tight budget (57 of 57 grid points that rendered whole under BASE or turn 5 still do). The price is a narrow band: when the real assembly needs more overhead than the minimum — about 60 characters for a source line, more with anchors or relations, about 600 with an alert shell — a node that turn 5 showed as a REFERENCE is now DROPPED WHOLE (example, core 300 / budget 1300: nodes of about 828–888 characters). It is never a cut and never silent (loud INFO `… never-fit … <id> (N chars)` and `capacity_empty`), but its trees are absent too, because the whole basin is the admit unit. The remedy is on file — a second pass restricted to never-fit roots, re-rendering that one basin in the reference form using the real candidate shell that `pith_provider_context` already computes — and is NOT to be implemented unless the tight regime proves reachable in production; nobody has measured the real constitutional-core size or the `cc_l1_budget` range.* (le-027 N-1.)

**C4 — open note for the Exec (wording as ruled):** *No drop line for pins; a dead-guard L1 exceeds the budget, and the count-only line is now the signal.* Pinned Stage-3 / un-Pithed lines sit outside the budget by design (identity is indivisible); a large pin, or a dead identity guard that pins everything, can push the L1 past `budget_chars` with no drop line, because drops are the only thing the other lines report. With this turn a *failed guard* raises `K of N items pinned because the identity guard failed …; L1 X chars vs budget B` once per call. A *healthy* large pin still has no line; folding pins into THE ONE rule remains an Exec call.

## 6. Open notes carried forward (unchanged)
C4 (above) · N-1 (above) · #817 deferred (checklist in `plan-001-audit.md` §11) · T1/#812 (source fix deletes the fork) · #829/#830 (shared fail-softs; the `cc_topology_export.py` mirror checks a node *attribute* `constitutional`, not `metadata['constitutional']` — le-027 N-3 — dormant since #147) · D3 wants · D4 deposit-path clips (#741) · D7 `MAX_QUEST_CHARS` until card-7 (`88ddfec`) and both hosts move together · #821 · N5 (`on_monitor_error` / an `ImportError` of `surface_resolver` drops the whole monitor stream, loudly). **Merge:** NG merges first (P329, merge = deploy) and is Josh-gated; the owner does not merge.
