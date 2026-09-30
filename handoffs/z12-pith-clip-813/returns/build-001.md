<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 return build-001
# What: audit disposition, what was built, evidence, deviations, staged .bashrc, decisions.
# Why: assignment build-813-pith-clip.md ("Outputs: handoffs/z12-pith-clip-813/returns/build-001.md").
# How: every number below is from a command run in this session; the full 35-row audit is in
#   ../plan-001-audit.md (committed before code).
# -------------------
-->

# #813 — Pith clip removal + character-cap audit — RETURN build-001

Lane `pith-clip-removal-813` · dispatch #10841 · worker seat (`worktree_write`) · returned **unreviewed**.
Related: [[NeuroGraph]] · [[Pith]] · [[NeuroGraph Is a Mind, Not a Database]] · [[Format-for-Purpose Principle]]

**Status: work returned, not accepted.** Nothing merged, wired, restarted or installed. `~/.bashrc` was
never written (sha256 identical before/after my exercise of the staged script on a temp copy).

## 1. Commits (all pushed by branch name)

| Repo | Branch | Base | Commit | What |
|---|---|---|---|---|
| NeuroGraph | `cc-laptop-pith-clip-813-20260930` | `e4ebf982` | `d4bc615` | plan + audit, **before any code** |
| NeuroGraph | same | | `5a8120a` | build: organism + contract + tests + fixtures + staged script |
| NeuroGraph | same | | *this return's commit* (branch head; see `git log -1`) | this file + plan amendment |
| docs | `cc-laptop-pith-clip-813-20260930` | `7cf85149` (see §7) | `6d41cbea` | plan pointer, before code |
| docs | same | | `8cba7963` | preflight + its test |
| Condensate | — | `4086540` | none | read-only via `git show master:rust_core/src/minitid.rs` |

## 2. What "no truncation" became (the rulings, concretely)

1. **Nodes render WHOLE.** `_pith_node_text` returns `_pith_node_raw_text` unchanged. `CC_PITH_PROVIDER_NODE_CHARS` is gone from the code.
2. **The budget is met by fewer whole assemblies.** `_pith_fit_statement` and the water-filling shortener are deleted; `_pith_fit_connected_line` is "whole render fits → admit, else `None`". `_pith_provider_admit` keeps the strict rank-order prefix and emits **one INFO line** whenever it drops anything:
   `pith provider_context: learned budget B chars met by dropping N whole assemblies (C chars rendered); kept K (U chars)`.
   `pith_stage3` (the L1 assembler) gets the same discipline: its keyframe fallback is deleted and it logs
   `pith stage3: L1 budget B chars met by dropping N whole items (C chars); kept K (U chars)`. No log when nothing is dropped.
3. **A keyframe carries its delta or does not apply.** `pith_stage2_keyframe` already returned `(keyframe, delta)`, but every production caller discarded the delta. A budgeted output has no room for the delta (that is what a binding budget refused), so a keyframe can never satisfy one: it is now called from **no** budgeted path (an AST test enforces this) and its docstring says so. It survives only for `pith_compress_history` (§5, D2).

## 3. Final disposition of the audit (full table with file:line and reasoning: `../plan-001-audit.md`)

| Cap | Class | Result in this branch |
|---|---|---|
| `CC_PITH_PROVIDER_NODE_CHARS` (700) | CUT | **removed** |
| budget-time prose shortening (`_pith_fit_*`) | CUT | **removed** → whole-or-drop |
| `_pith_provider_admit` tail drop | DROP-SILENTLY | **now loud** (INFO count + chars) |
| source label `[:80]` | CUT | **removed** |
| Stage 3 keyframe fallback | CUT | **removed** → drop whole + INFO |
| prefetch LOD keyframe (gated off) | CUT | **removed** |
| recall snippet 300 chars, *provider path* | CUT | **fixed**: `whole_content=True` (opt-in; default byte-identical) |
| recall snippet 300 chars, *gate-off `## Active Recall`* | CUT | **not changed** (no budget there → unbounded injection) — D1 |
| `MAX_INSTRUCTION_CHARS` 8000 | REJECT-LOUDLY | stays (closed `unavailable` + notice) |
| `MAX_QUEST_CHARS` 8000 | REJECT-LOUDLY | stays. **Not dead** — see §5 |
| `pith_compress_history` keyframe | CUT (no live caller) | not changed — D2 |
| `render_wants` `[:600]` / `_WANT_RE` 600 span | CUT / **DROP-SILENTLY** (long wants never captured) | not changed (outside provider context) — D3 |
| Commons deposit metadata `text[:2000]`; host `[:1000/:2000/:1500/:200]` and identical clips in `cc-ng-daemon.py` | CUT on the **deposit** path (LAW 7) | not changed (host excluded; substrate-volume decision) — D4 |
| member/overlap drops, `ROOTS/MEMBERS/DEPTH` | COUNT / DROP-SILENTLY (nodes) | not changed — D5 |
| `surfacing.py` 200-char `"..."` | CUT | not changed (shared with Syl's live `/assemble`; P329) — D6 |
| miniTID `MAX_PROVIDER_CONTEXT_CHARS` 40,000 | **REJECT-LOUDLY** | read-only. Oversized → `parse_provider_response` `None` → failure envelope: provider-facing `[Pith unavailable: …]` **and** raw deposit to the tract. Never cut, never silent. *Wording defect:* the message says "not a fresh provider_context envelope" and never says "oversized" |
| miniTID `MAX_PROVIDER_RESPONSE_BYTES` 256 KiB, `MAX_BODY` 20 MiB | REJECT-LOUDLY | read-only, no change proposed |
| miniTID `PITH_NOTICE_WHY_MAX` 200 (`...` at `:1264-1265`) | **CUT** (the provider-facing notice text; the raw deposit is uncut) | **proposal only** (below) |
| miniTID tool-tail byte envelope (64 KiB) | whole-message DROP, visible only in aggregate counts | proposal only |

**Rust proposals (no Rust edited):** (a) `PITH_NOTICE_WHY_MAX`: delete the `take(200)`+`"..."` branch and keep the whitespace flatten — safe because every `why` is a closed static string or a one-line transport/OS error, so the notice stays bounded by construction; update the assertion at `minitid.rs:3499-3514`. (b) `MAX_PROVIDER_CONTEXT_CHARS`: give the size rejection its own `Err("provider context exceeded N chars")` so an operator can tell size from schema. (c) Log count + bytes of tool-tail messages dropped by `bounded_current_episode`. The Condensate/miniTID change is a separate live-service decision.

## 4. Contract in both places, one change
* NG `docs/PITH_HOST_CONTRACT.md`: export block is **five** names; the "may shorten each member's prose" promise is replaced by whole-or-drop + the INFO log; changelog entry. (`tests/test_cc_pith_clip_813.py::test_host_contract_lists_five_exports_and_no_shortening_promise`.)
* docs `scripts/cc-ng-service.py`: `CC_PITH_PROVIDER_NODE_CHARS` removed from the required tuple; test updated, plus `test_retired_node_chars_export_is_not_required_but_is_tolerated`.
* **VPS host `cc_ng_host.py:509-511` not edited. Effect if it still exports the variable:** nothing functional — the value is copied into the `gates` block of `pith_metrics.jsonl` and ignored by the code. In `config`, `pith_effective_config()` reports it as `resolved: None`, `authority: "retired (#813: nodes render whole)"`, next to `env: "700"`, so a stale export is visible instead of hidden. (The laptop daemon's identical allow-list, `cc-ng-daemon.py:1990`, is likewise inert and also not edited.)

## 5. Things that contradicted the assignment or need a human eye
1. **`CC_PITH_PROVIDER_MAX_QUEST_CHARS` is not dead** at the versions I read. Condensate master still runs `extract_quest_focus` (`minitid.rs:1366`, `:1671`) and sends `quest_focus` on every provider request; both hosts forward it (`cc_ng_host.py:1027`, `cc-ng-daemon.py:1667`). I classified it REJECT-LOUDLY and left it. If the "Quest lane" was removed somewhere I did not read, say where before anyone removes the guard — D7.
2. **`_PITH_CONFIG_KEYS` keeps the retired name** (plan said remove). The suite asserts `env` keys == `resolved` keys == `_PITH_CONFIG_KEYS` == the host allow-list (`test_pith_metrics_concurrency.py:184`, `test_cc_host_pith_telemetry.py:79`), and the host is out of bounds. I followed the module's own precedent (`resolved[k] = None` + authority note). Delete the name from the tuple, the `resolved` dict and the host list together when the host is next edited.
3. **A deliberate refinement of "strict prefix":** an assembly that cannot fit even an *empty* envelope is skipped rather than ending admission — otherwise one giant top-ranked assembly blanks every other. A line that fits an empty envelope but not the remainder still ends admission (a lower-ranked line never jumps a dropped higher-ranked one). Covered by two tests; **reviewer: confirm you want this.**
4. **Consequence worth Josh's attention:** a single assembly whose *whole* render exceeds the largest possible budget (40,000 chars: the organism clamp and miniTID's `MAX_PROVIDER_CONTEXT_CHARS`) can never be shown. That is the honest price of "whole or absent"; it is loud (INFO line), not silent. Whether such a node should be split at ingest is a substrate question, not something I decided.
5. **Log volume:** `pith stage3` and the provider log fire on every recall where the budget binds. That is what "visible, never silent" asks for; if it is too chatty the fix is a rate limit, not silence.
6. **Counters:** `compressed_count` / `chars_saved` from Stage 3 now stay 0 (the fields remain; no schema churn). `test_pith_stage2.py` updated with the reason.

## 6. Evidence (this session)
* **P379:** `tests/test_cc_pith_clip_813.py::test_p379_module_under_test_is_the_worktree_copy` FAILS if `cc_ng_organism` is not this worktree's file; `NG_EMBED_*` scrubbed in the module preamble and run with `env -u NG_EMBED_REMOTE`.
* **RED first:** before implementing, the new file ran `14 failed, 8 passed` against the unchanged code (the golden and preamble tests passed, as they must; every behavioural test failed).
* **Golden vs BASE:** `tests/fixtures/pith_clip_813_golden_base.json` was generated by `tests/fixtures/gen_pith_clip_813_golden.py` from `git show e4ebf982:cc_ng_organism.py` (not from the branch). Four scenarios of well-formed short nodes (action→outcome→correction chain with anchors; two assemblies; a stale member with alert; the live-rail placeholder) render **byte-identical** on the branch.
* **Final targeted run** (once, after the last code change), NG:
  `pytest tests/test_cc_pith_clip_813.py tests/test_pith_provider_context.py tests/test_pith_stage{1,2,3,4,5}.py tests/test_pith_l1_provenance.py tests/test_pith_metrics_concurrency.py tests/test_pith_history_metrics.py tests/test_cc_host_pith_telemetry.py tests/test_cc_recall_dedup.py tests/test_cc_recall_unification.py tests/test_cc_region_confidence.py`
  → **223 passed, 3 skipped** (the skips are the host/daemon parity tests, which need `CC_DAEMON_UNDER_TEST`). Re-run with `CC_DAEMON_UNDER_TEST=<docs worktree>/scripts/cc-ng-daemon.py`: those 3 → **passed**.
  Not the full suite (per assignment). Earlier in the session the same set had 6 failures: five tests that pinned the removed behaviour (updated with reasons in their changelogs) and the config-parity test in §5.2.
* **docs:** `scripts/tests/test_cc_ng_service.py` + `test_pith_provider_context_wrapper.py` → **24 passed**, at the pushed commit `8cba7963`.
* **Not verified / limits:** no live service, socket, sidecar, checkpoint or tract was opened, so nothing here shows behaviour under real load or the real graph; the INFO wording is asserted by tests, not observed in a running daemon; Rust was read, not built.

## 7. Process notes (honest)
* My first docs push was rejected: `git pull --rebase` had rewritten the already-pushed pointer commit. I did **not** force-push; I realigned the local branch to the remote and cherry-picked the preflight commit on top. Consequence: the docs branch is based at `7cf85149` (docs `origin/main` when I started work), not `933f7158`, and is now 2 behind `origin/main` — the merger rebases.
* `git status` in Condensate shows one untracked file, `rust_core/MINITID_KISS_BUG.md`. It is not mine and I did not open or touch it.

## 8. STAGED `.bashrc` edit — NOT APPLIED
Script: `returns/bashrc-drop-node-chars.sh` (`apply | verify | reverse`; `BASHRC=` overrides the target; default `~/.bashrc`).
* **Backup:** `cp -p` to `~/.bashrc.bak-813-<YYYYmmdd-HHMMSS>`; pointer file `…bak-813.latest`.
* **Removal:** by line **pattern** `^export CC_PITH_PROVIDER_NODE_CHARS=` (never a line number); refuses unless exactly 1 line matches (0 or ≥2 → exit 1, nothing changed — tested).
* **Verify (names only):** `bash -n`, then prints the `CC_PITH_PROVIDER_*` **names** present, fails unless `NODE_CHARS` is absent and the other five present. It never prints values or file content.
* **Reverse:** byte-exact restore if the file is unchanged since `apply` (sha256 recorded); if another S4 change landed since, it re-appends only the one removed line and leaves the rest alone (tested).
* **Rehearsal on a locked temp copy of the real file** (real `~/.bashrc` untouched, sha256 identical before/after): exactly 1 match; the entire diff is `291d290 < export CC_PITH_PROVIDER_NODE_CHARS=700`; syntax ok; the 5 other provider exports present; reverse byte-exact.
* **Where in the S4 batch it runs:** **after both merges are DEPLOYED** — (1) the NeuroGraph merge (code no longer reads the variable; P329, NG merges first) **and** (2) the docs merge (preflight no longer requires it) — **and before the service restart** that re-reads the environment. Running it earlier makes the *current* preflight refuse launch (`missing canonical export`). Leaving the export in place is always safe, so if either merge is not live, do not run it. It rides S4's go with the other batched changes (P342); it is not a Josh item.

## 9. Undo
NG: revert `5a8120a` (and `d4bc615` if the docs should go too). docs: revert `8cba7963`. `.bashrc`: nothing to undo — never applied; after S4 use `reverse`.

## 10. Decisions I did not make (need the manager / Josh)
D1 gate-off `## Active Recall` snippet cap · D2 `compress_history` (no live Rust caller; retire the handler or make it lossless/identity) · D3 wants (`_WANT_RE` silently un-captures >600-char wants) · D4 deposit-path clips (LAW 7) · D5 silent member/overlap drops · D6 `surfacing.py` 200-char cut · D7 `MAX_QUEST_CHARS` premise. Detail: `../plan-001-audit.md` §8.

Pairing: a cross-family (non-glm) reviewer + LE must review before anything merges; this worker does not self-accept.


---
## Turn-2 corrections to this return (2026-09-30)
* **§3 row "recall snippet 300 chars, gate-off `## Active Recall`" and plan row #7 were wrong** that the non-provider path has no budget: the **Pith-ON L1 path** feeds the same 300-char strings to `pith_stage3` (a budget). Fixed in turn 2 (#816) for both Pith-ON streams and the gate-off path. My statement in §2/§3 that the provider path "renders whole nodes" was true of `provider_context` only, never of `cc_assemble_recall`.
* **§5.4:** "cannot fit 40,000" should read "cannot fit `learned_budget`" (checker-019 C2).
* **§5.1 / D7:** cite Condensate `origin/cc-laptop-minitid-card7-quest-removal-20260929` (`88ddfec`) as the Quest-removed blob; keep `MAX_QUEST_CHARS` until it and both hosts move together.
* **§6:** the 223-passed figure was correct for the turn-1 head; turn 2 supersedes it in `returns/build-002.md`.
