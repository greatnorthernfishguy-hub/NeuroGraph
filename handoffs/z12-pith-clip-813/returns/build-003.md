<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 3 return build-003
# What: #817 REVERTED and DEFERRED in both repos; proof; what was kept; final heads.
# Why: dispatch #11011; Chief ruling (docs 084b4161): pith_compress_history has two LIVE Python
#   callers, so its retirement moves to the post-track VPS/daemon lane.
# How: `git revert` per repo, each its own commit; conflicts resolved; targeted files run once.
# -------------------
-->

# #813 TURN 3 — RETURN build-003 (a small REVERT)

Lane `pith-clip-removal-813` · dispatch #11011 · worker seat · returned **unreviewed**; the delta pair (`bc4ae7a..HEAD`) judges the REVERTED diff.
Related: [[NeuroGraph]] · [[Pith]] · previous `build-002.md` (partly superseded — see its banner) · `../plan-001-audit.md` §11 · `../plan-002.md`.

**Nothing merged, wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a`, unchanged). Condensate read-only (`master` `4086540`).**

## 1. Why
`pith_compress_history` is **not** dead. It has two live Python callers, each via a call-time `from cc_ng_organism import pith_compress_history` that fails soft: `cc_ng_host.py:974` (VPS host) and `cc-ng-daemon.py:1617` (laptop daemon). My turn-2 claim "no live caller" was true of the **Rust** side only (Condensate master mentions it in a header comment) and I wrongly generalised it. Removing the function together with **both** handlers (LAW 3) spans two other lanes, and this track is laptop-only (P329: no VPS-affecting change). **#817 is DEFERRED to the post-track VPS/daemon lane: function + BOTH handlers removed TOGETHER.**

## 2. What was reverted (each its own commit)

| Repo | Revert commit | Reverts | Restores |
|---|---|---|---|
| NeuroGraph `cc-laptop-pith-clip-813-20260930` | `82cbbcd` | `afc9b3e` | `pith_compress_history`; `PithMetrics.history_calls / history_turns_in / history_turns_compressed / history_chars_in / history_chars_out / history_chars_saved / history_failures` (fields, `_reset_locked` lines, `snapshot()` lines) and `record_history_compression`; the docstring mentions; **`cc_ng_host._handle_compress_history` and its `_DISPATCH` entry, byte-for-byte**; `tests/test_cc_host_compress_history.py`; `tests/test_pith_history_metrics.py`; the contract's `compress_history` event / section / `history_*` counter table / health bullets |
| docs `cc-laptop-pith-clip-813-20260930` | `336954c3` | `771f006a` | the laptop daemon's `handle_compress_history` and its `DISPATCH` entry, and removes my `test_compress_history_is_retired_from_the_laptop_daemon` |

Then `d2f3b78` (NG) and `fd0ccc9c` (docs) mark #817 **DEFERRED** in `plan-001-audit.md` (row 10 + new §11), `plan-002.md`, `build-002.md` (banner), the contract (note on the restored section) and the vault pointer.

### Conflicts and how I resolved them
`git revert afc9b3e` conflicted in **two places**, both from later commits stacking on top of it; no code conflicted:
1. `cc_ng_organism.py` changelog header — later entries (#818, #819) sat in the same hunk. **Kept** both; dropped only the #817 entry; added a TURN 3 entry. The rest of the revert (function, `PithMetrics`, docstrings) applied cleanly.
2. `tests/test_cc_pith_clip_813.py` — the header block and the tail. The tail conflict block contained the four `test_817_*` tests **followed by** the #818/#819 tests, so taking "their" side would have deleted kept tests. I removed **only** the `test_817_*` functions and #817's header entry; verified afterwards: **6 `test_818_*` and 7 `test_819_*` functions still present**. The AST guard's tightened assertion was restored to `keyframe_callers <= {"pith_compress_history"}` (auto-reverted) and its comment now says #817 is deferred.
The contract merged cleanly: #817's removals were undone and the #816/#818/#819 paragraphs kept.

## 3. Proof

| Claim | Evidence (run this session) |
|---|---|
| `cc_ng_host.py` byte-identical to base | `git diff e4ebf982 HEAD --stat -- cc_ng_host.py` → **empty**; blob `e3b568c0…` is the same at `e4ebf982` and `HEAD` |
| function + counters + record method restored | `cc_ng_organism.py:4449 def pith_compress_history`, `:3745 history_calls`, `:3751 history_failures`, `:3833 def record_history_compression` |
| host handler restored | `cc_ng_host.py:974 def _handle_compress_history`, `:1425 "compress_history": _handle_compress_history` (the exact lines the Chief cited) |
| laptop handler restored | docs `scripts/cc-ng-daemon.py:1617 def handle_compress_history`, `:1722 'compress_history': handle_compress_history`; the file's blob equals the branch base (`7cf85149`) — **identical** |
| restored history tests are the original ones | `git diff e4ebf982 HEAD --stat` on both files → empty (identical to base) |
| no protected / vendored / shared / host file touched | changed vs base (non-`handoffs/`): `cc_ng_organism.py`, `docs/PITH_HOST_CONTRACT.md`, `tests/…` only; none of `neuro_foundation, openclaw_hook, stream_parser, activation_persistence, ng_*, surfacing, surface_resolver, neurograph_rpc, kiss_filter, tonic_thread, cc_ng_host` |

**Targeted run, once (worktree code only, `env -u NG_EMBED_REMOTE -u PYTHONPATH`, no daemon env):** the union of the turn-1/turn-2 sets plus the two restored history files — `test_cc_pith_clip_813, test_pith_provider_context, test_pith_stage1…5, test_pith_l1_provenance, test_pith_metrics_concurrency, test_cc_host_pith_telemetry, test_cc_recall_dedup, test_cc_recall_unification, test_cc_region_confidence, test_pith_history_metrics, test_cc_host_compress_history` → **261 passed, 3 skipped** (the 3 = host/daemon parity, which need `CC_DAEMON_UNDER_TEST`). **Parity tests run alone** with `CC_DAEMON_UNDER_TEST` → docs worktree daemon → **3 passed**. (They stay separate on purpose: the daemon import puts the primary checkout on `sys.path` and would make later tests run stale code — see `build-002.md` §8.)
**docs:** `scripts/tests/test_cc_ng_service.py` + `test_pith_provider_context_wrapper.py` → **24 passed** (25 minus the one retirement test the revert removes).
**Golden vs BASE `e4ebf982` unchanged and passing** (4 provider + 6 `cc_assemble_recall` scenarios, byte-identical). P379 preamble test passes (module under test is the worktree copy).
**Not verified:** no live service, socket, sidecar, checkpoint, tract or real graph; nothing built in Rust.

## 4. What is KEPT from turn 2 (unchanged by this revert)
Both Pith-ON streams **and** the gate-off path whole via the CC-only monitor route (shared `surfacing.py` / `surface_resolver.py` untouched); THE ONE budget rule (incl. removal of Stage 3's first-line overrun guard, **D8** still open for the reviewer); the fail-open un-Pithed renderer; F3a self-restore + F3b `--follow-symlinks` (script and tests); #818 loud drops; #819 trees + one-line reference (with the pre-PASS-2 dependency); audit row #7; F2/F6/F8; D7. Where #817's removal had simplified something the kept code touches: `pith_stage2_keyframe`'s docstring again names `pith_compress_history` as its only caller — accurate — and the complete-caller-set AST guard allows exactly that one caller.

## 5. Corrections to my earlier returns
* `build-002.md` §1 (`afc9b3e`, `771f006a`), §3 "#817", §4 row #10, §5 and §6 **D9** described the retirement — **superseded** (banner added). **D9 is moot**: `pith_stage2_keyframe` again has one caller, `pith_compress_history`.
* Row 10 of the audit stays a **CUT** (a lossy keyframe that discards its delta) and stays on the list until the VPS/daemon lane removes it, or rebuilds it lossless (keyframe + delta).
* Other turn-2 statements that mentioned the history counters as gone (`build-002.md` §3 removed-reference list) are likewise superseded.

## 6. Final heads (read with `git log -1` / `git ls-remote`)
* **docs** `cc-laptop-pith-clip-813-20260930`: **`fd0ccc9c08b965e6b9c68baebe6f78a09ed36931`** (revert `336954c3`, pointer `fd0ccc9c`; nothing follows it).
* **NeuroGraph** `cc-laptop-pith-clip-813-20260930`: the commit that adds **this file** and nothing else. Its parent is `d2f3b78` (markings) ← `82cbbcd` (revert) ← `33d212b` (turn-2 head). A file cannot name its own commit; the exact hash is in the reply and equals `git rev-parse HEAD`.

## 7. Deferred (record for the post-track VPS/daemon lane)
Remove `pith_compress_history`, `PithMetrics.history_*` + `record_history_compression`, **`cc_ng_host.py:974` + `:1425`**, **`cc-ng-daemon.py:1617` + `:1722`**, the two history test files and the contract's `compress_history` sections **together**, in one change; or rebuild it lossless. S4 order is unchanged from `build-002.md` §10.
