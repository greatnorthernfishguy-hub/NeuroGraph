```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, worker seat) - plan-004-r4f: return for the [R4f·seq] sequencing note
             (dispatch #11826, with the Exec Packet 439 confirmation of dispatch #11834 folded in).
-------------------
```

# plan-004 [R4f] - the two-step Phase 1 (sequencing note only) - RETURN

**Commits (branch `cc-laptop-want-text-repair-20260930`, pushed by name):**
- **[R4f·seq]** `cf3fb9b395839754b4a999b88b7f64da3d3deeb4` - the sequencing note on `plan-004.md`, **with the Exec Packet 439 confirmation folded in**. The addendum arrived before I had committed, so there is **no separate [R4f·P439] commit**.
- **pin-line follow-up** - the next commit: one changelog bullet (58) on `plan-004.md` recording the hash above, plus this file. Its own hash is the one reported in the reply (a file cannot carry its own hash).

**Source:** Chief-003's ruling on the P437 finding (`build-tool-007b.md` section 1, tool branch commit `b187ce0`), **CONFIRMED by the Executive (Exec Packet 439, via Chief-003)**. I did not read `build-tool-007b.md`: what it is cited for (the Graph-free classify, the probes P1-P4) is as stated in the dispatch.

**The hunks (all on `plan-004.md`; section and one line each):**
1. Changelog, bullets 56-57: the note and the Packet 439 confirmation (and 58 in the pin-line commit).
2. §0.1, new item 7 (the ruling map): Phase 1 runs as two separately-scheduled steps, CONFIRMED.
3. §4.4 "Tooling": one sentence - the 8 GiB heavy floor stands for step (1b); step (1a) gets its own floor only from a measured peak.
4. §6.6 Phase 1: an inline annotation on the "3G classify/rewrite, 6G V11" figure ("historical plan figure - now two steps") and the sequencing note as a new block: (1a) classify-light, (1b) rewrite + V1-V19 ending with V11; V11 stays THE gate; floors; CONFIRMED with the three Packet 439 points.
5. §7, the V11 line: one clause - V11 runs as the last step of (1b) at the heavy floor and remains THE gate before Phase 2.
6. §10: a recorded RULED line (not a remaining question) replacing the PENDING entry I first drafted.

**The three Packet 439 points, recorded in the note:** (i) light classify (no input Graph, no full vectors load, results proven identical to the Graph-based classify by the delta build's equivalence test) is dispatchable at ordinary free memory once the delta build and its pair land; (ii) rewrite + V11 is scheduled separately at the measured heavy floor, V11 stays canonical and a GATE and REMAINS in Phase 1 (plan-004 §6.6 and §7); (iii) the heavy floor is re-derived only from a measured V11 peak (probe P4) and stays 8 GiB until then.

**Nothing but sequencing text changed.** `git diff -U0` on the [R4f·seq] commit is six hunks; the three pre-existing lines it touches (Tooling, Phase 1, V11) are EXTENDED, never altered (verified: each old line is preserved inside its new line). Every "PENDING" / "one-line revert" marking is gone. The repair plan - what is repaired, gates P1-P10, V1-V19, the approvals, Phase 2 - is untouched. The §6.1 gate list does not restate Phase 1's order, so it has no hunk. `git diff --stat` shows `plan-004.md` only (plus this return in the pin-line commit). Plan only: no build, no data load, nothing applied, no checkpoint directory listed or opened, `neuro_foundation.py` and the vendored files untouched, no raw want text.
