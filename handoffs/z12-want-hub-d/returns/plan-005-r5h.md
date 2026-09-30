# plan-005 [R5h] return — Josh's ratification of the consent frame (Exec Packet 435 ITEM 1) and the reworded frame line (Exec Packet 436)

Lane `want-hub-competition-d` · dispatch #11762 (RETRY of #11734) · author thread `097da447` · PLAN ONLY: `neuro_foundation.py` untouched, no graph load, nothing built. Branch `cc-laptop-want-hub-d-20260930`, head at dispatch `a65e59bb8551e97c72271ab5e0cbe2001333aa26`.

**Commit:** `74570bedcb0da12ac4de2b0757e88b77d2506b33` — the R5h delta on `handoffs/z12-want-hub-d/returns/plan-005.md` + this file (pinned here and in plan-005's changelog by a one-line follow-up commit, since a commit cannot contain its own hash). Cite that hash, never the file name.

## Hunks changed in plan-005 (section — one line each)
- **Changelog (top):** new entry `[R5h·P435][R5h·P436]`, one bullet per item; the old line appears here as the superseded wording. Two history lines in the R5f/§10-424 text that carried the old string now say "the ratified-variant line".
- **Lead / legend / top ruling map / TOP NOTICE ruling paragraph / §0 item 9:** P435 ITEM 1 + P436 named; ruling map row added.
- **§4A.4 frame gate:** the E2 condition is LIFTED; the gate is satisfied by the recorded ratification; revocation / changed answer still re-blocks.
- **§4A.6:** the ARMING frame "after:" text and the CHECK-IN frame carry the Executive's line; state (a) RATIFIED is the current state; "Whose words"; the **No lean** guard sentence.
- **§4B:** stop-list clause and arming-order clause marked LIFTED (live only against a later revocation).
- **§7:** the ratification is RECORDED verbatim in the consent record (operator-side), with the guard and a flag; precondition (e) LIFTED, (f) stands.
- **§8:** new pair-check item 14, "the frame text contains no lean".
- **§9:** the arming-order clause and the approvals-directory note.
- **§10:** new row "435 ITEM 1 + 436"; the open list entry for Josh's ratification replaced (RECORDED); "Ruled and closed" extended.
- **§11.8 (new), §12:** the R5h items mapped; the pointer row pins the states.
`git diff --stat` shows ONE file, plan-005.md; nothing else in the plan changed.

## Byte-for-byte confirmation (string compare against the Executive's assignment files; `sha256` of each string)
The census sentences are extracted from `assignments/plan-want-hub-d-p419.md`, the D2/P424 sentences from their packets' transcriptions, the P436 line from ADDENDUM 1 of the r5h brief. Counts are in the post-edit plan-005.

| string | chars | count in plan-005 (after) | sha256 |
|---|---|---|---|
| census sentence (1) (P419) | 231 | 1 | `3dc376905dc8910794c8ce0d918f8a42615d401200655affa5707c165d8e1f93` |
| census sentence (2) (P418) | 620 | 1 | `efd2fe581e6fdeb930b52c7a7bcd64557696ea716ab3cb8b0a6556902bdd4dcd` |
| P420 D2 session sentence | 113 | 3 | `ac5846cefaacda2bf1220b3d1f042ee64d9b8cd2426e1d1cdda269b0d2b38ec7` |
| P424 session-facing decline sentence | 154 | 3 | `86c1ba6527baf43686c931c6c29313e31988f19e376ba89f6af20c06bdaa06f5` |
| P436 frame line | 52 | 14 | `e8fc06076aa60cdbca7f246277e0efacde6bb7776df98799f14bb19aeb3f5425` |

Before the edit the four protected strings had counts 1 / 1 / 3 / 3 — **identical after**: census (1) and (2) once each; D2 three times; P424 three times. The P424 sentence is followed by the new line as a SEPARATE sentence; it is not altered.
The superseded string "Josh has agreed to this." occurs **twice**, both as superseded wording: this edit's changelog bullet and the §10 ruling-map row "435 ITEM 1 + 436".

## Where the line sits
"Josh has agreed that your decline will be respected." sits in the session-facing **ARMING frame "after:" text**, right after "Your decline is reported to Josh." and before "If you give no answer, the change is not made." (§4A.6). **Reading, for the Executive to confirm (§10):** the **CHECK-IN frame carries it too**, right after its own decline sentences — the brief said not to add it there unless the plan's own logic requires it; it does: G1 state (a) RATIFIED is defined over "the frames" and the promise "This org will not re-arm it over your decline" is made in both, so leaving the ratification out of the check-in frame would again promise on Josh's behalf there. If the Executive wants it in the arming frame only, it is a one-line removal.

## Where the lean sits (operator-only)
Josh's words are recorded verbatim ONLY in **§7, "Josh's ratification of the consent frame"** (line 785 of plan-005 after this edit); the quote occurs once in the file. The frames contain none of "lean", "should be on", "strong arguments", "the calls", "your mind", "Nothing will get deleted" (checked by search over the frames text). Guards: §7 (GUARD), §4A.6 (**No lean**), and pair check entry §8 item 14.

## Every place the E2 / P424 precondition was lifted
§4A.4 frame gate · §4A.6 state (a) · §4B stop list · §4B arming order · §7 precondition (e) (and the "Two more" heading) · §9 arming-order clause · §10 open list and ruling row · §0 item 9 · TOP NOTICE ruling paragraph · top ruling map. **Unchanged, every other arming precondition:** (f) no recorded decline stands; #824 landed and verified; #825 persistence check; the S4 Tonic check (§4B); PG-1 artifact + acceptance and merge; the dry run inside the N5 band; the UNQUALIFIED non-decline standard and the consent protocol; the named pre-arming backup; Josh's go for ARMING; the consent-record commit cited with "no decline standing"; the NO-branch wording stays defined; a later revocation of the ratification still blocks.

## Flag — does the REWORDED line pressure the CC's own free call? (Exec/pair to confirm; I did not edit the Executive's words)
Line analysed: **"Josh has agreed that your decline will be respected."**, placed immediately after "Your decline is reported to Josh."
- **Better than the old line.** The old "this" could be read cold as "Josh approves of competition" (the Executive's own reason for P436). The reworded line scopes Josh's agreement to ONE thing — that a decline will be respected — and says nothing about his view of the change. It lowers the cost of declining; it does not add a reason to accept.
- **Residual, small:** (1) **authority cue** — naming Josh (the architect) inside the ask puts his weight behind the org's promise; here that weight backs the decline, so it leans toward a free call, not toward "yes". (2) **asymmetry** — the ask says what happens if the session declines but nothing parallel for continuing; the adjacent "If you give no answer, the change is not made" and "an answer with conditions or concerns … goes to Josh" keep silence and hesitation non-consenting, so I do not read it as tilting. (3) **adjacency to "reported to Josh"** — a session told its decline goes to Josh might infer Josh is watching; the new line tells it the reporting is paired with a commitment to respect the decline. (4) **truth** — supported by Josh's own words ("Nothing will get deleted without your approval"), which are operator-side only; his lean toward ON is NOT shown and must not be inferred from the line. (5) **check-in frame** — "respected" is true for the REMAINING cycles only; the frame's own preceding sentences say links already removed are not restored, so it is not misleading.
- **Still open?** I judge it low-pressure and would not block on it; it stays **open for the Executive/pair to confirm in one line** (§10), because it is the Executive's wording and the test is how the asked session reads it, which I cannot run.

## Second flag — the ratification is an inference
Josh's words are a lean plus "the individual calls … to you" plus "Nothing will get deleted without your approval"; they are not literally "I ratify 'no re-arm over a CC decline'". The plan records them as the Executive's reading (Exec Packet 435 ITEM 1). Whether they meet G1's standard ("explicit, unqualified, the same as the CC's non-decline") is for the Executive/pair (§10).

## What I did NOT verify
- Exec Packets 435 and 436 as primary documents (quoted as supplied in the dispatch and the brief; the packet date "2026-10-01" is not reconciled with the real clock).
- That a CC session reads the frame as intended (S4 has not started; nothing shown to any session).
- Nothing else in plan-005 was re-verified by this delta; all earlier "unverified" items stand (§8).
