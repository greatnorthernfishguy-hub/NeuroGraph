# checker-020 ROLE A (cross-family) — want-hub-competition-d plan REVISION 5 (targeted DELTA)

STATUS: INCOMPLETE - review in progress

- Seat: checker-020 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `want-hub-competition-d`. Dispatch #10907. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-d-rev2.md` including ADDENDUM 3 (docs branch `cc-laptop-daemon-recall-756-20260930`, not edited).
- Plan under review: `handoffs/z12-want-hub-d/returns/plan-005.md` at `9c521699725b30c85fbaad6a6f4b76c1b0cbae03` (a89c7b3d plus the in-place [R5b·P415] follow-up).
- Scope: targeted DELTA of REVISION 5, not a full round. Checks are ADDENDUM 3 ROLE A items (1)–(7) below.

## ADDENDUM 3 ROLE A items (in progress)

(1) each le-013 correction L1-L12 and checker-015 C1/C2 is really applied in the TEXT (verify each against the correction list in `reviews/le-013-want-hub-d-rev4.md`, not against plan-005's own table).

(2) the X7 table and section 4A.9: recompute from the derived JSON only (`probe-laptop.json`, `summary-laptop.json`) the cycle-1 MAX-links-lost-by-one-want figure (1,527 for `cc:want::ac4d8c6a7f50852c`, 48%), the cycles-2-10 maxima, the guardian margins (96.4% first cycle; worst 87.1% at cycle 21) under the HEIGHT key; state which you can and cannot reproduce.

(3) L1 pre-merge gate: the two-checkout REAL-graph golden as specified (base vs branch, all-defaults, copies of the laptop checkpoint and the staged VPS bundle, pinned sys.path[0], printed file+rev, identical removed-id hash / return value / full state hash) is executable as written (memory-capped, one load at a time, MemAvailable >= ~8 GB, read-only copies, hashes preserved) and is sequenced BEFORE any merge.

(4) L2: the serialized-bytes comparison (clock/saved_at excluded and LISTED) is well-defined, or the plan states plainly why byte-exact is impractical (never a silent narrowing).

(5) the P415 text: Q-E accepted with no ramp/cap/new parameter anywhere in the plan; both conditions are reporting triggers, not tunables (no env knob, no constant); the arming census sentence is a template using the dry run's recomputed figures; a decline stops arming and goes to Josh.

(6) L7/X8: ruled text present as ARMING preconditions incl. the verbatim CC-session response, 'pruned links are NOT restored: the plan names the pre-arming backup', the revocation path, the primary Packet 392 citation (id/ledger abd57423/sha256 f4bca117bdcd034c).

(7) anything that contradicts P399/P404/P409/P412/P415 or fails if built as written.

Authority: report_only. No build, no PR, no merge, no settle, no dispatch. Protected file `neuro_foundation.py` read only via `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:neuro_foundation.py`. No graph/checkpoint/tract load. Primary `/home/josh/NeuroGraph` not edited.
