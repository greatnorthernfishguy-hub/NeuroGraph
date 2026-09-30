# checker-021 — ROLE A short re-check of plan-005 after [R5c] fold

STATUS: INCOMPLETE - review in progress

Lane: `want-hub-competition-d` (dispatch #10996, session 52d39aba-db92-4bf2-b3b1-0e4c13f77d8c). Zone manager Z12. `report_only`. Cross-family ROLE A (fresh instance). Scoped to the CHANGED sections; not a full round.

Plan under review: `handoffs/z12-want-hub-d/returns/plan-005.md` at commit `1800d28ad6f82f8f503e45ef787041b52cdf4df1` (fold `25ae7a2` + pin-line `1800d28`). Inputs: `reviews/le-018-want-hub-d-rev5.md` (N1-N8), `reviews/checker-020-want-hub-d-rev5.md` (N1/N2), Exec P418 via `assignments/plan-want-hub-d-p418.md`.

ADDENDUM 4 ROLE A items:

(1) N1/#824: the automatic-refusal text no longer promises an untruthful signal (read the daemon `_guarded_save` at `039a3bf4` `:970-1011` yourself, `git show` from the docs branch/commit, and confirm the plan's description of BOTH refusal paths is accurate) and states plainly what the operator-level check does until #824 lands; #824 an ARMING precondition.

(2) N2/N3: BOTH census sentences appear VERBATIM (compare byte-for-byte with `assignments/plan-want-hub-d-p418.md` item 4 and the P415 sentence in `assignments/plan-want-hub-d-p415.md`), the consent protocol (explicit UNQUALIFIED non-decline only; qualified/silence/timeout -> Josh; verbatim record of text + response; an operator/Exec act, never a daemon->session relay; a stop request = a decline).

(3) N4: PG-1 artifact + named accepter recorded BEFORE Josh's merge ask + each load in its own fresh process + the >= ~8 GB gate + one load at a time.

(4) N5: the pre-accepted BAND stated exactly (single-want loss <= 55%/cycle; guardian margin >= +30 pts at every save; end-state p50/p90/max within +/-5 pts of -82/-90/-94%) as REPORTING thresholds only, in tooling, removed from the 4A.4 operator stop list.

(5) #825: the arming precondition proving on a COPY that pruned links stay pruned across SAVE AND RESTORE.

(6) the reporting column, the tie-break, the telemetry note, the pin.

(7) recompute the end-state per-want figures (p50/p90/max) from the derived JSON if you can and say whether -82/-90/-94% and the 1,662 figure reproduce; anything that contradicts a ruling.

Reviewer: checker-021 (grok-4.6). Worktree: `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930`. Branch: `cc-laptop-want-hub-d-20260930`. Protected file: read-only via `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:neuro_foundation.py`. Daemon: `git -C /home/josh/docs show 039a3bf4:scripts/cc-ng-daemon.py`. No graph or checkpoint load. Numbers only from the derived JSON named in the packet.
