# checker-015 ROLE A (cross-family) — want-hub-competition-d plan REVISION 4

STATUS: INCOMPLETE - review in progress

- Lane: `want-hub-competition-d`
- Dispatch: #10735
- Zone manager: Z12 (session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`)
- Checker: checker-015, grok-4.6, CROSS-FAMILY, `report_only`
- Role: ROLE A only (ADDENDUM 2 items (1)-(7))
- Plan under review: `handoffs/z12-want-hub-d/returns/plan-004.md`
- Plan pin: `618980ea5536232c0255c8cbfc3e9e685e5164fc`
- Plan sha256 (named in packet): `9ff61ac3ba3d24d7d1979b94663897db730110049d2a71bd4032924f9f36f2e8`
- Protected file: `neuro_foundation.py` at `e4ebf982b1989fd9066d610b94853bc68bf70d37` via `git show` only
- Authority: plan review only; no build, no merge, no settle, no dispatch, no primary-checkout edit

## P379 session start

- python: `/usr/bin/python3`
- `sys.path` includes `/home/josh/NeuroGraph` (from `PYTHONPATH`); NG modules were **not** imported
- NG-related in `sys.modules`: NONE
- `NG_EMBED_*`: none set
- Worktree: `/home/josh/NeuroGraph-worktrees/z12-want-hub-d-20260930`
- Branch: `cc-laptop-want-hub-d-20260930` @ `618980ea5536232c0255c8cbfc3e9e685e5164fc`

## ADDENDUM 2 ROLE A items

### (1) every le-012 correction C1-C11 is really applied

(use plan-004's own correction table and verify each row against the text, not the table)

STATUS: IN PROGRESS

### (2) NO contradiction of P399/P404

option (i) only, no predicate copy / removal loop in the orchestrator, additive kw-only default-None surface (`competing_ids`, `excluded_ids`, `max_removals`, static sort key on `to_prune`, removed-ids out-param), competing set passed as ids and NEVER inferred, explicit `raise`, validation before mutation; K and B env-read and passed as CALL ARGUMENTS with NO config key anywhere (`CC_SNN_CONFIG`, `DEFAULT_CONFIG`, `OPENCLAW_SNN_CONFIG` untouched), OFF at every restart by construction

STATUS: IN PROGRESS

### (3) implementability against the real code

read `_prune_synapses` at base `e4ebf982` (`git show e4ebf982:neuro_foundation.py`, read-only, PROTECTED) and say whether the described additive surface can be added with the default path provably unchanged (same return, same side effects) and whether the static sort key + `max_removals` + removed-ids out-param can be realised on the function's own `to_prune` list; the both-callers golden test (Door A `_structural_plasticity` and the Door B tail) and the all-defaults case

STATUS: IN PROGRESS

### (4) the C1 statement of the real `low_weight_steps` mutation is TRUE at `:3524-3529`

STATUS: IN PROGRESS

### (5) numbers

recompute from the derived JSON only where the rev-4 change alters them (the ordering is now a STATIC key: does the K=50+50 / B=5,000 cycle table and the per-cycle guardian-gate table still hold? recompute the worst cycle)

STATUS: IN PROGRESS

### (6) the dream-loop change

own try/except, `last_pass` still updated, unset env => skip + INFO "not armed"

STATUS: IN PROGRESS

### (7) anything that fails if built as written

STATUS: IN PROGRESS

## Verdict (pending)

Overall: PENDING

Numbered corrections: pending

Numbered not-verified: pending
