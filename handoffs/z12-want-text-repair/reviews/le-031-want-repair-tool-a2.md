STATUS: INCOMPLETE - review in progress

# le-031 — law-enforcer TINY RE-LOOK, ADDENDUM 3 (TURN A2 hardening of the 118-want repair one-shot tool)

Seat: le-031 (fresh neurograph-law-enforcer, report_only). Agent file sha256 verified = `6daf1621b844b9b72d567b329f2c9f40ca0b4516744608c125147e601c4acf23` (matches the dispatch).
Tool worktree: `/home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930`, branch `cc-laptop-want-repair-tool-20260930`, head `c099089503b275b39e9d2c93d7b4a08d3ae01194` (pull --rebase: up to date).
Diff: `git diff ab24e85c06a0b524828db895574e7544ca01ed09 e7bd634a70679dffef1eb65bb7d8c4df8490869d` on the tool + test files.
No raw want text in this file. Synthetic data under /tmp only.

## Checks (from ADDENDUM 3) — all PENDING

1. Each of le-029 C1-C8, N1 and checker-026 V18 really fixed (not just tested): C1, C2 (own erroring-Probes cases; tests stub `subprocess.run`), C3, C4, C5 (forge provisional packet), C7/C8, N1, V18 (empty mapping + tampered artifact hash FAIL) — PENDING
2. NEW gated `--step rollback` (second write path into live checkpoint): same gate stack as apply (P1/P4/P6/P7, `--expect-*`, `--josh-go`)? source ONLY the tool's own hash-verified pre-apply backup (never generation dir / `last_good/`)? verifies EVERY sha256 before replacing anything, atomic write, refuses on live identity != post-apply receipt? can it on ANY path alter a protected want / constitutional node / Choice Clause link, or restore over Syl's directories? torn-apply synthetic case end to end — PENDING
3. Inode evidence (Exec P428): st_dev/st_ino/st_nlink of the six files before/after; new inodes + link count 1 asserted; write guard refuses `st_nlink > 1`; generation partners only stat'd + hashed, generations dir NEVER listed/opened (read every os.listdir/scandir/glob/open) — PENDING
4. No regression: TURN A refusals hold; run test file ONCE (P379 preamble; worker reports 153 passed); worker's flags 1-5 and what they mean for TURN B — PENDING
5. LAW 1-8 on the diff (LAW 3 one shared parser fn; tool under `oneshot-tool/`, never merged; LAW 7 raw means complete); no protected/vendored/shared file, no raw want text in diff — PENDING

## Verdict

PENDING
