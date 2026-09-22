# E-T Ecosystem PUNCH LIST — moved

**The punchlist is no longer kept in this repo. The canonical record lives in the docs vault:**

### → `/home/josh/docs/punchlist/PUNCHLIST.md`

It is a small **index** plus per-domain shards — open `open/<domain>.md` for live work,
`done/<domain>.md` for completed items, `reference/doctrine.md` for principles, and
`journal/` for session narrative. Add a new item by appending a row to the matching
`open/<domain>.md` table, never to the index.

---

## Why this file is a stub

This repo carried a 578-line single-file `PUNCHLIST.md` that was a **pre-shard ancestor** of
that index. The master record was restructured on 2026-09-03 (closes #319) because the single
file had grown to 503 KB and could no longer be read in full; the copy here was never updated
to point at the new home, so the two records drifted in two repos with no cross-reference in
either direction. Its own header also claimed `Last updated: 2026-05-25` while the file was in
fact last committed 2026-08-13.

Reconciled 2026-09-22. Every item in the old file was checked by title text against the whole
`punchlist/` tree. All but six were already present there. The six that existed only here were
recovered into the shards with their numbers preserved verbatim:

| # | Item | Recovered to |
|---|---|---|
| 38 | TrollGuard SKILL.md fix (MOOT) | `punchlist/done/trollguard.md` |
| 40 | TrollGuard TypeScript hook (MOOT) | `punchlist/done/trollguard.md` |
| 41 | Elmer PRD section 6 rewrite (DONE — living docs) | `punchlist/done/triad.md` |
| 42 | Claude Code + NeuroGraph own instance (PARKED) | `punchlist/open/future.md` |
| 103 | Obsidian wiki-linking across vault (OPEN) | `punchlist/open/docs-housekeeping.md` |
| 103 | Origin Story Documentation (OPEN) | `punchlist/open/docs-housekeeping.md` |

## Nothing was lost — the old content is preserved twice

1. **Verbatim in the docs vault:** `/home/josh/docs/punchlist/ARCHIVE-NeuroGraph-PUNCHLIST_2026-08-13.md`
2. **In this repo's git history:** `git show b9b6016:PUNCHLIST.md`

To restore the old file in place: `git show b9b6016:PUNCHLIST.md > PUNCHLIST.md`

> ⚠ **Needs Josh's ratification.** Replacing this file with a stub was a judgement call made
> during the 2026-09-22 consolidation pass, tracked as **#457** in
> `punchlist/open/docs-housekeeping.md`. It is reversible with the one command above. The open
> question recorded there: should the other module repos ([[TID]], [[TrollGuard]], [[Elmer]],
> [[Immunis]], …) also be swept for orphaned pre-shard punchlist copies and stubbed the same
> way? That sweep has **not** been done.
