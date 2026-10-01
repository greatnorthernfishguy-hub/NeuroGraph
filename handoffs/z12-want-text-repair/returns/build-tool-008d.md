```
---- Changelog ----
[2026-10-01] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #12341, RETRY of the 12 ANCHOR_FAILED extraction) - build-tool-008d: the extraction ran
             ONCE and wrote the one private off-repo file (mode 0600). build-tool-008c.md (the not-run record of #12322) is untouched. This return
             carries ids, counts, lengths, offsets, hashes and booleans ONLY: no want text, no excerpt. Nothing fixed, guessed, re-tagged, edited or deleted (H-1).
-------------------
```

# build-tool-008d - the 12 ANCHOR_FAILED extraction (retry of #12322)

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #12341 (same-thread resume, Exec P454) - tool branch `cc-laptop-want-repair-tool-20260930`. Related: [[NeuroGraph]] - [[The Laws]]

## 0. Result

The one private file exists: **`/home/josh/backups/z12-want-text-repair-20261001T011502Z/review/anchor-failed-12-20261001T020218Z.md`**, sha256 **`8a7a9ec0d3dc3ee00c5e98dfdc6c1777d916a99f1a7729725d5756c40ad4780a`**, **20,539 bytes, mode 0600** (`stat` and `sha256sum` re-read from disk after the run; owner josh). It contains conversation text and is never pushed or pasted. **12 ids found (count check passed; the script STOPs if the count is not 12), all with the tool's re-derived outcome `ANCHOR_FAILED` and a `detail` equal to the report's (`text_count=2`).** The Executive decides each of the 12; I decided nothing.

## 1. Pre-flight (printed first) and what ran

| Time (UTC) | load1 | MemAvailable | SwapFree | `cc-ng-daemon.service` | daemon process | Result |
|---|---|---|---|---|---|---|
| (dispatch gate) 02:00:18 / 02:01:18 | 4.04 / 4.57 | 8,773,516 / 9,019,960 kB | - | inactive | none | quoted; other-worker-turns 1 |
| **02:01:44 (immediately before the launch)** | **5.20** | **8,772,728 kB (8.37 GiB)** | 9,122,156 kB | inactive | 0 | **PASS** (load < 6, MemAvailable >= 3 GiB, daemon inactive, no daemon process) |

One reading, no retry. **Command:** `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 systemd-run --user --scope -q -p MemoryMax=1G -p MemorySwapMax=0 python3 -B anchor_failed_extract.py` (script in the feasibility directory, **re-used unchanged: sha256 `bb659a33c7c8623dec8421e27ed8764172bae75d9b9be33296387afb3a127e10`**, the same as in build-tool-008c and recorded again by the script itself). rc 0, stderr empty (0 bytes), no OOM (the scope is gone, no event; `ru_maxrss` **497,508 kB = 0.474 GiB**, inside the 1 GiB cap). stdout carried only the JSON of ids, counts, hashes and booleans (P1/P379 header lines plus the result); captured to `anchor_failed_stdout.txt` sha256 `342cfa40837f4e6b0965c2f91c354b90736695cd42c06c66b7b12e36161aaae0`, stderr `e3b0c442...` (empty).

**Tool and input:** tool worktree HEAD `1f40fee3cb18049b17eb3a9d315c91dd1d50159b` (clean, 0 porcelain lines before and after); tool sha256 **`bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c`** (the tool the script imported read-only, recorded in its output); pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`, 0 porcelain lines (`--ignored`) after. The input is the run directory's verified copy: the script re-hashed all six files of `.../z12-want-text-repair-20261001T011502Z/copy/` (size and a full sha256 stream) against `copy-hashes.json` before reading and would STOP (exit 3) if any differed; it ran to completion, so **all six matched**. It used the tool's own `stream_graph_nodes`, `load_content_subset`, `load_pinned`, and `Classifier.classify_node` (re-deriving each record), `text_class` and `sha16`; the occurrence offsets are `str.find` over the same content strings the classifier saw (the tool's own `str.count` and an overlapping scan agree: 2 and 2 for every id).

## 2. Per-id numeric facts (no text). Offsets are 0-based character offsets of the want text in the source content

"Gap" = (second offset) - (first offset + want-text length); negative would mean overlap. "Context equal" = the 160 characters before / after the two occurrences compare equal.

| # | want node id | `source_node` id | class | want len | sha16 | source len | count (tool / overlap scan) | offsets | gap | adjacent (gap 0) | overlapping | before equal | after equal | identical both sides |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `cc:want::0849cc60e7e3f2f8` | `cc:conv::2b6ad17239bd0106473dd76baac874277e1ac22d` | A | 7,125 | 2850c8f0da245d7a | 128,729 | 2 / 2 | 7,404 ; 25,444 | 10,915 | no | no | yes | yes | yes |
| 2 | `cc:want::084c1b207df9b66a` | `cc:conv::b53acce9e8e0ff9b4fc43fe906b844b7a7cdd8e1` | A | 1,297 | 341234d113cba527 | 132,929 | 2 / 2 | 91,288 ; 100,868 | 8,283 | no | no | yes | yes | yes |
| 3 | `cc:want::1e84b21f862eb26a` | `cc:conv::91599a1552762ffd4c908ccfde3d33cc8a240cae` | A | 804 | 253cb3629743309e | 148,829 | 2 / 2 | 28,655 ; 53,181 | 23,722 | no | no | yes | yes | yes |
| 4 | `cc:want::32a2df6e49c81269` | `cc:conv::aaad5cea633fad3714786868f5634c6a7cc7bacd` | A | 664 | b199d0d6a26cc876 | 122,029 | 2 / 2 | 53,198 ; 108,533 | 54,671 | no | no | yes | yes | yes |
| 5 | `cc:want::6efce0eb6426ce7a` | `cc:conv::803865fed2924e91135b2befb1003c0fa15bff77` | C | 1,816 | e0eedda46ffd8241 | 148,724 | 2 / 2 | 111,186 ; 123,357 | 10,355 | no | no | yes | yes | yes |
| 6 | `cc:want::6f2de47a34887bda` | `cc:conv::b53acce9e8e0ff9b4fc43fe906b844b7a7cdd8e1` | C | 1,373 | 6f0c7f039fb69586 | 132,929 | 2 / 2 | 7,513 ; 17,780 | 8,894 | no | no | yes | yes | yes |
| 7 | `cc:want::819617483e25527f` | `cc:conv::7dc1e5f78806ef82b638392c9e360d3325acacd2` | A | 642 | 739a0508dc5cb0c0 | 144,048 | 2 / 2 | 72,151 ; 92,011 | 19,218 | no | no | yes | yes | yes |
| 8 | `cc:want::8e9856140b42e65f` | `cc:conv::eacc01a54d9bc6c794dc08e22f7099bb0307c5b7` | C | 1,085 | dd8fd62ee3c094bf | 144,009 | 2 / 2 | 123,690 ; 127,247 | 2,472 | no | no | yes | yes | yes |
| 9 | `cc:want::a8dad5b120a4f2b3` | `cc:conv::faae35d39f277fe30a2dcc0c2951e8b7103fded0` | C | 1,521 | bfbacb1a8eeeb386 | 134,353 | 2 / 2 | 64,744 ; 68,744 | 2,479 | no | no | yes | yes | yes |
| 10 | `cc:want::b6c8bf7af61eef63` | `cc:conv::803865fed2924e91135b2befb1003c0fa15bff77` | C | 884 | a22435206b6b43d9 | 148,724 | 2 / 2 | 60,325 ; 70,112 | 8,903 | no | no | yes | yes | yes |
| 11 | `cc:want::df4501e36d13910e` | `cc:conv::2b6ad17239bd0106473dd76baac874277e1ac22d` | A | 1,651 | 2e72c4fc25806dd4 | 128,729 | 2 / 2 | 3,606 ; 21,646 | 16,389 | no | no | yes | yes | yes |
| 12 | `cc:want::ed9162ca7c6df703` | `cc:conv::2b6ad17239bd0106473dd76baac874277e1ac22d` | A | 619 | 9856f800086a0c1a | 128,729 | 2 / 2 | 5,738 ; 23,778 | 17,421 | no | no | yes | yes | yes |

**Tallies (from the table):** 12 wants over **8 distinct source nodes** (`2b6ad172...` carries 3 of the wants, `b53acce9...` and `803865fe...` carry 2 each, the other five sources one each); class **A 7, B 0, C 5**; every `detail` equals the report's `text_count=2`; **every want text occurs exactly twice** (tool count 2, overlapping scan 2); **no pair is adjacent, none overlaps**; the gaps between the two occurrences range 2,472 to 54,671 characters; **for all 12, the 160 characters before AND after the first occurrence equal those of the second** (so the bounded windows alone do not distinguish the two occurrences; the offsets do). Want-text lengths 619 to 7,125; source-content lengths 122,029 to 148,829. I draw no conclusion from these facts.

## 3. What the private file contains per id (described, not reproduced)

The id; `source_node`; class and `detail`; want-text length, `sha16`, source-content length; the occurrence count and offsets; the gaps and the adjacency / overlap / context-equality booleans; the first 160 characters of the want text; for each occurrence 160 characters before and 160 after (delimited, JSON-escaped ASCII form); and the verbatim left-list block for that id copied from `review/left-list-20261001T011502Z.md`. **The left-list entries for these 12 carry no `outer opener +/-80` anchor line** (the booleans `left_list_has_outer_opener_anchor` are all false; every `left_list_block_present` is true): the tool records that anchor only after the want text anchors uniquely, and these 12 stop at the uniqueness check, so each verbatim block is 3 lines long (checked by line count, none starting with `outer opener`; the content of those lines is not reproduced here). The window is 160 characters as specified.

## 4. Source and copy unchanged; nothing else touched

- **Run-directory copy:** sizes equal what the tool recorded, inodes differ from the source's, and the recorded copy sha256s equal the kept COPY's manifest for all six files; verified (sha256 stream) by the script before any read. **Classify artifacts untouched:** the 8 reports and the two classify review files still have the sha256s recorded in build-tool-008 (`ec999a1f...`, `181515c2...`, `7dd13d2a...`, `0c7bf9ea...`, `f7f8e3a0...`, `279d2474...`, `a3e26631...`, `270af25c...`, left-list `59aacdf7...`, excerpts `960018c7...`); the new file is the only addition to `review/`.
- **Live CC source** (not opened this turn): size, inode and mtime still equal `copy_manifest.json` and what the tool recorded. **Kept COPY:** size + inode equal the manifest, mtimes `1790793124837064596` (vectors) and `1790793092513541531` (main), unchanged; not read.
- The daemon unit stayed `inactive`. Nothing under Syl's directories, the live tract, `~/.bashrc` or any generations directory; no `--apply`, rewrite, `phase2-backup`, rollback or `--josh-go*`; the only repo change is this return (and build-tool-008c.md is unedited).
- Measurement files, feasibility directory (off-repo): `anchor_failed_12_facts.json` (numeric facts only, sha256 `575d37e31cba77ee0a2b30c6a3e2e2db0a519c0d74b1809bf2c0fadda2bff95b`).

## 5. What I did NOT verify

- That the two occurrences are the "same" want in a semantic sense (they are textually identical in their 160-character neighbourhoods; I did not compare beyond that or characterise why the source repeats them).
- Anything about which of the two occurrences, if either, is the one the want node was minted from; whether these 12 would repair under any rule (that is the Executive's case-by-case decision, and nothing here fixes, guesses, re-tags, edits or deletes any node).
- The other 106 nodes of S (NONE) and everything downstream (rewrite, V11, T6), as ordered.

I have stopped.
