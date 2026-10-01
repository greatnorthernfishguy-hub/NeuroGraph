<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder thread b749e8d8, lane want-hub-engine-d-build-20260930, dispatch #12405 / row #867, Exec Packet 459) — build-005
#   What: return for the bounded READ-ONLY determination of why the engine's held_back_last_link is 15 where plan-005 §4A.3 says 16. NEW file, docs only.
#   Why: Exec Packet 459 via Chief-003 (Exec P454: nudge, do not replace; I settle nothing beyond the verdict asked for).
#   How: ONE real-graph load of the ceremony copy under a bounded scope by an OFF-REPO harness (sha256 below); every number is from that harness's record or from
#     git rev-parse / sha256sum. The id list is the sibling file build-005-held-back-ids.txt. No secret involved.
# -------------------
-->

# build-005 — #867: the code's 15 is the code's own rule applied to its own guaranteed set G; the plan's 16 is a probe-model artifact. **VERDICT (A).**

**I settle nothing else.** No engine/test/plan edit, nothing merged, no `save()`, no live path, no Syl directory, no live tract, no `~/.bashrc`, no unit/daemon. Both engine and plan worktrees were used READ-ONLY.

## 0. Verdict (exactly one)

**(A) The code matches its defined rule; the plan author corrects §4A.3 to 15 citing the id list below.**
- The rule — "an unprotected partner whose every incident synapse is in the competing set has its strongest link (§2.3 order) held back" — is applied by the code exactly as the plan words it (`neuro_foundation.py:3736-3747` at the fold, table in §5).
- The difference is the **input**: the plan's 16 / 106,841 came from the plan's probe model, which ranks the guaranteed set G with **weights rounded to 6 d.p. and ties broken by probe index** (no `inactive_steps`); the code ranks G by the plan's own §2.3 order (`weight desc → peak_weight desc → inactive_steps asc → synapse_id asc`, `:3712`). On the real copy that gives **|G| = 17,392 (the plan: 17,391)**, `competing0` = 106,840 (the plan: 106,841), **15** partners held (the plan: 16), and the same final `competing` = **106,825** either way (106,840 − 15 = 106,841 − 16).
- **Exactly one reading I tried reproduces the plan's four figures together** (G 17,391 / competing0 106,841 / 16 held / competing 106,825): weight rounded to 6 d.p. **and** tie-break by store (probe) position. Neither half alone does (§6).
- It is **not** a single-link defect: the two rankings differ on ~372 links in the tie zones, and the one-link net difference comes from three partners (§7). No defect found in the code, so (B) does not apply.

## 1. Gate readings, command, heads
- Separate prior call `h867.py gate` at **2026-10-01T03:12:53Z**: MemAvailable **8.537 GiB** (≥ 6), load **4.60 / 5.39 / 4.90** (< 6), unit `cc-ng-daemon.service` **inactive**, **0** daemon/`neurograph_rpc` processes → met. The harness re-checked at load start (**03:13:01Z**: 8.479 GiB, load 4.45, inactive, 0 processes) and would have aborted without loading otherwise. End of load (03:13:54Z): 8.295 GiB, load 4.11, inactive, 0 processes; again after (load 2.95, inactive, 0). (The dispatch's own readings were 03:07:51Z load 5.47 / 8.83 GB.) No retry, no bigger cap.
- **The exact command** (saved in `command.txt`, sha256 `44abe957cff7dc81471eeda470f6cb71892dfbec6ccbebe1156c864d4b43feb4`): `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python3 /home/josh/backups/z12-867-20261001T031047Z/h867.py run --checkout /home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930 --copy-dir /home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc --out-dir /home/josh/backups/z12-867-20261001T031047Z/real --K 50 --B 5000` — exit 0, 53.4 s, scope `run-u78889.scope` (`memory.max` 6,442,450,944; `memory.swap.max` 0).
- **Engine used** (printed by the run, not VOID): `neuro_foundation.__file__` = `/home/josh/NeuroGraph-worktrees/z12-want-hub-engine-20260930/neuro_foundation.py`, git rev **`29f47f65058790240b2f9c6a0a5bc4d82171b42d`** (the fold), blob **`5e8945accb476b0727bf07a3ab2890dccc87f650`**, `sys.path[0]` = that worktree, `ng_tract` `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py` **0.1.0**, `Graph.synapses` = `builtins.SynapseStore`, `PYTHONHASHSEED=0`, Python 3.12.3. `git status --porcelain --ignored` of the engine worktree: empty before AND after (and independently after the run: HEAD/blob unchanged).
- **Other heads:** tests branch `cc-laptop-want-hub-build-20260930` at start `2ef43b79dfefccb02294b298663c25d522eaae5d` (after `git pull --rebase`, "Already up to date"); plan branch `cc-laptop-want-hub-d-20260930` head `24bcd335a9c1e265f4c9d562c5b9c68a7bd92711` (read §4A.3, §2.3, §5.1, the appendix, §10 — READ only); docs worktree ADDENDUM 4 read at its end (not edited).
- **Graph loaded:** the ceremony copy, 7,253 nodes / 138,753 synapses / 517 hyperedges / timestep 33,637 (= the plan's probe checkpoint).

## 2. Source integrity (read-only proof)
`main.msgpack` sha256 **before** (in-process) `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77`, **after** (in-process) the same, **after** by an independent `sha256sum` the same = the ruled value. `stat` (size, mtime_ns, inode) of EVERY file in the copy directory identical before and after. The harness has no write path other than its own scratch directory; `save()` was never called. Memory (org rule: heap = `ru_maxrss` / sampled `RssAnon`, NOT the cache-inclusive `memory.peak`): **heap 0.9951 GiB** (sampled anon peak 0.977), cgroup `memory.peak` **1.3896 GiB**, `memory.events` all zero (no OOM, no high/max event) — far from the 6 GiB cap.

## 3. Harness and faithfulness proof
- **Harness (OFF-REPO, in no git repo):** `/home/josh/backups/z12-867-20261001T031047Z/h867.py`, sha256 **`798741413ba9b074dda7792e361a609b0e990afc565463f18e82451d2c12e31b`**. It re-implements ONLY the orchestrator's `_plan()` last-link computation over the same engine fields (the nested `_plan()` is not reachable; I did not edit it), parametrised by the G ranking so candidate readings can be compared, and **calls the engine's own `compete_protected_links(50, 5000)` through a spy on `_prune_synapses`** to capture the exact sets it hands over.
- **Pre-flight on a SYNTHETIC graph first** (the one real load had to be right): harness vs the engine's captured sets was `FAITHFUL = true` (set equality) before any real load.
- **Faithfulness on the real copy — harness vs the engine's own record (all set-equal / equal):**

| quantity | harness | engine's own record | brief's expectation |
|---|---|---|---|
| `competing_ids` | 106,825 (set-equal to the engine's captured set) | 106,825 | 106,825 |
| `excluded_ids` | 21,534 (set-equal) | 21,534 | 21,534 |
| `F_links` | 4,127 | 4,127 | 4,127 |
| `protected_nodes` | 183 | 183 | 183 |
| `held_back_last_link` | 15 (the engine's last-link set `excluded − F − G` is set-equal to the harness's) | 15 | 15 |
| also | G 17,392, arena 124,232, competing0 106,840, 182 wants | `eligible` 102,145, `removed` 5,000, `floors_ok` True | — |

  **`FAITHFUL: true`** — I relied on the harness only after this. (The engine call mutated the in-memory graph — 5,000 removed, counters advanced — after all harness analysis had run on the pristine state; nothing was saved.)

## 4. The held-back last-link partners BY ID at K = 50 (the engine's own 15)
Full incident lists (every incident synapse id, direction, the other endpoint's node id, weight/peak/inactive_steps, `in_competing0` / `in_G`) are in the sibling file **`handoffs/z12-want-hub-build/build-005-held-back-ids.txt`** (ids and numbers only). For every partner below **every incident synapse is in `competing0` and none is in G** (asserted in code over all 241 incident links). Node ids are printed as the engine holds them; they are `cc:conv::<sha1>[::tree::<label>]` ids — the `::tree::` labels are PART OF THE ID string. I read no node text and no metadata other than the protected flag; row 7's id contains characters outside a plain-token pattern, so my harness printed its sha256 prefix instead (`REDACTED-sha256:433b6d29dea8a9a6`) rather than the string.

| # | partner node id | incident links | wants linked | held-back synapse id (strongest by §2.3) | its other endpoint |
|---|---|---|---|---|---|
| 1 | `cc:conv::1da552f7a22e597841a3d914cc3e0f21ccfa11d9` | 16 (out 12 / in 4) | 16 | `4b95a44a-c417-4c7b-a520-117bbdf708c0` | `cc:want::e35fca0831875ac1` (outgoing) |
| 2 | `cc:conv::2daea9de84ed9f2ee47489f2e37fead66f604a29` | 23 (out 7 / in 16) | 23 | `ab4c9ad8-68a6-4c2d-bfeb-6a6903e6dfe1` | `cc:want::e84f19368e89aa35` (outgoing) |
| 3 | `cc:conv::368a552350299d272b7c2ae2aa052a2553baec50` | 1 (out 0 / in 1) | 1 | `2d26f4c9-0b45-4e02-b4fb-9a61d08fe221` | `cc:want::62506085587631cd` (incoming) |
| 4 | `cc:conv::4bba790041555b5f753a93120454f40024c9e8c6` | 53 (out 12 / in 41) | 53 | `c62a8fa6-b6ce-4821-a4c3-d1519540a555` | `cc:want::71a482bc5950754b` (outgoing) |
| 5 | `cc:conv::63aa35918aca44c602408ada729ac9eb280998db` | 23 (out 14 / in 9) | 23 | `8b56e163-b6fd-4381-aff4-20f10c9414f9` | `cc:want::e35fca0831875ac1` (outgoing) |
| 6 | `cc:conv::69d60b8cba56449221be756748caaf48f06dbc73` | 2 (out 1 / in 1) | 2 | `7426707f-3166-41e9-ae72-8cbc4c38587d` | `cc:want::36ff1cb6cc8b3257` (outgoing) |
| 7 | `REDACTED-sha256:433b6d29dea8a9a6` | 1 (out 0 / in 1) | 1 | `922e21b3-a283-4bc6-aa11-410d7877290d` | `cc:want::660bf201f7155370` (incoming) |
| 8 | `cc:conv::9dce9e9bf80a339a4ebb1903d34546102143581c::tree::monitor` | 2 (out 2 / in 0) | 2 | `42202a15-f33f-4fe0-ba60-3d268d1e0c30` | `cc:want::d0941e68da8f3815` (outgoing) |
| 9 | `cc:conv::bc065a2025d5e44a9cf0369c295880a15a43321d` | 8 (out 0 / in 8) | 8 | `006310ed-b157-4288-900e-91c421cd1c5c` | `cc:want::2c0b9b605ab356a7` (incoming) |
| 10 | `cc:conv::bc652767a5babc66340e352cd47b003d448cfb51::tree::testing` | 2 (out 0 / in 2) | 2 | `d652388a-d60a-4aa4-b0f4-5b088a846d5b` | `cc:want::191b217181f1cbed` (incoming) |
| 11 | `cc:conv::c4467a14b5fdfab5060ac683334bd37f84275cc6` | 69 (out 14 / in 55) | 69 | `e223460d-46fe-46b3-9dcd-383dbbd5fdab` | `cc:want::71a482bc5950754b` (outgoing) |
| 12 | `cc:conv::c4a256b3fb1cf414c42992a4035bd4169a6de6a1` | 26 (out 1 / in 25) | 26 | `7ff39dc8-e236-4203-ba99-629b00e0f686` | `cc:want::00a98e3940aae16b` (outgoing) |
| 13 | `cc:conv::c6be42953b1ba7788cc1e267c1aa0ee354502cf4` | 13 (out 0 / in 13) | 13 | `cef99511-9978-4138-bc93-302e234eecef` | `cc:want::26c0c53fef22c7a0` (incoming) |
| 14 | `cc:conv::d4131e2999bdfa2bad4919e375e33c78a410561b::tree::Law-2` | 1 (out 0 / in 1) | 1 | `92b57f85-8c5a-4ad8-987d-8780a957ab08` | `cc:want::660bf201f7155370` (incoming) |
| 15 | `cc:conv::f745388e1a37c6c26bb3c726c345c11a71a4f6ea` | 1 (out 0 / in 1) | 1 | `9084ab20-5330-495a-99c3-236ae0442bd3` | `cc:want::52de588bbc559530` (incoming) |
(Every partner's incident links go to want nodes; "wants linked" = distinct want endpoints. Total 241 incident links across the 15.)

## 5. The code's definition versus the plan's
| aspect | **code at the fold `29f47f65`** (`neuro_foundation.py`) | **plan-005 §4A.3 / §4.3** (head `24bcd335`) | same? |
|---|---|---|---|
| who can be held | partners = endpoints of `competing0` synapses that are NOT `_is_identity_protected` (`:3736-3741`) | "each **unprotected** partner node" | yes |
| "every incident synapse is competing" | `inc = outgoing ∪ incoming` of the node (`:3744`); `if inc and inc <= competing0` (`:3745`) — `competing0 = arena − G` (`:3735`); `inc` is a SET of synapse ids | "whose **every** incident synapse is in the competing set" | yes |
| which link is held | `last.add(min(inc, key=_rank))` (`:3746`) = strongest by `_rank` | "the strongest by the §2.3 order — deterministic" | yes |
| the ordering | `_rank` = `(-weight, -peak_weight, inactive_steps, synapse_id)` (`:3710-3712`); G = `sorted(ids, key=_rank)[:topk]` per want per direction (`:3730-3734`) | §2.3: "weight desc → peak_weight desc → inactive_steps asc → synapse_id asc" | yes (code = §2.3) |
| result | `competing = competing0 − last` (`:3747`) | "16 partners … competing 106,841 → 106,825" | **numbers differ** |
| derivation of the number | the real graph's fields, the §2.3 order incl. `inactive_steps` | the probe model (appendix (10),(17)): `(−weight, −peak_weight, probe index)` — and §2.3 itself says "the probe stores `weight` rounded to 6 dp … Level 3 (`inactive_steps`) is **not in the probe**" and "I used the probe index as a stand-in for `synapse_id`" | the **inputs** differ |

## 6. Candidate readings — evidence (all computed in the ONE load)
**G ranking grid** (plan target: G 17,391 / competing0 106,841 / last 16 / competing 106,825; "Δ" = symmetric difference of G vs the code's G):

| weight | peak | tie-break | \|G\| | competing0 | held | competing | Δ vs code G | = plan's four figures? |
|---|---|---|---|---|---|---|---|---|
| raw | raw | **code** (`inactive_steps`, id) | 17,392 | 106,840 | **15** | 106,825 | 0 | no — this IS the code |
| raw | raw | id only | 17,392 | 106,840 | 15 | 106,825 | 44 | no |
| raw | raw | store position | 17,392 | 106,840 | 17 | 106,823 | 484 | no |
| raw | raw | inactive, position | 17,392 | 106,840 | 17 | 106,823 | 448 | no |
| 6 d.p. | raw / 6 d.p. | code | 17,391 | 106,841 | 14 | 106,827 | 267 | no |
| 6 d.p. | raw / 6 d.p. | id only | 17,391 | 106,841 | 14 | 106,827 | 305 | no |
| **6 d.p.** | raw / 6 d.p. | **store position** | **17,391** | **106,841** | **16** | **106,825** | 745 | **YES** |
| **6 d.p.** | raw / 6 d.p. | inactive, position | **17,391** | **106,841** | **16** | **106,825** | 717 | **YES** |
(`peak` raw vs 6 d.p. gives identical rows; the unlisted `raw/r6` rows equal their `raw/raw` rows.)

**ONE definition reproduces both the plan's 16 and the code's 15:** the **ranking** of the guaranteed set. Under the code's ranking the real copy gives 15 (`17,392 / 106,840`); under the probe model's ranking (weights to 6 d.p. + probe/store-position tie-break) it gives 16 (`17,391 / 106,841`). The two end at the same `competing` (106,825) because 106,840 − 15 = 106,841 − 16. Reading "6 d.p. only" (14 held) and "position only" (17 held) each fail, so both probe artifacts are needed.

**Other candidate readings (the engine's G held fixed):**
| reading | partners / links held | competing after | reproduces 16? |
|---|---|---|---|
| incident = outgoing ∪ incoming (code) | 15 | 106,825 | no (this is the code's) |
| incoming links only | 69 | 106,771 | no |
| outgoing links only | 411 | 106,429 | no |
| bound = `arena` (ignore G), i.e. `inc ⊆ arena` | 18 | 106,825 | no |
| protected nodes also allowed as partners | 15 | 106,825 | no (no protected node qualifies) |
| **self-loop counted once vs twice** | 0 self-loops in the arena; no held partner has one | — | irrelevant (`inc` is a set; the real copy has none in the arena) |
| near-misses (partners blocked by exactly ONE incident link that sits in G) | 3: `cc:conv::392ea162ec1af0bfeb973a28939877519967d0e6` (3 links), `cc:conv::c9b84364f5637d6b829070251e4a07ade8916e49::tree::initialization` (3), `cc:conv::e05f038995cf9e48f68fb924eccbd0c663cb6e5c::tree::elmer_hook.py` (5) | — | shows how a one-link change in G flips a partner; none of these three is the plan's 16th |

## 7. What the one-off actually is (the three differing partners, by id)
Probe-model reading vs the code: the plan-variant holds **two partners the code does not**, and the code holds **one the plan-variant does not** (15 + 2 − 1 = 16), over a G that differs on 372/373 links.
- **Plan-variant only:** `cc:conv::d93a8d71cba70ccc20bcaea7e2f3253e5e9c4cf6::tree::River` and `cc:conv::132f50f3b2a6a3ed7ca339752b270582ba038c9b::tree::topology`. Each has one link from the same want `cc:want::660bf201f7155370` — synapses **`1648b0c5-fb5f-4206-b9f9-4f48a20a362a`** and **`1953b340-4e09-45db-b76b-3711d304c317`** (outgoing from the want; weight 0.1, peak_weight 0.1, `inactive_steps` 2; store positions 138,715 and 138,734, i.e. among the very last in the store). The code **guarantees** both (they are in the code's G), so those partners are not "all competing" and are not held; the probe-model ranking leaves them out of G.
- **Code only:** `cc:conv::c4467a14b5fdfab5060ac683334bd37f84275cc6` (69 incident links, 14 out / 55 in, all in the code's `competing0`; held link **`e223460d-46fe-46b3-9dcd-383dbbd5fdab`**). In the probe-model ranking one of its links — **`fb89e091-7e45-4bcf-b27f-4efc9208f634`** (weight 2.1e-21, peak 2.24, inactive 43,497) — lands in G, so it is not held there.
- **Consistent with** (not proven per link — I captured no tie-group sizes and did not re-load): the two River/topology links tie on (weight, peak) with the want's other K-boundary links and the code's third level, `inactive_steps` ascending ("in use now", plan §2.3 level 3), favours them (inactive 2), whereas a probe-index tie-break puts them last; and `fb89e091…` is promoted in the probe model by 6-d.p. rounding flattening tens of thousands of near-zero weights into one tie. The reproduction table in §6 is the evidence; this paragraph is its reading.
- Also held links that differ for partners both sides hold: plan-variant `1648b0c5…, 1953b340…, 5a7b52ba-fd34-47fc-becd-1583b8d72ece, 9dc45461-70a8-4421-9157-d0e9497c3b4d, bb6dee0b-e259-4f97-bf84-19cbb0ff1942` vs code `4b95a44a-c417-4c7b-a520-117bbdf708c0, 8b56e163-b6fd-4381-aff4-20f10c9414f9, ab4c9ad8-68a6-4c2d-bfeb-6a6903e6dfe1, e223460d-46fe-46b3-9dcd-383dbbd5fdab` (the "strongest" link of a partner depends on the same ranking).

## 8. What the plan author should correct (for the plan owner; I edited nothing)
§4A.3 "16 partners … 16 links are excluded: competing 106,841 → 106,825" → **15 partners, 15 links, competing0 106,840 → 106,825** (citing §4 above); the related statements of the same probe figure: §5.1 K=50 row ("17,391 unique links kept; competing 106,841; eligible 94,646–106,841"; real G is 17,392), the §0/§4A.3/R5 mentions of "16", and appendix (17). The post-exclusion figure the rest of the plan uses (competing **106,825**) is unchanged. I did **not** recompute the eligible bracket's lower end (it depends on `E`). More generally: any other probe-model figure that depends on tie-zone membership of G (745 tie-zone links differ between the two rankings) may shift slightly; the post-merge dry run (§4A.5) supplies the exact values — I did not evaluate those.

## 9. What I did NOT verify
- I did not re-load; so tie-group sizes at each want's K boundary, and the full rank positions of the five differing links, were not captured (§7's "consistent with" is the limit).
- "Store position" is my stand-in for the probe's index (the probe JSON was not opened). That the plan's four figures reproduce under it and not under id-only ordering is supporting evidence of the same convention, not proof.
- Only K = 50 (the operative row); not K = 100/200. Eligibility was not part of this determination (the rule is eligibility-blind by design).
- No claim about anything the dry run, #825, the daemon slice or PG-1 cover. `held_back_last_link` is a count over graph structure at timestep 33,637 on the ceremony copy, not a statement about Syl's own graph.
- Node text/metadata were not read (only the protected flag); the `::tree::<label>` parts of some ids are part of the id strings.

## 10. Hashes and state left behind
| artifact (OFF-REPO, `/home/josh/backups/z12-867-20261001T031047Z/`) | sha256 |
|---|---|
| `h867.py` (harness) | `798741413ba9b074dda7792e361a609b0e990afc565463f18e82451d2c12e31b` |
| `command.txt` | `44abe957cff7dc81471eeda470f6cb71892dfbec6ccbebe1156c864d4b43feb4` |
| `gate-before.json` | `279cd6598bba4da6040cdeae9db0a83a0aed5bb1d39982709241d5ecb9f0348d` |
| `real/record.json` (458 KB; full record incl. the grid and diffs) | `747b34576bf80a56d94785855f8b6bf7d1dab4a288c48132d3d5cc4419116490` |
| `real/held_back_ids.txt` (= the sibling file committed here) | `2a90ef643bd354006f047841748edada44c5a552e943c0520c1cc5544062960d` |
- Left in place: that scratch directory (also `smoke/` from the synthetic pre-flight). The ceremony copy, both engine/plan/tests worktrees' tracked files, and every checkpoint path are untouched. This commit adds `build-005.md` and `build-005-held-back-ids.txt` only. No worktree, branch, PR or unit created. Then STOP — I replace and settle nothing (Exec P454).
