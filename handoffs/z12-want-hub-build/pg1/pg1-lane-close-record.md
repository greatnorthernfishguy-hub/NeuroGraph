<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 zone manager, lane want-hub-engine-d-build-20260930) — PG-1 lane CLOSE record
#   What: the closing record of PG-1 (the real-graph default-path golden for the (d) engine fold): verdicts, the honest limits, where the evidence lives, source-intact proof and the scratch hash manifest recorded BEFORE disposal (#872).
#   Why: Exec Packet 465 via Chief-003 (close met; state honestly that the pair is a reproduction; corrected wall wording; scratch disposal approved only after the close, originals verified by hash first).
#   How: every hash from sha256sum / git rev-parse, taken 2026-10-01 ~04:10Z.
# -------------------
-->

# PG-1 — lane close record (outcome MET)

**Claim closed:** on the DEFAULT path (every new parameter at its default) `_prune_synapses()` at BASE `e4ebf982b1989fd9066d610b94853bc68bf70d37` and at FOLD `29f47f65058790240b2f9c6a0a5bc4d82171b42d` are identical on two real graphs (14 compared fields, serialized checkpoint bytes included); copy (b) removed the same 10,433 ids in the same non-sorted order under both.

## Cited verdicts
| who | return | verdict |
|---|---|---|
| builder ee0547f2 | #12384 (build-004, commit `b363e6fb5036c67a3fe1ee19e12e1905e88851ef`, artifact commit `2475dcf05b589eab84b5d8e9a2c283002d43723c`) | claim PASS both copies (ruled as a claim, MET) |
| acceptor le-039 | #12416 (`pg1/acceptance-le-039.md`, commit `00c1f49f3e7d77f87b3ffffc9ed1e319cccc803e`; hash fields corrected in `2e618317ab2ca28e3fc4c089cbcfe50591d12b3f`) | MET: PASS (a), PASS (b) from its own re-run |
| le-040 ROLE B | #12503 (`reviews/le-040-pg1.md`) | COMPLIANT |
| checker-031 ROLE A grok-4.6 | #12504 (`reviews/checker-031-pg1.md`) | PASS-WITH-NOTES |
| #867 determination | #12502 (`build-005.md`, commit `caa39edf8e0bbaf8b338b5a28c61250d5b9b4037`) | MET, verdict A (code's 15 stands; plan §4A.3 to be corrected to 15 by the plan author) |

## Stated honestly
- **(a) The pair is a REPRODUCTION, not independent verification:** the acceptor and both reviewers shared the checkouts, the python, the `ng_tract` binary and the driver's digest; no mutant was run on a real graph; the staged bundle is not Syl's graph. What it verifies: the BASE/FOLD default-path identity is reproducible by a second operator with its own harness.
- **(b) Wall sentence, corrected:** the fold adds **at least +0.67 s validation CPU; the remaining wall difference (0 to +1.9 s per call, one process per variant) is unexplained and within run-to-run spread**. (le-040/checker-031 note it is not shown to be noise; the old wording "within the noise" was withdrawn in `pg1/acceptance-record-notes.md` note 3.) Acceptability of the `_step_lock` hold is NOT judged here: the Executive accepted it for arming (Exec 459); improvement row #868.
- **(c) Where the evidence lives:** on the TESTS branch `cc-laptop-want-hub-build-20260930`, NOT the plan's designated paths. Artifact commit `2475dcf05b589eab84b5d8e9a2c283002d43723c`; acceptance commit `00c1f49f3e7d77f87b3ffffc9ed1e319cccc803e` (the Packet's "00f1c49f" was a transposition, verified from the branch). Notes: `pg1/acceptance-record-notes.md`. Acceptor harness: `pg1/acceptor-harness/`.
- Deviations (harmless/needs the Executive's note): copy (a) read in place by the builder (outcome-neutral); the gate lowered from the plan's ~8 GB text to the ruled 3.0/6 GiB after the restore measured 0.992 GiB; artifact path on the tests branch.
- **Not part of this lane, still owed before arming/merge:** the S4 Tonic check (P392/P399), the daemon slice, #825, the Syl-own backup ceremony (Exec 441), the post-merge N5 dry run (also precondition: the plan author corrects §4A.3 to 15), and Josh's ONE rollout call (P441). Merge is Josh's. #871 = S4 preflight check (the `ng_tract` binary in use must match a recorded install hash, or the install record is updated with a stated reason). Nothing merged, nothing armed.

## Sources intact BY HASH before scratch disposal (#872, Exec 465 item 4)
Ceremony backup `pre-placement-laptop-cc/` and staged bundle `vps-pull-staged/`, hashed 2026-10-01 ~04:10Z: `main.msgpack` **7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77** and `vectors.msgpack` **93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e** = the ruled ceremony values; `commons.msgpack` 986c8e93670e6d47d8db2cc18836abc0ea6df0e8cf97feba59bf8fea53495918; staged `main.msgpack` 8cf6ef22…90a1, `activations.json` 5fc6f7d9…a947, `manifest.json` 3c00aee4…d9f, `vectors.msgpack` bb3479b4…b019 = the values recorded by the builder's `records/copy-b.json` (src_sha256_before == after == dst). Syl's `~/NeuroGraph/data/checkpoints` and the live CC checkpoint were not touched.

## Scratch hash manifest (recorded BEFORE disposal; sha256, path relative to /home/josh/backups) — dirs `z12-pg1-20261001T021808Z` (builder, 1.2 GB) and `z12-pg1-accept-20261001T025235Z` (acceptor, 1.4 GB), 64 files
```
8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1  z12-pg1-20261001T021808Z/bundle-copy/main.msgpack
5fc6f7d9456a915cb03cac79098b79a1e1635c6af49731e70f102102053ea947  z12-pg1-20261001T021808Z/bundle-copy/main.msgpack.activations.json
3c00aee4c3c39fb05350f46ebe145b60bffdc833d406b2a9a18797d755d96d9f  z12-pg1-20261001T021808Z/bundle-copy/main.msgpack.manifest.json
bb3479b4e2b3795f4777b39292269e664fcd216f4a8816921449170b2ad4b019  z12-pg1-20261001T021808Z/bundle-copy/vectors.msgpack
dce32a6e3af0ba01d86301a2dfea3715def60b038ff56b1be57121b4bc3d966a  z12-pg1-20261001T021808Z/records/copy-b-after-part1.json
0dc501a96d29fce7d55a2fb6f36e0700ad981a7f0465eaa87a779fcd70920369  z12-pg1-20261001T021808Z/records/copy-b.json
41459d3c98a3617522726cc091c3d0dc6f3027318f3aada50270aa725cc2cf4c  z12-pg1-20261001T021808Z/records/gate-a-base.json
45d4ba5f74cdb773acf3b67a67d5ae9ef6b2f660308cce8e1fc505c212f40e28  z12-pg1-20261001T021808Z/records/gate-a-fold.json
d1fba415eb69e80e7eb040d28d5cfc1fdc65779d42cca7e1b3c2f4de68e28eb0  z12-pg1-20261001T021808Z/records/gate-b-base.json
de0c4c75b6179275c37cbef2640906f222dbb0a5ec41ea1586cb7d8b2f79667e  z12-pg1-20261001T021808Z/records/gate-b-fold.json
2810f213c8d377a1e1d3a011093c6b2a270e59a80b9df008e7ebf28842f473ef  z12-pg1-20261001T021808Z/records/gate-p2-first.json
6e91e7d97d11f831da85db7c6ec505ba10295ce3a21218c1d23cc1735fdac3b7  z12-pg1-20261001T021808Z/records/gate-p2-fold.json
994f1a330ce2331df3c3e8c85284d6a8cecd0778ca02061a808499a3892201df  z12-pg1-20261001T021808Z/records/part1-a-base.json
416acb6c276ae604eca8ac1cbe9e05afc221f685bb60875b777203b2b03b736a  z12-pg1-20261001T021808Z/records/part1-a-fold.json
a426e8cd649951d9e7f96ba7353201270ca23c8d3ce5e55f5e1fa30d860cfd26  z12-pg1-20261001T021808Z/records/part1-b-base.json
f4fa09b0c2ae21ed24c134267a06cd5adca2a76c55aa4d5db5a6795a3d898468  z12-pg1-20261001T021808Z/records/part1-b-fold.json
4b4d7f4ee628173c4f2d1feb46b237c579c7f5720081715fd41f6cb075f0b0ea  z12-pg1-20261001T021808Z/records/part2-first.json
83ac2be3f8c2d67398070d504201d75afebd51414038afbb59ef9cc0f53ada06  z12-pg1-20261001T021808Z/records/part2-fold.json
6faa8ec9c84dd963eaf25a0e865a44a95ad1b1a72f214fae5fd63d253bbf90a5  z12-pg1-20261001T021808Z/records/smoke-part1-base.json
6b517a260d2f9f020adbd75b8262da90af0c6df85dfbd641d0193e902a491069  z12-pg1-20261001T021808Z/records/smoke-part1-fold.json
131d5cdfc99d9e28ec1eda144265de320baa0426cb8cc347e0bc7b212b889dc0  z12-pg1-20261001T021808Z/records/smoke-part2b-first.json
b5ec8f17feeff12fcb62ec865b90998d0bf2c7a405e4f51d7d68af51916668e1  z12-pg1-20261001T021808Z/records/smoke-part2b-fold.json
fbc7f1d4c738869c08089227608580e9e6f53c5205cbfa2cdcf7941f570570bd  z12-pg1-20261001T021808Z/records/smoke-part2-first.json
6e2c8e1e10c1755f871bf774ab9644176145af6b33269da75267b594369a4249  z12-pg1-20261001T021808Z/records/smoke-part2-fold.json
26de35e29d5474657b887d4a8d46c3883291cbff4550c5702f6422f561b0a70a  z12-pg1-20261001T021808Z/records/src-a-after-part1.json
7f3ac987ea2a4bfc9e6b6e6e85fbc163d532e12dd4b96a278e0dc88298149f9a  z12-pg1-20261001T021808Z/records/src-a-after-part2.json
104cd898af5bda66c167a16ab836c23d748b609e60e5fd6d25c90ab142b9c173  z12-pg1-20261001T021808Z/records/src-a-before-part1.json
590ae326ff1187cfcc27c86e83681ef152345e9fd481e2a5c9b71373a470803f  z12-pg1-20261001T021808Z/records/src-b-vps-staged-after.json
902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce  z12-pg1-20261001T021808Z/scratch/a-base.msgpack
902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce  z12-pg1-20261001T021808Z/scratch/a-fold.msgpack
6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d  z12-pg1-20261001T021808Z/scratch/b-base.msgpack
6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d  z12-pg1-20261001T021808Z/scratch/b-fold.msgpack
962b45283337181e87a304f0d78f1d955a8665013f37380e8bb4dc63e90dc159  z12-pg1-20261001T021808Z/scratch/smoke-base.msgpack
962b45283337181e87a304f0d78f1d955a8665013f37380e8bb4dc63e90dc159  z12-pg1-20261001T021808Z/scratch/smoke-fold.msgpack
9fc608a0ba01f8ae8aad482c986961cdca878746d1d1615690af4bde2c434150  z12-pg1-20261001T021808Z/smoke-src2/main.msgpack
1a8aa5ae90aafb01137291875f6f86f82bd7f1ab764e7c83355a7231f143c736  z12-pg1-20261001T021808Z/smoke-src/main.msgpack
6fb44fcc802cd3fa18a3fb728a0599a58142151c82b2c2421b7f0dbce624e07b  z12-pg1-accept-20261001T025235Z/accept_harness.py
70bf6199c9a86bfae2c24f7d81543bfd330847225b76a6b4e5c626f69168bb5e  z12-pg1-accept-20261001T025235Z/compare_accept.py
7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77  z12-pg1-accept-20261001T025235Z/copy-a/main.msgpack
8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1  z12-pg1-accept-20261001T025235Z/copy-b/main.msgpack
5fc6f7d9456a915cb03cac79098b79a1e1635c6af49731e70f102102053ea947  z12-pg1-accept-20261001T025235Z/copy-b/main.msgpack.activations.json
3c00aee4c3c39fb05350f46ebe145b60bffdc833d406b2a9a18797d755d96d9f  z12-pg1-accept-20261001T025235Z/copy-b/main.msgpack.manifest.json
bb3479b4e2b3795f4777b39292269e664fcd216f4a8816921449170b2ad4b019  z12-pg1-accept-20261001T025235Z/copy-b/vectors.msgpack
85ad860c8e3dea2f26c5343b0a6aa1818ff338ee36976779def4d53af447e511  z12-pg1-accept-20261001T025235Z/records/accept-compare.json
ea7bcecc39eb717b0f5a8652dc5a6eea7f07c7d60aa08c18e35e816be5efe000  z12-pg1-accept-20261001T025235Z/records/crosscheck.json
cb86fe3fec352d4f2843d363d918f31099b10c1c565b4294688a5be214a318ad  z12-pg1-accept-20261001T025235Z/records/gate-a-base.json
32677af5dd5cc46c954207e3864661c1d8f5f81bca1a433028693d2138d5d9a3  z12-pg1-accept-20261001T025235Z/records/gate-a-fold.json
f3837ce71742a4fa220104844398526767b36a72b571a28d68694b3ba3dd76fa  z12-pg1-accept-20261001T025235Z/records/gate-b-base.json
4b6c718bfdfb509b34abd4dd64eff6ad3296e8c0ececfa20fac15233e7ae43ac  z12-pg1-accept-20261001T025235Z/records/gate-b-fold.json
d7a321c425e5aa89e573bbb9e21e1ce2e6a0bd604d7df449c2b3c0eaad01b2e8  z12-pg1-accept-20261001T025235Z/records/part1-a-base.json
53aed059ab2971f4f8ea7c1b22886b6aead3baf4459b62b0bf450b2aa952e908  z12-pg1-accept-20261001T025235Z/records/part1-a-fold.json
e5ebd4676403bb2c363d03256d5fed58a8b36b07e2bea552284a6cca2ef3cbe8  z12-pg1-accept-20261001T025235Z/records/part1-b-base.json
2a3984d450df8627c22bab10675d2ff6ff356a8ef7b596585259b7b34673a5db  z12-pg1-accept-20261001T025235Z/records/part1-b-fold.json
a46fbda2ce79ed035d1e9ffc43b12ae841e310449afe6181c88844fece88c6e3  z12-pg1-accept-20261001T025235Z/records/sha-a-source.txt
032876220d8d14e586635952e98bbb1a7f564ec1737af43b430db3770371851d  z12-pg1-accept-20261001T025235Z/records/sha-builder-bundlecopy.txt
a7ceca21b69776d180a8c28c1dc0476f19cfacc8f75baf5599e37c4e4756794d  z12-pg1-accept-20261001T025235Z/records/sha-builder-ckpts.txt
a7ceca21b69776d180a8c28c1dc0476f19cfacc8f75baf5599e37c4e4756794d  z12-pg1-accept-20261001T025235Z/records/sha-my-ckpts.txt
032876220d8d14e586635952e98bbb1a7f564ec1737af43b430db3770371851d  z12-pg1-accept-20261001T025235Z/records/sha-v-source.txt
8c72ac648feb4d7f4a013defe79218d92baf2a39dcf2f113c0284cbe0029fbc8  z12-pg1-accept-20261001T025235Z/records/stat-a-before.txt
993ae3d14be99a1a9ca8199cd0db58873b3961153a4e4ff0c7113899d2f12e53  z12-pg1-accept-20261001T025235Z/records/stat-v-before.txt
902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce  z12-pg1-accept-20261001T025235Z/scratch/a-base.msgpack
902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce  z12-pg1-accept-20261001T025235Z/scratch/a-fold.msgpack
6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d  z12-pg1-accept-20261001T025235Z/scratch/b-base.msgpack
6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d  z12-pg1-accept-20261001T025235Z/scratch/b-fold.msgpack
```
