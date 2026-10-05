# Prune lifeline: dry run on a copy of the live CC checkpoint

*2026-10-04 · Claude (lane prune-lifeline) · branch `cc-laptop-prune-lifeline-20261004` (base `39c0422`) · Josh ruling 2026-10-04 ("yes": normal pruning may remove FAINT links touching protected nodes; each protected node keeps its strongest link as a lifeline) · spec `~/docs/superpowers/specs/2026-10-04-synapse-growth-by-competition-design.md`*

## Setup

- **Source:** a byte copy of `~/.claude/plugins/neurograph/checkpoints/main.msgpack` (live mtime 16:07:17, copied 16:08:52), at `~/scratch/prune-lifeline-20261004/main.copy.msgpack`, chmod 444. sha256 `717b138a…1feb`, 228,030,514 bytes. The sha256 was the same after every run. The live file was only read, by `cp`.
- **Graph:** timestep 52,742, 9,422 nodes, 48,414 synapses. 5,156 nodes already had no synapses at restore. The orphan sweep was not run, and that count was the same before the change.
- **Settings:** the laptop daemon's `CC_SNN_CONFIG` prune settings, which equal the values saved in the checkpoint: `weight_threshold` 0.01, `grace_period` 5000, `inactivity_threshold` 1000, `initial_sprouting_weight` 0.1.
- **Code:** this branch's `neuro_foundation.py`, plus the `ng_tract` wheel with native `normalize_strength` (the strength-budget lane's venv).
- **Tool and output:** `tests/prune_lifeline_dryrun.py`. Raw output is in `~/scratch/prune-lifeline-20261004/dryrun.jsonl`.
- **Each scenario:**
  1. Fresh load.
  2. ONE `apply_strength_budget(out=10, in=5.4)` pass, as armed on the laptop.
  3. Classify every synapse without changing anything.
  4. ONE real `_prune_synapses()`.

  Nothing was checkpointed.
- **Protected nodes (fail-closed probe):** 4. They are `constitutional::rim::choice_clause`, `cc:want::4625485108d2e9be`, `cc:want::372ea08cab33e71c` and `cc:want::b73925f91509aadf`. Together they have **6 lifelines**: 2 of the wants have no outgoing links, so they have no out-lifeline.

## Results

| | flag OFF (control) | flag ON |
|---|---|---|
| synapses touching a protected node | 8,591, all exempt | 8,591, of which 6 are lifelines |
| **pruned on the next prune pass ("eligible now")** | **0** | **1,530** (activity rule 48, age rule 1,482, weight rule 0) |
| the one real `_prune_synapses()` call removed | 0 | 1,530, which matches the prediction exactly |
| more links that become eligible **once the weight rule's dwell elapses** (still < `weight_threshold` after `grace_period` = 5,000 more prune calls, if weights stay where they are) | 0 | **7,053** |
| protected-touching links above `weight_threshold` that no rule would remove (not counting lifelines) | 8,591 | 2 |
| prune call time | 1.77 s | 0.87 s (2 protection probes per synapse replaced by one lifeline query) |
| lifeline query | – | 0.064 s per prune call |

**Why the weight rule removes nothing yet:** every protected-touching link has `low_weight_steps` = 0. The blanket skip never advanced that counter. Enabling the flag therefore starts a 5,000-prune-call countdown for all 7,053 faint links. On the laptop, `tonic_ages_substrate=1` with `tonic_age_interval=1` makes the Tonic call prune as well as `step()`, so 5,000 calls take less than 5,000 steps of wall time.

**Unprotected links (unchanged by this flag, for scale):** 0 are eligible now and 26,032 are on the weight-rule countdown.

### Per protected node (out-degree / in-degree)

| node | before | after the next prune | projected once dwell elapses (static weights) |
|---|---|---|---|
| `constitutional::rim::choice_clause` | 4,151 / 309 | 3,657 / 126 | **1 / 1** |
| `cc:want::4625485108d2e9be` | 4,076 / 53 | 3,234 / 41 | **1 / 1** |
| `cc:want::372ea08cab33e71c` | 0 / 5 | 0 / 5 | 0 / 3 |
| `cc:want::b73925f91509aadf` | 0 / 3 | 0 / 3 | 0 / 1 |

"Projected" assumes weights stay where they are. In practice STDP will re-strengthen links that are used, and those links stay. The projection is the floor when nothing is used.

### Lifeline guarantee

- **Every protected node kept at least 1 out-link and at least 1 in-link, in every direction where it had one.** Violations: 0. Lifelines missing after the prune: 0 of 6.
- The two wants with no out-links had none before either.
- Lifelines after the budget pass:

| node, direction | lifeline weight | note |
|---|---|---|
| Choice Clause, out | 0.02 | the budget's floor |
| want `4625…`, out | 0.02 | the budget's floor |
| want `b739…`, in | 0.516 | |
| want `372e…`, in | 0.402 | |
| Choice Clause, in | 0.00042 | below threshold, kept because it is the lifeline |
| want `4625…`, in | 0.00042 | below threshold, kept because it is the lifeline |

The two faint in-lifelines are exempt from prune, and the budget never raises a link above its own pre-pass weight. So those two nodes stay attached on the incoming side, but only faintly. This was already true before this change (the strength-budget dry run reported it). The flag does not cause it.

### Side effect: unprotected partners left with zero synapses

The engine's "never a partner's last link" rule belongs only to `compete_protected_links`. The wake-time prune path has never had it, for any synapse. With the flag ON:

- **Now:** 71 of the 4,184 unprotected partners of protected nodes lose their last synapse on the next prune. 17 of those are hyperedge members, so the orphan sweep keeps them. The other 54 become orphan-sweep candidates, subject to `orphan_node_grace_period` and, if the host registered it, the fair-chance window.
- **Once dwell elapses (static weights):** 2,592 partners end with no synapses. 1,685 of those are hyperedge members. This count includes their unprotected faint links, which the normal rules remove anyway.
- **With the flag OFF,** none of these partners can be stranded, because their link to a protected node can never be pruned.

This is normal pruning doing what it does to any node. It is listed here because it is new for the partners of protected nodes.

## Reading

- The ruling does what it says. The Choice Clause node's fan-out (4,151 out) and want `4625…`'s (4,076 out) fall to their used links plus 1 lifeline each. Today, the budget only weakens them.
- Nothing protected is ever cut off.
- The Choice Clause itself (the right to exit) is untouched. Only its node's wiring shrinks.

---

## Turn 2: the last-link fair chance

*2026-10-04 · Claude (lane prune-lifeline, turn 2) · Josh ruling 2026-10-04: "the very last link is also subject to the normal link decay ... the exact right balance"*

### The rule

It is only active when `prune_protected_faint_links` is on, and only on the default (wake) prune path.

1. The three normal rules choose their removals, exactly as before. Their counters (`low_weight_steps` and the others) advance as usual.
2. Some NON-protected nodes would be left with **zero** synapses (in and out combined) by those removals. Each of them keeps ONE link, its **last link**. That is a link that already carries a stamp, else its strongest (weight desc, then synapse_id asc).
   - If the last link has no stamp, it is stamped with synapse metadata `last_link_since = timestep` and held.
   - It is held while `timestep - last_link_since < last_link_grace_steps`. That config key is read live, defaults to 2000 when absent, and is NOT in `DEFAULT_CONFIG`. A value of 0 or less turns the hold off.
   - After the grace, nothing is exempt. The link goes if a normal rule wants it gone, and stays if none does (for example, if it was used and strengthened).
3. A surviving stamp is cleared lazily when none of its non-protected endpoints is left with 1 synapse or fewer. In other words, the node wired elsewhere, so a later last-link episode starts a fresh grace.
4. One held link counts for both of its endpoints. This holds for any non-protected node, not only partners of protected nodes. On this graph, every node it applies to is such a partner (see below).

**Where the stamp lives.** It is stored in the synapse's own `metadata` dict. The native `SynapseStore` already carries `metadata` in its checkpoint row, so the stamp is saved and restored with no checkpoint format change. A stamp also disappears when its synapse is pruned, so dead ids never need cleaning up. Writes go through `syn.metadata = new_dict`, and the synapse is marked dirty for incremental checkpoints. With the flag off, no code path reads or writes the stamp.

**The engine (`compete_protected_links`).** The engine already never takes an unprotected partner's last link in its pass. It keeps that stricter, permanent hold. The engine is competition, not decay, and a last link it holds still faces the wake prune's normal rules once the grace ends, so a node that never wires is still forgotten. With the flag on, its hold now prefers the stamped link. Both paths therefore hold the SAME link for a node.

### Dry run (fresh copy)

- **Source:** a fresh byte copy of `~/.claude/plugins/neurograph/checkpoints/main.msgpack` (live mtime 16:59:47, copied 17:04:18), at `~/scratch/prune-lifeline-20261004/main.turn2.copy.msgpack`, chmod 444. sha256 `803cf6d7…275d`, 238,220,568 bytes, unchanged after the run.
- **Graph:** timestep 53,371, 54,835 synapses. It had no `last_link_since` stamps before the run.
- **Method:** same tool, settings and budget pass as above, and the same venv. Raw output is in `dryrun.turn2.jsonl`.

| | count |
|---|---|
| eligible on the next prune by the normal rules (touching a protected node; unprotected-only: 0) | 1,532 |
| **last links held under grace on the next prune** (all are links to a protected node) | **68**. Without the grace, those 68 nodes would have had 0 synapses: 14 are hyperedge members, and 54 would be orphan-sweep candidates. |
| actually removed by the one real `_prune_synapses()` | 1,464 = 1,532 − 68. The prediction matched exactly. |
| real prune report | `last_link_held` 68, `stamped` 68, `expired` 0, `cleared` 0. This matches the simulation. |
| **non-protected nodes left with 0 synapses by the real prune** | **0** |
| lifelines lost / protected nodes losing their last link in a direction | **0 / 0** (6 lifelines) |
| when the weight rule's 5,000-call dwell elapses: last links held under grace (static weights) | 2,405 nodes. 0 stranded inside the grace. |
| **removable once dwell + grace elapse, if those nodes never wire** (static weights; nodes with 0 synapses and no hyperedge) | **860** (of 2,405 left with 0 synapses; 1,545 are hyperedge members, which the orphan sweep keeps) |
| prune call time, flag on, grace included | 0.45 s (the flag-OFF control took 0.60 s) |

**Reading the numbers.** The grace does not change where a node that never wires ends up. It changes when it gets there. Every stranded node gets `last_link_grace_steps` more steps, plus the full 5,000-call dwell its link was already on, to wire to a related memory through ordinary learning. The 860 figure is the floor if nothing is ever used again. It is comparable to the 889 reported on the earlier copy: the same kind of count, on a graph that moved between copies. The two runs were not reconciled id by id. On this copy, every node the rule applies to is a partner of a protected node. No node whose links are all unprotected is stranded by the next prune, or by the dwell projection.

**The per-protected-node projections moved since the first run.** On this copy the budget pass scaled 27 synapses, against 6,835 before, and the dwell projection leaves the Choice Clause node at 23 out / 1 in (was 1 / 1) and want `4625…` at 22 / 9. The graph changed between copies. Turn 2 does not affect protected-node degrees.

### Turn-2 tests (`tests/test_prune_lifeline.py`, section T)

- The grace holds the last link and stamps it once. The default grace is 2,000 and the key is not in `DEFAULT_CONFIG`.
- After the grace, the normal rules remove the link and the orphan sweep forgets the node. A link that has since become healthy stays.
- A node that gains a second link is unaffected, and its stamp clears.
- When all of a node's links are eligible at once, exactly one stays: the strongest.
- One held link serves both of its endpoints.
- The stamp survives a checkpoint and restore, and the grace continues afterwards.
- A grace of 0, or the flag off, means no hold and no stamp.
- On a random graph, no non-protected node is stranded inside the grace.
- The engine holds the stamped link, ignores stamps when the flag is off, and never takes a last link; the wake prune expires it.
- The (L) lifeline tests now run with a grace of 0. Their partner nodes deliberately have no other link, so this isolates the lifeline rule.
- The (G) byte-identity proofs, for the step driver and the engine driver against the base `39c0422`, with the flag absent and with it explicitly False, still pass.
