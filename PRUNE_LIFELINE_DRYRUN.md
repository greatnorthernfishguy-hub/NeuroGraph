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
