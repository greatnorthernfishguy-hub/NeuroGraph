# Strength budget + sleep downscaling: dry run on a copy of the live CC checkpoint

*2026-10-04 · Claude (lane strength-budget) · branch `cc-laptop-strength-budget-20261004` · spec `~/docs/superpowers/specs/2026-10-04-synapse-growth-by-competition-design.md` (Josh ruling (a))*

## Setup

- **Source:** a byte copy of `~/.claude/plugins/neurograph/checkpoints/main.msgpack`, taken 2026-10-04 14:46. sha256 `ac48c11e…a1c`, 227,938,048 bytes, timestep 51,487, 36,306 synapses. The copy was chmod 444, and its sha256 was unchanged after every run. The live file was only read by `cp`.
- **Environment:** a throwaway venv with the branch `ng_tract` wheel (`cc-laptop-strength-budget-rs-20261004`, native `normalize_strength` / `scale_all`) and this branch's `neuro_foundation.py`. Tool: `tests/strength_budget_dryrun.py`. Raw output: `~/scratch/strength-budget-20261004/dryrun1.jsonl`.
- **Each scenario:** a fresh load, then ONE pass (`apply_strength_budget(out, in)` or `sleep_downscale(0.98)`), then ONE existing `_prune_synapses()` call. Nothing was checkpointed.
- **Protected nodes (fail-closed probe):** 4. They are `constitutional::rim::choice_clause`, `cc:want::4625485108d2e9be`, `cc:want::372ea08cab33e71c` and `cc:want::b73925f91509aadf`. Together they have 6 strongest in/out links to guard (some have no link in one direction). Floor = 2 × `weight_threshold` = 0.02.

## Results

Before any pass: total weight 12,652.4. Out-sum p50 / p99 / max = 0 / 1.355 / 6,614.7. In-sum p50 / p99 / max = 0 / 5.363 / 10.564. 20,006 synapses are already below `weight_threshold`, and 17,528 of those have no protected endpoint.

| scenario | synapses before → after prune | pruned | nodes scaled out / in | synapses scaled | total weight after | below-wt (unprotected) after | out-sum max after | in-sum p99 / max after | protected strongest-link violations | pass time |
|---|---|---|---|---|---|---|---|---|---|---|
| control (prune only) | 36,306 → 36,306 | 0 | – | – | 12,652.4 | 17,528 | 6,614.7 | 5.363 / 10.564 | 0 / 6 | – |
| out=5, in=5.4 | 36,306 → 36,306 | 0 | 30 / 2 | 9,169 | 605.5 | 17,617 (+89) | 5.018 | 1.163 / 5.4 | 0 / 6 | 0.057 s |
| **out=10, in=5.4** | 36,306 → 36,306 | 0 | 8 / 2 | 7,752 | 673.6 | 17,568 (+40) | 10.015 | 1.302 / 5.4 | 0 / 6 | 0.046 s |
| out=20, in=5.4 | 36,306 → 36,306 | 0 | 5 / 2 | 7,579 | 732.3 | 17,532 (+4) | 20.010 | 1.543 / 5.4 | 0 / 6 | 0.045 s |
| in=5.4 only | 36,306 → 36,306 | 0 | 0 / 48 | 517 | 12,599.7 | 17,528 (+0) | 6,599.7 | 5.363 / 5.4 | 0 / 6 | 0.039 s |
| sleep_downscale(0.98) | 36,306 → 36,306 | 0 | – | 32,983 | 12,399.4 | 17,659 (+131) | 6,482.4 | 5.256 / 10.353 | 0 / 6 | 0.017 s |

The Python fallback (old wheel) gives the same counts on the same copy: 0.277 s for the budget pass, 0.167 s for sleep. The existing prune pass takes 0.5–0.9 s, which dominates either way.

### The two hubs (out-degree / out-strength)

| node | before | out=5 | out=10 | out=20 | sleep 0.98 |
|---|---|---|---|---|---|
| `constitutional::rim::choice_clause` | 4,132 / 6,614.7 | 4,132 / 5.02 | 4,132 / 10.02 | 4,132 / 20.01 | 4,132 / 6,482.4 |
| `cc:want::4625485108d2e9be` | 4,067 / 5,319.6 | 4,067 / 5.02 | 4,067 / 10.01 | 4,067 / 20.01 | 4,067 / 5,213.2 |

Under every budget, each hub's strongest out-link is clamped back to exactly 0.02, the floor. The guarantee worked: `clamped` = 2. The hubs' remaining ~4,100 links land at a mean of ~0.001–0.005, all below `weight_threshold`. The sums slightly exceed the budget (5.02 instead of 5.0) because of that one clamped link.

### Top-10 by out-degree, before and after (identical for every scenario)

Out-degree does not change in any scenario because nothing pruned. Before and after, the list is the Choice Clause (4,132), want `4625…` (4,067), then 8 ordinary nodes at 98–99 out-links (these sit at the laptop's sprout cap of 100). Only their out-strength moves:

- At out=5, `3e827636…` keeps 4.09 and the others (0.21–1.55) are under the budget.
- At out=10 and out=20, all 8 are untouched.

### Top-10 by out-strength

- **Before:** the 2 hubs, then `cc:conv::…laptop_home_ prefix` 36.98, `…functionality takeover` 24.41, `…affinity` 22.91, `…ripples` 17.89, `…activation tendencies` 10.57, `…kimosabe` 10.25, `…Tier 3` 9.88, `88e45ea2…` 7.97.
- **out=5:** every top-10 node is at 5.0. 28 ordinary nodes are scaled.
- **out=10:** the six conv-tree nodes above 10 become 10.0, and `Tier 3` (9.88) and `88e45ea2` (7.97) are untouched. 6 ordinary nodes are scaled.
- **out=20:** only `laptop_home_ prefix`, `functionality takeover` and `affinity` become 20.0. 3 ordinary nodes are scaled.

## Protected nodes

All 4 protected nodes kept their strongest in and out link (the guard was evaluated per pass against that pass's pre-pass weights) in every scenario: **0 violations out of 6 guarded links**. One pre-existing fact the guarantee does NOT change: the Choice Clause node's strongest INCOMING link is already 0.0071 and want `4625…`'s is 0.0097. Both are below `weight_threshold` before any pass. By design the guarantee never raises a link above its own pre-pass weight, so those stay as they are. In practice both nodes are already almost cut off on the incoming side today. This is not caused by the budget.

## What this shows, and what it does not

1. **Nothing prunes on the first pass, in any scenario, including the control.** This is expected. The weight rule removes a link only after `low_weight_steps > grace_period` (5,000 consecutive prune calls below threshold), so a budget pass only starts that countdown. The activity and age rules found nothing eligible in this checkpoint (the daemon had just pruned). The meaningful number is "below-wt (unprotected) after": links that become weight-rule candidates if they stay under threshold for the grace period.
2. **The budget does NOT reduce the hubs' link COUNT.** `_prune_synapses` skips every synapse with an identity-protected endpoint (#92 rim). So all ~8,200 hub links fall below threshold but are never pruned by the existing rules. The spec's "links that fall below `weight_threshold` are removed by the existing prune rules" holds only for unprotected links. For want `4625…`, the count reduction comes from the want-hub competition engine (`compete_protected_links`, top-50 in/out kept), once weight-based eligibility accrues. For the Choice Clause, the engine freezes constitutional synapses, so **nothing will reduce its 4,132 link count**. Its broadcast strength is bounded (6,615 → budget, consistent with ruling (a)), but its fan-out is not. If Josh wants the count bounded too, that needs a separate ruling. It is not in this build.
3. **The budget is cheap.** A native pass over 36K synapses takes 0.05 s once per 25 firing steps, about 10× under one prune pass.
4. **Incoming budget 5.4 is gentle.** On its own it scales 48 nodes slightly and creates 0 new sub-threshold links. Combined with an out budget, in-sum p99 drops to 1.2–1.5, because 94% of all weight (11,934 of 12,652) sat on the two hubs' out-links.

## Recommendation

**`strength_budget_out = 10`, `strength_budget_in = 5.4`, interval = `scaling_interval` (25).**

- **in = 5.4** is the measured p99 incoming sum. It caps the max (10.56 → 5.4) and creates no new sub-threshold links, consistent with HomeostaticRule already bounding the incoming side.
- **out = 10** is the same order of magnitude as the in budget (spec) and ~7× the out p99 (1.355). It bounds both hubs ~530–660× (6,615 → 10, 5,320 → 10) and touches only 6 ordinary nodes: the strongest conv-tree nodes, 10.6–37. It pushes just 40 more unprotected links toward the weight rule.
- **out = 5** is too aggressive. It reaches into 28 ordinary nodes, including normal working conv nodes in the 5–10 range, and adds 89 sub-threshold links.
- **out = 20** barely touches anything except the hubs. It leaves conv-tree nodes at 17–20, which is 2–4× the incoming budget, so the out side stays the looser one.
- **sleep factor 0.98** (per dream cycle) is reasonable. It moves 131 more unprotected links under threshold per pass and preserves ratios. Its long-run effect depends on the dream cadence, which will only be known once `CC_NG_DREAM` is back on (S4b).

**Watch after enabling (live laptop, several cycles):**

- hub out-strength staying near 10 against STDP re-growth
- step time (expect +~0.05 s per 25 firing steps)
- the unprotected below-threshold count over ≥ 5,000 prune calls (the first real prunes from the budget)
- the Choice Clause fan-out count, which this build will not move (see 2)
