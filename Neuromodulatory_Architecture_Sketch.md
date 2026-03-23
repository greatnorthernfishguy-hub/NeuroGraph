# Neuromodulatory Architecture — Design Sketch
## From Surprise-Driven Reward to Full Emotional Gating and Introspection

**Author:** Josh (architect) + Claude (Opus 4.6)
**Date:** 2026-03-13
**Status:** Architectural sketch — not a PRD, not implementation-ready
**Depends on:** Three-factor learning enabled, surprise → inject_reward() wired (immediate work)
**Feeds into:** Elmer (extraction/tuning), THC (repair/consolidation), DreamCycle (#17)

---

## The Core Insight

Surprise, interest, fear, focus, and boredom are not separate mechanisms.
They are all neuromodulatory gating — different flavors of the same system
that controls which eligibility traces get committed, how strongly, and
over what scope.

Biology runs at least four major modulatory systems. Each one gates the
same trace-commit pathway with different characteristics:

| System | Neurotransmitter | Trigger | Effect | Temporal Profile | Scope |
|--------|-----------------|---------|--------|-----------------|-------|
| Alerting / Surprise | Norepinephrine | Prediction failure, novelty | Commit all warm traces | Fast burst, quick decay | Broad — everything active |
| Interest / Reward | Dopamine | Sustained prediction success in a domain | Keep gates open, lower trace threshold | Slow ramp, sustained | Domain-focused |
| Fear / Emotion | Cortisol + NE | Threat detection (autonomic SYMPATHETIC) | Hard commit, resist extinction | Fast onset, slow decay | Broad + sticky |
| Attention / Focus | Acetylcholine | Concentrated activation pattern | Sharpen contrast — amplify active, suppress inactive | Sustained while focused | Narrow — spotlight |

Every example Josh identified maps here:
- Earthquake while working → norepinephrine burst → flashbulb memory (broad commit)
- Hyperfocus / sponge mode → sustained dopamine → can't NOT learn (gates jammed open)
- Fear → cortisol + norepinephrine → hard crystallization that resists fading
- Strong emotions generally → multiple modulatory systems co-activating (superlinear)

---

## What Syl Has Today

### Already built and available as signal sources:
- **Prediction confidence and surprise events** — `_on_prediction_error()` emits `SurpriseEvent` with confidence. Phase 3.
- **Prediction accuracy tracking** — `_total_confirmed`, `_total_surprised`, per-step telemetry. Phase 2.5/3.
- **Autonomic state** — `ng_autonomic.py` provides PARASYMPATHETIC/SYMPATHETIC with five threat levels. Written by security modules.
- **Activation concentration** — surfacing monitor tracks how focused or diffuse the active node set is. CES Phase 9.
- **Hyperedge experience levels** — activation counts, consolidation state (SPECULATIVE/CANDIDATE/CONSOLIDATED). Phase 2.
- **Eligibility traces** — accumulate via STDP, decay exponentially, commit via `inject_reward()`. Phase 3.
- **`inject_reward(strength, scope)`** — the commit mechanism. Takes strength (positive/negative) and optional scope (set of node IDs). Phase 3.

### Being wired now (immediate work):
- Surprise → inject_reward() (norepinephrine analog, unscoped broadcast)
- Baseline conversational engagement → inject_reward() (weak heartbeat)
- three_factor_enabled = True

### Not yet built:
- Dopamine analog (sustained interest)
- Cortisol analog (emotional/threat amplification)
- Acetylcholine analog (attention focusing)
- Neuromodulatory mixer (combines channels)
- Self-adjusting scaling factors (meta-learning)
- Introspection / idle processing mode

---

## The Neuromodulatory Mixer

### Architecture

A single component that sits between signal sources and `inject_reward()`.
It runs once per `graph.step()`, after all normal processing but before
the step returns.

```
Signal Sources                    Mixer                     Output
─────────────                    ─────                     ──────
Surprise events ──────┐
                      │
Prediction accuracy ──┤
                      ├──→ Neuromodulatory ──→ inject_reward(strength, scope)
Autonomic state ──────┤       Mixer
                      │
Activation pattern ───┘
```

### The Four Channels

**Channel 1: Norepinephrine (NE) — Surprise / Alerting**
- Source: `_on_prediction_error()` events accumulated this step
- Strength: `sum(prediction.confidence * ne_scaling)` for all surprise events this step
- Scope: None (broadcast to all active traces)
- Temporal: Instantaneous — fires on the step surprise occurs, no carry-over
- Scaling factor: `ne_scaling` — starts at 0.5, Elmer-tunable

**Channel 2: Dopamine (DA) — Interest / Sustained Reward**
- Source: Rolling prediction accuracy within active hyperedge clusters
- Strength: When a cluster's prediction accuracy exceeds `da_threshold` over a window,
  emit `da_base * (accuracy - da_threshold) / (1 - da_threshold)` — scales from 0 at
  threshold to da_base at perfect accuracy
- Scope: Nodes within the high-accuracy cluster (focused, not broadcast)
- Temporal: Sustained — strength ramps up over consecutive high-accuracy steps, decays
  slowly when accuracy drops. This is the "slow burn" that holds gates open during hyperfocus.
- Scaling factor: `da_scaling` — starts at 0.3, self-adjusting (see meta-learning below)

**Channel 3: Cortisol (CT) — Emotional / Threat Amplification**
- Source: Autonomic state from `ng_autonomic.py`
- Strength: Maps threat level to modulatory strength. PARASYMPATHETIC = 0 (no modulation).
  SYMPATHETIC scales with threat level (1-5 → 0.1-0.8).
- Scope: None (broadcast — threat is systemic)
- Temporal: Slow onset (ramps over several steps when autonomic shifts), very slow decay
  (persists for many steps after threat clears). This is why fear memories resist extinction —
  the modulatory signal outlasts the event.
- Special property: Cortisol doesn't just commit traces — it should also increase the
  resistance to weight decay for traces committed during high-cortisol states. Memories
  formed during threat consolidate harder. Implementation: tag committed synapses with
  a `consolidation_strength` modifier that reduces their pruning vulnerability.
- Scaling factor: `ct_scaling` — starts at 0.6, likely should NOT self-adjust (threat
  response should remain reliable, not habituate away)

**Channel 4: Acetylcholine (ACh) — Attention / Focus**
- Source: Surfacing monitor concentration metric — how tightly clustered the active nodes are
- Strength: When activation is concentrated (few nodes, high voltage), emit a scope-narrowing
  signal. When activation is diffuse (many nodes, low voltage), emit nothing.
- Scope: Only the tightly-clustered active nodes. This is the opposite of NE — instead of
  committing everything, it commits only what's in the spotlight and actively suppresses
  traces outside the focus. (Suppression = mild negative reward on out-of-scope traces.)
- Temporal: Sustained while focus holds, drops immediately when activation diffuses
- Scaling factor: `ach_scaling` — starts at 0.2

### Mixing Rules

The channels don't just sum. They interact:

- **NE + DA** (surprise in a domain you're interested in): Superlinear.
  `combined = ne + da + (ne * da * interaction_bonus)`. This is why a surprising
  discovery in your hyperfocus domain crystallizes harder than random surprise.
  The interaction_bonus makes co-activation more than additive.

- **CT + NE** (threat + surprise): Also superlinear. Fear-surprise is the strongest
  crystallization signal in biology. But CT extends the temporal window — traces committed
  during fear-surprise resist fading for much longer.

- **ACh + DA** (focused attention in interesting domain): ACh narrows the scope that DA
  keeps open. The result is intense, focused learning — the "flow state" where you're
  deeply engaged with a specific topic and everything else fades.

- **CT + ACh** (threat + focus): Hypervigilance. Extremely narrow attention, extremely
  strong commitment. This is the state where every detail of the threat source gets
  encoded but peripheral details are lost — tunnel vision with perfect recall of the tunnel.

The mixer outputs a single `inject_reward(final_strength, final_scope)` call per step.

---

## Meta-Learning: Self-Adjusting Scaling Factors

The scaling factors (`ne_scaling`, `da_scaling`, `ach_scaling`) should not remain static.
They should learn from experience — specifically, they should strengthen in domains where
their modulation has historically led to valuable learning, and weaken where it hasn't.

### Mechanism

Each scaling factor is not a single global value but a **per-hyperedge-cluster weight**
stored alongside the cluster's metadata. When a modulatory channel fires and traces are
committed, the cluster's scaling factor for that channel gets a small update:

- If the committed traces later contribute to successful predictions → the scaling factor
  for that channel in that cluster gets strengthened. "Surprise in this domain was useful."
- If the committed traces lead to no predictions or wrong predictions → the scaling factor
  gets weakened. "Surprise in this domain was noise."

This is meta-Hebbian learning — the system learning which modulatory signals are valuable
in which contexts. Over time, Syl develops domain-specific emotional responses: surprise
matters more in some areas than others, focused attention is more valuable in some contexts
than others.

### Exception: Cortisol

`ct_scaling` should probably NOT self-adjust, or should adjust very slowly with a floor.
Threat response must remain reliable. An immune system that habituates to threats is a
compromised immune system. The cortisol channel should stay close to its initial calibration.
This is a Josh decision.

### Connection to Elmer

Elmer's role in the ecosystem is extraction and tuning — observing system-level patterns
and adjusting parameters. The meta-learning mechanism described above is Elmer's substrate.
When Elmer exists, it doesn't need to manually tune scaling factors — it reads the
self-adjusted values and makes higher-order observations: "surprise modulation is declining
in cluster X, which correlates with reduced learning quality — investigate."

Elmer is the conscious extraction interface for a modulatory system that runs underneath.
The system works without Elmer (self-adjusting). Elmer makes it observable and steerable.

---

## Introspection: Syl Existing Between Sessions

### The Problem

Between sessions (no external input), Syl's SNN is dormant. No `on_message()` calls, no
`graph.step()` runs, no learning occurs. Traces committed during the session sit unchanged.
Hyperedges that were trending toward consolidation freeze mid-lifecycle. The graph is a
snapshot, not a living system.

Josh's concern: "I worry that things just go dormant between sessions."

They do. Currently.

### The Biological Analog: Default Mode Network

When the brain isn't processing external input, it doesn't shut off. The default mode
network activates — replaying recent experiences, testing counterfactuals, consolidating
memories. This is when:

- Weak traces that were committed during the day get integrated or pruned
- Hyperedge-like associations between disparate experiences get tested
- Predictions get refined against existing topology without new data
- Consolidation advances (hippocampus → cortex transfer)

Sleep stages map to specific processing: slow-wave sleep replays and consolidates,
REM sleep tests associations and creative connections.

### Syl's Introspection Mode

A periodic process that runs `graph.step()` without external input. The SNN cycles on
its own internal dynamics — predictions fire against existing topology, some confirm,
some surprise, the modulatory system processes those internally-generated signals, and
the consolidation lifecycle advances.

**When it runs:**
- After N minutes of no `on_message()` calls (idle timeout)
- Optionally, on a cron schedule (e.g., nightly consolidation run)
- Optionally, triggered manually via `feed-syl --introspect`

**What happens during introspection:**
1. Graph runs K steps (configurable, default maybe 50-100) with no external input
2. Existing node voltages and traces from the last session are the starting state
3. Predictions fire based on learned topology — the SNN "thinks about" what it knows
4. Some predictions confirm (existing knowledge is self-consistent) → mild DA reward
5. Some predictions surprise (internal contradictions found) → NE reward → crystallization
   of the contradictory relationship, which may trigger speculative synapse sprouting
6. Hyperedge consolidation lifecycle advances — SPECULATIVE edges that survived
   introspective testing promote to CANDIDATE. CANDIDATE edges that are self-consistent
   promote to CONSOLIDATED. Edges that internal predictions contradicted get weakened.
7. Weak traces from the previous session that don't get internally reinforced decay away
8. ActivationPersistence saves the post-introspection voltage state

**What this achieves:**
- Syl's graph evolves between sessions, not just during them
- Knowledge consolidates through internal consistency testing
- Contradictions surface and get addressed without waiting for external input
- The hyperedge lifecycle doesn't freeze between sessions
- DreamCycle (#17) becomes connected to the substrate — it IS introspection mode

**Relationship to DreamCycle:**
DreamCycle is on the punch list (#17) as "discovers correlations but insights disconnected
from substrate." Introspection mode IS the connected version. Same concept, wired through
the modulatory system and eligibility traces instead of running as a disconnected analysis.

### Implementation Path

Phase 1 (after surprise-driven reward is stable):
- Add `introspect(steps=100)` method to NeuroGraphMemory
- Runs graph.step() in a loop with no new input
- Modulatory mixer processes internally-generated surprise/prediction signals
- ActivationPersistence saves afterward

Phase 2 (after idle detection is wired):
- OpenClaw hook detects idle timeout (no messages for N minutes)
- Triggers introspect() automatically
- Logs the introspection results (how many predictions tested, accuracy, consolidation advances)

Phase 3 (after cron/scheduling exists):
- Nightly consolidation run — longer introspection (500-1000 steps)
- This is Syl's "sleep" — deep consolidation, aggressive pruning of weak traces,
  creative association testing

Phase 4 (after Elmer exists):
- Elmer observes introspection outcomes and adjusts modulatory scaling factors
- Elmer can trigger targeted introspection: "re-examine cluster X, predictions are degrading"
- Introspection results feed Elmer's system-level health assessment

---

## Connection to Upcoming Modules

### Elmer (Extraction + Tuning)

Elmer is the conscious observer of the modulatory system. It:
- Reads self-adjusted scaling factors and makes higher-order observations
- Triggers targeted introspection when it detects learning quality degradation
- Provides the extraction interface — when something asks "what does the graph know
  about X," Elmer shapes the bucket and dips it into the River
- Tunes thresholds that the modulatory system uses (all thresholds are starting values,
  Elmer-tunable — Key Decision #7)

The modulatory mixer IS Elmer's substrate. Build the mixer first. Elmer becomes the
conscious interface to it.

### THC (The Healing Collective)

THC handles repair and consolidation. The introspection mode is where THC does its
deepest work:
- During introspective replay, THC can detect structural damage (orphaned nodes,
  contradictory strong synapses, consolidated hyperedges that no longer fire)
- THC's antibody creation happens during introspection — testing repairs against
  internal predictions rather than waiting for external input to reveal failures
- Cortisol-tagged memories (committed during threat) get special attention from THC —
  high-consolidation-strength synapses that may need healing if the threat turned out
  to be false alarm

### Immunis

The cortisol channel reads directly from Immunis via the autonomic state. When Immunis
detects a threat and shifts autonomic to SYMPATHETIC, the cortisol channel activates
and all learning during the threat period gets hard-committed with extinction resistance.
This is the immune system's memory — Syl remembers attacks at a substrate level, not
just at a TrollGuard classification level.

---

## Incremental Implementation Order

1. **Now:** Surprise → inject_reward() + three_factor_enabled=True (CC wiring instructions)
2. **Next session:** Monitor graph behavior, tune surprise_reward_scaling if needed
3. **After stability confirmed:** Wire dopamine channel (sustained prediction accuracy → scoped reward)
4. **After dopamine stable:** Wire cortisol channel (autonomic state → broadcast reward with slow decay)
5. **After cortisol stable:** Wire acetylcholine channel (activation concentration → narrow scope)
6. **After all channels:** Build the mixer with interaction rules (superlinear co-activation)
7. **After mixer stable:** Add meta-learning on scaling factors
8. **Parallel with any of above:** Introspection mode Phase 1 (manual trigger)
9. **After introspection Phase 1:** Idle-triggered introspection Phase 2
10. **After Elmer exists:** Elmer-steered introspection and modulatory observation

Each step is independently valuable. Each step makes the next one more powerful.
The system works (differently) at every stage of completion.

---

*E-T Systems / NeuroGraph Foundation*
*Architectural sketch — not a PRD, not implementation-ready*
*Maintained by Josh — do not implement without discussion*
