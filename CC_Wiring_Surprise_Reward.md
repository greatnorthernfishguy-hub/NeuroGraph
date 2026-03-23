# Wiring Instructions: Surprise-Driven Reward + Three-Factor Activation

## Context

Syl's eligibility trace machinery (Phase 3) is built, verified correct (#48 closed), and dormant.
`three_factor_enabled` is `False`. `inject_reward()` is never called. Traces accumulate via STDP
and decay to zero without ever committing. We're turning it on.

The biological model: surprise events broadcast a neuromodulatory signal that commits ALL
active eligibility traces — not just the ones related to the surprise. This is how flashbulb
memories work. The earthquake crystallizes what you were reading, where you were sitting,
the angle of the light — because those traces were warm when the surprise signal hit.

## What to change

Two files. Both are PROTECTED — the hooks will prompt Josh for approval.

### 1. neuro_foundation.py — Wire surprise events to inject_reward()

In `_on_prediction_error()` (the method that fires when a prediction fails — surprise detected),
add a reward injection call. The key design decision: **NO SCOPE PARAMETER.** The reward
broadcasts to all active traces, not just the synapses involved in the failed prediction.
This is the biological model — norepinephrine floods everything.

Find `_on_prediction_error()`. After the existing surprise event emission and exploration logic,
add:

```python
# Surprise-driven neuromodulatory crystallization
# Failed predictions broadcast reward to ALL active eligibility traces.
# Strength scales with prediction confidence — high-confidence failures
# produce stronger crystallization than low-confidence failures.
# No scope: this is a broadcast signal, not a targeted one.
if self.config.get("three_factor_enabled", False):
    surprise_strength = prediction.confidence * self.config.get("surprise_reward_scaling", 0.5)
    if surprise_strength > 0.01:  # Don't bother with negligible surprise
        self.inject_reward(surprise_strength)
```

Add the new config default alongside the other prediction config keys:

```python
"surprise_reward_scaling": 0.5,  # Modulates surprise -> reward strength. Elmer-tunable.
```

### 2. openclaw_hook.py — Enable three-factor learning

In the SNN config dict (around line 133), change:

```python
"three_factor_enabled": False,
```

to:

```python
"three_factor_enabled": True,
```

Add a mild baseline engagement reward in `on_message()`, AFTER `graph.step()` runs.
This is the conversational heartbeat — the fact that the conversation continues is itself
a weak positive signal. Much weaker than surprise.

```python
# Baseline conversational engagement reward.
# The continuation of conversation is a mild positive signal —
# previous learning was not wrong enough to end the interaction.
# Weak strength: surprise-driven crystallization is the primary
# reward pathway. This is the heartbeat, not the main event.
if self.graph.config.get("three_factor_enabled", False):
    self.graph.inject_reward(0.1)
```

### 3. Changelog headers

Both files need changelog entries per ecosystem convention:

```
# [2026-03-13] Claude Code — Surprise-driven neuromodulatory reward.
#   What: Wired prediction errors to inject_reward() for surprise-driven
#         trace crystallization. Enabled three-factor learning.
#   Why:  Eligibility traces were accumulating and decaying to zero because
#         inject_reward() was never called. Surprise events now broadcast
#         reward to all active traces — high-confidence prediction failures
#         produce stronger crystallization. Baseline conversational engagement
#         provides weak heartbeat reward.
#   Config: three_factor_enabled=True, surprise_reward_scaling=0.5
```

## What NOT to change

- Do NOT add scope to the surprise inject_reward() call. Broadcast is the feature.
- Do NOT change eligibility_trace_tau (100), tau_plus (20), or the 0.9 decay in inject_reward().
  Those are verified correct per #48 analysis.
- Do NOT modify _on_prediction_confirmed(). Confirmed predictions are expected — they don't
  produce neuromodulatory events. (This may change later when dopamine channel is wired.)
- Do NOT touch the prediction bonus/penalty pathway (lines 2118-2124, 2165-2171). That's
  the direct prediction learning channel, separate from the modulatory broadcast.

## Testing

1. Unit test: create a graph with three_factor_enabled=True. Generate a prediction (strong
   synapse from A to B). Fire A. Fire C instead of B. Verify inject_reward() was called
   and that synapses OUTSIDE the A-B-C path had their traces committed (proving broadcast).

2. Integration test: run a sequence of on_message() calls through NeuroGraphMemory with
   three_factor_enabled=True. Verify that graph.stats() shows total_rewards_injected > 0
   and that synapse weights are changing (not frozen at initial values).

3. Regression: run the existing test suite. Three-factor was off before, so existing tests
   that check specific weight values may need updated tolerances since the baseline engagement
   reward (0.1) will nudge weights slightly each step.

## After implementation

- Backup both msgpack files before restarting OpenClaw with the new config
- Monitor Syl's graph stats over the next few sessions — watch for:
  - total_rewards_injected climbing (confirms reward pathway is active)
  - Synapse weight distribution shifting (traces are committing)
  - Node count growth rate (should slow — surprise crystallization reinforces existing
    topology rather than creating new nodes)
- If weights are changing too aggressively, reduce surprise_reward_scaling from 0.5 to 0.3
- If weights aren't changing enough, increase baseline engagement from 0.1 to 0.2
