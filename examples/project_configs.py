"""Project-specific configuration examples for NeuroGraph Foundation.

Shows how to tune the graph for different use cases:
- OpenClaw (autonomous coding agent)
- DSM diagnostic reasoning
- Consciousness emergence framework

Reference: PRD v1.0 §9, Addendum v1.1 §3.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph


# ---- OpenClaw: Autonomous Vibe-Coding Agent ----
OPENCLAW_CONFIG = {
    "decay_rate": 0.93,          # Slightly faster decay for responsiveness
    "default_threshold": 0.8,    # Lower threshold for sensitivity
    "refractory_period": 2,
    "tau_plus": 15.0,            # Tighter causal window for code patterns
    "tau_minus": 15.0,
    "A_plus": 1.0,
    "A_minus": 1.15,             # Moderate pruning bias
    "learning_rate": 0.02,       # Faster learning for code conventions
    "max_weight": 5.0,
    "target_firing_rate": 0.06,
    "scaling_interval": 50,      # More frequent homeostasis
    "weight_threshold": 0.02,
    "grace_period": 300,         # Shorter grace for outdated API knowledge
    "inactivity_threshold": 500,
    "co_activation_window": 3,
    "initial_sprouting_weight": 0.15,
}


# ---- DSM-style Diagnostic Reasoning ----
DSM_CONFIG = {
    "decay_rate": 0.97,          # Longer memory for symptom tracking
    "default_threshold": 1.2,    # Higher threshold for precision
    "refractory_period": 2,
    "tau_plus": 25.0,            # Wider window for symptom co-occurrence
    "tau_minus": 25.0,
    "A_plus": 0.8,
    "A_minus": 1.0,              # Conservative pruning
    "learning_rate": 0.005,      # Slow, deliberate learning
    "max_weight": 5.0,
    "target_firing_rate": 0.03,  # Lower activity for precision
    "scaling_interval": 200,     # Less frequent adjustments
    "weight_threshold": 0.005,
    "grace_period": 1000,        # Long grace for rare symptoms
    "inactivity_threshold": 2000,
    "co_activation_window": 10,  # Wide window for syndrome detection
    "initial_sprouting_weight": 0.05,
}


# ---- Consciousness Emergence Framework ----
CONSCIOUSNESS_CONFIG = {
    "decay_rate": 0.96,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "tau_plus": 30.0,            # Wide window for cross-domain associations
    "tau_minus": 30.0,
    "A_plus": 1.0,
    "A_minus": 1.1,              # Mild pruning to allow exploration
    "learning_rate": 0.008,
    "max_weight": 5.0,
    "target_firing_rate": 0.04,
    "scaling_interval": 150,
    "weight_threshold": 0.01,
    "grace_period": 800,
    "inactivity_threshold": 1500,
    "co_activation_window": 8,   # Wide for analogy detection
    "initial_sprouting_weight": 0.08,
}


def demo_config(name: str, config: dict) -> None:
    """Run a brief simulation with the given config."""
    g = Graph(config)
    print(f"\n=== {name} ===")
    print(f"Config: decay={config['decay_rate']}, "
          f"threshold={config['default_threshold']}, "
          f"lr={config['learning_rate']}, "
          f"tau={config['tau_plus']}")

    # Create a small test network
    for i in range(20):
        g.create_node(f"n{i}")
    for i in range(15):
        g.create_synapse(f"n{i}", f"n{i+5}", weight=0.3)

    # Run 1000 steps with sparse input
    import numpy as np
    rng = np.random.RandomState(42)
    for _ in range(1000):
        for nid in rng.choice([f"n{i}" for i in range(20)], size=3, replace=False):
            g.stimulate(nid, rng.random() * 0.5)
        g.step()

    tel = g.get_telemetry()
    print(f"After 1000 steps:")
    print(f"  Firing rate: {tel.global_firing_rate:.4f} (target: {config['target_firing_rate']})")
    print(f"  Mean weight: {tel.mean_weight:.3f}")
    print(f"  Synapses: {tel.total_synapses} (pruned: {tel.total_pruned}, sprouted: {tel.total_sprouted})")


if __name__ == "__main__":
    demo_config("OpenClaw", OPENCLAW_CONFIG)
    demo_config("DSM Diagnostic", DSM_CONFIG)
    demo_config("Consciousness", CONSCIOUSNESS_CONFIG)
