"""Simple usage example for NeuroGraph Foundation.

Demonstrates creating a small graph, running simulation, and observing STDP learning.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph, ActivationMode


def main():
    # Create a graph with default configuration
    g = Graph()

    # Create nodes
    a = g.create_node("sensor_A", metadata={"type": "input"})
    b = g.create_node("sensor_B", metadata={"type": "input"})
    c = g.create_node("hidden_C", metadata={"type": "hidden"})
    d = g.create_node("output_D", metadata={"type": "output"})

    # Create synapses (A→C, B→C, C→D)
    syn_ac = g.create_synapse("sensor_A", "hidden_C", weight=0.5, delay=1)
    syn_bc = g.create_synapse("sensor_B", "hidden_C", weight=0.5, delay=1)
    syn_cd = g.create_synapse("hidden_C", "output_D", weight=0.5, delay=1)

    print("=== Initial State ===")
    print(f"A→C weight: {syn_ac.weight:.3f}")
    print(f"B→C weight: {syn_bc.weight:.3f}")
    print(f"C→D weight: {syn_cd.weight:.3f}")

    # Train: repeatedly fire A then C (causal), but not B
    print("\n=== Training: A→C causal pairing (50 rounds) ===")
    for i in range(50):
        g.stimulate("sensor_A", 1.5)
        g.step()
        g.step()  # Let spike propagate
        g.stimulate("hidden_C", 1.5)
        g.step()
        # Cool down
        g.step_n(3)

    print(f"A→C weight: {syn_ac.weight:.3f} (should increase)")
    print(f"B→C weight: {syn_bc.weight:.3f} (should stay similar)")
    print(f"C→D weight: {syn_cd.weight:.3f}")

    # Create a hyperedge grouping A, B, C
    print("\n=== Hyperedge Demo ===")
    he = g.create_hyperedge(
        {"sensor_A", "sensor_B", "hidden_C"},
        activation_threshold=0.6,
        output_targets=["output_D"],
        output_weight=1.0,
    )

    # Fire A and C (2/3 = 67% > 60% threshold) → hyperedge fires
    g.stimulate("sensor_A", 2.0)
    g.stimulate("hidden_C", 2.0)
    result = g.step()
    print(f"Fired nodes: {result.fired_node_ids}")
    print(f"Fired hyperedges: {result.fired_hyperedge_ids}")

    # Telemetry
    print("\n=== Telemetry ===")
    tel = g.get_telemetry()
    print(f"Nodes: {tel.total_nodes}")
    print(f"Synapses: {tel.total_synapses}")
    print(f"Hyperedges: {tel.total_hyperedges}")
    print(f"Global firing rate: {tel.global_firing_rate:.4f}")
    print(f"Mean weight: {tel.mean_weight:.3f}")

    # Checkpoint
    print("\n=== Checkpoint ===")
    g.checkpoint("/tmp/neurograph_example.json")
    print("Saved to /tmp/neurograph_example.json")

    g2 = Graph()
    g2.restore("/tmp/neurograph_example.json")
    print(f"Restored graph: {g2.get_telemetry().total_nodes} nodes, "
          f"{g2.get_telemetry().total_synapses} synapses")


if __name__ == "__main__":
    main()
