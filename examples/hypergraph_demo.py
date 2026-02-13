"""Hypergraph Engine demonstration.

Shows:
- Creating a hyperedge for syndrome detection
- Pattern completion: 4 of 5 symptoms → triggers 5th
- Hyperedge learning and adaptation
- Hierarchical hyperedges
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph, ActivationMode


def main():
    g = Graph({
        "default_threshold": 1.0,
        "decay_rate": 0.98,
        "refractory_period": 0,
        "he_member_weight_lr": 0.1,
        "he_threshold_lr": 0.02,
    })

    # ---- Syndrome Detection ----
    print("=== Syndrome Detection ===")
    symptoms = {
        "fever": g.create_node("fever").node_id,
        "cough": g.create_node("cough").node_id,
        "fatigue": g.create_node("fatigue").node_id,
        "aches": g.create_node("aches").node_id,
        "congestion": g.create_node("congestion").node_id,
    }
    diagnosis = g.create_node("flu_diagnosis")

    flu_syndrome = g.create_hyperedge(
        set(symptoms.values()),
        activation_threshold=0.6,
        output_targets=[diagnosis.node_id],
        output_weight=1.5,
        metadata={"label": "Flu Syndrome"},
    )
    flu_syndrome.pattern_completion_strength = 0.5

    print(f"Syndrome members: {list(symptoms.keys())}")
    print(f"Threshold: {flu_syndrome.activation_threshold}")
    print(f"Pattern completion: {flu_syndrome.pattern_completion_strength}")

    # ---- Pattern Completion ----
    print("\n=== Pattern Completion (4 of 5 symptoms) ===")
    # Present 4 symptoms (not congestion)
    for name in ["fever", "cough", "fatigue", "aches"]:
        g.stimulate(symptoms[name], 2.0)

    congestion_v_before = g.nodes[symptoms["congestion"]].voltage
    result = g.step()
    congestion_v_after = g.nodes[symptoms["congestion"]].voltage

    print(f"Fired hyperedges: {len(result.fired_hyperedge_ids)}")
    print(f"Flu syndrome fired: {flu_syndrome.hyperedge_id in result.fired_hyperedge_ids}")
    print(f"Congestion voltage: {congestion_v_before:.3f} → {congestion_v_after:.3f}")
    print(f"  (Pattern completion pre-charged the missing symptom)")

    # ---- Learning: Member Weight Adaptation ----
    print("\n=== Learning Over Time ===")
    print(f"Initial weights: { {k: f'{flu_syndrome.member_weights[v]:.2f}' for k, v in symptoms.items()} }")

    # Repeatedly present fever+cough+fatigue (core symptoms), not aches/congestion
    for _ in range(20):
        for name in ["fever", "cough", "fatigue"]:
            g.stimulate(symptoms[name], 2.0)
        g.step()

    print(f"After training:  { {k: f'{flu_syndrome.member_weights[v]:.2f}' for k, v in symptoms.items()} }")
    print("  (Core symptoms strengthened, peripheral ones weakened)")

    # ---- Threshold Learning ----
    print("\n=== Threshold Learning ===")
    t_before = flu_syndrome.activation_threshold
    # Positive reward
    for name in ["fever", "cough", "fatigue"]:
        g.stimulate(symptoms[name], 2.0)
    g.step()
    g.inject_reward(1.0)
    t_after = flu_syndrome.activation_threshold
    print(f"After reward: threshold {t_before:.3f} → {t_after:.3f} (more sensitive)")

    # ---- Hierarchical Hyperedges ----
    print("\n=== Hierarchical Hyperedges ===")
    respiratory = {
        "wheeze": g.create_node("wheeze").node_id,
        "shortness_of_breath": g.create_node("shortness_of_breath").node_id,
    }
    asthma_syndrome = g.create_hyperedge(
        set(respiratory.values()),
        activation_threshold=0.5,
        metadata={"label": "Asthma Syndrome"},
    )

    # Meta-hyperedge: "Respiratory Illness Category"
    resp_category = g.create_hierarchical_hyperedge(
        {flu_syndrome.hyperedge_id, asthma_syndrome.hyperedge_id},
        activation_threshold=0.3,
        metadata={"label": "Respiratory Illness Category"},
    )
    print(f"Level-0: Flu Syndrome ({len(flu_syndrome.member_nodes)} members)")
    print(f"Level-0: Asthma Syndrome ({len(asthma_syndrome.member_nodes)} members)")
    print(f"Level-1: Respiratory Category ({len(resp_category.member_nodes)} members, level={resp_category.level})")

    # ---- Telemetry ----
    print("\n=== Telemetry ===")
    tel = g.get_telemetry()
    print(f"Nodes: {tel.total_nodes}")
    print(f"Hyperedges: {tel.total_hyperedges}")
    print(f"Mean HE activation count: {tel.mean_he_activation_count:.1f}")

    # ---- Consolidation Demo ----
    print("\n=== Consolidation ===")
    # Create two near-identical hyperedges
    overlap_ids = [g.create_node(f"ov{i}").node_id for i in range(4)]
    he_x = g.create_hyperedge(set(overlap_ids[:3]))
    he_y = g.create_hyperedge(set(overlap_ids[:3]))  # Exact duplicate
    print(f"Before: {len(g.hyperedges)} hyperedges")
    merged = g.consolidate_hyperedges()
    print(f"After:  {len(g.hyperedges)} hyperedges (merged {merged})")


if __name__ == "__main__":
    main()
