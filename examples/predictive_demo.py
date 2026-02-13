#!/usr/bin/env python3
"""
Predictive Coding Engine Demo (Phase 3)

Demonstrates:
    1. Training a causal sequence (A→B→C)
    2. Introducing a prediction error (break the sequence)
    3. Watching the system discover an alternative pathway
    4. Using reward signals to guide learning
    5. Three-factor learning convergence

Reference: NeuroGraph Foundation PRD v1.0, Section 5.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ===========================================================================
# Demo 1: Training a Causal Sequence
# ===========================================================================

def demo_causal_sequence():
    separator("Demo 1: Training a Causal Sequence (A -> B -> C)")

    g = Graph(config={
        "prediction_threshold": 1.5,
        "prediction_window": 15,
        "prediction_chain_decay": 0.7,
        "prediction_max_chain_depth": 3,
        "prediction_pre_charge_factor": 0.3,
        "grace_period": 50000,
        "inactivity_threshold": 50000,
        "scaling_interval": 100000,
        # Higher learning rate and stronger LTP for demo convergence
        "learning_rate": 0.1,
        "A_plus": 1.2,
        "A_minus": 1.0,
    })

    # Create nodes
    for nid in ["A", "B", "C"]:
        g.create_node(node_id=nid)

    # Create initial weak synapses
    g.create_synapse("A", "B", weight=0.5)
    g.create_synapse("B", "C", weight=0.5)

    print("Initial synapse weights:")
    syn_ab = g._find_synapse("A", "B")
    syn_bc = g._find_synapse("B", "C")
    print(f"  A->B: {syn_ab.weight:.4f}")
    print(f"  B->C: {syn_bc.weight:.4f}")

    # Train: fire A, then quickly fire B (within 1-2 steps), then fire C.
    # The key for LTP is that pre fires BEFORE post with small Δt.
    print("\nTraining A->B->C sequence (200 iterations)...")
    for i in range(200):
        # Fire A
        g.stimulate("A", 2.0)
        g.step()
        # Fire B right after A (Δt=1 → strong LTP for A→B)
        g.stimulate("B", 2.0)
        g.step()
        # Fire C right after B (Δt=1 → strong LTP for B→C)
        g.stimulate("C", 2.0)
        g.step()
        # Cool-down: wait for refractory to clear
        for _ in range(5):
            g.step()

    print("\nAfter training:")
    print(f"  A->B: {syn_ab.weight:.4f}")
    print(f"  B->C: {syn_bc.weight:.4f}")

    # Test prediction
    if syn_ab.weight >= g.config["prediction_threshold"]:
        g.stimulate("A", 2.0)
        g.step()
        preds = g.get_predictions()
        print(f"\nPredictions after stimulating A:")
        for p in preds:
            print(f"  -> {p.target_node_id} "
                  f"(strength={p.strength:.3f}, "
                  f"confidence={p.confidence:.3f}, "
                  f"depth={p.chain_depth})")

        if any(p.target_node_id == "B" for p in preds):
            print("\n  [OK] System correctly predicts B after A fires!")
        if any(p.target_node_id == "C" for p in preds):
            print("  [OK] System predicts C through chain A->B->C!")
    else:
        print(f"\n  Weights not yet above threshold "
              f"({g.config['prediction_threshold']}) - more training needed")
        print("  (STDP asymmetry causes slow strengthening with default params)")

    return g


# ===========================================================================
# Demo 2: Prediction Error and Alternative Discovery
# ===========================================================================

def demo_prediction_error():
    separator("Demo 2: Prediction Error and Alternative Pathway Discovery")

    g = Graph(config={
        "prediction_threshold": 3.0,
        "prediction_window": 5,
        "prediction_chain_decay": 0.7,
        "prediction_pre_charge_factor": 0.3,
        "prediction_error_penalty": 0.02,
        "surprise_sprouting_weight": 0.1,
        "grace_period": 50000,
        "inactivity_threshold": 50000,
        "scaling_interval": 100000,
    })

    # Create trained A->B chain, but B has a very high threshold
    # so the propagated spike from A won't cause B to fire.
    # This simulates B being "blocked" — prediction is made but fails.
    for nid in ["A", "B", "D"]:
        g.create_node(node_id=nid)
    g.nodes["B"].threshold = 100.0  # B won't fire from propagation alone
    g.create_synapse("A", "B", weight=4.0)

    syn_ab = g._find_synapse("A", "B")
    initial_ab = syn_ab.weight
    print(f"Trained A->B weight: {initial_ab:.4f}")
    print(f"B threshold raised to 100.0 (simulating blocked path)")
    print(f"D exists but no A->D synapse yet")

    # Track events
    surprise_events = []
    g.register_event_handler("prediction_error",
                             lambda **kw: surprise_events.append(kw))
    sprout_events = []
    g.register_event_handler("surprise_sprouted",
                             lambda **kw: sprout_events.append(kw))

    # Fire A (predicts B), but fire D instead
    print("\nFiring A (predicts B)...")
    g.stimulate("A", 2.0)
    g.step()

    preds = g.get_predictions()
    print(f"Active predictions: {len(preds)}")
    for p in preds:
        print(f"  Expecting: {p.target_node_id} (confidence={p.confidence:.3f})")

    # Fire D (surprise!)
    print("\nFiring D instead of B (surprise!)...")
    g.stimulate("D", 2.0)
    g.step()

    # Let prediction for B expire
    print("Waiting for B prediction to expire...")
    for _ in range(10):
        g.step()

    print(f"\nResults:")
    print(f"  Prediction errors: {len(surprise_events)}")
    print(f"  A->B weight: {syn_ab.weight:.4f} (was {initial_ab:.4f})")
    if syn_ab.weight < initial_ab:
        print("  [OK] A->B weight weakened by prediction error!")

    syn_ad = g._find_synapse("A", "D")
    if syn_ad:
        print(f"  [OK] A->D created! weight={syn_ad.weight:.4f} "
              f"(mode={syn_ad.metadata.get('creation_mode', 'n/a')})")
    else:
        print("  A->D not created (D may not have fired in the right window)")

    if sprout_events:
        print(f"  Surprise sprouting events: {len(sprout_events)}")


# ===========================================================================
# Demo 3: Three-Factor Learning with Reward
# ===========================================================================

def demo_three_factor_learning():
    separator("Demo 3: Three-Factor Learning (STDP + Reward)")

    g = Graph(config={
        "three_factor_enabled": True,
        "prediction_threshold": 3.0,
        "eligibility_trace_tau": 100,
        "grace_period": 50000,
        "inactivity_threshold": 50000,
        "scaling_interval": 100000,
    })

    g.create_node(node_id="A")
    g.create_node(node_id="B")
    syn = g.create_synapse("A", "B", weight=1.0)

    print(f"Initial weight: {syn.weight:.4f}")
    print(f"Three-factor learning: ENABLED")
    print(f"STDP creates eligibility traces, reward commits changes")

    # Create causal pairing (A fires before B)
    print("\n--- Trial 1: Causal pairing A->B ---")
    g.stimulate("A", 2.0)
    g.step()
    g.step()
    g.stimulate("B", 2.0)
    g.step()

    print(f"After STDP: weight={syn.weight:.4f}, trace={syn.eligibility_trace:.6f}")
    print("  (In three-factor mode, weight change is deferred to trace)")

    # Inject positive reward
    print("\n--- Injecting positive reward (1.0) ---")
    g.inject_reward(1.0)
    print(f"After reward: weight={syn.weight:.4f}, trace={syn.eligibility_trace:.6f}")

    # Repeat with negative reward
    print("\n--- Trial 2: Another causal pairing ---")
    for _ in range(5):
        g.step()  # cool-down
    g.stimulate("A", 2.0)
    g.step()
    g.step()
    g.stimulate("B", 2.0)
    g.step()

    weight_before = syn.weight
    print(f"Before negative reward: weight={syn.weight:.4f}, "
          f"trace={syn.eligibility_trace:.6f}")

    g.inject_reward(-1.0)
    print(f"After negative reward: weight={syn.weight:.4f}, "
          f"trace={syn.eligibility_trace:.6f}")

    if syn.weight < weight_before:
        print("  [OK] Negative reward weakened the connection!")

    # Demonstrate scoped reward
    print("\n--- Scoped reward demonstration ---")
    g.create_node(node_id="C")
    g.create_node(node_id="D")
    syn_cd = g.create_synapse("C", "D", weight=1.0)

    # Create traces on both
    g.stimulate("A", 2.0)
    g.stimulate("C", 2.0)
    g.step()
    g.step()
    g.stimulate("B", 2.0)
    g.stimulate("D", 2.0)
    g.step()

    w_ab = syn.weight
    w_cd = syn_cd.weight
    print(f"A->B weight: {w_ab:.4f}, C->D weight: {w_cd:.4f}")

    g.inject_reward(1.0, scope={"A", "B"})
    print(f"After scoped reward (A,B only):")
    print(f"  A->B weight: {syn.weight:.4f} (changed: {syn.weight != w_ab})")
    print(f"  C->D weight: {syn_cd.weight:.4f} (changed: {syn_cd.weight != w_cd})")


# ===========================================================================
# Demo 4: Convergence Over Time
# ===========================================================================

def demo_convergence():
    separator("Demo 4: Prediction Accuracy Convergence")

    g = Graph(config={
        "prediction_threshold": 2.0,
        "prediction_window": 10,
        "prediction_chain_decay": 0.7,
        "prediction_pre_charge_factor": 0.3,
        "prediction_confirm_bonus": 0.01,
        "prediction_error_penalty": 0.02,
        "learning_rate": 0.1,
        "A_plus": 1.2,
        "A_minus": 1.0,
        "grace_period": 50000,
        "inactivity_threshold": 50000,
        "scaling_interval": 100000,
    })

    for nid in ["A", "B", "C"]:
        g.create_node(node_id=nid)
    g.create_synapse("A", "B", weight=0.5)
    g.create_synapse("B", "C", weight=0.5)

    print("Training A->B->C with prediction tracking...")
    print(f"{'Epoch':>6} | {'A->B wt':>8} | {'B->C wt':>8} | "
          f"{'Preds Made':>10} | {'Confirmed':>9} | {'Errors':>6} | {'Accuracy':>8}")
    print("-" * 75)

    for epoch in range(30):
        # Train sequence with tight causal timing for strong LTP
        for _ in range(20):
            # Fire A then B then C with small Δt
            g.stimulate("A", 2.0)
            g.step()
            g.stimulate("B", 2.0)
            g.step()
            g.stimulate("C", 2.0)
            g.step()
            # Cool-down
            for _ in range(5):
                g.step()

        tel = g.get_telemetry()
        syn_ab = g._find_synapse("A", "B")
        syn_bc = g._find_synapse("B", "C")

        w_ab = syn_ab.weight if syn_ab else 0
        w_bc = syn_bc.weight if syn_bc else 0

        print(f"{epoch+1:>6} | {w_ab:>8.4f} | {w_bc:>8.4f} | "
              f"{tel.total_predictions_made:>10} | "
              f"{tel.total_predictions_confirmed:>9} | "
              f"{tel.total_predictions_errors:>6} | "
              f"{tel.prediction_accuracy:>7.1%}")

    print(f"\nFinal telemetry:")
    tel = g.get_telemetry()
    print(f"  Total predictions made: {tel.total_predictions_made}")
    print(f"  Confirmed: {tel.total_predictions_confirmed}")
    print(f"  Errors: {tel.total_predictions_errors}")
    print(f"  Accuracy: {tel.prediction_accuracy:.1%}")
    print(f"  Novel sequences: {tel.total_novel_sequences}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    demo_causal_sequence()
    demo_prediction_error()
    demo_three_factor_learning()
    demo_convergence()

    print(f"\n{'='*60}")
    print("  All demos complete!")
    print(f"{'='*60}")
