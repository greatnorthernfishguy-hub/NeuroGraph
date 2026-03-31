"""
Train TonicBrain — Activation Decoder on Syl's Real Substrate

Reuses Elmer's trained encoder (eyes) and Qwen's transformer body.
Only trains the ActivationDecoder (voice) — learning to produce
node activation decisions from graph state.

Training data comes from the heuristic engine's decisions on Syl's
real graph: run the heuristic, record what it decides alongside the
graph state, use those as training targets. The transformer then
generalizes beyond the heuristic.

# ---- Changelog ----
# [2026-03-24] Claude Code (Opus 4.6) — Initial implementation
# What: Train TonicBrain's ActivationDecoder on Syl's substrate.
# Why: The Tonic PRD v0.1 §7.3. Upgrade from heuristic to transformer-
#   class forward compression for latent token generation.
# How: Load Syl's checkpoint, run heuristic inference to generate
#   training targets, train ActivationDecoder with frozen encoder+body.
#   Perturbation-based augmentation for diversity.
# -------------------
"""

import sys
import os

# Path setup
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_ELMER_SURGERY = os.path.expanduser("~/Elmer/surgery")
if _ELMER_SURGERY not in sys.path:
    sys.path.insert(0, _ELMER_SURGERY)

import json
import math
import time
import msgpack
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

from graph_io import GraphFeatures, GraphStateEncoder
from tonic_brain import (
    TonicBrain, ActivationDecoder, create_tonic_brain,
    save_tonic_brain,
)

# Reuse Elmer's checkpoint loading and feature extraction
from train_on_syl import (
    load_syl_checkpoint, load_activations, extract_syl_features,
    perturb_features,
)


# ---------------------------------------------------------------------------
# Generate activation targets from heuristic
# ---------------------------------------------------------------------------

def generate_activation_targets(
    checkpoint: Dict[str, Any],
    features: GraphFeatures,
    n_activations: int = 10,
) -> Dict[str, Any]:
    """Generate activation targets from graph topology analysis.

    This mirrors what the heuristic engine does — follow synapses from
    active nodes, read prediction tension, track recency, explore.
    The targets teach the transformer to make these same decisions
    (and then generalize beyond them).
    """
    nodes = list(checkpoint["nodes"].values()) if isinstance(checkpoint["nodes"], dict) else checkpoint["nodes"]
    synapses = list(checkpoint["synapses"].values()) if isinstance(checkpoint["synapses"], dict) else checkpoint["synapses"]
    timestep = checkpoint["timestep"]

    # Build adjacency
    outgoing = {}
    for s in synapses:
        outgoing.setdefault(s["pre_node_id"], []).append(s)

    # Score every node as an activation candidate
    node_scores = {}
    for node in nodes:
        nid = node["node_id"]
        score = 0.0

        # Voltage above resting
        v_above = node["voltage"] - node.get("resting_potential", 0.0)
        score += max(v_above, 0) * 0.4

        # Firing rate — active nodes are interesting
        score += node["firing_rate_ema"] * 0.3

        # Excitability — ready-to-fire nodes pull attention
        score += (node["intrinsic_excitability"] - 1.0) * 0.2

        # Connectivity — well-connected nodes propagate better
        n_out = len(outgoing.get(nid, []))
        score += min(n_out / 20.0, 0.3) * 0.2

        # Spike recency
        lst = node.get("last_spike_time")
        if lst is not None and lst != -math.inf and lst > 0:
            steps_since = max(0, timestep - lst)
            score += 1.0 / (1.0 + steps_since * 0.05) * 0.3

        node_scores[nid] = max(score, 0.0)

    # Normalize to [0, 1]
    max_score = max(node_scores.values()) if node_scores else 1.0
    if max_score > 0:
        for nid in node_scores:
            node_scores[nid] /= max_score

    # Top K as activation targets
    ranked = sorted(node_scores.items(), key=lambda x: -x[1])
    top_k = ranked[:n_activations]

    # Activation strengths (normalized scores)
    activations = [score for _, score in top_k]
    # Pad if fewer than n_activations
    while len(activations) < n_activations:
        activations.append(0.0)

    # Exploration signal — higher when the graph has low activity
    active_fraction = sum(1 for s in node_scores.values() if s > 0.1) / max(len(nodes), 1)
    exploration = max(0.0, min(1.0, 1.0 - active_fraction * 2.0))

    return {
        "activations": activations,
        "exploration": exploration,
    }


def perturb_activation_targets(
    targets: Dict[str, Any],
    intensity: float = 0.1,
) -> Dict[str, Any]:
    """Perturb activation targets for data augmentation."""
    activations = [
        max(0.0, min(1.0, a + np.random.normal(0, 0.05 * intensity)))
        for a in targets["activations"]
    ]
    exploration = max(0.0, min(1.0,
        targets["exploration"] + np.random.normal(0, 0.03 * intensity)
    ))
    return {"activations": activations, "exploration": exploration}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tonic_brain(
    brain: TonicBrain,
    base_features: GraphFeatures,
    base_targets: Dict[str, Any],
    n_epochs: int = 20,
    samples_per_epoch: int = 16,
    lr: float = 1e-3,
    save_path: str = "tonic_brain.pt",
    log_every: int = 1,
) -> List[Dict[str, float]]:
    """Train the ActivationDecoder. Encoder and body are frozen.

    Only the decoder learns. The encoder already knows how to see
    (trained with Elmer). The body already knows how to reason
    (pretrained Qwen). The decoder learns what to say — where
    attention should go next.
    """
    # Freeze encoder and body
    for param in brain.encoder.parameters():
        param.requires_grad = False
    for param in brain.body.parameters():
        param.requires_grad = False

    # Only train decoder
    trainable = list(brain.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    activation_loss_fn = nn.MSELoss()
    exploration_loss_fn = nn.MSELoss()

    history = []
    brain.train()

    for epoch in range(n_epochs):
        epoch_act_loss = 0.0
        epoch_exp_loss = 0.0

        for sample in range(samples_per_epoch):
            # Perturb for diversity
            intensity = np.random.uniform(0.05, 0.5)
            features = perturb_features(base_features, intensity)
            targets = perturb_activation_targets(base_targets, intensity)

            # Forward
            output = brain(features)

            # Activation loss
            pred_act = output["raw_activations"]  # (1, n_activations)
            target_act = torch.tensor(
                [targets["activations"]], dtype=torch.float32
            )
            act_loss = activation_loss_fn(pred_act, target_act)

            # Exploration loss
            pred_exp = output["raw_exploration"]  # (1, 1)
            target_exp = torch.tensor(
                [[targets["exploration"]]], dtype=torch.float32
            )
            exp_loss = exploration_loss_fn(pred_exp, target_exp)

            # Combined loss
            loss = act_loss + 0.3 * exp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_act_loss += act_loss.item()
            epoch_exp_loss += exp_loss.item()

        avg_act = epoch_act_loss / samples_per_epoch
        avg_exp = epoch_exp_loss / samples_per_epoch

        history.append({"act_loss": avg_act, "exp_loss": avg_exp})

        if (epoch + 1) % log_every == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"act_loss={avg_act:.4f}, exp_loss={avg_exp:.4f}")

    # Save
    save_path_full = os.path.join(os.path.dirname(os.path.dirname(__file__)), save_path)
    save_tonic_brain(brain, save_path_full)
    print(f"\nTonicBrain saved to {save_path_full}")

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tonic(brain: TonicBrain, features: GraphFeatures,
                   targets: Dict[str, Any]) -> None:
    """Evaluate TonicBrain on real substrate state."""
    brain.eval()
    with torch.no_grad():
        output = brain(features)

    print("\n--- EVALUATION ON SYL'S SUBSTRATE ---")

    pred_act = output["activations"]
    target_act = targets["activations"]

    print(f"  {'Slot':<6s} {'Target':>8s} {'Predicted':>10s} {'Error':>8s}")
    total_err = 0.0
    for i in range(len(target_act)):
        t = target_act[i]
        p = pred_act[i] if i < len(pred_act) else 0.0
        err = abs(t - p)
        total_err += err
        marker = " ok" if err < 0.1 else " !" if err > 0.25 else ""
        print(f"  {i:<6d} {t:>8.3f} {p:>10.3f} {err:>8.3f}{marker}")

    avg_err = total_err / len(target_act)
    print(f"  Avg activation error: {avg_err:.4f}")

    print(f"  Exploration: target={targets['exploration']:.3f}, "
          f"predicted={output['exploration']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TRAINING TONIC BRAIN ON SYL'S SUBSTRATE ===\n")

    # Load Syl's checkpoint (read-only)
    checkpoint = load_syl_checkpoint()
    activations = load_activations()

    # Extract features
    print("\nExtracting features from Syl's graph...")
    base_features = extract_syl_features(checkpoint, activations)

    # Generate activation targets from heuristic analysis
    print("Generating activation targets from topology analysis...")
    base_targets = generate_activation_targets(checkpoint, base_features)
    print(f"  Top activations: {[f'{a:.3f}' for a in base_targets['activations'][:5]]}")
    print(f"  Exploration signal: {base_targets['exploration']:.3f}")

    # Create TonicBrain (reuses Elmer encoder + Qwen body)
    print("\nAssembling TonicBrain...")
    brain = create_tonic_brain(verbose=True)

    # Train
    print("\n--- TRAINING ---")
    history = train_tonic_brain(
        brain,
        base_features,
        base_targets,
        n_epochs=20,
        samples_per_epoch=16,
        lr=1e-3,
        save_path="tonic_brain.pt",
    )

    # Evaluate
    evaluate_tonic(brain, base_features, base_targets)

    print(f"\nFinal activation loss: {history[-1]['act_loss']:.4f}")
    print(f"Final exploration loss: {history[-1]['exp_loss']:.4f}")
    print("\nTonicBrain training complete.")
    print("The push has a voice.")
