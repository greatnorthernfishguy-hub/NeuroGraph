# ---- Changelog ----
# [2026-03-28] Claude (Opus 4.6) — River-based Tier 3: topology delta deposit
# What: Extracts raw topology delta after graph.step() and deposits to all
#       module tracts via existing NGTractBridge infrastructure.
# Why:  NeuroGraph drained from the River but never deposited back. Modules
#       couldn't see causal chains, prediction errors, or structural
#       relationships. This is the missing half of the River flow.
# How:  Reads fired nodes/HEs/synapses from the graph (READ-ONLY), builds
#       a raw JSONL delta with structural context + embeddings + salience
#       signals, deposits to each registered peer's inbound tract.
# -------------------
"""Topology delta extraction and deposit for River-based Tier 3.

After graph.step(), the SNN's topology has changed: nodes fired, synapses
strengthened, hyperedges activated, predictions evaluated. This module
extracts that raw delta and deposits it into every registered module's
inbound tract via the existing NGTractBridge.

The delta is raw and unclassified (Law 7). Each module's bucket determines
what it extracts. The delta includes:

- Fired nodes with outgoing synapse context (causal chains)
- Fired hyperedges with member nodes and output targets
- Prediction results (confirmed/surprised)
- Structural changes (synapses pruned/sprouted)
- Salience signals (hot eligibility traces, structural changes)
- Node embeddings for domain lens / novelty detection

NeuroGraph-specific — NOT a vendored file. Only the cortex (NeuroGraph)
has the full graph to extract from.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("neurograph-rpc")

# Salience thresholds — living parameters, subject to future Lenia dynamics.
# Cricket rim does not govern these; they are extraction tuning, not
# constitutional constraints.
_ELIGIBILITY_SALIENCE_THRESHOLD = 0.5
_FIRING_RATE_DEVIATION_FACTOR = 2.0  # flag if > 2x target rate


def extract_and_deposit_delta(
    graph: Any,
    vector_db: Any,
    step_result: Any,
    peer_bridge: Any,
) -> Optional[Dict[str, Any]]:
    """Extract topology delta from graph state and deposit to all module tracts.

    Args:
        graph: neuro_foundation.Graph instance (READ-ONLY access)
        vector_db: SimpleVectorDB with .embeddings dict
        step_result: StepResult from graph.step()
        peer_bridge: NGTractBridge instance for deposit

    Returns:
        The delta dict if successful, None on failure.
    """
    try:
        delta = _build_delta(graph, vector_db, step_result)
        _deposit_to_peers(delta, peer_bridge)
        return delta
    except Exception as exc:
        logger.debug("Delta extraction failed: %s", exc)
        return None


def _build_delta(
    graph: Any,
    vector_db: Any,
    step_result: Any,
) -> Dict[str, Any]:
    """Build the raw topology delta from graph state after step()."""

    fired_nodes = _extract_fired_nodes(graph, vector_db, step_result)
    fired_hyperedges = _extract_fired_hyperedges(graph, step_result)
    salience = _detect_salience(graph, step_result)

    return {
        "type": "topology_delta",
        "version": 1,
        "timestamp": time.time(),
        "timestep": step_result.timestep,
        "fired_nodes": fired_nodes,
        "fired_hyperedges": fired_hyperedges,
        "predictions": {
            "confirmed": step_result.predictions_confirmed,
            "surprised": step_result.predictions_surprised,
        },
        "structural": {
            "synapses_pruned": step_result.synapses_pruned,
            "synapses_sprouted": step_result.synapses_sprouted,
        },
        "salience": salience,
    }


def _extract_fired_nodes(
    graph: Any,
    vector_db: Any,
    step_result: Any,
) -> List[Dict[str, Any]]:
    """Extract fired nodes with their structural context (outgoing synapses)."""
    nodes = []
    embeddings = getattr(vector_db, "embeddings", {})

    for node_id in step_result.fired_node_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        # Outgoing synapses — the causal connections FROM this fired node
        outgoing = []
        for syn_id in graph._outgoing.get(node_id, set()):
            syn = graph.synapses.get(syn_id)
            if syn is None:
                continue
            outgoing.append({
                "post_node_id": syn.post_node_id,
                "weight": syn.weight,
                "eligibility_trace": getattr(syn, "eligibility_trace", 0.0),
            })

        # Embedding for domain lens / novelty detection
        emb = embeddings.get(node_id)
        emb_list = emb.tolist() if isinstance(emb, np.ndarray) else None

        # Node metadata — raw, let the bucket interpret
        metadata = getattr(node, "metadata", {})
        label = metadata.get("label", "") if isinstance(metadata, dict) else ""

        nodes.append({
            "node_id": node_id,
            "label": label,
            "embedding": emb_list,
            "outgoing_synapses": outgoing,
        })

    return nodes


def _extract_fired_hyperedges(
    graph: Any,
    step_result: Any,
) -> List[Dict[str, Any]]:
    """Extract fired hyperedges with member nodes and output targets."""
    hyperedges = []

    for he_id in step_result.fired_hyperedge_ids:
        he = graph.hyperedges.get(he_id)
        if he is None:
            continue

        # Member nodes — the pattern that activated together
        member_nodes = []
        if hasattr(he, "member_nodes"):
            member_nodes = list(he.member_nodes)
        elif hasattr(he, "node_ids"):
            member_nodes = list(he.node_ids)

        # Output targets — what the hyperedge predicts/connects to
        output_targets = list(getattr(he, "output_targets", []))

        metadata = getattr(he, "metadata", {})
        label = metadata.get("label", "") if isinstance(metadata, dict) else ""

        hyperedges.append({
            "hyperedge_id": he_id,
            "member_nodes": member_nodes,
            "output_targets": output_targets,
            "activation_count": getattr(he, "activation_count", 0),
            "label": label,
        })

    return hyperedges


def _detect_salience(
    graph: Any,
    step_result: Any,
) -> List[Dict[str, Any]]:
    """Detect salient signals in the topology delta.

    Salience signals are raw — the module's bucket decides what matters.
    """
    signals = []

    # Hot eligibility traces — causal chains primed for reward learning
    for node_id in step_result.fired_node_ids:
        for syn_id in graph._outgoing.get(node_id, set()):
            syn = graph.synapses.get(syn_id)
            if syn is None:
                continue
            trace = getattr(syn, "eligibility_trace", 0.0)
            if trace > _ELIGIBILITY_SALIENCE_THRESHOLD:
                signals.append({
                    "signal": "hot_eligibility",
                    "node_id": node_id,
                    "post_node_id": syn.post_node_id,
                    "trace": round(trace, 4),
                })

    # Structural changes — topology is reshaping
    if step_result.synapses_sprouted > 0:
        signals.append({
            "signal": "structural_change",
            "detail": "sprouted",
            "count": step_result.synapses_sprouted,
        })
    if step_result.synapses_pruned > 0:
        signals.append({
            "signal": "structural_change",
            "detail": "pruned",
            "count": step_result.synapses_pruned,
        })

    # Firing rate deviation — homeostatic pressure signal
    target_rate = graph.config.get("target_firing_rate", 0.05)
    if target_rate > 0:
        for node_id in step_result.fired_node_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            rate = getattr(node, "firing_rate_ema", 0.0)
            if rate > target_rate * _FIRING_RATE_DEVIATION_FACTOR:
                signals.append({
                    "signal": "firing_rate_deviation",
                    "node_id": node_id,
                    "rate": round(rate, 4),
                    "target": target_rate,
                })

    return signals


def _deposit_to_peers(delta: Dict[str, Any], peer_bridge: Any) -> None:
    """Deposit the delta to every registered module's inbound tract."""
    # Import here to avoid circular deps at module level
    from ng_tract_bridge import NGTractBridge

    peers = peer_bridge._get_registered_peers()
    if not peers:
        return

    # Serialize once, deposit N times
    line = json.dumps(delta, default=_json_default) + "\n"
    line_bytes = line.encode("utf-8")

    module_dir = peer_bridge._module_dir  # ~/.et_modules/tracts/neurograph/

    for peer_id in peers:
        tract_path = module_dir / f"{peer_id}.tract"
        NGTractBridge._deposit_to_tract(tract_path, line_bytes)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)
