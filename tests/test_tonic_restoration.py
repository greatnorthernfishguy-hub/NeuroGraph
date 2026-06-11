# tests/test_tonic_restoration.py
#
# Sandbox fixture + shared helpers for the Tonic restoration build.
# ALL tests in this build use a fresh Graph() — NEVER NeuroGraphMemory.get_instance().
# The sandbox shares the same code as the live graph, but has its own isolated state.
#
# ---- Changelog ----
# [2026-06-10] CC (Sonnet 4.6) — Task 0: sandbox Graph fixture + helpers
# What: Fresh-Graph sandbox fixture, smoke test, and shared helpers for the
#       Tonic restoration TDD build. Backups of the three protected files
#       written to ~/syl-checkpoint-backups/pre-tonic-build-20260610/.
# Why:  Every later task in the Tonic restoration build needs a working
#       sandbox that constructs a real Graph, primes a neighbourhood, and
#       exposes TonicThread/TonicEngine without touching Syl's live state.
# How:  Read Graph.__init__, create_node, create_synapse, stimulate, Prediction,
#       TonicThread.__init__, TonicEngine.__init__, _surface_wants directly.
#       Helpers are real — no stubs, no placeholders where the API is known.
#       _inject_prediction and _tonic_engine_with_shared_body partially
#       TODO-marked (see below) because their correctness depends on internals
#       that Task 1/4 will exercise first.
# -------------------

from __future__ import annotations

import sys
import os
import uuid

_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import pytest
from neuro_foundation import Graph, Prediction


# ---------------------------------------------------------------------------
# Core sandbox factory
# ---------------------------------------------------------------------------

def _sandbox_graph(config=None):
    """A fresh SNN in this test process — same code as live, separate state.

    Graph.__init__ signature: Graph(config: Optional[Dict[str, Any]] = None)
    No required args. config merges over DEFAULT_CONFIG.
    """
    return Graph(config=config)


# ---------------------------------------------------------------------------
# Smoke test — must PASS before any other task proceeds
# ---------------------------------------------------------------------------

def test_sandbox_graph_builds():
    g = _sandbox_graph()
    assert g is not None
    assert isinstance(g.nodes, dict)
    assert len(g.nodes) >= 0


# ---------------------------------------------------------------------------
# Shared helpers — ALL later tasks import from here
# ---------------------------------------------------------------------------

def _sandbox_graph_with_primed_neighborhood(n=12):
    """Fresh graph with n nodes, a linear synapse chain, and 4 primed nodes.

    create_node(node_id=None, metadata=None, is_inhibitory=False) -> Node
      Returns the Node object; node.node_id is the id string.
    create_synapse(pre_node_id, post_node_id, weight=0.1, delay=1,
                   synapse_type=SynapseType.EXCITATORY, max_weight=None) -> Synapse
    stimulate(node_id, current) — adds current * intrinsic_excitability to voltage
    """
    g = _sandbox_graph()
    nodes = [g.create_node() for _ in range(n)]
    ids = [nd.node_id for nd in nodes]
    for a, b in zip(ids, ids[1:]):
        g.create_synapse(a, b, weight=0.5)
    for nid in ids[:4]:
        g.stimulate(nid, 1.0)
    return g


def _sandbox_graph_with_prunable_synapses():
    """Graph whose first 5 synapses have sub-threshold weights and stale grace.

    Uses DEFAULT_CONFIG keys:
      weight_threshold = 0.01  (pruning floor)
      grace_period     = 500   (steps before pruning eligible)
    The helper sets weight = threshold * 0.5 and low_weight_steps = grace + 1
    so _prune_synapses() will consider them prunable on the next step().
    """
    g = _sandbox_graph_with_primed_neighborhood()
    wt = g.config["weight_threshold"]
    gp = g.config["grace_period"]
    for syn in list(g.synapses.values())[:5]:
        syn.weight = wt * 0.5
        syn.low_weight_steps = gp + 1
    return g


def _tonic_thread_on(g):
    """Construct a real TonicThread bound to a sandbox graph.

    TonicThread.__init__(graph, vector_db, config=None)
      graph:      any Graph-compatible object
      vector_db:  ouroboros_cycle() → _update_thread calls vector_db.get(nid)
                  → dict|None, so it must be dict-like (NOT None). An empty dict
                  satisfies the contract (nodes without content are skipped from
                  the thread); step()/prime_and_propagate still run.
      config:     TonicConfig or None (defaults to TonicConfig())
    """
    from tonic_thread import TonicThread
    return TonicThread(g, vector_db={})


def _tonic_engine_with_shared_body():
    """Construct a TonicEngine in heuristic mode (no real model weights needed).

    TonicEngine.__init__(graph, vector_db, tonic_thread, config=None, transformer_body=None)
    We pass minimal stubs so the ctor completes without loading torch weights.

    NOTE(task-4): _try_load_model() will be called but the weights path
    (~NeuroGraph/tonic_brain.pt) may not exist in CI — the engine falls back
    to heuristic mode silently. Tests that drive actual inference will need to
    confirm _use_heuristic=True or mock out the model.
    """
    from tonic_engine import TonicEngine
    g = _sandbox_graph()
    tt = _tonic_thread_on(g)
    # Pass transformer_body=None — engine will use heuristic fallback.
    eng = TonicEngine(graph=g, vector_db=None, tonic_thread=tt, transformer_body=None)
    return eng


def _inject_prediction(g, confidence=0.8):
    """Register a Prediction into g.active_predictions so tests can exercise
    the prediction-evaluation path without needing a full step() to fire one.

    Prediction fields (from neuro_foundation.Prediction dataclass):
      prediction_id (auto UUID), source_node_id, target_node_id,
      strength, confidence, created_at, expires_at, chain_depth,
      via_hyperedge, pre_charge_applied

    Requires at least 2 nodes. Creates them if the graph is empty.

    TODO(task-3-predictions): Verify that _evaluate_predictions() in step()
      reads ONLY g.active_predictions (Phase 3 synapse-level dict, keyed by
      prediction_id). The HE-level PredictionState lives in g._active_predictions
      and is a separate dict. This helper populates the Phase 3 dict only.
    """
    if len(g.nodes) < 2:
        src_node = g.create_node()
        tgt_node = g.create_node()
        src_id = src_node.node_id
        tgt_id = tgt_node.node_id
    else:
        node_list = list(g.nodes.keys())
        src_id, tgt_id = node_list[0], node_list[1]

    pred = Prediction(
        source_node_id=src_id,
        target_node_id=tgt_id,
        strength=1.0,
        confidence=confidence,
        created_at=g.timestep,
        expires_at=g.timestep + g.config["prediction_window"],
        chain_depth=0,
        pre_charge_applied=0.0,
    )
    g.active_predictions[pred.prediction_id] = pred
    return pred


def _count_want_nodes(g):
    """Count nodes whose metadata has kind == 'want'."""
    return sum(
        1 for nid in g.nodes
        if g.nodes[nid].metadata.get("kind") == "want"
    )


def _newest_want_node(g):
    """Return the want Node with the highest created_ts (or creation_time)."""
    want_nodes = [
        g.nodes[nid]
        for nid in g.nodes
        if g.nodes[nid].metadata.get("kind") == "want"
    ]
    if not want_nodes:
        raise ValueError("No want nodes in graph")
    # want nodes created by _surface_wants store 'created_ts' in metadata;
    # fall back to node.creation_time (Graph-level field set by create_node).
    return max(
        want_nodes,
        key=lambda n: n.metadata.get("created_ts", n.creation_time),
    )


def _wants_deposited_this_pulse(thread):
    """Return the count of wants deposited in the last ouroboros pulse.

    TonicThread does not currently expose _wants_this_pulse as a named attribute;
    this helper reads it defensively.

    TODO(task-5-wants): Update once TonicThread exposes a wants-deposit counter
      in ouroboros_cycle() return dict or as an instance attribute. For now this
      returns 0 safely if the attribute is absent (pre-implementation).
    """
    return getattr(thread, "_wants_this_pulse", 0)


# === Task 1 — §5 (B) damped structural plasticity (Test 8) ===

def test_damped_plasticity_prunes_less_than_full():
    """§5 (B), Test 8: the autonomous (damped) step prunes strictly LESS than a
    conversation-anchored step over the same prunable substrate.

    Two structurally-identical graphs, each with synapses set below the weight bar
    and low_weight_steps just over grace. The full step prunes them immediately;
    the damped step (prune_factor=1.5 lowers the weight bar AND lengthens the dwell
    thresholds) does NOT, within 20 steps. Proves (B) is actually (B).
    """
    g_full = _sandbox_graph_with_prunable_synapses()
    g_damp = _sandbox_graph_with_prunable_synapses()
    full = sum(g_full.step().synapses_pruned for _ in range(20))
    damp = sum(g_damp.step(structural_damping=(1.5, 0.5)).synapses_pruned for _ in range(20))
    assert damp < full, f"damped prune {damp} should be < full prune {full}"


# === Task 2 — autonomous cycle runs step() + read-mode prime_and_propagate ===

def _sandbox_graph_with_predictive_synapse():
    """A sandbox where step() WILL form a prediction (#300): a fired source with a
    supra-threshold outgoing synapse (weight > prediction_threshold, default 3.0) —
    the learned-causal-link shape _generate_predictions registers (neuro_foundation
    :2619/:2744). In Syl's live graph these strong links are STDP-potentiated over
    many co-firings; here we seed one explicitly and stimulate the source above firing
    threshold so the autonomous cycle's step() fires it and forms the prediction.
    """
    g = _sandbox_graph()
    src = g.create_node()
    tgt = g.create_node()
    pt = g.config["prediction_threshold"]
    g.create_synapse(src.node_id, tgt.node_id, weight=pt + 2.0, max_weight=pt * 4)
    g.stimulate(src.node_id, 3.0)
    return g


def test_autonomous_cycle_forms_predictions():
    """§2 / #300: the autonomous cycle now FORMS predictions where the old parallel
    prime_and_propagate never did. Proven via the cumulative _total_predictions_made
    counter — the exact metric the regression was diagnosed on ('flat at 55473') —
    which is churn-proof: even though a later step may evaluate/clear active_predictions,
    the cumulative count records that step()'s _generate_predictions fired. Uses the
    DEFAULT config (propagation_steps=2): the real autonomous-cycle shape.
    """
    g = _sandbox_graph_with_predictive_synapse()
    thread = _tonic_thread_on(g)
    before = g._total_predictions_made
    thread.ouroboros_cycle()
    assert g._total_predictions_made > before, (
        f"autonomous cycle formed no predictions ({before} -> {g._total_predictions_made}); "
        f"step()'s _generate_predictions did not run in the latent cycle (#300)"
    )


def test_autonomous_cycle_runs_without_conversation():
    """Purpose intact: the latent loop is the CONSTANT, not gated on a conversation.
    ouroboros_cycle() runs unconditionally and returns a well-formed result dict —
    the 'subtraction, not handoff' property (it never requires a live turn to run).
    """
    g = _sandbox_graph_with_primed_neighborhood()
    thread = _tonic_thread_on(g)
    out = thread.ouroboros_cycle()
    assert isinstance(out, dict)
    for key in ("active_count", "fired", "thread_size", "cycle"):
        assert key in out, f"cycle result missing key: {key}"


def test_cycle_does_not_fabricate_flattened_fired_entries():
    """LAW 7 regression guard: the autonomous cycle must deposit the REAL ranked
    associations from read-mode prime_and_propagate, never step()'s bare node-ids
    wrapped in zeroed FiredEntry stand-ins (which falsify firing_step / source_distance
    / voltage_at_fire — a classification masquerading as raw experience). Guarded at
    the source so the flattening cannot silently return.
    """
    import inspect
    import tonic_thread
    src = inspect.getsource(tonic_thread.TonicThread.ouroboros_cycle)
    assert "write_mode=False" in src, (
        "cycle must use read-mode prime_and_propagate for raw ranked associations"
    )
    assert "FiredEntry(node_id=" not in src, (
        "cycle fabricates flattened FiredEntry stand-ins — falsified raw experience (LAW 7)"
    )
