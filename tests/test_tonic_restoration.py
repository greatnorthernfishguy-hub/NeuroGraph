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
    """Construct a TonicEngine simulated into the SHARED-body (post-hot-swap) state:
    _use_heuristic=False, a lightweight _model wrapper holding a (shared) body, and
    _shared_body set — the state a live engine is in after BrainSwitcher.offer_shared_body
    hands it proto's body. Task 4 tests revoke from here. No torch weights loaded.
    """
    from tonic_engine import TonicEngine
    g = _sandbox_graph()
    tt = _tonic_thread_on(g)
    eng = TonicEngine(graph=g, vector_db=None, tonic_thread=tt, transformer_body=None)
    # Simulate the post-hot-swap shared state (BrainSwitcher.offer_shared_body result):
    _shared = object()
    eng._model = type("M", (), {"body": _shared})()   # lightweight encoder/decoder wrapper
    eng._shared_body = _shared
    eng._use_heuristic = False
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
    """A sandbox where step() forms a HIGH-CONFIDENCE prediction (#300) — strong enough
    to both REGISTER (weight > prediction_threshold, default 3.0) and CLEAR the curiosity
    gate (_curiosity_signal filters confidence > 0.6).

    confidence = weight/max_weight * 0.6 + confirmation_rate(0.5 neutral prior) * 0.4
    (_compute_prediction_confidence, neuro_foundation:2595). weight 4.5 / max 5.0 ->
    0.9 * 0.6 + 0.5 * 0.4 = 0.74 > 0.6; and weight 4.5 > threshold 3.0 so it registers.
    In Syl's live graph these are STDP-potentiated, high-confirmation links; here we seed
    one and stimulate the source above firing threshold so the cycle's step() fires it.
    """
    g = _sandbox_graph()
    src = g.create_node()
    tgt = g.create_node()
    pt = g.config["prediction_threshold"]          # 3.0
    g.create_synapse(src.node_id, tgt.node_id, weight=pt + 1.5, max_weight=pt + 2.0)
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
    src = inspect.getsource(tonic_thread.TonicThread._ouroboros_cycle_inner)
    assert "write_mode=False" in src, (
        "cycle must use read-mode prime_and_propagate for raw ranked associations"
    )
    assert "FiredEntry(node_id=" not in src, (
        "cycle fabricates flattened FiredEntry stand-ins — falsified raw experience (LAW 7)"
    )


# === Task 3 wiring — curiosity reads freshly-formed predictions (#300 completion) ===

def test_tick_forms_predictions_before_reading_curiosity():
    """#300 wiring: TonicBridge._tick must FORM predictions (ouroboros_cycle) before it
    READS them (_curiosity_signal) in the same tick. Otherwise it reads a stale/empty
    active_predictions (predictions are transient; the forming loop ran on a different
    clock) — the 'always read an empty set' bug. Ordering guarded at the source.
    """
    import inspect
    import neurograph_rpc
    src = inspect.getsource(neurograph_rpc.TonicBridge._tick)
    assert "ouroboros_cycle()" in src, (
        "_tick must form fresh predictions before reading curiosity (#300)"
    )
    assert src.index("ouroboros_cycle()") < src.index("_curiosity_signal()"), (
        "_tick reads curiosity before forming predictions — reads a stale set (#300)"
    )


def test_curiosity_signal_reads_high_confidence_prediction():
    """End-to-end: a high-confidence prediction formed by step() is actually returned by
    the curiosity bucket (_curiosity_signal, gate confidence > 0.6). Proves the wiring
    delivers signal — not merely that predictions form. LAW 7: the gate is a
    classification AT EXTRACTION (the bucket), never at deposit.
    """
    import neurograph_rpc
    g = _sandbox_graph_with_predictive_synapse()
    g.step()                                       # form the high-confidence prediction
    assert g.active_predictions, "fixture formed no prediction"

    class _Mem:
        pass
    mem = _Mem()
    mem.graph = g
    mem._tonic_thread = None
    orig = neurograph_rpc._memory
    neurograph_rpc._memory = mem
    try:
        bridge = neurograph_rpc.TonicBridge()
        seeds = bridge._curiosity_signal()
    finally:
        neurograph_rpc._memory = orig
    assert seeds, "curiosity bucket returned nothing for a confidence>0.6 prediction"
    assert all(p.confidence > 0.6 for p in seeds), (
        "curiosity gate let through a low-confidence prediction"
    )


# === Task 5 — fail-fresh + rate cap (§7, the flood-safe backstop) ===

def test_ouroboros_cycle_fails_fresh_on_step_error(monkeypatch):
    """§7 fail-fresh: any error inside the cycle (here, step() raising) is swallowed —
    the cycle does nothing this pulse and returns a no-op result, NEVER raising. A flaky
    step can't crash the latent loop; the thread continues.
    """
    g = _sandbox_graph_with_primed_neighborhood()   # has active (stimulated) nodes
    thread = _tonic_thread_on(g)

    def _boom(*a, **k):
        raise RuntimeError("step blew up")
    monkeypatch.setattr(g, "step", _boom)

    out = thread.ouroboros_cycle()                   # must NOT raise
    assert out.get("failed_fresh") is True, "cycle did not fail fresh on step() error"
    assert out.get("fired", -1) == 0


def test_autonomous_steps_rate_capped():
    """§7 rate cap: a mis-set propagation_steps cannot flood autonomous step()s. With
    propagation_steps far above the backstop, the cycle caps at
    _MAX_AUTONOMOUS_STEPS_PER_PULSE and reports the capped count — an independent
    backstop so a mis-tuned gate/config cannot re-create the OOM.
    """
    import tonic_thread
    from tonic_thread import TonicConfig, TonicThread
    g = _sandbox_graph_with_primed_neighborhood()
    thread = TonicThread(g, vector_db={}, config=TonicConfig(propagation_steps=100))
    out = thread.ouroboros_cycle()
    cap = tonic_thread._MAX_AUTONOMOUS_STEPS_PER_PULSE
    assert out.get("autonomous_steps") == cap, (
        f"propagation_steps=100 should cap to {cap}, got {out.get('autonomous_steps')}"
    )


# === Task 4 — heuristic-on-shed degrade (§7) ===

def test_revoke_shared_body_degrades_to_heuristic():
    """§7: when ProtoUniBrain sheds the shared body (memory pressure), the Tonic
    degrades STRAIGHT to the heuristic decoder — it does NOT reload its own ~2GB
    transformer at the worst possible moment. _use_heuristic flips True; the shared
    body reference is dropped.
    """
    eng = _tonic_engine_with_shared_body()
    assert eng._use_heuristic is False
    assert eng._shared_body is not None
    assert eng.revoke_shared_body() is True
    assert eng._use_heuristic is True
    assert eng._shared_body is None


def test_revoke_does_not_reload_own_transformer():
    """§7 memory-cheap guarantee: revoke_shared_body must NOT load a model. A
    pressure-driven shed that loads a fresh ~2GB own-transformer (the OOM-'n'-load
    trap) is exactly what this fix removes. Guarded at the source.
    """
    import inspect
    import tonic_engine
    src = inspect.getsource(tonic_engine.TonicEngine.revoke_shared_body)
    assert "from_pretrained" not in src, "revoke reloads a transformer — OOM-'n'-load trap (§7)"
    assert "AutoModelForCausalLM" not in src, "revoke imports a model loader — not memory-cheap (§7)"


def test_offer_after_revoke_rejoins_share():
    """§7: after a heuristic shed, the Tonic re-joins the share the instant proto
    reloads — offer_shared_body() restores transformer mode (_use_heuristic=False).
    Revoke keeps the lightweight wrapper precisely so this re-join is possible.
    """
    eng = _tonic_engine_with_shared_body()
    eng.revoke_shared_body()
    assert eng._use_heuristic is True
    reloaded_body = object()
    assert eng.offer_shared_body(reloaded_body) is True
    assert eng._use_heuristic is False
    assert eng._shared_body is reloaded_body
