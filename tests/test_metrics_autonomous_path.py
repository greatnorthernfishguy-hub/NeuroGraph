"""
Substrate Metrics Pipeline — autonomous path (#320): metrics flow without a conversation.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — to_jsonl flag / no-conversation-dependency
# What: Proves _deposit_substrate_metrics(to_jsonl=...) — the gate (Commons) ALWAYS runs; the
#       neurograph.jsonl append runs ONLY when to_jsonl=True. The autonomous pulse calls with
#       to_jsonl=False so substrate metrics reach the Commons WITHOUT a conversation (Bunyan/
#       THC/Immunis health-monitor while idle) and the per-2s autonomous rate doesn't bloat jsonl.
# Why: [[feedback_no_conversation_dependency]] — nothing load-bearing may require an active turn.
# How: minimal fake _memory + step_result; recorder gate; tmp HOME so the jsonl probe is isolated.
# -------------------
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurograph_rpc as rpc


def _fake_step():
    return types.SimpleNamespace(
        fired_node_ids=["a", "b", "c"], fired_hyperedge_ids=["h"],
        synapses_pruned=0, synapses_sprouted=0,
        predictions_confirmed=10, predictions_surprised=0,
    )


def _fake_memory():
    g = types.SimpleNamespace(nodes={"n1": 1, "n2": 1}, synapses={"s1": 1})
    return types.SimpleNamespace(graph=g)


class _RecorderGate:
    def __init__(self):
        self.observed = []
    def observe(self, metrics):
        self.observed.append(metrics)


def _run(to_jsonl):
    """Run _deposit_substrate_metrics with a fake memory + recorder gate + isolated HOME.
    Returns (gate_observed_count, jsonl_written_bool)."""
    import tempfile
    orig_mem, orig_gate, orig_home = rpc._memory, rpc._metrics_gate, os.environ.get("HOME")
    rec = _RecorderGate()
    rpc._memory = _fake_memory()
    rpc._metrics_gate = rec
    tmp = tempfile.mkdtemp()
    os.environ["HOME"] = tmp
    try:
        rpc._deposit_substrate_metrics(_fake_step(), to_jsonl=to_jsonl)
    finally:
        rpc._memory, rpc._metrics_gate = orig_mem, orig_gate
        if orig_home is not None:
            os.environ["HOME"] = orig_home
    jsonl = os.path.join(tmp, ".et_modules", "shared_learning", "neurograph.jsonl")
    return len(rec.observed), os.path.exists(jsonl)


def test_gate_always_runs_jsonl_per_turn():
    # afterTurn path: to_jsonl=True → gate observes AND jsonl written
    n_obs, jsonl = _run(to_jsonl=True)
    assert n_obs == 1, "gate must observe on the per-turn path"
    assert jsonl, "to_jsonl=True writes neurograph.jsonl"


def test_autonomous_path_gate_only_no_jsonl():
    # autonomous path: to_jsonl=False → gate observes, jsonl SKIPPED (no per-2s bloat)
    n_obs, jsonl = _run(to_jsonl=False)
    assert n_obs == 1, "gate must observe on the AUTONOMOUS path (metrics flow without conversation)"
    assert not jsonl, "to_jsonl=False must NOT write jsonl (unbounded append would bloat at ~2s)"


def test_default_is_jsonl_on():
    """Default (no arg) keeps the historical per-turn jsonl behavior."""
    orig_mem, orig_gate, orig_home = rpc._memory, rpc._metrics_gate, os.environ.get("HOME")
    import tempfile
    rec = _RecorderGate(); rpc._memory = _fake_memory(); rpc._metrics_gate = rec
    tmp = tempfile.mkdtemp(); os.environ["HOME"] = tmp
    try:
        rpc._deposit_substrate_metrics(_fake_step())  # no to_jsonl arg
    finally:
        rpc._memory, rpc._metrics_gate = orig_mem, orig_gate
        if orig_home is not None:
            os.environ["HOME"] = orig_home
    assert os.path.exists(os.path.join(tmp, ".et_modules", "shared_learning", "neurograph.jsonl"))


if __name__ == "__main__":
    test_gate_always_runs_jsonl_per_turn();  print("PASS per-turn (to_jsonl=True): gate observes + jsonl written")
    test_autonomous_path_gate_only_no_jsonl(); print("PASS autonomous (to_jsonl=False): gate observes, NO jsonl (no conversation-dependency, no bloat)")
    test_default_is_jsonl_on();              print("PASS default keeps per-turn jsonl behavior")
    print("\nMetrics autonomous path (#320): ALL PASS — metrics flow without a conversation; jsonl stays per-turn")
