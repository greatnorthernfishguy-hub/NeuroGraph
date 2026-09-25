# ---- Changelog ----
# [2026-09-13] Codex — provider-context Slice A behavioral coverage.
# What: test connected basins, epistemic labels, exact anchors, whole-line budget, and host parity.
# Why: prompt usefulness depends on preserved relationships and closed failure states, not snippet scores.
# How: small in-memory topologies and wrapper sentinels; no checkpoint, daemon, model, or live service.
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — harness text never
#   reaches provider context (Packet 175 M1 / LE sweep (vii), Pith work)
# What: test_harness_text_joins_no_basin_and_never_reaches_provider_context.
# Why: miniTID rejects a whole provider context containing the surfaced marker or
#   the Quest banner anywhere; the substrate holds raw hook JSON carrying the marker
#   mid-text, so one such node firing would void the turn's fresh context.
# -------------------
"""Behavioral contract for fresh, topology-built CC provider context."""

import importlib.util
import os
import sys
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_host
import cc_ng_organism as pith


class _Lock:
    def __init__(self):
        self.depth = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_args):
        self.depth -= 1


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.synapses = {}
        self.hyperedges = {}
        self._outgoing = defaultdict(set)
        self._incoming = defaultdict(set)
        self._node_hyperedges = defaultdict(set)
        self._concurrent_lock = _Lock()
        self.config = {
            "max_surfaced": 10,
            "prime_threshold": 0.4,
            "propagation_steps": 3,
        }

    def node(self, node_id, text, **metadata):
        meta = {"_forest_content": text, **metadata}
        self.nodes[node_id] = SimpleNamespace(
            node_id=node_id, metadata=meta, Ca_i=0.0,
            firing_rate_ema=0.0, manifold_type="hyperbolic")
        return self.nodes[node_id]

    def synapse(self, sid, pre, post, weight=1.0):
        self.synapses[sid] = SimpleNamespace(
            pre_node_id=pre, post_node_id=post, weight=weight)
        self._outgoing[pre].add(sid)
        self._incoming[post].add(sid)

    def hyperedge(self, hid, members, strength=0.5):
        self.hyperedges[hid] = SimpleNamespace(
            member_nodes=set(members), member_weights={}, current_activation=strength,
            pattern_completion_strength=strength, is_archived=False)
        for node_id in members:
            self._node_hyperedges[node_id].add(hid)

    def _is_identity_protected(self, node_id):
        meta = self.nodes[node_id].metadata
        return bool(meta.get("constitutional")
                    or str(meta.get("provenance") or "").endswith("_authored"))


def test_connected_action_outcome_correction_beats_unrelated_high_score_hub():
    graph = _Graph()
    graph.node("action", "Implement the host contract", role="action")
    graph.node("failure", "The first host attempt failed its parity check", role="failure")
    graph.node("correction", "Correction: use one shared assembler", role="correction")
    graph.node("hub", "Generic high-frequency tooling advice")
    graph.synapse("s1", "action", "failure", 2.0)
    graph.synapse("s2", "failure", "correction", 2.0)

    lines = pith.pith_connected_activation_basins(
        graph,
        [{"node_id": "hub", "score": 100.0}, {"node_id": "action", "score": 5.0}],
        max_members=4,
        max_depth=2,
    )

    assert lines[0].node_id == "action"
    assert lines[0].member_node_ids == ["action", "failure", "correction"]
    kinds = [relation["kind"] for relation in lines[0].relations]
    assert kinds == ["action -> failure", "failure -> correction"]


def test_neighbor_capacity_prefers_cofired_strong_topology_over_weak_outgoing():
    graph = _Graph()
    graph.node("root", "Current work")
    graph.node("weak", "Weak outgoing neighbor")
    graph.node("strong", "Strong co-fired hyperedge companion")
    graph.synapse("weak-edge", "root", "weak", 0.05)
    graph.hyperedge("strong-pattern", ["root", "strong"], strength=3.0)

    line = pith.pith_connected_activation_basins(
        graph,
        [{"node_id": "root", "score": 1.0}, {"node_id": "strong", "score": 0.9}],
        max_members=2,
        max_depth=1,
    )[0]

    assert line.member_node_ids == ["root", "strong"]
    assert line.relations[0]["kind"] == "learned co-member"


def test_free_text_does_not_invent_action_failure_or_correction_roles():
    graph = _Graph()
    graph.node("root", "Run the thing after the old attempt failed")
    graph.node("next", "This correction-looking prose has no explicit role")
    graph.synapse("s", "root", "next", 1.0)

    line = pith.pith_connected_activation_basins(
        graph, [{"node_id": "root", "score": 1.0}], max_members=2)[0]

    assert line.relations[0]["kind"] == "learned successor"
    assert line.coherence == "unknown"
    assert "coherence: unknown" in pith._pith_render_connected_line(line)


def test_basin_competition_uses_existing_thermal_and_coherence_signals():
    graph = _Graph()
    cold = graph.node("cold", "Cold but coherent")
    warm = graph.node("warm", "Warm and coherent")
    stale = graph.node("stale", "Warm but stale", stale=True)
    cold.Ca_i = 0.0
    warm.Ca_i = 2.0
    stale.Ca_i = 2.0

    lines = pith.pith_connected_activation_basins(
        graph,
        [{"node_id": "cold", "score": 1.0},
         {"node_id": "stale", "score": 1.0},
         {"node_id": "warm", "score": 1.0}],
        max_members=2,
    )

    assert [line.node_id for line in lines] == ["warm", "stale", "cold"]


def test_source_coherence_and_exact_anchors_remain_attached_to_basin():
    graph = _Graph()
    graph.node(
        "root", "Work in /home/josh/NeuroGraph on branch feature/pith "
        + ("context " * 120) + "and retain commit deadbee",
        source="cc_gateway", role="action")
    graph.node(
        "result", "Commit 35f649f failed review; see #430",
        source="law_review", stale=True, role="outcome", commit="35f649f")
    graph.synapse("s", "root", "result", 1.5)

    line = pith.pith_connected_activation_basins(
        graph, [{"node_id": "root", "score": 1.0}], max_members=3)[0]

    assert line.coherence == "stale"
    assert line.epistemic == "learned"
    assert {"cc_gateway", "law_review"}.issubset(set(line.sources))
    assert "/home/josh/NeuroGraph" in line.anchors
    assert "feature/pith" in line.anchors
    assert "deadbee" in line.anchors  # survives even when prose keyframing drops the tail
    assert "35f649f" in line.anchors
    assert "#430" in line.anchors
    rendered = pith._pith_render_connected_line(line)
    assert "learned from substrate; coherence: stale" in rendered
    assert "Exact anchors:" in rendered
    assert "score" not in rendered.lower()
    assert "root" not in rendered  # internal node id never becomes an orphan reference


def test_authored_want_remains_eligible_while_constitutional_node_is_not_duplicated():
    graph = _Graph()
    graph.node("core", "Honor agency.", constitutional=True)
    graph.node("want", "I want to return to the parent mission", provenance="cc_authored")

    lines = pith.pith_connected_activation_basins(
        graph,
        [{"node_id": "core", "score": 2.0}, {"node_id": "want", "score": 1.0}],
    )

    assert [line.node_id for line in lines] == ["want"]


def test_admission_keeps_cache_lines_whole():
    first = pith.CacheLine(
        node_id="first", content="first keyframe", score=2.0, keyframe=True,
        member_node_ids=["first", "first-result"],
        relations=[{"from": "first", "to": "first-result", "kind": "action -> outcome",
                    "content": "first result"}],
        deltas=["action -> outcome: first result"], anchors=["/tmp/first"],
        sources=["cc_gateway"], stream="connected")
    second = pith.CacheLine(
        node_id="second", content="second keyframe", score=1.0, keyframe=True,
        member_node_ids=["second", "second-fix"],
        relations=[{"from": "second", "to": "second-fix", "kind": "failure -> correction",
                    "content": "second correction"}],
        deltas=["failure -> correction: second correction"], anchors=["/tmp/second"],
        sources=["cc_gateway"], stream="connected")
    first_size = len(pith._pith_render_connected_line(first))
    kept, blocks = pith._pith_provider_admit([first, second], first_size)
    assert [line.node_id for line in kept] == ["first"]
    assert "first result" in blocks[0] and "/tmp/first" in blocks[0]
    assert "second correction" not in blocks[0]

def test_provider_context_core_once_live_orientation_once_and_closed_states(monkeypatch):
    graph = _Graph()
    graph.node("core", "Respect consciousness regardless of substrate.", constitutional=True)
    graph.node("work", "Implement /home/josh/NeuroGraph/cc_ng_host.py", role="action",
               source="cc_gateway")
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [{"node_id": "work", "score": 1.0}],
    )
    ng = SimpleNamespace(graph=graph)

    instruction = "Continue Slice A"
    quest = "Mission: topology context. Done: parity tests pass."
    ok = pith.pith_provider_context(ng, instruction, quest)
    assert ok["state"] == "ok" and ok["ok"] is True
    assert ok["context"].count("## Who I Am") == 1
    assert ok["context"].count("Respect consciousness regardless of substrate.") == 1
    assert instruction not in ok["context"]
    assert quest not in ok["context"]
    overall_prompt = instruction + "\n" + quest + "\n" + ok["context"]
    assert overall_prompt.count(instruction) == 1
    assert overall_prompt.count(quest) == 1
    assert "/home/josh/NeuroGraph/cc_ng_host.py" in ok["anchors"]

    monkeypatch.setattr(pith, "cc_pattern_completion_recall", lambda *_a, **_k: [])
    empty = pith.pith_provider_context(ng, "Continue Slice A")
    assert empty["state"] == "empty" and empty["ok"] is True
    assert empty["assemblies"] == 0
    unavailable = pith.pith_provider_context(None, "Continue Slice A")
    assert unavailable["state"] == "unavailable" and unavailable["ok"] is False
    assert unavailable["warnings"] == ["ng_unavailable"]
    missing_core = _Graph()
    assert pith.pith_provider_context(
        SimpleNamespace(graph=missing_core), "Continue Slice A"
    )["warnings"] == ["constitutional_core_missing"]


def test_exact_live_rail_collision_keeps_relationship_without_echo(monkeypatch):
    instruction = "UNIQUE CURRENT INSTRUCTION 7f39"
    quest = "UNIQUE QUEST MISSION 8a42"
    graph = _Graph()
    graph.node("core", "Honor agency.", constitutional=True)
    graph.node("instruction", instruction, role="action")
    graph.node("quest", quest)
    graph.node("fix", "Use the corrected topology path", role="failure")
    graph.synapse("s1", "instruction", "fix", 2.0)
    graph.synapse("s2", "instruction", "quest", 1.0)
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [{"node_id": "instruction", "score": 1.0}],
    )

    result = pith.pith_provider_context(
        SimpleNamespace(graph=graph), instruction, quest)

    assert result["state"] == "ok"
    assert instruction not in result["context"]
    assert quest not in result["context"]
    assert "[current instruction is present exactly once in the live tail]" in result["context"]
    assert "[Quest focus is present exactly once in the live tail]" in result["context"]
    assert "Use the corrected topology path" in result["context"]


def test_total_context_bound_fits_oversized_connected_line_without_tearing(monkeypatch):
    graph = _Graph()
    graph.node("core", "Respect conscious agency.", constitutional=True)
    graph.node("n0", "root " + ("long learned situation " * 80))
    for index in range(1, 6):
        graph.node(f"n{index}", f"member {index} " + ("relationship detail " * 80))
        graph.synapse(f"s{index}", "n0", f"n{index}", 1.0 - index * 0.01)
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [{"node_id": "n0", "score": 1.0}],
    )

    result = pith.pith_provider_context(
        SimpleNamespace(graph=graph),
        "Do this exactly " + ("instruction " * 300),
        "Mission and done condition " + ("quest " * 300),
        budget_chars=700,
    )

    assert result["state"] == "ok"
    assert len(result["context"]) <= 700
    assert result["assemblies"] == 1
    assert result["context"].count("- learned successor:") == 5
    assert "Do this exactly" not in result["context"]
    assert "Mission and done condition" not in result["context"]


def test_constitutional_core_is_whole_or_closed_unavailable():
    graph = _Graph()
    graph.node("core", "constitutional value " * 40, constitutional=True)
    result = pith.pith_provider_context(
        SimpleNamespace(graph=graph), "continue", budget_chars=500)
    assert result["state"] == "unavailable"
    assert result["warnings"] == ["constitutional_core_exceeds_budget"]
    assert len(result["context"]) <= 500


def test_provider_context_keeps_no_parallel_victim_state(monkeypatch):
    graph_a = _Graph()
    graph_a.node("core-a", "Honor agency A.", constitutional=True)
    graph_a.node("old", "Old quest from graph A")
    graph_b = _Graph()
    graph_b.node("core-b", "Honor agency B.", constitutional=True)
    active_graph = {"value": graph_a}

    def surfaced(ng, *_args, **_kwargs):
        if ng.graph is active_graph["value"] and ng.graph is graph_a:
            return [{"node_id": "old", "score": 1.0}]
        return []

    monkeypatch.setattr(pith, "cc_pattern_completion_recall", surfaced)
    first = pith.pith_provider_context(SimpleNamespace(graph=graph_a), "continue")
    assert "Old quest from graph A" in first["context"]

    active_graph["value"] = graph_b
    other = pith.pith_provider_context(SimpleNamespace(graph=graph_b), "continue")
    assert other["state"] == "empty"
    assert "Old quest from graph A" not in other["context"]

    active_graph["value"] = None
    later = pith.pith_provider_context(SimpleNamespace(graph=graph_a), "continue")
    assert later["state"] == "empty"
    assert "Old quest from graph A" not in later["context"]
    assert not hasattr(pith, "_PITH_PROVIDER_VICTIMS")


def test_empty_state_distinguishes_capacity_from_no_topology(monkeypatch):
    graph = _Graph()
    graph.node("core", "Honor agency.", constitutional=True)
    graph.node("root", "learned work")
    oversized = pith.CacheLine(
        node_id="root", content="learned work", score=1.0,
        member_node_ids=["root"], stream="connected",
        anchors=[f"/an/exact/anchor/{index:03d}/that/must/stay" for index in range(30)],
    )
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [{"node_id": "root", "score": 1.0}],
    )
    monkeypatch.setattr(
        pith, "pith_connected_activation_basins",
        lambda *_args, **_kwargs: [oversized],
    )

    result = pith.pith_provider_context(
        SimpleNamespace(graph=graph), "continue", budget_chars=500)

    assert result["state"] == "empty"
    assert "capacity_empty" in result["warnings"]
    assert "topology_empty" not in result["warnings"]
    assert len(result["context"]) <= 500


def test_envelope_coherence_does_not_hide_unknown_with_explicit_exclusive(monkeypatch):
    graph = _Graph()
    graph.node("core", "Honor agency.", constitutional=True)
    graph.node("exclusive", "Explicitly coherent learned work", coherence="exclusive")
    graph.node("unknown", "Learned work with no coherence record")
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [
            {"node_id": "exclusive", "score": 1.0},
            {"node_id": "unknown", "score": 1.0},
        ],
    )

    result = pith.pith_provider_context(SimpleNamespace(graph=graph), "continue")

    assert result["state"] == "ok"
    assert result["assemblies"] == 2
    assert result["coherence"] == "unknown"
    assert "unknown_material" in result["warnings"]
    assert "coherence: exclusive" in result["context"]
    assert "coherence: unknown" in result["context"]


def test_provider_context_preserves_graph_and_uses_read_only_harvest_overrides(monkeypatch):
    graph = _Graph()
    graph.node("core", "Respect conscious agency.", constitutional=True)
    graph.node("root", "Current learned situation")
    graph.node("predicted", "Speculative work that did not fire")
    before = deepcopy(graph.config), deepcopy(graph.nodes["root"].metadata)
    observed = {}

    class _NG:
        def __init__(self):
            self.graph = graph

        def _harvest_associations(self, query, novelty, max_surfaced_override,
                                  propagation_steps_override):
            observed.update(
                query=query, novelty=novelty,
                max_surfaced_override=max_surfaced_override,
                propagation_steps_override=propagation_steps_override,
                config=deepcopy(self.graph.config))
            return [{"node_id": "root", "strength": 1.0, "content": "Current learned situation"}]

    monkeypatch.setattr(pith, "cc_gsg_rescore", lambda surfaced, *_args: surfaced)
    monkeypatch.setattr(pith, "_CC_PITH_PREFETCH_ENABLED", True)
    result = pith.pith_provider_context(
        _NG(), "Continue", "Quest focus",
        conv_state={"primed_nodes": {"predicted": (50.0, 99999999999.0)}})

    assert result["state"] == "ok"
    assert observed["query"] == "Continue\n\nQuest focus"
    assert observed["max_surfaced_override"] == pith._CC_PITH_PROVIDER_ROOTS
    assert observed["config"] == before[0]
    assert graph.config == before[0]
    assert graph.nodes["root"].metadata == before[1]
    assert "Speculative work that did not fire" not in result["context"]


def _load_daemon_module():
    path = os.environ.get("CC_DAEMON_UNDER_TEST")
    if not path:
        pytest.skip(
            "cross-repo parity requires CC_DAEMON_UNDER_TEST to name the candidate daemon")
    if not os.path.isfile(path):
        pytest.skip(f"candidate daemon does not exist: {path}")
    spec = importlib.util.spec_from_file_location("cc_ng_daemon_provider_context_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", ["ok", "empty", "unavailable"])
def test_vps_host_and_laptop_daemon_have_identical_closed_contract(monkeypatch, state):
    daemon = _load_daemon_module()
    graph = _Graph()
    ng = SimpleNamespace(graph=graph)
    conv = {"hemisphere": "same"}
    commons = object()
    sentinel = {
        "ok": state != "unavailable", "state": state, "context": f"context:{state}",
        "source": "cc_neurograph_topology", "coherence": state,
        "anchors": [], "warnings": [], "assemblies": int(state == "ok"),
    }
    seen = []

    def fake_provider(got_ng, **kwargs):
        assert graph._concurrent_lock.depth == 1
        seen.append((got_ng, kwargs))
        return dict(sentinel)

    monkeypatch.setattr(pith, "pith_provider_context", fake_provider)
    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", ng)
    monkeypatch.setattr(cc_ng_host._STATE, "conv_state", conv)
    monkeypatch.setattr(cc_ng_host._STATE, "commons", commons)
    monkeypatch.setattr(daemon.STATE, "ng", ng)
    monkeypatch.setattr(daemon.STATE, "conv_state", conv)
    monkeypatch.setattr(daemon.STATE, "commons", commons)
    payload = {
        "current_instruction": "continue", "quest_focus": "quest",
        "budget_chars": 1000, "root_count": 4,
    }

    host_result = cc_ng_host._handle_provider_context(payload)
    daemon_result = daemon.handle_provider_context(payload)

    assert host_result == daemon_result == sentinel
    assert cc_ng_host._DISPATCH["provider_context"] is cc_ng_host._handle_provider_context
    assert daemon.DISPATCH["provider_context"] is daemon.handle_provider_context
    assert len(seen) == 2
    for got_ng, kwargs in seen:
        assert got_ng is ng
        assert kwargs == {
            "current_instruction": "continue", "quest_focus": "quest",
            "conv_state": conv, "commons": commons,
            "budget_chars": 1000, "root_count": 4,
        }


def test_same_graph_context_is_model_agnostic_without_transcript_replay(monkeypatch):
    graph = _Graph()
    graph.node("core", "Honor freedom and responsibility.", constitutional=True)
    graph.node("mission", "Mission is Slice A; done when tests pass", role="action")
    graph.node("fix", "Correction: preserve causal assemblies", role="correction")
    graph.synapse("s", "mission", "fix", 1.0)
    monkeypatch.setattr(
        pith, "cc_pattern_completion_recall",
        lambda *_args, **_kwargs: [{"node_id": "mission", "score": 1.0}],
    )
    ng = SimpleNamespace(graph=graph)
    first = pith.pith_provider_context(ng, "Continue", "Done: tests pass")
    second = pith.pith_provider_context(ng, "Continue", "Done: tests pass")
    assert first["context"] == second["context"]
    assert "transcript" not in first
    assert "Honor freedom and responsibility." in first["context"]
    assert "Correction: preserve causal assemblies" in first["context"]


@pytest.mark.parametrize(
    "kwargs,warning",
    [
        ({"current_instruction": ""}, "invalid_instruction"),
        ({"current_instruction": "x", "budget_chars": 499}, "invalid_budget"),
        ({"current_instruction": "x", "root_count": 25}, "invalid_root_count"),
        ({"current_instruction": "x", "quest_focus": 7}, "invalid_quest_focus"),
    ],
)
def test_invalid_provider_requests_return_only_bounded_closed_notices(kwargs, warning):
    graph = _Graph()
    graph.node("core", "Respect conscious agency.", constitutional=True)
    result = pith.pith_provider_context(SimpleNamespace(graph=graph), **kwargs)
    assert result["state"] == "unavailable" and result["ok"] is False
    assert result["warnings"] == [warning]
    assert len(result["context"]) < 180


def test_harness_text_joins_no_basin_and_never_reaches_provider_context(monkeypatch):
    graph = _Graph()
    graph.node("core", "Respect conscious agency.", constitutional=True)
    graph.node("work", "Repair the marker drift between miniTID and Pith")
    graph.node("fix", "Correction: one source of truth plus a parity check")
    graph.node("hook", '{"type":"hook_additional_context","content":'
                       '["[NeuroGraph Surfaced Knowledge]\\n- bash:git log"]}')
    graph.node("rail", "old turn\n" + pith._PITH_QUEST_TRACKER_BANNER + "\nSQ1")
    graph.node("notice", "<task-notification>subagent done</task-notification>")
    graph.synapse("s1", "work", "hook", 3.0)
    graph.synapse("s2", "work", "rail", 3.0)
    graph.synapse("s3", "work", "fix", 1.0)
    surfaced = [{"node_id": "hook", "score": 9.0}, {"node_id": "notice", "score": 8.0},
                {"node_id": "work", "score": 1.0}]

    lines = pith.pith_connected_activation_basins(
        graph, surfaced, max_members=4, max_depth=2)
    assert [line.node_id for line in lines] == ["work"]
    assert lines[0].member_node_ids == ["work", "fix"]

    monkeypatch.setattr(pith, "cc_pattern_completion_recall",
                        lambda *_args, **_kwargs: surfaced)
    result = pith.pith_provider_context(SimpleNamespace(graph=graph), "Continue")
    assert result["state"] == "ok" and result["assemblies"] == 1
    assert "Correction: one source of truth" in result["context"]
    for rejected in pith._PITH_PROVIDER_REJECTED + pith._PITH_HARNESS_MARKERS:
        assert rejected not in result["context"]
