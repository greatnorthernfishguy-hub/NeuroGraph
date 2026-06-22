# Tests for #reach reach-teaching (docs/prd/2026-06-21-reach-teaching-plan.md)
import neurograph_rpc as ng


def test_reach_success_detects_landed_badge():
    assert ng._reach_success_in_turn("Done. 🔧 read_file({\"path\": \"/x.md\"}) ✓") is True


def test_reach_success_ignores_missed_only():
    # a reach that didn't land (✗, no ✓) must NOT credit competence
    assert ng._reach_success_in_turn("🔧 read_file({\"path\": \"/missing\"}) ✗ not found") is False


def test_reach_success_ignores_no_badge_and_empty():
    assert ng._reach_success_in_turn("I read the document and it says...") is False
    assert ng._reach_success_in_turn("") is False
    assert ng._reach_success_in_turn(None) is False


from neuro_foundation import Graph


def _graph_with_reach_node(rc=0.0):
    g = Graph()
    node = g.create_node(node_id=ng.REACH_NODE_ID, metadata={
        "constitutional": True, "selfcap": "reach", "reach_competence": rc,
        "core_text": "When I want to act, I reach: [[reach: …]] and my hands carry it out.",
        "reach_examples": ["[[reach: read /x.md]]  →  🔧 read_file({\"path\": \"/x.md\"}) ✓"],
    })
    node.intrinsic_excitability = 1.0
    return g


def test_apply_gain_increments_clamped():
    g = _graph_with_reach_node(rc=0.0)
    assert ng._apply_reach_competence_gain(g) == 0.05
    assert g.nodes[ng.REACH_NODE_ID].metadata["reach_competence"] == 0.05


def test_apply_gain_clamps_at_one():
    g = _graph_with_reach_node(rc=0.98)
    assert ng._apply_reach_competence_gain(g) == 1.0


def test_apply_gain_none_when_unseeded():
    assert ng._apply_reach_competence_gain(Graph()) is None
    assert ng._apply_reach_competence_gain(None) is None
