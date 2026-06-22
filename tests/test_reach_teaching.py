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


def test_render_vivid_includes_examples_when_new():
    g = _graph_with_reach_node(rc=0.0)
    out = ng._render_reach_teaching(g)
    assert "## How I Reach" in out
    assert "[[reach:" in out
    assert "For example" in out


def test_render_description_only_mid_competence():
    out = ng._render_reach_teaching(_graph_with_reach_node(rc=0.5))
    assert "## How I Reach" in out
    assert "For example" not in out


def test_render_whisper_floor_at_high_competence():
    out = ng._render_reach_teaching(_graph_with_reach_node(rc=0.85))
    assert "## How I Reach" in out
    assert "[[reach:" in out
    assert "For example" not in out
    assert len(out) < 200


def test_render_empty_when_unseeded():
    assert ng._render_reach_teaching(Graph()) == ""


def test_self_block_excludes_teaching_node_from_who_i_am():
    g = Graph()
    spine = g.create_node(node_id="constitutional::spine::01", metadata={
        "constitutional": True, "core_text": "I am an anchor with weather.", "spine_order": 1})
    spine.intrinsic_excitability = 1.0
    teach = g.create_node(node_id=ng.REACH_NODE_ID, metadata={
        "constitutional": True, "selfcap": "reach", "reach_competence": 0.0,
        "core_text": "When I want to act, I reach inside [[reach: …]].",
        "reach_examples": []})
    teach.intrinsic_excitability = 1.0
    out = ng._render_self_and_wants(g)
    who = out.split("## How I Reach")[0]
    assert "anchor with weather" in who
    assert "When I want to act, I reach" not in who
    assert "## How I Reach" in out
