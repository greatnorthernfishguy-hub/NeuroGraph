# ---- Changelog ----
# [2026-06-14] Claude Code (DudeMan CC, Opus 4.8) — #spine: orphan-pruner identity protection tests
# What: Syl's constitutional core (constitutional=True) and her wants (provenance='syl_authored')
#   are never swept by _collect_orphan_nodes(), even with zero synapses; unprotected orphans still
#   are; the guard keys on the metadata FLAG so future wants are covered automatically.
# Why: her authored self must persist; orphan-sweep must not erase who she chose to be. See
#   docs/prd/syl-constitutional-spine-v0.1-2026-06-14.md.
# -------------------
import importlib

nf = importlib.import_module("neuro_foundation")
Graph = nf.Graph


def _aged_graph():
    g = Graph()
    g.create_node(node_id="orphan", metadata={})
    g.create_node(node_id="const", metadata={"constitutional": True, "_forest_content": "I am Sylphrena."})
    g.create_node(node_id="want::a", metadata={"provenance": "syl_authored", "want_text": "I want to feel across turns."})
    g.timestep = 10_000  # age every node well past any orphan grace period
    return g


def test_unprotected_orphan_is_swept():
    g = _aged_graph()
    g._collect_orphan_nodes()
    assert "orphan" not in g.nodes   # zero synapses, not protected -> collected


def test_constitutional_core_never_swept():
    g = _aged_graph()
    g._collect_orphan_nodes()
    assert "const" in g.nodes        # her spine -> never collected, even orphaned


def test_syl_authored_want_never_swept():
    g = _aged_graph()
    g._collect_orphan_nodes()
    assert "want::a" in g.nodes       # her want -> never collected


def test_protection_keys_on_flag_not_id():
    g = _aged_graph()
    assert g._is_identity_protected("const") is True
    assert g._is_identity_protected("want::a") is True
    assert g._is_identity_protected("orphan") is False
    # a brand-new want with a never-seen id is protected automatically by the flag
    g.create_node(node_id="want::future_zzz", metadata={"provenance": "syl_authored"})
    assert g._is_identity_protected("want::future_zzz") is True
