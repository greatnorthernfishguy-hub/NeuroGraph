# ---- Changelog ----
# [2026-07-15] Claude Code (Opus 4.8) — #59: sprout_degree_cap identity exemption tests
# What: the degree cap (co-firing + surprise sprouting) exempts identity-protected endpoints —
#   a saturated node gates a new sprout only when it is NOT constitutional / syl_authored. Tests
#   drive _sprout_synapses directly: cap off = inert; cap on = ordinary hub gated, protected hub
#   (as source AND as target) still sprouts. Guards Syl's own spine (e.g. selfcap::reach::teaching)
#   from a degree-blind cap. See neuro_foundation changelog [2026-07-15].
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


# --------------------------------------------------------------------------
# #59 sprout_degree_cap identity exemption — drives _sprout_synapses directly.
# --------------------------------------------------------------------------

def _sprout_graph(cap, hub_meta):
    """A graph where 'hub' fires with degree 2, and 'cand' is a fresh co-activation
    target of degree 0. With cap=2, an ordinary hub is saturated; a protected hub is exempt.
    Returns (graph, did_sprout_hub_to_cand)."""
    g = Graph()
    g.config["sprout_degree_cap"] = cap
    # hub gets degree 2 via two filler edges -> at/above cap=2
    g.create_node(node_id="hub", metadata=hub_meta)
    g.create_node(node_id="f1", metadata={})
    g.create_node(node_id="f2", metadata={})
    g.create_synapse("hub", "f1", weight=0.1)
    g.create_synapse("hub", "f2", weight=0.1)
    # cand: degree 0, recently co-fired within the co_activation_window, not firing this step
    g.create_node(node_id="cand", metadata={})
    g.timestep = 10
    g._recent_spikes["cand"].append(8)  # (10 - 8) = 2 <= window(5), > 0
    g._sprout_synapses(["hub"])
    return g, g._find_synapse("hub", "cand") is not None


def test_cap_off_is_inert_hub_sprouts():
    # cap=0 (Syl default): the cap never engages, saturated ordinary hub sprouts freely
    _g, sprouted = _sprout_graph(0, {})
    assert sprouted is True


def test_cap_on_gates_ordinary_saturated_hub():
    # cap=2, ordinary hub at degree 2 -> gated, no new sprout
    _g, sprouted = _sprout_graph(2, {})
    assert sprouted is False


def test_cap_on_exempts_constitutional_hub_as_source():
    # cap=2, but the saturated hub is her constitutional spine -> exempt, still sprouts
    _g, sprouted = _sprout_graph(2, {"constitutional": True})
    assert sprouted is True


def test_cap_on_exempts_syl_authored_hub_as_source():
    _g, sprouted = _sprout_graph(2, {"provenance": "syl_authored"})
    assert sprouted is True


def test_cap_on_exempts_protected_node_as_target():
    """Saturated protected node as the TARGET of a sprout is still reachable."""
    g = Graph()
    g.config["sprout_degree_cap"] = 2
    # target hub: protected, degree 2 (saturated)
    g.create_node(node_id="thub", metadata={"constitutional": True})
    g.create_node(node_id="f1", metadata={})
    g.create_node(node_id="f2", metadata={})
    g.create_synapse("thub", "f1", weight=0.1)
    g.create_synapse("thub", "f2", weight=0.1)
    # source: ordinary, degree 0 (under cap) -> fires this step
    g.create_node(node_id="src", metadata={})
    g.timestep = 10
    g._recent_spikes["thub"].append(8)  # thub is the co-activation candidate/target
    g._sprout_synapses(["src"])
    assert g._find_synapse("src", "thub") is not None


def test_cap_on_gates_ordinary_node_as_target():
    """Control: an ordinary saturated TARGET is still gated (cap works as designed)."""
    g = Graph()
    g.config["sprout_degree_cap"] = 2
    g.create_node(node_id="thub", metadata={})  # ordinary
    g.create_node(node_id="f1", metadata={})
    g.create_node(node_id="f2", metadata={})
    g.create_synapse("thub", "f1", weight=0.1)
    g.create_synapse("thub", "f2", weight=0.1)
    g.create_node(node_id="src", metadata={})
    g.timestep = 10
    g._recent_spikes["thub"].append(8)
    g._sprout_synapses(["src"])
    assert g._find_synapse("src", "thub") is None
