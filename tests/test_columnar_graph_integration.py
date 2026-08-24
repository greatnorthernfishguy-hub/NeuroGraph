"""Graph-level regression tests for the #119 increment 3 columnar substrate.

The unit-level behaviour of the shims (IdInterner / SynapseStore /
SynapseMapping / CSRAdjacency / AdjacencyView) is covered in
``test_ng_columnar.py``.  This file locks down the five *integration* fixes
made where ``neuro_foundation.Graph`` drives those shims — each one a bug that
only appears through the Graph API, not the shim in isolation:

  A. ``create_synapse`` returns the live write-through handle (not a boxed,
     store-decoupled ``Synapse``), so callers see later plasticity updates.
  B. Serialization reads metadata via ``peek_metadata`` and must NOT densify
     the store's sparse ``_metadata`` map.
  C. Checkpointing snapshots synapses as immutable detached copies, immune to a
     concurrent tombstone/recycle of the underlying column row.
  D. ``_deserialize`` rebuilds the interner/store/CSR fresh, leaving no phantom
     nodes and no interner growth when restoring into a reused Graph.
  E. Edge insert/remove is O(1) in node degree — adjacency is derived from the
     store, so the hot paths must NOT materialise a node's whole neighbour set
     (the regression that made graph build/prune O(degree^2)).
"""

import ng_columnar
from neuro_foundation import Graph


# --- Fix A: create_synapse returns a live write-through handle --------------

def test_create_synapse_returns_live_write_through_handle():
    g = Graph()
    for n in ("A", "B"):
        g.create_node(node_id=n)
    syn = g.create_synapse("A", "B", weight=0.1)
    sid = syn.synapse_id

    # a store-side mutation (as plasticity would make) is visible on the handle
    g.synapses[sid].weight = 0.9
    assert syn.weight == 0.9

    # and the reverse: mutating the returned handle reaches the store
    syn.eligibility_trace = 0.5
    assert g.synapses[sid].eligibility_trace == 0.5


# --- Fix B: serialization must not densify the sparse metadata map ----------

def test_serialize_does_not_densify_sparse_metadata():
    g = Graph()
    for n in ("A", "B", "C"):
        g.create_node(node_id=n)
    g.create_synapse("A", "B")
    g.create_synapse("B", "C")

    store = g._synapse_store
    assert len(store._metadata) == 0            # nothing set -> sparse map empty

    g._serialize_full()

    # serializing every synapse must NOT have persisted an empty dict per row
    assert len(store._metadata) == 0


def test_serialize_preserves_real_metadata_without_extra_rows():
    g = Graph()
    for n in ("A", "B", "C"):
        g.create_node(node_id=n)
    syn = g.create_synapse("A", "B")
    g.create_synapse("B", "C")            # left metadata-free
    syn.metadata["tag"] = "keeper"        # densifies exactly one row (intended)

    store = g._synapse_store
    assert len(store._metadata) == 1

    data = g._serialize_full()

    assert data["synapses"][syn.synapse_id]["metadata"] == {"tag": "keeper"}
    # the metadata-free synapse serialized to an empty dict, still not persisted
    assert len(store._metadata) == 1


# --- Fix C: checkpoint snapshot is immune to row recycle -------------------

def test_checkpoint_snapshot_is_immune_to_row_recycle():
    g = Graph()
    for n in ("A", "B", "C"):
        g.create_node(node_id=n)
    s1 = g.create_synapse("A", "B", weight=0.5)
    sid1 = s1.synapse_id

    snap = dict(g.synapses.snapshot_items())
    assert snap[sid1].weight == 0.5
    # a snapshot value is a detached copy, not a live (store, row) view
    assert type(snap[sid1]).__name__ == "_DetachedSynapse"

    # tombstone sid1's row, then add a synapse that recycles the freed slot
    g.remove_synapse(sid1)
    g.create_synapse("B", "C", weight=0.9)

    # the pre-recycle snapshot still reads the old synapse, not the recycled one
    assert snap[sid1].weight == 0.5


# --- Fix D: restore into a reused Graph leaves no ghosts --------------------

def test_restore_into_reused_graph_leaves_no_phantom_nodes():
    g = Graph()
    for n in ("A", "B", "C", "D"):
        g.create_node(node_id=n)
    g.create_synapse("A", "B")
    g.create_synapse("C", "D")
    data = g._serialize_full()

    # reuse a Graph that already holds different, heavier state
    g2 = Graph()
    for n in ("X", "Y", "Z"):
        g2.create_node(node_id=n)
    g2.create_synapse("X", "Y")
    g2.create_synapse("Y", "Z")

    g2._deserialize(data)

    assert set(g2.nodes.keys()) == {"A", "B", "C", "D"}
    for ghost in ("X", "Y", "Z"):
        assert ghost not in g2._outgoing
        assert ghost not in g2._incoming
    # adjacency reports exactly the restored nodes, no phantoms
    assert len(g2._outgoing) == 4
    assert len(g2._outgoing["A"]) == 1
    (a_edge,) = g2._outgoing["A"]
    assert g2.synapses[a_edge].post_node_id == "B"


def test_repeated_restore_does_not_grow_interner():
    g = Graph()
    for n in ("A", "B"):
        g.create_node(node_id=n)
    g.create_synapse("A", "B")
    data = g._serialize_full()

    g2 = Graph()
    g2._deserialize(data)
    size1 = len(g2._node_interner)
    g2._deserialize(data)
    size2 = len(g2._node_interner)
    g2._deserialize(data)
    size3 = len(g2._node_interner)

    assert size1 == size2 == size3        # fresh interner each restore, no growth


# --- Fix E: edge mutation is O(1) in degree (derived adjacency) -------------

def test_create_synapse_does_not_materialize_adjacency(monkeypatch):
    """Guards the O(degree^2) build regression: create_synapse must not
    materialise a node's neighbour set (the store is adjacency's source of
    truth, so the old `_outgoing[pre].add(sid)` was both redundant and
    quadratic)."""
    g = Graph()
    for n in ("A", "B", "C"):
        g.create_node(node_id=n)

    calls = {"n": 0}
    orig = ng_columnar.AdjacencyView._materialize

    def spy(self, node_idx):
        calls["n"] += 1
        return orig(self, node_idx)

    monkeypatch.setattr(ng_columnar.AdjacencyView, "_materialize", spy)

    g.create_synapse("A", "B")
    g.create_synapse("A", "C")

    assert calls["n"] == 0


def test_remove_synapse_does_not_materialize_adjacency(monkeypatch):
    """Companion to the build guard: pruning must also be O(1) in degree."""
    g = Graph()
    for n in ("A", "B"):
        g.create_node(node_id=n)
    sid = g.create_synapse("A", "B").synapse_id

    calls = {"n": 0}
    orig = ng_columnar.AdjacencyView._materialize

    def spy(self, node_idx):
        calls["n"] += 1
        return orig(self, node_idx)

    monkeypatch.setattr(ng_columnar.AdjacencyView, "_materialize", spy)

    g.remove_synapse(sid)

    assert calls["n"] == 0


def test_derived_adjacency_reflects_add_and_remove():
    """Correctness backstop for Fix E: dropping the explicit adjacency writes
    must not change what `_outgoing`/`_incoming` report."""
    g = Graph()
    g.create_node(node_id="hub")
    sids = []
    for i in range(50):
        leaf = f"leaf{i}"
        g.create_node(node_id=leaf)
        sids.append(g.create_synapse("hub", leaf).synapse_id)

    assert g._outgoing["hub"] == set(sids)
    assert all(len(g._incoming[f"leaf{i}"]) == 1 for i in range(50))

    victim = sids[0]
    post = g.synapses[victim].post_node_id
    g.remove_synapse(victim)

    assert victim not in g._outgoing["hub"]
    assert len(g._outgoing["hub"]) == 49
    assert len(g._incoming[post]) == 0
