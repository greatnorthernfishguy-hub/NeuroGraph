"""Tests for ng_columnar.IdInterner — #119 increment 2/3 foundation."""

import pytest

from ng_columnar import IdInterner, INT32_MAX


def test_intern_assigns_dense_ascending_indices():
    it = IdInterner()
    assert it.intern("a") == 0
    assert it.intern("b") == 1
    assert it.intern("c") == 2
    assert len(it) == 3
    assert it.capacity == 3


def test_intern_is_idempotent():
    it = IdInterner()
    i = it.intern("node-x")
    assert it.intern("node-x") == i
    assert it.intern("node-x") == i
    assert len(it) == 1


def test_index_does_not_insert():
    it = IdInterner()
    it.intern("known")
    assert it.index("known") == 0
    assert it.index("unknown") == -1
    assert "unknown" not in it
    assert len(it) == 1  # index() lookup did not create a slot


def test_roundtrip_str_int_str():
    it = IdInterner()
    idx = it.intern("uuid-1234")
    assert it.id_of(idx) == "uuid-1234"
    assert it.id_of(999) is None
    assert it.id_of(-1) is None


def test_remove_tombstones_and_recycles_slot():
    it = IdInterner()
    a, b, c = it.intern("a"), it.intern("b"), it.intern("c")
    assert (a, b, c) == (0, 1, 2)
    assert it.remove("b") == 1
    assert "b" not in it
    assert len(it) == 2
    assert it.capacity == 3          # slot still allocated, tombstoned
    assert it.id_of(1) is None       # tombstoned slot reads back as None
    # next intern reuses the freed slot rather than extending
    assert it.intern("d") == 1
    assert it.id_of(1) == "d"
    assert it.capacity == 3          # no growth — slot was recycled


def test_remove_unknown_returns_negative_one():
    it = IdInterner()
    it.intern("a")
    assert it.remove("missing") == -1
    assert len(it) == 1


def test_reintern_after_remove_gets_fresh_index():
    it = IdInterner()
    it.intern("a")               # 0
    old = it.intern("gone")      # 1
    it.remove("gone")
    new = it.intern("gone")      # recycles slot 1
    assert new == old
    assert it.id_of(new) == "gone"


def test_live_indices_skips_tombstones():
    it = IdInterner()
    for s in ("a", "b", "c", "d"):
        it.intern(s)
    it.remove("b")
    it.remove("d")
    assert list(it.live_indices()) == [0, 2]


def test_capacity_stays_tight_under_churn():
    # Repeated add/remove of a rotating working set must not grow capacity
    # without bound — the recycling keeps downstream arrays compact.
    it = IdInterner()
    for s in ("a", "b", "c"):
        it.intern(s)
    assert it.capacity == 3
    for _ in range(1000):
        it.remove("c")
        assert it.intern("c") == 2   # always recycles the one free slot
    assert it.capacity == 3


def test_int32_guard(monkeypatch):
    # Simulate a full index space without allocating billions of entries:
    # the guard trips when the next fresh index would exceed INT32_MAX.
    it = IdInterner()

    class _HugeList(list):
        def __len__(self):
            return INT32_MAX + 1

    monkeypatch.setattr(it, "_idx_to_str", _HugeList())
    with pytest.raises(OverflowError):
        it.intern("overflow")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- SynapseStore / SynapseView ------------------------------------------

import numpy as np
from ng_columnar import SynapseStore, SynapseView


def _store():
    from ng_columnar import IdInterner
    nodes = IdInterner()
    return SynapseStore(nodes, initial_capacity=2), nodes


def test_add_and_read_roundtrip():
    store, nodes = _store()
    v = store.add("s1", "nodeA", "nodeB", weight=0.42, delay=3)
    assert v.synapse_id == "s1"
    assert v.pre_node_id == "nodeA"
    assert v.post_node_id == "nodeB"
    assert v.weight == pytest.approx(0.42, abs=1e-6)
    assert v.delay == 3
    assert "s1" in store
    assert len(store) == 1


def test_defaults_match_dataclass():
    store, _ = _store()
    v = store.add("s1", "a", "b")
    assert v.weight == pytest.approx(0.1)
    assert v.max_weight == pytest.approx(5.0)
    assert v.delay == 1
    assert v.salience == pytest.approx(1.0)
    assert v.peak_weight == pytest.approx(0.1)
    assert v.low_weight_steps == 0
    assert v.inactive_steps == 0


def test_endpoints_share_node_interner():
    store, nodes = _store()
    store.add("s1", "a", "b")
    store.add("s2", "a", "c")   # 'a' reused -> same interned index
    assert store.pre_idx[store.row_of("s1")] == store.pre_idx[store.row_of("s2")]
    assert nodes.index("a") == int(store.pre_idx[store.row_of("s1")])


def test_write_through_mutation_sticks():
    store, _ = _store()
    v = store.add("s1", "a", "b")
    v.weight = 2.5
    v.inactive_steps = 7
    # a fresh view onto the same row sees the change (columns, not a copy)
    v2 = store.view("s1")
    assert v2.weight == pytest.approx(2.5)
    assert v2.inactive_steps == 7


def test_synapse_type_enum_roundtrip():
    from neuro_foundation import SynapseType
    store, _ = _store()
    v = store.add("s1", "a", "b", synapse_type=SynapseType.INHIBITORY)
    assert v.synapse_type == SynapseType.INHIBITORY
    v.synapse_type = SynapseType.MODULATORY
    assert store.view("s1").synapse_type == SynapseType.MODULATORY


def test_metadata_is_sparse_and_mutable():
    store, _ = _store()
    v = store.add("s1", "a", "b")
    # no metadata passed -> nothing stored sparsely
    assert store._metadata == {}
    # accessing creates a persistent dict so in-place mutation sticks
    v.metadata["creation_mode"] = "surprise"
    assert store.view("s1").metadata["creation_mode"] == "surprise"
    # a synapse given metadata at add() carries it
    v2 = store.add("s2", "a", "b", metadata={"k": 1})
    assert v2.metadata == {"k": 1}


def test_capacity_grows_past_initial():
    store, _ = _store()               # initial_capacity=2
    for i in range(10):
        store.add(f"s{i}", "a", f"b{i}", weight=float(i))
    assert len(store) == 10
    for i in range(10):
        assert store.view(f"s{i}").weight == pytest.approx(float(i))


def test_remove_recycles_row_and_columns():
    store, _ = _store()
    store.add("s1", "a", "b", weight=9.0)
    row = store.row_of("s1")
    assert store.remove("s1") is True
    assert "s1" not in store
    assert len(store) == 0
    assert store.remove("s1") is False
    # a new synapse reuses the freed row, with columns reset to defaults
    v = store.add("s2", "a", "c")
    assert v.row == row
    assert v.weight == pytest.approx(0.1)      # not the stale 9.0
    assert store.view("s1") is None


def test_add_same_id_overwrites_in_place():
    store, _ = _store()
    r1 = store.add("s1", "a", "b", weight=1.0).row
    r2 = store.add("s1", "a", "c", weight=2.0).row
    assert r1 == r2                              # same row, overwritten
    assert len(store) == 1
    assert store.view("s1").post_node_id == "c"
    assert store.view("s1").weight == pytest.approx(2.0)


def test_views_and_rows_skip_tombstones():
    store, _ = _store()
    for i in range(4):
        store.add(f"s{i}", "a", "b")
    store.remove("s1")
    ids = sorted(v.synapse_id for v in store.views())
    assert ids == ["s0", "s2", "s3"]


def test_unknown_column_rejected():
    store, _ = _store()
    with pytest.raises(AttributeError):
        store.add("s1", "a", "b", nonsense=1)


# --- SynapseMapping (dict-compatible shim) --------------------------------

from ng_columnar import SynapseMapping
from neuro_foundation import Synapse, SynapseType as _ST


def _mapping():
    from ng_columnar import IdInterner
    nodes = IdInterner()
    return SynapseMapping(SynapseStore(nodes, initial_capacity=2))


def test_mapping_setitem_from_synapse_dataclass():
    m = _mapping()
    syn = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b",
                  weight=0.7, delay=4, synapse_type=_ST.INHIBITORY,
                  creation_time=12.0, eligibility_trace=0.5)
    m["s1"] = syn
    v = m["s1"]
    assert v.weight == pytest.approx(0.7)
    assert v.delay == 4
    assert v.synapse_type == _ST.INHIBITORY
    assert v.creation_time == pytest.approx(12.0)
    assert v.eligibility_trace == pytest.approx(0.5)
    assert v.pre_node_id == "a" and v.post_node_id == "b"


def test_mapping_all_scalar_fields_roundtrip():
    m = _mapping()
    syn = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b",
                  weight=1.1, max_weight=3.3, delay=2, last_update_time=9.0,
                  eligibility_trace=0.25, creation_time=4.0, peak_weight=1.5,
                  low_weight_steps=6, inactive_steps=8, salience=2.2)
    m["s1"] = syn
    v = m["s1"]
    for f in ("weight", "max_weight", "delay", "last_update_time",
              "eligibility_trace", "creation_time", "peak_weight",
              "low_weight_steps", "inactive_steps", "salience"):
        assert getattr(v, f) == pytest.approx(getattr(syn, f)), f


def test_mapping_getitem_missing_raises_keyerror():
    m = _mapping()
    with pytest.raises(KeyError):
        m["nope"]


def test_mapping_get_returns_default():
    m = _mapping()
    assert m.get("nope") is None
    assert m.get("nope", 42) == 42


def test_mapping_contains_len_iter():
    m = _mapping()
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b")
    m["s2"] = Synapse(synapse_id="s2", pre_node_id="a", post_node_id="c")
    assert "s1" in m and "x" not in m
    assert len(m) == 2
    assert sorted(iter(m)) == ["s1", "s2"]


def test_mapping_views_are_reiterable_and_lenable():
    m = _mapping()
    for i in range(3):
        m[f"s{i}"] = Synapse(synapse_id=f"s{i}", pre_node_id="a",
                             post_node_id=f"b{i}")
    vals = m.values()
    assert len(vals) == 3
    # re-iterable (not a one-shot generator)
    ids1 = sorted(v.synapse_id for v in vals)
    ids2 = sorted(v.synapse_id for v in vals)
    assert ids1 == ids2 == ["s0", "s1", "s2"]
    assert sorted(m.keys()) == ["s0", "s1", "s2"]
    assert sorted(k for k, _ in m.items()) == ["s0", "s1", "s2"]


def test_mapping_setitem_overwrites_in_place():
    m = _mapping()
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b", weight=1.0)
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="c", weight=2.0)
    assert len(m) == 1
    assert m["s1"].post_node_id == "c"
    assert m["s1"].weight == pytest.approx(2.0)


def test_mapping_delitem():
    m = _mapping()
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b")
    del m["s1"]
    assert "s1" not in m
    with pytest.raises(KeyError):
        del m["s1"]


def test_mapping_pop_returns_detached_snapshot():
    m = _mapping()
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b",
                      weight=3.0, metadata={"creation_mode": "surprise"})
    snap = m.pop("s1")
    assert "s1" not in m
    # snapshot survives row recycling
    m["s2"] = Synapse(synapse_id="s2", pre_node_id="a", post_node_id="c", weight=9.0)
    assert snap.synapse_id == "s1"
    assert snap.pre_node_id == "a" and snap.post_node_id == "b"
    assert snap.weight == pytest.approx(3.0)
    assert snap.metadata == {"creation_mode": "surprise"}


def test_mapping_pop_missing_default_and_keyerror():
    m = _mapping()
    assert m.pop("nope", None) is None
    assert m.pop("nope", "d") == "d"
    with pytest.raises(KeyError):
        m.pop("nope")


def test_mapping_clear():
    m = _mapping()
    for i in range(5):
        m[f"s{i}"] = Synapse(synapse_id=f"s{i}", pre_node_id="a", post_node_id=f"b{i}")
    m.clear()
    assert len(m) == 0
    assert list(m.values()) == []


def test_mapping_writes_through_view():
    m = _mapping()
    m["s1"] = Synapse(synapse_id="s1", pre_node_id="a", post_node_id="b")
    m["s1"].weight = 4.5
    m["s1"].inactive_steps = 11
    assert m["s1"].weight == pytest.approx(4.5)
    assert m["s1"].inactive_steps == 11


# --- fidelity: dynamics must match float64 dataclass baseline --------------

def test_iterated_trace_decay_matches_float64():
    """Repeated multiplicative decay must not drift from float64 (float32 would).

    eligibility_trace *= decay every step is exactly the pattern at
    neuro_foundation.py:2464/3863; float32 storage would visibly diverge.
    """
    store, _ = _store()
    v = store.add("s1", "a", "b", eligibility_trace=1.0)
    ref = 1.0
    decay = 0.97
    for _ in range(5000):
        v.eligibility_trace *= decay
        ref *= decay
    assert v.eligibility_trace == pytest.approx(ref, rel=1e-12, abs=1e-300)


def test_salience_ema_matches_float64():
    """salience EMA (neuro_foundation.py:2508) accumulated over many steps."""
    store, _ = _store()
    v = store.add("s1", "a", "b", salience=3.0)
    ref = 3.0
    sal_decay = 0.02
    for _ in range(5000):
        v.salience = 1.0 + (v.salience - 1.0) * (1.0 - sal_decay)
        ref = 1.0 + (ref - 1.0) * (1.0 - sal_decay)
    assert v.salience == pytest.approx(ref, rel=1e-12, abs=1e-300)


def test_all_float_columns_are_float64():
    store, _ = _store()
    import numpy as np
    for name in ("weight", "max_weight", "last_update_time", "eligibility_trace",
                 "creation_time", "peak_weight", "salience"):
        assert store._cols[name].dtype == np.float64, name


# --- CSRAdjacency + AdjacencyView (#119: replaces _outgoing/_incoming) ------

from ng_columnar import CSRAdjacency, AdjacencyView


def _adj():
    """Store with a small fixed topology + both direction views over it.

    Topology (pre -> post):  a->b, a->c, b->c   (node d has no edges)
    """
    store, nodes = _store()
    store.add("s_ab", "a", "b")
    store.add("s_ac", "a", "c")
    store.add("s_bc", "b", "c")
    nodes.intern("d")                       # edgeless but registered node
    csr = CSRAdjacency(store)
    out = AdjacencyView(csr, nodes, store, "out")
    inc = AdjacencyView(csr, nodes, store, "in")
    return store, nodes, csr, out, inc


def test_csr_out_rows_group_by_pre():
    store, nodes, csr, out, inc = _adj()
    a = nodes.index("a")
    rows = set(int(r) for r in csr.out_rows(a))
    assert rows == {store.row_of("s_ab"), store.row_of("s_ac")}
    assert csr.out_degree(a) == 2


def test_csr_in_rows_group_by_post():
    store, nodes, csr, out, inc = _adj()
    c = nodes.index("c")
    rows = set(int(r) for r in csr.in_rows(c))
    assert rows == {store.row_of("s_ac"), store.row_of("s_bc")}
    assert csr.in_degree(c) == 2


def test_adjacency_view_materializes_synapse_id_sets():
    store, nodes, csr, out, inc = _adj()
    assert out["a"] == {"s_ab", "s_ac"}
    assert out["b"] == {"s_bc"}
    assert inc["c"] == {"s_ac", "s_bc"}
    assert inc["b"] == {"s_ab"}


def test_edgeless_node_is_empty_not_missing():
    store, nodes, csr, out, inc = _adj()
    assert out["d"] == set()          # registered -> present, empty
    assert inc["d"] == set()
    assert "d" in out
    assert out.get("d", ()) == set()


def test_get_returns_default_for_unknown_node():
    store, nodes, csr, out, inc = _adj()
    assert out.get("nope", ()) == ()
    assert out.get("nope") is None
    assert "nope" not in out
    with pytest.raises(KeyError):
        out["nope"]


def test_union_pattern_matches_call_site():
    # neuro_foundation.py:1278 -> _outgoing.get(nid) | _incoming.get(nid)
    store, nodes, csr, out, inc = _adj()
    both = out.get("c", set()) | inc.get("c", set())
    assert both == {"s_ac", "s_bc"}   # c is only a post-endpoint here


def test_keys_iter_and_len_cover_all_nodes():
    store, nodes, csr, out, inc = _adj()
    assert set(out) == {"a", "b", "c", "d"}
    assert len(out) == 4


def test_version_cache_rebuilds_on_add():
    store, nodes, csr, out, inc = _adj()
    assert out["a"] == {"s_ab", "s_ac"}
    v0 = csr._built_version
    store.add("s_ad", "a", "d")            # bumps store.version
    assert out["a"] == {"s_ab", "s_ac", "s_ad"}   # view reflects it
    assert csr._built_version != v0


def test_version_cache_no_rebuild_without_change():
    store, nodes, csr, out, inc = _adj()
    out["a"]                                # force a build
    built = csr._built_version
    out["a"]; out["b"]; inc["c"]            # pure reads
    assert csr._built_version == built      # not rebuilt


def test_removal_reflected_after_rebuild():
    store, nodes, csr, out, inc = _adj()
    assert inc["c"] == {"s_ac", "s_bc"}
    store.remove("s_ac")
    assert out["a"] == {"s_ab"}
    assert inc["c"] == {"s_bc"}


def test_returned_set_add_discard_is_throwaway():
    # add_synapse does `_outgoing[pre].add(sid)`; the store is the real truth,
    # so mutating the materialised set must not corrupt the derived view.
    store, nodes, csr, out, inc = _adj()
    out["a"].add("garbage")
    out["a"].discard("s_ab")
    assert out["a"] == {"s_ab", "s_ac"}     # unchanged on re-derive


def test_setitem_registers_edgeless_node():
    store, nodes, csr, out, inc = _adj()
    assert "z" not in out
    out["z"] = set()                        # node-creation registration pattern
    assert "z" in out
    assert out["z"] == set()


def test_del_refuses_node_with_live_edges():
    store, nodes, csr, out, inc = _adj()
    with pytest.raises(RuntimeError):
        del out["a"]                        # a still has outgoing synapses
    assert "a" in out


def test_del_and_pop_edgeless_node():
    store, nodes, csr, out, inc = _adj()
    del out["d"]
    assert "d" not in out
    nodes.intern("e")
    assert out.pop("e") == set()
    assert "e" not in out
    assert out.pop("missing", "sentinel") == "sentinel"


def test_clear_is_noop():
    store, nodes, csr, out, inc = _adj()
    out.clear()
    assert out["a"] == {"s_ab", "s_ac"}     # derived edges untouched
    assert set(out) == {"a", "b", "c", "d"}


def test_empty_store_adjacency():
    store, nodes = _store()
    csr = CSRAdjacency(store)
    out = AdjacencyView(csr, nodes, store, "out")
    assert len(out) == 0
    assert out.get("anything", ()) == ()
    assert csr.out_degree(0) == 0


def test_direction_argument_validated():
    store, nodes = _store()
    csr = CSRAdjacency(store)
    with pytest.raises(ValueError):
        AdjacencyView(csr, nodes, store, "sideways")
