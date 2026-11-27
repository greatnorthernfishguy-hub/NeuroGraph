# ---- Changelog ----
# [2026-11-26] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1201) -- CREATE
# What: SimpleVectorDB.search with native_scan on against the same store with it off; the switch's default; the
#       organism's enabler; an unreadable row raises with the switch on.
# Why:  the seed scan as one native call (ng_tract.vdb_scan) must give search() the same answer as the Python loop.
# How:  two stores built from the same random unit vectors; ids, order and values compared (values to 2e-6).
# -------------------
import importlib

import numpy as np
import pytest

import ng_tract
from universal_ingestor import SimpleVectorDB

pytestmark = pytest.mark.skipif(not hasattr(ng_tract, "vdb_scan"), reason="this ng_tract wheel has no vdb_scan")


def _stores(n=2000, d=768, seed=11):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    a, b = SimpleVectorDB(), SimpleVectorDB()
    for i in range(n):
        a.insert("n%d" % i, m[i], "text %d" % i, {"i": i})
        b.insert("n%d" % i, m[i], "text %d" % i, {"i": i})
    b.native_scan = True
    return a, b, m


def test_default_is_the_python_loop():
    assert SimpleVectorDB.native_scan is False
    assert SimpleVectorDB().native_scan is False


@pytest.mark.parametrize("k,thr", [(10, 0.4), (5, 0.7), (50, 0.0), (10, -1.0), (3, 0.99)])
def test_search_gives_the_same_ids_order_and_values(k, thr):
    a, b, m = _stores()
    rng = np.random.default_rng(5)
    for j in (0, 17, 400, 1999):
        q = m[j] * 0.7 + rng.standard_normal(m.shape[1]).astype(np.float32) * 0.03
        ra, rb = a.search(q, k=k, threshold=thr), b.search(q, k=k, threshold=thr)
        assert [i for i, _ in ra] == [i for i, _ in rb]
        for (_, x), (_, y) in zip(ra, rb):
            assert abs(x - y) < 2e-6


def test_empty_store_and_no_hit():
    b = SimpleVectorDB()
    b.native_scan = True
    assert b.search(np.ones(8, dtype=np.float32)) == []
    b.insert("x", np.array([1.0, 0.0], dtype=np.float32))
    assert b.search(np.array([0.0, 1.0], dtype=np.float32), threshold=0.5) == []
    assert [i for i, _ in b.search(np.array([2.0, 0.0], dtype=np.float32), threshold=0.5)] == ["x"]


def test_float64_vectors_and_queries():
    b = SimpleVectorDB()
    b.native_scan = True
    b.insert("x", np.array([3.0, 4.0]))           # float64 in, float64 stored
    b.insert("y", np.array([4.0, 3.0], dtype=np.float32))
    r = dict(b.search(np.array([3.0, 4.0]), k=5, threshold=0.0))
    assert abs(r["x"] - 1.0) < 1e-6 and abs(r["y"] - 0.96) < 1e-6


def test_deleted_and_reinserted_ids_are_revalidated_as_before(monkeypatch):
    a, b, m = _stores(n=50, d=16)
    real = ng_tract.vdb_scan

    def scan_then_mutate(snapshot, q, thr):
        out = real(snapshot, q, thr)
        b.delete("n3")                                        # gone after the snapshot: must not be returned
        b.insert("n4", -m[4], "flipped", {})                 # re-inserted: re-scored against the new vector
        return out
    monkeypatch.setattr(ng_tract, "vdb_scan", scan_then_mutate)
    q3 = b.search(m[3], k=50, threshold=-1.0)
    ids = [i for i, _ in q3]
    assert "n3" not in ids and "n4" in ids
    assert abs(dict(q3)["n4"] - float(np.dot(m[3] / np.linalg.norm(m[3]), b.get_embedding("n4")))) < 2e-6


def test_an_unreadable_row_raises_with_the_switch_on():
    b = SimpleVectorDB()
    b.native_scan = True
    b.insert("x", np.ones(4, dtype=np.float32))
    b.embeddings["bad"] = [1.0, 0.0, 0.0, 0.0]                # not an array: the loop would have taken it
    with pytest.raises(TypeError):
        b.search(np.ones(4, dtype=np.float32), threshold=0.0)


def test_the_organism_switch(monkeypatch):
    import cc_ng_organism as org
    vdb = SimpleVectorDB()
    monkeypatch.setattr(org, "VDB_NATIVE_SCAN", False)
    assert org.cc_enable_native_vdb_scan(vdb) is False and vdb.native_scan is False
    monkeypatch.setattr(org, "VDB_NATIVE_SCAN", True)
    assert org.cc_enable_native_vdb_scan(None) is False
    assert org.cc_enable_native_vdb_scan(vdb) is True and vdb.native_scan is True
    monkeypatch.delattr(ng_tract, "vdb_scan")
    with pytest.raises(RuntimeError):
        org.cc_enable_native_vdb_scan(SimpleVectorDB())
