# ---- Changelog ----
# [2026-07-05] CC (laptop) — regression test for surfacing.py's hyperedge-scan race
# What: _score_node() iterated graph.hyperedges.values() directly on every call -- the
#   same class of race as lenia/graph_substrate.py's _hyperedge_similarity (#88/#341),
#   just not yet observed crashing here. Fixed by snapshotting under _step_lock, same
#   pattern as tests/test_graph_substrate_race.py.
# -------------------
import threading

from ces_config import load_ces_config
from neuro_foundation import Graph
from surfacing import SurfacingMonitor


class _StubVectorDB:
    def get(self, node_id):
        return None


def test_score_node_hyperedge_scan_no_race_under_concurrent_he_mutation():
    g = Graph()
    for i in range(50):
        g.create_node(node_id=f"n{i}")
    he = g.create_hyperedge(member_node_ids={f"n{i}" for i in range(10)})
    monitor = SurfacingMonitor(g, _StubVectorDB(), load_ces_config())

    stop = threading.Event()
    errors = []

    def writer():
        i = 100000
        while not stop.is_set():
            with g._step_lock:
                g.hyperedges[f"h{i}"] = he
                old = f"h{i - 60}"
                if old in g.hyperedges:
                    del g.hyperedges[old]
            i += 1

    def reader():
        try:
            node = g.nodes["n0"]
            for _ in range(5000):
                monitor._score_node("n0", node)
        except Exception as e:  # noqa: BLE001 -- catching the race is the point
            errors.append(repr(e))

    wt = threading.Thread(target=writer, daemon=True)
    rt = threading.Thread(target=reader, daemon=True)
    wt.start()
    rt.start()
    rt.join(timeout=30)
    stop.set()
    wt.join(timeout=5)

    assert not rt.is_alive(), "reader did not finish (possible deadlock)"
    assert errors == [], f"concurrent mutation raced: {errors[:3]}"
