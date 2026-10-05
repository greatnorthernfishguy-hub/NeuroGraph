# ---- Changelog ----
# [2026-10-04] Claude (lane vdb-lock-leak) — vector leak on orphan sweep + SimpleVectorDB thread safety (#270)
# What: proves (1) LEAK: a sweep that collects N nodes deletes exactly their N vectors through the
#   NeuroGraphMemory nodes_collected handler (registered by __init__ on a real NeuroGraphMemory), leaves
#   every other vector alone, carries node_ids on the event, keeps the existing count/timestep listener
#   shape working, and a failing delete never breaks step(); (2) CONCURRENCY: real threads hammer
#   search/get/count/all_ids/items_snapshot/capture_state while others insert, re-insert, delete and run
#   real graph sweeps whose handler deletes vectors -- zero exceptions, no torn entries, every search hit
#   exists when returned; CONTROLS run the same hammer against (a) this class with a no-op lock and
#   (b) a frozen copy of the pre-lane class, and assert the hammer DOES catch failures there;
#   (3) DEADLOCK GUARD: every store method completes while another thread holds the graph _step_lock,
#   a step() whose sweep deletes vectors completes after a vdb-lock holder releases, and a control store
#   that (wrongly) takes the graph lock under its own lock is caught by the same guard; (4) FORMAT: the
#   lock changes nothing on disk (msgpack + json byte-identical to the pre-lane class; save/load/save
#   identical) and pickling drops/re-creates the lock.
# Why: Josh 2026-10-04: fix the vector leak and make concurrent store access CORRECT, not merely
#   less likely to fail ("I don't like the concurrent risk at all, either form").
# How: pure in-process tests; small synthetic vectors; sys.setswitchinterval lowered during hammers.
# [2026-10-04] Claude (lane vdb-lock-leak, resume) — tightened the mid-search mutation assertion; added the
#   in-place graph.restore() persistence test for the handler registration.
# -------------------
"""Vector leak on orphan sweep + SimpleVectorDB leaf lock (#270)."""

import copy
import functools
import logging
import os
import pickle
import random
import shutil
import sys
import tempfile
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph
from universal_ingestor import SimpleVectorDB

DIM = 32


def _vec(rng):
    return rng.standard_normal(DIM).astype(np.float32)


def _orphan_graph():
    """A graph whose next sweep collects every unwired node (grace 0, timestep advanced)."""
    return Graph(config={"orphan_node_grace_period": 0})


def _owner_handler(vdb):
    """The REAL NeuroGraphMemory handler bound to a minimal owner holding `vdb`."""
    from openclaw_hook import NeuroGraphMemory
    owner = types.SimpleNamespace(vector_db=vdb)
    return functools.partial(NeuroGraphMemory._drop_collected_vectors, owner)


# ---------------------------------------------------------------------------
# Frozen copy of the PRE-LANE class (base 5246b63) — used ONLY as a control
# and as the on-disk format reference. Do not "fix" it.
# ---------------------------------------------------------------------------
class _PreLaneSimpleVectorDB:
    def __init__(self):
        self.embeddings, self.metadata, self.content = {}, {}, {}

    def insert(self, id, embedding, content="", metadata=None):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.embeddings[id] = embedding
        self.content[id] = content
        self.metadata[id] = metadata or {}

    def search(self, query_vector, k=5, threshold=0.7):
        if not self.embeddings:
            return []
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        results = []
        for id, emb in self.embeddings.items():
            sim = float(np.dot(query_vector, emb))
            if sim >= threshold:
                results.append((id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get(self, id):
        if id not in self.embeddings:
            return None
        return {"id": id, "embedding": self.embeddings[id], "content": self.content[id],
                "metadata": self.metadata[id]}

    def delete(self, id):
        if id not in self.embeddings:
            return False
        del self.embeddings[id]
        del self.content[id]
        del self.metadata[id]
        return True

    def count(self):
        return len(self.embeddings)

    def all_ids(self):
        return list(self.embeddings.keys())

    def items_snapshot(self):  # what an external caller did before: iterate the live dict
        return [(i, c, self.metadata[i]) for i, c in self.content.items()]

    def capture_state(self, detach=True):
        entries = {}
        for id in list(self.embeddings.keys()):
            emb = self.embeddings[id]
            meta = self.metadata.get(id, {})
            entries[id] = {"embedding": emb.astype(np.float32).tobytes(),
                           "content": self.content.get(id, ""),
                           "metadata": copy.deepcopy(meta) if detach else meta}
        return {"version": "1.0.0", "count": len(entries), "entries": entries}

    write_state = SimpleVectorDB.write_state

    # verbatim pre-lane load()
    def load(self, path):
        """Restore vector DB state from disk.

        Clears existing state before loading. Embeddings are restored
        as L2-normalized float32 numpy arrays.

        Args:
            path: File path to load from (.msgpack or .json).

        Returns:
            Number of entries loaded.
        """
        import numpy as np

        if path.endswith(".msgpack"):
            try:
                import msgpack
            except ImportError:
                raise ImportError("msgpack required for .msgpack deserialization")
            with open(path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
        else:
            import json
            import base64
            with open(path, "r") as f:
                json_data = json.load(f)
            # Convert base64 back to bytes
            data = {
                "version": json_data.get("version", "1.0.0"),
                "count": json_data.get("count", 0),
                "entries": {},
            }
            for id, entry in json_data.get("entries", {}).items():
                data["entries"][id] = {
                    "embedding": base64.b64decode(entry["embedding_b64"]),
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                }

        # Clear existing state
        self.embeddings.clear()
        self.content.clear()
        self.metadata.clear()

        # Restore entries
        for id, entry in data.get("entries", {}).items():
            embedding_bytes = entry["embedding"]
            vec = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
            # Vectors should already be normalized from when they were saved,
            # but re-normalize to be safe
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings[id] = vec
            self.content[id] = entry.get("content", "")
            self.metadata[id] = entry.get("metadata", {})

        return len(self.embeddings)


class _NoOpLock:
    def acquire(self, *a, **k):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _UnlockedSimpleVectorDB(SimpleVectorDB):
    """This lane's class with the lock stubbed out — proves the lock is load-bearing."""

    @staticmethod
    def _make_lock():
        return _NoOpLock()


# ---------------------------------------------------------------------------
# 1. The leak
# ---------------------------------------------------------------------------

def _seed(g, vdb, rng, n_orphan, n_wired, n_vector_only):
    orphans = ["orph_%d" % i for i in range(n_orphan)]
    wired = ["wired_%d" % i for i in range(n_wired)]
    vec_only = ["doc_%d" % i for i in range(n_vector_only)]
    for nid in orphans + wired:
        g.create_node(node_id=nid)
        vdb.insert(nid, _vec(rng), content="c:" + nid, metadata={"v": 0})
    for i in range(0, n_wired - 1, 2):
        g.create_synapse(wired[i], wired[i + 1], weight=0.5)
    for vid in vec_only:
        vdb.insert(vid, _vec(rng), content="c:" + vid, metadata={"creation_mode": "ingested"})
    g.timestep += 5
    return orphans, wired, vec_only


def test_sweep_deletes_exactly_the_collected_nodes_vectors_and_event_carries_node_ids():
    rng = np.random.default_rng(1)
    g, vdb = _orphan_graph(), SimpleVectorDB()
    orphans, wired, vec_only = _seed(g, vdb, rng, n_orphan=7, n_wired=6, n_vector_only=4)
    events = []
    g.register_event_handler("nodes_collected", lambda **kw: events.append(kw))
    g.register_event_handler("nodes_collected", _owner_handler(vdb))
    before = set(vdb.all_ids())

    removed = g._collect_orphan_nodes()

    assert removed == 7
    assert len(events) == 1 and events[0]["count"] == 7
    assert sorted(events[0]["node_ids"]) == sorted(orphans)
    assert "timestep" in events[0]
    after = set(vdb.all_ids())
    assert before - after == set(orphans)               # exactly the N collected vectors went
    assert after == set(wired) | set(vec_only)           # everything else untouched
    assert not (set(events[0]["node_ids"]) & after)      # no collected id keeps a vector
    for vid in after:                                    # survivors intact
        assert vdb.get(vid)["content"] == "c:" + vid


def test_sweep_through_step_deletes_vectors_and_daemon_style_listener_still_works():
    rng = np.random.default_rng(2)
    g, vdb = _orphan_graph(), SimpleVectorDB()
    orphans, wired, vec_only = _seed(g, vdb, rng, n_orphan=5, n_wired=4, n_vector_only=2)
    seen = []

    def _on_nodes_collected(count=0, timestep=0, **_):   # the laptop daemon's reap-logger signature
        seen.append((count, timestep))

    def _legacy_kw(**kw):                                 # the in-tree test listeners' signature
        seen.append(("kw", kw.get("count")))

    g.register_event_handler("nodes_collected", _on_nodes_collected)
    g.register_event_handler("nodes_collected", _legacy_kw)
    g.register_event_handler("nodes_collected", _owner_handler(vdb))
    g.step()
    assert set(orphans).isdisjoint(g.nodes)
    assert seen[0][0] == 5 and seen[1] == ("kw", 5)
    assert set(vdb.all_ids()) == set(wired) | set(vec_only)


def test_handler_without_node_ids_is_a_noop():
    vdb = SimpleVectorDB()
    vdb.insert("a", np.ones(DIM))
    _owner_handler(vdb)(count=3, timestep=9)              # an older engine's emit
    assert vdb.all_ids() == ["a"]


def test_failing_delete_never_breaks_step_and_warns_with_count_and_class_only(caplog):
    class _Boom(SimpleVectorDB):
        def delete(self, id):
            raise ValueError("secret-ish message " + id)

    rng = np.random.default_rng(3)
    g, vdb = _orphan_graph(), _Boom()
    orphans, _w, _v = _seed(g, vdb, rng, n_orphan=4, n_wired=2, n_vector_only=0)
    g.register_event_handler("nodes_collected", _owner_handler(vdb))
    with caplog.at_level(logging.WARNING, logger="neurograph"):
        g.step()                                          # must not raise
    assert set(orphans).isdisjoint(g.nodes)
    warns = [r for r in caplog.records if "vector delete failed" in r.getMessage()]
    assert len(warns) == 1 and warns[0].levelno == logging.WARNING
    msg = warns[0].getMessage()
    assert "4 of 4" in msg and "ValueError" in msg and "secret-ish" not in msg


def test_missing_vector_store_is_fail_soft(caplog):
    from openclaw_hook import NeuroGraphMemory
    owner = types.SimpleNamespace()                       # no vector_db at all
    with caplog.at_level(logging.WARNING, logger="neurograph"):
        NeuroGraphMemory._drop_collected_vectors(owner, node_ids=["a", "b"], count=2, timestep=1)
    msg = [r.getMessage() for r in caplog.records if "vector delete failed" in r.getMessage()][0]
    assert "2 of 2" in msg and "AttributeError" in msg


@pytest.fixture
def real_memory():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix="vdb_lock_leak_")
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False},
                                  "orphan_node_grace_period": 0})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def test_real_NeuroGraphMemory_registers_the_handler_and_a_step_drops_swept_vectors(real_memory):
    ng = real_memory
    handlers = ng.graph._event_handlers.get("nodes_collected", [])
    assert sum(1 for h in handlers if getattr(h, "__func__", None) is type(ng)._drop_collected_vectors) == 1
    rng = np.random.default_rng(4)
    ng.graph.config["orphan_node_grace_period"] = 0
    ids = ["leak_%d" % i for i in range(6)]
    for nid in ids:
        ng.graph.create_node(node_id=nid)
        ng.vector_db.insert(nid, _vec(rng), content=nid)
    ng.vector_db.insert("vector_only_doc", _vec(rng), content="doc")
    ng.graph.timestep += 5
    ng.graph.step()
    assert set(ids).isdisjoint(ng.graph.nodes)
    assert set(ids).isdisjoint(ng.vector_db.all_ids())
    assert "vector_only_doc" in ng.vector_db.all_ids()


def test_handler_survives_an_in_place_graph_restore(real_memory, tmp_path):
    """NeuroGraphMemory never replaces self.graph; Graph.restore() reloads IN PLACE and leaves
    _event_handlers alone, so the one registration made in __init__ still fires afterwards."""
    ng = real_memory
    ckpt = str(tmp_path / "g.msgpack")
    ng.graph.checkpoint(ckpt)
    graph_before = ng.graph
    ng.graph.restore(ckpt)
    assert ng.graph is graph_before
    handlers = ng.graph._event_handlers.get("nodes_collected", [])
    assert sum(1 for h in handlers if getattr(h, "__func__", None) is type(ng)._drop_collected_vectors) == 1
    rng = np.random.default_rng(7)
    ng.graph.config["orphan_node_grace_period"] = 0
    ids = ["post_restore_%d" % i for i in range(3)]
    for nid in ids:
        ng.graph.create_node(node_id=nid)
        ng.vector_db.insert(nid, _vec(rng), content=nid)
    ng.graph.timestep += 5
    ng.graph.step()
    assert set(ids).isdisjoint(ng.graph.nodes)
    assert set(ids).isdisjoint(ng.vector_db.all_ids())


# ---------------------------------------------------------------------------
# 2. Concurrency hammer
# ---------------------------------------------------------------------------

def _hammer(make_db, seconds, stop_on_first_failure=False, with_sweeps=True):
    """Readers + writers on one store. Returns the list of failures observed (strings)."""
    db = make_db()
    rng0 = np.random.default_rng(10)
    base_ids = ["n%d" % i for i in range(1500)]
    for vid in base_ids:
        db.insert(vid, _vec(rng0), content=vid + "|0", metadata={"tag": vid + "|0"})
    failures = []
    fail_lock = threading.Lock()
    stop = threading.Event()
    locked = isinstance(getattr(db, "_lock", None), type(threading.RLock()))

    def fail(msg):
        with fail_lock:
            failures.append(msg)
        if stop_on_first_failure:
            stop.set()

    def guarded(fn):
        def run():
            while not stop.is_set():
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - every exception is a finding
                    fail("%s: %s" % (fn.__name__, type(exc).__name__))
        return run

    def _consistent(vid, content, meta):
        return content.split("|")[0] == vid and meta.get("tag") == content

    q_rng = np.random.default_rng(11)
    queries = [_vec(q_rng) for _ in range(16)]

    def r_search():
        q = queries[random.randrange(len(queries))]
        if locked:
            with db._lock:  # check "exists at the moment returned" atomically with the return
                res = db.search(q, k=10, threshold=-1.0)
                gone = [i for i, _s in res if i not in db.embeddings]
        else:
            res = db.search(q, k=10, threshold=-1.0)
            gone = [i for i, _s in res if db.get(i) is None]
        if gone:
            fail("search returned %d id(s) that no longer exist" % len(gone))
        if res != sorted(res, key=lambda x: x[1], reverse=True):
            fail("search results not sorted")

    def r_get():
        for vid in random.sample(base_ids, 20):
            e = db.get(vid)
            if e is not None and not _consistent(vid, e["content"], e["metadata"]):
                fail("torn get(): %r / %r" % (e["content"], e["metadata"]))

    def r_snapshot():
        for vid, content, meta in db.items_snapshot():
            if not _consistent(vid, content, meta):
                fail("torn items_snapshot row")
                break
        db.count()
        db.all_ids()

    def r_capture():
        cap = db.capture_state(detach=True)
        if cap["count"] != len(cap["entries"]):
            fail("capture count != entries")
        for vid, e in cap["entries"].items():
            if not _consistent(vid, e["content"], e["metadata"]) or len(e["embedding"]) != DIM * 4:
                fail("torn capture entry")
                break

    w_rng = random.Random(12)

    def w_reinsert():
        vid = w_rng.choice(base_ids)
        ver = w_rng.randrange(1, 10 ** 6)
        db.insert(vid, np.ones(DIM, dtype=np.float32) * (1 + ver % 7), content="%s|%d" % (vid, ver),
                  metadata={"tag": "%s|%d" % (vid, ver)})

    def w_delete():
        db.delete(w_rng.choice(base_ids))

    churn = [0]

    def w_churn():
        churn[0] += 1
        vid = "churn_%d" % churn[0]
        db.insert(vid, np.ones(DIM, dtype=np.float32), content=vid + "|0", metadata={"tag": vid + "|0"})
        db.delete("churn_%d" % (churn[0] - 3))

    s_count = [0]

    def w_sweep():
        g = _orphan_graph()
        g.register_event_handler("nodes_collected", _owner_handler(db))
        s_count[0] += 1
        ids = ["sw%d_%d" % (s_count[0], i) for i in range(40)]
        for vid in ids:
            g.create_node(node_id=vid)
            db.insert(vid, np.ones(DIM, dtype=np.float32), content=vid + "|0", metadata={"tag": vid + "|0"})
        g.timestep += 5
        g.step()  # sweep under _step_lock -> handler -> vdb.delete (graph -> vdb)
        left = [vid for vid in ids if db.get(vid) is not None]
        if left:
            fail("sweep left %d vector(s) behind" % len(left))

    workers = [r_search, r_search, r_get, r_snapshot, r_capture, w_reinsert, w_delete, w_churn]
    if with_sweeps:
        workers.append(w_sweep)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        threads = [threading.Thread(target=guarded(fn), daemon=True) for fn in workers]
        for t in threads:
            t.start()
        stop.wait(seconds)
        stop.set()
        for t in threads:
            t.join(30)
        assert not any(t.is_alive() for t in threads), "hammer thread hung"
    finally:
        sys.setswitchinterval(old)
    return failures


def test_hammer_locked_store_has_zero_failures():
    failures = _hammer(SimpleVectorDB, seconds=4.0)
    assert failures == [], failures[:10]


def test_CONTROL_hammer_catches_the_no_op_lock_store():
    failures = _hammer(_UnlockedSimpleVectorDB, seconds=20.0, stop_on_first_failure=True)
    assert failures, "the hammer failed to catch an unlocked store -- it proves nothing"


def test_CONTROL_hammer_catches_the_pre_lane_class():
    failures = _hammer(_PreLaneSimpleVectorDB, seconds=20.0, stop_on_first_failure=True)
    assert failures, "the hammer failed to catch the original unlocked class -- it proves nothing"


# ---------------------------------------------------------------------------
# 3. Deadlock guard (the vdb lock is a LEAF)
# ---------------------------------------------------------------------------

def _all_store_ops(db, path):
    q = np.ones(DIM, dtype=np.float32)
    db.insert("x", q, content="x")
    db.search(q, k=3, threshold=-1.0)
    db.get("x")
    db.get_embedding("x")
    db.get_content("x")
    db.get_metadata("x")
    db.items_snapshot()
    db.count()
    db.all_ids()
    db.save(path)
    db.load(path)
    db.write_state(path, db.capture_state())
    db.delete("x")
    pickle.loads(pickle.dumps(db))


def _run_with_timeout(fn, timeout):
    err = []

    def body():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            err.append(exc)
    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout)
    return (not t.is_alive()), err


def test_every_store_op_completes_while_another_thread_holds_the_graph_step_lock(tmp_path):
    g, db = _orphan_graph(), SimpleVectorDB()
    held, release = threading.Event(), threading.Event()

    def holder():
        with g._step_lock:
            held.set()
            release.wait(30)
    h = threading.Thread(target=holder, daemon=True)
    h.start()
    assert held.wait(10)
    try:
        done, err = _run_with_timeout(lambda: _all_store_ops(db, str(tmp_path / "v.msgpack")), 20)
        assert done, "a vdb method blocked on the graph step lock: the vdb lock is not a leaf"
        assert not err, err
    finally:
        release.set()
        h.join(10)


def test_step_whose_sweep_deletes_vectors_completes_once_a_vdb_lock_holder_releases():
    rng = np.random.default_rng(5)
    g, db = _orphan_graph(), SimpleVectorDB()
    orphans, _w, _v = _seed(g, db, rng, n_orphan=3, n_wired=2, n_vector_only=0)
    g.register_event_handler("nodes_collected", _owner_handler(db))
    held, release = threading.Event(), threading.Event()

    def holder():  # hold the vdb lock; this thread never touches the graph
        with db._lock:
            held.set()
            release.wait(30)
    h = threading.Thread(target=holder, daemon=True)
    h.start()
    assert held.wait(10)
    stepper_done = threading.Event()
    s = threading.Thread(target=lambda: (g.step(), stepper_done.set()), daemon=True)
    s.start()
    time.sleep(0.3)
    assert not stepper_done.is_set()          # the sweep is waiting on the vdb lock (graph -> vdb)
    # ... and the graph lock it holds does not stop any OTHER vdb user: the order is one-way.
    release.set()
    assert stepper_done.wait(20), "step() never completed after the vdb lock was released"
    h.join(10)
    assert set(orphans).isdisjoint(db.all_ids())


def test_CONTROL_the_guard_catches_a_store_that_takes_the_graph_lock_under_its_own(tmp_path):
    g = _orphan_graph()

    class _BadStore(SimpleVectorDB):  # violates the leaf contract: vdb -> graph
        def count(self):
            with self._lock:
                if not g._step_lock.acquire(timeout=3):
                    raise TimeoutError("would deadlock")
                g._step_lock.release()
                return len(self.embeddings)

    db = _BadStore()
    held, release = threading.Event(), threading.Event()

    def holder():
        with g._step_lock:
            held.set()
            release.wait(30)
    h = threading.Thread(target=holder, daemon=True)
    h.start()
    assert held.wait(10)
    try:
        done, err = _run_with_timeout(lambda: _all_store_ops(db, str(tmp_path / "v.msgpack")), 20)
        assert (not done) or err, "the deadlock guard did not notice a vdb->graph lock order"
    finally:
        release.set()
        h.join(10)


# ---------------------------------------------------------------------------
# 4. On-disk format and pickling
# ---------------------------------------------------------------------------

def _fill(db):
    rng = np.random.default_rng(6)
    for i in range(50):
        db.insert("id_%d" % i, _vec(rng), content="text %d" % i,
                  metadata={"creation_mode": "conversational", "n": i, "nested": {"k": [i, i + 1]}})


@pytest.mark.parametrize("ext", [".msgpack", ".json"])
def test_written_bytes_identical_to_the_pre_lane_class_and_stable_across_save_load_save(tmp_path, ext):
    if ext == ".msgpack":
        pytest.importorskip("msgpack")
    new, old = SimpleVectorDB(), _PreLaneSimpleVectorDB()
    _fill(new)
    _fill(old)
    p_new, p_old, p_round = (str(tmp_path / (n + ext)) for n in ("new", "old", "round"))
    new.save(p_new)
    old.write_state(p_old, old.capture_state())
    with open(p_new, "rb") as a, open(p_old, "rb") as b:
        assert a.read() == b.read()
    # restore round-trip: load + save through this class == load + save through the pre-lane class
    # (load() re-normalizes, so the round-trip is compared against the base, not against p_new)
    again, again_old = SimpleVectorDB(), _PreLaneSimpleVectorDB()
    assert again.load(p_new) == 50 and again_old.load(p_old) == 50
    p_round_old = str(tmp_path / ("round_old" + ext))
    again.save(p_round)
    again_old.write_state(p_round_old, again_old.capture_state())
    with open(p_round, "rb") as a, open(p_round_old, "rb") as b:
        assert a.read() == b.read()


def test_pickle_and_deepcopy_drop_and_recreate_the_lock():
    db = SimpleVectorDB()
    _fill(db)
    state = db.__getstate__()
    assert "_lock" not in state
    for clone in (pickle.loads(pickle.dumps(db)), copy.deepcopy(db)):
        assert clone._lock is not db._lock
        assert isinstance(clone._lock, type(threading.RLock()))
        assert clone.all_ids() == db.all_ids()
        with clone._lock:                          # usable, and re-entrant
            assert clone.count() == 50


def test_search_drops_ids_deleted_and_rescores_ids_reinserted_during_scoring():
    db = SimpleVectorDB()
    for i in range(5):
        v = np.zeros(DIM, dtype=np.float32)
        v[0], v[1] = 1.0, 0.1 * i
        db.insert("s%d" % i, v)
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    real_dot = np.dot
    fired = []

    def dot_hook(a, b):  # mutate the store mid-scoring (after the snapshot, before re-validation)
        if not fired:
            fired.append(1)
            db.delete("s0")
            w = np.zeros(DIM, dtype=np.float32)
            w[2] = 1.0
            db.insert("s1", w)  # now orthogonal to q -> sim 0
        return real_dot(a, b)

    np_mod = sys.modules["universal_ingestor"].np
    orig = np_mod.dot
    np_mod.dot = dot_hook
    try:
        res = db.search(q, k=5, threshold=0.5)
    finally:
        np_mod.dot = orig
    ids = [i for i, _ in res]
    assert "s0" not in ids                       # deleted meanwhile -> dropped
    assert "s1" not in ids                       # re-inserted below threshold -> dropped on re-score
    assert ids == ["s2", "s3", "s4"]             # sims fall with i; the survivors, best first
