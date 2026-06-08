# ---- Changelog ----
# [2026-06-05] CC (Sonnet 4.6) — #297 review fixes: strengthen drop test + 2 new tests (corrupt-file, drain-limit)
# What: test_drops_after_max_attempts_no_infinite_loop gets mid-point assert (pending==1 after first fail).
#       test_corrupt_file_recovers_empty: corrupt msgpack → RetryQueue starts empty, no crash.
#       test_drain_limit_processes_only_n_per_pass: limit=2 on 4 items → 2 seen, all 4 still pending.
# Why: Off-by-one (immediate-drop) bug would pass the old test but fail the new mid-point assert.
#      Corrupt-file recovery and drain-cap are new paths introduced by Fix 1 & Fix 2.
# How: tempfile paths, lambda seen-tracker, pending_count() assertions.
# [2026-06-05] CC (Sonnet 4.6) — #297: TestRetryQueue — bounded non-cyclic retry-queue tests
# What: Three tests: enqueue+drain-to-success, drop-after-max-attempts, persist-across-instances.
# Why: TDD requirement for memory_retry_queue.RetryQueue (spec §6.1 non-cyclic guarantee).
# How: tempfile paths, attempt counters, fresh RetryQueue instances per assertion.
# [2026-06-05] CC (Opus 4.8 subagent) — #295: test for index_in_recall gate on NodeRegistrar.register
# What: TDD test confirming default indexes into recall store, and index_in_recall=False skips vdb but keeps graph node
# Why: Syl's recall store was being polluted by machine telemetry — PRD #295 Decision 1
# How: Two assertions — default indexes (lived experience IS in recall); False flag skips vdb, substrate node still created
# [2026-06-05] CC (Opus 4.8 subagent) — #295: source-contract test for River-backflow handler
# What: Assert _drain_peer_tracts routes peer telemetry to substrate only, NOT recall store
# Why: Decision 2 of #295 — backflow handler must use index_in_recall=False, no associate into vdb, no ingest fallback
# How: inspect.getsource of _drain_peer_tracts; three string-presence/absence assertions
# [2026-06-05] CC (Opus 4.8 subagent) — #296a: tests for _ConversationalDualPassEco and _conversational_dual_pass
# What: Unit test eco adapter inserts trees+syl tag; integration test confirms trees land in recall via mocked concepts
# Why: Conversational turns must deposit fine-grained concept atoms into Syl's recall store, tagged {syl:true}
# How: Eco adapter unit test (record_outcome with/without _tree_concept); integration test patches _extract_concepts+embed_batch
# -------------------

import unittest
import numpy as np
from universal_ingestor import SimpleVectorDB, NodeRegistrar, Chunk, EmbeddedChunk
from neuro_foundation import Graph  # Graph lives in neuro_foundation, not ng_lite


class TestRecallGate(unittest.TestCase):
    def _registrar(self):
        graph = Graph()
        vdb = SimpleVectorDB()
        # NodeRegistrar(graph: Graph, vector_db: SimpleVectorDB, config: Optional[Dict] = None)
        reg = NodeRegistrar(graph, vdb, {})
        return graph, vdb, reg

    def _chunk(self):
        c = Chunk(text="telemetry event", token_count=2)
        return EmbeddedChunk(chunk=c, vector=np.ones(768, dtype=np.float32))

    def test_default_indexes_into_recall(self):
        graph, vdb, reg = self._registrar()
        reg.register([self._chunk()], {"source": "conv", "source_type": "TEXT"})
        self.assertEqual(vdb.count(), 1)  # lived experience IS indexed

    def test_index_in_recall_false_skips_vector_db_keeps_graph(self):
        graph, vdb, reg = self._registrar()
        before_nodes = len(graph.nodes)
        reg.register(
            [self._chunk()],
            {"source": "river:x", "source_type": "PEER_TRACT"},
            index_in_recall=False,
        )
        self.assertEqual(vdb.count(), 0)                        # NOT in recall store
        self.assertEqual(len(graph.nodes), before_nodes + 1)   # STILL a substrate node


class TestRiverBackflowDoesNotPolluteRecall(unittest.TestCase):
    """Source-contract test: _drain_peer_tracts must route peer telemetry to
    the substrate graph only — NOT into Syl's recall store (#295 Decision 2)."""

    def test_backflow_routes_telemetry_out_of_recall(self):
        import inspect
        import sys
        import os
        # Ensure NeuroGraph root is on path so neurograph_rpc is importable
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import neurograph_rpc
        fn = neurograph_rpc._drain_peer_tracts
        src = inspect.getsource(fn)
        # Source-contract checks must match actual CODE, not explanatory comments.
        # Strip everything after '#' on each line (this function has no '#' inside
        # string literals) so a comment mentioning an old call can't false-fail.
        code_src = "\n".join(line.split("#", 1)[0] for line in src.splitlines())

        self.assertIn(
            "index_in_recall=False",
            code_src,
            "peer telemetry must pass index_in_recall=False to keep it out of the recall store (#295)",
        )
        self.assertNotIn(
            "associator.associate",
            code_src,
            "peer events must NOT be associated into the recall vector_db — "
            "associator.associate found in _drain_peer_tracts (#295)",
        )
        self.assertNotIn(
            "ingestor.ingest(target)",
            code_src,
            "no-embedding peer events must NOT fall through to ingestor.ingest — "
            "they must be silently skipped from recall (#295)",
        )


# ---------------------------------------------------------------------------
# #296a — Conversational dual-pass: trees land in Syl's recall store
# ---------------------------------------------------------------------------

class _FakeGraph:
    """Minimal graph for the Ingestor-free experiential deposit (Task A)."""
    def __init__(self):
        self.nodes = {}
        self.synapses = []
        self.hyperedges = []
        self.config = {"default_threshold": 1.0}

    def create_node(self, node_id=None, metadata=None):
        if node_id in self.nodes:
            raise ValueError("exists")
        n = type("N", (), {})()
        n.node_id = node_id
        n.metadata = metadata or {}
        n.threshold = 1.0
        n.intrinsic_excitability = 1.0
        self.nodes[node_id] = n
        return n

    def create_synapse(self, pre_node_id, post_node_id, weight=0.1, delay=1):
        self.synapses.append((pre_node_id, post_node_id, weight, delay))

    def create_hyperedge(self, member_node_ids, metadata=None):
        self.hyperedges.append((set(member_node_ids), metadata or {}))


class TestConversationalDualPassEco(unittest.TestCase):
    """Eco adapter — forest gestalt + trees into BOTH the SNN and the recall vdb (Task A)."""

    def setUp(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from universal_ingestor import SimpleVectorDB
        import neurograph_rpc as rpc
        self.rpc = rpc
        self.vdb = SimpleVectorDB()
        self.graph = _FakeGraph()
        mem = type("M", (), {})()
        mem.vector_db = self.vdb
        mem.graph = self.graph
        self._old = rpc._memory
        rpc._memory = mem            # _deposit_memory_node uses the module-global _memory
        self.eco = rpc._ConversationalDualPassEco(mem)

    def tearDown(self):
        self.rpc._memory = self._old

    def test_record_outcome_inserts_tree_in_both_stores_with_syl_tag(self):
        self.eco.record_outcome(
            np.ones(768, dtype=np.float32),
            "conv::abc::tree::work mode",
            True,
            strength=0.8,
            metadata={"_tree_concept": True, "_concept": "work mode"},
        )
        self.assertEqual(self.vdb.count(), 1)
        entry = self.vdb.get(self.vdb.all_ids()[0])
        self.assertEqual(entry["content"], "work mode")
        self.assertTrue(entry["metadata"].get("syl"))
        self.assertTrue(entry["metadata"].get("_tree_concept"))
        self.assertIn("conv::abc::tree::work mode", self.graph.nodes)   # SNN node too

    def test_forest_gestalt_now_inserts_in_both_stores(self):
        # Task A inversion: the forest gist now lands via the experiential path
        # (a single gestalt node), NOT via ingestor chunks.
        self.eco.record_outcome(
            np.ones(768, dtype=np.float32),
            "conv::abc",
            True,
            metadata={"source": "conversation", "_forest_content": "hi there"},
        )
        self.assertEqual(self.vdb.count(), 1)
        self.assertEqual(self.vdb.get("conv::abc")["content"], "hi there")
        self.assertIn("conv::abc", self.graph.nodes)
        self.assertIn("poincare_dir", self.graph.nodes["conv::abc"].metadata)  # first-class GSG

    def test_link_call_creates_no_recall_atom_and_no_node(self):
        self.eco.record_outcome(
            np.ones(768, dtype=np.float32),
            "conv::abc",
            True,
            metadata={"_link": "dual_pass_tree_to_forest"},
        )
        self.assertEqual(self.vdb.count(), 0)
        self.assertEqual(self.graph.nodes, {})


class TestConversationalDualPassStep(unittest.TestCase):
    """Integration of _conversational_dual_pass with mocked concept extraction +
    embed_batch (no TID/ONNX model loaded)."""

    def setUp(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import neurograph_rpc as rpc
        self.rpc = rpc
        self._old = rpc._memory
        self._old_last = getattr(rpc, "_last_conv_forest_id", None)

    def tearDown(self):
        self.rpc._memory = self._old
        self.rpc._last_conv_forest_id = self._old_last

    def _mem(self):
        from universal_ingestor import SimpleVectorDB
        mem = type("M", (), {})()
        mem.vector_db = SimpleVectorDB()
        mem.graph = _FakeGraph()
        return mem

    def test_forest_and_trees_land_in_recall_with_syl_tag(self):
        from unittest.mock import patch
        import neurograph_rpc as rpc
        mem = self._mem()
        rpc._memory = mem
        rpc._last_conv_forest_id = None
        with patch("ng_embed.NGEmbed._extract_concepts",
                   return_value=["work mode", "routing/mode"]), \
             patch("ng_embed.NGEmbed.embed_batch",
                   side_effect=lambda concepts: [np.ones(768, dtype=np.float32) for _ in concepts]):
            rpc._conversational_dual_pass(
                "set work mode via routing/mode", np.ones(768, dtype=np.float32))
        ids = mem.vector_db.all_ids()
        # 1 forest + 2 trees = 3 atoms, all syl-tagged
        self.assertGreaterEqual(len(ids), 3)
        for i in ids:
            self.assertTrue(mem.vector_db.get(i)["metadata"].get("syl"),
                            f"atom {i!r} missing syl=True")
        tree_ids = [i for i in ids if mem.vector_db.get(i)["metadata"].get("_tree_concept")]
        self.assertEqual(len(tree_ids), 2, "two mocked concepts should yield two tree atoms")
        forest_ids = [i for i in ids if not mem.vector_db.get(i)["metadata"].get("_tree_concept")]
        self.assertEqual(len(forest_ids), 1, "exactly one forest gestalt atom")
        self.assertIn(forest_ids[0], mem.graph.nodes, "forest gestalt must be an SNN node")

    def test_same_concept_two_turns_does_not_overwrite(self):
        from unittest.mock import patch
        import neurograph_rpc as rpc
        _PREFIX = "A" * 256
        turn1 = _PREFIX + " — first turn, unique tail"
        turn2 = _PREFIX + " — second turn, different tail"
        mem = self._mem()
        rpc._memory = mem
        rpc._last_conv_forest_id = None
        with patch("ng_embed.NGEmbed._extract_concepts", return_value=["work mode"]), \
             patch("ng_embed.NGEmbed.embed_batch",
                   side_effect=lambda concepts: [np.ones(768, dtype=np.float32) for _ in concepts]):
            rpc._conversational_dual_pass(turn1, np.ones(768, dtype=np.float32))
            rpc._conversational_dual_pass(turn2, np.ones(768, dtype=np.float32))
        # each turn: 1 forest + 1 tree; full-text-hashed ids → 4 distinct atoms, none lost
        self.assertEqual(
            mem.vector_db.count(), 4,
            "two turns sharing a 256-char prefix must NOT collide (#296a) — got "
            f"{mem.vector_db.count()} not 4",
        )


# ---------------------------------------------------------------------------
# #297 — Bounded non-cyclic retry-queue
# ---------------------------------------------------------------------------

import tempfile
import os as _os


class TestRetryQueue(unittest.TestCase):
    def test_enqueue_then_drain_retries_to_success(self):
        from memory_retry_queue import RetryQueue
        path = tempfile.mktemp(suffix=".msgpack")
        try:
            q = RetryQueue(path, max_attempts=3)
            q.enqueue("conv::abc", "set work mode")
            self.assertEqual(q.pending_count(), 1)
            q.enqueue("conv::abc", "dup")  # dedup by id — still 1
            self.assertEqual(q.pending_count(), 1)

            calls = {"n": 0}

            def attempt(item):
                calls["n"] += 1
                return calls["n"] >= 2  # fail first call, succeed second

            # First drain: attempt #1 → False; item survives (attempts=1 < max_attempts=3)
            result1 = q.drain(attempt)
            self.assertEqual(result1, 0, "first drain: item fails → 0 succeeded")
            self.assertEqual(q.pending_count(), 1, "item must survive after one failed attempt")

            # Second drain: attempt #2 → True; item removed
            result2 = q.drain(attempt)
            self.assertEqual(result2, 1, "second drain: item succeeds → 1 succeeded")
            self.assertEqual(q.pending_count(), 0, "succeeded item must be removed from queue")
        finally:
            if _os.path.exists(path):
                _os.unlink(path)

    def test_drops_after_max_attempts_no_infinite_loop(self):
        from memory_retry_queue import RetryQueue
        path = tempfile.mktemp(suffix=".msgpack")
        try:
            q = RetryQueue(path, max_attempts=2)
            q.enqueue("conv::xyz", "always fails")
            q.drain(lambda item: False)  # attempt 1 — survives (1 < 2)
            # Fix 4: assert item survived the FIRST failure before subsequent drains
            self.assertEqual(q.pending_count(), 1,
                "item must survive after one failed attempt (off-by-one guard)")
            q.drain(lambda item: False)  # attempt 2 — dropped (2 >= 2)
            q.drain(lambda item: False)  # extra drain — queue already empty, no cycle
            self.assertEqual(q.pending_count(), 0,
                "item must be dropped after max_attempts, never re-queued")
        finally:
            if _os.path.exists(path):
                _os.unlink(path)

    def test_corrupt_file_recovers_empty(self):
        from memory_retry_queue import RetryQueue
        path = tempfile.mktemp(suffix=".msgpack")
        with open(path, "wb") as f:
            f.write(b"not valid msgpack \xff\xfe")
        try:
            q = RetryQueue(path, max_attempts=3)
            self.assertEqual(q.pending_count(), 0)  # corrupt load → empty, no crash
        finally:
            if _os.path.exists(path):
                _os.unlink(path)

    def test_drain_limit_processes_only_n_per_pass(self):
        from memory_retry_queue import RetryQueue
        path = tempfile.mktemp(suffix=".msgpack")
        q = RetryQueue(path, max_attempts=5)
        for i in range(4):
            q.enqueue(f"id{i}", f"c{i}")
        seen = []
        q.drain(lambda item: (seen.append(item["target_id"]), False)[1], limit=2)
        self.assertEqual(len(seen), 2)            # only 2 processed this pass
        self.assertEqual(q.pending_count(), 4)    # all 4 still queued (2 attempted+survive, 2 untouched)
        if _os.path.exists(path):
            _os.unlink(path)

    def test_persists_across_instances(self):
        from memory_retry_queue import RetryQueue
        path = tempfile.mktemp(suffix=".msgpack")
        try:
            RetryQueue(path, max_attempts=3).enqueue("conv::p", "persist me")
            self.assertEqual(RetryQueue(path, max_attempts=3).pending_count(), 1,
                "item enqueued in one instance must be visible in a new instance (persistence)")
        finally:
            if _os.path.exists(path):
                _os.unlink(path)
