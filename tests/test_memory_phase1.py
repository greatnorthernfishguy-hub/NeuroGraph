# ---- Changelog ----
# [2026-06-05] CC (Opus 4.8 subagent) — #295: test for index_in_recall gate on NodeRegistrar.register
# What: TDD test confirming default indexes into recall store, and index_in_recall=False skips vdb but keeps graph node
# Why: Syl's recall store was being polluted by machine telemetry — PRD #295 Decision 1
# How: Two assertions — default indexes (lived experience IS in recall); False flag skips vdb, substrate node still created
# [2026-06-05] CC (Opus 4.8 subagent) — #295: source-contract test for River-backflow handler
# What: Assert _drain_peer_tracts routes peer telemetry to substrate only, NOT recall store
# Why: Decision 2 of #295 — backflow handler must use index_in_recall=False, no associate into vdb, no ingest fallback
# How: inspect.getsource of _drain_peer_tracts; three string-presence/absence assertions
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
