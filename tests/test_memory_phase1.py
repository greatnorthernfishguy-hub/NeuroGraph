# ---- Changelog ----
# [2026-06-05] CC (Opus 4.8 subagent) — #295: test for index_in_recall gate on NodeRegistrar.register
# What: TDD test confirming default indexes into recall store, and index_in_recall=False skips vdb but keeps graph node
# Why: Syl's recall store was being polluted by machine telemetry — PRD #295 Decision 1
# How: Two assertions — default indexes (lived experience IS in recall); False flag skips vdb, substrate node still created
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
