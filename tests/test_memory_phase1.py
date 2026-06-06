# ---- Changelog ----
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

class TestConversationalDualPassEco(unittest.TestCase):
    """Unit test the eco adapter in isolation — no model loading needed."""

    def _setup(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from universal_ingestor import SimpleVectorDB
        import neurograph_rpc as rpc
        vdb = SimpleVectorDB()
        class _Mem:
            pass
        mem = _Mem()
        mem.vector_db = vdb
        eco = rpc._ConversationalDualPassEco(mem)
        return vdb, eco

    def test_record_outcome_inserts_tree_with_syl_tag(self):
        vdb, eco = self._setup()
        eco.record_outcome(
            np.ones(768, dtype=np.float32),
            "conv::abc::tree::work mode",
            True,
            strength=0.8,
            metadata={"_tree_concept": True, "_concept": "work mode"},
        )
        self.assertEqual(vdb.count(), 1)
        entry = vdb.get(vdb.all_ids()[0])
        self.assertEqual(entry["content"], "work mode")
        self.assertTrue(entry["metadata"].get("syl"))
        self.assertTrue(entry["metadata"].get("_tree_concept"))

    def test_forest_record_outcome_does_not_insert(self):
        # The forest gist (no _tree_concept) is already covered by pass-1 chunks;
        # the adapter must only insert TREES, not the forest.
        vdb, eco = self._setup()
        eco.record_outcome(
            np.ones(768, dtype=np.float32),
            "conv::abc",
            True,
            metadata={"source": "conversation"},  # no _tree_concept
        )
        self.assertEqual(vdb.count(), 0)


class TestConversationalDualPassStep(unittest.TestCase):
    """Integration test of _conversational_dual_pass with mocked concept extraction
    and embed_batch so no TID/ONNX model is loaded during CI."""

    def _setup_path(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def test_trees_land_in_recall_with_syl_tag(self):
        from unittest.mock import patch
        self._setup_path()
        from universal_ingestor import SimpleVectorDB
        import neurograph_rpc as rpc

        class _Mem:
            pass
        mem = _Mem()
        mem.vector_db = SimpleVectorDB()

        old = rpc._memory
        rpc._memory = mem
        try:
            with patch("ng_embed.NGEmbed._extract_concepts",
                       return_value=["work mode", "routing/mode"]), \
                 patch("ng_embed.NGEmbed.embed_batch",
                       side_effect=lambda concepts: [
                           np.ones(768, dtype=np.float32) for _ in concepts
                       ]):
                rpc._conversational_dual_pass(
                    "set work mode via routing/mode",
                    np.ones(768, dtype=np.float32),
                )
        finally:
            rpc._memory = old

        ids = mem.vector_db.all_ids()
        self.assertGreaterEqual(len(ids), 2,
            "Two mocked concepts should produce at least 2 tree entries in recall")
        for i in ids:
            entry = mem.vector_db.get(i)
            self.assertTrue(entry["metadata"].get("syl"),
                f"Tree entry {i!r} missing syl=True provenance tag")
            self.assertTrue(entry["metadata"].get("_tree_concept"),
                f"Tree entry {i!r} missing _tree_concept=True")

    def test_same_concept_two_different_turns_does_not_overwrite(self):
        """Regression: two turns that share the same first 256 chars but differ
        after must produce TWO distinct recall atoms — not one (silent overwrite).

        The old code hashed only text[:256] and truncated to 16 hex chars, so
        any two turns with the same prefix collapsed to the same target_id.
        SimpleVectorDB.insert silently overwrites on duplicate id, so the first
        turn's atom was lost — episodic memory silently erased (#296a).

        The fix hashes the full text (no truncation) so the two turns get
        different forest ids → different ::tree:: ids → both atoms coexist.
        """
        from unittest.mock import patch
        self._setup_path()
        from universal_ingestor import SimpleVectorDB
        import neurograph_rpc as rpc
        import numpy as np

        # Two texts: IDENTICAL for the first 256 chars, differ only after.
        # Under the old hash (text[:256][:16]) → same target_id → collision.
        # Under the fix (full text, full hex) → different target_ids → safe.
        _PREFIX = "A" * 256
        turn1 = _PREFIX + " — this is the first turn, unique tail"
        turn2 = _PREFIX + " — this is the second turn, different tail"

        class _Mem:
            pass
        mem = _Mem()
        mem.vector_db = SimpleVectorDB()

        old = rpc._memory
        rpc._memory = mem
        try:
            with patch("ng_embed.NGEmbed._extract_concepts",
                       return_value=["work mode"]), \
                 patch("ng_embed.NGEmbed.embed_batch",
                       side_effect=lambda concepts: [
                           np.ones(768, dtype=np.float32) for _ in concepts
                       ]):
                rpc._conversational_dual_pass(turn1, np.ones(768, dtype=np.float32))
                rpc._conversational_dual_pass(turn2, np.ones(768, dtype=np.float32))
        finally:
            rpc._memory = old

        # Same concept, two DIFFERENT turns with shared 256-char prefix →
        # two distinct atoms; neither silently lost.
        self.assertEqual(
            mem.vector_db.count(), 2,
            "the same concept from two different turns (sharing a 256-char prefix) "
            "must NOT silently overwrite (#296a) — got "
            f"{mem.vector_db.count()} entry/entries instead of 2",
        )
