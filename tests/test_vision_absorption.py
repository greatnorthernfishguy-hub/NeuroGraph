"""Tests for vision_absorption (#82 Inc 1) — against a fresh Graph(), no _memory, no vdb.

# ---- Changelog ----
# [2026-09-06] DudeMan CC (Fable 5.1) — Created with the module.
#   What: forest+trees -> nodes/synapses/hyperedge; wrong-dim hard reject; prev-frame
#         delayed link; BTF OUTCOME round-trip via ng_tract; dispatcher split keeps
#         non-vision OUTCOME entries excluded (unchanged behaviour); no vdb ever touched.
#   Why:  The mother half of vision must be provable without the phone.
# -------------------
"""
import time
import unittest

import numpy as np

from neuro_foundation import Graph
import vision_absorption as va


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _frame(frame_id: str, n_trees: int = 3, seed: int = 0, dim: int = 768, image_ref=None):
    out = [{"module_id": va.VISION_PREFIX, "target_id": f"{va.VISION_PREFIX}::{frame_id}::forest",
            "embedding": _emb(seed, dim), "timestamp": time.time(),
            "metadata": {"kind": "forest", "frame_id": frame_id, "n_trees": n_trees,
                         "image_sha": "ab" * 32, **({"image_ref": image_ref} if image_ref else {})}}]
    for k in range(n_trees):
        out.append({"module_id": va.VISION_PREFIX, "target_id": f"{va.VISION_PREFIX}::{frame_id}::tree::{k}",
                    "embedding": _emb(seed * 100 + k + 1, dim), "timestamp": time.time(),
                    "metadata": {"kind": "tree", "frame_id": frame_id, "tree_index": k}})
    return out


class TestVisionAbsorption(unittest.TestCase):
    def setUp(self):
        va._last_frame_forest_id = None

    def test_forest_and_trees_become_topology(self):
        g = Graph()
        res = va.absorb_entries(g, _frame("f1", n_trees=3, image_ref="/tmp/x.jpg"))
        self.assertEqual(len(res), 1)
        r = res[0]
        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(len(r["tree_ids"]), 3)
        forest = g.nodes[r["forest_id"]]
        self.assertEqual(forest.metadata["modality"], "vision")
        self.assertEqual(forest.metadata["creation_mode"], "sensory")
        self.assertEqual(forest.metadata["_image_ref"], "/tmp/x.jpg")
        self.assertIn("poincare_dir", forest.metadata)
        # forest<->tree both directions = 6 synapses
        self.assertEqual(len(g.synapses), 6)
        # one hyperedge with all four members
        self.assertEqual(len(g.hyperedges), 1)
        he = next(iter(g.hyperedges.values()))
        self.assertEqual(set(he.member_nodes), {r["forest_id"], *r["tree_ids"]})
        self.assertEqual(he.metadata.get("modality"), "vision")
        # prompt binding: forest nudged, trees not
        self.assertGreater(forest.voltage, 0.0)
        self.assertEqual(g.nodes[r["tree_ids"][0]].voltage, 0.0)

    def test_wrong_dimension_is_a_hard_reject(self):
        g = Graph()
        res = va.absorb_entries(g, _frame("bad", n_trees=1, dim=512))
        self.assertEqual(res, [])
        self.assertEqual(len(g.nodes), 0)

    def test_trees_without_forest_are_skipped(self):
        g = Graph()
        entries = [e for e in _frame("orphan", n_trees=2) if "::tree::" in e["target_id"]]
        self.assertEqual(va.absorb_entries(g, entries), [])
        self.assertEqual(len(g.nodes), 0)

    def test_second_frame_gets_delayed_link_from_previous(self):
        g = Graph()
        r1 = va.absorb_entries(g, _frame("f1", n_trees=1, seed=1))[0]
        r2 = va.absorb_entries(g, _frame("f2", n_trees=1, seed=2))[0]
        links = [s for s in g.synapses.values()
                 if s.pre_node_id == r1["forest_id"] and s.post_node_id == r2["forest_id"]]
        self.assertEqual(len(links), 1)
        self.assertGreaterEqual(links[0].delay, 2)

    def test_btf_outcome_roundtrip_and_dispatcher_split(self):
        import msgpack
        import ng_tract
        payload = b""
        for e in _frame("btf", n_trees=2, seed=7):
            payload += bytes(ng_tract.write_outcome(
                timestamp=e["timestamp"], module_id=e["module_id"], target_id=e["target_id"],
                success=True, embedding=e["embedding"], metadata=msgpack.packb(e["metadata"])))
        # a non-vision OUTCOME entry that must stay excluded, exactly as today
        payload += bytes(ng_tract.write_outcome(timestamp=time.time(), module_id="elmer",
                                                target_id="x", success=True, embedding=_emb(9)))
        entries = list(ng_tract.TractReader(payload))
        vis, rest = va.split_vision_entries(entries)
        self.assertEqual(len(vis), 3)
        self.assertEqual(len(rest), 1)
        g = Graph()
        res = va.absorb_entries(g, vis)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(g.nodes[res[0]["forest_id"]].metadata["frame_id"], "btf")

    def test_never_touches_a_vector_db(self):
        # The function has no vdb parameter and imports no ingestor: prove by source.
        import inspect
        # Strip comments/docstrings: only CODE lines may not reference these.
        code = "\n".join(l for l in inspect.getsource(va).splitlines()
                         if l.strip() and not l.strip().startswith(("#", '"', "'", "*")))
        for forbidden in ("vector_db", "universal_ingestor", "import ng_embed", "from ng_embed", "SimpleVectorDB"):
            self.assertNotIn(forbidden, code)

    def test_store_image_body_dedupes(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            va._bodies_dir = lambda: pathlib.Path(d)  # type: ignore
            try:
                p1 = va.store_image_body(b"\xff\xd8jpegbytes")
                p2 = va.store_image_body(b"\xff\xd8jpegbytes")
                self.assertIsNotNone(p1)
                self.assertEqual(p1, p2)
                self.assertTrue(str(p1).endswith(".jpg"))
            finally:
                del va._bodies_dir


if __name__ == "__main__":
    unittest.main()
