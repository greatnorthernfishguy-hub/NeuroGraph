"""#82 Inc 2 / #410 — images surface AS images: resolver, CES L2 monitor, /assemble injection.

# ---- Changelog ----
# [2026-09-06] DudeMan CC (Fable 5.1) — Created with the change.
#   What: resolve_surface_item() returns an image item for a vision forest (and None when the
#         file is gone); text nodes still resolve substrate-first; CES L2 surfaces a vision node
#         instead of skipping it and prints the attached-image marker without describing it;
#         _vision_surface_messages() builds a proper image_url data-URL user message, honours
#         the count cap and the byte cap, and never emits a phantom block.
#   Why:  Without this half, #82 ends where #294 ended — storage, not sight.
# -------------------
"""
import base64
import importlib
import os
import tempfile
import unittest

from neuro_foundation import Graph
from ces_config import load_ces_config
from surface_resolver import resolve_surface_content, resolve_surface_item
from surfacing import SurfacingMonitor
import vision_absorption as va

_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


class _StubVDB:
    def __init__(self, entries=None):
        self._e = entries or {}

    def get(self, node_id):
        return self._e.get(node_id)


class _Step:
    def __init__(self, fired):
        self.fired_node_ids = list(fired)


def _vision_frame(g, image_ref, frame_id="f1"):
    e = [{"module_id": va.VISION_PREFIX, "target_id": f"{va.VISION_PREFIX}::{frame_id}::forest",
          "embedding": [0.1] * 768, "metadata": {"kind": "forest", "frame_id": frame_id, "image_ref": image_ref}},
         {"module_id": va.VISION_PREFIX, "target_id": f"{va.VISION_PREFIX}::{frame_id}::tree::0",
          "embedding": [0.2] * 768, "metadata": {"kind": "tree", "frame_id": frame_id, "tree_index": 0}}]
    return va.absorb_entries(g, e, nudge=0.0)[0]


class TestResolver(unittest.TestCase):
    def test_vision_forest_resolves_to_image_item(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_JPEG); path = f.name
        try:
            g = Graph(); r = _vision_frame(g, path)
            item = resolve_surface_item(g.nodes[r["forest_id"]], None)
            self.assertEqual(item, {"kind": "image", "image_ref": path})
            # trees carry no picture of their own
            self.assertIsNone(resolve_surface_item(g.nodes[r["tree_ids"][0]], None))
            # the text resolver is unchanged: a vision node has no text
            self.assertIsNone(resolve_surface_content(g.nodes[r["forest_id"]], None))
        finally:
            os.unlink(path)

    def test_missing_file_means_nothing_to_show(self):
        g = Graph(); r = _vision_frame(g, "/nonexistent/frame.jpg")
        self.assertIsNone(resolve_surface_item(g.nodes[r["forest_id"]], None))

    def test_text_nodes_still_substrate_first(self):
        g = Graph()
        n = g.create_node(node_id="t1", metadata={"_forest_content": "her actual turn, long enough to pass"})
        item = resolve_surface_item(n, {"content": "WANT"})
        self.assertEqual(item["kind"], "text")
        self.assertTrue(item["content"].startswith("her actual turn"))


class TestCESL2(unittest.TestCase):
    def test_vision_node_is_surfaced_not_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_JPEG); path = f.name
        try:
            g = Graph(); r = _vision_frame(g, path)
            mon = SurfacingMonitor(g, _StubVDB(), load_ces_config())
            mon.after_step(_Step([r["forest_id"]]))
            items = mon.get_surfaced()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["node_id"], r["forest_id"])
            self.assertEqual(items[0]["image_ref"], path)
            self.assertEqual(items[0]["content"], "")
            ctx = mon.format_context(items)
            self.assertIn("image attached", ctx)
            # LAW 7: the marker names that a picture is attached, never what is in it
            self.assertNotIn("dog", ctx); self.assertNotIn("photo of", ctx)
        finally:
            os.unlink(path)

    def test_text_node_via_l2_is_substrate_first(self):
        g = Graph()
        g.create_node(node_id="t1", metadata={"_forest_content": "her words on the substrate, not the shard"})
        mon = SurfacingMonitor(g, _StubVDB({"t1": {"content": "shard", "metadata": {}}}), load_ces_config())
        mon.after_step(_Step(["t1"]))
        items = mon.get_surfaced()
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["content"].startswith("her words"))
        self.assertNotIn("image_ref", items[0])

    def test_node_with_nothing_showable_is_still_skipped(self):
        g = Graph(); g.create_node(node_id="empty", metadata={})
        mon = SurfacingMonitor(g, _StubVDB(), load_ces_config())
        mon.after_step(_Step(["empty"]))
        self.assertEqual(mon.get_surfaced(), [])


class TestAssembleInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rpc = importlib.import_module("neurograph_rpc")

    def test_builds_image_url_user_message(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_JPEG); path = f.name
        try:
            msgs = self.rpc._vision_surface_messages([{"node_id": "x", "image_ref": path, "score": 1.23}])
            self.assertEqual(len(msgs), 1)
            m = msgs[0]
            self.assertEqual(m["role"], "user")
            kinds = [b["type"] for b in m["content"]]
            self.assertEqual(kinds, ["text", "image_url"])
            url = m["content"][1]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), _JPEG)
            self.assertIn("1.23", m["content"][0]["text"])
            # no caption anywhere
            self.assertNotIn("photo of", m["content"][0]["text"].lower())
        finally:
            os.unlink(path)

    def test_caps_and_skips(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_JPEG); path = f.name
        try:
            items = [{"image_ref": path}] * 5 + [{"image_ref": "/nope.jpg"}, {"content": "text only"}]
            self.assertEqual(len(self.rpc._vision_surface_messages(items, max_images=2)), 2)
            # byte cap: skipped, never a phantom block
            self.assertEqual(self.rpc._vision_surface_messages([{"image_ref": path}], max_bytes=4), [])
            self.assertEqual(self.rpc._vision_surface_messages([], ), [])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
