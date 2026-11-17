# ---- Changelog ----
# [2026-11-16] DudeMan CC (Claude Fable 5.2, laptop; NG suite repair #761) -- test_restore_warns_on_legacy_json builds its
#   simulated legacy file with a dict-shaped "synapses" map, as a real pre-#325 JSON checkpoint held. Commit 0f5447f
#   (2026-08-26, "perf: stream checkpoint synapses through native store (no dict inflation)") made _serialize_full() return
#   that map as pre-packed msgpack BYTES; json.dump(default=str) then wrote the repr of the bytes as a string, which is not
#   a legacy checkpoint at all (bulk_load: 'str' cannot be cast as 'dict'). The fixture now unpacks any packed sub-map
#   before json.dump. The assertion (restore reads it AND warns "legacy lossy-JSON") is unchanged.
# [2026-06-14] Claude Code (Opus 4.8) — #325 checkpoint msgpack-enforcer tests
# What: Graph.checkpoint() must REFUSE non-.msgpack paths (lossy JSON has no place on the
#   topology path); restore() must still READ a legacy .json but WARN. Empty graphs suffice —
#   the enforcer is about path/format, not node content.
# Why: a one-char path choice silently produced lossy-JSON topology persistence (#325).
# -------------------
import json
import os
import tempfile

import pytest

from neuro_foundation import Graph, CheckpointMode


def test_checkpoint_refuses_json():
    g = Graph()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = f.name
    try:
        with pytest.raises(ValueError, match="msgpack"):
            g.checkpoint(p, CheckpointMode.FULL)
    finally:
        os.unlink(p)


def test_checkpoint_refuses_extensionless():
    g = Graph()
    with pytest.raises(ValueError):
        g.checkpoint("/tmp/ng_topology_no_extension")


def test_checkpoint_refuses_json_for_all_modes():
    g = Graph()
    for mode in (CheckpointMode.FULL, CheckpointMode.INCREMENTAL, CheckpointMode.FORK):
        with pytest.raises(ValueError, match="msgpack"):
            g.checkpoint("/tmp/ng_topology.json", mode)


def test_checkpoint_msgpack_roundtrips():
    g = Graph()
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
        p = f.name
    try:
        g.checkpoint(p, CheckpointMode.FULL)         # must NOT raise
        g2 = Graph()
        g2.restore(p)                                # must NOT warn or raise
    finally:
        os.unlink(p)


def test_restore_warns_on_legacy_json(tmp_path):
    # Simulate a pre-#325 legacy JSON checkpoint and confirm restore() reads it but warns loudly.
    g = Graph()
    data = g._serialize_full()
    # [2026-11-16] a legacy JSON checkpoint held plain maps; today's serializer pre-packs some as msgpack bytes (0f5447f)
    import msgpack
    data = {k: (msgpack.unpackb(bytes(v), raw=False) if isinstance(v, (bytes, bytearray)) else v)
            for k, v in data.items()}
    p = tmp_path / "legacy_ng_lite_state.json"
    with open(p, "w") as f:
        json.dump(data, f, default=str)
    g2 = Graph()
    with pytest.warns(RuntimeWarning, match="legacy lossy-JSON"):
        g2.restore(str(p))
