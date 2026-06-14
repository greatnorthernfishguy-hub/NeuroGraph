# ---- Changelog ----
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
    p = tmp_path / "legacy_ng_lite_state.json"
    with open(p, "w") as f:
        json.dump(data, f, default=str)
    g2 = Graph()
    with pytest.warns(RuntimeWarning, match="legacy lossy-JSON"):
        g2.restore(str(p))
