# ---- Changelog ----
# [2026-11-18] DudeMan CC (#1243; Josh 2026-11-18 "you have my go and Proceed for ... #1243") -- created with the
#   PROTECTED activation_persistence.py commit.
# What: the activation sidecar's format follows its suffix. A `.msgpack` suffix writes msgpack (SNN state: msgpack only,
#       Format-for-Purpose); the default `.activations.json` suffix still writes the same JSON bytes as before. The
#       reader goes by the file's content, and with a msgpack suffix it falls back to the legacy `.activations.json`
#       file when the msgpack one does not exist yet (first start after the switch).
# Why:  #1243: SNN state held as JSON. Stage 1 changes no default, so nothing that names the file has to move yet.
# How:  a plain dict of node objects as the graph (the node-loop capture path); no daemon, no wheel needed.
# -------------------
"""#1243 stage 1: activation sidecar as msgpack when the suffix says so."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_persistence import ActivationPersistence  # noqa: E402
from ces_config import CESConfig  # noqa: E402

MSGPACK_SUFFIX = ".activations.msgpack"
LEGACY_SUFFIX = ".activations.json"


def _graph():
    nodes = {
        "a": SimpleNamespace(voltage=0.1 + 0.2, resting_potential=0.0, last_spike_time=41.0, intrinsic_excitability=1.25),
        "b": SimpleNamespace(voltage=-3.75, resting_potential=0.0, last_spike_time=7.0, intrinsic_excitability=0.5),
        "rest": SimpleNamespace(voltage=0.0, resting_potential=0.0, last_spike_time=0.0, intrinsic_excitability=1.0),
        "ünï::tree::x y": SimpleNamespace(voltage=18.01178581431049, resting_potential=0.0, last_spike_time=673706.0,
                                          intrinsic_excitability=4.098172631326843),
    }
    return SimpleNamespace(nodes=nodes, timestep=673708)


def _fresh_graph():
    g = _graph()
    for n in g.nodes.values():
        n.voltage = 0.0
        n.intrinsic_excitability = 1.0
    return g


def _ap(suffix=None):
    cfg = CESConfig()
    if suffix is not None:
        cfg.persistence.sidecar_suffix = suffix
    return ActivationPersistence(cfg)


def test_default_suffix_is_unchanged_and_still_writes_the_same_json(tmp_path):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    path = ap.write_state(ckpt, captured)
    assert path == ckpt + LEGACY_SUFFIX
    assert Path(path).read_bytes() == json.dumps(captured).encode("utf-8")


def test_msgpack_suffix_writes_msgpack_with_the_same_envelope(tmp_path):
    ap = _ap(MSGPACK_SUFFIX)
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    path = ap.write_state(ckpt, captured)
    assert path == ckpt + MSGPACK_SUFFIX
    raw = Path(path).read_bytes()
    assert raw[:1] != b"{"
    assert msgpack.unpackb(raw, raw=False) == captured          # every float exact, every key kept
    assert not Path(ckpt + LEGACY_SUFFIX).exists()              # the legacy file is not written beside it
    assert sorted(captured["entries"]) == ["a", "b", "ünï::tree::x y"]


def test_msgpack_sidecar_restores_exact_values(tmp_path):
    ap = _ap(MSGPACK_SUFFIX)
    ckpt = str(tmp_path / "main.msgpack")
    src = _graph()
    ap.write_state(ckpt, ap.capture_state(src))
    dst = _fresh_graph()
    assert _ap(MSGPACK_SUFFIX).restore(dst, ckpt) == 3
    for nid in ("a", "b", "ünï::tree::x y"):
        # restore applies decay for the seconds since the save: compare against the same formula
        assert abs(dst.nodes[nid].voltage - src.nodes[nid].voltage) <= abs(src.nodes[nid].voltage) * 1e-4
        assert dst.nodes[nid].intrinsic_excitability == src.nodes[nid].intrinsic_excitability
    assert dst.nodes["rest"].voltage == 0.0


def test_receipt_names_the_msgpack_path(tmp_path):
    ap = _ap(MSGPACK_SUFFIX)
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    receipt = ap.write_state(ckpt, captured, with_receipt=True)
    assert receipt == {"path": ckpt + MSGPACK_SUFFIX, "saved_at": captured["saved_at"]}


def test_msgpack_suffix_falls_back_to_the_legacy_json_file(tmp_path):
    """First start after the switch: only the old .activations.json exists. It is read, and left in place."""
    ckpt = str(tmp_path / "main.msgpack")
    src = _graph()
    _ap().write_state(ckpt, _ap().capture_state(src))            # the old engine's file
    legacy = Path(ckpt + LEGACY_SUFFIX)
    before = legacy.read_bytes()
    dst = _fresh_graph()
    assert _ap(MSGPACK_SUFFIX).restore(dst, ckpt) == 3
    assert dst.nodes["b"].intrinsic_excitability == 0.5
    assert legacy.read_bytes() == before
    assert not Path(ckpt + MSGPACK_SUFFIX).exists()              # restore writes nothing


def test_msgpack_file_wins_over_a_stale_legacy_file(tmp_path):
    ckpt = str(tmp_path / "main.msgpack")
    old = _graph()
    old.nodes["a"].intrinsic_excitability = 9.0
    _ap().write_state(ckpt, _ap().capture_state(old))            # stale legacy file
    _ap(MSGPACK_SUFFIX).write_state(ckpt, _ap(MSGPACK_SUFFIX).capture_state(_graph()))
    dst = _fresh_graph()
    assert _ap(MSGPACK_SUFFIX).restore(dst, ckpt) == 3
    assert dst.nodes["a"].intrinsic_excitability == 1.25


def test_reader_goes_by_content_not_by_suffix(tmp_path):
    """Rollback: the old default suffix pointed at a file that holds msgpack (or the reverse) still restores."""
    ckpt = str(tmp_path / "main.msgpack")
    captured = _ap().capture_state(_graph())
    Path(ckpt + LEGACY_SUFFIX).write_bytes(msgpack.packb(captured, use_bin_type=True))
    dst = _fresh_graph()
    assert _ap().restore(dst, ckpt) == 3
    Path(ckpt + MSGPACK_SUFFIX).write_text(json.dumps(captured))
    dst = _fresh_graph()
    assert _ap(MSGPACK_SUFFIX).restore(dst, ckpt) == 3


def test_unreadable_sidecar_restores_nothing_and_does_not_raise(tmp_path):
    ckpt = str(tmp_path / "main.msgpack")
    Path(ckpt + MSGPACK_SUFFIX).write_bytes(b"\x84\xa7version")   # torn msgpack
    assert _ap(MSGPACK_SUFFIX).restore(_fresh_graph(), ckpt) == 0
    Path(ckpt + MSGPACK_SUFFIX).write_bytes(b"{incomplete")
    assert _ap(MSGPACK_SUFFIX).restore(_fresh_graph(), ckpt) == 0


def test_decay_applies_the_same_to_both_formats(tmp_path):
    ckpt_j, ckpt_m = str(tmp_path / "j.msgpack"), str(tmp_path / "m.msgpack")
    captured = _ap().capture_state(_graph())
    captured["saved_at"] = time.time() - 7200.0                   # two hours ago
    _ap().write_state(ckpt_j, captured)
    _ap(MSGPACK_SUFFIX).write_state(ckpt_m, captured)
    gj, gm = _fresh_graph(), _fresh_graph()
    assert _ap().restore(gj, ckpt_j) == _ap(MSGPACK_SUFFIX).restore(gm, ckpt_m) == 3
    for nid in ("a", "b"):
        assert abs(gj.nodes[nid].voltage - gm.nodes[nid].voltage) < 1e-6
        assert abs(gm.nodes[nid].voltage) < abs(_graph().nodes[nid].voltage)
