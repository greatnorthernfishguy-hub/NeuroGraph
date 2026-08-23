# ---- Changelog ----
# [2026-08-23] Claude Code (Opus 4.8) — poincare_dir compact-bytes round-trip (#119)
# What: unit tests for pack_poincare_dir / poincare_dir_array — the #119
#   increment-1 footprint fix that replaces the ~24 KB/node Python list of
#   768 boxed floats with a 3072-byte float32 buffer (~8× smaller, ~837 MB
#   reclaimed across ~40K nodes).
# Why: the fix touches every GSG read-path; before it can ride a live respawn
#   it must be proven lossless AND backward-compatible with pre-#119 checkpoints
#   that still hold the legacy Python-list form (decoded until the one-time
#   _gsg_backfill_existing_nodes pass re-saves).
# How: pure round-trip at the helper boundary — no fixtures/models needed since
#   pack/unpack are self-contained. Also asserts the compact size, the legacy
#   list decode path, the empty/None guards, and that the recovered array is
#   usable by the GSG math (dot / norm / layer-norm scale).
# -------------------
"""Unit tests for #119 poincare_dir compact-bytes storage."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from neuro_foundation import pack_poincare_dir, poincare_dir_array


def _unit(dim=768, seed=7):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_pack_returns_compact_bytes():
    d = _unit()
    packed = pack_poincare_dir(d)
    assert isinstance(packed, bytes)
    # float32 buffer: exactly 4 bytes/element, ~8× smaller than the boxed list.
    assert len(packed) == d.size * 4 == 3072


def test_round_trip_is_lossless():
    d = _unit()
    packed = pack_poincare_dir(d)
    out = poincare_dir_array({"poincare_dir": packed})
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == d.shape
    # float32 in -> float32 bytes -> float32 out: bit-exact.
    np.testing.assert_array_equal(out, d)


def test_pack_accepts_list_and_ndarray():
    d = _unit()
    from_list = pack_poincare_dir(d.tolist())
    from_arr = pack_poincare_dir(d)
    assert from_list == from_arr


def test_legacy_list_form_still_decodes():
    # Pre-#119 checkpoints store a Python list; readers must still work until
    # the backfill re-saves them in compact form.
    d = _unit()
    out = poincare_dir_array({"poincare_dir": d.tolist()})
    assert out is not None
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, d, rtol=0, atol=1e-6)


def test_absent_and_empty_metadata_return_none():
    assert poincare_dir_array(None) is None
    assert poincare_dir_array({}) is None
    assert poincare_dir_array({"poincare_dir": None}) is None


def test_recovered_array_supports_gsg_math():
    # GSG read-paths only dot / norm / scale the direction — confirm the
    # zero-copy frombuffer array supports all three despite being read-only.
    d = _unit()
    out = poincare_dir_array({"poincare_dir": pack_poincare_dir(d)})
    assert float(np.dot(out, out)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-5)
    for layer_norm in (0.70, 0.50, 0.30):
        pt = out * layer_norm  # produces a fresh writable array; no mutation of `out`
        assert float(np.linalg.norm(pt)) == pytest.approx(layer_norm, abs=1e-5)


def test_bytes_path_is_zero_copy_readonly():
    # frombuffer returns a read-only view; document/lock that contract so a
    # future in-place mutation in a read-path fails loudly here first.
    d = _unit()
    out = poincare_dir_array({"poincare_dir": pack_poincare_dir(d)})
    assert out.flags.writeable is False
