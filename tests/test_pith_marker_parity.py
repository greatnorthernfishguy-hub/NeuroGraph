# tests/test_pith_marker_parity.py
#
# ---- Changelog ----
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — Pith marker
#   parity with miniTID (Packet 175 M1 / LE sweep (vii), Pith work)
# What: pins cc_ng_organism's _PITH_HARNESS_MARKERS to miniTID's
#   is_synthetic_harness_text MARKERS, and _PITH_PROVIDER_REJECTED to the
#   strings miniTID's provider_context_is_usable rejects.
# Why: the two lists drifted across the language boundary (Pith lacked the
#   surfaced marker), and one surfaced line voids miniTID's whole fresh
#   context (LE sweep 001 (vii)).  One source of truth is miniTID; this test
#   reads it rather than holding a third copy.
# How: parses Condensate rust_core/src/minitid.rs (CONDENSATE_MINITID_RS, or
#   ~/Condensate/rust_core/src/minitid.rs) and resolves its &str consts.
# -------------------
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism as pith

_MINITID_RS = os.environ.get(
    "CONDENSATE_MINITID_RS",
    os.path.expanduser("~/Condensate/rust_core/src/minitid.rs"))


@pytest.fixture(scope="module")
def minitid_src():
    if not os.path.isfile(_MINITID_RS):
        pytest.skip(f"miniTID source not found at {_MINITID_RS}; "
                    "set CONDENSATE_MINITID_RS to check marker parity")
    with open(_MINITID_RS, encoding="utf-8") as f:
        return f.read()


def _fn_body(src, name):
    start = src.index(f"fn {name}(")
    return src[start:src.index("\n}\n", start)]


def _resolve(src, token):
    token = token.strip()
    if token.startswith('"'):
        return token[1:-1]
    match = re.search(rf'const {token}: &str =\s*"([^"]*)";', src)
    assert match, f"miniTID const {token} not found"
    return match.group(1)


def test_harness_markers_match_minitid(minitid_src):
    body = _fn_body(minitid_src, "is_synthetic_harness_text")
    array = re.search(r"const MARKERS: &\[&str\] = &\[(.*?)\];", body, re.S)
    assert array, "miniTID MARKERS array not found"
    markers = tuple(_resolve(minitid_src, token)
                    for token in array.group(1).split(",") if token.strip())
    assert markers, "parsed no miniTID markers"
    assert pith._PITH_HARNESS_MARKERS == markers


def test_provider_rejections_match_minitid(minitid_src):
    body = _fn_body(minitid_src, "provider_context_is_usable")
    tokens = re.findall(r'\.contains\(([A-Z_]+|"[^"]*")\)', body)
    rejected = tuple(_resolve(minitid_src, token) for token in tokens)
    assert rejected, "parsed no miniTID provider rejections"
    assert pith._PITH_PROVIDER_REJECTED == rejected
