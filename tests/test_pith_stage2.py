# tests/test_pith_stage2.py
#
# ---- Changelog ----
# [2026-07-10] Claude Code (Opus 4.8) — Pith Stage 2 tests (concept-aware keyframe)
# What: Direct tests of the EXTRACTIVE pith_stage2_keyframe() (keep the highest-
#   information segments, not the head) + pith_stage3's graceful-degradation
#   fill. Headline case: a greeting+payload item keyframes to the PAYLOAD, not
#   the greeting -- the failure mode of first-sentence compression. Plus:
#   high-signal-beats-filler, marker/bounds/delta, short passthrough, empty,
#   in-stage3 compression (lod = retained fraction, not dropped), strict-prefix
#   break when even a keyframe overflows, pinned never compressed, metrics.
# Why: docs/prd/Pith_PRD_v0.1.md, Stage 2 -- softens Stage 3's hard budget
#   cliff into graceful degradation, keeping the concepts an item carries.
# How: plain CacheLine construction + pith_stage2_keyframe()/pith_stage3();
#   _PITH_METRICS reset() in an autouse fixture, same pattern as the Stage-1/3
#   suites.
# -------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cc_ng_organism import (
    CacheLine,
    pith_stage2_keyframe,
    pith_stage3,
    _PITH_METRICS,
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    _PITH_METRICS.reset()
    yield
    _PITH_METRICS.reset()


# ---------------------------------------------------------------------------
# pith_stage2_keyframe() -- concept-aware extraction
# ---------------------------------------------------------------------------

def test_short_content_untouched():
    content = "a short line of text well under the default budget"
    kf, delta = pith_stage2_keyframe(content)
    assert kf == content
    assert delta == ""
    assert "⋯" not in kf


def test_keyframe_drops_greeting_keeps_payload():
    # THE case first-sentence compression gets wrong: the greeting is the head,
    # the payload is buried. Concept-aware extraction must keep the payload
    # (dense in identifiers/numbers) and drop the social pre/postamble.
    content = (
        "Good morning, my friend! Hope you slept well and had a nice coffee.\n"
        "The save_guard refuses to overwrite main.msgpack when node_count drops "
        "below 50% of last_good, writing a timestamped backup to "
        "checkpoints/last_good first.\n"
        "Anyway, thanks so much and talk soon, cheers!"
    )
    kf, delta = pith_stage2_keyframe(content, max_chars=160)

    assert "save_guard" in kf and "main.msgpack" in kf, "payload must survive"
    assert "Good morning" not in kf, "greeting must not survive"
    assert "cheers" not in kf, "sign-off must not survive"
    assert kf.rstrip().endswith("]"), "must carry the ⋯[+N] marker"
    # the dropped greeting/sign-off land in the delta for later expansion
    assert "Good morning" in delta or "cheers" in delta


def test_keyframe_prefers_high_signal_over_longer_filler():
    # A long low-signal filler line vs a short fact-dense line -- the fact line
    # wins despite being shorter, proving selection is by information, not length.
    filler = ("we then proceeded to discuss the general vibe of things at some "
              "length in a rather rambling and non specific manner for a while")
    fact = "commit bc26d5b shipped pith_stage3 with CC_PITH_L1_BUDGET=4000"
    content = filler + "\n" + fact + "\n" + filler
    kf, _ = pith_stage2_keyframe(content, max_chars=120)

    assert "bc26d5b" in kf or "CC_PITH_L1_BUDGET" in kf, "fact-dense line should win"


def test_keyframe_marker_bounds_and_delta():
    head = "## Design Doc: Widget Factory calibration procedure"
    prose = ("The calibration routine sweeps torque from 0.1 to 4.0 Nm across "
              "12 fixtures and logs residuals to widget_cal.json each pass. ") * 4
    content = head + "\n" + prose
    max_chars = 220

    kf, delta = pith_stage2_keyframe(content, max_chars=max_chars)

    marker_idx = kf.rfind(" ⋯[+")
    assert marker_idx != -1 and kf.endswith("]")
    body = kf[:marker_idx]
    assert len(body) <= max_chars
    assert delta != ""


def test_empty_and_whitespace_content():
    assert pith_stage2_keyframe("") == ("", "")
    assert pith_stage2_keyframe("   \n\t  ") == ("", "")
    assert pith_stage2_keyframe(None) == ("", "")


# ---------------------------------------------------------------------------
# pith_stage3() graceful-degradation integration
# ---------------------------------------------------------------------------

def _long_item(node_id, score, stream="pattern"):
    # A fact-dense long body so the keyframe is meaningfully shorter than full.
    body = ("The reconcile pass at timestep 16371 pruned 76 entities and "
            "compacted outcomes.tract to 949k rows while preserving the "
            "watermark. ") * 6
    return CacheLine.from_surfaced(node_id, "## Reconcile\n" + body,
                                    score=score, stream=stream)


def test_graceful_degradation_keeps_overflow_item_as_keyframe():
    top = CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    overflow = _long_item("mid", score=5.0)
    tiny = CacheLine.from_surfaced("low", "z" * 10, score=1.0, stream="pattern")

    out = pith_stage3([top, overflow, tiny], budget_chars=300)
    ids = [l.node_id for l in out]

    assert "mid" in ids, "overflow item should be kept as a keyframe, not dropped"
    mid = out[ids.index("mid")]
    assert 0.0 < mid.lod < 1.0, "lod records the retained fraction"
    assert mid.keyframe is True
    assert "⋯[+" in mid.content
    assert mid.deltas and mid.deltas[0] != ""
    assert _PITH_METRICS.compressed_count == 1
    assert _PITH_METRICS.chars_saved > 0


def test_break_when_even_keyframe_overflows():
    top = CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    overflow = _long_item("mid", score=5.0)

    # Only ~30 chars left after "top" -- too small for even a keyframe -- so
    # "mid" is dropped and strict rank-prefix stops the fill there.
    out = pith_stage3([top, overflow], budget_chars=80)
    ids = [l.node_id for l in out]

    assert ids == ["top"]
    assert _PITH_METRICS.compressed_count == 0
    assert _PITH_METRICS.ranked_dropped == 1


def test_pinned_never_compressed():
    pinned_content = "## Pinned Heading\n" + ("word " * 200)
    pinned = CacheLine.from_surfaced("pin", pinned_content, score=0.0,
                                      pinned=True, stream="pattern")
    top = CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    overflow = _long_item("mid", score=5.0)

    out = pith_stage3([pinned, top, overflow], budget_chars=300)
    ids = [l.node_id for l in out]

    pin_line = out[ids.index("pin")]
    assert pin_line.content == pinned_content, "pinned content untouched verbatim"
    assert pin_line.lod == 1.0, "pinned line never entered the keyframe branch"
    assert "⋯[+" not in pin_line.content
    # a non-pinned peer still degraded under the same budget
    assert out[ids.index("mid")].lod < 1.0
    assert _PITH_METRICS.compressed_count == 1


def test_metrics_snapshot_consistent():
    top = CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    overflow = _long_item("mid", score=5.0)

    pith_stage3([top, overflow], budget_chars=300)
    snap = _PITH_METRICS.snapshot()

    assert snap["compressed_count"] == 1
    assert snap["chars_saved"] > 0
