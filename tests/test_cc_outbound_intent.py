# ---- Changelog ----
# [2026-11-24] DudeMan CC (Claude Fable 5.2, laptop; SQ11, punch #1250 item D) -- the CC's outbound intent deposit
# What: a deposited want is one readable experience frame with source "cc_outbound" and the WHOLE text; frames
#       append; empty text writes nothing; the path comes from CC_NG_OUTBOUND_TRACT; a failure is a WARNING without
#       the text.
# Why:  the deposit half of the CC's own-turn channel (docs/prd/2026-11-24-cc-own-turn-channel-proposal.md).
# How:  the real crate writer and reader on a temp file.
# -------------------
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ng_tract  # noqa: E402
import cc_ng_organism as org  # noqa: E402


def _frames(path):
    with open(path, "rb") as f:
        return [(e.entry_type, e.source, e.content) for e in ng_tract.TractReader(f.read())]


def test_a_deposit_is_one_whole_raw_frame_and_frames_append(tmp_path):
    path = str(tmp_path / "sub" / "cc_outbound.tract")
    long_text = "open question: " + "why does the queue saturate? " * 400          # ~12k chars, kept whole
    assert org.cc_deposit_outbound_intent(long_text, path) is True
    assert org.cc_deposit_outbound_intent("a second want", path) is True
    assert _frames(path) == [(ng_tract.ENTRY_EXPERIENCE, "cc_outbound", long_text),
                             (ng_tract.ENTRY_EXPERIENCE, "cc_outbound", "a second want")]


def test_empty_text_writes_nothing(tmp_path):
    path = str(tmp_path / "cc_outbound.tract")
    for bad in ("", "   \n", None, 7):
        assert org.cc_deposit_outbound_intent(bad, path) is False
    assert not os.path.exists(path)


def test_the_path_comes_from_the_environment(tmp_path, monkeypatch):
    path = str(tmp_path / "from_env.tract")
    monkeypatch.setenv("CC_NG_OUTBOUND_TRACT", path)
    assert org.cc_outbound_tract_path() == path
    assert org.cc_deposit_outbound_intent("via env") is True and _frames(path)[0][2] == "via env"
    monkeypatch.delenv("CC_NG_OUTBOUND_TRACT")
    assert org.cc_outbound_tract_path().endswith("/.et_modules/shared_learning/cc_outbound.tract")


def test_a_failed_deposit_is_a_warning_without_the_text(tmp_path, caplog):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    with caplog.at_level(logging.WARNING):
        assert org.cc_deposit_outbound_intent("the secret want", str(blocker / "under_a_file" / "t.tract")) is False
    assert "deposit_failed" in caplog.text and "secret" not in caplog.text
