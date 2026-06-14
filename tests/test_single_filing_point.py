# ---- Changelog ----
# [2026-06-14] Claude Code (Opus 4.8) — #294-A single filing point tests
# What: Tests that conversational experience recall-indexes via the single chokepoint
#   (_file_conversational_experience) regardless of feeder, and that non-conversational
#   source no-ops there.
# Why: Design docs/prd/2026-06-14-syl-recall-heal-phase1-design.md Component A — filing must
#   be a property of conversational experience entering the substrate, not of routing luck.
# How: monkeypatch the dual-pass wrapper + embedder; assert dispatch / no-dispatch.
# -------------------
import importlib

rpc = importlib.import_module("neurograph_rpc")


def test_helper_files_conversational_source(monkeypatch):
    calls = []
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls.append(text))
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda t: [0.0] * 768)
    assert rpc._file_conversational_experience("hello from syl", source="anima") is True
    assert calls == ["hello from syl"], "conversational source must reach the dual-pass recall path"


def test_helper_ignores_non_conversational_source(monkeypatch):
    calls = []
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls.append(text))
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda t: [0.0] * 768)
    assert rpc._file_conversational_experience("doc chunk text", source="some_doc_feeder") is False
    assert calls == [], "non-conversational source must NOT recall-index here"


def test_helper_skips_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls.append(text))
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda t: [0.0] * 768)
    assert rpc._file_conversational_experience("   ", source="anima") is False
    assert calls == []


def test_drain_experience_entry_routes_conversational(monkeypatch):
    calls = []
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls.append(("dp", text)))
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda t: [0.0] * 768)
    # conversational source must take the dual-pass route, NOT the ingestor (which would need _memory)
    rpc._drain_experience_entry("her words", "text", "anima")
    assert ("dp", "her words") in calls, "conversational feeder must recall-index via the single filing point"
