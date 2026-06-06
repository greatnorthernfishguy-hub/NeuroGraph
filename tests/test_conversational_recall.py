# ---- Changelog ----
# [2026-06-06] CC (Opus 4.8) — #294 conversational recall: drain-time absorb tests
# What: Unit tests for _absorb_conversational_experience filter/route logic.
# Why: Anima-sourced experience frames must reach recall (forest+trees); peer
#      telemetry and non-experience frames must not.
# How: Duck-typed fake entries + monkeypatched sinks; no substrate, no ng_tract.
# -------------------
import types
import importlib

import neurograph_rpc as rpc


class _FakeEntry:
    """Duck-typed stand-in for ng_tract.PyExperienceEntry / PyOutcomeEntry."""
    def __init__(self, entry_type, source, content):
        self.entry_type = entry_type
        self.source = source
        self.content = content


def _install_capturing_sinks(monkeypatch):
    calls = {"forest": [], "trees": []}
    dummy_ingestor = types.SimpleNamespace(ingest=lambda text: calls["forest"].append(text))
    dummy_memory = types.SimpleNamespace(ingestor=dummy_ingestor, _message_count=0)
    monkeypatch.setattr(rpc, "_memory", dummy_memory)
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls["trees"].append(text))
    # _absorb embeds before dual-pass; stub embedding so no model is loaded.
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda text: [0.0])
    return calls, dummy_memory


def test_absorbs_anima_experience_into_forest_and_trees(monkeypatch):
    calls, _ = _install_capturing_sinks(monkeypatch)
    entries = [_FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"the load-bearing joke")]
    n = rpc._absorb_conversational_experience(entries)
    assert n == 1
    assert calls["forest"] == ["the load-bearing joke"]
    assert calls["trees"] == ["the load-bearing joke"]


def test_skips_peer_module_and_non_experience_frames(monkeypatch):
    calls, _ = _install_capturing_sinks(monkeypatch)
    entries = [
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "elmer", b"substrate telemetry"),   # peer module → skip
        _FakeEntry(rpc._ENTRY_OUTCOME, "anima", b"an outcome, not experience"),  # wrong type → skip
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"   "),                    # blank → skip
    ]
    n = rpc._absorb_conversational_experience(entries)
    assert n == 0
    assert calls["forest"] == []
    assert calls["trees"] == []


def test_decodes_bytes_and_increments_message_count(monkeypatch):
    calls, mem = _install_capturing_sinks(monkeypatch)
    entries = [
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "animus", "already a str"),          # legacy dir-name + str content
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"bytes content"),
    ]
    n = rpc._absorb_conversational_experience(entries)
    assert n == 2
    assert calls["forest"] == ["already a str", "bytes content"]
    assert mem._message_count == 2


def test_empty_or_none_is_safe(monkeypatch):
    _install_capturing_sinks(monkeypatch)
    assert rpc._absorb_conversational_experience([]) == 0
    assert rpc._absorb_conversational_experience(None) == 0
