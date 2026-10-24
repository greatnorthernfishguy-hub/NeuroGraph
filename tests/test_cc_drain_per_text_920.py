# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop) — #920 per text (#1173 fix set, Josh's approval 2026-10-23)
# What: drain_ingest_tract(per_text=True): retries counted per turn TEXT (sha1, the #907 forest id) so the copies of one
#   text share one budget; once a text is quarantined, every later copy (retry tract or main tract) goes straight to
#   <retry tract>.quarantine whole with no attempt; prepare_ingest_tract(quarantine_base=) skips it (no TID call); the
#   index <retry>.quarantine.texts is rebuilt from the quarantine file when missing or stale; nothing is dropped.
# Why: per raw frame, 14 copies of one poison text cost 14 x 6 TID timeouts (dev-log 2026-10-23_poison-920 §2 open point 1).
# How: the #920 test fixtures (real ng_tract on tmp files, a fake dual pass, a fake ng_embed); nothing under ~/.claude.
# -------------------
import hashlib
import json
import logging
import os
import sys
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import cc_ng_organism
from test_cc_poison_retry_920 import env, _frame, _write, _read, _frames_of, _FakeNGEmbed  # noqa: F401 (fixture)


def _sha1(t):
    return hashlib.sha1(t.encode()).hexdigest()


def _retry(path, state, max_retries=2, **kw):
    return cc_ng_organism.drain_ingest_tract(
        types.SimpleNamespace(nodes={}), None, state, tract_path=path, return_consumed=True,
        hold_on_failure=True, retry_tract_path=path, max_retries=max_retries, per_text=True, **kw)


def _main(path, retry, state, **kw):
    return cc_ng_organism.drain_ingest_tract(
        types.SimpleNamespace(nodes={}), None, state, tract_path=path, return_consumed=True,
        hold_on_failure=True, retry_tract_path=retry, per_text=True, **kw)


def test_copies_share_one_budget_then_go_straight_to_quarantine(env, caplog):
    a, b, c = (_frame(env.tmp, "POISON same text", i) for i in range(3))   # 3 copies (distinct stamps)
    assert len({a, b, c}) == 3
    retry = _write(env.tmp / "retry.tract", [a, b, c])
    caplog.set_level(logging.WARNING, logger="cc_ng_organism")
    st = {"last_forest_id": None}
    _retry(retry, st, max_retries=2, max_entries=8)
    # copy a: retry 1 (rotated); copy b: retry 2 -> quarantined (shared budget); copy c: straight to quarantine
    attempts = [t for t, _ in env.calls]
    assert attempts == ["POISON same text", "POISON same text"]
    assert _frames_of(_read(retry + ".quarantine")) == [b, c]
    assert _frames_of(_read(retry)) == [a]
    _retry(retry, st, max_retries=2, max_entries=8)
    assert [t for t, _ in env.calls] == attempts                              # no further TID attempt
    assert _frames_of(_read(retry + ".quarantine")) == [b, c, a]
    assert _read(retry) == b""
    assert st["quarantined_total"] == 3 and st["quarantined_copies_total"] == 2
    idx = json.load(open(retry + ".quarantine.texts"))
    assert idx["texts"] == {_sha1("POISON same text"): 3}
    assert idx["size"] == os.path.getsize(retry + ".quarantine")
    assert json.load(open(retry + ".retries")) == {}
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("QUARANTINED:" in m for m in msgs) == 1                        # ONE loud ERROR per text
    assert sum("copy of a QUARANTINED turn" in m for m in msgs) == 2
    assert not any("POISON" in m for m in msgs)                               # never the text


def test_main_tract_copy_of_a_quarantined_text_never_attempted(env):
    q = _frame(env.tmp, "POISON once", 0)
    retry = _write(env.tmp / "retry.tract", [])
    with open(retry + ".quarantine", "wb") as f:
        f.write(q)
    good = _frame(env.tmp, "GOOD turn", 1)
    copy = _frame(env.tmp, "POISON once", 2)
    main = _write(env.tmp / "turns.tract", [good, copy])
    st = {"last_forest_id": None}
    (n, consumed), = [_main(main, retry, st)]
    assert n == 1 and consumed == good + copy and _read(main) == b""
    assert [t for t, _ in env.calls] == ["GOOD turn"]
    assert _frames_of(_read(retry + ".quarantine")) == [q, copy]
    assert _read(retry) == b""                                                # not parked: quarantined directly


def test_prepare_skips_quarantined_text_and_prepared_only_drain_moves_it(env):
    q = _frame(env.tmp, "POISON once", 0)
    retry = _write(env.tmp / "retry.tract", [])
    with open(retry + ".quarantine", "wb") as f:
        f.write(q)
    copy = _frame(env.tmp, "POISON once", 1)
    good = _frame(env.tmp, "GOOD turn", 2)
    main = _write(env.tmp / "turns.tract", [copy, good])
    cache = {}
    summ = cc_ng_organism.prepare_ingest_tract(main, cache, max_entries=4, quarantine_base=retry)
    assert _FakeNGEmbed.calls == ["GOOD turn"]                                # no TID call for the quarantined text
    assert summ["quarantined_skipped"] == 1 and summ["prepared"] == 1
    st = {"last_forest_id": None}
    (n, consumed) = _main(main, retry, st, prepared=cache, prepared_only=True)
    assert n == 1 and _read(main) == b""                                      # did not stop at the unprepared copy
    assert _frames_of(_read(retry + ".quarantine")) == [q, copy]


def test_prepare_without_quarantine_base_is_unchanged(env):
    q = _frame(env.tmp, "POISON once", 0)
    retry = _write(env.tmp / "retry.tract", [])
    with open(retry + ".quarantine", "wb") as f:
        f.write(q)
    main = _write(env.tmp / "turns.tract", [_frame(env.tmp, "POISON once", 1)])
    summ = cc_ng_organism.prepare_ingest_tract(main, {}, max_entries=4)
    assert "quarantined_skipped" not in summ and _FakeNGEmbed.calls == ["POISON once"]


def test_index_rebuilt_when_missing_or_stale(env):
    q = _frame(env.tmp, "POISON once", 0)
    retry = _write(env.tmp / "retry.tract", [])
    with open(retry + ".quarantine", "wb") as f:
        f.write(q)
    assert cc_ng_organism._drain_qtexts_load(retry) == {_sha1("POISON once"): 1}     # no index yet
    with open(retry + ".quarantine.texts", "w") as f:
        json.dump({"size": 1, "texts": {}}, f)                                        # stale (size mismatch)
    assert cc_ng_organism._drain_qtexts_load(retry) == {_sha1("POISON once"): 1}
    with open(retry + ".quarantine.texts", "w") as f:
        f.write("{not json")
    assert cc_ng_organism._drain_qtexts_load(retry) == {_sha1("POISON once"): 1}
    cc_ng_organism._drain_qtexts_save(retry, {"x": 7})
    assert cc_ng_organism._drain_qtexts_load(retry) == {"x": 7}                       # a matching index is trusted


def test_legacy_per_frame_keys_dropped_and_success_ends_text_count(env):
    good = _frame(env.tmp, "GOOD turn", 1)
    retry = _write(env.tmp / "retry.tract", [good])
    with open(retry + ".retries", "w") as f:
        json.dump({hashlib.sha256(b"old frame").hexdigest(): 4, "text:" + _sha1("GOOD turn"): 1}, f)
    (n, _c) = _retry(retry, {"last_forest_id": None})
    assert n == 1 and json.load(open(retry + ".retries")) == {}


def test_per_text_off_keeps_per_frame_counting(env):
    a, b = _frame(env.tmp, "POISON same", 0), _frame(env.tmp, "POISON same", 1)
    retry = _write(env.tmp / "retry.tract", [a, b])
    cc_ng_organism.drain_ingest_tract(types.SimpleNamespace(nodes={}), None, {"last_forest_id": None},
                                      tract_path=retry, hold_on_failure=True, retry_tract_path=retry, max_retries=2)
    counts = json.load(open(retry + ".retries"))
    assert counts == {hashlib.sha256(a).hexdigest(): 1, hashlib.sha256(b).hexdigest(): 1}
    assert not os.path.exists(retry + ".quarantine.texts")


def test_conservation_mixed(env):
    texts = ["GOOD a", "POISON x", "POISON x", "GOOD b", "POISON x", "POISON y", "POISON y", "GOOD c"]
    frames = [_frame(env.tmp, t, i) for i, t in enumerate(texts)]
    retry = _write(env.tmp / "retry.tract", frames)
    st = {"last_forest_id": None}
    absorbed = 0
    for _ in range(6):
        (n, _c) = _retry(retry, st, max_retries=2, max_entries=8)
        absorbed += n
    left, quar = _frames_of(_read(retry)), _frames_of(_read(retry + ".quarantine"))
    assert absorbed == 3 and left == []
    assert sorted(quar) == sorted(f for f, t in zip(frames, texts) if t.startswith("POISON"))
    assert sum(1 for t, _ in env.calls if t.startswith("POISON")) == 4       # 2 texts x max_retries, copies free


def test_quarantine_append_failure_attempts_the_copy_and_drops_nothing(env, monkeypatch):
    q = _frame(env.tmp, "POISON once", 0)
    retry = _write(env.tmp / "retry.tract", [])
    with open(retry + ".quarantine", "wb") as f:
        f.write(q)
    copy = _frame(env.tmp, "POISON once", 1)
    main = _write(env.tmp / "turns.tract", [copy])
    monkeypatch.setattr(cc_ng_organism, "_drain_quarantine_frame", lambda p, f: (False, None, "OSError"))
    _main(main, retry, {"last_forest_id": None})
    assert [t for t, _ in env.calls] == ["POISON once"]                        # attempted normally instead
    assert _frames_of(_read(retry)) == [copy]                                  # parked whole in the retry tract
