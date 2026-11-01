# ---- Changelog ----
# [2026-10-31] DudeMan CC (Claude Opus 5.5, laptop; lane concepts-stage1, #1203; Josh 2026-10-31 "You are a go")
# What: tests for concept formation stage 1 — the switch and parameters, the concept index, recognition at encoding,
#   sleep-time recognition of the existing past.
# Why: docs/superpowers/plans/2026-10-31-concept-formation-stage1.md (spec 2026-10-31-concept-formation-and-recall-design).
# How: plain Graph + SimpleVectorDB with hand-made vectors, as test_cc_bind_reinforce_1173.py builds its graph.
# -------------------
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest

import cc_ng_organism

_CONCEPT_ENV = ("CC_NG_CONCEPTS", "CC_NG_CONCEPT_MATCH", "CC_NG_CONCEPT_DRIFT", "CC_NG_CONCEPT_ETA",
                "CC_NG_CONCEPT_CEILING", "CC_NG_CONCEPT_BIRTH_UP", "CC_NG_CONCEPT_BIRTH_DOWN",
                "CC_NG_CONCEPT_REACTIVATE", "CC_NG_CONCEPT_BACKLOG_PER_SLEEP", "CC_NG_CONCEPT_BACKLOG_BATCH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _CONCEPT_ENV:
        monkeypatch.delenv(k, raising=False)


# ---- Task 2: switch and parameters ----

def test_concepts_off_by_default():
    assert cc_ng_organism.cc_concepts_enabled() is False


def test_concepts_on(monkeypatch):
    monkeypatch.setenv("CC_NG_CONCEPTS", "1")
    assert cc_ng_organism.cc_concepts_enabled() is True


def test_params_defaults():
    p = cc_ng_organism.cc_concept_params()
    assert p == {"match": 0.60, "drift": 0.05, "eta": 0.2, "ceiling": 1.0, "birth_up": 0.2, "birth_down": 0.15,
                 "reactivate": 0.5, "backlog": 200, "batch": 25}
    assert isinstance(p["backlog"], int) and isinstance(p["batch"], int)


def test_params_clamped(monkeypatch):
    monkeypatch.setenv("CC_NG_CONCEPT_MATCH", "2")
    monkeypatch.setenv("CC_NG_CONCEPT_DRIFT", "abc")
    monkeypatch.setenv("CC_NG_CONCEPT_BACKLOG_PER_SLEEP", "-5")
    monkeypatch.setenv("CC_NG_CONCEPT_BACKLOG_BATCH", "5000")
    monkeypatch.setenv("CC_NG_CONCEPT_BIRTH_UP", "0")
    p = cc_ng_organism.cc_concept_params()
    assert p["match"] == 0.95
    assert p["drift"] == 0.05
    assert p["backlog"] == 0
    assert p["batch"] == 1000
    assert p["birth_up"] == 0.01


# ---- Task 3: the concept index and nearest-concept lookup ----

import numpy as np
from universal_ingestor import SimpleVectorDB


def _unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def zero_centre(monkeypatch):
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_centre", lambda dim: np.zeros(dim, dtype=np.float32))


def _concept_row(vdb, cid, vec, label="c"):
    vdb.insert(cid, np.asarray(vec, dtype=np.float32), label, {"cc": True, "_cc_concept": True})


def test_empty_index_returns_no_match(zero_centre):
    assert cc_ng_organism._cc_concept_nearest(SimpleVectorDB(), _unit(1, 0, 0), 0.6) is None


def test_match_above_threshold(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    _concept_row(vdb, "cc:concept::b", _unit(0, 1, 0))
    vdb.insert("cc:conv::x::tree::q", _unit(1, 0.05, 0), "not a concept", {"cc": True})
    hit = cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0.1, 0), 0.6)
    assert hit is not None and hit[0] == "cc:concept::a" and hit[1] > 0.9


def test_below_threshold_none(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 1.5, 0), 0.6) is None


def test_centred_not_raw(monkeypatch):
    centre = np.asarray([0.6, 0.6, 0.0], dtype=np.float32)
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_centre", lambda dim: centre)
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0.2, 0))
    q = _unit(0.2, 1, 0)
    assert float(np.dot(_unit(1, 0.2, 0), q)) >= 0.35       # raw cosine is sizeable ...
    assert cc_ng_organism._cc_concept_nearest(vdb, q, 0.35) is None   # ... centred it is not a match


def test_dim_mismatch_skipped(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0, 0), 0.6) is None


def test_index_rebuilds_after_vdb_change(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6)[0] == "cc:concept::a"
    _concept_row(vdb, "cc:concept::a", _unit(0, 0, 1))      # same object, same count, new vector
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6) is None
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 0, 1), 0.6)[0] == "cc:concept::a"


def test_index_after_load(zero_centre, tmp_path):
    other = SimpleVectorDB()
    _concept_row(other, "cc:concept::z", _unit(0, 1, 0))
    path = str(tmp_path / "v.msgpack")
    other.save(path)
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6)[0] == "cc:concept::a"
    vdb.load(path)                                          # same object, same concept count, different rows
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6) is None
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 1, 0), 0.6)[0] == "cc:concept::z"
