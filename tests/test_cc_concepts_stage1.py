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
