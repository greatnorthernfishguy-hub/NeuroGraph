# Tests for #reach reach-teaching (docs/prd/2026-06-21-reach-teaching-plan.md)
import neurograph_rpc as ng


def test_reach_success_detects_landed_badge():
    assert ng._reach_success_in_turn("Done. 🔧 read_file({\"path\": \"/x.md\"}) ✓") is True


def test_reach_success_ignores_missed_only():
    # a reach that didn't land (✗, no ✓) must NOT credit competence
    assert ng._reach_success_in_turn("🔧 read_file({\"path\": \"/missing\"}) ✗ not found") is False


def test_reach_success_ignores_no_badge_and_empty():
    assert ng._reach_success_in_turn("I read the document and it says...") is False
    assert ng._reach_success_in_turn("") is False
    assert ng._reach_success_in_turn(None) is False
