# ---- Changelog ----
# [2026-07-05] CC (laptop) — Tool-call deposits: Commons-only, never the main substrate
# What: Tests for _handle_post_tool_use()/_deposit_tool_experience() in cc_ng_host.py.
# Why:  Verifies the fix for CC's own "[NeuroGraph Surfaced Knowledge]" hook context
#   surfacing literal "tool:Edit file:..."/"bash:..." strings verbatim -- tool-call
#   telemetry must land ONLY in CC's isolated Commons medium, never the main graph/
#   vector_db (which is what SurfacingMonitor draws from). Mirrors Syl's TID Substrate
#   Peninsula (#97, 2026-06-30) pattern.
# How:  Real NeuroGraphMemory (temp workspace) + real Commons (cc_ng_organism's
#   get_cc_commons singleton) -- no mocks, since the whole point is verifying which
#   real object gets a real new entry.
# -------------------

import hashlib
import tempfile
import shutil

import pytest


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_tool_deposit_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def _commons_has_target(target_id):
    from cc_ng_organism import get_cc_commons
    commons = get_cc_commons("/tmp/cc_tool_deposit_test_workspace")
    return any(
        syn.target_id == target_id for syn in commons._ng.synapses.values()
    )


def test_post_tool_use_deposits_to_commons_not_main_graph(cc_ng):
    import cc_ng_host

    cc_ng_host._STATE.cc_ng = cc_ng
    before_nodes = len(cc_ng.graph.nodes)

    data = {
        "tool_name": "Read",
        "tool_input": {"file_path": "/home/josh/NeuroGraph/some_unique_marker_file.py"},
        "tool_response": "content of some unique marker file that should never enter the main graph",
    }
    result = cc_ng_host._handle_post_tool_use(data)
    assert result == {"ok": True}

    # Main graph must be untouched -- no on_message()-driven node creation.
    assert len(cc_ng.graph.nodes) == before_nodes

    # But the experience did land in CC's Commons (still gets the "SNN magic").
    experience = (
        "tool:Read file:/home/josh/NeuroGraph/some_unique_marker_file.py content:"
        "content of some unique marker file that should never enter the main graph"
    )
    target_id = f"cc:experience:{hashlib.sha256(experience.encode()).hexdigest()[:16]}"
    assert _commons_has_target(target_id)


def test_post_tool_use_bash_command_deposits_to_commons_not_main_graph(cc_ng):
    import cc_ng_host

    cc_ng_host._STATE.cc_ng = cc_ng
    before_nodes = len(cc_ng.graph.nodes)

    data = {
        "tool_name": "Bash",
        "tool_input": {"command": "echo unique_marker_bash_command_12345"},
        "tool_response": "unique_marker_bash_command_12345",
    }
    cc_ng_host._handle_post_tool_use(data)

    assert len(cc_ng.graph.nodes) == before_nodes

    experience = "bash:echo unique_marker_bash_command_12345 result:unique_marker_bash_command_12345"
    target_id = f"cc:experience:{hashlib.sha256(experience.encode()).hexdigest()[:16]}"
    assert _commons_has_target(target_id)
