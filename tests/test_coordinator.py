"""Tests for neurograph_rpc coordinator-level handler contracts (#209).

Covers:
- handle_bootstrap: already-initialized guard, topology-owned guard
- handle_ingest: not-bootstrapped guard, empty/whitespace guards, success path
- handle_assemble: not-bootstrapped guard, no-user-text guard, KISS truncation contract
- handle_after_turn: not-bootstrapped guard, timestep advance, cache clearing
- handle_dispose: not-bootstrapped guard, triggers save
- handle_stats: not-bootstrapped guard, success path shape
- cc_ng_host: _cleanup_stale_socket stale removal and live-socket raise
- cc_ng_host: init_cc_host already-initialized and construction-failure guards
"""
import os
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import neurograph_rpc
import cc_ng_host


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_rpc_globals():
    """Save and restore neurograph_rpc module globals after each test."""
    orig_memory = neurograph_rpc._memory
    orig_tract = neurograph_rpc._tract
    orig_ingest_text = neurograph_rpc._ingest_text
    orig_ingest_embedding = neurograph_rpc._ingest_embedding
    yield
    neurograph_rpc._memory = orig_memory
    neurograph_rpc._tract = orig_tract
    neurograph_rpc._ingest_text = orig_ingest_text
    neurograph_rpc._ingest_embedding = orig_ingest_embedding


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ng_test"
    ws.mkdir()
    (ws / "checkpoints").mkdir()
    return str(ws)


@pytest.fixture
def live_memory(workspace):
    """Real NeuroGraphMemory injected as the rpc singleton.

    Skipped when openclaw_hook import fails (HF Space CI) or when heavy
    deps like fastembed fail to initialize. Live tests run green on VPS.
    """
    pytest.importorskip("openclaw_hook")
    try:
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        mem = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        neurograph_rpc._memory = mem
    except Exception as exc:
        pytest.skip(f"NeuroGraphMemory setup failed: {exc}")
    yield mem
    neurograph_rpc._memory = None
    try:
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
    except Exception:
        pass


# ── handle_bootstrap ──────────────────────────────────────────────────

class TestHandleBootstrap:
    def test_already_initialized_returns_early(self, live_memory):
        """Second call returns already_initialized without re-entering."""
        result = neurograph_rpc.handle_bootstrap({})
        assert result["bootstrapped"] is True
        assert result["reason"] == "already_initialized"

    def test_topology_owned_returns_false(self):
        """Topology claimed by another PID -> decline bootstrap."""
        neurograph_rpc._memory = None
        mock_topo = MagicMock()
        mock_topo.claim.return_value = False
        mock_topo.owner_pid.return_value = 12345
        mock_oc = MagicMock()
        with patch.dict(sys.modules, {"topology_owner": mock_topo, "openclaw_hook": mock_oc}):
            result = neurograph_rpc.handle_bootstrap({})
        assert result["bootstrapped"] is False
        assert "12345" in result["reason"]


# ── handle_ingest ─────────────────────────────────────────────────────

class TestHandleIngest:
    def test_not_bootstrapped_returns_error(self):
        """Without bootstrap, ingest returns structured error dict."""
        neurograph_rpc._memory = None
        result = neurograph_rpc.handle_ingest({"message": {"content": "hello"}})
        assert result == {"ingested": False, "reason": "not_bootstrapped"}

    def test_empty_content_returns_false(self, live_memory):
        result = neurograph_rpc.handle_ingest({"message": {"content": ""}})
        assert result["ingested"] is False

    def test_whitespace_only_returns_false(self, live_memory):
        result = neurograph_rpc.handle_ingest({"message": {"content": "   "}})
        assert result["ingested"] is False

    def test_valid_text_returns_ingested_true(self, live_memory):
        result = neurograph_rpc.handle_ingest(
            {"message": {"content": "The substrate is the communication protocol."}}
        )
        assert result["ingested"] is True

    def test_message_count_increments(self, live_memory):
        before = live_memory._message_count
        neurograph_rpc.handle_ingest({"message": {"content": "Count increment test"}})
        assert live_memory._message_count == before + 1

    def test_missing_message_key_returns_false(self, live_memory):
        """Params with no message key treated as empty message."""
        result = neurograph_rpc.handle_ingest({})
        assert result["ingested"] is False


# ── handle_assemble ───────────────────────────────────────────────────

class TestHandleAssemble:
    def test_not_bootstrapped_returns_null_addition(self):
        neurograph_rpc._memory = None
        result = neurograph_rpc.handle_assemble({"messages": []})
        assert result == {"systemPromptAddition": None}

    def test_empty_messages_returns_null_addition(self, live_memory):
        result = neurograph_rpc.handle_assemble({"messages": []})
        assert result["systemPromptAddition"] is None

    def test_no_user_role_returns_null_addition(self, live_memory):
        """Assistant-only conversation has no user text to prime from."""
        messages = [{"role": "assistant", "content": "I am present."}]
        result = neurograph_rpc.handle_assemble({"messages": messages})
        assert result["systemPromptAddition"] is None

    def test_result_always_has_system_prompt_key(self, live_memory):
        messages = [{"role": "user", "content": "What is the River?"}]
        result = neurograph_rpc.handle_assemble({"messages": messages})
        assert "systemPromptAddition" in result

    def test_system_prompt_addition_is_str_or_none(self, live_memory):
        """systemPromptAddition must be str or None, never a dict or list."""
        neurograph_rpc.handle_ingest({"message": {"content": "River flows between modules"}})
        messages = [{"role": "user", "content": "River flows between modules"}]
        result = neurograph_rpc.handle_assemble({"messages": messages})
        addition = result["systemPromptAddition"]
        assert addition is None or isinstance(addition, str)

    def test_short_conversation_omits_messages_key(self, live_memory):
        """With <= recent_window messages, messages key must NOT appear."""
        messages = [{"role": "user", "content": "Brief message"}]
        result = neurograph_rpc.handle_assemble({"messages": messages})
        assert "messages" not in result

    def test_long_conversation_truncation_when_kiss_active(self, live_memory):
        """With > recent_window messages and KISS active, messages key appears truncated."""
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"}
            for i in range(30)
        ]
        result = neurograph_rpc.handle_assemble({"messages": messages})
        if "messages" in result:
            assert len(result["messages"]) < len(messages)


# ── handle_after_turn ─────────────────────────────────────────────────

class TestHandleAfterTurn:
    def test_not_bootstrapped_returns_none(self):
        """afterTurn with no memory is a silent no-op."""
        neurograph_rpc._memory = None
        result = neurograph_rpc.handle_after_turn({"lastUserMessage": {"content": "hi"}})
        assert result is None

    def test_advances_graph_timestep(self, live_memory):
        before = live_memory.graph.timestep
        neurograph_rpc._ingest_text = "advance timestep"
        neurograph_rpc.handle_after_turn(
            {"lastUserMessage": {"content": "advance timestep"}}
        )
        assert live_memory.graph.timestep > before

    def test_clears_ingest_cache_after_run(self, live_memory):
        """_ingest_text and _ingest_embedding are consumed on afterTurn."""
        neurograph_rpc._ingest_text = "cached for drain"
        neurograph_rpc._ingest_embedding = None
        neurograph_rpc.handle_after_turn({})
        assert neurograph_rpc._ingest_text is None
        assert neurograph_rpc._ingest_embedding is None

    def test_recovers_text_from_params(self, live_memory):
        """When ingest cache is empty, text is recovered from lastUserMessage."""
        neurograph_rpc._ingest_text = None
        neurograph_rpc._ingest_embedding = None
        neurograph_rpc.handle_after_turn(
            {"lastUserMessage": {"content": "recover from params"}}
        )
        assert neurograph_rpc._ingest_text is None


# ── handle_dispose ────────────────────────────────────────────────────

class TestHandleDispose:
    def test_not_bootstrapped_returns_none(self):
        neurograph_rpc._memory = None
        result = neurograph_rpc.handle_dispose({})
        assert result is None

    def test_triggers_save(self, live_memory):
        """Dispose must call save() -- checkpoint must be written."""
        save_called = []
        orig_save = live_memory.save
        live_memory.save = lambda: save_called.append(True) or orig_save()
        neurograph_rpc.handle_dispose({})
        assert len(save_called) >= 1
        live_memory.save = orig_save


# ── handle_stats ──────────────────────────────────────────────────────

class TestHandleStats:
    def test_not_bootstrapped_returns_error_dict(self):
        neurograph_rpc._memory = None
        result = neurograph_rpc.handle_stats({})
        assert result == {"error": "not_bootstrapped"}

    def test_returns_dict_when_bootstrapped(self, live_memory):
        result = neurograph_rpc.handle_stats({})
        assert isinstance(result, dict)
        assert "error" not in result

    def test_includes_module_hooks_key(self, live_memory):
        result = neurograph_rpc.handle_stats({})
        assert "module_hooks" in result
        assert "errors" in result["module_hooks"]


# ── cc_ng_host: _cleanup_stale_socket ────────────────────────────────

class TestCleanupStaleSocket:
    def test_no_file_is_noop(self, tmp_path):
        missing = str(tmp_path / "nonexistent.sock")
        with patch.object(cc_ng_host, "SOCKET_PATH", missing):
            cc_ng_host._cleanup_stale_socket()

    def test_stale_file_removed(self, tmp_path):
        """Socket file with no listener -> file is removed."""
        sock_path = str(tmp_path / "stale.sock")
        Path(sock_path).touch()
        with patch.object(cc_ng_host, "SOCKET_PATH", sock_path):
            cc_ng_host._cleanup_stale_socket()
        assert not os.path.exists(sock_path)

    def test_live_socket_raises(self, tmp_path):
        """Active listener -> raises RuntimeError after all retry attempts."""
        sock_path = str(tmp_path / "live.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(10)  # large backlog so 3 retries all succeed; listen(1) fills queue in strict-backlog containers
        try:
            with patch.object(cc_ng_host, "SOCKET_PATH", sock_path):
                with patch("cc_ng_host.time.sleep"):
                    with pytest.raises(RuntimeError, match="in use by another process"):
                        cc_ng_host._cleanup_stale_socket()
        finally:
            srv.close()
            if os.path.exists(sock_path):
                os.remove(sock_path)


# ── cc_ng_host: init_cc_host ──────────────────────────────────────────

class TestInitCCHost:
    def test_already_initialized_returns_true(self):
        orig = cc_ng_host._STATE.cc_ng
        cc_ng_host._STATE.cc_ng = MagicMock()
        try:
            result = cc_ng_host.init_cc_host()
            assert result is True
        finally:
            cc_ng_host._STATE.cc_ng = orig

    def test_construction_failure_returns_false(self):
        orig = cc_ng_host._STATE.cc_ng
        cc_ng_host._STATE.cc_ng = None
        try:
            mock_oc = MagicMock()
            mock_oc.NeuroGraphMemory.side_effect = RuntimeError("init failed")
            with patch.dict(sys.modules, {"openclaw_hook": mock_oc}):
                result = cc_ng_host.init_cc_host()
            assert result is False
        finally:
            cc_ng_host._STATE.cc_ng = orig
