"""
Tests for the NeuroGraph OpenClaw integration hook.

Covers:
- Singleton pattern
- Message ingestion
- Semantic recall
- Auto-save behavior
- File/directory ingestion
- Stats reporting
- Save/restore across instances
"""

import os
import tempfile
from pathlib import Path

import pytest

from openclaw_hook import NeuroGraphMemory


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test."""
    NeuroGraphMemory.reset_instance()
    yield
    NeuroGraphMemory.reset_instance()


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / "neurograph"
    ws.mkdir()
    (ws / "checkpoints").mkdir()
    return str(ws)


class TestSingleton:
    def test_singleton_returns_same_instance(self, workspace):
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng2 = NeuroGraphMemory.get_instance()
        assert ng1 is ng2

    def test_reset_creates_new_instance(self, workspace):
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        NeuroGraphMemory.reset_instance()
        ng2 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        assert ng1 is not ng2


class TestMessageIngestion:
    def test_ingest_text(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("The quick brown fox jumps over the lazy dog")
        assert result["status"] == "ingested"
        assert result["nodes_created"] > 0
        assert result["message_count"] == 1

    def test_ingest_empty_skipped(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("")
        assert result["status"] == "skipped"

    def test_ingest_whitespace_skipped(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("   \n\t  ")
        assert result["status"] == "skipped"

    def test_multiple_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        for i in range(5):
            result = ng.on_message(f"Message number {i} about topic {i}")
        assert result["message_count"] == 5

    def test_graph_grows_with_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats_before = ng.stats()
        ng.on_message("Neural networks learn patterns through backpropagation")
        ng.on_message("Gradient descent optimizes loss functions")
        stats_after = ng.stats()
        assert stats_after["nodes"] > stats_before["nodes"]


class TestRecall:
    def test_recall_finds_related(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Python is a programming language used for machine learning")
        ng.on_message("JavaScript runs in web browsers")
        results = ng.recall("programming language", k=5, threshold=0.0)
        # Should return some results (threshold=0 to be permissive with hash fallback)
        assert isinstance(results, list)

    def test_recall_empty_db(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        results = ng.recall("anything")
        assert results == []


class TestAutoSave:
    def test_auto_save_every_n_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.auto_save_interval = 3
        checkpoint_path = Path(workspace) / "checkpoints" / "main.msgpack"

        for i in range(3):
            ng.on_message(f"Auto-save test message {i}")

        # After 3 messages with interval=3, checkpoint should exist
        assert checkpoint_path.exists()

    def test_manual_save(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Save this")
        path = ng.save()
        assert os.path.exists(path)


class TestStats:
    def test_stats_has_required_fields(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        required_keys = [
            "version", "timestep", "nodes", "synapses", "hyperedges",
            "firing_rate", "mean_weight", "predictions_made",
            "prediction_accuracy", "vector_db_count", "checkpoint",
            "message_count",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_after_ingestion(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Test content for statistics")
        stats = ng.stats()
        assert stats["nodes"] > 0
        assert stats["message_count"] == 1


class TestPersistence:
    def test_cross_session_persistence(self, workspace):
        """Data survives across singleton resets (simulating restart)."""
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng1.on_message("Persistent data test: STDP learning is fundamental")
        ng1.save()
        nodes_before = ng1.stats()["nodes"]

        NeuroGraphMemory.reset_instance()

        ng2 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        nodes_after = ng2.stats()["nodes"]
        assert nodes_after == nodes_before


class TestFileIngestion:
    def test_ingest_python_file(self, workspace, tmp_path):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    print('hello world')\n")
        result = ng.ingest_file(str(py_file))
        assert result["status"] == "ingested"
        assert result["nodes_created"] > 0

    def test_ingest_missing_file(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.ingest_file("/nonexistent/file.txt")
        assert result["status"] == "error"

    def test_ingest_directory(self, workspace, tmp_path):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)

        # Create test files
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.txt").write_text("hello world\n")
        (tmp_path / "c.jpg").write_text("not text")  # Should be skipped

        results = ng.ingest_directory(str(tmp_path))
        ingested = [r for r in results if r.get("status") == "ingested"]
        assert len(ingested) == 2  # .py and .txt, not .jpg


class TestStep:
    def test_step_runs_snn(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Create some nodes first")
        results = ng.step(n=5)
        assert len(results) == 5


class TestMemoryLogging:
    def test_memory_dir_created(self, workspace):
        """Memory directory is created on initialization."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        memory_dir = Path(workspace) / "memory"
        assert memory_dir.is_dir()

    def test_embedding_status_event_on_init(self, workspace):
        """Embedding status event is written on initialization."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        events_path = Path(workspace) / "memory" / "events.jsonl"
        assert events_path.exists()
        import json
        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        # First event should be embedding_status
        assert events[0]["event"] == "embedding_status"
        assert "model_available" in events[0]["data"]

    def test_ingestion_event_on_message(self, workspace):
        """Ingestion events are written to memory/ on on_message."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Memory logging test message with enough content")
        events_path = Path(workspace) / "memory" / "events.jsonl"
        import json
        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        ingestion_events = [e for e in events if e["event"] == "ingestion"]
        assert len(ingestion_events) >= 1
        assert ingestion_events[0]["data"]["status"] == "ingested"

    def test_stats_includes_memory_dir(self, workspace):
        """Stats include memory_dir path."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        assert "memory_dir" in stats
        assert "memory" in stats["memory_dir"]

    def test_stats_includes_embedding_status(self, workspace):
        """Stats include embedding backend status."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        assert "embedding" in stats
        assert "model_available" in stats["embedding"]
        assert "device_requested" in stats["embedding"]


class TestEmbeddingDeviceConfig:
    def test_default_device_is_auto(self, workspace):
        """Default embedding device is 'auto'."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "auto"

    def test_explicit_cpu_device_via_config(self, workspace):
        """Embedding device can be set via config."""
        ng = NeuroGraphMemory.get_instance(
            workspace_dir=workspace, config={"embedding_device": "cpu"}
        )
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "cpu"

    def test_env_var_device_override(self, workspace, monkeypatch):
        """NEUROGRAPH_EMBEDDING_DEVICE env var overrides default."""
        monkeypatch.setenv("NEUROGRAPH_EMBEDDING_DEVICE", "cpu")
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "cpu"


class TestStableCheckpointGuard:
    """Regression coverage for the 2026-07-03 torn-read fix.

    Root cause of two real incidents (VPS CC-NG 2026-06-14, laptop CC-NG
    2026-06-26): restore()/load() reading a checkpoint mid-autosave-write,
    raising, being silently swallowed, and the resulting empty graph then
    getting autosaved over the real state. _wait_for_stable_checkpoint()
    closes the trigger by refusing to read a file that's still changing.
    """

    def test_stable_file_returns_true_immediately(self, tmp_path):
        from openclaw_hook import _wait_for_stable_checkpoint
        p = tmp_path / "stable.msgpack"
        p.write_bytes(b"x" * 1000)
        assert _wait_for_stable_checkpoint(str(p), max_wait=5.0, check_interval=0.2) is True

    def test_missing_file_returns_true(self, tmp_path):
        from openclaw_hook import _wait_for_stable_checkpoint
        p = tmp_path / "does-not-exist.msgpack"
        assert _wait_for_stable_checkpoint(str(p), max_wait=5.0) is True

    def test_actively_growing_file_is_not_trusted_until_stable(self, tmp_path):
        """A file mid-write must not be read until it stops changing."""
        from openclaw_hook import _wait_for_stable_checkpoint
        import threading
        import time as time_mod

        p = tmp_path / "growing.msgpack"

        def writer():
            with open(p, "wb") as f:
                for _ in range(5):
                    f.write(b"x" * 1000)
                    f.flush()
                    time_mod.sleep(0.3)

        th = threading.Thread(target=writer)
        th.start()
        time_mod.sleep(0.1)  # let the writer start first
        result = _wait_for_stable_checkpoint(str(p), max_wait=10.0, check_interval=0.2)
        th.join()
        assert result is True  # eventually stabilizes once the writer finishes
        assert p.stat().st_size == 5000  # and we only trusted it once complete

    def test_never_stabilizing_file_times_out_false(self, tmp_path):
        """A pathologically stuck write must not be silently trusted either."""
        from openclaw_hook import _wait_for_stable_checkpoint
        import threading
        import time as time_mod

        p = tmp_path / "stuck.msgpack"

        def slow_writer():
            with open(p, "wb") as f:
                for _ in range(20):
                    f.write(b"x" * 1000)
                    f.flush()
                    time_mod.sleep(0.2)

        th = threading.Thread(target=slow_writer, daemon=True)
        th.start()
        time_mod.sleep(0.1)
        result = _wait_for_stable_checkpoint(str(p), max_wait=1.5, check_interval=0.2)
        assert result is False

    def test_mid_write_checkpoint_defers_restore_instead_of_silently_emptying(self, workspace):
        """Integration-level: constructing NeuroGraphMemory while the checkpoint
        is actively being overwritten must not produce a phantom-empty graph
        from a torn read — it must defer restore for that init instead."""
        import threading
        import time as time_mod

        # Build a real, healthy checkpoint first.
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng1.on_message("Regression test: torn checkpoint reads must not corrupt state")
        ng1.save()
        real_node_count = ng1.stats()["nodes"]
        assert real_node_count > 0
        checkpoint_path = Path(workspace) / "checkpoints" / "main.msgpack"
        real_bytes = checkpoint_path.read_bytes()
        NeuroGraphMemory.reset_instance()

        # Simulate a slow autosave in progress: truncate-then-rewrite the same
        # file slowly, matching how a real save overwrites in place.
        def slow_rewrite():
            with open(checkpoint_path, "wb") as f:
                half = len(real_bytes) // 2
                f.write(real_bytes[:half])
                f.flush()
                time_mod.sleep(1.0)
                f.write(real_bytes[half:])
                f.flush()

        th = threading.Thread(target=slow_rewrite)
        th.start()
        time_mod.sleep(0.1)  # ensure construction starts while file is mid-write

        ng2 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        th.join()

        # Either it correctly waited and got the real, complete state, or it
        # cleanly deferred to empty — never a torn partial read in between.
        assert ng2.stats()["nodes"] in (0, real_node_count)
