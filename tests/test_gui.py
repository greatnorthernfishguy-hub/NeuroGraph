"""
Tests for NeuroGraph GUI non-GUI logic.

Tests the infrastructure classes from neurograph_gui.py without requiring
tkinter or a display server.  All filesystem operations use tmp_path,
git and NeuroGraphMemory calls are mocked.

Covers:
  - GUIConfig: load/save, defaults, directory creation
  - FileWatcher: ignore patterns, stability detection
  - GitUpdater: clone, fetch/compare, pull/deploy (mocked git)
  - IngestionManager: ingest + move to ingested/
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the non-GUI classes directly from neurograph_gui.py
# ---------------------------------------------------------------------------

_gui_path = Path(__file__).resolve().parent.parent / "neurograph_gui.py"
_loader = importlib.machinery.SourceFileLoader("neurograph_gui_mod", str(_gui_path))
_spec = importlib.util.spec_from_loader("neurograph_gui_mod", _loader)
_mod = importlib.util.module_from_spec(_spec)
_loader.exec_module(_mod)

GUIConfig = _mod.GUIConfig
FileWatcher = _mod.FileWatcher
GitUpdater = _mod.GitUpdater
IngestionManager = _mod.IngestionManager
SUPPORTED_EXTENSIONS = _mod.SUPPORTED_EXTENSIONS
DEFAULT_CONFIG = _mod.DEFAULT_CONFIG


# ===================================================================
# GUIConfig
# ===================================================================

class TestGUIConfig:
    """Tests for GUIConfig persistence and defaults."""

    def test_default_values(self, tmp_path):
        cfg = GUIConfig(config_path=str(tmp_path / "config.json"))
        assert cfg.get("watcher_enabled") is True
        assert cfg.get("watcher_stability_seconds") == 1.0
        assert "inbox" in cfg.get("inbox_path")

    def test_get_with_custom_default(self, tmp_path):
        cfg = GUIConfig(config_path=str(tmp_path / "config.json"))
        assert cfg.get("nonexistent", "fallback") == "fallback"

    def test_set_and_get(self, tmp_path):
        cfg = GUIConfig(config_path=str(tmp_path / "config.json"))
        cfg.set("my_key", 42)
        assert cfg.get("my_key") == 42

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "config.json"
        cfg = GUIConfig(config_path=str(path))
        cfg.set("hello", "world")
        cfg.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["hello"] == "world"

    def test_load_existing_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"inbox_path": "/custom/inbox"}))
        cfg = GUIConfig(config_path=str(path))
        assert cfg.get("inbox_path") == "/custom/inbox"

    def test_load_corrupt_json_uses_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("not valid json{{{")
        cfg = GUIConfig(config_path=str(path))
        # Should fall back to DEFAULT_CONFIG
        assert cfg.get("watcher_enabled") is True

    def test_ensure_directories(self, tmp_path):
        cfg = GUIConfig(config_path=str(tmp_path / "config.json"))
        inbox = tmp_path / "inbox"
        ingested = tmp_path / "ingested"
        repo = tmp_path / "repo"
        cfg.set("inbox_path", str(inbox))
        cfg.set("ingested_path", str(ingested))
        cfg.set("repo_path", str(repo))
        cfg.set("log_path", str(tmp_path / "logs" / "gui.log"))
        cfg.ensure_directories()
        assert inbox.is_dir()
        assert ingested.is_dir()
        assert repo.is_dir()
        assert (tmp_path / "logs").is_dir()

    def test_save_then_reload(self, tmp_path):
        path = tmp_path / "config.json"
        cfg1 = GUIConfig(config_path=str(path))
        cfg1.set("test_val", "abc")
        cfg1.save()

        cfg2 = GUIConfig(config_path=str(path))
        assert cfg2.get("test_val") == "abc"


# ===================================================================
# FileWatcher
# ===================================================================

class TestFileWatcherIgnore:
    """Tests for FileWatcher.should_ignore() static method."""

    def test_hidden_files_ignored(self):
        assert FileWatcher.should_ignore("/inbox/.hidden.txt") is True
        assert FileWatcher.should_ignore("/inbox/.DS_Store") is True

    def test_temp_files_ignored(self):
        assert FileWatcher.should_ignore("/inbox/file.tmp") is True
        assert FileWatcher.should_ignore("/inbox/download.part") is True
        assert FileWatcher.should_ignore("/inbox/chrome.crdownload") is True
        assert FileWatcher.should_ignore("/inbox/file.swp") is True
        assert FileWatcher.should_ignore("/inbox/backup~") is True

    def test_unsupported_extensions_ignored(self):
        assert FileWatcher.should_ignore("/inbox/photo.jpg") is True
        assert FileWatcher.should_ignore("/inbox/archive.zip") is True
        assert FileWatcher.should_ignore("/inbox/data.csv") is True

    def test_supported_extensions_not_ignored(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert FileWatcher.should_ignore(f"/inbox/file{ext}") is False

    def test_no_extension_not_ignored(self):
        # Files without extension are not ignored (ext == "")
        assert FileWatcher.should_ignore("/inbox/Makefile") is False


class TestFileWatcherLifecycle:
    """Tests for FileWatcher start/stop and stability detection."""

    def test_start_stop(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        results = []
        watcher = FileWatcher(
            inbox_path=str(inbox),
            on_file_ready=lambda p: results.append(p),
            on_error=lambda e: None,
            stability_seconds=0.1,
        )
        watcher.start()
        assert watcher.is_running
        watcher.stop()
        assert not watcher.is_running

    def test_detects_stable_file(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        results = []
        watcher = FileWatcher(
            inbox_path=str(inbox),
            on_file_ready=lambda p: results.append(p),
            on_error=lambda e: None,
            stability_seconds=0.1,
        )
        watcher.start()
        try:
            # Drop a file
            test_file = inbox / "test.md"
            test_file.write_text("# Hello")
            # Wait for stability detection
            time.sleep(1.5)
            assert len(results) >= 1
            assert "test.md" in results[0]
        finally:
            watcher.stop()

    def test_ignores_hidden_file(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        results = []
        watcher = FileWatcher(
            inbox_path=str(inbox),
            on_file_ready=lambda p: results.append(p),
            on_error=lambda e: None,
            stability_seconds=0.1,
        )
        watcher.start()
        try:
            hidden = inbox / ".hidden"
            hidden.write_text("secret")
            time.sleep(1.5)
            assert len(results) == 0
        finally:
            watcher.stop()


# ===================================================================
# GitUpdater
# ===================================================================

class TestGitUpdater:
    """Tests for GitUpdater with mocked subprocess calls."""

    def _make_updater(self, tmp_path, on_status=None, on_complete=None, on_error=None):
        return GitUpdater(
            repo_url="https://github.com/test/repo.git",
            repo_path=str(tmp_path / "repo"),
            skill_dir=str(tmp_path / "skill"),
            workspace_dir=str(tmp_path / "workspace"),
            on_status=on_status or (lambda s: None),
            on_complete=on_complete or (lambda r: None),
            on_error=on_error or (lambda e: None),
        )

    def test_ensure_repo_clones_when_missing(self, tmp_path):
        updater = self._make_updater(tmp_path)
        with mock.patch.object(updater, "_run_git") as mock_git:
            result = updater.ensure_repo()
            assert result is True
            mock_git.assert_called_once()
            args = mock_git.call_args[0]
            assert "clone" in args

    def test_ensure_repo_skips_if_exists(self, tmp_path):
        repo = tmp_path / "repo" / ".git"
        repo.mkdir(parents=True)
        updater = self._make_updater(tmp_path)
        with mock.patch.object(updater, "_run_git") as mock_git:
            result = updater.ensure_repo()
            assert result is True
            mock_git.assert_not_called()

    def test_get_local_commit(self, tmp_path):
        updater = self._make_updater(tmp_path)
        mock_result = mock.Mock()
        mock_result.stdout = "abc1234\n"
        with mock.patch.object(updater, "_run_git", return_value=mock_result):
            assert updater.get_local_commit() == "abc1234"

    def test_check_worker_no_updates(self, tmp_path):
        results = []
        updater = self._make_updater(
            tmp_path, on_complete=lambda r: results.append(r),
        )
        repo_dir = tmp_path / "repo" / ".git"
        repo_dir.mkdir(parents=True)

        def fake_git(*args, **kwargs):
            r = mock.Mock()
            if "rev-parse" in args:
                r.stdout = "abc1234\n"
            elif "rev-list" in args:
                r.stdout = "0\n"
            elif "log" in args:
                r.stdout = ""
            else:
                r.stdout = ""
            return r

        with mock.patch.object(updater, "_run_git", side_effect=fake_git):
            updater._check_worker()

        assert len(results) == 1
        assert results[0]["has_updates"] is False
        assert results[0]["commits_behind"] == 0

    def test_check_worker_has_updates(self, tmp_path):
        results = []
        updater = self._make_updater(
            tmp_path, on_complete=lambda r: results.append(r),
        )
        repo_dir = tmp_path / "repo" / ".git"
        repo_dir.mkdir(parents=True)

        call_count = [0]

        def fake_git(*args, **kwargs):
            r = mock.Mock()
            if "rev-parse" in args:
                call_count[0] += 1
                if call_count[0] == 2:
                    r.stdout = "abc1234\n"  # local
                else:
                    r.stdout = "def5678\n"  # remote
            elif "rev-list" in args:
                r.stdout = "3\n"
            elif "log" in args:
                r.stdout = "def5678 commit 3\nabc5555 commit 2\nabc4444 commit 1\n"
            else:
                r.stdout = ""
            return r

        with mock.patch.object(updater, "_run_git", side_effect=fake_git):
            updater._check_worker()

        assert len(results) == 1
        assert results[0]["has_updates"] is True
        assert results[0]["commits_behind"] == 3

    def test_check_worker_error(self, tmp_path):
        errors = []
        updater = self._make_updater(
            tmp_path, on_error=lambda e: errors.append(e),
        )
        with mock.patch.object(
            updater, "ensure_repo", side_effect=RuntimeError("network down")
        ):
            updater._check_worker()

        assert len(errors) == 1
        assert "network down" in errors[0]


# ===================================================================
# IngestionManager
# ===================================================================

class TestIngestionManager:
    """Tests for IngestionManager file movement logic."""

    def _make_manager(self, tmp_path, on_result=None, on_error=None):
        return IngestionManager(
            workspace_dir=str(tmp_path / "workspace"),
            ingested_path=str(tmp_path / "ingested"),
            on_result=on_result or (lambda r: None),
            on_error=on_error or (lambda e: None),
        )

    def test_move_to_ingested_creates_date_dir(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        test_file = tmp_path / "inbox" / "test.md"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("hello")

        dest = mgr._move_to_ingested(str(test_file))

        assert not test_file.exists()
        assert Path(dest).exists()
        assert Path(dest).name == "test.md"
        # Check dated directory
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in dest

    def test_move_handles_name_collision(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        date_dir = tmp_path / "ingested" / today
        date_dir.mkdir(parents=True)
        (date_dir / "test.md").write_text("first")

        test_file = tmp_path / "test.md"
        test_file.write_text("second")
        dest = mgr._move_to_ingested(str(test_file))

        assert Path(dest).name == "test_1.md"
        assert Path(dest).read_text() == "second"

    def test_move_handles_multiple_collisions(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        date_dir = tmp_path / "ingested" / today
        date_dir.mkdir(parents=True)
        (date_dir / "test.md").write_text("first")
        (date_dir / "test_1.md").write_text("second")

        test_file = tmp_path / "test.md"
        test_file.write_text("third")
        dest = mgr._move_to_ingested(str(test_file))

        assert Path(dest).name == "test_2.md"

    def test_ingest_file_worker_success(self, tmp_path):
        results = []
        errors = []
        mgr = self._make_manager(
            tmp_path,
            on_result=lambda r: results.append(r),
            on_error=lambda e: errors.append(e),
        )

        # Create test file
        test_file = tmp_path / "inbox" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("print('hello')")

        # Mock NeuroGraphMemory
        mock_ng = mock.Mock()
        mock_ng.ingest_file.return_value = {
            "status": "ingested",
            "nodes_created": 3,
            "synapses_created": 5,
        }
        mgr._ng = mock_ng

        mgr._ingest_file_worker(str(test_file))

        assert len(results) == 1
        assert results[0]["nodes_created"] == 3
        assert "moved_to" in results[0]
        assert not test_file.exists()
        assert len(errors) == 0

    def test_ingest_file_worker_error(self, tmp_path):
        results = []
        errors = []
        mgr = self._make_manager(
            tmp_path,
            on_result=lambda r: results.append(r),
            on_error=lambda e: errors.append(e),
        )

        test_file = tmp_path / "inbox" / "bad.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("x")

        mock_ng = mock.Mock()
        mock_ng.ingest_file.return_value = {
            "status": "error",
            "reason": "file too small",
        }
        mgr._ng = mock_ng

        mgr._ingest_file_worker(str(test_file))

        assert len(errors) == 1
        assert "bad.py" in errors[0]
        # File should NOT be moved on error
        assert test_file.exists()
        assert len(results) == 0

    def test_batch_ingest(self, tmp_path):
        results = []
        mgr = self._make_manager(
            tmp_path, on_result=lambda r: results.append(r),
        )

        inbox = tmp_path / "inbox"
        inbox.mkdir(parents=True)
        (inbox / "a.md").write_text("# A")
        (inbox / "b.txt").write_text("B content")

        mock_ng = mock.Mock()
        mock_ng.ingest_file.return_value = {
            "status": "ingested",
            "nodes_created": 1,
            "synapses_created": 1,
        }
        mgr._ng = mock_ng

        mgr._ingest_batch_worker([str(inbox / "a.md"), str(inbox / "b.txt")])

        assert len(results) == 2
        assert not (inbox / "a.md").exists()
        assert not (inbox / "b.txt").exists()


# ===================================================================
# Integration-style tests
# ===================================================================

class TestPipelineIntegration:
    """Test the full watcher -> ingest -> move pipeline (mocked ingestion)."""

    def test_file_drop_pipeline(self, tmp_path):
        """Simulate file drop -> watcher detect -> ingest -> move."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        ingested = tmp_path / "ingested"

        results = []

        # Build an IngestionManager with mocked NeuroGraphMemory
        mgr = IngestionManager(
            workspace_dir=str(tmp_path / "workspace"),
            ingested_path=str(ingested),
            on_result=lambda r: results.append(r),
            on_error=lambda e: None,
        )
        mock_ng = mock.Mock()
        mock_ng.ingest_file.return_value = {
            "status": "ingested",
            "nodes_created": 2,
            "synapses_created": 4,
        }
        mgr._ng = mock_ng

        # Simulate the FileWatcher calling on_file_ready, which in turn
        # triggers ingestion synchronously (in test we call the worker directly)
        test_file = inbox / "notes.md"
        test_file.write_text("# My Notes\nSome content here.")

        # Directly invoke ingestion worker (watcher would call this)
        mgr._ingest_file_worker(str(test_file))

        assert len(results) == 1
        assert results[0]["nodes_created"] == 2
        assert not test_file.exists()
        # Check file was moved to ingested/
        ingested_files = list(ingested.rglob("notes.md"))
        assert len(ingested_files) == 1
