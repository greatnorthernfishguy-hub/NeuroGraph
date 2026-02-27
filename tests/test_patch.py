"""
Tests for neurograph-patch selective code patching tool.

Covers:
- Diff detection (CHANGED, IDENTICAL, MISSING)
- Backup creation and naming
- Patch execution (copy, chmod)
- Validation (pass, fail with rollback)
- Rollback from manifest
- Patch manifest persistence
- Dry-run and --list modes
- Schema check integration
"""

import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the patch module by loading the script file directly
import importlib.util
import types

_patch_script = Path(__file__).resolve().parent.parent / "neurograph-patch"
_loader = importlib.machinery.SourceFileLoader("neurograph_patch", str(_patch_script))
_spec = importlib.util.spec_from_loader("neurograph_patch", _loader)
ngpatch = types.ModuleType(_spec.name)
ngpatch.__spec__ = _spec
_loader.exec_module(ngpatch)


@pytest.fixture
def mock_deploy(tmp_path):
    """Set up a mock deployment environment.

    Creates:
        repo/     — fake repo with source files
        skill/    — fake skill dir with deployed files
        workspace/ — workspace with checkpoints/ and memory/
        bin/      — fake bin dir
    """
    repo = tmp_path / "repo"
    skill = tmp_path / "skill"
    workspace = tmp_path / "workspace"
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"

    for d in [repo, skill, skill / "scripts", workspace, workspace / "checkpoints",
              workspace / "memory", bin_dir, home]:
        d.mkdir(parents=True, exist_ok=True)

    # Create source files in repo (must cover all MANIFEST entries)
    (repo / "neuro_foundation.py").write_text("# neuro v2\nprint('updated')\n")
    (repo / "universal_ingestor.py").write_text("# ingestor v2\nclass EmbeddingEngine: pass\n")
    (repo / "openclaw_hook.py").write_text("# hook v2\nclass NeuroGraphMemory: pass\n")
    (repo / "neurograph_migrate.py").write_text("# migrate v2\n")
    (repo / "SKILL.md").write_text("# SKILL v2\n")
    (repo / "feed-syl").write_text("#!/usr/bin/env python3\n# feed-syl v2\n")
    (repo / "neurograph_gui.py").write_text("# gui v2\n")
    (repo / "neurograph.desktop").write_text("[Desktop Entry]\nName=NeuroGraph\n")
    (repo / "ng_lite.py").write_text("# ng_lite v2\n")
    (repo / "ng_bridge.py").write_text("# ng_bridge v2\n")
    (repo / "ng_peer_bridge.py").write_text("# ng_peer_bridge v2\n")
    (repo / "et_modules").mkdir(parents=True, exist_ok=True)
    (repo / "et_modules" / "__init__.py").write_text("# et_modules init\n")
    (repo / "et_modules" / "manager.py").write_text("# et_modules manager v2\n")
    (repo / "et_module.json").write_text('{"module_id": "neurograph"}\n')

    # Create deployed files (some identical, some different)
    (skill / "neuro_foundation.py").write_text("# neuro v1\nprint('old')\n")
    (skill / "universal_ingestor.py").write_text("# ingestor v2\nclass EmbeddingEngine: pass\n")  # identical
    (skill / "openclaw_hook.py").write_text("# hook v1\n")
    (skill / "scripts" / "openclaw_hook.py").write_text("# hook v1\n")
    (skill / "neurograph_migrate.py").write_text("# migrate v2\n")  # identical
    (skill / "SKILL.md").write_text("# SKILL v2\n")  # identical
    (bin_dir / "feed-syl").write_text("#!/usr/bin/env python3\n# feed-syl v1\n")
    (home / "feed-syl").write_text("#!/usr/bin/env python3\n# feed-syl v1\n")

    paths = {
        "repo": repo,
        "skill": skill,
        "workspace": workspace,
        "bin": bin_dir,
        "home": home,
    }
    return paths


# ---------------------------------------------------------------------------
# Diff Detection
# ---------------------------------------------------------------------------

class TestDiffDetection:
    def test_detects_changed_file(self, mock_deploy):
        changes = ngpatch.detect_changes(mock_deploy)
        by_name = {c["name"]: c for c in changes}
        assert by_name["neuro_foundation.py"]["status"] == "CHANGED"

    def test_detects_identical_file(self, mock_deploy):
        changes = ngpatch.detect_changes(mock_deploy)
        by_name = {c["name"]: c for c in changes}
        assert by_name["universal_ingestor.py"]["status"] == "IDENTICAL"
        assert by_name["neurograph_migrate.py"]["status"] == "IDENTICAL"
        assert by_name["SKILL.md"]["status"] == "IDENTICAL"

    def test_detects_missing_file(self, mock_deploy):
        # Remove a deployed file
        (mock_deploy["skill"] / "neuro_foundation.py").unlink()
        changes = ngpatch.detect_changes(mock_deploy)
        by_name = {c["name"]: c for c in changes}
        assert by_name["neuro_foundation.py"]["status"] == "MISSING"

    def test_multi_target_file(self, mock_deploy):
        changes = ngpatch.detect_changes(mock_deploy)
        by_name = {c["name"]: c for c in changes}
        hook = by_name["openclaw_hook.py"]
        assert len(hook["targets"]) == 2
        assert hook["status"] == "CHANGED"

    def test_filter_files(self, mock_deploy):
        changes = ngpatch.detect_changes(mock_deploy, filter_files=["neuro_foundation.py"])
        assert len(changes) == 1
        assert changes[0]["name"] == "neuro_foundation.py"


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_created(self, mock_deploy):
        target = mock_deploy["skill"] / "neuro_foundation.py"
        original_content = target.read_text()
        ts = int(time.time())
        bp = ngpatch.backup_file(target, ts)
        assert bp is not None
        assert Path(bp).exists()
        assert Path(bp).read_text() == original_content
        assert f".backup-{ts}" in bp

    def test_backup_returns_none_for_missing(self, tmp_path):
        bp = ngpatch.backup_file(tmp_path / "nonexistent.py", 12345)
        assert bp is None

    def test_multiple_backups_coexist(self, mock_deploy):
        target = mock_deploy["skill"] / "neuro_foundation.py"
        bp1 = ngpatch.backup_file(target, 1000000)
        bp2 = ngpatch.backup_file(target, 1000001)
        assert bp1 != bp2
        assert Path(bp1).exists()
        assert Path(bp2).exists()


# ---------------------------------------------------------------------------
# Patch Execution
# ---------------------------------------------------------------------------

class TestPatchExecution:
    def test_changed_file_is_overwritten(self, mock_deploy):
        result = ngpatch.run_patch(mock_deploy, no_backup=True)
        # neuro_foundation.py was CHANGED, should now match repo
        deployed = (mock_deploy["skill"] / "neuro_foundation.py").read_text()
        source = (mock_deploy["repo"] / "neuro_foundation.py").read_text()
        assert deployed == source

    def test_identical_file_not_touched(self, mock_deploy):
        target = mock_deploy["skill"] / "universal_ingestor.py"
        mtime_before = target.stat().st_mtime
        # Small sleep to ensure mtime would differ if touched
        time.sleep(0.05)
        ngpatch.run_patch(mock_deploy, no_backup=True)
        mtime_after = target.stat().st_mtime
        assert mtime_before == mtime_after

    def test_multi_target_both_updated(self, mock_deploy):
        ngpatch.run_patch(mock_deploy, no_backup=True)
        source = (mock_deploy["repo"] / "openclaw_hook.py").read_text()
        assert (mock_deploy["skill"] / "openclaw_hook.py").read_text() == source
        assert (mock_deploy["skill"] / "scripts" / "openclaw_hook.py").read_text() == source

    def test_feed_syl_executable(self, mock_deploy):
        ngpatch.run_patch(mock_deploy, no_backup=True)
        for loc in [mock_deploy["bin"] / "feed-syl", mock_deploy["home"] / "feed-syl"]:
            assert loc.stat().st_mode & stat.S_IXUSR

    def test_filter_patches_only_specified(self, mock_deploy):
        ngpatch.run_patch(mock_deploy, filter_files=["neuro_foundation.py"], no_backup=True)
        # neuro_foundation patched
        assert (mock_deploy["skill"] / "neuro_foundation.py").read_text() == \
               (mock_deploy["repo"] / "neuro_foundation.py").read_text()
        # openclaw_hook NOT patched (still old)
        assert (mock_deploy["skill"] / "openclaw_hook.py").read_text() == "# hook v1\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_good_skill_dir(self):
        """Validation passes against the actual repo (it has valid Python)."""
        repo = Path(__file__).resolve().parent.parent
        valid, msg = ngpatch.validate_patch(repo)
        assert valid, f"Validation failed: {msg}"

    def test_validate_bad_skill_dir(self, tmp_path):
        """Validation fails with broken Python."""
        (tmp_path / "neuro_foundation.py").write_text("raise SyntaxError('broken')")
        (tmp_path / "universal_ingestor.py").write_text("pass")
        (tmp_path / "openclaw_hook.py").write_text("pass")
        valid, msg = ngpatch.validate_patch(tmp_path)
        assert not valid


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_auto_rollback_on_validation_failure(self, mock_deploy):
        """If patched code is invalid, files are auto-restored from backup."""
        # Write invalid Python to repo source
        (mock_deploy["repo"] / "neuro_foundation.py").write_text("def broken(:\n")
        original = (mock_deploy["skill"] / "neuro_foundation.py").read_text()

        result = ngpatch.run_patch(mock_deploy)
        assert result == 1  # should fail

        # File should be restored to original
        assert (mock_deploy["skill"] / "neuro_foundation.py").read_text() == original

    def test_manual_rollback_from_manifest(self, mock_deploy):
        """--rollback restores from the most recent patch manifest."""
        # First, do a successful patch (mock validation to pass)
        with mock.patch.object(ngpatch, "validate_patch", return_value=(True, "OK")):
            ngpatch.run_patch(mock_deploy)

        # Verify patch was applied
        assert (mock_deploy["skill"] / "neuro_foundation.py").read_text() == \
               (mock_deploy["repo"] / "neuro_foundation.py").read_text()

        # Now rollback
        with mock.patch.object(ngpatch, "validate_patch", return_value=(True, "OK")):
            result = ngpatch.run_rollback(mock_deploy)
        assert result == 0

        # File should be back to v1
        assert (mock_deploy["skill"] / "neuro_foundation.py").read_text() == "# neuro v1\nprint('old')\n"

    def test_rollback_no_manifest(self, mock_deploy):
        result = ngpatch.run_rollback(mock_deploy)
        assert result == 1


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_written(self, mock_deploy):
        with mock.patch.object(ngpatch, "validate_patch", return_value=(True, "OK")):
            ngpatch.run_patch(mock_deploy)
        history = mock_deploy["workspace"] / "patch_history"
        manifests = list(history.glob("patch-*.json"))
        assert len(manifests) == 1
        data = json.loads(manifests[0].read_text())
        assert "files_patched" in data
        assert "timestamp" in data

    def test_manifest_contains_hashes(self, mock_deploy):
        with mock.patch.object(ngpatch, "validate_patch", return_value=(True, "OK")):
            ngpatch.run_patch(mock_deploy)
        history = mock_deploy["workspace"] / "patch_history"
        manifest = json.loads(list(history.glob("patch-*.json"))[0].read_text())
        for entry in manifest["files_patched"]:
            assert "source_hash" in entry
            assert "previous_hash" in entry


# ---------------------------------------------------------------------------
# Dry Run and List
# ---------------------------------------------------------------------------

class TestDryRunAndList:
    def test_dry_run_no_changes(self, mock_deploy):
        original = (mock_deploy["skill"] / "neuro_foundation.py").read_text()
        ngpatch.run_patch(mock_deploy, dry_run=True)
        assert (mock_deploy["skill"] / "neuro_foundation.py").read_text() == original

    def test_list_returns_zero(self, mock_deploy):
        result = ngpatch.run_list(mock_deploy)
        assert result == 0

    def test_nothing_to_patch(self, mock_deploy):
        """When all files are identical, run_patch reports nothing to do."""
        # Make all deployed files match repo
        for name, targets, _ in ngpatch.MANIFEST:
            source = mock_deploy["repo"] / name
            for t in targets:
                target = ngpatch.expand_target(t, mock_deploy)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(target))

        result = ngpatch.run_patch(mock_deploy)
        assert result == 0


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = ngpatch.file_hash(f)
        h2 = ngpatch.file_hash(f)
        assert h1 == h2

    def test_hash_none_for_missing(self, tmp_path):
        assert ngpatch.file_hash(tmp_path / "nonexistent") is None

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert ngpatch.file_hash(f1) != ngpatch.file_hash(f2)
