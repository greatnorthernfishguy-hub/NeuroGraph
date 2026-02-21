"""Tests for neurograph_paths — unified path resolution."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from neurograph_paths import (
    get_neurograph_home,
    get_checkpoint_dir,
    get_checkpoint_path,
    write_conf,
    read_conf,
)


# ------------------------------------------------------------------
# get_neurograph_home resolution order
# ------------------------------------------------------------------

class TestResolutionOrder:
    """NEUROGRAPH_HOME env > conf file > NEUROGRAPH_WORKSPACE_DIR > default."""

    def test_env_var_highest_priority(self, tmp_path):
        """NEUROGRAPH_HOME env var wins over everything."""
        env_dir = str(tmp_path / "from_env")
        conf_dir = str(tmp_path / "from_conf")

        # Write conf pointing elsewhere
        conf_file = tmp_path / "test.conf"
        conf_file.write_text(json.dumps({"neurograph_home": conf_dir}))

        with patch.dict(os.environ, {"NEUROGRAPH_HOME": env_dir}, clear=False):
            result = get_neurograph_home()
        assert str(result) == str(Path(env_dir).resolve())

    def test_conf_file_second_priority(self, tmp_path):
        """~/.neurograph.conf wins when no env var set."""
        conf_dir = str(tmp_path / "from_conf")
        conf_file = tmp_path / "test.conf"
        conf_file.write_text(json.dumps({"neurograph_home": conf_dir}))

        env = {k: v for k, v in os.environ.items()
               if k not in ("NEUROGRAPH_HOME", "NEUROGRAPH_WORKSPACE_DIR")}
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", str(conf_file)):
                result = get_neurograph_home()
        assert str(result) == str(Path(conf_dir).resolve())

    def test_legacy_env_third_priority(self, tmp_path):
        """NEUROGRAPH_WORKSPACE_DIR honored when no NEUROGRAPH_HOME or conf."""
        legacy_dir = str(tmp_path / "from_legacy")

        env = {k: v for k, v in os.environ.items() if k != "NEUROGRAPH_HOME"}
        env["NEUROGRAPH_WORKSPACE_DIR"] = legacy_dir
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", str(tmp_path / "nonexistent.conf")):
                result = get_neurograph_home()
        assert str(result) == str(Path(legacy_dir).resolve())

    def test_default_fallback(self, tmp_path):
        """Falls back to ~/.neurograph when nothing is configured."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("NEUROGRAPH_HOME", "NEUROGRAPH_WORKSPACE_DIR")}
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", str(tmp_path / "nonexistent.conf")):
                result = get_neurograph_home()
        assert result == Path("~/.neurograph").expanduser().resolve()

    def test_corrupt_conf_falls_through(self, tmp_path):
        """Corrupt conf file doesn't crash — falls to next priority."""
        conf_file = tmp_path / "bad.conf"
        conf_file.write_text("this is not json {{{")

        env = {k: v for k, v in os.environ.items()
               if k not in ("NEUROGRAPH_HOME", "NEUROGRAPH_WORKSPACE_DIR")}
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", str(conf_file)):
                result = get_neurograph_home()
        # Should fall through to default without crashing
        assert result == Path("~/.neurograph").expanduser().resolve()

    def test_empty_conf_value_falls_through(self, tmp_path):
        """Conf with empty neurograph_home falls to next priority."""
        conf_file = tmp_path / "empty.conf"
        conf_file.write_text(json.dumps({"neurograph_home": ""}))

        env = {k: v for k, v in os.environ.items()
               if k not in ("NEUROGRAPH_HOME", "NEUROGRAPH_WORKSPACE_DIR")}
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", str(conf_file)):
                result = get_neurograph_home()
        assert result == Path("~/.neurograph").expanduser().resolve()


# ------------------------------------------------------------------
# Convenience accessors
# ------------------------------------------------------------------

class TestConvenience:
    def test_checkpoint_dir(self, tmp_path):
        with patch.dict(os.environ, {"NEUROGRAPH_HOME": str(tmp_path)}, clear=False):
            assert get_checkpoint_dir() == (tmp_path / "checkpoints").resolve()

    def test_checkpoint_path(self, tmp_path):
        with patch.dict(os.environ, {"NEUROGRAPH_HOME": str(tmp_path)}, clear=False):
            assert get_checkpoint_path() == (tmp_path / "checkpoints" / "main.msgpack").resolve()


# ------------------------------------------------------------------
# Config file read/write
# ------------------------------------------------------------------

class TestConfFile:
    def test_write_and_read(self, tmp_path):
        conf = str(tmp_path / "test.conf")
        write_conf("/home/josh/neurograph", conf_path=conf)
        assert read_conf(conf_path=conf) == "/home/josh/neurograph"

    def test_read_missing_returns_none(self, tmp_path):
        assert read_conf(conf_path=str(tmp_path / "nope.conf")) is None

    def test_write_creates_parent_dirs(self, tmp_path):
        conf = str(tmp_path / "deep" / "nested" / "test.conf")
        write_conf("/some/path", conf_path=conf)
        assert Path(conf).exists()
        assert read_conf(conf_path=conf) == "/some/path"

    def test_write_expands_tilde(self, tmp_path):
        conf = str(tmp_path / "test.conf")
        write_conf("~/my_neurograph", conf_path=conf)
        result = read_conf(conf_path=conf)
        assert result == str(Path("~/my_neurograph").expanduser())

    def test_conf_is_valid_json(self, tmp_path):
        conf = str(tmp_path / "test.conf")
        write_conf("/data/ng", conf_path=conf)
        data = json.loads(Path(conf).read_text())
        assert "neurograph_home" in data


# ------------------------------------------------------------------
# All components agree
# ------------------------------------------------------------------

class TestConsistency:
    def test_all_resolve_same_path(self, tmp_path):
        """When conf file exists, all accessors return paths under it."""
        conf = str(tmp_path / "test.conf")
        write_conf(str(tmp_path / "unified"), conf_path=conf)

        env = {k: v for k, v in os.environ.items()
               if k not in ("NEUROGRAPH_HOME", "NEUROGRAPH_WORKSPACE_DIR")}
        with patch.dict(os.environ, env, clear=True):
            with patch("neurograph_paths._CONF_FILE", conf):
                home = get_neurograph_home()
                ckpt_dir = get_checkpoint_dir()
                ckpt_path = get_checkpoint_path()

        assert ckpt_dir == home / "checkpoints"
        assert ckpt_path == home / "checkpoints" / "main.msgpack"
