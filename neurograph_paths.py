"""Unified path resolution for all NeuroGraph components.

Every NeuroGraph tool (openclaw_hook, feed-syl, neurograph CLI, deploy.sh)
MUST use this module to determine where data lives on disk. This prevents
components from silently creating duplicate installs in different locations.

Resolution order (first match wins):
    1. NEUROGRAPH_HOME environment variable
    2. ~/.neurograph.conf JSON config file  {"neurograph_home": "/path/..."}
    3. NEUROGRAPH_WORKSPACE_DIR env var (backward compat)
    4. Default: ~/.neurograph
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


_CONF_FILE = "~/.neurograph.conf"
_DEFAULT_HOME = "~/.neurograph"


def get_neurograph_home() -> Path:
    """Return the canonical NeuroGraph data directory.

    All checkpoints, state, and runtime data live under this directory.
    Every component MUST call this instead of hardcoding a path.
    """
    # 1. Explicit env var (highest priority — session/script overrides)
    env_home = os.environ.get("NEUROGRAPH_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    # 2. Persistent config file (survives across sessions)
    conf_path = Path(_CONF_FILE).expanduser()
    if conf_path.is_file():
        try:
            data = json.loads(conf_path.read_text())
            home = data.get("neurograph_home", "").strip()
            if home:
                return Path(home).expanduser().resolve()
        except (json.JSONDecodeError, OSError):
            pass  # Corrupt or unreadable — fall through

    # 3. Legacy env var (backward compat with pre-unification installs)
    legacy = os.environ.get("NEUROGRAPH_WORKSPACE_DIR")
    if legacy:
        return Path(legacy).expanduser().resolve()

    # 4. Default
    return Path(_DEFAULT_HOME).expanduser().resolve()


def get_checkpoint_dir() -> Path:
    """Return the checkpoints subdirectory."""
    return get_neurograph_home() / "checkpoints"


def get_checkpoint_path() -> Path:
    """Return the default checkpoint file path."""
    return get_checkpoint_dir() / "main.msgpack"


def write_conf(neurograph_home: str, conf_path: Optional[str] = None) -> Path:
    """Write the canonical config file so all components agree on the path.

    Args:
        neurograph_home: Absolute or expandable path to data directory.
        conf_path: Override config file location (for testing).

    Returns:
        Path to the written config file.
    """
    target = Path(conf_path or _CONF_FILE).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {"neurograph_home": str(Path(neurograph_home).expanduser())}
    target.write_text(json.dumps(data, indent=2) + "\n")
    return target


def read_conf(conf_path: Optional[str] = None) -> Optional[str]:
    """Read the configured neurograph_home from the config file.

    Returns:
        The configured path string, or None if no config file exists.
    """
    target = Path(conf_path or _CONF_FILE).expanduser()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text())
        return data.get("neurograph_home")
    except (json.JSONDecodeError, OSError):
        return None
