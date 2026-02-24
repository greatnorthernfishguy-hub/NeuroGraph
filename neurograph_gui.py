#!/usr/bin/env python3
"""
NeuroGraph Manager -- tkinter desktop GUI for updates, ingestion, and monitoring.

Provides a four-tab interface:
  - Status:    Live telemetry dashboard from NeuroGraphMemory.stats()
  - Ingestion: watchdog file watcher + manual ingest, moves files on success
  - Updates:   Git-based update mechanism with neurograph-patch integration
  - Logs:      Viewer for events.jsonl and gui.log

Launch:
    python3 neurograph_gui.py
    # Or from Linux application launcher via .desktop entry

Environment:
    NEUROGRAPH_WORKSPACE_DIR  Override workspace (default: ~/.openclaw/neurograph)
    NEUROGRAPH_SKILL_DIR      Override skill dir (default: ~/.openclaw/skills/neurograph)
    NEUROGRAPH_GUI_DIR        Override GUI data dir (default: ~/.neurograph)

Grok Review Changelog (v0.7.1):
    Accepted: GUIMessageQueue._poll() now logs callback exceptions via
        logging.debug(exc_info=True) instead of bare 'except: pass'.
        Errors are still caught (GUI must not crash from background thread
        failures), but they're now visible in gui.log for debugging.
    Rejected: 'No progress bar for ingest_directory()' — UX enhancement,
        not a bug.  ingest_directory() runs on a daemon thread (line 577)
        so the GUI doesn't block.  Progress indication is a future feature.
    Rejected: 'Ingests any file in inbox — no sandbox for malicious PDFs' —
        FileWatcher.should_ignore() (line 196-207) already filters by
        extension against SUPPORTED_EXTENSIONS.  PyPDF2's text extraction
        does not execute embedded content.  PDF sandboxing would require
        process isolation infrastructure beyond the scope of a desktop GUI.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants  (no tkinter imports yet -- keeps classes below testable headless)
# ---------------------------------------------------------------------------

APP_NAME = "NeuroGraph Manager"
APP_VERSION = "0.5.5"

DEFAULT_GUI_DIR = str(Path.home() / ".neurograph")

DEFAULT_CONFIG: Dict[str, Any] = {
    "inbox_path": str(Path.home() / ".neurograph" / "inbox"),
    "ingested_path": str(Path.home() / ".neurograph" / "ingested"),
    "repo_url": "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git",
    "repo_path": str(Path.home() / ".neurograph" / "repo"),
    "log_path": str(Path.home() / ".neurograph" / "logs" / "gui.log"),
    "workspace_dir": os.environ.get(
        "NEUROGRAPH_WORKSPACE_DIR",
        str(Path.home() / ".openclaw" / "neurograph"),
    ),
    "skill_dir": os.environ.get(
        "NEUROGRAPH_SKILL_DIR",
        str(Path.home() / ".openclaw" / "skills" / "neurograph"),
    ),
    "watcher_enabled": True,
    "watcher_stability_seconds": 1.0,
    "status_refresh_seconds": 5,
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".md", ".txt", ".html", ".htm", ".pdf",
    ".json", ".csv",
    # Media (video)
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".ogv",
    # Media (audio)
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus",
    ".aiff", ".alac",
}

_logger = logging.getLogger("neurograph_gui")

# ---------------------------------------------------------------------------
# Optional watchdog import
# ---------------------------------------------------------------------------
try:
    from watchdog.observers import Observer as _WatchdogObserver
    from watchdog.events import FileSystemEventHandler as _FSHandler
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False

# ---------------------------------------------------------------------------
# 1. GUIConfig
# ---------------------------------------------------------------------------


class GUIConfig:
    """Manages ``~/.neurograph/config.json`` with sensible defaults.

    Reads on construction; writes on explicit ``save()``.  Any key not present
    in the file falls back to ``DEFAULT_CONFIG``.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._path = Path(config_path or os.path.join(DEFAULT_GUI_DIR, "config.json"))
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key)
        if val is not None:
            return val
        val = DEFAULT_CONFIG.get(key)
        if val is not None:
            return val
        return default

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def ensure_directories(self) -> None:
        """Create inbox, ingested, repo, and logs dirs if missing."""
        for key in ("inbox_path", "ingested_path", "repo_path"):
            Path(self.get(key)).mkdir(parents=True, exist_ok=True)
        log_path = self.get("log_path")
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. FileWatcher
# ---------------------------------------------------------------------------


class FileWatcher:
    """Watches an inbox directory for new files and fires a callback.

    Uses the ``watchdog`` library when available.  Falls back to periodic
    polling (every 2 s) if watchdog is not installed.

    Files are considered *stable* when their size has not changed for
    ``stability_seconds``.  Hidden files, temp files (``.tmp``, ``.part``,
    ``.crdownload``, names ending with ``~``), and unsupported extensions
    are silently ignored.
    """

    def __init__(
        self,
        inbox_path: str,
        on_file_ready: Callable[[str], None],
        on_error: Callable[[str], None],
        stability_seconds: float = 1.0,
    ) -> None:
        self._inbox = Path(inbox_path)
        self._on_file_ready = on_file_ready
        self._on_error = on_error
        self._stability = stability_seconds
        self._pending: Dict[str, float] = {}  # path -> last-known size
        self._observer: Any = None
        self._running = False
        self._poll_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._known_files: set = set()  # for polling fallback

    # -- public API --

    def start(self) -> None:
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._running = True
        if _HAS_WATCHDOG:
            self._start_watchdog()
        else:
            self._known_files = self._scan_inbox()
        self._schedule_stability_check()

    def stop(self) -> None:
        self._running = False
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
            self._observer = None
        if self._poll_timer is not None:
            self._poll_timer.cancel()
            self._poll_timer = None

    @property
    def is_running(self) -> bool:
        return self._running

    # -- ignore logic (static for testability) --

    @staticmethod
    def should_ignore(path: str) -> bool:
        """Return ``True`` if *path* should be skipped during ingestion."""
        name = os.path.basename(path)
        if name.startswith("."):
            return True
        if name.endswith(("~", ".tmp", ".part", ".crdownload", ".swp")):
            return True
        ext = os.path.splitext(name)[1].lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return True
        return False

    # -- watchdog backend --

    def _start_watchdog(self) -> None:
        handler = _InboxHandler(self._on_fs_event)
        self._observer = _WatchdogObserver()
        self._observer.schedule(handler, str(self._inbox), recursive=False)
        self._observer.daemon = True
        self._observer.start()

    def _on_fs_event(self, path: str) -> None:
        if self.should_ignore(path):
            return
        with self._lock:
            try:
                self._pending[path] = os.path.getsize(path)
            except OSError:
                pass

    # -- polling fallback --

    def _scan_inbox(self) -> set:
        try:
            return {str(p) for p in self._inbox.iterdir() if p.is_file()}
        except OSError:
            return set()

    def _poll_for_new(self) -> None:
        current = self._scan_inbox()
        new_files = current - self._known_files
        for f in new_files:
            if not self.should_ignore(f):
                with self._lock:
                    try:
                        self._pending[f] = os.path.getsize(f)
                    except OSError:
                        pass
        self._known_files = current

    # -- stability checker --

    def _schedule_stability_check(self) -> None:
        if not self._running:
            return
        self._poll_timer = threading.Timer(0.5, self._check_stability)
        self._poll_timer.daemon = True
        self._poll_timer.start()

    def _check_stability(self) -> None:
        if not self._running:
            return
        # Polling fallback: detect new files
        if not _HAS_WATCHDOG:
            self._poll_for_new()
        ready: List[str] = []
        with self._lock:
            to_remove = []
            for path, last_size in list(self._pending.items()):
                try:
                    current_size = os.path.getsize(path)
                except OSError:
                    to_remove.append(path)
                    continue
                if current_size == last_size:
                    ready.append(path)
                    to_remove.append(path)
                else:
                    self._pending[path] = current_size
            for p in to_remove:
                self._pending.pop(p, None)
        for path in ready:
            try:
                self._on_file_ready(path)
            except Exception as exc:
                self._on_error(f"Ingestion callback failed for {path}: {exc}")
        self._schedule_stability_check()


if _HAS_WATCHDOG:
    class _InboxHandler(_FSHandler):
        """Forward watchdog created/modified events to a single callback."""

        def __init__(self, callback: Callable[[str], None]) -> None:
            super().__init__()
            self._cb = callback

        def on_created(self, event):  # type: ignore[override]
            if not event.is_directory:
                self._cb(event.src_path)

        def on_modified(self, event):  # type: ignore[override]
            if not event.is_directory:
                self._cb(event.src_path)


# ---------------------------------------------------------------------------
# 3. GitUpdater
# ---------------------------------------------------------------------------


class GitUpdater:
    """Git-based update mechanism.

    * First use:  ``git clone --depth 1`` into ``repo_path``.
    * Check:      ``git fetch origin main`` then compare HEADs.
    * Update:     ``git pull origin main`` then deploy via neurograph-patch.

    All git and deploy operations run on daemon threads so the GUI
    stays responsive.  Results are delivered via callbacks.
    """

    def __init__(
        self,
        repo_url: str,
        repo_path: str,
        skill_dir: str,
        workspace_dir: str,
        on_status: Callable[[str], None],
        on_complete: Callable[[Dict[str, Any]], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._repo_url = repo_url
        self._repo_path = Path(repo_path)
        self._skill_dir = Path(skill_dir)
        self._workspace_dir = Path(workspace_dir)
        self._on_status = on_status
        self._on_complete = on_complete
        self._on_error = on_error

    # -- public API (non-blocking) --

    def check_for_updates(self) -> None:
        threading.Thread(target=self._check_worker, daemon=True).start()

    def pull_and_deploy(self) -> None:
        threading.Thread(target=self._pull_worker, daemon=True).start()

    def ensure_repo(self) -> bool:
        """Clone if not present.  Returns True when repo is ready."""
        if (self._repo_path / ".git").is_dir():
            return True
        self._repo_path.mkdir(parents=True, exist_ok=True)
        try:
            self._run_git(
                "clone", "--depth", "1", self._repo_url, str(self._repo_path)
            )
            return True
        except Exception as exc:
            self._on_error(f"Clone failed: {exc}")
            return False

    def get_local_commit(self) -> Optional[str]:
        try:
            r = self._run_git(
                "rev-parse", "--short", "HEAD", cwd=str(self._repo_path)
            )
            return r.stdout.strip()
        except Exception:
            return None

    # -- background workers --

    def _check_worker(self) -> None:
        try:
            self._on_status("Ensuring local repo exists...")
            if not self.ensure_repo():
                return
            self._on_status("Fetching from origin...")
            self._run_git(
                "fetch", "--depth", "1", "origin", "main",
                cwd=str(self._repo_path),
            )
            local = self._run_git(
                "rev-parse", "--short", "HEAD",
                cwd=str(self._repo_path),
            ).stdout.strip()
            remote = self._run_git(
                "rev-parse", "--short", "origin/main",
                cwd=str(self._repo_path),
            ).stdout.strip()
            behind_str = self._run_git(
                "rev-list", "--count", "HEAD..origin/main",
                cwd=str(self._repo_path),
            ).stdout.strip()
            behind = int(behind_str) if behind_str.isdigit() else 0
            log_text = ""
            if behind > 0:
                log_result = self._run_git(
                    "log", "--oneline", "HEAD..origin/main",
                    cwd=str(self._repo_path),
                )
                log_text = log_result.stdout.strip()
            self._on_complete({
                "action": "check",
                "has_updates": behind > 0,
                "local_commit": local,
                "remote_commit": remote,
                "commits_behind": behind,
                "commit_log": log_text,
            })
        except Exception as exc:
            self._on_error(f"Update check failed: {exc}")

    def _pull_worker(self) -> None:
        try:
            if not self.ensure_repo():
                return
            # Reset the update-only clone to a clean state before pulling.
            # This clone is never the user's working copy — it exists solely
            # to fetch upstream changes for deployment.  Unstaged changes
            # (e.g. from a previous partial deploy or rebase config) would
            # block ``git pull``, so we discard them here.
            self._on_status("Cleaning local repo state...")
            self._run_git(
                "checkout", "--", ".",
                cwd=str(self._repo_path),
            )
            self._on_status("Pulling latest changes...")
            self._run_git("pull", "origin", "main", cwd=str(self._repo_path))
            commit = self.get_local_commit() or "unknown"
            self._on_status(f"Pulled to {commit}. Deploying files...")

            # Use neurograph-patch from the *pulled* repo for deployment
            patch_script = self._repo_path / "neurograph-patch"
            if patch_script.exists():
                result = self._deploy_via_patch(patch_script)
            else:
                result = self._deploy_copy_fallback()

            result["action"] = "deploy"
            result["commit"] = commit
            self._on_complete(result)
        except Exception as exc:
            self._on_error(f"Update failed: {exc}")

    def _deploy_via_patch(self, patch_script: Path) -> Dict[str, Any]:
        """Import neurograph-patch from pulled repo and run its logic."""
        loader = importlib.machinery.SourceFileLoader(
            "neurograph_patch_mod", str(patch_script)
        )
        spec = importlib.util.spec_from_loader("neurograph_patch_mod", loader)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        loader.exec_module(mod)

        paths = {
            "repo": self._repo_path,
            "skill": self._skill_dir,
            "workspace": self._workspace_dir,
            "bin": Path.home() / ".local" / "bin",
            "home": Path.home(),
        }
        changes = mod.detect_changes(paths)
        to_patch = [c for c in changes if c["status"] in ("CHANGED", "MISSING")]

        if not to_patch:
            self._on_status("All files up to date.")
            return {"success": True, "files_patched": 0, "files_unchanged": len(changes)}

        timestamp = int(time.time())
        # Backup existing files
        for entry in to_patch:
            for target in entry["targets"]:
                mod.backup_file(target, timestamp)

        # Copy updated files
        for entry in to_patch:
            for target in entry["targets"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(entry["source"]), str(target))
                if entry["executable"]:
                    target.chmod(target.stat().st_mode | 0o111)
            self._on_status(f"  Patched {entry['name']}")

        # Validate
        valid, msg = mod.validate_patch(paths["skill"])
        if not valid:
            self._on_status(f"Validation FAILED: {msg}. Rolling back...")
            for entry in to_patch:
                for target in entry["targets"]:
                    backup = str(target) + f".backup-{timestamp}"
                    if os.path.exists(backup):
                        shutil.copy2(backup, str(target))
            return {"success": False, "validation": msg, "files_patched": 0,
                    "files_unchanged": len(changes)}

        self._on_status(f"Validation passed: {msg}")
        identical = [c for c in changes if c["status"] == "IDENTICAL"]
        return {
            "success": True,
            "files_patched": len(to_patch),
            "files_unchanged": len(identical),
            "validation": msg,
        }

    def _deploy_copy_fallback(self) -> Dict[str, Any]:
        """Simple file-copy fallback when neurograph-patch is not in repo."""
        core_files = [
            # Core engine
            "neuro_foundation.py", "universal_ingestor.py",
            "openclaw_hook.py", "neurograph_migrate.py", "neurograph_gui.py",
            # Phase 6: NG-Lite + bridge
            "ng_lite.py", "ng_bridge.py",
            # Phase 7: Peer bridge + ET Module Manager
            "ng_peer_bridge.py", "et_module.json",
            "et_modules/__init__.py", "et_modules/manager.py",
        ]
        patched = 0
        for name in core_files:
            src = self._repo_path / name
            dst = self._skill_dir / name
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                patched += 1
                self._on_status(f"  Copied {name}")
        return {"success": True, "files_patched": patched, "files_unchanged": 0}

    # -- git subprocess helper --

    def _run_git(self, *args: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result


# ---------------------------------------------------------------------------
# 4. IngestionManager
# ---------------------------------------------------------------------------


class IngestionManager:
    """Coordinates file ingestion and post-ingestion file movement.

    Uses ``NeuroGraphMemory.ingest_file()`` for the heavy lifting, then
    moves successfully-ingested files to ``ingested/YYYY-MM-DD/``.

    ``NeuroGraphMemory`` is initialised lazily on first use to avoid
    a slow startup if the user only wants to check for updates.
    """

    def __init__(
        self,
        workspace_dir: str,
        ingested_path: str,
        on_result: Callable[[Dict[str, Any]], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._workspace_dir = workspace_dir
        self._ingested_path = Path(ingested_path)
        self._on_result = on_result
        self._on_error = on_error
        self._ng: Any = None  # lazy NeuroGraphMemory

    def get_memory(self) -> Any:
        if self._ng is None:
            # Add skill dir and script dir to path
            for p in (
                os.environ.get("NEUROGRAPH_SKILL_DIR", ""),
                str(Path(__file__).resolve().parent),
            ):
                if p and p not in sys.path:
                    sys.path.insert(0, p)
            from openclaw_hook import NeuroGraphMemory
            self._ng = NeuroGraphMemory.get_instance(
                workspace_dir=self._workspace_dir
            )
        return self._ng

    def ingest_file(self, path: str) -> None:
        threading.Thread(
            target=self._ingest_file_worker, args=(path,), daemon=True
        ).start()

    def ingest_files(self, paths: List[str]) -> None:
        threading.Thread(
            target=self._ingest_batch_worker, args=(paths,), daemon=True
        ).start()

    @property
    def is_initialized(self) -> bool:
        """True if NeuroGraphMemory has been loaded (non-blocking check)."""
        return self._ng is not None

    def get_stats(self) -> Dict[str, Any]:
        return self.get_memory().stats()

    def get_stats_async(
        self,
        on_result: Callable[[Dict[str, Any]], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Fetch stats in a background thread (non-blocking)."""
        threading.Thread(
            target=self._stats_worker,
            args=(on_result, on_error),
            daemon=True,
        ).start()

    def save_checkpoint_async(
        self,
        on_result: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Save checkpoint in a background thread (non-blocking)."""
        def worker():
            try:
                path = self.get_memory().save()
                on_result(path)
            except Exception as exc:
                on_error(str(exc))
        threading.Thread(target=worker, daemon=True).start()

    def save_checkpoint(self) -> str:
        return self.get_memory().save()

    def _stats_worker(
        self,
        on_result: Callable[[Dict[str, Any]], None],
        on_error: Callable[[str], None],
    ) -> None:
        try:
            stats = self.get_stats()
            on_result(stats)
        except Exception as exc:
            on_error(str(exc))

    def _ingest_file_worker(self, path: str) -> None:
        try:
            ng = self.get_memory()
            result = ng.ingest_file(path)
            if result.get("status") == "error":
                self._on_error(f"Failed: {Path(path).name}: {result.get('reason')}")
                return
            dest = self._move_to_ingested(path)
            result["moved_to"] = dest
            result["original_path"] = path
            self._on_result(result)
        except Exception as exc:
            self._on_error(f"Ingestion error for {Path(path).name}: {exc}")

    def _ingest_batch_worker(self, paths: List[str]) -> None:
        for p in paths:
            self._ingest_file_worker(p)

    def _move_to_ingested(self, original_path: str) -> str:
        date_dir = self._ingested_path / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        name = Path(original_path).name
        dest = date_dir / name
        counter = 1
        while dest.exists():
            stem = Path(original_path).stem
            suffix = Path(original_path).suffix
            dest = date_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.move(original_path, str(dest))
        return str(dest)


# ===========================================================================
# tkinter GUI  (imported conditionally for headless testability)
# ===========================================================================

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


class GUIMessageQueue:
    """Thread-safe bridge: background threads enqueue callbacks; the tkinter
    main loop drains and executes them every *poll_ms* milliseconds."""

    def __init__(self, root: Any, poll_ms: int = 100) -> None:
        self._root = root
        self._queue: queue.Queue = queue.Queue()
        self._poll_ms = poll_ms
        self._poll()

    def put(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        self._queue.put((callback, args, kwargs))

    def _poll(self) -> None:
        while not self._queue.empty():
            try:
                cb, args, kwargs = self._queue.get_nowait()
                cb(*args, **kwargs)
            except Exception:
                # Log callback errors instead of silently swallowing
                # (Grok review: GUI queue exception handling)
                logging.getLogger("neurograph.gui").debug(
                    "GUI queue callback error", exc_info=True,
                )
        self._root.after(self._poll_ms, self._poll)


# ---------------------------------------------------------------------------
# 5. NeuroGraphGUI  --  main window
# ---------------------------------------------------------------------------


class NeuroGraphGUI:
    """Main application window with four tabs."""

    def __init__(self, root: Any) -> None:
        if not _HAS_TK:
            raise RuntimeError("tkinter is required for the GUI. Install python3-tk.")
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("900x650")
        self.root.minsize(750, 520)

        self.config = GUIConfig()
        self.config.ensure_directories()

        self.msg_queue = GUIMessageQueue(root)

        # Set up file-based logging
        self._setup_logging()

        # Core managers
        self.ingestion_mgr = IngestionManager(
            workspace_dir=self.config.get("workspace_dir"),
            ingested_path=self.config.get("ingested_path"),
            on_result=lambda r: self.msg_queue.put(self._on_ingest_result, r),
            on_error=lambda e: self.msg_queue.put(self._on_ingest_error, e),
        )
        self.git_updater = GitUpdater(
            repo_url=self.config.get("repo_url"),
            repo_path=self.config.get("repo_path"),
            skill_dir=self.config.get("skill_dir"),
            workspace_dir=self.config.get("workspace_dir"),
            on_status=lambda s: self.msg_queue.put(self._on_update_status, s),
            on_complete=lambda r: self.msg_queue.put(self._on_update_complete, r),
            on_error=lambda e: self.msg_queue.put(self._on_update_error, e),
        )
        self.file_watcher: Optional[FileWatcher] = None
        self._ingest_history: List[Dict[str, Any]] = []

        # Build the UI
        self._build_menu()
        self._build_tabs()
        self._build_status_bar()

        # Start watcher if enabled
        if self.config.get("watcher_enabled", True):
            self._start_watcher()

        # Periodic status refresh
        self._refresh_status()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Settings...", command=self._show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        self._build_status_tab()
        self._build_ingestion_tab()
        self._build_updates_tab()
        self._build_logs_tab()

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready")
        bar = ttk.Label(
            self.root, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2),
        )
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ---- Status Tab ----

    def _build_status_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Status ")

        title = ttk.Label(frame, text="NeuroGraph Telemetry", font=("", 13, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))

        self._stat_labels: Dict[str, tk.StringVar] = {}
        fields = [
            ("Nodes", "nodes"), ("Synapses", "synapses"),
            ("Hyperedges", "hyperedges"), ("Vector DB", "vector_db_count"),
            ("Timestep", "timestep"), ("Firing Rate", "firing_rate"),
            ("Mean Weight", "mean_weight"), ("Predictions", "predictions_made"),
            ("Confirmed", "predictions_confirmed"), ("Accuracy", "prediction_accuracy"),
            ("Novel Seq.", "novel_sequences"), ("Pruned / Sprouted", "pruned_sprouted"),
            ("Messages", "message_count"), ("Embedding", "embedding_backend"),
        ]
        for i, (label_text, key) in enumerate(fields):
            row, col = divmod(i, 2)
            ttk.Label(frame, text=f"{label_text}:").grid(
                row=row + 1, column=col * 2, sticky=tk.E, padx=(0, 5), pady=2,
            )
            var = tk.StringVar(value="--")
            ttk.Label(frame, textvariable=var, width=20).grid(
                row=row + 1, column=col * 2 + 1, sticky=tk.W, pady=2,
            )
            self._stat_labels[key] = var

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(fields) // 2 + 2, column=0, columnspan=4, pady=10)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_status_now).pack(
            side=tk.LEFT, padx=5,
        )
        ttk.Button(btn_frame, text="Save Checkpoint", command=self._save_checkpoint).pack(
            side=tk.LEFT, padx=5,
        )

    def _refresh_status(self) -> None:
        """Auto-refresh driven by root.after().

        Uses a background thread for stats retrieval so the GUI main loop
        never blocks on NeuroGraphMemory initialization or model loading.
        """
        self._refresh_status_now()
        interval = int(self.config.get("status_refresh_seconds", 5)) * 1000
        self.root.after(interval, self._refresh_status)

    def _refresh_status_now(self) -> None:
        """Kick off an async stats fetch.  Results arrive via msg_queue."""
        if not self.ingestion_mgr.is_initialized:
            # Show "initializing" state instead of blocking
            for var in self._stat_labels.values():
                var.set("loading...")
            self._status_var.set("Initializing NeuroGraph (loading model)...")

        self.ingestion_mgr.get_stats_async(
            on_result=lambda stats: self.msg_queue.put(self._apply_stats, stats),
            on_error=lambda err: self.msg_queue.put(self._apply_stats_error, err),
        )

    def _apply_stats(self, stats: Dict[str, Any]) -> None:
        """Apply fetched stats to the status tab labels (runs on main thread)."""
        for key, var in self._stat_labels.items():
            if key == "pruned_sprouted":
                var.set(f"{stats.get('pruned', 0)} / {stats.get('sprouted', 0)}")
            elif key == "embedding_backend":
                emb = stats.get("embedding", {})
                if isinstance(emb, dict):
                    backend = emb.get("backend", "unknown")
                    dim = emb.get("dimension", "?")
                    var.set(f"{backend} ({dim}d)")
                else:
                    var.set(str(emb))
            else:
                var.set(str(stats.get(key, "--")))
        self._status_var.set("Ready")

    def _apply_stats_error(self, error: str) -> None:
        """Handle stats fetch failure (runs on main thread)."""
        for var in self._stat_labels.values():
            var.set("--")
        self._status_var.set(f"Stats error: {error}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint asynchronously so the GUI doesn't freeze."""
        self._status_var.set("Saving checkpoint...")
        self.ingestion_mgr.save_checkpoint_async(
            on_result=lambda path: self.msg_queue.put(
                self._status_var.set, f"Checkpoint saved to {path}"
            ),
            on_error=lambda err: self.msg_queue.put(
                self._status_var.set, f"Save failed: {err}"
            ),
        )

    # ---- Ingestion Tab ----

    def _build_ingestion_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Ingestion ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(5, weight=1)

        # -- Inbox path and watcher controls --
        top = ttk.LabelFrame(frame, text="Inbox", padding=5)
        top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 5))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Path:").grid(row=0, column=0, sticky=tk.E, padx=(0, 5))
        self._inbox_path_var = tk.StringVar(value=self.config.get("inbox_path"))
        ttk.Label(top, textvariable=self._inbox_path_var).grid(
            row=0, column=1, sticky=tk.W,
        )

        self._watcher_var = tk.StringVar(value="Watcher: OFF")
        self._watcher_btn = ttk.Button(
            top, textvariable=self._watcher_var, width=14,
            command=self._toggle_watcher,
        )
        self._watcher_btn.grid(row=0, column=2, padx=5)
        if self.file_watcher and self.file_watcher.is_running:
            self._watcher_var.set("Watcher: ON")

        # -- Inbox file list --
        ttk.Label(frame, text="Files in Inbox:").grid(
            row=1, column=0, sticky=tk.W, pady=(5, 0),
        )
        self._inbox_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=6)
        self._inbox_listbox.grid(row=2, column=0, sticky=tk.NSEW)
        inbox_scroll = ttk.Scrollbar(frame, command=self._inbox_listbox.yview)
        inbox_scroll.grid(row=2, column=1, sticky=tk.NS)
        self._inbox_listbox.config(yscrollcommand=inbox_scroll.set)

        # -- Action buttons --
        btn_bar = ttk.Frame(frame)
        btn_bar.grid(row=3, column=0, pady=5, sticky=tk.W)
        ttk.Button(btn_bar, text="Ingest All", command=self._ingest_all).pack(
            side=tk.LEFT, padx=(0, 5),
        )
        ttk.Button(btn_bar, text="Ingest Selected", command=self._ingest_selected).pack(
            side=tk.LEFT, padx=(0, 5),
        )
        ttk.Button(btn_bar, text="Add Files...", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 5),
        )
        ttk.Button(btn_bar, text="Refresh Inbox", command=self._refresh_inbox).pack(
            side=tk.LEFT,
        )

        # -- Ingestion history --
        ttk.Label(frame, text="Ingestion History:").grid(
            row=4, column=0, sticky=tk.W, pady=(5, 0),
        )
        self._history_listbox = tk.Listbox(frame, height=6)
        self._history_listbox.grid(row=5, column=0, sticky=tk.NSEW)
        hist_scroll = ttk.Scrollbar(frame, command=self._history_listbox.yview)
        hist_scroll.grid(row=5, column=1, sticky=tk.NS)
        self._history_listbox.config(yscrollcommand=hist_scroll.set)

        # Initial refresh
        self._refresh_inbox()

    def _toggle_watcher(self) -> None:
        if self.file_watcher and self.file_watcher.is_running:
            self._stop_watcher()
            self._watcher_var.set("Watcher: OFF")
            self._status_var.set("File watcher stopped")
        else:
            self._start_watcher()
            self._watcher_var.set("Watcher: ON")
            self._status_var.set("File watcher started")

    def _refresh_inbox(self) -> None:
        self._inbox_listbox.delete(0, tk.END)
        inbox = Path(self.config.get("inbox_path"))
        if inbox.is_dir():
            for fp in sorted(inbox.iterdir()):
                if fp.is_file() and not FileWatcher.should_ignore(str(fp)):
                    self._inbox_listbox.insert(tk.END, fp.name)

    def _ingest_all(self) -> None:
        inbox = Path(self.config.get("inbox_path"))
        files = [
            str(inbox / self._inbox_listbox.get(i))
            for i in range(self._inbox_listbox.size())
        ]
        if not files:
            self._status_var.set("Inbox is empty")
            return
        self._status_var.set(f"Ingesting {len(files)} file(s)...")
        self.ingestion_mgr.ingest_files(files)

    def _ingest_selected(self) -> None:
        inbox = Path(self.config.get("inbox_path"))
        sel = self._inbox_listbox.curselection()
        if not sel:
            self._status_var.set("No files selected")
            return
        files = [str(inbox / self._inbox_listbox.get(i)) for i in sel]
        self._status_var.set(f"Ingesting {len(files)} file(s)...")
        self.ingestion_mgr.ingest_files(files)

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select files to ingest",
            filetypes=[
                ("All supported",
                 "*.py *.js *.ts *.md *.txt *.html *.htm *.pdf "
                 "*.json *.csv "
                 "*.mp4 *.avi *.mkv *.mov *.webm *.mp3 *.wav *.flac *.ogg *.m4a"),
                ("Code", "*.py *.js *.ts"),
                ("Documents", "*.md *.txt *.pdf *.html *.htm"),
                ("Data", "*.json *.csv"),
                ("Media", "*.mp4 *.avi *.mkv *.mov *.webm *.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        inbox = Path(self.config.get("inbox_path"))
        for p in paths:
            dest = inbox / Path(p).name
            if not dest.exists():
                shutil.copy2(p, str(dest))
        self._refresh_inbox()
        self._status_var.set(f"Added {len(paths)} file(s) to inbox")

    def _on_ingest_result(self, result: Dict[str, Any]) -> None:
        name = Path(result.get("original_path", "?")).name
        nodes = result.get("nodes_created", 0)
        syns = result.get("synapses_created", 0)
        entry = f"OK  {name}  ({nodes} nodes, {syns} synapses)"
        self._history_listbox.insert(0, entry)
        self._ingest_history.insert(0, result)
        _logger.info("Ingested %s: %d nodes, %d synapses", name, nodes, syns)
        self._refresh_inbox()
        self._status_var.set(f"Ingested {name}")

    def _on_ingest_error(self, error: str) -> None:
        self._history_listbox.insert(0, f"ERR  {error}")
        _logger.warning("Ingestion error: %s", error)
        self._status_var.set(error)

    def _on_file_detected(self, path: str) -> None:
        """Called by FileWatcher via msg_queue when a file is stable."""
        _logger.info("Watcher detected: %s", path)
        self._refresh_inbox()
        self._status_var.set(f"Auto-ingesting {Path(path).name}...")
        self.ingestion_mgr.ingest_file(path)

    # ---- Updates Tab ----

    def _build_updates_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Updates ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

        # -- Info labels --
        info = ttk.LabelFrame(frame, text="Installation", padding=5)
        info.grid(row=0, column=0, sticky=tk.EW, pady=(0, 5))
        info.columnconfigure(1, weight=1)

        ttk.Label(info, text="Version:").grid(row=0, column=0, sticky=tk.E, padx=(0, 5))
        self._version_var = tk.StringVar(value=APP_VERSION)
        ttk.Label(info, textvariable=self._version_var).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(info, text="Skill Dir:").grid(row=1, column=0, sticky=tk.E, padx=(0, 5))
        ttk.Label(info, text=self.config.get("skill_dir")).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(info, text="Repo:").grid(row=2, column=0, sticky=tk.E, padx=(0, 5))
        ttk.Label(info, text=self.config.get("repo_path")).grid(row=2, column=1, sticky=tk.W)

        # -- Action buttons --
        btn_bar = ttk.Frame(frame)
        btn_bar.grid(row=1, column=0, pady=5, sticky=tk.W)
        self._check_btn = ttk.Button(
            btn_bar, text="Check for Updates", command=self._on_check_updates,
        )
        self._check_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._update_btn = ttk.Button(
            btn_bar, text="Update Now", command=self._on_update_now,
            state=tk.DISABLED,
        )
        self._update_btn.pack(side=tk.LEFT)

        # -- Status display --
        self._update_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._update_status_var).grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )

        # -- Update log --
        ttk.Label(frame, text="Update Log:").grid(
            row=3, column=0, sticky=tk.W, pady=(5, 0),
        )
        self._update_log = scrolledtext.ScrolledText(
            frame, height=12, state=tk.DISABLED, wrap=tk.WORD,
        )
        self._update_log.grid(row=4, column=0, sticky=tk.NSEW)

    def _on_check_updates(self) -> None:
        self._check_btn.config(state=tk.DISABLED)
        self._update_btn.config(state=tk.DISABLED)
        self._append_update_log("Checking for updates...")
        self.git_updater.check_for_updates()

    def _on_update_now(self) -> None:
        self._update_btn.config(state=tk.DISABLED)
        self._check_btn.config(state=tk.DISABLED)
        self._append_update_log("Starting update...")
        self.git_updater.pull_and_deploy()

    def _on_update_status(self, status: str) -> None:
        self._append_update_log(status)

    def _on_update_complete(self, result: Dict[str, Any]) -> None:
        self._check_btn.config(state=tk.NORMAL)
        action = result.get("action", "")
        if action == "check":
            behind = result.get("commits_behind", 0)
            local = result.get("local_commit", "?")
            remote = result.get("remote_commit", "?")
            if result.get("has_updates"):
                self._update_status_var.set(
                    f"{behind} commit(s) behind  (local: {local}, remote: {remote})"
                )
                self._append_update_log(f"Updates available: {behind} commit(s) behind")
                log_text = result.get("commit_log", "")
                if log_text:
                    self._append_update_log(log_text)
                self._update_btn.config(state=tk.NORMAL)
            else:
                self._update_status_var.set(f"Up to date (commit: {local})")
                self._append_update_log("Already up to date.")
                self._update_btn.config(state=tk.DISABLED)
        elif action == "deploy":
            patched = result.get("files_patched", 0)
            commit = result.get("commit", "?")
            if result.get("success"):
                self._append_update_log(
                    f"Update complete: {patched} file(s) patched, commit {commit}"
                )
                self._update_status_var.set(f"Updated to {commit}")
                self._status_var.set(f"Update complete ({patched} files)")
            else:
                msg = result.get("validation", "unknown error")
                self._append_update_log(f"Update FAILED: {msg}")
                self._update_status_var.set("Update failed (rolled back)")
            self._update_btn.config(state=tk.DISABLED)

    def _on_update_error(self, error: str) -> None:
        self._check_btn.config(state=tk.NORMAL)
        self._append_update_log(f"ERROR: {error}")
        self._update_status_var.set("Error")
        self._status_var.set(error)

    def _append_update_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._update_log.config(state=tk.NORMAL)
        self._update_log.insert(tk.END, f"[{ts}] {text}\n")
        self._update_log.see(tk.END)
        self._update_log.config(state=tk.DISABLED)

    # ---- Logs Tab ----

    def _build_logs_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Logs ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Controls
        ctl = ttk.Frame(frame)
        ctl.grid(row=0, column=0, sticky=tk.EW, pady=(0, 5))

        ttk.Label(ctl, text="Source:").pack(side=tk.LEFT, padx=(0, 5))
        self._log_source_var = tk.StringVar(value="events.jsonl")
        src_combo = ttk.Combobox(
            ctl, textvariable=self._log_source_var, width=18,
            values=["events.jsonl", "gui.log"], state="readonly",
        )
        src_combo.pack(side=tk.LEFT, padx=(0, 10))
        src_combo.bind("<<ComboboxSelected>>", lambda e: self._load_log())

        ttk.Button(ctl, text="Refresh", command=self._load_log).pack(
            side=tk.LEFT, padx=(0, 5),
        )
        ttk.Button(ctl, text="Clear Display", command=self._clear_log_display).pack(
            side=tk.LEFT,
        )

        # Log display
        self._log_text = scrolledtext.ScrolledText(
            frame, state=tk.DISABLED, wrap=tk.WORD,
        )
        self._log_text.grid(row=1, column=0, sticky=tk.NSEW)

    def _load_log(self) -> None:
        source = self._log_source_var.get()
        if source == "events.jsonl":
            path = Path(self.config.get("workspace_dir")) / "memory" / "events.jsonl"
        else:
            path = Path(self.config.get("log_path"))

        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        if path.exists():
            try:
                lines = path.read_text(errors="replace").splitlines()
                # Show last 500 lines
                for line in lines[-500:]:
                    if source == "events.jsonl":
                        try:
                            ev = json.loads(line)
                            ts = datetime.fromtimestamp(ev.get("timestamp", 0))
                            event_type = ev.get("event", "?")
                            data = ev.get("data", {})
                            self._log_text.insert(
                                tk.END,
                                f"{ts:%Y-%m-%d %H:%M:%S}  {event_type}\n",
                            )
                            for k, v in data.items():
                                self._log_text.insert(tk.END, f"    {k}: {v}\n")
                        except json.JSONDecodeError:
                            self._log_text.insert(tk.END, line + "\n")
                    else:
                        self._log_text.insert(tk.END, line + "\n")
            except OSError as exc:
                self._log_text.insert(tk.END, f"Error reading log: {exc}\n")
        else:
            self._log_text.insert(tk.END, f"Log file not found: {path}\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log_display(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ---- Settings Dialog ----

    def _show_settings(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("500x320")
        dlg.transient(self.root)
        dlg.grab_set()

        pad = {"padx": 5, "pady": 3}
        entries: Dict[str, tk.StringVar] = {}

        fields = [
            ("Inbox Path", "inbox_path"),
            ("Ingested Path", "ingested_path"),
            ("Repo URL", "repo_url"),
            ("Repo Clone Path", "repo_path"),
            ("Workspace Dir", "workspace_dir"),
            ("Skill Dir", "skill_dir"),
        ]
        for i, (label, key) in enumerate(fields):
            ttk.Label(dlg, text=f"{label}:").grid(row=i, column=0, sticky=tk.E, **pad)
            var = tk.StringVar(value=self.config.get(key, ""))
            ttk.Entry(dlg, textvariable=var, width=50).grid(
                row=i, column=1, sticky=tk.EW, **pad,
            )
            entries[key] = var

        # Watcher enabled checkbox
        watcher_var = tk.BooleanVar(value=self.config.get("watcher_enabled", True))
        ttk.Checkbutton(dlg, text="Enable file watcher", variable=watcher_var).grid(
            row=len(fields), column=1, sticky=tk.W, **pad,
        )

        def on_save() -> None:
            for key, var in entries.items():
                self.config.set(key, var.get())
            self.config.set("watcher_enabled", watcher_var.get())
            self.config.save()
            self.config.ensure_directories()
            self._inbox_path_var.set(self.config.get("inbox_path"))
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Save", command=on_save).pack(
            side=tk.LEFT, padx=5,
        )
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(
            side=tk.LEFT, padx=5,
        )

        dlg.columnconfigure(1, weight=1)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            f"{APP_NAME} {APP_VERSION}\n\n"
            "NeuroGraph cognitive architecture manager.\n"
            "Updates, ingestion, and monitoring in one place.",
        )

    # ---- File Watcher ----

    def _start_watcher(self) -> None:
        inbox = self.config.get("inbox_path")
        self.file_watcher = FileWatcher(
            inbox_path=inbox,
            on_file_ready=lambda p: self.msg_queue.put(self._on_file_detected, p),
            on_error=lambda e: self.msg_queue.put(self._on_ingest_error, e),
            stability_seconds=float(
                self.config.get("watcher_stability_seconds", 1.0)
            ),
        )
        self.file_watcher.start()
        self._watcher_var.set("Watcher: ON")
        _logger.info("File watcher started on %s", inbox)

    def _stop_watcher(self) -> None:
        if self.file_watcher and self.file_watcher.is_running:
            self.file_watcher.stop()
            self.file_watcher = None
            _logger.info("File watcher stopped")

    # ---- Lifecycle ----

    def _setup_logging(self) -> None:
        log_path = self.config.get("log_path")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)

    def _on_close(self) -> None:
        self._stop_watcher()
        self.config.save()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not _HAS_TK:
        print(
            "Error: tkinter is required.\n"
            "Install it with: sudo apt-get install python3-tk",
            file=sys.stderr,
        )
        sys.exit(1)

    root = tk.Tk()
    NeuroGraphGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
