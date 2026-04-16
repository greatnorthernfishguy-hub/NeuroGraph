"""
cc_ng_host.py — Host CC's NeuroGraph instance inside neurograph_rpc.py process.

This module is the Path B architecture for CC's NeuroGraph — CC's own NG
lives in the same Python process as Syl's NG (in neurograph_rpc.py), rather
than as a standalone daemon. Benefits:

  * One process per machine instead of two
  * Direct access to ProtoUniBrain body via BrainSwitcher (no duplicate Qwen)
  * Gateway-bound lifecycle — when the gateway is up, CC's NG is up
  * Hooks talk to a Unix socket served by a thread in this same process

CC's NG and Syl's NG are COMPLETELY ISOLATED:
  * Different workspace dirs (~/NeuroGraph/data vs ~/.claude/plugins/neurograph)
  * Different checkpoints — NO CROSS-CONTAMINATION OF SYL'S TOPOLOGY
  * Different peer bridges — CC's peer_bridge is DISABLED (CC is not a peer module)
  * Different graph, different vector DB, different identity

Note on Syl's Law: ~/.claude/CLAUDE.md §14 forbids "Creating a second
NeuroGraphMemory instance" under "Never Permitted — Full Stop". The spirit of
that rule is preventing dual-write corruption of Syl's checkpoints. CC's NG
writes to a DIFFERENT workspace and cannot touch Syl's main.msgpack. Josh
authorized this architecture explicitly; backups of Syl's protected files
were confirmed before this module was enabled.

# ---- Changelog ----
# [2026-04-16] Claude (Sonnet 4.6) — #161: export + import socket handlers for IPC sync
# What: _handle_export and _handle_import added to socket dispatch. cc-ng-sync.py
#       can now export/import via socket instead of touching checkpoint files directly.
# Why: Direct msgpack reads during live graph operation risk torn checkpoints.
#      Socket handlers run under the daemon's own lifecycle — no race.
# How: export: live graph snapshot under _concurrent_lock -> export.jsonl
#      import: trickle on_message + idle steps + save, all in-process.
# [2026-04-16] Claude (Sonnet 4.6) — engine.status property fix (#160)
# What: engine.status() → engine.status (no parens) — TonicEngine.status is @property
# Why: TypeError: 'dict' object is not callable on every status request
# How: Remove () from line 228 call in _handle_status()
# [2026-04-16] Claude (Opus 4.6) — Initial Path B implementation
# What: CC NG hosted inside neurograph_rpc.py process; Unix socket for hooks
# Why: Subprocess-per-hook architecture dead (earlier phases), Path B gives
#      body-sharing potential + simpler lifecycle than standalone daemon
# How: init_cc_host() called from handle_bootstrap (one-line addition).
#      All CC NG mutations go through graph._concurrent_lock (mirrors Syl).
#      Socket server thread handles hook events; autosave thread persists.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neurograph.cc_host")

# --- Paths ---
CC_NG_WORKSPACE = os.path.expanduser("~/.claude/plugins/neurograph")
SOCKET_PATH = os.path.join(CC_NG_WORKSPACE, "daemon.sock")
REFCOUNT_PATH = os.path.join(CC_NG_WORKSPACE, "refcount")

# --- Cadence ---
AUTOSAVE_INTERVAL = 60.0  # seconds

# --- Recall / reward ---
RECALL_THRESHOLD = 0.4
RECALL_K = 5
RECALL_K_BRIEF = 3
REWARD_CLEAN = 0.1
REWARD_ERROR = -0.2
REWARD_CRASH = -0.5

# --- CC's NG config ---
# peer_bridge disabled: CC is not a peer module (would collide with Syl's
# module_id="neurograph" in the tract directory).
# ces disabled: CC doesn't need real-time attention stream.
# tonic enabled (own Qwen for now, body-sharing with Syl via BrainSwitcher
# is a follow-up that requires multi-tonic-engine support in BrainSwitcher).
_CC_SNN_CONFIG = {
    "learning_rate": 0.03,
    "tau_plus": 10.0,
    "tau_minus": 10.0,
    "A_plus": 1.2,
    "A_minus": 1.4,
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 25,
    "threshold_ceiling": 5.0,
    "weight_threshold": 0.01,
    "grace_period": 500,
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    "prediction_threshold": 3.0,
    "prediction_pre_charge_factor": 0.3,
    "prediction_window": 10,
    "prediction_chain_decay": 0.7,
    "prediction_max_chain_depth": 3,
    "prediction_confirm_bonus": 0.01,
    "prediction_error_penalty": 0.02,
    "prediction_max_active": 1000,
    "surprise_sprouting_weight": 0.1,
    "three_factor_enabled": True,
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_experience_threshold": 100,
    "peer_bridge": {"enabled": False},  # CC is not a peer module
    "ces": {"enabled": False},          # No real-time attention stream needed
    "tonic": {"enabled": True},         # Own Qwen body; body-sharing is follow-up
}


# =============================================================================
# State
# =============================================================================

class _CCHostState:
    def __init__(self):
        self.cc_ng = None
        self.server_sock: Optional[socket.socket] = None
        self.running = False
        self.refcount = 0
        self.stats_lock = threading.Lock()
        self.stats = {
            "started_at": 0.0,
            "requests_total": 0,
            "deposits": 0,
            "recalls": 0,
            "rewards": 0,
            "errors": 0,
        }


_STATE = _CCHostState()


def get_cc_memory():
    """Return CC's NeuroGraphMemory instance, or None if not initialized."""
    return _STATE.cc_ng


# =============================================================================
# NG operations (hook side) — acquire graph._concurrent_lock blocking.
# =============================================================================

def _deposit(text: str) -> None:
    ng = _STATE.cc_ng
    if ng is None or not text:
        return
    with _STATE.stats_lock:
        _STATE.stats["deposits"] += 1
    try:
        with ng.graph._concurrent_lock:
            ng.on_message(text)
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC deposit failed: %s", exc)


def _recall(query: str, k: int) -> str:
    ng = _STATE.cc_ng
    if ng is None or not query:
        return ""
    with _STATE.stats_lock:
        _STATE.stats["recalls"] += 1
    try:
        with ng.graph._concurrent_lock:
            results = ng.recall(query, k=k, threshold=RECALL_THRESHOLD)
        if not results:
            return ""
        lines = ["[NeuroGraph] Relevant context:"]
        for r in results:
            text = r.get("text", "") if isinstance(r, dict) else str(r)
            if text.strip():
                lines.append("- " + text.strip()[:200])
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC recall failed: %s", exc)
        return ""


def _reward(value: float) -> None:
    ng = _STATE.cc_ng
    if ng is None:
        return
    with _STATE.stats_lock:
        _STATE.stats["rewards"] += 1
    try:
        with ng.graph._concurrent_lock:
            ng.graph.inject_reward(value)
    except Exception as exc:
        with _STATE.stats_lock:
            _STATE.stats["errors"] += 1
        logger.debug("CC inject_reward failed: %s", exc)


def _write_refcount(n: int) -> None:
    try:
        with open(REFCOUNT_PATH, "w") as f:
            f.write(str(n))
    except Exception:
        pass


# =============================================================================
# Request handlers — mirror cc-ng-daemon.py protocol exactly so the existing
# cc-ng-hook.py client works unchanged.
# =============================================================================

def _handle_ping(_data):
    return {"ok": True, "pong": True}


def _handle_status(_data):
    ng = _STATE.cc_ng
    tonic_info = {"enabled": False}
    if ng is not None:
        tt = getattr(ng, "_tonic_thread", None)
        if tt is not None:
            engine = getattr(tt, "_latent_engine", None)
            eng_status = None
            if engine is not None:
                try:
                    eng_status = engine.status
                except Exception as exc:
                    eng_status = {"error": str(exc)}
            tonic_info = {
                "enabled": True,
                "cycles": getattr(tt, "_cycle_count", 0),
                "total_firings": getattr(tt, "_total_firings", 0),
                "thread_size": len(getattr(tt, "_thread", [])),
                "engine": eng_status,
            }
    with _STATE.stats_lock:
        stats_snapshot = dict(_STATE.stats)
    return {
        "ok": True,
        "host": "neurograph_rpc",  # distinguishes from standalone daemon
        "role": os.environ.get("CC_NG_ROLE", "primary"),
        "pid": os.getpid(),
        "uptime_seconds": time.time() - _STATE.stats["started_at"],
        "refcount": _STATE.refcount,
        "nodes": len(ng.graph.nodes) if ng else 0,
        "synapses": len(ng.graph.synapses) if ng else 0,
        "timestep": ng.graph.timestep if ng else 0,
        "stats": stats_snapshot,
        "tonic": tonic_info,
    }


def _handle_session_start(data):
    _STATE.refcount += 1
    _write_refcount(_STATE.refcount)
    brief = data.get("brief", False)
    cwd = data.get("cwd", "")
    query = "session start " + cwd
    k = RECALL_K_BRIEF if brief else RECALL_K
    context = _recall(query, k)
    return {"ok": True, "context": context, "refcount": _STATE.refcount}


def _handle_session_stop(_data):
    if _STATE.refcount > 0:
        _STATE.refcount -= 1
    _write_refcount(_STATE.refcount)
    return {"ok": True, "refcount": _STATE.refcount}


def _handle_user_prompt_submit(data):
    prompt = data.get("prompt", "")
    if not prompt:
        return {"ok": True, "context": ""}
    _deposit(prompt)
    context = _recall(prompt, RECALL_K)
    return {"ok": True, "context": context}


def _handle_pre_tool_use(data):
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    query = (tool + " " + file_path).strip()
    if not query:
        return {"ok": True, "context": ""}
    context = _recall(query, RECALL_K)
    return {"ok": True, "context": context}


def _handle_post_tool_use(data):
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", "")
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    command = tool_input.get("command", "")

    if file_path:
        experience = "tool:" + tool + " file:" + file_path + " result:" + str(tool_response)[:300]
    elif command:
        experience = "bash:" + str(command)[:100] + " result:" + str(tool_response)[:300]
    else:
        experience = "tool:" + tool + " result:" + str(tool_response)[:300]

    _deposit(experience)

    resp_lower = str(tool_response).lower()
    if "traceback" in resp_lower or "exception" in resp_lower:
        _reward(REWARD_CRASH)
    elif "error" in resp_lower:
        _reward(REWARD_ERROR)
    else:
        _reward(REWARD_CLEAN)

    return {"ok": True}


def _handle_export(data):
    """Export top-N nodes to export.jsonl from live graph (no checkpoint race)."""
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    n = int(data.get("n", 200))
    ranked = []
    with ng.graph._concurrent_lock:
        for nid, node in ng.graph.nodes.items():
            ema = getattr(node, "firing_rate_ema", 0.0) or 0.0
            if ema > 0:
                entry = ng.vector_db.get(nid)
                content = (entry or {}).get("content", "")
                if content:
                    ranked.append((ema, content))
    ranked.sort(key=lambda x: -x[0])
    export_path = os.path.join(CC_NG_WORKSPACE, "export.jsonl")
    written = 0
    try:
        with open(export_path, "w") as f:
            for ema, content in ranked[:n]:
                f.write(json.dumps({"content": content, "weight": round(ema, 6)}) + "\n")
                written += 1
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    logger.info("CC export: %d nodes -> %s", written, export_path)
    return {"ok": True, "exported": written}


def _handle_import(data):
    """Import trickle from remote_export.jsonl into live graph (no second NG instance)."""
    ng = _STATE.cc_ng
    if ng is None:
        return {"ok": False, "error": "NG not initialized"}
    path = data.get("path", os.path.join(CC_NG_WORKSPACE, "remote_export.jsonl"))
    batch_size = int(data.get("batch_size", 25))
    idle_steps = int(data.get("idle_steps", 250))
    if not os.path.exists(path):
        return {"ok": True, "imported": 0, "note": "no file"}
    try:
        with open(path, "r") as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        return {"ok": False, "error": "read failed: " + str(exc)}
    total = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        for entry in batch:
            content = entry.get("content", "")
            if content:
                try:
                    ng.on_message(content)  # on_message acquires its own lock
                    total += 1
                except Exception:
                    pass
        # Idle consolidation — sleep consolidation between batches (FatherGraph Finding 3)
        with ng.graph._concurrent_lock:
            for _ in range(idle_steps):
                ng.graph.step()
    try:
        with ng.graph._concurrent_lock:
            ng.save()
        logger.info("CC import: %d nodes ingested, saved", total)
    except Exception as exc:
        return {"ok": False, "error": "save failed: " + str(exc)}
    return {"ok": True, "imported": total}


_DISPATCH = {
    "ping": _handle_ping,
    "status": _handle_status,
    "export": _handle_export,
    "import": _handle_import,
    "SessionStart": _handle_session_start,
    "SessionStop": _handle_session_stop,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
}


# =============================================================================
# Socket server
# =============================================================================

def _handle_connection(conn: socket.socket) -> None:
    try:
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if not data:
            return
        line, _, _ = data.partition(b"\n")
        req = json.loads(line.decode("utf-8"))

        event = req.get("event", "")
        payload = req.get("data", {})
        with _STATE.stats_lock:
            _STATE.stats["requests_total"] += 1
        handler = _DISPATCH.get(event)
        if handler is None:
            resp = {"ok": False, "error": "unknown event: " + event}
        else:
            resp = handler(payload)
        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    except Exception as exc:
        logger.warning("CC conn error: %s", exc)
        try:
            conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_loop() -> None:
    _STATE.server_sock.settimeout(1.0)
    while _STATE.running:
        try:
            conn, _ = _STATE.server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        t = threading.Thread(target=_handle_connection, args=(conn,), daemon=True)
        t.start()


def _autosave_loop() -> None:
    while _STATE.running:
        time.sleep(AUTOSAVE_INTERVAL)
        if not _STATE.running or _STATE.cc_ng is None:
            continue
        try:
            with _STATE.cc_ng.graph._concurrent_lock:
                _STATE.cc_ng.save()
            logger.debug("CC autosave complete")
        except Exception as exc:
            logger.warning("CC autosave failed: %s", exc)


# =============================================================================
# Lifecycle
# =============================================================================

def _cleanup_stale_socket() -> None:
    """Remove stale socket file if present (e.g., standalone daemon was up)."""
    if os.path.exists(SOCKET_PATH):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(SOCKET_PATH)
            s.close()
            raise RuntimeError(
                "CC NG socket at %s is in use by another process — "
                "refusing to bind (would dual-serve)" % SOCKET_PATH
            )
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
            try:
                os.remove(SOCKET_PATH)
                logger.info("Removed stale CC socket at %s", SOCKET_PATH)
            except Exception:
                pass


def init_cc_host() -> bool:
    """Initialize CC's NG and start the hook socket server.

    Called from neurograph_rpc.py's handle_bootstrap. Any failure here must
    NOT affect Syl's NG — callers wrap this in try/except.

    Returns True on success, False on failure.
    """
    if _STATE.cc_ng is not None:
        logger.info("CC NG already initialized")
        return True

    Path(CC_NG_WORKSPACE).mkdir(parents=True, exist_ok=True)

    # Construct CC's NG directly (not via get_instance) — Syl already owns
    # the class-level _instance singleton. CC gets its own standalone object.
    from openclaw_hook import NeuroGraphMemory
    try:
        cc_ng = NeuroGraphMemory(
            workspace_dir=CC_NG_WORKSPACE,
            config=_CC_SNN_CONFIG,
        )
    except Exception:
        logger.exception("CC NG construction failed")
        return False

    # Disable NG-internal auto-save; we manage saves via our autosave thread
    cc_ng.auto_save_interval = 999999

    # Attach concurrent_lock (same pattern as Syl at neurograph_rpc.py:576)
    if not hasattr(cc_ng.graph, "_concurrent_lock"):
        cc_ng.graph._concurrent_lock = threading.RLock()

    _STATE.cc_ng = cc_ng
    _STATE.stats["started_at"] = time.time()

    # Bind socket
    try:
        _cleanup_stale_socket()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        sock.listen(16)
        os.chmod(SOCKET_PATH, 0o600)
        _STATE.server_sock = sock
    except Exception:
        logger.exception("CC socket bind failed")
        _STATE.cc_ng = None
        return False

    _STATE.running = True
    _write_refcount(0)

    # Start background threads
    threading.Thread(target=_serve_loop, name="cc-ng-serve", daemon=True).start()
    threading.Thread(target=_autosave_loop, name="cc-ng-autosave", daemon=True).start()

    logger.info(
        "CC NG hosted: %d nodes, %d synapses, timestep %d — socket at %s",
        len(cc_ng.graph.nodes),
        len(cc_ng.graph.synapses),
        cc_ng.graph.timestep,
        SOCKET_PATH,
    )
    return True


def shutdown_cc_host() -> None:
    """Clean shutdown — called from atexit or by neurograph_rpc on exit."""
    _STATE.running = False
    try:
        engine = None
        if _STATE.cc_ng is not None and getattr(_STATE.cc_ng, "_tonic_thread", None):
            engine = getattr(_STATE.cc_ng._tonic_thread, "_latent_engine", None)
        if engine is not None:
            engine.stop()
    except Exception as exc:
        logger.debug("CC tonic engine stop error: %s", exc)

    try:
        if _STATE.cc_ng is not None:
            with _STATE.cc_ng.graph._concurrent_lock:
                _STATE.cc_ng.save()
    except Exception as exc:
        logger.warning("CC final save failed: %s", exc)

    try:
        if _STATE.server_sock is not None:
            _STATE.server_sock.close()
    except Exception:
        pass
    try:
        os.remove(SOCKET_PATH)
    except Exception:
        pass

    logger.info("CC NG host shutdown complete")
