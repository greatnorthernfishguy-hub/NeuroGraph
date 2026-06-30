"""
TID Substrate Peninsula — Commons side.

The Commons-half of TID's substrate peninsula. Runs in NeuroGraph's process alongside
the Commons singleton. Accepts the intra-module msgpack connection from the TID-side
peninsula, receives TID's routing-outcome deposits, forwards them to get_commons().deposit(),
and pushes enhanced topology back to TID-side after the enhance loop salts the Commons.

Intra-module IPC — LAW 1 governs inter-module communication; a module's two halves
communicating through their own private socket is not that. The Commons is never
transmitted; only thin deposit payloads and bucketed recommendations cross the boundary.

# ---- Changelog ----
# [2026-06-30] Claude Code (Sonnet 4.6) — #97 TID Commons valence routing: peninsula Commons-side
# What: New file. Commons-half of TID's substrate peninsula.
# Why: Gives TID's compute body Commons-enriched routing intelligence without a cross-module
#      bridge or a second NGLite instance in TID's process. One Commons, two address spaces —
#      the peninsula IS TID's substrate participation, not a parallel substrate.
# How: Unix socket server (path: TID_PENINSULA_SOCK env var, LAW 5). Threading daemon,
#      consistent with NG's existing pulse threads. 4-byte length-prefixed msgpack framing.
#      Receive loop: TID deposit → get_commons().deposit(). push_enhanced() called from
#      _run_commons_enhance_scoop() after each enhance cycle to bucket fresh enhanced content
#      and send to TID-side. One direction at a time (send lock).
# -------------------
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np

logger = logging.getLogger("ng.tid_peninsula")

# LAW 5 — socket path from env, never hardcoded.
_SOCK_PATH: str = os.environ.get("TID_PENINSULA_SOCK", "/tmp/tid-peninsula.sock")
# How many enhanced recs to push back per cycle.
_PUSH_TOP_K: int = int(os.environ.get("TID_PENINSULA_PUSH_TOP_K", "10"))


def _recv_frame(conn: socket.socket) -> Optional[bytes]:
    """Read one length-prefixed msgpack frame. Returns None on EOF/error."""
    try:
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        length = struct.unpack(">I", header)[0]
        body = b""
        while len(body) < length:
            chunk = conn.recv(length - len(body))
            if not chunk:
                return None
            body += chunk
        return body
    except OSError:
        return None


def _send_frame(conn: socket.socket, payload: bytes) -> bool:
    """Send one length-prefixed msgpack frame. Returns False on error."""
    try:
        header = struct.pack(">I", len(payload))
        conn.sendall(header + payload)
        return True
    except OSError:
        return False


class TIDPeninsulaCommons:
    """Commons-side half of TID's substrate peninsula.

    One instance, created at NG startup. Manages the persistent socket connection to
    the TID-side half. Thread-safe — the server thread and the push path (called from
    the scan-drain pulse thread) both hold _send_lock before writing to the socket.
    """

    def __init__(self) -> None:
        self._client: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._last_push_ts: float = 0.0  # watermark — only push content newer than last push

    def start(self) -> None:
        """Start the socket server in a daemon thread. Idempotent."""
        if self._server_thread is not None and self._server_thread.is_alive():
            return
        self._server_thread = threading.Thread(
            target=self._serve,
            name="tid-peninsula-commons",
            daemon=True,
        )
        self._server_thread.start()
        logger.info("TID peninsula (Commons-side) started on %s", _SOCK_PATH)

    def _serve(self) -> None:
        """Accept connections from TID-side, one at a time. Loops on disconnect."""
        # Remove stale socket file from a previous run.
        try:
            os.unlink(_SOCK_PATH)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("TID peninsula: could not remove stale socket: %s", exc)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(_SOCK_PATH)
            srv.listen(1)
            srv.settimeout(2.0)  # allow clean shutdown checks
        except OSError as exc:
            logger.error("TID peninsula: socket bind failed: %s", exc)
            return

        logger.info("TID peninsula: listening for TID-side connection")
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                logger.debug("TID peninsula: accept error: %s", exc)
                break

            logger.info("TID peninsula: TID-side connected")
            with self._send_lock:
                self._client = conn
            try:
                self._handle_client(conn)
            finally:
                with self._send_lock:
                    if self._client is conn:
                        self._client = None
                try:
                    conn.close()
                except OSError:
                    pass
                logger.info("TID peninsula: TID-side disconnected; waiting for reconnect")

    def _handle_client(self, conn: socket.socket) -> None:
        """Receive deposits from TID-side and forward to the Commons."""
        while True:
            raw = _recv_frame(conn)
            if raw is None:
                break
            try:
                msg: Dict[str, Any] = msgpack.unpackb(raw, raw=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("TID peninsula: bad msgpack frame: %s", exc)
                continue

            if msg.get("type") != "deposit":
                logger.debug("TID peninsula: unknown message type: %s", msg.get("type"))
                continue

            self._forward_deposit(msg)

    def _forward_deposit(self, msg: Dict[str, Any]) -> None:
        """Deposit TID's routing outcome into the Commons. Fail-soft."""
        try:
            from commons import get_commons
            commons = get_commons()
            if commons is None:
                return

            emb_list = msg.get("embedding")
            target_id = msg.get("target_id", "")
            success = bool(msg.get("success", True))
            strength = float(msg.get("strength", 1.0))
            metadata = msg.get("metadata") or {}

            if not emb_list or not target_id:
                logger.debug("TID peninsula: deposit missing embedding or target_id")
                return

            embedding = np.array(emb_list, dtype=np.float32)
            commons.deposit(
                embedding=embedding,
                target_id=target_id,
                success=success,
                strength=strength,
                metadata=metadata,
            )
            logger.debug("TID peninsula: deposited %s (success=%s)", target_id, success)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TID peninsula: deposit failed: %s", exc)

    def push_enhanced(self) -> None:
        """Bucket recently-enhanced content from the Commons and send to TID-side.

        Called from _run_commons_enhance_scoop() after each enhance cycle. If no TID-side
        is connected, returns silently. Fail-soft — a push failure never breaks the pulse.
        """
        with self._send_lock:
            client = self._client
        if client is None:
            return

        try:
            from commons import get_commons
            commons = get_commons()
            if commons is None:
                return

            # Bucket enhanced deposits newer than the last push (watermark).
            # bucket_recent sorted desc + early-break makes this O(new content), not O(all).
            # "enhanced:" prefix is set by CommonsEnhancer on its return deposits.
            rows: List[Tuple] = commons.bucket_recent(
                limit=_PUSH_TOP_K,
                since=self._last_push_ts,
                with_metadata=True,
            )
            recs = []
            for row in rows:
                target_id = row[0]
                weight = float(row[1])
                reasoning = row[2] if len(row) > 2 else ""
                if not target_id.startswith("enhanced:"):
                    continue
                recs.append([target_id, weight, reasoning or ""])

            if not recs:
                return

            payload = msgpack.packb({"type": "enhanced", "recs": recs}, use_bin_type=True)
            with self._send_lock:
                if self._client is not client:
                    return  # client changed between check and send
                ok = _send_frame(self._client, payload)
            if ok:
                self._last_push_ts = time.time()
            else:
                logger.debug("TID peninsula: push_enhanced send failed (client gone)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("TID peninsula: push_enhanced failed: %s", exc)


# Module-level singleton — created at NeuroGraph startup via start_tid_peninsula().
_peninsula: Optional[TIDPeninsulaCommons] = None


def start_tid_peninsula() -> None:
    """Create and start the TID peninsula Commons-side. Called once at NG startup."""
    global _peninsula
    if _peninsula is not None:
        return
    _peninsula = TIDPeninsulaCommons()
    _peninsula.start()


def tid_peninsula_push_enhanced() -> None:
    """Push fresh enhanced recs to TID-side. Called after each enhance scoop."""
    if _peninsula is not None:
        _peninsula.push_enhanced()
