# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1, #1184) — cc_ng_host releases _concurrent_lock after capture
# What: _save_releasing_after_capture against fake NeuroGraphMemory objects (with and without after_capture).
# Why:  #1161 / #1123 port: the hosted CC mind's autosave must not hold the lock through the write, and must still
#       work if the PROTECTED openclaw_hook commit is not installed.
# How:  AST-extract the two helpers (no NG/model imports); another thread probes the lock during the "write".
# -------------------
import ast
import threading
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def _helpers():
    tree = ast.parse((ROOT / "cc_ng_host.py").read_text())
    want = {"_save_accepts_after_capture", "_save_releasing_after_capture"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert len(nodes) == 2
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "cc_ng_host.py", "exec"), ns)
    return ns["_save_releasing_after_capture"]


def _lock_free_from_other_thread(lock) -> bool:
    out = []
    t = threading.Thread(target=lambda: out.append(lock.acquire(timeout=1) and (lock.release() or True)))
    t.start(); t.join()
    return out[0]


def test_lock_released_after_capture_before_write():
    save = _helpers()
    lock = threading.RLock()
    seen = {}

    class NG:
        graph = NS(_concurrent_lock=lock)

        def save(self, *, with_receipt=False, after_capture=None):
            seen["held_during_capture"] = not _lock_free_from_other_thread(lock)
            after_capture()
            seen["free_during_write"] = _lock_free_from_other_thread(lock)
            return "ok"

    assert save(NG()) == "ok"
    assert seen == {"held_during_capture": True, "free_during_write": True}
    assert _lock_free_from_other_thread(lock)


def test_raise_before_callback_still_releases():
    save = _helpers()
    lock = threading.RLock()

    class NG:
        graph = NS(_concurrent_lock=lock)

        def save(self, *, after_capture=None):
            raise RuntimeError("refused")

    try:
        save(NG())
    except RuntimeError:
        pass
    assert _lock_free_from_other_thread(lock)


def test_old_save_without_after_capture_holds_lock_whole_save():
    save = _helpers()
    lock = threading.RLock()
    seen = {}

    class NG:
        graph = NS(_concurrent_lock=lock)

        def save(self):
            seen["held"] = not _lock_free_from_other_thread(lock)
            return "ok"

    assert save(NG()) == "ok"
    assert seen == {"held": True}
    assert _lock_free_from_other_thread(lock)
