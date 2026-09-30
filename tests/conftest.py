"""Repo-wide pytest guard — lane commons-test-isolation-738, ADDENDUM A (#738/#754,
Executive Packet 373).

# ---- Changelog ----
# [2026-09-30] worker-001 (Claude Sonnet 5) — ADDENDUM A, Executive Packet 373 widening
# What: New file. (A1) Patches commons.Commons.persist at collection time so any persist
#       whose target resolves to the literal real checkpoint
#       ~/NeuroGraph/data/checkpoints/commons.msgpack raises, records the offending test
#       nodeid, and fails the pytest session at the end if any block was left unresolved
#       (handle_after_turn swallows the raise in a bare except, so the raise alone would
#       pass silently — see neurograph_rpc.py:3796-3800 in this lane's first commit).
#       (A2) Registers the `slow` and `timeout` markers via pytest_configure so an
#       environment without pytest-timeout installed does not warn/error on
#       tests/test_snn.py's new markers.
# Why: #738 recurred from ANOTHER seat's unguarded pytest run after this lane's per-file
#      fixture fix landed — a per-run fixture only protects the file it's defined in. The
#      guard needs to live in the repo (tests/conftest.py loads for every test file in this
#      directory) so it applies regardless of which test file or seat runs.
# How: Class-level monkeypatch of commons.Commons.persist (no restoration — this file only
#      loads under pytest, never in a production import of commons.py). The protected path
#      is computed from the literal default string, never from NG_COMMONS_CHECKPOINT_PATH
#      or neurograph_rpc's constant, so the env-var override this lane's (a) added cannot
#      defeat the guard. Blocked attempts are recorded in a module-level list keyed by the
#      current test's nodeid (tracked via an autouse fixture); pytest_sessionfinish inspects
#      that list and flips session.exitstatus nonzero if anything is still recorded. The
#      record is resettable (reset_blocked_persist_attempts()) specifically so this file's
#      own guard test can provoke a block, assert it was recorded, and clear it before the
#      real session ends.
# -------------------
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Literal default — deliberately NOT read from neurograph_rpc._COMMONS_CHECKPOINT_PATH or
# NG_COMMONS_CHECKPOINT_PATH, so the env-var override added in this lane's (a) cannot be
# used to defeat this guard.
_REAL_COMMONS_PATH = os.path.realpath(
    os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack")
)

_blocked_persist_attempts = []  # list[(nodeid, resolved_path)]
_current_test_nodeid = [None]   # single-slot holder, set by the autouse tracker fixture below

_GUARD_STATUS = {"installed": False, "reason": None}


def reset_blocked_persist_attempts():
    """Test-facing hook — clears the block record.

    Exists so tests/test_commons_checkpoint_isolation_738.py can provoke a block (proving
    the guard fires and is recorded) and then clear it, so exercising the guard does not
    itself fail the real pytest session.
    """
    _blocked_persist_attempts.clear()


def get_blocked_persist_attempts():
    """Test-facing hook — read-only snapshot of the current block record."""
    return list(_blocked_persist_attempts)


def real_commons_path():
    """Test-facing hook — the literal protected path this guard computed at import time."""
    return _REAL_COMMONS_PATH


@pytest.fixture(autouse=True)
def _track_current_test_nodeid(request):
    _current_test_nodeid[0] = request.node.nodeid
    yield
    _current_test_nodeid[0] = None


def pytest_configure(config):
    # A2 (#754): register both markers so @pytest.mark.slow / @pytest.mark.timeout(...) in
    # tests/test_snn.py never produce PytestUnknownMarkWarning, whether or not the
    # pytest-timeout plugin (which would otherwise register `timeout` itself) is installed.
    config.addinivalue_line(
        "markers", "slow: marks a test as slow/long-running (row #754) — not selected by default runs"
    )
    config.addinivalue_line(
        "markers",
        "timeout: pytest-timeout per-test timeout marker, registered here so environments "
        "without pytest-timeout installed do not warn/error on it",
    )

    # A1: repo-wide persist guard.
    try:
        import commons
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        _GUARD_STATUS["installed"] = False
        _GUARD_STATUS["reason"] = f"commons module not importable: {exc!r}"
        print(
            f"[conftest #738 guard] commons not importable ({exc!r}) — "
            "repo-wide persist guard DISABLED for this session",
            file=sys.stderr,
        )
        return

    if getattr(commons.Commons.persist, "_is_738_guard", False):
        # Already patched (e.g. conftest re-executed in-process by a nested pytest run) —
        # do not double-wrap.
        _GUARD_STATUS["installed"] = True
        return

    _orig_persist = commons.Commons.persist

    def _guarded_persist(self, filepath):
        resolved = os.path.realpath(os.path.expanduser(filepath))
        if resolved == _REAL_COMMONS_PATH:
            _blocked_persist_attempts.append((_current_test_nodeid[0], resolved))
            raise RuntimeError(
                f"#738 repo-wide guard (tests/conftest.py): refused Commons.persist() to "
                f"the REAL checkpoint {resolved} from test {_current_test_nodeid[0]!r}. "
                "This must never be written by a test run."
            )
        return _orig_persist(self, filepath)

    _guarded_persist._is_738_guard = True
    commons.Commons.persist = _guarded_persist
    _GUARD_STATUS["installed"] = True


def pytest_sessionfinish(session, exitstatus):
    if _blocked_persist_attempts:
        print("\n" + "=" * 78, file=sys.stderr)
        print(
            "#738 REPO-WIDE GUARD: blocked Commons.persist() attempt(s) to the REAL "
            "commons checkpoint that were never cleared:",
            file=sys.stderr,
        )
        for nodeid, path in _blocked_persist_attempts:
            print(f"  - {nodeid!r} attempted persist to {path}", file=sys.stderr)
        print(
            "handle_after_turn swallows this RuntimeError in a bare except (neurograph_rpc.py "
            "~:3796-3800), so without this sessionfinish check the offending test would have "
            "passed silently. Failing the session instead.",
            file=sys.stderr,
        )
        print("=" * 78, file=sys.stderr)
        session.exitstatus = 1
