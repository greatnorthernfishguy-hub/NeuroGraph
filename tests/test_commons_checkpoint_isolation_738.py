"""
Regression guard — lane commons-test-isolation-738 (rows #738/#742).

# ---- Changelog ----
# [2026-09-30] worker-001 (Claude Sonnet 5) — lane commons-test-isolation-738, rows #738/#742
# What: New file. Proves (1) the default commons checkpoint path is unchanged when
#       NG_COMMONS_CHECKPOINT_PATH is unset, (2) the env var override is honored at
#       import time (subprocess, controlled env, no live graph started), and (3) a
#       handle_after_turn-driven auto-save persist lands under a redirected tmp path
#       and never touches the real ~/NeuroGraph/data/checkpoints/commons.msgpack.
# Why: #738 — a standalone run of tests/test_coordinator.py wrote the REAL commons
#       checkpoint because _COMMONS_CHECKPOINT_PATH was a hardcoded module constant
#       with no override channel, and _last_save_time inits to 0.0 so the very first
#       afterTurn in a fresh process hits the time-based auto-save trigger. This file
#       is the guard that would have caught it before it ever reached a real run.
# How: (1)/(2) run `python3 -c "import neurograph_rpc; ..."` in a subprocess with a
#      controlled, minimal environment so the module-level `os.environ.get(...)` read
#      is genuinely exercised (not just re-read in-process, where a prior import could
#      already be cached in sys.modules). (3) drives the real, public
#      neurograph_rpc.handle_after_turn() with a minimal hand-built fake `_memory` (not
#      a MagicMock — magic-method auto-mocking made several branches non-deterministic
#      in a spike of this test) and monkeypatches out the River/tract/embedding side
#      helpers that are irrelevant to the checkpoint-path contract under test, so the
#      test stays cheap, deterministic, and offline (no embedding model load, no real
#      tract file writes). Real-path safety is asserted both directions: the redirected
#      tmp file exists and is non-trivial, and the real file's stat/sha256 are unchanged
#      across the whole test.
# -------------------
"""

import hashlib
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import neurograph_rpc
import commons as commons_mod

REAL_COMMONS_PATH = os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack")


def _real_file_fingerprint():
    """(exists, size, mtime_ns, sha256) for the REAL checkpoint — never written by this file."""
    if not os.path.exists(REAL_COMMONS_PATH):
        return (False, None, None, None)
    st = os.stat(REAL_COMMONS_PATH)
    with open(REAL_COMMONS_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return (True, st.st_size, st.st_mtime_ns, digest)


def _subprocess_env(extra=None):
    """A controlled, minimal env for the import-time subprocess checks below.

    Deliberately NOT a copy of the parent env — the whole point of these two tests is
    to prove what the module resolves to under a *specific* env, so an accidentally
    inherited NG_COMMONS_CHECKPOINT_PATH (or NG_EMBED_REMOTE, per #732 discipline) must
    not leak in. PATH/HOME/PYTHONPATH are the only carry-overs needed to run python3
    and resolve `~` and local imports at all.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", os.path.expanduser("~")),
        "PYTHONPATH": REPO_ROOT,
    }
    if extra:
        env.update(extra)
    return env


class TestDefaultPathUnchangedWhenEnvUnset:
    def test_resolves_to_prior_hardcoded_default(self):
        before = _real_file_fingerprint()

        proc = subprocess.run(
            [sys.executable, "-c", "import neurograph_rpc; print(neurograph_rpc._COMMONS_CHECKPOINT_PATH)"],
            cwd=REPO_ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, f"import failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        resolved = proc.stdout.strip().splitlines()[-1]
        assert resolved == REAL_COMMONS_PATH == os.path.expanduser(
            "~/NeuroGraph/data/checkpoints/commons.msgpack"
        )
        # Import alone must never touch the real file.
        assert _real_file_fingerprint() == before


class TestEnvOverrideHonored:
    def test_override_is_honored_at_import_time(self, tmp_path):
        before = _real_file_fingerprint()
        override = str(tmp_path / "override-commons.msgpack")

        proc = subprocess.run(
            [sys.executable, "-c", "import neurograph_rpc; print(neurograph_rpc._COMMONS_CHECKPOINT_PATH)"],
            cwd=REPO_ROOT,
            env=_subprocess_env({"NG_COMMONS_CHECKPOINT_PATH": override}),
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, f"import failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        resolved = proc.stdout.strip().splitlines()[-1]
        assert resolved == override
        assert resolved != REAL_COMMONS_PATH
        assert _real_file_fingerprint() == before


class _FakeGraph:
    """Just enough surface for handle_after_turn's happy path — no real SNN, no embeddings."""

    def __init__(self):
        self.config = {}          # three_factor_enabled defaults False -> skip inject_reward
        self.nodes = {}
        self.timestep = 0

    def step(self):
        self.timestep += 1
        return _StepResult()

    def discover_hyperedges(self, fired_node_ids):
        return []

    def get_stats(self):
        return {}


class _StepResult:
    predictions_confirmed = 0
    predictions_surprised = 0
    fired_node_ids = []


class _FakeMemory:
    """Minimal stand-in for NeuroGraphMemory — cheap, offline, no checkpoint I/O of its own."""

    def __init__(self):
        self.graph = _FakeGraph()
        self._message_count = 10
        self.auto_save_interval = 10   # count_trigger True deterministically (no wall-clock dependency)
        self._substrate_novelty_ema = 0.0
        self._surfacing_monitor = None
        self.saved = False

    def save(self):
        self.saved = True


@pytest.fixture
def _isolated_commons_env(tmp_path, monkeypatch):
    """Redirect the checkpoint path + give handle_after_turn a clean, fast path through
    the parts of the function that are not the subject of this test (River/tract deposits,
    conversational filing, embeddings, probation, pass-2 retries). None of those are what
    #738 was about; leaving them live would make this test slow, non-deterministic, and
    liable to write real tract files outside this lane's declared scope.
    """
    redirected = tmp_path / "commons.msgpack"
    monkeypatch.setattr(neurograph_rpc, "_COMMONS_CHECKPOINT_PATH", str(redirected))
    monkeypatch.setattr(neurograph_rpc, "_last_save_time", 0.0)
    monkeypatch.setattr(neurograph_rpc, "_tract", None)
    monkeypatch.setattr(neurograph_rpc, "_lenia_kill_switch", None)
    monkeypatch.setattr(neurograph_rpc, "_lenia_engine", None)

    for name in (
        "_drain_peer_tracts",
        "_deposit_topology_to_river",
        "_deposit_experience_to_river",
        "_file_conversational_experience",
        "_deposit_surfacing_outcome",
        "_check_outbound_intent",
        "_drain_pass2_retries",
        "_update_probation",
        "_anticipate",
    ):
        monkeypatch.setattr(neurograph_rpc, name, lambda *a, **k: None)

    orig_commons_singleton = commons_mod._commons
    commons_mod._commons = None
    yield redirected
    commons_mod._commons = orig_commons_singleton


class TestHandleAfterTurnPersistLandsUnderRedirectedPath:
    def test_persist_writes_redirected_file_and_never_touches_real_path(
        self, _isolated_commons_env, monkeypatch
    ):
        redirected = _isolated_commons_env
        before = _real_file_fingerprint()

        fake_memory = _FakeMemory()
        monkeypatch.setattr(neurograph_rpc, "_memory", fake_memory)
        monkeypatch.setattr(neurograph_rpc, "_ingest_text", None)
        monkeypatch.setattr(neurograph_rpc, "_ingest_embedding", None)

        assert not redirected.exists()

        neurograph_rpc.handle_after_turn({"lastUserMessage": {"content": "regression guard 738"}})

        assert fake_memory.saved is True, "the memory-side save() must still run"
        assert redirected.exists(), "commons persist must land under the redirected tmp path"
        assert redirected.stat().st_size > 0

        after = _real_file_fingerprint()
        assert after == before, (
            "the real commons checkpoint must be byte-identical before/after a "
            "handle_after_turn-driven auto-save — this is the exact failure mode of #738"
        )


class TestNegativeControlAgainstPreFixLine:
    """Actually executes the pre-fix constant expression (not a re-import of the full,
    now-fixed neurograph_rpc.py — that module no longer contains the bug) as a standalone
    scratch script OUTSIDE the repo tree, with HOME pointed at a throwaway scratch
    directory so it can never resolve into the real ~/NeuroGraph path even by accident.

    This empirically shows the pre-fix line — `os.path.expanduser("~/NeuroGraph/data/
    checkpoints/commons.msgpack")`, no env read — ignores NG_COMMONS_CHECKPOINT_PATH
    entirely and resolves relative to $HOME no matter what override a test sets. That is
    exactly the mechanism of #738: on a real machine HOME is the real home directory, so
    that unconditional resolution IS the real checkpoint path, and no test-side fixture
    (tmp_path, monkeypatch, env var) could ever have redirected it. The fixed line, run
    the same way, honors the override. The contrast is the guard.
    """

    _PRE_FIX_LINE = (
        'import os; '
        'print(os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack"))'
    )
    _FIXED_LINE = (
        'import os; '
        'print(os.path.expanduser(os.environ.get('
        '"NG_COMMONS_CHECKPOINT_PATH", "~/NeuroGraph/data/checkpoints/commons.msgpack")))'
    )

    def test_prefix_line_ignores_override_fixed_line_honors_it(self, tmp_path):
        scratch_home = tmp_path / "scratch_home"
        scratch_home.mkdir()
        scratch_script_dir = tmp_path / "scratch_outside_repo"
        scratch_script_dir.mkdir()
        redirect_target = str(tmp_path / "redirected-commons.msgpack")

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(scratch_home),
            "NG_COMMONS_CHECKPOINT_PATH": redirect_target,
        }

        real_before = _real_file_fingerprint()

        pre_fix = subprocess.run(
            [sys.executable, "-c", self._PRE_FIX_LINE],
            cwd=str(scratch_script_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        fixed = subprocess.run(
            [sys.executable, "-c", self._FIXED_LINE],
            cwd=str(scratch_script_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert pre_fix.returncode == 0 and fixed.returncode == 0
        pre_fix_resolved = pre_fix.stdout.strip()
        fixed_resolved = fixed.stdout.strip()

        # Pre-fix: ignores the override, resolves under scratch $HOME (on a real
        # machine this would be the real ~/NeuroGraph — the actual #738 mechanism).
        assert pre_fix_resolved == str(scratch_home / "NeuroGraph/data/checkpoints/commons.msgpack")
        assert pre_fix_resolved != redirect_target

        # Fixed: honors the override, unconditionally.
        assert fixed_resolved == redirect_target

        # Neither subprocess touched anything under scratch_home or the real path
        # (these are print-only expression evaluations, but assert it rather than assume it).
        assert not (scratch_home / "NeuroGraph").exists()
        assert _real_file_fingerprint() == real_before
