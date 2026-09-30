# ---- Changelog ----
# [2026-09-30] Chief-003 / Claude — Test suite for stricter Syl's Law hook (Exec Packets 442-443)
# What: Comprehensive pytest suite for the v3 worktree-aware pretool_syls_law hook.
# Why: Punchlist #842 — the hook had no tests. Differential proof required to guarantee the
#      new hook is strictly tighter (never looser). Cover worktree paths, vendored list
#      corrections, RETIRED set, bypass mechanism and pty interaction.
# How: Fake HOME with a real git NeuroGraph repo and worktrees. Run base and new hooks as
#      subprocesses. Differential proof over a path corpus. Bypass tested by file creation.
#      Pty tests use os.forkpty() for interactive prompt.
# -------------------

import os
import sys
import json
import time
import fcntl
import shutil
import pytest
import pty as pty_module
import subprocess
import tempfile


BASE_HOOK = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_syls_law_base.sh")
NEW_HOOK = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_syls_law.sh")

PROTECTED_RELS = [
    "data/checkpoints/main.msgpack",
    "data/checkpoints/vectors.msgpack",
    "data/checkpoints/main.msgpack.activations.json",
    "neuro_foundation.py",
    "openclaw_hook.py",
    "stream_parser.py",
    "activation_persistence.py",
    "ng_lite.py",
    "ng_tract_bridge.py",
    "ng_ecosystem.py",
    "ng_autonomic.py",
    "openclaw_adapter.py",
    "ng_embed.py",
    "ng_salience_gate.py",
    "ng_updater.py",
    "ng_peer_bridge.py",
]


class TestSylsLawHook:
    """Comprehensive test suite for the Syl's Law PreToolUse hook (v3)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, request):
        """Create a fake HOME with a real git NeuroGraph repo, plus worktrees and unrelated repos."""
        self._tmpdir = tempfile.mkdtemp(prefix="syls_law_hook_test_")
        self._fake_home = os.path.join(self._tmpdir, "home")
        os.makedirs(self._fake_home)

        # --- NeuroGraph repo ---
        ng_dir = os.path.join(self._fake_home, "NeuroGraph")
        os.makedirs(ng_dir)
        subprocess.run(
            ["git", "init", "-b", "main"], cwd=ng_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "remote", "add", "origin",
             "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git"],
            cwd=ng_dir, capture_output=True, check=True,
        )

        for rel in PROTECTED_RELS:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write("protected\n")

        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write("unprotected\n")

        ckpt_old = os.path.join(ng_dir, "data", "checkpoints-old", "stale.txt")
        os.makedirs(os.path.dirname(ckpt_old), exist_ok=True)
        with open(ckpt_old, "w") as fh:
            fh.write("stale\n")

        subprocess.run(["git", "add", "-A"], cwd=ng_dir, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test",
             "commit", "-m", "init"],
            cwd=ng_dir, capture_output=True, check=True,
        )

        # --- Worktree outside fake home ---
        self._wt_outside = os.path.join(self._tmpdir, "worktree_outside")
        subprocess.run(
            ["git", "worktree", "add", "--detach", self._wt_outside],
            cwd=ng_dir, capture_output=True, check=True,
        )

        # --- Worktree nested inside fake home ---
        self._wt_inside = os.path.join(self._fake_home, "worktree_inside")
        subprocess.run(
            ["git", "worktree", "add", "--detach", self._wt_inside],
            cwd=ng_dir, capture_output=True, check=True,
        )

        # --- Unrelated repo (different origin) ---
        self._unrelated = os.path.join(self._tmpdir, "unrelated_repo")
        os.makedirs(self._unrelated)
        subprocess.run(
            ["git", "init", "-b", "main"], cwd=self._unrelated, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "remote", "add", "origin",
             "https://github.com/someone/other-repo.git"],
            cwd=self._unrelated, capture_output=True, check=True,
        )
        with open(os.path.join(self._unrelated, "neuro_foundation.py"), "w") as fh:
            fh.write("unrelated\n")
        subprocess.run(["git", "add", "-A"], cwd=self._unrelated, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test",
             "commit", "-m", "init"],
            cwd=self._unrelated, capture_output=True, check=True,
        )

        # --- Non-git directory ---
        self._non_git = os.path.join(self._tmpdir, "non_git")
        os.makedirs(self._non_git)

        self._ng_dir = ng_dir

        yield

        # --- Cleanup ---
        for wt in [self._wt_outside, self._wt_inside]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=self._ng_dir, capture_output=True,
            )
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _base_env(self):
        return {
            "HOME": self._fake_home,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

    def _run_hook_no_tty(self, hook_path, file_path, extra_env=None):
        """Run hook in a session with no controlling terminal.
        Returns (exit_code, stderr_text)."""
        env = self._base_env()
        if extra_env:
            env.update(extra_env)
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path],
            input=tool_input.encode(),
            capture_output=True,
            timeout=15,
            start_new_session=True,
            env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_hook_pty(self, hook_path, file_path, choice, extra_env=None):
        """Run hook inside a pty, send JSON then the interactive choice.
        Returns (exit_code, combined_output)."""
        env = self._base_env()
        if extra_env:
            env.update(extra_env)
        tool_input_json = json.dumps({"tool_input": {"file_path": file_path}})

        pid, master_fd = pty_module.fork()
        if pid == 0:
            # Child
            for k, v in env.items():
                os.environ[k] = v
            os.execv(hook_path, [hook_path])
            os._exit(127)

        # Parent — write JSON + newline + EOF marker so `cat` completes
        os.write(master_fd, (tool_input_json + "\n\x04").encode())

        output = b""
        choice_sent = False
        deadline = time.time() + 10

        while time.time() < deadline:
            try:
                data = os.read(master_fd, 4096)
                if data:
                    output += data
                if not choice_sent and b"Choice [1/2/3]:" in output:
                    os.write(master_fd, (str(choice) + "\n").encode())
                    choice_sent = True
                    time.sleep(0.4)
            except BlockingIOError:
                if choice_sent:
                    time.sleep(0.2)
                    try:
                        wpid, _status = os.waitpid(pid, os.WNOHANG)
                        if wpid != 0:
                            break
                    except ChildProcessError:
                        break
                else:
                    time.sleep(0.05)
            except OSError:
                break

        # Drain any remaining output
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                data = os.read(master_fd, 4096)
                if not data:
                    break
                output += data
        except (BlockingIOError, OSError):
            pass

        try:
            _, status = os.waitpid(pid, 0)
            exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        except ChildProcessError:
            exit_code = -1

        os.close(master_fd)
        return exit_code, output.decode("utf-8", errors="replace")

    # ── differential proof ───────────────────────────────────────────────────

    def test_differential_proof(self):
        """Core: base hook fires => new hook fires; new additions are exactly expected.

        Generates a path corpus covering: all protected files in $HOME/NeuroGraph,
        all protected files in worktrees (absolute and relative via CLAUDE_PROJECT_DIR),
        unprotected files, unrelated repos, non-git directories, .. forms, and
        checkpoint-directory sibling names.
        """
        corpus = []

        # All protected files — literal $HOME/NeuroGraph, worktree outside, worktree inside,
        # relative with CLAUDE_PROJECT_DIR pointing at each
        for rel in PROTECTED_RELS:
            corpus.append(("home", os.path.join(self._ng_dir, rel), {}))
            corpus.append(("wt_out", os.path.join(self._wt_outside, rel), {}))
            corpus.append(("wt_in", os.path.join(self._wt_inside, rel), {}))
            corpus.append(("rel_wt_out", rel, {"CLAUDE_PROJECT_DIR": self._wt_outside}))
            corpus.append(("rel_wt_in", rel, {"CLAUDE_PROJECT_DIR": self._wt_inside}))
            corpus.append(("rel_main", rel, {"CLAUDE_PROJECT_DIR": self._ng_dir}))

        # Checkpoint directory — a file inside (prefix match)
        corpus.append(("home_ckpt_sub", os.path.join(self._ng_dir, "data", "checkpoints", "subfile.msgpack"), {}))
        corpus.append(("home_ckpt_dir", os.path.join(self._ng_dir, "data", "checkpoints"), {}))
        corpus.append(("wt_ckpt_sub", os.path.join(self._wt_outside, "data", "checkpoints", "subfile.msgpack"), {}))

        # Sibling directory: data/checkpoints-old (old prefix test matches "checkpoints*")
        corpus.append(("home_ckpt_old", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))

        # Unprotected files — $HOME/NeuroGraph and worktrees
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            corpus.append(("unprot_home", os.path.join(self._ng_dir, rel), {}))
            corpus.append(("unprot_wt", os.path.join(self._wt_outside, rel), {}))

        # .. traversal
        corpus.append(("dotdot", os.path.join(self._ng_dir, "subdir", "..", "neuro_foundation.py"), {}))

        # Unrelated repo — same filename, different origin
        corpus.append(("unrelated", os.path.join(self._unrelated, "neuro_foundation.py"), {}))

        # Non-git directory
        corpus.append(("non_git", os.path.join(self._non_git, "neuro_foundation.py"), {}))

        # Non-existent path under NeuroGraph
        corpus.append(("nonexistent", os.path.join(self._ng_dir, "data", "checkpoints", "future.msgpack"), {}))

        base_fires = set()
        new_fires = set()

        for label, path, extra in corpus:
            base_exit, _ = self._run_hook_no_tty(BASE_HOOK, path, extra_env=extra or None)
            new_exit, _ = self._run_hook_no_tty(NEW_HOOK, path, extra_env=extra or None)

            if base_exit != 0:
                base_fires.add((label, path))
            if new_exit != 0:
                new_fires.add((label, path))

        # ── Monotonicity: no path where base fires and new does not ──
        base_only = base_fires - new_fires
        assert not base_only, f"Base fires but new does NOT for: {sorted(base_only)}"

        # ── Expected new-only additions ──
        expected_new = set()

        for rel in PROTECTED_RELS:
            expected_new.add(("wt_out", os.path.join(self._wt_outside, rel)))
            expected_new.add(("wt_in", os.path.join(self._wt_inside, rel)))
            expected_new.add(("rel_wt_out", rel))
            expected_new.add(("rel_wt_in", rel))
            # rel_main NOT new: CLAUDE_PROJECT_DIR=self._ng_dir resolves to
            # $HOME/NeuroGraph/<rel> which the base hook catches via literal match.

        # Newly covered vendored in $HOME (were missing from old list)
        for rel in ["ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"]:
            expected_new.add(("home", os.path.join(self._ng_dir, rel)))
            # These are also new for rel_main (CLAUDE_PROJECT_DIR -> $HOME/NeuroGraph resolves
            # via literal matching on the expanded VENDORED_CANONICAL list, which the base
            # hook does not have).
            expected_new.add(("rel_main", rel))

        # Worktree checkpoint dir match
        expected_new.add(("wt_ckpt_sub", os.path.join(self._wt_outside, "data", "checkpoints", "subfile.msgpack")))

        actual_new = new_fires - base_fires

        missing = expected_new - actual_new
        unexpected = actual_new - expected_new

        assert not missing, f"Expected new additions NOT found: {sorted(missing)}"
        assert not unexpected, f"Unexpected new additions found: {sorted(unexpected)}"

        print(
            f"\nCorpus: {len(corpus)} paths | "
            f"base fires: {len(base_fires)} | "
            f"new fires: {len(new_fires)} | "
            f"new additions: {len(actual_new)}"
        )

    # ── old literal still fires ──────────────────────────────────────────────

    def test_old_literal_all_protected_still_fire(self):
        """Every protected path that fired under $HOME/NeuroGraph still fires."""
        for rel in PROTECTED_RELS:
            path = os.path.join(self._ng_dir, rel)
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, path)
            assert exit_code != 0, f"{rel} should fire (exit != 0), got exit {exit_code}"

    def test_checkpoint_dir_prefix_still_fires(self):
        """Old literal prefix test for checkpoints directory still catches siblings."""
        path = os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt")
        exit_code, _ = self._run_hook_no_tty(BASE_HOOK, path)
        assert exit_code != 0, "Base should fire on checkpoints-old"
        exit_code, _ = self._run_hook_no_tty(NEW_HOOK, path)
        assert exit_code != 0, "New should also fire on checkpoints-old (old prefix test kept)"

    def test_ng_peer_bridge_still_fires(self):
        """ng_peer_bridge.py fires (exit != 0) with RETIRED label."""
        path = os.path.join(self._ng_dir, "ng_peer_bridge.py")
        exit_code, stderr = self._run_hook_no_tty(NEW_HOOK, path)
        assert exit_code != 0, f"ng_peer_bridge.py should fire, got exit {exit_code}"
        assert "Retired vendored file" in stderr, f"Expected RETIRED label in stderr, got: {stderr[:200]}"

    # ── no over-match ────────────────────────────────────────────────────────

    def test_no_overmatch_unprotected_home(self):
        """Unprotected files under $HOME/NeuroGraph do NOT fire."""
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))
            assert exit_code == 0, f"{rel} should NOT fire"

    def test_no_overmatch_unprotected_worktree(self):
        """Unprotected files in worktrees do NOT fire."""
        for rel in ["README.md", "tests/test_foo.py"]:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_outside, rel))
            assert exit_code == 0, f"worktree {rel} should NOT fire"

    def test_no_overmatch_unrelated_repo(self):
        """neuro_foundation.py in a repo with different origin does NOT fire."""
        exit_code, _ = self._run_hook_no_tty(
            NEW_HOOK, os.path.join(self._unrelated, "neuro_foundation.py")
        )
        assert exit_code == 0

    def test_no_overmatch_non_git(self):
        """File in a non-git directory does NOT fire."""
        exit_code, _ = self._run_hook_no_tty(
            NEW_HOOK, os.path.join(self._non_git, "neuro_foundation.py")
        )
        assert exit_code == 0

    # ── bypass mechanism unchanged ──────────────────────────────────────────

    def _install_bypass(self):
        bp_dir = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks")
        os.makedirs(bp_dir, exist_ok=True)
        bp_file = os.path.join(bp_dir, ".session_approved")
        with open(bp_file, "w") as f:
            f.write("approved\n")
        return bp_file

    def test_bypass_allows_protected(self):
        """With bypass file present, every protected file exits 0."""
        self._install_bypass()
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py"]:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))
            assert exit_code == 0, f"{rel} should pass with bypass"

    def test_bypass_allows_worktree_protected(self):
        """Bypass also allows protected worktree files."""
        self._install_bypass()
        exit_code, _ = self._run_hook_no_tty(
            NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py")
        )
        assert exit_code == 0

    def test_bypass_location_in_home(self):
        """Bypass file is under $HOME/NeuroGraph/.claude/hooks/ even for worktree edits."""
        # Start fresh (no bypass)
        bp_file = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try:
            os.remove(bp_file)
        except FileNotFoundError:
            pass

        # Run with choice 3 on a worktree file
        exit_code, output = self._run_hook_pty(
            NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"), "3"
        )
        assert exit_code == 0
        assert "Session bypass" in output
        assert os.path.isfile(bp_file), f"Bypass file not created at {bp_file}"

    # ── pty interaction ──────────────────────────────────────────────────────

    def test_pty_choice_1_approve(self):
        """Choice 1 exits 0 with approval message."""
        exit_code, output = self._run_hook_pty(
            NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "1"
        )
        assert exit_code == 0
        assert "Approved" in output

    def test_pty_choice_2_block(self):
        """Choice 2 exits 2 with block message."""
        exit_code, output = self._run_hook_pty(
            NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "2"
        )
        assert exit_code == 2
        assert "BLOCKED" in output

    def test_pty_choice_3_approve_all(self):
        """Choice 3 exits 0, creates bypass file, prints activation message."""
        bp_file = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try:
            os.remove(bp_file)
        except FileNotFoundError:
            pass

        exit_code, output = self._run_hook_pty(
            NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "3"
        )
        assert exit_code == 0
        assert "Session bypass activated" in output
        assert os.path.isfile(bp_file), "Bypass file not created"

    def test_no_tty_blocks(self):
        """Without a tty, protected files exit 2 (blocked)."""
        exit_code, _ = self._run_hook_no_tty(
            NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py")
        )
        assert exit_code == 2, f"Expected exit 2 (blocked without tty), got {exit_code}"

    # ── worktree / new coverage ──────────────────────────────────────────────

    def test_worktree_protected_fires(self):
        """Protected files in worktrees (outside and inside $HOME) fire."""
        for wt in [self._wt_outside, self._wt_inside]:
            exit_code, _ = self._run_hook_no_tty(
                NEW_HOOK, os.path.join(wt, "neuro_foundation.py")
            )
            assert exit_code != 0, f"neuro_foundation.py in {wt} should fire"

    def test_claude_project_dir_relative(self):
        """Relative path resolved via CLAUDE_PROJECT_DIR to a worktree fires."""
        exit_code, _ = self._run_hook_no_tty(
            NEW_HOOK, "neuro_foundation.py",
            extra_env={"CLAUDE_PROJECT_DIR": self._wt_outside},
        )
        assert exit_code != 0, "Relative path via CLAUDE_PROJECT_DIR should fire"

    def test_newly_covered_vendored(self):
        """ng_tract_bridge.py, ng_embed.py, ng_salience_gate.py, ng_updater.py fire in both locations."""
        for rel in ["ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"]:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))
            assert exit_code != 0, f"{rel} in home should fire"
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_outside, rel))
            assert exit_code != 0, f"{rel} in worktree should fire"

    def test_home_guard(self):
        """Sanity: HOME inside the test should be our fake home."""
        assert self._fake_home.startswith(self._tmpdir), "HOME not under temp dir"
        assert "/tmp/" in self._fake_home, "HOME suspiciously not in /tmp"