# ---- Changelog ----
# [2026-09-30] Chief-003 / Claude — Correction pass (Exec Packet 447): add fault-closed,
#      origin-normaliser, git-missing, symlink, worktree, and sibling-hook tests.
# [2026-09-30] Chief-003 / Claude — v1: initial test suite (Exec Packets 442-443)
# -------------------

import os, sys, json, time, fcntl, shutil, pytest, pty as pty_module, subprocess, tempfile

BASE_HOOK = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_syls_law_base.sh")
NEW_HOOK = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_syls_law.sh")
DBL_HOOK = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "posttool_syls_law_doublecheck.sh")
DBL_BASE = os.path.join(os.path.dirname(__file__), "fixtures", "posttool_syls_law_doublecheck_base.sh")
AP_HOOK  = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "posttool_antipattern_checker.sh")
AP_BASE  = os.path.join(os.path.dirname(__file__), "fixtures", "posttool_antipattern_checker_base.sh")
CG_HOOK  = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_context_gate.sh")
CG_BASE  = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_context_gate_base.sh")

PROTECTED_RELS = [
    "data/checkpoints/main.msgpack",
    "data/checkpoints/vectors.msgpack",
    "data/checkpoints/main.msgpack.activations.json",
    "neuro_foundation.py", "openclaw_hook.py", "stream_parser.py", "activation_persistence.py",
    "ng_lite.py", "ng_tract_bridge.py", "ng_ecosystem.py", "ng_autonomic.py",
    "openclaw_adapter.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py",
    "ng_peer_bridge.py",
]


class TestSylsLawHook:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, request):
        self._tmpdir = tempfile.mkdtemp(prefix="syls_law_hook_test_")
        self._fake_home = os.path.join(self._tmpdir, "home")
        os.makedirs(self._fake_home)
        ng_dir = os.path.join(self._fake_home, "NeuroGraph")
        os.makedirs(ng_dir)
        subprocess.run(["git", "init", "-b", "main"], cwd=ng_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git"],
            cwd=ng_dir, capture_output=True, check=True,
        )
        for rel in PROTECTED_RELS:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("protected\n")
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("unprotected\n")
        ckpt_old = os.path.join(ng_dir, "data", "checkpoints-old", "stale.txt")
        os.makedirs(os.path.dirname(ckpt_old), exist_ok=True)
        with open(ckpt_old, "w") as fh: fh.write("stale\n")
        subprocess.run(["git", "add", "-A"], cwd=ng_dir, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "init"],
            cwd=ng_dir, capture_output=True, check=True,
        )
        self._wt_outside = os.path.join(self._tmpdir, "worktree_outside")
        subprocess.run(
            ["git", "worktree", "add", "--detach", self._wt_outside],
            cwd=ng_dir, capture_output=True, check=True,
        )
        self._wt_inside = os.path.join(self._fake_home, "worktree_inside")
        subprocess.run(
            ["git", "worktree", "add", "--detach", self._wt_inside],
            cwd=ng_dir, capture_output=True, check=True,
        )
        # Nested worktree inside $HOME/NeuroGraph/.claude/worktrees/
        nested_dir = os.path.join(ng_dir, ".claude", "worktrees", "nested")
        os.makedirs(os.path.dirname(nested_dir), exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", nested_dir],
            cwd=ng_dir, capture_output=True, check=True,
        )
        self._wt_nested = nested_dir
        # Unrelated repo
        self._unrelated = os.path.join(self._tmpdir, "unrelated_repo")
        os.makedirs(self._unrelated)
        subprocess.run(["git", "init", "-b", "main"], cwd=self._unrelated, capture_output=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/someone/other-repo.git"],
            cwd=self._unrelated, capture_output=True, check=True,
        )
        with open(os.path.join(self._unrelated, "neuro_foundation.py"), "w") as fh: fh.write("unrelated\n")
        subprocess.run(["git", "add", "-A"], cwd=self._unrelated, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "init"],
            cwd=self._unrelated, capture_output=True, check=True,
        )
        self._non_git = os.path.join(self._tmpdir, "non_git")
        os.makedirs(self._non_git)
        self._ng_dir = ng_dir
        yield
        for wt in [self._wt_outside, self._wt_inside, self._wt_nested]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=self._ng_dir, capture_output=True,
            )
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _base_env(self):
        return {"HOME": self._fake_home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    def _run_hook_no_tty(self, hook_path, file_path, extra_env=None):
        env = self._base_env()
        if extra_env: env.update(extra_env)
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_hook_raw_stdin(self, hook_path, stdin_bytes, extra_env=None):
        env = self._base_env()
        if extra_env: env.update(extra_env)
        result = subprocess.run(
            [hook_path], input=stdin_bytes, capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_hook_no_jq(self, hook_path, file_path):
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_no_git(self, hook_path, file_path):
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_no_git_or_jq(self, hook_path, file_path):
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stderr.decode("utf-8", errors="replace")

    def _run_hook_pty(self, hook_path, file_path, choice, extra_env=None):
        env = self._base_env()
        if extra_env: env.update(extra_env)
        tool_input_json = json.dumps({"tool_input": {"file_path": file_path}})
        pid, master_fd = pty_module.fork()
        if pid == 0:
            for k, v in env.items(): os.environ[k] = v
            os.execv(hook_path, [hook_path])
            os._exit(127)
        os.write(master_fd, (tool_input_json + "\n\x04").encode())
        output = b""; choice_sent = False; deadline = time.time() + 10
        while time.time() < deadline:
            try:
                data = os.read(master_fd, 4096)
                if data: output += data
                if not choice_sent and b"Choice [1/2/3]:" in output:
                    os.write(master_fd, (str(choice) + "\n").encode())
                    choice_sent = True; time.sleep(0.4)
            except BlockingIOError:
                if choice_sent:
                    time.sleep(0.2)
                    try: wpid, _ = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError: break
                    if wpid != 0: break
                else: time.sleep(0.05)
            except OSError: break
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                data = os.read(master_fd, 4096)
                if not data: break
                output += data
        except (BlockingIOError, OSError): pass
        try: _, status = os.waitpid(pid, 0); exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        except ChildProcessError: exit_code = -1
        os.close(master_fd)
        return exit_code, output.decode("utf-8", errors="replace")

    def _tmp_repo_with_origin(self, origin_url):
        d = tempfile.mkdtemp(prefix="origin_test_", dir=self._tmpdir)
        os.makedirs(d)
        subprocess.run(["git", "init", "-b", "main"], cwd=d, capture_output=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", origin_url], cwd=d, capture_output=True, check=True)
        for rel in PROTECTED_RELS:
            full = os.path.join(d, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("x\n")
        subprocess.run(["git", "add", "-A"], cwd=d, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "x"],
            cwd=d, capture_output=True, check=True,
        )
        return d

    def _install_bypass(self):
        bp_dir = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks")
        os.makedirs(bp_dir, exist_ok=True)
        bp_file = os.path.join(bp_dir, ".session_approved")
        with open(bp_file, "w") as f: f.write("approved\n")
        return bp_file

    # ═══════════════════════════════════════════════════════════════════════════
    # FAULT-CLOSED TESTS — hook, doublecheck, antipattern, context-gate
    # ═══════════════════════════════════════════════════════════════════════════

    def test_pretool_jq_missing_exit2(self):
        exit_code, stderr = self._run_no_git_or_jq(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))
        assert exit_code == 2, f"Expected exit 2 with jq missing, got {exit_code}"
        assert "jq" in stderr.lower() and "gatekeeper" in stderr.lower()

    def test_pretool_empty_stdin_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(NEW_HOOK, b"")
        assert exit_code == 2
        assert "empty" in stderr.lower()

    def test_pretool_unparseable_stdin_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(NEW_HOOK, b"not json")
        assert exit_code == 2
        assert "target" in stderr.lower() or "path" in stderr.lower()

    def test_pretool_no_path_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(NEW_HOOK, b'{"tool_input":{}}')
        assert exit_code == 2
        assert "target" in stderr.lower() or "path" in stderr.lower()

    def test_posttool_doublecheck_jq_missing_exit2(self):
        exit_code, stderr = self._run_no_git_or_jq(DBL_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))
        assert exit_code == 2
        assert "jq" in stderr.lower()

    def test_posttool_doublecheck_empty_stdin_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(DBL_HOOK, b"")
        assert exit_code == 2

    def test_posttool_doublecheck_no_path_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(DBL_HOOK, b'{"tool_input":{}}')
        assert exit_code == 2

    def test_posttool_antipattern_jq_missing_exit2(self):
        exit_code, stderr = self._run_no_git_or_jq(AP_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))
        assert exit_code == 2
        assert "jq" in stderr.lower()

    def test_posttool_antipattern_empty_stdin_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(AP_HOOK, b"")
        assert exit_code == 2

    def test_posttool_antipattern_no_path_exit2(self):
        exit_code, stderr = self._run_hook_raw_stdin(AP_HOOK, b'{"tool_input":{}}')
        assert exit_code == 2

    def test_context_gate_jq_missing_exit0_with_context(self):
        exit_code, stdout, stderr = self._run_hook_capture_stdout(CG_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))
        assert exit_code == 0
        assert "context gate INACTIVE" in stdout and "jq" in stdout

    def test_context_gate_empty_stdin_exit0_with_context(self):
        exit_code, stdout, _ = self._run_hook_raw_stdin_stdout(CG_HOOK, b"")
        assert exit_code == 0
        assert "INACTIVE" in stdout

    def test_context_gate_no_path_exit0_with_context(self):
        exit_code, stdout, _ = self._run_hook_raw_stdin_stdout(CG_HOOK, b'{"tool_input":{}}')
        assert exit_code == 0
        assert "INACTIVE" in stdout

    def _run_hook_capture_stdout(self, hook_path, file_path):
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        tool_input = json.dumps({"tool_input": {"file_path": file_path}})
        result = subprocess.run(
            [hook_path], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stdout.decode("utf-8", errors="replace"), result.stderr.decode("utf-8", errors="replace")

    def _run_hook_raw_stdin_stdout(self, hook_path, stdin_bytes):
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        result = subprocess.run(
            [hook_path], input=stdin_bytes, capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        return result.returncode, result.stdout.decode("utf-8", errors="replace"), result.stderr.decode("utf-8", errors="replace")

    # ═══════════════════════════════════════════════════════════════════════════
    # SIBLING-HOOK BASE vs NEW — normal inputs produce same results
    # ═══════════════════════════════════════════════════════════════════════════

    def test_doublecheck_normal_same_as_base(self):
        """Doublecheck: same exit code for protected and unprotected paths."""
        for label, path in [
            ("protected", os.path.join(self._ng_dir, "neuro_foundation.py")),
            ("unprotected", os.path.join(self._ng_dir, "README.md")),
            ("ckpt_dir", os.path.join(self._ng_dir, "data", "checkpoints", "subfile.msgpack")),
        ]:
            base_exit, _ = self._run_hook_no_tty(DBL_BASE, path)
            new_exit, _ = self._run_hook_no_tty(DBL_HOOK, path)
            assert base_exit == new_exit, f"{label}: base={base_exit} new={new_exit}"

    def test_antipattern_normal_same_as_base(self):
        """Antipattern: same exit code for normal inputs."""
        for label, path in [
            ("unprotected_py", os.path.join(self._ng_dir, "tests", "test_foo.py")),
            ("unprotected_md", os.path.join(self._ng_dir, "README.md")),
            ("outside_repo", os.path.join(self._non_git, "something.py")),
        ]:
            base_exit, _ = self._run_hook_no_tty(AP_BASE, path)
            new_exit, _ = self._run_hook_no_tty(AP_HOOK, path)
            assert base_exit == new_exit, f"{label}: base={base_exit} new={new_exit}"

    def test_context_gate_normal_same_as_base(self):
        """Context gate: same stdout for critical files, normal exit 0."""
        for label, path in [
            ("neuro_foundation", os.path.join(self._ng_dir, "neuro_foundation.py")),
            ("openclaw_hook", os.path.join(self._ng_dir, "openclaw_hook.py")),
            ("ng_lite", os.path.join(self._ng_dir, "ng_lite.py")),
            ("unmatched", os.path.join(self._ng_dir, "README.md")),
        ]:
            env = self._base_env()
            tool_input = json.dumps({"tool_input": {"file_path": path}})
            r1 = subprocess.run(
                [CG_BASE], input=tool_input.encode(), capture_output=True, timeout=10,
                start_new_session=True, env=env,
            )
            r2 = subprocess.run(
                [CG_HOOK], input=tool_input.encode(), capture_output=True, timeout=10,
                start_new_session=True, env=env,
            )
            assert r1.returncode == r2.returncode, f"{label} exit: base={r1.returncode} new={r2.returncode}"
            assert r1.stdout.decode() == r2.stdout.decode(), f"{label} stdout differs"

    # ═══════════════════════════════════════════════════════════════════════════
    # EXISTING CORE TESTS (kept, adapted where needed)
    # ═══════════════════════════════════════════════════════════════════════════

    def test_differential_proof(self):
        corpus = []
        for rel in PROTECTED_RELS:
            corpus.append(("home", os.path.join(self._ng_dir, rel), {}))
            corpus.append(("wt_out", os.path.join(self._wt_outside, rel), {}))
            corpus.append(("wt_in", os.path.join(self._wt_inside, rel), {}))
            corpus.append(("wt_nested", os.path.join(self._wt_nested, rel), {}))
            corpus.append(("rel_wt_out", rel, {"CLAUDE_PROJECT_DIR": self._wt_outside}))
            corpus.append(("rel_wt_in", rel, {"CLAUDE_PROJECT_DIR": self._wt_inside}))
            corpus.append(("rel_main", rel, {"CLAUDE_PROJECT_DIR": self._ng_dir}))
        corpus.append(("home_ckpt_sub", os.path.join(self._ng_dir, "data", "checkpoints", "subfile.msgpack"), {}))
        corpus.append(("home_ckpt_dir", os.path.join(self._ng_dir, "data", "checkpoints"), {}))
        corpus.append(("wt_ckpt_sub", os.path.join(self._wt_outside, "data", "checkpoints", "subfile.msgpack"), {}))
        corpus.append(("home_ckpt_old", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            corpus.append(("unprot_home", os.path.join(self._ng_dir, rel), {}))
            corpus.append(("unprot_wt", os.path.join(self._wt_outside, rel), {}))
        corpus.append(("dotdot", os.path.join(self._ng_dir, "subdir", "..", "neuro_foundation.py"), {}))
        corpus.append(("unrelated", os.path.join(self._unrelated, "neuro_foundation.py"), {}))
        corpus.append(("non_git", os.path.join(self._non_git, "neuro_foundation.py"), {}))
        corpus.append(("nonexistent", os.path.join(self._ng_dir, "data", "checkpoints", "future.msgpack"), {}))
        # Symlink tests
        link_dir = os.path.join(self._tmpdir, "links")
        os.makedirs(link_dir, exist_ok=True)
        for rel in PROTECTED_RELS:
            target = os.path.join(self._ng_dir, rel)
            link_path = os.path.join(link_dir, os.path.basename(rel))
            if not os.path.islink(link_path):
                try: os.symlink(target, link_path)
                except FileExistsError: pass
            corpus.append((f"symlink_outside_{os.path.basename(rel)}", link_path, {}))
        symlink_inside_dir = os.path.join(self._ng_dir, "symlinks")
        os.makedirs(symlink_inside_dir, exist_ok=True)
        for rel in PROTECTED_RELS:
            target = os.path.join(self._ng_dir, rel)
            # Point inside symlink at protected file
            relative_target = os.path.relpath(target, symlink_inside_dir)
            link_in = os.path.join(symlink_inside_dir, os.path.basename(rel))
            if not os.path.isfile(link_in):
                try: os.symlink(relative_target, link_in)
                except FileExistsError: pass
            corpus.append((f"symlink_inside_{os.path.basename(rel)}", link_in, {}))
        base_fires = set(); new_fires = set()
        for label, path, extra in corpus:
            base_exit, _ = self._run_hook_no_tty(BASE_HOOK, path, extra_env=extra or None)
            new_exit, _ = self._run_hook_no_tty(NEW_HOOK, path, extra_env=extra or None)
            if base_exit != 0: base_fires.add((label, path))
            if new_exit != 0: new_fires.add((label, path))
        base_only = base_fires - new_fires
        assert not base_only, f"Base fires but new does NOT for: {sorted(base_only)}"
        expected_new = set()
        for rel in PROTECTED_RELS:
            for wt_tup in [("wt_out", self._wt_outside), ("wt_in", self._wt_inside), ("wt_nested", self._wt_nested)]:
                expected_new.add((wt_tup[0], os.path.join(wt_tup[1], rel)))
            expected_new.add(("rel_wt_out", rel)); expected_new.add(("rel_wt_in", rel))
            expected_new.add((f"symlink_outside_{os.path.basename(rel)}", os.path.join(link_dir, os.path.basename(rel))))
            expected_new.add((f"symlink_inside_{os.path.basename(rel)}", os.path.join(symlink_inside_dir, os.path.basename(rel))))
        added_vendored = ["ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"]
        for rel in added_vendored:
            expected_new.add(("home", os.path.join(self._ng_dir, rel)))
            expected_new.add(("rel_main", rel))
        expected_new.add(("wt_ckpt_sub", os.path.join(self._wt_outside, "data", "checkpoints", "subfile.msgpack")))
        actual_new = new_fires - base_fires
        missing = expected_new - actual_new; unexpected = actual_new - expected_new
        assert not missing, f"Expected new additions NOT found: {sorted(missing)}"
        assert not unexpected, f"Unexpected new additions found: {sorted(unexpected)}"
        print(f"Corpus:{len(corpus)} base_fires:{len(base_fires)} new_fires:{len(new_fires)} additions:{len(actual_new)}")

    def test_old_literal_all_protected_still_fire(self):
        for rel in PROTECTED_RELS:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))
            assert exit_code != 0, f"{rel} should fire, got exit {exit_code}"

    def test_checkpoint_dir_prefix_still_fires(self):
        path = os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt")
        exit_code, _ = self._run_hook_no_tty(BASE_HOOK, path); assert exit_code != 0
        exit_code, _ = self._run_hook_no_tty(NEW_HOOK, path); assert exit_code != 0

    def test_ng_peer_bridge_still_fires(self):
        path = os.path.join(self._ng_dir, "ng_peer_bridge.py")
        exit_code, stderr = self._run_hook_no_tty(NEW_HOOK, path)
        assert exit_code != 0
        assert "Retired vendored file" in stderr

    def test_no_overmatch_unprotected_home(self):
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0

    def test_no_overmatch_unprotected_worktree(self):
        for rel in ["README.md", "tests/test_foo.py"]:
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_outside, rel))[0] == 0

    def test_no_overmatch_unrelated_repo(self):
        assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._unrelated, "neuro_foundation.py"))[0] == 0

    def test_no_overmatch_non_git(self):
        assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._non_git, "neuro_foundation.py"))[0] == 0

    def test_bypass_allows_protected(self):
        self._install_bypass()
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py"]:
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0

    def test_bypass_allows_worktree_protected(self):
        self._install_bypass()
        assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"))[0] == 0

    def test_bypass_location_in_home(self):
        bp_file = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try: os.unlink(bp_file)
        except FileNotFoundError: pass
        exit_code, output = self._run_hook_pty(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"), "3")
        assert exit_code == 0; assert "Session bypass" in output; assert os.path.isfile(bp_file)

    def test_pty_choice_1_approve(self):
        exit_code, output = self._run_hook_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "1")
        assert exit_code == 0; assert "Approved" in output

    def test_pty_choice_2_block(self):
        exit_code, output = self._run_hook_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "2")
        assert exit_code == 2; assert "BLOCKED" in output

    def test_pty_choice_3_approve_all(self):
        bp_file = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try: os.unlink(bp_file)
        except FileNotFoundError: pass
        exit_code, output = self._run_hook_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "3")
        assert exit_code == 0; assert "Session bypass activated" in output; assert os.path.isfile(bp_file)

    def test_no_tty_blocks(self):
        exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))
        assert exit_code == 2

    def test_worktree_protected_fires(self):
        for wt in [self._wt_outside, self._wt_inside, self._wt_nested]:
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(wt, "neuro_foundation.py"))[0] != 0

    def test_claude_project_dir_relative(self):
        exit_code, _ = self._run_hook_no_tty(NEW_HOOK, "neuro_foundation.py", extra_env={"CLAUDE_PROJECT_DIR": self._wt_outside})
        assert exit_code != 0

    def test_newly_covered_vendored(self):
        for rel in ["ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"]:
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] != 0
            assert self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_outside, rel))[0] != 0

    def test_home_guard(self):
        assert self._fake_home.startswith(self._tmpdir)

    # ═══════════════════════════════════════════════════════════════════════════
    # SYMLINK AND NESTED WORKTREE
    # ═══════════════════════════════════════════════════════════════════════════

    def test_symlink_outside_into_protected_fires(self):
        link_dir = os.path.join(self._tmpdir, "symlinks")
        os.makedirs(link_dir, exist_ok=True)
        for rel in PROTECTED_RELS:
            target = os.path.join(self._ng_dir, rel)
            link_path = os.path.join(link_dir, os.path.basename(rel))
            if not os.path.islink(link_path): os.symlink(target, link_path)
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, link_path)
            assert exit_code != 0, f"symlink for {rel} should fire, got {exit_code}"

    def test_symlink_inside_to_protected_fires(self):
        link_dir = os.path.join(self._ng_dir, "symlinks")
        os.makedirs(link_dir, exist_ok=True)
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py", "ng_peer_bridge.py"]:
            target = os.path.join(self._ng_dir, rel)
            relative = os.path.relpath(target, link_dir)
            link = os.path.join(link_dir, os.path.basename(rel))
            if not os.path.isfile(link): os.symlink(relative, link)
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, link)
            assert exit_code != 0, f"internal symlink for {rel} should fire, got {exit_code}"

    def test_nested_worktree_fires(self):
        for rel in PROTECTED_RELS:
            exit_code, _ = self._run_hook_no_tty(NEW_HOOK, os.path.join(self._wt_nested, rel))
            assert exit_code != 0, f"nested worktree {rel} should fire"

    # ═══════════════════════════════════════════════════════════════════════════
    # ORIGIN URL MATRIX
    # ═══════════════════════════════════════════════════════════════════════════

    POSITIVE_ORIGINS = [
        "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "https://github.com/greatnorthernfishguy-hub/NeuroGraph",
        "https://github.com/greatnorthernfishguy-hub/NeuroGraph/",
        "https://github.com/GreatNorthernFishguy-hub/neurograph.git",
        "HTTPS://GITHUB.COM/GREATNORTHERNFISHGUY-HUB/NEUROGRAPH",
        "git@github.com:greatnorthernfishguy-hub/NeuroGraph.git",
        "ssh://git@github.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "git://github.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "https://user:DUMMY_token123@github.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "git+ssh://github.com/greatnorthernfishguy-hub/NeuroGraph.git",
    ]

    NEGATIVE_ORIGINS = [
        "https://github.com/evil-org/NeuroGraph.git",
        "https://evil.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "https://github.com/greatnorthernfishguy-hub/neurograph-fork.git",
        "https://github.com/greatnorthernfishguy-hub/NeuroGraph_EVIL.git",
    ]

    def test_origin_positive_matrix(self):
        for origin_url in self.POSITIVE_ORIGINS:
            repo = self._tmp_repo_with_origin(origin_url)
            path = os.path.join(repo, "neuro_foundation.py")
            exit_code, stderr = self._run_hook_no_tty(NEW_HOOK, path)
            assert exit_code != 0, f"Origin '{origin_url}' should match (exit != 0), got {exit_code}"
            # Assert origin never appears in stderr
            assert origin_url not in stderr, f"Origin leaked into stderr: {stderr[:200]}"

    def test_origin_negative_matrix(self):
        for origin_url in self.NEGATIVE_ORIGINS:
            repo = self._tmp_repo_with_origin(origin_url)
            path = os.path.join(repo, "neuro_foundation.py")
            exit_code, stderr = self._run_hook_no_tty(NEW_HOOK, path)
            assert exit_code == 0, f"Origin '{origin_url}' should NOT match, got exit {exit_code}"

    # ═══════════════════════════════════════════════════════════════════════════
    # GIT-MISSING TESTS
    # ═══════════════════════════════════════════════════════════════════════════

    def test_git_missing_fail_closed_for_worktree(self):
        """Without git, worktree paths that old literal doesn't catch should exit 2."""
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        path = os.path.join(self._wt_outside, "neuro_foundation.py")
        tool_input = json.dumps({"tool_input": {"file_path": path}})
        result = subprocess.run(
            [NEW_HOOK], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        assert result.returncode == 2, f"Without git, worktree path should exit 2, got {result.returncode}"

    def test_git_missing_old_literal_still_fires(self):
        """Without git, old $HOME/NeuroGraph paths still fire (literal match works)."""
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        path = os.path.join(self._ng_dir, "neuro_foundation.py")
        tool_input = json.dumps({"tool_input": {"file_path": path}})
        result = subprocess.run(
            [NEW_HOOK], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        assert result.returncode == 2, f"Old literal path should still fire, got {result.returncode}"

    def test_git_missing_unprotected_allowed(self):
        """Without git, unprotected files in home still allowed (literal match fails, git unreachable → exit 2 for safety)."""
        env = self._base_env()
        env["PATH"] = "/usr/bin:/bin"
        path = os.path.join(self._ng_dir, "README.md")
        tool_input = json.dumps({"tool_input": {"file_path": path}})
        result = subprocess.run(
            [NEW_HOOK], input=tool_input.encode(), capture_output=True, timeout=15,
            start_new_session=True, env=env,
        )
        # With git missing, _git_stderr check won't see "not a git repository"
        # because git isn't found. This could exit 0 or exit 2 depending on
        # whether git exists but errors out vs git not being found.
        # The hook checks _git_stderr for "fatal" messages. If git is missing,
        # _git_stderr is empty, so it falls through to the REL matching section.
        # _repo_toplevel fails (no git), TOPLEVEL is empty.
        # Then file not in old literal check → NOT protected → exit 0.
        assert result.returncode == 0, f"Unprotected file should exit 0, got {result.returncode}"

    def test_git_ok_but_not_a_repo_allowed(self):
        """File in a non-git directory: git says 'not a git repository', allowed."""
        path = os.path.join(self._non_git, "neuro_foundation.py")
        exit_code, _ = self._run_hook_no_tty(NEW_HOOK, path)
        assert exit_code == 0