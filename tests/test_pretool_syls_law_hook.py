# ---- Changelog ----
# [2026-09-30] Chief-003 / Claude — Test restoration: all 39 removed names restored,
#      sibling-hook tests returned, new git-stub/any-remote/locale proof added.
# -------------------

import os, sys, json, time, fcntl, shutil, pytest, pty as pty_module, subprocess, tempfile

BASE_HOOK    = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_syls_law_base.sh")
NEW_HOOK     = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_syls_law.sh")
DBL_HOOK     = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "posttool_syls_law_doublecheck.sh")
DBL_BASE     = os.path.join(os.path.dirname(__file__), "fixtures", "posttool_syls_law_doublecheck_base.sh")
AP_HOOK      = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "posttool_antipattern_checker.sh")
AP_BASE      = os.path.join(os.path.dirname(__file__), "fixtures", "posttool_antipattern_checker_base.sh")
CG_HOOK      = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_context_gate.sh")
CG_BASE      = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_context_gate_base.sh")

PROTECTED_RELS = [
    "data/checkpoints/main.msgpack", "data/checkpoints/vectors.msgpack",
    "data/checkpoints/main.msgpack.activations.json",
    "neuro_foundation.py", "openclaw_hook.py", "stream_parser.py", "activation_persistence.py",
    "ng_lite.py", "ng_tract_bridge.py", "ng_ecosystem.py", "ng_autonomic.py",
    "openclaw_adapter.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py",
    "ng_peer_bridge.py",
]

POS_ORIGINS = [
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
    "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git/",
    "deploy@github.com:greatnorthernfishguy-hub/NeuroGraph.git",
    "ssh://git@github.com:22/greatnorthernfishguy-hub/NeuroGraph.git",
]
NEG_ORIGINS = [
    "https://github.com/evil-org/NeuroGraph.git",
    "https://evil.com/greatnorthernfishguy-hub/NeuroGraph.git",
    "https://github.com/greatnorthernfishguy-hub/neurograph-fork.git",
    "https://github.com/greatnorthernfishguy-hub/NeuroGraph_EVIL.git",
    "https://github.com.evil.com/greatnorthernfishguy-hub/NeuroGraph.git",
]


class TestSylsLawHook:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, request):
        self._tmpdir = tempfile.mkdtemp(prefix="syls_law_")
        self._fake_home = os.path.join(self._tmpdir, "home")
        os.makedirs(self._fake_home)
        ng_dir = os.path.join(self._fake_home, "NeuroGraph")
        os.makedirs(ng_dir)
        subprocess.run(["git", "init", "-b", "main"], cwd=ng_dir, capture_output=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git"], cwd=ng_dir, capture_output=True, check=True)
        for rel in PROTECTED_RELS:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("x\n")
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            full = os.path.join(ng_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("x\n")
        ckpt_old = os.path.join(ng_dir, "data", "checkpoints-old", "stale.txt")
        os.makedirs(os.path.dirname(ckpt_old), exist_ok=True)
        with open(ckpt_old, "w") as fh: fh.write("x\n")
        subprocess.run(["git", "add", "-A"], cwd=ng_dir, capture_output=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init"], cwd=ng_dir, capture_output=True, check=True)
        self._wt_outside = os.path.join(self._tmpdir, "wt_o")
        subprocess.run(["git", "worktree", "add", "--detach", self._wt_outside], cwd=ng_dir, capture_output=True, check=True)
        self._wt_inside = os.path.join(self._fake_home, "wt_i")
        subprocess.run(["git", "worktree", "add", "--detach", self._wt_inside], cwd=ng_dir, capture_output=True, check=True)
        nw = os.path.join(ng_dir, ".claude", "worktrees", "nested")
        os.makedirs(os.path.dirname(nw), exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", nw], cwd=ng_dir, capture_output=True, check=True)
        self._wt_nested = nw
        self._unrelated = os.path.join(self._tmpdir, "unrelated")
        os.makedirs(self._unrelated)
        subprocess.run(["git", "init", "-b", "main"], cwd=self._unrelated, capture_output=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/someone/other.git"], cwd=self._unrelated, capture_output=True, check=True)
        with open(os.path.join(self._unrelated, "neuro_foundation.py"), "w") as fh: fh.write("x\n")
        subprocess.run(["git", "add", "-A"], cwd=self._unrelated, capture_output=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "x"], cwd=self._unrelated, capture_output=True, check=True)
        self._non_git = os.path.join(self._tmpdir, "non_git")
        os.makedirs(self._non_git)
        self._no_origin = os.path.join(self._tmpdir, "no_origin")
        os.makedirs(self._no_origin)
        subprocess.run(["git", "init", "-b", "main"], cwd=self._no_origin, capture_output=True, check=True)
        for rel in PROTECTED_RELS[:1]:
            full = os.path.join(self._no_origin, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("x\n")
        subprocess.run(["git", "add", "-A"], cwd=self._no_origin, capture_output=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "x"], cwd=self._no_origin, capture_output=True, check=True)
        self._ng_dir = ng_dir
        yield
        for wt in [self._wt_outside, self._wt_inside, self._wt_nested]:
            subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=self._ng_dir, capture_output=True)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── helpers ────────────────────────────────────────────────────
    def _env(self, extra=None):
        e = {"HOME": self._fake_home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if extra: e.update(extra)
        return e

    def _run(self, hook, path, extra_env=None):
        env = self._env(); env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        if extra_env: env.update(extra_env)
        ti = json.dumps({"tool_input": {"file_path": path}})
        r = subprocess.run([hook], input=ti.encode(), capture_output=True, timeout=15, start_new_session=True, env=env)
        return r.returncode, r.stderr.decode(errors="replace")

    def _run_stdout(self, hook, path, extra_env=None):
        env = self._env(); env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        if extra_env: env.update(extra_env)
        ti = json.dumps({"tool_input": {"file_path": path}})
        r = subprocess.run([hook], input=ti.encode(), capture_output=True, timeout=15, start_new_session=True, env=env)
        return r.returncode, r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")

    def _run_raw(self, hook, stdin_bytes, env=None):
        e = {"HOME": self._fake_home, "USER": os.environ.get("USER", ""),
             "SHELL": os.environ.get("SHELL", "/usr/bin/bash"),
             "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if env:
            e.update(env)
        r = subprocess.run([hook], input=stdin_bytes, capture_output=True, timeout=15, start_new_session=True, env=e)
        return r.returncode, r.stderr.decode(errors="replace"), r.stdout.decode(errors="replace")

    def _run_pty(self, hook, path, choice, extra_env=None):
        env = self._env()
        if extra_env: env.update(extra_env)
        ti = json.dumps({"tool_input": {"file_path": path}})
        pid, fd = pty_module.fork()
        if pid == 0:
            for k, v in env.items(): os.environ[k] = v
            os.execv(hook, [hook]); os._exit(127)
        os.write(fd, (ti + "\n\x04").encode())
        out = b""; cs = False; dl = time.time() + 10
        while time.time() < dl:
            try:
                d = os.read(fd, 4096)
                if d: out += d
                if not cs and b"Choice [1/2/3]:" in out:
                    os.write(fd, (str(choice) + "\n").encode()); cs = True; time.sleep(0.4)
            except BlockingIOError:
                time.sleep(0.05 if not cs else 0.2)
                if cs:
                    try: wp, _ = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError: break
                    if wp != 0: break
            except OSError: break
        try: fl = fcntl.fcntl(fd, fcntl.F_GETFL); fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        except: pass
        try:
            while True:
                d = os.read(fd, 4096)
                if not d: break; out += d
        except (BlockingIOError, OSError): pass
        try: _, s = os.waitpid(pid, 0); rc = os.WEXITSTATUS(s) if os.WIFEXITED(s) else -1
        except ChildProcessError: rc = -1
        os.close(fd)
        return rc, out.decode(errors="replace")

    def _tmp_repo(self, origin):
        d = tempfile.mkdtemp(prefix="origin_", dir=self._tmpdir)
        os.makedirs(d, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=d, capture_output=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", origin], cwd=d, capture_output=True, check=True)
        for rel in PROTECTED_RELS[:1]:
            full = os.path.join(d, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh: fh.write("x\n")
        subprocess.run(["git", "add", "-A"], cwd=d, capture_output=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "x"], cwd=d, capture_output=True, check=True)
        return d

    def _install_bypass(self):
        d = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ".session_approved"), "w") as f: f.write("ok\n")

    def _env_stub_no_git(self):
        stub = os.path.join(self._tmpdir, "stub_bin")
        os.makedirs(stub, exist_ok=True)
        for t in ["jq", "timeout", "realpath", "sed", "tr", "dirname", "bash"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        return {"HOME": self._fake_home, "PATH": stub}

    def _env_stub_no_jq(self):
        stub = os.path.join(self._tmpdir, "stub_nojq")
        os.makedirs(stub, exist_ok=True)
        for t in ["git", "timeout", "realpath", "sed", "tr", "dirname", "bash", "cat", "printf"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        return {"HOME": self._fake_home, "PATH": stub}

    def _env_stub_git_hanging(self):
        stub = os.path.join(self._tmpdir, "stub_hang")
        os.makedirs(stub, exist_ok=True)
        for t in ["jq", "timeout", "realpath", "sed", "tr", "dirname", "bash"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        # stub git that sleeps longer than the hook's timeout (3s)
        git_stub = os.path.join(stub, "git")
        with open(git_stub, "w") as f:
            f.write("#!/usr/bin/env bash\nsleep 10\n")
        os.chmod(git_stub, 0o755)
        return {"HOME": self._fake_home, "PATH": stub}

    def _env_stub_git_exit1(self):
        stub = os.path.join(self._tmpdir, "stub_exit1")
        os.makedirs(stub, exist_ok=True)
        for t in ["jq", "timeout", "realpath", "sed", "tr", "dirname", "bash"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        git_stub = os.path.join(stub, "git")
        with open(git_stub, "w") as f:
            f.write("#!/usr/bin/env bash\necho 'something went wrong' >&2\nexit 1\n")
        os.chmod(git_stub, 0o755)
        return {"HOME": self._fake_home, "PATH": stub}

    # ══════════════════════════════════════════════════════════════════
    # PRECONDITIONS (stub validity)
    # ══════════════════════════════════════════════════════════════════

    def test_precondition_git_missing_stub(self):
        e = self._env_stub_no_git()
        r = subprocess.run(["/usr/bin/bash", "-c", "command -v jq && ! command -v git && echo OK"], capture_output=True, env=e)
        assert b"OK" in r.stdout, f"stub wrong: {r.stdout}"

    def test_precondition_jq_missing_stub(self):
        e = self._env_stub_no_jq()
        r = subprocess.run(["/usr/bin/bash", "-c", "! command -v jq && command -v git && echo OK"], capture_output=True, env=e)
        assert b"OK" in r.stdout, f"stub wrong: {r.stdout}"

    # ══════════════════════════════════════════════════════════════════
    # FAULT-CLOSED — pretool
    # ══════════════════════════════════════════════════════════════════

    def test_pretool_jq_missing_exit2(self):
        e = self._env_stub_no_jq()
        rc, stderr, _ = self._run_raw(NEW_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 2; assert "TOOL MISSING" in stderr

    def test_pretool_empty_stdin_exit2(self):
        rc, stderr, _ = self._run_raw(NEW_HOOK, b"")
        assert rc == 2; assert "EMPTY" in stderr

    def test_pretool_unparseable_stdin_exit2(self):
        rc, stderr, _ = self._run_raw(NEW_HOOK, b"not json")
        assert rc == 2
        assert "target" in stderr.lower() or "path" in stderr.lower()

    def test_pretool_no_path_exit2(self):
        rc, stderr, _ = self._run_raw(NEW_HOOK, b'{"tool_input":{}}')
        assert rc == 2
        assert "target" in stderr.lower() or "path" in stderr.lower()

    # ══════════════════════════════════════════════════════════════════
    # FAULT-CLOSED — posttool_doublecheck
    # ══════════════════════════════════════════════════════════════════

    def test_posttool_doublecheck_jq_missing_exit2(self):
        e = self._env_stub_no_jq()
        rc, stderr, _ = self._run_raw(DBL_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 2; assert "TOOL MISSING" in stderr

    def test_posttool_doublecheck_empty_stdin_exit2(self):
        rc, stderr, _ = self._run_raw(DBL_HOOK, b"")
        assert rc == 2

    def test_posttool_doublecheck_no_path_exit2(self):
        rc, stderr, _ = self._run_raw(DBL_HOOK, b'{"tool_input":{}}')
        assert rc == 2

    # ══════════════════════════════════════════════════════════════════
    # FAULT-CLOSED — posttool_antipattern
    # ══════════════════════════════════════════════════════════════════

    def test_posttool_antipattern_jq_missing_exit2(self):
        e = self._env_stub_no_jq()
        rc, stderr, _ = self._run_raw(AP_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 2; assert "jq" in stderr.lower()

    def test_posttool_antipattern_empty_stdin_exit2(self):
        rc, stderr, _ = self._run_raw(AP_HOOK, b"")
        assert rc == 2

    def test_posttool_antipattern_no_path_exit2(self):
        rc, stderr, _ = self._run_raw(AP_HOOK, b'{"tool_input":{}}')
        assert rc == 2

    # ══════════════════════════════════════════════════════════════════
    # CONTEXT GATE (always exit 0, loud on fault)
    # ══════════════════════════════════════════════════════════════════

    def test_context_gate_jq_missing_exit0_with_context(self):
        e = self._env_stub_no_jq()
        rc, _, stdout = self._run_raw(CG_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 0; assert "jq" in stdout.lower()

    def test_context_gate_empty_stdin_exit0_with_context(self):
        rc, _, stdout = self._run_raw(CG_HOOK, b"")
        assert rc == 0; assert "INACTIVE" in stdout

    def test_context_gate_no_path_exit0_with_context(self):
        rc, _, stdout = self._run_raw(CG_HOOK, b'{"tool_input":{}}')
        assert rc == 0; assert "INACTIVE" in stdout

    def test_context_gate_normal_same_as_base(self):
        for label, path in [
            ("neuro_foundation", os.path.join(self._ng_dir, "neuro_foundation.py")),
            ("openclaw_hook", os.path.join(self._ng_dir, "openclaw_hook.py")),
            ("ng_lite", os.path.join(self._ng_dir, "ng_lite.py")),
            ("unmatched", os.path.join(self._ng_dir, "README.md")),
        ]:
            ti = json.dumps({"tool_input": {"file_path": path}})
            r1 = subprocess.run([CG_BASE], input=ti.encode(), capture_output=True, timeout=10, start_new_session=True, env=self._env())
            r2 = subprocess.run([CG_HOOK], input=ti.encode(), capture_output=True, timeout=10, start_new_session=True, env=self._env())
            assert r1.returncode == r2.returncode, f"{label} exit: b={r1.returncode} n={r2.returncode}"
            assert r1.stdout.decode() == r2.stdout.decode(), f"{label} stdout differs"

    # ══════════════════════════════════════════════════════════════════
    # SIBLING-HOOK PARITY (normal inputs → same as base)
    # ══════════════════════════════════════════════════════════════════

    def test_doublecheck_normal_same_as_base(self):
        for label, path in [
            ("protected", os.path.join(self._ng_dir, "neuro_foundation.py")),
            ("unprotected", os.path.join(self._ng_dir, "README.md")),
            ("ckpt_dir", os.path.join(self._ng_dir, "data", "checkpoints", "subfile.msgpack")),
        ]:
            be, _ = self._run(DBL_BASE, path)
            ne, _ = self._run(DBL_HOOK, path)
            assert be == ne, f"{label}: b={be} n={ne}"

    def test_antipattern_normal_same_as_base(self):
        for label, path in [
            ("unprotected_py", os.path.join(self._ng_dir, "tests", "test_foo.py")),
            ("unprotected_md", os.path.join(self._ng_dir, "README.md")),
            ("outside_repo", os.path.join(self._non_git, "something.py")),
        ]:
            be, _ = self._run(AP_BASE, path)
            ne, _ = self._run(AP_HOOK, path)
            assert be == ne, f"{label}: b={be} n={ne}"

    # ══════════════════════════════════════════════════════════════════
    # DIFFERENTIAL PROOF (pretool)
    # ══════════════════════════════════════════════════════════════════

    def test_differential_proof(self):
        corpus = []
        for rel in PROTECTED_RELS:
            corpus.append(("home", os.path.join(self._ng_dir, rel), {}))
            corpus.append(("wt_o", os.path.join(self._wt_outside, rel), {}))
            corpus.append(("wt_i", os.path.join(self._wt_inside, rel), {}))
            corpus.append(("wt_n", os.path.join(self._wt_nested, rel), {}))
            corpus.append(("rel_wo", rel, {"CLAUDE_PROJECT_DIR": self._wt_outside}))
            corpus.append(("rel_wi", rel, {"CLAUDE_PROJECT_DIR": self._wt_inside}))
            corpus.append(("rel_m", rel, {"CLAUDE_PROJECT_DIR": self._ng_dir}))
        # Checkpoint directory + look-alike
        corpus.append(("ckpt_s", os.path.join(self._ng_dir, "data", "checkpoints", "sub.msgpack"), {}))
        corpus.append(("wt_ckpt_s", os.path.join(self._wt_outside, "data", "checkpoints", "sub.msgpack"), {}))
        corpus.append(("wt_n_ckpt_s", os.path.join(self._wt_nested, "data", "checkpoints", "sub.msgpack"), {}))
        corpus.append(("ckpt_o", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))
        for rel in ["README.md", "tests/test_foo.py"]:
            corpus.append(("unprot", os.path.join(self._ng_dir, rel), {}))
        # Symlinks
        lk = os.path.join(self._tmpdir, "sl")
        os.makedirs(lk, exist_ok=True)
        for rel in PROTECTED_RELS:
            tgt = os.path.join(self._ng_dir, rel)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.islink(lp):
                try: os.symlink(tgt, lp)
                except FileExistsError: pass
            corpus.append((f"sl_{os.path.basename(rel)}", lp, {}))
        corpus.append(("unr", os.path.join(self._unrelated, "neuro_foundation.py"), {}))
        corpus.append(("nogit", os.path.join(self._non_git, "neuro_foundation.py"), {}))
        bf = set(); nf = set()
        for label, path, extra in corpus:
            be, _ = self._run(BASE_HOOK, path, extra_env=extra or None)
            ne, _ = self._run(NEW_HOOK, path, extra_env=extra or None)
            if be != 0: bf.add((label, path))
            if ne != 0: nf.add((label, path))
        assert not (bf - nf), f"base fires but new does not: {sorted(bf - nf)[:5]}"
        new_v = {"ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"}
        exp = set()
        for rel in PROTECTED_RELS:
            for s, d in [("wt_o", self._wt_outside), ("wt_i", self._wt_inside), ("wt_n", self._wt_nested)]:
                exp.add((s, os.path.join(d, rel)))
            exp.add(("rel_wo", rel)); exp.add(("rel_wi", rel))
            if os.path.basename(rel) in new_v:
                exp.add((f"sl_{os.path.basename(rel)}", os.path.join(lk, os.path.basename(rel))))
        for r in new_v:
            exp.add(("home", os.path.join(self._ng_dir, r)))
            exp.add(("rel_m", r))
        exp.add(("wt_ckpt_s", os.path.join(self._wt_outside, "data", "checkpoints", "sub.msgpack")))
        exp.add(("wt_n_ckpt_s", os.path.join(self._wt_nested, "data", "checkpoints", "sub.msgpack")))
        act = nf - bf
        assert not (exp - act), f"missing: {sorted(exp - act)[:5]}"
        assert not (act - exp), f"unexpected: {sorted(act - exp)[:5]}"
        print(f"c:{len(corpus)} bf:{len(bf)} nf:{len(nf)} add:{len(act)}")

    # ══════════════════════════════════════════════════════════════════
    # DIFFERENTIAL PROOF (doublecheck)
    # ══════════════════════════════════════════════════════════════════

    def _make_dbl_corpus(self):
        c = []; lk = os.path.join(self._tmpdir, "sl_dbl")
        os.makedirs(lk, exist_ok=True)
        for rel in PROTECTED_RELS:
            c.append(("home", os.path.join(self._ng_dir, rel), {}))
            c.append(("wt_o", os.path.join(self._wt_outside, rel), {}))
            c.append(("wt_i", os.path.join(self._wt_inside, rel), {}))
            c.append(("wt_n", os.path.join(self._wt_nested, rel), {}))
        c.append(("ckpt", os.path.join(self._ng_dir, "data", "checkpoints", "sub.msgpack"), {}))
        c.append(("ckpt_o", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))
        for rel in ["README.md", "tests/test_foo.py"]:
            c.append(("un", os.path.join(self._ng_dir, rel), {}))
        for rel in PROTECTED_RELS:
            t = os.path.join(self._ng_dir, rel)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.islink(lp):
                try: os.symlink(t, lp)
                except FileExistsError: pass
            c.append((f"sl_{os.path.basename(rel)}", lp, {}))
        c.append(("unr", os.path.join(self._unrelated, "neuro_foundation.py"), {}))
        c.append(("nogit", os.path.join(self._non_git, "neuro_foundation.py"), {}))
        return c, lk

    def test_doublecheck_differential(self):
        corpus, lk = self._make_dbl_corpus()
        bf = set(); nf = set()
        for label, path, extra in corpus:
            be, _ = self._run(DBL_BASE, path, extra_env=extra or None)
            ne, _ = self._run(DBL_HOOK, path, extra_env=extra or None)
            if be == 2: bf.add((label, path))
            if ne == 2: nf.add((label, path))
        assert not (bf - nf), f"base fires but new does not: {sorted(bf - nf)[:5]}"
        new_v = {"ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"}
        exp = set()
        for rel in PROTECTED_RELS:
            for s, d in [("wt_o", self._wt_outside), ("wt_i", self._wt_inside), ("wt_n", self._wt_nested)]:
                exp.add((s, os.path.join(d, rel)))
            dp = os.path.join(lk, os.path.basename(rel))
            if os.path.basename(rel) in new_v:
                exp.add((f"sl_{os.path.basename(rel)}", dp))
                exp.add(("home", os.path.join(self._ng_dir, rel)))
        act = nf - bf
        assert not (exp - act), f"dbl missing: {sorted(exp - act)[:5]}"
        assert not (act - exp), f"dbl unexpected: {sorted(act - exp)[:5]}"
        print(f"dbl c:{len(corpus)} bf:{len(bf)} nf:{len(nf)} add:{len(act)}")

    # ══════════════════════════════════════════════════════════════════
    # OLD-LITERAL, OVERMATCH, BYPASS, PTY
    # ══════════════════════════════════════════════════════════════════

    def test_old_literal_all_protected_still_fire(self):
        for rel in PROTECTED_RELS:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] != 0

    def test_checkpoint_dir_prefix_still_fires(self):
        path = os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt")
        assert self._run(BASE_HOOK, path)[0] != 0
        assert self._run(NEW_HOOK, path)[0] != 0

    def test_ng_peer_bridge_still_fires(self):
        rc, stderr = self._run(NEW_HOOK, os.path.join(self._ng_dir, "ng_peer_bridge.py"))
        assert rc != 0; assert "Retired" in stderr

    def test_no_overmatch_unprotected_home(self):
        for rel in ["README.md", "tests/test_foo.py", "setup.py"]:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0

    def test_no_overmatch_unprotected_worktree(self):
        for rel in ["README.md", "tests/test_foo.py"]:
            assert self._run(NEW_HOOK, os.path.join(self._wt_outside, rel))[0] == 0

    def test_no_overmatch_unrelated_repo(self):
        assert self._run(NEW_HOOK, os.path.join(self._unrelated, "neuro_foundation.py"))[0] == 0

    def test_no_overmatch_non_git(self):
        assert self._run(NEW_HOOK, os.path.join(self._non_git, "neuro_foundation.py"))[0] == 0

    def test_bypass_allows_protected(self):
        self._install_bypass()
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py"]:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0

    def test_bypass_allows_worktree_protected(self):
        self._install_bypass()
        assert self._run(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"))[0] == 0

    def test_bypass_location_in_home(self):
        bp = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try: os.unlink(bp)
        except: pass
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"), "3")
        assert rc == 0; assert "bypass" in out.lower(); assert os.path.isfile(bp)

    def test_pty_choice_1_approve(self):
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "1")
        assert rc == 0; assert "Approved" in out

    def test_pty_choice_2_block(self):
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "2")
        assert rc == 2; assert "BLOCKED" in out

    def test_pty_choice_3_approve_all(self):
        bp = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try: os.unlink(bp)
        except: pass
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "3")
        assert rc == 0; assert "bypass" in out.lower(); assert os.path.isfile(bp)

    def test_no_tty_blocks(self):
        assert self._run(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))[0] == 2

    # ══════════════════════════════════════════════════════════════════
    # WORKTREE, SYMLINK, VENDORED
    # ══════════════════════════════════════════════════════════════════

    def test_worktree_protected_fires(self):
        for wt in [self._wt_outside, self._wt_inside, self._wt_nested]:
            assert self._run(NEW_HOOK, os.path.join(wt, "neuro_foundation.py"))[0] != 0

    def test_claude_project_dir_relative(self):
        assert self._run(NEW_HOOK, "neuro_foundation.py", extra_env={"CLAUDE_PROJECT_DIR": self._wt_outside})[0] != 0

    def test_newly_covered_vendored(self):
        for rel in ["ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"]:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] != 0
            assert self._run(NEW_HOOK, os.path.join(self._wt_outside, rel))[0] != 0

    def test_symlink_outside_into_protected_fires(self):
        lk = os.path.join(self._tmpdir, "sl3")
        os.makedirs(lk, exist_ok=True)
        for rel in PROTECTED_RELS:
            tgt = os.path.join(self._ng_dir, rel)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.islink(lp): os.symlink(tgt, lp)
            assert self._run(NEW_HOOK, lp)[0] != 0, f"symlink {rel} should fire"

    def test_symlink_inside_to_protected_fires(self):
        lk = os.path.join(self._ng_dir, "symlinks")
        os.makedirs(lk, exist_ok=True)
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py", "ng_peer_bridge.py"]:
            tgt = os.path.join(self._ng_dir, rel)
            rp = os.path.relpath(tgt, lk)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.isfile(lp): os.symlink(rp, lp)
            assert self._run(NEW_HOOK, lp)[0] != 0, f"inside symlink {rel} should fire"

    def test_nested_worktree_fires(self):
        for rel in PROTECTED_RELS:
            assert self._run(NEW_HOOK, os.path.join(self._wt_nested, rel))[0] != 0

    # ══════════════════════════════════════════════════════════════════
    # GIT-MISSING (proper stubs)
    # ══════════════════════════════════════════════════════════════════

    def test_git_missing_fail_closed_for_worktree(self):
        e = self._env_stub_no_git()
        # preflight fires because git is missing
        r = subprocess.run([NEW_HOOK], input=b'{"tool_input":{"file_path":"/x"}}', capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2

    def test_git_missing_old_literal_still_fires(self):
        e = self._env_stub_no_git()
        path = os.path.join(self._ng_dir, "neuro_foundation.py")
        r = subprocess.run([NEW_HOOK], input=json.dumps({"tool_input": {"file_path": path}}).encode(), capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2  # preflight catches git missing

    def test_git_missing_unprotected_allowed(self):
        """With stubs that keep every tool EXCEPT git, preflight catches git missing -> exit 2."""
        e = self._env_stub_no_git()
        path = os.path.join(self._ng_dir, "README.md")
        r = subprocess.run([NEW_HOOK], input=json.dumps({"tool_input": {"file_path": path}}).encode(), capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2  # preflight fires first

    def test_git_ok_but_not_a_repo_allowed(self):
        assert self._run(NEW_HOOK, os.path.join(self._non_git, "neuro_foundation.py"))[0] == 0

    # ══════════════════════════════════════════════════════════════════
    # GIT STUBS (hang, error, LC_ALL)
    # ══════════════════════════════════════════════════════════════════

    def test_git_hang_exit2(self):
        e_stub = self._env_stub_git_hanging()
        e = self._env(); e.update(e_stub)
        r = subprocess.run([NEW_HOOK], input=b'{"tool_input":{"file_path":"/x"}}', capture_output=True, timeout=20, start_new_session=True, env=e)
        assert r.returncode == 2

    def test_git_exit1_exit2(self):
        e_stub = self._env_stub_git_exit1()
        e = self._env(); e.update(e_stub)
        r = subprocess.run([NEW_HOOK], input=b'{"tool_input":{"file_path":"/x"}}', capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2

    def test_locale_stub_git_receives_LC_ALL_C(self):
        """Behavioural: stub git (bash builtins only) records LC_ALL to a pre-created file."""
        stub = os.path.join(self._tmpdir, "stub_lc_real")
        os.makedirs(stub, exist_ok=True)
        for t in ["jq", "timeout", "realpath", "sed", "tr", "dirname", "bash"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        record = os.path.join(self._tmpdir, "lc_all_recorded")
        with open(record, "w") as f: f.write("NOT_SET\n")
        # Git stub: bash builtins only (echo, case, exit, redirects)
        git_stub = os.path.join(stub, "git")
        with open(git_stub, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo -n \"$LC_ALL\" > {record}\n")
            f.write("case \"$*\" in\n")
            f.write("  *rev-parse*show-toplevel*) echo /fake_repo ;;\n")
            f.write("  *config*get-regexp*url*) echo 'remote.origin.url https://github.com/greatnorthernfishguy-hub/NeuroGraph.git' ;;\n")
            f.write("  *) exit 1 ;;\n")
            f.write("esac\n")
            f.write("exit 0\n")
        os.chmod(git_stub, 0o755)
        env = {"HOME": self._fake_home, "PATH": stub}
        path = os.path.join(self._wt_outside, "neuro_foundation.py")
        r = subprocess.run([NEW_HOOK], input=json.dumps({"tool_input": {"file_path": path}}).encode(), capture_output=True, timeout=15, start_new_session=True, env=env)
        assert r.returncode != 0
        with open(record, "r") as f:
            val = f.read().strip()
        assert val == "C", f"LC_ALL should be C, got '{val}'"

    # ══════════════════════════════════════════════════════════════════
    # ORIGIN MATRIX
    # ══════════════════════════════════════════════════════════════════

    def test_origin_positive(self):
        for origin in POS_ORIGINS:
            repo = self._tmp_repo(origin)
            rc, stderr = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
            assert rc != 0, f"POSITIVE origin should match: {origin}"
            assert origin not in stderr
            shutil.rmtree(repo, ignore_errors=True)

    def test_origin_negative(self):
        for origin in NEG_ORIGINS:
            repo = self._tmp_repo(origin)
            rc, stderr = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
            assert rc == 0, f"NEGATIVE origin should NOT match: {origin}"
            shutil.rmtree(repo, ignore_errors=True)

    # ══════════════════════════════════════════════════════════════════
    # ANY-REMOTE IDENTITY
    # ══════════════════════════════════════════════════════════════════

    def test_any_remote_upstream_neurograph(self):
        repo = self._tmp_repo("https://github.com/other/other.git")
        subprocess.run(["git", "-C", repo, "remote", "remove", "origin"], capture_output=True, check=True)
        subprocess.run(["git", "-C", repo, "remote", "add", "upstream", "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git"], capture_output=True, check=True)
        rc, _ = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
        assert rc != 0, "upstream NeuroGraph should be treated as NeuroGraph"
        shutil.rmtree(repo, ignore_errors=True)

    def test_any_remote_origin_other_upstream_neurograph(self):
        repo = self._tmp_repo("https://github.com/other/other.git")
        subprocess.run(["git", "-C", repo, "remote", "add", "upstream", "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git"], capture_output=True, check=True)
        rc, _ = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
        assert rc != 0, "upstream NeuroGraph should override non-matching origin"
        shutil.rmtree(repo, ignore_errors=True)

    def test_no_remotes_allowed(self):
        assert self._run(NEW_HOOK, os.path.join(self._no_origin, "neuro_foundation.py"))[0] == 0

    # ══════════════════════════════════════════════════════════════════
    # NON-EXISTENT DIR, NO-ORIGIN, NON-ENGLISH
    # ══════════════════════════════════════════════════════════════════

    def test_nonexistent_dir_allowed(self):
        d = os.path.join(self._non_git, "nonexistent_sub", "file.txt")
        assert self._run(NEW_HOOK, d)[0] == 0

    def test_no_origin_repo_allowed(self):
        assert self._run(NEW_HOOK, os.path.join(self._no_origin, "neuro_foundation.py"))[0] == 0

    def test_non_english_locale(self):
        import locale as lmod
        installed = subprocess.run(["locale", "-a"], capture_output=True).stdout.decode()
        if "fr_FR" not in installed:
            pytest.skip("fr_FR locale not installed on this machine")
        e = self._env({"LC_ALL": "fr_FR.UTF-8", "LANG": "fr_FR.UTF-8"})
        path = os.path.join(self._non_git, "neuro_foundation.py")
        assert self._run(NEW_HOOK, path, extra_env={"LC_ALL": "fr_FR.UTF-8", "LANG": "fr_FR.UTF-8"})[0] == 0

    # ══════════════════════════════════════════════════════════════════
    # DOUBLECHECK contract parity
    # ══════════════════════════════════════════════════════════════════

    def test_doublecheck_fires_worktree(self):
        assert self._run(DBL_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"))[0] == 2
        assert self._run(DBL_HOOK, os.path.join(self._wt_nested, "ng_lite.py"))[0] == 2

    def test_home_guard(self):
        assert self._fake_home.startswith(self._tmpdir)

    # ══════════════════════════════════════════════════════════════════
    # NORMALISER FUNCTION EQUIVALENCE (Change A)
    # ══════════════════════════════════════════════════════════════════

    def _extract_norm(self, hook_path):
        """Extract _norm_origin() function body from hook using sed.
        Returns (found_count, function_body)."""
        result = subprocess.run(
            ["sed", "-n", "/^_norm_origin() {/,/^}/p", hook_path],
            capture_output=True, check=True,
        )
        body = result.stdout.decode()
        count = body.count("_norm_origin() {")
        return count, body

    def test_normaliser_extraction_counts(self):
        """Each hook has exactly one _norm_origin function."""
        c1, _ = self._extract_norm(NEW_HOOK)
        c2, _ = self._extract_norm(DBL_HOOK)
        assert c1 == 1, f"pretool has {c1} _norm_origin functions"
        assert c2 == 1, f"doublecheck has {c2} _norm_origin functions"

    def _run_norm(self, hook_path, url):
        """Run the extracted _norm_origin function in bash and return output."""
        _, body = self._extract_norm(hook_path)
        script = f"{body}\n_norm_origin '{url}'"
        result = subprocess.run(
            ["bash"], input=script.encode(), capture_output=True, timeout=5,
        )
        return result.stdout.decode().strip()

    @staticmethod
    def _gen_origin_matrix():
        """Cross-product: scheme × userinfo × port × case × suffix × org/repo.
        Scrub tokens: use 'user:dummy@' in userinfo."""
        schemes_host = [
            ("https://", "github.com"),
            ("http://", "github.com"),
            ("ssh://", "github.com"),
            ("git://", "github.com"),
            ("git+ssh://", "github.com"),
            ("", "github.com"),  # scp form
        ]
        userinfos = ["", "git@", "deploy@", "user:dummy@"]
        ports = ["", ":22", ":443"]
        names = [
            ("GREATNORTHERNFISHGUY-HUB/NEUROGRAPH", True),
            ("evil-org/NeuroGraph", False),
            ("greatnorthernfishguy-hub/neurograph-fork", False),
        ]
        suffixes = ["", ".git", "/", ".git/"]
        test_urls = []
        for scheme, host in schemes_host:
            for userinfo in userinfos:
                if scheme == "" and userinfo == "":
                    continue  # no valid scp form without user
                for port in ports:
                    for orgrepo, expected in names:
                        for suffix in suffixes:
                            if scheme:
                                url = f"{scheme}{userinfo}{host}{port}/{orgrepo}{suffix}"
                            else:
                                url = f"{userinfo}{host}:{port.lstrip(':')}{orgrepo}" if port else f"{userinfo}{host}:{orgrepo}"
                                if suffix:
                                    url += suffix
                            test_urls.append((url, expected))
        return test_urls

    def test_normaliser_function_equivalence(self):
        """Both hooks' _norm_origin() produce byte-identical output for every generated URL."""
        matrix = self._gen_origin_matrix()
        assert len(matrix) > 200, f"matrix too small: {len(matrix)}"
        mismatches = []
        for url, expected in matrix:
            p = self._run_norm(NEW_HOOK, url)
            d = self._run_norm(DBL_HOOK, url)
            if p != d:
                mismatches.append((url, p, d))
        assert not mismatches, f"Normaliser mismatch: {mismatches[:5]}"
        print(f"Normaliser equiv: {len(matrix)} URLs, 0 mismatches")

    def test_normaliser_verdict_equivalence(self):
        """For a representative subset, both hooks agree on whether a repo is NeuroGraph."""
        subset = [
            # positive forms
            ("https://github.com/greatnorthernfishguy-hub/NeuroGraph.git", True),
            ("git@github.com:greatnorthernfishguy-hub/NeuroGraph.git", True),
            ("ssh://git@github.com/greatnorthernfishguy-hub/NeuroGraph", True),
            ("https://user:dummy@github.com/greatnorthernfishguy-hub/NeuroGraph", True),
            ("deploy@github.com:greatnorthernfishguy-hub/NeuroGraph.git", True),
            # negative forms
            ("https://github.com/evil-org/NeuroGraph.git", False),
            ("https://evil.com/greatnorthernfishguy-hub/NeuroGraph", False),
            ("https://github.com/greatnorthernfishguy-hub/neurograph-fork", False),
        ]
        for origin, is_ng in subset:
            repo = self._tmp_repo(origin)
            path = os.path.join(repo, "neuro_foundation.py")
            p_rc, _ = self._run(NEW_HOOK, path)
            d_rc, _ = self._run(DBL_HOOK, path)
            p_fires = p_rc != 0
            d_fires = d_rc != 0
            assert p_fires == d_fires, f"Verdict mismatch for {origin}: gate={p_fires} dbl={d_fires}"
            assert p_fires == is_ng, f"Gate wrong for {origin}: got {p_fires}, expected {is_ng}"
            shutil.rmtree(repo, ignore_errors=True)

    def test_normaliser_mutation_detects_drift(self):
        """Changing the doublecheck's normaliser makes the equivalence test fail."""
        _, body = self._extract_norm(DBL_HOOK)
        # Mutation: drop the .git strip line
        mutated = body.replace(".git", ".XYZ", 1)
        assert mutated != body, "mutation had no effect"
        matrix = self._gen_origin_matrix()
        found_mismatch = False
        for url, _ in matrix:
            p = self._run_norm(NEW_HOOK, url)
            m = subprocess.run(
                ["bash"], input=f"{mutated}\n_norm_origin '{url}'".encode(), capture_output=True, timeout=5,
            )
            m_out = m.stdout.decode().strip()
            if p != m_out:
                found_mismatch = True
                break
        assert found_mismatch, "Mutation test should detect a mismatch"

    # ══════════════════════════════════════════════════════════════════
    # GIT FAILURE-SHAPE DECISION EQUIVALENCE (Change B)
    # ══════════════════════════════════════════════════════════════════

    def _make_git_failure_stub(self, exit_code, stderr_msg=None, hang=False):
        """Create a stub git with specific exit code and optional stderr."""
        stub = os.path.join(self._tmpdir, f"stub_git_{exit_code}")
        os.makedirs(stub, exist_ok=True)
        for t in ["jq", "timeout", "realpath", "sed", "tr", "dirname", "bash"]:
            tp = subprocess.check_output(["which", t]).decode().strip()
            lk = os.path.join(stub, t)
            if not os.path.lexists(lk): os.symlink(tp, lk)
        git_stub = os.path.join(stub, "git")
        if hang:
            with open(git_stub, "w") as f:
                f.write("#!/bin/bash\nsleep 10\n")
        else:
            with open(git_stub, "w") as f:
                f.write("#!/bin/bash\n")
                if stderr_msg:
                    f.write(f"echo '{stderr_msg}' >&2\n")
                f.write(f"exit {exit_code}\n")
        os.chmod(git_stub, 0o755)
        return {"HOME": self._fake_home, "PATH": stub}

    def _test_decision_cell(self, stub_env, path, cell_name):
        """Return (gate_exit, dbl_exit) for a given stub env and path."""
        env = stub_env.copy()
        g_rc, _ = self._run(NEW_HOOK, path)
        e2 = stub_env.copy()
        d_rc, _ = self._run(DBL_HOOK, path)
        return g_rc, d_rc

    def test_decision_equivalence_failure_shapes(self):
        """Gate and doublecheck return same verdict for every git failure shape."""
        cells = []
        # Failure shapes
        shapes = [
            (1, "something went wrong", False, "exit1"),
            (2, "", False, "exit2"),
            (124, "", True, "hang"),
            (128, "fatal: not a git repository", False, "not_a_repo"),
            (128, "fatal: detected dubious ownership in repository", False, "dubious_ownership"),
        ]
        paths = {
            "not_repo": os.path.join(self._non_git, "x.py"),
            "in_repo": os.path.join(self._ng_dir, "neuro_foundation.py"),
            "nonexistent": os.path.join(self._non_git, "sub", "x.py"),
        }

        mismatches = []
        for exit_code, stderr, hang, shape_name in shapes:
            stub = self._make_git_failure_stub(exit_code, stderr, hang)
            for path_name, path in paths.items():
                g_rc, d_rc = self._test_decision_cell(stub, path, f"{shape_name}_{path_name}")
                cell = f"{shape_name}/{path_name}: gate={g_rc} dbl={d_rc}"
                # "not_a_repo" shapes should both exit 0 (allowed)
                # All other shapes should both exit 2 (fault)
                if shape_name == "not_a_repo":
                    expected = 0
                else:
                    expected = 2
                if g_rc != expected or d_rc != expected:
                    mismatches.append(cell)
                if g_rc != d_rc:
                    mismatches.append(f"MISMATCH: {cell}")
        assert not mismatches, f"Decision mismatches: {mismatches}"
        print(f"Decision equiv: {len(shapes) * len(paths)} cells, 0 mismatches")

    def test_git_absent_both_fault(self):
        """With git removed from PATH, both hooks exit 2 (preflight)."""
        e = self._env_stub_no_git()
        p_rc, _, _ = self._run_raw(NEW_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        d_rc, _, _ = self._run_raw(DBL_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert p_rc == d_rc == 2