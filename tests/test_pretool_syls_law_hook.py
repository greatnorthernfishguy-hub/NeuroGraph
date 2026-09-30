# ---- Changelog ----
# [2026-09-30] Chief-003 / Claude — Fold pass: doublecheck worktree-aware + FIX-UP defects
# -------------------

import os, sys, json, time, fcntl, shutil, pytest, pty as pty_module, subprocess, tempfile

BASE_HOOK = os.path.join(os.path.dirname(__file__), "fixtures", "pretool_syls_law_base.sh")
NEW_HOOK = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "pretool_syls_law.sh")
DBL_HOOK = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "hooks", "posttool_syls_law_doublecheck.sh")
DBL_BASE = os.path.join(os.path.dirname(__file__), "fixtures", "posttool_syls_law_doublecheck_base.sh")

PROTECTED_RELS = [
    "data/checkpoints/main.msgpack", "data/checkpoints/vectors.msgpack",
    "data/checkpoints/main.msgpack.activations.json",
    "neuro_foundation.py", "openclaw_hook.py", "stream_parser.py", "activation_persistence.py",
    "ng_lite.py", "ng_tract_bridge.py", "ng_ecosystem.py", "ng_autonomic.py",
    "openclaw_adapter.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py",
    "ng_peer_bridge.py",
]

POS = [
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
NEG = [
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
        for rel in ["README.md", "tests/test_foo.py"]:
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
        # No-origin repo
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

    def _env(self, extra=None):
        e = {"HOME": self._fake_home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if extra: e.update(extra)
        return e

    def _run(self, hook, path, env=None):
        ti = json.dumps({"tool_input": {"file_path": path}})
        r = subprocess.run([hook], input=ti.encode(), capture_output=True, timeout=15, start_new_session=True, env=env or self._env())
        return r.returncode, r.stderr.decode(errors="replace")

    def _run_raw(self, hook, stdin_bytes, env=None):
        r = subprocess.run([hook], input=stdin_bytes, capture_output=True, timeout=15, start_new_session=True, env=env or self._env())
        return r.returncode, r.stderr.decode(errors="replace"), r.stdout.decode(errors="replace")

    def _run_pty(self, hook, path, choice):
        env = self._env()
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

    def _env_no_git(self):
        return {"HOME": self._fake_home, "PATH": "/usr/bin:/bin"}

    def _env_stub_no_git(self):
        """PATH that keeps jq and timeout but excludes git."""
        jq_dir = os.path.dirname(subprocess.check_output(["command", "-v", "jq"]).decode().strip())
        timeout_dir = os.path.dirname(subprocess.check_output(["command", "-v", "timeout"]).decode().strip())
        return {"HOME": self._fake_home, "PATH": f"{jq_dir}:{timeout_dir}:/usr/bin:/bin"}

    # ═══════════════════════════════════════════════════════════════════
    # PRECONDITION: jq/git stubs
    # ═══════════════════════════════════════════════════════════════════

    def test_precondition_git_missing_stub_has_jq_not_git(self):
        e = self._env_stub_no_git()
        r = subprocess.run(["bash", "-c", "command -v jq && ! command -v git && echo OK"], capture_output=True, env=e)
        assert b"OK" in r.stdout, f"stub wrong: {r.stdout}"

    def test_precondition_jq_missing_stub_lacks_jq(self):
        e = self._env_no_git()
        r = subprocess.run(["bash", "-c", "command -v jq; echo EXIT:$?"], capture_output=True, env=e)
        assert b"EXIT:1" in r.stdout, f"jq still found: {r.stdout}"

    # ═══════════════════════════════════════════════════════════════════
    # FIX-UP: git-missing (proper stubs)
    # ═══════════════════════════════════════════════════════════════════

    def test_git_missing_preflight_exit2(self):
        e = self._env_stub_no_git()
        r = subprocess.run([NEW_HOOK], input=b'{"tool_input":{"file_path":"/x"}}', capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2, f"git missing preflight should exit 2, got {r.returncode}"

    def test_git_missing_old_literal_fires_preflight(self):
        """Old literal paths still fire even with preflight (preflight guards for pretool)."""
        e = self._env_stub_no_git()
        # With git absent, preflight exits 2 before ANY check
        path = os.path.join(self._ng_dir, "neuro_foundation.py")
        r = subprocess.run([NEW_HOOK], input=json.dumps({"tool_input": {"file_path": path}}).encode(), capture_output=True, timeout=15, start_new_session=True, env=e)
        assert r.returncode == 2

    # ═══════════════════════════════════════════════════════════════════
    # FIX-UP: non-existent directory
    # ═══════════════════════════════════════════════════════════════════

    def test_nonexistent_dir_allowed(self):
        """Write into a non-existent directory outside any repo is allowed."""
        d = os.path.join(self._non_git, "nonexistent_sub", "file.txt")
        rc, _ = self._run(NEW_HOOK, d)
        assert rc == 0, f"nonexistent dir should be allowed, got {rc}"

    # ═══════════════════════════════════════════════════════════════════
    # FIX-UP: no-origin repo allowed
    # ═══════════════════════════════════════════════════════════════════

    def test_no_origin_repo_allowed(self):
        """Repo with no remote at all → not NeuroGraph, allowed."""
        path = os.path.join(self._no_origin, "neuro_foundation.py")
        rc, _ = self._run(NEW_HOOK, path)
        assert rc == 0, f"no-origin repo should exit 0, got {rc}"

    # ═══════════════════════════════════════════════════════════════════
    # FIX-UP: LC_ALL=C locale independence
    # ═══════════════════════════════════════════════════════════════════

    def test_non_english_locale(self):
        """Unprotected non-git path allowed even with non-English locale."""
        e = self._env({"LC_ALL": "fr_FR.UTF-8", "LANG": "fr_FR.UTF-8"})
        path = os.path.join(self._non_git, "neuro_foundation.py")
        rc, _ = self._run(NEW_HOOK, path, env=e)
        assert rc == 0, f"non-English locale should allow, got {rc}"

    # ═══════════════════════════════════════════════════════════════════
    # FIX-UP: origin positive/negative matrix (includes new forms)
    # ═══════════════════════════════════════════════════════════════════

    def test_origin_positive_matrix(self):
        for origin in POS:
            repo = self._tmp_repo(origin)
            rc, stderr = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
            assert rc != 0, f"POSITIVE origin should match: {origin}"
            assert origin not in stderr
            shutil.rmtree(repo, ignore_errors=True)

    def test_origin_negative_matrix(self):
        for origin in NEG:
            repo = self._tmp_repo(origin)
            rc, stderr = self._run(NEW_HOOK, os.path.join(repo, "neuro_foundation.py"))
            assert rc == 0, f"NEGATIVE origin should NOT match: {origin}"
            shutil.rmtree(repo, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════════
    # PRETOOL: existing core tests
    # ═══════════════════════════════════════════════════════════════════

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
        corpus.append(("ckpt_s", os.path.join(self._ng_dir, "data", "checkpoints", "sub.msgpack"), {}))
        corpus.append(("ckpt_o", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))
        for rel in ["README.md", "tests/test_foo.py"]:
            corpus.append(("unprot", os.path.join(self._ng_dir, rel), {}))
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
            be, _ = self._run(BASE_HOOK, path, env=self._env(extra or None))
            ne, _ = self._run(NEW_HOOK, path, env=self._env(extra or None))
            if be != 0: bf.add((label, path))
            if ne != 0: nf.add((label, path))
        assert not (bf - nf), f"base fires but new does not: {sorted(bf - nf)[:5]}"
        new_vendored = {"ng_tract_bridge.py", "ng_embed.py", "ng_salience_gate.py", "ng_updater.py"}
        exp = set()
        for rel in PROTECTED_RELS:
            for wt_src, wt_dir in [("wt_o", self._wt_outside), ("wt_i", self._wt_inside), ("wt_n", self._wt_nested)]:
                exp.add((wt_src, os.path.join(wt_dir, rel)))
            exp.add(("rel_wo", rel)); exp.add(("rel_wi", rel))
            if os.path.basename(rel) in new_vendored:
                exp.add((f"sl_{os.path.basename(rel)}", os.path.join(lk, os.path.basename(rel))))
        for r in new_vendored:
            exp.add(("home", os.path.join(self._ng_dir, r)))
            exp.add(("rel_m", r))
        exp.add(("wt_o", os.path.join(self._wt_outside, "data", "checkpoints", "sub.msgpack")))
        act = nf - bf
        assert not (exp - act), f"missing: {sorted(exp - act)[:5]}"
        assert not (act - exp), f"unexpected: {sorted(act - exp)[:5]}"
        print(f"c:{len(corpus)} bf:{len(bf)} nf:{len(nf)} add:{len(act)}")

    def test_old_literal_still_fires(self):
        for rel in PROTECTED_RELS:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] != 0

    def test_peer_bridge_retired(self):
        rc, stderr = self._run(NEW_HOOK, os.path.join(self._ng_dir, "ng_peer_bridge.py"))
        assert rc != 0; assert "Retired" in stderr

    def test_no_overmatch(self):
        for rel in ["README.md", "tests/test_foo.py"]:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0
        for rel in ["README.md"]:
            assert self._run(NEW_HOOK, os.path.join(self._wt_outside, rel))[0] == 0
        assert self._run(NEW_HOOK, os.path.join(self._unrelated, "neuro_foundation.py"))[0] == 0
        assert self._run(NEW_HOOK, os.path.join(self._non_git, "neuro_foundation.py"))[0] == 0

    def test_bypass(self):
        self._install_bypass()
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack"]:
            assert self._run(NEW_HOOK, os.path.join(self._ng_dir, rel))[0] == 0
        assert self._run(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"))[0] == 0

    def test_bypass_location(self):
        bp = os.path.join(self._fake_home, "NeuroGraph", ".claude", "hooks", ".session_approved")
        try: os.unlink(bp)
        except: pass
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"), "3")
        assert rc == 0; assert "bypass" in out.lower(); assert os.path.isfile(bp)

    def test_pty_approve(self):
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "1")
        assert rc == 0; assert "Approved" in out

    def test_pty_block(self):
        rc, out = self._run_pty(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"), "2")
        assert rc == 2; assert "BLOCKED" in out

    def test_no_tty_blocks(self):
        assert self._run(NEW_HOOK, os.path.join(self._ng_dir, "neuro_foundation.py"))[0] == 2

    def test_fault_jq_missing(self):
        e = self._env_no_git()
        rc, stderr = self._run_raw(NEW_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 2; assert "TOOL MISSING" in stderr

    def test_fault_empty_stdin(self):
        rc, stderr, _ = self._run_raw(NEW_HOOK, b"")
        assert rc == 2; assert "EMPTY" in stderr

    def test_fault_no_path(self):
        rc, stderr, _ = self._run_raw(NEW_HOOK, b'{"tool_input":{}}')
        assert rc == 2

    def test_symlink_outside(self):
        lk = os.path.join(self._tmpdir, "sl2")
        os.makedirs(lk, exist_ok=True)
        for rel in ["neuro_foundation.py", "data/checkpoints/main.msgpack", "ng_lite.py", "ng_peer_bridge.py"]:
            t = os.path.join(self._ng_dir, rel)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.islink(lp):
                os.symlink(t, lp)
            assert self._run(NEW_HOOK, lp)[0] != 0

    def test_nested_worktree(self):
        for rel in PROTECTED_RELS:
            assert self._run(NEW_HOOK, os.path.join(self._wt_nested, rel))[0] != 0

    # ═══════════════════════════════════════════════════════════════════
    # DOUBLECHECK: differential + contract parity
    # ═══════════════════════════════════════════════════════════════════

    def _make_dbl_corpus(self):
        c = []
        for rel in PROTECTED_RELS:
            c.append(("home", os.path.join(self._ng_dir, rel), {}))
            c.append(("wt_o", os.path.join(self._wt_outside, rel), {}))
            c.append(("wt_i", os.path.join(self._wt_inside, rel), {}))
            c.append(("wt_n", os.path.join(self._wt_nested, rel), {}))
        c.append(("ckpt", os.path.join(self._ng_dir, "data", "checkpoints", "sub.msgpack"), {}))
        c.append(("ckpt_o", os.path.join(self._ng_dir, "data", "checkpoints-old", "stale.txt"), {}))
        for rel in ["README.md", "tests/test_foo.py"]:
            c.append(("un", os.path.join(self._ng_dir, rel), {}))
        lk = os.path.join(self._tmpdir, "sl_dbl")
        os.makedirs(lk, exist_ok=True)
        for rel in PROTECTED_RELS:
            t = os.path.join(self._ng_dir, rel)
            lp = os.path.join(lk, os.path.basename(rel))
            if not os.path.islink(lp):
                try: os.symlink(t, lp)
                except FileExistsError: pass
            c.append((f"sl_{os.path.basename(rel)}", lp, {}))
        c.append(("unr", os.path.join(self._unrelated, "neuro_foundation.py"), {}))
        c.append(("nogit", os.path.join(self._non_git, "neuro_foundation.py"), {}))
        return c

    def test_doublecheck_differential(self):
        corpus = self._make_dbl_corpus()
        bf = set(); nf = set()
        for label, path, extra in corpus:
            be, _ = self._run(DBL_BASE, path, env=self._env(extra or None))
            ne, _ = self._run(DBL_HOOK, path, env=self._env(extra or None))
            if be == 2: bf.add((label, path))
            if ne == 2: nf.add((label, path))
        assert not (bf - nf), f"dbl base fires but new does not: {sorted(bf - nf)[:5]}"
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

    def test_doublecheck_parity_protected(self):
        """Base and new both exit 2 with same stderr for a literal protected hit."""
        path = os.path.join(self._ng_dir, "neuro_foundation.py")
        brc, bstderr = self._run(DBL_BASE, path)
        nrc, nstderr = self._run(DBL_HOOK, path)
        assert brc == 2 and nrc == 2
        assert "PROTECTED FILE WAS MODIFIED" in bstderr
        assert "PROTECTED FILE WAS MODIFIED" in nstderr

    def test_doublecheck_parity_unprotected(self):
        """Both exit 0 for unprotected file."""
        path = os.path.join(self._ng_dir, "README.md")
        assert self._run(DBL_BASE, path)[0] == 0
        assert self._run(DBL_HOOK, path)[0] == 0

    def test_doublecheck_fires_worktree(self):
        assert self._run(DBL_HOOK, os.path.join(self._wt_outside, "neuro_foundation.py"))[0] == 2
        assert self._run(DBL_HOOK, os.path.join(self._wt_nested, "ng_lite.py"))[0] == 2

    def test_doublecheck_fault_jq_missing(self):
        e = self._env_no_git()
        rc, stderr, _ = self._run_raw(DBL_HOOK, b'{"tool_input":{"file_path":"/x"}}', env=e)
        assert rc == 2; assert "TOOL MISSING" in stderr

    def test_home_guard(self):
        assert self._fake_home.startswith(self._tmpdir)