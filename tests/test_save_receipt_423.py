"""#423 phase A — focused tests for the opt-in save receipt.

# ---- Changelog ----
# [2026-09-11] Claude Code (Opus 5) — #423 save-receipt tests
# What: exercises NeuroGraphMemory.save()'s real body — legacy string/exception
#   contract, strict primary acceptance, quarantine, and every component
#   failure mode (graph, vectors, activations, manifest, generation).
# Why: the receipt exists to make swallowed failures visible; only tests that
#   drive the ACTUAL save body can show it does. A test against a reimplemented
#   copy would prove nothing about the shipped code.
# How: openclaw_hook imports neuro_foundation and the embedding stack, which we
#   must not load, so the module-level receipt helpers and the save() body are
#   AST-extracted and compiled into a private namespace. checkpoint_guardian is
#   stdlib-only, so the REAL atomic writers, manifest and generation ring run —
#   real hardlinks, real rotation. Graph, vector DB, save gate and activation
#   persistence are disposable fakes over tmp_path. Nothing constructs
#   NeuroGraphMemory and no model is ever loaded.
# -------------------
"""

import ast
import json
import os
import shutil
import sys
import time
import types
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import checkpoint_guardian  # stdlib-only; safe to import for real

HOOK_PATH = os.path.join(REPO_ROOT, "openclaw_hook.py")

# Module-level helpers the save body leans on.
_HELPERS = (
    "_sync_receipt_artifacts",
    "_file_identity",
    "_same_inode",
    "_identity_unchanged",
    "_stream_sha256",
    "_sidecar_defect",
    "_verify_generation",
    "_new_receipt_components",
)


def _build_namespace():
    """Compile openclaw_hook's receipt helpers + save() body in isolation.

    AST extraction rather than import: openclaw_hook pulls in neuro_foundation
    and the embedding stack at module scope. We take the real source of the
    real functions, so the code under test is the code that ships.
    """
    tree = ast.parse(Path(HOOK_PATH).read_text(), filename=HOOK_PATH)

    wanted = []
    save_fn = None
    receipt_components = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS:
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("_RECEIPT_COMPONENTS",
                                                        "_HASH_CHUNK"):
                    wanted.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "NeuroGraphMemory":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "save":
                    save_fn = sub

    assert save_fn is not None, "save() not found in NeuroGraphMemory"
    missing = {h for h in _HELPERS} - {
        n.name for n in wanted if isinstance(n, ast.FunctionDef)}
    assert not missing, f"helpers not found in openclaw_hook: {sorted(missing)}"
    del receipt_components

    module = ast.Module(
        body=[ast.ImportFrom(module="__future__",
                             names=[ast.alias(name="annotations", asname=None)],
                             level=0)] + wanted + [save_fn],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    ns = {
        "__name__": "_ng_save_under_test",
        "__file__": HOOK_PATH,
        "os": os, "json": json, "time": time, "hashlib": __import__("hashlib"),
        "Path": Path,
        "Any": object, "Dict": dict, "List": list, "Optional": object,
        "logger": _Logger(),
        "CheckpointMode": types.SimpleNamespace(FULL="FULL"),
        "atomic_file_write": checkpoint_guardian.atomic_file_write,
        "quarantine_save": checkpoint_guardian.quarantine_save,
        "write_manifest": checkpoint_guardian.write_manifest,
        "rotate_generations": checkpoint_guardian.rotate_generations,
        "best_effort_git_hash": checkpoint_guardian.best_effort_git_hash,
    }
    exec(compile(module, HOOK_PATH, "exec"), ns)
    return ns


NS = _build_namespace()
save = NS["save"]


# ---- disposable fakes (no NG, no model, no real graph) ----

class FakeGraph:
    def __init__(self, nodes=12, fail=False):
        self.nodes = {f"n{i}": i for i in range(nodes)}
        self.synapses = {"s0": 1, "s1": 2}
        self.hyperedges = {"h0": 1}
        self.timestep = 4242
        self.fail = fail
        self.payload = b"graph-bytes-v1"

    def checkpoint(self, path, mode=None):
        if self.fail:
            raise IOError("graph checkpoint write exploded")
        with open(path, "wb") as f:
            f.write(self.payload)
        return path


class FakeVectorDB:
    def __init__(self, fail=False, entries=7):
        self.fail = fail
        self.entries = entries
        self.payload = b"vector-bytes-v1"

    def save(self, path):
        if self.fail:
            raise IOError("vector DB write exploded")
        with open(path, "wb") as f:
            f.write(self.payload)
        return self.entries

    def count(self):
        return self.entries


class FakeGate:
    """Stands in for checkpoint_guardian.SaveGate."""

    def __init__(self, permit=True, reason="refused-by-test"):
        self._permit = permit
        self._reason = reason

    def permit(self, nodes, live_synapses=None, live_hyperedges=None,
               wires_own_deposits=None):
        return (True, "ok") if self._permit else (False, self._reason)


class FakeActivation:
    """Mirrors ActivationPersistence.save()'s real contract.

    It writes IN PLACE, swallows its own write exceptions, returns the path
    either way, and assigns ``_last_save_time`` ONLY after the write context
    exits cleanly. ``capture`` failures are raised from OUTSIDE that try, so
    they propagate — same as the real writer.
    """

    def __init__(self, mode="ok"):
        self.mode = mode
        self._last_save_time = None

    def save(self, graph, checkpoint_path):
        path = str(checkpoint_path) + ".activations.json"
        if self.mode == "raise_capture":
            raise RuntimeError("capture() exploded before the writer's try")
        data = {"version": "1.0", "saved_at": time.time(),
                "timestep": graph.timestep, "entries": {"n0": 0.5}}
        try:
            if self.mode != "noop":
                with open(path, "w") as f:
                    if self.mode == "truncate":
                        f.write(json.dumps(data)[:18])
                        raise IOError("disk full mid-dump")
                    json.dump(data, f)
            if self.mode == "swallow_flush":
                # Complete, well-formed file on disk — but the writer's own
                # success signal never advances.
                raise IOError("flush failed on close")
            if self.mode == "noop":
                raise IOError("never opened the file")
            self._last_save_time = data["saved_at"]
        except Exception:
            pass  # exactly what the real writer does: log and carry on
        return path


class FakeSelf:
    def __init__(self, tmp, graph=None, vdb=None, gate=None, activation="ok"):
        self._checkpoint_dir = Path(tmp)
        self._checkpoint_path = Path(tmp) / "main.msgpack"
        self._vector_db_path = Path(tmp) / "vectors.msgpack"
        self.graph = graph or FakeGraph()
        self.vector_db = vdb or FakeVectorDB()
        self._save_gate = FakeGate() if gate is None else gate
        self._activation_persistence = (
            FakeActivation(activation) if isinstance(activation, str)
            else activation)

    def _guardian_meaningful_nodes(self):
        return len(self.graph.nodes)


class ReceiptTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="ng423-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def sut(self, **kw):
        return FakeSelf(self.tmp, **kw)


# ---- legacy contract: unchanged ----

class TestLegacyContract(ReceiptTestCase):
    def test_primary_returns_checkpoint_path_string(self):
        me = self.sut()
        out = save(me)
        self.assertIsInstance(out, str)
        self.assertEqual(out, str(me._checkpoint_path))
        self.assertTrue(os.path.exists(out))

    def test_quarantine_returns_quarantine_path_string(self):
        me = self.sut(gate=FakeGate(permit=False))
        out = save(me)
        self.assertIsInstance(out, str)
        self.assertIn("quarantine", out)
        # primary left untouched
        self.assertFalse(me._checkpoint_path.exists())

    def test_graph_write_exception_still_propagates(self):
        me = self.sut(graph=FakeGraph(fail=True))
        with self.assertRaises(IOError):
            save(me)

    def test_activation_capture_exception_still_propagates(self):
        me = self.sut(activation="raise_capture")
        with self.assertRaises(RuntimeError):
            save(me)

    def test_vector_failure_is_still_swallowed_and_returns_primary(self):
        """The defect the receipt exists to expose — legacy is unchanged."""
        me = self.sut(vdb=FakeVectorDB(fail=True))
        out = save(me)
        self.assertEqual(out, str(me._checkpoint_path))

    def test_legacy_mode_does_no_verification_work(self):
        """with_receipt=False must not stat or hash anything extra."""
        calls = {"identity": 0, "hash": 0}
        real_identity, real_hash = NS["_file_identity"], NS["_stream_sha256"]

        def counting_identity(p):
            calls["identity"] += 1
            return real_identity(p)

        def counting_hash(p):
            calls["hash"] += 1
            return real_hash(p)

        NS["_file_identity"], NS["_stream_sha256"] = counting_identity, counting_hash
        try:
            save(self.sut())
            self.assertEqual(calls, {"identity": 0, "hash": 0})
            save(self.sut(), with_receipt=True)
            self.assertGreater(calls["identity"], 0)
        finally:
            NS["_file_identity"], NS["_stream_sha256"] = real_identity, real_hash


# ---- receipt: the accepting case ----

class TestStrictPrimary(ReceiptTestCase):
    def test_clean_primary_save_is_accepted(self):
        r = save(self.sut(), with_receipt=True)
        self.assertEqual(r["outcome"], "primary")
        self.assertTrue(r["accepted"], r["not_accepted_because"])
        self.assertEqual(r["not_accepted_because"], [])
        for name in ("graph", "vectors", "activations", "manifest", "generation"):
            self.assertEqual(r["components"][name]["status"], "saved", name)
        for name, member in r["components"]["generation"]["members"].items():
            self.assertEqual(member["state"], "match", name)

    def test_msgpack_members_are_proven_by_hardlink_not_metadata(self):
        r = save(self.sut(), with_receipt=True)
        members = r["components"]["generation"]["members"]
        self.assertEqual(members["graph"]["evidence"], "hardlink")
        self.assertEqual(members["vectors"]["evidence"], "hardlink")
        # sidecars are copy2'd, so they must be proven by content
        self.assertEqual(members["manifest"]["evidence"], "content-sha256")
        self.assertEqual(members["activations"]["evidence"], "content-sha256")

    def test_receipt_carries_legacy_path_and_file_identity(self):
        me = self.sut()
        r = save(me, with_receipt=True)
        self.assertEqual(r["path"], str(me._checkpoint_path))
        ident = r["components"]["graph"]["identity"]
        self.assertEqual(ident["size"], len(me.graph.payload))
        self.assertIn("inode", ident)


# ---- receipt: quarantine ----

class TestQuarantine(ReceiptTestCase):
    def test_quarantine_is_never_accepted(self):
        r = save(self.sut(gate=FakeGate(permit=False)), with_receipt=True)
        self.assertEqual(r["outcome"], "quarantine")
        self.assertFalse(r["accepted"])
        self.assertIn("quarantine", r["path"])

    def test_omitted_components_are_not_attempted(self):
        r = save(self.sut(gate=FakeGate(permit=False)), with_receipt=True)
        for name in ("activations", "manifest", "generation"):
            self.assertEqual(r["components"][name]["status"], "not_attempted", name)
        self.assertEqual(r["components"]["graph"]["status"], "saved")
        self.assertEqual(r["components"]["vectors"]["status"], "saved")

    def test_quarantine_vector_failure_is_visible(self):
        r = save(self.sut(gate=FakeGate(permit=False), vdb=FakeVectorDB(fail=True)),
                 with_receipt=True)
        self.assertEqual(r["outcome"], "quarantine")
        self.assertEqual(r["components"]["vectors"]["status"], "failed")
        self.assertIn("exploded", r["components"]["vectors"]["error"])

    def test_quarantine_graph_failure_reports_failed_not_raise(self):
        me = self.sut(gate=FakeGate(permit=False), graph=FakeGraph(fail=True))
        r = save(me, with_receipt=True)
        self.assertEqual(r["outcome"], "failed")
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["graph"]["status"], "failed")


# ---- receipt: component failures ----

class TestComponentFailures(ReceiptTestCase):
    def test_graph_failure_yields_failed_outcome(self):
        r = save(self.sut(graph=FakeGraph(fail=True)), with_receipt=True)
        self.assertEqual(r["outcome"], "failed")
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["graph"]["status"], "failed")
        # nothing downstream may claim success
        for name in ("vectors", "activations", "manifest", "generation"):
            self.assertNotEqual(r["components"][name]["status"], "saved", name)

    def test_vector_failure_blocks_acceptance_on_primary_path(self):
        r = save(self.sut(vdb=FakeVectorDB(fail=True)), with_receipt=True)
        self.assertEqual(r["outcome"], "primary")
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["vectors"]["status"], "failed")
        self.assertTrue(any("vectors" in b for b in r["not_accepted_because"]))

    def test_truncated_sidecar_is_not_accepted(self):
        r = save(self.sut(activation="truncate"), with_receipt=True)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["activations"]["status"], "failed")

    def test_complete_sidecar_with_swallowed_flush_error_is_not_accepted(self):
        """A well-formed file on disk is NOT enough.

        The writer's flush failed after json.dump wrote a complete document,
        so the file parses and its mtime is fresh — but the writer never set
        its success signal. Stat-based checking would pass this; the token
        check must not.
        """
        me = self.sut(activation="swallow_flush")
        r = save(me, with_receipt=True)
        sidecar = str(me._checkpoint_path) + ".activations.json"
        self.assertTrue(os.path.exists(sidecar))
        with open(sidecar) as f:
            self.assertIn("entries", json.load(f))  # complete and well-formed
        self.assertIsNone(me._activation_persistence._last_save_time)
        self.assertEqual(r["components"]["activations"]["status"], "failed")
        self.assertFalse(r["accepted"])

    def test_sidecar_never_written_is_not_accepted(self):
        r = save(self.sut(activation="noop"), with_receipt=True)
        self.assertEqual(r["components"]["activations"]["status"], "failed")
        self.assertFalse(r["accepted"])

    def test_missing_ces_is_not_applicable_but_still_acceptable(self):
        r = save(self.sut(activation=None), with_receipt=True)
        self.assertEqual(r["components"]["activations"]["status"], "not_applicable")
        self.assertTrue(r["accepted"], r["not_accepted_because"])

    def test_manifest_failure_blocks_acceptance_and_skips_generation(self):
        real = NS["write_manifest"]

        def boom(*a, **k):
            raise IOError("manifest write exploded")

        NS["write_manifest"] = boom
        try:
            r = save(self.sut(), with_receipt=True)
        finally:
            NS["write_manifest"] = real
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["manifest"]["status"], "failed")
        self.assertEqual(r["components"]["generation"]["status"], "not_attempted")

    def test_generation_failure_blocks_acceptance(self):
        real = NS["rotate_generations"]

        def boom(*a, **k):
            raise OSError("rotation exploded")

        NS["rotate_generations"] = boom
        try:
            r = save(self.sut(), with_receipt=True)
        finally:
            NS["rotate_generations"] = real
        self.assertFalse(r["accepted"])
        self.assertEqual(r["components"]["manifest"]["status"], "saved")
        self.assertEqual(r["components"]["generation"]["status"], "failed")

    def test_no_guardian_fails_acceptance_rather_than_assuming_success(self):
        me = self.sut(gate=False)  # falsy-but-not-None handled below
        me._save_gate = None
        r = save(me, with_receipt=True)
        self.assertEqual(r["outcome"], "primary")
        self.assertFalse(r["accepted"])
        self.assertFalse(r["guardian_available"])
        for name in ("manifest", "generation"):
            self.assertEqual(r["components"][name]["status"], "not_applicable", name)
        self.assertEqual(r["components"]["graph"]["status"], "saved")


# ---- the generation ring cannot be trusted on its own ----

class TestGenerationEvidence(ReceiptTestCase):
    def test_stale_vectors_in_ring_cannot_count_as_accepted(self):
        """The partial-ring trap: vectors fail, but a previous vectors.msgpack
        is still on disk, so rotate_generations hardlinks the STALE one."""
        me = self.sut()
        save(me, with_receipt=True)                 # generation 1: good
        self.assertTrue(me._vector_db_path.exists())

        me.vector_db = FakeVectorDB(fail=True)      # generation 2: vectors fail
        r = save(me, with_receipt=True)

        self.assertEqual(r["components"]["vectors"]["status"], "failed")
        self.assertEqual(r["components"]["generation"]["status"], "failed")
        members = r["components"]["generation"]["members"]
        self.assertEqual(members["vectors"]["state"], "stale")
        # the stale file really is sitting in the ring, looking healthy
        self.assertIsNotNone(members["vectors"]["identity"])
        self.assertFalse(r["accepted"])

    def test_missing_member_is_detected(self):
        ident = NS["_file_identity"]
        src = os.path.join(self.tmp, "main.msgpack")
        Path(src).write_bytes(b"abc")
        gen = os.path.join(self.tmp, "gen")
        os.makedirs(gen)
        out = NS["_verify_generation"](gen, [("graph", "main.msgpack", src, ident(src))])
        self.assertFalse(out["ok"])
        self.assertEqual(out["members"]["graph"]["state"], "missing")

    def test_absent_generation_directory_is_not_evidence(self):
        out = NS["_verify_generation"](os.path.join(self.tmp, "nope"), [])
        self.assertFalse(out["ok"])
        self.assertIn("missing", out["error"])

    def test_corrupt_copy_with_identical_size_and_mtime_is_caught(self):
        """Metadata equality is NOT content proof.

        shutil.copy2 preserves size and mtime_ns exactly, so a substituted
        member of the same length with a copied timestamp is indistinguishable
        from the real one by stat alone. Only content settles it.
        """
        ident = NS["_file_identity"]
        src = os.path.join(self.tmp, "sidecar.json")
        Path(src).write_bytes(b"AAAAAAAAAA")
        written = ident(src)

        gen = os.path.join(self.tmp, "gen")
        os.makedirs(gen)
        member = os.path.join(gen, "sidecar.json")
        Path(member).write_bytes(b"BBBBBBBBBB")        # same length, other bytes
        st = os.stat(src)
        os.utime(member, ns=(st.st_atime_ns, st.st_mtime_ns))

        m_ident = ident(member)
        self.assertEqual(m_ident["size"], written["size"])
        self.assertEqual(m_ident["mtime_ns"], written["mtime_ns"])
        self.assertNotEqual(m_ident["inode"], written["inode"])

        out = NS["_verify_generation"](gen, [("manifest", "sidecar.json", src, written)])
        self.assertFalse(out["ok"], "same size+mtime must not pass as evidence")
        self.assertEqual(out["members"]["manifest"]["state"], "stale")
        self.assertEqual(out["members"]["manifest"]["evidence"], "content-sha256")

    def test_honest_copy_passes_by_content(self):
        ident = NS["_file_identity"]
        src = os.path.join(self.tmp, "sidecar.json")
        Path(src).write_bytes(b"real-payload")
        written = ident(src)
        gen = os.path.join(self.tmp, "gen")
        os.makedirs(gen)
        shutil.copy2(src, os.path.join(gen, "sidecar.json"))
        out = NS["_verify_generation"](gen, [("manifest", "sidecar.json", src, written)])
        self.assertTrue(out["ok"])
        self.assertEqual(out["members"]["manifest"]["state"], "match")

    def test_source_changed_after_write_is_unverifiable(self):
        ident = NS["_file_identity"]
        src = os.path.join(self.tmp, "sidecar.json")
        Path(src).write_bytes(b"first")
        written = ident(src)
        gen = os.path.join(self.tmp, "gen")
        os.makedirs(gen)
        shutil.copy2(src, os.path.join(gen, "sidecar.json"))
        time.sleep(0.01)
        Path(src).write_bytes(b"someone else wrote this")   # source moved
        out = NS["_verify_generation"](gen, [("manifest", "sidecar.json", src, written)])
        self.assertFalse(out["ok"])
        self.assertEqual(out["members"]["manifest"]["state"], "unverifiable")


class TestNoPrivateModel(unittest.TestCase):
    def test_receipt_reports_only_observed_disk_state(self):
        """No component may be reported saved without an artifact behind it."""
        import tempfile
        tmp = tempfile.mkdtemp(prefix="ng423-")
        try:
            me = FakeSelf(tmp)
            r = save(me, with_receipt=True)
            for name, comp in r["components"].items():
                if comp["status"] != "saved":
                    continue
                if name == "generation":
                    self.assertTrue(os.path.isdir(comp["path"]), name)
                else:
                    self.assertTrue(os.path.exists(comp["path"]), name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAcceptanceFlush(ReceiptTestCase):
    def test_receipt_flushes_files_and_publication_directories(self):
        import stat
        from unittest.mock import patch
        calls = []
        real = os.fsync
        def sync(fd):
            calls.append(stat.S_ISDIR(os.fstat(fd).st_mode))
            return real(fd)
        with patch.object(os, 'fsync', side_effect=sync):
            result = save(self.sut(), with_receipt=True)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['components']['generation']['durability'], 'fsynced')
        self.assertIn(False, calls)
        self.assertIn(True, calls)
        first_directory = calls.index(True)
        self.assertTrue(all(calls[first_directory:]))

    def test_file_flush_failure_refuses_acceptance(self):
        from unittest.mock import patch
        with patch.object(os, 'fsync', side_effect=OSError('storage failure')):
            result = save(self.sut(), with_receipt=True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['components']['generation']['status'], 'failed')

    def test_directory_flush_failure_refuses_acceptance(self):
        import stat
        from unittest.mock import patch
        real = os.fsync
        def sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('directory publication failure')
            return real(fd)
        with patch.object(os, 'fsync', side_effect=sync):
            result = save(self.sut(), with_receipt=True)
        self.assertFalse(result['accepted'])

    def test_legacy_save_does_not_gain_receipt_flushes(self):
        from unittest.mock import patch
        with patch.object(os, 'fsync', side_effect=AssertionError('receipt-only operation')):
            self.assertIsInstance(save(self.sut()), str)
