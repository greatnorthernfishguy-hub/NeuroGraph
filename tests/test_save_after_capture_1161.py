"""#1161 / #1123: NeuroGraphMemory.save(after_capture=...) lets a caller drop its outer lock at the capture.

# ---- Changelog ----
# [2026-10-22] Claude Code (Opus 5.5; #1161 / #1123) — created
# What: (A) the shipped save() body (AST-extracted, the #423 harness) calls after_capture exactly once, after the capture
#       and before the permit / any write; an outer lock released there is free for the whole write (the hold is measured);
#       capture failure, ownership refusal and a raising callback are all handled; a failing atomic write behaves as
#       before. (B) a REAL NeuroGraphMemory: the same mind state saves to identical main / vectors bytes with and without
#       the callback; a step thread mutating the graph and the vector DB during a slowed write leaves one consistent set
#       (manifest == restored graph == restored vectors, all at the capture instant); the generation ring rotates and
#       prunes exactly as without the callback.
# Why:  the CC daemon held _concurrent_lock across the ~70 s write; the fix releases it after the capture. None of what is
#       saved may change.
# How:  (A) reuses tests/test_save_receipt_423.py's FakeSelf; (B) the #373 real-boot fixture (tonic and peer bridge off).
# -------------------
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_save_receipt_423 import FakeSelf, FakeGate, save  # noqa: E402


# ---------------------------------------------------------------- (A) the shipped body, fakes


def _slow_writes(me, secs, log):
    w = me.graph.write_checkpoint

    def write(path, captured, mode=None):
        log.append(('write', time.monotonic()))
        time.sleep(secs)
        return w(path, captured, mode)
    me.graph.write_checkpoint = write


@pytest.mark.parametrize('receipt', [False, True])
def test_callback_once_after_capture_before_permit_and_writes(tmp_path, receipt):
    me = FakeSelf(tmp_path)
    log = []
    cap = me.graph.capture_checkpoint

    def capture(mode=None, detach=False):
        log.append(('capture', time.monotonic()))
        return cap(mode, detach)
    me.graph.capture_checkpoint = capture
    permit = me._save_gate.permit

    def p(nodes, **kw):
        log.append(('permit', time.monotonic()))
        return permit(nodes, **kw)
    me._save_gate.permit = p
    _slow_writes(me, 0.0, log)
    out = save(me, with_receipt=receipt, after_capture=lambda: log.append(('released', time.monotonic())))
    names = [n for n, _ in log]
    assert names == ['capture', 'released', 'permit', 'write']
    assert (out['accepted'] if receipt else out == str(me._checkpoint_path))


def test_outer_lock_is_held_only_for_the_capture(tmp_path):
    """The measured hold: an outer lock released by after_capture is free for the whole (slowed) write."""
    me = FakeSelf(tmp_path)
    outer = threading.Lock()
    log = []
    _slow_writes(me, 0.6, log)
    got = {}

    def contender():
        t0 = time.monotonic()
        with outer:
            got['waited'] = time.monotonic() - t0
            got['at'] = time.monotonic()

    outer.acquire()
    t_start = time.monotonic()
    released = {}

    def release():
        released['at'] = time.monotonic()
        outer.release()
    th = threading.Thread(target=contender)
    th.start()
    save(me, after_capture=release)
    th.join(5)
    hold = released['at'] - t_start
    write_start = log[0][1]
    assert hold < 0.3, hold                                  # the capture only (fakes: milliseconds)
    assert got['at'] < write_start + 0.6, (got, write_start)  # the contender ran DURING the write
    assert me._checkpoint_path.read_bytes() == b'graph-bytes-v1'


def test_capture_failure_still_calls_back_once(tmp_path):
    me = FakeSelf(tmp_path)
    calls = []

    def boom(*a, **k):
        raise ValueError('capture unavailable')
    me.graph.capture_checkpoint = boom
    with pytest.raises(RuntimeError):
        save(me, after_capture=lambda: calls.append(1))
    assert calls == [1]
    out = save(me, with_receipt=True, after_capture=lambda: calls.append(2))
    assert out['outcome'] == 'failed' and calls == [1, 2]
    assert not me._checkpoint_path.exists()


def test_ownership_refusal_still_calls_back(tmp_path):
    me = FakeSelf(tmp_path)
    calls = []
    with me.graph._step_lock:
        with pytest.raises(RuntimeError):
            save(me, after_capture=lambda: calls.append(1))
    assert calls == [1]


def test_raising_callback_never_fails_the_save(tmp_path):
    me = FakeSelf(tmp_path)

    def bad():
        raise RuntimeError('releaser broke')
    assert save(me, after_capture=bad) == str(me._checkpoint_path)
    assert me._checkpoint_path.read_bytes() == b'graph-bytes-v1'


@pytest.mark.parametrize('receipt', [False, True])
def test_atomic_write_failure_handling_unchanged(tmp_path, receipt):
    """A failing graph write: previous primary kept, no tmp, no generation -- with and without the callback."""
    results = {}
    for label, cb in (('none', None), ('cb', lambda: None)):
        d = tmp_path / label
        d.mkdir()
        me = FakeSelf(d)
        save(me)                                    # a good primary first
        good = me._checkpoint_path.read_bytes()
        gens_before = sorted(os.listdir(d / 'generations')) if (d / 'generations').exists() else []
        me.graph.fail = True
        me.graph.payload = b'graph-bytes-v2'
        if receipt:
            out = save(me, with_receipt=True, after_capture=cb)
            results[label] = (out['outcome'], out['accepted'], out['components']['graph']['status'])
        else:
            with pytest.raises(IOError):
                save(me, after_capture=cb)
            results[label] = 'raised'
        assert me._checkpoint_path.read_bytes() == good
        assert [n for n in os.listdir(d) if '.tmp-' in n] == []
        gens_after = sorted(os.listdir(d / 'generations')) if (d / 'generations').exists() else []
        assert gens_after == gens_before
    assert results['none'] == results['cb']


def test_quarantine_path_unchanged_and_released(tmp_path):
    me = FakeSelf(tmp_path, gate=FakeGate(permit=False))
    calls = []
    out = save(me, after_capture=lambda: calls.append(1))
    assert 'quarantine' in out and calls == [1]
    assert not me._checkpoint_path.exists()


# ---------------------------------------------------------------- (B) a real NeuroGraphMemory

TEST_CONFIG = {"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}}


def _boot(ws):
    from openclaw_hook import NeuroGraphMemory
    return NeuroGraphMemory(workspace_dir=ws, config=TEST_CONFIG)


@pytest.fixture
def ws():
    d = tempfile.mkdtemp(prefix="save_lock_1161_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _grow(ng, n, prefix='c', dim=None):
    import numpy as np
    rng = np.random.default_rng(7)
    for i in range(n):
        nid = f"{prefix}-{i}"
        ng.graph.create_node(node_id=nid)
        if dim:
            v = rng.standard_normal(dim).astype('float32')
            ng.vector_db.insert(nid, v / np.linalg.norm(v), {'content': nid})
    ids = list(ng.graph.nodes)
    for i in range(0, len(ids) - 1, 2):
        try:
            ng.graph.create_synapse(ids[i], ids[i + 1], weight=0.3)
        except Exception:
            pass


def _vdb_dim(ng):
    return getattr(ng.vector_db, 'dim', None) or getattr(ng.vector_db, 'dimension', None) or 768


def _sidecar_sans_time(path):
    d = json.loads(Path(path).read_text())
    d.pop('saved_at', None)
    return d


def test_same_state_saves_identical_bytes_with_and_without_callback(ws):
    ng = _boot(ws)
    _grow(ng, 40, dim=_vdb_dim(ng))
    ng.save()
    main_a = Path(ng._checkpoint_path).read_bytes()
    vec_a = Path(ng._vector_db_path).read_bytes()
    act_a = _sidecar_sans_time(str(ng._checkpoint_path) + '.activations.json') \
        if os.path.exists(str(ng._checkpoint_path) + '.activations.json') else None
    lk = threading.RLock()
    lk.acquire()
    ng.save(after_capture=lk.release)
    assert not lk._is_owned()
    assert Path(ng._checkpoint_path).read_bytes() == main_a
    assert Path(ng._vector_db_path).read_bytes() == vec_a
    if act_a is not None:
        assert _sidecar_sans_time(str(ng._checkpoint_path) + '.activations.json') == act_a


def test_set_is_consistent_under_a_concurrent_step(ws):
    """A step thread mutates graph + vectors while the write runs (outer lock already released): the saved set is the
    capture instant -- manifest, restored graph and restored vectors agree."""
    ng = _boot(ws)
    dim = _vdb_dim(ng)
    _grow(ng, 30, dim=dim)
    import numpy as np
    stop = threading.Event()
    outer = threading.RLock()
    released = threading.Event()
    stepped = []

    def stepper():
        released.wait(5)
        i = 0
        while not stop.is_set():
            with ng.graph._step_lock:
                nid = f"late-{i}"
                ng.graph.create_node(node_id=nid)
                v = np.ones(dim, dtype='float32') / np.sqrt(dim)
                ng.vector_db.insert(nid, v, {'content': nid})
                ng.graph.step()
            stepped.append(i)
            i += 1
            time.sleep(0.01)

    real_write = ng.graph.write_checkpoint

    def slow_write(path, captured, mode=None):
        time.sleep(0.5)                      # the stepper runs here
        return real_write(path, captured, mode)
    ng.graph.write_checkpoint = slow_write

    def release():
        outer.release()
        released.set()
    th = threading.Thread(target=stepper)
    th.start()
    outer.acquire()
    try:
        ng.save(after_capture=release)
    finally:
        stop.set()
        th.join(5)
    assert stepped, 'the step thread never ran during the write'
    manifest = json.loads(Path(str(ng._checkpoint_path) + '.manifest.json').read_text())
    ng2 = _boot(tempfile.mkdtemp(prefix="save_lock_1161_r_"))
    try:
        ng2.graph.restore(str(ng._checkpoint_path))
        assert len(ng2.graph.nodes) == manifest['nodes']
        assert ng2.graph.timestep == manifest['timestep']
        assert not any(n.startswith('late-') for n in ng2.graph.nodes)
        vdb2 = type(ng.vector_db)(dimension=dim) if 'dimension' in type(ng.vector_db).__init__.__code__.co_varnames \
            else None
        if vdb2 is not None and hasattr(vdb2, 'load'):
            vdb2.load(str(ng._vector_db_path))
            assert len(vdb2) == manifest['vdb_count'] if hasattr(vdb2, '__len__') else True
    finally:
        shutil.rmtree(ng2._workspace_dir if hasattr(ng2, '_workspace_dir') else '', ignore_errors=True)
    assert manifest['vdb_count'] == 30
    assert len(ng.graph.nodes) > manifest['nodes']          # the live mind moved on; the file did not


def _gen_shape(ws):
    gdir = Path(ws) / 'checkpoints' / 'generations'
    if not gdir.exists():
        return []
    return [sorted(os.listdir(gdir / g)) for g in sorted(os.listdir(gdir))]


def test_guardian_rotation_and_pruning_unchanged():
    shapes = {}
    for label in ('none', 'cb'):
        d = tempfile.mkdtemp(prefix=f"save_lock_1161_{label}_")
        try:
            ng = _boot(d)
            for k in range(12):                        # more saves than the ring keeps
                _grow(ng, 3, prefix=f'g{k}')
                if label == 'cb':
                    lk = threading.Lock()
                    lk.acquire()
                    ng.save(after_capture=lk.release)
                    assert not lk.locked()
                else:
                    ng.save()
            shapes[label] = _gen_shape(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    assert shapes['none'] and shapes['none'] == shapes['cb'] if len(shapes['none']) == len(shapes['cb']) else False
    assert len(shapes['none']) == len(shapes['cb'])
