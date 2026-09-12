"""Canonical producer captures a common set and publishes outside mutation lock."""
# ---- Changelog ----
# [2026-09-11] Codex — common capture, failure and concurrent publication tests.
# What: real extracted producer with isolated fakes and thread barriers.
# Why: separate correct writers must still agree on the same state and lock order.
# How: resume mutations before first write; verify files, manifest and receipt.
# -------------------
import json
import threading
from types import MethodType

import pytest

from .test_save_receipt_423 import FakeSelf, FakeGate, save

@pytest.mark.parametrize('receipt,quarantine',[(True,False),(False,False),(True,True),(False,True)])
def test_saved_set_uses_capture_even_when_mutation_resumes(tmp_path,receipt,quarantine):
    me=FakeSelf(tmp_path,gate=FakeGate(permit=not quarantine))
    lock=me.graph._step_lock
    original_graph=me.graph.payload; original_vectors=me.vector_db.payload
    original_write=me.graph.write_checkpoint
    seen=[]
    def write_graph(path,captured,mode=None):
        assert not lock._is_owned()
        # Mutation resumes BEFORE the first artifact writer runs.
        with lock:
            me.graph.payload=b'later graph'; me.vector_db.payload=b'later vectors'
            me.graph.nodes['later']=999; me.graph.timestep=9999
            me.vector_db.entries=88
        seen.append('graph')
        return original_write(path,captured,mode)
    me.graph.write_checkpoint=write_graph
    vector_write=me.vector_db.write_state
    def write_vectors(path,captured):
        assert not lock._is_owned()
        seen.append('vectors')
        assert captured['count']==7
        return vector_write(path,captured)
    me.vector_db.write_state=write_vectors
    activation_write=me._activation_persistence.write_state
    def write_activation(path,captured,**kwargs):
        assert not lock._is_owned()
        assert captured['timestep']==4242
        seen.append('activations')
        return activation_write(path,captured,**kwargs)
    me._activation_persistence.write_state=write_activation
    gate=me._save_gate.permit
    def permit(nodes,**kw):
        assert not lock._is_owned()
        assert nodes==12
        return gate(nodes,**kw)
    me._save_gate.permit=permit
    result=save(me,with_receipt=receipt)
    if quarantine:
        path=result['path'] if receipt else result
        from pathlib import Path
        assert Path(path).read_bytes()==original_graph
        vector_paths=list((tmp_path/'quarantine').glob('*vectors*'))
        assert any(p.read_bytes()==original_vectors for p in vector_paths)
        assert seen==['graph','vectors']
    else:
        assert me._checkpoint_path.read_bytes()==original_graph
        assert me._vector_db_path.read_bytes()==original_vectors
        sidecar=json.loads((tmp_path/'main.msgpack.activations.json').read_text())
        assert sidecar['timestep']==4242
        manifest=json.loads((tmp_path/'main.msgpack.manifest.json').read_text())
        assert manifest['nodes']==12 and manifest['timestep']==4242
        assert manifest['vdb_count']==7
        assert seen==['graph','vectors','activations']
        if receipt:assert result['accepted']

@pytest.mark.parametrize('component',['graph','vectors','activations'])
def test_capture_failure_is_component_specific_and_writes_nothing(tmp_path,component):
    me=FakeSelf(tmp_path)
    def fail(*a,**k):raise ValueError('capture unavailable')
    if component=='graph':me.graph.capture_checkpoint=fail
    elif component=='vectors':me.vector_db.capture_state=fail
    else:me._activation_persistence.capture_state=fail
    result=save(me,with_receipt=True)
    assert result['outcome']=='failed' and not result['accepted']
    assert result['components'][component]['status']=='failed'
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError,match='capture unavailable'):save(me)


def test_save_rejects_caller_holding_mutation_lock_without_waiting_for_publication(tmp_path):
    me=FakeSelf(tmp_path)
    ready=threading.Event(); release=threading.Event()
    def publication_owner():
        with me._save_publication_lock:
            ready.set(); release.wait(2)
    worker=threading.Thread(target=publication_owner);worker.start()
    assert ready.wait(1)
    try:
        with me.graph._step_lock:
            result=save(me,with_receipt=True)
            assert not result['accepted']
            assert 'holding graph mutation lock' in result['components']['graph']['error']
            with pytest.raises(RuntimeError,match='holding graph mutation lock'):save(me)
        assert not list(tmp_path.iterdir())
    finally:
        release.set();worker.join(2)


def test_concurrent_saves_serialize_capture_through_publication(tmp_path):
    me=FakeSelf(tmp_path)
    entered=threading.Event(); release=threading.Event(); second_started=threading.Event()
    captures=[]; results=[]; failures=[]
    original_capture=me._capture_checkpoint_state
    def capture():
        captures.append(threading.current_thread().name)
        return original_capture()
    me._capture_checkpoint_state=capture
    original_write=me.graph.write_checkpoint
    def write(path,captured,mode=None):
        assert not me.graph._step_lock._is_owned()
        if threading.current_thread().name=='first':
            entered.set()
            assert release.wait(2)
        return original_write(path,captured,mode)
    me.graph.write_checkpoint=write
    def run(second=False):
        if second:second_started.set()
        try:results.append(save(me,with_receipt=True))
        except BaseException as exc:failures.append(exc)
    first=threading.Thread(target=run,name='first');first.start()
    assert entered.wait(1)
    second=threading.Thread(target=run,args=(True,),name='second');second.start()
    assert second_started.wait(1)
    try:
        assert captures==['first']
        # Disk writer's pause does not prevent graph mutation.
        with me.graph._step_lock:me.graph.payload=b'next generation'
    finally:
        release.set();first.join(3);second.join(3)
    assert not first.is_alive() and not second.is_alive() and not failures
    assert captures==['first','second']
    assert len(results)==2 and all(r['accepted'] for r in results)
    assert me._checkpoint_path.read_bytes()==b'next generation'


def test_associate_options_never_modify_checkpoint_configuration():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    source=Path(__file__).parents[1]/'openclaw_hook.py'
    cls=next(c for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef) and c.name=='NeuroGraphMemory')
    fn=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='associate')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns={};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
    config={'max_surfaced':10,'propagation_steps':3}
    def harvest(text,**options):
        # A capture or another concurrent query sees only the real configuration.
        assert config=={'max_surfaced':10,'propagation_steps':3}
        assert options=={'max_surfaced_override':7,'propagation_steps_override':5}
        return ['result']
    fake=SimpleNamespace(graph=SimpleNamespace(config=config),_harvest_associations=harvest)
    assert ns['associate'](fake,'question',k=7,steps=5)==['result']

@pytest.mark.parametrize('receipt',[False,True])
@pytest.mark.parametrize('failure',['vectors','activations'])
def test_incomplete_write_retains_previous_generation(tmp_path,monkeypatch,receipt,failure):
    from .test_save_receipt_423 import NS
    me=FakeSelf(tmp_path)
    assert save(me,with_receipt=True)['accepted']
    # Capture the actual last-good artifacts and ring before injecting failure.
    ring=tmp_path/'generations'
    before={str(p.relative_to(ring)):p.read_bytes() for p in ring.rglob('*') if p.is_file()}
    assert before
    manifest=(tmp_path/'main.msgpack.manifest.json').read_bytes()
    me.graph.payload=b'new graph';me.vector_db.payload=b'new vectors'
    def forbidden(*a,**kw):raise AssertionError('incomplete set must not publish manifest/generation')
    monkeypatch.setitem(NS,'write_manifest',forbidden)
    monkeypatch.setitem(NS,'rotate_generations',forbidden)
    def sidecar_failure(path,captured,**kwargs):
        assert not me.graph._step_lock._is_owned()
        # Realistic torn sidecar on its existing legacy in-place writer path.
        from pathlib import Path
        Path(str(path)+'.activations.json').write_text('{incomplete')
        raise OSError('sidecar write failed')
    if failure=='vectors':me.vector_db.fail=True
    else:me._activation_persistence.write_state=sidecar_failure
    if not receipt and failure=='activations':
        with pytest.raises(OSError,match='sidecar write failed'):save(me)
    else:
        result=save(me,with_receipt=receipt)
        if receipt:
            assert not result['accepted']
            assert result['components'][failure]['status']=='failed'
            assert result['components']['generation']['status']=='not_attempted'
            assert result['components']['manifest']['status']=='not_attempted'
    assert me._checkpoint_path.read_bytes()==b'new graph'
    if failure=='activations':assert me._vector_db_path.read_bytes()==b'new vectors'
    assert (tmp_path/'main.msgpack.manifest.json').read_bytes()==manifest
    after={str(p.relative_to(ring)):p.read_bytes() for p in ring.rglob('*') if p.is_file()}
    assert after==before

@pytest.mark.parametrize('quarantine',[False,True])
def test_payloads_released_after_component_writes(tmp_path,quarantine):
    me=FakeSelf(tmp_path,gate=FakeGate(permit=not quarantine))
    original_capture=me._capture_checkpoint_state
    holder=[]
    def capture():
        captured=original_capture();holder.append(captured);return captured
    me._capture_checkpoint_state=capture
    vector_write=me.vector_db.write_state
    def vectors(path,captured):
        assert 'graph' not in holder[0]
        return vector_write(path,captured)
    me.vector_db.write_state=vectors
    activation_write=me._activation_persistence.write_state
    def activation(path,captured):
        assert 'graph' not in holder[0] and 'vectors' not in holder[0]
        return activation_write(path,captured,**kwargs)
    me._activation_persistence.write_state=activation
    save(me,with_receipt=True)
    assert set(holder[0])=={'counts'}


def test_activation_receipt_accepts_same_clock_value(tmp_path):
    me=FakeSelf(tmp_path)
    ap=me._activation_persistence
    capture=ap.capture_state
    def same_time(graph):
        data=capture(graph);data['saved_at']=12.0;return data
    ap.capture_state=same_time
    ap._last_save_time=12.0
    assert save(me,with_receipt=True)['accepted']


def test_activation_receipt_without_token_fails_closed(tmp_path):
    me=FakeSelf(tmp_path)
    ap=me._activation_persistence;write=ap.write_state
    def missing_token(path,captured,**kw):
        report=write(path,captured,**kw);report.pop('saved_at');return report
    ap.write_state=missing_token
    result=save(me,with_receipt=True)
    assert not result['accepted']
    assert result['components']['activations']['status']=='failed'
    assert result['components']['generation']['status']=='not_attempted'
