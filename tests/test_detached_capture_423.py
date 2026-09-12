"""#423 canonical capture/write contracts using extracted code and disposable fakes."""
# ---- Changelog ----
# [2026-09-11] Codex — replace constructor-based native draft with isolated tests.
# What: exercise real persistence methods without importing Graph or embedding modules.
# Why: offline repair must not instantiate NG/models or touch live state.
# How: AST extraction; detached mutable payloads, independent msgpack decoding, barriers.
# -------------------
import ast
import copy
import enum
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import msgpack
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
class Mode(enum.Enum):
    FULL = 'full'
    INCREMENTAL = 'incremental'
    FORK = 'fork'


def methods(file, cls, names):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    body = [f for f in node.body if isinstance(f, ast.FunctionDef) and f.name in names]
    ns = dict(copy=copy, CheckpointMode=Mode, msgpack=msgpack, Dict=Dict,
              Any=Any, Optional=Optional, time=time, json=json, logger=logging.getLogger('test'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), file, 'exec'), ns)
    return {f.name: ns[f.name] for f in body}


class GraphFake:
    def __init__(self):
        self._step_lock = threading.RLock()
        self.payload = {'nodes': {'n': {'voltage': .7, 'metadata': {'nested': [1]},
                                      'array': np.array([2.])}},
                        'config': {'nested': [3]}, 'synapses': msgpack.packb({'s': {'weight': .5}})}
        self._dirty_nodes, self._dirty_synapses, self._dirty_hyperedges = {'n'}, {'s'}, {'h'}
    def _serialize_full(self):
        assert self._step_lock._is_owned()
        return dict(self.payload)
    def _serialize_incremental(self):
        return {'incremental': True, 'nodes': self.payload['nodes'], 'synapses': {'s': {'weight': .5}}}

for name, fn in methods('neuro_foundation.py', 'Graph', {'capture_checkpoint','write_checkpoint','checkpoint'}).items():
    setattr(GraphFake, name, fn)

class VectorFake:
    def __init__(self):
        self.embeddings = {'n': np.array([1.,2.], dtype=np.float32)}
        self.content = {'n': 'raw experience'}
        self.metadata = {'n': {'nested': [1]}}
for name, fn in methods('universal_ingestor.py', 'SimpleVectorDB', {'capture_state','write_state','save'}).items():
    setattr(VectorFake, name, fn)

class ActivationFake:
    def __init__(self):
        self._cfg = SimpleNamespace(max_entries=100)
        self._last_save_time = None
    def _sidecar_path_for(self, path):
        return str(path)+'.activations.json'
for name, fn in methods('activation_persistence.py', 'ActivationPersistence', {'capture','capture_state','write_state','save'}).items():
    setattr(ActivationFake, name, fn)

@pytest.mark.parametrize('mode', [Mode.FULL, Mode.FORK, Mode.INCREMENTAL])
def test_graph_detachment(mode):
    g=GraphFake(); cap=g.capture_checkpoint(mode, detach=True)
    g.payload['nodes']['n']['metadata']['nested'].append(9)
    g.payload['nodes']['n']['array'][0]=8
    assert cap['nodes']['n']['metadata']['nested']==[1]
    assert cap['nodes']['n']['array'][0]==2
    assert not g._step_lock._is_owned()
    if mode==Mode.INCREMENTAL:
        assert not g._dirty_nodes
    if mode==Mode.FORK:
        assert cap['_fork'] is True

@pytest.mark.parametrize('packed', [bytes, bytearray])
def test_native_synapse_bytes_are_preserved_and_detached(tmp_path, packed):
    g=GraphFake(); del g.payload['nodes']['n']['array']
    g.payload['synapses']=packed(g.payload['synapses'])
    cap=g.capture_checkpoint(detach=True)
    if packed is bytearray:
        g.payload['synapses'][:]=b'bad'
    path=str(tmp_path/'state.msgpack');g.write_checkpoint(path,cap)
    decoded=msgpack.unpackb(Path(path).read_bytes(),raw=False)
    assert decoded['synapses']=={'s':{'weight':.5}}


def test_checkpoint_rejects_extension_before_capture(tmp_path):
    g=GraphFake(); g._serialize_full=lambda:pytest.fail('capture ran before extension validation')
    with pytest.raises(ValueError):g.checkpoint(str(tmp_path/'bad.json'))


def test_capture_failure_releases_lock():
    g=GraphFake()
    def fail():raise RuntimeError('capture failure')
    g._serialize_full=fail
    with pytest.raises(RuntimeError):g.capture_checkpoint(detach=True)
    assert not g._step_lock._is_owned()


def test_writer_runs_outside_lock_and_does_not_read_live_graph(tmp_path):
    g=GraphFake();del g.payload['nodes']['n']['array']
    original=g.write_checkpoint
    def write(path,cap,mode):
        assert not g._step_lock._is_owned()
        g.payload['nodes']['n']['voltage']=.1
        return original(path,cap,mode)
    g.write_checkpoint=write
    # Explicit detachment is the coordinated producer's contract.
    cap=g.capture_checkpoint(detach=True);write(str(tmp_path/'a.msgpack'),cap,Mode.FULL)
    assert msgpack.unpackb((tmp_path/'a.msgpack').read_bytes(),raw=False)['nodes']['n']['voltage']==.7


def test_capture_waits_for_completed_mutation():
    g=GraphFake();finished=threading.Event();entered=threading.Event()
    def capture():entered.set();g.capture_checkpoint(detach=True);finished.set()
    with g._step_lock:
        t=threading.Thread(target=capture);t.start();assert entered.wait(1)
        assert not finished.wait(.05)
    t.join(1);assert finished.is_set()

@pytest.mark.parametrize('suffix',['msgpack','json'])
def test_vector_capture_detaches_and_roundtrips(tmp_path,suffix):
    v=VectorFake();cap=v.capture_state(detach=True)
    v.embeddings['n'][0]=9;v.metadata['n']['nested'].append(2);v.content['n']='changed'
    path=str(tmp_path/('v.'+suffix));assert v.write_state(path,cap)==1
    assert cap['entries']['n']['content']=='raw experience'
    assert cap['entries']['n']['metadata']=={'nested':[1]}
    assert np.frombuffer(cap['entries']['n']['embedding'],dtype=np.float32)[0]==1
    decoded=msgpack.unpackb(Path(path).read_bytes(),raw=False) if suffix=='msgpack' else json.loads(Path(path).read_text())
    assert decoded['count']==1


def test_vector_legacy_save_delegates(tmp_path):
    v=VectorFake();assert v.save(str(tmp_path/'v.msgpack'))==1


def test_activation_capture_uses_captured_values(tmp_path):
    ap=ActivationFake();node=SimpleNamespace(voltage=.5,resting_potential=0,last_spike_time=2,intrinsic_excitability=1)
    g=SimpleNamespace(nodes={'n':node},timestep=7)
    cap=ap.capture_state(g);node.voltage=.9;g.timestep=9
    path=ap.write_state(str(tmp_path/'g.msgpack'),cap)
    data=json.loads(Path(path).read_text());assert data['timestep']==7
    assert data['entries']['n']['voltage']==.5
    assert ap._last_save_time==data['saved_at']


def test_activation_failure_propagates_and_does_not_claim_success(tmp_path):
    ap=ActivationFake()
    with pytest.raises(OSError):ap.write_state(str(tmp_path/'missing'/'g.msgpack'),{'entries':{},'saved_at':1})
    assert ap._last_save_time is None

@pytest.mark.parametrize('write_mode',[False,True])
def test_actual_propagation_cannot_prime_during_capture(write_mode):
    from typing import List, Set, Tuple
    tree=ast.parse((ROOT/'neuro_foundation.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Graph')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prime_and_propagate')
    ns=dict(List=List,Set=Set,Tuple=Tuple,Dict=Dict,PropagationResult=lambda **kw:SimpleNamespace(fired_entries=[],**kw))
    # Annotations must stay inert without loading the SNN type tree.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'prime_under_test','exec'),ns)
    node=SimpleNamespace(voltage=0.,resting_potential=0.,refractory_remaining=0,intrinsic_excitability=1)
    graph=SimpleNamespace(_step_lock=threading.RLock(),nodes={'n':node},hyperedges={},active_predictions={},_active_predictions={},timestep=0,config={'decay_rate':.9,'he_experience_threshold':1})
    entered=threading.Event();done=threading.Event();errors=[]
    def prime():
        entered.set()
        try:ns['prime_and_propagate'](graph,['n'],[1.],steps=0,write_mode=write_mode)
        except Exception as exc:errors.append(exc)
        finally:done.set()
    with graph._step_lock:
        worker=threading.Thread(target=prime);worker.start();assert entered.wait(1)
        assert not done.wait(.05)
        assert node.voltage==0
    worker.join(1);assert done.is_set();assert not errors
    assert node.voltage==(1. if write_mode else 0.)


def test_disappearing_vector_refuses_capture():
    class Disappearing(dict):
        def keys(self):
            keys=list(super().keys());self.clear();return keys
    v=VectorFake();v.embeddings=Disappearing(v.embeddings)
    with pytest.raises(KeyError):v.capture_state(detach=True)


def test_harvest_honors_query_options_without_mutating_config():
    source=ROOT/'openclaw_hook.py'
    cls=next(c for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef) and c.name=='NeuroGraphMemory')
    fn=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='_harvest_associations')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns={'logger':logging.getLogger('test')};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
    config={'max_surfaced':10,'propagation_steps':3}
    def embed(text):
        assert config=={'max_surfaced':10,'propagation_steps':3}
        return [1.]
    def propagate(**kw):
        assert kw['steps']==5
        return SimpleNamespace(fired_entries=[SimpleNamespace(node_id=str(i),firing_step=i,voltage_at_fire=1,was_predicted=False) for i in range(9)])
    fake=SimpleNamespace(graph=SimpleNamespace(config=config,nodes={str(i):None for i in range(9)},prime_and_propagate=propagate),ingestor=SimpleNamespace(embedder=SimpleNamespace(embed_text=embed)),vector_db=SimpleNamespace(search=lambda *a,**k:[(str(i),1.) for i in range(9)],get=lambda n:{'content':n,'metadata':{}}))
    result=ns['_harvest_associations'](fake,'query',max_surfaced_override=7,propagation_steps_override=5)
    assert len(result)==7


def test_activation_explicit_receipt_does_not_require_clock_advance(tmp_path):
    ap=ActivationFake(); cap={'entries':{},'saved_at':12.0,'timestep':7}
    ap._last_save_time=12.0
    receipt=ap.write_state(str(tmp_path/'g.msgpack'),cap,with_receipt=True)
    assert receipt=={'path':str(tmp_path/'g.msgpack.activations.json'),'saved_at':12.0}
    assert json.loads(Path(receipt['path']).read_text())['saved_at']==receipt['saved_at']
