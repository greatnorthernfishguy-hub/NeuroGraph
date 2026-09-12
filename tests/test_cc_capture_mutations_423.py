"""Real CC application functions extracted via AST; no NG constructors/models."""
import ast
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE = Path(__file__).parents[1] / 'cc_ng_organism.py'

def functions(*names):
    tree = ast.parse(SOURCE.read_text())
    wanted = {'_cc_mutation_lock', *names}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    ns = dict(logger=logging.getLogger('test'), time=time, Optional=object,
              _CC_CONV_THRESHOLD_BOOST=2, _CC_CONV_NOVELTY_DAMPENING=.5,
              _CC_CONV_PROBATION_PERIOD=4, _CC_CONV_PROBATION_REQUIRE_SPIKE=False,
              _CC_CONV_SYNAPSE_DELAY_MAX=3, _CC_KISS_GATE_ENABLED=False,
              _cc_embed_to_poincare_dir=lambda x:x, _cc_has_ever_fired=lambda n:False,
              cc_anticipate=lambda *a:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns

class Graph:
    def __init__(self):
        self._step_lock=threading.RLock()
        self.nodes={}
        self.config={}
    def create_node(self,node_id,metadata):
        assert self._step_lock._is_owned()
        n=SimpleNamespace(metadata=metadata)
        self.nodes[node_id]=n
        return n
    def create_synapse(self,*a,**k):
        assert self._step_lock._is_owned()
    def create_hyperedge(self,*a,**k):
        assert self._step_lock._is_owned()

class VDB:
    def __init__(self,g,fail=False):self.g,self.fail=g,fail
    def insert(self,**kw):
        assert self.g._step_lock._is_owned()
        if self.fail:raise OSError('insert failed')

@pytest.fixture
def packer(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'neuro_foundation',SimpleNamespace(pack_poincare_dir=lambda x:b'packed'))

def test_deposit_and_vector_share_lock(packer):
    ns=functions('_cc_deposit_memory_node')
    g=Graph()
    ns['_cc_deposit_memory_node'](g,VDB(g),'a',[1],'text',{})
    assert g.nodes['a'].metadata['probation_remaining']==4
    assert not g._step_lock._is_owned()

def test_partial_insert_failure_propagates(packer):
    ns=functions('_cc_deposit_memory_node')
    g=Graph()
    with pytest.raises(OSError,match='insert failed'):
        ns['_cc_deposit_memory_node'](g,VDB(g,True),'a',[1],'text',{})
    assert 'a' in g.nodes # retained partial application; no fabricated rollback
    assert not g._step_lock._is_owned()

@pytest.mark.parametrize('fail_insert,extract_failed',[(False,False),(True,False),(False,True)])
def test_dual_pass_outcome_and_embedding_outside_lock(packer,monkeypatch,fail_insert,extract_failed):
    import sys
    ns=functions('_cc_deposit_memory_node','_CCConversationalDualPassEco','run_conversational_dual_pass','_cc_bind_conversational_topology')
    ns['_cc_concept_passes_floor']=lambda c:True
    g=Graph()
    class Embed:
        def dual_record_outcome(self,ecosystem,embedding,target_id,metadata,**kw):
            assert not g._step_lock._is_owned()
            ecosystem.record_outcome(embedding,target_id,True,metadata=metadata)
            assert not g._step_lock._is_owned() # tree extraction/model phase
            return {'tree_ids':[],'extraction_failed':extract_failed}
    monkeypatch.setitem(sys.modules,'ng_embed',SimpleNamespace(NGEmbed=SimpleNamespace(get_instance=lambda:Embed())))
    assert ns['run_conversational_dual_pass'](g,VDB(g,fail_insert),'text',[1],{}) is (not fail_insert and not extract_failed)

def test_probation_and_kiss_keep_existing_clock_semantics():
    ns=functions('_cc_kiss_reinforce_node','cc_update_probation')
    g=Graph()
    class Metadata(dict):
        def __setitem__(self,k,v):
            assert g._step_lock._is_owned()
            super().__setitem__(k,v)
    g.nodes['a']=SimpleNamespace(metadata=Metadata(probation_remaining=2,probation_total=4),threshold=3,intrinsic_excitability=.5)
    assert ns['_cc_kiss_reinforce_node'](g,'a')
    assert g.nodes['a'].metadata['probation_remaining']==1
    assert ns['cc_update_probation'](g)==['a']
    assert g.nodes['a'].metadata['graduated'] is True
    assert not hasattr(g,'timestep') # neither path requires/advances graph clock

def test_geometry_embedding_outside_lock_and_stale_result_discarded(packer,monkeypatch):
    import sys
    ns=functions('cc_stamp_missing_geometry')
    g=Graph()
    n=SimpleNamespace(metadata={'_forest_content':'original'})
    g.nodes['a']=n
    def embed(text):
        assert not g._step_lock._is_owned()
        with g._step_lock:n.metadata['_forest_content']='changed'
        return [1]
    monkeypatch.setitem(sys.modules,'ng_embed',SimpleNamespace(embed=embed))
    assert ns['cc_stamp_missing_geometry'](g)==0
    assert 'poincare_dir' not in n.metadata

def test_deposit_waits_for_capture_lock(packer):
    ns=functions('_cc_deposit_memory_node')
    g=Graph(); started=threading.Event(); done=threading.Event()
    def apply():
        started.set()
        ns['_cc_deposit_memory_node'](g,VDB(g),'a',[1],'text',{})
        done.set()
    with g._step_lock:
        t=threading.Thread(target=apply); t.start()
        assert started.wait(1)
        assert not done.wait(.03)
        assert not g.nodes
    t.join(2)
    assert done.is_set()

@pytest.mark.parametrize('tree_failure',[False,True])
def test_canonical_ngembed_does_not_swallow_deposit_exception(tree_failure):
    """Execute actual vendored dual-pass method, without constructing its embedder."""
    tree=ast.parse((SOURCE.parent/'ng_embed.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NGEmbed')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='dual_record_outcome')
    ns={}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'ng_embed.py','exec'),ns)
    class Eco:
        def record_outcome_broadcast(self,*a,metadata=None,**kw):
            if not tree_failure or (metadata or {}).get('_tree_concept'):
                raise OSError('vector insertion failed')
            return {'deposited':True}
    fake=SimpleNamespace(_extract_concepts=lambda text:['concept'],embed_batch=lambda concepts:[[1]])
    with pytest.raises(OSError,match='vector insertion failed'):
        ns['dual_record_outcome'](fake,Eco(),'text',[1],'target',True,metadata={})
