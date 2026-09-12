"""Detached canonical serializers preserve learned and temporal state without Graph startup."""
# ---- Changelog ----
# [2026-09-11] Codex — audit #423 borrowed-state detachment in every checkpoint mode.
# What: Mutate nested/shared live state after capture; compare against legacy deep copy.
# Why: Removing the expensive outer deepcopy must not leave snapshot aliases behind.
# How: Extract actual serializers into a fake host; no Graph imports or constructors.
# -------------------
import ast
import copy
import enum
import math
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import msgpack
import pytest

SOURCE = Path(__file__).parents[1] / 'neuro_foundation.py'
METHODS = {'capture_checkpoint', '_serialize_full', '_serialize_incremental',
           '_serialize_node', '_serialize_hyperedge', '_serialize_prediction',
           '_serialize_prediction_state', 'write_checkpoint'}


class Mode(enum.Enum):
    FULL = 'full'
    INCREMENTAL = 'incremental'
    FORK = 'fork'


class SerializerFake:
    pass


def install_serializers():
    cls = next(n for n in ast.parse(SOURCE.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'Graph')
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0)] + body,
                        type_ignores=[])
    ns = dict(copy=copy, math=math, msgpack=msgpack, CheckpointMode=Mode,
              HomeostaticRule=type('HomeostaticFake', (), {}))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    for name in METHODS:
        setattr(SerializerFake, name, ns[name])


install_serializers()


class SynapsesFake(dict):
    packed_type = bytes
    last_packed = None

    def to_checkpoint_msgpack(self):
        self.last_packed = self.packed_type(msgpack.packb(dict(self), use_bin_type=True))
        return self.last_packed

    def serialize_one(self, sid):
        # Even a backend returning a borrowed nested metadata dict must detach it.
        return dict(self[sid])


def fixture():
    shared = {'nested': [1, {'x': [2]}]}
    spikes = [1, 4]
    node = SimpleNamespace(node_id='n', voltage=.7, threshold=.8, resting_potential=0.,
        refractory_remaining=1, refractory_period=2, last_spike_time=4,
        spike_history=SimpleNamespace(to_list=lambda: list(spikes), capacity=10),
        firing_rate_ema=.2, intrinsic_excitability=.9, metadata=shared,
        is_inhibitory=False, Ca_i=.1, diffpc_layer=1, pred_weights={'m': .5},
        pred_error_ema=.1, manifold_type='hyperbolic', creation_time=1)
    he = SimpleNamespace(hyperedge_id='h', member_nodes={'n'}, member_weights={'n': .5},
        activation_threshold=.6, activation_mode=SimpleNamespace(name='WEIGHTED_THRESHOLD'),
        current_activation=.3, output_targets=['m'], output_weight=.4, metadata=shared,
        is_learnable=True, refractory_period=2, refractory_remaining=1, activation_count=3,
        pattern_completion_strength=.2, child_hyperedges={'child'}, level=1,
        recent_activation_ema=.2, is_archived=False,
        consolidation_state=SimpleNamespace(value='speculative'), creation_time=1)
    archived = copy.deepcopy(he)
    archived.hyperedge_id = 'a'
    archived.metadata = shared
    archived.is_archived = True
    pred = SimpleNamespace(prediction_id='p', source_node_id='n', target_node_id='m',
        strength=.4, confidence=.5, created_at=1, expires_at=5, chain_depth=1,
        via_hyperedge='h', pre_charge_applied=True)
    graph = SerializerFake()
    graph.__dict__.update(_step_lock=threading.RLock(), nodes={'n': node},
        hyperedges={'h': he}, _archived_hyperedges={'a': archived}, config={'nested': shared},
        timestep=4, synapses=SynapsesFake({'s': {'weight': .5, 'metadata': shared}}),
        active_predictions={'p': pred},
        _prediction_outcomes=[SimpleNamespace(prediction=pred, confirmed=True, resolved_at=4,
                                              actual_firing_nodes=['n'])],
        _synapse_confirmation_history={'s': deque([True, False])},
        _novel_sequence_log=deque([{'sequence': ['n', 'm'], 'metadata': shared}]),
        _reward_history=deque([{'scope': ['n'], 'metadata': shared}]),
        _active_predictions={'hp': SimpleNamespace(hyperedge_id='h', predicted_targets={'m'},
            prediction_strength=.5, prediction_timestamp=1, prediction_window=5, confirmed_targets={'n'})},
        _prediction_counter=7, _he_last_fired_step={'h': 4},
        _he_output_candidates={'h': {'m': 3}}, _delay_buffer={5: [('n', .2)]},
        _recent_spikes={'n': deque([1, 4])}, _steps_since_last_fire=0, _plasticity_rules=[],
        _dirty_nodes={'n'}, _dirty_synapses={'s'}, _dirty_hyperedges={'h'})
    for name in ('total_pruned total_sprouted total_he_discovered total_he_consolidated '
                 'total_predictions_made total_predictions_confirmed total_predictions_errors '
                 'total_novel_sequences total_rewards_injected total_predictions total_confirmed '
                 'total_surprised he_adapt_candidate_count he_adapt_candidate_ema '
                 'he_adapt_consolidated_count he_adapt_consolidated_age he_survival_ema '
                 'total_he_state_transitions total_he_substrate_culled').split():
        setattr(graph, '_' + name, 1)
    return graph, shared, spikes


def mutate_every_live_collection(g, shared, spikes):
    shared['nested'][1]['x'].append(99)
    shared['new'] = True
    spikes.append(9)
    n = g.nodes['n']
    n.pred_weights['m'] = 9
    n.voltage = 9
    for he in (g.hyperedges['h'], g._archived_hyperedges['a']):
        he.member_nodes.add('new')
        he.member_weights['n'] = 9
        he.output_targets.append('new')
        he.child_hyperedges.add('new')
    g.config['extra'] = []
    g._prediction_outcomes[0].actual_firing_nodes.append('new')
    g._prediction_outcomes[0].prediction.strength = 9
    g._synapse_confirmation_history['s'].append(True)
    g._novel_sequence_log[0]['sequence'].append('new')
    g._reward_history[0]['scope'].append('new')
    g._active_predictions['hp'].predicted_targets.add('new')
    g._active_predictions['hp'].confirmed_targets.add('new')
    g._he_last_fired_step['h'] = 9
    g._he_output_candidates['h']['m'] = 9
    g._delay_buffer[5].append(('new', 9))
    g._recent_spikes['n'].append(9)
    g.nodes.clear()
    g.hyperedges.clear()
    g._archived_hyperedges.clear()
    g.synapses.clear()


@pytest.mark.parametrize('mode', list(Mode))
def test_all_serialized_fields_match_legacy_then_survive_live_mutation(mode):
    g, shared, spikes = fixture()
    with g._step_lock:
        legacy = copy.deepcopy(g._serialize_incremental() if mode == Mode.INCREMENTAL else g._serialize_full())
    if mode == Mode.FORK:
        legacy['_fork'] = True
    cap = g.capture_checkpoint(mode)
    assert cap == legacy
    assert msgpack.packb(cap, use_bin_type=True) == msgpack.packb(legacy, use_bin_type=True)
    mutate_every_live_collection(g, shared, spikes)
    assert cap == legacy
    assert not g._step_lock._is_owned()


@pytest.mark.parametrize('mode', list(Mode))
def test_shared_subtrees_copied_once_across_nodes_edges_and_histories(mode):
    g, shared, _ = fixture()
    cap = g.capture_checkpoint(mode)
    detached = cap['nodes']['n']['metadata']
    assert detached is cap['hyperedges']['h']['metadata']
    assert detached is not shared
    if mode == Mode.INCREMENTAL:
        assert detached is cap['synapses']['s']['metadata']
    else:
        assert detached is cap['config']['nested']
        assert detached is cap['archived_hyperedges']['a']['metadata']
        assert detached is cap['novel_sequence_log'][0]['metadata']
        assert detached is cap['reward_history'][0]['metadata']


@pytest.mark.parametrize('mode', list(Mode))
def test_explicit_detach_false_retains_legacy_borrowing(mode):
    g, shared, _ = fixture()
    cap = g.capture_checkpoint(mode, detach=False)
    assert cap['nodes']['n']['metadata'] is shared
    assert cap['hyperedges']['h']['member_weights'] is g.hyperedges['h'].member_weights
    shared['nested'].append('later')
    assert cap['nodes']['n']['metadata']['nested'][-1] == 'later'


@pytest.mark.parametrize('mode', list(Mode))
def test_recursive_metadata_detaches_with_shared_memo(mode):
    g, shared, _ = fixture()
    # Cyclic metadata capture retains deepcopy behavior; msgpack writing a cycle
    # still fails as before. This is a capture test, not a new schema promise.
    g.synapses = SynapsesFake({'s': {'weight': .5}})
    shared['self'] = shared
    cap = g.capture_checkpoint(mode)
    detached = cap['nodes']['n']['metadata']
    assert detached['self'] is detached
    assert cap['hyperedges']['h']['metadata'] is detached
    shared['nested'].append(99)
    assert detached['nested'] == [1, {'x': [2]}]


def test_borrowed_copy_failure_preserves_incremental_dirty_flags_and_releases_lock():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError('cannot capture')
    g, shared, _ = fixture()
    shared['bad'] = Uncopyable()
    with pytest.raises(RuntimeError, match='cannot capture'):
        g.capture_checkpoint(Mode.INCREMENTAL)
    assert g._dirty_nodes == {'n'}
    assert g._dirty_synapses == {'s'}
    assert g._dirty_hyperedges == {'h'}
    assert not g._step_lock._is_owned()


@pytest.mark.parametrize('mode', [Mode.FULL, Mode.FORK])
@pytest.mark.parametrize('packed_type', [bytes, bytearray])
def test_fresh_native_synapse_payload_is_not_copied_and_splices(mode, packed_type, tmp_path):
    g, _, _ = fixture()
    g.synapses.packed_type = packed_type
    cap = g.capture_checkpoint(mode)
    assert cap['synapses'] is g.synapses.last_packed
    assert isinstance(cap['synapses'], packed_type)
    path = tmp_path / 'state.msgpack'
    g.write_checkpoint(str(path), cap, mode)
    decoded = msgpack.unpackb(path.read_bytes(), raw=False)
    expected = msgpack.unpackb(bytes(g.synapses.last_packed), raw=False)
    assert decoded['synapses'] == expected
