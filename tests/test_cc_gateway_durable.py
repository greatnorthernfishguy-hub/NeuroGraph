# ---- Changelog ----
# [2026-09-11] Codex — #423 durable raw gateway transport failure boundaries.
# What: fakes + real tract bytes, no NG/embedding initialization.
# Why: learning attempts and checkpoint acceptance are separate crash boundaries.
# How: extract production functions, SQLite inspection, concurrency and fault injection.
# -------------------
import ast
import glob
import hashlib
import logging
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
import uuid

import ng_tract
import pytest


@pytest.fixture
def rig(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'cc_ng_organism.py'
    tree = ast.parse(source.read_text())
    ns = dict(os=os, glob=glob, uuid=uuid, logger=logging.getLogger('test'),
              _CC_CALLOSUM_LEG1_ENABLED=True,
              _CC_GATEWAY_CONDUIT_GLOB='*_cc_gateway.*.tract')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('drain_gateway_conduit', '_apply_gateway_experience')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
    monkeypatch.setitem(sys.modules, 'ng_embed', SimpleNamespace(embed=lambda text: text))
    monkeypatch.setitem(sys.modules, 'cc_refeed', SimpleNamespace(should_pause_for_load=lambda ceiling: False))
    monkeypatch.setenv('MACHINE_ID', 'vps')
    seen = []
    ns['run_conversational_dual_pass'] = lambda g, v, text, emb, state: seen.append(text) or True
    graph = SimpleNamespace(_concurrent_lock=threading.RLock())
    conduit = tmp_path / 'conduit'; conduit.mkdir()
    journal = tmp_path / 'local' / 'delivery.sqlite3'
    state = {}
    receipt = dict(outcome='primary', accepted=True)
    def add(name='laptop_cc_gateway.1.tract', texts=('same words',)):
        path = conduit / name
        for text in texts:
            ng_tract.deposit_experience(raw=text.encode(), source='cc_gateway', tract_paths=[str(path)])
        return path
    def drain(**kw):
        args = dict(conduit_dir=str(conduit), journal_path=str(journal),
                    save_callback=lambda: receipt)
        args.update(kw)
        return ns['drain_gateway_conduit'](graph, None, state, **args)
    return SimpleNamespace(**locals())


def test_receipt_loss_retries_without_learning_and_retains_raw(rig):
    path = rig.add(texts=('same', 'same')); raw = path.read_bytes()
    rig.receipt.update(outcome='quarantine', accepted=False)
    first = rig.drain()
    assert first['retained'] == 1 and not first['accepted']
    assert path.read_bytes() == raw and rig.seen == ['same', 'same']
    rig.receipt.update(outcome='primary', accepted=True)
    assert rig.drain()['accepted'] and not path.exists()
    response = rig.drain()  # socket response was lost after deletion
    assert response['accepted_files'] == [dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest())]
    assert rig.seen == ['same', 'same']
    with sqlite3.connect(rig.journal) as db:
        assert db.execute('SELECT raw FROM files').fetchone()[0] == raw
        assert len(db.execute('SELECT start,end FROM records').fetchall()) == 2


@pytest.mark.parametrize('receipt', [None, 'primary.msgpack', {'accepted': True, 'outcome': 'quarantine'},
                                     {'accepted': False, 'outcome': 'failed'}])
def test_nonacceptance_never_removes_input(rig, receipt):
    path = rig.add()
    assert not rig.drain(save_callback=lambda: receipt)['accepted']
    assert path.exists()
    assert rig.seen == ['same words']


def test_save_exception_then_retry_does_not_relearn(rig):
    path = rig.add()
    def fail(): raise OSError('disk full')
    assert not rig.drain(save_callback=fail)['ok']
    assert path.exists()
    assert rig.drain()['accepted']
    assert len(rig.seen) == 1


def test_partial_mutation_is_uncertain_even_same_process(rig):
    path = rig.add(texts=('first', 'second'))
    def partial(*args):
        rig.seen.append(args[2]); raise RuntimeError('after mutation')
    rig.ns['run_conversational_dual_pass'] = partial
    assert rig.drain()['uncertain'] == 1
    assert rig.drain()['uncertain'] == 1
    assert rig.seen == ['first'] and path.exists()


def test_crash_between_attempt_and_application_requires_reconciliation(rig):
    path = rig.add()
    def interrupt(*args): raise KeyboardInterrupt()
    rig.ns['run_conversational_dual_pass'] = interrupt
    with pytest.raises(KeyboardInterrupt): rig.drain()
    assert rig.drain()['uncertain'] == 1
    assert path.exists() and not rig.seen


def test_restart_or_different_graph_never_trusts_prior_applied(rig):
    path = rig.add()
    rig.receipt.update(accepted=False, outcome='quarantine')
    rig.drain()
    rig.state.clear()  # new graph incarnation/process binding
    rig.receipt.update(accepted=True, outcome='primary')
    assert rig.drain()['uncertain'] == 1
    assert rig.seen == ['same words'] and path.exists()


def test_restart_accepted_receipt_and_new_file_do_not_repeat_learning(rig):
    first = rig.add(texts=('first',)); first_raw = first.read_bytes()
    rig.drain(); rig.state.clear()
    second = rig.add(name='laptop_cc_gateway.2.tract', texts=('second',))
    second_raw = second.read_bytes()
    result = rig.drain()
    assert result['accepted'] and result['all_done'] and result['uncertain'] == 0
    assert rig.seen == ['first', 'second']
    assert result['accepted_files'] == [
        dict(name=first.name, sha256=hashlib.sha256(first_raw).hexdigest()),
        dict(name=second.name, sha256=hashlib.sha256(second_raw).hexdigest())]


def test_restart_finishes_terminal_receipt_cleanup_without_relearning(rig, monkeypatch):
    path = rig.add()
    unlink = os.unlink
    with monkeypatch.context() as scoped:
        def fail(target, *args, **kwargs):
            if str(target) == str(path): raise OSError('cleanup interrupted')
            return unlink(target, *args, **kwargs)
        scoped.setattr(os, 'unlink', fail)
        assert not rig.drain()['ok']
    rig.state.clear()
    result = rig.drain()
    assert result['accepted'] and result['uncertain'] == 0
    assert not path.exists() and rig.seen == ['same words']


def test_same_filename_two_accepted_digests_remain_exact_receipt_identities(rig):
    path = rig.add(texts=('first',)); first_raw = path.read_bytes()
    rig.drain()
    path = rig.add(texts=('second',)); second_raw = path.read_bytes()
    rig.drain(); rig.state.clear()
    result = rig.drain()
    assert result['accepted'] and result['all_done']
    assert {r['sha256'] for r in result['accepted_files']} == {
        hashlib.sha256(first_raw).hexdigest(), hashlib.sha256(second_raw).hexdigest()}
    assert {r['name'] for r in result['accepted_files']} == {path.name}
    assert rig.seen == ['first', 'second']


def test_corrupt_suffix_is_parsed_before_any_learning(rig):
    path = rig.add(); path.write_bytes(path.read_bytes() + b'broken tail')
    assert rig.drain()['retained'] == 1
    assert not rig.seen and path.exists()


def test_changed_bytes_same_filename_is_distinct_experience(rig):
    path = rig.add(texts=('first',)); rig.drain()
    path = rig.add(texts=('second',)); raw = path.read_bytes()
    result = rig.drain()
    assert rig.seen == ['first', 'second']
    assert dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest()) in result['accepted_files']


def test_source_change_during_save_is_not_deleted(rig):
    path = rig.add()
    def change():
        path.write_bytes(b'replacement'); return rig.receipt
    result = rig.drain(save_callback=change)
    assert path.read_bytes() == b'replacement'
    assert result['accepted_files'] == []


def test_concurrent_deliveries_apply_once(rig):
    rig.add(texts=('first', 'second'))
    barrier = threading.Barrier(3)
    results = []
    def run():
        barrier.wait(); results.append(rig.drain())
    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join(timeout=10); assert not thread.is_alive()
    assert rig.seen == ['first', 'second']
    assert all(r['accepted'] for r in results)


def test_gate_missing_contract_and_self_exclusion_are_inert(rig):
    path = rig.add()
    rig.ns['_CC_CALLOSUM_LEG1_ENABLED'] = False
    assert rig.drain()['disabled'] and not rig.journal.exists()
    rig.ns['_CC_CALLOSUM_LEG1_ENABLED'] = True
    assert not rig.drain(save_callback=None)['ok'] and not rig.journal.exists()
    assert rig.drain(exclude_prefix='laptop_')['accepted_files'] == []
    assert path.exists() and not rig.seen


def test_load_backpressure_retains_full_file_then_continues(rig, monkeypatch):
    path = rig.add(texts=('first', 'second')); raw = path.read_bytes()
    monkeypatch.setitem(sys.modules, 'cc_refeed', SimpleNamespace(should_pause_for_load=lambda c: True))
    assert rig.drain()['retained'] == 1
    assert path.read_bytes() == raw and rig.seen == ['first']
    assert rig.drain()['accepted']
    assert rig.seen == ['first', 'second']


def test_crash_after_save_before_journal_acceptance_is_uncertain_on_restart(rig):
    path = rig.add()
    def crash_after_save():
        # Simulates save reaching disk followed by process death before receipt.
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): rig.drain(save_callback=crash_after_save)
    rig.state.clear()
    assert rig.drain()['uncertain'] == 1
    assert path.exists() and rig.seen == ['same words']


def test_ack_cleanup_failure_retries_without_learning(rig, monkeypatch):
    path = rig.add()
    unlink = os.unlink
    with monkeypatch.context() as scoped:
        def fail(target, *args, **kwargs):
            if str(target) == str(path): raise OSError('unlink interrupted')
            return unlink(target, *args, **kwargs)
        scoped.setattr(os, 'unlink', fail)
        assert not rig.drain()['ok']
    assert path.exists()
    assert rig.drain()['accepted']
    assert rig.seen == ['same words']


def test_journal_unavailable_prevents_learning(rig):
    path = rig.add()
    rig.journal.parent.mkdir()
    rig.journal.write_bytes(b'not a sqlite database')
    assert not rig.drain()['ok']
    assert path.exists() and rig.seen == []


def test_host_uses_opt_in_receipt_and_new_event_alias(rig, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'cc_ng_host.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_handle_drain_conduit')
    calls = []
    def save(**kwargs):
        calls.append(kwargs); return rig.receipt
    ng = SimpleNamespace(graph=rig.graph, vector_db=None, save=save)
    ns = dict(os=os, _STATE=SimpleNamespace(cc_ng=ng, conv_state=rig.state),
              CC_NG_WORKSPACE=str(rig.tmp_path / 'host'), logger=logging.getLogger('test'))
    monkeypatch.setitem(sys.modules, 'cc_ng_organism', SimpleNamespace(drain_gateway_conduit=rig.ns['drain_gateway_conduit']))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
    rig.add()
    response = ns['_handle_drain_conduit'](dict(conduit_dir=str(rig.conduit), batch_size=1, idle_steps=0))
    assert response['accepted'] and calls == [{'with_receipt': True}]
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert any(any(isinstance(k, ast.Constant) and k.value == 'drain_conduit_durable' and
                       isinstance(v, ast.Name) and v.id == '_handle_drain_conduit'
                       for k, v in zip(d.keys, d.values)) for d in handlers)


def test_one_save_for_multiple_files_and_one_graph_lock_per_record(rig):
    lock_events = []
    class Lock:
        def __enter__(self): lock_events.append('enter')
        def __exit__(self, *args): lock_events.append('exit')
    rig.graph._concurrent_lock = Lock()
    rig.add(texts=('first', 'second'))
    rig.add(name='laptop_cc_gateway.2.tract', texts=('third',))
    saves = []
    def save():
        saves.append(True)
        assert len(rig.seen) == 3
        assert len(list(rig.conduit.glob('*.tract'))) == 2
        return rig.receipt
    assert rig.drain(save_callback=save, batch_size=25, idle_steps=250)['accepted']
    assert len(saves) == 1 and lock_events == ['enter', 'exit'] * 4


def test_valid_input_can_acknowledge_alongside_invalid_retained_input(rig):
    rig.add()
    bad = rig.conduit / 'laptop_cc_gateway.bad.tract'; bad.write_bytes(b'bad')
    result = rig.drain()
    assert result['accepted'] and not result['all_done'] and not result['ok']
    assert len(result['accepted_files']) == 1 and bad.exists()


def test_failed_directory_durability_prevents_learning(rig, monkeypatch):
    path = rig.add()
    def fail(*args): raise OSError('fsync failed')
    monkeypatch.setattr(os, 'fsync', fail)
    assert not rig.drain()['ok']
    assert path.exists() and not rig.seen


def test_journal_cannot_live_in_git_conduit(rig):
    path = rig.add()
    result = rig.drain(journal_path=str(rig.conduit / 'journal.sqlite3'))
    assert not result['ok'] and not rig.seen and path.exists()



def test_restart_adopts_unattempted_file_after_load_gate(rig, monkeypatch):
    first = rig.add(texts=('first',))
    second = rig.add(name='laptop_cc_gateway.2.tract', texts=('second',))
    monkeypatch.setitem(sys.modules, 'cc_refeed', SimpleNamespace(should_pause_for_load=lambda c: True))
    initial = rig.drain()
    assert initial['accepted'] and initial['retained'] == 1
    assert not first.exists() and second.exists() and rig.seen == ['first']
    rig.state.clear()
    result = rig.drain()
    assert result['accepted'] and result['all_done'] and not result['uncertain']
    assert rig.seen == ['first', 'second'] and not second.exists()
    assert len(result['accepted_files']) == 2


def test_restart_adopts_raw_retained_before_first_attempt(rig, monkeypatch):
    path = rig.add()
    class InterruptedReader:
        def __init__(self, raw):
            raise KeyboardInterrupt()
    with monkeypatch.context() as scoped:
        scoped.setattr(ng_tract, 'TractReader', InterruptedReader)
        with pytest.raises(KeyboardInterrupt):
            rig.drain()
    with sqlite3.connect(rig.journal) as db:
        assert db.execute('SELECT count(*) FROM files').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM records').fetchone()[0] == 0
    rig.state.clear()
    assert rig.drain()['accepted']
    assert rig.seen == ['same words'] and not path.exists()
