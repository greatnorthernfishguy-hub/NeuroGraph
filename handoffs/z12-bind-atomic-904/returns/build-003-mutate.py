#!/usr/bin/env python3
"""#904 mutation runner (build-001 table + the FOLD-UP additions). Applies ONE mutation at a time to a SCRATCH copy of cc_ng_organism.py
or cc-ng-daemon.py (never the worktree file), asserts every replacement matched EXACTLY once, runs the relevant test file(s) once, and
reports KILLED / SURVIVED. Load gate < 6 before every run."""
import os
import re
import subprocess
import sys

NG = '/home/josh/NeuroGraph-worktrees/z12-bind-atomic-904-20261001'
DOCS = '/home/josh/docs/.claude/worktrees/daemon-deposit-904-20261001'
SCR = '/tmp/z12-904-fold-scratch'
ORG = NG + '/cc_ng_organism.py'
DAEMON = DOCS + '/scripts/cc-ng-daemon.py'
ENV = dict(os.environ, HOME=SCR + '/home', PYTHONUSERBASE='/home/josh/.local', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
for k in ('CC_NG_BATCH_SIZE', 'CC_NG_IDLE_STEPS', 'CC_NG_DRAIN_HOLD_ON_FAILURE', 'Z12_904_ORG_UNDER_TEST', 'Z12_904_DAEMON_UNDER_TEST', 'NG_EMBED_REMOTE'):
    ENV.pop(k, None)

ROLL = '                    try:\n                        if hid in graph.hyperedges:\n                            graph.remove_hyperedge(hid)'
M = [
    # ---- build-001 table (the ruled mutations) ----
    ('N1 failure logged at debug again', 'org', [('(logger.warning if rolled_back else logger.error)(', 'logger.debug(')]),
    ('N2 returns True after a partial write', 'org',
     [('            else "ROLLBACK FAILED, a partial write REMAINS",\n        )\n        return False',
       '            else "ROLLBACK FAILED, a partial write REMAINS",\n        )\n        return True')]),
    ('N3 rollback skipped', 'org', [('rolled_back = eco.rollback()', 'rolled_back = True')]),
    ('N4 vdb entries not deleted on rollback', 'org', [('                        vdb.delete(node_id)', '                        pass')]),
    ('N5 state pointer not restored', 'org', [('                state["last_forest_id"] = prior_last', '                pass')]),
    ('N6 forest-absent returns silently again', 'org',
     [('            _site("forest_absent")\n            raise RuntimeError("forest node absent at bind")', '            return None')]),
    ('N7 the post-condition (truth rule) removed', 'org',
     [('            raise RuntimeError("bind wrote less than a turn with trees needs")', '            pass')]),
    ('N8 site counter dropped', 'org', [('                _failures[site] = _failures.get(site, 0) + 1', '                pass')]),
    ('N9 born-unbound warning dropped', 'org',
     [('        if (not tree_ids and not window_ids and status["sequence"] != "written"',
       '        if (False and not tree_ids and not window_ids and status["sequence"] != "written"')]),
    ('N10a swallow back: tree synapse', 'org',
     [('            _syn("tree_synapse", forest_id, tid, 0.2)\n            _syn("tree_synapse", tid, forest_id, 0.15)\n            status["tree_pairs"] += 1',
       '            try:\n                _syn("tree_synapse", forest_id, tid, 0.2)\n                _syn("tree_synapse", tid, forest_id, 0.15)\n                status["tree_pairs"] += 1\n            except Exception:\n                pass')]),
    ('N10b swallow back: window synapse', 'org',
     [('            _syn("window_synapse", forest_id, wid, 0.2)\n            _syn("window_synapse", wid, forest_id, 0.15)\n            status["window_pairs"] += 1',
       '            try:\n                _syn("window_synapse", forest_id, wid, 0.2)\n                _syn("window_synapse", wid, forest_id, 0.15)\n                status["window_pairs"] += 1\n            except Exception:\n                pass')]),
    ('N10c swallow back: hyperedge (debug)', 'org',
     [('            _site("hyperedge")\n            he = graph.create_hyperedge(\n                member_node_ids=set([forest_id] + tree_ids + window_ids),\n                metadata={"creation_mode": "conversational", "cc": True},\n            )\n            if journal is not None:\n                journal["hyperedges"].append(getattr(he, "hyperedge_id", None))\n            status["hyperedge"] = True',
       '            _site("hyperedge")\n            try:\n                he = graph.create_hyperedge(\n                    member_node_ids=set([forest_id] + tree_ids + window_ids),\n                    metadata={"creation_mode": "conversational", "cc": True},\n                )\n                if journal is not None:\n                    journal["hyperedges"].append(getattr(he, "hyperedge_id", None))\n                status["hyperedge"] = True\n            except Exception as exc:\n                logger.debug("hyperedge failed: %s", exc)')]),
    ('N10d swallow back: window chain', 'org',
     [('                _syn("window_chain", window_ids[i], window_ids[i + 1], 0.2, delay=d)\n                status["window_chain"] += 1',
       '                try:\n                    _syn("window_chain", window_ids[i], window_ids[i + 1], 0.2, delay=d)\n                    status["window_chain"] += 1\n                except Exception:\n                    pass')]),
    ('N10e swallow back: sequence link', 'org',
     [('            _syn("sequence_link", last_id, forest_id, 0.2, delay=d)\n            status["sequence"] = "written"',
       '            try:\n                _syn("sequence_link", last_id, forest_id, 0.2, delay=d)\n                status["sequence"] = "written"\n            except Exception:\n                pass')]),
    ('N11 True after a failed tree-bind (swallow tree site AND no post-condition)', 'org',
     [('            _syn("tree_synapse", forest_id, tid, 0.2)\n            _syn("tree_synapse", tid, forest_id, 0.15)\n            status["tree_pairs"] += 1',
       '            try:\n                _syn("tree_synapse", forest_id, tid, 0.2)\n                _syn("tree_synapse", tid, forest_id, 0.15)\n                status["tree_pairs"] += 1\n            except Exception:\n                pass'),
      ('            raise RuntimeError("bind wrote less than a turn with trees needs")', '            pass')]),
    ('N12 the drain truncates a failed entry with the flag on', 'org',
     [('        if hold_on_failure:\n            # Truncate only what precedes the first failed entry (or the whole\n            # walked span when nothing failed: safe_offset == consumed_offset).\n            consumed_offset = safe_offset',
       '        if hold_on_failure:\n            pass')]),
    ('N13a want swallow back: surface_wants', 'org',
     [('        _cc_report_want_synapse_failures("surface_wants", syn_attempted, syn_failed, syn_exc_types)', '        pass')]),
    ('N13b want swallow back: surface_wants_for_graph', 'org',
     [('        _cc_report_want_synapse_failures("surface_wants_for_graph", syn_attempted, syn_failed, syn_exc_types)', '        pass')]),
    # ---- FOLD-UP additions (NG) ----
    ('M20 rollback leaves the hyperedge (C3/F11)', 'org', [('                        graph.remove_hyperedge(hid)', '                        pass')]),
    ('F1 re-stamp restore skipped entirely (C4)', 'org', [('            for snap in reversed(self.restores):', '            for snap in []:')]),
    ('F2 stamp left in place: no snapshot taken (C4)', 'org', [('                snap = self._snapshot_existing(node_id, entry, meta)', '                snap = None')]),
    ('F3 restore metadata only (threshold/excitability left re-stamped)', 'org',
     [('                        for a, before in snap["attrs_before"].items():\n                            if same(getattr(node, a), snap["attrs_wrote"][a]):\n                                setattr(node, a, before)\n', '')]),
    ('F4 vdb entry of a pre-existing node not restored', 'org', [('                    if snap["vdb"] is not None:', '                    if False:')]),
    # ---- ROUND 2 (le-053): a NARROW, CONDITIONAL restore; the honest sentence; the printed count; the tie ----
    ('W1 restore UNCONDITIONALLY (a sweep between snapshot and rollback is erased)', 'org',
     [('                            if same(getattr(node, a), snap["attrs_wrote"][a]):', '                            if True:'),
      ('                            if same(cur, snap["meta_wrote"][k]):', '                            if True:')]),
    ('W2 WHOLE-DICT restore (clear + refill: erases keys the call never wrote)', 'org',
     [('                        for k in snap["keys"]:\n                            cur = node.metadata[k] if k in node.metadata else absent\n                            if same(cur, snap["meta_wrote"][k]):\n                                if snap["meta_before"][k] is absent:\n                                    node.metadata.pop(k, None)\n                                else:\n                                    node.metadata[k] = snap["meta_before"][k]\n',
       '                        node.metadata.clear()\n                        node.metadata.update({k: v for k, v in snap["meta_before"].items() if v is not absent})\n')]),
    ('W3 a key the call ADDED is not deleted', 'org', [('                                    node.metadata.pop(k, None)', '                                    pass')]),
    ('W4 threshold/excitability restored but the metadata keys are not', 'org',
     [('                        for k in snap["keys"]:\n                            cur = node.metadata[k]', '                        for k in []:\n                            cur = node.metadata[k]')]),
    ('S1 the "nothing of this turn remains" sentence is back beside kept entries (F1-1)', 'org',
     [('            ("rolled back, nothing of this turn remains" if not eco.vdb_unprovable', '            ("rolled back, nothing of this turn remains" if True')]),
    ('U1 the unrestorable count is not printed (F1-4)', 'org',
     [('n_forest, n_tree, n_window, n_syn, n_he, eco.vdb_unprovable, eco.unrestorable,', 'n_forest, n_tree, n_window, n_syn, n_he, eco.vdb_unprovable, 0,')]),
    ('T1 the deposit gains a NEW stamped key (F1-3 tie)', 'org',
     [('        node.metadata["novelty_dampening"] = _CC_CONV_NOVELTY_DAMPENING', '        node.metadata["novelty_dampening"] = _CC_CONV_NOVELTY_DAMPENING\n        node.metadata["extra_stamp"] = 1')]),
    ('T2 the restore list drops a key the deposit writes (F1-3 tie)', 'org',
     [('_STAMPED_META_KEYS = ("probation_remaining", "probation_total", "novelty_dampening", "poincare_dir")', '_STAMPED_META_KEYS = ("probation_remaining", "probation_total", "poincare_dir")')]),
    ('F6 exact embedding put-back dropped (a re-insert re-normalises)', 'org',
     [('                            store[snap["id"]] = emb', '                            pass')]),
    ('V1 delete SKIPPED when the vdb probe raised for a NEW node (checker C2)', 'org',
     [('                        vdb_fresh = True\n                    else:', '                        vdb_fresh = False\n                    else:')]),
    ('V2 a pre-existing node whose probe raised has its entry DELETED (never-delete violated)', 'org',
     [('                        self.vdb_unprovable += 1', '                        self.vdb_unprovable += 1\n                        vdb_fresh = True')]),
    # ---- daemon (build-001) ----
    ('D1 _deposit ignores the False', 'daemon', [("            if not ingested:\n                dual_pass_code = 'dual_pass_false'\n", '')]),
    ('D2 the daemon never passes the flag', 'daemon',
     [("_hold_kw = {'hold_on_failure': True} if DRAIN_HOLD_ON_FAILURE else {}", '_hold_kw = {}')]),
    ('D3 pointer persisted but NOT loaded', 'daemon', [("        self.conv_state['last_forest_id'] = _load_conv_pointer(self.ng.graph)\n", '')]),
    ('D4 pointer loaded WITHOUT the id-exists check', 'daemon', [('    if not present:\n', '    if False:\n')]),
    ('D5 the default is flipped on in code (absent = True)', 'daemon',
     [("    raw = env.get('CC_NG_DRAIN_HOLD_ON_FAILURE')\n    if raw is None:\n        return False",
       "    raw = env.get('CC_NG_DRAIN_HOLD_ON_FAILURE')\n    if raw is None:\n        return True")]),
    ('D6 pointer read AFTER the save (runs ahead of the saved graph)', 'daemon',
     [('        _write_conv_pointer(_pointer_before_save)', "        _write_conv_pointer(STATE.conv_state.get('last_forest_id'))")]),
    # ---- daemon (FOLD-UP: C8/F8 and item 6) ----
    ('D7 the flag is read at IMPORT again (C8/F8)', 'daemon',
     [('DRAIN_HOLD_ON_FAILURE = False   # set ONCE', 'DRAIN_HOLD_ON_FAILURE = _read_drain_hold_env()   # set ONCE')]),
    ('D8 main() never reads the flag (C8/F8)', 'daemon',
     [("    _init_drain_hold()   # #904 (C8/F8): once, after logging is configured\n", '')]),
    ('D9 the flag is read BEFORE logging is configured (order swapped)', 'daemon',
     [("def main():\n    _setup_logging()\n", "def main():\n    _init_drain_hold()\n    _setup_logging()\n"),
      ("    _init_drain_hold()   # #904 (C8/F8): once, after logging is configured\n", '')]),
    ('D10 a MISSING sidecar is a WARNING again (item 6)', 'daemon',
     [('        logger.info("CC conversational pointer not restored: reason=conv_pointer_missing', '        logger.warning("CC conversational pointer not restored: reason=conv_pointer_missing')]),
    ('D11 a CORRUPT sidecar downgraded to INFO (item 6)', 'daemon',
     [('        logger.warning("CC conversational pointer not restored: reason=conv_pointer_corrupt exc_type=%s -- the first forest after "',
       '        logger.info("CC conversational pointer not restored: reason=conv_pointer_corrupt exc_type=%s -- the first forest after "')]),
    ('Q2 the INFO claims "clean start" again (round 2)', 'daemon',
     [('reason=conv_pointer_missing -- no pointer sidecar found; the "', 'reason=conv_pointer_missing -- clean start, no sidecar yet; the "')]),
    ('D12 a STALE id downgraded to INFO (item 6)', 'daemon',
     [('        logger.warning("CC conversational pointer not restored: reason=conv_pointer_stale_id', '        logger.info("CC conversational pointer not restored: reason=conv_pointer_stale_id')]),
]


def load():
    with open('/proc/loadavg') as f:
        return float(f.read().split()[0])


def run(cmd, extra_env, cwd):
    """GATE, not wait (Exec P500): the START LOAD is recorded; at >= 6 the run is REFUSED and the whole table stops."""
    L = load()
    if L >= 6:
        print('REFUSED   start_load=%.2f (>= 6): NOT RUN; stopping (resume later with the mutation names as arguments)' % L)
        sys.exit(9)
    p = subprocess.run(cmd, cwd=cwd, env=dict(ENV, **extra_env), capture_output=True, text=True, timeout=600)
    tail = [l for l in p.stdout.splitlines() if re.search(r'\d+ (passed|failed)', l)]
    failed = sorted(set(re.findall(r'^FAILED (\S+)', p.stdout, re.M)))
    return ('start_load=%.2f %s' % (L, tail[-1] if tail else p.stdout[-200:])), failed


def pt(path):
    return [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '-rA', path]


def main(only):
    org_src, dae_src = open(ORG).read(), open(DAEMON).read()
    results = []
    for name, target, pairs in M:
        if only and not any(o in name for o in only):
            continue
        src = org_src if target == 'org' else dae_src
        for old, new in pairs:
            assert src.count(old) == 1, 'mutation %s matched %d times (must be exactly 1): %r' % (name, src.count(old), old[:70])
            src = src.replace(old, new)
        safe = re.sub(r'[^A-Za-z0-9]+', '_', name)[:40]
        if target == 'org':
            path = '%s/mut/%s_cc_ng_organism.py' % (SCR, safe)
            open(path, 'w').write(src)
            summary, failed = run(pt('tests/test_cc_bind_atomic_904.py'),
                                  {'Z12_904_ORG_UNDER_TEST': path, 'Z12_904_DAEMON_UNDER_TEST': DAEMON}, NG)
        else:
            path = '%s/mut/%s_cc-ng-daemon.py' % (SCR, safe)
            open(path, 'w').write(src)
            s1, f1 = run(pt('scripts/tests/test_cc_ng_daemon_deposit_904.py'), {'Z12_904_DAEMON_UNDER_TEST': path}, DOCS)
            s2, f2 = run(pt('tests/test_cc_bind_atomic_904.py'), {'Z12_904_DAEMON_UNDER_TEST': path}, NG)
            summary, failed = 'docs[%s] ng[%s]' % (s1, s2), f1 + f2
        verdict = 'KILLED' if failed else 'SURVIVED'
        short = [f.split('::')[-1][:60] for f in failed[:2]]
        print('%-9s %-78s %s\n          by: %s%s' % (verdict, name, summary, short, ' ...(+%d)' % (len(failed) - 2) if len(failed) > 2 else ''))
        sys.stdout.flush()
        results.append((name, verdict))
    k = sum(1 for _, v in results if v == 'KILLED')
    print('\n%d/%d mutations KILLED' % (k, len(results)))
    for n, v in results:
        if v != 'KILLED':
            print('SURVIVED:', n)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
