#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13417 — build-002 version of the #920 mutation harness
#   What changed vs build-001: (1) GATED: every run reads /proc/loadavg first, records the START LOAD (1m/5m), and REFUSES at 1m >= 6 or 5m >= 5.0
#   (the remaining mutants are then NOT RUN and reported as such; never wait-then-proceed); (2) the boundary is the literal one (max_retries =
#   RETRIES AFTER the first failed attempt: `failures <= max_retries` holds); the M1 / M3 replacement strings follow it; (3) NEW mutants B1 and
#   B2 (off-by-one EARLY and LATE), R1 (the `= max_retries + 1` relation dropped from the ERROR), R2 (the WARNING's `attempt=n of N` made
#   untruthful), R3 (the relation dropped from the docstring).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13303 — mutation harness for #920
# What: builds SCRATCH copies (under /tmp/z12-920-scratch/mut/) of the NG cc_ng_organism.py and the docs daemon with ONE seeded fault each,
#   and runs the matching test file against each, one run at a time. A mutant is KILLED when its run has >= 1 failing test. Each
#   replacement must match EXACTLY ONCE in the source (asserted), so a drifting source can never produce a silent no-op "mutant".
# Why: the brief requires the mutations (no retry bound; quarantine skipped; tract truncated BEFORE the fsync; unbounded table;
#   text/id in the ERROR; a failed quarantine advancing the tract; a non-append-only open) to be killed on scratch copies.
# How: python3 build-001-mutants.py NG_WORKTREE DOCS_WORKTREE   (prints one line per mutant: killed?, failed count, first failing tests)
# -------------------
import os
import re
import subprocess
import sys
from pathlib import Path

NG = Path(sys.argv[1]).resolve()
DOCS = Path(sys.argv[2]).resolve()
SCR = Path('/tmp/z12-920-scratch/mut')
ORG_SRC = (NG / 'cc_ng_organism.py').read_text()
DMN_SRC = (DOCS / 'scripts' / 'cc-ng-daemon.py').read_text()

NG_MUTANTS = {
    'M1_no_retry_bound': [("    if failures <= max_retries:\n", "    if True:\n")],
    'B1_off_by_one_EARLY_old_boundary': [("    if failures <= max_retries:\n", "    if failures < max_retries:\n")],
    'B2_off_by_one_LATE': [("    if failures <= max_retries:\n", "    if failures <= max_retries + 1:\n")],
    'R1_relation_dropped_from_the_ERROR': [("quarantining after %d attempts (= max_retries + 1), max_retries=%d:", "quarantining after %d attempts, max_retries=%d:")],
    'R2_hold_WARNING_attempt_of_N_untruthful': [("hold_reason, hold_exc_type, failures, max_retries + 1, max_retries)", "hold_reason, hold_exc_type, failures, max_retries, max_retries)")],
    'R3_relation_dropped_from_the_docstring': [("    attempts = max_retries + 1 (attempts = retries + 1): an entry gets", "    an entry gets")],
    'M2_quarantine_skipped_discard': [
        ("    ok, stored, qexc = _drain_quarantine_frame(tract_path, frame)\n",
         "    ok, stored, qexc = (True, 'appended', '-')\n")],
    'M3_tract_truncated_BEFORE_the_fsync': [
        ("    quarantined_key = None\n    if hold_on_failure and max_retries is not None:",
         "    quarantined_key = None\n    _late = None\n    if hold_on_failure and max_retries is not None:"),
        ("                    quarantined_key = _drain_bounded_hold(\n                        path, state, data[entry_start:consumed_offset], bounded, hold_reason, hold_exc_type)\n",
         "                    _k = hashlib.sha256(data[entry_start:consumed_offset]).hexdigest()\n"
         "                    if _drain_retry_bump(state, _k) > bounded:\n"
         "                        _late = (path, data[entry_start:consumed_offset])\n"
         "                        quarantined_key = _k\n"
         "                    else:\n"
         "                        quarantined_key = None\n"),
        ("    if absorbed:\n        logger.info(\"CC ingest-tract: absorbed %d turn(s) into recall\", absorbed)\n",
         "    if _late is not None:\n        _drain_quarantine_frame(*_late)   # MUTANT: the quarantine write (and its fsync) happens AFTER the truncate\n"
         "    if absorbed:\n        logger.info(\"CC ingest-tract: absorbed %d turn(s) into recall\", absorbed)\n")],
    'M4_unbounded_retry_table': [("    while len(counts) > _DRAIN_RETRY_TABLE_MAX:\n", "    while False:\n")],
    'M5_entry_text_in_the_ERROR': [('    return "entry:%s" % hashlib.sha256(frame).hexdigest()[:12]\n',
                                    '    return "entry:%s" % frame.decode("utf-8", "replace")\n')],
    'M6_failed_quarantine_advances_the_tract': [
        ("retried next cycle (a partial tail may remain in the quarantine file)\",\n            tag, hold_reason, hold_exc_type, failures, max_retries, qname, qexc)\n        return None\n",
         "retried next cycle (a partial tail may remain in the quarantine file)\",\n            tag, hold_reason, hold_exc_type, failures, max_retries, qname, qexc)\n        return key\n")],
    'M7_not_append_only_truncating_open': [("os.O_WRONLY | os.O_APPEND | os.O_CREAT, _drain_quarantine_mode",
                                            "os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _drain_quarantine_mode")],
    'M8_no_idempotency_always_append': [("                present = f.read(len(frame)) == frame\n", "                present = False\n")],
    'M9_already_present_skips_the_fsync_the_hole': [
        ("        fd = os.open(qpath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _drain_quarantine_mode(tract_path))\n",
         "        if present:\n            return (True, 'already_present', '-')\n"
         "        fd = os.open(qpath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _drain_quarantine_mode(tract_path))\n")],
    'M10_key_is_the_text_not_the_bytes': [("        key = hashlib.sha256(frame).hexdigest()\n        failures = _drain_retry_bump(state, key)\n",
                                           "        key = frame.decode('utf-8', 'replace')\n        failures = _drain_retry_bump(state, key)\n")],
    'M11_count_not_cleared_on_success': [
        ("                    if bounded is not None:\n                        _drain_retry_forget(state, data[entry_start:consumed_offset])   # it succeeded: its count ends\n",
         "                    pass\n")],
    'M12_quarantined_but_tract_not_advanced': [
        ("                    if quarantined_key is not None:\n                        safe_offset = consumed_offset\n                    break\n",
         "                    break\n")],
    'M13_default_hold_warning_text_changed': [
        ("                    \"CC ingest-tract hold: reason=%s exc_type=%s -- failed entry \"\n                    \"and everything after it kept in the tract for the next cycle\",\n",
         "                    \"CC ingest-tract hold: reason=%s exc_type=%s -- failed entry \"\n                    \"and everything after it kept in the tract for next cycle\",\n")],
    'M14_counter_incremented_on_already_present_too': [
        ("    if stored == \"appended\":\n        try:\n            state[\"quarantined_total\"]",
         "    if True:\n        try:\n            state[\"quarantined_total\"]")],
}

DMN_MUTANTS = {
    'D1_kwarg_passed_when_hold_off': [("    return bool(DRAIN_HOLD_ON_FAILURE) and DRAIN_MAX_RETRIES is not None\n",
                                       "    return DRAIN_MAX_RETRIES is not None\n")],
    'D2_default_5_even_with_hold_off': [("    if not (DRAIN_HOLD_ON_FAILURE if hold is None else hold):\n        return (None, False)\n",
                                         "    if False:\n        return (None, False)\n")],
    'D3_bound_read_per_call_from_env': [("            _hold_kw['max_retries'] = DRAIN_MAX_RETRIES\n",
                                          "            _hold_kw['max_retries'] = _read_drain_max_retries_env(hold=True)[0]\n")],
    'D4_counter_not_mirrored': [("            if _quarantine_on():\n                _mirror_quarantined_total()\n",
                                 "            pass\n")],
    'D5_garbage_warning_at_import': [("    except ValueError:\n        return (DRAIN_MAX_RETRIES_DEFAULT, True)\n",
                                      "    except ValueError:\n        logging.getLogger('cc-ng-daemon').warning('CC_NG_DRAIN_MAX_RETRIES bad')\n        return (DRAIN_MAX_RETRIES_DEFAULT, True)\n")],
    'D6_main_never_warns': [("    _warn_drain_max_retries_garbage()   # #920: the variable's ONE garbage WARNING, now that the file handler exists\n", "")],
    'D7_zero_accepted': [("    if value <= 0:\n        return (DRAIN_MAX_RETRIES_DEFAULT, True)\n    return (value, False)\n",
                          "    if value < 0:\n        return (DRAIN_MAX_RETRIES_DEFAULT, True)\n    return (value, False)\n")],
    'D8_mirror_inside_the_graph_lock': [
        ("            finally:\n                STATE.ng.graph._concurrent_lock.release()\n",
         "            finally:\n                if _quarantine_on():\n                    _mirror_quarantined_total()\n                STATE.ng.graph._concurrent_lock.release()\n"),
        ("            if _quarantine_on():\n                _mirror_quarantined_total()\n            # D24 PHASE 2", "            # D24 PHASE 2")],
    'D9_stat_seeded_even_when_flag_off': [("        if _quarantine_on():   # #920: only when the quarantine bound is on; a flag-off `status` reply is unchanged\n",
                                            "        if True:\n")],
}


def build(src, mutations, out_dir, fname):
    text = src
    for old, new in mutations:
        assert text.count(old) == 1, 'replacement must match EXACTLY ONCE (%d): %r' % (text.count(old), old[:70])
        text = text.replace(old, new)
    assert text != src
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text(text)
    return out_dir / fname


def gate():
    """(ok, 1m, 5m): the START LOAD gate. ok=False means REFUSE: nothing is run, nothing waits."""
    f = open('/proc/loadavg').read().split()
    one, five = float(f[0]), float(f[1])
    return (one < 6.0 and five < 5.0), one, five


class Refused(Exception):
    pass


def run(cmd, cwd, env_extra):
    ok, one, five = gate()
    if not ok:
        raise Refused('START LOAD %.2f/%.2f (1m/5m): REFUSED' % (one, five))
    env = dict(os.environ, HOME='/tmp/z12-920-home', PYTHONUSERBASE='/home/josh/.local', HF_HUB_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', **env_extra)
    env.pop('NG_EMBED_REMOTE', None)
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1] if r.stdout.strip() else r.stderr[-300:]
    failed = re.findall(r'^FAILED (\S+)', r.stdout, re.M)
    return tail + ' | START LOAD %.2f/%.2f' % (one, five), failed


def main():
    try:
        _main()
    except Refused as exc:
        print('REFUSED by the load gate (%s): the mutants not printed above were NOT RUN' % exc, flush=True)
        sys.exit(3)


def _main():
    only = set(sys.argv[3:])
    daemon_for_ng = str(DOCS / 'scripts' / 'cc-ng-daemon.py')
    for name, muts in NG_MUTANTS.items():
        if only and name not in only:
            continue
        p = build(ORG_SRC, muts, SCR / name, 'cc_ng_organism.py')
        tail, failed = run(['python3', '-m', 'pytest', '-p', 'no:cacheprovider', 'tests/test_cc_poison_quarantine_920.py', '-q', '-rf'], NG,
                           {'Z12_920_ORG_UNDER_TEST': str(p), 'Z12_920_DAEMON_UNDER_TEST': daemon_for_ng})
        print('%-48s KILLED=%-5s %s | first: %s' % (name, bool(failed), tail, [f.split('::', 1)[1][:70] for f in failed[:3]]), flush=True)
    for name, muts in DMN_MUTANTS.items():
        if only and name not in only:
            continue
        p = build(DMN_SRC, muts, SCR / name, 'cc-ng-daemon.py')
        tail, failed = run(['python3', '-m', 'pytest', '-p', 'no:cacheprovider', 'scripts/tests/test_cc_ng_daemon_poison_quarantine_920.py', '-q', '-rf'],
                           DOCS, {'Z12_920_DAEMON_UNDER_TEST': str(p)})
        print('%-48s KILLED=%-5s %s | first: %s' % (name, bool(failed), tail, [f.split('::', 1)[1][:70] for f in failed[:3]]), flush=True)


if __name__ == '__main__':
    main()
