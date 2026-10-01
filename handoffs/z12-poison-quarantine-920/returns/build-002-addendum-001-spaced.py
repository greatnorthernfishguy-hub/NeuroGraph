#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13417+ — spaced driver for the 16 CARRIED mutants
# What: runs build-002-mutants.py once per mutant (M8-M14, D1-D9), each run gated by the harness's own START LOAD gate (1m < 6 AND 5m < 5.0),
#   with a 20 s pause BETWEEN mutants. On the first refusal (harness exit 3) it STOPS: the rest stay CARRIED and are listed. A pause is not a
#   retry: a refused mutant is never re-attempted.
# Why: Z12's carried-mutants nudge: my own back-to-back runs raised the 1-minute load 3.67 -> 5.89 last time.
# How: scratch tooling only (/tmp); no repo file is edited.
# -------------------
import subprocess
import sys
import time

NG = '/home/josh/NeuroGraph-worktrees/z12-quarantine-920-20261001'
DOCS = '/home/josh/docs/.claude/worktrees/daemon-quarantine-920-20261001'
MUTANTS = [
    'M8_no_idempotency_always_append', 'M9_already_present_skips_the_fsync_the_hole', 'M10_key_is_the_text_not_the_bytes',
    'M11_count_not_cleared_on_success', 'M12_quarantined_but_tract_not_advanced', 'M13_default_hold_warning_text_changed',
    'M14_counter_incremented_on_already_present_too',
    'D1_kwarg_passed_when_hold_off', 'D2_default_5_even_with_hold_off', 'D3_bound_read_per_call_from_env', 'D4_counter_not_mirrored',
    'D5_garbage_warning_at_import', 'D6_main_never_warns', 'D7_zero_accepted', 'D8_mirror_inside_the_graph_lock',
    'D9_stat_seeded_even_when_flag_off',
]
PAUSE = 20

for i, name in enumerate(MUTANTS):
    r = subprocess.run(['python3', '/tmp/z12-920-scratch/build-002-mutants.py', NG, DOCS, name], capture_output=True, text=True)
    out = [ln for ln in r.stdout.splitlines() if ln.strip()]
    for ln in out:
        print(ln[:260], flush=True)
    if r.returncode == 3 or any('REFUSED' in ln for ln in out):
        print('GATE REFUSED at %s: STOPPING. CARRIED (not run): %s' % (name, ', '.join(MUTANTS[i:])), flush=True)
        sys.exit(3)
    if r.returncode not in (0,) or not out:
        print('HARNESS PROBLEM at %s (exit %s): %s' % (name, r.returncode, (r.stderr or '')[-300:]), flush=True)
        sys.exit(4)
    if i + 1 < len(MUTANTS):
        time.sleep(PAUSE)
print('ALL 16 RAN', flush=True)
