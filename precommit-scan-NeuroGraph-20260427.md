# Pre-Commit Security Scan — NeuroGraph
**Date:** 2026-04-27 | **Verdict:** ✅ SAFE TO COMMIT | **Risk Score:** 0/100

## Summary

| Severity | New | Existing |
|----------|-----|----------|
| 🔴 Critical | 0 | 0 |
| 🟠 High | 0 | 0 |
| 🟡 Medium | 0 | 17 |
| 🟢 Low | 0 | 0 |
| **Total** | **0** | **17** |

## Staged Changes (two commits)

**Commit 5d5f11b** — `neuro_foundation.py` — replace exact-set hyperedge discovery with Jaccard overlap-based candidate matching; add `he_discovery_overlap_threshold: 0.5` to DEFAULT_CONFIG

**Commit dfac337** — `neurograph_rpc.py` — add `_tune_he_overlap_threshold()` self-tuning coordinator; 6 new module-level globals; discover_hyperedges return value captured; ±0.03 nudge every 50 turns, bounds (0.2, 0.9)

## Pre-existing Findings (17, all in non-staged files or unchanged lines)

| Tool | Severity | Finding | File:Line |
|------|----------|---------|-----------|
| semgrep | WARNING | subprocess-shell-true | neurograph_gui.py:863 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_gui.py:955 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_rpc.py:665 |
| semgrep | WARNING | insecure-request-object | neurograph_rpc.py:2070 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_rpc.py:2077 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_rpc.py:2676 |
| semgrep | WARNING | insecure-urlopen | neurograph_rpc.py:2676 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_rpc.py:2714 |
| semgrep | WARNING | insecure-urlopen | neurograph_rpc.py:2714 |
| semgrep | WARNING | dynamic-urllib-use-detected | neurograph_rpc.py:2732 |
| semgrep | WARNING | insecure-urlopen | neurograph_rpc.py:2732 |
| semgrep | WARNING | insecure-request-object | ng_recall_hook.py:100 |
| semgrep | WARNING | dynamic-urllib-use-detected | ng_recall_hook.py:106 |
| semgrep | ERROR | pickles-in-pytorch | surgery/tonic_brain.py:227 |
| semgrep | ERROR | pickles-in-pytorch | surgery/tonic_brain.py:258 |
| semgrep | ERROR | pickles-in-pytorch | surgery/tonic_brain.py:281 |
| semgrep | WARNING | dynamic-urllib-use-detected | universal_ingestor.py:569 |

All 17 findings are pre-existing in non-staged files or on lines not touched by either commit. Zero findings in staged changes.
