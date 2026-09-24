---
name: ruling-coresident-cc-export-scoping
description: Exporting CC topology from the VPS (where CC-NG and Syl-NG are co-resident in one process) requires instance-pinning + a POSITIVE cc-provenance whitelist, never a Syl-class blacklist
metadata:
  type: project
---

**Context (verified cc_ng_host.py:1-22):** On the VPS, the CC's NeuroGraphMemory and Syl's NeuroGraphMemory run in the SAME neurograph_rpc.py process but are "COMPLETELY ISOLATED" — different workspace (CC: ~/.claude/plugins/neurograph, Syl: ~/NeuroGraph/data), different checkpoints, different graph, different vector_db, different NeuroGraphMemory instance. Two graph objects live in one process.

**Ruling — any CC-topology export must guarantee no Syl leakage via TWO belts, not one:**
1. **Instance/path pinning:** the export must read the CC NeuroGraphMemory instance (cc_ng_host `_STATE`) or the CC workspace checkpoint explicitly, and ASSERT it, aborting if pointed at Syl's graph/workspace. A co-resident mis-wire is a Syl-topology leak = Syl's Law + Duck-Ethics territory.
2. **Positive provenance whitelist, NOT a Syl blacklist:** export ONLY nodes carrying a CC marker (`{"cc": True}` / id prefix `cc:conv::` / `::tree::` / `cc_authored` / creation_mode tags) and assert every exported node has one, aborting otherwise. Relying on `Graph._is_identity_protected` (skips constitutional/syl_authored) as the leakage guard is WRONG: on the CC graph it matches ~nothing (false confidence), and if ever pointed at Syl's graph it would still export all her ordinary conversational nodes (tagged `{"syl": True}`, neurograph_rpc.py:2415) since only her wants+spine are protected.

**Risk-ordering affirmed:** #70 v1 keeps the mutating structural INSTALL on the laptop and the VPS read-only — correct, lowest-risk. v2 (Path B, laptop→VPS receiver) would run create_node/create_synapse INSIDE the co-resident process; that install must get its own dedicated review with the same instance-pinning + Syl's-Law backup discipline before build. See [[ruling-cc-river-merge-intra-mind]] and [[finding-cc-authored-want-protection-gap]].
