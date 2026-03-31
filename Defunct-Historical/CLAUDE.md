# NeuroGraph Foundation

## Project Overview
The NeuroGraph Foundation is a reusable, project-agnostic cognitive architecture that provides AI systems with dynamic learning, causal reasoning, and self-optimizing knowledge management through the integration of:

1. **Semantic Memory Layer** (Vector Database) - Content storage and fuzzy similarity retrieval
2. **Structural Memory Layer** (Hypergraph Engine) - N-ary relationships and concept clusters  
3. **Temporal Dynamics Layer** (Spiking Neural Network with STDP) - Real-time learning and causal inference

## Documentation
Complete product requirements and technical specifications are in `/docs/`:
- `NeuroGraph_Foundation_PRD_v1_0_docx.pdf` - Core architecture (v1.0)
- `NeuroGraph_Foundation__Addendums.pdf` - Universal Ingestor & Memory Hierarchy (v1.1-1.2)

**CRITICAL**: Read both PRD documents in `/docs/` before starting any implementation.

## Implementation Status

All phases COMPLETE. See `CHANGELOG.md` for detailed per-phase build history.

| Phase | Version | Key Files |
|-------|---------|-----------|
| 1: Core Foundation | v0.1.0 | `neuro_foundation.py` |
| 1.5: Stability | v0.1.5 | `neuro_foundation.py` |
| 2: Hypergraph Engine | v0.2.0 | `neuro_foundation.py` |
| 2.5: Predictive Infra | v0.2.5 | `neuro_foundation.py` |
| 3: Predictive Coding | v0.3.0 | `neuro_foundation.py` |
| 3.5: State Persistence | v0.3.5 | `neuro_foundation.py` |
| 4: Universal Ingestor | v0.4.0 | `universal_ingestor.py` |
| 5: Deployment | v0.5.0 | `openclaw_hook.py`, `neurograph_migrate.py`, `deploy.sh`, CLIs |
| 5.5: Management GUI | v0.5.5 | `neurograph_gui.py` |
| 6: NG-Lite & Bridge | v0.6.0 | `ng_lite.py`, `ng_bridge.py` |
| 7: ET Module Manager | v0.7.0 | `ng_peer_bridge.py`, `et_modules/manager.py` |
| 9: CES | v0.9.0 | `ces_config.py`, `stream_parser.py`, `surfacing.py`, `ces_monitoring.py` |

## File Structure
```
NeuroGraph/
├── neuro_foundation.py              # Core implementation (Phase 1 + 1.5 + 2 + 2.5 + 3 + 3.5)
├── universal_ingestor.py            # Phase 4: Universal Ingestor System
├── openclaw_hook.py                 # Phase 5: OpenClaw integration singleton
├── neurograph_migrate.py            # Phase 5: Checkpoint migration framework
├── neurograph                       # Phase 5: neurograph management CLI
├── feed-syl                         # Phase 5: Ingestion/status CLI tool
├── deploy.sh                        # Phase 5: One-command deployment script
├── SKILL.md                         # Phase 5: OpenClaw skill definition
├── neurograph_gui.py                # Phase 5.5: tkinter management GUI
├── neurograph.desktop               # Phase 5.5: Linux desktop entry
├── ng_lite.py                       # Phase 6: Lightweight learning substrate (vendorable)
├── ng_bridge.py                     # Phase 6: NGSaaSBridge (Tier 3 bridge to full NeuroGraph)
├── ng_peer_bridge.py                # Phase 7: NGPeerBridge (Tier 2 cross-module learning)
├── et_module.json                   # Phase 7: Module manifest for ET ecosystem
├── et_modules/
│   ├── __init__.py
│   └── manager.py                   # Phase 7: ET Module Manager
├── ces_config.py                    # Phase 9: CES centralized configuration
├── stream_parser.py                 # Phase 9: Real-time stream parser (Ollama + fallback)
├── activation_persistence.py        # Phase 9: Cross-session activation state sidecar
├── surfacing.py                     # Phase 9: Surfacing monitor for prompt injection
├── ces_monitoring.py                # Phase 9: Health context, logger, HTTP dashboard
├── tests/
│   ├── __init__.py
│   ├── test_snn.py                  # SNN dynamics tests (12)
│   ├── test_stdp.py                 # STDP learning tests (9)
│   ├── test_hypergraph.py           # Phase 1 + 1.5 hyperedge tests (18)
│   ├── test_hypergraph_phase2.py    # Phase 2 hypergraph engine tests (26)
│   ├── test_phase25.py              # Phase 2.5 predictive infrastructure tests (26)
│   ├── test_prediction.py           # Phase 3 predictive coding tests (38)
│   ├── test_phase35.py              # Phase 3.5 prediction persistence tests (20)
│   ├── test_ingestor.py             # Phase 4 universal ingestor tests (88)
│   ├── test_integration.py          # End-to-end tests (18)
│   ├── test_migration.py            # Phase 5: Migration framework tests (32)
│   ├── test_openclaw_hook.py        # Phase 5: OpenClaw hook tests (18)
│   ├── test_gui.py                  # Phase 5.5: GUI non-GUI logic tests (25)
│   ├── test_ng_lite.py              # Phase 6: NG-Lite + NGSaaSBridge tests (52)
│   ├── test_et_modules.py           # Phase 7: ET Module Manager + NGPeerBridge tests (45)
│   └── test_ces.py                  # Phase 9: CES test suite (64)
├── examples/
│   ├── simple_usage.py              # Basic example
│   ├── project_configs.py           # OpenClaw, DSM, Consciousness configs
│   ├── hypergraph_demo.py           # Phase 2 hypergraph features demo
│   ├── ingest_code.py               # Phase 4: Ingest Python code example
│   ├── ingest_document.py           # Phase 4: Ingest markdown document example
│   └── ingest_multi_source.py       # Phase 4: Multi-source integration example
├── docs/                            # PRD documentation
├── requirements.txt
├── .gitignore
└── CLAUDE.md                        # This file
```

## Architecture Notes

### Key Design Principles
1. **Sparse by default** - No dense matrices, use sparse representations
2. **Dynamic topology** - Nodes/edges created/destroyed at runtime
3. **Pluggable plasticity** - Learning rules are swappable strategy objects
4. **Persistence-native** - All state is serializable

### Critical Implementation Details

**STDP Asymmetry** (Section 3.1.1):
- A_minus MUST be > A_plus (ratio 1.05-1.2) for stability
- Without this, network saturates

**Weight-dependent STDP**:
- LTP scaled by `(max_weight - w)/max_weight`
- Prevents runaway potentiation

**Homeostatic Plasticity** (Section 3.2):
- Use multiplicative synaptic scaling, NOT normalization
- Normalization destroys learned structure
- Scaling preserves relative weight ratios

**Refractory Periods**:
- Mandatory 2-step rest after firing
- Prevents unrealistic rapid re-firing

## Configuration Defaults (Section 9)
```python
DEFAULT_CONFIG = {
    # Phase 1: Core SNN
    'decay_rate': 0.95,
    'default_threshold': 1.0,
    'refractory_period': 2,
    'tau_plus': 20.0,
    'tau_minus': 20.0,
    'A_plus': 1.0,
    'A_minus': 1.2,
    'learning_rate': 0.01,
    'max_weight': 5.0,
    'target_firing_rate': 0.05,
    'scaling_interval': 100,
    'weight_threshold': 0.01,
    'grace_period': 500,
    'inactivity_threshold': 1000,
    'co_activation_window': 5,
    'initial_sprouting_weight': 0.1,
    # Phase 2: Hypergraph Engine
    'he_pattern_completion_strength': 0.3,
    'he_member_weight_lr': 0.05,
    'he_threshold_lr': 0.01,
    'he_discovery_window': 10,
    'he_discovery_min_co_fires': 5,
    'he_discovery_min_nodes': 3,
    'he_consolidation_overlap': 0.8,
    'he_member_evolution_window': 50,
    'he_member_evolution_min_co_fires': 10,
    'he_member_evolution_initial_weight': 0.3,
    # Phase 2.5: Predictive Infrastructure
    'prediction_window': 5,
    'prediction_ema_alpha': 0.01,
    'he_experience_threshold': 100,
}
```

## Testing Requirements (Section 9)

### Acceptance Criteria - Phase 1
1. 1K-node graph runs 10K steps without explosion or silent death
2. STDP correctly strengthens causal sequences (A→B strengthens when A fires before B)
3. STDP correctly weakens acausal pairs (A→B weakens when A fires after B)
4. Firing rates stabilize within 2× target after homeostatic regulation
5. ≥30% of speculative synapses are pruned within grace period
6. No memory leaks over 100K steps

## Implementation Guidelines

### What Claude Code Should Build

1. **Start with the core data structures** (Section 2.2):
   - Node class with all properties from PRD Table 2.2.1
   - Synapse class with all properties from Table 2.2.2
   - Hyperedge class with all properties from Table 2.2.3
   - Graph container

2. **Implement SNN simulation** (Section 2.2.4):
   - Voltage decay
   - Current injection
   - Spike propagation with delays
   - Refractory period enforcement

3. **Add STDP plasticity** (Section 3.1):
   - Follow mathematical specification exactly
   - Implement weight-dependent scaling
   - Add temporal aliasing handling

4. **Add homeostatic mechanisms** (Section 3.2):
   - Synaptic scaling (multiplicative)
   - Intrinsic excitability adjustment
   - Threshold adaptation

5. **Add structural plasticity** (Section 3.3):
   - Weight-based pruning
   - Co-activation sprouting
   - Age-based cleanup

6. **Comprehensive docstrings**:
   - Reference PRD sections
   - Explain failure modes and mitigations
   - Include example usage

### Code Style
- Use NumPy for vectorized operations where possible
- Sparse representations (dicts, sets) for topology
- Type hints throughout
- Comprehensive error handling
- Performance-critical sections should be optimized

## Dependencies
```
numpy>=1.24.0
scipy>=1.10.0           # For sparse matrices if needed
msgpack>=1.0.0          # For efficient serialization
sentence-transformers>=2.2.0  # For embedding generation (Phase 4)
beautifulsoup4>=4.12.0  # For URL/HTML extraction (Phase 4)
PyPDF2>=3.0.0           # For PDF extraction (Phase 4)
watchdog>=3.0.0         # For GUI file system monitoring (Phase 5.5)
```

## Notes for Claude Code

- **Read the PRDs first** - They contain critical implementation details
- **Follow the phased approach** - Don't try to build everything at once
- **Pay attention to failure modes** - PRD Section 3.1.2 lists common issues
- **Test as you go** - Each phase has acceptance criteria
- **Ask questions** - If anything in the PRDs is unclear, ask before implementing

## Changelog

See `CHANGELOG.md` for detailed per-phase build history (Phases 1–9).
