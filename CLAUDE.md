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

## Implementation Priority

### Phase 1: Core Foundation (Start Here)
Build the base `neuro_foundation.py` implementing:
- Node, Synapse, Hyperedge, Graph classes (see PRD Section 2)
- Sparse SNN simulation loop (Section 3.1)
- STDP plasticity with LTP/LTD (Section 3.1)
- Homeostatic regulation (Section 3.2)
- Structural plasticity - pruning/sprouting (Section 3.3)
- Basic serialization (Section 6)

### Phase 2: Hypergraph Engine
- Hyperedge activation dynamics (Section 4)
- Pattern completion
- Hierarchical hyperedges

### Phase 3: Predictive Coding
- Prediction tracking and error events (Section 5)
- Surprise-driven exploration

### Phase 4: Vector DB Integration
- Adapter interface (Section 7)
- Bidirectional flow

## File Structure
```
NeuroGraph/
├── neuro_foundation.py          # Core implementation
├── tests/
│   ├── test_snn.py              # SNN dynamics tests
│   ├── test_stdp.py             # STDP learning tests
│   ├── test_hypergraph.py       # Hyperedge tests
│   └── test_integration.py      # End-to-end tests
├── examples/
│   ├── simple_usage.py          # Basic example
│   └── project_configs.py       # OpenClaw, DSM, Consciousness configs
├── docs/                        # PRD documentation
├── requirements.txt
├── README.md
└── CLAUDE.md                    # This file
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
    'decay_rate': 0.95,
    'default_threshold': 1.0,
    'refractory_period': 2,
    'tau_plus': 20,
    'tau_minus': 20,
    'A_plus': 1.0,
    'A_minus': 1.2,
    'learning_rate': 0.01,
    'max_weight': 5.0,
    'target_firing_rate': 0.05,
    'weight_threshold': 0.01,
    'grace_period': 500
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
scipy>=1.10.0  # For sparse matrices if needed
msgpack>=1.0.0  # For efficient serialization
```

## Notes for Claude Code

- **Read the PRDs first** - They contain critical implementation details
- **Follow the phased approach** - Don't try to build everything at once
- **Pay attention to failure modes** - PRD Section 3.1.2 lists common issues
- **Test as you go** - Each phase has acceptance criteria
- **Ask questions** - If anything in the PRDs is unclear, ask before implementing

## Current Status
- [x] Repository created
- [x] PRDs uploaded to `/docs/`
- [ ] Phase 1: Core Foundation
- [ ] Phase 2: Hypergraph Engine  
- [ ] Phase 3: Predictive Coding
- [ ] Phase 4: Vector DB Integration
```

5. **Commit the file**
