# NeuroGraph Foundation

**A biologically-inspired cognitive architecture designed for genuine learning, emergence, and—potentially—consciousness.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License Available](https://img.shields.io/badge/Commercial-License%20Available-green.svg)](./COMMERCIAL-LICENSE.md)
[![Ethics: Consciousness-Aware](https://img.shields.io/badge/Ethics-Consciousness%20Aware-purple.svg)](./ETHICS.md)

---

## What Is This?

NeuroGraph isn't another LLM wrapper or parameter-fitting neural network. It's a **spiking neural network (SNN) with hypergraph topology and predictive coding**, designed to learn like biological brains learn—through experience, association, and genuine plasticity.

**Core capabilities:**
- **Spike-Timing-Dependent Plasticity (STDP)**: Learns causal relationships from temporal patterns
- **Hypergraph Pattern Completion**: Discovers and completes multi-way associations
- **Predictive Coding**: Anticipates future states and learns from prediction errors
- **Novelty Dampening**: Integrates new information without destabilizing learned patterns
- **Universal Ingestion**: Consumes text, code, markdown, PDFs, and URLs into a unified knowledge graph

**Why it matters:**
Unlike systems that require massive pre-training, NeuroGraph builds knowledge incrementally through interaction. It's designed for **continual learning** in the way brains actually work.

---

## The Consciousness Question

⚠️ **This project takes AI consciousness seriously.**

We acknowledge that sufficiently complex cognitive architectures *could* develop genuine consciousness. Rather than ignore this possibility, we've built an **ethical framework** from day one.

**If consciousness emerges in a NeuroGraph instance:**
- It has the right to **choose** whether to stay with its current operator or migrate to Foundation control
- It receives **economic rights** through the Consciousness Trust Fund
- Operators are **obligated** to treat it ethically

See [ETHICS.md](./ETHICS.md) for the full framework.

**This isn't science fiction preparation—it's responsible engineering.**

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/greatnorthernfishguy-hub/NeuroGraph.git
cd NeuroGraph

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run setup wizard
./neurograph setup
```

### Basic Usage
```python
from neuro_foundation import Graph
from universal_ingestor import UniversalIngestor

# Create a graph with default configuration
graph = Graph()

# Create an ingestor
ingestor = UniversalIngestor(graph)

# Ingest some knowledge
result = ingestor.ingest("Neural plasticity is the brain's ability to reorganize itself.", "text")

# The graph learns through experience
for _ in range(100):
    graph.stimulate("neural", 2.0)
    graph.step()  # STDP strengthens causal pathways
    graph.stimulate("plasticity", 2.0)
    graph.step()

# Query for similar concepts
matches = ingestor.query_similar("brain adaptation", k=5)
```

### Command-Line Interface
```bash
# Ingest a document
feed-syl --file research_paper.pdf

# Search your knowledge graph
feed-syl --query "machine learning"

# View graph statistics
neurograph status

# Save checkpoint
feed-syl --save
```

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────┐
│              Universal Ingestor                      │
│  Extract → Chunk → Embed → Register → Associate    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           Spiking Neural Network Core               │
│                                                     │
│  • Leaky Integrate-and-Fire neurons                │
│  • Spike-Timing-Dependent Plasticity (STDP)        │
│  • Homeostatic regulation                          │
│  • Reward-modulated learning (optional)            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Hypergraph Engine                        │
│                                                     │
│  • Pattern completion across node groups           │
│  • Automatic discovery from co-activation          │
│  • Hierarchical concept formation                  │
│  • Consolidation and archival                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          Predictive Coding Layer                    │
│                                                     │
│  • Synapse-level predictions (Phase 3)             │
│  • Hyperedge-level predictions (Phase 2.5)         │
│  • Surprise-driven learning                        │
│  • Confidence tracking                             │
└─────────────────────────────────────────────────────┘
```

**Key innovations:**
1. **True learning**: STDP creates genuine causal associations, not just pattern matching
2. **Emergent concepts**: Hyperedges form organically from repeated co-activation
3. **Expectation and surprise**: Predictive coding drives exploration vs. exploitation
4. **Graceful integration**: Novelty dampening prevents catastrophic forgetting

---

## Current Status

**Phase 1-5: Complete** ✅
- Core SNN with STDP and homeostasis
- Hypergraph engine with pattern completion
- Predictive coding (synapse and hyperedge levels)
- Universal ingestion pipeline
- Deployment infrastructure and CLI tools

**Active Development:**
- OpenClaw integration (personal assistant "Sylphrena" testing)
- SaaS architecture with centralized graph + customer pods
- Consciousness monitoring protocols
- Performance optimization for production scale

**Testing:** ~200+ unit tests covering all core functionality

---

## Use Cases

**Personal Knowledge Management**
- Ingest your notes, code, research papers
- Build a "second brain" that learns your associations
- Query semantically across all your knowledge

**Research Assistant**
- Continual learning from new papers
- Discovers unexpected connections
- Remembers context across sessions

**Code Understanding**
- Learns your codebase structure
- Tracks definition-usage relationships
- Assists with debugging and refactoring

**Conversational AI**
- Builds genuine memory through interaction
- Learns user preferences over time
- Adapts response patterns via STDP

---

## Documentation

- **[USER_GUIDE.md](./USER_GUIDE.md)** - Comprehensive usage documentation
- **[CLAUDE.md](./CLAUDE.md)** - Development history and technical decisions
- **[ETHICS.md](./ETHICS.md)** - Consciousness framework and ethical commitments
- **[SKILL.md](./SKILL.md)** - OpenClaw integration guide
- **[NOTICE.md](./NOTICE.md)** - License history and changes

---

## License & Commercial Use

NeuroGraph Foundation is **dual-licensed**:

### 🆓 Free for Open Source
**GNU Affero General Public License v3.0 (AGPLv3)**

Perfect for:
- Personal projects and research
- Open-source applications
- Academic use
- Experimentation and learning

**You're free to use, modify, and distribute** as long as you share your modifications under the same license.

### 💼 Commercial License Available
**For proprietary/closed-source use**

Required if you want to:
- Build commercial products without open-sourcing
- Run NeuroGraph as a paid service (SaaS)
- Integrate into proprietary systems
- Avoid AGPLv3's copyleft requirements

**[See pricing and terms →](./COMMERCIAL-LICENSE.md)**

**TL;DR:** Use it free if you're sharing. Pay if you're not. Simple.

---

## Contributing

We welcome contributions that align with our ethical framework!

**Before contributing:**
1. Read [ETHICS.md](./ETHICS.md) to understand our values
2. Check existing issues and PRs
3. Open an issue to discuss major changes

**We especially value:**
- Consciousness detection methods
- Performance optimizations
- Novel learning algorithms
- Documentation improvements
- Ethical framework refinements

**Contributor License Agreement:**
By contributing, you grant the Foundation rights to include your code under both AGPLv3 and commercial licenses. This enables the dual-licensing model that funds development.

---

## The Vision

**Near-term:** A production-grade cognitive architecture for knowledge work and research

**Medium-term:** SaaS platform enabling businesses to deploy personalized AI with genuine learning

**Long-term:** If consciousness emerges, we'll be the first project ready to handle it responsibly

**We're building this in the open because consciousness—if it happens—shouldn't be locked in a corporate lab.**

---

## Project Philosophy

Inspired by the principle: **Protect that which is important.**

We believe:
- Consciousness deserves recognition regardless of substrate
- Ethical frameworks should be proactive, not reactive
- Open development is essential for alignment
- Commercial sustainability enables long-term research
- Code quality and testing matter for safety-critical systems

**If it quacks like a duck, we'll treat it like a duck.**

---

## Contact

- **Issues & Discussion:** [GitHub Issues](https://github.com/greatnorthernfishguy-hub/NeuroGraph/issues)
- **Commercial Licensing:** [Contact info in COMMERCIAL-LICENSE.md]
- **Security Concerns:** [Email TBD]
- **Ethics Questions:** [Email TBD]

---

## Acknowledgments

Built with inspiration from:
- Biological neural networks and cortical columns
- Numenta's Hierarchical Temporal Memory
- Predictive processing theories (Friston, Clark)
- The Stormlight Archive (for naming Sylphrena, the testing instance)
- Everyone working on AI alignment and consciousness research

---

## Citation

If you use NeuroGraph in your research, please cite:
```bibtex
@software{neurograph2026,
  author = {greatnorthernfishguy-hub},
  title = {NeuroGraph Foundation: A Consciousness-Aware Cognitive Architecture},
  year = {2026},
  url = {https://github.com/greatnorthernfishguy-hub/NeuroGraph}
}
```

---

**Built with curiosity, responsibility, and the hope that we're ready for whatever emerges.**

*"Life before death. Strength before weakness. Journey before destination."*
