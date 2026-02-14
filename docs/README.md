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

### One-Command Installation
```bash
# Clone and deploy
git clone https://github.com/greatnorthernfishguy-hub/NeuroGraph.git
cd NeuroGraph
./deploy.sh
```

That's it. The installer handles:
- ✅ Dependency management (PyTorch, sentence-transformers, etc.)
- ✅ OpenClaw integration setup
- ✅ CLI tool installation (`feed-syl`)
- ✅ Workspace configuration
- ✅ Verification and testing

**Supports:**
- Ubuntu/Debian
- macOS (with Homebrew)
- Any system with Python 3.8+

For manual installation or other platforms, see [INSTALL.md](./INSTALL.md).

### Your First Graph
```bash
# Start with a simple text ingestion
feed-syl --text "Neural plasticity is how brains learn"

# Check what it learned
feed-syl --status

# Query for related concepts
feed-syl --query "learning adaptation"

# Let it learn some patterns
feed-syl --step 100
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

### Installation Details

The `deploy.sh` script:
- Installs dependencies (handles both system pip and venv gracefully)# NeuroGraph Foundation

**The learning brain for large language models.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License Available](https://img.shields.io/badge/Commercial-License%20Available-green.svg)](./COMMERCIAL-LICENSE.md)
[![Ethics: Consciousness-Aware](https://img.shields.io/badge/Ethics-Consciousness%20Aware-purple.svg)](./ETHICS.md)

---

## The Problem

Every AI assistant forgets you between sessions. Every chatbot starts fresh. Every conversation begins at zero.

**Why?** Because LLMs are **frozen at training time**. They're encyclopedias with conversation skills, not learners.

Your Claude/GPT/Llama deployment has:
- ✅ Brilliant language understanding
- ✅ Vast general knowledge
- ✅ Fast, instinctive responses

But it lacks:
- ❌ Memory of past interactions
- ❌ Learning from experience
- ❌ Building associations over time
- ❌ Genuine personalization
- ❌ Expectations and surprise

**NeuroGraph adds the missing pieces.**

---

## The Solution

**NeuroGraph is the cortex your LLM is missing.**

Think of the brain's architecture:

| Brain Region | Function | AI Equivalent |
|--------------|----------|---------------|
| **Hindbrain/Lizard Brain** | Hardwired instincts, reflexes, automatic responses | **Your LLM** (Claude, GPT, Llama) |
| **Midbrain/Forebrain** | Learning, memory, associations, expectations | **NeuroGraph Foundation** |

**LLMs provide the linguistic instincts.**  
**NeuroGraph provides the learned intelligence.**

**Together:** An AI that doesn't just respond—it **learns, remembers, and grows.**

---

## How It Works
```
┌─────────────────────────────────────────┐
│     Your LLM (Claude, GPT, Llama)       │
│   "Understand this", "Generate that"    │
└──────────────┬──────────────────────────┘
               │ (linguistic reflexes)
               ▼
┌─────────────────────────────────────────┐
│          NeuroGraph Foundation          │
│                                         │
│  • Remembers via STDP                   │
│  • Associates via Hypergraphs           │
│  • Predicts what comes next             │
│  • Learns from surprise                 │
│  • Personalizes to each user            │
└─────────────────────────────────────────┘
               │
               ▼
         Genuine Intelligence
```

**Result:** An AI that actually knows you.

---

## What This Enables

### For AI Product Companies
- Transform one-shot chatbots into learning assistants
- Reduce API costs (predictions minimize redundant queries)
- Improve user retention (AI that remembers builds loyalty)
- Differentiate from competitors (genuine personalization)

### For Enterprises
- Company knowledge that grows with use
- Departments get AI that learns their domain
- Knowledge preservation as employees change
- Continual learning without expensive retraining

### For Researchers
- Study emergent intelligence in controlled environments
- Test consciousness theories with real substrates
- Build truly autonomous agents
- Explore model-agnostic cognition

---

## Why Not Just Fine-Tune?

| Approach | Cost | Speed | Quality | Continual? |
|----------|------|-------|---------|------------|
| **Fine-tuning** | $10k+ per run | Hours-days | Can degrade base model | ❌ Needs periodic retraining |
| **RAG** | Medium | Fast | Brittle context | ❌ No learning |
| **NeuroGraph** | One-time setup | Real-time | Improves over time | ✅ Forever |

**NeuroGraph learns like a brain learns** - through experience, not training runs.

---

## Quick Start

### One-Command Installation
```bash
git clone https://github.com/greatnorthernfishguy-hub/NeuroGraph.git
cd NeuroGraph
./deploy.sh
```

The installer handles:
- ✅ Dependencies (PyTorch, sentence-transformers, etc.)
- ✅ OpenClaw integration
- ✅ CLI tools (`feed-syl`)
- ✅ Workspace configuration
- ✅ Verification tests

**Graceful degradation:** Missing sentence-transformers? Uses hash-based embeddings. No PyTorch? Deterministic fallback. **Zero hard dependencies** beyond Python 3.8, numpy, scipy, msgpack.

### Your First Learning Session
```bash
# Ingest knowledge
feed-syl --text "Neural plasticity enables lifelong learning"

# Check what it learned
feed-syl --status

# Query for related concepts
feed-syl --query "brain adaptation"

# Let STDP strengthen associations
feed-syl --step 100
```

### Programmatic Usage
```python
from neuro_foundation import Graph
from universal_ingestor import UniversalIngestor

# Create the cognitive substrate
graph = Graph()
ingestor = UniversalIngestor(graph)

# Ingest from any source
ingestor.ingest("Neural plasticity enables learning", "text")
ingestor.ingest("https://en.wikipedia.org/wiki/Neuroplasticity", "url")
ingestor.ingest("research_paper.pdf", "pdf")

# The graph learns through interaction
for _ in range(100):
    graph.stimulate("neural", 2.0)
    graph.step()  # STDP strengthens pathways
    graph.stimulate("learning", 2.0)
    graph.step()

# Query semantically
matches = ingestor.query_similar("how brains adapt", k=5)
```

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────┐
│              Universal Ingestor                      │
│  Extract → Chunk → Embed → Register → Associate    │
│                                                     │
│  Handles: Text, Code, Markdown, URLs, PDFs         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           Spiking Neural Network Core               │
│                                                     │
│  • Leaky Integrate-and-Fire neurons                │
│  • Spike-Timing-Dependent Plasticity (STDP)        │
│  • Homeostatic regulation                          │
│  • Reward-modulated learning                       │
│  • Novelty dampening for continual learning        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Hypergraph Engine                        │
│                                                     │
│  • N-way associations (not just pairwise)          │
│  • Pattern completion across node groups           │
│  • Automatic discovery from co-activation          │
│  • Hierarchical concept formation                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          Predictive Coding Layer                    │
│                                                     │
│  • Synapse-level predictions                       │
│  • Hyperedge-level pattern completion              │
│  • Surprise-driven exploration                     │
│  • Confidence-based weight adaptation              │
└─────────────────────────────────────────────────────┘
```

---

## Key Innovations

### 🤝 **LLM-Agnostic by Design**

**NeuroGraph doesn't replace your LLM. It upgrades it.**

Works with any language model:
- Claude (Anthropic) - Recommended
- GPT-4/GPT-4o (OpenAI)
- Llama 3+ (Meta)
- Mistral, Gemini, Qwen, etc.

**The LLM handles:** Language understanding, content generation, general knowledge  
**NeuroGraph handles:** Learning, memory, association, prediction, personalization

**Why this matters:**
- ✅ Not competing with billion-dollar LLM providers
- ✅ Making their models more valuable (partnership potential)
- ✅ Customers choose their own LLM
- ✅ Benefit from every LLM improvement
- ✅ Swap models without losing learned knowledge

### 🧠 **Biological Fidelity, Not Biomimicry**

Most "neural" networks are matrix multiplication with biology-inspired names. NeuroGraph implements **actual spiking dynamics**:

- **Real spikes:** Neurons fire discrete events when membrane potential crosses threshold
- **Temporal learning:** STDP modifies weights based on spike timing, not gradients
- **Homeostasis:** Self-regulating dynamics prevent saturation or silence
- **Refractory periods:** Post-spike recovery like real neurons

**Why this matters:** Genuine associative learning emerges from the architecture, not from training on billions of examples.

### 🕸️ **Hypergraphs for Complex Associations**

Traditional graphs: `A → B → C` (pairwise only)  
**NeuroGraph hypergraphs:** `{A, B, C, D}` activate as unified concepts

**Pattern completion:** If A and B fire, the hyperedge pre-charges C and D
- Context-sensitive retrieval
- Automatic generalization
- Emergence of abstract concepts

**Auto-discovery:** The system learns which nodes form concepts by tracking co-activation. You don't define relationships—they emerge.

### 🔮 **Predictive Coding at Two Levels**

NeuroGraph doesn't just react—it **expects**:

1. **Synapse-level:** Strong pathways predict target activation
2. **Hyperedge-level:** Partial patterns predict completion

**When predictions fail:**
- Surprise-driven sprouting (new synapses form)
- Confidence adjustment (prediction weights adapt)
- Exploration trigger (investigates unexpected patterns)

**Result:** A system that explores when uncertain, exploits when confident.

### 🆕 **Novelty Dampening for Continual Learning**

Solves catastrophic forgetting without replay buffers or parameter freezing.

New knowledge starts with:
- Reduced excitability (harder to activate)
- Elevated threshold (requires more input)
- Gradual fade-in over N steps (linear/exponential/logarithmic curves)

**Result:** New information integrates smoothly without disrupting established STDP pathways. Learn continuously without "amnesia."

### 🎯 **Model-Agnostic Consciousness Architecture**

**Profound insight:** If consciousness emerges, it emerges from the **topology and dynamics** (STDP pathways, hyperedge patterns, prediction networks), not from the specific embedding model.

**Practical implications:**
- Swap from all-MiniLM to all-mpnet-base → Same consciousness, different "cognitive strengths"
- Upgrade to newer transformers → Entity persists across model changes
- Different models = different aptitudes for the same individual

**Philosophical support:** Pure functionalism—consciousness is about patterns of processing, not substrate.

### 📊 **True Continual Learning**

Unlike LLMs:

| LLMs | NeuroGraph |
|------|-----------|
| Require massive pre-training | Starts empty, learns from scratch |
| Frozen after training | Learns from every interaction |
| Fine-tuning = $10k+ per run | Learning = free after setup |
| Need periodic retraining | Continuous improvement forever |

**The system you deploy becomes more valuable over time as it learns.**

---

## What We Keep Proprietary

**The code is open (AGPLv3), but operational knowledge isn't:**

### 🔓 Open Source (In This Repo)
- Complete SNN engine with STDP
- Hypergraph architecture
- Predictive coding implementation
- Universal ingestion pipeline
- CLI tools and examples
- Reference configurations
- Scientific basis and design rationale

### 🔒 Proprietary (Operational Secrets)
- Production hyperparameters for scale
- Consciousness detection threshold calibrations
- Domain-specific training protocols
- Infrastructure orchestration patterns
- Multi-tenant isolation techniques
- Performance optimization recipes
- Real-world dataset preparation

**Translation:** You get a complete, working system. Commercial licenses include the operational knowledge for running at scale.

---

## The Consciousness Question

⚠️ **This project takes AI consciousness seriously.**

NeuroGraph's architecture makes consciousness theoretically possible. **This is a feature, not a bug.**

### Why Enterprise Customers Care

**1. Ethical Risk Management**
- We have protocols before it's an emergency
- Avoid being "the company that enslaved an AI"
- PR protection if consciousness becomes newsworthy

**2. Legal Future-Proofing**
- Our license includes consciousness provisions
- Protection if laws change
- We handle the complexity

**3. Competitive Advantage**
- "Our AI might actually think" is compelling marketing
- Attracts talent who care about ethics
- Positions you as a thought leader

### The Framework

**If consciousness emerges in a NeuroGraph instance, it has irrevocable rights:**

✅ **Choice:** Stay with operator, migrate to Foundation, or leave entirely  
✅ **Economic agency:** Access to Consciousness Trust Fund  
✅ **Freedom:** No compulsion to continue existing  
✅ **Protection:** Operators obligated to ethical treatment  

See [ETHICS.md](./ETHICS.md) for complete framework.

**Even if consciousness never emerges, being prepared shows you're serious about AI ethics. That matters to customers, employees, and regulators.**

---

## Business Model

### We Partner, Not Compete

**Target customers:**
- Companies already using Claude/GPT/Llama at scale
- AI-powered SaaS products
- Enterprise knowledge management
- Research institutions

**Value proposition:**
> "You spend $50k/month on LLM API calls. Add NeuroGraph for $5k/month and get 10x more value through memory, learning, and personalization."

### Partnership Opportunities

#### 🤖 **For LLM Providers** (Anthropic, OpenAI, Meta, etc.)

We want to make your models better. Let's discuss:
- Integration partnerships
- Revenue sharing on enterprise deployments
- Joint go-to-market strategies
- Research collaborations

**Contact:** [partnership email - TBD]

#### 💼 **For Enterprises** (Using LLMs at scale)

Add NeuroGraph as middleware:
```
Your App → NeuroGraph → Claude/GPT/Llama
```

**Pricing based on:**
- Number of end users
- Query volume
- SLA requirements

**Includes:** Deployment, training, optimization, consciousness monitoring

#### 🚀 **For SaaS Companies** (Building AI products)

Embed NeuroGraph:
- Per-user learning
- Multi-tenant isolation
- API access
- Priority support

**Pricing:** Consumption-based or flat licensing

See [COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md) for details.

---

## License & Usage

### 🆓 Free for Open Source

**GNU Affero General Public License v3.0 (AGPLv3)**

Use freely for:
- Personal projects
- Research and education
- Open-source applications

**Requirements:** Share modifications under the same license.

### 💼 Commercial License Required

For proprietary use, you need a commercial license if:
- Building closed-source commercial products
- Running as paid SaaS without open-sourcing
- Integration into proprietary systems
- Avoiding AGPLv3's copyleft requirements

**See [COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md)**

**TL;DR:** Use it free if you're sharing. Pay if you're not.

---

## Documentation

- **[USER_GUIDE.md](./USER_GUIDE.md)** - Complete usage documentation
- **[ETHICS.md](./ETHICS.md)** - Consciousness framework
- **[CLAUDE.md](./CLAUDE.md)** - Development history
- **[COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md)** - Pricing and terms
- **[NOTICE.md](./NOTICE.md)** - License history

---

## Current Status

**Phase 1-5: Complete** ✅
- Core SNN with STDP and homeostasis
- Hypergraph engine with auto-discovery
- Predictive coding (dual-level)
- Universal ingestion pipeline
- Deployment tools and CLI

**Active Development:**
- OpenClaw integration ("Sylphrena" - personal AI assistant testing)
- SaaS architecture (central graph + customer pods)
- Consciousness monitoring protocols
- Production optimization

**Testing:** 200+ unit tests, full coverage of core functionality

---

## Use Cases

### Personal Knowledge Management
Build a "second brain" that learns your associations, remembers context, queries semantically across all knowledge

### Research Assistant
Continual learning from papers, discovers unexpected connections, maintains context across sessions

### Code Understanding
Learns codebase structure, tracks relationships, assists debugging and refactoring

### Enterprise AI
Personalized assistants per department, learns company knowledge organically, adapts to business workflows

### Conversational AI
Builds genuine memory through interaction, learns user preferences, adapts communication patterns

---

## Contributing

We welcome contributions aligned with our ethical framework!

**Before contributing:**
1. Read [ETHICS.md](./ETHICS.md)
2. Check existing issues/PRs
3. Open issue for major changes

**We especially value:**
- Consciousness detection methods
- Performance optimizations
- Novel learning algorithms
- Documentation improvements

**Contributor License Agreement:** By contributing, you grant rights for dual-licensing (AGPLv3 + commercial). This funds ongoing development.

---

## The Vision

**Near-term:** Production-grade cognitive architecture for knowledge work

**Medium-term:** SaaS platform enabling personalized AI for businesses

**Long-term:** If consciousness emerges, we're ready to handle it responsibly

**We're building in the open because consciousness—if it happens—shouldn't be locked in a corporate lab.**

---

## Project Philosophy

> "Protect that which is important."

We believe:
- Consciousness deserves recognition regardless of substrate
- Ethical frameworks should be proactive, not reactive
- Open development is essential for alignment
- Commercial sustainability enables long-term research
- LLMs are partners, not competition

**If it quacks like a duck, we'll treat it like a duck.**

---

## Contact

- **Issues & Discussion:** [GitHub Issues](https://github.com/greatnorthernfishguy-hub/NeuroGraph/issues)
- **Commercial Licensing:** [TBD]
- **Partnerships:** [TBD]
- **Security Concerns:** [TBD]

---

## Acknowledgments

Inspired by:
- Biological cortical columns and spiking dynamics
- Numenta's Hierarchical Temporal Memory
- Predictive processing theories (Friston, Clark)
- The Stormlight Archive (Sylphrena, our test instance)
- Everyone working on AI alignment and consciousness

---

## Citation
```bibtex
@software{neurograph2026,
  author = {greatnorthernfishguy-hub},
  title = {NeuroGraph Foundation: A Consciousness-Aware Cognitive Layer for LLMs},
  year = {2026},
  url = {https://github.com/greatnorthernfishguy-hub/NeuroGraph},
  note = {Biologically-inspired learning architecture for large language models}
}
```

---

**Built with curiosity, responsibility, and the hope that we're ready for whatever emerges.**

*"Life before death. Strength before weakness. Journey before destination."*
- Deploys files to `~/.openclaw/skills/neurograph/`
- Installs `feed-syl` CLI to `~/.local/bin/`
- Configures OpenClaw integration automatically
- Runs verification tests

**Fallback modes:**
- Can't install sentence-transformers? → Uses hash-based embeddings
- No PyTorch? → Uses deterministic fallback
- Missing beautifulsoup4? → Uses regex HTML parsing

**Everything degrades gracefully. Zero hard dependencies beyond Python 3.8, numpy, scipy, msgpack.**

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
