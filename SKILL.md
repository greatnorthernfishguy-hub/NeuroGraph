---
name: neurograph
description: "NeuroGraph is a cognitive memory system that remembers context across conversations using a spiking neural network substrate. Invoke when the user asks about memory, remember, context, what was learned, what the system knows, Syl, graph associations, knowledge, or wants to ingest or feed content into the knowledge base."
metadata: { "openclaw": { "emoji": "🧠", "requires": {} } }
hook: openclaw_hook.py::get_instance
---

# NeuroGraph Cognitive Memory

## Metadata
name: neurograph
version: 0.4.0
autoload: true

## Description
NeuroGraph provides dynamic cognitive memory for the OpenClaw AI assistant.
It combines three memory layers:
- **Semantic Memory** (Vector DB) — content similarity via embeddings
- **Structural Memory** (Hypergraph) — N-ary concept relationships
- **Temporal Memory** (Spiking Neural Network) — causal learning via STDP

## Environment Variables
- `NEUROGRAPH_WORKSPACE_DIR` — Working directory for checkpoints and data (default: `~/.openclaw/neurograph`)

## Capabilities
- Automatic content ingestion (text, code, markdown, URLs, PDFs)
- STDP-based causal learning across conversations
- Semantic similarity recall
- Prediction generation and surprise-driven exploration
- Cross-session persistence via checkpoints

## Integration
NeuroGraph auto-loads as an OpenClaw skill. On each message:
1. Content is ingested through the 5-stage pipeline
2. The SNN runs one learning step (STDP + homeostatic + structural plasticity)
3. Predictions are evaluated and surprise exploration triggered
4. State auto-saves every 10 messages

## CLI
Use `feed-syl` for manual ingestion and status:
```bash
feed-syl --status              # Show graph statistics
feed-syl --text "content"      # Ingest text
feed-syl --file path/to/file   # Ingest a file
feed-syl --workspace           # Ingest OpenClaw workspace docs
feed-syl --query "search term" # Semantic search
```

## Files
- `neuro_foundation.py` — Core SNN engine with STDP, hypergraph, predictions
- `universal_ingestor.py` — 5-stage ingestion pipeline
- `openclaw_hook.py` — OpenClaw integration singleton
