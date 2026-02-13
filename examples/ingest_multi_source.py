"""Example: Ingest diverse sources and show cross-domain integration.

Demonstrates the Universal Ingestor consuming multiple source types
(code, markdown, plain text) and building an integrated knowledge graph
with cross-domain associations and novelty dampening.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph
from universal_ingestor import (
    SimpleVectorDB,
    UniversalIngestor,
    SourceType,
    ChunkStrategy,
)


# Three different sources to integrate
CODE_SOURCE = """\
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []

    def forward(self, inputs):
        activation = inputs
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backpropagate(self, loss):
        gradient = loss
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
"""

MARKDOWN_SOURCE = """\
# Neural Network Architecture

Neural networks consist of layers of interconnected nodes. Each connection
has a weight that is adjusted during training.

## Forward Propagation

Data flows forward through the network, with each layer transforming the
input through weighted sums and activation functions.

## Backpropagation

Errors are propagated backward through the network to update weights.
The gradient of the loss function determines how each weight should change.
"""

TEXT_SOURCE = """\
Deep learning has revolutionized artificial intelligence. The key insight
is that multiple layers of representation allow networks to learn
hierarchical features from raw data.

Training deep networks requires large datasets and significant compute.
Techniques like batch normalization and skip connections help with
training stability and gradient flow.
"""


def main():
    graph = Graph()
    vector_db = SimpleVectorDB()

    # Consciousness-style config for cross-domain exploration
    config = {
        "chunking": {
            "strategy": ChunkStrategy.SEMANTIC,
            "max_chunk_tokens": 100,
        },
        "embedding": {"use_model": False, "dimension": 64},
        "registration": {
            "novelty_dampening": 0.1,
            "probation_period": 20,
        },
        "association": {
            "similarity_threshold": 0.3,
            "sequential_weight": 0.25,
            "parent_child_weight": 0.4,
            "min_cluster_size": 3,
            "cluster_similarity_threshold": 0.3,
        },
    }

    ingestor = UniversalIngestor(graph, vector_db, config)

    # Ingest all three sources
    print("=== Ingesting Code ===")
    r1 = ingestor.ingest(CODE_SOURCE, source_type=SourceType.CODE)
    print(f"  Chunks: {r1.chunks_created}, Nodes: {len(r1.nodes_created)}, "
          f"Synapses: {len(r1.synapses_created)}")

    print("\n=== Ingesting Markdown ===")
    r2 = ingestor.ingest(MARKDOWN_SOURCE, source_type=SourceType.MARKDOWN)
    print(f"  Chunks: {r2.chunks_created}, Nodes: {len(r2.nodes_created)}, "
          f"Synapses: {len(r2.synapses_created)}")

    print("\n=== Ingesting Plain Text ===")
    r3 = ingestor.ingest(TEXT_SOURCE)
    print(f"  Chunks: {r3.chunks_created}, Nodes: {len(r3.nodes_created)}, "
          f"Synapses: {len(r3.synapses_created)}")

    # Overall statistics
    total_nodes = len(graph.nodes)
    total_synapses = len(graph.synapses)
    total_hyperedges = len(graph.hyperedges)
    print(f"\n=== Integrated Graph ===")
    print(f"  Total nodes:      {total_nodes}")
    print(f"  Total synapses:   {total_synapses}")
    print(f"  Total hyperedges: {total_hyperedges}")
    print(f"  Vector DB entries: {vector_db.count()}")

    # Analyze synapse types
    print(f"\n=== Relationship Types ===")
    modes = {}
    for syn in graph.synapses.values():
        mode = syn.metadata.get("creation_mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1
    for mode, count in sorted(modes.items()):
        print(f"  {mode}: {count}")

    # Show cross-domain connections
    print(f"\n=== Cross-Domain Query ===")
    for query in ["neural network training", "gradient computation", "layer architecture"]:
        results = ingestor.query_similar(query, k=3, threshold=0.0)
        print(f"  '{query}':")
        for r in results:
            src_type = r["metadata"].get("source_type", "?")
            preview = r["content"][:50].replace("\n", " ")
            print(f"    [{src_type}] sim={r['similarity']:.3f}: {preview}...")

    # Simulate probation advancement
    print(f"\n=== Probation Simulation ===")
    for step in range(20):
        graduated = ingestor.update_probation()
        if graduated:
            print(f"  Step {step + 1}: {len(graduated)} node(s) graduated to full integration")

    # Show final node excitability distribution
    excitabilities = [n.intrinsic_excitability for n in graph.nodes.values()]
    print(f"\n=== Final Excitability Distribution ===")
    print(f"  Min: {min(excitabilities):.3f}")
    print(f"  Max: {max(excitabilities):.3f}")
    print(f"  Mean: {sum(excitabilities)/len(excitabilities):.3f}")
    print(f"  Graduated: {sum(1 for e in excitabilities if e >= 1.0)}/{len(excitabilities)}")

    # Show ingestion log
    print(f"\n=== Ingestion Log ===")
    for i, log in enumerate(ingestor.ingestion_log):
        print(f"  [{i+1}] {log.source_type.name}: {log.chunks_created} chunks, "
              f"{len(log.nodes_created)} nodes")


if __name__ == "__main__":
    main()
