"""Example: Ingest Python code and show learned structure.

Demonstrates the Universal Ingestor with code-aware chunking (OpenClaw config),
showing how function/class definitions become nodes and how structural
relationships (sequential, definition→usage) become synapses.
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


# Sample Python code to ingest
SAMPLE_CODE = '''\
import math
from typing import List

class Vector:
    """A simple 2D vector class."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y


def compute_angle(v1: Vector, v2: Vector) -> float:
    """Compute angle between two vectors in radians."""
    cos_theta = v1.dot(v2) / (v1.magnitude() * v2.magnitude())
    return math.acos(max(-1, min(1, cos_theta)))


def batch_normalize(vectors: List[Vector]) -> List[Vector]:
    """Normalize a list of vectors."""
    return [v.normalize() for v in vectors]
'''


def main():
    # Create graph and vector DB
    graph = Graph()
    vector_db = SimpleVectorDB()

    # Configure for code ingestion (OpenClaw-style)
    config = {
        "chunking": {
            "strategy": ChunkStrategy.CODE_AWARE,
        },
        "embedding": {"use_model": False, "dimension": 64},
        "registration": {
            "novelty_dampening": 0.3,
            "probation_period": 10,
        },
        "association": {
            "similarity_threshold": 0.5,
            "sequential_weight": 0.3,
        },
    }

    ingestor = UniversalIngestor(graph, vector_db, config)

    # Ingest the code
    print("Ingesting Python code...")
    result = ingestor.ingest(SAMPLE_CODE, source_type=SourceType.CODE)

    print(f"\n=== Ingestion Result ===")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Nodes created:  {len(result.nodes_created)}")
    print(f"  Synapses:       {len(result.synapses_created)}")
    print(f"  Hyperedges:     {len(result.hyperedges_created)}")

    # Show the created nodes
    print(f"\n=== Created Nodes ===")
    for nid in result.nodes_created:
        node = graph.nodes[nid]
        preview = node.metadata.get("chunk_text_preview", "")[:60]
        chunk_type = node.metadata.get("type", "unknown")
        name = node.metadata.get("name", "")
        dampening = node.intrinsic_excitability
        print(f"  [{chunk_type}] {name or 'preamble'} "
              f"(dampening={dampening:.2f}): {preview}...")

    # Show synapses and their types
    print(f"\n=== Created Synapses ===")
    for sid in result.synapses_created:
        syn = graph.synapses[sid]
        mode = syn.metadata.get("creation_mode", "unknown")
        print(f"  {syn.pre_node_id[:8]}... → {syn.post_node_id[:8]}... "
              f"[{mode}] weight={syn.weight:.3f}")

    # Run a few simulation steps to show probation progression
    print(f"\n=== Probation Progression ===")
    sample_node = graph.nodes[result.nodes_created[0]]
    for step in range(10):
        graduated = ingestor.update_probation()
        if graduated:
            print(f"  Step {step + 1}: {len(graduated)} node(s) graduated")
    print(f"  Final excitability: {sample_node.intrinsic_excitability:.2f}")

    # Query similar content
    print(f"\n=== Similarity Query ===")
    query = "vector normalization"
    similar = ingestor.query_similar(query, k=3, threshold=0.0)
    print(f"  Query: '{query}'")
    for r in similar:
        print(f"  sim={r['similarity']:.3f}: {r['content'][:60]}...")


if __name__ == "__main__":
    main()
