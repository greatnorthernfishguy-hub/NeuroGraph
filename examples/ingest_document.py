"""Example: Ingest a markdown document and show concept extraction.

Demonstrates hierarchical chunking (DSM-style config) where heading
structure is preserved and parent-child relationships become synapses.
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


# Sample clinical/academic document
SAMPLE_DOCUMENT = """\
# Generalized Anxiety Disorder

Generalized Anxiety Disorder (GAD) is characterized by persistent and
excessive worry about a variety of different things. People with GAD may
anticipate disaster and may be overly concerned about money, health,
family, work, or other issues.

## Diagnostic Criteria

The following criteria must be met for a diagnosis of GAD:

Excessive anxiety and worry occurring more days than not for at least
6 months about a number of events or activities.

The individual finds it difficult to control the worry. The anxiety and
worry are associated with three or more of the following symptoms.

## Symptoms

### Psychological Symptoms

Restlessness or feeling keyed up or on edge. Being easily fatigued.
Difficulty concentrating or mind going blank. Irritability.

### Physical Symptoms

Muscle tension and sleep disturbance (difficulty falling or staying asleep,
or restless unsatisfying sleep).

## Treatment Approaches

### Cognitive Behavioral Therapy

CBT is the gold standard treatment for GAD. It involves identifying and
challenging anxious thoughts, developing coping strategies, and gradual
exposure to anxiety-provoking situations.

### Pharmacotherapy

SSRIs and SNRIs are first-line pharmacological treatments. Buspirone may
also be effective. Benzodiazepines should be used cautiously due to
dependence risk.

## Comorbidity

GAD frequently co-occurs with major depressive disorder, other anxiety
disorders, and substance use disorders.
"""


def main():
    graph = Graph({"default_threshold": 1.2})
    vector_db = SimpleVectorDB()

    # DSM-style hierarchical chunking
    config = {
        "chunking": {
            "strategy": ChunkStrategy.HIERARCHICAL,
        },
        "embedding": {"use_model": False, "dimension": 64},
        "registration": {
            "novelty_dampening": 0.05,
            "probation_period": 100,
        },
        "association": {
            "similarity_threshold": 0.3,
            "sequential_weight": 0.2,
            "parent_child_weight": 0.6,
            "min_cluster_size": 3,
            "cluster_similarity_threshold": 0.4,
        },
    }

    ingestor = UniversalIngestor(graph, vector_db, config)

    print("Ingesting clinical document...")
    result = ingestor.ingest(SAMPLE_DOCUMENT, source_type=SourceType.MARKDOWN)

    print(f"\n=== Ingestion Result ===")
    print(f"  Source type:    {result.source_type.name}")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Nodes created:  {len(result.nodes_created)}")
    print(f"  Synapses:       {len(result.synapses_created)}")
    print(f"  Hyperedges:     {len(result.hyperedges_created)}")

    # Show hierarchical structure
    print(f"\n=== Document Structure (Nodes) ===")
    for nid in result.nodes_created:
        node = graph.nodes[nid]
        heading = node.metadata.get("heading", "(no heading)")
        level = node.metadata.get("level", "-")
        parent = node.metadata.get("parent_chunk_id", None)
        indent = "  " * (int(level) if isinstance(level, int) else 0)
        excit = node.intrinsic_excitability
        print(f"  {indent}[L{level}] {heading} (excitability={excit:.3f})")

    # Show relationships
    print(f"\n=== Relationships ===")
    creation_modes = {}
    for sid in result.synapses_created:
        syn = graph.synapses[sid]
        mode = syn.metadata.get("creation_mode", "unknown")
        creation_modes[mode] = creation_modes.get(mode, 0) + 1

    for mode, count in sorted(creation_modes.items()):
        print(f"  {mode}: {count} synapses")

    # Show hyperedges (clusters)
    if result.hyperedges_created:
        print(f"\n=== Concept Clusters (Hyperedges) ===")
        for hid in result.hyperedges_created:
            he = graph.hyperedges[hid]
            print(f"  Cluster: {len(he.member_nodes)} members")

    # Query for related concepts
    print(f"\n=== Concept Queries ===")
    for query in ["anxiety symptoms", "treatment medication", "diagnosis criteria"]:
        results = ingestor.query_similar(query, k=2, threshold=0.0)
        print(f"  '{query}':")
        for r in results:
            heading = r["metadata"].get("heading", "?")
            print(f"    sim={r['similarity']:.3f} → {heading}")


if __name__ == "__main__":
    main()
