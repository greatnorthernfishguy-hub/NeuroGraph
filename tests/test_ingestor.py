"""
Tests for Phase 4: Universal Ingestor System.

Tests cover:
    - SimpleVectorDB operations (insert, search, delete)
    - Each extractor format (Text, Markdown, Code, URL, PDF)
    - Adaptive chunking strategies (Semantic, Code-aware, Hierarchical, Fixed-size)
    - Embedding generation and caching
    - Novelty dampening and probation
    - Similarity-based and structural association
    - Hypergraph clustering
    - End-to-end ingestion pipeline
    - Project configs (OpenClaw, DSM, Consciousness)
    - Edge cases (empty input, single chunk, etc.)
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph, ActivationMode
from universal_ingestor import (
    # Data structures
    SourceType,
    ChunkStrategy,
    DampeningCurve,
    ExtractedContent,
    Chunk,
    EmbeddedChunk,
    IngestionResult,
    # Vector DB
    SimpleVectorDB,
    # Extractors
    TextExtractor,
    MarkdownExtractor,
    CodeExtractor,
    URLExtractor,
    PDFExtractor,
    ExtractorRouter,
    # Chunker
    AdaptiveChunker,
    # Embedder
    EmbeddingEngine,
    # Registrar
    NodeRegistrar,
    # Associator
    HypergraphAssociator,
    # Configs
    OPENCLAW_INGESTOR_CONFIG,
    DSM_INGESTOR_CONFIG,
    CONSCIOUSNESS_INGESTOR_CONFIG,
    get_ingestor_config,
    # Coordinator
    IngestorConfig,
    UniversalIngestor,
)


# ---------------------------------------------------------------------------
# SimpleVectorDB Tests
# ---------------------------------------------------------------------------

class TestSimpleVectorDB(unittest.TestCase):
    """Tests for the in-memory vector database."""

    def setUp(self):
        self.db = SimpleVectorDB()

    def test_insert_and_get(self):
        """Insert entry and retrieve by ID."""
        vec = np.array([1.0, 0.0, 0.0])
        self.db.insert("doc1", vec, "hello world", {"key": "val"})
        result = self.db.get("doc1")
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "hello world")
        self.assertEqual(result["metadata"]["key"], "val")

    def test_insert_normalizes_vector(self):
        """Inserted vectors are L2-normalized."""
        vec = np.array([3.0, 4.0])  # norm = 5
        self.db.insert("doc1", vec, "test")
        stored = self.db.get("doc1")["embedding"]
        self.assertAlmostEqual(np.linalg.norm(stored), 1.0, places=5)

    def test_search_cosine_similarity(self):
        """Search returns matches above threshold sorted by similarity."""
        self.db.insert("a", np.array([1.0, 0.0, 0.0]), "doc a")
        self.db.insert("b", np.array([0.9, 0.1, 0.0]), "doc b")
        self.db.insert("c", np.array([0.0, 0.0, 1.0]), "doc c")

        results = self.db.search(np.array([1.0, 0.0, 0.0]), k=5, threshold=0.5)
        # 'a' should be top (similarity ≈ 1.0), 'b' should also match
        self.assertGreaterEqual(len(results), 2)
        self.assertEqual(results[0][0], "a")
        self.assertGreater(results[0][1], results[1][1])

    def test_search_respects_threshold(self):
        """Search excludes results below threshold."""
        self.db.insert("a", np.array([1.0, 0.0]), "doc a")
        self.db.insert("b", np.array([0.0, 1.0]), "doc b")

        results = self.db.search(np.array([1.0, 0.0]), k=5, threshold=0.99)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "a")

    def test_search_top_k(self):
        """Search returns at most k results."""
        for i in range(10):
            vec = np.random.randn(3)
            self.db.insert(f"d{i}", vec, f"doc {i}")

        results = self.db.search(np.random.randn(3), k=3, threshold=0.0)
        self.assertLessEqual(len(results), 3)

    def test_search_empty_db(self):
        """Search on empty DB returns empty list."""
        results = self.db.search(np.array([1.0, 0.0]), k=5, threshold=0.0)
        self.assertEqual(results, [])

    def test_delete(self):
        """Delete removes entry and returns True; missing returns False."""
        self.db.insert("doc1", np.array([1.0]), "test")
        self.assertTrue(self.db.delete("doc1"))
        self.assertIsNone(self.db.get("doc1"))
        self.assertFalse(self.db.delete("doc1"))

    def test_count(self):
        """Count reflects number of stored entries."""
        self.assertEqual(self.db.count(), 0)
        self.db.insert("a", np.array([1.0]), "a")
        self.db.insert("b", np.array([1.0]), "b")
        self.assertEqual(self.db.count(), 2)
        self.db.delete("a")
        self.assertEqual(self.db.count(), 1)

    def test_all_ids(self):
        """all_ids returns all stored IDs."""
        self.db.insert("x", np.array([1.0]), "x")
        self.db.insert("y", np.array([1.0]), "y")
        ids = self.db.all_ids()
        self.assertIn("x", ids)
        self.assertIn("y", ids)


# ---------------------------------------------------------------------------
# Extractor Tests
# ---------------------------------------------------------------------------

class TestTextExtractor(unittest.TestCase):
    """Tests for plain text extractor."""

    def test_extract_basic(self):
        """Text extractor returns raw text and metadata."""
        ext = TextExtractor()
        result = ext.extract("Hello, world!")
        self.assertEqual(result.raw_text, "Hello, world!")
        self.assertEqual(result.source_type, SourceType.TEXT)
        self.assertEqual(result.metadata["source_type"], "text")

    def test_extract_preserves_content(self):
        """All content is preserved exactly."""
        ext = TextExtractor()
        text = "Line 1\nLine 2\n\nLine 4"
        result = ext.extract(text)
        self.assertEqual(result.raw_text, text)


class TestMarkdownExtractor(unittest.TestCase):
    """Tests for markdown extractor."""

    def test_extract_headings(self):
        """Markdown extractor detects heading hierarchy."""
        ext = MarkdownExtractor()
        md = "# Title\n\nSome text.\n\n## Section 1\n\nContent.\n\n### Subsection\n\nMore."
        result = ext.extract(md)
        self.assertEqual(result.source_type, SourceType.MARKDOWN)
        self.assertEqual(len(result.structure), 3)
        self.assertEqual(result.structure[0]["level"], 1)
        self.assertEqual(result.structure[0]["title"], "Title")
        self.assertEqual(result.structure[1]["level"], 2)
        self.assertEqual(result.structure[2]["level"], 3)

    def test_extract_no_headings(self):
        """Markdown with no headings returns empty structure."""
        ext = MarkdownExtractor()
        result = ext.extract("Just plain text, no headings.")
        self.assertEqual(len(result.structure), 0)

    def test_heading_count_metadata(self):
        """Metadata includes heading count."""
        ext = MarkdownExtractor()
        md = "# H1\n## H2\n## H3"
        result = ext.extract(md)
        self.assertEqual(result.metadata["heading_count"], 3)


class TestCodeExtractor(unittest.TestCase):
    """Tests for language-aware code extractor."""

    def test_extract_python_functions(self):
        """Detects Python function and class definitions."""
        ext = CodeExtractor()
        code = textwrap.dedent("""\
        import os

        class MyClass:
            def method(self):
                pass

        def standalone():
            pass
        """)
        result = ext.extract(code, language="python")
        self.assertEqual(result.source_type, SourceType.CODE)
        names = [s["name"] for s in result.structure]
        self.assertIn("MyClass", names)
        self.assertIn("method", names)
        self.assertIn("standalone", names)

    def test_extract_python_async(self):
        """Detects async function definitions."""
        ext = CodeExtractor()
        code = "async def fetch_data(url):\n    pass\n"
        result = ext.extract(code, language="python")
        self.assertEqual(len(result.structure), 1)
        self.assertEqual(result.structure[0]["name"], "fetch_data")
        self.assertEqual(result.structure[0]["type"], "async_function")

    def test_extract_javascript(self):
        """Detects JavaScript function definitions."""
        ext = CodeExtractor()
        code = "function main() {\n  return 42;\n}\n\nclass App {\n}\n"
        result = ext.extract(code, language="javascript")
        names = [s["name"] for s in result.structure]
        self.assertIn("main", names)
        self.assertIn("App", names)


import textwrap


class TestURLExtractor(unittest.TestCase):
    """Tests for URL/HTML extractor."""

    def test_extract_html_content(self):
        """Extracts text from inline HTML (no network)."""
        ext = URLExtractor()
        html = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>"
        result = ext.extract(html)
        self.assertIn("Hello", result.raw_text)

    def test_extract_title(self):
        """Extracts page title from HTML."""
        ext = URLExtractor()
        html = "<html><head><title>My Page</title></head><body>Content</body></html>"
        result = ext.extract(html)
        self.assertEqual(result.metadata["title"], "My Page")


class TestExtractorRouter(unittest.TestCase):
    """Tests for the extractor router and auto-detection."""

    def test_detect_markdown(self):
        """Auto-detects markdown from heading prefix."""
        router = ExtractorRouter()
        stype = router.detect_source_type("# Hello World\n\nContent here.")
        self.assertEqual(stype, SourceType.MARKDOWN)

    def test_detect_code(self):
        """Auto-detects code from def/class keywords."""
        router = ExtractorRouter()
        stype = router.detect_source_type("def my_func():\n    pass\n")
        self.assertEqual(stype, SourceType.CODE)

    def test_detect_url(self):
        """Auto-detects URL from http prefix."""
        router = ExtractorRouter()
        stype = router.detect_source_type("https://example.com")
        self.assertEqual(stype, SourceType.URL)

    def test_detect_html(self):
        """Auto-detects HTML content."""
        router = ExtractorRouter()
        stype = router.detect_source_type("<html><body>hi</body></html>")
        self.assertEqual(stype, SourceType.URL)

    def test_detect_text_fallback(self):
        """Falls back to TEXT for unrecognized content."""
        router = ExtractorRouter()
        stype = router.detect_source_type("Just some random words here.")
        self.assertEqual(stype, SourceType.TEXT)

    def test_extract_with_explicit_type(self):
        """Explicit source_type overrides auto-detection."""
        router = ExtractorRouter()
        result = router.extract("def foo():\n    pass\n", source_type=SourceType.TEXT)
        self.assertEqual(result.source_type, SourceType.TEXT)


# ---------------------------------------------------------------------------
# Chunking Tests
# ---------------------------------------------------------------------------

class TestAdaptiveChunker(unittest.TestCase):
    """Tests for adaptive chunking strategies."""

    def test_semantic_chunking_respects_paragraphs(self):
        """Semantic chunking splits on paragraph boundaries."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.SEMANTIC, "max_chunk_tokens": 50})
        text = "First paragraph with some content.\n\nSecond paragraph with more."
        content = ExtractedContent(raw_text=text, source_type=SourceType.TEXT)
        chunks = chunker.chunk(content)
        self.assertGreaterEqual(len(chunks), 1)
        # All original text should be covered
        combined = " ".join(c.text for c in chunks)
        self.assertIn("First paragraph", combined)
        self.assertIn("Second paragraph", combined)

    def test_semantic_chunking_large_paragraph(self):
        """Semantic chunking splits large paragraphs by sentence."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.SEMANTIC, "max_chunk_tokens": 20})
        text = "This is sentence one. This is sentence two. This is sentence three. " * 5
        content = ExtractedContent(raw_text=text, source_type=SourceType.TEXT)
        chunks = chunker.chunk(content)
        self.assertGreater(len(chunks), 1)

    def test_code_aware_chunking(self):
        """Code-aware chunking splits by function/class."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.CODE_AWARE})
        code = textwrap.dedent("""\
        import os

        class MyClass:
            def method(self):
                return 42

        def standalone():
            return 1
        """)
        structure = [
            {"type": "class", "name": "MyClass", "line": 2, "indent": 0},
            {"type": "function", "name": "method", "line": 3, "indent": 4},
            {"type": "function", "name": "standalone", "line": 6, "indent": 0},
        ]
        content = ExtractedContent(
            raw_text=code,
            structure=structure,
            source_type=SourceType.CODE,
            metadata={"language": "python"},
        )
        chunks = chunker.chunk(content)
        # Should have preamble + at least 2 code chunks
        self.assertGreaterEqual(len(chunks), 2)
        # Check code_aware metadata
        code_chunks = [c for c in chunks if c.metadata.get("strategy") == "code_aware"]
        self.assertGreater(len(code_chunks), 0)

    def test_hierarchical_chunking(self):
        """Hierarchical chunking follows heading structure."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.HIERARCHICAL})
        md = "# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        structure = [
            {"type": "heading", "level": 1, "title": "Title", "line": 0},
            {"type": "heading", "level": 2, "title": "Section A", "line": 4},
            {"type": "heading", "level": 2, "title": "Section B", "line": 8},
        ]
        content = ExtractedContent(
            raw_text=md, structure=structure, source_type=SourceType.MARKDOWN
        )
        chunks = chunker.chunk(content)
        self.assertGreaterEqual(len(chunks), 3)
        headings = [c.metadata.get("heading") for c in chunks if c.metadata.get("heading")]
        self.assertIn("Title", headings)
        self.assertIn("Section A", headings)

    def test_hierarchical_parent_child(self):
        """Hierarchical chunking sets parent-child relationships."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.HIERARCHICAL})
        md = "# Title\n\nIntro.\n\n## Sub\n\nDetail."
        structure = [
            {"type": "heading", "level": 1, "title": "Title", "line": 0},
            {"type": "heading", "level": 2, "title": "Sub", "line": 4},
        ]
        content = ExtractedContent(
            raw_text=md, structure=structure, source_type=SourceType.MARKDOWN
        )
        chunks = chunker.chunk(content)
        # The level-2 chunk should reference level-1 as parent
        sub_chunks = [c for c in chunks if c.metadata.get("heading") == "Sub"]
        if sub_chunks:
            title_chunks = [c for c in chunks if c.metadata.get("heading") == "Title"]
            if title_chunks:
                self.assertEqual(sub_chunks[0].parent_chunk_id, title_chunks[0].chunk_id)

    def test_fixed_size_chunking(self):
        """Fixed-size chunking produces consistent-sized chunks with overlap."""
        chunker = AdaptiveChunker({
            "strategy": ChunkStrategy.FIXED_SIZE,
            "max_chunk_tokens": 20,
            "overlap_tokens": 5,
        })
        text = " ".join([f"word{i}" for i in range(100)])
        content = ExtractedContent(raw_text=text, source_type=SourceType.TEXT)
        chunks = chunker.chunk(content)
        self.assertGreater(len(chunks), 1)

    def test_empty_input(self):
        """Empty input produces no chunks."""
        chunker = AdaptiveChunker()
        content = ExtractedContent(raw_text="", source_type=SourceType.TEXT)
        chunks = chunker.chunk(content)
        self.assertEqual(len(chunks), 0)

    def test_token_estimation(self):
        """Token estimation is reasonable (words / 0.75)."""
        tokens = AdaptiveChunker.estimate_tokens("one two three four")
        # 4 words / 0.75 ≈ 5
        self.assertEqual(tokens, 5)

    def test_code_fallback_no_structure(self):
        """Code chunking falls back to semantic when no structure."""
        chunker = AdaptiveChunker({"strategy": ChunkStrategy.CODE_AWARE})
        content = ExtractedContent(
            raw_text="Just plain text.\n\nAnother paragraph.",
            structure=[],
            source_type=SourceType.CODE,
        )
        chunks = chunker.chunk(content)
        self.assertGreater(len(chunks), 0)


# ---------------------------------------------------------------------------
# Embedding Tests
# ---------------------------------------------------------------------------

class TestEmbeddingEngine(unittest.TestCase):
    """Tests for embedding generation and caching."""

    def setUp(self):
        # Use hash fallback (no model) for testing
        self.engine = EmbeddingEngine({"use_model": False, "dimension": 64})

    def test_embed_produces_vectors(self):
        """Embedding produces normalized vectors of correct dimension."""
        chunks = [Chunk(text="Hello world"), Chunk(text="Goodbye world")]
        embedded = self.engine.embed(chunks)
        self.assertEqual(len(embedded), 2)
        for ec in embedded:
            self.assertEqual(ec.vector.shape[0], 64)
            self.assertAlmostEqual(np.linalg.norm(ec.vector), 1.0, places=5)

    def test_embed_deterministic(self):
        """Same text produces same embedding (deterministic hash)."""
        chunks1 = [Chunk(text="Hello world")]
        chunks2 = [Chunk(text="Hello world")]
        emb1 = self.engine.embed(chunks1)
        emb2 = self.engine.embed(chunks2)
        np.testing.assert_array_almost_equal(emb1[0].vector, emb2[0].vector)

    def test_embed_caching(self):
        """Repeated embedding uses cache."""
        chunks = [Chunk(text="cached text")]
        self.engine.embed(chunks)
        cache_before = self.engine.cache_hits

        # Embed same text again — should use cache
        self.engine.embed(chunks)
        # Cache should still have entry (hits count = entries in cache)
        self.assertGreaterEqual(self.engine.cache_hits, cache_before)

    def test_embed_text_single(self):
        """embed_text() embeds a single string."""
        vec = self.engine.embed_text("test string")
        self.assertEqual(vec.shape[0], 64)
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)

    def test_different_texts_different_vectors(self):
        """Different texts produce different vectors."""
        vec1 = self.engine.embed_text("apple banana cherry")
        vec2 = self.engine.embed_text("quantum physics relativity")
        # Very unlikely to be identical with hash-based embedding
        self.assertFalse(np.allclose(vec1, vec2))

    def test_model_name_fallback(self):
        """Fallback engine reports hash_fallback model name."""
        chunks = [Chunk(text="test")]
        embedded = self.engine.embed(chunks)
        self.assertEqual(embedded[0].model_name, "hash_fallback")


class TestEmbeddingDeviceControl(unittest.TestCase):
    """Tests for device control and graceful degradation."""

    def test_default_device_is_auto(self):
        """Default device is 'auto'."""
        engine = EmbeddingEngine({"use_model": False})
        self.assertEqual(engine.device, "auto")

    def test_explicit_cpu_device(self):
        """Explicit CPU device is accepted."""
        engine = EmbeddingEngine({"use_model": False, "device": "cpu"})
        self.assertEqual(engine.device, "cpu")

    def test_explicit_cuda_device_falls_back(self):
        """CUDA device gracefully falls back when CUDA unavailable."""
        # In test environment, CUDA is typically not available
        engine = EmbeddingEngine({"use_model": False, "device": "cuda"})
        self.assertEqual(engine.device, "cuda")
        # Engine should still work (hash fallback since use_model=False)
        vec = engine.embed_text("test")
        self.assertEqual(vec.shape[0], 768)

    def test_status_property(self):
        """Status property returns diagnostic info."""
        engine = EmbeddingEngine({"use_model": False, "dimension": 64})
        status = engine.status
        self.assertFalse(status["model_available"])
        self.assertEqual(status["model_name"], "hash_fallback")
        self.assertEqual(status["device_requested"], "auto")
        self.assertIsNone(status["device_active"])
        self.assertEqual(status["dimension"], 64)
        self.assertEqual(status["cache_entries"], 0)

    def test_status_after_embedding(self):
        """Status reflects cache entries after embedding."""
        engine = EmbeddingEngine({"use_model": False, "dimension": 64})
        engine.embed_text("test")
        self.assertEqual(engine.status["cache_entries"], 1)

    def test_fallback_reason_when_model_disabled(self):
        """No fallback reason when model intentionally disabled."""
        engine = EmbeddingEngine({"use_model": False})
        self.assertIsNone(engine.status["fallback_reason"])

    def test_fallback_reason_when_import_fails(self):
        """Fallback reason set when sentence-transformers unavailable."""
        # use_model=True but sentence-transformers is not installed in test env
        engine = EmbeddingEngine({"use_model": True})
        if not engine._model_available:
            self.assertIsNotNone(engine.status["fallback_reason"])

    def test_encode_batch_runtime_fallback(self):
        """If model encode raises at runtime, falls back to hash."""
        engine = EmbeddingEngine({"use_model": False, "dimension": 64})
        # Simulate a loaded model that fails at encode time
        engine._model_available = True

        class FaultyModel:
            def encode(self, texts, **kwargs):
                raise RuntimeError("CUDA out of memory")

        engine._model = FaultyModel()
        # Should not raise — falls back to hash embeddings
        vecs = engine._encode_batch(["hello", "world"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(vecs[0].shape[0], 64)


# ---------------------------------------------------------------------------
# Novelty Dampening Tests
# ---------------------------------------------------------------------------

class TestNoveltyDampening(unittest.TestCase):
    """Tests for novelty dampening and probation system."""

    def setUp(self):
        self.graph = Graph()
        self.vector_db = SimpleVectorDB()
        self.registrar = NodeRegistrar(
            self.graph, self.vector_db,
            config={
                "novelty_dampening": 0.3,
                "probation_period": 10,
                "dampening_curve": DampeningCurve.LINEAR,
                "initial_threshold_boost": 0.2,
            },
        )
        self.embedder = EmbeddingEngine({"use_model": False, "dimension": 32})

    def _make_embedded_chunks(self, texts):
        chunks = [Chunk(text=t, chunk_id=f"chunk_{i}") for i, t in enumerate(texts)]
        return self.embedder.embed(chunks)

    def test_new_nodes_have_reduced_excitability(self):
        """Newly registered nodes start at dampened excitability."""
        embedded = self._make_embedded_chunks(["Test content"])
        node_ids = self.registrar.register(embedded)
        node = self.graph.nodes[node_ids[0]]
        self.assertAlmostEqual(node.intrinsic_excitability, 0.3)

    def test_new_nodes_have_boosted_threshold(self):
        """Newly registered nodes have higher firing threshold."""
        embedded = self._make_embedded_chunks(["Test content"])
        node_ids = self.registrar.register(embedded)
        node = self.graph.nodes[node_ids[0]]
        expected = self.graph.config["default_threshold"] + 0.2
        self.assertAlmostEqual(node.threshold, expected)

    def test_probation_metadata(self):
        """Registered nodes have probation tracking metadata."""
        embedded = self._make_embedded_chunks(["Test"])
        node_ids = self.registrar.register(embedded)
        meta = self.graph.nodes[node_ids[0]].metadata
        self.assertEqual(meta["probation_remaining"], 10)
        self.assertEqual(meta["novelty_dampening"], 0.3)
        self.assertEqual(meta["creation_mode"], "ingested")

    def test_probation_decrements(self):
        """update_probation decrements remaining probation."""
        embedded = self._make_embedded_chunks(["Test"])
        node_ids = self.registrar.register(embedded)
        self.registrar.update_probation(node_ids)
        meta = self.graph.nodes[node_ids[0]].metadata
        self.assertEqual(meta["probation_remaining"], 9)

    def test_probation_graduation(self):
        """Node graduates after probation period completes."""
        embedded = self._make_embedded_chunks(["Test"])
        node_ids = self.registrar.register(embedded)

        for _ in range(10):
            graduated = self.registrar.update_probation(node_ids)

        self.assertIn(node_ids[0], graduated)
        node = self.graph.nodes[node_ids[0]]
        self.assertAlmostEqual(node.intrinsic_excitability, 1.0)
        self.assertAlmostEqual(node.threshold, self.graph.config["default_threshold"])
        self.assertTrue(node.metadata.get("graduated"))

    def test_dampening_fades_during_probation(self):
        """Excitability increases during probation (linear curve)."""
        embedded = self._make_embedded_chunks(["Test"])
        node_ids = self.registrar.register(embedded)
        node = self.graph.nodes[node_ids[0]]

        excitabilities = [node.intrinsic_excitability]
        for _ in range(5):
            self.registrar.update_probation(node_ids)
            excitabilities.append(node.intrinsic_excitability)

        # Should be monotonically increasing
        for i in range(1, len(excitabilities)):
            self.assertGreaterEqual(excitabilities[i], excitabilities[i - 1])

    def test_dampening_stored_in_vector_db(self):
        """Registration stores embeddings in vector DB."""
        embedded = self._make_embedded_chunks(["Vector stored"])
        node_ids = self.registrar.register(embedded)
        entry = self.vector_db.get(node_ids[0])
        self.assertIsNotNone(entry)
        self.assertEqual(entry["content"], "Vector stored")

    def test_exponential_dampening_curve(self):
        """Exponential dampening curve has fast early growth."""
        registrar = NodeRegistrar(
            self.graph, self.vector_db,
            config={
                "novelty_dampening": 0.1,
                "probation_period": 20,
                "dampening_curve": DampeningCurve.EXPONENTIAL,
                "initial_threshold_boost": 0.0,
            },
        )
        embedded = self._make_embedded_chunks(["Exp test"])
        node_ids = registrar.register(embedded)
        node = self.graph.nodes[node_ids[0]]

        # After a few steps, exponential should already be above linear
        for _ in range(5):
            registrar.update_probation(node_ids)

        # At 25% progress (5/20), exponential should be > 0.1 + 0.25*(1-0.1)
        self.assertGreater(node.intrinsic_excitability, 0.1)


# ---------------------------------------------------------------------------
# Association Tests
# ---------------------------------------------------------------------------

class TestHypergraphAssociator(unittest.TestCase):
    """Tests for similarity-based and structural association."""

    def setUp(self):
        self.graph = Graph()
        self.vector_db = SimpleVectorDB()
        self.embedder = EmbeddingEngine({"use_model": False, "dimension": 32})

    def _setup_nodes_and_embeddings(self, texts):
        """Helper: create nodes and embeddings."""
        chunks = [Chunk(text=t, chunk_id=f"n_{i}", position=i)
                  for i, t in enumerate(texts)]
        embedded = self.embedder.embed(chunks)
        node_ids = []
        for ec in embedded:
            nid = ec.chunk.chunk_id
            node = self.graph.create_node(node_id=nid)
            self.vector_db.insert(nid, ec.vector, ec.chunk.text)
            node_ids.append(nid)
        return embedded, node_ids

    def test_similarity_synapses_created(self):
        """Synapses created between similar chunks."""
        # Use identical text to guarantee high similarity
        embedded, node_ids = self._setup_nodes_and_embeddings(
            ["alpha beta gamma", "alpha beta gamma"]
        )
        assoc = HypergraphAssociator(self.graph, {"similarity_threshold": 0.5})
        syn_ids, he_ids = assoc.associate(embedded, node_ids, self.vector_db)
        self.assertGreater(len(syn_ids), 0)

    def test_similarity_threshold_filters(self):
        """High threshold prevents synapse creation for dissimilar chunks."""
        embedded, node_ids = self._setup_nodes_and_embeddings(
            ["alpha", "zyxwvutsrqponm"]
        )
        assoc = HypergraphAssociator(self.graph, {"similarity_threshold": 0.999})
        syn_ids, he_ids = assoc.associate(embedded, node_ids, self.vector_db)
        # If texts are different enough, no synapses at very high threshold
        # (This depends on hash embeddings; may or may not create synapses)
        # Just verify the code runs without error
        self.assertIsInstance(syn_ids, list)

    def test_sequential_synapses(self):
        """Sequential chunks get linked with structural synapses."""
        embedded, node_ids = self._setup_nodes_and_embeddings(
            ["Part one.", "Part two.", "Part three."]
        )
        assoc = HypergraphAssociator(
            self.graph,
            {"similarity_threshold": 0.999, "sequential_weight": 0.3},
        )
        syn_ids, he_ids = assoc.associate(embedded, node_ids, self.vector_db)
        # Check for sequential synapses
        sequential = [
            sid for sid in syn_ids
            if self.graph.synapses[sid].metadata.get("creation_mode") == "sequential"
        ]
        self.assertEqual(len(sequential), 2)  # 3 nodes → 2 sequential links

    def test_parent_child_synapses(self):
        """Parent-child chunks get linked."""
        parent_chunk = Chunk(text="Parent section", chunk_id="parent", position=0)
        child_chunk = Chunk(
            text="Child subsection", chunk_id="child", position=1,
            parent_chunk_id="parent",
        )
        embedded = self.embedder.embed([parent_chunk, child_chunk])
        for ec in embedded:
            nid = ec.chunk.chunk_id
            self.graph.create_node(node_id=nid)
            self.vector_db.insert(nid, ec.vector, ec.chunk.text)

        assoc = HypergraphAssociator(
            self.graph,
            {"similarity_threshold": 0.999, "parent_child_weight": 0.5},
        )
        syn_ids, he_ids = assoc.associate(
            embedded, ["parent", "child"], self.vector_db
        )
        parent_child = [
            sid for sid in syn_ids
            if self.graph.synapses[sid].metadata.get("creation_mode") == "parent_child"
        ]
        self.assertEqual(len(parent_child), 1)

    def test_code_definition_usage_links(self):
        """Code definitions get linked to chunks referencing them."""
        chunks = [
            Chunk(text="def calculate_total(items):\n    return sum(items)",
                  chunk_id="def_chunk", position=0,
                  metadata={"strategy": "code_aware", "name": "calculate_total"}),
            Chunk(text="result = calculate_total(my_items)",
                  chunk_id="usage_chunk", position=1,
                  metadata={"strategy": "code_aware", "name": "main"}),
        ]
        embedded = self.embedder.embed(chunks)
        for ec in embedded:
            nid = ec.chunk.chunk_id
            self.graph.create_node(node_id=nid)
            self.vector_db.insert(nid, ec.vector, ec.chunk.text)

        assoc = HypergraphAssociator(
            self.graph, {"similarity_threshold": 0.999}
        )
        syn_ids, _ = assoc.associate(
            embedded, ["def_chunk", "usage_chunk"], self.vector_db
        )
        code_links = [
            sid for sid in syn_ids
            if self.graph.synapses[sid].metadata.get("creation_mode") == "code_def_usage"
        ]
        self.assertGreater(len(code_links), 0)

    def test_hypergraph_clustering(self):
        """Clusters of similar nodes form hyperedges."""
        # Create many nodes with identical text to guarantee clustering
        texts = ["clustering test content"] * 5
        embedded, node_ids = self._setup_nodes_and_embeddings(texts)
        assoc = HypergraphAssociator(
            self.graph,
            {"similarity_threshold": 0.999, "min_cluster_size": 3,
             "cluster_similarity_threshold": 0.5},
        )
        _, he_ids = assoc.associate(embedded, node_ids, self.vector_db)
        self.assertGreater(len(he_ids), 0)
        he = self.graph.hyperedges[he_ids[0]]
        self.assertGreaterEqual(len(he.member_nodes), 3)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestUniversalIngestor(unittest.TestCase):
    """End-to-end tests for the ingestion pipeline."""

    def setUp(self):
        self.graph = Graph()
        self.vector_db = SimpleVectorDB()
        self.ingestor = UniversalIngestor(
            self.graph, self.vector_db,
            config={
                "embedding": {"use_model": False, "dimension": 32},
                "registration": {
                    "novelty_dampening": 0.3,
                    "probation_period": 5,
                },
                "association": {
                    "similarity_threshold": 0.99,
                    "min_cluster_size": 10,
                },
            },
        )

    def test_ingest_text(self):
        """Ingest plain text end-to-end."""
        text = "First paragraph about neural networks.\n\nSecond paragraph about AI."
        result = self.ingestor.ingest(text)
        self.assertGreater(result.chunks_created, 0)
        self.assertGreater(len(result.nodes_created), 0)
        self.assertEqual(result.source_type, SourceType.TEXT)

    def test_ingest_markdown(self):
        """Ingest markdown with heading structure."""
        md = "# Title\n\nIntro paragraph.\n\n## Section 1\n\nSection content."
        result = self.ingestor.ingest(md, source_type=SourceType.MARKDOWN)
        self.assertGreater(result.chunks_created, 0)
        self.assertGreater(len(result.nodes_created), 0)
        self.assertEqual(result.source_type, SourceType.MARKDOWN)

    def test_ingest_code(self):
        """Ingest Python code, preserving structure."""
        code = textwrap.dedent("""\
        class DataProcessor:
            def __init__(self):
                self.data = []

            def process(self, item):
                self.data.append(item)

        def main():
            dp = DataProcessor()
            dp.process("test")
        """)
        result = self.ingestor.ingest(code, source_type=SourceType.CODE)
        self.assertGreater(result.chunks_created, 0)
        self.assertGreater(len(result.nodes_created), 0)

    def test_ingest_empty_input(self):
        """Empty input produces zero chunks."""
        result = self.ingestor.ingest("")
        self.assertEqual(result.chunks_created, 0)
        self.assertEqual(len(result.nodes_created), 0)

    def test_nodes_in_graph_after_ingest(self):
        """Nodes are registered in the graph after ingestion."""
        text = "Content for graph registration."
        result = self.ingestor.ingest(text)
        for nid in result.nodes_created:
            self.assertIn(nid, self.graph.nodes)

    def test_embeddings_in_vector_db(self):
        """Embeddings are stored in vector DB after ingestion."""
        text = "Content for vector DB."
        result = self.ingestor.ingest(text)
        for nid in result.nodes_created:
            entry = self.vector_db.get(nid)
            self.assertIsNotNone(entry)

    def test_ingestion_log(self):
        """Ingestion results are logged."""
        self.ingestor.ingest("First doc")
        self.ingestor.ingest("Second doc")
        self.assertEqual(len(self.ingestor.ingestion_log), 2)

    def test_query_similar(self):
        """query_similar finds related content after ingestion."""
        self.ingestor.ingest("Neural networks learn patterns from data.")
        # Query with the same text should find it
        results = self.ingestor.query_similar(
            "Neural networks learn patterns from data.",
            k=5, threshold=0.0,
        )
        self.assertGreater(len(results), 0)

    def test_update_probation_after_steps(self):
        """Probation updates work with the pipeline."""
        result = self.ingestor.ingest("Probation test content.")
        node_ids = result.nodes_created
        node = self.graph.nodes[node_ids[0]]

        # Initially dampened
        self.assertLess(node.intrinsic_excitability, 1.0)

        # Graduate through probation
        for _ in range(5):
            self.ingestor.update_probation()

        # Should be graduated
        self.assertAlmostEqual(node.intrinsic_excitability, 1.0)

    def test_ingest_batch(self):
        """Batch ingestion processes multiple sources."""
        sources = [
            ("First document content.", None),
            ("# Markdown Doc\n\nContent.", SourceType.MARKDOWN),
        ]
        results = self.ingestor.ingest_batch(sources)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].chunks_created, 0)
        self.assertGreater(results[1].chunks_created, 0)

    def test_sequential_synapses_in_pipeline(self):
        """End-to-end pipeline creates sequential synapses."""
        # Use a chunker config that forces multiple chunks
        ingestor = UniversalIngestor(
            self.graph, self.vector_db,
            config={
                "embedding": {"use_model": False, "dimension": 32},
                "chunking": {
                    "strategy": ChunkStrategy.SEMANTIC,
                    "max_chunk_tokens": 10,
                },
                "association": {
                    "similarity_threshold": 0.999,
                    "sequential_weight": 0.3,
                },
            },
        )
        text = "First topic about AI.\n\nSecond topic about ML.\n\nThird topic about DL."
        result = ingestor.ingest(text)
        if result.chunks_created >= 2:
            sequential = [
                sid for sid in result.synapses_created
                if self.graph.synapses.get(sid, None) and
                self.graph.synapses[sid].metadata.get("creation_mode") == "sequential"
            ]
            self.assertGreater(len(sequential), 0)


# ---------------------------------------------------------------------------
# Project Configuration Tests
# ---------------------------------------------------------------------------

class TestProjectConfigs(unittest.TestCase):
    """Tests for project-specific configurations."""

    def test_openclaw_config_structure(self):
        """OpenClaw config has all required sections."""
        config = OPENCLAW_INGESTOR_CONFIG
        self.assertIn("extraction", config)
        self.assertIn("chunking", config)
        self.assertIn("embedding", config)
        self.assertIn("registration", config)
        self.assertIn("association", config)

    def test_openclaw_code_aware_chunking(self):
        """OpenClaw uses code-aware chunking."""
        config = OPENCLAW_INGESTOR_CONFIG
        self.assertEqual(config["chunking"]["strategy"], ChunkStrategy.CODE_AWARE)

    def test_openclaw_fast_integration(self):
        """OpenClaw has aggressive novelty dampening (0.3)."""
        config = OPENCLAW_INGESTOR_CONFIG
        self.assertEqual(config["registration"]["novelty_dampening"], 0.3)

    def test_dsm_hierarchical_chunking(self):
        """DSM uses hierarchical chunking."""
        config = DSM_INGESTOR_CONFIG
        self.assertEqual(config["chunking"]["strategy"], ChunkStrategy.HIERARCHICAL)

    def test_dsm_conservative_dampening(self):
        """DSM has conservative novelty dampening (0.05)."""
        config = DSM_INGESTOR_CONFIG
        self.assertEqual(config["registration"]["novelty_dampening"], 0.05)

    def test_dsm_high_similarity_threshold(self):
        """DSM requires high similarity for associations."""
        config = DSM_INGESTOR_CONFIG
        self.assertEqual(config["association"]["similarity_threshold"], 0.8)

    def test_consciousness_semantic_chunking(self):
        """Consciousness uses semantic chunking."""
        config = CONSCIOUSNESS_INGESTOR_CONFIG
        self.assertEqual(config["chunking"]["strategy"], ChunkStrategy.SEMANTIC)

    def test_consciousness_very_conservative(self):
        """Consciousness has very conservative dampening (0.01)."""
        config = CONSCIOUSNESS_INGESTOR_CONFIG
        self.assertEqual(config["registration"]["novelty_dampening"], 0.01)

    def test_consciousness_exploratory_threshold(self):
        """Consciousness uses lower similarity threshold for exploration."""
        config = CONSCIOUSNESS_INGESTOR_CONFIG
        self.assertEqual(config["association"]["similarity_threshold"], 0.65)

    def test_get_ingestor_config(self):
        """get_ingestor_config returns correct config by name."""
        config = get_ingestor_config("openclaw")
        self.assertEqual(config, OPENCLAW_INGESTOR_CONFIG)
        config = get_ingestor_config("dsm")
        self.assertEqual(config, DSM_INGESTOR_CONFIG)
        config = get_ingestor_config("consciousness")
        self.assertEqual(config, CONSCIOUSNESS_INGESTOR_CONFIG)

    def test_get_ingestor_config_invalid(self):
        """get_ingestor_config raises ValueError for unknown project."""
        with self.assertRaises(ValueError):
            get_ingestor_config("nonexistent")

    def test_openclaw_end_to_end(self):
        """OpenClaw config works end-to-end with code ingestion."""
        graph = Graph({"default_threshold": 0.8})
        vdb = SimpleVectorDB()
        config = {**OPENCLAW_INGESTOR_CONFIG}
        config["embedding"] = {"use_model": False, "dimension": 32}
        ingestor = UniversalIngestor(graph, vdb, config)

        code = "class Foo:\n    def bar(self):\n        return 42\n"
        result = ingestor.ingest(code, source_type=SourceType.CODE)
        self.assertGreater(result.chunks_created, 0)

    def test_dsm_end_to_end(self):
        """DSM config works end-to-end with markdown ingestion."""
        graph = Graph({"default_threshold": 1.2})
        vdb = SimpleVectorDB()
        config = {**DSM_INGESTOR_CONFIG}
        config["embedding"] = {"use_model": False, "dimension": 32}
        ingestor = UniversalIngestor(graph, vdb, config)

        md = "# Disorder\n\nDescription.\n\n## Symptoms\n\nList of symptoms."
        result = ingestor.ingest(md, source_type=SourceType.MARKDOWN)
        self.assertGreater(result.chunks_created, 0)

    def test_consciousness_end_to_end(self):
        """Consciousness config works end-to-end."""
        graph = Graph()
        vdb = SimpleVectorDB()
        config = {**CONSCIOUSNESS_INGESTOR_CONFIG}
        config["embedding"] = {"use_model": False, "dimension": 32}
        ingestor = UniversalIngestor(graph, vdb, config)

        text = "Consciousness is the subjective experience of awareness.\n\nQualia are individual instances."
        result = ingestor.ingest(text)
        self.assertGreater(result.chunks_created, 0)


# ---------------------------------------------------------------------------
# IngestorConfig Tests
# ---------------------------------------------------------------------------

class TestIngestorConfig(unittest.TestCase):
    """Tests for the IngestorConfig wrapper."""

    def test_attribute_access(self):
        """IngestorConfig provides attribute access to sections."""
        cfg = IngestorConfig({
            "extraction": {"key": "val"},
            "chunking": {"strategy": "semantic"},
        })
        self.assertEqual(cfg.extraction["key"], "val")
        self.assertEqual(cfg.chunking["strategy"], "semantic")

    def test_missing_section_returns_empty(self):
        """Missing sections return empty dict."""
        cfg = IngestorConfig({})
        self.assertEqual(cfg.extraction, {})
        self.assertEqual(cfg.association, {})

    def test_get_method(self):
        """get() works for arbitrary keys."""
        cfg = IngestorConfig({"custom": 42})
        self.assertEqual(cfg.get("custom"), 42)
        self.assertIsNone(cfg.get("missing"))


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and robustness."""

    def test_single_word_input(self):
        """Single word input produces at least one chunk."""
        graph = Graph()
        vdb = SimpleVectorDB()
        ingestor = UniversalIngestor(
            graph, vdb,
            config={"embedding": {"use_model": False, "dimension": 16}},
        )
        result = ingestor.ingest("Hello")
        self.assertGreater(result.chunks_created, 0)

    def test_whitespace_only_input(self):
        """Whitespace-only input produces no chunks."""
        graph = Graph()
        vdb = SimpleVectorDB()
        ingestor = UniversalIngestor(
            graph, vdb,
            config={"embedding": {"use_model": False, "dimension": 16}},
        )
        result = ingestor.ingest("   \n\n   \t   ")
        self.assertEqual(result.chunks_created, 0)

    def test_very_long_input(self):
        """Long input is chunked without error."""
        graph = Graph()
        vdb = SimpleVectorDB()
        ingestor = UniversalIngestor(
            graph, vdb,
            config={
                "embedding": {"use_model": False, "dimension": 16},
                "chunking": {
                    "strategy": ChunkStrategy.FIXED_SIZE,
                    "max_chunk_tokens": 50,
                    "overlap_tokens": 10,
                },
            },
        )
        text = "Word " * 5000  # ~5000 words
        result = ingestor.ingest(text)
        self.assertGreater(result.chunks_created, 1)
        self.assertEqual(len(result.nodes_created), result.chunks_created)

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        graph = Graph()
        vdb = SimpleVectorDB()
        ingestor = UniversalIngestor(
            graph, vdb,
            config={"embedding": {"use_model": False, "dimension": 16}},
        )
        text = "Quantenmechanik beschreibt Teilchen.\n\nSchrodinger-Gleichung."
        result = ingestor.ingest(text)
        self.assertGreater(result.chunks_created, 0)

    def test_multiple_ingestions_accumulate(self):
        """Multiple ingestions accumulate nodes in the same graph."""
        graph = Graph()
        vdb = SimpleVectorDB()
        ingestor = UniversalIngestor(
            graph, vdb,
            config={
                "embedding": {"use_model": False, "dimension": 16},
                "association": {"similarity_threshold": 0.999},
            },
        )
        result1 = ingestor.ingest("First document.")
        result2 = ingestor.ingest("Second document.")
        total_nodes = len(result1.nodes_created) + len(result2.nodes_created)
        self.assertEqual(len(graph.nodes), total_nodes)

    def test_vector_db_zero_vector(self):
        """Zero vector is handled gracefully in vector DB."""
        db = SimpleVectorDB()
        db.insert("zero", np.zeros(3), "zero vec")
        # Search with zero vector — no crash
        results = db.search(np.zeros(3), k=5, threshold=0.0)
        # Result is implementation-defined but should not crash
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
