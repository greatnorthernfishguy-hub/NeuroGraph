"""
NeuroGraph Foundation — Phase 4: Universal Ingestor System

Five-stage pipeline for consuming arbitrary data sources and transforming them
into fully integrated knowledge within the NeuroGraph Foundation.

Pipeline stages (PRD Addendum §2):
    1. Extract  — Convert raw input to structured text
    2. Chunk    — Segment into semantically meaningful units
    3. Embed    — Vector representations via sentence-transformers
    4. Register — Insert into Vector DB, create SNN nodes with novelty dampening
    5. Associate — Create synapses/hyperedges via similarity and structure

Reference: NeuroGraph Foundation PRD Addendum v1.1-1.2 (Universal Ingestor).

Grok Review Changelog (v0.7.1):
    Accepted: Added outer try/except in embed() as defense-in-depth around
        _encode_batch() — if something unexpected bypasses the inner catch,
        the batch falls back to per-chunk hash embeddings rather than
        propagating up to the caller.
    Rejected: 'PDF extraction loses images/tables' — PyPDF2 is a text
        extraction library by design. Image/table extraction would require
        pdfplumber or similar, which is a feature request (not a bug).
        Will consider for a future phase if demand warrants.
    Rejected: 'No graceful degrade if model load fails' — _try_load_model()
        already catches ImportError AND broad Exception, sets _model_available
        = False, and falls back to hash. _encode_batch() additionally wraps
        runtime model errors. Both paths were implemented since Phase 4.
    Rejected: 'Cache uses SHA256 on vectors — why?' — Cache key is SHA256 of
        the TEXT, not the vector (_cache_key hashes text.encode("utf-8")).
        Text is the correct key: same text always maps to the same embedding
        within a model session. Using text directly as key would consume more
        memory for large chunks.
    Rejected: 'Chunker OOM for >10k tokens' — Semantic chunker already splits
        oversized paragraphs by sentence boundaries (lines 619-634).
        Hierarchical chunker calls _split_large_section() for oversized
        headings. Fixed-size chunker uses a sliding window. All paths are
        bounded by max_chunk_tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import textwrap
import uuid
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from neuro_foundation import (
    ActivationMode,
    Graph,
    Node,
    Synapse,
    SynapseType,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SourceType(Enum):
    """Supported input source types."""
    URL = auto()
    PDF = auto()
    CODE = auto()
    MARKDOWN = auto()
    TEXT = auto()
    JSON = auto()
    CSV = auto()
    HTML = auto()
    MEDIA = auto()


class ChunkStrategy(Enum):
    """Chunking strategies (PRD Addendum §2, Stage 2)."""
    SEMANTIC = auto()
    CODE_AWARE = auto()
    HIERARCHICAL = auto()
    FIXED_SIZE = auto()


class DampeningCurve(Enum):
    """Novelty dampening fade curves (PRD Addendum §4.3)."""
    LINEAR = auto()
    EXPONENTIAL = auto()
    LOGARITHMIC = auto()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedContent:
    """Output of Stage 1: Extract (PRD Addendum §2).

    Attributes:
        raw_text: Full extracted text.
        metadata: Source metadata (filename, url, language, etc.).
        structure: Structural hints (headings, functions, sections).
        source_type: Detected or specified source type.
    """
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    structure: List[Dict[str, Any]] = field(default_factory=list)
    source_type: SourceType = SourceType.TEXT


@dataclass
class Chunk:
    """A semantically meaningful segment of content (PRD Addendum §2, Stage 2).

    Attributes:
        chunk_id: Unique identifier.
        text: The chunk text content.
        metadata: Chunk metadata (position, type, parent, language, etc.).
        parent_chunk_id: ID of parent chunk for hierarchical relationships.
        position: Ordinal position in the original document.
        token_count: Approximate token count.
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None
    position: int = 0
    token_count: int = 0


@dataclass
class EmbeddedChunk:
    """A chunk with its vector representation (PRD Addendum §2, Stage 3).

    Attributes:
        chunk: The original Chunk.
        vector: Normalized embedding vector.
        model_name: Name of the embedding model used.
    """
    chunk: Chunk = field(default_factory=Chunk)
    vector: np.ndarray = field(default_factory=lambda: np.array([]))
    model_name: str = ""


@dataclass
class IngestionResult:
    """Result of a full ingestion pipeline run (PRD Addendum §4.1).

    Attributes:
        source: Original source identifier.
        source_type: Detected source type.
        chunks_created: Number of chunks produced.
        nodes_created: List of created node IDs.
        synapses_created: List of created synapse IDs.
        hyperedges_created: List of created hyperedge IDs.
        embeddings_cached: Number of embeddings served from cache.
        metadata: Additional result metadata.
    """
    source: str = ""
    source_type: SourceType = SourceType.TEXT
    chunks_created: int = 0
    nodes_created: List[str] = field(default_factory=list)
    synapses_created: List[str] = field(default_factory=list)
    hyperedges_created: List[str] = field(default_factory=list)
    embeddings_cached: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simple Vector DB (PRD §7)
# ---------------------------------------------------------------------------

class SimpleVectorDB:
    """In-memory vector database using numpy (PRD §7).

    Provides cosine-similarity search over normalized vectors with
    content and metadata storage.

    Example::

        db = SimpleVectorDB()
        db.insert("doc1", np.array([0.1, 0.9, 0.2]), "hello world", {"source": "test"})
        results = db.search(np.array([0.1, 0.9, 0.2]), k=5, threshold=0.5)
    """

    def __init__(self) -> None:
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.content: Dict[str, str] = {}

    def insert(
        self,
        id: str,
        embedding: np.ndarray,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store embedding with content and metadata.

        The embedding is L2-normalized before storage for fast cosine
        similarity via dot product.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.embeddings[id] = embedding
        self.content[id] = content
        self.metadata[id] = metadata or {}

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Cosine similarity search, return top-k above threshold.

        Returns:
            List of (id, similarity) tuples sorted by descending similarity.
        """
        if not self.embeddings:
            return []

        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        results: List[Tuple[str, float]] = []
        for id, emb in self.embeddings.items():
            sim = float(np.dot(query_vector, emb))
            if sim >= threshold:
                results.append((id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve embedding, content, and metadata by ID."""
        if id not in self.embeddings:
            return None
        return {
            "id": id,
            "embedding": self.embeddings[id],
            "content": self.content[id],
            "metadata": self.metadata[id],
        }

    def delete(self, id: str) -> bool:
        """Remove an entry by ID. Returns True if found and deleted."""
        if id not in self.embeddings:
            return False
        del self.embeddings[id]
        del self.content[id]
        del self.metadata[id]
        return True

    def count(self) -> int:
        """Return number of stored entries."""
        return len(self.embeddings)

    def all_ids(self) -> List[str]:
        """Return all stored IDs."""
        return list(self.embeddings.keys())


# ---------------------------------------------------------------------------
# Stage 1: Extractors (PRD Addendum §2, Stage 1)
# ---------------------------------------------------------------------------

class BaseExtractor:
    """Base class for content extractors."""

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        raise NotImplementedError


class TextExtractor(BaseExtractor):
    """Plain text fallback extractor."""

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        # source is the raw text itself
        return ExtractedContent(
            raw_text=source,
            metadata={"source_type": "text", "length": len(source)},
            structure=[],
            source_type=SourceType.TEXT,
        )


class MarkdownExtractor(BaseExtractor):
    """Markdown extractor preserving heading hierarchy."""

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        structure = []
        current_section = None
        lines = source.split("\n")

        for i, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    "type": "heading",
                    "level": level,
                    "title": title,
                    "line": i,
                }
                structure.append(current_section)

        return ExtractedContent(
            raw_text=source,
            metadata={
                "source_type": "markdown",
                "heading_count": len(structure),
                "length": len(source),
            },
            structure=structure,
            source_type=SourceType.MARKDOWN,
        )


class CodeExtractor(BaseExtractor):
    """Language-aware code extractor using regex-based parsing.

    Extracts function/class definitions for Python, JavaScript, and
    generic code files. Uses AST-like structural detection without
    requiring language-specific parsers.
    """

    # Patterns for common languages
    _PYTHON_PATTERNS = [
        (r'^(class\s+\w+[^:]*:)', "class"),
        (r'^(def\s+\w+\s*\([^)]*\)[^:]*:)', "function"),
        (r'^(async\s+def\s+\w+\s*\([^)]*\)[^:]*:)', "async_function"),
    ]
    _JS_PATTERNS = [
        (r'^(class\s+\w+)', "class"),
        (r'^((?:export\s+)?(?:async\s+)?function\s+\w+)', "function"),
        (r'^(const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)', "arrow_function"),
    ]

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        language = kwargs.get("language", "python")
        structure = self._parse_structure(source, language)

        return ExtractedContent(
            raw_text=source,
            metadata={
                "source_type": "code",
                "language": language,
                "length": len(source),
                "definitions": len(structure),
            },
            structure=structure,
            source_type=SourceType.CODE,
        )

    def _parse_structure(
        self, source: str, language: str
    ) -> List[Dict[str, Any]]:
        patterns = self._PYTHON_PATTERNS
        if language in ("javascript", "js", "typescript", "ts"):
            patterns = self._JS_PATTERNS

        structure = []
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern, def_type in patterns:
                if re.match(pattern, stripped):
                    # Extract name
                    name_match = re.search(
                        r'(?:class|def|function|const|async\s+def)\s+(\w+)',
                        stripped,
                    )
                    name = name_match.group(1) if name_match else "unknown"
                    indent = len(line) - len(line.lstrip())
                    structure.append({
                        "type": def_type,
                        "name": name,
                        "line": i,
                        "indent": indent,
                    })
                    break

        return structure


class URLExtractor(BaseExtractor):
    """URL/web content extractor.

    Uses beautifulsoup4 for HTML parsing when available,
    falls back to regex-based tag stripping.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        # Try to fetch URL content
        try:
            import urllib.request
            with urllib.request.urlopen(source, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            # If fetching fails, treat source as HTML content
            html = source

        text = self._html_to_text(html)
        title = self._extract_title(html)

        return ExtractedContent(
            raw_text=text,
            metadata={
                "source_type": "url",
                "url": source,
                "title": title,
                "length": len(text),
            },
            structure=[],
            source_type=SourceType.URL,
        )

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: regex-based tag stripping
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""


class PDFExtractor(BaseExtractor):
    """PDF text extractor with layout awareness.

    Uses PyPDF2 when available, returns informative error otherwise.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        text, page_structure = self._extract_pdf(source)

        return ExtractedContent(
            raw_text=text,
            metadata={
                "source_type": "pdf",
                "path": source,
                "pages": len(page_structure),
                "length": len(text),
            },
            structure=page_structure,
            source_type=SourceType.PDF,
        )

    def _extract_pdf(self, path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            pages = []
            all_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                all_text.append(page_text)
                pages.append({
                    "type": "page",
                    "page_number": i + 1,
                    "length": len(page_text),
                })
            return "\n\n".join(all_text), pages
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF extraction. "
                "Install with: pip install PyPDF2>=3.0.0"
            )
        except Exception as e:
            raise ValueError(f"Failed to extract PDF '{path}': {e}")


class JSONExtractor(BaseExtractor):
    """JSON file extractor that preserves structure as navigation hints.

    Extracts all text content from JSON data (string values, keys) and
    reports the top-level structure (keys, nesting depth) so that the
    chunker can produce meaningful segments.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        try:
            data = json.loads(source)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON — fall back to plain text
            return TextExtractor().extract(source, **kwargs)

        text_parts: List[str] = []
        structure: List[Dict[str, Any]] = []
        self._walk(data, text_parts, structure, depth=0, path="$")

        raw_text = "\n".join(text_parts) if text_parts else source

        return ExtractedContent(
            raw_text=raw_text,
            metadata={
                "source_type": "json",
                "length": len(source),
                "top_level_type": type(data).__name__,
                "top_level_keys": (
                    list(data.keys())[:50] if isinstance(data, dict) else None
                ),
                "structure_entries": len(structure),
            },
            structure=structure,
            source_type=SourceType.JSON,
        )

    def _walk(
        self,
        obj: Any,
        text_parts: List[str],
        structure: List[Dict[str, Any]],
        depth: int,
        path: str,
    ) -> None:
        if isinstance(obj, dict):
            structure.append({
                "type": "object",
                "path": path,
                "depth": depth,
                "keys": list(obj.keys())[:50],
            })
            for key, value in obj.items():
                text_parts.append(f"{key}: {self._value_repr(value)}")
                if isinstance(value, (dict, list)):
                    self._walk(value, text_parts, structure, depth + 1, f"{path}.{key}")
        elif isinstance(obj, list):
            structure.append({
                "type": "array",
                "path": path,
                "depth": depth,
                "length": len(obj),
            })
            for i, item in enumerate(obj[:200]):  # cap iteration
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, (dict, list)):
                    self._walk(item, text_parts, structure, depth + 1, f"{path}[{i}]")
                else:
                    text_parts.append(str(item))
        elif isinstance(obj, str):
            text_parts.append(obj)
        else:
            text_parts.append(str(obj))

    @staticmethod
    def _value_repr(value: Any) -> str:
        if isinstance(value, str):
            return value[:500] if len(value) > 500 else value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return f"[...{len(value)} items]"
        if isinstance(value, dict):
            return f"{{...{len(value)} keys}}"
        if value is None:
            return "null"
        return str(value)[:200]


class CSVExtractor(BaseExtractor):
    """CSV file extractor that preserves column structure.

    Reads the CSV using the ``csv`` stdlib module and produces text
    that represents each row as key-value pairs (using header column
    names when available).  Structure entries record column names and
    row count.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        import csv
        import io

        reader = csv.reader(io.StringIO(source))
        rows = list(reader)

        if not rows:
            return ExtractedContent(
                raw_text=source,
                metadata={"source_type": "csv", "length": len(source), "rows": 0},
                structure=[],
                source_type=SourceType.CSV,
            )

        # Heuristic: first row is a header if it doesn't look numeric
        header = rows[0]
        has_header = not all(self._looks_numeric(cell) for cell in header if cell.strip())

        if has_header:
            data_rows = rows[1:]
            columns = header
        else:
            data_rows = rows
            columns = [f"col_{i}" for i in range(len(header))]

        text_parts: List[str] = []
        # Emit column names as a header line
        text_parts.append("Columns: " + ", ".join(columns))

        for row_idx, row in enumerate(data_rows[:5000]):  # cap to prevent OOM
            pairs = []
            for col_idx, cell in enumerate(row):
                col_name = columns[col_idx] if col_idx < len(columns) else f"col_{col_idx}"
                if cell.strip():
                    pairs.append(f"{col_name}={cell.strip()}")
            if pairs:
                text_parts.append(f"Row {row_idx + 1}: " + "; ".join(pairs))

        structure = [{
            "type": "table",
            "columns": columns,
            "row_count": len(data_rows),
            "has_header": has_header,
        }]

        return ExtractedContent(
            raw_text="\n".join(text_parts),
            metadata={
                "source_type": "csv",
                "length": len(source),
                "columns": columns,
                "rows": len(data_rows),
                "has_header": has_header,
            },
            structure=structure,
            source_type=SourceType.CSV,
        )

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        try:
            float(value.strip())
            return True
        except (ValueError, AttributeError):
            return False


class HTMLExtractor(BaseExtractor):
    """Local HTML file extractor.

    Unlike ``URLExtractor`` which fetches from a URL first, this extractor
    operates directly on HTML content (typically read from a local file).
    Uses BeautifulSoup when available, falls back to regex tag stripping.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        text = self._html_to_text(source)
        title = self._extract_title(source)
        headings = self._extract_headings(source)

        return ExtractedContent(
            raw_text=text,
            metadata={
                "source_type": "html",
                "title": title,
                "length": len(text),
                "heading_count": len(headings),
            },
            structure=headings,
            source_type=SourceType.HTML,
        )

    def _html_to_text(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _extract_title(self, html: str) -> str:
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_headings(self, html: str) -> List[Dict[str, Any]]:
        headings: List[Dict[str, Any]] = []
        for match in re.finditer(r'<(h[1-6])[^>]*>(.*?)</\1>', html, re.DOTALL | re.IGNORECASE):
            level = int(match.group(1)[1])
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            if title:
                headings.append({
                    "type": "heading",
                    "level": level,
                    "title": title,
                })
        return headings


# Media format registry
MEDIA_EXTENSIONS: Dict[str, str] = {
    # Video
    ".mp4": "video", ".avi": "video", ".mkv": "video", ".mov": "video",
    ".webm": "video", ".wmv": "video", ".flv": "video", ".m4v": "video",
    ".mpg": "video", ".mpeg": "video", ".3gp": "video", ".ogv": "video",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".m4a": "audio", ".aac": "audio", ".wma": "audio", ".opus": "audio",
    ".aiff": "audio", ".alac": "audio",
}


class MediaReferenceExtractor(BaseExtractor):
    """Creates reference nodes for audio/video media files.

    Does NOT transcode or transcribe content.  Instead, creates a
    metadata-rich reference node that can be linked to other knowledge
    in the graph through synaptic associations.

    Supports both local file paths and URLs.
    """

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        is_url = bool(re.match(r'^https?://', source, re.IGNORECASE))
        is_file = os.path.isfile(source)

        if is_file:
            return self._extract_file(source)
        elif is_url:
            return self._extract_url(source)
        else:
            # Treat as a path reference even if file doesn't exist yet
            return self._extract_reference(source)

    def _extract_file(self, path: str) -> ExtractedContent:
        p = Path(path)
        ext = p.suffix.lower()
        media_type = MEDIA_EXTENSIONS.get(ext, "unknown")

        try:
            stat = p.stat()
            size_bytes = stat.st_size
            size_human = self._human_size(size_bytes)
        except OSError:
            size_bytes = 0
            size_human = "unknown"

        # Build a textual description for the node
        description = (
            f"[Media Reference: {p.name}]\n"
            f"Type: {media_type}/{ext.lstrip('.')}\n"
            f"Filename: {p.name}\n"
            f"Size: {size_human}\n"
            f"Path: {path}\n"
        )

        return ExtractedContent(
            raw_text=description,
            metadata={
                "source_type": "media",
                "media_type": media_type,
                "format": ext.lstrip("."),
                "filename": p.name,
                "file_path": str(p.resolve()),
                "size_bytes": size_bytes,
                "size_human": size_human,
                "length": len(description),
                "is_reference": True,
            },
            structure=[{
                "type": "media_reference",
                "media_type": media_type,
                "format": ext.lstrip("."),
                "filename": p.name,
            }],
            source_type=SourceType.MEDIA,
        )

    def _extract_url(self, url: str) -> ExtractedContent:
        # Parse the URL to extract filename and extension
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        path_part = unquote(parsed.path)
        filename = Path(path_part).name if path_part else url
        ext = Path(path_part).suffix.lower() if path_part else ""
        media_type = MEDIA_EXTENSIONS.get(ext, "media")

        description = (
            f"[Media Reference: {filename}]\n"
            f"Type: {media_type}/{ext.lstrip('.') if ext else 'unknown'}\n"
            f"Filename: {filename}\n"
            f"URL: {url}\n"
        )

        return ExtractedContent(
            raw_text=description,
            metadata={
                "source_type": "media",
                "media_type": media_type,
                "format": ext.lstrip(".") if ext else "unknown",
                "filename": filename,
                "url": url,
                "length": len(description),
                "is_reference": True,
            },
            structure=[{
                "type": "media_reference",
                "media_type": media_type,
                "format": ext.lstrip(".") if ext else "unknown",
                "url": url,
            }],
            source_type=SourceType.MEDIA,
        )

    def _extract_reference(self, source: str) -> ExtractedContent:
        """Handle a path that doesn't exist on disk (future/external reference)."""
        p = Path(source)
        ext = p.suffix.lower()
        media_type = MEDIA_EXTENSIONS.get(ext, "unknown")

        description = (
            f"[Media Reference: {p.name}]\n"
            f"Type: {media_type}/{ext.lstrip('.') if ext else 'unknown'}\n"
            f"Filename: {p.name}\n"
            f"Reference: {source}\n"
        )

        return ExtractedContent(
            raw_text=description,
            metadata={
                "source_type": "media",
                "media_type": media_type,
                "format": ext.lstrip(".") if ext else "unknown",
                "filename": p.name,
                "reference": source,
                "length": len(description),
                "is_reference": True,
            },
            structure=[{
                "type": "media_reference",
                "media_type": media_type,
                "format": ext.lstrip(".") if ext else "unknown",
            }],
            source_type=SourceType.MEDIA,
        )

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0  # type: ignore[assignment]
        return f"{size_bytes:.1f} PB"


class ExtractorRouter:
    """Routes source input to the appropriate extractor (PRD Addendum §2).

    Auto-detects source type based on content analysis or uses explicit
    source_type parameter.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._extractors: Dict[SourceType, BaseExtractor] = {
            SourceType.TEXT: TextExtractor(),
            SourceType.MARKDOWN: MarkdownExtractor(),
            SourceType.CODE: CodeExtractor(),
            SourceType.URL: URLExtractor(),
            SourceType.PDF: PDFExtractor(),
            SourceType.JSON: JSONExtractor(),
            SourceType.CSV: CSVExtractor(),
            SourceType.HTML: HTMLExtractor(),
            SourceType.MEDIA: MediaReferenceExtractor(),
        }

    def detect_source_type(self, source: str) -> SourceType:
        """Auto-detect source type from content analysis."""
        # URL detection — check for media URLs first
        if re.match(r'^https?://', source, re.IGNORECASE):
            from urllib.parse import urlparse, unquote
            parsed = urlparse(source)
            url_ext = Path(unquote(parsed.path)).suffix.lower()
            if url_ext in MEDIA_EXTENSIONS:
                return SourceType.MEDIA
            return SourceType.URL

        # File path detection
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == ".pdf":
                return SourceType.PDF
            if ext in (".py", ".js", ".ts", ".java", ".go", ".rs", ".c",
                        ".cpp", ".rb", ".php"):
                return SourceType.CODE
            if ext in (".md", ".markdown"):
                return SourceType.MARKDOWN
            if ext == ".json":
                return SourceType.JSON
            if ext == ".csv":
                return SourceType.CSV
            if ext in (".html", ".htm"):
                return SourceType.HTML
            if ext in MEDIA_EXTENSIONS:
                return SourceType.MEDIA

        # Content-based detection
        if source.strip().startswith(("# ", "## ", "### ")):
            return SourceType.MARKDOWN
        if re.search(r'^(def |class |import |from |function )', source, re.MULTILINE):
            return SourceType.CODE
        if source.strip().startswith(("<html", "<!DOCTYPE", "<head")):
            return SourceType.HTML
        # JSON detection: starts with { or [
        stripped = source.strip()
        if stripped and stripped[0] in ('{', '['):
            try:
                json.loads(stripped[:1000] if len(stripped) > 1000 else stripped)
                return SourceType.JSON
            except (json.JSONDecodeError, ValueError):
                pass

        return SourceType.TEXT

    def extract(
        self,
        source: str,
        source_type: Optional[SourceType] = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        """Extract content from source using appropriate extractor."""
        if source_type is None:
            source_type = self.detect_source_type(source)

        extractor = self._extractors.get(source_type, self._extractors[SourceType.TEXT])

        # For file paths, read the file first.
        # Exceptions: PDF (handles its own reading), URL (fetches from network),
        # MEDIA (handles its own file/URL detection).
        if (os.path.isfile(source)
                and source_type not in (SourceType.PDF, SourceType.URL, SourceType.MEDIA)):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            result = extractor.extract(content, **kwargs)
            result.metadata["file_path"] = source
        else:
            result = extractor.extract(source, **kwargs)

        result.source_type = source_type
        return result


# ---------------------------------------------------------------------------
# Stage 2: Adaptive Chunker (PRD Addendum §2, Stage 2)
# ---------------------------------------------------------------------------

class AdaptiveChunker:
    """Segments content into semantically meaningful chunks.

    Supports four strategies:
        - SEMANTIC: Respects paragraph/sentence boundaries (200-500 tokens)
        - CODE_AWARE: Chunks by function/class/module boundaries
        - HIERARCHICAL: Follows document heading hierarchy
        - FIXED_SIZE: Fixed-size with overlap (fallback)

    Token estimation uses word count / 0.75 as a heuristic.

    Args:
        config: Chunking configuration with keys:
            - strategy: ChunkStrategy (default SEMANTIC)
            - min_chunk_tokens: Minimum chunk size (default 200)
            - max_chunk_tokens: Maximum chunk size (default 500)
            - overlap_tokens: Overlap between fixed-size chunks (default 64)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.strategy = self.config.get("strategy", ChunkStrategy.SEMANTIC)
        self.min_tokens = self.config.get("min_chunk_tokens", 200)
        self.max_tokens = self.config.get("max_chunk_tokens", 500)
        self.overlap_tokens = self.config.get("overlap_tokens", 64)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text (words / 0.75 heuristic)."""
        words = len(text.split())
        return max(1, int(words / 0.75))

    def chunk(
        self,
        content: ExtractedContent,
        strategy: Optional[ChunkStrategy] = None,
    ) -> List[Chunk]:
        """Chunk extracted content using the specified or configured strategy."""
        strat = strategy or self.strategy
        if strat == ChunkStrategy.CODE_AWARE:
            return self._chunk_code(content)
        elif strat == ChunkStrategy.HIERARCHICAL:
            return self._chunk_hierarchical(content)
        elif strat == ChunkStrategy.FIXED_SIZE:
            return self._chunk_fixed(content)
        else:
            return self._chunk_semantic(content)

    def _chunk_semantic(self, content: ExtractedContent) -> List[Chunk]:
        """Semantic chunking: respect paragraph and sentence boundaries."""
        text = content.raw_text.strip()
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[Chunk] = []
        current_text = ""
        position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (current_text + "\n\n" + para).strip() if current_text else para
            tokens = self.estimate_tokens(candidate)

            if tokens > self.max_tokens and current_text:
                # Flush current buffer
                chunks.append(self._make_chunk(
                    current_text, position, {"strategy": "semantic"}
                ))
                position += 1
                current_text = para
            elif tokens > self.max_tokens:
                # Single paragraph too large — split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    cand2 = (current_text + " " + sent).strip() if current_text else sent
                    if self.estimate_tokens(cand2) > self.max_tokens and current_text:
                        chunks.append(self._make_chunk(
                            current_text, position, {"strategy": "semantic"}
                        ))
                        position += 1
                        current_text = sent
                    else:
                        current_text = cand2
            else:
                current_text = candidate

        if current_text.strip():
            chunks.append(self._make_chunk(
                current_text, position, {"strategy": "semantic"}
            ))

        return chunks

    def _chunk_code(self, content: ExtractedContent) -> List[Chunk]:
        """Code-aware chunking: split by function/class definitions."""
        text = content.raw_text
        structure = content.structure
        if not structure:
            # Fallback to semantic if no structure detected
            return self._chunk_semantic(content)

        lines = text.split("\n")
        chunks: List[Chunk] = []
        position = 0

        # Build ranges from structure entries
        ranges: List[Tuple[int, int, Dict[str, Any]]] = []
        for i, entry in enumerate(structure):
            start = entry["line"]
            end = structure[i + 1]["line"] if i + 1 < len(structure) else len(lines)
            ranges.append((start, end, entry))

        # Add preamble (before first definition)
        if ranges and ranges[0][0] > 0:
            preamble = "\n".join(lines[:ranges[0][0]]).strip()
            if preamble:
                chunks.append(self._make_chunk(
                    preamble, position,
                    {"strategy": "code_aware", "type": "preamble"},
                ))
                position += 1

        for start, end, entry in ranges:
            chunk_text = "\n".join(lines[start:end]).rstrip()
            if chunk_text.strip():
                meta = {
                    "strategy": "code_aware",
                    "type": entry.get("type", "unknown"),
                    "name": entry.get("name", "unknown"),
                    "language": content.metadata.get("language", "unknown"),
                }
                chunks.append(self._make_chunk(chunk_text, position, meta))
                position += 1

        if not chunks:
            return self._chunk_semantic(content)

        return chunks

    def _chunk_hierarchical(self, content: ExtractedContent) -> List[Chunk]:
        """Hierarchical chunking: follow heading structure."""
        text = content.raw_text
        structure = content.structure
        if not structure:
            return self._chunk_semantic(content)

        lines = text.split("\n")
        chunks: List[Chunk] = []
        position = 0

        # Build section ranges from headings
        heading_lines = [s["line"] for s in structure]
        parent_ids: Dict[int, str] = {}  # heading level → most recent chunk_id

        # Add preamble before first heading
        if heading_lines and heading_lines[0] > 0:
            preamble = "\n".join(lines[:heading_lines[0]]).strip()
            if preamble:
                c = self._make_chunk(
                    preamble, position,
                    {"strategy": "hierarchical", "type": "preamble"},
                )
                chunks.append(c)
                position += 1

        for i, entry in enumerate(structure):
            start = entry["line"]
            end = structure[i + 1]["line"] if i + 1 < len(structure) else len(lines)
            section_text = "\n".join(lines[start:end]).strip()

            if not section_text:
                continue

            level = entry.get("level", 1)
            parent_id = None
            # Find parent: nearest heading with lower level
            for lv in range(level - 1, 0, -1):
                if lv in parent_ids:
                    parent_id = parent_ids[lv]
                    break

            meta = {
                "strategy": "hierarchical",
                "type": "section",
                "heading": entry.get("title", ""),
                "level": level,
            }

            # If section is too large, sub-chunk it
            if self.estimate_tokens(section_text) > self.max_tokens:
                sub_chunks = self._split_large_section(
                    section_text, position, meta, parent_id
                )
                for sc in sub_chunks:
                    chunks.append(sc)
                    position += 1
                if sub_chunks:
                    parent_ids[level] = sub_chunks[0].chunk_id
            else:
                c = self._make_chunk(section_text, position, meta)
                c.parent_chunk_id = parent_id
                chunks.append(c)
                parent_ids[level] = c.chunk_id
                position += 1

        if not chunks:
            return self._chunk_semantic(content)

        return chunks

    def _chunk_fixed(self, content: ExtractedContent) -> List[Chunk]:
        """Fixed-size chunking with overlap (512 tokens, 64 overlap)."""
        text = content.raw_text.strip()
        if not text:
            return []

        words = text.split()
        # Convert token counts to word counts (approximate)
        chunk_words = int(self.max_tokens * 0.75)
        overlap_words = int(self.overlap_tokens * 0.75)

        chunks: List[Chunk] = []
        position = 0
        start = 0

        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(self._make_chunk(
                chunk_text, position, {"strategy": "fixed_size"}
            ))
            position += 1
            start += chunk_words - overlap_words
            if start >= len(words):
                break

        return chunks

    def _split_large_section(
        self,
        text: str,
        start_position: int,
        base_meta: Dict[str, Any],
        parent_id: Optional[str],
    ) -> List[Chunk]:
        """Split an oversized section into sub-chunks by paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[Chunk] = []
        current = ""
        pos = start_position
        first_id: Optional[str] = None

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            candidate = (current + "\n\n" + para).strip() if current else para
            if self.estimate_tokens(candidate) > self.max_tokens and current:
                c = self._make_chunk(current, pos, {**base_meta, "sub_chunk": True})
                c.parent_chunk_id = parent_id
                chunks.append(c)
                if first_id is None:
                    first_id = c.chunk_id
                pos += 1
                current = para
            else:
                current = candidate

        if current.strip():
            c = self._make_chunk(current, pos, {**base_meta, "sub_chunk": True})
            c.parent_chunk_id = parent_id
            chunks.append(c)

        return chunks

    def _make_chunk(
        self,
        text: str,
        position: int,
        metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a Chunk with estimated token count."""
        return Chunk(
            text=text,
            position=position,
            token_count=self.estimate_tokens(text),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Stage 3: Embedding Engine (PRD Addendum §2, Stage 3)
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Converts chunks into vector representations.

    Uses sentence-transformers/all-mpnet-base-v2 when available.
    Falls back to a deterministic hash-based embedding for environments
    without the model installed (testing, lightweight deployments).

    Supports explicit device control via the ``device`` config key:
    - ``"auto"`` (default): let sentence-transformers pick the best device
    - ``"cpu"``: force CPU inference (safe for CUDA-less environments)
    - ``"cuda"``: request GPU; falls back to CPU if CUDA is unavailable

    Caching avoids recomputation of identical text.

    Args:
        config: Embedding configuration with keys:
            - model_name: Model identifier (default "all-mpnet-base-v2")
            - dimension: Embedding dimension (default 768)
            - cache_size: Max cache entries (default 10000)
            - use_model: Force model loading (default True, falls back if unavailable)
            - device: Device selection — "auto", "cpu", or "cuda" (default "auto")
    """

    _logger = logging.getLogger("neurograph.embedding")

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.model_name = self.config.get("model_name", "all-mpnet-base-v2")
        self.dimension = self.config.get("dimension", 768)
        self.cache_size = self.config.get("cache_size", 10000)
        self.use_model = self.config.get("use_model", True)
        self.device = self.config.get("device", "auto")
        self._model = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model_available = False
        self._active_device: Optional[str] = None  # actual device after resolution
        self._fallback_reason: Optional[str] = None

        if self.use_model:
            self._try_load_model()

    def _resolve_device(self) -> str:
        """Resolve the requested device to an actual device string.

        Handles graceful degradation: if 'cuda' is requested but unavailable,
        falls back to 'cpu' with a warning rather than hard-failing.

        Returns:
            Device string suitable for SentenceTransformer (or None for auto).
        """
        if self.device == "auto":
            return None  # let sentence-transformers auto-detect
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    self._logger.warning(
                        "CUDA requested but not available (torch.cuda.is_available()=False). "
                        "Falling back to CPU."
                    )
                    return "cpu"
            except ImportError:
                self._logger.warning(
                    "CUDA requested but PyTorch not installed. Falling back to CPU."
                )
                return "cpu"
        return self.device  # "cpu" or any explicit device string

    @staticmethod
    def _suppress_provider_warnings() -> None:
        """Suppress API key warnings from HuggingFace inference providers.

        sentence-transformers v5+ and transformers v5+ added inference provider
        backends (OpenAI, Google, Voyage, etc.) that emit noisy warnings when
        their API keys aren't set — even when we only use local torch models.
        NeuroGraph uses ONLY local torch-based embeddings; TID controls all
        external API calls.  No provider API keys are needed or used.

        This sets environment variables and warning filters BEFORE import to
        prevent those warnings from reaching the user.

        Grok review v0.7.1: Broadened filters to catch all warning categories
        (not just UserWarning) and plural "KEYS" patterns that were escaping.
        """
        import os
        import warnings

        # Tell transformers to only log errors, not provider-related warnings
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        # Disable HuggingFace Hub telemetry and inference provider checks
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
        # Prevent tokenizers parallelism warning
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Filter out API key warnings that slip through from provider packages
        # (openai, google-generativeai, voyageai) if they happen to be installed.
        # Use no category restriction to catch UserWarning, FutureWarning,
        # DeprecationWarning, RuntimeWarning — providers use various types.
        # Use re.IGNORECASE-equivalent patterns for case-insensitive matching.
        for pattern in [
            r"(?i).*api.?keys?.*",
            r"(?i).*set up your.*api.*",
            r"(?i).*openai.*",
            r"(?i).*google.*api.*",
            r"(?i).*voyage.*",
            r"(?i).*inference.?provider.*",
            r"(?i).*provider.*backend.*",
            r"(?i).*api.?key.*not.?set.*",
        ]:
            warnings.filterwarnings("ignore", message=pattern)

    def _try_load_model(self) -> None:
        """Attempt to load the sentence-transformers model.

        Suppresses API key warnings from inference providers (OpenAI, Google,
        Voyage) that were added in sentence-transformers v5+ / transformers v5+.
        NeuroGraph uses local torch-based embeddings only — no external API
        keys are needed.

        Logs the outcome so silent failures are visible to operators.
        """
        # Suppress provider warnings BEFORE any HuggingFace imports
        self._suppress_provider_warnings()

        try:
            from sentence_transformers import SentenceTransformer

            resolved_device = self._resolve_device()
            # Build kwargs — use backend="torch" if the version supports it
            # (v5+) to explicitly avoid API provider backends
            kwargs: Dict[str, Any] = {}
            if resolved_device is not None:
                kwargs["device"] = resolved_device
            try:
                # sentence-transformers v5+ accepts backend parameter
                self._model = SentenceTransformer(
                    self.model_name, backend="torch", **kwargs
                )
            except TypeError:
                # Older versions don't have the backend parameter
                self._model = SentenceTransformer(self.model_name, **kwargs)

            self._model_available = True
            self._active_device = str(self._model.device)
            # Update dimension from loaded model
            self.dimension = self._model.get_sentence_embedding_dimension()
            self._logger.info(
                "Loaded embedding model '%s' on device '%s' (local torch backend)",
                self.model_name, self._active_device,
            )
        except ImportError:
            self._model_available = False
            self._fallback_reason = "sentence-transformers not installed"
            self._logger.warning(
                "sentence-transformers not installed. "
                "Using deterministic hash-based fallback embeddings."
            )
        except Exception as exc:
            self._model_available = False
            self._fallback_reason = str(exc)
            self._logger.warning(
                "Failed to load embedding model '%s': %s. "
                "Using deterministic hash-based fallback embeddings.",
                self.model_name, exc,
            )

    def embed(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Embed a list of chunks, using cache where possible.

        Returns list of EmbeddedChunk with normalized vectors.
        Individual chunk failures fall back to hash embedding rather than
        failing the entire batch (Grok review: batch resilience).
        """
        results: List[EmbeddedChunk] = []
        uncached: List[Tuple[int, Chunk]] = []

        # Check cache first
        for i, chunk in enumerate(chunks):
            cache_key = self._cache_key(chunk.text)
            if cache_key in self._cache:
                vec = self._cache[cache_key]
                # Move to end for LRU
                self._cache.move_to_end(cache_key)
                results.append(EmbeddedChunk(
                    chunk=chunk,
                    vector=vec,
                    model_name=self.model_name if self._model_available else "hash_fallback",
                ))
            else:
                results.append(None)  # type: ignore[arg-type]
                uncached.append((i, chunk))

        if uncached:
            texts = [c.text for _, c in uncached]
            try:
                vectors = self._encode_batch(texts)
            except Exception as exc:
                self._logger.warning(
                    "Batch embed failed (%s), falling back to per-chunk hash", exc,
                )
                vectors = [self._hash_embed(t) for t in texts]
            for (idx, chunk), vec in zip(uncached, vectors):
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                cache_key = self._cache_key(chunk.text)
                self._update_cache(cache_key, vec)
                results[idx] = EmbeddedChunk(
                    chunk=chunk,
                    vector=vec,
                    model_name=self.model_name if self._model_available else "hash_fallback",
                )

        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns normalized vector."""
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        vectors = self._encode_batch([text])
        vec = vectors[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._update_cache(cache_key, vec)
        return vec

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts into vectors.

        If the model is loaded but encoding fails at runtime (e.g. CUDA
        out-of-memory, driver error), falls back to hash embeddings for this
        batch rather than crashing the entire pipeline.
        """
        if self._model_available and self._model is not None:
            try:
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                return [np.array(e) for e in embeddings]
            except Exception as exc:
                self._logger.warning(
                    "Model encode failed (%s). Falling back to hash embeddings "
                    "for this batch of %d texts.",
                    exc, len(texts),
                )
                return [self._hash_embed(t) for t in texts]
        else:
            return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding fallback.

        Produces consistent vectors from text content using SHA-256 seeding.
        Useful for testing without model dependencies.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _cache_key(self, text: str) -> str:
        """Create cache key from text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _update_cache(self, key: str, vec: np.ndarray) -> None:
        """Add to cache with LRU eviction."""
        self._cache[key] = vec
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @property
    def cache_hits(self) -> int:
        """Number of entries currently in cache."""
        return len(self._cache)

    @property
    def status(self) -> Dict[str, Any]:
        """Return embedding engine status for diagnostics.

        Useful for OpenClaw and CLI tools to verify the embedding backend
        without running a full ingestion cycle.
        """
        return {
            "model_available": self._model_available,
            "model_name": self.model_name if self._model_available else "hash_fallback",
            "device_requested": self.device,
            "device_active": self._active_device,
            "dimension": self.dimension,
            "cache_entries": len(self._cache),
            "fallback_reason": self._fallback_reason,
        }


# ---------------------------------------------------------------------------
# Stage 4: Node Registrar with Novelty Dampening (PRD Addendum §4.2-4.3)
# ---------------------------------------------------------------------------

class NodeRegistrar:
    """Registers chunks as SNN nodes with novelty dampening.

    New nodes start at reduced synaptic weight and higher threshold
    (novelty dampening) to prevent destabilizing existing STDP-learned
    pathways.  During a probation period, dampening fades and the node
    graduates to full integration.

    Args:
        graph: The NeuroGraph to register nodes into.
        vector_db: SimpleVectorDB for embedding storage.
        config: Registration configuration with keys:
            - novelty_dampening: Initial weight reduction factor (default 0.3)
            - probation_period: Steps before full integration (default 10)
            - dampening_curve: DampeningCurve (default LINEAR)
            - initial_threshold_boost: Extra threshold for new nodes (default 0.2)
    """

    def __init__(
        self,
        graph: Graph,
        vector_db: SimpleVectorDB,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.vector_db = vector_db
        self.config = config or {}
        self.novelty_dampening = self.config.get("novelty_dampening", 0.3)
        self.probation_period = self.config.get("probation_period", 10)
        self.dampening_curve = self.config.get(
            "dampening_curve", DampeningCurve.LINEAR
        )
        self.initial_threshold_boost = self.config.get(
            "initial_threshold_boost", 0.2
        )

    def register(
        self,
        embedded_chunks: List[EmbeddedChunk],
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Register embedded chunks as nodes in the graph and vector DB.

        Returns list of created node IDs.
        """
        created_ids: List[str] = []
        src_meta = source_metadata or {}

        for ec in embedded_chunks:
            node_id = ec.chunk.chunk_id

            # Create node with novelty dampening
            node_meta = {
                **ec.chunk.metadata,
                **src_meta,
                "chunk_text_preview": ec.chunk.text[:200],
                "token_count": ec.chunk.token_count,
                "position": ec.chunk.position,
                "creation_mode": "ingested",
                "probation_remaining": self.probation_period,
                "novelty_dampening": self.novelty_dampening,
                "dampening_curve": self.dampening_curve.name,
            }
            if ec.chunk.parent_chunk_id:
                node_meta["parent_chunk_id"] = ec.chunk.parent_chunk_id

            node = self.graph.create_node(node_id=node_id, metadata=node_meta)

            # Apply novelty dampening: higher threshold for new nodes
            base_threshold = self.graph.config.get("default_threshold", 1.0)
            node.threshold = base_threshold + self.initial_threshold_boost

            # Reduce intrinsic excitability by dampening factor
            # dampening=0.3 means node starts at 30% effectiveness
            node.intrinsic_excitability = self.novelty_dampening

            # Store in vector DB
            self.vector_db.insert(
                id=node_id,
                embedding=ec.vector,
                content=ec.chunk.text,
                metadata=node_meta,
            )

            created_ids.append(node_id)

        return created_ids

    def get_dampening_factor(self, node: Node) -> float:
        """Compute current dampening factor for a node based on probation progress.

        Returns a value in [novelty_dampening, 1.0] that increases as the
        node progresses through its probation period.
        """
        prob_remaining = node.metadata.get("probation_remaining", 0)
        prob_total = node.metadata.get("probation_remaining", 0)

        # If no probation tracking, node is fully integrated
        initial_dampening = node.metadata.get("novelty_dampening", 1.0)
        if prob_remaining <= 0:
            return 1.0

        # Total probation = probation_period from config
        total = self.probation_period
        if total <= 0:
            return 1.0

        elapsed = total - prob_remaining
        progress = elapsed / total  # 0.0 to 1.0

        curve_name = node.metadata.get("dampening_curve", "LINEAR")
        curve = DampeningCurve[curve_name] if isinstance(curve_name, str) else self.dampening_curve

        if curve == DampeningCurve.LINEAR:
            factor = initial_dampening + (1.0 - initial_dampening) * progress
        elif curve == DampeningCurve.EXPONENTIAL:
            # Fast early growth
            factor = initial_dampening + (1.0 - initial_dampening) * (
                1.0 - math.exp(-3.0 * progress)
            )
        elif curve == DampeningCurve.LOGARITHMIC:
            # Slow early, fast late
            factor = initial_dampening + (1.0 - initial_dampening) * (
                math.log1p(progress * (math.e - 1)) / 1.0
            )
        else:
            factor = initial_dampening + (1.0 - initial_dampening) * progress

        return min(1.0, max(initial_dampening, factor))

    def update_probation(self, node_ids: Optional[List[str]] = None) -> List[str]:
        """Advance probation for nodes, graduating those that complete it.

        Call this each simulation step (or periodically) to fade dampening.
        Returns list of node IDs that graduated this call.
        """
        graduated: List[str] = []
        ids = node_ids or list(self.graph.nodes.keys())

        for nid in ids:
            node = self.graph.nodes.get(nid)
            if node is None:
                continue

            prob = node.metadata.get("probation_remaining")
            if prob is None or prob <= 0:
                continue

            # Decrement probation
            node.metadata["probation_remaining"] = prob - 1

            if node.metadata["probation_remaining"] <= 0:
                # Graduate: restore full excitability and threshold
                node.intrinsic_excitability = 1.0
                node.threshold = self.graph.config.get("default_threshold", 1.0)
                node.metadata["graduated"] = True
                graduated.append(nid)
            else:
                # Update excitability based on dampening curve
                factor = self.get_dampening_factor(node)
                node.intrinsic_excitability = factor

        return graduated


# ---------------------------------------------------------------------------
# Stage 5: Hypergraph Associator (PRD Addendum §2, Stage 5)
# ---------------------------------------------------------------------------

class HypergraphAssociator:
    """Creates initial relationships via similarity and structural cues.

    Three association strategies:
        1. Similarity-based: Vector similarity → synapses (if sim > threshold)
        2. Structural: Sequential chunks, parent-child, code def→usage
        3. Hypergraph clustering: Group related chunks into hyperedges

    Args:
        graph: The NeuroGraph to create associations in.
        config: Association configuration with keys:
            - similarity_threshold: Min cosine similarity for synapse (default 0.7)
            - similarity_weight_scale: Scale similarity to synapse weight (default 1.0)
            - sequential_weight: Weight for sequential chunk links (default 0.3)
            - parent_child_weight: Weight for parent-child links (default 0.5)
            - min_cluster_size: Min nodes for hyperedge cluster (default 3)
            - cluster_similarity_threshold: Min avg similarity for cluster (default 0.6)
    """

    def __init__(
        self,
        graph: Graph,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.similarity_weight_scale = self.config.get("similarity_weight_scale", 1.0)
        self.sequential_weight = self.config.get("sequential_weight", 0.3)
        self.parent_child_weight = self.config.get("parent_child_weight", 0.5)
        self.min_cluster_size = self.config.get("min_cluster_size", 3)
        self.cluster_sim_threshold = self.config.get(
            "cluster_similarity_threshold", 0.6
        )

    def associate(
        self,
        embedded_chunks: List[EmbeddedChunk],
        node_ids: List[str],
        vector_db: SimpleVectorDB,
    ) -> Tuple[List[str], List[str]]:
        """Create associations from embedded chunks.

        Returns:
            Tuple of (synapse_ids, hyperedge_ids) created.
        """
        synapse_ids: List[str] = []
        hyperedge_ids: List[str] = []

        # 1. Similarity-based synapses
        sim_synapses = self._create_similarity_synapses(embedded_chunks, node_ids)
        synapse_ids.extend(sim_synapses)

        # 2. Structural synapses
        struct_synapses = self._create_structural_synapses(embedded_chunks, node_ids)
        synapse_ids.extend(struct_synapses)

        # 3. Hypergraph clustering
        he_ids = self._create_clusters(embedded_chunks, node_ids)
        hyperedge_ids.extend(he_ids)

        return synapse_ids, hyperedge_ids

    def _create_similarity_synapses(
        self,
        embedded_chunks: List[EmbeddedChunk],
        node_ids: List[str],
    ) -> List[str]:
        """Create synapses between chunks with high cosine similarity."""
        created: List[str] = []
        n = len(embedded_chunks)
        if n < 2:
            return created

        # Compute pairwise similarities
        for i in range(n):
            for j in range(i + 1, n):
                vec_i = embedded_chunks[i].vector
                vec_j = embedded_chunks[j].vector
                sim = float(np.dot(vec_i, vec_j))

                if sim >= self.similarity_threshold:
                    weight = sim * self.similarity_weight_scale
                    weight = min(weight, self.graph.config.get("max_weight", 5.0))

                    # Apply novelty dampening to initial weight
                    nid_i = node_ids[i]
                    nid_j = node_ids[j]
                    dampening_i = self._get_node_dampening(nid_i)
                    dampening_j = self._get_node_dampening(nid_j)
                    dampened_weight = weight * min(dampening_i, dampening_j)

                    # Bidirectional: create synapse in both directions
                    syn1 = self.graph.create_synapse(
                        nid_i, nid_j,
                        weight=dampened_weight,
                    )
                    syn1.metadata["creation_mode"] = "similarity"
                    syn1.metadata["similarity"] = sim
                    created.append(syn1.synapse_id)

                    syn2 = self.graph.create_synapse(
                        nid_j, nid_i,
                        weight=dampened_weight,
                    )
                    syn2.metadata["creation_mode"] = "similarity"
                    syn2.metadata["similarity"] = sim
                    created.append(syn2.synapse_id)

        return created

    def _create_structural_synapses(
        self,
        embedded_chunks: List[EmbeddedChunk],
        node_ids: List[str],
    ) -> List[str]:
        """Create synapses from structural relationships.

        - Sequential: adjacent chunks in original document
        - Parent-child: hierarchical chunk relationships
        - Code: definition → usage links
        """
        created: List[str] = []

        # Sort by position for sequential linking
        sorted_pairs = sorted(
            zip(embedded_chunks, node_ids),
            key=lambda x: x[0].chunk.position,
        )

        # Sequential links
        for i in range(len(sorted_pairs) - 1):
            ec_a, nid_a = sorted_pairs[i]
            ec_b, nid_b = sorted_pairs[i + 1]

            dampening = min(
                self._get_node_dampening(nid_a),
                self._get_node_dampening(nid_b),
            )
            weight = self.sequential_weight * dampening

            syn = self.graph.create_synapse(nid_a, nid_b, weight=weight)
            syn.metadata["creation_mode"] = "sequential"
            created.append(syn.synapse_id)

        # Parent-child links
        for ec, nid in zip(embedded_chunks, node_ids):
            parent_id = ec.chunk.parent_chunk_id
            if parent_id and parent_id in self.graph.nodes:
                dampening = min(
                    self._get_node_dampening(nid),
                    self._get_node_dampening(parent_id),
                )
                weight = self.parent_child_weight * dampening

                # Parent → child
                syn = self.graph.create_synapse(parent_id, nid, weight=weight)
                syn.metadata["creation_mode"] = "parent_child"
                created.append(syn.synapse_id)

        # Code definition → usage links
        created.extend(self._create_code_links(embedded_chunks, node_ids))

        return created

    def _create_code_links(
        self,
        embedded_chunks: List[EmbeddedChunk],
        node_ids: List[str],
    ) -> List[str]:
        """Link code definitions to chunks that reference them."""
        created: List[str] = []

        # Collect definition names
        definitions: Dict[str, str] = {}  # name → node_id
        for ec, nid in zip(embedded_chunks, node_ids):
            meta = ec.chunk.metadata
            if meta.get("strategy") == "code_aware" and meta.get("name"):
                name = meta["name"]
                definitions[name] = nid

        if not definitions:
            return created

        # Find usage references
        for ec, nid in zip(embedded_chunks, node_ids):
            for def_name, def_nid in definitions.items():
                if def_nid == nid:
                    continue  # Skip self-reference
                # Check if this chunk references the definition
                if def_name in ec.chunk.text:
                    dampening = min(
                        self._get_node_dampening(def_nid),
                        self._get_node_dampening(nid),
                    )
                    weight = 0.4 * dampening

                    syn = self.graph.create_synapse(def_nid, nid, weight=weight)
                    syn.metadata["creation_mode"] = "code_def_usage"
                    syn.metadata["definition"] = def_name
                    created.append(syn.synapse_id)

        return created

    def _create_clusters(
        self,
        embedded_chunks: List[EmbeddedChunk],
        node_ids: List[str],
    ) -> List[str]:
        """Create hyperedge clusters from highly similar groups.

        Uses a simple greedy clustering: for each node, find all nodes
        above the cluster similarity threshold and form a hyperedge if
        the group meets minimum size.
        """
        created: List[str] = []
        n = len(embedded_chunks)
        if n < self.min_cluster_size:
            return created

        # Build similarity matrix
        used_in_cluster: Set[int] = set()
        clusters: List[Set[int]] = []

        for i in range(n):
            if i in used_in_cluster:
                continue
            cluster = {i}
            for j in range(i + 1, n):
                if j in used_in_cluster:
                    continue
                sim = float(np.dot(
                    embedded_chunks[i].vector,
                    embedded_chunks[j].vector,
                ))
                if sim >= self.cluster_sim_threshold:
                    cluster.add(j)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                used_in_cluster.update(cluster)

        # Create hyperedges for clusters
        for cluster_indices in clusters:
            member_ids = {node_ids[i] for i in cluster_indices}
            member_weights = {nid: 1.0 for nid in member_ids}

            he = self.graph.create_hyperedge(
                member_node_ids=member_ids,
                member_weights=member_weights,
                activation_threshold=0.6,
                activation_mode=ActivationMode.WEIGHTED_THRESHOLD,
                metadata={
                    "creation_mode": "ingestion_cluster",
                    "cluster_size": len(member_ids),
                },
            )
            created.append(he.hyperedge_id)

        return created

    def _get_node_dampening(self, node_id: str) -> float:
        """Get current novelty dampening factor for a node."""
        node = self.graph.nodes.get(node_id)
        if node is None:
            return 1.0
        return node.intrinsic_excitability


# ---------------------------------------------------------------------------
# Project Configurations (PRD Addendum §3)
# ---------------------------------------------------------------------------

OPENCLAW_INGESTOR_CONFIG = {
    "extraction": {},
    "chunking": {
        "strategy": ChunkStrategy.CODE_AWARE,
        "min_chunk_tokens": 100,
        "max_chunk_tokens": 500,
    },
    "embedding": {
        "model_name": "all-mpnet-base-v2",
        "use_model": True,
        "device": "auto",
    },
    "registration": {
        "novelty_dampening": 0.3,
        "probation_period": 10,
        "dampening_curve": DampeningCurve.LINEAR,
        "initial_threshold_boost": 0.15,
    },
    "association": {
        "similarity_threshold": 0.7,
        "similarity_weight_scale": 1.0,
        "sequential_weight": 0.3,
        "parent_child_weight": 0.5,
        "min_cluster_size": 3,
        "cluster_similarity_threshold": 0.65,
    },
}

DSM_INGESTOR_CONFIG = {
    "extraction": {},
    "chunking": {
        "strategy": ChunkStrategy.HIERARCHICAL,
        "min_chunk_tokens": 200,
        "max_chunk_tokens": 500,
    },
    "embedding": {
        "model_name": "all-mpnet-base-v2",
        "use_model": True,
    },
    "registration": {
        "novelty_dampening": 0.05,
        "probation_period": 100,
        "dampening_curve": DampeningCurve.LOGARITHMIC,
        "initial_threshold_boost": 0.3,
    },
    "association": {
        "similarity_threshold": 0.8,
        "similarity_weight_scale": 0.8,
        "sequential_weight": 0.2,
        "parent_child_weight": 0.6,
        "min_cluster_size": 3,
        "cluster_similarity_threshold": 0.75,
    },
}

CONSCIOUSNESS_INGESTOR_CONFIG = {
    "extraction": {},
    "chunking": {
        "strategy": ChunkStrategy.SEMANTIC,
        "min_chunk_tokens": 200,
        "max_chunk_tokens": 500,
    },
    "embedding": {
        "model_name": "all-mpnet-base-v2",
        "use_model": True,
    },
    "registration": {
        "novelty_dampening": 0.01,
        "probation_period": 500,
        "dampening_curve": DampeningCurve.EXPONENTIAL,
        "initial_threshold_boost": 0.4,
    },
    "association": {
        "similarity_threshold": 0.65,
        "similarity_weight_scale": 1.2,
        "sequential_weight": 0.25,
        "parent_child_weight": 0.4,
        "min_cluster_size": 3,
        "cluster_similarity_threshold": 0.55,
    },
}


def get_ingestor_config(project: str) -> Dict[str, Any]:
    """Get ingestor configuration for a named project.

    Args:
        project: One of "openclaw", "dsm", "consciousness".

    Returns:
        Configuration dictionary suitable for UniversalIngestor.
    """
    configs = {
        "openclaw": OPENCLAW_INGESTOR_CONFIG,
        "dsm": DSM_INGESTOR_CONFIG,
        "consciousness": CONSCIOUSNESS_INGESTOR_CONFIG,
    }
    name = project.lower().strip()
    if name not in configs:
        raise ValueError(
            f"Unknown project '{project}'. Choose from: {list(configs.keys())}"
        )
    return configs[name]


# ---------------------------------------------------------------------------
# Universal Ingestor Coordinator (PRD Addendum §4.1)
# ---------------------------------------------------------------------------

class IngestorConfig:
    """Configuration container for the Universal Ingestor.

    Wraps a nested dictionary with attribute access for pipeline stage configs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    @property
    def extraction(self) -> Dict[str, Any]:
        return self._config.get("extraction", {})

    @property
    def chunking(self) -> Dict[str, Any]:
        return self._config.get("chunking", {})

    @property
    def embedding(self) -> Dict[str, Any]:
        return self._config.get("embedding", {})

    @property
    def registration(self) -> Dict[str, Any]:
        return self._config.get("registration", {})

    @property
    def association(self) -> Dict[str, Any]:
        return self._config.get("association", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


class UniversalIngestor:
    """Main coordinator that orchestrates the five-stage ingestion pipeline
    (PRD Addendum §4.1).

    Pipeline: Extract → Chunk → Embed → Register → Associate

    Example::

        graph = Graph()
        vector_db = SimpleVectorDB()
        ingestor = UniversalIngestor(graph, vector_db)
        result = ingestor.ingest("# Hello\\n\\nSome markdown content")

    Args:
        neuro_graph: The NeuroGraph Graph instance.
        vector_db: SimpleVectorDB instance for embedding storage.
        config: Configuration dict or IngestorConfig (see project configs).
    """

    def __init__(
        self,
        neuro_graph: Graph,
        vector_db: SimpleVectorDB,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.neuro_graph = neuro_graph
        self.vector_db = vector_db

        if isinstance(config, IngestorConfig):
            self.config = config
        else:
            self.config = IngestorConfig(config)

        # Pipeline stages
        self.extractor = ExtractorRouter(self.config.extraction)
        self.chunker = AdaptiveChunker(self.config.chunking)
        self.embedder = EmbeddingEngine(self.config.embedding)
        self.registrar = NodeRegistrar(
            neuro_graph, vector_db, self.config.registration
        )
        self.associator = HypergraphAssociator(neuro_graph, self.config.association)

        # State tracking
        self.ingestion_log: List[IngestionResult] = []

    def ingest(
        self,
        source: str,
        source_type: Optional[SourceType] = None,
        **kwargs: Any,
    ) -> IngestionResult:
        """Main ingestion entry point. Executes all 5 pipeline stages.

        Args:
            source: Input data — text content, file path, or URL.
            source_type: Explicit source type (auto-detected if None).
            **kwargs: Additional arguments passed to the extractor.

        Returns:
            IngestionResult with details of created nodes, synapses, and hyperedges.
        """
        # Stage 1: Extract
        extracted = self.extractor.extract(source, source_type=source_type, **kwargs)

        # Stage 2: Chunk
        chunks = self.chunker.chunk(extracted)

        if not chunks:
            result = IngestionResult(
                source=source[:200],
                source_type=extracted.source_type,
                chunks_created=0,
            )
            self.ingestion_log.append(result)
            return result

        # Stage 3: Embed
        cache_before = self.embedder.cache_hits
        embedded_chunks = self.embedder.embed(chunks)
        cache_hits = max(0, self.embedder.cache_hits - cache_before - len(chunks))

        # Stage 4: Register
        source_meta = {
            "source": source[:200],
            "source_type": extracted.source_type.name,
        }
        node_ids = self.registrar.register(embedded_chunks, source_meta)

        # Stage 5: Associate
        synapse_ids, hyperedge_ids = self.associator.associate(
            embedded_chunks, node_ids, self.vector_db
        )

        result = IngestionResult(
            source=source[:200],
            source_type=extracted.source_type,
            chunks_created=len(chunks),
            nodes_created=node_ids,
            synapses_created=synapse_ids,
            hyperedges_created=hyperedge_ids,
            embeddings_cached=cache_hits,
            metadata={
                "extraction_metadata": extracted.metadata,
                "structure_count": len(extracted.structure),
            },
        )
        self.ingestion_log.append(result)
        return result

    def ingest_batch(
        self,
        sources: List[Tuple[str, Optional[SourceType]]],
    ) -> List[IngestionResult]:
        """Ingest multiple sources sequentially.

        Args:
            sources: List of (source, source_type) tuples.

        Returns:
            List of IngestionResult objects.
        """
        results = []
        for source, stype in sources:
            result = self.ingest(source, source_type=stype)
            results.append(result)
        return results

    def update_probation(self) -> List[str]:
        """Advance probation for all ingested nodes.

        Should be called after each Graph.step() to fade novelty dampening.

        Returns:
            List of node IDs that graduated this call.
        """
        return self.registrar.update_probation()

    def query_similar(
        self,
        text: str,
        k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Query the vector DB for chunks similar to the given text.

        Args:
            text: Query text.
            k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of result dicts with id, similarity, content, metadata.
        """
        query_vec = self.embedder.embed_text(text)
        results = self.vector_db.search(query_vec, k=k, threshold=threshold)

        output = []
        for id, similarity in results:
            entry = self.vector_db.get(id)
            if entry:
                output.append({
                    "id": id,
                    "similarity": similarity,
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                })
        return output
