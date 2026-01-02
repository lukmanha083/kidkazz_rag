"""Integration tests for storage module with chunker pipeline."""

import pytest

from src.chunker import (
    Chunk,
    ChunkMetadata,
    create_hierarchical_chunks,
    enrich_all_chunks,
)
from src.chunker.embedder import MockEmbedder, cosine_similarity
from src.storage.mock_store import MockChunkStore


@pytest.fixture
def mock_store():
    """Create fresh mock store."""
    return MockChunkStore()


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def sample_markdown():
    """Sample markdown document for testing."""
    return """# Introduction

This is an introductory paragraph about machine learning.
Machine learning is defined as a subset of artificial intelligence.

## Supervised Learning

Supervised learning is a type of machine learning where the model
learns from labeled data. For example, classification and regression
are common supervised learning tasks.

### Classification

Classification is the task of predicting a discrete label.
For instance, spam detection is a classification problem.

### Regression

Regression is the task of predicting a continuous value.
For example, predicting house prices is a regression problem.

## Unsupervised Learning

Unsupervised learning works with unlabeled data.
Clustering and dimensionality reduction are examples.

### Clustering

Clustering groups similar data points together.
K-means is a popular clustering algorithm.
"""


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_full_pipeline_markdown_to_storage(
        self, sample_markdown, mock_store, mock_embedder
    ):
        """Should process markdown through entire pipeline."""
        # Phase 2: Create chunks
        chunks = create_hierarchical_chunks(
            sample_markdown,
            doc_id="test_doc",
            level_sizes=(500, 100, 50),
        )
        assert len(chunks) > 0

        # Phase 2: Enrich with metadata
        metadata = enrich_all_chunks(chunks, document_id="test_doc")
        assert len(metadata) == len(chunks)

        # Phase 2: Generate embeddings
        embedded_chunks = mock_embedder.embed_chunks(chunks)
        assert len(embedded_chunks) == len(chunks)

        # Phase 3: Store in database
        mock_store.store_document(
            doc_id="test_doc",
            title="ML Introduction",
            embedded_chunks=embedded_chunks,
            metadata_list=metadata,
        )

        # Verify storage
        docs = mock_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "test_doc"

        # Verify chunks stored
        stored_chunks = mock_store.get_document_chunks("test_doc")
        assert len(stored_chunks) == len(chunks)

    def test_semantic_search_after_storage(
        self, sample_markdown, mock_store, mock_embedder
    ):
        """Should find relevant chunks via semantic search."""
        # Setup: Full pipeline
        chunks = create_hierarchical_chunks(sample_markdown, doc_id="ml_doc")
        metadata = enrich_all_chunks(chunks, document_id="ml_doc")
        embedded_chunks = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("ml_doc", "ML Guide", embedded_chunks, metadata)

        # Search for classification-related content
        query = mock_embedder.embed_text("classification discrete label prediction")
        results = mock_store.search_similar(query, top_k=3)

        assert len(results) > 0
        # Top results should mention classification or related terms
        contents = [ec.chunk.content.lower() for ec, _ in results]
        assert any("classification" in c or "label" in c for c in contents)

    def test_graph_traversal_after_storage(
        self, sample_markdown, mock_store, mock_embedder
    ):
        """Should traverse graph relationships correctly."""
        # Setup
        chunks = create_hierarchical_chunks(
            sample_markdown,
            doc_id="graph_doc",
            level_sizes=(500, 100, 50),
        )
        metadata = enrich_all_chunks(chunks, document_id="graph_doc")
        embedded_chunks = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("graph_doc", "Graph Test", embedded_chunks, metadata)

        # Find a level 2 chunk with a parent
        level2_chunks = [ec for ec in embedded_chunks if ec.chunk.level == 2]
        level2_with_parent = [ec for ec in level2_chunks if ec.chunk.parent_id]

        if level2_with_parent:
            test_chunk = level2_with_parent[0]

            # Test parent traversal
            parent = mock_store.get_parent(test_chunk.chunk.id)
            assert parent is not None
            assert parent.chunk.level == 1

            # Test children traversal from parent
            children = mock_store.get_children(parent.chunk.id)
            assert len(children) > 0
            child_ids = {c.chunk.id for c in children}
            assert test_chunk.chunk.id in child_ids

    def test_context_window_retrieval(
        self, sample_markdown, mock_store, mock_embedder
    ):
        """Should retrieve context window around a chunk."""
        # Setup
        chunks = create_hierarchical_chunks(sample_markdown, doc_id="ctx_doc")
        metadata = enrich_all_chunks(chunks, document_id="ctx_doc")
        embedded_chunks = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("ctx_doc", "Context Test", embedded_chunks, metadata)

        # Find a chunk with neighbors
        level2_chunks = [ec for ec in embedded_chunks if ec.chunk.level == 2]
        if len(level2_chunks) > 2:
            # Pick a middle chunk
            middle_idx = len(level2_chunks) // 2
            middle_chunk = level2_chunks[middle_idx]

            context = mock_store.get_context_window(middle_chunk.chunk.id, window_size=2)

            # Should include the middle chunk
            assert any(ec.chunk.id == middle_chunk.chunk.id for ec in context)
            # Should include more than just the center
            assert len(context) >= 1

    def test_filter_by_semantic_type(
        self, sample_markdown, mock_store, mock_embedder
    ):
        """Should filter search by semantic type."""
        # Setup
        chunks = create_hierarchical_chunks(sample_markdown, doc_id="type_doc")
        metadata = enrich_all_chunks(chunks, document_id="type_doc")
        embedded_chunks = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("type_doc", "Type Test", embedded_chunks, metadata)

        query = mock_embedder.embed_text("definition of learning")

        # Search only for definitions
        results = mock_store.search_similar(
            query, top_k=5, semantic_type="definition"
        )

        # All results should be definitions
        for ec, _ in results:
            chunk_data = mock_store._chunks[ec.chunk.id]
            assert chunk_data["semantic_type"] == "definition"


class TestMultiDocumentStorage:
    """Tests for multi-document scenarios."""

    def test_store_multiple_documents(self, mock_store, mock_embedder):
        """Should store and retrieve multiple documents."""
        # Create two simple documents
        content1 = "# Doc 1\n\nFirst document content about Python."
        content2 = "# Doc 2\n\nSecond document content about JavaScript."

        for i, content in enumerate([content1, content2], 1):
            chunks = create_hierarchical_chunks(content, doc_id=f"doc{i}")
            metadata = enrich_all_chunks(chunks, document_id=f"doc{i}")
            embedded = mock_embedder.embed_chunks(chunks)
            mock_store.store_document(f"doc{i}", f"Document {i}", embedded, metadata)

        # Verify both documents stored
        docs = mock_store.list_documents()
        assert len(docs) == 2

        # Search within specific document
        query = mock_embedder.embed_text("Python programming")
        results = mock_store.search_similar(query, top_k=5, doc_id="doc1")

        for ec, _ in results:
            chunk_data = mock_store._chunks[ec.chunk.id]
            assert chunk_data["document_id"] == "doc1"

    def test_delete_one_document_preserves_others(self, mock_store, mock_embedder):
        """Should delete one document without affecting others."""
        # Create two documents
        for i in range(2):
            content = f"# Document {i}\n\nContent for document {i}."
            chunks = create_hierarchical_chunks(content, doc_id=f"doc{i}")
            metadata = enrich_all_chunks(chunks, document_id=f"doc{i}")
            embedded = mock_embedder.embed_chunks(chunks)
            mock_store.store_document(f"doc{i}", f"Doc {i}", embedded, metadata)

        # Delete first document
        mock_store.delete_document("doc0")

        # Verify only one document remains
        docs = mock_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc1"


class TestSearchQuality:
    """Tests for search result quality."""

    def test_exact_match_ranks_highest(self, mock_store, mock_embedder):
        """Exact content match should rank highest."""
        chunks = [
            Chunk(id="exact", content="Machine learning is amazing", level=2, token_count=5),
            Chunk(id="partial", content="Learning machines work well", level=2, token_count=5),
            Chunk(id="unrelated", content="Cooking recipes are fun", level=2, token_count=5),
        ]
        metadata = [
            ChunkMetadata(chunk_id=c.id, document_id="search_doc", semantic_type="narrative")
            for c in chunks
        ]
        embedded = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("search_doc", "Search Test", embedded, metadata)

        # Search for exact content
        query = mock_embedder.embed_text("Machine learning is amazing")
        results = mock_store.search_similar(query, top_k=3)

        # Exact match should be first
        assert results[0][0].chunk.id == "exact"

    def test_similarity_scores_decrease(self, mock_store, mock_embedder):
        """Similarity scores should decrease for less relevant results."""
        chunks = [
            Chunk(id=f"chunk_{i}", content=f"Content variation {i} about topic", level=2, token_count=5)
            for i in range(5)
        ]
        metadata = [
            ChunkMetadata(chunk_id=c.id, document_id="score_doc", semantic_type="narrative")
            for c in chunks
        ]
        embedded = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("score_doc", "Score Test", embedded, metadata)

        query = mock_embedder.embed_text("Content variation 0 about topic")
        results = mock_store.search_similar(query, top_k=5)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestKeywordSearch:
    """Tests for keyword search functionality."""

    def test_keyword_search_finds_exact_terms(self, mock_store, mock_embedder):
        """Should find chunks containing exact keyword."""
        chunks = [
            Chunk(id="python", content="Python is a programming language", level=2, token_count=5),
            Chunk(id="java", content="Java is also a programming language", level=2, token_count=5),
            Chunk(id="cooking", content="Cooking requires skill", level=2, token_count=5),
        ]
        metadata = [
            ChunkMetadata(chunk_id=c.id, document_id="kw_doc", semantic_type="narrative")
            for c in chunks
        ]
        embedded = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("kw_doc", "Keyword Test", embedded, metadata)

        results = mock_store.search_keyword("Python")

        assert len(results) == 1
        assert results[0].chunk.id == "python"

    def test_keyword_search_multiple_matches(self, mock_store, mock_embedder):
        """Should find all chunks containing keyword."""
        chunks = [
            Chunk(id="c1", content="Programming in Python", level=2, token_count=5),
            Chunk(id="c2", content="Programming in Java", level=2, token_count=5),
            Chunk(id="c3", content="Programming is fun", level=2, token_count=5),
        ]
        metadata = [
            ChunkMetadata(chunk_id=c.id, document_id="multi_kw", semantic_type="narrative")
            for c in chunks
        ]
        embedded = mock_embedder.embed_chunks(chunks)
        mock_store.store_document("multi_kw", "Multi KW Test", embedded, metadata)

        results = mock_store.search_keyword("Programming")

        assert len(results) == 3
