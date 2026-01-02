"""Tests for embedding generation."""

import pytest

from src.chunker.chunker import Chunk
from src.chunker.embedder import (
    ChunkEmbedder,
    EmbeddedChunk,
    MockEmbedder,
    cosine_similarity,
    find_similar_chunks,
)


class TestMockEmbedder:
    """Tests for MockEmbedder class."""

    def test_initialization(self):
        """Should initialize with default values."""
        embedder = MockEmbedder()
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.get_embedding_dim() == 384

    def test_custom_dimension(self):
        """Should accept custom dimension."""
        embedder = MockEmbedder(embedding_dim=768)
        assert embedder.get_embedding_dim() == 768

    def test_embed_text_returns_correct_dimension(self):
        """Should return embedding of correct dimension."""
        embedder = MockEmbedder(embedding_dim=384)
        result = embedder.embed_text("Hello world")
        assert len(result) == 384

    def test_embed_text_deterministic(self):
        """Same text should produce same embedding."""
        embedder = MockEmbedder()
        e1 = embedder.embed_text("Hello world")
        e2 = embedder.embed_text("Hello world")
        assert e1 == e2

    def test_embed_text_different_for_different_text(self):
        """Different text should produce different embeddings."""
        embedder = MockEmbedder()
        e1 = embedder.embed_text("Hello")
        e2 = embedder.embed_text("World")
        assert e1 != e2

    def test_embed_empty_text(self):
        """Should return zero vector for empty text."""
        embedder = MockEmbedder(embedding_dim=10)
        result = embedder.embed_text("")
        assert result == [0.0] * 10

    def test_embed_whitespace_text(self):
        """Should return zero vector for whitespace."""
        embedder = MockEmbedder(embedding_dim=10)
        result = embedder.embed_text("   ")
        assert result == [0.0] * 10

    def test_embeddings_are_normalized(self):
        """Embeddings should be unit vectors."""
        embedder = MockEmbedder()
        embedding = embedder.embed_text("Test text")
        norm = sum(v * v for v in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.001

    def test_embed_texts_generator(self):
        """Should generate embeddings for multiple texts."""
        embedder = MockEmbedder(embedding_dim=10)
        texts = ["Text 1", "Text 2", "Text 3"]
        results = list(embedder.embed_texts(texts))
        assert len(results) == 3
        for r in results:
            assert len(r) == 10

    def test_embed_chunk(self):
        """Should embed a chunk."""
        embedder = MockEmbedder()
        chunk = Chunk(id="c1", content="Test content", level=2, token_count=5)
        result = embedder.embed_chunk(chunk)

        assert isinstance(result, EmbeddedChunk)
        assert result.chunk is chunk
        assert len(result.embedding) == 384
        assert result.model_name == embedder.model_name

    def test_embed_chunks(self):
        """Should embed multiple chunks."""
        embedder = MockEmbedder()
        chunks = [
            Chunk(id="c1", content="First", level=2, token_count=5),
            Chunk(id="c2", content="Second", level=2, token_count=5),
        ]
        results = embedder.embed_chunks(chunks)

        assert len(results) == 2
        assert all(isinstance(r, EmbeddedChunk) for r in results)

    def test_embed_chunks_empty_list(self):
        """Should handle empty chunk list."""
        embedder = MockEmbedder()
        results = embedder.embed_chunks([])
        assert results == []


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk dataclass."""

    def test_create_embedded_chunk(self):
        """Should create embedded chunk."""
        chunk = Chunk(id="c1", content="Test", level=2, token_count=5)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )
        assert ec.chunk is chunk
        assert ec.embedding == [0.1, 0.2, 0.3]
        assert ec.model_name == "test-model"

    def test_embedding_dim_property(self):
        """Should return embedding dimension."""
        chunk = Chunk(id="c1", content="Test", level=2, token_count=5)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1] * 384,
            model_name="test",
        )
        assert ec.embedding_dim == 384

    def test_to_dict(self):
        """Should convert to dictionary."""
        chunk = Chunk(
            id="c1",
            content="Test content",
            level=2,
            token_count=5,
            parent_id="p1",
            section_path=["A", "B"],
        )
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2],
            model_name="model",
        )
        d = ec.to_dict()

        assert d["chunk_id"] == "c1"
        assert d["content"] == "Test content"
        assert d["embedding"] == [0.1, 0.2]
        assert d["model_name"] == "model"
        assert d["level"] == 2
        assert d["parent_id"] == "p1"
        assert d["section_path"] == ["A", "B"]


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert abs(cosine_similarity(v1, v2) + 1.0) < 0.001

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        v1 = [1.0, 0.5, 0.0]
        v2 = [1.0, 0.6, 0.0]
        sim = cosine_similarity(v1, v2)
        assert sim > 0.9

    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        assert cosine_similarity(v1, v2) == 0.0

    def test_different_dimensions_raises(self):
        """Should raise for vectors of different dimensions."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            cosine_similarity(v1, v2)


class TestFindSimilarChunks:
    """Tests for find_similar_chunks function."""

    def test_finds_most_similar(self):
        """Should return most similar chunks first."""
        embedder = MockEmbedder(embedding_dim=10)

        chunks = [
            Chunk(id="c1", content="apple fruit", level=2, token_count=5),
            Chunk(id="c2", content="banana fruit", level=2, token_count=5),
            Chunk(id="c3", content="car vehicle", level=2, token_count=5),
        ]
        embedded = embedder.embed_chunks(chunks)

        # Query similar to "apple fruit"
        query = embedder.embed_text("apple fruit")
        results = find_similar_chunks(query, embedded, top_k=3)

        # First result should be exact match
        assert results[0][0].chunk.id == "c1"
        assert results[0][1] > 0.99  # Near-identical

    def test_respects_top_k(self):
        """Should return at most top_k results."""
        embedder = MockEmbedder(embedding_dim=10)
        chunks = [
            Chunk(id=f"c{i}", content=f"Content {i}", level=2, token_count=5)
            for i in range(10)
        ]
        embedded = embedder.embed_chunks(chunks)

        query = embedder.embed_text("Content 5")
        results = find_similar_chunks(query, embedded, top_k=3)
        assert len(results) <= 3

    def test_respects_threshold(self):
        """Should filter by similarity threshold."""
        embedder = MockEmbedder(embedding_dim=10)
        chunks = [
            Chunk(id="c1", content="exact match", level=2, token_count=5),
            Chunk(id="c2", content="completely different", level=2, token_count=5),
        ]
        embedded = embedder.embed_chunks(chunks)

        query = embedder.embed_text("exact match")
        # Exact match should have similarity 1.0 (passes any threshold)
        results = find_similar_chunks(query, embedded, top_k=10, threshold=0.9999)
        # At least the exact match should be returned
        assert len(results) >= 1
        assert results[0][0].chunk.id == "c1"
        assert results[0][1] > 0.999  # First result is near-identical

    def test_empty_chunks_list(self):
        """Should return empty for empty chunk list."""
        query = [0.1] * 10
        results = find_similar_chunks(query, [], top_k=5)
        assert results == []

    def test_returns_tuples(self):
        """Should return (EmbeddedChunk, similarity) tuples."""
        embedder = MockEmbedder(embedding_dim=10)
        chunks = [Chunk(id="c1", content="Test", level=2, token_count=5)]
        embedded = embedder.embed_chunks(chunks)

        query = embedder.embed_text("Test")
        results = find_similar_chunks(query, embedded, top_k=1)

        assert len(results) == 1
        assert isinstance(results[0][0], EmbeddedChunk)
        assert isinstance(results[0][1], float)


class TestChunkEmbedderInitialization:
    """Tests for ChunkEmbedder initialization (without loading model)."""

    def test_initialization(self):
        """Should initialize without loading model."""
        embedder = ChunkEmbedder()
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder._initialized is False

    def test_custom_model_name(self):
        """Should accept custom model name."""
        embedder = ChunkEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_embedding_dim_without_init(self):
        """Should return dimension without initializing model."""
        embedder = ChunkEmbedder()
        dim = embedder.get_embedding_dim()
        assert dim == 384  # Default for bge-small

    def test_get_embedding_dim_custom_model(self):
        """Should return correct dimension for known models."""
        embedder = ChunkEmbedder(model_name="BAAI/bge-base-en-v1.5")
        assert embedder.get_embedding_dim() == 768

        embedder = ChunkEmbedder(model_name="BAAI/bge-large-en-v1.5")
        assert embedder.get_embedding_dim() == 1024
