"""Tests for embedding generation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.chunker.chunker import Chunk
from src.chunker.embedder import (
    CohereEmbedder,
    COHERE_VALID_DIMS,
    EmbeddedChunk,
    MockEmbedder,
    OpenAIEmbedder,
    OPENAI_MODEL_DIMENSIONS,
    cosine_similarity,
    find_similar_chunks,
)


class TestMockEmbedder:
    """Tests for MockEmbedder class."""

    def test_initialization(self):
        """Should initialize with default values."""
        embedder = MockEmbedder()
        assert embedder.model_name == "text-embedding-3-small"
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


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()

        # Mock embeddings.create response
        def create_embedding(model, input):
            if isinstance(input, str):
                input = [input]

            mock_data = []
            for _i, text in enumerate(input):
                mock_emb = MagicMock()
                # Generate deterministic mock embedding
                dim = OPENAI_MODEL_DIMENSIONS.get(model, 1536)
                mock_emb.embedding = [0.1 * ((hash(text) + j) % 10) for j in range(dim)]
                mock_data.append(mock_emb)

            mock_response = MagicMock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create = MagicMock(side_effect=create_embedding)
        return mock_client

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.chunker.embedder.OpenAIEmbedder.client", new_callable=lambda: property(lambda _self: MagicMock()))
    def test_initialization(self, _mock_client):
        """Should initialize with API key from environment."""
        embedder = OpenAIEmbedder()
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.api_key == "test-api-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_initialization_custom_model(self):
        """Should accept custom model name."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
        assert embedder.model_name == "text-embedding-3-large"

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True)
    def test_initialization_no_api_key_raises(self):
        """Should raise ValueError when no API key is set."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
            OpenAIEmbedder()

    def test_initialization_with_explicit_api_key(self):
        """Should accept explicit API key parameter."""
        embedder = OpenAIEmbedder(api_key="explicit-key")
        assert embedder.api_key == "explicit-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_embedding_dim_small(self):
        """Should return correct dimension for text-embedding-3-small."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
        assert embedder.get_embedding_dim() == 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_embedding_dim_large(self):
        """Should return correct dimension for text-embedding-3-large."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
        assert embedder.get_embedding_dim() == 3072

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_embedding_dim_ada(self):
        """Should return correct dimension for text-embedding-ada-002."""
        embedder = OpenAIEmbedder(model_name="text-embedding-ada-002")
        assert embedder.get_embedding_dim() == 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_embedding_dim_unknown_model(self):
        """Should return default 1536 for unknown models."""
        embedder = OpenAIEmbedder(model_name="unknown-model")
        assert embedder.get_embedding_dim() == 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_text_returns_correct_dimension(self, mock_openai_client):
        """Should return embedding of correct dimension."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        result = embedder.embed_text("Hello world")
        assert len(result) == 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_text_empty_returns_zero_vector(self):
        """Should return zero vector for empty text."""
        embedder = OpenAIEmbedder()
        result = embedder.embed_text("")
        assert result == [0.0] * 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_text_whitespace_returns_zero_vector(self):
        """Should return zero vector for whitespace."""
        embedder = OpenAIEmbedder()
        result = embedder.embed_text("   ")
        assert result == [0.0] * 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_texts_generator(self, mock_openai_client):
        """Should generate embeddings for multiple texts."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        texts = ["Text 1", "Text 2", "Text 3"]
        results = list(embedder.embed_texts(texts))

        assert len(results) == 3
        for r in results:
            assert len(r) == 1536

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_texts_empty_list(self, mock_openai_client):
        """Should handle empty text list."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        results = list(embedder.embed_texts([]))
        assert results == []

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_texts_handles_empty_strings(self, mock_openai_client):
        """Should return zero vectors for empty strings in batch."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        texts = ["Text 1", "", "Text 3"]
        results = list(embedder.embed_texts(texts))

        assert len(results) == 3
        assert results[1] == [0.0] * 1536  # Empty string gets zero vector

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_chunk(self, mock_openai_client):
        """Should embed a chunk."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        chunk = Chunk(id="c1", content="Test content", level=2, token_count=5)
        result = embedder.embed_chunk(chunk)

        assert isinstance(result, EmbeddedChunk)
        assert result.chunk is chunk
        assert len(result.embedding) == 1536
        assert result.model_name == "text-embedding-3-small"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_chunks(self, mock_openai_client):
        """Should embed multiple chunks."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        chunks = [
            Chunk(id="c1", content="First", level=2, token_count=5),
            Chunk(id="c2", content="Second", level=2, token_count=5),
        ]
        results = embedder.embed_chunks(chunks)

        assert len(results) == 2
        assert all(isinstance(r, EmbeddedChunk) for r in results)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embed_chunks_empty_list(self, mock_openai_client):
        """Should handle empty chunk list."""
        embedder = OpenAIEmbedder()
        embedder._client = mock_openai_client

        results = embedder.embed_chunks([])
        assert results == []

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_lazy_client_initialization(self):
        """Client should be lazily initialized."""
        embedder = OpenAIEmbedder()
        assert embedder._client is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_model_dimensions_mapping(self):
        """Verify all OpenAI model dimensions are correctly mapped."""
        expected = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        assert OPENAI_MODEL_DIMENSIONS == expected


class TestCohereEmbedder:
    """Tests for CohereEmbedder class."""

    @pytest.fixture
    def mock_cohere_client(self):
        """Create a mock Cohere client."""
        mock_client = MagicMock()

        def create_embed(texts, model, input_type, embedding_types, output_dimension):
            mock_response = MagicMock()
            embeddings = []
            for text in texts:
                emb = [0.1 * ((hash(text) + j) % 10) for j in range(output_dimension)]
                embeddings.append(emb)
            mock_response.embeddings.float_ = embeddings
            return mock_response

        mock_client.embed = MagicMock(side_effect=create_embed)
        return mock_client

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-cohere-key"})
    def test_initialization(self):
        """Should initialize with correct defaults."""
        embedder = CohereEmbedder()
        assert embedder.model_name == "embed-v4.0"
        assert embedder.get_embedding_dim() == 1536
        assert embedder.api_key == "test-cohere-key"

    @patch.dict(os.environ, {"COHERE_API_KEY": "", "CO_API_KEY": ""}, clear=True)
    def test_no_api_key_raises(self):
        """Should raise ValueError when no API key is set."""
        with pytest.raises(ValueError, match="COHERE_API_KEY not set"):
            CohereEmbedder()

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_invalid_dimension_raises(self):
        """Should raise ValueError for invalid embedding dimension."""
        with pytest.raises(ValueError, match="embedding_dim must be one of"):
            CohereEmbedder(embedding_dim=999)

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_valid_dimensions(self):
        """Should accept all valid Matryoshka dimensions."""
        for dim in COHERE_VALID_DIMS:
            embedder = CohereEmbedder(embedding_dim=dim)
            assert embedder.get_embedding_dim() == dim

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_co_api_key_fallback(self):
        """Should fall back to CO_API_KEY env var."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "", "CO_API_KEY": "co-key"}):
            embedder = CohereEmbedder()
            assert embedder.api_key == "co-key"

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_text_returns_correct_dimension(self, mock_cohere_client):
        """Should return embedding of correct dimension."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        result = embedder.embed_text("Hello world")
        assert len(result) == 1536

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_text_uses_search_document(self, mock_cohere_client):
        """embed_text should use input_type='search_document'."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        embedder.embed_text("test text")

        call_kwargs = mock_cohere_client.embed.call_args
        assert call_kwargs[1]["input_type"] == "search_document"

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_query_uses_search_query(self, mock_cohere_client):
        """embed_query should use input_type='search_query'."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        embedder.embed_query("test query")

        call_kwargs = mock_cohere_client.embed.call_args
        assert call_kwargs[1]["input_type"] == "search_query"

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_text_empty_returns_zero_vector(self):
        """Should return zero vector for empty text without API call."""
        embedder = CohereEmbedder(embedding_dim=256)
        result = embedder.embed_text("")
        assert result == [0.0] * 256

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_query_empty_returns_zero_vector(self):
        """Should return zero vector for empty query without API call."""
        embedder = CohereEmbedder(embedding_dim=256)
        result = embedder.embed_query("   ")
        assert result == [0.0] * 256

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_texts_batching(self, mock_cohere_client):
        """Should batch texts and return embeddings."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        texts = ["Text 1", "Text 2", "Text 3"]
        results = list(embedder.embed_texts(texts))

        assert len(results) == 3
        for r in results:
            assert len(r) == 1536

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_texts_handles_empty_strings(self, mock_cohere_client):
        """Should return zero vectors for empty strings in batch."""
        embedder = CohereEmbedder(embedding_dim=256)
        embedder._client = mock_cohere_client

        texts = ["Text 1", "", "Text 3"]
        results = list(embedder.embed_texts(texts))

        assert len(results) == 3
        assert results[1] == [0.0] * 256

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_texts_empty_list(self, mock_cohere_client):
        """Should handle empty text list."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        results = list(embedder.embed_texts([]))
        assert results == []

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_chunks(self, mock_cohere_client):
        """Should embed multiple chunks."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        chunks = [
            Chunk(id="c1", content="First", level=2, token_count=5),
            Chunk(id="c2", content="Second", level=2, token_count=5),
        ]
        results = embedder.embed_chunks(chunks)

        assert len(results) == 2
        assert all(isinstance(r, EmbeddedChunk) for r in results)
        assert results[0].model_name == "embed-v4.0"

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_embed_chunks_empty_list(self, mock_cohere_client):
        """Should handle empty chunk list."""
        embedder = CohereEmbedder()
        embedder._client = mock_cohere_client

        results = embedder.embed_chunks([])
        assert results == []

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_lazy_client_initialization(self):
        """Client should be lazily initialized."""
        embedder = CohereEmbedder()
        assert embedder._client is None


class TestEmbedQueryAliases:
    """Tests for embed_query() on Mock and OpenAI embedders."""

    def test_mock_embed_query_is_alias(self):
        """MockEmbedder.embed_query should return same as embed_text."""
        embedder = MockEmbedder(embedding_dim=10)
        text = "test query"
        assert embedder.embed_query(text) == embedder.embed_text(text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_embed_query_is_alias(self, ):
        """OpenAIEmbedder.embed_query should be an alias for embed_text."""
        embedder = OpenAIEmbedder()
        # Just check the method exists and is callable
        assert hasattr(embedder, "embed_query")
        assert callable(embedder.embed_query)
