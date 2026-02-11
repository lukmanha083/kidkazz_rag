"""Tests for CohereReranker."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.chunker.chunker import Chunk
from src.chunker.embedder import EmbeddedChunk


class TestCohereReranker:
    """Tests for CohereReranker class."""

    @pytest.fixture
    def mock_cohere_client(self):
        """Create a mock Cohere client for reranking."""
        mock_client = MagicMock()

        def create_rerank(model, query, documents, top_n):
            mock_response = MagicMock()
            # Return top_n results in reverse order with decreasing scores
            results = []
            for i in range(min(top_n, len(documents))):
                item = MagicMock()
                item.index = i
                item.relevance_score = 0.9 - (i * 0.1)
                results.append(item)
            mock_response.results = results
            return mock_response

        mock_client.rerank = MagicMock(side_effect=create_rerank)
        return mock_client

    @pytest.fixture
    def sample_chunk_results(self):
        """Create sample (EmbeddedChunk, score) tuples."""
        chunks = [
            Chunk(id=f"c{i}", content=f"Content about topic {i}", level=2, token_count=10)
            for i in range(5)
        ]
        return [
            (EmbeddedChunk(chunk=c, embedding=[0.1] * 10, model_name="test"), 0.5 + i * 0.05)
            for i, c in enumerate(chunks)
        ]

    @pytest.fixture
    def sample_summary_results(self):
        """Create sample summary dicts."""
        return [
            {"summary_id": f"s{i}", "content": f"Summary about topic {i}", "score": 0.5 + i * 0.05}
            for i in range(5)
        ]

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_initialization(self):
        """Should initialize with correct defaults."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()
        assert reranker.model_name == "rerank-v3.5"
        assert reranker.api_key == "test-key"

    @patch.dict(os.environ, {"COHERE_API_KEY": "", "CO_API_KEY": ""}, clear=True)
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_no_api_key_raises(self):
        """Should raise ValueError when no API key is set."""
        from src.chunker.reranker import CohereReranker
        with pytest.raises(ValueError, match="COHERE_API_KEY not set"):
            CohereReranker()

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_rerank_chunks(self, mock_cohere_client, sample_chunk_results):
        """Should rerank chunk results and return top_n."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()
        reranker._client = mock_cohere_client

        results = reranker.rerank_chunks("test query", sample_chunk_results, top_n=3)

        assert len(results) == 3
        # Verify results are (EmbeddedChunk, float) tuples
        for ec, score in results:
            assert isinstance(ec, EmbeddedChunk)
            assert isinstance(score, float)

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_rerank_summaries(self, mock_cohere_client, sample_summary_results):
        """Should rerank summary results with updated scores."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()
        reranker._client = mock_cohere_client

        results = reranker.rerank_summaries("test query", sample_summary_results, top_n=3)

        assert len(results) == 3
        for r in results:
            assert "reranked" in r
            assert r["reranked"] is True
            assert isinstance(r["score"], float)

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_rerank_empty_results(self, mock_cohere_client):
        """Should return empty list for empty input."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()
        reranker._client = mock_cohere_client

        assert reranker.rerank_chunks("query", [], top_n=5) == []
        assert reranker.rerank_summaries("query", [], top_n=5) == []

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_rerank_fallback_on_error(self, sample_chunk_results):
        """Should fall back to original order on API error."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()

        # Create a client that raises an error
        mock_client = MagicMock()
        mock_client.rerank.side_effect = Exception("API error")
        reranker._client = mock_client

        results = reranker.rerank_chunks("query", sample_chunk_results, top_n=3)

        # Should return first 3 of original results
        assert len(results) == 3
        assert results[0][0].chunk.id == "c0"

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("src.chunker.reranker.COHERE_AVAILABLE", True)
    def test_rerank_summaries_fallback_on_error(self, sample_summary_results):
        """Should fall back to original order on API error for summaries."""
        from src.chunker.reranker import CohereReranker
        reranker = CohereReranker()

        mock_client = MagicMock()
        mock_client.rerank.side_effect = Exception("API error")
        reranker._client = mock_client

        results = reranker.rerank_summaries("query", sample_summary_results, top_n=3)

        assert len(results) == 3
        assert results[0]["summary_id"] == "s0"
