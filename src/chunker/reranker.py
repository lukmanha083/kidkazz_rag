"""Cohere Rerank v3.5 for post-search result reranking."""

import logging
import os
from typing import Optional

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CohereReranker:
    """Reranks search results using Cohere Rerank v3.5."""

    DEFAULT_MODEL = "rerank-v3.5"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere package is not installed. Run: pip install cohere"
            )

        from dotenv import load_dotenv
        load_dotenv()

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY not set. Add it to .env file or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load Cohere client."""
        if self._client is None:
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def rerank_chunks(
        self,
        query: str,
        results: list[tuple],
        top_n: int = 5,
    ) -> list[tuple]:
        """Rerank chunk search results.

        Args:
            query: Original search query
            results: List of (EmbeddedChunk, score) tuples from vector search
            top_n: Number of results to return after reranking

        Returns:
            Reranked list of (EmbeddedChunk, relevance_score) tuples
        """
        if not results:
            return []

        documents = [ec.chunk.content for ec, _score in results]

        try:
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(top_n, len(results)),
            )

            reranked = []
            for item in response.results:
                original_ec, _original_score = results[item.index]
                reranked.append((original_ec, item.relevance_score))

            return reranked

        except Exception as e:
            logger.warning("Reranking failed, falling back to original order: %s", e)
            return results[:top_n]

    def rerank_summaries(
        self,
        query: str,
        results: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """Rerank summary search results.

        Args:
            query: Original search query
            results: List of summary dicts with "content" and "score" keys
            top_n: Number of results to return after reranking

        Returns:
            Reranked list of summary dicts with updated "score" and "reranked" flag
        """
        if not results:
            return []

        documents = [r.get("content", "") for r in results]

        try:
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(top_n, len(results)),
            )

            reranked = []
            for item in response.results:
                summary = dict(results[item.index])
                summary["score"] = item.relevance_score
                summary["reranked"] = True
                reranked.append(summary)

            return reranked

        except Exception as e:
            logger.warning("Summary reranking failed, falling back to original order: %s", e)
            return results[:top_n]
