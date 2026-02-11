"""Embedding generation with multiple backend support.

Supports:
- OpenAI (API-based, production)
- Mock (testing)
"""

import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .chunker import Chunk


# Default model configuration
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 32

# OpenAI model dimensions
OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass
class EmbeddedChunk:
    """Chunk with embedding vector attached."""

    chunk: Chunk
    embedding: list[float]
    model_name: str

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk.id,
            "content": self.chunk.content,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "level": self.chunk.level,
            "parent_id": self.chunk.parent_id,
            "child_ids": self.chunk.child_ids,
            "prev_id": self.chunk.prev_id,
            "next_id": self.chunk.next_id,
            "section_path": self.chunk.section_path,
            "metadata": self.chunk.metadata,
        }


class MockEmbedder:
    """
    Mock embedder for testing without API dependencies.

    Generates deterministic pseudo-embeddings based on text hash.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_dim: int = 384,
    ):
        """
        Initialize mock embedder.

        Args:
            model_name: Model name (for compatibility)
            embedding_dim: Dimension of mock embeddings
        """
        self.model_name = model_name
        self._embedding_dim = embedding_dim

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic mock embedding."""
        if not text.strip():
            return [0.0] * self._embedding_dim

        # Use hash to generate deterministic values
        text_hash = hash(text)
        embedding = []
        for i in range(self._embedding_dim):
            # Generate values between -1 and 1
            value = ((text_hash + i * 31) % 10000) / 5000 - 1.0
            embedding.append(value)

        # Normalize to unit vector
        norm = sum(v * v for v in embedding) ** 0.5
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query (alias for embed_text)."""
        return self.embed_text(text)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,  # noqa: ARG002 - API compatibility
    ) -> Iterator[list[float]]:
        """Generate mock embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size (unused, for API compatibility)

        Yields:
            Mock embedding vectors
        """
        for text in texts:
            yield self.embed_text(text)

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Generate mock embedding for a chunk."""
        return EmbeddedChunk(
            chunk=chunk,
            embedding=self.embed_text(chunk.content),
            model_name=self.model_name,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = DEFAULT_BATCH_SIZE,  # noqa: ARG002 - API compatibility
        show_progress: bool = False,  # noqa: ARG002 - API compatibility
    ) -> list[EmbeddedChunk]:
        """Generate mock embeddings for multiple chunks.

        Args:
            chunks: List of chunks to embed
            batch_size: Batch size (unused, for API compatibility)
            show_progress: Progress indicator (unused, for API compatibility)

        Returns:
            List of EmbeddedChunks with mock embeddings
        """
        return [self.embed_chunk(chunk) for chunk in chunks]


class OpenAIEmbedder:
    """
    OpenAI API embedder for higher quality embeddings.

    Uses OpenAI's text-embedding models via API. Requires OPENAI_API_KEY.
    Higher quality than local models but requires API calls and costs money.
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        from dotenv import load_dotenv
        load_dotenv()

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env file or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def get_embedding_dim(self) -> int:
        """Get embedding dimension for the configured model."""
        return OPENAI_MODEL_DIMENSIONS.get(self.model_name, 1536)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            return [0.0] * self.get_embedding_dim()

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query (alias for embed_text)."""
        return self.embed_text(text)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Iterator[list[float]]:
        """
        Generate embeddings for multiple texts.

        OpenAI API supports batch embedding natively (up to 2048 inputs).

        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls (max 2048)

        Yields:
            Embedding vectors
        """
        if not texts:
            return

        zero_vector = [0.0] * self.get_embedding_dim()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Separate empty and non-empty texts
            non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            # Get embeddings for non-empty texts
            embeddings_map = {}
            if non_empty_texts:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=non_empty_texts
                )
                for idx, emb_data in zip(non_empty_indices, response.data, strict=True):
                    embeddings_map[idx] = emb_data.embedding

            # Yield in order
            for j in range(len(batch)):
                if j in embeddings_map:
                    yield embeddings_map[j]
                else:
                    yield zero_vector

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Generate embedding for a single chunk."""
        return EmbeddedChunk(
            chunk=chunk,
            embedding=self.embed_text(chunk.content),
            model_name=self.model_name,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,  # noqa: ARG002 - API compatibility
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for multiple chunks.

        Args:
            chunks: List of chunks to embed
            batch_size: Batch size for API calls
            show_progress: Progress indicator (reserved for future)

        Returns:
            List of EmbeddedChunks with embeddings
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = list(self.embed_texts(texts, batch_size=batch_size))

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                model_name=self.model_name,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]


COHERE_VALID_DIMS = {256, 512, 1024, 1536}


class CohereEmbedder:
    """Cohere Embed v4 embedder with Matryoshka dimension support.

    Uses separate input_type for documents vs queries as required by Cohere.
    """

    DEFAULT_MODEL = "embed-v4.0"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        embedding_dim: int = 1536,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        self.model_name = model_name
        self._embedding_dim = embedding_dim
        self.api_key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY not set. Add it to .env file or pass api_key parameter."
            )

        if embedding_dim not in COHERE_VALID_DIMS:
            raise ValueError(
                f"embedding_dim must be one of {sorted(COHERE_VALID_DIMS)}, got {embedding_dim}"
            )

    @property
    def client(self):
        """Lazy-load Cohere client."""
        if self._client is None:
            import cohere
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Call Cohere embed API for a batch of texts."""
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type,
            embedding_types=["float"],
            output_dimension=self._embedding_dim,
        )
        return [list(v) for v in response.embeddings.float_]

    def embed_text(self, text: str) -> list[float]:
        """Embed text as a document (for ingestion/storage)."""
        if not text.strip():
            return [0.0] * self._embedding_dim
        return self._embed_batch([text], input_type="search_document")[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed text as a search query (for retrieval)."""
        if not text.strip():
            return [0.0] * self._embedding_dim
        return self._embed_batch([text], input_type="search_query")[0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 96,
    ) -> Iterator[list[float]]:
        """Embed multiple texts as documents in batches."""
        if not texts:
            return

        zero_vector = [0.0] * self._embedding_dim

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            embeddings_map: dict[int, list[float]] = {}
            if non_empty_texts:
                results = self._embed_batch(non_empty_texts, input_type="search_document")
                for idx, emb in zip(non_empty_indices, results, strict=True):
                    embeddings_map[idx] = emb

            for j in range(len(batch)):
                if j in embeddings_map:
                    yield embeddings_map[j]
                else:
                    yield zero_vector

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Embed a single chunk as a document."""
        return EmbeddedChunk(
            chunk=chunk,
            embedding=self.embed_text(chunk.content),
            model_name=self.model_name,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 96,
        show_progress: bool = False,  # noqa: ARG002 - API compatibility
    ) -> list[EmbeddedChunk]:
        """Embed multiple chunks as documents."""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = list(self.embed_texts(texts, batch_size=batch_size))

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                model_name=self.model_name,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_chunks(
    query_embedding: list[float],
    embedded_chunks: list[EmbeddedChunk],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[EmbeddedChunk, float]]:
    """
    Find most similar chunks to a query embedding.

    Args:
        query_embedding: Query vector
        embedded_chunks: List of embedded chunks to search
        top_k: Number of results to return
        threshold: Minimum similarity threshold

    Returns:
        List of (chunk, similarity) tuples, sorted by similarity
    """
    similarities: list[tuple[EmbeddedChunk, float]] = []

    for ec in embedded_chunks:
        sim = cosine_similarity(query_embedding, ec.embedding)
        if sim >= threshold:
            similarities.append((ec, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]
