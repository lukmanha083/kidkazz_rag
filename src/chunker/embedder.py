"""FastEmbed integration for CPU-optimized embedding generation."""

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .chunker import Chunk


# Default model configuration
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32

# Model dimension mapping
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
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


class ChunkEmbedder:
    """
    CPU-optimized embedding generator using FastEmbed.

    FastEmbed uses ONNX Runtime with INT8 quantization for fast CPU inference.
    No GPU required - optimized for AMD Ryzen CPUs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the embedding model
            cache_dir: Directory to cache model files
            threads: Number of CPU threads (None = auto-detect)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            from fastembed import TextEmbedding

            kwargs = {"model_name": self.model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            if self.threads:
                kwargs["threads"] = self.threads

            self._model = TextEmbedding(**kwargs)
            self._initialized = True
        except ImportError as e:
            raise ImportError(
                "FastEmbed is not installed. Run: pip install fastembed"
            ) from e

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension for the configured model.

        Returns:
            Embedding dimension (e.g., 384 for bge-small)
        """
        return MODEL_DIMENSIONS.get(self.model_name, 384)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        self._ensure_initialized()

        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.get_embedding_dim()

        # FastEmbed returns a generator, get first result
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Iterator[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Yields:
            Embedding vectors
        """
        self._ensure_initialized()

        if not texts:
            return

        # Handle empty texts by replacing with placeholder
        processed_texts = [t if t.strip() else " " for t in texts]

        for embedding in self._model.embed(processed_texts, batch_size=batch_size):
            yield embedding.tolist()

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: Chunk to embed

        Returns:
            EmbeddedChunk with embedding attached
        """
        embedding = self.embed_text(chunk.content)
        return EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            model_name=self.model_name,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for multiple chunks.

        Args:
            chunks: List of chunks to embed
            batch_size: Batch size for processing
            show_progress: Show progress indicator

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
            for chunk, embedding in zip(chunks, embeddings)
        ]


class MockEmbedder:
    """
    Mock embedder for testing without FastEmbed dependency.

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

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Iterator[list[float]]:
        """Generate mock embeddings for multiple texts."""
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
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> list[EmbeddedChunk]:
        """Generate mock embeddings for multiple chunks."""
        return [self.embed_chunk(chunk) for chunk in chunks]


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

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
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
