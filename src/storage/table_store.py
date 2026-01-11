"""Table storage with multi-vector retrieval support.

This module provides storage and retrieval for parsed tables using
a multi-vector approach: embed table summaries for retrieval, store
raw markdown for LLM synthesis.
"""

import logging
from typing import Optional, Protocol

from src.chunker.table_parser import ParsedTable
from src.chunker.table_summarizer import TableSummary

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol for embedders that can embed text."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string."""
        ...


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class TableStore:
    """Storage and retrieval for tables with multi-vector approach.

    Uses in-memory storage for now. Can be extended to use Helix-DB
    for production deployments.
    """

    def __init__(self, embedder: EmbedderProtocol):
        """Initialize table store with embedder.

        Args:
            embedder: Embedder for generating summary embeddings
        """
        self.embedder = embedder

        # In-memory storage
        self._tables: dict[str, dict] = {}  # table_id -> table data
        self._embeddings: dict[str, list[float]] = {}  # table_id -> embedding
        self._summaries: dict[str, str] = {}  # table_id -> summary text
        self._doc_tables: dict[str, list[str]] = {}  # doc_id -> [table_ids]
        self._concept_tables: dict[str, list[str]] = {}  # concept -> [table_ids]

    def store_table(
        self,
        table: ParsedTable,
        summary: TableSummary,
        document_id: str,
    ) -> str:
        """Store table with summary embedding for retrieval.

        Args:
            table: Parsed table to store
            summary: LLM-generated summary
            document_id: Source document ID

        Returns:
            Table ID for retrieval
        """
        table_id = f"table_{table.source_chunk_id}"

        # Embed the summary for retrieval
        embedding = self.embedder.embed_text(summary.summary_text)

        # Store table data
        self._tables[table_id] = {
            "table_id": table_id,
            "raw_markdown": table.raw_markdown,
            "column_names": table.column_names,
            "rows": table.rows,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "column_types": table.column_types,
            "has_header_row": table.has_header_row,
            "surrounding_context": table.surrounding_context,
            "source_chunk_id": table.source_chunk_id,
            "document_id": document_id,
            "key_columns": summary.key_columns,
            "key_values": summary.key_values,
        }

        # Store embedding and summary
        self._embeddings[table_id] = embedding
        self._summaries[table_id] = summary.summary_text

        # Track document -> tables relationship
        if document_id not in self._doc_tables:
            self._doc_tables[document_id] = []
        self._doc_tables[document_id].append(table_id)

        logger.info(f"Stored table {table_id} with {table.row_count} rows")
        return table_id

    def get_table(self, table_id: str) -> Optional[ParsedTable]:
        """Get a table by its ID.

        Args:
            table_id: Table identifier

        Returns:
            ParsedTable if found, None otherwise
        """
        if table_id not in self._tables:
            return None

        data = self._tables[table_id]
        return ParsedTable(
            raw_markdown=data["raw_markdown"],
            column_names=data["column_names"],
            rows=data["rows"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            column_types=data["column_types"],
            has_header_row=data["has_header_row"],
            surrounding_context=data["surrounding_context"],
            source_chunk_id=data["source_chunk_id"],
        )

    def search_tables(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[tuple[ParsedTable, float]]:
        """Search tables by semantic similarity.

        Args:
            query: Search query text
            top_k: Maximum number of results
            doc_id: Optional filter by document

        Returns:
            List of (ParsedTable, similarity_score) tuples
        """
        if not self._tables:
            return []

        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Calculate similarities
        scores: list[tuple[str, float]] = []
        for table_id, embedding in self._embeddings.items():
            # Filter by doc_id if specified
            if doc_id and self._tables[table_id].get("document_id") != doc_id:
                continue

            similarity = cosine_similarity(query_embedding, embedding)
            scores.append((table_id, similarity))

        # Sort by similarity (descending) and take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        # Convert to ParsedTable objects
        results: list[tuple[ParsedTable, float]] = []
        for table_id, score in top_results:
            table = self.get_table(table_id)
            if table:
                results.append((table, score))

        return results

    def link_table_to_concept(self, table_id: str, concept_name: str) -> None:
        """Link a table to a concept for graph-based retrieval.

        Args:
            table_id: Table identifier
            concept_name: Concept name to link
        """
        if table_id not in self._tables:
            logger.warning(f"Cannot link: table {table_id} not found")
            return

        concept_key = concept_name.lower()
        if concept_key not in self._concept_tables:
            self._concept_tables[concept_key] = []

        if table_id not in self._concept_tables[concept_key]:
            self._concept_tables[concept_key].append(table_id)
            logger.info(f"Linked table {table_id} to concept {concept_name}")

    def get_tables_for_concept(self, concept_name: str) -> list[ParsedTable]:
        """Get all tables linked to a concept.

        Args:
            concept_name: Concept name

        Returns:
            List of tables linked to the concept
        """
        concept_key = concept_name.lower()
        table_ids = self._concept_tables.get(concept_key, [])

        tables: list[ParsedTable] = []
        for table_id in table_ids:
            table = self.get_table(table_id)
            if table:
                tables.append(table)

        return tables

    def list_tables(self, doc_id: Optional[str] = None) -> list[dict]:
        """List all stored tables.

        Args:
            doc_id: Optional filter by document ID

        Returns:
            List of table metadata dictionaries
        """
        results: list[dict] = []

        for table_id, data in self._tables.items():
            # Filter by doc_id if specified
            if doc_id and data.get("document_id") != doc_id:
                continue

            results.append({
                "table_id": table_id,
                "column_names": data["column_names"],
                "row_count": data["row_count"],
                "column_count": data["column_count"],
                "document_id": data.get("document_id"),
                "summary": self._summaries.get(table_id, ""),
            })

        return results

    def delete_table(self, table_id: str) -> bool:
        """Delete a table and its embedding.

        Args:
            table_id: Table identifier

        Returns:
            True if deleted, False if not found
        """
        if table_id not in self._tables:
            return False

        # Get document ID before deletion
        doc_id = self._tables[table_id].get("document_id")

        # Remove from storage
        del self._tables[table_id]
        self._embeddings.pop(table_id, None)
        self._summaries.pop(table_id, None)

        # Remove from document mapping
        if doc_id and doc_id in self._doc_tables:
            self._doc_tables[doc_id] = [
                tid for tid in self._doc_tables[doc_id] if tid != table_id
            ]

        # Remove from concept mappings
        for concept in self._concept_tables:
            self._concept_tables[concept] = [
                tid for tid in self._concept_tables[concept] if tid != table_id
            ]

        logger.info(f"Deleted table {table_id}")
        return True
