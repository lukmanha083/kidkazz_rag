"""Table summarizer using LLM for semantic embedding generation.

This module provides LLM-powered summarization of parsed tables
to enable semantic search through table content.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .table_parser import ParsedTable

logger = logging.getLogger(__name__)


@dataclass
class TableSummary:
    """LLM-generated summary of a table for embedding."""

    table_id: str
    summary_text: str  # Natural language description
    key_columns: list[str] = field(default_factory=list)
    key_values: list[str] = field(default_factory=list)  # Notable data points
    embedding: Optional[list[float]] = None


class TableSummarizer:
    """Generate natural language summaries of tables using LLM."""

    def __init__(self, provider: str = "anthropic"):
        """Initialize summarizer with LLM provider.

        Args:
            provider: LLM provider name (anthropic, openai, etc.)
        """
        self.provider = provider
        self.client = self._init_client(provider)

    def _init_client(self, provider: str):
        """Initialize LLM client based on provider.

        Args:
            provider: Provider name

        Returns:
            Configured LLM client
        """
        if provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic()
            except ImportError:
                logger.warning("anthropic package not installed, using mock client")
                return None
        elif provider == "openai":
            try:
                import openai
                return openai.OpenAI()
            except ImportError:
                logger.warning("openai package not installed, using mock client")
                return None
        else:
            logger.warning(f"Unknown provider {provider}, using mock client")
            return None

    def summarize(self, table: ParsedTable) -> TableSummary:
        """Generate a summary for semantic search.

        Args:
            table: Parsed table to summarize

        Returns:
            TableSummary with natural language description
        """
        # Generate table ID
        table_id = f"table_{table.source_chunk_id}"

        # Extract key columns and values
        key_columns = self._identify_key_columns(table)
        key_values = self._extract_key_values(table)

        # Generate summary using LLM
        summary_text = self._generate_summary(table)

        return TableSummary(
            table_id=table_id,
            summary_text=summary_text,
            key_columns=key_columns,
            key_values=key_values,
        )

    def _generate_summary(self, table: ParsedTable) -> str:
        """Generate natural language summary using LLM.

        Args:
            table: Parsed table

        Returns:
            Summary text string
        """
        if self.client is None:
            # Fallback for testing or when client not available
            return self._generate_fallback_summary(table)

        prompt = table.to_summary_prompt()

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._generate_fallback_summary(table)

    def _generate_fallback_summary(self, table: ParsedTable) -> str:
        """Generate a basic summary without LLM.

        Args:
            table: Parsed table

        Returns:
            Basic summary string
        """
        cols = ", ".join(table.column_names[:3])
        if len(table.column_names) > 3:
            cols += f", and {len(table.column_names) - 3} more"

        return (
            f"Table with {table.row_count} rows and {table.column_count} columns. "
            f"Columns: {cols}. "
            f"{table.surrounding_context[:100] if table.surrounding_context else ''}"
        ).strip()

    def _identify_key_columns(self, table: ParsedTable) -> list[str]:
        """Identify the most important columns.

        Heuristic: first column + any numeric columns (up to 3 total).

        Args:
            table: Parsed table

        Returns:
            List of key column names
        """
        key = []

        # Always include first column
        if table.column_names:
            key.append(table.column_names[0])

        # Add numeric columns
        for i, col_type in enumerate(table.column_types):
            if col_type == "numeric" and table.column_names[i] not in key:
                key.append(table.column_names[i])
                if len(key) >= 3:
                    break

        return key[:3]  # Max 3 key columns

    def _extract_key_values(self, table: ParsedTable) -> list[str]:
        """Extract notable values from the table.

        Includes values from first row (often important) and
        last row (often totals/summaries).

        Args:
            table: Parsed table

        Returns:
            List of key values
        """
        values = []

        # First row values
        if table.rows:
            values.extend(table.rows[0][:3])

        # Last row values (often totals)
        if len(table.rows) > 1:
            values.extend(table.rows[-1][:3])

        return values
