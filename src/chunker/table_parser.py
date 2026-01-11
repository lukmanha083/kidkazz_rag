"""Table parser for extracting and structuring markdown tables.

This module provides functionality to parse markdown tables into structured
representations suitable for LLM processing and semantic search.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedTable:
    """Structured representation of a markdown table."""

    raw_markdown: str
    column_names: list[str]
    rows: list[list[str]]
    row_count: int
    column_count: int
    column_types: list[str]  # "text", "numeric", "date", "mixed"
    has_header_row: bool
    surrounding_context: str  # Text before table
    source_chunk_id: str

    def to_markdown_kv(self) -> str:
        """Convert to Markdown-KV format (best for LLM accuracy).

        Research shows Markdown-KV format achieves 60.7% accuracy
        for table comprehension tasks, outperforming raw markdown.

        Returns:
            String with key-value pairs for each cell
        """
        lines = []
        for row in self.rows:
            for col_name, value in zip(self.column_names, row, strict=False):
                lines.append(f"- {col_name}: {value}")
            lines.append("")  # Blank line between rows
        return "\n".join(lines)

    def to_summary_prompt(self) -> str:
        """Generate prompt for LLM summarization.

        Returns:
            Prompt string for generating table summary
        """
        return f"""Summarize this table in 2-3 sentences.

Table columns: {', '.join(self.column_names)}
Number of rows: {self.row_count}
Context: {self.surrounding_context[:200] if self.surrounding_context else 'None'}

Table:
{self.raw_markdown}

Provide a natural language summary describing:
1. What this table contains
2. The key relationships or comparisons shown
3. Any notable values or patterns
"""


def parse_markdown_table(content: str, chunk_id: str) -> Optional[ParsedTable]:
    """Parse a markdown table from content into structured form.

    Args:
        content: Text content that may contain a markdown table
        chunk_id: Source chunk ID for tracking

    Returns:
        ParsedTable if a valid table is found, None otherwise
    """
    if not content or not content.strip():
        return None

    # Regex pattern for markdown tables
    # Matches: | Header | ... |\n|---|...|\n| Row | ... |
    table_pattern = r'\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)*)'
    match = re.search(table_pattern, content)

    if not match:
        return None

    header_row = match.group(1)
    body = match.group(2)

    # Parse column names
    columns = [c.strip() for c in header_row.split('|') if c.strip()]

    if not columns:
        return None

    # Parse rows
    rows = []
    for line in body.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Split by | and filter empty strings
        cells = [c.strip() for c in line.split('|')]
        # Remove first and last empty strings from split
        cells = [c for c in cells if c or cells.index(c) not in (0, len(cells) - 1)]
        # Keep only non-empty or intentionally empty cells
        if cells:
            # Pad or truncate to match column count
            while len(cells) < len(columns):
                cells.append("")
            cells = cells[:len(columns)]
            rows.append(cells)

    if not rows:
        return None

    # Infer column types
    column_types = infer_column_types(columns, rows)

    # Extract surrounding context (text before table)
    table_start = content.find(match.group(0))
    context_before = content[:table_start].strip()
    # Take last 200 chars of context
    surrounding_context = context_before[-200:] if len(context_before) > 200 else context_before

    return ParsedTable(
        raw_markdown=match.group(0).strip(),
        column_names=columns,
        rows=rows,
        row_count=len(rows),
        column_count=len(columns),
        column_types=column_types,
        has_header_row=True,
        surrounding_context=surrounding_context,
        source_chunk_id=chunk_id,
    )


def infer_column_types(columns: list[str], rows: list[list[str]]) -> list[str]:
    """Infer column data types from content.

    Args:
        columns: List of column names
        rows: List of row data

    Returns:
        List of type strings: "text", "numeric", or "date"
    """
    types = []

    for col_idx in range(len(columns)):
        values = []
        for row in rows:
            if col_idx < len(row):
                values.append(row[col_idx])

        if not values:
            types.append("text")
            continue

        # Check if date (look for date patterns)
        date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'
        date_count = sum(1 for v in values if re.match(date_pattern, v.strip()))
        if date_count > len(values) * 0.5:
            types.append("date")
            continue

        # Check if numeric (numbers, currency, percentages)
        numeric_pattern = r'^[\d,.$%€¥£()\-+\s]+$'
        numeric_count = sum(
            1 for v in values
            if v.strip() and re.match(numeric_pattern, v.strip())
        )
        if numeric_count >= len(values) * 0.8:
            types.append("numeric")
            continue

        # Default to text
        types.append("text")

    return types
