"""Tests for table parser module.

TDD tests for Phase 2: Advanced Table Processing.
Tests cover ParsedTable dataclass and parsing functions.
"""

import pytest


# ============================================================================
# Tests for ParsedTable Dataclass
# ============================================================================


class TestParsedTableDataclass:
    """Tests for ParsedTable dataclass structure."""

    def test_parsed_table_exists(self):
        """ParsedTable should be importable."""
        from src.chunker.table_parser import ParsedTable

        assert ParsedTable is not None

    def test_parsed_table_has_required_fields(self):
        """ParsedTable should have all required fields."""
        from src.chunker.table_parser import ParsedTable

        table = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="Some context",
            source_chunk_id="chunk_001",
        )

        assert table.raw_markdown == "| A | B |\n|---|---|\n| 1 | 2 |"
        assert table.column_names == ["A", "B"]
        assert table.rows == [["1", "2"]]
        assert table.row_count == 1
        assert table.column_count == 2
        assert table.column_types == ["text", "text"]
        assert table.has_header_row is True
        assert table.surrounding_context == "Some context"
        assert table.source_chunk_id == "chunk_001"

    def test_to_markdown_kv_method(self):
        """ParsedTable should have to_markdown_kv method."""
        from src.chunker.table_parser import ParsedTable

        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $100 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$100"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        result = table.to_markdown_kv()

        assert "- Method: FIFO" in result
        assert "- Cost: $100" in result

    def test_to_markdown_kv_multiple_rows(self):
        """to_markdown_kv should handle multiple rows."""
        from src.chunker.table_parser import ParsedTable

        table = ParsedTable(
            raw_markdown="",
            column_names=["Item", "Price"],
            rows=[["Apple", "$1"], ["Banana", "$2"]],
            row_count=2,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        result = table.to_markdown_kv()

        assert "- Item: Apple" in result
        assert "- Price: $1" in result
        assert "- Item: Banana" in result
        assert "- Price: $2" in result

    def test_to_summary_prompt_method(self):
        """ParsedTable should have to_summary_prompt method."""
        from src.chunker.table_parser import ParsedTable

        table = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="Context about the table",
            source_chunk_id="chunk_001",
        )

        prompt = table.to_summary_prompt()

        assert "A" in prompt and "B" in prompt  # Column names
        assert "1" in prompt  # Row count
        assert "Context" in prompt  # Context included


# ============================================================================
# Tests for parse_markdown_table Function
# ============================================================================


class TestParseMarkdownTable:
    """Tests for parse_markdown_table function."""

    def test_function_exists(self):
        """parse_markdown_table should be importable."""
        from src.chunker.table_parser import parse_markdown_table

        assert callable(parse_markdown_table)

    def test_parses_simple_table(self):
        """Should parse a simple markdown table."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
Some text before the table.

| Method | Cost | Ending Inventory |
|--------|------|------------------|
| FIFO   | $500 | $200             |
| LIFO   | $600 | $100             |

Some text after.
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert result.column_names == ["Method", "Cost", "Ending Inventory"]
        assert len(result.rows) == 2
        assert result.rows[0] == ["FIFO", "$500", "$200"]
        assert result.rows[1] == ["LIFO", "$600", "$100"]
        assert result.row_count == 2
        assert result.column_count == 3

    def test_returns_none_for_no_table(self):
        """Should return None when no table is found."""
        from src.chunker.table_parser import parse_markdown_table

        content = "This is just plain text without any table."
        result = parse_markdown_table(content, "chunk_001")

        assert result is None

    def test_extracts_surrounding_context(self):
        """Should extract text before the table as context."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
The following table shows inventory methods:

| Method | Description |
|--------|-------------|
| FIFO   | First In    |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert "inventory methods" in result.surrounding_context.lower()

    def test_preserves_source_chunk_id(self):
        """Should preserve the source chunk ID."""
        from src.chunker.table_parser import parse_markdown_table

        content = "| A |\n|---|\n| 1 |"
        result = parse_markdown_table(content, "my_chunk_123")

        assert result is not None
        assert result.source_chunk_id == "my_chunk_123"

    def test_handles_alignment_markers(self):
        """Should handle various alignment markers in separator row."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| Left | Center | Right |
|:-----|:------:|------:|
| A    | B      | C     |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert result.column_names == ["Left", "Center", "Right"]
        assert result.rows[0] == ["A", "B", "C"]

    def test_handles_empty_cells(self):
        """Should handle tables with empty cells."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| A | B | C |
|---|---|---|
| 1 |   | 3 |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert len(result.rows[0]) == 3

    def test_handles_single_column_table(self):
        """Should handle single column tables."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| Item |
|------|
| A    |
| B    |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert result.column_count == 1
        assert result.row_count == 2


# ============================================================================
# Tests for infer_column_types Function
# ============================================================================


class TestInferColumnTypes:
    """Tests for column type inference."""

    def test_function_exists(self):
        """infer_column_types should be importable."""
        from src.chunker.table_parser import infer_column_types

        assert callable(infer_column_types)

    def test_detects_numeric_columns(self):
        """Should detect numeric columns."""
        from src.chunker.table_parser import infer_column_types

        columns = ["Amount"]
        rows = [["$100"], ["$200"], ["$300"]]

        types = infer_column_types(columns, rows)

        assert types[0] == "numeric"

    def test_detects_text_columns(self):
        """Should detect text columns."""
        from src.chunker.table_parser import infer_column_types

        columns = ["Name"]
        rows = [["Alice"], ["Bob"], ["Charlie"]]

        types = infer_column_types(columns, rows)

        assert types[0] == "text"

    def test_detects_date_columns(self):
        """Should detect date columns."""
        from src.chunker.table_parser import infer_column_types

        columns = ["Date"]
        rows = [["2024-01-15"], ["2024-02-20"], ["2024-03-25"]]

        types = infer_column_types(columns, rows)

        assert types[0] == "date"

    def test_handles_mixed_numeric(self):
        """Should detect numeric when majority are numeric."""
        from src.chunker.table_parser import infer_column_types

        columns = ["Value"]
        rows = [["100"], ["200"], ["300"], ["400"], ["N/A"]]

        types = infer_column_types(columns, rows)

        assert types[0] == "numeric"

    def test_handles_multiple_columns(self):
        """Should handle multiple columns correctly."""
        from src.chunker.table_parser import infer_column_types

        columns = ["Name", "Amount", "Date"]
        rows = [
            ["Alice", "$100", "2024-01-15"],
            ["Bob", "$200", "2024-02-20"],
        ]

        types = infer_column_types(columns, rows)

        assert types[0] == "text"
        assert types[1] == "numeric"
        assert types[2] == "date"


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestTableParserEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_table(self):
        """Should handle malformed tables gracefully."""
        from src.chunker.table_parser import parse_markdown_table

        content = "| A | B\n|---\n| 1"  # Missing delimiters
        result = parse_markdown_table(content, "chunk_001")

        # Malformed tables should return None
        assert result is None

    def test_handles_empty_content(self):
        """Should handle empty content."""
        from src.chunker.table_parser import parse_markdown_table

        result = parse_markdown_table("", "chunk_001")

        assert result is None

    def test_handles_table_only_header(self):
        """Should handle table with only header row."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| A | B | C |
|---|---|---|
"""
        result = parse_markdown_table(content, "chunk_001")

        # Header-only tables should return a ParsedTable with zero rows
        assert result is not None
        assert result.row_count == 0
        assert result.column_names == ["A", "B", "C"]

    def test_handles_unicode_content(self):
        """Should handle unicode characters in tables."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| 名前 | 金額 |
|------|------|
| 商品A | ¥1000 |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert "名前" in result.column_names
        assert "¥1000" in result.rows[0]

    def test_handles_special_characters(self):
        """Should handle special characters in cell values."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| Formula | Result |
|---------|--------|
| A + B   | C = D  |
| 50%     | (100)  |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert result.row_count == 2


# ============================================================================
# Tests for Integration with Chunk Metadata
# ============================================================================


class TestTableParserIntegration:
    """Tests for integration with chunk processing."""

    def test_works_with_has_table_flag(self):
        """Should correctly identify tables for chunks with has_table flag."""
        from src.chunker.table_parser import parse_markdown_table

        # Simulate chunk content that has_table=True
        content = """
## Inventory Methods Comparison

| Method | COGS | Ending Inventory |
|--------|------|------------------|
| FIFO   | $500 | $200             |
| LIFO   | $600 | $100             |

As shown in the table above, FIFO results in lower COGS.
"""
        result = parse_markdown_table(content, "chunk_with_table")

        assert result is not None
        assert result.row_count == 2
        assert "Method" in result.column_names

    def test_multiple_tables_returns_first(self):
        """Should handle content with multiple tables (return first)."""
        from src.chunker.table_parser import parse_markdown_table

        content = """
| Table 1 |
|---------|
| A       |

Some text between.

| Table 2 |
|---------|
| B       |
"""
        result = parse_markdown_table(content, "chunk_001")

        assert result is not None
        assert "Table 1" in result.column_names
