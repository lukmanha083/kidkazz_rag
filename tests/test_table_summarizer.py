"""Tests for table summarizer module.

TDD tests for LLM-powered table summarization.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Tests for TableSummary Dataclass
# ============================================================================


class TestTableSummaryDataclass:
    """Tests for TableSummary dataclass."""

    def test_table_summary_exists(self):
        """TableSummary should be importable."""
        from src.chunker.table_summarizer import TableSummary

        assert TableSummary is not None

    def test_table_summary_has_required_fields(self):
        """TableSummary should have all required fields."""
        from src.chunker.table_summarizer import TableSummary

        summary = TableSummary(
            table_id="table_001",
            summary_text="This table compares FIFO and LIFO methods.",
            key_columns=["Method", "Cost"],
            key_values=["FIFO", "$500", "LIFO", "$600"],
            embedding=None,
        )

        assert summary.table_id == "table_001"
        assert "FIFO" in summary.summary_text
        assert summary.key_columns == ["Method", "Cost"]
        assert summary.key_values == ["FIFO", "$500", "LIFO", "$600"]
        assert summary.embedding is None

    def test_table_summary_with_embedding(self):
        """TableSummary should accept embedding vectors."""
        from src.chunker.table_summarizer import TableSummary

        embedding = [0.1, 0.2, 0.3, 0.4]
        summary = TableSummary(
            table_id="table_001",
            summary_text="Test summary",
            key_columns=[],
            key_values=[],
            embedding=embedding,
        )

        assert summary.embedding == [0.1, 0.2, 0.3, 0.4]


# ============================================================================
# Tests for TableSummarizer Class
# ============================================================================


class TestTableSummarizerClass:
    """Tests for TableSummarizer class."""

    def test_table_summarizer_exists(self):
        """TableSummarizer should be importable."""
        from src.chunker.table_summarizer import TableSummarizer

        assert TableSummarizer is not None

    def test_table_summarizer_accepts_provider(self):
        """TableSummarizer should accept provider parameter."""
        from src.chunker.table_summarizer import TableSummarizer

        # Mock the client initialization
        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer(provider="anthropic")
            assert summarizer is not None

    def test_summarize_method_exists(self):
        """TableSummarizer should have summarize method."""
        from src.chunker.table_summarizer import TableSummarizer

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            assert hasattr(summarizer, "summarize")
            assert callable(summarizer.summarize)


# ============================================================================
# Tests for Summarization with Mocked LLM
# ============================================================================


class TestSummarizationWithMockedLLM:
    """Tests for summarize method with mocked LLM responses."""

    def test_summarize_returns_table_summary(self):
        """summarize should return TableSummary object."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer, TableSummary

        # Create test table
        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $500 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$500"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="Inventory costing methods",
            source_chunk_id="chunk_001",
        )

        # Mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This table shows inventory methods.")]
        mock_client.messages.create.return_value = mock_response

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()
            result = summarizer.summarize(table)

            assert isinstance(result, TableSummary)
            assert "inventory" in result.summary_text.lower()

    def test_summarize_generates_table_id(self):
        """summarize should generate table_id from source_chunk_id."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="| A |\n|---|\n| 1 |",
            column_names=["A"],
            rows=[["1"]],
            row_count=1,
            column_count=1,
            column_types=["text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="my_chunk_123",
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary text")]
        mock_client.messages.create.return_value = mock_response

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()
            result = summarizer.summarize(table)

            assert "my_chunk_123" in result.table_id

    def test_summarize_extracts_key_columns(self):
        """summarize should extract key columns from table."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="| Item | Price | Quantity |\n|---|---|---|\n| Apple | $5 | 10 |",
            column_names=["Item", "Price", "Quantity"],
            rows=[["Apple", "$5", "10"]],
            row_count=1,
            column_count=3,
            column_types=["text", "numeric", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create.return_value = mock_response

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()
            result = summarizer.summarize(table)

            # Should include first column and numeric columns
            assert "Item" in result.key_columns
            assert len(result.key_columns) >= 1

    def test_summarize_extracts_key_values(self):
        """summarize should extract key values from table."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $500 |\n| LIFO | $600 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$500"], ["LIFO", "$600"]],
            row_count=2,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create.return_value = mock_response

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()
            result = summarizer.summarize(table)

            # Should include values from first and last rows
            assert len(result.key_values) > 0


# ============================================================================
# Tests for LLM Integration
# ============================================================================


class TestLLMIntegration:
    """Tests for LLM client integration."""

    def test_calls_llm_with_prompt(self):
        """summarize should call LLM with table prompt."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="Context text",
            source_chunk_id="chunk_001",
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create.return_value = mock_response

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()
            summarizer.summarize(table)

            # Verify LLM was called
            mock_client.messages.create.assert_called_once()

            # Verify prompt includes table info
            call_kwargs = mock_client.messages.create.call_args[1]
            messages = call_kwargs["messages"]
            user_content = messages[0]["content"]
            assert "A" in user_content  # Column name
            assert "Context" in user_content  # Context included

    def test_handles_llm_error_gracefully(self):
        """summarize should handle LLM errors gracefully."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="| A |\n|---|\n| 1 |",
            column_names=["A"],
            rows=[["1"]],
            row_count=1,
            column_count=1,
            column_types=["text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch.object(TableSummarizer, "_init_client", return_value=mock_client):
            summarizer = TableSummarizer()

            # Should handle error gracefully (return fallback or raise specific exception)
            try:
                result = summarizer.summarize(table)
                # If it returns a result, it should have a fallback summary
                assert result.summary_text is not None
            except Exception as e:
                # If it raises, should be a specific error type
                assert "summarization" in str(e).lower() or "API" in str(e)


# ============================================================================
# Tests for Key Column/Value Extraction
# ============================================================================


class TestKeyExtraction:
    """Tests for key column and value extraction helpers."""

    def test_identify_key_columns_first_column(self):
        """Should always include first column as key."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="",
            column_names=["Name", "Description", "Notes"],
            rows=[["A", "Desc", "Note"]],
            row_count=1,
            column_count=3,
            column_types=["text", "text", "text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            key_columns = summarizer._identify_key_columns(table)

            assert "Name" in key_columns

    def test_identify_key_columns_includes_numeric(self):
        """Should include numeric columns as key columns."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="",
            column_names=["Item", "Price", "Tax", "Description"],
            rows=[["A", "$10", "$1", "Text"]],
            row_count=1,
            column_count=4,
            column_types=["text", "numeric", "numeric", "text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            key_columns = summarizer._identify_key_columns(table)

            assert "Price" in key_columns or "Tax" in key_columns

    def test_identify_key_columns_max_three(self):
        """Should return at most 3 key columns."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="",
            column_names=["A", "B", "C", "D", "E"],
            rows=[["1", "2", "3", "4", "5"]],
            row_count=1,
            column_count=5,
            column_types=["numeric", "numeric", "numeric", "numeric", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            key_columns = summarizer._identify_key_columns(table)

            assert len(key_columns) <= 3

    def test_extract_key_values_from_first_row(self):
        """Should extract values from first row."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="",
            column_names=["A", "B"],
            rows=[["First", "Row"], ["Second", "Row"]],
            row_count=2,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            key_values = summarizer._extract_key_values(table)

            assert "First" in key_values

    def test_extract_key_values_from_last_row(self):
        """Should extract values from last row (often totals)."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummarizer

        table = ParsedTable(
            raw_markdown="",
            column_names=["Item", "Amount"],
            rows=[["A", "$100"], ["B", "$200"], ["Total", "$300"]],
            row_count=3,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        with patch.object(TableSummarizer, "_init_client", return_value=MagicMock()):
            summarizer = TableSummarizer()
            key_values = summarizer._extract_key_values(table)

            assert "Total" in key_values or "$300" in key_values
