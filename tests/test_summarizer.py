"""Tests for document summarization module."""

import pytest
from unittest.mock import MagicMock, patch


class TestSummaryDataclass:
    """Tests for Summary dataclass."""

    def test_summary_creation(self):
        """Should create a Summary with required fields."""
        from src.chunker.summarizer import Summary

        summary = Summary(
            summary_id="summary_doc1_document",
            content="This is a test summary.",
            level="document",
            source_id="doc1",
            document_id="doc1",
        )

        assert summary.summary_id == "summary_doc1_document"
        assert summary.content == "This is a test summary."
        assert summary.level == "document"
        assert summary.source_id == "doc1"
        assert summary.document_id == "doc1"

    def test_summary_word_count_auto_calculated(self):
        """Should auto-calculate word count if not provided."""
        from src.chunker.summarizer import Summary

        summary = Summary(
            summary_id="test",
            content="One two three four five",
            level="section",
            source_id="chunk1",
            document_id="doc1",
        )

        assert summary.word_count == 5

    def test_summary_with_key_points(self):
        """Should store key points as list."""
        from src.chunker.summarizer import Summary

        summary = Summary(
            summary_id="test",
            content="Summary content",
            level="chapter",
            source_id="chunk1",
            document_id="doc1",
            key_points=["Point 1", "Point 2", "Point 3"],
        )

        assert len(summary.key_points) == 3
        assert "Point 1" in summary.key_points

    def test_summary_with_embedding(self):
        """Should store embedding vector."""
        from src.chunker.summarizer import Summary

        embedding = [0.1, 0.2, 0.3, 0.4]
        summary = Summary(
            summary_id="test",
            content="Summary content",
            level="section",
            source_id="chunk1",
            document_id="doc1",
            embedding=embedding,
        )

        assert summary.embedding == embedding

    def test_summary_parent_summary_id(self):
        """Should store parent summary ID for hierarchy."""
        from src.chunker.summarizer import Summary

        summary = Summary(
            summary_id="summary_chunk1_section",
            content="Section summary",
            level="section",
            source_id="chunk1",
            document_id="doc1",
            parent_summary_id="summary_chapter1_chapter",
        )

        assert summary.parent_summary_id == "summary_chapter1_chapter"


class TestConceptTypeCoercion:
    """Tests for concept type coercion validator."""

    def test_valid_concept_type(self):
        """Should accept valid concept types."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="FIFO",
            concept_type="method",
            definition="First-in, first-out inventory method",
        )
        assert concept.concept_type == ConceptType.METHOD

    def test_coerce_unknown_type_to_term(self):
        """Should coerce unknown types to 'term'."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="Some Concept",
            concept_type="unknown_type",
            definition="A concept definition",
        )
        assert concept.concept_type == ConceptType.TERM

    def test_coerce_technique_to_method(self):
        """Should map 'technique' to 'method'."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="Crop Rotation",
            concept_type="technique",
            definition="A farming technique",
        )
        assert concept.concept_type == ConceptType.METHOD

    def test_coerce_neural_network_to_model(self):
        """Should map 'neural_network' to 'model'."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="CNN",
            concept_type="neural_network",
            definition="Convolutional Neural Network",
        )
        assert concept.concept_type == ConceptType.MODEL

    def test_coerce_illness_to_disease(self):
        """Should map 'illness' to 'disease' for veterinary concepts."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="Mastitis",
            concept_type="illness",
            definition="Inflammation of udder tissue",
        )
        assert concept.concept_type == ConceptType.DISEASE

    def test_coerce_species_to_organism(self):
        """Should map 'species' to 'organism' for biology concepts."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="E. coli",
            concept_type="species",
            definition="Escherichia coli bacteria",
        )
        assert concept.concept_type == ConceptType.ORGANISM

    def test_handles_enum_directly(self):
        """Should accept ConceptType enum directly."""
        from src.chunker.summarizer import ExtractedConcept, ConceptType

        concept = ExtractedConcept(
            name="Safety Stock",
            concept_type=ConceptType.TERM,
            definition="Buffer inventory",
        )
        assert concept.concept_type == ConceptType.TERM


class TestPydanticModels:
    """Tests for Pydantic output models."""

    def test_section_summary_output(self):
        """SectionSummaryOutput should validate correctly."""
        from src.chunker.summarizer import SectionSummaryOutput

        output = SectionSummaryOutput(
            summary="This section covers the basics.",
            key_points=["Point A", "Point B"],
        )

        assert output.summary == "This section covers the basics."
        assert len(output.key_points) == 2

    def test_chapter_summary_output(self):
        """ChapterSummaryOutput should validate correctly."""
        from src.chunker.summarizer import ChapterSummaryOutput

        output = ChapterSummaryOutput(
            chapter_title="Chapter 3: Inventory Valuation",
            summary="This chapter introduces key concepts.",
            key_points=["Concept 1", "Concept 2", "Concept 3"],
        )

        assert "chapter" in output.summary.lower()
        assert len(output.key_points) == 3
        assert output.chapter_title == "Chapter 3: Inventory Valuation"

    def test_chapter_summary_output_default_title(self):
        """ChapterSummaryOutput should default chapter_title to empty string."""
        from src.chunker.summarizer import ChapterSummaryOutput

        output = ChapterSummaryOutput(
            summary="This chapter covers methods.",
            key_points=["Method A"],
        )

        assert output.chapter_title == ""

    def test_document_summary_output(self):
        """DocumentSummaryOutput should validate correctly."""
        from src.chunker.summarizer import DocumentSummaryOutput

        output = DocumentSummaryOutput(
            summary="This document provides a comprehensive overview.",
            key_points=["Topic 1", "Topic 2"],
        )

        assert "document" in output.summary.lower()


class TestDocumentSummarizer:
    """Tests for DocumentSummarizer class."""

    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", False)
    @patch("src.chunker.summarizer.instructor", None)
    def test_init_raises_without_instructor(self):
        """Should raise ImportError if instructor not available."""
        from src.chunker.summarizer import DocumentSummarizer

        with pytest.raises(ImportError):
            DocumentSummarizer()

    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True)
    @patch("src.chunker.summarizer.instructor")
    def test_init_with_custom_provider(self, mock_instructor):
        """Should accept custom provider."""
        mock_instructor.from_provider.return_value = MagicMock()

        from src.chunker.summarizer import DocumentSummarizer
        DocumentSummarizer(provider="anthropic/claude-opus-4-20250514")

        mock_instructor.from_provider.assert_called_once_with("anthropic/claude-opus-4-20250514")

    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True)
    @patch("src.chunker.summarizer.instructor")
    def test_summarize_section_returns_summary(self, mock_instructor):
        """Should return Summary using extractive method (no LLM call)."""
        from src.chunker.summarizer import DocumentSummarizer

        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client

        summarizer = DocumentSummarizer()
        summary = summarizer.summarize_section(
            chunk_content="FIFO is an inventory valuation method. It assumes first items purchased are first sold. This provides accurate cost tracking.",
            chunk_id="chunk_001",
            document_id="inventory_book",
            document_title="Inventory Accounting",
            section_path=["Chapter 1", "FIFO Method"],
        )

        assert summary.level == "section"
        assert summary.source_id == "chunk_001"
        assert summary.document_id == "inventory_book"
        assert summary.content  # non-empty
        # No LLM call should be made for sections
        mock_client.create.assert_not_called()

    @patch("src.chunker.summarizer.extractive_summarize", None)
    @patch("src.chunker.summarizer.extract_keywords", None)
    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True)
    @patch("src.chunker.summarizer.instructor")
    def test_summarize_section_handles_no_extractive(self, mock_instructor):
        """Should return fallback summary when extractive libs unavailable."""
        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client

        from src.chunker.summarizer import DocumentSummarizer

        summarizer = DocumentSummarizer()
        summary = summarizer.summarize_section(
            chunk_content="Some content about inventory management and cost tracking methods.",
            chunk_id="chunk_001",
            document_id="doc1",
            document_title="Test Doc",
        )

        # Should still return a summary (fallback), no LLM call
        assert summary.level == "section"
        assert summary.source_id == "chunk_001"
        assert summary.content  # non-empty
        mock_client.create.assert_not_called()

    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True)
    @patch("src.chunker.summarizer.instructor")
    def test_summarize_chapter_uses_section_summaries(self, mock_instructor):
        """Should use section summaries for chapter summary."""
        from src.chunker.summarizer import DocumentSummarizer, ChapterSummaryOutput, Summary

        mock_client = MagicMock()
        mock_client.create.return_value = ChapterSummaryOutput(
            chapter_title="Chapter 5: Inventory Valuation",
            summary="This chapter covers inventory valuation methods including FIFO and LIFO.",
            key_points=["FIFO", "LIFO", "Weighted Average"],
        )
        mock_instructor.from_provider.return_value = mock_client

        summarizer = DocumentSummarizer()

        # Create section summaries
        section_summaries = [
            Summary(
                summary_id="s1",
                content="FIFO section summary",
                level="section",
                source_id="sec1",
                document_id="doc1",
            ),
            Summary(
                summary_id="s2",
                content="LIFO section summary",
                level="section",
                source_id="sec2",
                document_id="doc1",
            ),
        ]

        chapter_summary = summarizer.summarize_chapter(
            chunk_id="chapter_001",
            chunk_content="Chapter content...",
            section_summaries=section_summaries,
            document_id="doc1",
            document_title="Inventory Accounting",
            chapter_title="Inventory Valuation",
        )

        assert chapter_summary.level == "chapter"
        assert "inventory" in chapter_summary.content.lower()
        # Chapter title should be stored as sentinel in key_points
        assert any(kp.startswith("__title__:") for kp in chapter_summary.key_points)
        assert "__title__:Chapter 5: Inventory Valuation" in chapter_summary.key_points

    @patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True)
    @patch("src.chunker.summarizer.instructor")
    def test_generate_all_summaries_creates_hierarchy(self, mock_instructor):
        """Should generate summaries at all levels.

        Sections use extractive methods (no LLM calls).
        Only chapter + document use LLM (2 API calls).
        """
        from src.chunker.summarizer import (
            DocumentSummarizer,
            ChapterSummaryOutput,
            DocumentSummaryOutput,
        )

        # Mock: only 2 LLM calls (chapter + document), sections are extractive
        mock_client = MagicMock()
        mock_client.create.side_effect = [
            # Chapter summary (1 LLM call)
            ChapterSummaryOutput(
                chapter_title="Chapter 1: Methods",
                summary="Chapter summary",
                key_points=["C", "D"],
            ),
            # Document summary (1 LLM call)
            DocumentSummaryOutput(summary="Document summary", key_points=["E", "F", "G"]),
        ]
        mock_instructor.from_provider.return_value = mock_client

        summarizer = DocumentSummarizer()

        chunks = [
            {"id": "l1_1", "content": "Chapter content", "level": 1, "child_ids": ["l2_1", "l2_2"]},
            {"id": "l2_1", "content": "Section 1 about FIFO method. It is commonly used.", "level": 2, "section_path": ["Ch1", "Sec1"]},
            {"id": "l2_2", "content": "Section 2 about LIFO method. It provides tax benefits.", "level": 2, "section_path": ["Ch1", "Sec2"]},
        ]

        summaries, concepts = summarizer.generate_all_summaries(
            document_id="doc1",
            document_title="Test Document",
            chunks=chunks,
        )

        # Should have: 2 sections + 1 chapter + 1 document = 4 summaries
        assert len(summaries) == 4

        # Check levels
        levels = [s.level for s in summaries]
        assert levels.count("section") == 2
        assert levels.count("chapter") == 1
        assert levels.count("document") == 1

        # Only 2 LLM calls (chapter + document), not 4
        assert mock_client.create.call_count == 2

        # Concepts should be a list (may be empty in mock)
        assert isinstance(concepts, list)


class TestSummaryStorageQueries:
    """Tests for summary storage query classes."""

    def test_add_summary_query(self):
        """AddSummary query should format parameters correctly."""
        from src.storage.queries import AddSummary

        props = {
            "summary_id": "summary_doc1_document",
            "content": "Test summary content",
            "level": "document",
            "source_id": "doc1",
            "document_id": "doc1",
            "key_points": '["Point 1"]',
            "word_count": 10,
        }

        query = AddSummary(props)
        params = query.query()

        assert params[0]["summary_id"] == "summary_doc1_document"
        assert params[0]["level"] == "document"

    def test_get_summary_query(self):
        """GetSummary query should format parameters correctly."""
        from src.storage.queries import GetSummary

        query = GetSummary("summary_doc1_document")
        params = query.query()

        assert params[0]["summary_id"] == "summary_doc1_document"

    def test_get_document_summaries_query(self):
        """GetDocumentSummaries query should format parameters correctly."""
        from src.storage.queries import GetDocumentSummaries

        query = GetDocumentSummaries("doc1", level="chapter")
        params = query.query()

        assert params[0]["document_id"] == "doc1"
        assert params[0]["level"] == "chapter"

    def test_search_similar_summaries_query(self):
        """SearchSimilarSummaries query should format parameters correctly."""
        from src.storage.queries import SearchSimilarSummaries

        embedding = [0.1, 0.2, 0.3]
        query = SearchSimilarSummaries(
            query_embedding=embedding,
            limit=5,
            level="document",
        )
        params = query.query()

        # HelixQL uses query_vec and top_k parameter names
        assert params[0]["query_vec"] == embedding
        assert params[0]["top_k"] == 5
        # level is used for post-processing, not in HelixQL query
        assert query.level == "document"

    def test_delete_document_summaries_query(self):
        """DeleteDocumentSummaries query should format parameters correctly."""
        from src.storage.queries import DeleteDocumentSummaries

        query = DeleteDocumentSummaries("doc1")
        params = query.query()

        assert params[0]["document_id"] == "doc1"


class TestChapterTitleExtraction:
    """Tests for chapter title extraction helpers."""

    def test_extract_chapter_title_from_key_points(self):
        """Should extract chapter title from __title__: sentinel."""
        from src.chunker.summarizer import DocumentSummarizer, Summary

        summary = Summary(
            summary_id="test",
            content="Summary content",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
            key_points=["__title__:Chapter 2: Safety Stock", "Point A", "Point B"],
        )
        title = DocumentSummarizer._extract_chapter_title(summary)
        assert title == "Chapter 2: Safety Stock"

    def test_extract_chapter_title_returns_none_when_missing(self):
        """Should return None when no title sentinel present."""
        from src.chunker.summarizer import DocumentSummarizer, Summary

        summary = Summary(
            summary_id="test",
            content="Summary content",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
            key_points=["Point A", "Point B"],
        )
        title = DocumentSummarizer._extract_chapter_title(summary)
        assert title is None

    def test_get_display_key_points_filters_sentinel(self):
        """Should filter out __title__: sentinel from key_points."""
        from src.chunker.summarizer import DocumentSummarizer, Summary

        summary = Summary(
            summary_id="test",
            content="Summary content",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
            key_points=["__title__:Chapter 3: FIFO", "Point A", "Point B"],
        )
        display = DocumentSummarizer._get_display_key_points(summary)
        assert display == ["Point A", "Point B"]

    def test_format_chapter_for_doc_context_with_title(self):
        """Should use actual title instead of generic 'Chapter N'."""
        from src.chunker.summarizer import DocumentSummarizer, Summary

        summary = Summary(
            summary_id="test",
            content="This chapter covers cost methods.",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
            key_points=["__title__:Chapter 5: Cost Methods", "FIFO", "LIFO"],
        )
        context = DocumentSummarizer._format_chapter_for_doc_context(summary, 0)
        assert context.startswith("Chapter 5: Cost Methods:")
        assert "FIFO" in context
        assert "__title__" not in context

    def test_format_chapter_for_doc_context_without_title(self):
        """Should fall back to generic 'Chapter N' when no title."""
        from src.chunker.summarizer import DocumentSummarizer, Summary

        summary = Summary(
            summary_id="test",
            content="Content here.",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
            key_points=["Point A"],
        )
        context = DocumentSummarizer._format_chapter_for_doc_context(summary, 2)
        assert context.startswith("Chapter 3:")

    def test_result_to_summary_stores_title_sentinel(self):
        """_result_to_summary should store chapter_title as sentinel in key_points."""
        from src.chunker.summarizer import ChapterSummaryOutput, DocumentSummarizer

        output = ChapterSummaryOutput(
            chapter_title="Chapter 1: Introduction",
            summary="Introduction to the topic.",
            key_points=["Overview", "Scope"],
        )

        # Need a real summarizer instance
        with patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True), \
             patch("src.chunker.summarizer.instructor") as mock_instructor:
            mock_instructor.from_provider.return_value = MagicMock()
            summarizer = DocumentSummarizer()

        summary = summarizer._result_to_summary(
            result=output,
            summary_id="summary_ch1_chapter",
            level="chapter",
            source_id="ch1",
            document_id="doc1",
        )

        assert "__title__:Chapter 1: Introduction" in summary.key_points
        assert "Overview" in summary.key_points
        assert "Scope" in summary.key_points

    def test_result_to_summary_no_title_for_non_chapter(self):
        """_result_to_summary should not store title sentinel for non-chapter levels."""
        from src.chunker.summarizer import DocumentSummaryOutput, DocumentSummarizer

        output = DocumentSummaryOutput(
            summary="Document overview.",
            key_points=["Topic 1", "Topic 2"],
        )

        with patch("src.chunker.summarizer.INSTRUCTOR_AVAILABLE", True), \
             patch("src.chunker.summarizer.instructor") as mock_instructor:
            mock_instructor.from_provider.return_value = MagicMock()
            summarizer = DocumentSummarizer()

        summary = summarizer._result_to_summary(
            result=output,
            summary_id="summary_doc1_document",
            level="document",
            source_id="doc1",
            document_id="doc1",
        )

        assert not any(kp.startswith("__title__:") for kp in summary.key_points)


class TestCLIChapterHelpers:
    """Tests for CLI chapter title/sorting helpers."""

    def test_extract_chapter_title_from_summary_dict(self):
        """Should extract title from summary dict key_points."""
        from src.cli.commands.summarize import _extract_chapter_title_from_summary

        summary = {
            "key_points": ["__title__:Chapter 4: Warehousing", "Point A"],
        }
        assert _extract_chapter_title_from_summary(summary) == "Chapter 4: Warehousing"

    def test_extract_chapter_title_from_summary_dict_json_string(self):
        """Should handle key_points stored as JSON string."""
        import json
        from src.cli.commands.summarize import _extract_chapter_title_from_summary

        summary = {
            "key_points": json.dumps(["__title__:Part II: Methods", "Point B"]),
        }
        assert _extract_chapter_title_from_summary(summary) == "Part II: Methods"

    def test_extract_chapter_title_returns_none_when_missing(self):
        """Should return None when no title sentinel."""
        from src.cli.commands.summarize import _extract_chapter_title_from_summary

        summary = {"key_points": ["Point A", "Point B"]}
        assert _extract_chapter_title_from_summary(summary) is None

    def test_get_display_key_points_filters_sentinel(self):
        """Should filter out title sentinel from display."""
        from src.cli.commands.summarize import _get_display_key_points_from_summary

        summary = {
            "key_points": ["__title__:Chapter 1: Intro", "FIFO", "LIFO"],
        }
        result = _get_display_key_points_from_summary(summary)
        assert result == ["FIFO", "LIFO"]

    def test_chapter_sort_key_numeric(self):
        """Should extract trailing number from source_id."""
        from src.cli.commands.summarize import _chapter_sort_key

        assert _chapter_sort_key({"source_id": "inv_l1_5"})[0] == 5
        assert _chapter_sort_key({"source_id": "inv_l1_12"})[0] == 12
        assert _chapter_sort_key({"source_id": "wh_l1_0"})[0] == 0

    def test_chapter_sort_key_no_number(self):
        """Should fall back to high number for non-numeric source_ids."""
        from src.cli.commands.summarize import _chapter_sort_key

        assert _chapter_sort_key({"source_id": "abc"})[0] == 999999

    def test_chapter_sort_key_ordering(self):
        """Chapters should sort by numeric suffix."""
        from src.cli.commands.summarize import _chapter_sort_key

        chapters = [
            {"source_id": "inv_l1_10"},
            {"source_id": "inv_l1_2"},
            {"source_id": "inv_l1_1"},
            {"source_id": "inv_l1_5"},
        ]
        sorted_chapters = sorted(chapters, key=_chapter_sort_key)
        assert [c["source_id"] for c in sorted_chapters] == [
            "inv_l1_1", "inv_l1_2", "inv_l1_5", "inv_l1_10"
        ]


class TestMCPFormatterChapterTitle:
    """Tests for MCP formatter chapter title extraction."""

    def test_format_summary_extracts_chapter_title(self):
        """Should extract chapter_title from key_points sentinel."""
        from src.mcp_server.formatters import format_summary

        summary = {
            "summary_id": "s1",
            "content": "Chapter content",
            "level": "chapter",
            "source_id": "ch1",
            "document_id": "doc1",
            "key_points": ["__title__:Chapter 2: Costs", "Point A", "Point B"],
            "word_count": 10,
            "created_at": 0,
        }
        result = format_summary(summary)
        assert result["chapter_title"] == "Chapter 2: Costs"
        assert result["key_points"] == ["Point A", "Point B"]
        assert "__title__" not in str(result["key_points"])

    def test_format_summary_no_title_when_absent(self):
        """Should not include chapter_title when no sentinel present."""
        from src.mcp_server.formatters import format_summary

        summary = {
            "summary_id": "s1",
            "content": "Content",
            "level": "document",
            "source_id": "doc1",
            "document_id": "doc1",
            "key_points": ["Point A"],
            "word_count": 5,
            "created_at": 0,
        }
        result = format_summary(summary)
        assert "chapter_title" not in result
        assert result["key_points"] == ["Point A"]

    def test_format_summary_handles_json_string_key_points(self):
        """Should handle key_points as JSON string."""
        import json
        from src.mcp_server.formatters import format_summary

        summary = {
            "summary_id": "s1",
            "content": "Content",
            "level": "chapter",
            "source_id": "ch1",
            "document_id": "doc1",
            "key_points": json.dumps(["__title__:Part I: Basics", "Intro"]),
            "word_count": 5,
            "created_at": 0,
        }
        result = format_summary(summary)
        assert result["chapter_title"] == "Part I: Basics"
        assert result["key_points"] == ["Intro"]
