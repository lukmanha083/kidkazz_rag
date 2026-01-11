"""Tests for metadata enrichment."""

import pytest

from src.chunker.chunker import Chunk
from src.chunker.metadata import (
    ChunkMetadata,
    detect_content_features,
    enrich_all_chunks,
    enrich_chunk_metadata,
    extract_topic_tags,
    filter_chunks_by_topic,
    filter_chunks_by_type,
    infer_semantic_type,
)


class TestInferSemanticType:
    """Tests for infer_semantic_type function."""

    def test_detects_definition(self):
        """Should detect definition content."""
        content = "A function is defined as a mapping between sets."
        assert infer_semantic_type(content) == "definition"

    def test_detects_definition_is_a(self):
        """Should detect 'is a' pattern."""
        content = "A variable is a named storage location."
        assert infer_semantic_type(content) == "definition"

    def test_detects_example(self):
        """Should detect example content."""
        content = "For example, consider the following case."
        assert infer_semantic_type(content) == "example"

    def test_detects_example_eg(self):
        """Should detect 'for instance' pattern."""
        content = "For instance, consider the case of sorting algorithms."
        assert infer_semantic_type(content) == "example"

    def test_detects_procedure(self):
        """Should detect procedure/steps content."""
        # Need 2+ different procedure patterns to match
        content = "First, open the file. Then, read the data."
        assert infer_semantic_type(content) == "procedure"

    def test_detects_procedure_first_then(self):
        """Should detect first/then pattern."""
        content = "First, initialize the variable. Then, assign a value."
        assert infer_semantic_type(content) == "procedure"

    def test_detects_theorem(self):
        """Should detect theorem content."""
        content = "Theorem 1: For all x, if P(x) then Q(x)."
        assert infer_semantic_type(content) == "theorem"

    def test_detects_proof(self):
        """Should detect proof content."""
        content = "Proof: Assume the contrary. Q.E.D."
        assert infer_semantic_type(content) == "theorem"

    def test_defaults_to_narrative(self):
        """Should default to narrative for general content."""
        content = "This paragraph describes the history of computing."
        assert infer_semantic_type(content) == "narrative"

    def test_empty_content(self):
        """Should return narrative for empty content."""
        assert infer_semantic_type("") == "narrative"


class TestExtractTopicTags:
    """Tests for extract_topic_tags function."""

    def test_extracts_bold_text(self):
        """Should extract bold emphasized terms."""
        content = "The **quick sort** algorithm is efficient."
        tags = extract_topic_tags(content)
        assert "quick sort" in tags

    def test_extracts_italic_text(self):
        """Should extract italic emphasized terms."""
        content = "This is called _dynamic programming_."
        tags = extract_topic_tags(content)
        assert "dynamic programming" in tags

    def test_extracts_defined_terms(self):
        """Should extract terms after 'is a'."""
        content = "A hash table is a data structure."
        tags = extract_topic_tags(content)
        assert any("data structure" in t for t in tags)

    def test_extracts_capitalized_phrases(self):
        """Should extract multi-word capitalized phrases."""
        content = "The Quick Sort Algorithm was invented by Hoare."
        tags = extract_topic_tags(content)
        assert any("quick sort" in t.lower() for t in tags)

    def test_limits_max_tags(self):
        """Should respect max_tags limit."""
        content = "**Term1** **Term2** **Term3** **Term4** **Term5** **Term6**"
        tags = extract_topic_tags(content, max_tags=3)
        assert len(tags) <= 3

    def test_empty_content(self):
        """Should return empty list for empty content."""
        tags = extract_topic_tags("")
        assert tags == []

    def test_no_duplicates(self):
        """Should not return duplicate tags."""
        content = "**Python** is great. I love **Python**."
        tags = extract_topic_tags(content)
        assert len(tags) == len(set(tags))


class TestDetectContentFeatures:
    """Tests for detect_content_features function."""

    def test_detects_table(self):
        """Should detect markdown table."""
        content = "| A | B |\n|---|---|\n| 1 | 2 |"
        features = detect_content_features(content)
        assert features["has_table"] is True

    def test_detects_code_block(self):
        """Should detect code block."""
        content = "```python\nprint('hello')\n```"
        features = detect_content_features(content)
        assert features["has_code"] is True

    def test_detects_inline_code(self):
        """Should detect inline code."""
        content = "Use the `print()` function."
        features = detect_content_features(content)
        assert features["has_code"] is True

    def test_detects_math_dollar(self):
        """Should detect $ math."""
        content = "The formula is $x^2 + y^2$."
        features = detect_content_features(content)
        assert features["has_math"] is True

    def test_detects_math_double_dollar(self):
        """Should detect $$ math block."""
        content = "$$\nE = mc^2\n$$"
        features = detect_content_features(content)
        assert features["has_math"] is True

    def test_detects_bullet_list(self):
        """Should detect bullet list."""
        content = "- Item 1\n- Item 2"
        features = detect_content_features(content)
        assert features["has_list"] is True

    def test_detects_numbered_list(self):
        """Should detect numbered list."""
        content = "1. First\n2. Second"
        features = detect_content_features(content)
        assert features["has_list"] is True

    def test_no_features(self):
        """Should return all False for plain text."""
        content = "Just some plain text here."
        features = detect_content_features(content)
        assert not any(features.values())


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_create_metadata(self):
        """Should create metadata with required fields."""
        meta = ChunkMetadata(
            chunk_id="test_1",
            document_id="doc_1",
            semantic_type="definition",
        )
        assert meta.chunk_id == "test_1"
        assert meta.document_id == "doc_1"
        assert meta.semantic_type == "definition"

    def test_default_values(self):
        """Should have sensible defaults."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        assert meta.topic_tags == []
        assert meta.child_ids == []
        assert meta.sibling_ids == []
        assert meta.hierarchy_level == 2

    def test_to_dict(self):
        """Should convert to dictionary."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="example",
            has_code=True,
        )
        d = meta.to_dict()
        assert d["chunk_id"] == "test"
        assert d["semantic_type"] == "example"
        assert d["has_code"] is True


class TestEnrichChunkMetadata:
    """Tests for enrich_chunk_metadata function."""

    def test_enriches_chunk(self):
        """Should create metadata from chunk."""
        chunk = Chunk(
            id="chunk_1",
            content="A function is defined as a mapping.",
            level=2,
            token_count=10,
            parent_id="parent_1",
            section_path=["Chapter 1", "Section 1.1"],
        )
        meta = enrich_chunk_metadata(chunk, "doc_1", sequence_position=5)

        assert meta.chunk_id == "chunk_1"
        assert meta.document_id == "doc_1"
        assert meta.semantic_type == "definition"
        assert meta.sequence_position == 5
        assert meta.parent_id == "parent_1"
        assert meta.section_path == ["Chapter 1", "Section 1.1"]

    def test_includes_sibling_ids(self):
        """Should include sibling IDs if provided."""
        chunk = Chunk(id="c2", content="text", level=2, token_count=5)
        meta = enrich_chunk_metadata(
            chunk, "doc", 0, sibling_ids=["c1", "c3"]
        )
        assert meta.sibling_ids == ["c1", "c3"]

    def test_detects_features(self):
        """Should detect content features."""
        chunk = Chunk(
            id="c1",
            content="```python\ncode\n```",
            level=2,
            token_count=5,
        )
        meta = enrich_chunk_metadata(chunk, "doc", 0)
        assert meta.has_code is True


class TestEnrichAllChunks:
    """Tests for enrich_all_chunks function."""

    def test_enriches_all(self):
        """Should enrich all chunks."""
        chunks = [
            Chunk(id="c1", content="First chunk", level=2, token_count=5),
            Chunk(id="c2", content="Second chunk", level=2, token_count=5),
        ]
        metadata = enrich_all_chunks(chunks, "doc_1")
        assert len(metadata) == 2
        assert metadata[0].chunk_id == "c1"
        assert metadata[1].chunk_id == "c2"

    def test_sets_sequence_positions(self):
        """Should set correct sequence positions."""
        chunks = [
            Chunk(id="c1", content="First", level=2, token_count=5),
            Chunk(id="c2", content="Second", level=2, token_count=5),
            Chunk(id="c3", content="Third", level=2, token_count=5),
        ]
        metadata = enrich_all_chunks(chunks, "doc")
        assert metadata[0].sequence_position == 0
        assert metadata[1].sequence_position == 1
        assert metadata[2].sequence_position == 2

    def test_detects_siblings(self):
        """Should detect sibling chunks."""
        parent = Chunk(
            id="p1",
            content="Parent",
            level=1,
            token_count=10,
            child_ids=["c1", "c2"],
        )
        c1 = Chunk(id="c1", content="Child 1", level=2, token_count=5, parent_id="p1")
        c2 = Chunk(id="c2", content="Child 2", level=2, token_count=5, parent_id="p1")
        chunks = [parent, c1, c2]

        metadata = enrich_all_chunks(chunks, "doc")

        # c1's siblings should include c2, and vice versa
        c1_meta = next(m for m in metadata if m.chunk_id == "c1")
        c2_meta = next(m for m in metadata if m.chunk_id == "c2")
        assert "c2" in c1_meta.sibling_ids
        assert "c1" in c2_meta.sibling_ids


class TestFilterChunksByType:
    """Tests for filter_chunks_by_type function."""

    def test_filters_by_type(self):
        """Should filter chunks by semantic type."""
        chunks = [
            Chunk(id="c1", content="is defined as", level=2, token_count=5),
            Chunk(id="c2", content="For example", level=2, token_count=5),
            Chunk(id="c3", content="is a type", level=2, token_count=5),
        ]
        metadata = enrich_all_chunks(chunks, "doc")

        definitions = filter_chunks_by_type(chunks, metadata, "definition")
        assert len(definitions) == 2  # c1 and c3

        examples = filter_chunks_by_type(chunks, metadata, "example")
        assert len(examples) == 1  # c2

    def test_returns_empty_for_no_match(self):
        """Should return empty list if no matches."""
        chunks = [Chunk(id="c1", content="plain text", level=2, token_count=5)]
        metadata = enrich_all_chunks(chunks, "doc")
        result = filter_chunks_by_type(chunks, metadata, "theorem")
        assert result == []


class TestFilterChunksByTopic:
    """Tests for filter_chunks_by_topic function."""

    def test_filters_by_topic(self):
        """Should filter chunks containing topic tag."""
        chunks = [
            Chunk(id="c1", content="**Python** is great", level=2, token_count=5),
            Chunk(id="c2", content="**Java** is popular", level=2, token_count=5),
            Chunk(id="c3", content="**Python** and **Java**", level=2, token_count=5),
        ]
        metadata = enrich_all_chunks(chunks, "doc")

        python_chunks = filter_chunks_by_topic(chunks, metadata, "python")
        assert len(python_chunks) == 2  # c1 and c3

    def test_case_insensitive(self):
        """Should be case insensitive."""
        chunks = [
            Chunk(id="c1", content="**PYTHON** rules", level=2, token_count=5),
        ]
        metadata = enrich_all_chunks(chunks, "doc")

        result = filter_chunks_by_topic(chunks, metadata, "python")
        assert len(result) == 1

    def test_returns_empty_for_no_match(self):
        """Should return empty if topic not found."""
        chunks = [Chunk(id="c1", content="plain text", level=2, token_count=5)]
        metadata = enrich_all_chunks(chunks, "doc")
        result = filter_chunks_by_topic(chunks, metadata, "nonexistent")
        assert result == []


class TestExtractHeaderInfo:
    """Tests for extracting header info from chunk content (TDD)."""

    def test_extract_header_info_h1(self):
        """Should extract h1 header info."""
        from src.chunker.metadata import extract_header_info

        content = "# Introduction\n\nThis is the introduction."
        result = extract_header_info(content)

        assert result["header_text"] == "Introduction"
        assert result["header_level"] == 1
        assert result["block_type"] == "Header"

    def test_extract_header_info_h2(self):
        """Should extract h2 header info."""
        from src.chunker.metadata import extract_header_info

        content = "## Getting Started\n\nLet's begin."
        result = extract_header_info(content)

        assert result["header_text"] == "Getting Started"
        assert result["header_level"] == 2
        assert result["block_type"] == "Header"

    def test_extract_header_info_h3(self):
        """Should extract h3 header info."""
        from src.chunker.metadata import extract_header_info

        content = "### Installation Steps\n\nFollow these steps."
        result = extract_header_info(content)

        assert result["header_text"] == "Installation Steps"
        assert result["header_level"] == 3
        assert result["block_type"] == "Header"

    def test_extract_header_info_h6(self):
        """Should extract h6 header info."""
        from src.chunker.metadata import extract_header_info

        content = "###### Deep Nested Section\n\nContent here."
        result = extract_header_info(content)

        assert result["header_text"] == "Deep Nested Section"
        assert result["header_level"] == 6
        assert result["block_type"] == "Header"

    def test_extract_header_info_no_header(self):
        """Should return empty dict for content without header."""
        from src.chunker.metadata import extract_header_info

        content = "This is just plain text.\n\nNo header here."
        result = extract_header_info(content)

        assert result == {}

    def test_extract_header_info_empty_content(self):
        """Should return empty dict for empty content."""
        from src.chunker.metadata import extract_header_info

        result = extract_header_info("")
        assert result == {}

    def test_extract_header_info_header_only(self):
        """Should extract header even if content is header only."""
        from src.chunker.metadata import extract_header_info

        content = "# Title Only"
        result = extract_header_info(content)

        assert result["header_text"] == "Title Only"
        assert result["header_level"] == 1

    def test_extract_header_info_strips_whitespace(self):
        """Should strip whitespace from header text."""
        from src.chunker.metadata import extract_header_info

        content = "##   Spaced Title   \n\nContent"
        result = extract_header_info(content)

        assert result["header_text"] == "Spaced Title"

    def test_extract_header_info_ignores_non_first_line_headers(self):
        """Should only detect headers on the first line."""
        from src.chunker.metadata import extract_header_info

        content = "Some intro text\n\n## This header is not first"
        result = extract_header_info(content)

        assert result == {}

    def test_extract_header_info_invalid_header_no_space(self):
        """Should not match headers without space after #."""
        from src.chunker.metadata import extract_header_info

        content = "#NoSpaceHeader\n\nContent"
        result = extract_header_info(content)

        assert result == {}


class TestEnrichChunkMetadataWithHeaders:
    """Tests for header metadata in enrichment pipeline (TDD)."""

    def test_enrich_chunk_metadata_extracts_header(self):
        """Should extract header info during enrichment."""
        chunk = Chunk(
            id="test_1",
            content="# Chapter 1: Introduction\n\nThis is the chapter content.",
            level=1,
            token_count=15,
        )
        result = enrich_chunk_metadata(chunk, "doc_1", 0)

        assert result.header_text == "Chapter 1: Introduction"
        assert result.header_level == 1
        assert result.block_type == "Header"

    def test_enrich_chunk_metadata_no_header(self):
        """Should leave header fields None when no header present."""
        chunk = Chunk(
            id="test_1",
            content="Just some plain text without any header.",
            level=2,
            token_count=10,
        )
        result = enrich_chunk_metadata(chunk, "doc_1", 0)

        assert result.header_text is None
        assert result.header_level is None
        assert result.block_type is None

    def test_enrich_all_chunks_preserves_headers(self):
        """Should preserve header info across all chunks."""
        chunks = [
            Chunk(
                id="c1",
                content="# Main Title\n\nIntro text.",
                level=1,
                token_count=10,
            ),
            Chunk(
                id="c2",
                content="## Section One\n\nSection content.",
                level=2,
                token_count=10,
            ),
            Chunk(
                id="c3",
                content="Plain paragraph content.",
                level=2,
                token_count=5,
            ),
        ]

        metadata_list = enrich_all_chunks(chunks, "doc_1")

        # First chunk has h1
        assert metadata_list[0].header_text == "Main Title"
        assert metadata_list[0].header_level == 1

        # Second chunk has h2
        assert metadata_list[1].header_text == "Section One"
        assert metadata_list[1].header_level == 2

        # Third chunk has no header
        assert metadata_list[2].header_text is None
        assert metadata_list[2].header_level is None


class TestHeaderMetadataFields:
    """Tests for header-related metadata fields (TDD)."""

    def test_chunk_metadata_has_header_text_field(self):
        """ChunkMetadata should have header_text field."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            header_text="Introduction",
        )
        assert meta.header_text == "Introduction"

    def test_chunk_metadata_header_text_defaults_to_none(self):
        """header_text should default to None."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        assert meta.header_text is None

    def test_chunk_metadata_has_header_level_field(self):
        """ChunkMetadata should have header_level field."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            header_level=2,
        )
        assert meta.header_level == 2

    def test_chunk_metadata_header_level_defaults_to_none(self):
        """header_level should default to None."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        assert meta.header_level is None

    def test_chunk_metadata_has_block_type_field(self):
        """ChunkMetadata should have block_type field."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            block_type="Header",
        )
        assert meta.block_type == "Header"

    def test_chunk_metadata_block_type_defaults_to_none(self):
        """block_type should default to None."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        assert meta.block_type is None

    def test_to_dict_includes_header_fields(self):
        """to_dict should include header fields."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            header_text="Chapter 1",
            header_level=1,
            block_type="Header",
        )
        d = meta.to_dict()
        assert d["header_text"] == "Chapter 1"
        assert d["header_level"] == 1
        assert d["block_type"] == "Header"

    def test_to_dict_includes_none_header_fields(self):
        """to_dict should include None values for unset header fields."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        d = meta.to_dict()
        assert "header_text" in d
        assert "header_level" in d
        assert "block_type" in d
        assert d["header_text"] is None
        assert d["header_level"] is None
        assert d["block_type"] is None
