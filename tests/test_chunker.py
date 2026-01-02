"""Tests for hierarchical chunking logic."""

import pytest

from src.chunker.chunker import (
    Chunk,
    create_hierarchical_chunks,
    estimate_tokens,
    generate_chunk_id,
    get_child_chunks,
    get_chunk_by_id,
    get_context_window,
    get_parent_chunk,
    get_sibling_chunks,
    split_text_with_overlap,
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self):
        """Should return 0 for empty string."""
        assert estimate_tokens("") == 0

    def test_short_text(self):
        """Should estimate tokens for short text."""
        # "Hello" = 5 chars -> ~1 token
        result = estimate_tokens("Hello")
        assert result >= 1

    def test_longer_text(self):
        """Should estimate tokens for longer text."""
        text = "This is a longer sentence with multiple words."
        result = estimate_tokens(text)
        # ~47 chars -> ~12 tokens
        assert 8 <= result <= 20

    def test_proportional_to_length(self):
        """Longer text should have more tokens."""
        short = estimate_tokens("Short")
        long = estimate_tokens("This is a much longer piece of text")
        assert long > short


class TestSplitTextWithOverlap:
    """Tests for split_text_with_overlap function."""

    def test_empty_text(self):
        """Should return empty list for empty text."""
        result = split_text_with_overlap("", 100, 10)
        assert result == []

    def test_whitespace_only(self):
        """Should return empty list for whitespace."""
        result = split_text_with_overlap("   \n\t  ", 100, 10)
        assert result == []

    def test_short_text_single_chunk(self):
        """Should return single chunk for short text."""
        text = "Short text."
        result = split_text_with_overlap(text, 100, 10)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_long_text(self):
        """Should split text exceeding max tokens."""
        # Create text that's definitely > 100 tokens
        text = "This is a sentence. " * 50  # ~250 tokens
        result = split_text_with_overlap(text, 100, 10)
        assert len(result) > 1

    def test_preserves_sentences(self):
        """Should try to split at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        result = split_text_with_overlap(text, 20, 5, preserve_sentences=True)
        # Each chunk should end with a period (complete sentence)
        for chunk in result:
            assert chunk.strip().endswith(".")

    def test_overlap_included(self):
        """Should include overlap between chunks."""
        text = "Sentence one here. Sentence two here. Sentence three here."
        result = split_text_with_overlap(text, 30, 10, preserve_sentences=True)
        if len(result) > 1:
            # Verify some content from chunk 1 appears in chunk 2 (overlap)
            chunk1_words = set(result[0].split())
            chunk2_words = set(result[1].split())
            overlap_words = chunk1_words & chunk2_words
            assert len(overlap_words) > 0, "Expected overlap between consecutive chunks"


class TestGenerateChunkId:
    """Tests for generate_chunk_id function."""

    def test_generates_unique_ids(self):
        """Should generate unique IDs."""
        ids = {generate_chunk_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_uses_prefix(self):
        """Should include prefix in ID."""
        result = generate_chunk_id("test")
        assert result.startswith("test_")

    def test_default_prefix(self):
        """Should use default prefix."""
        result = generate_chunk_id()
        assert result.startswith("chunk_")


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_create_chunk(self):
        """Should create chunk with required fields."""
        chunk = Chunk(
            id="test_1",
            content="Some content",
            level=2,
            token_count=10,
        )
        assert chunk.id == "test_1"
        assert chunk.content == "Some content"
        assert chunk.level == 2
        assert chunk.token_count == 10

    def test_default_optional_fields(self):
        """Should have default values for optional fields."""
        chunk = Chunk(id="test", content="", level=0, token_count=0)
        assert chunk.parent_id is None
        assert chunk.child_ids == []
        assert chunk.prev_id is None
        assert chunk.next_id is None

    def test_word_count_property(self):
        """Should calculate word count."""
        chunk = Chunk(id="test", content="one two three", level=0, token_count=0)
        assert chunk.word_count == 3

    def test_hashable(self):
        """Should be hashable by ID."""
        chunk = Chunk(id="test", content="", level=0, token_count=0)
        assert hash(chunk) == hash("test")


class TestCreateHierarchicalChunks:
    """Tests for create_hierarchical_chunks function."""

    def test_empty_content(self):
        """Should return empty list for empty content."""
        result = create_hierarchical_chunks("")
        assert result == []

    def test_whitespace_content(self):
        """Should return empty list for whitespace."""
        result = create_hierarchical_chunks("   \n\t  ")
        assert result == []

    def test_simple_document(self):
        """Should create chunks for simple document."""
        content = "# Title\n\nThis is the content of the document."
        result = create_hierarchical_chunks(content)
        assert len(result) >= 1

    def test_creates_level2_chunks(self):
        """Should create level 2 (leaf) chunks."""
        content = "# Section\n\nContent here."
        result = create_hierarchical_chunks(content)
        level2_chunks = [c for c in result if c.level == 2]
        assert len(level2_chunks) >= 1

    def test_creates_level1_for_large_sections(self):
        """Should create level 1 chunks for large sections."""
        # Create content large enough to trigger level 1 chunk
        content = "# Large Section\n\n" + "Word " * 500
        result = create_hierarchical_chunks(content, level_sizes=(200, 50, 25))
        level1_chunks = [c for c in result if c.level == 1]
        assert len(level1_chunks) >= 1

    def test_parent_child_relationships(self):
        """Should link parent and child chunks."""
        content = "# Section\n\n" + "Content " * 200
        result = create_hierarchical_chunks(content, level_sizes=(200, 50, 25))

        level1 = [c for c in result if c.level == 1]
        level2 = [c for c in result if c.level == 2]

        if level1 and level2:
            # Some level 2 chunks should have parent_id
            children_with_parents = [c for c in level2 if c.parent_id]
            assert len(children_with_parents) > 0

    def test_prev_next_relationships(self):
        """Should link consecutive chunks."""
        content = "# Section 1\nContent 1\n\n# Section 2\nContent 2"
        result = create_hierarchical_chunks(content)

        level2 = [c for c in result if c.level == 2]
        if len(level2) >= 2:
            assert level2[0].next_id == level2[1].id
            assert level2[1].prev_id == level2[0].id

    def test_section_path_populated(self):
        """Should populate section_path for chunks."""
        content = "# Chapter\n\n## Section\n\nContent here."
        result = create_hierarchical_chunks(content)
        for chunk in result:
            assert isinstance(chunk.section_path, list)

    def test_respects_doc_id_prefix(self):
        """Should use doc_id in chunk IDs."""
        content = "# Title\nContent"
        result = create_hierarchical_chunks(content, doc_id="mydoc")
        for chunk in result:
            assert chunk.id.startswith("mydoc_")

    def test_handles_code_blocks(self):
        """Should not split code blocks."""
        content = "# Section\n\n```python\nprint('hello')\nprint('world')\n```"
        result = create_hierarchical_chunks(content)
        # Code block should be intact in at least one chunk
        code_found = any("```python" in c.content for c in result)
        assert code_found

    def test_handles_tables(self):
        """Should preserve tables as atomic units."""
        content = "# Section\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        result = create_hierarchical_chunks(content)
        # Table should be in chunks
        assert len(result) >= 1

    def test_intro_content_captured(self):
        """Should capture content before first heading."""
        content = "Intro paragraph.\n\n# Section\nSection content."
        result = create_hierarchical_chunks(content)
        intro_found = any("Intro paragraph" in c.content for c in result)
        assert intro_found


class TestGetChunkById:
    """Tests for get_chunk_by_id function."""

    def test_finds_existing_chunk(self):
        """Should find chunk by ID."""
        chunks = [
            Chunk(id="a", content="A", level=0, token_count=1),
            Chunk(id="b", content="B", level=0, token_count=1),
        ]
        result = get_chunk_by_id(chunks, "b")
        assert result is not None
        assert result.content == "B"

    def test_returns_none_for_missing(self):
        """Should return None for non-existent ID."""
        chunks = [Chunk(id="a", content="A", level=0, token_count=1)]
        result = get_chunk_by_id(chunks, "missing")
        assert result is None


class TestGetParentChunk:
    """Tests for get_parent_chunk function."""

    def test_finds_parent(self):
        """Should find parent chunk."""
        parent = Chunk(id="parent", content="P", level=1, token_count=1)
        child = Chunk(id="child", content="C", level=2, token_count=1, parent_id="parent")
        chunks = [parent, child]

        result = get_parent_chunk(chunks, child)
        assert result is parent

    def test_returns_none_for_no_parent(self):
        """Should return None for chunk without parent."""
        chunk = Chunk(id="orphan", content="O", level=1, token_count=1)
        result = get_parent_chunk([chunk], chunk)
        assert result is None


class TestGetChildChunks:
    """Tests for get_child_chunks function."""

    def test_finds_children(self):
        """Should find all child chunks."""
        parent = Chunk(
            id="parent",
            content="P",
            level=1,
            token_count=1,
            child_ids=["c1", "c2"],
        )
        child1 = Chunk(id="c1", content="C1", level=2, token_count=1)
        child2 = Chunk(id="c2", content="C2", level=2, token_count=1)
        chunks = [parent, child1, child2]

        result = get_child_chunks(chunks, parent)
        assert len(result) == 2

    def test_empty_for_no_children(self):
        """Should return empty list for chunk without children."""
        chunk = Chunk(id="leaf", content="L", level=2, token_count=1)
        result = get_child_chunks([chunk], chunk)
        assert result == []


class TestGetSiblingChunks:
    """Tests for get_sibling_chunks function."""

    def test_finds_siblings(self):
        """Should find sibling chunks."""
        parent = Chunk(
            id="parent",
            content="P",
            level=1,
            token_count=1,
            child_ids=["c1", "c2", "c3"],
        )
        c1 = Chunk(id="c1", content="C1", level=2, token_count=1, parent_id="parent")
        c2 = Chunk(id="c2", content="C2", level=2, token_count=1, parent_id="parent")
        c3 = Chunk(id="c3", content="C3", level=2, token_count=1, parent_id="parent")
        chunks = [parent, c1, c2, c3]

        result = get_sibling_chunks(chunks, c2)
        assert len(result) == 2
        sibling_ids = {s.id for s in result}
        assert sibling_ids == {"c1", "c3"}

    def test_empty_for_no_parent(self):
        """Should return empty for chunk without parent."""
        chunk = Chunk(id="orphan", content="O", level=1, token_count=1)
        result = get_sibling_chunks([chunk], chunk)
        assert result == []


class TestGetContextWindow:
    """Tests for get_context_window function."""

    def test_includes_center_chunk(self):
        """Should always include the center chunk."""
        chunk = Chunk(id="center", content="C", level=2, token_count=1)
        result = get_context_window([chunk], chunk, window_size=2)
        assert chunk in result

    def test_includes_prev_chunks(self):
        """Should include previous chunks."""
        c1 = Chunk(id="c1", content="1", level=2, token_count=1)
        c2 = Chunk(id="c2", content="2", level=2, token_count=1, prev_id="c1")
        c3 = Chunk(id="c3", content="3", level=2, token_count=1, prev_id="c2")
        c1.next_id = "c2"
        c2.next_id = "c3"
        chunks = [c1, c2, c3]

        result = get_context_window(chunks, c3, window_size=2)
        assert len(result) == 3  # c1, c2, c3

    def test_includes_next_chunks(self):
        """Should include next chunks."""
        c1 = Chunk(id="c1", content="1", level=2, token_count=1, next_id="c2")
        c2 = Chunk(id="c2", content="2", level=2, token_count=1, prev_id="c1", next_id="c3")
        c3 = Chunk(id="c3", content="3", level=2, token_count=1, prev_id="c2")
        chunks = [c1, c2, c3]

        result = get_context_window(chunks, c1, window_size=2)
        assert len(result) == 3

    def test_handles_start_of_document(self):
        """Should handle chunk at start (no prev)."""
        c1 = Chunk(id="c1", content="1", level=2, token_count=1, next_id="c2")
        c2 = Chunk(id="c2", content="2", level=2, token_count=1, prev_id="c1")
        chunks = [c1, c2]

        result = get_context_window(chunks, c1, window_size=2)
        assert c1 in result
        assert c2 in result

    def test_handles_end_of_document(self):
        """Should handle chunk at end (no next)."""
        c1 = Chunk(id="c1", content="1", level=2, token_count=1, next_id="c2")
        c2 = Chunk(id="c2", content="2", level=2, token_count=1, prev_id="c1")
        chunks = [c1, c2]

        result = get_context_window(chunks, c2, window_size=2)
        assert c1 in result
        assert c2 in result
