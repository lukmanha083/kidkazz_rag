"""Tests for markdown structure parsing."""

import pytest

from src.chunker.parser import (
    MarkdownSection,
    SpecialBlock,
    build_section_path,
    extract_special_blocks,
    flatten_sections,
    get_heading_hierarchy,
    get_section_by_line,
    parse_markdown_structure,
)


class TestParseMarkdownStructure:
    """Tests for parse_markdown_structure function."""

    def test_empty_content(self):
        """Should handle empty content."""
        result = parse_markdown_structure("")
        assert result.level == 0
        assert result.content == ""
        assert result.children == []

    def test_content_without_headings(self):
        """Should return single root with all content."""
        content = "This is some text without headings.\nMore text here."
        result = parse_markdown_structure(content)
        assert result.level == 0
        assert result.content == content
        assert result.children == []

    def test_single_heading(self):
        """Should parse single heading."""
        content = "# Title\n\nSome content here."
        result = parse_markdown_structure(content)
        assert len(result.children) == 1
        assert result.children[0].heading == "Title"
        assert result.children[0].level == 1
        assert "Some content here" in result.children[0].content

    def test_multiple_headings_same_level(self):
        """Should parse multiple headings at same level."""
        content = "# First\nContent 1\n# Second\nContent 2\n# Third\nContent 3"
        result = parse_markdown_structure(content)
        assert len(result.children) == 3
        assert result.children[0].heading == "First"
        assert result.children[1].heading == "Second"
        assert result.children[2].heading == "Third"

    def test_nested_headings(self):
        """Should create nested structure for h2 under h1."""
        content = "# Chapter\n\n## Section\n\nContent here"
        result = parse_markdown_structure(content)
        assert len(result.children) == 1
        chapter = result.children[0]
        assert chapter.heading == "Chapter"
        assert len(chapter.children) == 1
        assert chapter.children[0].heading == "Section"

    def test_deeply_nested_headings(self):
        """Should handle h1 > h2 > h3 nesting."""
        content = "# Level 1\n## Level 2\n### Level 3\nDeep content"
        result = parse_markdown_structure(content)
        l1 = result.children[0]
        assert l1.heading == "Level 1"
        l2 = l1.children[0]
        assert l2.heading == "Level 2"
        l3 = l2.children[0]
        assert l3.heading == "Level 3"
        assert "Deep content" in l3.content

    def test_sibling_sections_with_children(self):
        """Should handle multiple h2 under h1."""
        content = "# Chapter\n## Sec 1\nText 1\n## Sec 2\nText 2"
        result = parse_markdown_structure(content)
        chapter = result.children[0]
        assert len(chapter.children) == 2
        assert chapter.children[0].heading == "Sec 1"
        assert chapter.children[1].heading == "Sec 2"

    def test_content_before_first_heading(self):
        """Should capture content before first heading in root."""
        content = "Intro text\n\n# Chapter\nChapter content"
        result = parse_markdown_structure(content)
        assert "Intro text" in result.content
        assert len(result.children) == 1

    def test_heading_with_special_chars(self):
        """Should handle headings with special characters."""
        content = "# What is `code`?\n\nAnswer here."
        result = parse_markdown_structure(content)
        assert result.children[0].heading == "What is `code`?"

    def test_full_content_property(self):
        """Should return full content including children."""
        content = "# Parent\nParent text\n## Child\nChild text"
        result = parse_markdown_structure(content)
        parent = result.children[0]
        full = parent.full_content
        assert "Parent text" in full
        assert "Child text" in full

    def test_word_count_property(self):
        """Should calculate word count correctly."""
        content = "# Title\nOne two three four five"
        result = parse_markdown_structure(content)
        assert result.children[0].word_count == 5


class TestExtractSpecialBlocks:
    """Tests for extract_special_blocks function."""

    def test_empty_content(self):
        """Should return empty list for empty content."""
        result = extract_special_blocks("")
        assert result == []

    def test_code_block_backticks(self):
        """Should extract code block with triple backticks."""
        content = "Text\n```python\nprint('hello')\n```\nMore text"
        result = extract_special_blocks(content)
        assert len(result) == 1
        assert result[0].block_type == "code"
        assert result[0].language == "python"
        assert "print('hello')" in result[0].content

    def test_code_block_tildes(self):
        """Should extract code block with tildes."""
        content = "~~~\ncode here\n~~~"
        result = extract_special_blocks(content)
        assert len(result) == 1
        assert result[0].block_type == "code"

    def test_table_detection(self):
        """Should extract markdown table."""
        content = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = extract_special_blocks(content)
        assert len(result) == 1
        assert result[0].block_type == "table"

    def test_math_block_double_dollar(self):
        """Should extract $$ math block."""
        content = "Text\n$$\nx^2 + y^2 = z^2\n$$\nMore text"
        result = extract_special_blocks(content)
        assert len(result) == 1
        assert result[0].block_type == "math"

    def test_math_block_latex_brackets(self):
        """Should extract \\[ \\] math block."""
        content = "Text\n\\[\nE = mc^2\n\\]\nMore"
        result = extract_special_blocks(content)
        assert len(result) == 1
        assert result[0].block_type == "math"

    def test_multiple_blocks(self):
        """Should extract multiple different blocks."""
        content = "```\ncode\n```\n\n| a |\n|---|\n| b |\n\n$$\nmath\n$$"
        result = extract_special_blocks(content)
        assert len(result) == 3
        types = {b.block_type for b in result}
        assert types == {"code", "table", "math"}

    def test_preserves_line_numbers(self):
        """Should track start and end line numbers."""
        content = "line 0\n```\ncode\n```\nline 4"
        result = extract_special_blocks(content)
        assert result[0].start_line == 1
        assert result[0].end_line == 3


class TestBuildSectionPath:
    """Tests for build_section_path function."""

    def test_path_to_child(self):
        """Should build path from root to child."""
        content = "# Chapter\n## Section"
        root = parse_markdown_structure(content)
        section = root.children[0].children[0]
        path = build_section_path(root, section)
        assert path == ["Chapter", "Section"]

    def test_path_to_root_child(self):
        """Should build path for direct child of root."""
        content = "# Chapter"
        root = parse_markdown_structure(content)
        chapter = root.children[0]
        path = build_section_path(root, chapter)
        assert path == ["Chapter"]

    def test_path_not_found(self):
        """Should return empty list if target not found."""
        content = "# Chapter"
        root = parse_markdown_structure(content)
        other = MarkdownSection("Other", 1, "", 0, 0)
        path = build_section_path(root, other)
        assert path == []


class TestFlattenSections:
    """Tests for flatten_sections function."""

    def test_empty_document(self):
        """Should return empty list for doc without headings."""
        root = parse_markdown_structure("Just text")
        result = flatten_sections(root)
        assert result == []

    def test_flattens_nested_structure(self):
        """Should flatten in document order."""
        content = "# A\n## A1\n## A2\n# B\n## B1"
        root = parse_markdown_structure(content)
        result = flatten_sections(root)
        headings = [s.heading for s in result]
        assert headings == ["A", "A1", "A2", "B", "B1"]


class TestGetSectionByLine:
    """Tests for get_section_by_line function."""

    def test_finds_section(self):
        """Should find section containing line."""
        content = "# Title\nLine 1\nLine 2"
        root = parse_markdown_structure(content)
        section = get_section_by_line(root, 1)
        assert section is not None
        assert section.heading == "Title"

    def test_returns_none_for_root_only(self):
        """Should return None for content before headings."""
        content = "Intro\n# Title"
        root = parse_markdown_structure(content)
        section = get_section_by_line(root, 0)
        assert section is None


class TestGetHeadingHierarchy:
    """Tests for get_heading_hierarchy function."""

    def test_returns_heading(self):
        """Should return section heading."""
        content = "# Title"
        root = parse_markdown_structure(content)
        result = get_heading_hierarchy(root.children[0])
        assert result == ["Title"]

    def test_root_returns_empty(self):
        """Should return empty for root."""
        root = parse_markdown_structure("# Title")
        result = get_heading_hierarchy(root)
        assert result == []
