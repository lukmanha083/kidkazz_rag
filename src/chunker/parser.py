"""Markdown structure extraction and parsing."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarkdownSection:
    """Represents a section in the markdown document."""

    heading: str
    level: int  # 1-6 for h1-h6, 0 for root
    content: str
    start_line: int
    end_line: int
    children: list["MarkdownSection"] = field(default_factory=list)

    @property
    def full_content(self) -> str:
        """Get content including all children."""
        parts = [self.content]
        for child in self.children:
            parts.append(child.full_content)
        return "\n".join(parts)

    @property
    def word_count(self) -> int:
        """Get word count of this section (excluding children)."""
        return len(self.content.split())


@dataclass
class SpecialBlock:
    """Represents a special block (table, code, math) that shouldn't be split."""

    block_type: str  # "table", "code", "math"
    content: str
    start_line: int
    end_line: int
    language: Optional[str] = None  # For code blocks


def parse_markdown_structure(content: str) -> MarkdownSection:
    """
    Parse markdown into hierarchical structure based on headings.

    Args:
        content: Raw markdown content

    Returns:
        Root MarkdownSection with nested children
    """
    lines = content.split("\n")
    root = MarkdownSection(
        heading="",
        level=0,
        content="",
        start_line=0,
        end_line=len(lines) - 1,
        children=[],
    )

    # Find all headings with their positions
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    headings: list[tuple[int, int, str]] = []  # (line_num, level, text)

    for i, line in enumerate(lines):
        match = heading_pattern.match(line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append((i, level, text))

    if not headings:
        # No headings, entire content is root
        root.content = content
        return root

    # Build hierarchy
    section_stack: list[MarkdownSection] = [root]

    for i, (line_num, level, text) in enumerate(headings):
        # Determine end line (start of next heading or end of document)
        if i + 1 < len(headings):
            end_line = headings[i + 1][0] - 1
        else:
            end_line = len(lines) - 1

        # Extract content between heading and next heading
        content_lines = lines[line_num + 1 : end_line + 1]
        section_content = "\n".join(content_lines).strip()

        section = MarkdownSection(
            heading=text,
            level=level,
            content=section_content,
            start_line=line_num,
            end_line=end_line,
            children=[],
        )

        # Find parent (closest section with lower level)
        while len(section_stack) > 1 and section_stack[-1].level >= level:
            section_stack.pop()

        section_stack[-1].children.append(section)
        section_stack.append(section)

    # Root content is everything before first heading
    first_heading_line = headings[0][0]
    root.content = "\n".join(lines[:first_heading_line]).strip()

    return root


def extract_special_blocks(content: str) -> list[SpecialBlock]:
    """
    Extract tables, code blocks, and math blocks as atomic units.

    Args:
        content: Markdown content

    Returns:
        List of special blocks that shouldn't be split during chunking
    """
    blocks: list[SpecialBlock] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks (``` or ~~~)
        code_match = re.match(r"^(`{3,}|~{3,})(\w*)?", line)
        if code_match:
            fence = code_match.group(1)
            language = code_match.group(2) or None
            start_line = i
            i += 1

            # Find closing fence
            while i < len(lines) and not lines[i].startswith(fence):
                i += 1

            end_line = i
            block_content = "\n".join(lines[start_line : end_line + 1])
            blocks.append(
                SpecialBlock(
                    block_type="code",
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                )
            )
            i += 1
            continue

        # Tables (lines with |)
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|?[\s\-:|]+\|", lines[i + 1]):
            start_line = i
            # Find table extent
            while i < len(lines) and "|" in lines[i]:
                i += 1
            end_line = i - 1
            block_content = "\n".join(lines[start_line : end_line + 1])
            blocks.append(
                SpecialBlock(
                    block_type="table",
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            continue

        # Math blocks ($$...$$)
        if line.strip() == "$$":
            start_line = i
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                i += 1
            end_line = i
            block_content = "\n".join(lines[start_line : end_line + 1])
            blocks.append(
                SpecialBlock(
                    block_type="math",
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            i += 1
            continue

        # LaTeX math blocks (\[...\])
        if line.strip() == "\\[":
            start_line = i
            i += 1
            while i < len(lines) and lines[i].strip() != "\\]":
                i += 1
            end_line = i
            block_content = "\n".join(lines[start_line : end_line + 1])
            blocks.append(
                SpecialBlock(
                    block_type="math",
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            i += 1
            continue

        i += 1

    return blocks


def get_heading_hierarchy(section: MarkdownSection) -> list[str]:
    """
    Get breadcrumb path for a section.

    Args:
        section: The section to get hierarchy for

    Returns:
        List of heading texts from root to this section
    """
    if section.level == 0:
        return []
    return [section.heading]


def build_section_path(
    root: MarkdownSection, target_section: MarkdownSection
) -> list[str]:
    """
    Build full path from root to target section.

    Args:
        root: Root section of document
        target_section: Section to find path to

    Returns:
        List of heading texts forming the path
    """

    def find_path(current: MarkdownSection, target: MarkdownSection) -> Optional[list[str]]:
        if current is target:
            return [current.heading] if current.heading else []

        for child in current.children:
            path = find_path(child, target)
            if path is not None:
                if current.heading:
                    return [current.heading, *path]
                return path

        return None

    path = find_path(root, target_section)
    return path if path else []


def flatten_sections(root: MarkdownSection) -> list[MarkdownSection]:
    """
    Flatten hierarchical sections into a list (depth-first order).

    Args:
        root: Root section

    Returns:
        List of all sections in document order
    """
    result: list[MarkdownSection] = []

    def traverse(section: MarkdownSection) -> None:
        if section.level > 0:  # Skip root
            result.append(section)
        for child in section.children:
            traverse(child)

    traverse(root)
    return result


def get_section_by_line(root: MarkdownSection, line_num: int) -> Optional[MarkdownSection]:
    """
    Find the section containing a specific line number.

    Args:
        root: Root section
        line_num: Line number to find

    Returns:
        The most specific section containing that line, or None
    """

    def find_section(section: MarkdownSection) -> Optional[MarkdownSection]:
        if section.start_line <= line_num <= section.end_line:
            # Check children first for more specific match
            for child in section.children:
                result = find_section(child)
                if result:
                    return result
            return section if section.level > 0 else None
        return None

    return find_section(root)
