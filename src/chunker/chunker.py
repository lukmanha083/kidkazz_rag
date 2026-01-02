"""Hierarchical chunking logic for vector + graph storage."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .parser import MarkdownSection, extract_special_blocks, parse_markdown_structure


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.
    More accurate than word count, less overhead than actual tokenization.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Rough estimate: 4 chars per token (works well for English)
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    """A chunk with content and relationships for vector + graph storage."""

    id: str
    content: str
    level: int  # 0=doc, 1=section, 2=leaf
    token_count: int

    # Graph relationships
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None

    # Source information
    source_section: Optional[str] = None
    section_path: list[str] = field(default_factory=list)

    # Additional metadata (extensible)
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Get word count of chunk content."""
        return len(self.content.split())

    def __hash__(self) -> int:
        return hash(self.id)


def generate_chunk_id(prefix: str = "chunk") -> str:
    """Generate a unique chunk ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def split_text_with_overlap(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    preserve_sentences: bool = True,
) -> list[str]:
    """
    Split text into chunks with overlap, respecting sentence boundaries.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks
        preserve_sentences: Try to split at sentence boundaries

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    tokens_estimate = estimate_tokens(text)
    if tokens_estimate <= max_tokens:
        return [text]

    chunks: list[str] = []

    if preserve_sentences:
        # Split into sentences
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_text = ""
                overlap_count = 0
                for s in reversed(current_chunk):
                    s_tokens = estimate_tokens(s)
                    if overlap_count + s_tokens <= overlap_tokens:
                        overlap_text = s + " " + overlap_text
                        overlap_count += s_tokens
                    else:
                        break

                current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
                current_tokens = overlap_count

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))
    else:
        # Simple character-based splitting
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token

        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap_chars

    return chunks


def create_hierarchical_chunks(
    content: str,
    level_sizes: tuple[int, int, int] = (2048, 512, 256),
    overlap_ratio: float = 0.15,
    doc_id: Optional[str] = None,
) -> list[Chunk]:
    """
    Create multi-level hierarchical chunks optimized for vector + graph.

    Creates three levels of chunks:
    - Level 0: Document-level reference (not stored as chunk, just metadata)
    - Level 1: Section-level chunks (for context synthesis)
    - Level 2: Leaf-level chunks (for vector search)

    Args:
        content: Markdown content to chunk
        level_sizes: Target token sizes for (level1, level2, unused)
        overlap_ratio: Overlap ratio between consecutive chunks
        doc_id: Optional document ID prefix

    Returns:
        List of chunks with hierarchical relationships
    """
    if not content.strip():
        return []

    doc_prefix = doc_id or "doc"
    level1_size, level2_size, _ = level_sizes
    overlap_tokens = int(level2_size * overlap_ratio)

    # Parse document structure
    root = parse_markdown_structure(content)
    special_blocks = extract_special_blocks(content)

    # Track all chunks
    all_chunks: list[Chunk] = []
    chunk_counter = {"level1": 0, "level2": 0}

    def is_in_special_block(text: str) -> bool:
        """Check if text contains a special block that shouldn't be split."""
        for block in special_blocks:
            if block.content in text:
                return True
        return False

    def create_level2_chunks(
        section_content: str,
        parent_chunk: Chunk,
        section_path: list[str],
    ) -> list[Chunk]:
        """Create level 2 (leaf) chunks from section content."""
        chunks: list[Chunk] = []

        # Check for special blocks
        for block in special_blocks:
            if block.content in section_content:
                # Create atomic chunk for special block
                block_tokens = estimate_tokens(block.content)
                if block_tokens > 0:
                    chunk_counter["level2"] += 1
                    chunk = Chunk(
                        id=f"{doc_prefix}_l2_{chunk_counter['level2']}",
                        content=block.content,
                        level=2,
                        token_count=block_tokens,
                        parent_id=parent_chunk.id,
                        section_path=section_path.copy(),
                        metadata={"block_type": block.block_type},
                    )
                    chunks.append(chunk)
                    parent_chunk.child_ids.append(chunk.id)
                # Remove block from content to process rest
                section_content = section_content.replace(block.content, "")

        # Split remaining content
        remaining_content = section_content.strip()
        if remaining_content:
            text_chunks = split_text_with_overlap(
                remaining_content,
                max_tokens=level2_size,
                overlap_tokens=overlap_tokens,
                preserve_sentences=True,
            )

            for text in text_chunks:
                if text.strip():
                    chunk_counter["level2"] += 1
                    chunk = Chunk(
                        id=f"{doc_prefix}_l2_{chunk_counter['level2']}",
                        content=text.strip(),
                        level=2,
                        token_count=estimate_tokens(text),
                        parent_id=parent_chunk.id,
                        section_path=section_path.copy(),
                    )
                    chunks.append(chunk)
                    parent_chunk.child_ids.append(chunk.id)

        return chunks

    def process_section(
        section: MarkdownSection,
        path: list[str],
    ) -> list[Chunk]:
        """Process a section and its children recursively."""
        chunks: list[Chunk] = []
        current_path = path + [section.heading] if section.heading else path

        # Get full content of this section (including children)
        full_content = section.full_content
        section_tokens = estimate_tokens(full_content)

        # Create level 1 chunk for this section if substantial
        if section_tokens >= level2_size and section.level > 0:
            chunk_counter["level1"] += 1
            level1_chunk = Chunk(
                id=f"{doc_prefix}_l1_{chunk_counter['level1']}",
                content=full_content[:level1_size * 4],  # Truncate if too long
                level=1,
                token_count=min(section_tokens, level1_size),
                source_section=section.heading,
                section_path=current_path.copy(),
            )
            chunks.append(level1_chunk)

            # Create level 2 chunks from section content
            level2_chunks = create_level2_chunks(
                section.content,  # Just this section's direct content
                level1_chunk,
                current_path,
            )
            chunks.extend(level2_chunks)

        elif section.content.strip():
            # Small section - create single level 2 chunk
            chunk_counter["level2"] += 1
            chunk = Chunk(
                id=f"{doc_prefix}_l2_{chunk_counter['level2']}",
                content=section.content.strip(),
                level=2,
                token_count=estimate_tokens(section.content),
                source_section=section.heading,
                section_path=current_path.copy(),
            )
            chunks.append(chunk)

        # Process children
        for child in section.children:
            child_chunks = process_section(child, current_path)
            chunks.extend(child_chunks)

        return chunks

    # Process root content (before first heading)
    if root.content.strip():
        chunk_counter["level2"] += 1
        intro_chunk = Chunk(
            id=f"{doc_prefix}_l2_{chunk_counter['level2']}",
            content=root.content.strip(),
            level=2,
            token_count=estimate_tokens(root.content),
            source_section="Introduction",
            section_path=["Introduction"],
        )
        all_chunks.append(intro_chunk)

    # Process all sections
    for section in root.children:
        section_chunks = process_section(section, [])
        all_chunks.extend(section_chunks)

    # Link prev/next relationships (sequential order)
    level2_chunks = [c for c in all_chunks if c.level == 2]
    for i, chunk in enumerate(level2_chunks):
        if i > 0:
            chunk.prev_id = level2_chunks[i - 1].id
        if i < len(level2_chunks) - 1:
            chunk.next_id = level2_chunks[i + 1].id

    # Link level 1 chunks prev/next
    level1_chunks = [c for c in all_chunks if c.level == 1]
    for i, chunk in enumerate(level1_chunks):
        if i > 0:
            chunk.prev_id = level1_chunks[i - 1].id
        if i < len(level1_chunks) - 1:
            chunk.next_id = level1_chunks[i + 1].id

    return all_chunks


def get_chunk_by_id(chunks: list[Chunk], chunk_id: str) -> Optional[Chunk]:
    """Find a chunk by its ID."""
    for chunk in chunks:
        if chunk.id == chunk_id:
            return chunk
    return None


def get_parent_chunk(chunks: list[Chunk], chunk: Chunk) -> Optional[Chunk]:
    """Get the parent chunk of a given chunk."""
    if chunk.parent_id:
        return get_chunk_by_id(chunks, chunk.parent_id)
    return None


def get_child_chunks(chunks: list[Chunk], chunk: Chunk) -> list[Chunk]:
    """Get all child chunks of a given chunk."""
    return [get_chunk_by_id(chunks, cid) for cid in chunk.child_ids if get_chunk_by_id(chunks, cid)]


def get_sibling_chunks(chunks: list[Chunk], chunk: Chunk) -> list[Chunk]:
    """Get sibling chunks (same parent) of a given chunk."""
    if not chunk.parent_id:
        return []

    parent = get_chunk_by_id(chunks, chunk.parent_id)
    if not parent:
        return []

    return [
        get_chunk_by_id(chunks, cid)
        for cid in parent.child_ids
        if cid != chunk.id and get_chunk_by_id(chunks, cid)
    ]


def get_context_window(
    chunks: list[Chunk],
    chunk: Chunk,
    window_size: int = 1,
) -> list[Chunk]:
    """
    Get a context window around a chunk (prev/next neighbors).

    Args:
        chunks: All chunks
        chunk: Center chunk
        window_size: Number of chunks before and after

    Returns:
        List of chunks in the context window (including center)
    """
    result: list[Chunk] = []

    # Get previous chunks
    current = chunk
    prev_chunks: list[Chunk] = []
    for _ in range(window_size):
        if current.prev_id:
            prev = get_chunk_by_id(chunks, current.prev_id)
            if prev:
                prev_chunks.insert(0, prev)
                current = prev
            else:
                break
        else:
            break

    result.extend(prev_chunks)
    result.append(chunk)

    # Get next chunks
    current = chunk
    for _ in range(window_size):
        if current.next_id:
            next_chunk = get_chunk_by_id(chunks, current.next_id)
            if next_chunk:
                result.append(next_chunk)
                current = next_chunk
            else:
                break
        else:
            break

    return result
