"""Metadata enrichment for vector search + graph traversal."""

import re
from dataclasses import dataclass, field
from typing import Optional

from .chunker import Chunk


# Patterns for semantic type detection
DEFINITION_PATTERNS = [
    r"\bis\s+defined\s+as\b",
    r"\brefers\s+to\b",
    r"\bmeans\b",
    r"\bis\s+a\b",
    r"\bare\s+called\b",
    r"^Definition:",
    r"^Def\.",
]

EXAMPLE_PATTERNS = [
    r"\bfor\s+example\b",
    r"\bfor\s+instance\b",
    r"\bsuch\s+as\b",
    r"\be\.g\.\b",
    r"^Example:",
    r"^Example\s+\d+",
]

PROCEDURE_PATTERNS = [
    r"\bstep\s+\d+\b",
    r"\bfirst,?\s",
    r"\bthen,?\s",
    r"\bfinally,?\s",
    r"\bnext,?\s",
    r"^\d+\.\s+",
    r"^-\s+",
    r"^Algorithm:",
    r"^Procedure:",
]

THEOREM_PATTERNS = [
    r"^Theorem\s*\d*[:.]",
    r"^Lemma\s*\d*[:.]",
    r"^Corollary\s*\d*[:.]",
    r"^Proof[:.]",
    r"\bprove\s+that\b",
    r"\bQ\.E\.D\.",
]


@dataclass
class ChunkMetadata:
    """Complete metadata for vector search + graph traversal."""

    # Identity
    chunk_id: str
    document_id: str

    # Vector search optimization
    semantic_type: str  # definition, example, procedure, theorem, narrative
    topic_tags: list[str] = field(default_factory=list)
    token_count: int = 0
    word_count: int = 0

    # Graph traversal optimization
    hierarchy_level: int = 2
    section_path: list[str] = field(default_factory=list)
    sequence_position: int = 0

    # Relationships (for Helix-DB graph)
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)
    sibling_ids: list[str] = field(default_factory=list)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None

    # Content analysis
    has_table: bool = False
    has_code: bool = False
    has_math: bool = False
    has_list: bool = False

    # Header/block metadata (from Reducto block mode)
    header_text: Optional[str] = None  # The actual header text if this is a header block
    header_level: Optional[int] = None  # 1-6 for header level (h1-h6)
    block_type: Optional[str] = None  # Block type from Reducto: "Header", "Text", "Table", etc.

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "semantic_type": self.semantic_type,
            "topic_tags": self.topic_tags,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "hierarchy_level": self.hierarchy_level,
            "section_path": self.section_path,
            "sequence_position": self.sequence_position,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "sibling_ids": self.sibling_ids,
            "prev_id": self.prev_id,
            "next_id": self.next_id,
            "has_table": self.has_table,
            "has_code": self.has_code,
            "has_math": self.has_math,
            "has_list": self.has_list,
            "header_text": self.header_text,
            "header_level": self.header_level,
            "block_type": self.block_type,
        }


def infer_semantic_type(content: str) -> str:
    """
    Classify chunk content as definition/example/procedure/theorem/narrative.

    Args:
        content: Chunk text content

    Returns:
        Semantic type string
    """
    # Check patterns in priority order (use IGNORECASE for all patterns)
    for pattern in THEOREM_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return "theorem"

    for pattern in DEFINITION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return "definition"

    for pattern in EXAMPLE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return "example"

    # Check for procedure (multiple indicators needed)
    procedure_matches = sum(
        1 for pattern in PROCEDURE_PATTERNS
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
    )
    if procedure_matches >= 2:
        return "procedure"

    # Default to narrative
    return "narrative"


def extract_topic_tags(content: str, max_tags: int = 5) -> list[str]:
    """
    Extract key topics/entities from chunk content.

    Uses simple heuristics:
    - Capitalized words (potential proper nouns/concepts)
    - Bold/italic text (emphasized concepts)
    - Words after "is a" or "refers to" (defined terms)

    Args:
        content: Chunk text content
        max_tags: Maximum number of tags to return

    Returns:
        List of topic tags
    """
    tags: set[str] = set()

    # Extract bold text (**text** or __text__)
    bold_pattern = r"\*\*([^*]+)\*\*|__([^_]+)__"
    for match in re.finditer(bold_pattern, content):
        tag = (match.group(1) or match.group(2)).strip().lower()
        if 2 <= len(tag) <= 50:
            tags.add(tag)

    # Extract italic text (*text* or _text_)
    italic_pattern = r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)"
    for match in re.finditer(italic_pattern, content):
        tag = (match.group(1) or match.group(2)).strip().lower()
        if 2 <= len(tag) <= 50:
            tags.add(tag)

    # Extract defined terms (after "is a", "refers to", etc.)
    definition_pattern = r"(?:is\s+a|refers\s+to|called|known\s+as)\s+([A-Za-z][a-z]+(?:\s+[a-z]+)?)"
    for match in re.finditer(definition_pattern, content, re.IGNORECASE):
        tag = match.group(1).strip().lower()
        if 2 <= len(tag) <= 50:
            tags.add(tag)

    # Extract capitalized multi-word phrases (likely concepts)
    concept_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    for match in re.finditer(concept_pattern, content):
        tag = match.group(1).strip().lower()
        if 2 <= len(tag) <= 50:
            tags.add(tag)

    # Sort by length (shorter = more specific) and limit
    sorted_tags = sorted(tags, key=len)[:max_tags]
    return sorted_tags


def detect_content_features(content: str) -> dict[str, bool]:
    """
    Detect special content features (table, code, math, list).

    Args:
        content: Chunk text content

    Returns:
        Dictionary of feature flags
    """
    return {
        "has_table": bool(re.search(r"\|.*\|.*\|", content)),
        "has_code": bool(re.search(r"```|`[^`]+`", content)),
        "has_math": bool(re.search(r"\$\$|\$[^$]+\$|\\\\?\[|\\\\?\(", content)),
        "has_list": bool(re.search(r"^[\s]*[-*+]\s|^[\s]*\d+\.\s", content, re.MULTILINE)),
    }


# Pattern for markdown headers (h1-h6)
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def extract_header_info(content: str) -> dict:
    """
    Extract header metadata from chunk content if it starts with a markdown header.

    Args:
        content: Chunk text content

    Returns:
        Dictionary with header_text, header_level, block_type if header found,
        empty dict otherwise.
    """
    if not content:
        return {}

    # Only check the first line
    first_line = content.split("\n", 1)[0]
    match = HEADER_PATTERN.match(first_line)

    if match:
        header_hashes = match.group(1)
        header_text = match.group(2).strip()
        return {
            "header_text": header_text,
            "header_level": len(header_hashes),
            "block_type": "Header",
        }

    return {}


def enrich_chunk_metadata(
    chunk: Chunk,
    document_id: str,
    sequence_position: int,
    sibling_ids: Optional[list[str]] = None,
) -> ChunkMetadata:
    """
    Create complete metadata for a chunk.

    Args:
        chunk: The chunk to enrich
        document_id: Document identifier
        sequence_position: Position in document order
        sibling_ids: IDs of sibling chunks (same parent)

    Returns:
        ChunkMetadata with all fields populated
    """
    content_features = detect_content_features(chunk.content)
    header_info = extract_header_info(chunk.content)

    return ChunkMetadata(
        chunk_id=chunk.id,
        document_id=document_id,
        semantic_type=infer_semantic_type(chunk.content),
        topic_tags=extract_topic_tags(chunk.content),
        token_count=chunk.token_count,
        word_count=chunk.word_count,
        hierarchy_level=chunk.level,
        section_path=chunk.section_path.copy(),
        sequence_position=sequence_position,
        parent_id=chunk.parent_id,
        child_ids=chunk.child_ids.copy(),
        sibling_ids=sibling_ids or [],
        prev_id=chunk.prev_id,
        next_id=chunk.next_id,
        has_table=content_features["has_table"],
        has_code=content_features["has_code"],
        has_math=content_features["has_math"],
        has_list=content_features["has_list"],
        header_text=header_info.get("header_text"),
        header_level=header_info.get("header_level"),
        block_type=header_info.get("block_type"),
    )


def enrich_all_chunks(
    chunks: list[Chunk],
    document_id: str,
) -> list[ChunkMetadata]:
    """
    Enrich all chunks with metadata.

    Args:
        chunks: List of chunks to enrich
        document_id: Document identifier

    Returns:
        List of ChunkMetadata for all chunks
    """
    # Build parent -> children mapping for sibling detection
    parent_children: dict[Optional[str], list[str]] = {}
    for chunk in chunks:
        parent_id = chunk.parent_id
        if parent_id not in parent_children:
            parent_children[parent_id] = []
        parent_children[parent_id].append(chunk.id)

    # Enrich each chunk
    metadata_list: list[ChunkMetadata] = []
    for i, chunk in enumerate(chunks):
        # Get sibling IDs (same parent, excluding self)
        siblings = parent_children.get(chunk.parent_id, [])
        sibling_ids = [sid for sid in siblings if sid != chunk.id]

        metadata = enrich_chunk_metadata(
            chunk=chunk,
            document_id=document_id,
            sequence_position=i,
            sibling_ids=sibling_ids,
        )
        metadata_list.append(metadata)

    return metadata_list


def filter_chunks_by_type(
    chunks: list[Chunk],
    metadata_list: list[ChunkMetadata],
    semantic_type: str,
) -> list[Chunk]:
    """
    Filter chunks by semantic type.

    Args:
        chunks: All chunks
        metadata_list: Corresponding metadata
        semantic_type: Type to filter for

    Returns:
        Filtered list of chunks
    """
    return [
        chunk
        for chunk, meta in zip(chunks, metadata_list, strict=True)
        if meta.semantic_type == semantic_type
    ]


def filter_chunks_by_topic(
    chunks: list[Chunk],
    metadata_list: list[ChunkMetadata],
    topic: str,
) -> list[Chunk]:
    """
    Filter chunks that contain a specific topic tag.

    Args:
        chunks: All chunks
        metadata_list: Corresponding metadata
        topic: Topic to search for

    Returns:
        Filtered list of chunks
    """
    topic_lower = topic.lower()
    return [
        chunk
        for chunk, meta in zip(chunks, metadata_list, strict=True)
        if topic_lower in [t.lower() for t in meta.topic_tags]
    ]
