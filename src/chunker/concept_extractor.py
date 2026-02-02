"""Concept extraction using Instructor + LLM.

This module provides LLM-powered extraction of concepts, definitions,
and relationships from textbook content.
"""

import logging
import re
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables (for ANTHROPIC_API_KEY)
load_dotenv()

# Optional import for instructor
try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None  # type: ignore
    INSTRUCTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class ConceptType(str, Enum):
    """Types of concepts that can be extracted from textbooks."""

    TERM = "term"  # Vocabulary: COGS, Depreciation, Liability
    METHOD = "method"  # Techniques: FIFO, LIFO, Weighted Average
    PRINCIPLE = "principle"  # Rules: Matching Principle, Revenue Recognition
    FORMULA = "formula"  # Calculations: COGS = Begin + Purchases - End
    ACCOUNT = "account"  # Ledger accounts: Inventory, Cost of Sales


class ExtractedConcept(BaseModel):
    """A concept extracted from textbook content."""

    name: str = Field(description="Canonical display name (e.g., 'Cost of Goods Sold')")
    concept_type: ConceptType = Field(description="Type of concept")
    definition: str = Field(
        description="1-2 sentence definition explaining the concept"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations (e.g., ['COGS', 'cost of sales'])",
    )


class ConceptRelation(BaseModel):
    """A relationship between two concepts."""

    from_concept: str = Field(description="Source concept name")
    to_concept: str = Field(description="Target concept name")
    relation_type: str = Field(
        description="Relationship type: uses, requires, calculated_from, "
        "component_of, recorded_in, supersedes"
    )


class ChunkExtraction(BaseModel):
    """All concepts and relationships extracted from a single chunk."""

    defined_concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description="Concepts that are DEFINED in this chunk (have definitions)",
    )
    mentioned_concepts: list[str] = Field(
        default_factory=list,
        description="Concept names that are mentioned but not defined here",
    )
    relationships: list[ConceptRelation] = Field(
        default_factory=list,
        description="Relationships between concepts found in this chunk",
    )


# ============================================================================
# Utility Functions
# ============================================================================


def slugify(name: str) -> str:
    """
    Convert concept name to URL-safe ID.

    Args:
        name: Concept name to slugify

    Returns:
        Lowercase, hyphenated slug

    Examples:
        >>> slugify("Cost of Goods Sold")
        'cost-of-goods-sold'
        >>> slugify("FIFO (First-In, First-Out)")
        'fifo-first-in-first-out'
    """
    if not name:
        return ""

    # Convert to lowercase and strip whitespace
    slug = name.lower().strip()

    # Remove special characters (keep alphanumeric, spaces, hyphens)
    slug = re.sub(r"[^\w\s-]", "", slug)

    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)

    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)

    # Strip leading/trailing hyphens
    slug = slug.strip("-")

    return slug


# ============================================================================
# ConceptExtractor Class
# ============================================================================


class ConceptExtractor:
    """Extract concepts from chunks using Instructor + LLM."""

    def __init__(
        self,
        provider: str = "anthropic/claude-3-5-haiku-20241022",
        max_retries: int = 2,
    ):
        """
        Initialize extractor.

        Args:
            provider: Instructor provider string (e.g., "anthropic/claude-3-5-haiku-20241022")
            max_retries: Number of retries on validation failure

        Raises:
            ImportError: If instructor is not installed
        """
        if not INSTRUCTOR_AVAILABLE or instructor is None:
            raise ImportError(
                "instructor is not installed. "
                "Install with: pip install 'kidkazz[concepts]'"
            )

        self.client = instructor.from_provider(provider)
        self.max_retries = max_retries

    def extract_from_chunk(
        self,
        content: str,
        section_path: list[str],
        document_title: str,
        existing_concepts: Optional[list[str]] = None,
        header_text: Optional[str] = None,
        header_level: Optional[int] = None,
        semantic_type: Optional[str] = None,
    ) -> ChunkExtraction:
        """
        Extract concepts and relationships from a single chunk.

        Args:
            content: Chunk text content
            section_path: Breadcrumb path ["Chapter 5", "Inventory", "COGS Methods"]
            document_title: Source document name for context
            existing_concepts: Known concepts to reference (avoid duplicates)
            header_text: Header text from chunk metadata (if any)
            header_level: Header level 1-6 (h1-h6) from chunk metadata
            semantic_type: Semantic type from metadata (definition, example, etc.)

        Returns:
            ChunkExtraction with defined concepts, mentioned concepts, relationships
        """
        # Build context with header information
        context_parts = [f"Document: {document_title}"]
        context_parts.append(f"Section: {' > '.join(section_path)}")

        if header_text:
            header_indicator = f"h{header_level}" if header_level else "header"
            context_parts.append(f"Current Header ({header_indicator}): {header_text}")

        if semantic_type:
            context_parts.append(f"Content Type: {semantic_type}")

        context = "\n".join(context_parts)
        existing = existing_concepts or []

        system_prompt = (
            "You are extracting concepts from a textbook. "
            "Identify terms, methods, principles, formulas, and accounts. "
            "For each concept that is DEFINED (explained) in this text, extract:\n"
            "- The canonical name\n"
            "- The type (term, method, principle, formula, account)\n"
            "- A 1-2 sentence definition\n"
            "- Any aliases or abbreviations\n\n"
            "Also identify relationships between concepts (uses, requires, "
            "calculated_from, component_of, recorded_in, supersedes).\n\n"
            f"Already known concepts (reference but don't redefine): {existing}"
        )

        try:
            return self.client.create(
                response_model=ChunkExtraction,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nContent:\n{content}"},
                ],
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.warning(f"Extraction failed for chunk: {e}")
            return ChunkExtraction()

    def extract_from_chunks(
        self,
        chunks: list[dict],  # [{content, section_path, ...}]
        document_title: str,
        metadata_list: Optional[list] = None,
    ) -> tuple[list[ExtractedConcept], list[ConceptRelation]]:
        """
        Extract all concepts from a document's chunks.

        Args:
            chunks: List of chunk dicts with 'content' and 'section_path' keys
            document_title: Document title for context
            metadata_list: Optional list of ChunkMetadata objects with header info

        Returns:
            Tuple of (deduplicated concepts, all relationships)
        """
        if not chunks:
            return [], []

        all_concepts: dict[str, ExtractedConcept] = {}  # lowercase name -> concept
        all_relations: list[ConceptRelation] = []
        known_names: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting concepts from chunk {i + 1}/{len(chunks)}")

            # Get metadata for this chunk if available
            header_text = None
            header_level = None
            semantic_type = None

            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                header_text = getattr(meta, "header_text", None)
                header_level = getattr(meta, "header_level", None)
                semantic_type = getattr(meta, "semantic_type", None)

            extraction = self.extract_from_chunk(
                content=chunk["content"],
                section_path=chunk.get("section_path", []),
                document_title=document_title,
                existing_concepts=known_names,
                header_text=header_text,
                header_level=header_level,
                semantic_type=semantic_type,
            )

            # Deduplicate concepts by lowercase name
            for concept in extraction.defined_concepts:
                name_lower = concept.name.lower()
                if name_lower not in all_concepts:
                    all_concepts[name_lower] = concept
                    known_names.append(concept.name)
                else:
                    # Merge aliases from duplicate
                    existing = all_concepts[name_lower]
                    for alias in concept.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

            # Collect all relationships
            all_relations.extend(extraction.relationships)

        return list(all_concepts.values()), all_relations

    def extract_from_chunks_with_mapping(
        self,
        chunks: list[dict],  # [{content, section_path, chunk_id, ...}]
        document_title: str,
        metadata_list: Optional[list] = None,
    ) -> tuple[list[ExtractedConcept], list[ConceptRelation], dict[int, ChunkExtraction]]:
        """
        Extract concepts from chunks while preserving per-chunk mapping.

        Args:
            chunks: List of chunk dicts with 'content', 'section_path', 'chunk_id' keys
            document_title: Document title for context
            metadata_list: Optional list of ChunkMetadata objects with header info

        Returns:
            Tuple of:
            - deduplicated concepts (list)
            - all relationships (list)
            - chunk_index -> ChunkExtraction mapping (for creating DefinesConcept/MentionsConcept edges)
        """
        if not chunks:
            return [], [], {}

        all_concepts: dict[str, ExtractedConcept] = {}  # lowercase name -> concept
        all_relations: list[ConceptRelation] = []
        chunk_extractions: dict[int, ChunkExtraction] = {}  # chunk_index -> extraction
        known_names: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting concepts from chunk {i + 1}/{len(chunks)}")

            # Get metadata for this chunk if available
            header_text = None
            header_level = None
            semantic_type = None

            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                header_text = getattr(meta, "header_text", None)
                header_level = getattr(meta, "header_level", None)
                semantic_type = getattr(meta, "semantic_type", None)

            extraction = self.extract_from_chunk(
                content=chunk["content"],
                section_path=chunk.get("section_path", []),
                document_title=document_title,
                existing_concepts=known_names,
                header_text=header_text,
                header_level=header_level,
                semantic_type=semantic_type,
            )

            # Store per-chunk extraction for linking
            chunk_extractions[i] = extraction

            # Deduplicate concepts by lowercase name
            for concept in extraction.defined_concepts:
                name_lower = concept.name.lower()
                if name_lower not in all_concepts:
                    all_concepts[name_lower] = concept
                    known_names.append(concept.name)
                else:
                    # Merge aliases from duplicate
                    existing = all_concepts[name_lower]
                    for alias in concept.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

            # Collect all relationships
            all_relations.extend(extraction.relationships)

        return list(all_concepts.values()), all_relations, chunk_extractions


# ============================================================================
# Semantic Type Filtering
# ============================================================================


def filter_chunks_by_semantic_type(
    chunks: list[dict],
    metadata_list: list,
    semantic_types: Optional[list[str]] = None,
) -> tuple[list[dict], list]:
    """
    Filter chunks by semantic type for prioritized extraction.

    Args:
        chunks: List of chunk dicts
        metadata_list: List of ChunkMetadata objects
        semantic_types: List of semantic types to include (e.g., ["definition", "theorem"])
                       If None or empty, returns all chunks.

    Returns:
        Tuple of (filtered_chunks, filtered_metadata)
    """
    if not semantic_types:
        return chunks, metadata_list

    filtered_chunks = []
    filtered_metadata = []

    # Use strict=False to allow graceful handling when lists have different lengths
    for chunk, meta in zip(chunks, metadata_list, strict=False):
        if getattr(meta, "semantic_type", None) in semantic_types:
            filtered_chunks.append(chunk)
            filtered_metadata.append(meta)

    return filtered_chunks, filtered_metadata


# ============================================================================
# Header Hierarchy Relationship Inference
# ============================================================================


def infer_relationships_from_headers(
    concepts: list[ExtractedConcept],
    metadata_list: list,
    concept_chunk_map: dict[str, str],
) -> list[ConceptRelation]:
    """
    Infer parent-child relationships between concepts based on header hierarchy.

    When a concept is extracted from an h2 section that follows an h1 section
    containing another concept, we infer a parent-child relationship.

    Args:
        concepts: List of extracted concepts
        metadata_list: List of ChunkMetadata objects with header info
        concept_chunk_map: Mapping of concept name -> chunk_id

    Returns:
        List of inferred ConceptRelation objects
    """
    if not concepts or not metadata_list or not concept_chunk_map:
        return []

    # Build chunk_id -> metadata mapping
    chunk_metadata: dict[str, tuple] = {}
    for meta in metadata_list:
        chunk_id = getattr(meta, "chunk_id", None)
        if chunk_id:
            header_level = getattr(meta, "header_level", None)
            header_text = getattr(meta, "header_text", None)
            chunk_metadata[chunk_id] = (header_level, header_text)

    # Build concept -> (chunk_id, header_level) mapping
    concept_levels: dict[str, tuple[str, Optional[int]]] = {}
    for concept in concepts:
        chunk_id = concept_chunk_map.get(concept.name)
        if chunk_id and chunk_id in chunk_metadata:
            header_level, _ = chunk_metadata[chunk_id]
            concept_levels[concept.name] = (chunk_id, header_level)

    # Find parent relationships based on header hierarchy
    relations = []

    # Track header stack for hierarchy
    header_stack: list[tuple[str, int]] = []  # [(concept_name, header_level)]

    # Sort concepts by their chunk position (assuming chunk_ids are ordered)
    sorted_concepts = sorted(
        [(c.name, concept_levels.get(c.name, (None, None))) for c in concepts],
        key=lambda x: x[1][0] if x[1][0] else "",
    )

    for concept_name, (_chunk_id, level) in sorted_concepts:
        if level is None:
            continue

        # Pop concepts with same or higher level (lower number = higher in hierarchy)
        while header_stack and header_stack[-1][1] >= level:
            header_stack.pop()

        # If there's a parent concept in the stack, create relationship
        if header_stack:
            parent_name = header_stack[-1][0]
            relations.append(
                ConceptRelation(
                    from_concept=parent_name,
                    to_concept=concept_name,
                    relation_type="parent_of",
                )
            )

        # Push current concept to stack
        header_stack.append((concept_name, level))

    return relations
