# Chunker Module - LlamaIndex-style hierarchical chunking
from .chunker import (
    Chunk,
    create_hierarchical_chunks,
    estimate_tokens,
    get_child_chunks,
    get_chunk_by_id,
    get_context_window,
    get_parent_chunk,
    get_sibling_chunks,
    split_text_with_overlap,
)
from .embedder import (
    EmbeddedChunk,
    MockEmbedder,
    OpenAIEmbedder,
    cosine_similarity,
    find_similar_chunks,
)
from .metadata import (
    ChunkMetadata,
    detect_content_features,
    enrich_all_chunks,
    enrich_chunk_metadata,
    extract_topic_tags,
    filter_chunks_by_topic,
    filter_chunks_by_type,
    infer_semantic_type,
)
from .parser import (
    MarkdownSection,
    SpecialBlock,
    build_section_path,
    extract_special_blocks,
    flatten_sections,
    get_heading_hierarchy,
    get_section_by_line,
    parse_markdown_structure,
)

__all__ = [
    # Chunker
    "Chunk",
    "create_hierarchical_chunks",
    "estimate_tokens",
    "get_child_chunks",
    "get_chunk_by_id",
    "get_context_window",
    "get_parent_chunk",
    "get_sibling_chunks",
    "split_text_with_overlap",
    # Embedder
    "EmbeddedChunk",
    "MockEmbedder",
    "OpenAIEmbedder",
    "cosine_similarity",
    "find_similar_chunks",
    # Metadata
    "ChunkMetadata",
    "detect_content_features",
    "enrich_all_chunks",
    "enrich_chunk_metadata",
    "extract_topic_tags",
    "filter_chunks_by_topic",
    "filter_chunks_by_type",
    "infer_semantic_type",
    # Parser
    "MarkdownSection",
    "SpecialBlock",
    "build_section_path",
    "extract_special_blocks",
    "flatten_sections",
    "get_heading_hierarchy",
    "get_section_by_line",
    "parse_markdown_structure",
]
