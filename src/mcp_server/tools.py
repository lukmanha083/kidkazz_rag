"""MCP tool implementations for KidKazz RAG."""

import logging
from typing import Any, Optional

from .config import ServerState
from .formatters import (
    format_chunk,
    format_chunk_list,
    format_concept,
    format_concept_list,
    format_concept_with_context,
    format_document,
    format_document_stats,
    format_optional_chunk,
    format_search_results,
)

logger = logging.getLogger(__name__)


def register_tools(mcp: Any, state: ServerState) -> None:
    """Register all MCP tools with the server.

    Args:
        mcp: FastMCP server instance
        state: Server state with store and embedder
    """

    @mcp.tool()
    def search_semantic(
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search for document chunks semantically similar to the query.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)
            doc_id: Filter to specific document (optional)
            level: Filter by hierarchy level - 1 (section) or 2 (leaf) (optional)
            semantic_type: Filter by type - definition, example, procedure, theorem, narrative (optional)
            threshold: Minimum similarity score 0.0-1.0 (default: 0.0)
            tags: Filter by document tags - documents must have ALL specified tags (optional)

        Returns:
            List of matching chunks with content, metadata, and similarity scores
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"search_semantic: query='{query_preview}', top_k={top_k}, tags={tags}")

        # Generate query embedding server-side
        query_embedding = state.embedder.embed_text(query)

        # Search storage
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
            tags=tags,
        )

        logger.info(f"search_semantic: found {len(results)} results")
        return format_search_results(results)

    @mcp.tool()
    def search_keyword(
        keyword: str,
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for document chunks containing a specific keyword or phrase.

        Args:
            keyword: Text to search for
            doc_id: Filter to specific document (optional)
            case_sensitive: Whether search is case-sensitive (default: False)

        Returns:
            List of matching chunks with content and metadata
        """
        logger.info(f"search_keyword: keyword='{keyword}', doc_id={doc_id}")

        results = state.store.search_keyword(
            keyword=keyword,
            doc_id=doc_id,
            case_sensitive=case_sensitive,
        )

        logger.info(f"search_keyword: found {len(results)} results")
        return format_chunk_list(results)

    @mcp.tool()
    def get_chunk(chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific chunk by its ID.

        Args:
            chunk_id: Unique identifier of the chunk (e.g., "textbook_l2_5")

        Returns:
            Chunk with content, metadata, and embedding info, or null if not found
        """
        logger.info(f"get_chunk: chunk_id={chunk_id}")

        result = state.store.get_chunk(chunk_id)
        return format_optional_chunk(result)

    @mcp.tool()
    def get_context_window(
        chunk_id: str,
        window_size: int = 2,
    ) -> list[dict[str, Any]]:
        """Get a chunk with its surrounding context (previous and next chunks).

        Args:
            chunk_id: Center chunk ID
            window_size: Number of chunks before and after (default: 2)

        Returns:
            List of chunks in document order, centered on the specified chunk
        """
        logger.info(f"get_context_window: chunk_id={chunk_id}, window_size={window_size}")

        results = state.store.get_context_window(chunk_id, window_size=window_size)

        logger.info(f"get_context_window: found {len(results)} chunks")
        return format_chunk_list(results)

    @mcp.tool()
    def get_parent(chunk_id: str) -> Optional[dict[str, Any]]:
        """Get the parent chunk of a given chunk (Level 2 -> Level 1).

        Useful for expanding context to understand the broader section.

        Args:
            chunk_id: Child chunk ID

        Returns:
            Parent chunk with content and metadata, or null if no parent
        """
        logger.info(f"get_parent: chunk_id={chunk_id}")

        result = state.store.get_parent(chunk_id)
        return format_optional_chunk(result)

    @mcp.tool()
    def get_children(chunk_id: str) -> list[dict[str, Any]]:
        """Get all child chunks of a given chunk (Level 1 -> Level 2).

        Useful for drilling down into section details.

        Args:
            chunk_id: Parent chunk ID

        Returns:
            List of child chunks with content and metadata
        """
        logger.info(f"get_children: chunk_id={chunk_id}")

        results = state.store.get_children(chunk_id)

        logger.info(f"get_children: found {len(results)} children")
        return format_chunk_list(results)

    @mcp.tool()
    def get_siblings(chunk_id: str) -> list[dict[str, Any]]:
        """Get sibling chunks (same parent, same level).

        Useful for exploring related content in the same section.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of sibling chunks (excluding the specified chunk)
        """
        logger.info(f"get_siblings: chunk_id={chunk_id}")

        results = state.store.get_siblings(chunk_id)

        logger.info(f"get_siblings: found {len(results)} siblings")
        return format_chunk_list(results)

    @mcp.tool()
    def list_documents(
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List all documents in the knowledge base.

        Args:
            tags: Filter by document tags - documents must have ALL specified tags (optional)

        Returns:
            List of documents with doc_id, title, tags, chunk_count, and created_at
        """
        logger.info(f"list_documents: tags={tags}")

        docs = state.store.list_documents(tags=tags)

        logger.info(f"list_documents: found {len(docs)} documents")
        return [format_document(doc) for doc in docs]

    @mcp.tool()
    def get_document_chunks(
        doc_id: str,
        level: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get all chunks from a specific document.

        Args:
            doc_id: Document identifier
            level: Filter by hierarchy level - 1 or 2 (optional)

        Returns:
            List of chunks sorted by sequence position
        """
        logger.info(f"get_document_chunks: doc_id={doc_id}, level={level}")

        results = state.store.get_document_chunks(doc_id, level=level)

        logger.info(f"get_document_chunks: found {len(results)} chunks")
        return format_chunk_list(results)

    @mcp.tool()
    def get_document_stats(doc_id: str) -> Optional[dict[str, Any]]:
        """Get statistics and summary for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Statistics including total_chunks, level_counts, type_counts, total_tokens
        """
        logger.info(f"get_document_stats: doc_id={doc_id}")

        stats = state.store.get_document_stats(doc_id)

        if stats is None:
            logger.warning(f"get_document_stats: document '{doc_id}' not found")
            return None

        return format_document_stats(stats)

    # ========== CONCEPT TOOLS ==========

    @mcp.tool()
    def list_concepts(
        doc_id: Optional[str] = None,
        concept_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        List all extracted concepts from the knowledge base.

        Args:
            doc_id: Filter concepts from a specific document (optional)
            concept_type: Filter by type: term, method, principle, formula, account (optional)

        Returns:
            List of concepts with their definitions and metadata.
        """
        logger.info(f"list_concepts: doc_id={doc_id}, type={concept_type}")

        concepts = state.store.list_concepts(doc_id=doc_id)

        # Filter by type if specified
        if concept_type:
            concepts = [
                c for c in concepts
                if c.get("concept_type") == concept_type
            ]

        logger.info(f"list_concepts: found {len(concepts)} concepts")
        return format_concept_list(concepts)

    @mcp.tool()
    def get_concept(
        name: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a specific concept by name.

        Args:
            name: The concept name (e.g., "Cost of Goods Sold" or "COGS")

        Returns:
            Concept details including definition, type, aliases, and sources.
            Returns None if concept not found.
        """
        import json

        logger.info(f"get_concept: name='{name}'")

        # Try exact name match first
        concept = state.store.get_concept_by_name(name)

        # Try slug match
        if not concept:
            from src.chunker.concept_extractor import slugify
            concept = state.store.get_concept(slugify(name))

        # Try alias match (search through all concepts)
        if not concept:
            all_concepts = state.store.list_concepts()
            name_lower = name.lower()
            for c in all_concepts:
                aliases_raw = c.get("aliases", "[]")
                try:
                    aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
                except json.JSONDecodeError:
                    aliases = []
                if name_lower in [a.lower() for a in aliases]:
                    concept = c
                    break

        if not concept:
            logger.info(f"get_concept: concept '{name}' not found")
            return None

        return format_concept(concept)

    @mcp.tool()
    def get_concept_with_citations(
        name: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get a concept with its source citations (chunks where it's defined).

        This is the recommended tool for answering "What is X?" questions,
        as it provides both the definition and the source references.

        Args:
            name: The concept name

        Returns:
            Concept with citations showing where it's defined in source documents.
            Includes document ID and section path for each citation.
        """
        logger.info(f"get_concept_with_citations: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            from src.chunker.concept_extractor import slugify
            concept = state.store.get_concept(slugify(name))

        if not concept:
            logger.info(f"get_concept_with_citations: concept '{name}' not found")
            return None

        concept_id = concept.get("concept_id")
        definition_chunks = state.store.get_concept_definition_chunks(concept_id)
        related = state.store.get_related_concepts(concept_id)

        return format_concept_with_context(concept, definition_chunks, related)

    @mcp.tool()
    def get_related_concepts(
        name: str,
        include_reverse: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get concepts that are related to a given concept.

        Relationships include: uses, requires, calculated_from, component_of,
        recorded_in, supersedes.

        Args:
            name: The concept name to find relationships for
            include_reverse: Also include concepts that relate TO this one

        Returns:
            List of related concepts. Use this to understand how concepts
            connect across different documents (e.g., inventory concepts
            linking to accounting concepts).
        """
        logger.info(f"get_related_concepts: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            from src.chunker.concept_extractor import slugify
            concept = state.store.get_concept(slugify(name))

        if not concept:
            logger.info(f"get_related_concepts: concept '{name}' not found")
            return []

        concept_id = concept.get("concept_id")
        related = state.store.get_related_concepts(concept_id)

        # Include reverse relationships if requested
        if include_reverse:
            dependents = state.store.get_concept_dependents(concept_id)
            # Merge and deduplicate by concept_id
            seen_ids = {c.get("concept_id") for c in related}
            for dep in dependents:
                if dep.get("concept_id") not in seen_ids:
                    related.append(dep)
                    seen_ids.add(dep.get("concept_id"))

        logger.info(f"get_related_concepts: found {len(related)} related concepts")
        return format_concept_list(related)

    @mcp.tool()
    def search_concepts(
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for concepts by name, definition, or aliases.

        Args:
            query: Search text (matches concept names, definitions, aliases)
            top_k: Maximum number of results to return

        Returns:
            List of matching concepts sorted by relevance.
        """
        import json

        logger.info(f"search_concepts: query='{query}', top_k={top_k}")

        all_concepts = state.store.list_concepts()

        # Simple text search with scoring
        query_lower = query.lower()
        matches = []

        for c in all_concepts:
            score = 0
            name = c.get("name", "").lower()
            definition = c.get("definition", "").lower()
            aliases_raw = c.get("aliases", "[]")
            try:
                aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
            except json.JSONDecodeError:
                aliases = []
            aliases_lower = [a.lower() for a in aliases]

            # Exact match = highest score
            if query_lower == name:
                score += 100
            elif query_lower in name:
                score += 50

            # Alias match
            if any(query_lower == a for a in aliases_lower):
                score += 80
            elif any(query_lower in a for a in aliases_lower):
                score += 40

            # Definition match
            if query_lower in definition:
                score += 20

            if score > 0:
                matches.append((c, score))

        # Sort by score and limit
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:top_k]

        logger.info(f"search_concepts: found {len(matches)} matches")
        return [
            {**format_concept(c), "relevance_score": score}
            for c, score in matches
        ]

    @mcp.tool()
    def explain_concept_cross_document(
        name: str,
    ) -> dict[str, Any]:
        """
        Get comprehensive explanation of a concept across all documents.

        This tool provides:
        1. The concept definition with source citation
        2. Related concepts from all documents (cross-textbook links)
        3. Relevant chunks for additional context

        Use this for questions like "Explain X and how it relates to Y"
        or "What are all the aspects of X?"

        Args:
            name: The concept name

        Returns:
            Comprehensive concept information with cross-document context.
        """
        import json

        logger.info(f"explain_concept_cross_document: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            from src.chunker.concept_extractor import slugify
            concept = state.store.get_concept(slugify(name))

        if not concept:
            logger.info(f"explain_concept_cross_document: concept '{name}' not found")
            return {"error": f"Concept '{name}' not found"}

        concept_id = concept.get("concept_id")

        # Get citations
        definition_chunks = state.store.get_concept_definition_chunks(concept_id)

        # Get related concepts
        related = state.store.get_related_concepts(concept_id)

        # Get citations for related concepts (cross-document context)
        related_context = []
        for rel_concept in related[:5]:  # Limit to avoid too much data
            rel_id = rel_concept.get("concept_id")
            rel_chunks = state.store.get_concept_definition_chunks(rel_id)
            if rel_chunks:
                section_path_raw = rel_chunks[0].get("section_path", "[]")
                try:
                    section_path = json.loads(section_path_raw) if isinstance(section_path_raw, str) else section_path_raw
                except json.JSONDecodeError:
                    section_path = []
                related_context.append({
                    "concept": format_concept(rel_concept),
                    "citation": {
                        "document_id": rel_chunks[0].get("document_id", ""),
                        "section_path": section_path,
                    }
                })

        # Format citations
        citations = []
        for c in definition_chunks[:3]:
            section_path_raw = c.get("section_path", "[]")
            try:
                section_path = json.loads(section_path_raw) if isinstance(section_path_raw, str) else section_path_raw
            except json.JSONDecodeError:
                section_path = []
            content = c.get("content", "")
            citations.append({
                "document_id": c.get("document_id", ""),
                "section_path": section_path,
                "content_preview": content[:300] if content else "",
            })

        return {
            "concept": format_concept(concept),
            "definition": concept.get("definition", ""),
            "citations": citations,
            "related_concepts_with_sources": related_context,
        }

    @mcp.tool()
    def get_concept_graph_dot(
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Get concept graph in DOT format for visualization.

        The DOT format can be rendered using Graphviz or online tools
        like https://dreampuf.github.io/GraphvizOnline/

        Args:
            doc_id: Filter to concepts from a specific document (optional)

        Returns:
            DOT format string representing the concept graph.
        """
        from src.cli.graph_viz import concepts_to_dot

        logger.info(f"get_concept_graph_dot: doc_id={doc_id}")

        concepts = state.store.list_concepts(doc_id=doc_id)

        if not concepts:
            return 'digraph Empty { label="No concepts found"; }'

        # Get relations
        relations: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        for concept in concepts:
            concept_id = concept.get("concept_id")
            related = state.store.get_related_concepts(concept_id)

            for rel_concept in related:
                from_name = concept.get("name")
                to_name = rel_concept.get("name")
                key = (from_name, to_name)
                if key not in seen:
                    seen.add(key)
                    relations.append((from_name, to_name, "relates_to"))

        title = f"Concepts from {doc_id}" if doc_id else "Knowledge Graph"
        return concepts_to_dot(concepts, relations, title)
