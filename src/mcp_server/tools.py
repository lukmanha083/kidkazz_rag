"""MCP tool implementations for KidKazz RAG."""

import logging
from typing import Any, Optional

from .config import ServerState
from .formatters import (
    extract_chunk_images,
    format_chunk,
    format_chunk_list,
    format_concept,
    format_concept_list,
    format_concept_with_context,
    format_document,
    format_document_stats,
    format_optional_chunk,
    format_search_result,
    format_search_results,
    format_summary,
    format_summary_hierarchy,
    format_summary_list,
    format_summary_search_results,
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
        has_table: Optional[bool] = None,
        has_code: Optional[bool] = None,
        has_math: Optional[bool] = None,
        has_image: Optional[bool] = None,
        header_level: Optional[int] = None,
        include_images: bool = False,
    ) -> list:
        """Search for document chunks semantically similar to the query.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)
            doc_id: Filter to specific document (optional)
            level: Filter by hierarchy level - 1 (section) or 2 (leaf) (optional)
            semantic_type: Filter by type - definition, example, procedure, theorem, narrative (optional)
            threshold: Minimum similarity score 0.0-1.0 (default: 0.0)
            tags: Filter by document tags - documents must have ALL specified tags (optional)
            has_table: Filter to chunks containing tables (optional)
            has_code: Filter to chunks containing code blocks (optional)
            has_math: Filter to chunks containing math equations (optional)
            has_image: Filter to chunks containing images (optional)
            header_level: Filter to chunks with specific header level 1-6 (optional)
            include_images: Return actual image data (PNG/JPG) alongside text results.
                When True, chunks with images will have ImageContent blocks interleaved
                with their text metadata. Use this when you need to see table/figure images.
                (default: False)

        Returns:
            List of matching chunks with content, metadata, and similarity scores.
            When include_images=True, Image objects are interleaved after each chunk
            that contains images.
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"search_semantic: query='{query_preview}', top_k={top_k}, tags={tags}")

        # Generate query embedding server-side
        query_embedding = state.embedder.embed_query(query)

        # Fetch more candidates when reranking
        reranker = state.reranker
        fetch_k = top_k * 4 if reranker else top_k

        # Search storage
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=fetch_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
            tags=tags,
            has_table=has_table,
            has_code=has_code,
            has_math=has_math,
            has_image=has_image,
            header_level=header_level,
        )

        # Rerank if enabled
        if reranker and results:
            results = reranker.rerank_chunks(query, results, top_n=top_k)

        logger.info(f"search_semantic: found {len(results)} results")

        if not include_images:
            return format_search_results(results)

        # Mixed response: text metadata interleaved with images
        mixed: list = []
        for ec, score in results:
            mixed.append(format_search_result(ec, score))
            mixed.extend(extract_chunk_images(ec.chunk.content))
        return mixed

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
    def get_chunk_images(chunk_id: str) -> list:
        """Get actual images (PNG/JPG) embedded in a chunk.

        Use this after finding a chunk with has_image=True to see the actual
        table or figure images. Returns ImageContent blocks that Claude can
        view directly.

        Args:
            chunk_id: Unique identifier of the chunk (e.g., "textbook_l2_42")

        Returns:
            List containing metadata dict and Image objects for each image
            found in the chunk. Returns error dict if chunk not found.
        """
        logger.info(f"get_chunk_images: chunk_id={chunk_id}")

        result = state.store.get_chunk(chunk_id)
        if result is None:
            return [{"error": f"Chunk '{chunk_id}' not found"}]

        images = extract_chunk_images(result.chunk.content)
        if not images:
            return [{"chunk_id": chunk_id, "images_found": 0}]

        response: list = [{"chunk_id": chunk_id, "images_found": len(images)}]
        response.extend(images)
        return response

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

    # =========================================================================
    # Summary Tools (Document Summarization Feature)
    # =========================================================================

    @mcp.tool()
    def get_document_summary(doc_id: str) -> Optional[dict[str, Any]]:
        """Get the document-level summary for a document.

        Returns the top-level summary that describes the entire document's
        scope, purpose, and main themes. This is the highest level in the
        summary hierarchy.

        Args:
            doc_id: Document identifier

        Returns:
            Document summary with content and key points, or null if not found
        """
        logger.info(f"get_document_summary: doc_id={doc_id}")

        summaries = state.store.get_document_summaries(doc_id, level="document")

        if not summaries:
            logger.info(f"get_document_summary: no summary found for '{doc_id}'")
            return None

        return format_summary(summaries[0])

    @mcp.tool()
    def get_chapter_summaries(doc_id: str) -> list[dict[str, Any]]:
        """Get all chapter-level summaries for a document.

        Returns summaries for each major section (L1 chunks) of the document.
        Chapter summaries aggregate section summaries and describe the main
        themes and learning objectives of each chapter.

        Args:
            doc_id: Document identifier

        Returns:
            List of chapter summaries with content and key points
        """
        logger.info(f"get_chapter_summaries: doc_id={doc_id}")

        summaries = state.store.get_document_summaries(doc_id, level="chapter")

        logger.info(f"get_chapter_summaries: found {len(summaries)} chapters")
        return format_summary_list(summaries)

    @mcp.tool()
    def get_section_summaries(doc_id: str) -> list[dict[str, Any]]:
        """Get all section-level summaries for a document.

        Returns summaries for each section (L2 chunks) of the document.
        Section summaries are the most granular level, describing specific
        topics and concepts within each section.

        Args:
            doc_id: Document identifier

        Returns:
            List of section summaries with content and key points
        """
        logger.info(f"get_section_summaries: doc_id={doc_id}")

        summaries = state.store.get_document_summaries(doc_id, level="section")

        logger.info(f"get_section_summaries: found {len(summaries)} sections")
        return format_summary_list(summaries)

    @mcp.tool()
    def search_summaries(
        query: str,
        top_k: int = 5,
        level: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search summaries semantically to find relevant document sections.

        Use this to quickly find which parts of documents are relevant to a
        topic. More efficient than searching full chunks when you need an
        overview rather than specific details.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)
            level: Filter by summary level: "document", "chapter", "section" (optional)
            doc_id: Filter to summaries from a specific document (optional)

        Returns:
            List of matching summaries with content, key points, and similarity scores
        """
        displayed_query = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"search_summaries: query='{displayed_query}', top_k={top_k}, level={level}, doc_id={doc_id}")

        # Embed query for vector search
        query_embedding = state.embedder.embed_query(query)

        # Fetch more candidates when reranking
        reranker = state.reranker
        fetch_k = top_k * 4 if reranker else top_k

        results = state.store.search_summaries(
            query_embedding=query_embedding,
            limit=fetch_k,
            level=level,
            document_id=doc_id,
        )

        # Rerank if enabled
        if reranker and results:
            results = reranker.rerank_summaries(query, results, top_n=top_k)

        logger.info(f"search_summaries: found {len(results)} results")
        return format_summary_search_results(results)

    @mcp.tool()
    def get_summary_hierarchy(doc_id: str) -> dict[str, Any]:
        """Get the complete summary hierarchy for a document as a tree.

        Returns a hierarchical structure showing how document, chapter, and
        section summaries relate to each other. Useful for understanding
        document structure and navigating to specific sections.

        Args:
            doc_id: Document identifier

        Returns:
            Hierarchical tree with document summary at root, chapters as children,
            and sections as grandchildren. Each node includes content and key_points.
        """
        logger.info(f"get_summary_hierarchy: doc_id={doc_id}")

        summaries = state.store.get_document_summaries(doc_id)

        if not summaries:
            logger.info(f"get_summary_hierarchy: no summaries found for '{doc_id}'")
            return {"document_id": doc_id, "hierarchy": None}

        logger.info(f"get_summary_hierarchy: building tree from {len(summaries)} summaries")
        return format_summary_hierarchy(summaries, doc_id)

    @mcp.tool()
    def list_summarized_documents() -> list[dict[str, Any]]:
        """List all documents that have summaries generated.

        Use this to discover which documents have been summarized and are
        available for summary-based search and navigation.

        Returns:
            List of documents with their summary counts
        """
        logger.info("list_summarized_documents: fetching list")

        docs = state.store.list_summarized_documents()

        logger.info(f"list_summarized_documents: found {len(docs)} documents")
        return docs

    # ========================================================================
    # SKILL TOOLS (procedural how-to knowledge)
    # ========================================================================

    @mcp.tool()
    def get_skill(skill_id: str) -> dict[str, Any]:
        """Get a procedural skill by its ID, including all ordered steps.

        A skill is a named, executable procedure extracted from a textbook
        (e.g., "Deploy app to Kubernetes"). Steps include action text, code
        snippets, and expected outputs.

        Args:
            skill_id: Slugified skill identifier (e.g., "deploy-app-to-kubernetes")

        Returns:
            Dict with skill metadata and an ordered 'steps' list. Returns
            {"error": ...} if not found.
        """
        logger.info(f"get_skill: skill_id={skill_id}")

        skill = state.store.get_skill(skill_id)
        if not skill:
            return {"error": f"Skill '{skill_id}' not found"}

        steps = state.store.get_skill_steps(skill_id)
        return _format_skill_with_steps(skill, steps)

    @mcp.tool()
    def get_skill_by_name(name: str) -> dict[str, Any]:
        """Get a skill by its display name.

        Useful when you know the name but not the slug.

        Args:
            name: Display name (e.g., "Deploy app to Kubernetes")

        Returns:
            Dict with skill metadata and ordered steps, or {"error": ...}.
        """
        logger.info(f"get_skill_by_name: name={name!r}")

        skill = state.store.get_skill_by_name(name)
        if not skill:
            return {"error": f"Skill with name {name!r} not found"}

        skill_id = skill.get("skill_id", "")
        steps = state.store.get_skill_steps(skill_id) if skill_id else []
        return _format_skill_with_steps(skill, steps)

    @mcp.tool()
    def list_skills(
        doc_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List available skills, optionally filtered by document or domain.

        Args:
            doc_id: Filter to skills extracted from a specific document
            domain: Filter by domain (programming, erp, stem, agriculture, veterinary, general)

        Returns:
            List of skill summary dicts (id, name, goal, difficulty, step_count, has_code).
        """
        logger.info(f"list_skills: doc_id={doc_id}, domain={domain}")

        skills = state.store.list_skills(doc_id=doc_id)
        if domain:
            skills = [s for s in skills if s.get("domain") == domain]

        return [
            {
                "skill_id": s.get("skill_id"),
                "name": s.get("name"),
                "goal": s.get("goal"),
                "domain": s.get("domain"),
                "difficulty": s.get("difficulty"),
                "step_count": s.get("step_count", 0),
                "has_code": bool(s.get("has_code")),
                "source_document_id": s.get("source_document_id"),
            }
            for s in skills
        ]

    @mcp.tool()
    def search_skills(
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over skills. Returns the most relevant how-to procedures.

        Use this when a user asks "how do I X?" — returns the skill(s)
        whose name/goal/success_criteria best match the query.

        Args:
            query: Natural language query
            top_k: Number of results (default: 5)
            doc_id: Optional filter by source document

        Returns:
            List of {skill, score, steps} dicts sorted by similarity.
        """
        logger.info(f"search_skills: query={query[:50]!r}, top_k={top_k}")

        query_embedding = state.embedder.embed_query(query)
        results = state.store.search_skills(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
        )

        output = []
        for skill, score in results:
            skill_id = skill.get("skill_id", "")
            steps = state.store.get_skill_steps(skill_id) if skill_id else []
            entry = _format_skill_with_steps(skill, steps)
            entry["score"] = score
            output.append(entry)
        return output

    @mcp.tool()
    def get_skill_prerequisites(skill_id: str) -> dict[str, Any]:
        """Get what must be learned before attempting this skill.

        Returns prerequisite concepts (knowledge needed) and prerequisite
        skills (other procedures that should be mastered first).

        Args:
            skill_id: Skill identifier

        Returns:
            Dict with 'prerequisite_concepts', 'prerequisite_skills', and
            'produces_concepts' lists.
        """
        logger.info(f"get_skill_prerequisites: skill_id={skill_id}")

        skill = state.store.get_skill(skill_id)
        if not skill:
            return {"error": f"Skill '{skill_id}' not found"}

        # HelixChunkStore returns node with 'id' (internal Helix ID);
        # MockChunkStore stores by skill_id directly. Fall back so both work.
        internal_id = skill.get("id") or skill_id

        return {
            "skill_id": skill_id,
            "skill_name": skill.get("name", ""),
            "prerequisite_concepts": state.store.get_skill_prerequisite_concepts(internal_id),
            "prerequisite_skills": state.store.get_skill_prerequisite_skills(internal_id),
            "produces_concepts": state.store.get_skill_produced_concepts(internal_id),
        }


def _format_skill_with_steps(skill: dict, steps: list[dict]) -> dict[str, Any]:
    """Shape a Skill node + its steps into a clean MCP response dict."""
    import json as _json

    common_failures_raw = skill.get("common_failures", "[]")
    try:
        common_failures = _json.loads(common_failures_raw) if isinstance(common_failures_raw, str) else common_failures_raw
    except (ValueError, TypeError):
        common_failures = []

    return {
        "skill_id": skill.get("skill_id"),
        "name": skill.get("name"),
        "goal": skill.get("goal"),
        "domain": skill.get("domain"),
        "difficulty": skill.get("difficulty"),
        "success_criteria": skill.get("success_criteria"),
        "common_failures": common_failures,
        "source_document_id": skill.get("source_document_id"),
        "anchor_header": skill.get("anchor_header"),
        "step_count": skill.get("step_count", len(steps)),
        "steps": [
            {
                "step_number": s.get("step_number"),
                "action": s.get("action"),
                "code_content": s.get("code_content"),
                "code_language": s.get("code_language"),
                "expected_output": s.get("expected_output"),
                "is_optional": bool(s.get("is_optional")),
            }
            for s in steps
        ],
    }
