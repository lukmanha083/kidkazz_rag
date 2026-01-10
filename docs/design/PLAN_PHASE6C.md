# Phase 6C: Concept Extraction - MCP Tools

**Status: ✅ Complete**

## Overview

Add MCP tools for concept queries, enabling Claude Code to search concepts, get definitions with citations, and traverse concept relationships.

## Prerequisites

- Phase 6A complete (core infrastructure)
- Phase 6B complete (CLI integration)
- MCP server operational (Phase 4)

---

## Implementation Details

### 1. Concept Formatters (`src/mcp_server/formatters.py`)

Add formatting functions for concept data:

```python
def format_concept(concept: dict) -> dict:
    """Format a single concept for MCP output."""
    import json

    return {
        "concept_id": concept.get("concept_id", ""),
        "name": concept.get("name", ""),
        "definition": concept.get("definition", ""),
        "concept_type": concept.get("concept_type", ""),
        "aliases": json.loads(concept.get("aliases", "[]")),
        "source_documents": json.loads(concept.get("source_documents", "[]")),
    }


def format_concept_list(concepts: list[dict]) -> list[dict]:
    """Format a list of concepts."""
    return [format_concept(c) for c in concepts]


def format_concept_with_context(
    concept: dict,
    definition_chunks: list[dict],
    related_concepts: list[dict],
) -> dict:
    """Format concept with full context for rich responses."""
    import json

    formatted_chunks = []
    for chunk in definition_chunks:
        section_path = json.loads(chunk.get("section_path", "[]"))
        formatted_chunks.append({
            "document_id": chunk.get("document_id", ""),
            "section_path": section_path,
            "content_preview": chunk.get("content", "")[:200] + "...",
            "chunk_id": chunk.get("chunk_id", ""),
        })

    return {
        "concept": format_concept(concept),
        "citations": formatted_chunks,
        "related_concepts": format_concept_list(related_concepts),
    }
```

### 2. Concept Tools (`src/mcp_server/tools.py`)

Add tools inside `register_tools()`:

```python
def register_tools(mcp: FastMCP, state: "ServerState") -> None:
    """Register all MCP tools."""

    # ... existing tools ...

    # ========== CONCEPT TOOLS ==========

    @mcp.tool()
    def list_concepts(
        doc_id: Optional[str] = None,
        concept_type: Optional[str] = None,
    ) -> list[dict]:
        """
        List all extracted concepts from the knowledge base.

        Args:
            doc_id: Filter concepts from a specific document (optional)
            concept_type: Filter by type: term, method, principle, formula, account (optional)

        Returns:
            List of concepts with their definitions and metadata.
        """
        import json

        logger.info(f"list_concepts: doc_id={doc_id}, type={concept_type}")

        concepts = state.store.list_concepts(doc_id=doc_id)

        # Filter by type if specified
        if concept_type:
            concepts = [
                c for c in concepts
                if c.get("concept_type") == concept_type
            ]

        return format_concept_list(concepts)

    @mcp.tool()
    def get_concept(
        name: str,
    ) -> Optional[dict]:
        """
        Get detailed information about a specific concept by name.

        Args:
            name: The concept name (e.g., "Cost of Goods Sold" or "COGS")

        Returns:
            Concept details including definition, type, aliases, and sources.
            Returns None if concept not found.
        """
        from src.chunker.concept_extractor import slugify

        logger.info(f"get_concept: name='{name}'")

        # Try exact name match first
        concept = state.store.get_concept_by_name(name)

        # Try slug match
        if not concept:
            concept = state.store.get_concept(slugify(name))

        # Try alias match (search through all concepts)
        if not concept:
            import json
            all_concepts = state.store.list_concepts()
            name_lower = name.lower()
            for c in all_concepts:
                aliases = json.loads(c.get("aliases", "[]"))
                if name_lower in [a.lower() for a in aliases]:
                    concept = c
                    break

        if not concept:
            return None

        return format_concept(concept)

    @mcp.tool()
    def get_concept_with_citations(
        name: str,
    ) -> Optional[dict]:
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
        from src.chunker.concept_extractor import slugify

        logger.info(f"get_concept_with_citations: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            concept = state.store.get_concept(slugify(name))

        if not concept:
            return None

        concept_id = concept.get("concept_id")
        definition_chunks = state.store.get_concept_definition_chunks(concept_id)
        related = state.store.get_related_concepts(concept_id)

        return format_concept_with_context(concept, definition_chunks, related)

    @mcp.tool()
    def get_related_concepts(
        name: str,
        include_reverse: bool = False,
    ) -> list[dict]:
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
        from src.chunker.concept_extractor import slugify

        logger.info(f"get_related_concepts: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            concept = state.store.get_concept(slugify(name))

        if not concept:
            return []

        concept_id = concept.get("concept_id")
        related = state.store.get_related_concepts(concept_id)

        return format_concept_list(related)

    @mcp.tool()
    def search_concepts(
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
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
            aliases = json.loads(c.get("aliases", "[]"))
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

        return [
            {**format_concept(c), "relevance_score": score}
            for c, score in matches
        ]

    @mcp.tool()
    def get_concept_chunks(
        name: str,
        include_mentions: bool = True,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get all chunks related to a concept.

        Args:
            name: The concept name
            include_mentions: Include chunks that mention the concept (not just define it)
            limit: Maximum number of chunks to return

        Returns:
            List of chunks where the concept is defined or mentioned,
            useful for getting full context about a concept.
        """
        from src.chunker.concept_extractor import slugify

        logger.info(f"get_concept_chunks: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            concept = state.store.get_concept(slugify(name))

        if not concept:
            return []

        concept_id = concept.get("concept_id")
        chunks = state.store.get_concept_definition_chunks(concept_id)

        # Format chunks
        return format_chunk_list([
            # Convert to EmbeddedChunk-like format for existing formatter
            {"chunk": chunk} for chunk in chunks[:limit]
        ])

    @mcp.tool()
    def explain_concept_cross_document(
        name: str,
    ) -> dict:
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
        from src.chunker.concept_extractor import slugify
        import json

        logger.info(f"explain_concept_cross_document: name='{name}'")

        concept = state.store.get_concept_by_name(name)
        if not concept:
            concept = state.store.get_concept(slugify(name))

        if not concept:
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
                related_context.append({
                    "concept": format_concept(rel_concept),
                    "citation": {
                        "document_id": rel_chunks[0].get("document_id", ""),
                        "section_path": json.loads(rel_chunks[0].get("section_path", "[]")),
                    }
                })

        return {
            "concept": format_concept(concept),
            "definition": concept.get("definition", ""),
            "citations": [
                {
                    "document_id": c.get("document_id", ""),
                    "section_path": json.loads(c.get("section_path", "[]")),
                    "content_preview": c.get("content", "")[:300],
                }
                for c in definition_chunks[:3]
            ],
            "related_concepts_with_sources": related_context,
        }
```

### 3. Update Resources (`src/mcp_server/resources.py`)

Add concept tools to SCHEMA_INFO:

```python
SCHEMA_INFO = {
    # ... existing schema ...

    "concept_tools": {
        "description": "Tools for querying extracted concepts and their relationships",
        "tools": [
            {
                "name": "list_concepts",
                "description": "List all concepts, optionally filtered by document or type",
                "parameters": ["doc_id (optional)", "concept_type (optional)"],
            },
            {
                "name": "get_concept",
                "description": "Get detailed info about a specific concept by name",
                "parameters": ["name"],
            },
            {
                "name": "get_concept_with_citations",
                "description": "Get concept with source citations - use for 'What is X?' questions",
                "parameters": ["name"],
            },
            {
                "name": "get_related_concepts",
                "description": "Get concepts related to a given concept (cross-document)",
                "parameters": ["name", "include_reverse (optional)"],
            },
            {
                "name": "search_concepts",
                "description": "Search concepts by name, definition, or aliases",
                "parameters": ["query", "top_k (optional)"],
            },
            {
                "name": "get_concept_chunks",
                "description": "Get chunks where a concept is defined or mentioned",
                "parameters": ["name", "include_mentions (optional)", "limit (optional)"],
            },
            {
                "name": "explain_concept_cross_document",
                "description": "Get comprehensive cross-document explanation of a concept",
                "parameters": ["name"],
            },
        ],
    },
}
```

---

## Tool Usage Examples

### Example 1: "What is COGS?"

Claude calls `get_concept_with_citations("Cost of Goods Sold")`:

```json
{
  "concept": {
    "concept_id": "cost-of-goods-sold",
    "name": "Cost of Goods Sold",
    "definition": "The direct costs attributable to goods sold during a period, calculated as Beginning Inventory + Purchases - Ending Inventory.",
    "concept_type": "formula",
    "aliases": ["COGS", "cost of sales"]
  },
  "citations": [
    {
      "document_id": "inventory_textbook",
      "section_path": ["Chapter 5", "Inventory Valuation", "COGS Methods"],
      "content_preview": "Cost of Goods Sold (COGS) represents the direct costs..."
    }
  ],
  "related_concepts": [
    {"name": "FIFO", "concept_type": "method"},
    {"name": "Journal Entry", "concept_type": "term"}
  ]
}
```

### Example 2: "How does COGS relate to accounting?"

Claude calls `explain_concept_cross_document("COGS")`:

```json
{
  "concept": {...},
  "definition": "The direct costs attributable to goods sold...",
  "citations": [
    {"document_id": "inventory_textbook", "section_path": ["Chapter 5", "COGS"]}
  ],
  "related_concepts_with_sources": [
    {
      "concept": {"name": "Journal Entry", "concept_type": "term"},
      "citation": {"document_id": "accounting_textbook", "section_path": ["Chapter 3", "Recording"]}
    },
    {
      "concept": {"name": "Income Statement", "concept_type": "term"},
      "citation": {"document_id": "accounting_textbook", "section_path": ["Chapter 2", "Financial Statements"]}
    }
  ]
}
```

---

## Test Plan

### test_mcp_concept_tools.py (~20 tests)

- `list_concepts` returns all concepts
- `list_concepts` with doc_id filter
- `list_concepts` with concept_type filter
- `get_concept` for existing concept
- `get_concept` for non-existent concept
- `get_concept` by alias
- `get_concept_with_citations` returns citations
- `get_related_concepts` returns related
- `get_related_concepts` for concept with no relations
- `search_concepts` exact match
- `search_concepts` partial match
- `search_concepts` alias match
- `search_concepts` definition match
- `get_concept_chunks` returns chunks
- `explain_concept_cross_document` full context

---

## Verification

1. **Start MCP server:**
   ```bash
   python -m src.mcp_server
   ```

2. **Test via Claude Code:**
   - Ask "What is COGS?"
   - Ask "How does inventory valuation relate to accounting?"
   - Ask "List all concepts from the inventory textbook"

3. **Run tests:**
   ```bash
   PYTHONPATH=. pytest tests/test_mcp_concept_tools.py -v -m mcp
   ```

---

## Files Changed/Created

### New Files
- `tests/test_mcp_concept_tools.py`

### Modified Files
- `src/mcp_server/formatters.py` (add concept formatters)
- `src/mcp_server/tools.py` (add concept tools)
- `src/mcp_server/resources.py` (document new tools)

---

## Next Phase

After Phase 6C, proceed to Phase 6D: Graph Visualization.
