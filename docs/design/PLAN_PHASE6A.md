# Phase 6A: Concept Extraction - Core Infrastructure

## Overview

Build the foundational infrastructure for concept extraction: Pydantic models, LLM-powered extractor, Helix-DB schema extension, and query classes.

## Prerequisites

- Phase 3 complete (Helix-DB storage layer)
- Phase 4 complete (MCP server)
- Anthropic API key available

## Dependencies

**pyproject.toml additions:**
```toml
[project.optional-dependencies]
concepts = [
    "instructor>=1.0.0",
]
all = [
    # ... existing deps
    "instructor>=1.0.0",
]
```

---

## Implementation Details

### 1. Pydantic Models (`src/chunker/concept_extractor.py`)

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConceptType(str, Enum):
    """Types of concepts that can be extracted."""
    TERM = "term"              # Vocabulary: COGS, Depreciation, Liability
    METHOD = "method"          # Techniques: FIFO, LIFO, Weighted Average
    PRINCIPLE = "principle"    # Rules: Matching Principle, Revenue Recognition
    FORMULA = "formula"        # Calculations: COGS = Begin + Purchases - End
    ACCOUNT = "account"        # Ledger accounts: Inventory, Cost of Sales


class ExtractedConcept(BaseModel):
    """A concept extracted from textbook content."""

    name: str = Field(
        description="Canonical display name (e.g., 'Cost of Goods Sold')"
    )
    concept_type: ConceptType = Field(
        description="Type of concept"
    )
    definition: str = Field(
        description="1-2 sentence definition explaining the concept"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations (e.g., ['COGS', 'cost of sales'])"
    )


class ConceptRelation(BaseModel):
    """A relationship between two concepts."""

    from_concept: str = Field(
        description="Source concept name"
    )
    to_concept: str = Field(
        description="Target concept name"
    )
    relation_type: str = Field(
        description="Relationship type: uses, requires, calculated_from, "
                    "component_of, recorded_in, supersedes"
    )


class ChunkExtraction(BaseModel):
    """All concepts and relationships extracted from a single chunk."""

    defined_concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description="Concepts that are DEFINED in this chunk (have definitions)"
    )
    mentioned_concepts: list[str] = Field(
        default_factory=list,
        description="Concept names that are mentioned but not defined here"
    )
    relationships: list[ConceptRelation] = Field(
        default_factory=list,
        description="Relationships between concepts found in this chunk"
    )
```

### 2. ConceptExtractor Class (`src/chunker/concept_extractor.py`)

```python
import instructor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract concepts from chunks using Instructor + Claude."""

    def __init__(
        self,
        provider: str = "anthropic/claude-sonnet-4-20250514",
        max_retries: int = 2,
    ):
        """
        Initialize extractor.

        Args:
            provider: Instructor provider string (e.g., "anthropic/claude-sonnet-4-20250514")
            max_retries: Number of retries on validation failure
        """
        self.client = instructor.from_provider(provider)
        self.max_retries = max_retries

    def extract_from_chunk(
        self,
        content: str,
        section_path: list[str],
        document_title: str,
        existing_concepts: Optional[list[str]] = None,
    ) -> ChunkExtraction:
        """
        Extract concepts and relationships from a single chunk.

        Args:
            content: Chunk text content
            section_path: Breadcrumb path ["Chapter 5", "Inventory", "COGS Methods"]
            document_title: Source document name for context
            existing_concepts: Known concepts to reference (avoid duplicates)

        Returns:
            ChunkExtraction with defined concepts, mentioned concepts, relationships
        """
        context = f"Document: {document_title}\nSection: {' > '.join(section_path)}"
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
                    {"role": "user", "content": f"{context}\n\nContent:\n{content}"}
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
    ) -> tuple[list[ExtractedConcept], list[ConceptRelation]]:
        """
        Extract all concepts from a document's chunks.

        Args:
            chunks: List of chunk dicts with 'content' and 'section_path' keys
            document_title: Document title for context

        Returns:
            Tuple of (deduplicated concepts, all relationships)
        """
        all_concepts: dict[str, ExtractedConcept] = {}  # name -> concept
        all_relations: list[ConceptRelation] = []
        known_names: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting concepts from chunk {i+1}/{len(chunks)}")

            extraction = self.extract_from_chunk(
                content=chunk["content"],
                section_path=chunk.get("section_path", []),
                document_title=document_title,
                existing_concepts=known_names,
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


def slugify(name: str) -> str:
    """Convert concept name to URL-safe ID."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    return slug
```

### 3. Schema Extension (`db/schema.hx`)

Add after existing node/edge definitions:

```hx
// ========== CONCEPT GRAPH ==========

// Concept node: Extracted entity from textbooks
N::Concept {
    INDEX concept_id: String,      // Slugified: "cost-of-goods-sold"
    INDEX name: String,            // Display: "Cost of Goods Sold"
    definition: String,            // Brief definition
    concept_type: String,          // term, method, principle, formula, account
    source_documents: String,      // JSON array of doc_ids
    aliases: String,               // JSON array: ["COGS", "cost of sales"]
}

// Chunk defines a concept (contains definition)
E::DefinesConcept {
    From: Chunk,
    To: Concept,
}

// Chunk mentions a concept (references without defining)
E::MentionsConcept {
    From: Chunk,
    To: Concept,
}

// Concept relates to another concept
E::RelatesTo {
    From: Concept,
    To: Concept,
}
```

### 4. HelixQL Queries (`db/queries.hx`)

Add after existing queries:

```hx
// ========== CONCEPT MUTATIONS ==========

// Add a new concept
QUERY AddConcept(
    concept_id: String,
    name: String,
    definition: String,
    concept_type: String,
    source_documents: String,
    aliases: String
) =>
    concept <- AddN<Concept>({
        concept_id: concept_id,
        name: name,
        definition: definition,
        concept_type: concept_type,
        source_documents: source_documents,
        aliases: aliases
    })
    RETURN concept

// Link chunk to concept it defines
QUERY LinkChunkDefinesConcept(chunk_id: ID, concept_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    concept <- N<Concept>(concept_id)
    AddE<DefinesConcept>::From(chunk)::To(concept)
    RETURN chunk

// Link chunk to concept it mentions
QUERY LinkChunkMentionsConcept(chunk_id: ID, concept_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    concept <- N<Concept>(concept_id)
    AddE<MentionsConcept>::From(chunk)::To(concept)
    RETURN chunk

// Link two concepts with a relationship
QUERY LinkConceptRelatesTo(from_id: ID, to_id: ID) =>
    from_concept <- N<Concept>(from_id)
    to_concept <- N<Concept>(to_id)
    AddE<RelatesTo>::From(from_concept)::To(to_concept)
    RETURN from_concept

// ========== CONCEPT QUERIES ==========

// Get concept by name
QUERY GetConceptByName(name: String) =>
    concepts <- N<Concept>::WHERE(_::{name}::EQ(name))
    RETURN concepts

// Get concept by concept_id
QUERY GetConceptById(concept_id: String) =>
    concepts <- N<Concept>::WHERE(_::{concept_id}::EQ(concept_id))
    RETURN concepts

// List all concepts
QUERY ListConcepts() =>
    concepts <- N<Concept>
    RETURN concepts

// List concepts from a specific document
QUERY ListDocumentConcepts(document_id: String) =>
    chunks <- N<Chunk>::WHERE(_::{document_id}::EQ(document_id))
    concepts <- chunks::Out<DefinesConcept>
    RETURN concepts

// ========== CONCEPT TRAVERSALS ==========

// Get chunks that define a concept (for citations)
QUERY GetConceptDefinitionChunks(concept_id: ID) =>
    chunks <- N<Concept>(concept_id)::In<DefinesConcept>
    RETURN chunks

// Get chunks that mention a concept
QUERY GetConceptMentionChunks(concept_id: ID) =>
    chunks <- N<Concept>(concept_id)::In<MentionsConcept>
    RETURN chunks

// Get related concepts (one hop out)
QUERY GetRelatedConcepts(concept_id: ID) =>
    related <- N<Concept>(concept_id)::Out<RelatesTo>
    RETURN related

// Get concepts that relate TO this one (reverse)
QUERY GetConceptDependents(concept_id: ID) =>
    dependents <- N<Concept>(concept_id)::In<RelatesTo>
    RETURN dependents

// ========== CONCEPT DELETION ==========

// Delete a concept
QUERY DropConcept(concept_id: ID) =>
    DROP N<Concept>(concept_id)
    RETURN "Removed concept"
```

### 5. Python Query Classes (`src/storage/queries.py`)

Add after existing query classes:

```python
# ========== CONCEPT QUERIES ==========

class AddConcept(_get_query_base_class()):
    """Add a concept node to the database."""

    def __init__(
        self,
        concept_id: str,
        name: str,
        definition: str,
        concept_type: str,
        source_documents: list[str],
        aliases: list[str],
    ) -> None:
        super().__init__()
        self.concept_id = concept_id
        self.name = name
        self.definition = definition
        self.concept_type = concept_type
        self.source_documents = source_documents
        self.aliases = aliases

    def query(self) -> list[dict[str, Any]]:
        return [{
            "concept_id": self.concept_id,
            "name": self.name,
            "definition": self.definition,
            "concept_type": self.concept_type,
            "source_documents": json.dumps(self.source_documents),
            "aliases": json.dumps(self.aliases),
        }]

    def response(self, response: Any) -> QueryResult:
        return QueryResult(success=True, data={"concept_id": self.concept_id})


class GetConceptByName(_get_query_base_class()):
    """Get concept by name."""

    def __init__(self, name: str) -> None:
        super().__init__(endpoint="GetConceptByName")
        self.name = name

    def query(self) -> list[dict[str, Any]]:
        return [{"name": self.name}]

    def response(self, response: Any) -> QueryResult:
        if isinstance(response, list) and len(response) > 0:
            return QueryResult(success=True, data={"node": response[0]})
        return QueryResult(success=False, error="Concept not found")


class GetConceptById(_get_query_base_class()):
    """Get concept by concept_id."""

    def __init__(self, concept_id: str) -> None:
        super().__init__(endpoint="GetConceptById")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        if isinstance(response, list) and len(response) > 0:
            return QueryResult(success=True, data={"node": response[0]})
        return QueryResult(success=False, error="Concept not found")


class ListConcepts(_get_query_base_class()):
    """List all concepts."""

    def __init__(self) -> None:
        super().__init__(endpoint="ListConcepts")

    def query(self) -> list[dict[str, Any]]:
        return [{}]

    def response(self, response: Any) -> QueryResult:
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class LinkChunkDefinesConcept(_get_query_base_class()):
    """Link chunk to concept it defines."""

    def __init__(self, chunk_id: str, concept_id: str) -> None:
        super().__init__(endpoint="LinkChunkDefinesConcept")
        self.chunk_id = chunk_id
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        return [{"chunk_id": self.chunk_id, "concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        return QueryResult(success=True)


class LinkChunkMentionsConcept(_get_query_base_class()):
    """Link chunk to concept it mentions."""

    def __init__(self, chunk_id: str, concept_id: str) -> None:
        super().__init__(endpoint="LinkChunkMentionsConcept")
        self.chunk_id = chunk_id
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        return [{"chunk_id": self.chunk_id, "concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        return QueryResult(success=True)


class LinkConceptRelatesTo(_get_query_base_class()):
    """Link two concepts with a relationship."""

    def __init__(self, from_id: str, to_id: str) -> None:
        super().__init__(endpoint="LinkConceptRelatesTo")
        self.from_id = from_id
        self.to_id = to_id

    def query(self) -> list[dict[str, Any]]:
        return [{"from_id": self.from_id, "to_id": self.to_id}]

    def response(self, response: Any) -> QueryResult:
        return QueryResult(success=True)


class GetConceptDefinitionChunks(_get_query_base_class()):
    """Get chunks that define a concept (for citations)."""

    def __init__(self, concept_id: str) -> None:
        super().__init__(endpoint="GetConceptDefinitionChunks")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetRelatedConcepts(_get_query_base_class()):
    """Get concepts related to a given concept."""

    def __init__(self, concept_id: str) -> None:
        super().__init__(endpoint="GetRelatedConcepts")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])
```

### 6. Storage Client Extension (`src/storage/client.py`)

Add methods to `HelixChunkStore`:

```python
# Concept storage methods

def store_concept(
    self,
    concept_id: str,
    name: str,
    definition: str,
    concept_type: str,
    source_documents: list[str],
    aliases: list[str],
) -> Optional[str]:
    """Store a concept and return its internal ID."""
    self._ensure_connected()
    query = AddConcept(
        concept_id=concept_id,
        name=name,
        definition=definition,
        concept_type=concept_type,
        source_documents=source_documents,
        aliases=aliases,
    )
    result = self._execute_query(query)
    if result.success:
        return self._extract_node_id(result.data)
    return None

def get_concept(self, concept_id: str) -> Optional[dict]:
    """Get concept by concept_id."""
    self._ensure_connected()
    query = GetConceptById(concept_id)
    result = self._execute_query(query)
    if result.success and result.data:
        return result.data.get("node")
    return None

def get_concept_by_name(self, name: str) -> Optional[dict]:
    """Get concept by name."""
    self._ensure_connected()
    query = GetConceptByName(name)
    result = self._execute_query(query)
    if result.success and result.data:
        return result.data.get("node")
    return None

def list_concepts(self, doc_id: Optional[str] = None) -> list[dict]:
    """List all concepts, optionally filtered by document."""
    self._ensure_connected()
    if doc_id:
        query = ListDocumentConcepts(doc_id)
    else:
        query = ListConcepts()
    result = self._execute_query(query)
    if result.success:
        return result.data or []
    return []

def link_chunk_defines_concept(
    self,
    chunk_internal_id: str,
    concept_internal_id: str,
) -> bool:
    """Create DefinesConcept edge from chunk to concept."""
    self._ensure_connected()
    query = LinkChunkDefinesConcept(chunk_internal_id, concept_internal_id)
    result = self._execute_query(query)
    return result.success

def link_chunk_mentions_concept(
    self,
    chunk_internal_id: str,
    concept_internal_id: str,
) -> bool:
    """Create MentionsConcept edge from chunk to concept."""
    self._ensure_connected()
    query = LinkChunkMentionsConcept(chunk_internal_id, concept_internal_id)
    result = self._execute_query(query)
    return result.success

def link_concept_relates_to(
    self,
    from_concept_id: str,
    to_concept_id: str,
) -> bool:
    """Create RelatesTo edge between concepts."""
    self._ensure_connected()
    query = LinkConceptRelatesTo(from_concept_id, to_concept_id)
    result = self._execute_query(query)
    return result.success

def get_concept_definition_chunks(self, concept_id: str) -> list[dict]:
    """Get chunks that define a concept (for citations)."""
    self._ensure_connected()
    # First get concept internal ID
    concept = self.get_concept(concept_id)
    if not concept:
        return []
    internal_id = self._extract_node_id(concept)
    if not internal_id:
        return []

    query = GetConceptDefinitionChunks(internal_id)
    result = self._execute_query(query)
    if result.success:
        return result.data or []
    return []

def get_related_concepts(self, concept_id: str) -> list[dict]:
    """Get concepts related to a given concept."""
    self._ensure_connected()
    concept = self.get_concept(concept_id)
    if not concept:
        return []
    internal_id = self._extract_node_id(concept)
    if not internal_id:
        return []

    query = GetRelatedConcepts(internal_id)
    result = self._execute_query(query)
    if result.success:
        return result.data or []
    return []
```

---

## Test Plan

### test_concept_extractor.py (~20 tests)

- ConceptType enum values
- ExtractedConcept model validation
- ConceptRelation model validation
- ChunkExtraction model validation
- slugify function
- ConceptExtractor initialization (mock provider)
- extract_from_chunk with mock responses
- extract_from_chunks deduplication
- Error handling for failed extraction

### test_concept_queries.py (~15 tests)

- AddConcept query payload
- GetConceptByName query payload
- GetConceptById query payload
- ListConcepts query payload
- LinkChunkDefinesConcept query payload
- LinkChunkMentionsConcept query payload
- LinkConceptRelatesTo query payload
- GetConceptDefinitionChunks query payload
- GetRelatedConcepts query payload
- Response processing for all query types

---

## Verification

1. **Schema deployment:**
   ```bash
   helix push dev
   ```

2. **Unit tests:**
   ```bash
   PYTHONPATH=. pytest tests/test_concept_extractor.py -v
   PYTHONPATH=. pytest tests/test_concept_queries.py -v
   ```

3. **Manual extraction test:**
   ```python
   from src.chunker.concept_extractor import ConceptExtractor

   extractor = ConceptExtractor()
   result = extractor.extract_from_chunk(
       content="COGS (Cost of Goods Sold) is calculated as...",
       section_path=["Chapter 5", "COGS"],
       document_title="Inventory Textbook",
   )
   print(result)
   ```

---

## Files Changed/Created

### New Files
- `src/chunker/concept_extractor.py`
- `tests/test_concept_extractor.py`
- `tests/test_concept_queries.py`

### Modified Files
- `pyproject.toml` (add instructor dependency)
- `db/schema.hx` (add Concept node, edges)
- `db/queries.hx` (add concept queries)
- `src/storage/queries.py` (add Python query classes)
- `src/storage/client.py` (add concept methods)
- `src/storage/schema.py` (add schema constants)

---

## Next Phase

After Phase 6A, proceed to Phase 6B: CLI and Ingestion Integration.
