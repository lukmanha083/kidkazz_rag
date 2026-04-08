"""Distill textbook knowledge into specialist agent configs for agent_ex.

Reads a document's concept graph + chapter summaries from Helix-DB and produces
an AgentConfig-compatible JSON that agent_ex can import as a specialist agent.

The specialist agent uses kidkazz_rag's MCP tools at runtime to answer domain
questions grounded in textbook content.
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional instructor import (same pattern as summarizer.py)
# ---------------------------------------------------------------------------
try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None  # type: ignore[assignment]
    INSTRUCTOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class AgentIdentity(BaseModel):
    """LLM-generated agent identity fields."""

    role: str = Field(description="One-sentence role description starting with 'an expert in...'")
    personality: str = Field(
        description="Communication style in 2-5 words (e.g., 'precise and pedagogical')"
    )
    goal: str = Field(description="2-3 sentence goal describing what the agent helps users do")
    success_criteria: str = Field(description="1-2 sentence criteria for a good answer")
    constraints: list[str] = Field(
        description="3-6 rules the agent must follow (cite sources, handle uncertainty, etc.)"
    )
    scope: str = Field(description="One sentence describing the agent's knowledge boundary")
    tool_guidance: str = Field(
        description="Markdown-formatted guide for when/how to use each MCP tool"
    )


class ToolCall(BaseModel):
    """A single tool call in a few-shot example."""

    tool: str = Field(description="MCP tool name")
    args: dict[str, Any] = Field(description="Tool arguments")


class ToolExample(BaseModel):
    """A few-shot example showing how to use MCP tools."""

    user: str = Field(description="User question")
    assistant_thinking: str = Field(description="Brief reasoning about which tools to use")
    tool_calls: list[ToolCall] = Field(description="Ordered list of tool calls")
    response_sketch: str = Field(description="Brief sketch of what the response would contain")


class ToolExampleSet(BaseModel):
    """Set of few-shot tool usage examples."""

    examples: list[ToolExample] = Field(description="3-5 diverse usage examples")


# ---------------------------------------------------------------------------
# MCP tool catalog (source of truth: src/mcp_server/tools.py)
# ---------------------------------------------------------------------------

# Tools a specialist agent should have access to
TOOL_IDS = [
    "search_semantic",
    "search_keyword",
    "get_chunk",
    "get_context_window",
    "get_parent",
    "get_children",
    "search_concepts",
    "get_concept",
    "get_concept_with_citations",
    "get_related_concepts",
    "get_document_summary",
    "get_chapter_summaries",
    "search_summaries",
    "list_concepts",
]

# Brief descriptions for the LLM to understand each tool
TOOL_CATALOG = [
    ("search_semantic", "Vector search for relevant text passages. Filters: doc_id, tags, has_table, has_code, has_math, has_image, header_level, semantic_type"),
    ("search_keyword", "Full-text keyword search across chunks"),
    ("get_chunk", "Get a specific chunk by ID"),
    ("get_context_window", "Get a chunk with surrounding context (parent + siblings)"),
    ("get_parent", "Get parent chunk of a given chunk"),
    ("get_children", "Get child chunks of a given chunk"),
    ("search_concepts", "Search concepts by name or keyword with relevance scoring"),
    ("get_concept", "Get a concept by exact name"),
    ("get_concept_with_citations", "Get concept definition with source chunk citations and related concepts"),
    ("get_related_concepts", "Get concepts related to a given concept with relationship types"),
    ("get_document_summary", "Get document-level summary"),
    ("get_chapter_summaries", "Get all chapter-level summaries for a document"),
    ("search_summaries", "Semantic search across summaries with level/doc filters"),
    ("list_concepts", "List all concepts, optionally filtered by document or type"),
]


# ---------------------------------------------------------------------------
# TextbookDistiller
# ---------------------------------------------------------------------------


class TextbookDistiller:
    """Distill a document's knowledge base into a specialist agent config.

    Uses 2 LLM calls:
    1. Identity generation (role, goal, constraints, tool_guidance)
    2. Few-shot example generation (tool usage patterns)

    Everything else is computed from stored data (no LLM needed).
    """

    def __init__(self, store: Any, provider: str = "openai/gpt-4o-mini"):
        self.store = store
        self.provider = provider
        self.client = None

    def _ensure_instructor(self) -> None:
        """Lazily initialize the instructor client, raising on missing dep."""
        if self.client is not None:
            return
        if not INSTRUCTOR_AVAILABLE or instructor is None:
            raise ImportError(
                "instructor is not installed. "
                "Install with: pip install 'kidkazz[concepts]'"
            )
        self.client = instructor.from_provider(self.provider)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def distill(
        self,
        doc_id: str,
        top_k: int = 20,
        num_examples: int = 5,
    ) -> dict[str, Any]:
        """Distill a document into an AgentConfig-compatible dict.

        Args:
            doc_id: Document identifier in the knowledge base
            top_k: Number of top concepts to include in expertise
            num_examples: Number of few-shot tool examples to generate

        Returns:
            Dict matching agent_ex AgentConfig fields + _distill_metadata
        """
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if num_examples < 1:
            raise ValueError("num_examples must be a positive integer")

        logger.info("Gathering document data for %s...", doc_id)

        # 1. Gather data (no LLM)
        doc_title = self._get_doc_title(doc_id)
        doc_summary = self._get_doc_summary(doc_id)

        if not doc_summary:
            raise ValueError(
                f"No document-level summary found for '{doc_id}'. "
                "Run 'kidkazz summarize generate' first."
            )

        chapter_summaries = self._get_chapter_summaries(doc_id)
        concepts = self._get_concepts(doc_id)
        ranked_concepts = self._rank_concepts(concepts, top_k)
        prerequisites = self._build_prerequisites(ranked_concepts)

        logger.info(
            "Found %d concepts, %d chapters. Generating identity...",
            len(concepts),
            len(chapter_summaries),
        )

        self._ensure_instructor()

        # 2. Generate identity (LLM call 1)
        identity = self._generate_identity(
            doc_title, doc_id, doc_summary, chapter_summaries, ranked_concepts
        )

        logger.info("Generating %d tool usage examples...", num_examples)

        # 3. Generate examples (LLM call 2)
        examples = self._generate_examples(
            doc_id, doc_title, ranked_concepts[:10], doc_summary, num_examples
        )

        # 4. Assemble output
        return self._assemble_config(
            doc_id=doc_id,
            doc_title=doc_title,
            identity=identity,
            examples=examples,
            ranked_concepts=ranked_concepts,
            all_concepts=concepts,
            chapter_summaries=chapter_summaries,
            prerequisites=prerequisites,
        )

    # ------------------------------------------------------------------
    # Data gathering (no LLM)
    # ------------------------------------------------------------------

    def _get_doc_title(self, doc_id: str) -> str:
        """Get document title from metadata."""
        docs = self.store.list_documents()
        for doc in docs:
            if doc.get("doc_id") == doc_id:
                return doc.get("title", doc_id)
        return doc_id

    def _get_doc_summary(self, doc_id: str) -> Optional[dict]:
        """Get document-level summary."""
        summaries = self.store.get_document_summaries(doc_id, level="document")
        return summaries[0] if summaries else None

    def _get_chapter_summaries(self, doc_id: str) -> list[dict]:
        """Get chapter-level summaries."""
        return self.store.get_document_summaries(doc_id, level="chapter")

    def _get_concepts(self, doc_id: str) -> list[dict]:
        """Get all concepts for a document."""
        return self.store.list_concepts(doc_id=doc_id)

    def _rank_concepts(self, concepts: list[dict], top_k: int) -> list[dict]:
        """Rank concepts by connection count (most connected = most important).

        For each concept, queries related concepts and counts connections.
        Caches the related data in ``_related`` so ``_build_prerequisites``
        can reuse it without re-querying.
        """
        ranked = []
        skipped = 0
        for concept in concepts:
            concept_id = concept.get("concept_id", "")
            if not concept_id:
                skipped += 1
                continue

            try:
                related = self.store.get_related_concepts_with_types(concept_id)
            except Exception as e:
                logger.warning("Failed to get related concepts for %s: %s", concept_id, e)
                related = []

            ranked.append({
                **concept,
                "_connection_count": len(related),
                "_related": related,
            })

        if skipped:
            logger.warning("Skipped %d concepts with missing concept_id", skipped)

        ranked.sort(key=lambda c: c.get("_connection_count", 0), reverse=True)
        return ranked[:top_k]

    def _build_prerequisites(self, ranked_concepts: list[dict]) -> list[dict]:
        """Build prerequisite chains from 'requires'/'calculated_from' edges.

        Uses cached ``_related`` data from ``_rank_concepts`` to avoid
        redundant store queries.
        """
        prerequisites = []
        prerequisite_types = {"requires", "calculated_from"}

        for concept in ranked_concepts:
            concept_id = concept.get("concept_id", "")
            if not concept_id:
                continue

            related = concept.get("_related", [])

            deps = [
                r["concept"].get("name", r["concept"].get("concept_id", ""))
                for r in related
                if r.get("relation_type", "").lower() in prerequisite_types
            ]

            if deps:
                prerequisites.append({
                    "from": concept.get("name", concept_id),
                    "requires": deps,
                })

        return prerequisites

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _generate_identity(
        self,
        doc_title: str,
        doc_id: str,
        doc_summary: dict,
        chapter_summaries: list[dict],
        ranked_concepts: list[dict],
    ) -> AgentIdentity:
        """LLM call 1: Generate agent identity from document knowledge."""
        # Build context for the LLM
        summary_text = doc_summary.get("content", "")
        key_points = doc_summary.get("key_points", [])
        if isinstance(key_points, str):
            try:
                key_points = json.loads(key_points)
            except (json.JSONDecodeError, TypeError):
                key_points = [key_points] if key_points else []

        chapter_overview = "\n".join(
            f"- {ch.get('content', '')[:150]}"
            for ch in chapter_summaries[:15]
        )

        concept_list = ", ".join(
            c.get("name", "") for c in ranked_concepts[:15]
        )

        concept_types = Counter(
            c.get("concept_type", "unknown") for c in ranked_concepts
        )
        type_summary = ", ".join(
            f"{count} {ctype}" for ctype, count in concept_types.most_common(5)
        )

        tool_catalog_text = "\n".join(
            f"- {name}: {desc}" for name, desc in TOOL_CATALOG
        )

        prompt = f"""You are creating a specialist AI agent configuration for a textbook knowledge base.

DOCUMENT: "{doc_title}" (ID: {doc_id})

DOCUMENT SUMMARY:
{summary_text}

KEY POINTS:
{chr(10).join(f'- {kp}' for kp in key_points[:10])}

CHAPTERS ({len(chapter_summaries)} total):
{chapter_overview}

TOP CONCEPTS ({len(ranked_concepts)} most connected):
{concept_list}

CONCEPT TYPE DISTRIBUTION: {type_summary}

AVAILABLE MCP TOOLS (the agent uses these to search the knowledge base):
{tool_catalog_text}

Generate the agent's identity configuration. The tool_guidance should be a practical
markdown guide telling the agent WHEN and HOW to use each tool for different question types
(factual, procedural, comparative, overview). Reference the doc_id '{doc_id}' in tool guidance
where appropriate for filtering."""

        return self.client.chat.completions.create(
            model=self.provider.split("/")[-1],
            response_model=AgentIdentity,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )

    def _generate_examples(
        self,
        doc_id: str,
        doc_title: str,
        top_concepts: list[dict],
        doc_summary: dict,
        num_examples: int,
    ) -> list[dict]:
        """LLM call 2: Generate few-shot tool usage examples."""
        concept_context = "\n".join(
            f"- {c.get('name', '')} ({c.get('concept_type', '')}): {c.get('definition', '')[:100]}"
            for c in top_concepts
        )

        tool_catalog_text = "\n".join(
            f"- {name}: {desc}" for name, desc in TOOL_CATALOG
        )

        prompt = f"""Generate {num_examples} diverse few-shot examples showing how a specialist agent
would use MCP tools to answer questions about "{doc_title}" (doc_id: "{doc_id}").

TOP CONCEPTS IN THIS KNOWLEDGE BASE:
{concept_context}

AVAILABLE MCP TOOLS:
{tool_catalog_text}

Create examples covering these question types:
1. Factual/definition lookup (use search_concepts + get_concept_with_citations)
2. Formula/calculation (use search_semantic with has_math=true)
3. Comparison between concepts (use get_concept_with_citations + get_related_concepts)
4. Chapter/topic overview (use get_chapter_summaries or search_summaries)
5. Procedure/how-to (use search_semantic + get_context_window)

Use REAL concept names from the list above. Use doc_id="{doc_id}" in search filters.
Each example should show realistic tool_calls with actual argument values."""

        result = self.client.chat.completions.create(
            model=self.provider.split("/")[-1],
            response_model=ToolExampleSet,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )

        return [ex.model_dump() for ex in result.examples]

    # ------------------------------------------------------------------
    # Assembly (no LLM)
    # ------------------------------------------------------------------

    def _assemble_config(
        self,
        doc_id: str,
        doc_title: str,
        identity: AgentIdentity,
        examples: list[dict],
        ranked_concepts: list[dict],
        all_concepts: list[dict],
        chapter_summaries: list[dict],
        prerequisites: list[dict],
    ) -> dict[str, Any]:
        """Assemble all pieces into an AgentConfig-compatible JSON dict."""
        # Build expertise from top concept names
        expertise = [
            c.get("name", "") for c in ranked_concepts[:10] if c.get("name")
        ]

        # Build name from document title
        name = _slugify(doc_title) + "_specialist"

        # Concept type distribution
        type_dist = Counter(
            c.get("concept_type", "unknown") for c in all_concepts
        )

        # Top concepts by connection count
        top_by_connections = [
            {"name": c.get("name", ""), "connections": c.get("_connection_count", 0)}
            for c in ranked_concepts[:10]
        ]

        return {
            # AgentConfig fields
            "name": name,
            "description": (
                f"Specialist agent distilled from '{doc_title}' "
                f"with {len(all_concepts)} concepts across "
                f"{len(chapter_summaries)} chapters"
            ),
            "role": identity.role,
            "expertise": expertise,
            "personality": identity.personality,
            "goal": identity.goal,
            "success_criteria": identity.success_criteria,
            "constraints": identity.constraints,
            "scope": identity.scope,
            "tool_ids": TOOL_IDS,
            "tool_guidance": identity.tool_guidance,
            "tool_examples": examples,
            "output_format": (
                "## Answer\n"
                "[Clear explanation with definitions and context]\n\n"
                "## Sources\n"
                "- Chapter X, Section Y: \"relevant quote\"\n"
                "- Concept: [concept_name] - [definition]\n\n"
                "## Related Concepts\n"
                "- [concept_1]: [brief relevance]\n"
                "- [concept_2]: [brief relevance]"
            ),
            "system_prompt": None,
            "provider": self.provider.split("/")[0] if "/" in self.provider else "openai",
            "model": self.provider.split("/", 1)[1] if "/" in self.provider else self.provider,
            "disabled_builtins": [],
            "intervention_pipeline": [],
            "sandbox": {},
            "execution_mode": "interactive",
            "budget": {},
            # Metadata (stripped by agent_ex on import)
            "_distill_metadata": {
                "source_document_id": doc_id,
                "source_document_title": doc_title,
                "total_concepts": len(all_concepts),
                "total_chapters": len(chapter_summaries),
                "concept_types": dict(type_dist.most_common()),
                "top_concepts_by_connections": top_by_connections,
                "prerequisite_skills": prerequisites,
                "distilled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "provider": self.provider,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    result = text.strip("_")[:80]
    return result or "unnamed"


def _sanitize_filename(doc_id: str) -> str:
    """Sanitize a doc_id for use in a filename (prevent path traversal)."""
    return re.sub(r"[^\w\-]", "_", doc_id)[:100]
