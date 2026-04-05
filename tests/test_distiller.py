"""Tests for textbook knowledge distillation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.distiller.distiller import (
    TOOL_IDS,
    AgentIdentity,
    TextbookDistiller,
    ToolExampleSet,
    _slugify,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """Create a mock store with realistic data."""
    store = MagicMock()

    store.list_documents.return_value = [
        {"doc_id": "test_book", "title": "Test Book on Inventory"},
    ]

    store.get_document_summaries.side_effect = lambda doc_id, level=None: {
        "document": [
            {
                "summary_id": "summary_test_book_document",
                "content": "This book covers inventory management fundamentals.",
                "level": "document",
                "key_points": json.dumps([
                    "FIFO and LIFO valuation methods",
                    "Safety stock calculation",
                    "Economic Order Quantity formula",
                ]),
            }
        ],
        "chapter": [
            {
                "summary_id": "summary_ch1_chapter",
                "content": "Chapter 1 introduces inventory basics.",
                "level": "chapter",
            },
            {
                "summary_id": "summary_ch2_chapter",
                "content": "Chapter 2 covers valuation methods.",
                "level": "chapter",
            },
        ],
    }.get(level, [])

    store.list_concepts.return_value = [
        {
            "concept_id": "fifo",
            "name": "FIFO",
            "concept_type": "method",
            "definition": "First-In, First-Out inventory valuation method",
            "aliases": '["First In First Out"]',
        },
        {
            "concept_id": "lifo",
            "name": "LIFO",
            "concept_type": "method",
            "definition": "Last-In, First-Out inventory valuation method",
            "aliases": "[]",
        },
        {
            "concept_id": "safety-stock",
            "name": "Safety Stock",
            "concept_type": "term",
            "definition": "Buffer inventory to prevent stockouts",
            "aliases": "[]",
        },
        {
            "concept_id": "eoq",
            "name": "Economic Order Quantity",
            "concept_type": "formula",
            "definition": "Optimal order quantity that minimizes total inventory costs",
            "aliases": '["EOQ"]',
        },
        {
            "concept_id": "holding-cost",
            "name": "Holding Cost",
            "concept_type": "term",
            "definition": "Cost of storing inventory over time",
            "aliases": '["carrying cost"]',
        },
    ]

    def mock_related(concept_id):
        relations = {
            "fifo": [
                {"concept": {"name": "LIFO", "concept_id": "lifo"}, "relation_type": "relates_to"},
                {"concept": {"name": "COGS", "concept_id": "cogs"}, "relation_type": "calculated_from"},
            ],
            "eoq": [
                {"concept": {"name": "Holding Cost", "concept_id": "holding-cost"}, "relation_type": "requires"},
                {"concept": {"name": "Ordering Cost", "concept_id": "ordering-cost"}, "relation_type": "requires"},
            ],
            "safety-stock": [
                {"concept": {"name": "Reorder Point", "concept_id": "reorder-point"}, "relation_type": "calculated_from"},
            ],
            "lifo": [],
            "holding-cost": [],
        }
        return relations.get(concept_id, [])

    store.get_related_concepts_with_types.side_effect = mock_related
    return store


@pytest.fixture
def mock_identity():
    """Pre-built identity for assembly tests."""
    return AgentIdentity(
        role="an expert in Inventory Management",
        personality="precise and pedagogical",
        goal="Help users understand inventory concepts using textbook knowledge.",
        success_criteria="Every answer cites at least one source.",
        constraints=[
            "Always cite sources",
            "Prefer textbook definitions over general knowledge",
        ],
        scope="Test Book on Inventory",
        tool_guidance="1. Use search_concepts first\n2. Use get_concept_with_citations for details",
    )


# ---------------------------------------------------------------------------
# Tests: _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello_world"

    def test_special_chars(self):
        assert _slugify("Inventory & Accounting: A Guide!") == "inventory_accounting_a_guide"

    def test_truncation(self):
        result = _slugify("a" * 100)
        assert len(result) <= 80

    def test_underscores(self):
        assert _slugify("one__two___three") == "one_two_three"


# ---------------------------------------------------------------------------
# Tests: _rank_concepts
# ---------------------------------------------------------------------------


class TestRankConcepts:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_ranks_by_connection_count(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=3)

        # FIFO has 2 connections, EOQ has 2, Safety Stock has 1
        assert len(ranked) == 3
        assert ranked[0]["_connection_count"] >= ranked[1]["_connection_count"]
        assert ranked[1]["_connection_count"] >= ranked[2]["_connection_count"]

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_top_k_limits_results(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=2)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# Tests: _build_prerequisites
# ---------------------------------------------------------------------------


class TestBuildPrerequisites:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_extracts_requires_edges(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)
        prereqs = distiller._build_prerequisites(ranked)

        # EOQ requires Holding Cost and Ordering Cost
        eoq_prereq = next((p for p in prereqs if p["from"] == "Economic Order Quantity"), None)
        assert eoq_prereq is not None
        assert "Holding Cost" in eoq_prereq["requires"]
        assert "Ordering Cost" in eoq_prereq["requires"]

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_extracts_calculated_from_edges(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)
        prereqs = distiller._build_prerequisites(ranked)

        # FIFO is calculated_from COGS, Safety Stock from Reorder Point
        fifo_prereq = next((p for p in prereqs if p["from"] == "FIFO"), None)
        assert fifo_prereq is not None
        assert "COGS" in fifo_prereq["requires"]

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_skips_non_prerequisite_edges(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        # LIFO has no relations, so no prerequisites
        ranked = [{"concept_id": "lifo", "name": "LIFO"}]
        prereqs = distiller._build_prerequisites(ranked)
        assert prereqs == []


# ---------------------------------------------------------------------------
# Tests: _assemble_config
# ---------------------------------------------------------------------------


class TestAssembleConfig:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_output_has_all_agent_config_fields(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)

        result = distiller._assemble_config(
            doc_id="test_book",
            doc_title="Test Book on Inventory",
            identity=mock_identity,
            examples=[{"user": "test", "tool_calls": []}],
            ranked_concepts=ranked,
            all_concepts=concepts,
            chapter_summaries=[{"content": "ch1"}, {"content": "ch2"}],
            prerequisites=[],
        )

        # Required AgentConfig fields
        assert "name" in result
        assert "role" in result
        assert "expertise" in result
        assert "personality" in result
        assert "goal" in result
        assert "success_criteria" in result
        assert "constraints" in result
        assert "scope" in result
        assert "tool_ids" in result
        assert "tool_guidance" in result
        assert "tool_examples" in result
        assert "output_format" in result
        assert "provider" in result
        assert "model" in result
        assert "execution_mode" in result
        assert "budget" in result
        assert "intervention_pipeline" in result
        assert "sandbox" in result
        assert "disabled_builtins" in result

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_name_is_slugified(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        result = distiller._assemble_config(
            doc_id="test_book",
            doc_title="Test Book on Inventory",
            identity=mock_identity,
            examples=[],
            ranked_concepts=[],
            all_concepts=[],
            chapter_summaries=[],
            prerequisites=[],
        )
        assert result["name"] == "test_book_on_inventory_specialist"

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_tool_ids_are_valid(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        result = distiller._assemble_config(
            doc_id="test_book",
            doc_title="Test",
            identity=mock_identity,
            examples=[],
            ranked_concepts=[],
            all_concepts=[],
            chapter_summaries=[],
            prerequisites=[],
        )
        assert result["tool_ids"] == TOOL_IDS
        assert "search_semantic" in result["tool_ids"]
        assert "get_concept_with_citations" in result["tool_ids"]

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_metadata_present(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)

        result = distiller._assemble_config(
            doc_id="test_book",
            doc_title="Test Book",
            identity=mock_identity,
            examples=[],
            ranked_concepts=ranked,
            all_concepts=concepts,
            chapter_summaries=[{"content": "ch1"}],
            prerequisites=[{"from": "EOQ", "requires": ["Holding Cost"]}],
        )

        meta = result["_distill_metadata"]
        assert meta["source_document_id"] == "test_book"
        assert meta["total_concepts"] == 5
        assert meta["total_chapters"] == 1
        assert len(meta["prerequisite_skills"]) == 1
        assert "concept_types" in meta
        assert "distilled_at" in meta

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_output_is_json_serializable(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        result = distiller._assemble_config(
            doc_id="test_book",
            doc_title="Test",
            identity=mock_identity,
            examples=[{"user": "q", "tool_calls": [{"tool": "search_concepts", "args": {}}]}],
            ranked_concepts=[],
            all_concepts=[],
            chapter_summaries=[],
            prerequisites=[],
        )
        # Must not raise
        serialized = json.dumps(result, indent=2)
        deserialized = json.loads(serialized)
        assert deserialized["name"] == result["name"]


# ---------------------------------------------------------------------------
# Tests: _get_doc_title
# ---------------------------------------------------------------------------


class TestGetDocTitle:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_returns_title_when_found(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        assert distiller._get_doc_title("test_book") == "Test Book on Inventory"

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_returns_doc_id_when_not_found(self, mock_instructor, mock_store):
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        assert distiller._get_doc_title("nonexistent") == "nonexistent"
