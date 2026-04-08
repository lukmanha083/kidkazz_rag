"""Tests for textbook knowledge distillation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.cli.commands.distill import _safe_filename
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
            "id": "helix_fifo",
            "concept_id": "fifo",
            "name": "FIFO",
            "concept_type": "method",
            "definition": "First-In, First-Out inventory valuation method",
            "aliases": '["First In First Out"]',
        },
        {
            "id": "helix_lifo",
            "concept_id": "lifo",
            "name": "LIFO",
            "concept_type": "method",
            "definition": "Last-In, First-Out inventory valuation method",
            "aliases": "[]",
        },
        {
            "id": "helix_safety_stock",
            "concept_id": "safety-stock",
            "name": "Safety Stock",
            "concept_type": "term",
            "definition": "Buffer inventory to prevent stockouts",
            "aliases": "[]",
        },
        {
            "id": "helix_eoq",
            "concept_id": "eoq",
            "name": "Economic Order Quantity",
            "concept_type": "formula",
            "definition": "Optimal order quantity that minimizes total inventory costs",
            "aliases": '["EOQ"]',
        },
        {
            "id": "helix_holding_cost",
            "concept_id": "holding-cost",
            "name": "Holding Cost",
            "concept_type": "term",
            "definition": "Cost of storing inventory over time",
            "aliases": '["carrying cost"]',
        },
    ]

    # Related data keyed by internal Helix ID (used by _direct methods)
    _related_by_internal_id = {
        "helix_fifo": [
            {"concept": {"name": "LIFO", "concept_id": "lifo"}, "relation_type": "relates_to"},
            {"concept": {"name": "COGS", "concept_id": "cogs"}, "relation_type": "calculated_from"},
        ],
        "helix_eoq": [
            {"concept": {"name": "Holding Cost", "concept_id": "holding-cost"}, "relation_type": "requires"},
            {"concept": {"name": "Ordering Cost", "concept_id": "ordering-cost"}, "relation_type": "requires"},
        ],
        "helix_safety_stock": [
            {"concept": {"name": "Reorder Point", "concept_id": "reorder-point"}, "relation_type": "calculated_from"},
        ],
        "helix_lifo": [],
        "helix_holding_cost": [],
    }

    # Legacy method (keyed by concept_id string)
    def mock_related(concept_id):
        relations = {
            "fifo": _related_by_internal_id["helix_fifo"],
            "eoq": _related_by_internal_id["helix_eoq"],
            "safety-stock": _related_by_internal_id["helix_safety_stock"],
            "lifo": [],
            "holding-cost": [],
        }
        return relations.get(concept_id, [])

    store.get_related_concepts_with_types.side_effect = mock_related

    # New direct methods (keyed by internal Helix ID)
    store.count_concept_edges_direct.side_effect = lambda iid: len(
        _related_by_internal_id.get(iid, [])
    )
    store.get_related_concepts_with_types_direct.side_effect = lambda iid: (
        _related_by_internal_id.get(iid, [])
    )

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


# ---------------------------------------------------------------------------
# Tests: _slugify edge cases
# ---------------------------------------------------------------------------


class TestSlugifyEdgeCases:
    def test_empty_string_returns_unnamed(self):
        assert _slugify("") == "unnamed"

    def test_whitespace_only_returns_unnamed(self):
        assert _slugify("   ") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        assert _slugify("!!!@@@") == "unnamed"


# ---------------------------------------------------------------------------
# Tests: _safe_filename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_normal_doc_id(self):
        assert _safe_filename("my_book_123") == "my_book_123"

    def test_path_traversal_stripped(self):
        result = _safe_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_special_chars_replaced(self):
        result = _safe_filename("book/with spaces & stuff")
        assert "/" not in result
        assert " " not in result

    def test_truncation(self):
        result = _safe_filename("a" * 200)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# Tests: provider/model extraction
# ---------------------------------------------------------------------------


class TestProviderModel:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_provider_extracted_from_slash_format(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="anthropic/claude-3-haiku")
        result = distiller._assemble_config(
            doc_id="test",
            doc_title="Test",
            identity=mock_identity,
            examples=[],
            ranked_concepts=[],
            all_concepts=[],
            chapter_summaries=[],
            prerequisites=[],
        )
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-3-haiku"

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_default_provider_when_no_slash(self, mock_instructor, mock_store, mock_identity):
        distiller = TextbookDistiller(mock_store, provider="gpt-4o-mini")
        result = distiller._assemble_config(
            doc_id="test",
            doc_title="Test",
            identity=mock_identity,
            examples=[],
            ranked_concepts=[],
            all_concepts=[],
            chapter_summaries=[],
            prerequisites=[],
        )
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tests: distill() entry point
# ---------------------------------------------------------------------------


class TestDistillEntryPoint:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_raises_on_missing_summary(self, mock_instructor, mock_store):
        """distill() should raise ValueError when no document summary exists."""
        # Override to return empty for document level
        mock_store.get_document_summaries.side_effect = lambda doc_id, level=None: []

        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        with pytest.raises(ValueError, match="No document-level summary"):
            distiller.distill("test_book")

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", False)
    def test_raises_on_missing_instructor(self, mock_store):
        """distill() should raise ImportError when instructor is missing."""
        # Need a store with valid data so we get past data gathering
        mock_store.get_document_summaries.side_effect = lambda doc_id, level=None: {
            "document": [{"summary_id": "s", "content": "test", "key_points": "[]"}],
            "chapter": [],
        }.get(level, [])

        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        with pytest.raises(ImportError, match="instructor is not installed"):
            distiller.distill("test_book")


# ---------------------------------------------------------------------------
# Tests: _rank_concepts two-phase ranking
# ---------------------------------------------------------------------------


class TestRankConceptsTwoPhase:
    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_related_data_cached_on_ranked_concepts(self, mock_instructor, mock_store):
        """_rank_concepts should store _related so _build_prerequisites can reuse it."""
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)

        # All ranked concepts should have _related key
        for c in ranked:
            assert "_related" in c
            assert isinstance(c["_related"], list)

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_build_prerequisites_uses_cached_related(self, mock_instructor, mock_store):
        """_build_prerequisites should not call any store methods."""
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=5)

        # Reset call counts after ranking
        mock_store.count_concept_edges_direct.reset_mock()
        mock_store.get_related_concepts_with_types_direct.reset_mock()

        distiller._build_prerequisites(ranked)

        # Should not have made any additional calls
        mock_store.count_concept_edges_direct.assert_not_called()
        mock_store.get_related_concepts_with_types_direct.assert_not_called()

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_phase1_counts_all_concepts(self, mock_instructor, mock_store):
        """Phase 1 should call count_concept_edges_direct for all concepts."""
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        distiller._rank_concepts(concepts, top_k=2)

        # count_concept_edges_direct called for all 5 concepts
        assert mock_store.count_concept_edges_direct.call_count == 5

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_phase2_fetches_only_top_k(self, mock_instructor, mock_store):
        """Phase 2 should call get_related_concepts_with_types_direct only for top_k."""
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=2)

        # Only 2 full fetches (top_k=2), not 5
        assert mock_store.get_related_concepts_with_types_direct.call_count == 2
        assert len(ranked) == 2

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_ranked_by_connection_count(self, mock_instructor, mock_store):
        """Concepts should be ranked by connection count descending."""
        distiller = TextbookDistiller(mock_store, provider="openai/gp-4o-mini")
        concepts = mock_store.list_concepts.return_value
        ranked = distiller._rank_concepts(concepts, top_k=3)

        counts = [c["_connection_count"] for c in ranked]
        assert counts == sorted(counts, reverse=True)
        # FIFO and EOQ have 2 connections each, Safety Stock has 1
        assert ranked[0]["_connection_count"] >= ranked[-1]["_connection_count"]

    @patch("src.distiller.distiller.INSTRUCTOR_AVAILABLE", True)
    @patch("src.distiller.distiller.instructor")
    def test_concepts_without_internal_id_skipped(self, mock_instructor, mock_store):
        """Concepts without 'id' field should get count 0, not raise."""
        distiller = TextbookDistiller(mock_store, provider="openai/gpt-4o-mini")
        concepts = [
            {"concept_id": "no-id", "name": "No ID Concept"},  # Missing "id"
            *mock_store.list_concepts.return_value,
        ]
        ranked = distiller._rank_concepts(concepts, top_k=3)

        # Should still produce results without error
        assert len(ranked) == 3
