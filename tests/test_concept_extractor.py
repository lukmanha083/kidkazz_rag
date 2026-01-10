"""Tests for concept extraction module (TDD - tests written first)."""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Tests for Pydantic Models
# ============================================================================


class TestConceptType:
    """Tests for ConceptType enum."""

    def test_has_term_type(self):
        """Should have TERM type for vocabulary concepts."""
        from src.chunker.concept_extractor import ConceptType

        assert ConceptType.TERM == "term"

    def test_has_method_type(self):
        """Should have METHOD type for techniques."""
        from src.chunker.concept_extractor import ConceptType

        assert ConceptType.METHOD == "method"

    def test_has_principle_type(self):
        """Should have PRINCIPLE type for rules."""
        from src.chunker.concept_extractor import ConceptType

        assert ConceptType.PRINCIPLE == "principle"

    def test_has_formula_type(self):
        """Should have FORMULA type for calculations."""
        from src.chunker.concept_extractor import ConceptType

        assert ConceptType.FORMULA == "formula"

    def test_has_account_type(self):
        """Should have ACCOUNT type for ledger accounts."""
        from src.chunker.concept_extractor import ConceptType

        assert ConceptType.ACCOUNT == "account"

    def test_is_string_enum(self):
        """Should be a string enum for JSON serialization."""
        from src.chunker.concept_extractor import ConceptType

        assert isinstance(ConceptType.TERM.value, str)


class TestExtractedConcept:
    """Tests for ExtractedConcept Pydantic model."""

    def test_create_basic_concept(self):
        """Should create concept with required fields."""
        from src.chunker.concept_extractor import ConceptType, ExtractedConcept

        concept = ExtractedConcept(
            name="Cost of Goods Sold",
            concept_type=ConceptType.FORMULA,
            definition="The direct costs attributable to goods sold.",
        )
        assert concept.name == "Cost of Goods Sold"
        assert concept.concept_type == ConceptType.FORMULA
        assert concept.definition == "The direct costs attributable to goods sold."

    def test_aliases_default_empty(self):
        """Should have empty aliases by default."""
        from src.chunker.concept_extractor import ConceptType, ExtractedConcept

        concept = ExtractedConcept(
            name="FIFO",
            concept_type=ConceptType.METHOD,
            definition="First-In, First-Out inventory method.",
        )
        assert concept.aliases == []

    def test_create_with_aliases(self):
        """Should accept aliases list."""
        from src.chunker.concept_extractor import ConceptType, ExtractedConcept

        concept = ExtractedConcept(
            name="Cost of Goods Sold",
            concept_type=ConceptType.FORMULA,
            definition="Direct costs of goods sold.",
            aliases=["COGS", "cost of sales"],
        )
        assert concept.aliases == ["COGS", "cost of sales"]

    def test_validates_required_name(self):
        """Should require name field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ConceptType, ExtractedConcept

        with pytest.raises(ValidationError):
            ExtractedConcept(
                concept_type=ConceptType.TERM,
                definition="Some definition",
            )

    def test_validates_required_concept_type(self):
        """Should require concept_type field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ExtractedConcept

        with pytest.raises(ValidationError):
            ExtractedConcept(
                name="Test",
                definition="Some definition",
            )

    def test_validates_required_definition(self):
        """Should require definition field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ConceptType, ExtractedConcept

        with pytest.raises(ValidationError):
            ExtractedConcept(
                name="Test",
                concept_type=ConceptType.TERM,
            )


class TestConceptRelation:
    """Tests for ConceptRelation Pydantic model."""

    def test_create_relation(self):
        """Should create relation with all fields."""
        from src.chunker.concept_extractor import ConceptRelation

        relation = ConceptRelation(
            from_concept="COGS",
            to_concept="FIFO",
            relation_type="uses",
        )
        assert relation.from_concept == "COGS"
        assert relation.to_concept == "FIFO"
        assert relation.relation_type == "uses"

    def test_validates_required_from_concept(self):
        """Should require from_concept field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ConceptRelation

        with pytest.raises(ValidationError):
            ConceptRelation(
                to_concept="FIFO",
                relation_type="uses",
            )

    def test_validates_required_to_concept(self):
        """Should require to_concept field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ConceptRelation

        with pytest.raises(ValidationError):
            ConceptRelation(
                from_concept="COGS",
                relation_type="uses",
            )

    def test_validates_required_relation_type(self):
        """Should require relation_type field."""
        from pydantic import ValidationError

        from src.chunker.concept_extractor import ConceptRelation

        with pytest.raises(ValidationError):
            ConceptRelation(
                from_concept="COGS",
                to_concept="FIFO",
            )

    def test_accepts_various_relation_types(self):
        """Should accept various relation types."""
        from src.chunker.concept_extractor import ConceptRelation

        relation_types = [
            "uses",
            "requires",
            "calculated_from",
            "component_of",
            "recorded_in",
            "supersedes",
        ]
        for rel_type in relation_types:
            relation = ConceptRelation(
                from_concept="A",
                to_concept="B",
                relation_type=rel_type,
            )
            assert relation.relation_type == rel_type


class TestChunkExtraction:
    """Tests for ChunkExtraction Pydantic model."""

    def test_create_empty_extraction(self):
        """Should create extraction with empty defaults."""
        from src.chunker.concept_extractor import ChunkExtraction

        extraction = ChunkExtraction()
        assert extraction.defined_concepts == []
        assert extraction.mentioned_concepts == []
        assert extraction.relationships == []

    def test_create_with_defined_concepts(self):
        """Should accept list of defined concepts."""
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptType,
            ExtractedConcept,
        )

        concept = ExtractedConcept(
            name="FIFO",
            concept_type=ConceptType.METHOD,
            definition="First-In, First-Out method.",
        )
        extraction = ChunkExtraction(defined_concepts=[concept])
        assert len(extraction.defined_concepts) == 1
        assert extraction.defined_concepts[0].name == "FIFO"

    def test_create_with_mentioned_concepts(self):
        """Should accept list of mentioned concept names."""
        from src.chunker.concept_extractor import ChunkExtraction

        extraction = ChunkExtraction(mentioned_concepts=["Inventory", "Balance Sheet"])
        assert extraction.mentioned_concepts == ["Inventory", "Balance Sheet"]

    def test_create_with_relationships(self):
        """Should accept list of relationships."""
        from src.chunker.concept_extractor import ChunkExtraction, ConceptRelation

        relation = ConceptRelation(
            from_concept="COGS",
            to_concept="FIFO",
            relation_type="uses",
        )
        extraction = ChunkExtraction(relationships=[relation])
        assert len(extraction.relationships) == 1
        assert extraction.relationships[0].from_concept == "COGS"

    def test_create_full_extraction(self):
        """Should create extraction with all fields populated."""
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptRelation,
            ConceptType,
            ExtractedConcept,
        )

        concept = ExtractedConcept(
            name="COGS",
            concept_type=ConceptType.FORMULA,
            definition="Cost of Goods Sold.",
            aliases=["Cost of Sales"],
        )
        relation = ConceptRelation(
            from_concept="COGS",
            to_concept="Inventory",
            relation_type="calculated_from",
        )
        extraction = ChunkExtraction(
            defined_concepts=[concept],
            mentioned_concepts=["Beginning Inventory", "Ending Inventory"],
            relationships=[relation],
        )

        assert len(extraction.defined_concepts) == 1
        assert len(extraction.mentioned_concepts) == 2
        assert len(extraction.relationships) == 1


# ============================================================================
# Tests for slugify function
# ============================================================================


class TestSlugify:
    """Tests for slugify utility function."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        from src.chunker.concept_extractor import slugify

        assert slugify("Cost of Goods Sold") == "cost-of-goods-sold"

    def test_replace_spaces(self):
        """Should replace spaces with hyphens."""
        from src.chunker.concept_extractor import slugify

        assert slugify("Hello World") == "hello-world"

    def test_remove_special_chars(self):
        """Should remove special characters."""
        from src.chunker.concept_extractor import slugify

        assert slugify("FIFO (First-In, First-Out)") == "fifo-first-in-first-out"

    def test_strip_whitespace(self):
        """Should strip leading/trailing whitespace."""
        from src.chunker.concept_extractor import slugify

        assert slugify("  Test Concept  ") == "test-concept"

    def test_collapse_multiple_hyphens(self):
        """Should collapse multiple hyphens to single."""
        from src.chunker.concept_extractor import slugify

        result = slugify("Test   Multiple   Spaces")
        assert "--" not in result

    def test_empty_string(self):
        """Should handle empty string."""
        from src.chunker.concept_extractor import slugify

        assert slugify("") == ""

    def test_already_slug(self):
        """Should not modify already slugified string."""
        from src.chunker.concept_extractor import slugify

        assert slugify("already-a-slug") == "already-a-slug"


# ============================================================================
# Tests for ConceptExtractor class
# ============================================================================


class TestConceptExtractorInit:
    """Tests for ConceptExtractor initialization."""

    def test_default_provider(self):
        """Should use default Anthropic provider."""
        import src.chunker.concept_extractor as extractor_module

        mock_instructor = MagicMock()
        mock_instructor.from_provider.return_value = MagicMock()

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                mock_instructor.from_provider.assert_called_once()
                call_args = mock_instructor.from_provider.call_args[0][0]
                assert "anthropic" in call_args.lower() or "claude" in call_args.lower()

    def test_custom_provider(self):
        """Should accept custom provider."""
        import src.chunker.concept_extractor as extractor_module

        mock_instructor = MagicMock()
        mock_instructor.from_provider.return_value = MagicMock()

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor(provider="openai/gpt-4")
                mock_instructor.from_provider.assert_called_with("openai/gpt-4")

    def test_default_max_retries(self):
        """Should have default max_retries of 2."""
        import src.chunker.concept_extractor as extractor_module

        mock_instructor = MagicMock()
        mock_instructor.from_provider.return_value = MagicMock()

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                assert extractor.max_retries == 2

    def test_custom_max_retries(self):
        """Should accept custom max_retries."""
        import src.chunker.concept_extractor as extractor_module

        mock_instructor = MagicMock()
        mock_instructor.from_provider.return_value = MagicMock()

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor(max_retries=5)
                assert extractor.max_retries == 5

    def test_raises_import_error_when_instructor_not_available(self):
        """Should raise ImportError when instructor not installed."""
        import src.chunker.concept_extractor as extractor_module

        with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", False):
            with pytest.raises(ImportError) as exc_info:
                extractor_module.ConceptExtractor()
            assert "instructor is not installed" in str(exc_info.value)


class TestConceptExtractorExtractFromChunk:
    """Tests for ConceptExtractor.extract_from_chunk method."""

    def test_returns_chunk_extraction(self):
        """Should return ChunkExtraction object."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptType,
            ExtractedConcept,
        )

        mock_extraction = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="COGS",
                    concept_type=ConceptType.FORMULA,
                    definition="Cost of Goods Sold.",
                )
            ]
        )

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = mock_extraction
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                result = extractor.extract_from_chunk(
                    content="COGS is Cost of Goods Sold...",
                    section_path=["Chapter 5", "COGS"],
                    document_title="Inventory Textbook",
                )

                assert isinstance(result, ChunkExtraction)
                assert len(result.defined_concepts) == 1

    def test_passes_content_to_llm(self):
        """Should pass content in message to LLM."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                extractor.extract_from_chunk(
                    content="Test content about FIFO method",
                    section_path=["Chapter 1"],
                    document_title="Test Doc",
                )

                # Check that create was called with messages containing content
                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                assert "Test content about FIFO method" in user_message["content"]

    def test_includes_section_path_in_context(self):
        """Should include section path in context."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                extractor.extract_from_chunk(
                    content="Test content",
                    section_path=["Chapter 5", "Inventory", "COGS Methods"],
                    document_title="Test Doc",
                )

                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                assert "Chapter 5" in user_message["content"]
                assert "COGS Methods" in user_message["content"]

    def test_includes_existing_concepts(self):
        """Should include existing concepts to avoid duplicates."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                extractor.extract_from_chunk(
                    content="Test content",
                    section_path=["Chapter 1"],
                    document_title="Test Doc",
                    existing_concepts=["FIFO", "LIFO", "Inventory"],
                )

                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                system_message = next(m for m in messages if m["role"] == "system")
                assert "FIFO" in system_message["content"]
                assert "LIFO" in system_message["content"]

    def test_handles_extraction_failure(self):
        """Should return empty extraction on failure."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.side_effect = Exception("API Error")
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                result = extractor.extract_from_chunk(
                    content="Test content",
                    section_path=["Chapter 1"],
                    document_title="Test Doc",
                )

                assert isinstance(result, ChunkExtraction)
                assert result.defined_concepts == []
                assert result.relationships == []

    def test_uses_max_retries(self):
        """Should pass max_retries to client.create."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor(max_retries=3)
                extractor.extract_from_chunk(
                    content="Test",
                    section_path=[],
                    document_title="Doc",
                )

                call_kwargs = mock_client.create.call_args[1]
                assert call_kwargs["max_retries"] == 3


class TestConceptExtractorExtractFromChunks:
    """Tests for ConceptExtractor.extract_from_chunks method."""

    def test_returns_concepts_and_relations(self):
        """Should return tuple of (concepts, relations)."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptRelation,
            ConceptType,
            ExtractedConcept,
        )

        mock_extraction = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="FIFO",
                    concept_type=ConceptType.METHOD,
                    definition="First-In, First-Out.",
                )
            ],
            relationships=[
                ConceptRelation(
                    from_concept="COGS",
                    to_concept="FIFO",
                    relation_type="uses",
                )
            ],
        )

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = mock_extraction
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [{"content": "Test", "section_path": ["Ch1"]}]
                concepts, relations = extractor.extract_from_chunks(chunks, "Test Doc")

                assert isinstance(concepts, list)
                assert isinstance(relations, list)
                assert len(concepts) == 1
                assert len(relations) == 1

    def test_deduplicates_concepts_by_name(self):
        """Should deduplicate concepts with same name (case-insensitive)."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptType,
            ExtractedConcept,
        )

        extraction1 = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="FIFO",
                    concept_type=ConceptType.METHOD,
                    definition="First definition.",
                )
            ]
        )
        extraction2 = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="fifo",  # Same concept, different case
                    concept_type=ConceptType.METHOD,
                    definition="Second definition.",
                    aliases=["First-In First-Out"],
                )
            ]
        )

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.side_effect = [extraction1, extraction2]
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [
                    {"content": "Chunk 1", "section_path": ["Ch1"]},
                    {"content": "Chunk 2", "section_path": ["Ch2"]},
                ]
                concepts, _ = extractor.extract_from_chunks(chunks, "Test Doc")

                # Should only have 1 concept (deduplicated)
                assert len(concepts) == 1

    def test_merges_aliases_on_duplicate(self):
        """Should merge aliases when concepts are duplicated."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptType,
            ExtractedConcept,
        )

        extraction1 = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="FIFO",
                    concept_type=ConceptType.METHOD,
                    definition="First definition.",
                    aliases=["First-In First-Out"],
                )
            ]
        )
        extraction2 = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="FIFO",
                    concept_type=ConceptType.METHOD,
                    definition="Second definition.",
                    aliases=["FIFO Method"],
                )
            ]
        )

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.side_effect = [extraction1, extraction2]
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [
                    {"content": "Chunk 1", "section_path": ["Ch1"]},
                    {"content": "Chunk 2", "section_path": ["Ch2"]},
                ]
                concepts, _ = extractor.extract_from_chunks(chunks, "Test Doc")

                # Should have merged aliases
                assert len(concepts) == 1
                assert "First-In First-Out" in concepts[0].aliases
                assert "FIFO Method" in concepts[0].aliases

    def test_accumulates_relations_from_all_chunks(self):
        """Should accumulate relationships from all chunks."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptRelation,
        )

        extraction1 = ChunkExtraction(
            relationships=[
                ConceptRelation(from_concept="A", to_concept="B", relation_type="uses")
            ]
        )
        extraction2 = ChunkExtraction(
            relationships=[
                ConceptRelation(
                    from_concept="B", to_concept="C", relation_type="requires"
                )
            ]
        )

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.side_effect = [extraction1, extraction2]
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [
                    {"content": "Chunk 1", "section_path": []},
                    {"content": "Chunk 2", "section_path": []},
                ]
                _, relations = extractor.extract_from_chunks(chunks, "Test Doc")

                assert len(relations) == 2
                assert any(r.from_concept == "A" for r in relations)
                assert any(r.from_concept == "B" for r in relations)

    def test_passes_known_concepts_to_subsequent_chunks(self):
        """Should pass previously found concepts to avoid redefinition."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import (
            ChunkExtraction,
            ConceptType,
            ExtractedConcept,
        )

        extraction1 = ChunkExtraction(
            defined_concepts=[
                ExtractedConcept(
                    name="FIFO",
                    concept_type=ConceptType.METHOD,
                    definition="First-In First-Out.",
                )
            ]
        )
        extraction2 = ChunkExtraction()  # Second chunk finds nothing new

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.side_effect = [extraction1, extraction2]
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [
                    {"content": "Chunk 1", "section_path": []},
                    {"content": "Chunk 2", "section_path": []},
                ]
                extractor.extract_from_chunks(chunks, "Test Doc")

                # Second call should have "FIFO" in existing_concepts context
                second_call = mock_client.create.call_args_list[1]
                messages = second_call[1]["messages"]
                system_message = next(m for m in messages if m["role"] == "system")
                assert "FIFO" in system_message["content"]

    def test_handles_empty_chunks_list(self):
        """Should handle empty chunks list."""
        import src.chunker.concept_extractor as extractor_module

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                concepts, relations = extractor.extract_from_chunks([], "Test Doc")

                assert concepts == []
                assert relations == []
                mock_client.create.assert_not_called()

    def test_handles_chunk_without_section_path(self):
        """Should handle chunks without section_path key."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                chunks = [{"content": "Test content"}]  # No section_path
                concepts, relations = extractor.extract_from_chunks(chunks, "Test Doc")

                # Should not raise, should use empty section_path
                assert concepts == []
                assert relations == []
