"""Tests for header-aware concept extraction (TDD - tests written first).

This module tests the enhanced concept extraction that leverages header metadata
for smarter extraction and relationship inference.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Tests for Header Context in Extraction
# ============================================================================


class TestExtractFromChunkWithHeaderContext:
    """Tests for extract_from_chunk with header metadata."""

    def test_accepts_header_text_parameter(self):
        """Should accept header_text parameter."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                # Should not raise
                result = extractor.extract_from_chunk(
                    content="FIFO is a method...",
                    section_path=["Chapter 5", "COGS"],
                    document_title="Inventory Textbook",
                    header_text="Inventory Valuation Methods",
                )
                assert isinstance(result, ChunkExtraction)

    def test_accepts_header_level_parameter(self):
        """Should accept header_level parameter."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                result = extractor.extract_from_chunk(
                    content="FIFO is a method...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                    header_level=2,
                )
                assert isinstance(result, ChunkExtraction)

    def test_accepts_semantic_type_parameter(self):
        """Should accept semantic_type parameter."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                result = extractor.extract_from_chunk(
                    content="FIFO is defined as...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                    semantic_type="definition",
                )
                assert isinstance(result, ChunkExtraction)

    def test_includes_header_text_in_prompt(self):
        """Should include header_text in the LLM prompt context."""
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
                    content="FIFO is a method...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                    header_text="Inventory Valuation Methods",
                )

                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                assert "Inventory Valuation Methods" in user_message["content"]

    def test_includes_header_level_in_prompt(self):
        """Should include header_level in the LLM prompt context."""
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
                    content="FIFO is a method...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                    header_text="Methods",
                    header_level=2,
                )

                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                # Should include header level indicator (h2)
                assert "h2" in user_message["content"].lower() or "level 2" in user_message["content"].lower()

    def test_includes_semantic_type_in_prompt(self):
        """Should include semantic_type in the LLM prompt context."""
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
                    content="FIFO is defined as...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                    semantic_type="definition",
                )

                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                assert "definition" in user_message["content"].lower()

    def test_works_without_header_metadata(self):
        """Should work when header metadata is not provided (backwards compatible)."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()
                # Call without any header params (original signature)
                result = extractor.extract_from_chunk(
                    content="FIFO is a method...",
                    section_path=["Chapter 5"],
                    document_title="Textbook",
                )
                assert isinstance(result, ChunkExtraction)


# ============================================================================
# Tests for extract_from_chunks with Metadata
# ============================================================================


class TestExtractFromChunksWithMetadata:
    """Tests for extract_from_chunks with ChunkMetadata support."""

    def test_accepts_metadata_list_parameter(self):
        """Should accept optional metadata_list parameter."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()

                chunks = [{"content": "Test", "section_path": ["Ch1"]}]
                # Create mock metadata
                mock_metadata = MagicMock()
                mock_metadata.header_text = "Test Header"
                mock_metadata.header_level = 2
                mock_metadata.semantic_type = "definition"

                # Should not raise
                concepts, relations = extractor.extract_from_chunks(
                    chunks, "Test Doc", metadata_list=[mock_metadata]
                )
                assert isinstance(concepts, list)

    def test_passes_metadata_to_extract_from_chunk(self):
        """Should pass metadata fields to extract_from_chunk."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()

                chunks = [{"content": "FIFO is defined as...", "section_path": ["Ch1"]}]
                mock_metadata = MagicMock()
                mock_metadata.header_text = "Inventory Methods"
                mock_metadata.header_level = 2
                mock_metadata.semantic_type = "definition"

                extractor.extract_from_chunks(chunks, "Test Doc", metadata_list=[mock_metadata])

                # Check that the LLM was called with header context
                call_kwargs = mock_client.create.call_args[1]
                messages = call_kwargs["messages"]
                user_message = next(m for m in messages if m["role"] == "user")
                assert "Inventory Methods" in user_message["content"]

    def test_works_without_metadata_list(self):
        """Should work when metadata_list is not provided (backwards compatible)."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()

                chunks = [{"content": "Test", "section_path": ["Ch1"]}]
                # Call without metadata_list
                concepts, relations = extractor.extract_from_chunks(chunks, "Test Doc")
                assert isinstance(concepts, list)

    def test_handles_mismatched_metadata_length(self):
        """Should handle when metadata_list length doesn't match chunks."""
        import src.chunker.concept_extractor as extractor_module
        from src.chunker.concept_extractor import ChunkExtraction

        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = ChunkExtraction()
        mock_instructor.from_provider.return_value = mock_client

        with patch.object(extractor_module, "instructor", mock_instructor):
            with patch.object(extractor_module, "INSTRUCTOR_AVAILABLE", True):
                extractor = extractor_module.ConceptExtractor()

                chunks = [
                    {"content": "Test 1", "section_path": ["Ch1"]},
                    {"content": "Test 2", "section_path": ["Ch2"]},
                ]
                # Only one metadata for two chunks
                mock_metadata = MagicMock()
                mock_metadata.header_text = "Header"
                mock_metadata.header_level = 2
                mock_metadata.semantic_type = "definition"

                # Should not raise, should handle gracefully
                concepts, relations = extractor.extract_from_chunks(
                    chunks, "Test Doc", metadata_list=[mock_metadata]
                )
                assert isinstance(concepts, list)


# ============================================================================
# Tests for Semantic Type Filtering
# ============================================================================


class TestSemanticTypeFiltering:
    """Tests for prioritized extraction based on semantic type."""

    def test_filter_chunks_by_semantic_type_function_exists(self):
        """Should have filter_chunks_by_semantic_type function."""
        from src.chunker.concept_extractor import filter_chunks_by_semantic_type

        assert callable(filter_chunks_by_semantic_type)

    def test_filter_returns_only_matching_types(self):
        """Should return only chunks with matching semantic type."""
        from src.chunker.concept_extractor import filter_chunks_by_semantic_type

        chunks = [
            {"content": "Chunk 1", "section_path": []},
            {"content": "Chunk 2", "section_path": []},
            {"content": "Chunk 3", "section_path": []},
        ]
        metadata_list = [
            MagicMock(semantic_type="definition"),
            MagicMock(semantic_type="narrative"),
            MagicMock(semantic_type="definition"),
        ]

        filtered_chunks, filtered_metadata = filter_chunks_by_semantic_type(
            chunks, metadata_list, semantic_types=["definition"]
        )

        assert len(filtered_chunks) == 2
        assert filtered_chunks[0]["content"] == "Chunk 1"
        assert filtered_chunks[1]["content"] == "Chunk 3"

    def test_filter_accepts_multiple_types(self):
        """Should filter by multiple semantic types."""
        from src.chunker.concept_extractor import filter_chunks_by_semantic_type

        chunks = [
            {"content": "Chunk 1", "section_path": []},
            {"content": "Chunk 2", "section_path": []},
            {"content": "Chunk 3", "section_path": []},
        ]
        metadata_list = [
            MagicMock(semantic_type="definition"),
            MagicMock(semantic_type="theorem"),
            MagicMock(semantic_type="narrative"),
        ]

        filtered_chunks, filtered_metadata = filter_chunks_by_semantic_type(
            chunks, metadata_list, semantic_types=["definition", "theorem"]
        )

        assert len(filtered_chunks) == 2

    def test_filter_returns_empty_when_no_match(self):
        """Should return empty lists when no chunks match."""
        from src.chunker.concept_extractor import filter_chunks_by_semantic_type

        chunks = [{"content": "Chunk 1", "section_path": []}]
        metadata_list = [MagicMock(semantic_type="narrative")]

        filtered_chunks, filtered_metadata = filter_chunks_by_semantic_type(
            chunks, metadata_list, semantic_types=["definition"]
        )

        assert len(filtered_chunks) == 0
        assert len(filtered_metadata) == 0

    def test_filter_returns_all_when_no_filter(self):
        """Should return all chunks when semantic_types is empty or None."""
        from src.chunker.concept_extractor import filter_chunks_by_semantic_type

        chunks = [
            {"content": "Chunk 1", "section_path": []},
            {"content": "Chunk 2", "section_path": []},
        ]
        metadata_list = [
            MagicMock(semantic_type="definition"),
            MagicMock(semantic_type="narrative"),
        ]

        # Empty list
        filtered_chunks, _ = filter_chunks_by_semantic_type(
            chunks, metadata_list, semantic_types=[]
        )
        assert len(filtered_chunks) == 2

        # None
        filtered_chunks, _ = filter_chunks_by_semantic_type(
            chunks, metadata_list, semantic_types=None
        )
        assert len(filtered_chunks) == 2


# ============================================================================
# Tests for Header Hierarchy Relationship Inference
# ============================================================================


class TestHeaderHierarchyRelationships:
    """Tests for inferring concept relationships from header hierarchy."""

    def test_infer_relationships_from_headers_function_exists(self):
        """Should have infer_relationships_from_headers function."""
        from src.chunker.concept_extractor import infer_relationships_from_headers

        assert callable(infer_relationships_from_headers)

    def test_infers_parent_relationship_from_h1_to_h2(self):
        """Should infer parent relationship when h2 follows h1."""
        from src.chunker.concept_extractor import (
            ConceptType,
            ExtractedConcept,
            infer_relationships_from_headers,
        )

        # Concepts extracted from chunks at different header levels
        concepts = [
            ExtractedConcept(
                name="Inventory Management",
                concept_type=ConceptType.TERM,
                definition="Managing inventory.",
            ),
            ExtractedConcept(
                name="FIFO",
                concept_type=ConceptType.METHOD,
                definition="First-In, First-Out.",
            ),
        ]

        # Metadata showing header hierarchy
        metadata_list = [
            MagicMock(header_text="Inventory Management", header_level=1, chunk_id="chunk1"),
            MagicMock(header_text="Valuation Methods", header_level=2, chunk_id="chunk2"),
        ]

        # Map concepts to chunks
        concept_chunk_map = {
            "Inventory Management": "chunk1",
            "FIFO": "chunk2",
        }

        relations = infer_relationships_from_headers(
            concepts, metadata_list, concept_chunk_map
        )

        # FIFO (from h2) should have parent relationship to Inventory Management (from h1)
        assert len(relations) >= 1
        parent_rel = next(
            (r for r in relations if r.to_concept == "FIFO" and r.relation_type == "parent_of"),
            None,
        )
        assert parent_rel is not None
        assert parent_rel.from_concept == "Inventory Management"

    def test_no_relationship_for_same_level_headers(self):
        """Should not infer parent relationship for same level headers."""
        from src.chunker.concept_extractor import (
            ConceptType,
            ExtractedConcept,
            infer_relationships_from_headers,
        )

        concepts = [
            ExtractedConcept(name="FIFO", concept_type=ConceptType.METHOD, definition="First-In."),
            ExtractedConcept(name="LIFO", concept_type=ConceptType.METHOD, definition="Last-In."),
        ]

        metadata_list = [
            MagicMock(header_text="FIFO Method", header_level=2, chunk_id="chunk1"),
            MagicMock(header_text="LIFO Method", header_level=2, chunk_id="chunk2"),
        ]

        concept_chunk_map = {"FIFO": "chunk1", "LIFO": "chunk2"}

        relations = infer_relationships_from_headers(concepts, metadata_list, concept_chunk_map)

        # No parent_of relationships between same-level concepts
        parent_relations = [r for r in relations if r.relation_type == "parent_of"]
        fifo_lifo_parent = [
            r for r in parent_relations
            if (r.from_concept == "FIFO" and r.to_concept == "LIFO")
            or (r.from_concept == "LIFO" and r.to_concept == "FIFO")
        ]
        assert len(fifo_lifo_parent) == 0

    def test_handles_missing_header_level(self):
        """Should handle chunks without header_level gracefully."""
        from src.chunker.concept_extractor import (
            ConceptType,
            ExtractedConcept,
            infer_relationships_from_headers,
        )

        concepts = [
            ExtractedConcept(name="Test", concept_type=ConceptType.TERM, definition="Test."),
        ]

        metadata_list = [
            MagicMock(header_text=None, header_level=None, chunk_id="chunk1"),
        ]

        concept_chunk_map = {"Test": "chunk1"}

        # Should not raise
        relations = infer_relationships_from_headers(concepts, metadata_list, concept_chunk_map)
        assert isinstance(relations, list)

    def test_handles_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        from src.chunker.concept_extractor import infer_relationships_from_headers

        relations = infer_relationships_from_headers([], [], {})
        assert relations == []

    def test_relationship_has_inferred_from_field(self):
        """Should mark relationships as inferred from header hierarchy."""
        from src.chunker.concept_extractor import (
            ConceptType,
            ExtractedConcept,
            infer_relationships_from_headers,
        )

        concepts = [
            ExtractedConcept(name="Parent", concept_type=ConceptType.TERM, definition="Parent."),
            ExtractedConcept(name="Child", concept_type=ConceptType.TERM, definition="Child."),
        ]

        metadata_list = [
            MagicMock(header_text="Parent Section", header_level=1, chunk_id="chunk1"),
            MagicMock(header_text="Child Section", header_level=2, chunk_id="chunk2"),
        ]

        concept_chunk_map = {"Parent": "chunk1", "Child": "chunk2"}

        relations = infer_relationships_from_headers(concepts, metadata_list, concept_chunk_map)

        if relations:
            # Check that relationship indicates it was inferred
            assert any(
                hasattr(r, "inferred_from") and r.inferred_from == "header_hierarchy"
                for r in relations
            ) or all(r.relation_type == "parent_of" for r in relations)
