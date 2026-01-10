"""Tests for concept-related HelixQL query classes (TDD - tests written first)."""

import json

import pytest


# ============================================================================
# Tests for AddConcept query
# ============================================================================


class TestAddConcept:
    """Tests for AddConcept query."""

    def test_creates_query_payload(self):
        """Should create valid query payload matching HelixQL signature."""
        from src.storage.queries import AddConcept

        query = AddConcept(
            concept_id="cost-of-goods-sold",
            name="Cost of Goods Sold",
            definition="The direct costs attributable to goods sold.",
            concept_type="formula",
            source_documents=["inventory_textbook"],
            aliases=["COGS", "cost of sales"],
        )
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "cost-of-goods-sold"
        assert payload[0]["name"] == "Cost of Goods Sold"
        assert payload[0]["definition"] == "The direct costs attributable to goods sold."
        assert payload[0]["concept_type"] == "formula"

    def test_serializes_lists_as_json(self):
        """Should serialize source_documents and aliases as JSON strings."""
        from src.storage.queries import AddConcept

        query = AddConcept(
            concept_id="fifo",
            name="FIFO",
            definition="First-In, First-Out method.",
            concept_type="method",
            source_documents=["doc1", "doc2"],
            aliases=["First-In First-Out"],
        )
        payload = query.query()

        # Should be JSON strings
        source_docs = json.loads(payload[0]["source_documents"])
        aliases = json.loads(payload[0]["aliases"])
        assert source_docs == ["doc1", "doc2"]
        assert aliases == ["First-In First-Out"]

    def test_handles_empty_lists(self):
        """Should handle empty source_documents and aliases."""
        from src.storage.queries import AddConcept

        query = AddConcept(
            concept_id="test",
            name="Test",
            definition="Test definition.",
            concept_type="term",
            source_documents=[],
            aliases=[],
        )
        payload = query.query()

        source_docs = json.loads(payload[0]["source_documents"])
        aliases = json.loads(payload[0]["aliases"])
        assert source_docs == []
        assert aliases == []

    def test_response_processing(self):
        """Should process response correctly."""
        from src.storage.queries import AddConcept

        query = AddConcept(
            concept_id="test",
            name="Test",
            definition="Test def.",
            concept_type="term",
            source_documents=[],
            aliases=[],
        )
        result = query.response({"status": "ok"})

        assert result.success is True
        assert result.data["concept_id"] == "test"


# ============================================================================
# Tests for GetConceptByName query
# ============================================================================


class TestGetConceptByName:
    """Tests for GetConceptByName query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetConceptByName

        query = GetConceptByName("Cost of Goods Sold")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["name"] == "Cost of Goods Sold"
        assert query.endpoint == "GetConceptByName"

    def test_response_found(self):
        """Should return success when concept found."""
        from src.storage.queries import GetConceptByName

        query = GetConceptByName("FIFO")
        result = query.response([{"name": "FIFO", "concept_type": "method"}])

        assert result.success is True
        assert result.data["node"]["name"] == "FIFO"

    def test_response_not_found(self):
        """Should return failure when no results."""
        from src.storage.queries import GetConceptByName

        query = GetConceptByName("NonExistent")
        result = query.response([])

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_response_none(self):
        """Should return failure when response is None."""
        from src.storage.queries import GetConceptByName

        query = GetConceptByName("Test")
        result = query.response(None)

        assert result.success is False


# ============================================================================
# Tests for GetConceptById query
# ============================================================================


class TestGetConceptById:
    """Tests for GetConceptById query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetConceptById

        query = GetConceptById("cost-of-goods-sold")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "cost-of-goods-sold"
        assert query.endpoint == "GetConceptById"

    def test_response_found(self):
        """Should return success when concept found."""
        from src.storage.queries import GetConceptById

        query = GetConceptById("fifo")
        result = query.response([{"concept_id": "fifo", "name": "FIFO"}])

        assert result.success is True
        assert result.data["node"]["concept_id"] == "fifo"

    def test_response_not_found(self):
        """Should return failure when no results."""
        from src.storage.queries import GetConceptById

        query = GetConceptById("non-existent")
        result = query.response([])

        assert result.success is False
        assert result.error is not None


# ============================================================================
# Tests for ListConcepts query
# ============================================================================


class TestListConcepts:
    """Tests for ListConcepts query."""

    def test_creates_empty_params(self):
        """Should create empty parameters."""
        from src.storage.queries import ListConcepts

        query = ListConcepts()
        payload = query.query()

        assert len(payload) == 1
        assert payload[0] == {}
        assert query.endpoint == "ListConcepts"

    def test_response_with_concepts(self):
        """Should return concepts list from response."""
        from src.storage.queries import ListConcepts

        query = ListConcepts()
        result = query.response([
            {"concept_id": "fifo", "name": "FIFO"},
            {"concept_id": "lifo", "name": "LIFO"},
        ])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import ListConcepts

        query = ListConcepts()
        result = query.response([])

        assert result.success is True
        assert result.data == []

    def test_response_none(self):
        """Should handle None response."""
        from src.storage.queries import ListConcepts

        query = ListConcepts()
        result = query.response(None)

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for ListDocumentConcepts query
# ============================================================================


class TestListDocumentConcepts:
    """Tests for ListDocumentConcepts query."""

    def test_creates_query_payload(self):
        """Should create valid query payload with document_id."""
        from src.storage.queries import ListDocumentConcepts

        query = ListDocumentConcepts("inventory_textbook")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["document_id"] == "inventory_textbook"
        assert query.endpoint == "ListDocumentConcepts"

    def test_response_with_concepts(self):
        """Should return concepts list from response."""
        from src.storage.queries import ListDocumentConcepts

        query = ListDocumentConcepts("doc_1")
        result = query.response([
            {"concept_id": "cogs", "name": "COGS"},
        ])

        assert result.success is True
        assert len(result.data) == 1

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import ListDocumentConcepts

        query = ListDocumentConcepts("doc_1")
        result = query.response([])

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for LinkChunkDefinesConcept query
# ============================================================================


class TestLinkChunkDefinesConcept:
    """Tests for LinkChunkDefinesConcept query."""

    def test_creates_link_payload(self):
        """Should create payload for linking chunk to concept it defines."""
        from src.storage.queries import LinkChunkDefinesConcept

        query = LinkChunkDefinesConcept("chunk_internal_123", "concept_internal_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_internal_123"
        assert payload[0]["concept_id"] == "concept_internal_456"
        assert query.endpoint == "LinkChunkDefinesConcept"

    def test_response_success(self):
        """Should return success for link operation."""
        from src.storage.queries import LinkChunkDefinesConcept

        query = LinkChunkDefinesConcept("c1", "concept1")
        result = query.response({})

        assert result.success is True


# ============================================================================
# Tests for LinkChunkMentionsConcept query
# ============================================================================


class TestLinkChunkMentionsConcept:
    """Tests for LinkChunkMentionsConcept query."""

    def test_creates_link_payload(self):
        """Should create payload for linking chunk to concept it mentions."""
        from src.storage.queries import LinkChunkMentionsConcept

        query = LinkChunkMentionsConcept("chunk_789", "concept_321")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_789"
        assert payload[0]["concept_id"] == "concept_321"
        assert query.endpoint == "LinkChunkMentionsConcept"

    def test_response_success(self):
        """Should return success for link operation."""
        from src.storage.queries import LinkChunkMentionsConcept

        query = LinkChunkMentionsConcept("c1", "concept1")
        result = query.response({})

        assert result.success is True


# ============================================================================
# Tests for LinkConceptRelatesTo query
# ============================================================================


class TestLinkConceptRelatesTo:
    """Tests for LinkConceptRelatesTo query."""

    def test_creates_link_payload(self):
        """Should create payload for linking two concepts."""
        from src.storage.queries import LinkConceptRelatesTo

        query = LinkConceptRelatesTo("concept_from_id", "concept_to_id")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["from_id"] == "concept_from_id"
        assert payload[0]["to_id"] == "concept_to_id"
        assert query.endpoint == "LinkConceptRelatesTo"

    def test_response_success(self):
        """Should return success for link operation."""
        from src.storage.queries import LinkConceptRelatesTo

        query = LinkConceptRelatesTo("from1", "to1")
        result = query.response({})

        assert result.success is True


# ============================================================================
# Tests for GetConceptDefinitionChunks query
# ============================================================================


class TestGetConceptDefinitionChunks:
    """Tests for GetConceptDefinitionChunks query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetConceptDefinitionChunks

        query = GetConceptDefinitionChunks("concept_internal_123")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "concept_internal_123"
        assert query.endpoint == "GetConceptDefinitionChunks"

    def test_response_with_chunks(self):
        """Should return chunks list from response."""
        from src.storage.queries import GetConceptDefinitionChunks

        query = GetConceptDefinitionChunks("concept_1")
        result = query.response([
            {"chunk_id": "chunk1", "content": "Definition..."},
            {"chunk_id": "chunk2", "content": "More detail..."},
        ])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import GetConceptDefinitionChunks

        query = GetConceptDefinitionChunks("concept_1")
        result = query.response([])

        assert result.success is True
        assert result.data == []

    def test_response_none(self):
        """Should handle None response."""
        from src.storage.queries import GetConceptDefinitionChunks

        query = GetConceptDefinitionChunks("concept_1")
        result = query.response(None)

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for GetConceptMentionChunks query
# ============================================================================


class TestGetConceptMentionChunks:
    """Tests for GetConceptMentionChunks query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetConceptMentionChunks

        query = GetConceptMentionChunks("concept_internal_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "concept_internal_456"
        assert query.endpoint == "GetConceptMentionChunks"

    def test_response_with_chunks(self):
        """Should return chunks list from response."""
        from src.storage.queries import GetConceptMentionChunks

        query = GetConceptMentionChunks("concept_1")
        result = query.response([
            {"chunk_id": "chunk1", "content": "Mentions concept..."},
        ])

        assert result.success is True
        assert len(result.data) == 1

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import GetConceptMentionChunks

        query = GetConceptMentionChunks("concept_1")
        result = query.response([])

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for GetRelatedConcepts query
# ============================================================================


class TestGetRelatedConcepts:
    """Tests for GetRelatedConcepts query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetRelatedConcepts

        query = GetRelatedConcepts("concept_internal_789")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "concept_internal_789"
        assert query.endpoint == "GetRelatedConcepts"

    def test_response_with_related(self):
        """Should return related concepts list from response."""
        from src.storage.queries import GetRelatedConcepts

        query = GetRelatedConcepts("cogs_concept")
        result = query.response([
            {"concept_id": "fifo", "name": "FIFO"},
            {"concept_id": "lifo", "name": "LIFO"},
        ])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import GetRelatedConcepts

        query = GetRelatedConcepts("isolated_concept")
        result = query.response([])

        assert result.success is True
        assert result.data == []

    def test_response_none(self):
        """Should handle None response."""
        from src.storage.queries import GetRelatedConcepts

        query = GetRelatedConcepts("concept_1")
        result = query.response(None)

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for GetConceptDependents query
# ============================================================================


class TestGetConceptDependents:
    """Tests for GetConceptDependents query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        from src.storage.queries import GetConceptDependents

        query = GetConceptDependents("concept_internal_123")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "concept_internal_123"
        assert query.endpoint == "GetConceptDependents"

    def test_response_with_dependents(self):
        """Should return dependent concepts list from response."""
        from src.storage.queries import GetConceptDependents

        query = GetConceptDependents("fifo_concept")
        result = query.response([
            {"concept_id": "cogs", "name": "COGS"},
        ])

        assert result.success is True
        assert len(result.data) == 1

    def test_response_empty(self):
        """Should handle empty response."""
        from src.storage.queries import GetConceptDependents

        query = GetConceptDependents("root_concept")
        result = query.response([])

        assert result.success is True
        assert result.data == []


# ============================================================================
# Tests for DropConcept query
# ============================================================================


class TestDropConcept:
    """Tests for DropConcept query."""

    def test_creates_drop_payload(self):
        """Should create payload for HelixQL DropConcept."""
        from src.storage.queries import DropConcept

        query = DropConcept("concept_internal_123")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["concept_id"] == "concept_internal_123"
        assert query.endpoint == "DropConcept"

    def test_response_success(self):
        """Should return success for drop operation."""
        from src.storage.queries import DropConcept

        query = DropConcept("concept_1")
        result = query.response("Removed concept")

        assert result.success is True
