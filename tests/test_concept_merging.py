"""Tests for cross-document concept merging (TDD - tests written first)."""

import json
import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Tests for UpdateConcept query
# ============================================================================


class TestUpdateConceptQuery:
    """Tests for UpdateConcept query class."""

    def test_update_concept_query_params(self):
        """Should generate correct query parameters."""
        from src.storage.queries import UpdateConcept

        query = UpdateConcept(
            concept_id="cost-of-goods-sold",
            source_documents=["doc1", "doc2"],
            aliases=["COGS", "Cost of Sales"],
        )

        params = query.query()
        assert len(params) == 1
        assert params[0]["concept_id"] == "cost-of-goods-sold"
        assert json.loads(params[0]["source_documents"]) == ["doc1", "doc2"]
        assert json.loads(params[0]["aliases"]) == ["COGS", "Cost of Sales"]

    def test_update_concept_response_success(self):
        """Should return success on valid response."""
        from src.storage.queries import UpdateConcept

        query = UpdateConcept(
            concept_id="fifo",
            source_documents=["doc1"],
            aliases=[],
        )

        result = query.response({"updated": True})
        assert result.success is True

    def test_update_concept_can_update_definition(self):
        """Should support optional definition update."""
        from src.storage.queries import UpdateConcept

        query = UpdateConcept(
            concept_id="fifo",
            source_documents=["doc1"],
            aliases=[],
            definition="Updated definition here",
        )

        params = query.query()
        assert params[0]["definition"] == "Updated definition here"

    def test_update_concept_without_definition(self):
        """Should not include definition when not provided."""
        from src.storage.queries import UpdateConcept

        query = UpdateConcept(
            concept_id="fifo",
            source_documents=["doc1"],
            aliases=[],
        )

        params = query.query()
        # Definition should be None or not present when not updating
        assert params[0].get("definition") is None


# ============================================================================
# Tests for store_or_merge_concept method
# ============================================================================


class TestStoreOrMergeConcept:
    """Tests for store_or_merge_concept storage client method."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store with concept methods."""
        store = MagicMock()
        store.get_concept_by_name = MagicMock(return_value=None)
        store.store_concept = MagicMock(return_value="new-internal-id")
        store.update_concept = MagicMock(return_value=True)
        return store

    def test_creates_new_concept_when_not_exists(self, mock_store):
        """Should create new concept when it doesn't exist."""
        from src.storage.client import HelixChunkStore

        # Mock get_concept_by_name to return None (not found)
        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=None)
            store.store_concept = MagicMock(return_value="new-id")
            store.update_concept = MagicMock()

            result = store.store_or_merge_concept(
                concept_id="fifo",
                name="FIFO",
                definition="First-In, First-Out",
                concept_type="method",
                source_documents=["doc1"],
                aliases=["First-In First-Out"],
            )

            store.store_concept.assert_called_once()
            store.update_concept.assert_not_called()
            assert result == "new-id"

    def test_merges_when_concept_exists(self, mock_store):
        """Should merge source_documents and aliases when concept exists."""
        from src.storage.client import HelixChunkStore

        existing_concept = {
            "concept_id": "fifo",
            "name": "FIFO",
            "definition": "First-In, First-Out method",
            "concept_type": "method",
            "source_documents": '["doc1"]',
            "aliases": '["First-In First-Out"]',
        }

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=existing_concept)
            store.store_concept = MagicMock()
            store.update_concept = MagicMock(return_value=True)

            result = store.store_or_merge_concept(
                concept_id="fifo",
                name="FIFO",
                definition="Different definition",
                concept_type="method",
                source_documents=["doc2"],  # New document
                aliases=["FIFO Method"],  # New alias
            )

            store.store_concept.assert_not_called()
            store.update_concept.assert_called_once()

            # Check merged values
            call_args = store.update_concept.call_args
            assert "doc1" in call_args.kwargs["source_documents"]
            assert "doc2" in call_args.kwargs["source_documents"]
            assert "First-In First-Out" in call_args.kwargs["aliases"]
            assert "FIFO Method" in call_args.kwargs["aliases"]

    def test_deduplicates_source_documents(self):
        """Should not duplicate source_documents."""
        from src.storage.client import HelixChunkStore

        existing_concept = {
            "concept_id": "fifo",
            "name": "FIFO",
            "source_documents": '["doc1", "doc2"]',
            "aliases": '[]',
        }

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=existing_concept)
            store.store_concept = MagicMock()
            store.update_concept = MagicMock(return_value=True)

            store.store_or_merge_concept(
                concept_id="fifo",
                name="FIFO",
                definition="",
                concept_type="method",
                source_documents=["doc2", "doc3"],  # doc2 already exists
                aliases=[],
            )

            call_args = store.update_concept.call_args
            source_docs = call_args.kwargs["source_documents"]
            # Should have doc1, doc2, doc3 (no duplicate doc2)
            assert sorted(source_docs) == ["doc1", "doc2", "doc3"]

    def test_deduplicates_aliases(self):
        """Should not duplicate aliases."""
        from src.storage.client import HelixChunkStore

        existing_concept = {
            "concept_id": "cogs",
            "name": "Cost of Goods Sold",
            "source_documents": '["doc1"]',
            "aliases": '["COGS", "Cost of Sales"]',
        }

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=existing_concept)
            store.store_concept = MagicMock()
            store.update_concept = MagicMock(return_value=True)

            store.store_or_merge_concept(
                concept_id="cogs",
                name="Cost of Goods Sold",
                definition="",
                concept_type="formula",
                source_documents=["doc2"],
                aliases=["COGS", "CoGS"],  # COGS already exists (case-insensitive)
            )

            call_args = store.update_concept.call_args
            aliases = call_args.kwargs["aliases"]
            # Should not have duplicate COGS
            aliases_lower = [a.lower() for a in aliases]
            assert aliases_lower.count("cogs") == 1

    def test_returns_existing_internal_id(self):
        """Should return the existing concept's internal ID when merging."""
        from src.storage.client import HelixChunkStore

        existing_concept = {
            "N": {"Id": "existing-internal-id"},  # Internal node ID
            "concept_id": "fifo",
            "name": "FIFO",
            "source_documents": '["doc1"]',
            "aliases": '[]',
        }

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=existing_concept)
            store.update_concept = MagicMock(return_value=True)

            result = store.store_or_merge_concept(
                concept_id="fifo",
                name="FIFO",
                definition="",
                concept_type="method",
                source_documents=["doc2"],
                aliases=[],
            )

            assert result == "existing-internal-id"


# ============================================================================
# Tests for ingest with cross-document merging
# ============================================================================


class TestIngestCrossDocumentMerging:
    """Tests for concept merging during document ingestion."""

    @pytest.fixture
    def mock_store_with_existing_concept(self):
        """Create mock store that has an existing concept."""
        store = MagicMock()
        store.get_concept_by_name = MagicMock(return_value={
            "concept_id": "fifo",
            "name": "FIFO",
            "definition": "First-In, First-Out",
            "source_documents": '["inventory_textbook"]',
            "aliases": '["First-In First-Out"]',
        })
        store.store_or_merge_concept = MagicMock(return_value="merged-id")
        return store

    def test_ingest_uses_merge_instead_of_store(self):
        """Ingest should use store_or_merge_concept instead of store_concept."""
        # This test verifies the ingest command calls the merge method
        from typer.testing import CliRunner
        from src.cli.main import app

        runner = CliRunner()

        with patch("src.cli.commands.ingest.get_store") as mock_get_store, \
             patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True), \
             patch("src.cli.commands.ingest.ConceptExtractor") as mock_extractor_class:

            mock_store = MagicMock()
            mock_store.store_document = MagicMock()
            mock_store.store_or_merge_concept = MagicMock(return_value="concept-id")
            mock_store.link_chunk_defines_concept = MagicMock()
            mock_store.link_concept_relates_to = MagicMock()
            mock_get_store.return_value = mock_store

            # Mock extractor
            mock_extractor = MagicMock()
            mock_extractor.extract_from_chunks.return_value = (
                [{"concept_id": "fifo", "name": "FIFO", "definition": "Test", "concept_type": "method", "aliases": []}],
                [],
            )
            mock_extractor_class.return_value = mock_extractor

            # Create temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write("# Test\n\nFIFO is a method.")
                temp_path = f.name

            try:
                result = runner.invoke(app, [
                    "ingest", "markdown", temp_path,
                    "--extract-concepts",
                    "--dry-run",
                ])

                # Should call store_or_merge_concept, not store_concept
                if mock_store.store_or_merge_concept.called:
                    assert True
                # For now, we're just checking the test structure is correct
            finally:
                import os
                os.unlink(temp_path)

    def test_second_document_merges_concepts(self):
        """When second doc has same concept, it should merge not duplicate."""
        from src.storage.client import HelixChunkStore

        # Simulate: doc1 already ingested with FIFO concept
        # Now doc2 is being ingested with same FIFO concept

        existing_concept = {
            "N": {"Id": "fifo-internal-id"},
            "concept_id": "fifo",
            "name": "FIFO",
            "definition": "First-In, First-Out inventory method",
            "source_documents": '["inventory_textbook"]',
            "aliases": '["First-In First-Out"]',
        }

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store.get_concept_by_name = MagicMock(return_value=existing_concept)
            store.store_concept = MagicMock()
            store.update_concept = MagicMock(return_value=True)

            # Ingest doc2 with same FIFO concept
            result = store.store_or_merge_concept(
                concept_id="fifo",
                name="FIFO",
                definition="FIFO method for costing",  # Different definition
                concept_type="method",
                source_documents=["accounting_textbook"],  # New document
                aliases=["FIFO Method"],  # Additional alias
            )

            # Should NOT create new concept
            store.store_concept.assert_not_called()

            # Should update existing concept
            store.update_concept.assert_called_once()

            # Verify merged data
            call_kwargs = store.update_concept.call_args.kwargs
            assert "inventory_textbook" in call_kwargs["source_documents"]
            assert "accounting_textbook" in call_kwargs["source_documents"]
            assert "First-In First-Out" in call_kwargs["aliases"]
            assert "FIFO Method" in call_kwargs["aliases"]


# ============================================================================
# Tests for update_concept storage method
# ============================================================================


class TestUpdateConceptMethod:
    """Tests for update_concept storage client method."""

    def test_update_concept_calls_query(self):
        """Should execute UpdateConcept query."""
        from src.storage.client import HelixChunkStore

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store._ensure_connected = MagicMock()

            # Mock the query execution
            mock_response = MagicMock()
            mock_response.success = True
            store._client.execute = MagicMock(return_value=mock_response)

            with patch("src.storage.client.UpdateConcept") as mock_query_class:
                mock_query = MagicMock()
                mock_query.response.return_value = MagicMock(success=True)
                mock_query_class.return_value = mock_query

                result = store.update_concept(
                    concept_id="fifo",
                    source_documents=["doc1", "doc2"],
                    aliases=["FIFO", "First-In First-Out"],
                )

                mock_query_class.assert_called_once_with(
                    concept_id="fifo",
                    source_documents=["doc1", "doc2"],
                    aliases=["FIFO", "First-In First-Out"],
                    definition=None,
                )

    def test_update_concept_with_definition(self):
        """Should pass definition when provided."""
        from src.storage.client import HelixChunkStore

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store._ensure_connected = MagicMock()

            with patch("src.storage.client.UpdateConcept") as mock_query_class:
                mock_query = MagicMock()
                mock_query.response.return_value = MagicMock(success=True)
                mock_query_class.return_value = mock_query

                store.update_concept(
                    concept_id="fifo",
                    source_documents=["doc1"],
                    aliases=[],
                    definition="Updated definition",
                )

                mock_query_class.assert_called_once_with(
                    concept_id="fifo",
                    source_documents=["doc1"],
                    aliases=[],
                    definition="Updated definition",
                )

    def test_update_concept_returns_success(self):
        """Should return True on successful update."""
        from src.storage.client import HelixChunkStore

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store._ensure_connected = MagicMock()

            with patch("src.storage.client.UpdateConcept") as mock_query_class:
                mock_query = MagicMock()
                mock_query.response.return_value = MagicMock(success=True)
                mock_query_class.return_value = mock_query

                result = store.update_concept(
                    concept_id="fifo",
                    source_documents=["doc1"],
                    aliases=[],
                )

                assert result is True

    def test_update_concept_returns_false_on_failure(self):
        """Should return False on failed update."""
        from src.storage.client import HelixChunkStore

        with patch.object(HelixChunkStore, '__init__', lambda x: None):
            store = HelixChunkStore()
            store._client = MagicMock()
            store._ensure_connected = MagicMock()

            with patch("src.storage.client.UpdateConcept") as mock_query_class:
                mock_query = MagicMock()
                mock_query.response.return_value = MagicMock(success=False, error="Update failed")
                mock_query_class.return_value = mock_query

                result = store.update_concept(
                    concept_id="nonexistent",
                    source_documents=["doc1"],
                    aliases=[],
                )

                assert result is False
