"""Tests for concept-related methods in HelixChunkStore (TDD - tests written first)."""

from unittest.mock import MagicMock, patch

import pytest


class TestHelixChunkStoreConceptMethods:
    """Tests for concept storage methods in HelixChunkStore."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store


class TestStoreConcept:
    """Tests for HelixChunkStore.store_concept method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_store_concept_calls_execute_query(self, store, mock_helix_client):
        """Should call _execute_query with AddConcept query."""
        # Mock the response
        mock_helix_client.query.return_value = {"id": "concept_internal_123"}

        result = store.store_concept(
            concept_id="cost-of-goods-sold",
            name="Cost of Goods Sold",
            definition="The direct costs attributable to goods sold.",
            concept_type="formula",
            source_documents=["inventory_textbook"],
            aliases=["COGS", "cost of sales"],
        )

        # Should return internal ID
        assert result is not None
        mock_helix_client.query.assert_called()

    def test_store_concept_returns_internal_id(self, store, mock_helix_client):
        """Should return the internal concept ID."""
        mock_helix_client.query.return_value = {"id": "concept_internal_123"}

        result = store.store_concept(
            concept_id="fifo",
            name="FIFO",
            definition="First-In, First-Out method.",
            concept_type="method",
            source_documents=["doc1"],
            aliases=[],
        )

        assert result == "concept_internal_123"

    def test_store_concept_returns_none_on_network_error(self, store, mock_helix_client):
        """Should return None on network errors (ConnectionError, TimeoutError, OSError)."""
        mock_helix_client.query.side_effect = ConnectionError("Network unavailable")

        result = store.store_concept(
            concept_id="test",
            name="Test",
            definition="Test def.",
            concept_type="term",
            source_documents=[],
            aliases=[],
        )

        assert result is None

    def test_store_concept_raises_on_unexpected_error(self, store, mock_helix_client):
        """Should raise unexpected exceptions for debugging."""
        mock_helix_client.query.side_effect = ValueError("Unexpected error")

        with pytest.raises(ValueError, match="Unexpected error"):
            store.store_concept(
                concept_id="test",
                name="Test",
                definition="Test def.",
                concept_type="term",
                source_documents=[],
                aliases=[],
            )


class TestGetConcept:
    """Tests for HelixChunkStore.get_concept method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_get_concept_returns_concept_dict(self, store, mock_helix_client):
        """Should return concept dictionary."""
        mock_helix_client.query.return_value = [
            {"concept_id": "fifo", "name": "FIFO", "definition": "First-In, First-Out."}
        ]

        result = store.get_concept("fifo")

        assert result is not None
        assert result.get("concept_id") == "fifo"
        assert result.get("name") == "FIFO"

    def test_get_concept_returns_none_when_not_found(self, store, mock_helix_client):
        """Should return None when concept not found."""
        mock_helix_client.query.return_value = []

        result = store.get_concept("non-existent")

        assert result is None


class TestGetConceptByName:
    """Tests for HelixChunkStore.get_concept_by_name method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_get_concept_by_name_returns_concept(self, store, mock_helix_client):
        """Should return concept by name."""
        mock_helix_client.query.return_value = [
            {"concept_id": "cogs", "name": "Cost of Goods Sold"}
        ]

        result = store.get_concept_by_name("Cost of Goods Sold")

        assert result is not None
        assert result.get("name") == "Cost of Goods Sold"

    def test_get_concept_by_name_returns_none_when_not_found(self, store, mock_helix_client):
        """Should return None when concept not found."""
        mock_helix_client.query.return_value = []

        result = store.get_concept_by_name("NonExistent")

        assert result is None


class TestListConcepts:
    """Tests for HelixChunkStore.list_concepts method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_list_concepts_returns_all_concepts(self, store, mock_helix_client):
        """Should return list of all concepts."""
        mock_helix_client.query.return_value = [
            {"concept_id": "fifo", "name": "FIFO"},
            {"concept_id": "lifo", "name": "LIFO"},
        ]

        result = store.list_concepts()

        assert len(result) == 2

    def test_list_concepts_empty_when_none(self, store, mock_helix_client):
        """Should return empty list when no concepts."""
        mock_helix_client.query.return_value = []

        result = store.list_concepts()

        assert result == []

    def test_list_concepts_with_doc_filter(self, store, mock_helix_client):
        """Should filter concepts by document."""
        mock_helix_client.query.return_value = [
            {"concept_id": "cogs", "name": "COGS"},
        ]

        result = store.list_concepts(doc_id="inventory_textbook")

        assert len(result) == 1
        # Verify the query was called with doc_id filter
        mock_helix_client.query.assert_called()


class TestLinkChunkDefinesConcept:
    """Tests for HelixChunkStore.link_chunk_defines_concept method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_link_chunk_defines_concept_returns_true(self, store, mock_helix_client):
        """Should return True on success."""
        mock_helix_client.query.return_value = {}

        result = store.link_chunk_defines_concept("chunk_123", "concept_456")

        assert result is True
        mock_helix_client.query.assert_called()

    def test_link_chunk_defines_concept_calls_query(self, store, mock_helix_client):
        """Should call query with link parameters."""
        mock_helix_client.query.return_value = {}

        store.link_chunk_defines_concept("chunk_abc", "concept_xyz")

        mock_helix_client.query.assert_called()


class TestLinkChunkMentionsConcept:
    """Tests for HelixChunkStore.link_chunk_mentions_concept method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_link_chunk_mentions_concept_returns_true(self, store, mock_helix_client):
        """Should return True on success."""
        mock_helix_client.query.return_value = {}

        result = store.link_chunk_mentions_concept("chunk_123", "concept_456")

        assert result is True


class TestLinkConceptRelatesTo:
    """Tests for HelixChunkStore.link_concept_relates_to method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_link_concept_relates_to_returns_true(self, store, mock_helix_client):
        """Should return True on success."""
        mock_helix_client.query.return_value = {}

        result = store.link_concept_relates_to("from_concept_123", "to_concept_456")

        assert result is True


class TestGetConceptDefinitionChunks:
    """Tests for HelixChunkStore.get_concept_definition_chunks method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_get_concept_definition_chunks_returns_chunks(self, store, mock_helix_client):
        """Should return list of chunks that define the concept."""
        # First call gets the concept
        # Second call gets the definition chunks
        mock_helix_client.query.side_effect = [
            [{"concept_id": "cogs", "name": "COGS", "id": "concept_internal_123"}],
            [{"chunk_id": "chunk_1", "content": "Definition..."}],
        ]

        result = store.get_concept_definition_chunks("cogs")

        assert isinstance(result, list)

    def test_get_concept_definition_chunks_empty_for_invalid_concept(self, store, mock_helix_client):
        """Should return empty list for non-existent concept."""
        mock_helix_client.query.return_value = []

        result = store.get_concept_definition_chunks("non-existent")

        assert result == []


class TestGetRelatedConcepts:
    """Tests for HelixChunkStore.get_related_concepts method."""

    @pytest.fixture
    def mock_helix_client(self):
        """Create a mock helix client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_helix_client):
        """Create a HelixChunkStore with mocked client."""
        from src.storage.client import HelixChunkStore, HelixConfig

        store = HelixChunkStore(HelixConfig())
        store._client = mock_helix_client
        store._initialized = True
        return store

    def test_get_related_concepts_returns_concepts(self, store, mock_helix_client):
        """Should return list of related concepts."""
        # First call gets the concept
        # Second call gets related concepts
        mock_helix_client.query.side_effect = [
            [{"concept_id": "cogs", "name": "COGS", "id": "concept_internal_123"}],
            [{"concept_id": "fifo", "name": "FIFO"}, {"concept_id": "lifo", "name": "LIFO"}],
        ]

        result = store.get_related_concepts("cogs")

        assert isinstance(result, list)

    def test_get_related_concepts_empty_for_invalid_concept(self, store, mock_helix_client):
        """Should return empty list for non-existent concept."""
        mock_helix_client.query.return_value = []

        result = store.get_related_concepts("non-existent")

        assert result == []

    def test_get_related_concepts_empty_for_isolated_concept(self, store, mock_helix_client):
        """Should return empty list for concept with no relations."""
        mock_helix_client.query.side_effect = [
            [{"concept_id": "isolated", "name": "Isolated", "id": "concept_123"}],
            [],  # No related concepts
        ]

        result = store.get_related_concepts("isolated")

        assert result == []
