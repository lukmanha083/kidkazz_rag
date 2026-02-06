"""Tests for extractive summarization and keyword extraction."""

import pytest


class TestExtractiveSummarize:
    """Tests for extractive_summarize()."""

    def test_normal_text(self):
        """Should extract key sentences from multi-sentence text."""
        from src.chunker.extractive import extractive_summarize

        text = (
            "Inventory valuation is crucial for accurate financial reporting. "
            "The FIFO method assumes the first items purchased are the first sold. "
            "LIFO assumes the last items purchased are the first sold. "
            "The weighted average method calculates a mean cost per unit. "
            "Each method produces different gross profit figures. "
            "Companies must choose a method consistent with their industry."
        )

        result = extractive_summarize(text, sentence_count=2)

        assert result  # non-empty
        assert isinstance(result, str)
        # Should be shorter than the original
        assert len(result) < len(text)

    def test_empty_text(self):
        """Should return empty string for empty input."""
        from src.chunker.extractive import extractive_summarize

        assert extractive_summarize("") == ""
        assert extractive_summarize("   ") == ""

    def test_short_text(self):
        """Should return the text itself when fewer sentences than requested."""
        from src.chunker.extractive import extractive_summarize

        text = "This is a single sentence about inventory."
        result = extractive_summarize(text, sentence_count=3)

        assert result  # non-empty
        assert isinstance(result, str)

    def test_sentence_count_respected(self):
        """Should extract fewer sentences than the original text."""
        from src.chunker.extractive import extractive_summarize

        text = (
            "First sentence about topic A. "
            "Second sentence about topic B. "
            "Third sentence about topic C. "
            "Fourth sentence about topic D. "
            "Fifth sentence about topic E."
        )

        result = extractive_summarize(text, sentence_count=2)
        # Result should be shorter than original
        assert len(result) < len(text)


class TestExtractKeywords:
    """Tests for extract_keywords()."""

    def test_normal_text(self):
        """Should extract keywords from text."""
        from src.chunker.extractive import extract_keywords

        text = (
            "Inventory valuation methods include FIFO, LIFO, and weighted average. "
            "The FIFO method is most commonly used in retail businesses. "
            "LIFO provides tax advantages during periods of rising prices."
        )

        keywords = extract_keywords(text, max_keywords=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        for kw in keywords:
            assert "name" in kw
            assert "score" in kw
            assert isinstance(kw["name"], str)
            assert isinstance(kw["score"], float)

    def test_empty_text(self):
        """Should return empty list for empty input."""
        from src.chunker.extractive import extract_keywords

        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_max_keywords_limit(self):
        """Should respect max_keywords parameter."""
        from src.chunker.extractive import extract_keywords

        text = (
            "Python programming language supports object-oriented programming, "
            "functional programming, and procedural programming paradigms. "
            "It has extensive standard library and active community support."
        )

        keywords = extract_keywords(text, max_keywords=3)
        assert len(keywords) <= 3


class TestDeduplicateKeywords:
    """Tests for keyword normalization and deduplication."""

    def test_merges_singular_plural(self):
        """Should merge 'cost' and 'costs' into one keyword."""
        from src.chunker.extractive import _deduplicate_keywords

        keywords = [("costs", 0.05), ("cost", 0.03), ("pricing", 0.1)]
        result = _deduplicate_keywords(keywords)
        names = [name for name, _ in result]

        # "cost" and "costs" should merge into one
        assert len(result) == 2
        assert "cost" in names  # lower score wins
        assert "pricing" in names

    def test_normalize_ies_suffix(self):
        """Should normalize 'ies' to 'y' (e.g., inventories -> inventory)."""
        from src.chunker.extractive import _normalize_keyword

        assert _normalize_keyword("inventories") == "inventory"
        assert _normalize_keyword("policies") == "policy"

    def test_normalize_preserves_short_words(self):
        """Should not strip 's' from short words like 'is' or 'as'."""
        from src.chunker.extractive import _normalize_keyword

        assert _normalize_keyword("is") == "is"
        assert _normalize_keyword("as") == "as"

    def test_normalize_preserves_double_s(self):
        """Should not strip 's' from words ending in 'ss'."""
        from src.chunker.extractive import _normalize_keyword

        assert _normalize_keyword("loss") == "loss"
        assert _normalize_keyword("process") == "process"


class TestFallbackSummarize:
    """Tests for _fallback_summarize()."""

    def test_normal_text(self):
        """Should return first N sentences."""
        from src.chunker.extractive import _fallback_summarize

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _fallback_summarize(text, sentence_count=2)

        assert "First sentence." in result
        assert "Second sentence." in result

    def test_empty_text(self):
        """Should return empty string for empty input."""
        from src.chunker.extractive import _fallback_summarize

        assert _fallback_summarize("") == ""
        assert _fallback_summarize("   ") == ""

    def test_fewer_sentences_than_requested(self):
        """Should return all text when fewer sentences exist."""
        from src.chunker.extractive import _fallback_summarize

        text = "Only one sentence here."
        result = _fallback_summarize(text, sentence_count=5)

        assert result == "Only one sentence here."
