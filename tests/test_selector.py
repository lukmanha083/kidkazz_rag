"""Tests for PDF selection and tool recommendation."""

import pytest
from pathlib import Path

from src.pdf_converter.selector import (
    select_pdf,
    calculate_tool_scores,
    get_tool_recommendation,
    SCORING_RULES,
)


class TestSelectPdf:
    """Tests for select_pdf function."""

    def test_empty_list_returns_none(self):
        """Should return None when no PDFs available."""
        result = select_pdf([])
        assert result is None

    def test_single_pdf_auto_selects(self, single_pdf_path):
        """Should auto-select when only one PDF."""
        result = select_pdf(single_pdf_path)
        assert result == str(single_pdf_path[0])

    def test_valid_choice_selects_correct_pdf(self, sample_pdf_paths):
        """Should select the PDF at the chosen index."""
        result = select_pdf(sample_pdf_paths, choice="2")
        assert result == str(sample_pdf_paths[1])

    def test_first_choice_selects_first_pdf(self, sample_pdf_paths):
        """Should select first PDF when choice is 1."""
        result = select_pdf(sample_pdf_paths, choice="1")
        assert result == str(sample_pdf_paths[0])

    def test_last_choice_selects_last_pdf(self, sample_pdf_paths):
        """Should select last PDF when choice matches length."""
        result = select_pdf(sample_pdf_paths, choice="3")
        assert result == str(sample_pdf_paths[2])

    def test_invalid_choice_defaults_to_first(self, sample_pdf_paths):
        """Should default to first PDF on invalid choice."""
        result = select_pdf(sample_pdf_paths, choice="invalid")
        assert result == str(sample_pdf_paths[0])

    def test_out_of_range_choice_defaults_to_first(self, sample_pdf_paths):
        """Should default to first PDF when choice is out of range."""
        result = select_pdf(sample_pdf_paths, choice="99")
        assert result == str(sample_pdf_paths[0])

    def test_zero_choice_defaults_to_first(self, sample_pdf_paths):
        """Should default to first PDF when choice is 0."""
        result = select_pdf(sample_pdf_paths, choice="0")
        assert result == str(sample_pdf_paths[0])

    def test_negative_choice_defaults_to_first(self, sample_pdf_paths):
        """Should default to first PDF when choice is negative."""
        result = select_pdf(sample_pdf_paths, choice="-1")
        assert result == str(sample_pdf_paths[0])

    def test_none_choice_defaults_to_first(self, sample_pdf_paths):
        """Should default to first PDF when no choice provided."""
        result = select_pdf(sample_pdf_paths, choice=None)
        assert result == str(sample_pdf_paths[0])


class TestCalculateToolScores:
    """Tests for calculate_tool_scores function."""

    def test_math_content_favors_nougat(self):
        """Math content should score highest for nougat when quality is priority."""
        # Use scanned + quality to maximize nougat advantage
        scores = calculate_tool_scores("math", "scanned", "quality")
        assert scores["nougat"] >= scores["marker"]
        assert scores["nougat"] >= scores["docling"]

    def test_tables_content_favors_docling(self):
        """Table content should score highest for docling."""
        scores = calculate_tool_scores("tables", "digital", "balance")
        assert scores["docling"] >= scores["marker"]
        assert scores["docling"] >= scores["nougat"]

    def test_text_content_favors_marker(self):
        """Text content should score highest for marker."""
        scores = calculate_tool_scores("text", "digital", "speed")
        assert scores["marker"] >= scores["docling"]
        assert scores["marker"] >= scores["nougat"]

    def test_speed_priority_boosts_marker(self):
        """Speed priority should boost marker score."""
        speed_scores = calculate_tool_scores("mixed", "digital", "speed")
        quality_scores = calculate_tool_scores("mixed", "digital", "quality")
        assert speed_scores["marker"] > quality_scores["marker"]

    def test_quality_priority_boosts_nougat_and_docling(self):
        """Quality priority should boost nougat and docling."""
        quality_scores = calculate_tool_scores("mixed", "digital", "quality")
        speed_scores = calculate_tool_scores("mixed", "digital", "speed")
        assert quality_scores["nougat"] > speed_scores["nougat"]
        assert quality_scores["docling"] > speed_scores["docling"]

    def test_scanned_pdf_boosts_nougat_and_docling(self):
        """Scanned PDFs should boost nougat and docling scores."""
        scanned_scores = calculate_tool_scores("text", "scanned", "balance")
        digital_scores = calculate_tool_scores("text", "digital", "balance")
        assert scanned_scores["nougat"] > digital_scores["nougat"]
        assert scanned_scores["docling"] > digital_scores["docling"]

    def test_all_scores_are_non_negative(self):
        """All scores should be non-negative."""
        for content in ["math", "tables", "text", "mixed"]:
            for pdf_type in ["scanned", "digital", "mixed"]:
                for priority in ["speed", "quality", "balance"]:
                    scores = calculate_tool_scores(content, pdf_type, priority)
                    for tool, score in scores.items():
                        assert score >= 0, f"Negative score for {tool}"

    def test_unknown_content_type_still_works(self):
        """Unknown content type should not crash."""
        scores = calculate_tool_scores("unknown", "digital", "balance")
        assert "marker" in scores
        assert "docling" in scores
        assert "nougat" in scores


class TestGetToolRecommendation:
    """Tests for get_tool_recommendation function."""

    def test_math_heavy_recommends_nougat(self):
        """Math-heavy academic content should recommend nougat."""
        result = get_tool_recommendation("math", "scanned", "quality")
        assert result == "nougat"

    def test_tables_heavy_recommends_docling(self):
        """Table-heavy content should recommend docling."""
        result = get_tool_recommendation("tables", "digital", "quality")
        assert result == "docling"

    def test_text_with_speed_recommends_marker(self):
        """Text content with speed priority should recommend marker."""
        result = get_tool_recommendation("text", "digital", "speed")
        assert result == "marker"

    def test_mixed_content_recommends_docling(self):
        """Mixed content should typically recommend docling."""
        result = get_tool_recommendation("mixed", "mixed", "balance")
        assert result == "docling"

    def test_returns_valid_tool_name(self):
        """Should always return a valid tool name."""
        valid_tools = {"marker", "docling", "nougat"}
        for content in ["math", "tables", "text", "mixed"]:
            for pdf_type in ["scanned", "digital", "mixed"]:
                for priority in ["speed", "quality", "balance"]:
                    result = get_tool_recommendation(content, pdf_type, priority)
                    assert result in valid_tools


class TestScoringRulesConsistency:
    """Tests to ensure scoring rules are consistent."""

    def test_all_tools_in_content_type_rules(self):
        """All tools should be scored for each content type."""
        for content_type, scores in SCORING_RULES["content_type"].items():
            for tool in ["marker", "docling", "nougat"]:
                assert tool in scores or scores.get(tool, 0) == 0

    def test_all_tools_in_pdf_type_rules(self):
        """All tools should be scored for each PDF type."""
        for pdf_type, scores in SCORING_RULES["pdf_type"].items():
            for tool in ["marker", "docling", "nougat"]:
                assert tool in scores or scores.get(tool, 0) == 0

    def test_all_tools_in_priority_rules(self):
        """All tools should be scored for each priority."""
        for priority, scores in SCORING_RULES["priority"].items():
            for tool in ["marker", "docling", "nougat"]:
                assert tool in scores or scores.get(tool, 0) == 0
