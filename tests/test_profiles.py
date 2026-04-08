"""Tests for extraction profiles."""

import pytest

from src.chunker.profiles import (
    ExtractionProfile,
    PROFILES,
    get_profile,
    list_profiles,
)


class TestExtractionProfile:
    def test_all_builtin_profiles_exist(self):
        expected = {"general", "programming", "erp", "stem", "agriculture", "veterinary"}
        assert set(PROFILES.keys()) == expected

    def test_all_profiles_have_concept_types(self):
        for name, profile in PROFILES.items():
            assert profile.concept_types, f"Profile '{name}' has empty concept_types"

    def test_all_profiles_have_display_name(self):
        for name, profile in PROFILES.items():
            assert profile.display_name, f"Profile '{name}' has empty display_name"

    def test_general_matches_original_5_types(self):
        """General profile must match the original 5 concept types for backward compat."""
        general = get_profile("general")
        assert general.concept_types == ["term", "method", "principle", "formula", "account"]

    def test_general_has_no_extraction_hints(self):
        """General profile should not add extra hints (backward compat)."""
        general = get_profile("general")
        assert general.extraction_hints == ""

    def test_programming_has_code_hints(self):
        prog = get_profile("programming")
        assert "code" in prog.extraction_hints.lower()
        assert "CLI" in prog.extraction_hints
        assert "algorithm" in prog.concept_types

    def test_erp_has_business_types(self):
        erp = get_profile("erp")
        assert "account" in erp.concept_types
        assert "process" in erp.concept_types
        assert "ERP" in erp.extraction_hints

    def test_veterinary_has_medical_types(self):
        vet = get_profile("veterinary")
        assert "disease" in vet.concept_types
        assert "treatment" in vet.concept_types
        assert "vaccine" in vet.concept_types

    def test_profile_is_frozen(self):
        profile = get_profile("general")
        with pytest.raises(AttributeError):
            profile.name = "modified"


class TestGetProfile:
    def test_returns_correct_profile(self):
        profile = get_profile("programming")
        assert profile.name == "programming"

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown profile 'nonexistent'"):
            get_profile("nonexistent")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            get_profile("bad_name")


class TestListProfiles:
    def test_returns_sorted_list(self):
        names = list_profiles()
        assert names == sorted(names)
        assert "general" in names
        assert "programming" in names

    def test_count(self):
        assert len(list_profiles()) == 6
