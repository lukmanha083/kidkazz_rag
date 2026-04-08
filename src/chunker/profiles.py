"""Book-type-aware extraction profiles for domain-specific concept extraction.

Each profile customizes which concept types the LLM looks for and provides
domain-specific hints for the extraction and summarization prompts.

Usage:
    from src.chunker.profiles import get_profile

    profile = get_profile("programming")
    # profile.concept_types -> ["term", "method", "algorithm", ...]
    # profile.extraction_hints -> "Look for code examples, CLI commands, ..."
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractionProfile:
    """Extraction profile for a specific book type.

    Attributes:
        name: Machine name (e.g., "programming", "erp")
        display_name: Human-readable label (e.g., "CS / Programming")
        concept_types: Concept type names the LLM should look for
        extraction_hints: Extra text appended to LLM system prompts
    """

    name: str
    display_name: str
    concept_types: list[str] = field(default_factory=list)
    extraction_hints: str = ""


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_GENERAL_TYPES = ["term", "method", "principle", "formula", "account"]

PROFILES: dict[str, ExtractionProfile] = {
    "general": ExtractionProfile(
        name="general",
        display_name="General",
        concept_types=_GENERAL_TYPES,
        extraction_hints="",
    ),
    "programming": ExtractionProfile(
        name="programming",
        display_name="CS / Programming",
        concept_types=[
            "term",
            "method",
            "algorithm",
            "data_structure",
            "pattern",
            "framework",
            "language",
            "protocol",
            "architecture",
            "function",
            "formula",
            "principle",
        ],
        extraction_hints=(
            "This is a programming / computer science textbook. Pay special attention to:\n"
            "- Code examples and snippets (preserve language identifiers)\n"
            "- CLI commands and their flags/arguments (e.g., kubectl, docker, git)\n"
            "- API signatures, function prototypes, and method calls\n"
            "- Configuration file formats (YAML, JSON, TOML, INI)\n"
            "- Design patterns and architectural concepts\n"
            "- Library and framework names as distinct concepts\n"
            "- Error messages and debugging patterns\n"
            "When a code block demonstrates a concept, note it as the concept's definition."
        ),
    ),
    "erp": ExtractionProfile(
        name="erp",
        display_name="ERP / Business",
        concept_types=[
            "term",
            "method",
            "principle",
            "formula",
            "account",
            "process",
            "system",
        ],
        extraction_hints=(
            "This is an ERP / business systems textbook covering modules like "
            "accounting, warehouse management, human resources, and procurement. "
            "Pay special attention to:\n"
            "- ERP module names and their functions\n"
            "- Business transactions and document flows\n"
            "- General ledger accounts and chart of accounts structure\n"
            "- Workflow and approval processes\n"
            "- Report names and KPIs\n"
            "- Integration points between modules"
        ),
    ),
    "stem": ExtractionProfile(
        name="stem",
        display_name="STEM / Science",
        concept_types=[
            "term",
            "method",
            "principle",
            "formula",
            "theory",
            "law",
            "process",
            "organism",
            "structure",
            "system",
            "unit",
        ],
        extraction_hints=(
            "This is a STEM / science textbook. Pay special attention to:\n"
            "- Scientific notation, equations, and units of measurement\n"
            "- Theorems, proofs, and derivations\n"
            "- Experimental procedures and methodologies\n"
            "- Physical/chemical/biological processes\n"
            "- Named laws and principles (e.g., Newton's Laws, Le Chatelier's Principle)\n"
            "- Organisms, species, and taxonomic classifications"
        ),
    ),
    "agriculture": ExtractionProfile(
        name="agriculture",
        display_name="Agriculture / Farming",
        concept_types=[
            "term",
            "method",
            "principle",
            "practice",
            "crop",
            "breed",
            "pest",
            "tool",
            "nutrient",
            "system",
            "process",
        ],
        extraction_hints=(
            "This is an agriculture / farming textbook. Pay special attention to:\n"
            "- Farming and ranching practices (crop rotation, irrigation, grazing)\n"
            "- Soil science terms and nutrient cycles\n"
            "- Crop varieties and planting specifications\n"
            "- Livestock breeds and management practices\n"
            "- Pest and disease identification and control\n"
            "- Equipment and machinery names\n"
            "- Yield calculations and economic analysis"
        ),
    ),
    "veterinary": ExtractionProfile(
        name="veterinary",
        display_name="Veterinary Science",
        concept_types=[
            "term",
            "method",
            "principle",
            "disease",
            "symptom",
            "treatment",
            "vaccine",
            "drug",
            "pathogen",
            "condition",
            "anatomy",
            "procedure",
            "feed",
            "behavior",
        ],
        extraction_hints=(
            "This is a veterinary science textbook. Pay special attention to:\n"
            "- Disease names, causative agents, and clinical signs\n"
            "- Differential diagnosis lists\n"
            "- Treatment protocols and drug dosages\n"
            "- Vaccine schedules and prevention strategies\n"
            "- Anatomical structures and body systems\n"
            "- Surgical and diagnostic procedures\n"
            "- Nutritional requirements and feed formulations\n"
            "- Animal behavior and welfare indicators"
        ),
    ),
}


def get_profile(name: str) -> ExtractionProfile:
    """Get an extraction profile by name.

    Args:
        name: Profile name (programming, erp, stem, agriculture, veterinary, general)

    Returns:
        ExtractionProfile for the given name

    Raises:
        ValueError: If profile name is not recognized
    """
    profile = PROFILES.get(name)
    if profile is None:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return profile


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(PROFILES)
