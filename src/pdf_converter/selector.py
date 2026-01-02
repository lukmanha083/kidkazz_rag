"""PDF selection and tool recommendation logic."""

from pathlib import Path
from typing import Optional


# Tool metadata: (display_name, description, estimated_time)
TOOL_META = {
    "docling": ("Docling", "Mixed content, good tables", "~2-3 min/100pg"),
    "marker": ("Marker", "Fast, clean layouts", "~1-2 min/100pg"),
    "nougat": ("Nougat", "Academic, math-heavy", "~5-10 min/100pg"),
}

# Scoring rules for tool recommendation
SCORING_RULES = {
    "content_type": {
        "math": {"nougat": 3, "marker": 1, "docling": 0},
        "tables": {"docling": 3, "marker": 2, "nougat": 0},
        "text": {"marker": 3, "docling": 2, "nougat": 0},
        "mixed": {"docling": 3, "marker": 2, "nougat": 1},
    },
    "pdf_type": {
        "scanned": {"nougat": 2, "docling": 2, "marker": 0},
        "digital": {"marker": 2, "docling": 1, "nougat": 0},
        "mixed": {"docling": 2, "marker": 0, "nougat": 0},
    },
    "priority": {
        "speed": {"marker": 3, "docling": 1, "nougat": 0},
        "quality": {"nougat": 2, "docling": 2, "marker": 0},
        "balance": {"docling": 2, "marker": 2, "nougat": 0},
    },
}


def select_pdf(pdf_list: list, choice: Optional[str] = None) -> Optional[str]:
    """
    Select a PDF from the detected files.

    Args:
        pdf_list: List of Path objects pointing to PDF files
        choice: Optional pre-selected choice (1-indexed string)

    Returns:
        Path string of selected PDF, or None if no PDFs available
    """
    if not pdf_list:
        return None

    if len(pdf_list) == 1:
        return str(pdf_list[0])

    if choice is not None:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(pdf_list):
                return str(pdf_list[idx])
        except (ValueError, IndexError):
            pass

    # Default to first PDF
    return str(pdf_list[0])


def calculate_tool_scores(
    content_type: str,
    pdf_type: str,
    priority: str
) -> dict[str, int]:
    """
    Calculate tool scores based on user preferences.

    Args:
        content_type: One of "math", "tables", "text", "mixed"
        pdf_type: One of "scanned", "digital", "mixed"
        priority: One of "speed", "quality", "balance"

    Returns:
        Dictionary of tool names to scores
    """
    scores = {"docling": 0, "marker": 0, "nougat": 0}

    # Apply content type scoring
    if content_type in SCORING_RULES["content_type"]:
        for tool, pts in SCORING_RULES["content_type"][content_type].items():
            scores[tool] += pts

    # Apply PDF type scoring
    if pdf_type in SCORING_RULES["pdf_type"]:
        for tool, pts in SCORING_RULES["pdf_type"][pdf_type].items():
            scores[tool] += pts

    # Apply priority scoring
    if priority in SCORING_RULES["priority"]:
        for tool, pts in SCORING_RULES["priority"][priority].items():
            scores[tool] += pts

    return scores


def get_tool_recommendation(
    content_type: str,
    pdf_type: str,
    priority: str
) -> str:
    """
    Get recommended tool based on user preferences.

    Args:
        content_type: One of "math", "tables", "text", "mixed"
        pdf_type: One of "scanned", "digital", "mixed"
        priority: One of "speed", "quality", "balance"

    Returns:
        Recommended tool name ("marker", "docling", or "nougat")
    """
    scores = calculate_tool_scores(content_type, pdf_type, priority)
    return max(scores, key=scores.get)
