"""LLM-based enrichment of detected procedural skills.

Takes raw skills produced by `src/chunker/skills.py` (phases A+B) and adds
LLM-generated metadata:

- Canonical imperative name
- Goal / success criteria
- Difficulty
- Prerequisite concepts (linked to the existing concept graph)
- Produced concepts
- Common failure modes

Raw steps from phase B are preserved verbatim — the LLM only enriches the
surrounding metadata. This keeps code/CLI snippets exactly as the textbook
wrote them.

Mirrors the summarizer pattern: optional Instructor import, Pydantic models
for structured output, profile-aware system prompts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

from .skills import AssembledSkill, RawStep

if TYPE_CHECKING:
    from .profiles import ExtractionProfile

logger = logging.getLogger(__name__)

# Optional import for instructor (same pattern as summarizer.py)
try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None  # type: ignore[assignment]
    INSTRUCTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pydantic models for LLM output
# ---------------------------------------------------------------------------


class CommonFailure(BaseModel):
    """A known failure mode for a skill."""

    symptom: str = Field(description="Observable symptom or error message")
    resolution: str = Field(description="How to resolve or work around the failure")


class EnrichedSkillMetadata(BaseModel):
    """LLM-generated metadata attached to an AssembledSkill.

    Raw steps are NOT part of this model — they come from phase B. This
    model only carries the metadata the LLM should derive.
    """

    name: str = Field(
        description=(
            "Canonical imperative title, max 8 words "
            "(e.g., 'Deploy app to Kubernetes', 'Calculate safety stock')."
        ),
    )
    goal: str = Field(
        description="1 sentence describing what the user accomplishes by executing this skill."
    )
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Approximate difficulty level for someone new to the domain."
    )
    success_criteria: str = Field(
        description="1 sentence describing how to verify the skill executed successfully."
    )
    common_failures: list[CommonFailure] = Field(
        default_factory=list,
        description="Known failure modes with resolutions (0-4 items).",
    )
    prerequisite_concepts: list[str] = Field(
        default_factory=list,
        description=(
            "Names of concepts that must be understood before attempting this skill. "
            "Use exact names from the KNOWN CONCEPTS list when possible."
        ),
    )
    prerequisite_skills: list[str] = Field(
        default_factory=list,
        description="Names of other skills that should be learned first.",
    )
    produces_concepts: list[str] = Field(
        default_factory=list,
        description=(
            "Names of concepts this skill creates or modifies in the world "
            "(e.g., 'running pod', 'approved purchase order')."
        ),
    )


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class SkillSynthesizer:
    """Enrich AssembledSkills with LLM-generated metadata via Instructor."""

    def __init__(
        self,
        provider: str = "openai/gpt-4o-mini",
        max_retries: int = 2,
        profile: "Optional[ExtractionProfile]" = None,
    ) -> None:
        """
        Initialize synthesizer.

        Args:
            provider: Instructor provider string (e.g., "openai/gpt-4o-mini")
            max_retries: Number of retries on validation failure
            profile: Extraction profile for domain-specific prompts

        Raises:
            ImportError: If instructor is not installed
        """
        if not INSTRUCTOR_AVAILABLE or instructor is None:
            raise ImportError(
                "instructor is not installed. "
                "Install with: pip install 'kidkazz[concepts]'"
            )
        self.provider = provider
        self.client = instructor.from_provider(provider)
        self.max_retries = max_retries
        self.profile = profile

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        base = (
            "You are analyzing a PROCEDURAL SKILL extracted from a textbook. "
            "You will be given:\n"
            "1. An anchor header (e.g., 'How to Deploy an Application')\n"
            "2. An ordered list of parsed steps with code snippets where present\n"
            "3. A list of known concepts from the same textbook\n\n"
            "Your job is to produce structured metadata ABOUT the skill — "
            "not to rewrite the steps. Steps are already correct. Return:\n"
            "- A concise imperative name\n"
            "- The concrete goal the user achieves\n"
            "- A difficulty estimate\n"
            "- Success criteria (how to verify it worked)\n"
            "- Common failures the textbook warns about\n"
            "- Prerequisite concepts (match exact names from KNOWN CONCEPTS)\n"
            "- Prerequisite skills\n"
            "- Produced concepts (what the skill creates)\n\n"
            "If the textbook doesn't mention something, leave the list empty. "
            "Do NOT invent prerequisites that aren't implied by the source."
        )
        if self.profile and self.profile.extraction_hints:
            base += f"\n\n{self.profile.extraction_hints}"
        return base

    def _user_prompt(
        self,
        skill: AssembledSkill,
        known_concepts: list[str],
    ) -> str:
        steps_text = "\n".join(self._format_step(s) for s in skill.steps)
        concepts_text = ", ".join(known_concepts[:100]) if known_concepts else "(none extracted yet)"
        return (
            f"ANCHOR HEADER: {skill.boundary.anchor_header}\n\n"
            f"STEPS:\n{steps_text}\n\n"
            f"KNOWN CONCEPTS:\n{concepts_text}"
        )

    @staticmethod
    def _format_step(step: RawStep) -> str:
        parts = [f"{step.step_number}. {step.action}"]
        if step.code_content:
            lang = step.code_language or ""
            parts.append(f"   ```{lang}\n{step.code_content}\n   ```")
        if step.expected_output:
            parts.append(f"   Output: {step.expected_output}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        skill: AssembledSkill,
        known_concepts: Optional[list[str]] = None,
    ) -> EnrichedSkillMetadata:
        """Call the LLM to produce enriched metadata for a single skill.

        Raises the underlying Instructor exception on failure (caller
        decides how to handle — retry, skip, fall back to defaults).
        """
        return self.client.chat.completions.create(
            model=self.provider.split("/", 1)[-1],
            response_model=EnrichedSkillMetadata,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(skill, known_concepts or [])},
            ],
            max_retries=self.max_retries,
        )

    def enrich_all(
        self,
        skills: list[AssembledSkill],
        known_concepts: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Enrich a list of skills, producing combined dicts (raw + metadata).

        Returns a list of dicts with shape:
            {
                "boundary": {...},
                "steps": [...],
                "metadata": {...} | None,   # None if enrichment failed
                "enrichment_error": str | None,
            }
        """
        results: list[dict[str, Any]] = []
        for skill in skills:
            entry: dict[str, Any] = {
                "boundary": skill.boundary.to_dict(),
                "steps": [s.to_dict() for s in skill.steps],
                "metadata": None,
                "enrichment_error": None,
            }
            try:
                enriched = self.enrich(skill, known_concepts)
                entry["metadata"] = enriched.model_dump()
            except Exception as e:  # noqa: BLE001 - log and move on
                logger.warning(
                    "Enrichment failed for skill %r: %s",
                    skill.boundary.anchor_header, e,
                )
                entry["enrichment_error"] = str(e)
            results.append(entry)
        return results
