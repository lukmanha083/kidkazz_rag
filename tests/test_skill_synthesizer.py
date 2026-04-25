"""Tests for LLM-based skill enrichment (Instructor mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.chunker.skills import AssembledSkill, RawStep, SkillBoundary


def _sample_skill() -> AssembledSkill:
    boundary = SkillBoundary(
        anchor_header="How to Deploy an App",
        anchor_header_level=2,
        chunk_ids=["c1", "c2"],
        start_chunk_id="c1",
        end_chunk_id="c2",
        document_id="kubernetes_guide",
    )
    steps = [
        RawStep(
            step_number=1,
            action="Create a deployment manifest",
            code_content="apiVersion: apps/v1\nkind: Deployment",
            code_language="yaml",
        ),
        RawStep(
            step_number=2,
            action="Apply the manifest",
            code_content="kubectl apply -f deploy.yaml",
            code_language="bash",
            expected_output="deployment.apps/my-app created",
        ),
    ]
    return AssembledSkill(boundary=boundary, steps=steps, raw_markdown="...")


@patch("src.chunker.skill_synthesizer.INSTRUCTOR_AVAILABLE", False)
def test_init_raises_without_instructor():
    from src.chunker.skill_synthesizer import SkillSynthesizer

    with pytest.raises(ImportError, match="instructor is not installed"):
        SkillSynthesizer()


@patch("src.chunker.skill_synthesizer.INSTRUCTOR_AVAILABLE", True)
@patch("src.chunker.skill_synthesizer.instructor")
class TestSkillSynthesizer:
    def test_enrich_single_skill(self, mock_instructor):
        from src.chunker.skill_synthesizer import EnrichedSkillMetadata, SkillSynthesizer

        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client

        expected = EnrichedSkillMetadata(
            name="Deploy app to Kubernetes",
            goal="Run a pod matching the deployment spec",
            difficulty="intermediate",
            success_criteria="Pods in Running state",
            common_failures=[],
            prerequisite_concepts=["Pod", "Deployment"],
            prerequisite_skills=["Install kubectl"],
            produces_concepts=["Running Pod"],
        )
        mock_client.chat.completions.create.return_value = expected

        synth = SkillSynthesizer(provider="openai/gpt-4o-mini")
        result = synth.enrich(_sample_skill(), known_concepts=["Pod", "Deployment", "Service"])

        assert result.name == "Deploy app to Kubernetes"
        assert result.difficulty == "intermediate"
        assert result.prerequisite_concepts == ["Pod", "Deployment"]
        assert mock_client.chat.completions.create.call_count == 1

    def test_system_prompt_includes_profile_hints(self, mock_instructor):
        from src.chunker.profiles import get_profile
        from src.chunker.skill_synthesizer import SkillSynthesizer

        mock_instructor.from_provider.return_value = MagicMock()

        profile = get_profile("programming")
        synth = SkillSynthesizer(provider="openai/gpt-4o-mini", profile=profile)
        prompt = synth._system_prompt()
        assert "programming" in prompt.lower() or "code" in prompt.lower()

    def test_user_prompt_formats_steps(self, mock_instructor):
        from src.chunker.skill_synthesizer import SkillSynthesizer

        mock_instructor.from_provider.return_value = MagicMock()
        synth = SkillSynthesizer()
        prompt = synth._user_prompt(_sample_skill(), known_concepts=["Pod"])

        assert "ANCHOR HEADER: How to Deploy an App" in prompt
        assert "1. Create a deployment manifest" in prompt
        assert "2. Apply the manifest" in prompt
        assert "```yaml" in prompt
        assert "kubectl apply" in prompt
        assert "Pod" in prompt

    def test_enrich_all_handles_failures(self, mock_instructor):
        from src.chunker.skill_synthesizer import SkillSynthesizer

        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client
        # First call succeeds, second raises
        from src.chunker.skill_synthesizer import EnrichedSkillMetadata

        good = EnrichedSkillMetadata(
            name="Good Skill",
            goal="Do something",
            difficulty="beginner",
            success_criteria="Works",
        )
        mock_client.chat.completions.create.side_effect = [good, RuntimeError("rate limit")]

        synth = SkillSynthesizer()
        skills = [_sample_skill(), _sample_skill()]
        results = synth.enrich_all(skills)

        assert len(results) == 2
        assert results[0]["metadata"]["name"] == "Good Skill"
        assert results[0]["enrichment_error"] is None
        assert results[1]["metadata"] is None
        assert "rate limit" in results[1]["enrichment_error"]

    def test_enrich_all_preserves_raw_steps(self, mock_instructor):
        """LLM only fills metadata; raw steps must pass through unchanged."""
        from src.chunker.skill_synthesizer import EnrichedSkillMetadata, SkillSynthesizer

        mock_client = MagicMock()
        mock_instructor.from_provider.return_value = mock_client
        mock_client.chat.completions.create.return_value = EnrichedSkillMetadata(
            name="x", goal="x", difficulty="beginner", success_criteria="x",
        )

        synth = SkillSynthesizer()
        skills = [_sample_skill()]
        results = synth.enrich_all(skills)

        # Steps dict should match original code/language verbatim
        assert results[0]["steps"][1]["code_content"] == "kubectl apply -f deploy.yaml"
        assert results[0]["steps"][1]["code_language"] == "bash"
