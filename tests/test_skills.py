"""Tests for procedural skill extraction (detection + assembly)."""

from src.chunker.skills import (
    AssembledSkill,
    RawStep,
    SkillBoundary,
    assemble_skill_steps,
    detect_skill_boundaries,
    extract_skills,
)


def _chunk(
    chunk_id: str,
    content: str,
    header_text: str | None = None,
    header_level: int | None = None,
    prev_id: str | None = None,
    next_id: str | None = None,
    document_id: str = "doc1",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "header_text": header_text,
        "header_level": header_level,
        "prev_id": prev_id,
        "next_id": next_id,
        "document_id": document_id,
    }


# ---------------------------------------------------------------------------
# Phase A: Detection
# ---------------------------------------------------------------------------


class TestDetectSkillBoundaries:
    def test_empty_chunks(self):
        assert detect_skill_boundaries([]) == []

    def test_no_anchor_headers(self):
        chunks = [
            _chunk("c1", "Some narrative text."),
            _chunk("c2", "More text without step markers."),
        ]
        assert detect_skill_boundaries(chunks) == []

    def test_how_to_header_with_numbered_steps(self):
        chunks = [
            _chunk(
                "c1",
                "## How to Deploy\n\n1. Create manifest\n2. Apply manifest",
                header_text="How to Deploy",
                header_level=2,
                next_id="c2",
            ),
            _chunk(
                "c2",
                "Now the pods are running.",
                prev_id="c1",
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1
        assert boundaries[0].anchor_header == "How to Deploy"
        assert boundaries[0].start_chunk_id == "c1"

    def test_to_verb_colon_header(self):
        chunks = [
            _chunk(
                "c1",
                "To install kubectl:\n\n1. Download binary\n2. Make executable",
                header_text="To install kubectl:",
                header_level=3,
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1

    def test_stops_at_sibling_header(self):
        chunks = [
            _chunk(
                "c1",
                "1. Step one\n2. Step two",
                header_text="How to Deploy",
                header_level=2,
                next_id="c2",
            ),
            _chunk(
                "c2",
                "Different topic here",
                header_text="How to Delete",
                header_level=2,
                prev_id="c1",
                next_id="c3",
            ),
            _chunk("c3", "More", prev_id="c2"),
        ]
        boundaries = detect_skill_boundaries(chunks)
        # Both anchors detected; first should stop at second's header
        assert len(boundaries) >= 1
        first = boundaries[0]
        assert "c2" not in first.chunk_ids

    def test_insufficient_signal_rejected(self):
        chunks = [
            _chunk(
                "c1",
                "Just some prose without steps or code.",
                header_text="How to Relax",
                header_level=2,
            ),
        ]
        assert detect_skill_boundaries(chunks) == []

    def test_code_fence_plus_one_step_accepted(self):
        chunks = [
            _chunk(
                "c1",
                "## How to Build\n\n1. Run the command\n\n```bash\nmake build\n```",
                header_text="How to Build",
                header_level=2,
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1

    def test_walks_next_id_chain(self):
        chunks = [
            _chunk(
                "c1",
                "1. First step",
                header_text="How to X",
                header_level=2,
                next_id="c2",
            ),
            _chunk("c2", "2. Second step\n3. Third step", prev_id="c1", next_id="c3"),
            _chunk("c3", "```yaml\nkey: value\n```", prev_id="c2"),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1
        assert boundaries[0].chunk_ids == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Phase B: Assembly
# ---------------------------------------------------------------------------


class TestAssembleSkillSteps:
    def test_parses_numbered_steps(self):
        chunks = [
            _chunk(
                "c1",
                "1. First do this\n2. Then do that\n3. Finally do the other",
                header_text="How to X",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to X",
            anchor_header_level=2,
            chunk_ids=["c1"],
            start_chunk_id="c1",
            end_chunk_id="c1",
        )
        result = assemble_skill_steps(boundary, chunks)
        assert len(result.steps) == 3
        assert result.steps[0].action == "First do this"
        assert result.steps[1].action == "Then do that"
        assert result.steps[2].action == "Finally do the other"

    def test_extracts_code_block(self):
        chunks = [
            _chunk(
                "c1",
                "1. Run the apply command:\n\n```bash\nkubectl apply -f deploy.yaml\n```",
                header_text="How to Deploy",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to Deploy",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        assert len(result.steps) == 1
        step = result.steps[0]
        assert "Run the apply command" in step.action
        assert step.code_content == "kubectl apply -f deploy.yaml"
        assert step.code_language == "bash"

    def test_extracts_expected_output_marker(self):
        chunks = [
            _chunk(
                "c1",
                "1. Check status:\n\n```bash\nkubectl get pods\n```\n\nOutput: STATUS Running",
                header_text="How to Check",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to Check",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        assert len(result.steps) == 1
        assert "STATUS Running" in result.steps[0].expected_output

    def test_extracts_output_from_second_fence(self):
        chunks = [
            _chunk(
                "c1",
                "1. Run:\n\n```bash\necho hi\n```\n\n```console\nhi\n```",
                header_text="How to Echo",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to Echo",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        assert result.steps[0].code_content == "echo hi"
        assert result.steps[0].expected_output == "hi"

    def test_step_word_markers(self):
        chunks = [
            _chunk(
                "c1",
                "Step 1: Open the file\nStep 2: Read the data\nStep 3: Close the file",
                header_text="How to Process",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to Process",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        assert len(result.steps) == 3
        assert result.steps[0].action == "Open the file"
        assert result.steps[1].action == "Read the data"

    def test_detects_step_word_markers_at_boundary(self):
        """Phase A should count 'Step N:' markers, not only '1.' markers."""
        chunks = [
            _chunk(
                "c1",
                "Step 1: Open\nStep 2: Read\nStep 3: Close",
                header_text="How to Process",
                header_level=2,
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1

    def test_detects_paren_step_markers_at_boundary(self):
        """Phase A should count '1)' markers, not only '1.' markers."""
        chunks = [
            _chunk(
                "c1",
                "1) First\n2) Second\n3) Third",
                header_text="How to Setup",
                header_level=2,
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1

    def test_detects_boundary_with_only_code_fences(self):
        """Two code fences + imperative text should pass the body-signal gate."""
        chunks = [
            _chunk(
                "c1",
                "Run the build:\n\n```bash\nmake\n```\n\nRun tests:\n\n```bash\nmake test\n```",
                header_text="How to Build",
                header_level=2,
            ),
        ]
        boundaries = detect_skill_boundaries(chunks)
        assert len(boundaries) == 1

    def test_renumbers_sequentially(self):
        chunks = [
            _chunk(
                "c1",
                "5. Skip number\n6. Another\n7. Third",
                header_text="How to X",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to X",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        assert [s.step_number for s in result.steps] == [1, 2, 3]

    def test_fallback_imperative_parser(self):
        chunks = [
            _chunk(
                "c1",
                "Run the installer.\n\nConfigure the credentials.\n\nStart the service.",
                header_text="How to Setup",
                header_level=2,
            ),
        ]
        boundary = SkillBoundary(
            anchor_header="How to Setup",
            anchor_header_level=2,
            chunk_ids=["c1"],
        )
        result = assemble_skill_steps(boundary, chunks)
        # Fallback should find 3 imperative paragraphs
        assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestExtractSkills:
    def test_full_pipeline(self):
        chunks = [
            _chunk(
                "c1",
                "## How to Deploy an App\n\nThis section shows how to deploy.",
                header_text="How to Deploy an App",
                header_level=2,
                next_id="c2",
            ),
            _chunk(
                "c2",
                "1. Create the manifest:\n\n```yaml\nkind: Deployment\n```",
                prev_id="c1",
                next_id="c3",
            ),
            _chunk(
                "c3",
                "2. Apply it:\n\n```bash\nkubectl apply -f deploy.yaml\n```",
                prev_id="c2",
            ),
        ]
        skills = extract_skills(chunks)
        assert len(skills) == 1
        skill = skills[0]
        assert skill.boundary.anchor_header == "How to Deploy an App"
        assert len(skill.steps) == 2
        assert skill.steps[0].code_language == "yaml"
        assert skill.steps[1].code_language == "bash"
        assert "kubectl apply" in skill.steps[1].code_content

    def test_multiple_skills_in_document(self):
        chunks = [
            _chunk(
                "c1",
                "1. Step A\n2. Step B",
                header_text="How to Deploy",
                header_level=2,
                next_id="c2",
            ),
            _chunk(
                "c2",
                "1. Step X\n2. Step Y",
                header_text="How to Delete",
                header_level=2,
                prev_id="c1",
            ),
        ]
        skills = extract_skills(chunks)
        assert len(skills) == 2
        assert skills[0].boundary.anchor_header == "How to Deploy"
        assert skills[1].boundary.anchor_header == "How to Delete"
