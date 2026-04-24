"""Tests for skill storage (MockChunkStore) and MCP tool formatting."""

import json
from unittest.mock import MagicMock

import pytest

from src.storage.mock_store import MockChunkStore


def _skill_props(**overrides) -> dict:
    base = {
        "skill_id": "deploy-app",
        "name": "Deploy app",
        "goal": "Running pods",
        "domain": "programming",
        "difficulty": "intermediate",
        "source_document_id": "doc1",
        "anchor_header": "How to Deploy",
        "source_chunk_ids": json.dumps(["c1", "c2"]),
        "success_criteria": "Pods running",
        "common_failures": json.dumps([]),
        "step_count": 2,
        "has_code": 1,
        "created_at": 123456,
    }
    base.update(overrides)
    return base


def _step(step_id: str, step_number: int, action: str, **overrides) -> dict:
    base = {
        "step_id": step_id,
        "skill_id": "deploy-app",
        "step_number": step_number,
        "action": action,
        "code_content": "",
        "code_language": "",
        "expected_output": "",
        "is_optional": 0,
    }
    base.update(overrides)
    return base


class TestMockSkillStorage:
    def test_round_trip_store_and_get(self):
        store = MockChunkStore()
        steps = [
            _step("s1", 1, "Create manifest", code_content="kind: Deployment", code_language="yaml"),
            _step("s2", 2, "Apply it", code_content="kubectl apply", code_language="bash"),
        ]
        sid = store.store_skill(_skill_props(), steps)
        assert sid == "deploy-app"

        skill = store.get_skill("deploy-app")
        assert skill is not None
        assert skill["name"] == "Deploy app"
        assert skill["difficulty"] == "intermediate"

        out_steps = store.get_skill_steps("deploy-app")
        assert len(out_steps) == 2
        assert out_steps[0]["step_number"] == 1
        assert out_steps[1]["code_language"] == "bash"

    def test_get_skill_by_name(self):
        store = MockChunkStore()
        store.store_skill(_skill_props(), [])
        found = store.get_skill_by_name("Deploy app")
        assert found is not None
        assert found["skill_id"] == "deploy-app"
        assert store.get_skill_by_name("Nonexistent") is None

    def test_list_skills_filter_by_doc(self):
        store = MockChunkStore()
        store.store_skill(_skill_props(skill_id="a", name="A", source_document_id="doc1"), [])
        store.store_skill(_skill_props(skill_id="b", name="B", source_document_id="doc2"), [])

        all_skills = store.list_skills()
        assert len(all_skills) == 2

        doc1_skills = store.list_skills(doc_id="doc1")
        assert len(doc1_skills) == 1
        assert doc1_skills[0]["skill_id"] == "a"

    def test_steps_sorted_by_number(self):
        store = MockChunkStore()
        steps = [
            _step("s3", 3, "Third"),
            _step("s1", 1, "First"),
            _step("s2", 2, "Second"),
        ]
        store.store_skill(_skill_props(), steps)
        ordered = store.get_skill_steps("deploy-app")
        assert [s["step_number"] for s in ordered] == [1, 2, 3]

    def test_link_prerequisite_concept(self):
        store = MockChunkStore()
        store.store_skill(_skill_props(), [])
        assert store.link_skill_requires_concept("deploy-app", "pod-concept")
        prereqs = store.get_skill_prerequisite_concepts("deploy-app")
        assert {"concept_id": "pod-concept"} in prereqs

    def test_link_produces_concept(self):
        store = MockChunkStore()
        store.store_skill(_skill_props(), [])
        store.link_skill_produces_concept("deploy-app", "running-pod")
        produces = store.get_skill_produced_concepts("deploy-app")
        assert produces == [{"concept_id": "running-pod"}]

    def test_link_requires_skill(self):
        store = MockChunkStore()
        store.store_skill(_skill_props(), [])
        store.store_skill(_skill_props(skill_id="install", name="Install kubectl"), [])
        store.link_skill_requires_skill("deploy-app", "install")
        prereqs = store.get_skill_prerequisite_skills("deploy-app")
        assert prereqs == [{"skill_id": "install"}]

    def test_clear_resets_skill_state(self):
        """clear() must reset skill-related maps so tests don't leak state."""
        store = MockChunkStore()
        store.store_skill(_skill_props(), [_step("s1", 1, "A")])
        store.link_skill_requires_concept("deploy-app", "pod")
        store.link_skill_produces_concept("deploy-app", "running-pod")
        store.link_skill_requires_skill("deploy-app", "install")

        assert store.list_skills()
        assert store.get_skill_steps("deploy-app")

        store.clear()

        assert store.list_skills() == []
        assert store.get_skill("deploy-app") is None
        assert store.get_skill_steps("deploy-app") == []
        assert store.get_skill_prerequisite_concepts("deploy-app") == []
        assert store.get_skill_produced_concepts("deploy-app") == []
        assert store.get_skill_prerequisite_skills("deploy-app") == []


class TestMCPFormatter:
    def test_format_skill_with_steps(self):
        from src.mcp_server.tools import _format_skill_with_steps

        skill = {
            "skill_id": "deploy",
            "name": "Deploy",
            "goal": "ship it",
            "domain": "programming",
            "difficulty": "beginner",
            "success_criteria": "works",
            "common_failures": json.dumps([{"symptom": "x", "resolution": "y"}]),
            "source_document_id": "doc1",
            "anchor_header": "How to Deploy",
            "step_count": 1,
        }
        steps = [
            {
                "step_number": 1,
                "action": "Run",
                "code_content": "make",
                "code_language": "bash",
                "expected_output": "done",
                "is_optional": 0,
            },
        ]
        out = _format_skill_with_steps(skill, steps)
        assert out["name"] == "Deploy"
        assert out["common_failures"] == [{"symptom": "x", "resolution": "y"}]
        assert out["step_count"] == 1
        assert len(out["steps"]) == 1
        assert out["steps"][0]["is_optional"] is False

    def test_format_handles_invalid_json_common_failures(self):
        from src.mcp_server.tools import _format_skill_with_steps

        skill = {"skill_id": "x", "name": "X", "common_failures": "not-json"}
        out = _format_skill_with_steps(skill, [])
        assert out["common_failures"] == []


class TestGenerateSkillId:
    def test_slugify_via_cli(self):
        from src.cli.commands.skills import _slugify

        assert _slugify("Deploy App to Kubernetes") == "deploy-app-to-kubernetes"
        assert _slugify("How to Use `kubectl`!") == "how-to-use-kubectl"
        assert _slugify("   ") == "unnamed-skill"
