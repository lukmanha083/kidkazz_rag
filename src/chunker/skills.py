"""Procedural skill extraction from textbook chunks.

Detects skill boundaries in documents (multi-chunk "How to" sections) and
re-assembles the fragmented chunks back into ordered, executable steps with
code snippets and expected outputs.

Two phases:

- Phase A (detect): find anchor headers + walk sibling chunks to determine
  skill boundaries. See `detect_skill_boundaries`.
- Phase B (assemble): concatenate the raw markdown in a boundary and parse
  it into RawStep objects. See `assemble_skill_steps`.

Enrichment (LLM-based naming, prerequisites, verification) lives in
`skill_synthesizer.py`. Storage orchestration (DB writes, edge linking)
is added later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Header patterns that strongly anchor a skill boundary
_ANCHOR_HEADER_PATTERNS = [
    re.compile(r"^(How\s+to|Steps\s+to|Tutorial|Walkthrough|Procedure|Exercise)\b", re.IGNORECASE),
    # "To deploy an application:" — starts with "To " then an imperative verb + colon
    re.compile(r"^To\s+[a-z]\w+.*:$", re.IGNORECASE),
]

# Markers that indicate a numbered/lettered step start inside content
_STEP_MARKER_PATTERNS = [
    re.compile(r"^\s*(\d+)\.\s+(.+)", re.MULTILINE),        # "1. Do thing"
    re.compile(r"^\s*(\d+)\)\s+(.+)", re.MULTILINE),        # "1) Do thing"
    re.compile(r"^\s*Step\s+(\d+)[:\.\s]\s*(.+)", re.MULTILINE | re.IGNORECASE),
]

# Expected output markers (Phase B)
_OUTPUT_MARKERS = re.compile(
    r"^\s*(?:Output|Returns|You\s+should\s+see|Expected\s+output|Result)[:\s]\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

# Code fence regex (captures language + content)
_CODE_FENCE = re.compile(
    r"```(\w*)\s*\n(.*?)\n```",
    re.DOTALL,
)

# Procedural body signals (used for density check)
_PROCEDURAL_SIGNALS = [
    re.compile(r"^\s*\d+[\.\)]\s+\S", re.MULTILINE),
    re.compile(r"^\s*Step\s+\d+", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^```\w+", re.MULTILINE),
    re.compile(r"\bFirst,\s", re.IGNORECASE),
    re.compile(r"\bThen,\s", re.IGNORECASE),
    re.compile(r"\bFinally,\s", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillBoundary:
    """A contiguous range of chunks that together form a single skill."""

    anchor_header: str                   # Text of the heading that anchored detection
    anchor_header_level: int             # h1-h6 level of the anchor
    chunk_ids: list[str] = field(default_factory=list)
    start_chunk_id: str = ""
    end_chunk_id: str = ""
    document_id: str = ""

    def to_dict(self) -> dict:
        return {
            "anchor_header": self.anchor_header,
            "anchor_header_level": self.anchor_header_level,
            "chunk_ids": self.chunk_ids,
            "start_chunk_id": self.start_chunk_id,
            "end_chunk_id": self.end_chunk_id,
            "document_id": self.document_id,
        }


@dataclass
class RawStep:
    """A parsed step from a skill's raw markdown."""

    step_number: int
    action: str                          # Narrative text describing the step
    code_content: str = ""               # Verbatim code from a following fence
    code_language: str = ""              # Language label from the fence
    expected_output: str = ""            # "Output:"/"You should see:" snippet

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "code_content": self.code_content,
            "code_language": self.code_language,
            "expected_output": self.expected_output,
        }


@dataclass
class AssembledSkill:
    """A skill boundary + its parsed steps. Intermediate artifact (pre-LLM)."""

    boundary: SkillBoundary
    steps: list[RawStep] = field(default_factory=list)
    raw_markdown: str = ""               # Source used for parsing (for debugging)

    def to_dict(self) -> dict:
        return {
            "boundary": self.boundary.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Phase A: Detection
# ---------------------------------------------------------------------------


def _is_anchor_header(header_text: Optional[str]) -> bool:
    """Return True if `header_text` matches any anchor pattern."""
    if not header_text:
        return False
    text = header_text.strip()
    return any(p.match(text) for p in _ANCHOR_HEADER_PATTERNS)


def _count_signals(content: str) -> int:
    """Count how many procedural signals appear in a chunk's content."""
    return sum(1 for p in _PROCEDURAL_SIGNALS if p.search(content))


def _chunk_to_meta(chunk: Any) -> dict:
    """Extract the relevant metadata dict from an EmbeddedChunk or dict.

    Accepts both `EmbeddedChunk` objects (with `.chunk.metadata`) and plain
    dicts (as produced by tests/fixtures).
    """
    if hasattr(chunk, "chunk"):
        c = chunk.chunk
        meta = dict(c.metadata or {})
        meta.setdefault("chunk_id", c.id)
        meta.setdefault("content", c.content)
        meta.setdefault("prev_id", c.prev_id)
        meta.setdefault("next_id", c.next_id)
        meta.setdefault("level", c.level)
        return meta
    return dict(chunk)


def detect_skill_boundaries(chunks: list[Any]) -> list[SkillBoundary]:
    """Find skill boundaries in a document's chunks.

    A boundary starts at a chunk whose `header_text` matches an anchor
    pattern, and extends forward via `next_id` until a sibling or shallower
    header ends it, or procedural signal density drops.

    Args:
        chunks: List of chunks for a single document. Can be EmbeddedChunk
            objects or dicts with chunk_id/content/header_text/header_level
            /prev_id/next_id fields.

    Returns:
        List of SkillBoundary objects, one per detected skill.
    """
    if not chunks:
        return []

    # Normalize and index by chunk_id for forward traversal
    metas = [_chunk_to_meta(c) for c in chunks]
    by_id = {m["chunk_id"]: m for m in metas if m.get("chunk_id")}
    # Preserve input order for anchor scan
    ordered = metas

    boundaries: list[SkillBoundary] = []
    consumed: set[str] = set()  # Chunks already part of a boundary

    for anchor in ordered:
        cid = anchor.get("chunk_id")
        if not cid or cid in consumed:
            continue
        header_text = anchor.get("header_text")
        anchor_level = anchor.get("header_level")
        if not _is_anchor_header(header_text):
            continue

        # Walk forward collecting body chunks
        collected: list[dict] = [anchor]
        current_id = anchor.get("next_id")
        weak_streak = 0
        while current_id:
            nxt = by_id.get(current_id)
            if nxt is None:
                break
            # Stop at a sibling or shallower header
            nxt_level = nxt.get("header_level")
            nxt_text = nxt.get("header_text")
            if (
                nxt_text
                and isinstance(anchor_level, int)
                and isinstance(nxt_level, int)
                and nxt_level <= anchor_level
            ):
                break

            collected.append(nxt)

            # Density check: stop if 3 consecutive chunks have no signals
            if _count_signals(nxt.get("content", "")) == 0:
                weak_streak += 1
                if weak_streak >= 3:
                    # Trim the trailing weak chunks — they don't belong
                    collected = collected[: -weak_streak]
                    break
            else:
                weak_streak = 0

            current_id = nxt.get("next_id")

        # Minimum body signal: combined content must have >=2 step markers
        # OR >=1 step marker + >=1 code fence. Fallback imperative parsing
        # handles the "code fence only" case at phase B.
        combined = "\n\n".join(m.get("content", "") for m in collected)
        numbered_count = sum(
            len(p.findall(combined)) for p in _STEP_MARKER_PATTERNS
        )
        code_count = len(_CODE_FENCE.findall(combined))
        # Accept if: 2+ step markers, OR 1+ step marker + 1+ code fence,
        # OR 2+ code fences (imperative fallback will find steps)
        if numbered_count < 2 and not (numbered_count >= 1 and code_count >= 1) and code_count < 2:
            continue

        chunk_ids = [m["chunk_id"] for m in collected if m.get("chunk_id")]
        boundary = SkillBoundary(
            anchor_header=str(header_text).strip(),
            anchor_header_level=int(anchor_level or 0),
            chunk_ids=chunk_ids,
            start_chunk_id=chunk_ids[0] if chunk_ids else "",
            end_chunk_id=chunk_ids[-1] if chunk_ids else "",
            document_id=anchor.get("document_id", ""),
        )
        boundaries.append(boundary)
        consumed.update(chunk_ids)

    return boundaries


# ---------------------------------------------------------------------------
# Phase B: Assembly
# ---------------------------------------------------------------------------


def _split_on_step_markers(text: str) -> list[tuple[int, str]]:
    """Split text at numbered/stepped markers.

    Returns a list of (step_number, segment_text) tuples. The first segment
    before any marker is dropped (it's usually the anchor's intro paragraph).
    """
    # Find all marker positions
    matches: list[tuple[int, int, str]] = []  # (start, number, following_text)
    for pat in _STEP_MARKER_PATTERNS:
        for m in pat.finditer(text):
            try:
                num = int(m.group(1))
            except (ValueError, IndexError):
                continue
            matches.append((m.start(), num))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0])

    segments: list[tuple[int, str]] = []
    for i, (start, num) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        segment = text[start:end]
        # Strip the marker itself from the segment's first line
        segment = re.sub(r"^\s*(?:\d+[\.\)]|Step\s+\d+[:\.\s])\s*", "", segment, count=1, flags=re.IGNORECASE)
        segments.append((num, segment))
    return segments


def _extract_code_and_output(segment: str) -> tuple[str, str, str, str]:
    """Extract action text, code block, language, expected output from a segment.

    Returns:
        (action_text, code_content, code_language, expected_output)
    """
    # Find the first code fence
    fence_match = _CODE_FENCE.search(segment)
    code_content = ""
    code_language = ""
    if fence_match:
        code_language = fence_match.group(1) or ""
        code_content = fence_match.group(2).strip("\n")

    # Action = text before the first code fence (or full text if no fence)
    if fence_match:
        action_text = segment[: fence_match.start()].strip()
    else:
        action_text = segment.strip()

    # Look for expected output AFTER the first fence
    expected_output = ""
    if fence_match:
        after_fence = segment[fence_match.end():]
        # Option 1: explicit "Output:" marker
        out_match = _OUTPUT_MARKERS.search(after_fence)
        if out_match:
            # Capture the marker's own line + any immediately following
            # fenced output block
            remaining = after_fence[out_match.start():]
            output_fence = _CODE_FENCE.search(remaining)
            if output_fence and output_fence.start() < 200:
                expected_output = output_fence.group(2).strip("\n")
            else:
                # Just the rest of the marker line
                expected_output = out_match.group(1).strip()
        else:
            # Option 2: second fence labeled console/output immediately after
            second_fence = _CODE_FENCE.search(after_fence)
            if second_fence and second_fence.start() < 150:
                lang = (second_fence.group(1) or "").lower()
                if lang in {"console", "output", "stdout", "text"}:
                    expected_output = second_fence.group(2).strip("\n")

    # Collapse whitespace in action text
    action_text = re.sub(r"\s+\n", "\n", action_text).strip()
    return action_text, code_content, code_language, expected_output


def assemble_skill_steps(
    boundary: SkillBoundary,
    chunks: list[Any],
) -> AssembledSkill:
    """Parse a skill boundary's chunks back into ordered steps.

    Concatenates chunk contents in the boundary's `chunk_ids` order, then
    runs a deterministic parser to extract numbered steps with code blocks
    and expected outputs.

    Args:
        boundary: The SkillBoundary to assemble.
        chunks: All chunks for the document (used to look up content).

    Returns:
        AssembledSkill with boundary + parsed RawStep list.
    """
    # Build id -> content map
    metas = [_chunk_to_meta(c) for c in chunks]
    by_id = {m["chunk_id"]: m for m in metas if m.get("chunk_id")}

    parts: list[str] = []
    for cid in boundary.chunk_ids:
        m = by_id.get(cid)
        if m and m.get("content"):
            parts.append(m["content"])
    raw = "\n\n".join(parts)

    segments = _split_on_step_markers(raw)
    steps: list[RawStep] = []

    if segments:
        for num, segment in segments:
            action, code, lang, output = _extract_code_and_output(segment)
            if not action and not code:
                continue
            steps.append(RawStep(
                step_number=num,
                action=action,
                code_content=code,
                code_language=lang,
                expected_output=output,
            ))
        # Renumber sequentially (in case the source skipped numbers)
        steps = [
            RawStep(
                step_number=i + 1,
                action=s.action,
                code_content=s.code_content,
                code_language=s.code_language,
                expected_output=s.expected_output,
            )
            for i, s in enumerate(steps)
        ]
    else:
        # Fallback: no explicit step markers. Treat each imperative
        # paragraph as a step.
        steps = _fallback_parse_imperatives(raw)

    return AssembledSkill(boundary=boundary, steps=steps, raw_markdown=raw)


_IMPERATIVE_VERBS = (
    "run", "create", "install", "configure", "deploy", "apply", "build",
    "open", "click", "navigate", "add", "remove", "update", "start", "stop",
    "write", "save", "edit", "restart", "check", "verify", "test", "set",
    "export", "import", "generate", "copy", "paste", "type", "enter",
)


def _fallback_parse_imperatives(raw: str) -> list[RawStep]:
    """Split on paragraphs; keep ones starting with an imperative verb."""
    paragraphs = re.split(r"\n\s*\n", raw)
    steps: list[RawStep] = []
    pending_action = ""

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # Check if paragraph is a code fence
        fence_match = _CODE_FENCE.match(stripped)
        if fence_match and pending_action:
            # Attach to the most recent step
            steps[-1] = RawStep(
                step_number=steps[-1].step_number,
                action=steps[-1].action,
                code_content=fence_match.group(2).strip("\n"),
                code_language=fence_match.group(1) or "",
                expected_output=steps[-1].expected_output,
            )
            continue

        first_word = stripped.split()[0].lower().rstrip(",.:")
        if first_word in _IMPERATIVE_VERBS:
            steps.append(RawStep(
                step_number=len(steps) + 1,
                action=stripped,
            ))
            pending_action = stripped

    return steps


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def extract_skills(chunks: list[Any]) -> list[AssembledSkill]:
    """Run detection + assembly in one call.

    Args:
        chunks: All chunks for a single document.

    Returns:
        List of AssembledSkill objects (one per detected boundary).
    """
    boundaries = detect_skill_boundaries(chunks)
    return [assemble_skill_steps(b, chunks) for b in boundaries]
